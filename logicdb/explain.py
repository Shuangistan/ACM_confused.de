"""Turning a proof DAG into something a person can check.

The distinction this module exists to preserve: an explanation here is a
*rendering of the actual derivation*, not a plausible story generated alongside
the answer. If the text says the conclusion rests on three facts and a rule, it
rests on exactly those, because the text is produced by walking the structure
that produced the answer. That is the whole reason for putting the logic in a
database instead of leaving it inside a model.

Four things get rendered, and the last two are the ones reviewers actually use:

* **The derivation** -- what concluded what, from which facts, by which rule.
* **Provenance** -- every leaf traces to a source, and every rule to whoever
  approved it. A proof that bottoms out in unattributed assertions has not
  explained anything.
* **Counterfactuals** -- which single facts, if different, would change the
  answer. In review this is what people reach for first: not "why this
  conclusion" but "what would have to be true for it to be otherwise".
* **Assumptions** -- the independence assumption behind noisy-OR, any unknowns,
  and any approximation. Stated rather than buried in a number.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .engine import DerivedAtom, Evaluation, Support
from .facts import FactStore, Truth
from .probability import Answer, ProbabilityComputer
from .program import RuleBase, RuleRecord, RuleStatus
from .syntax import Atom, Vocabulary

INDENT = "  "


# --------------------------------------------------------------------------
# Phrasing
# --------------------------------------------------------------------------


def phrase_atom(atom: Atom, vocabulary: Vocabulary | None) -> str:
    """Render an atom as a sentence when the pack supplies a template.

    Predicate soup is readable by the person who wrote the rules and by nobody
    else. Since the reviewer of a proposed rule may be a domain expert rather
    than a logician, `overdrawn(app_17)` is rendered as "application 17 is
    overdrawn" whenever a `phrase` was declared.
    """
    if vocabulary is not None:
        decl = vocabulary.get(atom.signature)
        if decl is not None and decl.phrase:
            try:
                return decl.phrase.format(*[str(t) for t in atom.terms])
            except (IndexError, KeyError):
                pass
    return str(atom)


# --------------------------------------------------------------------------
# Counterfactuals
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Counterfactual:
    """A fact that, alone, would change the conclusion."""

    atom: Atom
    current: Truth
    alternative: Truth
    conclusion_holds_now: bool
    conclusion_holds_then: bool

    def render(self, vocabulary: Vocabulary | None = None) -> str:
        subject = phrase_atom(self.atom, vocabulary)
        then = "would hold" if self.conclusion_holds_then else "would not hold"
        return f"if {subject} were {self.alternative.value}, the conclusion {then}"


def find_counterfactuals(
    goal: Atom,
    facts: FactStore,
    rulebase: RuleBase,
    critical: bool = False,
    limit: int = 6,
) -> list[Counterfactual]:
    """Single-fact changes that would flip the conclusion.

    Deliberately one-at-a-time. Searching combinations would find more, but a
    reviewer cannot hold "if these four things were jointly different" in their
    head, and an explanation nobody can check is not an explanation.
    """
    from .engine import evaluate

    baseline = evaluate(rulebase, facts, critical)
    holds_now = baseline.holds(goal)

    out: list[Counterfactual] = []
    for record in list(facts):
        if record.truth is Truth.UNKNOWN:
            continue  # unknowns are handled by the question ranking, not here
        flipped = Truth.FALSE if record.truth is Truth.TRUE else Truth.TRUE
        probe = facts.with_assignment({record.atom: flipped})
        holds_then = evaluate(rulebase, probe, critical).holds(goal)
        if holds_then != holds_now:
            out.append(
                Counterfactual(
                    atom=record.atom,
                    current=record.truth,
                    alternative=flipped,
                    conclusion_holds_now=holds_now,
                    conclusion_holds_then=holds_then,
                )
            )
        if len(out) >= limit:
            break
    return out


# --------------------------------------------------------------------------
# Proof rendering
# --------------------------------------------------------------------------


@dataclass
class ProofNode:
    """One step of a rendered derivation."""

    atom: Atom
    kind: str                      # "fact" | "derived" | "unknown" | "absent"
    text: str
    detail: str = ""
    probability: float | None = None
    rule_id: str | None = None
    children: list["ProofNode"] = field(default_factory=list)

    def walk(self):
        yield self
        for child in self.children:
            yield from child.walk()


class Explainer:
    """Renders evaluations as proofs, provenance and counterfactuals."""

    def __init__(
        self,
        rulebase: RuleBase,
        vocabulary: Vocabulary | None = None,
        max_depth: int = 12,
    ) -> None:
        self.rulebase = rulebase
        self.vocabulary = vocabulary
        self.max_depth = max_depth

    # -- structure ---------------------------------------------------------
    def uncertain_atoms(self, facts: FactStore, critical: bool = False) -> set[Atom]:
        """Atoms that are genuinely unknown, before any assumption is applied.

        Needed because the evaluation carried on an `Answer` has its unknowns
        pinned to one end of the interval. Rendering that evaluation naively
        would report an assumed value as an established one -- telling a reviewer
        "confirmed not to hold" about a fact nobody ever supplied. That is the
        closed-world error reappearing in the explanation, which is the worst
        place for it, since the explanation is what the reviewer actually trusts.
        """
        from .engine import evaluate

        unpinned = evaluate(self.rulebase, facts, critical)
        uncertain = {r.atom for r in facts if r.truth is Truth.UNKNOWN}
        uncertain |= {b.would_derive for b in unpinned.blocked}
        return uncertain

    def proof_tree(
        self,
        goal: Atom,
        evaluation: Evaluation,
        _seen: set[Atom] | None = None,
        _depth: int = 0,
        uncertain: set[Atom] | None = None,
    ) -> ProofNode:
        seen = set() if _seen is None else _seen
        uncertain = uncertain or set()
        computer = ProbabilityComputer(evaluation)

        node = evaluation.derived.get(goal)
        phrase = phrase_atom(goal, self.vocabulary)

        if node is None:
            truth = evaluation.facts.truth_of(goal)
            record = evaluation.facts.record_of(goal)
            if truth is Truth.TRUE:
                return ProofNode(
                    atom=goal, kind="fact", text=phrase,
                    detail=f"source: {record.source}" if record else "asserted",
                    probability=record.confidence if record else 1.0,
                )
            if truth is Truth.UNKNOWN:
                return ProofNode(
                    atom=goal, kind="unknown", text=phrase,
                    detail=(
                        f"not known ({record.source})" if record
                        else "never recorded, so treated as unknown rather than false"
                    ),
                )
            return ProofNode(
                atom=goal, kind="absent", text=phrase,
                detail=f"does not hold ({record.source})" if record else "does not hold",
                probability=0.0,
            )

        if goal in seen or _depth >= self.max_depth:
            return ProofNode(
                atom=goal, kind="derived", text=phrase,
                detail="(already shown above)",
                probability=computer.probability_of(goal),
            )
        seen = seen | {goal}

        children: list[ProofNode] = []
        for support in node.supports:
            children.append(
                self._render_support(support, evaluation, seen, _depth + 1, uncertain)
            )

        return ProofNode(
            atom=goal, kind="derived", text=phrase,
            detail=f"{len(node.supports)} supporting rule(s)",
            probability=computer.probability_of(goal),
            children=children,
        )

    def _render_support(
        self,
        support: Support,
        evaluation: Evaluation,
        seen: set[Atom],
        depth: int,
        uncertain: set[Atom],
    ) -> ProofNode:
        record = self.rulebase.get(support.rule_id)
        premises = [
            self.proof_tree(atom, evaluation, seen, depth, uncertain)
            if not negated
            else self._render_negated(atom, uncertain)
            for atom, negated in support.premises
        ]
        return ProofNode(
            atom=support.rule.head,
            kind="derived",
            text=f"by rule {support.rule_id}: {support.rule}",
            detail=self._rule_provenance(record, support.strength),
            probability=support.strength,
            rule_id=support.rule_id,
            children=premises,
        )

    def _render_negated(self, atom: Atom, uncertain: set[Atom]) -> ProofNode:
        """Render `not X`, distinguishing established from assumed.

        "X is confirmed not to hold" and "X was never supplied, so this bound
        assumes it does not hold" are entirely different claims, and only the
        first is evidence. Collapsing them is how an interval endpoint gets read
        as a finding.
        """
        phrase = phrase_atom(atom, self.vocabulary)
        if atom in uncertain:
            return ProofNode(
                atom=atom,
                kind="unknown",
                text=f"not {phrase}",
                detail=(
                    "ASSUMED, not established -- this is unknown, and this end of "
                    "the interval is what follows if it does not hold"
                ),
            )
        return ProofNode(
            atom=atom,
            kind="absent",
            text=f"not {phrase}",
            detail="confirmed not to hold",
        )

    def _rule_provenance(self, record: RuleRecord | None, strength: float) -> str:
        if record is None:
            return f"strength {strength:.2f}"
        bits = [f"strength {strength:.2f}", f"origin {record.origin.value}"]
        if record.status is RuleStatus.APPROVED:
            bits.append(f"approved by {record.approved_by}")
        else:
            bits.append(f"status {record.status.value} -- NOT yet approved")
        if record.source_citation:
            bits.append(f"cites {record.source_citation}")
        if record.stats.observations:
            bits.append(
                f"right {record.stats.successes}/{record.stats.observations} in use"
            )
        return "; ".join(bits)

    # -- text --------------------------------------------------------------
    def render_proof(
        self,
        goal: Atom,
        evaluation: Evaluation,
        uncertain: set[Atom] | None = None,
    ) -> str:
        return "\n".join(
            self._render_lines(
                self.proof_tree(goal, evaluation, uncertain=uncertain), 0
            )
        )

    def _render_lines(self, node: ProofNode, depth: int) -> list[str]:
        marker = {
            "fact": "[fact]", "derived": "[rule]",
            "unknown": "[UNKNOWN]", "absent": "[not]",
        }[node.kind]
        probability = (
            f"  p={node.probability:.2f}" if node.probability is not None else ""
        )
        lines = [f"{INDENT * depth}{marker} {node.text}{probability}"]
        if node.detail:
            lines.append(f"{INDENT * (depth + 1)}({node.detail})")
        for child in node.children:
            lines.extend(self._render_lines(child, depth + 1))
        return lines

    def explain_answer(
        self, answer: Answer, facts: FactStore, critical: bool = False
    ) -> str:
        """The full account: conclusion, proof, what is missing, what would change it."""
        goal_phrase = phrase_atom(answer.goal, self.vocabulary)
        lines = [f"Question: does {goal_phrase}?", ""]

        if answer.is_uncovered:
            lines += [
                f"Answer: no applicable logic. P = {answer.bounds}",
                "",
                "The rule base has nothing that bears on this case. This is not a "
                "'no' -- it is an absence of applicable logic, and the difference "
                "matters. A rule needs to be proposed and approved before this "
                "question can be answered.",
            ]
            return "\n".join(lines)

        lines.append(f"Answer: P = {answer.bounds}")
        if answer.bounds.width > 0.01:
            lines.append(
                f"  The interval is {answer.bounds.width:.2f} wide because facts "
                f"are missing. It is not a confidence interval over a model -- it "
                f"spans every way the unknowns could turn out."
            )
        lines.append("")

        if len(answer.outcomes) > 1:
            lines.append("Competing outcomes (not normalised, so ignorance stays visible):")
            for name, bounds in sorted(
                answer.outcomes.items(), key=lambda kv: -kv[1].point
            ):
                lines.append(f"  {name:32s} {bounds}")
            lines.append("")

        uncertain = self.uncertain_atoms(facts, critical)
        lines.append("Derivation:")
        lines.append(self.render_proof(answer.goal, answer.evaluation, uncertain))
        lines.append("")

        if answer.ask:
            lines.append("Missing information, ranked by whether it would change anything:")
            for candidate in answer.ask:
                tag = "DECISIVE" if candidate.decisive else (
                    f"narrows by {candidate.information_gain:.2f}"
                )
                lines.append(
                    f"  {phrase_atom(candidate.atom, self.vocabulary)}   [{tag}]"
                )
                lines.append(
                    f"      if true -> {candidate.if_true}, "
                    f"if false -> {candidate.if_false}"
                )
            lines.append("")

        counterfactuals = find_counterfactuals(
            answer.goal, facts, self.rulebase, critical
        )
        if counterfactuals:
            lines.append("What would change the conclusion:")
            for cf in counterfactuals:
                lines.append(f"  {cf.render(self.vocabulary)}")
            lines.append("")

        if answer.assumptions:
            lines.append("Assumptions made in reaching this:")
            for assumption in answer.assumptions:
                lines.append(f"  - {assumption}")
            lines.append("")

        unapproved = self._unapproved_rules(answer.evaluation)
        if unapproved:
            lines.append(
                f"Governance: {len(unapproved)} rule(s) in this derivation are not "
                f"approved ({', '.join(unapproved)}). This answer is provisional."
            )

        return "\n".join(lines)

    def _unapproved_rules(self, evaluation: Evaluation) -> list[str]:
        out = []
        for rule_id in sorted(evaluation.rules_used):
            record = self.rulebase.get(rule_id)
            if record is not None and record.status is not RuleStatus.APPROVED:
                out.append(f"{rule_id} [{record.status.value}]")
        return out

    # -- provenance --------------------------------------------------------
    def provenance_report(self, goal: Atom, evaluation: Evaluation) -> str:
        """Every fact and rule the conclusion rests on, with its source.

        This is the artifact that answers "can responsibility for this decision
        be identified?" -- not as a score, but as a list with a named source
        against every line, or a gap where one is missing.
        """
        tree = self.proof_tree(goal, evaluation)
        facts_seen: dict[str, str] = {}
        rules_seen: dict[str, str] = {}

        for node in tree.walk():
            if node.kind == "fact":
                facts_seen[str(node.atom)] = node.detail
            elif node.rule_id:
                record = self.rulebase.get(node.rule_id)
                if record is not None:
                    who = record.approved_by or "NOBODY"
                    rules_seen[node.rule_id] = (
                        f"{record.origin.value}, {record.status.value}, "
                        f"approved by {who}"
                    )

        lines = [f"Provenance for {phrase_atom(goal, self.vocabulary)}", "", "Facts:"]
        lines += [f"  {atom:40s} {source}" for atom, source in sorted(facts_seen.items())] or [
            "  (none)"
        ]
        lines += ["", "Rules:"]
        lines += [f"  {rid:10s} {detail}" for rid, detail in sorted(rules_seen.items())] or [
            "  (none)"
        ]

        unattributed = [
            f"{rid} ({detail})" for rid, detail in rules_seen.items() if "NOBODY" in detail
        ]
        if unattributed:
            lines += [
                "",
                "GAP: the following rules have no approver, so responsibility for "
                "this conclusion cannot be fully traced:",
            ] + [f"  {u}" for u in unattributed]
        return "\n".join(lines)


__all__ = ["Counterfactual", "Explainer", "ProofNode", "find_counterfactuals", "phrase_atom"]
