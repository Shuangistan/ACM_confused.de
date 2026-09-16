"""The deterministic agent: fixed procedures for extraction and rule mining.

Deterministic on purpose. When the claim being demonstrated is that decisions
are reproducible and explainable, the default proposal path should not itself be
a coin flip -- and a reviewer comparing two runs should see the same candidate
rules, not a different sample of them.

The mining search is deliberately shallow: conjunctions of at most three
observables, filtered by support and precision, ranked, deduplicated, and
pruned of anything a more general accepted rule already covers. A deeper search
would find more rules and better-fitting ones, and that would be a mistake. The
output is a review queue for a human, and a queue of two hundred
near-duplicate five-term conjunctions is worse than a queue of eight readable
ones: the reviewer stops reading, and rule-level oversight degrades into exactly
the rubber-stamping the design exists to avoid. The constraint is on the human's
attention, not the CPU.
"""

from __future__ import annotations

import itertools
import re
from dataclasses import dataclass
from typing import Any, Iterable

from ..facts import FactStore, Truth
from ..packs import DomainPack
from ..parser import ParseError, parse_rule
from ..program import RuleOrigin
from ..syntax import Atom, Literal, Rule, Var, Vocabulary
from .base import LabelledCase, RuleProposal


@dataclass
class MiningConfig:
    """Thresholds governing what gets proposed.

    The defaults are tuned for reviewability rather than coverage. `min_support`
    of 20 keeps out rules that fit a handful of cases by luck; `min_precision`
    of 0.70 keeps out anything close enough to chance that a reviewer could not
    defend approving it; `max_body` of 3 keeps rules readable.
    """

    max_body: int = 3
    min_support: int = 20
    min_precision: float = 0.70
    #: A longer rule must beat the shorter one it extends by this much, or it is
    #: dropped as needless complexity. Without it the search returns a general
    #: rule plus a dozen barely-better refinements of it.
    min_gain_over_subsumer: float = 0.05
    max_proposals: int = 8
    #: Require a rule to beat the target's base rate by this margin. A rule that
    #: merely restates how common the outcome is has explained nothing.
    min_lift: float = 0.10


class ScriptedAgent:
    """Proposes facts and rules by fixed procedure."""

    def __init__(
        self,
        pack: DomainPack,
        config: MiningConfig | None = None,
        agent_id: str = "scripted-agent-v1",
    ) -> None:
        self.pack = pack
        self.config = config or MiningConfig()
        self._agent_id = agent_id
        #: Clauses the last extraction could not use, as (text, reason). Read by
        #: the review UI so a malformed policy annotation is visible rather than
        #: silently absent from the rule base.
        self.skipped: list[tuple[str, str]] = []

    @property
    def agent_id(self) -> str:
        return self._agent_id

    # -- facts -------------------------------------------------------------
    def extract_facts(self, record: dict[str, Any], entity: str) -> FactStore:
        """Delegates to the pack's declared mappings.

        The agent has no discretion here, which is the point: extraction is
        auditable because it is a table, not a judgement.
        """
        return self.pack.facts_from(record, entity)

    # -- mining ------------------------------------------------------------
    def mine_rules(
        self,
        cases: list[LabelledCase],
        target: str,
        existing: set[str] | None = None,
    ) -> list[RuleProposal]:
        """Induce candidate rules for `target` from labelled cases.

        Returns proposals, not rules. Nothing here enters the rule base; every
        candidate carries its support and precision so a human can judge whether
        a correlation is also a reason.

        Implemented over bitsets. Each literal is compiled once into an integer
        whose bits mark the cases where it holds, so scoring a conjunction is a
        couple of AND operations and a popcount rather than a scan over every
        case. The naive version took ~38 seconds on a thousand cases, which is
        long enough that a reviewer clicking "mine" assumes it has hung.

        Cases where a premise is UNKNOWN are simply absent from that literal's
        bitset, so they fall out of both support and hits. That is deliberate:
        counting them as failures would penalise a rule for the data being
        incompletely recorded, and bias the search toward rules over
        well-recorded fields regardless of whether those are the right fields.
        """
        existing = existing or set()
        if not cases:
            return []

        literals = self._usable_literals(cases)
        if not literals:
            return []

        positives = sum(1 for c in cases if c.label == target)
        if positives == 0:
            return []
        base_rate = positives / len(cases)

        bits = self._compile_bitsets(cases, literals)
        positive_bits = 0
        for index, case in enumerate(cases):
            if case.label == target:
                positive_bits |= 1 << index

        scored: list[_Candidate] = []
        # Level-wise, Apriori-style. Support is anti-monotone -- adding a literal
        # can only shrink it -- so a combination already below the threshold can
        # never be rescued by extending it, and its supersets need never be
        # built.
        frontier: list[tuple[tuple[tuple[str, bool], ...], int]] = [
            ((literal,), bits[literal]) for literal in literals
        ]
        for _ in range(self.config.max_body):
            survivors: list[tuple[tuple[tuple[str, bool], ...], int]] = []
            for combo, mask in frontier:
                support = mask.bit_count()
                if support < self.config.min_support:
                    continue
                survivors.append((combo, mask))

                precision = (mask & positive_bits).bit_count() / support
                if precision < self.config.min_precision:
                    continue
                if precision - base_rate < self.config.min_lift:
                    # No better than knowing how common the outcome is.
                    continue
                scored.append(
                    _Candidate(
                        literals=combo,
                        support=support,
                        precision=precision,
                        examples=self._sample(mask & positive_bits, cases),
                        counterexamples=self._sample(mask & ~positive_bits, cases),
                    )
                )

            frontier = []
            for combo, mask in survivors:
                last = literals.index(combo[-1])
                for literal in literals[last + 1 :]:
                    # A body naming the same predicate twice is redundant, and one
                    # containing both `p` and `not p` is unsatisfiable.
                    if any(name == literal[0] for name, _ in combo):
                        continue
                    frontier.append((combo + (literal,), mask & bits[literal]))
            if not frontier:
                break

        scored.sort(key=lambda c: (-c.precision, -c.support, len(c.literals)))
        kept = self._prune_subsumed(scored)

        proposals: list[RuleProposal] = []
        for candidate in kept:
            try:
                rule = self._build_rule(candidate.literals, target)
            except ValueError:
                continue  # unsafe (all-negative) body
            if rule.canonical_key() in existing:
                continue
            proposals.append(
                RuleProposal(
                    rule=rule,
                    origin=RuleOrigin.MINED,
                    strength=round(candidate.precision, 3),
                    rationale=(
                        f"fires on {candidate.support} of {len(cases)} cases and is "
                        f"correct on {candidate.precision:.0%} of those, against a "
                        f"base rate of {base_rate:.0%}"
                    ),
                    proposed_by=self._agent_id,
                    support=candidate.support,
                    precision=round(candidate.precision, 3),
                    examples=candidate.examples,
                    counterexamples=candidate.counterexamples,
                )
            )
            if len(proposals) >= self.config.max_proposals:
                break
        return proposals

    def _usable_literals(self, cases: list[LabelledCase]) -> list[tuple[str, bool]]:
        """Literals worth searching: each varying observable, positive and negated.

        Negated literals are included because real eligibility logic is full of
        them -- "provided they do not hold savings above the limit" is a
        condition, not an afterthought. A miner restricted to positive
        conjunctions cannot express such a rule at all, so it approximates it
        with whatever correlates, and hands the reviewer a rule that is wrong in
        a way the evidence cannot reveal.

        A predicate true everywhere or nowhere carries no information and is
        dropped; sorting keeps the search deterministic.
        """
        counts: dict[str, int] = {}
        for case in cases:
            for record in case.facts:
                if record.truth is Truth.TRUE:
                    counts[record.atom.predicate] = counts.get(record.atom.predicate, 0) + 1
        usable = [
            name
            for name in self.pack.observable_predicates()
            if 0 < counts.get(name, 0) < len(cases)
        ]
        return [(name, negated) for name in sorted(usable) for negated in (False, True)]

    def _compile_bitsets(
        self, cases: list[LabelledCase], literals: list[tuple[str, bool]]
    ) -> dict[tuple[str, bool], int]:
        """One pass over the data: for each literal, the cases where it holds.

        An UNKNOWN value sets no bit for either polarity, which is how unknowns
        end up excluded from support rather than counted against the rule.
        """
        masks = {literal: 0 for literal in literals}
        names = {name for name, _ in literals}
        for index, case in enumerate(cases):
            bit = 1 << index
            for name in names:
                truth = case.facts.truth_of(_atom_for(name, case.entity))
                if truth is Truth.TRUE and (name, False) in masks:
                    masks[(name, False)] |= bit
                elif truth is Truth.FALSE and (name, True) in masks:
                    masks[(name, True)] |= bit
        return masks

    @staticmethod
    def _sample(mask: int, cases: list[LabelledCase], limit: int = 5) -> tuple[str, ...]:
        """A few entities from a bitset, so a reviewer can spot-check."""
        out: list[str] = []
        index = 0
        while mask and len(out) < limit:
            if mask & 1:
                out.append(cases[index].entity)
            mask >>= 1
            index += 1
        return tuple(out)

    def _prune_subsumed(self, scored: list["_Candidate"]) -> list["_Candidate"]:
        """Drop longer rules that barely improve on a shorter one already kept.

        Keeps the queue readable. A reviewer offered `high_risk <- overdrawn`
        alongside `high_risk <- overdrawn, thin_file` and
        `high_risk <- overdrawn, thin_file, young` is being asked to do
        model selection by hand, which is not what their judgement is for.
        """
        kept: list[_Candidate] = []
        for candidate in scored:
            subsumer = next(
                (
                    k for k in kept
                    if set(k.literals).issubset(set(candidate.literals))
                ),
                None,
            )
            if subsumer is not None:
                gain = candidate.precision - subsumer.precision
                if gain < self.config.min_gain_over_subsumer:
                    continue
            kept.append(candidate)
        return kept

    def _build_rule(self, literals: Iterable[tuple[str, bool]], target: str) -> Rule:
        var = Var("A")
        body = tuple(
            Literal(Atom(name, (var,)), negated=negated) for name, negated in literals
        )
        rule = Rule(Atom(target, (var,)), body)
        if all(lit.negated for lit in body):
            # Safety: a head variable needs a positive binding occurrence. An
            # all-negative body has none, so the rule would range over nothing.
            raise ValueError("all-negative body")
        return rule

    # -- documents ---------------------------------------------------------
    #: The scripted agent reads machine-readable annotations, not prose. A
    #: policy document here carries an explicit `<!-- rule: ... -->` comment
    #: beside the clause it formalises. That is genuine extraction with genuine
    #: provenance -- the citation points at a real section a reviewer can check
    #: -- but it is not natural-language understanding, and pretending otherwise
    #: would overstate what this backend does. Reading untagged prose is the
    #: LLM backend's job.
    _RULE_COMMENT = re.compile(
        r"<!--\s*rule:\s*(?P<rule>[^>]+?)\s*(?:\|\s*p\s*=\s*(?P<p>[0-9.]+))?\s*-->"
    )
    _HEADING = re.compile(r"^#{1,6}\s*(?P<heading>.+?)\s*$", re.MULTILINE)

    def extract_from_document(
        self, name: str, text: str, vocabulary: Vocabulary
    ) -> list[RuleProposal]:
        """Draft rules from an annotated policy document, citing their clause.

        Clauses that fail to parse are recorded in `self.skipped` rather than
        discarded quietly, so a policy author can be told which annotation was
        malformed. The commonest cause is a body of only negated literals, which
        binds no variable and therefore describes no one.
        """
        self.skipped = []
        proposals: list[RuleProposal] = []
        headings = [(m.start(), m.group("heading")) for m in self._HEADING.finditer(text)]

        for match in self._RULE_COMMENT.finditer(text):
            rule_text = match.group("rule").strip()
            if not rule_text.endswith("."):
                rule_text += "."
            try:
                rule = parse_rule(rule_text)
            except (ParseError, ValueError) as exc:
                # A malformed annotation is a defect in the document, not in the
                # rule base, so one bad clause must not block extraction of the
                # rest. But skipping in silence is worse: a policy author sees
                # seven rules appear from eight clauses and has no way to learn
                # which one failed or why. The skip is recorded and returned.
                self.skipped.append((rule_text, str(exc).splitlines()[0]))
                continue

            clause = "preamble"
            for position, heading in headings:
                if position < match.start():
                    clause = heading
                else:
                    break

            strength = float(match.group("p")) if match.group("p") else 0.9
            proposals.append(
                RuleProposal(
                    rule=rule,
                    origin=RuleOrigin.DOCUMENT,
                    strength=strength,
                    rationale=f"encodes a clause of {name}",
                    proposed_by=self._agent_id,
                    source_citation=f"{name}, {clause}",
                )
            )
        return proposals


@dataclass(frozen=True)
class _Candidate:
    literals: tuple[tuple[str, bool], ...]
    support: int
    precision: float
    examples: tuple[str, ...]
    counterexamples: tuple[str, ...]


def _atom_for(predicate: str, entity: str) -> Atom:
    from ..syntax import Const

    return Atom(predicate, (Const(entity),))


__all__ = ["MiningConfig", "ScriptedAgent"]
