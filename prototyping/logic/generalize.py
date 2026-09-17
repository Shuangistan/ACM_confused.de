"""Merging near-duplicate rules into one that covers both.

Without this the database is a cache. Each case shape produces its own clause,
the next case differs somewhere, nothing matches, and a model is asked again —
so coverage creeps instead of climbing, however deterministic the engine is.

Two mechanisms, both mechanical:

**Dropping literals.** Two approved clauses concluding the same thing, alike
except for a condition each, generalise to their shared core:

    deny(c) <- arrears(c), disputed(c).
    deny(c) <- arrears(c), no_notice(c).
    ------------------------------------
    deny(c) <- arrears(c).

**Variabilising constants.** Two clauses differing only in a value generalise
by replacing it with a variable — Plotkin's least general generalisation,
restricted to the case where the clauses are otherwise identical:

    reorder(c, toner) <- low(c, toner).
    reorder(c, paper) <- low(c, paper).
    -----------------------------------
    reorder(c, X) <- low(c, X).

What this cannot do is worth stating plainly, because it is the failure this
system actually exhibits. `reorder_toner` and `reorder_paper` are different
*predicates*, and no anti-unification merges those — the thing that would have
to change is the extractor putting the entity in an argument rather than in the
predicate's name. Generalisation can widen a rule; it cannot repair a
vocabulary.

Nothing here is applied automatically. A generalisation always fires on strictly
more cases than its sources, so it always risks deciding something nobody
intended. Each candidate is replayed over the stored cases and handed to a
person with the consequences attached.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations

from .engine import Engine
from .facts import FactStore, Truth
from .parse import ParseError, parse_atom
from .syntax import Atom, Const, Literal, Rule, Term, UnsafeRule, Var


@dataclass
class Candidate:
    """A proposed generalisation, and what it would have done."""

    rule: Rule
    sources: tuple[str, ...]
    how: str                      # "dropped-literals" | "variabilised"
    #: Cases the sources already decided, which this still decides.
    keeps: list[str] = field(default_factory=list)
    #: Cases nothing decided before and this now would. The gain, and the risk.
    gains: list[str] = field(default_factory=list)
    #: Cases where it fires and the case recorded a different conclusion.
    conflicts: list[str] = field(default_factory=list)
    #: Conditions removed, and how many the narrowest source had. A clause cut
    #: from three conditions to one is a different proposition, not a tidy-up.
    dropped: int = 0
    narrowest_source: int = 0
    #: How many cases the replay could actually run on. "No conflicts" over
    #: four cases is not the same claim as over four hundred, and a reviewer
    #: shown only the first number will read it as the second.
    evidence: int = 0

    @property
    def reach(self) -> float:
        """Share of the narrowest source's conditions this drops."""
        if not self.narrowest_source:
            return 0.0
        return self.dropped / self.narrowest_source

    @property
    def safe(self) -> bool:
        """Nothing recorded contradicts it — which is not the same as right."""
        return not self.conflicts

    def caution(self) -> str:
        """What a reviewer should be told before they read the numbers."""
        if self.conflicts:
            return (f"contradicts {len(self.conflicts)} recorded decision(s) — "
                    f"do not approve without reading them")
        if self.reach >= 0.5:
            return (f"drops {self.dropped} of {self.narrowest_source} "
                    f"conditions; this is a materially wider claim, not a merge")
        if self.evidence < 10:
            return (f"only {self.evidence} case(s) to test against — no "
                    f"conflicts here means little")
        return f"no conflicts across {self.evidence} cases"

    def summary(self) -> str:
        return (f"replaces {len(self.sources)} rules; keeps {len(self.keeps)} "
                f"decisions, newly decides {len(self.gains)}, "
                f"conflicts with {len(self.conflicts)} of {self.evidence}")


# --------------------------------------------------------------------------
# Forming candidates
# --------------------------------------------------------------------------
def _body_key(lit: Literal) -> str:
    return str(lit)


def _drop_literals(a: Rule, b: Rule) -> Rule | None:
    """The shared core of two clauses with the same head."""
    if str(a.head) != str(b.head):
        return None
    shared = [l for l in a.body if _body_key(l) in {_body_key(x) for x in b.body}]
    if not shared or len(shared) == len(a.body) == len(b.body):
        return None                      # identical, or nothing in common
    try:
        return Rule(a.head, tuple(shared))
    except (UnsafeRule, ValueError):
        return None                      # dropping left a variable unbound


def _anti_unify_terms(x: Term, y: Term, mapping: dict, counter: list) -> Term:
    if x == y:
        return x
    key = (str(x), str(y))
    if key not in mapping:
        mapping[key] = Var(f"G{counter[0]}")
        counter[0] += 1
    return mapping[key]


def _variabilise(a: Rule, b: Rule) -> Rule | None:
    """Replace the values two otherwise-identical clauses differ in."""
    if a.head.signature != b.head.signature or len(a.body) != len(b.body):
        return None
    mapping: dict = {}
    counter = [0]

    def au(p: Atom, q: Atom) -> Atom | None:
        if p.signature != q.signature:
            return None
        return Atom(p.predicate, tuple(
            _anti_unify_terms(s, t, mapping, counter) for s, t in zip(p.terms, q.terms)))

    head = au(a.head, b.head)
    if head is None:
        return None
    body = []
    for la, lb in zip(sorted(a.body, key=str), sorted(b.body, key=str)):
        if la.negated != lb.negated:
            return None
        atom = au(la.atom, lb.atom)
        if atom is None:
            return None
        body.append(Literal(atom, la.negated))
    if not mapping:
        return None                      # nothing differed; not a generalisation
    try:
        return Rule(head, tuple(body))
    except (UnsafeRule, ValueError):
        return None


def propose(rules: list[tuple[str, Rule]]) -> list[Candidate]:
    """Every generalisation two approved rules admit, deduplicated."""
    seen: dict[str, Candidate] = {}
    for (ka, ra), (kb, rb) in combinations(rules, 2):
        for how, made in (("dropped-literals", _drop_literals(ra, rb)),
                          ("variabilised", _variabilise(ra, rb))):
            if made is None:
                continue
            key = made.canonical_key()
            if key in {r.canonical_key() for _, r in rules}:
                continue                 # already in the base
            if key in seen:
                seen[key].sources = tuple(sorted(set(seen[key].sources) | {ka, kb}))
            else:
                narrowest = min(len(ra.body), len(rb.body))
                seen[key] = Candidate(
                    made, (ka, kb), how,
                    dropped=max(0, narrowest - len(made.body)),
                    narrowest_source=narrowest)
    return list(seen.values())


# --------------------------------------------------------------------------
# Replaying it over what has actually happened
# --------------------------------------------------------------------------
def _facts_of(case: dict) -> FactStore:
    store = FactStore()
    for row in case.get("facts") or []:
        text, truth = str(row[0]), (row[1] if len(row) > 1 else "true")
        bare = text[4:].strip() if text.lower().startswith("not ") else text
        negated = truth == "false" or text.lower().startswith("not ")
        try:
            atom = parse_atom(bare)
        except (ParseError, ValueError):
            continue
        if atom.is_ground():
            store.assert_fact(atom, Truth.FALSE if negated else Truth.TRUE)
    return store


def replay(candidate: Candidate, cases: list[dict],
           base: list[tuple[str, Rule]]) -> Candidate:
    """Run the candidate over the stored cases and record what changes.

    The case log is the only evidence available, and it is evidence about what
    has been seen rather than about what is true. So this reports rather than
    judges: what the candidate keeps, what it newly decides, and where it
    contradicts a conclusion already recorded. A person decides whether the
    gain is worth the reach.
    """
    engine = Engine()
    others = [(k, r) for k, r in base if k not in candidate.sources]
    head_pred = candidate.rule.head.predicate

    for case in cases:
        facts = _facts_of(case)
        if len(facts) == 0:
            continue
        candidate.evidence += 1
        try:
            with_cand = engine.evaluate(others + [("cand", candidate.rule)], facts)
            without = engine.evaluate(base, facts)
        except Exception:  # noqa: BLE001
            continue

        fires = any(a.predicate == head_pred for a in with_cand.derived)
        fired_before = any(a.predicate == head_pred for a in without.derived)
        recorded = {str(c).split("(")[0] for c in (case.get("conclusions") or [])}

        if fires and fired_before:
            candidate.keeps.append(case["case_id"])
        elif fires and not fired_before:
            # New ground. A gain if nothing contradicts it; a conflict if the
            # case was decided the other way at the time.
            if recorded and head_pred not in recorded:
                candidate.conflicts.append(case["case_id"])
            else:
                candidate.gains.append(case["case_id"])
    return candidate


def suggestions(rules: list[tuple[str, Rule]], cases: list[dict]) -> list[Candidate]:
    """Candidates worth a person's time, most useful first."""
    out = [replay(c, cases, rules) for c in propose(rules)]
    # Conflicts first, then reach: a candidate that drops most of a body is
    # shown below one that merely merges, even if it would decide more.
    out.sort(key=lambda c: (len(c.conflicts), c.reach, -len(c.gains)))
    return out


__all__ = ["Candidate", "propose", "replay", "suggestions"]
