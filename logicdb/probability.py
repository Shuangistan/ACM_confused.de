"""Probability over proofs: noisy-OR, bounds under missing facts, and what to ask.

Three things happen here, and the third is the one that matters most in practice.

**1. Combining evidence (noisy-OR).** An atom supported by several rules is more
credible than one supported by a single rule. Each support is treated as an
independent potential cause:

    P(h) = 1 - Π_r (1 - p_r · P(body_r))

The independence assumption is a real modelling commitment and frequently a
wrong one -- two mined rules resting on correlated attributes are not
independent evidence. So it is stated in the explanation rather than buried in
the arithmetic, and `Answer.assumptions` carries it to the surface.

**2. Bounds instead of a point estimate.** When a fact is unknown, the honest
answer is not a number but an interval. Everything is computed twice: once with
the unknowns pushed toward the lowest answer, once toward the highest. A narrow
interval means the missing facts do not matter here. A wide one means the system
is guessing, and says so.

The naive way to get exact bounds is to evaluate all 2^k assignments of the k
unknowns. That is avoided where it can be: for an unknown appearing only
positively, P is monotone increasing in it, so its minimising value is FALSE and
its maximising value is TRUE -- no search needed. Only unknowns appearing in
both polarities are enumerated, and there are usually very few. The tests check
this fast path against exhaustive enumeration, because a shortcut that disagrees
with the definition is worse than no shortcut.

**3. What to ask.** Given bounds, the useful output is not the interval but the
next question. For each unknown the answer is recomputed with it resolved both
ways, and the unknowns are ranked by whether resolving them would actually
change the decision. This is what turns "critical information may be missing"
from a caveat into "ask about the guarantor, and don't bother with the other
three".
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field

from .engine import Engine, Evaluation
from .facts import FactStore, Truth
from .program import RuleBase, RuleRecord
from .syntax import Atom, Literal, Rule

#: Above this many ambiguous unknowns, exhaustive enumeration is refused and the
#: bounds are reported as approximate. Better to say so than to hang.
MAX_ENUMERATION_BITS = 14


# --------------------------------------------------------------------------
# Core combination
# --------------------------------------------------------------------------


def noisy_or(contributions: list[float]) -> float:
    """Combine independent supports: 1 - Π (1 - c)."""
    remaining = 1.0
    for c in contributions:
        remaining *= 1.0 - max(0.0, min(1.0, c))
    return 1.0 - remaining


class ProbabilityComputer:
    """Assigns a probability to every atom in an evaluation.

    Recursion is handled by refusing circular support: while computing P(h), any
    path that arrives back at `h` contributes nothing. Without this, a recursive
    rule base could inflate its own confidence by citing itself, which is both
    wrong and exactly the kind of failure that is invisible in the output.
    """

    def __init__(self, evaluation: Evaluation) -> None:
        self.evaluation = evaluation
        self._memo: dict[Atom, float] = {}

    def probability_of(self, atom: Atom) -> float:
        return self._resolve(atom, set())

    def _resolve(self, atom: Atom, visiting: set[Atom]) -> float:
        if atom in self._memo:
            return self._memo[atom]
        if atom in visiting:
            return 0.0  # circular support contributes nothing

        node = self.evaluation.derived.get(atom)
        if node is None:
            truth = self.evaluation.facts.truth_of(atom)
            record = self.evaluation.facts.record_of(atom)
            if truth is Truth.TRUE:
                value = record.confidence if record else 1.0
            else:
                # FALSE and UNKNOWN both give 0 here. Unknowns are not handled by
                # smearing a probability over them -- they are handled by the
                # bounds machinery below, which pins them and re-evaluates.
                value = 0.0
            self._memo[atom] = value
            return value

        visiting.add(atom)
        contributions: list[float] = []
        for support in node.supports:
            body = 1.0
            for premise, negated in support.premises:
                p = self._resolve(premise, visiting)
                body *= (1.0 - p) if negated else p
                if body == 0.0:
                    break
            contributions.append(support.strength * body)
        visiting.discard(atom)

        value = noisy_or(contributions)
        self._memo[atom] = value
        return value


# --------------------------------------------------------------------------
# Polarity analysis -- how an unknown can move the answer
# --------------------------------------------------------------------------


def predicate_polarities(
    rules: list[RuleRecord], goal_signature: str
) -> dict[str, set[int]]:
    """Signs under which each predicate can influence `goal_signature`.

    `{+1}` means raising it can only raise the answer, `{-1}` only lower it,
    `{+1, -1}` means it can do either depending on the case and must be searched
    rather than reasoned about. Computed from the rule base rather than from one
    evaluation, because the derivation itself changes as unknowns are pinned --
    a polarity derived from a single evaluation would be valid only for that
    evaluation.
    """
    incoming: dict[str, list[tuple[str, int]]] = {}
    for record in rules:
        head = record.head_signature
        for lit in record.rule.body:
            incoming.setdefault(head, []).append((lit.signature, -1 if lit.negated else 1))

    polarities: dict[str, set[int]] = {goal_signature: {1}}
    frontier = [(goal_signature, 1)]
    seen: set[tuple[str, int]] = {(goal_signature, 1)}

    while frontier:
        signature, sign = frontier.pop()
        for body_sig, edge_sign in incoming.get(signature, ()):
            new_sign = sign * edge_sign
            state = (body_sig, new_sign)
            polarities.setdefault(body_sig, set()).add(new_sign)
            if state not in seen:
                seen.add(state)
                frontier.append(state)

    return polarities


# --------------------------------------------------------------------------
# Answers
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Bounds:
    """A probability interval. `width` is the price of the missing facts."""

    lower: float
    upper: float
    exact: bool = True

    @property
    def width(self) -> float:
        return self.upper - self.lower

    @property
    def point(self) -> float:
        return (self.lower + self.upper) / 2.0

    def __str__(self) -> str:
        if self.width < 1e-9:
            return f"{self.lower:.2f}"
        marker = "" if self.exact else " (approx)"
        return f"[{self.lower:.2f}, {self.upper:.2f}]{marker}"


@dataclass(frozen=True)
class AskCandidate:
    """An unknown fact, and what resolving it would do."""

    atom: Atom
    if_true: Bounds
    if_false: Bounds
    #: True when resolving this fact alone moves the answer across the decision
    #: threshold. The only unknowns genuinely worth a human's time.
    decisive: bool
    #: How much the interval narrows, averaged over the two resolutions.
    information_gain: float

    def __str__(self) -> str:
        mark = "DECISIVE" if self.decisive else f"narrows by {self.information_gain:.2f}"
        return f"{self.atom}: true -> {self.if_true}, false -> {self.if_false}  [{mark}]"


@dataclass
class Answer:
    """The system's response to one query."""

    goal: Atom
    bounds: Bounds
    evaluation: Evaluation
    #: Competing outcomes (e.g. approve vs decline), each with its own bounds.
    #: Deliberately not normalised -- see `is_uncovered`.
    outcomes: dict[str, Bounds] = field(default_factory=dict)
    ask: list[AskCandidate] = field(default_factory=list)
    assumptions: list[str] = field(default_factory=list)
    unknowns_considered: tuple[Atom, ...] = ()


    @property
    def is_uncovered(self) -> bool:
        """True when no outcome has meaningful support.

        This is why outcome probabilities are never normalised. Normalising a
        pair of near-zero numbers would turn "the rule base has nothing to say
        about this case" into a confident-looking 50/50. Keeping them raw makes
        ignorance visible, and this flag is the trigger for asking the agent to
        propose a rule.
        """
        if not self.outcomes:
            return self.bounds.upper < 0.2
        return all(b.upper < 0.2 for b in self.outcomes.values())

    def best_question(self) -> AskCandidate | None:
        return self.ask[0] if self.ask else None


# --------------------------------------------------------------------------
# The solver
# --------------------------------------------------------------------------


class ProbabilisticSolver:
    """Answers queries with bounds, and says what would narrow them."""

    def __init__(
        self,
        rulebase: RuleBase,
        engine: Engine | None = None,
        threshold: float = 0.5,
    ) -> None:
        self.rulebase = rulebase
        self.engine = engine or Engine()
        #: Probability above which an outcome counts as the decision. Used only
        #: to decide which unknowns are decisive, never to hide the interval.
        self.threshold = threshold

    # -- single evaluation -------------------------------------------------
    def _probability_under(
        self,
        goal: Atom,
        facts: FactStore,
        critical: bool,
        assignment: dict[Atom, Truth],
        rules: list[RuleRecord],
    ) -> float:
        store = facts.with_assignment(assignment) if assignment else facts
        evaluation = self.engine.evaluate(self.rulebase, store, critical, rules=rules)
        return ProbabilityComputer(evaluation).probability_of(goal)

    # -- bounds ------------------------------------------------------------
    def bounds_for(
        self,
        goal: Atom,
        facts: FactStore,
        critical: bool = False,
        unknowns: list[Atom] | None = None,
        pinned: dict[Atom, Truth] | None = None,
    ) -> Bounds:
        """Exact bounds on P(goal), given what is unknown.

        Unknowns that can only push the answer one way are set directly. Only
        those that can push both ways are searched, and the search is over that
        subset alone.
        """
        rules = self.rulebase.active_for(critical)
        pinned = dict(pinned or {})
        unknown_atoms = [
            a for a in (unknowns if unknowns is not None else facts.unknowns())
            if a not in pinned
        ]

        if not unknown_atoms:
            p = self._probability_under(goal, facts, critical, pinned, rules)
            return Bounds(p, p, exact=True)

        polarities = predicate_polarities(rules, goal.signature)

        monotone_min: dict[Atom, Truth] = {}
        monotone_max: dict[Atom, Truth] = {}
        ambiguous: list[Atom] = []

        for atom in unknown_atoms:
            signs = polarities.get(atom.signature, set())
            if signs == {1}:
                monotone_min[atom] = Truth.FALSE
                monotone_max[atom] = Truth.TRUE
            elif signs == {-1}:
                monotone_min[atom] = Truth.TRUE
                monotone_max[atom] = Truth.FALSE
            elif not signs:
                # Cannot reach the goal at all; its value is irrelevant. Pinned
                # to FALSE purely so the evaluation is deterministic.
                monotone_min[atom] = Truth.FALSE
                monotone_max[atom] = Truth.FALSE
            else:
                ambiguous.append(atom)

        exact = True
        if len(ambiguous) > MAX_ENUMERATION_BITS:
            exact = False
            # Keep the ones most likely to matter -- those reachable at all --
            # and pin the rest. The result is reported as approximate.
            ambiguous = ambiguous[:MAX_ENUMERATION_BITS]
            for atom in unknown_atoms:
                if atom not in monotone_min and atom not in ambiguous:
                    monotone_min[atom] = Truth.FALSE
                    monotone_max[atom] = Truth.FALSE

        lowest, highest = 1.0, 0.0
        for combo in itertools.product(
            (Truth.FALSE, Truth.TRUE), repeat=len(ambiguous)
        ):
            extra = dict(zip(ambiguous, combo))
            low = self._probability_under(
                goal, facts, critical, {**pinned, **monotone_min, **extra}, rules
            )
            high = self._probability_under(
                goal, facts, critical, {**pinned, **monotone_max, **extra}, rules
            )
            lowest = min(lowest, low, high)
            highest = max(highest, low, high)

        return Bounds(lowest, highest, exact=exact)

    def bounds_brute_force(
        self,
        goal: Atom,
        facts: FactStore,
        critical: bool = False,
        unknowns: list[Atom] | None = None,
    ) -> Bounds:
        """Bounds by evaluating every assignment. The definition, not the method.

        Exponential and therefore unusable in production, but it is what
        `bounds_for` is tested against. A fast path nobody checks against the
        definition is a guess.
        """
        rules = self.rulebase.active_for(critical)
        unknown_atoms = list(
            unknowns if unknowns is not None else facts.unknowns()
        )
        if not unknown_atoms:
            p = self._probability_under(goal, facts, critical, {}, rules)
            return Bounds(p, p)

        lowest, highest = 1.0, 0.0
        for combo in itertools.product(
            (Truth.FALSE, Truth.TRUE), repeat=len(unknown_atoms)
        ):
            assignment = dict(zip(unknown_atoms, combo))
            p = self._probability_under(goal, facts, critical, assignment, rules)
            lowest = min(lowest, p)
            highest = max(highest, p)
        return Bounds(lowest, highest)

    # -- what to ask -------------------------------------------------------
    def rank_questions(
        self,
        goal: Atom,
        facts: FactStore,
        critical: bool = False,
        unknowns: list[Atom] | None = None,
        limit: int = 5,
    ) -> list[AskCandidate]:
        """Rank unknown facts by whether resolving them would change anything.

        Decisive unknowns -- those that move the answer across the threshold --
        come first, because those are the only ones where an answer changes what
        happens. Everything else is ranked by how much it narrows the interval.
        An unknown that changes nothing is not offered at all: a system that asks
        for information it will not use trains people to stop answering.
        """
        unknown_atoms = list(unknowns if unknowns is not None else facts.unknowns())
        if not unknown_atoms:
            return []

        candidates: list[AskCandidate] = []
        for atom in unknown_atoms:
            others = [a for a in unknown_atoms if a != atom]
            if_true = self.bounds_for(
                goal, facts, critical, unknowns=others, pinned={atom: Truth.TRUE}
            )
            if_false = self.bounds_for(
                goal, facts, critical, unknowns=others, pinned={atom: Truth.FALSE}
            )

            decisive = (if_true.lower > self.threshold) != (if_false.lower > self.threshold) or (
                if_true.upper > self.threshold
            ) != (if_false.upper > self.threshold)

            base_width = self.bounds_for(goal, facts, critical, unknowns=unknown_atoms).width
            gain = base_width - (if_true.width + if_false.width) / 2.0

            if decisive or gain > 1e-6:
                candidates.append(
                    AskCandidate(
                        atom=atom,
                        if_true=if_true,
                        if_false=if_false,
                        decisive=decisive,
                        information_gain=max(0.0, gain),
                    )
                )

        candidates.sort(key=lambda c: (not c.decisive, -c.information_gain, str(c.atom)))
        return candidates[:limit]

    # -- the full answer ---------------------------------------------------
    def answer(
        self,
        goal: Atom,
        facts: FactStore,
        critical: bool = False,
        competing: list[Atom] | None = None,
    ) -> Answer:
        """Answer a query: bounds, competing outcomes, and the next question."""
        unknown_atoms = facts.unknowns()
        bounds = self.bounds_for(goal, facts, critical, unknowns=unknown_atoms)

        evaluation = self.engine.evaluate(
            self.rulebase,
            facts.with_assignment({a: Truth.FALSE for a in unknown_atoms}),
            critical,
        )

        outcomes: dict[str, Bounds] = {str(goal): bounds}
        for other in competing or ():
            if other != goal:
                outcomes[str(other)] = self.bounds_for(
                    other, facts, critical, unknowns=unknown_atoms
                )

        ask = self.rank_questions(goal, facts, critical, unknowns=unknown_atoms)

        assumptions: list[str] = []
        multi = [a for a in evaluation.derived.values() if len(a.supports) > 1]
        if multi:
            assumptions.append(
                f"{len(multi)} conclusion(s) combine several supporting rules by "
                f"noisy-OR, which treats those rules as independent evidence. "
                f"Rules resting on correlated attributes would overstate the "
                f"result."
            )
        if unknown_atoms:
            assumptions.append(
                f"{len(unknown_atoms)} fact(s) are unknown; the interval spans "
                f"every way they could resolve rather than assuming a value."
            )
        if not bounds.exact:
            assumptions.append(
                "Too many interacting unknowns to enumerate exactly; the interval "
                "is an approximation and may be narrower than the truth."
            )

        return Answer(
            goal=goal,
            bounds=bounds,
            evaluation=evaluation,
            outcomes=outcomes,
            ask=ask,
            assumptions=assumptions,
            unknowns_considered=tuple(unknown_atoms),
        )


__all__ = [
    "Answer",
    "AskCandidate",
    "Bounds",
    "MAX_ENUMERATION_BITS",
    "ProbabilisticSolver",
    "ProbabilityComputer",
    "noisy_or",
    "predicate_polarities",
]
