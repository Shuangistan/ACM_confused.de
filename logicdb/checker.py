"""Properties of the rule base as a whole, checked by exhausting its inputs.

Safety, stratification and vocabulary are checked on every insertion, and they
are properties of a rule in isolation. Three properties are not, and they are
the ones that waste a reviewer's time or embarrass the system in front of an
applicant:

* **Contradiction** -- some input derives two competing outcomes at once.
* **Redundancy** -- a rule whose removal changes no outcome on any input. It
  still costs a review and still appears in proofs.
* **Dead rules** -- approved, but no input can make the body fire. Usually a
  sign the rule was written against a misunderstanding of the vocabulary.

Each is an existential question over all inputs, which is the shape a SAT solver
is built for. We enumerate instead, for a reason that is not laziness: every
predicate in the shipped packs has arity one, so grounding on a single entity
leaves a propositional problem over ten to nineteen observables, and running the
real engine over every assignment has a property no encoding can match -- it
cannot disagree with production behaviour, because it *is* production
behaviour. There is no translation to get wrong, and getting it wrong is easy:
a solver reads `h <- b` as material implication and admits models where `h`
holds for no reason, while the stratified semantics takes the unique minimal
model.

Enumeration dies exponentially all the same. Past roughly twenty-five
observables, or any predicate of arity two, this needs a solver -- and when that
happens the enumerator should be kept as the oracle the solver is validated
against, exactly as it is for the probability bounds.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field

from .engine import Engine
from .facts import FactStore, Truth
from .program import RuleBase, RuleRecord, RuleStatus
from .syntax import Atom, Const, Vocabulary

#: Above this, enumeration is refused rather than run for hours. Reported, never
#: silently skipped: a checker that quietly gives up reads exactly like a
#: checker that found nothing.
MAX_INPUTS = 1 << 20


@dataclass
class Contradiction:
    """An input under which two competing outcomes both hold."""

    outcomes: tuple[str, ...]
    true_observables: tuple[str, ...]
    rules_used: tuple[str, ...]

    def describe(self) -> str:
        given = ", ".join(self.true_observables) or "nothing"
        return (
            f"{' and '.join(self.outcomes)} both hold when {given} are true "
            f"(via {', '.join(self.rules_used)})"
        )


@dataclass
class CheckReport:
    entity: str
    inputs_examined: int
    observables: tuple[str, ...]
    contradictions: list[Contradiction] = field(default_factory=list)
    dead_rules: list[str] = field(default_factory=list)
    redundant_rules: list[tuple[str, str]] = field(default_factory=list)
    #: Set when the input space was too large to exhaust. The report is then a
    #: statement about what was checked, not about the rule base.
    skipped: str | None = None

    @property
    def clean(self) -> bool:
        return not (self.contradictions or self.dead_rules or self.redundant_rules)

    def describe(self) -> str:
        if self.skipped:
            return f"not checked: {self.skipped}"
        lines = [
            f"exhausted {self.inputs_examined:,} inputs over "
            f"{len(self.observables)} observables"
        ]
        if self.clean:
            lines.append("  no contradictions, no dead rules, no redundant rules")
            return "\n".join(lines)
        for c in self.contradictions[:5]:
            lines.append(f"  CONTRADICTION: {c.describe()}")
        if len(self.contradictions) > 5:
            lines.append(f"  ... and {len(self.contradictions) - 5} more")
        for rule_id in self.dead_rules:
            lines.append(f"  DEAD: {rule_id} can never fire")
        for rule_id, why in self.redundant_rules:
            lines.append(f"  REDUNDANT: {rule_id} -- {why}")
        return "\n".join(lines)


class Checker:
    """Exhausts a rule base's inputs and reports what only that can reveal."""

    def __init__(
        self,
        vocabulary: Vocabulary,
        outcomes: list[str],
        entity: str = "_probe",
        max_inputs: int = MAX_INPUTS,
    ) -> None:
        self.vocabulary = vocabulary
        self.outcomes = outcomes
        self.entity = entity
        self.max_inputs = max_inputs
        self.engine = Engine()
        self.observables = tuple(
            d.name for d in vocabulary.observables() if d.arity == 1
        )

    def _facts_for(self, assignment: tuple[bool, ...]) -> FactStore:
        store = FactStore()
        for name, value in zip(self.observables, assignment):
            store.assert_fact(
                Atom(name, (Const(self.entity),)),
                Truth.TRUE if value else Truth.FALSE,
                source="checker",
            )
        return store

    def check(self, rulebase: RuleBase, deep: bool = True) -> CheckReport:
        """Run every check. `deep=False` skips redundancy, which costs N times more."""
        k = len(self.observables)
        total = 1 << k
        report = CheckReport(
            entity=self.entity, inputs_examined=0, observables=self.observables
        )
        if total > self.max_inputs:
            report.skipped = (
                f"2^{k} = {total:,} inputs exceeds the {self.max_inputs:,} cap; "
                f"this rule base needs a solver, not enumeration"
            )
            return report

        outcome_atoms = [Atom(o, (Const(self.entity),)) for o in self.outcomes]
        approved = [
            r for r in rulebase
            if r.status in (RuleStatus.APPROVED, RuleStatus.PROVISIONAL)
        ]
        fired: set[str] = set()
        #: outcome-set per input, so redundancy can be judged by comparison
        baseline: list[frozenset[str]] = []

        for assignment in itertools.product((False, True), repeat=k):
            facts = self._facts_for(assignment)
            ev = self.engine.evaluate(rulebase, facts)
            fired |= ev.rules_used
            holding = frozenset(a.predicate for a in outcome_atoms if ev.holds(a))
            baseline.append(holding)
            if len(holding) > 1:
                report.contradictions.append(
                    Contradiction(
                        outcomes=tuple(sorted(holding)),
                        true_observables=tuple(
                            n for n, v in zip(self.observables, assignment) if v
                        ),
                        rules_used=tuple(sorted(ev.rules_used)),
                    )
                )
        report.inputs_examined = total
        report.dead_rules = sorted(r.rule_id for r in approved if r.rule_id not in fired)

        if deep:
            # A dead rule is trivially redundant. Reporting it twice tells the
            # reviewer nothing and makes the report look worse than it is.
            live = [r for r in approved if r.rule_id not in report.dead_rules]
            report.redundant_rules = self._redundant(rulebase, live, baseline)
        return report

    def _redundant(
        self,
        rulebase: RuleBase,
        approved: list[RuleRecord],
        baseline: list[frozenset[str]],
    ) -> list[tuple[str, str]]:
        """Rules whose removal changes no outcome on any input.

        Defined by behaviour rather than by syntactic subsumption, because
        behaviour is what a reviewer is being asked about. A rule that changes
        nothing still costs a review and still clutters every proof it appears
        in.
        """
        out: list[tuple[str, str]] = []
        k = len(self.observables)
        for record in approved:
            reduced = RuleBase(self.vocabulary)
            for other in rulebase:
                if other.rule_id != record.rule_id:
                    reduced.add(other)
            changed = False
            for i, assignment in enumerate(itertools.product((False, True), repeat=k)):
                ev = self.engine.evaluate(reduced, self._facts_for(assignment))
                holding = frozenset(
                    o for o in self.outcomes
                    if ev.holds(Atom(o, (Const(self.entity),)))
                )
                if holding != baseline[i]:
                    changed = True
                    break
            if not changed:
                out.append(
                    (record.rule_id, "removing it changes no outcome on any input")
                )
        return out


__all__ = ["Checker", "CheckReport", "Contradiction", "MAX_INPUTS"]
