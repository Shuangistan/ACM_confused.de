"""The node that maintains the logic, and the mechanical screen in front of it.

The decision side answers cases. This side answers a different question: *is the
rule base still the right rule base?* Three things feed it, and each arrives on
its own schedule rather than per decision:

* **Audit verdicts.** A sampled decision comes back right or wrong. That verdict
  belongs to the rules that produced it, and moves their posteriors.
* **Gaps.** A case the base cannot answer is a request for logic.
* **Degradation.** A rule whose posterior has fallen far enough to matter comes
  back for re-review -- the one circumstance in which an approved rule is looked
  at again, justified because the evidence changed rather than because the
  reviewer is being asked to repeat themselves.

Between this node and the human sits `screen`, and its placement is the point.
Dead, redundant and contradictory rules are found by exhausting the input space,
which is mechanical, so they should never consume a reviewer. Reviewer attention
is the resource the whole design exists to spend well; spending it on a rule
that provably changes nothing is the same waste as per-decision rubber-stamping,
one level up.

What this node will not do is approve anything. It proposes, retires under
instruction, and keeps the queue honest. The gate stays where it is.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .checker import CheckReport, Checker
from .governance import Governor
from .node import Decision
from .packs import DomainPack
from .program import RuleRecord, RuleStatus


@dataclass
class ScreenResult:
    """What the mechanical screen did to a batch of proposals."""

    admitted: list[RuleRecord] = field(default_factory=list)
    #: Rejected before any human saw them, with the reason.
    filtered: list[tuple[RuleRecord, str]] = field(default_factory=list)
    report: CheckReport | None = None

    def describe(self) -> str:
        lines = [
            f"screen: {len(self.admitted)} admitted to the queue, "
            f"{len(self.filtered)} filtered mechanically"
        ]
        for record, why in self.filtered:
            lines.append(f"  filtered {record.rule_id}: {why}")
        return "\n".join(lines)


@dataclass
class MaintenanceReport:
    """The state of the rule base, not of the cycle that produced it.

    Audit tallies live on `CycleReport`: they describe what one pass did,
    whereas this describes what is now true of the logic and would still be
    true if nobody ran another case.
    """

    flagged_for_rereview: list[str] = field(default_factory=list)
    retirement_candidates: list[tuple[str, str]] = field(default_factory=list)

    def describe(self) -> str:
        lines = ["maintenance:"]
        if self.flagged_for_rereview:
            lines.append(
                f"  re-review, evidence degraded: "
                f"{', '.join(self.flagged_for_rereview)}"
            )
        if self.retirement_candidates:
            for rule_id, why in self.retirement_candidates:
                lines.append(f"  retirement candidate {rule_id}: {why}")
        if not self.flagged_for_rereview and not self.retirement_candidates:
            lines.append("  nothing needs a person's attention")
        return "\n".join(lines)


class LogicMaintainer:
    """Keeps the rule base current. Proposes and retires; never approves."""

    def __init__(
        self,
        pack: DomainPack,
        governor: Governor,
        checker: Checker | None = None,
    ) -> None:
        self.pack = pack
        self.governor = governor
        self.checker = checker or Checker(pack.vocabulary, pack.outcomes)

    # -- audit verdicts in -------------------------------------------------
    def record_audit(self, decision: Decision, correct: bool, label: str | None = None) -> bool:
        """Attach an auditor's verdict to the rules that produced the decision.

        This is the return path that makes sampling worth doing. Without it the
        sampler selects cases for review and the review changes nothing, which
        is oversight theatre with extra steps.
        """
        if decision.query_id is None:
            return False
        self.governor.record_outcome(decision.query_id, correct=correct, label=label)
        return True

    # -- the mechanical screen --------------------------------------------
    def screen(self, proposals: list[RuleRecord]) -> ScreenResult:
        """Filter proposals a machine can reject, before a person sees them.

        Each candidate is tested against the *approved* base plus itself: a rule
        that changes no outcome anywhere adds nothing, and one that can never
        fire is simply wrong. Both are mechanical facts, and a reviewer asked to
        adjudicate them has been handed a job that was never theirs.
        """
        from .program import RuleBase

        result = ScreenResult()
        approved = [
            r for r in self.governor.rulebase if r.status is RuleStatus.APPROVED
        ]
        for candidate in proposals:
            trial = RuleBase(self.pack.vocabulary)
            for record in approved:
                trial.add(record)
            trial.add(candidate)
            report = self.checker.check(trial, deep=True)
            if report.skipped:
                # Too large to exhaust. Admit rather than filter: refusing a
                # proposal because we could not check it would silently make
                # the queue depend on pack size.
                result.admitted.append(candidate)
                continue
            if candidate.rule_id in report.dead_rules:
                result.filtered.append((candidate, "can never fire"))
                continue
            redundant = {rid for rid, _ in report.redundant_rules}
            if candidate.rule_id in redundant:
                result.filtered.append(
                    (candidate, "changes no outcome on any input")
                )
                continue
            if report.contradictions:
                result.filtered.append(
                    (
                        candidate,
                        f"admits {len(report.contradictions)} input(s) deriving "
                        f"two outcomes at once, e.g. "
                        f"{report.contradictions[0].describe()}",
                    )
                )
                continue
            result.admitted.append(candidate)
        return result

    # -- periodic sweep ----------------------------------------------------
    def sweep(self) -> MaintenanceReport:
        """What the rule base needs from a person now, and nothing else."""
        report = MaintenanceReport()
        report.flagged_for_rereview = [
            r.rule_id for r in self.governor.rules_needing_rereview()
        ]
        check = self.checker.check(self.governor.rulebase, deep=True)
        if not check.skipped:
            report.retirement_candidates = [
                (rid, "can never fire") for rid in check.dead_rules
            ] + list(check.redundant_rules)
        return report


__all__ = ["LogicMaintainer", "MaintenanceReport", "ScreenResult"]
