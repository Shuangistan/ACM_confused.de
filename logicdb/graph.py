"""The structure as a LangGraph state graph.

Same architecture as `system.py`, expressed as a graph rather than as nested
loops. Three things are genuinely better here and one thing must be protected.

**The approval gate is an `interrupt`.** Offline human approval is exactly what
`interrupt()` is for: the graph pauses at the gate, its state is checkpointed,
and it resumes when a person answers -- possibly in another process, on another
day. `system.py` hand-rolled that as a synchronous callback, which was a
simplification of what production would need.

**Cases fan out.** Deciding a case is independent of deciding any other, so
`Send` dispatches them concurrently instead of walking a list.

**The topology is inspectable.** The edges *are* the control flow, rather than
being implied by `if` statements spread across two files.

What must be protected is the property the whole project rests on: the graph
schedules, and the engine decides. No node here computes an answer, chooses
between competing outcomes, or writes an explanation -- each one calls into the
same `Engine`, `Governor` and `Checker` used everywhere else. Unplug LangGraph
and `system.py` produces identical results from identical inputs, which is the
test that this remained a scheduling change rather than a semantic one.

State is deliberately plain: dicts of primitives, not domain objects. Anything
in graph state gets checkpointed and must survive a round trip, and a rule
record that serialises subtly wrong would corrupt the audit trail rather than
raise. The durable record stays in SQLite, where it already was.
"""

from __future__ import annotations

import operator
from dataclasses import dataclass, field
from typing import Annotated, Any, Callable, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, Overwrite, Send, interrupt

from .agents.base import LabelledCase, validate_proposals
from .maintenance import MaintenanceReport
from .program import RuleRecord
from .system import System

#: Returns (approve, note) for one rule. Stands where a person stands.
Reviewer = Callable[[RuleRecord], tuple[bool, str]]
#: Returns (was_correct, observed_label) for one decided case.
Auditor = Callable[[Any], tuple[bool, str | None]]


class ScreeningState(TypedDict, total=False):
    """What flows along the edges.

    `outcomes` carries one small dict per decided case rather than a `Decision`
    object: the fan-out reducer has to merge it, the checkpointer has to store
    it, and neither should be handed an object graph that reaches the rule base.
    """

    cycle: int
    cases: list[dict[str, Any]]
    #: Written concurrently by the fan-out, so it needs a reducer.
    outcomes: Annotated[list[dict[str, Any]], operator.add]
    gaps: list[dict[str, Any]]
    staged: list[str]
    admitted: list[str]
    filtered: list[tuple[str, str]]
    report: dict[str, Any]
    reports: Annotated[list[dict[str, Any]], operator.add]
    max_cycles: int


@dataclass
class CycleReport:
    """What one pass of the graph did."""

    index: int
    decided: int = 0
    automatic: int = 0
    audited: int = 0
    audit_verdicts: int = 0
    audits_wrong: int = 0
    uncovered: int = 0
    blocked: int = 0
    proposed: int = 0
    screened_out: int = 0
    queued: int = 0
    approved: int = 0
    rejected: int = 0
    maintenance: MaintenanceReport | None = None

    @property
    def productive(self) -> bool:
        """Whether anything changed. The loop's stop test."""
        return bool(self.queued or self.approved or self.rejected)

    def describe(self) -> str:
        lines = [
            f"cycle {self.index}",
            f"  decisions : {self.decided} decided, {self.automatic} stood without "
            f"a person, {self.uncovered} uncovered, {self.blocked} blocked",
            f"  audits    : {self.audited} sampled, {self.audit_verdicts} verdicts "
            f"returned, {self.audits_wrong} found wrong",
            f"  logic     : {self.proposed} proposed, {self.screened_out} screened "
            f"out mechanically, {self.queued} reached the queue "
            f"({self.approved} approved, {self.rejected} rejected)",
        ]
        if self.maintenance is not None:
            for line in self.maintenance.describe().splitlines():
                lines.append(f"  {line}")
        return "\n".join(lines)


@dataclass
class GraphRun:
    """The result of driving the graph to a stop."""

    reports: list[CycleReport] = field(default_factory=list)
    interrupted: bool = False
    pending_review: list[dict[str, Any]] = field(default_factory=list)

    def describe(self) -> str:
        lines = [r.describe() for r in self.reports]
        if self.interrupted:
            lines.append(
                f"paused at the gate: {len(self.pending_review)} rule(s) "
                f"awaiting a person"
            )
        return "\n\n".join(lines)


def build_graph(
    system: System,
    labelled: list[LabelledCase] | None = None,
    auditor: Auditor | None = None,
):
    """Compile the screening graph over an assembled `System`.

    Dependencies are closed over rather than carried in state. They are objects
    with database handles; putting them in state would mean checkpointing a
    connection, and the graph has no business owning the rule base.
    """

    # -- nodes -------------------------------------------------------------
    def open_cycle(state: ScreeningState) -> dict[str, Any]:
        """Start a pass. Clears the previous pass's per-case results."""
        return {
            "cycle": state.get("cycle", 0) + 1,
            "outcomes": Overwrite([]),
            "gaps": [],
            "staged": [],
            "admitted": [],
            "filtered": [],
        }

    def fan_out(state: ScreeningState) -> list[Send]:
        """One `decide_case` task per case. The cases do not interact."""
        return [Send("decide_case", {"record": r}) for r in state["cases"]]

    def decide_case(payload: dict[str, Any]) -> dict[str, Any]:
        """Route one case. The engine decides; this records what happened."""
        record = payload["record"]
        decision = system.node.decide(record)
        entry: dict[str, Any] = {
            "entity": decision.entity,
            "query_id": decision.query_id,
            "uncovered": decision.uncovered,
            "blocked": bool(decision.blocked_on),
            "took_effect": decision.took_effect,
            "sampled": bool(
                decision.assessment is not None and decision.assessment.sampled
            ),
            "mode": (
                decision.assessment.mode.value if decision.assessment else None
            ),
        }
        # The audit verdict is taken here, while the decision object is in
        # hand; only its result travels on, because the object does not need to.
        if entry["sampled"] and auditor is not None:
            correct, label = auditor(decision)
            entry["audited"] = True
            entry["correct"] = bool(correct)
            entry["label"] = label
        if decision.uncovered or decision.blocked_on:
            entry["record"] = record
        return {"outcomes": [entry]}

    def triage(state: ScreeningState) -> dict[str, Any]:
        """Tally the pass, post audit verdicts, and collect the gaps."""
        report = CycleReport(index=state["cycle"])
        gaps: list[dict[str, Any]] = []
        for entry in state["outcomes"]:
            if entry["blocked"]:
                report.blocked += 1
                continue
            if entry["uncovered"]:
                report.uncovered += 1
                if "record" in entry:
                    gaps.append(entry["record"])
                continue
            report.decided += 1
            if entry["took_effect"]:
                report.automatic += 1
            if entry["sampled"]:
                report.audited += 1
            if entry.get("audited") and entry.get("query_id"):
                system.governor.record_outcome(
                    entry["query_id"], correct=entry["correct"], label=entry.get("label")
                )
                report.audit_verdicts += 1
                if not entry["correct"]:
                    report.audits_wrong += 1
        return {"gaps": gaps, "report": _pack(report)}

    def has_gaps(state: ScreeningState) -> str:
        """A gap is a request for logic. No gap, nothing to propose."""
        if state["gaps"] and labelled:
            return "propose"
        return "maintain"

    def propose(state: ScreeningState) -> dict[str, Any]:
        """Ask the agent for logic covering what the base could not answer."""
        report = _unpack(state["report"])
        known = {r.rule.canonical_key() for r in system.governor.rulebase}
        proposals = system.agent.mine_rules(
            labelled, target=system.pack.goal_predicate, existing=known
        )
        accepted, _bad = validate_proposals(proposals, system.pack.vocabulary)
        report.proposed = len(accepted)
        staged: list[str] = []
        for proposal in accepted:
            result = system.governor.propose(
                proposal.rule,
                origin=proposal.origin,
                proposed_by=system.agent.agent_id,
                critical_context=True,
                strength=proposal.strength,
                source_citation=proposal.source_citation,
                mining_support=proposal.support,
                mining_precision=proposal.precision,
                notes=proposal.rationale,
            )
            if result.record is not None:
                staged.append(result.record.rule_id)
        return {"staged": staged, "report": _pack(report)}

    def screen(state: ScreeningState) -> dict[str, Any]:
        """Reject mechanically what a person should never be asked to judge."""
        report = _unpack(state["report"])
        records = [
            r for r in (system.governor.rulebase.get(i) for i in state["staged"])
            if r is not None
        ]
        result = system.maintainer.screen(records)
        for record, why in result.filtered:
            system.governor.reject(record.rule_id, by="checker", reason=why)
        report.screened_out = len(result.filtered)
        report.queued = len(result.admitted)
        return {
            "admitted": [r.rule_id for r in result.admitted],
            "filtered": [(r.rule_id, why) for r, why in result.filtered],
            "report": _pack(report),
        }

    def review(state: ScreeningState) -> dict[str, Any]:
        """The gate. Pauses here until a person answers.

        This is the one node that does not return on its own. `interrupt`
        checkpoints the graph and raises; the caller resumes with a verdict per
        rule. A queue that empties itself would not be a gate.
        """
        report = _unpack(state["report"])
        if not state["admitted"]:
            return {"report": _pack(report)}

        queue = []
        for rule_id in state["admitted"]:
            record = system.governor.rulebase.get(rule_id)
            if record is None:
                continue
            queue.append(
                {
                    "rule_id": record.rule_id,
                    "rule": str(record.rule),
                    "origin": record.origin.value,
                    "strength": record.strength,
                    "cites": record.source_citation,
                    "support": record.mining_support,
                    "precision": record.mining_precision,
                }
            )
        verdicts: dict[str, Any] = interrupt({"review_queue": queue}) or {}

        for rule_id in state["admitted"]:
            verdict = verdicts.get(rule_id)
            if verdict is None:
                continue  # left in the queue; a gate may be answered in parts
            approve, note = verdict
            if approve:
                system.governor.approve(rule_id, by="reviewer", note=note)
                report.approved += 1
            else:
                system.governor.reject(rule_id, by="reviewer", reason=note)
                report.rejected += 1
        return {"report": _pack(report)}

    def maintain(state: ScreeningState) -> dict[str, Any]:
        """What the rule base needs from a person now, independent of this pass."""
        report = _unpack(state["report"])
        report.maintenance = system.maintainer.sweep()
        return {"report": _pack(report), "reports": [_pack(report)]}

    def again(state: ScreeningState) -> str:
        """Stop when a pass changed nothing, or at the backstop."""
        report = _unpack(state["report"])
        if not report.productive:
            return END
        if state["cycle"] >= state.get("max_cycles", 5):
            return END
        return "open_cycle"

    # -- topology ----------------------------------------------------------
    graph = StateGraph(ScreeningState)
    graph.add_node("open_cycle", open_cycle)
    graph.add_node("decide_case", decide_case)
    graph.add_node("triage", triage)
    graph.add_node("propose", propose)
    graph.add_node("screen", screen)
    graph.add_node("review", review)
    graph.add_node("maintain", maintain)

    graph.add_edge(START, "open_cycle")
    graph.add_conditional_edges("open_cycle", fan_out, ["decide_case"])
    graph.add_edge("decide_case", "triage")
    graph.add_conditional_edges("triage", has_gaps, ["propose", "maintain"])
    graph.add_edge("propose", "screen")
    graph.add_edge("screen", "review")
    graph.add_edge("review", "maintain")
    graph.add_conditional_edges("maintain", again, ["open_cycle", END])
    return graph


# --------------------------------------------------------------------------
# CycleReport <-> plain dict. State must stay serialisable; reports are how the
# run is read afterwards, so they travel as data and are rebuilt at the edges.
# --------------------------------------------------------------------------
_REPORT_FIELDS = (
    "index", "decided", "automatic", "audited", "audit_verdicts", "audits_wrong",
    "uncovered", "blocked", "proposed", "screened_out", "queued", "approved",
    "rejected",
)


def _pack(report: CycleReport) -> dict[str, Any]:
    out = {name: getattr(report, name) for name in _REPORT_FIELDS}
    if report.maintenance is not None:
        out["maintenance"] = {
            "flagged_for_rereview": list(report.maintenance.flagged_for_rereview),
            "retirement_candidates": [
                list(pair) for pair in report.maintenance.retirement_candidates
            ],
        }
    return out


def _unpack(data: dict[str, Any]) -> CycleReport:
    report = CycleReport(**{k: data[k] for k in _REPORT_FIELDS if k in data})
    if "maintenance" in data:
        report.maintenance = MaintenanceReport(
            flagged_for_rereview=list(data["maintenance"]["flagged_for_rereview"]),
            retirement_candidates=[
                (a, b) for a, b in data["maintenance"]["retirement_candidates"]
            ],
        )
    return report


def run_graph(
    system: System,
    cases: list[dict[str, Any]],
    labelled: list[LabelledCase] | None = None,
    auditor: Auditor | None = None,
    reviewer: Reviewer | None = None,
    thread_id: str = "screening",
    max_cycles: int = 5,
    checkpointer=None,
) -> GraphRun:
    """Drive the graph, answering the gate with `reviewer`.

    With no `reviewer` the run stops at the first gate and hands back the queue,
    which is the honest production shape: the people who approve rules are not
    sitting inside the process. Supplying one collapses that wait to a function
    call, which is what a demo and a test need.
    """
    from langgraph.checkpoint.memory import InMemorySaver

    app = build_graph(system, labelled, auditor).compile(
        checkpointer=checkpointer or InMemorySaver()
    )
    config = {"configurable": {"thread_id": thread_id}}
    state: Any = {
        "cycle": 0, "cases": cases, "outcomes": [], "gaps": [],
        "staged": [], "admitted": [], "filtered": [], "reports": [],
        "max_cycles": max_cycles,
    }

    run = GraphRun()
    while True:
        result = app.invoke(state, config)
        pending = result.get("__interrupt__")
        if not pending:
            run.reports = [_unpack(d) for d in result.get("reports", [])]
            return run
        queue = pending[0].value["review_queue"]
        if reviewer is None:
            run.interrupted = True
            run.pending_review = queue
            snapshot = app.get_state(config).values
            run.reports = [_unpack(d) for d in snapshot.get("reports", [])]
            return run
        verdicts = {}
        for item in queue:
            record = system.governor.rulebase.get(item["rule_id"])
            if record is not None:
                verdicts[item["rule_id"]] = list(reviewer(record))
        state = Command(resume=verdicts)


__all__ = ["Auditor", "CycleReport", "GraphRun", "Reviewer", "ScreeningState",
           "build_graph", "run_graph"]
