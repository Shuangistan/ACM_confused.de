"""Rule governance: criticality routing, approve-once, and impact analysis.

This module is the project's actual answer to "what does meaningful human
control look like". Per-decision oversight does not scale and collapses into
rubber-stamping, especially where the reviewer cannot independently check the
output. Rule-level oversight amortises instead: review a rule once, and it
governs every future case whose facts match it.

The rules of the road, all enforced here rather than by convention:

* **Criticality attaches to the query; approval attaches to the rule.** A
  critical query blocks until every rule in its derivation is approved. A
  non-critical query proceeds on provisional logic and leaves a reviewable trace.

* **A critical decision can only ever rest on approved rules.** Enforced by
  handing the engine a filtered rule set, so an unapproved rule cannot
  contribute even accidentally. `verify_invariant` re-checks this after the
  fact, because a guarantee worth making is worth auditing.

* **Approve once, never again.** Lookup is by canonical key, so the same logic
  re-proposed in different syntax is recognised as already settled. Without
  this the whole amortisation argument silently degrades into one review per
  proposal.

* **Rejection is retroactive.** A provisional rule that gets rejected may
  already have influenced decisions. Those are enumerated, not quietly
  forgotten.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Iterable

from .engine import Engine, Evaluation
from .facts import FactStore, Truth
from .probability import Answer, ProbabilisticSolver
from .program import RuleBase, RuleOrigin, RuleRecord, RuleStatus
from .syntax import Atom, Rule, Vocabulary


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


# --------------------------------------------------------------------------
# Criticality
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class CriticalityRule:
    """A declared condition under which a query counts as critical.

    Kept as data on the pack rather than as code, so the framework stays
    domain-neutral and so the definition of "critical" is itself reviewable --
    which it should be, since it decides what gets scrutinised.
    """

    rule_id: str
    description: str
    #: Predicate over the query context: entity attributes, goal, requester.
    #: Expressed as simple field comparisons so a pack can declare it in YAML.
    field: str
    op: str
    value: Any

    def holds_for(self, context: dict[str, Any]) -> bool:
        actual = context.get(self.field)
        if actual is None:
            return False
        try:
            if self.op == "eq":
                return actual == self.value
            if self.op == "ne":
                return actual != self.value
            if self.op == "gt":
                return actual > self.value
            if self.op == "gte":
                return actual >= self.value
            if self.op == "lt":
                return actual < self.value
            if self.op == "lte":
                return actual <= self.value
            if self.op == "in":
                return actual in self.value
            if self.op == "contains":
                return self.value in actual
        except TypeError:
            return False
        raise ValueError(f"unknown criticality operator {self.op!r}")


@dataclass
class CriticalityPolicy:
    """Decides whether a query is critical, and says why.

    Defaults to critical when nothing is declared. Deliberate: the failure of
    treating a consequential decision as routine is far worse than the failure
    of asking for an unnecessary review, so the safe default is the strict one.
    """

    rules: list[CriticalityRule] = field(default_factory=list)
    default_critical: bool = True

    def assess(self, context: dict[str, Any]) -> tuple[bool, str]:
        for rule in self.rules:
            if rule.holds_for(context):
                return True, f"{rule.rule_id}: {rule.description}"
        if self.default_critical:
            return True, "no criticality rule matched, and the default is critical"
        return False, "no criticality rule matched; treated as routine"


# --------------------------------------------------------------------------
# Proposals and decisions
# --------------------------------------------------------------------------


class ProposalOutcome(str, Enum):
    ALREADY_APPROVED = "already_approved"
    """The same logic was settled previously. No human is asked twice."""

    ALREADY_REJECTED = "already_rejected"
    ALREADY_PENDING = "already_pending"
    QUEUED_ONLINE = "queued_online"
    """Critical context: a human must rule on it before the query can proceed."""

    APPLIED_PROVISIONALLY = "applied_provisionally"
    """Non-critical context: usable now, reviewed later."""

    REJECTED_INVALID = "rejected_invalid"
    """Failed vocabulary, safety or stratification checks. Never reached a human."""


@dataclass
class ProposalResult:
    outcome: ProposalOutcome
    record: RuleRecord | None
    message: str

    @property
    def usable(self) -> bool:
        return self.record is not None and self.record.status in (
            RuleStatus.APPROVED,
            RuleStatus.PROVISIONAL,
        )


@dataclass
class QueryRecord:
    """One question put to the system, and what it rested on."""

    query_id: str
    goal: Atom
    entity: str
    critical: bool
    criticality_reason: str
    asked_by: str
    asked_at: str = field(default_factory=_now)
    rule_ids: tuple[str, ...] = ()
    provisional_rule_ids: tuple[str, ...] = ()
    lower: float = 0.0
    upper: float = 0.0
    blocked_on: tuple[str, ...] = ()
    status: str = "answered"       # "answered" | "blocked_for_approval" | "uncovered"
    outcome_label: str | None = None
    confirmed_correct: bool | None = None

    @property
    def is_provisional(self) -> bool:
        return bool(self.provisional_rule_ids)


@dataclass
class ImpactReport:
    """What a rejected rule had already affected."""

    rule_id: str
    rule: Rule
    rejected_by: str
    reason: str
    affected_queries: list[QueryRecord] = field(default_factory=list)

    def render(self) -> str:
        lines = [
            f"Rule {self.rule_id} rejected by {self.rejected_by}: {self.reason}",
            f"  {self.rule}",
        ]
        if not self.affected_queries:
            lines.append("  No decisions used it. Nothing to revisit.")
            return "\n".join(lines)
        lines.append(
            f"  It was used provisionally in {len(self.affected_queries)} decision(s), "
            f"which now need revisiting:"
        )
        for q in self.affected_queries:
            lines.append(
                f"    {q.query_id}  {q.goal}  asked by {q.asked_by} at {q.asked_at}"
            )
        return "\n".join(lines)


class BlockedForApproval(Exception):
    """A critical query needs logic no human has approved yet.

    An exception rather than a return value because it must not be possible to
    ignore it by forgetting to check a flag. The query genuinely cannot proceed.
    """

    def __init__(self, query: QueryRecord, pending: list[RuleRecord]) -> None:
        self.query = query
        self.pending = pending
        names = ", ".join(r.rule_id for r in pending)
        super().__init__(
            f"critical query {query.query_id} needs approval of: {names}. "
            f"A critical decision may not rest on unreviewed logic."
        )


# --------------------------------------------------------------------------
# The governor
# --------------------------------------------------------------------------


class Governor:
    """Routes proposals and queries according to criticality and approval state."""

    def __init__(
        self,
        rulebase: RuleBase,
        criticality: CriticalityPolicy | None = None,
        vocabulary: Vocabulary | None = None,
        threshold: float = 0.5,
        store: Any = None,
    ) -> None:
        self.rulebase = rulebase
        self.criticality = criticality or CriticalityPolicy()
        self.vocabulary = vocabulary
        self.solver = ProbabilisticSolver(rulebase, threshold=threshold)
        #: Prior queries, loaded from the store when there is one. Impact
        #: analysis reaches only as far back as this does, so a governor with no
        #: store can only account for the current session.
        self.queries: dict[str, QueryRecord] = (
            store.load_queries() if store is not None else {}
        )
        self.impacts: list[ImpactReport] = []
        self.store = store
        self._query_counter = _highest_query_number(self.queries)

    # -- persistence helpers ----------------------------------------------
    def _persist_rule(self, record: RuleRecord, event: str, actor: str,
                      detail: str = "") -> None:
        if self.store is None:
            return
        self.store.save_rule(record)
        self.store.log_rule_event(record, event, actor, detail)

    def _persist_query(self, query: QueryRecord, facts: FactStore | None = None) -> None:
        if self.store is not None:
            self.store.save_query(query, facts)

    def _persist_stats(self, rule_ids: Iterable[str]) -> None:
        if self.store is None:
            return
        for rule_id in rule_ids:
            record = self.rulebase.get(rule_id)
            if record is not None:
                self.store.save_rule(record)

    # -- proposals ---------------------------------------------------------
    def propose(
        self,
        rule: Rule,
        origin: RuleOrigin,
        proposed_by: str,
        critical_context: bool,
        strength: float = 0.7,
        source_citation: str | None = None,
        mining_support: int | None = None,
        mining_precision: float | None = None,
        notes: str = "",
    ) -> ProposalResult:
        """Route a proposed rule.

        The approve-once check comes first, before any validation or queueing.
        A rule whose logic has already been settled must never reappear in front
        of a human, whatever syntax it arrives in this time.
        """
        key = rule.canonical_key()
        existing = self.rulebase.find_by_key(key)

        if existing is not None:
            if existing.status is RuleStatus.APPROVED:
                return ProposalResult(
                    ProposalOutcome.ALREADY_APPROVED, existing,
                    f"identical logic already approved as {existing.rule_id} by "
                    f"{existing.approved_by}; not re-queued",
                )
            if existing.status is RuleStatus.REJECTED:
                return ProposalResult(
                    ProposalOutcome.ALREADY_REJECTED, existing,
                    f"identical logic already rejected as {existing.rule_id}: "
                    f"{existing.rejected_reason}",
                )
            if existing.status in (RuleStatus.PENDING_ONLINE, RuleStatus.PROVISIONAL):
                # Already awaiting review. A critical context escalates it from
                # the offline queue to the blocking one, but does not duplicate it.
                if critical_context and existing.status is RuleStatus.PROVISIONAL:
                    existing.status = RuleStatus.PENDING_ONLINE
                    self._persist_rule(
                        existing, "escalated", proposed_by,
                        "a critical query needs this rule",
                    )
                    return ProposalResult(
                        ProposalOutcome.QUEUED_ONLINE, existing,
                        f"{existing.rule_id} was awaiting offline review; a critical "
                        f"query now needs it, so it has been escalated to blocking",
                    )
                return ProposalResult(
                    ProposalOutcome.ALREADY_PENDING, existing,
                    f"identical logic already awaiting review as {existing.rule_id}",
                )

        record = RuleRecord(
            rule=rule,
            rule_id=self.rulebase.next_rule_id(),
            strength=strength,
            status=RuleStatus.PENDING_ONLINE if critical_context else RuleStatus.PROVISIONAL,
            origin=origin,
            proposed_by=proposed_by,
            source_citation=source_citation,
            mining_support=mining_support,
            mining_precision=mining_precision,
            notes=notes,
        )

        try:
            stored = self.rulebase.add(record)
        except ValueError as exc:
            # Vocabulary, safety or stratification failure. Rejected mechanically,
            # so a malformed proposal never consumes a reviewer's attention.
            return ProposalResult(
                ProposalOutcome.REJECTED_INVALID, None,
                f"proposal rejected without review: {exc}",
            )

        if stored.status is RuleStatus.PENDING_ONLINE:
            self._persist_rule(
                stored, "queued_online", proposed_by,
                f"proposed by {proposed_by} in a critical context",
            )
            return ProposalResult(
                ProposalOutcome.QUEUED_ONLINE, stored,
                f"{stored.rule_id} needs approval before this critical query can "
                f"proceed",
            )
        self._persist_rule(
            stored, "applied_provisionally", proposed_by,
            f"proposed by {proposed_by}; usable for routine queries pending review",
        )
        return ProposalResult(
            ProposalOutcome.APPLIED_PROVISIONALLY, stored,
            f"{stored.rule_id} applied provisionally and queued for offline review",
        )

    # -- review ------------------------------------------------------------
    def approve(self, rule_id: str, by: str, note: str = "") -> RuleRecord:
        """Approve a rule, permanently.

        The amortisation claim lives here: from now on this rule is used without
        further review, however many cases it touches.
        """
        record = self.rulebase.get(rule_id)
        if record is None:
            raise KeyError(f"no rule {rule_id}")
        record.approve(by)
        if note:
            record.notes = (record.notes + "\n" + note).strip()
        # Written through immediately. An approval that lives only in memory is
        # the one failure that would silently undo the entire amortisation
        # argument: on the next restart the reviewer is asked again.
        self._persist_rule(record, "approved", by, note)
        return record

    def reject(self, rule_id: str, by: str, reason: str) -> ImpactReport:
        """Reject a rule and enumerate what it already affected.

        A provisional rule may have influenced real decisions before anyone
        looked at it. That exposure is the cost of the offline review lane, and
        the honest way to run it is to make the cost visible rather than let
        rejection quietly erase the history.
        """
        record = self.rulebase.get(rule_id)
        if record is None:
            raise KeyError(f"no rule {rule_id}")
        record.reject(by, reason)
        self._persist_rule(record, "rejected", by, reason)

        # Prefer the store: its history spans every session, while the in-memory
        # log only covers this one. A rejection that under-reports its own blast
        # radius is worse than no report.
        if self.store is not None:
            affected_ids = self.store.queries_using_rule(rule_id, provisional_only=True)
            affected = [self.queries[qid] for qid in affected_ids if qid in self.queries]
        else:
            affected = [
                q for q in self.queries.values() if rule_id in q.provisional_rule_ids
            ]
        report = ImpactReport(
            rule_id=rule_id, rule=record.rule, rejected_by=by,
            reason=reason, affected_queries=affected,
        )
        self.impacts.append(report)
        return report

    def review_queue(self) -> list[RuleRecord]:
        return self.rulebase.review_queue()

    # -- queries -----------------------------------------------------------
    def next_query_id(self) -> str:
        self._query_counter += 1
        return f"q{self._query_counter:05d}"

    def ask(
        self,
        goal: Atom,
        facts: FactStore,
        entity: str,
        asked_by: str,
        context: dict[str, Any] | None = None,
        competing: list[Atom] | None = None,
    ) -> tuple[Answer, QueryRecord]:
        """Answer a query, enforcing the governance rules.

        Raises `BlockedForApproval` when the query is critical and the logic it
        would need has not been approved. That is not an error condition -- it is
        the system working.
        """
        context = dict(context or {})
        context.setdefault("goal", goal.predicate)
        context.setdefault("entity", entity)
        critical, reason = self.criticality.assess(context)

        query = QueryRecord(
            query_id=self.next_query_id(),
            goal=goal,
            entity=entity,
            critical=critical,
            criticality_reason=reason,
            asked_by=asked_by,
        )

        if critical:
            blocking = self._rules_that_would_help_but_are_unapproved(goal, facts)
            if blocking:
                query.status = "blocked_for_approval"
                query.blocked_on = tuple(r.rule_id for r in blocking)
                self.queries[query.query_id] = query
                for record in blocking:
                    if record.status is RuleStatus.PROVISIONAL:
                        record.status = RuleStatus.PENDING_ONLINE
                        self._persist_rule(
                            record, "escalated", asked_by,
                            f"blocking critical query {query.query_id}",
                        )
                self._persist_query(query, facts)
                raise BlockedForApproval(query, blocking)

        answer = self.solver.answer(goal, facts, critical=critical, competing=competing)

        used = sorted(answer.evaluation.rules_used)
        provisional = [
            rid for rid in used
            if (r := self.rulebase.get(rid)) and r.status is RuleStatus.PROVISIONAL
        ]
        query.rule_ids = tuple(used)
        query.provisional_rule_ids = tuple(provisional)
        query.lower, query.upper = answer.bounds.lower, answer.bounds.upper
        query.status = "uncovered" if answer.is_uncovered else "answered"

        # Reuse is counted per distinct query, which is the amortisation figure:
        # how many decisions one review has governed.
        for rid in used:
            record = self.rulebase.get(rid)
            if record is not None:
                record.stats.reuse_count += 1
                record.stats.last_used_at = query.asked_at

        self.queries[query.query_id] = query
        # The facts go down with the query, so this decision can be replayed
        # exactly rather than re-derived against a rule base that has since moved.
        self._persist_query(query, facts)
        self._persist_stats(used)
        return answer, query

    def _rules_that_would_help_but_are_unapproved(
        self, goal: Atom, facts: FactStore
    ) -> list[RuleRecord]:
        """Unapproved rules that would contribute if they were approved.

        Found by evaluating with the non-critical rule set and seeing which
        unapproved rules fire. Only rules that actually bear on this case are
        surfaced, so a reviewer is not asked to rule on logic irrelevant to the
        query that is waiting.
        """
        # Every rule that could still become usable -- not just the ones already
        # usable for routine queries. A rule sitting in PENDING_ONLINE is
        # invisible to `active_for(critical=False)`, and considering only that
        # set produced a quietly false answer: a critical query would report "no
        # applicable logic" while eight relevant rules sat in the review queue.
        # "Nothing bears on this case" and "what bears on it has not been
        # reviewed" are different states, and this is the distinction the whole
        # system is built to preserve.
        candidates = [
            r for r in self.rulebase.all_records()
            if r.status in (
                RuleStatus.PROPOSED,
                RuleStatus.PENDING_ONLINE,
                RuleStatus.PROVISIONAL,
                RuleStatus.APPROVED,
            )
        ]
        if not candidates:
            return []
        evaluation = Engine().evaluate(
            self.rulebase, facts, critical=False, rules=candidates
        )
        out = []
        for rid in sorted(evaluation.rules_used):
            record = self.rulebase.get(rid)
            if record is not None and record.status is not RuleStatus.APPROVED:
                out.append(record)
        return out

    # -- outcomes and learning --------------------------------------------
    def record_outcome(self, query_id: str, correct: bool, label: str | None = None) -> None:
        """Feed a confirmed outcome back to the rules that produced it.

        This is what "improved during interaction" means concretely: strengths
        stop being author guesses and become tallies a reviewer can inspect.
        """
        query = self.queries.get(query_id)
        if query is None:
            raise KeyError(f"no query {query_id}")
        query.confirmed_correct = correct
        query.outcome_label = label
        for rid in query.rule_ids:
            record = self.rulebase.get(rid)
            if record is None:
                continue
            if correct:
                record.stats.successes += 1
            else:
                record.stats.failures += 1
        self._persist_query(query)
        self._persist_stats(query.rule_ids)

    def rules_needing_rereview(
        self, min_observations: int = 8, degradation: float = 0.2
    ) -> list[RuleRecord]:
        """Approved rules whose evidence has turned against them.

        The single exception to approve-once, and it is not a re-litigation of
        the original decision: the reviewer is being shown new evidence, not
        asked to repeat themselves. Without this, "approved forever" would mean
        a rule that has since been wrong forty times stays in force.
        """
        out = []
        for record in self.rulebase.by_status(RuleStatus.APPROVED):
            if record.stats.observations < min_observations:
                continue
            if record.strength - record.stats.posterior_mean() >= degradation:
                out.append(record)
        return out

    # -- audit -------------------------------------------------------------
    def verify_invariant(self) -> list[str]:
        """Re-check the governance guarantee over everything answered so far.

        The engine is supposed to make violations impossible by construction.
        This audits it anyway: a guarantee worth making is worth checking, and
        a silent regression here would be invisible in every other output.
        """
        violations: list[str] = []
        for query in self.queries.values():
            if not query.critical or query.status != "answered":
                continue
            for rid in query.rule_ids:
                record = self.rulebase.get(rid)
                if record is None:
                    violations.append(
                        f"{query.query_id}: used rule {rid}, which no longer exists"
                    )
                elif record.status is not RuleStatus.APPROVED:
                    violations.append(
                        f"{query.query_id} is critical but used {rid}, whose status "
                        f"is {record.status.value}"
                    )
        return violations

    def amortisation_report(self) -> str:
        """How much decision-making each act of review has bought."""
        approved = self.rulebase.by_status(RuleStatus.APPROVED)
        answered = [q for q in self.queries.values() if q.status == "answered"]
        if not approved:
            return "No approved rules yet, so no oversight has been amortised."

        total_reuse = sum(r.stats.reuse_count for r in approved)
        top = sorted(approved, key=lambda r: -r.stats.reuse_count)[:5]
        lines = [
            f"Reviews performed:        {len(approved)}",
            f"Decisions answered:       {len(answered)}",
            f"Rule applications:        {total_reuse}",
            # The headline claim is decisions, not rule firings. Several rules
            # fire per decision, so dividing applications by reviews would
            # overstate this by that factor -- both are shown so the difference
            # is visible rather than a matter of which one got labelled.
            f"Decisions per review:     {len(answered) / len(approved):.1f}",
            f"Applications per rule:    {total_reuse / len(approved):.1f}",
            "",
            "Most-reused rules:",
        ]
        for record in top:
            lines.append(
                f"  {record.rule_id}  used {record.stats.reuse_count:4d}x  "
                f"approved by {record.approved_by}  {record.rule}"
            )
        return "\n".join(lines)

    def governance_summary(self) -> dict[str, Any]:
        queries = list(self.queries.values())
        return {
            "rules_total": len(self.rulebase),
            "rules_approved": len(self.rulebase.by_status(RuleStatus.APPROVED)),
            "rules_provisional": len(self.rulebase.by_status(RuleStatus.PROVISIONAL)),
            "rules_pending_online": len(
                self.rulebase.by_status(RuleStatus.PENDING_ONLINE)
            ),
            "rules_rejected": len(self.rulebase.by_status(RuleStatus.REJECTED)),
            "queries_total": len(queries),
            "queries_critical": sum(1 for q in queries if q.critical),
            "queries_blocked": sum(
                1 for q in queries if q.status == "blocked_for_approval"
            ),
            "queries_provisional": sum(1 for q in queries if q.is_provisional),
            "queries_uncovered": sum(1 for q in queries if q.status == "uncovered"),
            "invariant_violations": len(self.verify_invariant()),
        }


def _highest_query_number(queries: dict[str, QueryRecord]) -> int:
    """Resume numbering after a restart rather than colliding with history."""
    highest = 0
    for query_id in queries:
        if query_id.startswith("q") and query_id[1:].isdigit():
            highest = max(highest, int(query_id[1:]))
    return highest


def open_governor(
    db_path: str | Any,
    vocabulary: Vocabulary,
    criticality: CriticalityPolicy | None = None,
    threshold: float = 0.5,
) -> Governor:
    """Open a persistent logic database, restoring rules, approvals and history.

    The whole point of persistence here: a governor opened on an existing
    database already knows which rules were approved and by whom, so nobody is
    asked to re-approve settled logic, and a rule rejected next month can still
    name the decisions it touched last month.
    """
    from .store import Store

    store = Store(db_path)
    store.save_vocabulary(vocabulary)
    rulebase = store.load_rulebase(vocabulary)
    return Governor(
        rulebase,
        criticality=criticality,
        vocabulary=vocabulary,
        threshold=threshold,
        store=store,
    )


__all__ = [
    "BlockedForApproval",
    "CriticalityPolicy",
    "CriticalityRule",
    "Governor",
    "ImpactReport",
    "ProposalOutcome",
    "ProposalResult",
    "QueryRecord",
    "open_governor",
]
