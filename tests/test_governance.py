"""Tests for rule governance.

`test_critical_decision_never_rests_on_unapproved_logic` is the one that matters.
It is the property the whole design is for, and it is the kind of guarantee that
could regress silently -- everything would still answer, just without the
protection. So it is checked directly and then re-audited over a simulated run.
"""

from __future__ import annotations

import pytest

from logicdb.facts import FactStore, Truth
from logicdb.governance import (
    BlockedForApproval,
    CriticalityPolicy,
    CriticalityRule,
    Governor,
    ProposalOutcome,
)
from logicdb.parser import parse_atom, parse_rule
from logicdb.program import RuleBase, RuleOrigin, RuleRecord, RuleStatus
from logicdb.syntax import PredicateDecl, Vocabulary


@pytest.fixture
def vocab() -> Vocabulary:
    return Vocabulary(
        [
            PredicateDecl("overdrawn", 1, "observable"),
            PredicateDecl("thin_file", 1, "observable"),
            PredicateDecl("guarantor", 1, "observable"),
            PredicateDecl("large_amount", 1, "observable"),
            PredicateDecl("decline", 1, "derived"),
            PredicateDecl("approve", 1, "derived"),
        ]
    )


@pytest.fixture
def governor(vocab: Vocabulary) -> Governor:
    policy = CriticalityPolicy(
        rules=[
            CriticalityRule(
                rule_id="large_sums",
                description="large amounts are reviewed online",
                field="amount",
                op="gte",
                value=10_000,
            )
        ],
        default_critical=False,
    )
    return Governor(RuleBase(vocab), criticality=policy, vocabulary=vocab)


def facts_for(vocab: Vocabulary, entity: str = "app1", **kw: str) -> FactStore:
    store = FactStore(vocab)
    for predicate, state in kw.items():
        store.assert_fact(parse_atom(f"{predicate}({entity})"), Truth(state), source="test")
    return store


# --------------------------------------------------------------------------
# Criticality routing
# --------------------------------------------------------------------------


def test_non_critical_proposal_applies_immediately(governor) -> None:
    result = governor.propose(
        parse_rule("decline(A) <- overdrawn(A), thin_file(A)."),
        origin=RuleOrigin.MINED, proposed_by="agent", critical_context=False,
    )
    assert result.outcome is ProposalOutcome.APPLIED_PROVISIONALLY
    assert result.record.status is RuleStatus.PROVISIONAL
    assert result.usable, "non-critical work must not stall waiting for review"


def test_critical_proposal_blocks(governor) -> None:
    result = governor.propose(
        parse_rule("decline(A) <- overdrawn(A), thin_file(A)."),
        origin=RuleOrigin.MINED, proposed_by="agent", critical_context=True,
    )
    assert result.outcome is ProposalOutcome.QUEUED_ONLINE
    assert result.record.status is RuleStatus.PENDING_ONLINE
    assert not result.usable


def test_critical_query_escalates_a_provisional_rule(governor) -> None:
    """A rule good enough for routine work is not automatically good enough here."""
    first = governor.propose(
        parse_rule("decline(A) <- overdrawn(A), thin_file(A)."),
        origin=RuleOrigin.MINED, proposed_by="agent", critical_context=False,
    )
    assert first.record.status is RuleStatus.PROVISIONAL

    second = governor.propose(
        parse_rule("decline(X) <- thin_file(X), overdrawn(X)."),
        origin=RuleOrigin.MINED, proposed_by="agent", critical_context=True,
    )
    assert second.outcome is ProposalOutcome.QUEUED_ONLINE
    assert second.record.rule_id == first.record.rule_id, "same rule, not a duplicate"
    assert second.record.status is RuleStatus.PENDING_ONLINE


def test_criticality_policy_reads_context(governor) -> None:
    critical, reason = governor.criticality.assess({"amount": 25_000})
    assert critical and "large_sums" in reason
    routine, _ = governor.criticality.assess({"amount": 500})
    assert not routine


def test_default_is_critical_when_nothing_declared() -> None:
    """Treating a consequential decision as routine is the worse failure."""
    critical, reason = CriticalityPolicy().assess({})
    assert critical
    assert "default" in reason


# --------------------------------------------------------------------------
# Approve once
# --------------------------------------------------------------------------


def test_approved_rule_is_never_queued_again(governor) -> None:
    """The requirement in one test: approve once, never asked again."""
    proposal = governor.propose(
        parse_rule("decline(A) <- overdrawn(A), thin_file(A)."),
        origin=RuleOrigin.MINED, proposed_by="agent", critical_context=True,
    )
    governor.approve(proposal.record.rule_id, by="reviewer-1")

    again = governor.propose(
        parse_rule("decline(Z) <- thin_file(Z), overdrawn(Z)."),
        origin=RuleOrigin.MINED, proposed_by="agent", critical_context=True,
    )
    assert again.outcome is ProposalOutcome.ALREADY_APPROVED
    assert again.record.rule_id == proposal.record.rule_id
    assert governor.review_queue() == [], "nothing should be awaiting review"


def test_rejected_rule_is_not_reconsidered(governor) -> None:
    proposal = governor.propose(
        parse_rule("decline(A) <- overdrawn(A)."),
        origin=RuleOrigin.MINED, proposed_by="agent", critical_context=False,
    )
    governor.reject(proposal.record.rule_id, by="reviewer-1", reason="too crude")

    again = governor.propose(
        parse_rule("decline(Q) <- overdrawn(Q)."),
        origin=RuleOrigin.MINED, proposed_by="agent", critical_context=False,
    )
    assert again.outcome is ProposalOutcome.ALREADY_REJECTED
    assert not again.usable


def test_invalid_proposals_never_reach_a_human(governor) -> None:
    """A malformed proposal must not consume reviewer attention."""
    result = governor.propose(
        parse_rule("decline(A) <- hallucinated_predicate(A)."),
        origin=RuleOrigin.MINED, proposed_by="agent", critical_context=True,
    )
    assert result.outcome is ProposalOutcome.REJECTED_INVALID
    assert governor.review_queue() == []


# --------------------------------------------------------------------------
# The governance invariant
# --------------------------------------------------------------------------


def test_critical_query_blocks_on_unapproved_logic(governor, vocab) -> None:
    governor.propose(
        parse_rule("decline(A) <- overdrawn(A), thin_file(A)."),
        origin=RuleOrigin.MINED, proposed_by="agent", critical_context=False,
    )
    facts = facts_for(vocab, overdrawn="true", thin_file="true", guarantor="false")

    with pytest.raises(BlockedForApproval) as exc:
        governor.ask(
            parse_atom("decline(app1)"), facts, entity="app1",
            asked_by="officer", context={"amount": 50_000},
        )
    assert exc.value.pending, "the block must name what needs approving"


def test_same_query_proceeds_once_approved(governor, vocab) -> None:
    proposal = governor.propose(
        parse_rule("decline(A) <- overdrawn(A), thin_file(A)."),
        origin=RuleOrigin.MINED, proposed_by="agent", critical_context=False,
    )
    facts = facts_for(vocab, overdrawn="true", thin_file="true", guarantor="false")
    context = {"amount": 50_000}

    with pytest.raises(BlockedForApproval):
        governor.ask(parse_atom("decline(app1)"), facts, "app1", "officer", context)

    governor.approve(proposal.record.rule_id, by="reviewer-1")

    answer, query = governor.ask(
        parse_atom("decline(app1)"), facts, "app1", "officer", context
    )
    assert query.status == "answered"
    assert query.critical
    assert answer.bounds.lower > 0


def test_second_similar_query_needs_no_further_approval(governor, vocab) -> None:
    """Amortisation: one review, many decisions. The core claim of the design."""
    proposal = governor.propose(
        parse_rule("decline(A) <- overdrawn(A), thin_file(A)."),
        origin=RuleOrigin.MINED, proposed_by="agent", critical_context=True,
    )
    governor.approve(proposal.record.rule_id, by="reviewer-1")

    for i in range(5):
        entity = f"app{i}"
        facts = facts_for(vocab, entity, overdrawn="true", thin_file="true",
                          guarantor="false")
        _, query = governor.ask(
            parse_atom(f"decline({entity})"), facts, entity, "officer",
            {"amount": 50_000},
        )
        assert query.status == "answered"

    assert governor.review_queue() == [], "no further reviews were required"
    assert proposal.record.stats.reuse_count == 5


def test_critical_decision_never_rests_on_unapproved_logic(governor, vocab) -> None:
    """The invariant, audited over a mixed run of routine and critical queries.

    Routine queries are allowed to use provisional logic; critical ones are not.
    After a run containing both, no critical query may have touched an
    unapproved rule.
    """
    approved = governor.propose(
        parse_rule("decline(A) <- overdrawn(A), thin_file(A)."),
        origin=RuleOrigin.MINED, proposed_by="agent", critical_context=True,
    )
    governor.approve(approved.record.rule_id, by="reviewer-1")
    governor.propose(
        parse_rule("decline(A) <- large_amount(A), overdrawn(A)."),
        origin=RuleOrigin.MINED, proposed_by="agent", critical_context=False,
    )

    blocked_count = 0
    for i in range(12):
        entity = f"app{i}"
        facts = facts_for(
            vocab, entity,
            overdrawn="true",
            thin_file="true" if i % 2 == 0 else "false",
            large_amount="true",
            guarantor="false",
        )
        amount = 50_000 if i % 3 == 0 else 500
        try:
            governor.ask(
                parse_atom(f"decline({entity})"), facts, entity, "officer",
                {"amount": amount},
            )
        except BlockedForApproval:
            blocked_count += 1

    assert blocked_count > 0, "some critical queries should have been blocked"
    assert governor.verify_invariant() == []


def test_invariant_audit_detects_a_violation(governor, vocab) -> None:
    """The audit must actually be capable of failing, or it proves nothing."""
    proposal = governor.propose(
        parse_rule("decline(A) <- overdrawn(A), thin_file(A)."),
        origin=RuleOrigin.MINED, proposed_by="agent", critical_context=True,
    )
    governor.approve(proposal.record.rule_id, by="reviewer-1")
    facts = facts_for(vocab, overdrawn="true", thin_file="true", guarantor="false")
    governor.ask(parse_atom("decline(app1)"), facts, "app1", "officer", {"amount": 50_000})

    assert governor.verify_invariant() == []
    # Retroactively un-approve, simulating a regression in the routing logic.
    proposal.record.status = RuleStatus.PROVISIONAL
    assert governor.verify_invariant() != []


# --------------------------------------------------------------------------
# Impact analysis
# --------------------------------------------------------------------------


def test_rejecting_a_provisional_rule_enumerates_affected_decisions(
    governor, vocab
) -> None:
    """The cost of the offline lane must be visible, not quietly erased."""
    proposal = governor.propose(
        parse_rule("decline(A) <- overdrawn(A), thin_file(A)."),
        origin=RuleOrigin.MINED, proposed_by="agent", critical_context=False,
    )
    for i in range(3):
        entity = f"app{i}"
        facts = facts_for(vocab, entity, overdrawn="true", thin_file="true",
                          guarantor="false")
        governor.ask(
            parse_atom(f"decline({entity})"), facts, entity, "officer", {"amount": 100}
        )

    report = governor.reject(proposal.record.rule_id, by="reviewer-1",
                             reason="confounded with postcode")
    assert len(report.affected_queries) == 3
    assert "revisiting" in report.render()


def test_rejecting_an_unused_rule_reports_no_impact(governor) -> None:
    proposal = governor.propose(
        parse_rule("decline(A) <- large_amount(A)."),
        origin=RuleOrigin.MINED, proposed_by="agent", critical_context=False,
    )
    report = governor.reject(proposal.record.rule_id, by="reviewer-1", reason="nope")
    assert report.affected_queries == []
    assert "Nothing to revisit" in report.render()


# --------------------------------------------------------------------------
# Learning from outcomes
# --------------------------------------------------------------------------


def test_outcomes_update_rule_evidence(governor, vocab) -> None:
    proposal = governor.propose(
        parse_rule("decline(A) <- overdrawn(A), thin_file(A)."),
        origin=RuleOrigin.MINED, proposed_by="agent", critical_context=True,
    )
    governor.approve(proposal.record.rule_id, by="reviewer-1")

    for i in range(10):
        entity = f"app{i}"
        facts = facts_for(vocab, entity, overdrawn="true", thin_file="true",
                          guarantor="false")
        _, query = governor.ask(
            parse_atom(f"decline({entity})"), facts, entity, "officer",
            {"amount": 50_000},
        )
        governor.record_outcome(query.query_id, correct=i < 7)

    stats = proposal.record.stats
    assert (stats.successes, stats.failures) == (7, 3)
    assert 0.5 < stats.posterior_mean() < 0.8


def test_degraded_rule_is_flagged_for_rereview(governor, vocab) -> None:
    """The one exception to approve-once, and only because the evidence changed."""
    proposal = governor.propose(
        parse_rule("decline(A) <- overdrawn(A), thin_file(A)."),
        origin=RuleOrigin.MINED, proposed_by="agent", critical_context=True,
        strength=0.9,
    )
    governor.approve(proposal.record.rule_id, by="reviewer-1")

    for i in range(12):
        entity = f"app{i}"
        facts = facts_for(vocab, entity, overdrawn="true", thin_file="true",
                          guarantor="false")
        _, query = governor.ask(
            parse_atom(f"decline({entity})"), facts, entity, "officer",
            {"amount": 50_000},
        )
        governor.record_outcome(query.query_id, correct=i < 4)

    flagged = governor.rules_needing_rereview()
    assert proposal.record in flagged


def test_healthy_rule_is_not_flagged(governor, vocab) -> None:
    proposal = governor.propose(
        parse_rule("decline(A) <- overdrawn(A), thin_file(A)."),
        origin=RuleOrigin.MINED, proposed_by="agent", critical_context=True,
        strength=0.8,
    )
    governor.approve(proposal.record.rule_id, by="reviewer-1")
    for i in range(12):
        entity = f"app{i}"
        facts = facts_for(vocab, entity, overdrawn="true", thin_file="true",
                          guarantor="false")
        _, query = governor.ask(
            parse_atom(f"decline({entity})"), facts, entity, "officer",
            {"amount": 50_000},
        )
        governor.record_outcome(query.query_id, correct=i < 11)
    assert governor.rules_needing_rereview() == []


# --------------------------------------------------------------------------
# Uncovered cases
# --------------------------------------------------------------------------


def test_uncovered_query_is_reported_as_such(governor, vocab) -> None:
    """No applicable logic is not the same as a negative answer."""
    proposal = governor.propose(
        parse_rule("decline(A) <- overdrawn(A), thin_file(A)."),
        origin=RuleOrigin.MINED, proposed_by="agent", critical_context=True,
    )
    governor.approve(proposal.record.rule_id, by="reviewer-1")

    facts = facts_for(vocab, overdrawn="false", thin_file="false", guarantor="false")
    answer, query = governor.ask(
        parse_atom("decline(app1)"), facts, "app1", "officer", {"amount": 50_000}
    )
    assert answer.is_uncovered
    assert query.status == "uncovered"


def test_critical_query_blocks_on_pending_rules_not_just_provisional(
    governor, vocab
) -> None:
    """A pending rule must block a critical query, not vanish from consideration.

    Regression test. `active_for(critical=False)` returns only PROVISIONAL and
    APPROVED rules, so a rule proposed in a critical context (PENDING_ONLINE)
    was invisible when deciding whether to block. The critical query then
    reported "no applicable logic" while a directly relevant rule sat in the
    review queue -- a false statement, and precisely the absence-versus-denial
    conflation this system exists to prevent.
    """
    proposal = governor.propose(
        parse_rule("decline(A) <- overdrawn(A), thin_file(A)."),
        origin=RuleOrigin.DOCUMENT, proposed_by="agent", critical_context=True,
    )
    assert proposal.record.status is RuleStatus.PENDING_ONLINE

    facts = facts_for(vocab, overdrawn="true", thin_file="true", guarantor="false")
    with pytest.raises(BlockedForApproval) as exc:
        governor.ask(
            parse_atom("decline(app1)"), facts, "app1", "officer",
            context={"amount": 50_000},
        )
    assert proposal.record.rule_id in [r.rule_id for r in exc.value.pending]


def test_rejected_rules_do_not_block_a_critical_query(governor, vocab) -> None:
    """A settled refusal must not resurface as a blocker."""
    proposal = governor.propose(
        parse_rule("decline(A) <- overdrawn(A), thin_file(A)."),
        origin=RuleOrigin.MINED, proposed_by="agent", critical_context=False,
    )
    governor.reject(proposal.record.rule_id, by="reviewer", reason="too crude")

    facts = facts_for(vocab, overdrawn="true", thin_file="true", guarantor="false")
    answer, query = governor.ask(
        parse_atom("decline(app1)"), facts, "app1", "officer", {"amount": 50_000}
    )
    assert answer.is_uncovered and query.status == "uncovered"
