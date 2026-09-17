"""The two claims the paper makes that nothing else guards.

The gate's rule — approved logic proceeds, anything else at a consequential tier
waits — is the whole argument. Approve-once is the mechanism that makes it
bearable. Both were verified by hand at a shell prompt, which is how a claim
survives until the day it quietly stops being true.

Neither test needs a model. The gate is a pure function of tier and coverage;
the store is a database.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent import ladder  # noqa: E402
from agent.nodes.gate import AUDIT_RATE, gate, may_proceed  # noqa: E402
from logic.parse import parse_rule  # noqa: E402
from logic.store import APPROVED, REJECTED, RuleRow, Store  # noqa: E402


def state(tier: str, covered: bool, case_id: str = "case-1") -> dict:
    return {"tier": tier, "covered": covered, "case_id": case_id}


# ---------------------------------------------------------------------------
# The gate
#
# The tier sets what is at stake. What decides whether a person is asked is
# whether approved logic already covers the case.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("tier", [
    ladder.HUMAN_ONLY, ladder.HUMAN_APPROVAL,
    ladder.AI_WITH_OVERSIGHT, ladder.AI_RECOMMENDS,
])
def test_approved_logic_proceeds_at_every_tier(tier: str) -> None:
    """Including `human_only`. A rule an expert approved is their decision."""
    out = gate(state(tier, covered=True))
    assert out["blocked"] is False
    assert may_proceed({**state(tier, True), **out}) == "respond"


@pytest.mark.parametrize("tier", [ladder.HUMAN_ONLY, ladder.HUMAN_APPROVAL])
def test_an_uncovered_consequential_case_waits(tier: str) -> None:
    out = gate(state(tier, covered=False))
    assert out["blocked"] is True
    assert out["block_reason"]
    assert may_proceed({**state(tier, False), **out}) == "park"


@pytest.mark.parametrize("tier", [ladder.AI_WITH_OVERSIGHT, ladder.AI_RECOMMENDS])
def test_an_uncovered_routine_case_proceeds(tier: str) -> None:
    out = gate(state(tier, covered=False))
    assert out["blocked"] is False


def test_a_missing_tier_fails_towards_asking_a_person() -> None:
    """An absent tier must not read as a permissive one."""
    out = gate({"covered": False, "case_id": "x"})
    assert out["blocked"] is True


# ---------------------------------------------------------------------------
# Audit sampling
# ---------------------------------------------------------------------------
def test_sampling_is_reproducible_not_random() -> None:
    """Asked why a case was checked, "the random number generator" is not an
    answer anyone should accept."""
    runs = {gate(state(ladder.AI_WITH_OVERSIGHT, True, "same-case"))["sampled"]
            for _ in range(25)}
    assert len(runs) == 1


def test_the_rate_is_highest_where_a_wrong_decision_hurts_most() -> None:
    assert (AUDIT_RATE[ladder.HUMAN_ONLY]
            > AUDIT_RATE[ladder.HUMAN_APPROVAL]
            > AUDIT_RATE[ladder.AI_RECOMMENDS]
            >= AUDIT_RATE[ladder.AI_WITH_OVERSIGHT])


def test_no_tier_ever_stops_being_watched() -> None:
    """A rule right a thousand times can start being wrong when the population
    shifts underneath it."""
    assert all(rate > 0 for rate in AUDIT_RATE.values())


def test_roughly_the_stated_proportion_is_pulled() -> None:
    rate = AUDIT_RATE[ladder.HUMAN_ONLY]
    pulled = sum(gate(state(ladder.HUMAN_ONLY, True, f"case-{i}"))["sampled"]
                 for i in range(400))
    assert abs(pulled / 400 - rate) < 0.08


# ---------------------------------------------------------------------------
# Approve-once
#
# Without canonical identity, "approve once" degrades into approving trivial
# variants forever, and the amortisation the design rests on collapses to one
# review per proposal.
# ---------------------------------------------------------------------------
@pytest.fixture
def store() -> Store:
    return Store(":memory:")


def put(store: Store, text: str) -> str:
    rule = parse_rule(text)
    store.upsert_rule(RuleRow(rule.canonical_key(), str(rule),
                              rule.head.predicate, proposed_by="agent"))
    return rule.canonical_key()


def test_the_same_logic_in_different_syntax_is_one_row(store: Store) -> None:
    a = put(store, "decline(A) <- overdrawn(A), not guarantor(A).")
    b = put(store, "decline(X) <- not guarantor(X), overdrawn(X).")
    assert a == b
    assert len(store.rules()) == 1


def test_re_proposing_an_approved_rule_leaves_it_approved(store: Store) -> None:
    """The failure this prevents: an agent re-proposes, the row resets to
    pending, and a reviewer is asked to bless what they already blessed."""
    key = put(store, "decline(A) <- overdrawn(A), not guarantor(A).")
    store.set_status(key, APPROVED, "SP", "checked")
    put(store, "decline(Z) <- not guarantor(Z), overdrawn(Z).")
    assert store.get_rule(key).status == APPROVED
    assert store.get_rule(key).approved_by == "SP"


def test_re_proposing_a_rejected_rule_does_not_revive_it(store: Store) -> None:
    key = put(store, "decline(A) <- overdrawn(A), not guarantor(A).")
    store.set_status(key, REJECTED, "SP", "fires on cases it should not")
    put(store, "decline(Q) <- overdrawn(Q), not guarantor(Q).")
    assert store.get_rule(key).status == REJECTED


def test_a_genuinely_different_rule_is_a_second_row(store: Store) -> None:
    put(store, "decline(A) <- overdrawn(A), not guarantor(A).")
    put(store, "decline(A) <- overdrawn(A), guarantor(A).")
    assert len(store.rules()) == 2


def test_only_approved_rules_are_active(store: Store) -> None:
    approved = put(store, "decline(A) <- overdrawn(A), not guarantor(A).")
    put(store, "decline(A) <- arrears(A).")
    store.set_status(approved, APPROVED, "SP")
    assert [r.canonical_key for r in store.active_rules()] == [approved]


def test_every_change_leaves_a_trail(store: Store) -> None:
    key = put(store, "decline(A) <- overdrawn(A), not guarantor(A).")
    store.set_status(key, APPROVED, "SP", "checked")
    events = [(e["event"], e["actor"]) for e in store.events(key)]
    assert ("proposed", "agent") in events
    assert ("approved", "SP") in events


# ---------------------------------------------------------------------------
# Parked decisions
# ---------------------------------------------------------------------------
def test_a_parked_decision_survives_in_the_database(store: Store) -> None:
    store.record_case("c1", "evict?", [["in_arrears(c)", "true", "x"]], [], [],
                      "uncovered", tier=ladder.HUMAN_ONLY, status="waiting")
    assert [c["case_id"] for c in store.waiting()] == ["c1"]


def test_settling_it_once_works_and_twice_does_not(store: Store) -> None:
    store.record_case("c1", "evict?", [["in_arrears(c)", "true", "x"]], [], [],
                      "uncovered", tier=ladder.HUMAN_ONLY, status="waiting")
    assert store.settle_case("c1", "approved", "SP", "reviewed", "the answer")
    assert not store.settle_case("c1", "approved", "SP")
    assert store.get_case("c1")["status"] == "approved"
    assert store.get_case("c1")["reply"] == "the answer"


def test_coverage_counts_only_what_the_rules_decided(store: Store) -> None:
    for i in range(3):
        store.record_case(f"r{i}", "q", [["a(c)", "true", "x"]], ["b(c)"], ["k"],
                          "rules", covered=True)
    store.record_case("u1", "q", [["a(c)", "true", "x"]], [], [], "uncovered")
    assert store.stats()["coverage"] == pytest.approx(0.75)
