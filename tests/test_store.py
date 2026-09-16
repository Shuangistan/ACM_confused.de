"""Tests for persistence.

`test_approval_survives_restart` is the point of the module. "Approve once and
never again" is a promise about the future, so an approval that lives only in
memory quietly breaks it: the next time the process starts, the reviewer is
asked the same question again, and the amortisation argument -- one review, many
decisions -- becomes one review per restart.

The other property worth guarding is that impact analysis reaches backwards. A
rule rejected today may have influenced decisions made weeks ago, in a different
session. If rejection can only see the current process, it under-reports its own
blast radius, which is worse than not reporting at all.
"""

from __future__ import annotations

import pytest

from logicdb.facts import FactStore, Truth
from logicdb.governance import (
    BlockedForApproval,
    CriticalityPolicy,
    CriticalityRule,
    open_governor,
)
from logicdb.parser import parse_atom, parse_rule
from logicdb.program import RuleOrigin, RuleStatus
from logicdb.store import Store
from logicdb.syntax import PredicateDecl, Vocabulary


@pytest.fixture
def vocab() -> Vocabulary:
    return Vocabulary(
        [
            PredicateDecl("overdrawn", 1, "observable", phrase="{0} is overdrawn"),
            PredicateDecl("thin_file", 1, "observable", phrase="{0} has a thin file"),
            PredicateDecl("guarantor", 1, "observable", phrase="{0} has a guarantor"),
            PredicateDecl("large_amount", 1, "observable"),
            PredicateDecl("high_risk", 1, "derived"),
            PredicateDecl("decline", 1, "derived"),
        ]
    )


@pytest.fixture
def policy() -> CriticalityPolicy:
    return CriticalityPolicy(
        rules=[
            CriticalityRule("large_sums", "large amounts reviewed online",
                            "amount", "gte", 10_000)
        ],
        default_critical=False,
    )


@pytest.fixture
def db(tmp_path):
    return tmp_path / "logic.sqlite"


def facts_for(vocab: Vocabulary, entity: str, **kw: str) -> FactStore:
    store = FactStore(vocab)
    for predicate, state in kw.items():
        store.assert_fact(
            parse_atom(f"{predicate}({entity})"), Truth(state), source=f"{predicate} src"
        )
    return store


RISK_RULE = "high_risk(A) <- overdrawn(A), thin_file(A)."
DECLINE_RULE = "decline(A) <- high_risk(A), not guarantor(A)."


# --------------------------------------------------------------------------
# The core promise
# --------------------------------------------------------------------------


def test_approval_survives_restart(db, vocab, policy) -> None:
    """Approve once, restart, and nobody is asked again."""
    gov = open_governor(db, vocab, policy)
    proposal = gov.propose(parse_rule(RISK_RULE), RuleOrigin.MINED, "agent",
                           critical_context=True, strength=0.8)
    gov.approve(proposal.record.rule_id, by="reviewer-jd", note="checked s4.2")
    gov.store.close()

    reopened = open_governor(db, vocab, policy)
    restored = reopened.rulebase.get(proposal.record.rule_id)
    assert restored is not None
    assert restored.status is RuleStatus.APPROVED
    assert restored.approved_by == "reviewer-jd"
    assert reopened.review_queue() == [], "nothing should be awaiting review"

    # And re-proposing the same logic, spelled differently, is still settled.
    again = reopened.propose(
        parse_rule("high_risk(Z) <- thin_file(Z), overdrawn(Z)."),
        RuleOrigin.MINED, "agent", critical_context=True,
    )
    assert again.outcome.value == "already_approved"
    assert reopened.review_queue() == []
    reopened.store.close()


def test_critical_query_works_after_restart(db, vocab, policy) -> None:
    """A rule approved in one session governs critical queries in the next."""
    gov = open_governor(db, vocab, policy)
    for text in (RISK_RULE, DECLINE_RULE):
        p = gov.propose(parse_rule(text), RuleOrigin.SEED, "pack",
                        critical_context=True, strength=0.85)
        gov.approve(p.record.rule_id, by="policy-team")
    gov.store.close()

    reopened = open_governor(db, vocab, policy)
    facts = facts_for(reopened.vocabulary, "app1", overdrawn="true",
                      thin_file="true", guarantor="false")
    answer, query = reopened.ask(
        parse_atom("decline(app1)"), facts, "app1", "officer", {"amount": 50_000}
    )
    assert query.status == "answered"
    assert query.critical
    assert answer.bounds.lower > 0
    reopened.store.close()


def test_unapproved_rule_still_blocks_after_restart(db, vocab, policy) -> None:
    """Persistence must not accidentally launder a provisional rule into an approved one."""
    gov = open_governor(db, vocab, policy)
    gov.propose(parse_rule(RISK_RULE), RuleOrigin.MINED, "agent",
                critical_context=False, strength=0.8)
    gov.propose(parse_rule(DECLINE_RULE), RuleOrigin.MINED, "agent",
                critical_context=False, strength=0.85)
    gov.store.close()

    reopened = open_governor(db, vocab, policy)
    facts = facts_for(reopened.vocabulary, "app1", overdrawn="true",
                      thin_file="true", guarantor="false")
    with pytest.raises(BlockedForApproval):
        reopened.ask(parse_atom("decline(app1)"), facts, "app1", "officer",
                     {"amount": 50_000})
    reopened.store.close()


# --------------------------------------------------------------------------
# History reaches backwards
# --------------------------------------------------------------------------


def test_impact_analysis_spans_sessions(db, vocab, policy) -> None:
    """A rule rejected today must name decisions it affected in an earlier session."""
    gov = open_governor(db, vocab, policy)
    risk = gov.propose(parse_rule(RISK_RULE), RuleOrigin.MINED, "agent",
                       critical_context=False, strength=0.8)
    decline = gov.propose(parse_rule(DECLINE_RULE), RuleOrigin.SEED, "pack",
                          critical_context=True, strength=0.85)
    gov.approve(decline.record.rule_id, by="policy-team")
    for i in range(3):
        entity = f"app_{i}"
        facts = facts_for(gov.vocabulary, entity, overdrawn="true",
                          thin_file="true", guarantor="false")
        gov.ask(parse_atom(f"decline({entity})"), facts, entity, "officer",
                {"amount": 500})
    gov.store.close()

    reopened = open_governor(db, vocab, policy)
    report = reopened.reject(risk.record.rule_id, by="reviewer-jd",
                             reason="confounded with postcode")
    assert len(report.affected_queries) == 3, (
        "rejection must see decisions made before this process started"
    )
    reopened.store.close()


def test_query_log_reloads(db, vocab, policy) -> None:
    gov = open_governor(db, vocab, policy)
    p = gov.propose(parse_rule(RISK_RULE), RuleOrigin.MINED, "agent",
                    critical_context=True, strength=0.8)
    gov.approve(p.record.rule_id, by="jd")
    facts = facts_for(gov.vocabulary, "app1", overdrawn="true", thin_file="true")
    _, query = gov.ask(parse_atom("high_risk(app1)"), facts, "app1", "officer",
                       {"amount": 50_000})
    gov.store.close()

    reopened = open_governor(db, vocab, policy)
    assert query.query_id in reopened.queries
    restored = reopened.queries[query.query_id]
    assert restored.critical
    assert restored.rule_ids == query.rule_ids
    reopened.store.close()


def test_query_ids_do_not_collide_after_restart(db, vocab, policy) -> None:
    """Numbering resumes rather than overwriting earlier decisions."""
    gov = open_governor(db, vocab, policy)
    p = gov.propose(parse_rule(RISK_RULE), RuleOrigin.MINED, "agent",
                    critical_context=True, strength=0.8)
    gov.approve(p.record.rule_id, by="jd")
    ids = []
    for i in range(3):
        facts = facts_for(gov.vocabulary, f"a{i}", overdrawn="true", thin_file="true")
        _, q = gov.ask(parse_atom(f"high_risk(a{i})"), facts, f"a{i}", "officer",
                       {"amount": 50_000})
        ids.append(q.query_id)
    gov.store.close()

    reopened = open_governor(db, vocab, policy)
    facts = facts_for(reopened.vocabulary, "a9", overdrawn="true", thin_file="true")
    _, q = reopened.ask(parse_atom("high_risk(a9)"), facts, "a9", "officer",
                        {"amount": 50_000})
    assert q.query_id not in ids
    assert len(reopened.queries) == 4
    reopened.store.close()


def test_replay_uses_the_facts_the_decision_actually_saw(db, vocab, policy) -> None:
    """Auditing a past decision must not re-derive it against today's rule base."""
    gov = open_governor(db, vocab, policy)
    p = gov.propose(parse_rule(RISK_RULE), RuleOrigin.MINED, "agent",
                    critical_context=True, strength=0.8)
    gov.approve(p.record.rule_id, by="jd")
    facts = facts_for(gov.vocabulary, "app1", overdrawn="true", thin_file="true",
                      guarantor="unknown")
    _, query = gov.ask(parse_atom("high_risk(app1)"), facts, "app1", "officer",
                       {"amount": 50_000})
    gov.store.close()

    reopened = open_governor(db, vocab, policy)
    replayed = reopened.store.replay_facts(query.query_id, reopened.vocabulary)
    assert replayed.truth_of(parse_atom("overdrawn(app1)")) is Truth.TRUE
    assert replayed.truth_of(parse_atom("guarantor(app1)")) is Truth.UNKNOWN, (
        "an unknown must be replayed as unknown, not silently as false"
    )
    reopened.store.close()


# --------------------------------------------------------------------------
# Stats and the audit trail
# --------------------------------------------------------------------------


def test_reuse_and_evidence_survive_restart(db, vocab, policy) -> None:
    gov = open_governor(db, vocab, policy)
    p = gov.propose(parse_rule(RISK_RULE), RuleOrigin.MINED, "agent",
                    critical_context=True, strength=0.8)
    gov.approve(p.record.rule_id, by="jd")
    for i in range(5):
        facts = facts_for(gov.vocabulary, f"a{i}", overdrawn="true", thin_file="true")
        _, q = gov.ask(parse_atom(f"high_risk(a{i})"), facts, f"a{i}", "officer",
                       {"amount": 50_000})
        gov.record_outcome(q.query_id, correct=i < 4)
    gov.store.close()

    reopened = open_governor(db, vocab, policy)
    restored = reopened.rulebase.get(p.record.rule_id)
    assert restored.stats.reuse_count == 5
    assert (restored.stats.successes, restored.stats.failures) == (4, 1)
    assert reopened.store.amortisation()["rule_applications"] == 5
    reopened.store.close()


def test_rule_history_is_append_only(db, vocab, policy) -> None:
    """The governance record keeps every transition, not just the final state."""
    gov = open_governor(db, vocab, policy)
    p = gov.propose(parse_rule(RISK_RULE), RuleOrigin.MINED, "agent",
                    critical_context=False, strength=0.8)
    gov.propose(parse_rule("high_risk(Z) <- thin_file(Z), overdrawn(Z)."),
                RuleOrigin.MINED, "agent", critical_context=True)
    gov.approve(p.record.rule_id, by="reviewer-jd", note="fine")

    history = gov.store.rule_history(p.record.rule_id)
    events = [h["event"] for h in history]
    assert events == ["applied_provisionally", "escalated", "approved"]
    assert history[-1]["actor"] == "reviewer-jd"
    gov.store.close()


def test_rejection_is_recorded_with_its_reason(db, vocab, policy) -> None:
    gov = open_governor(db, vocab, policy)
    p = gov.propose(parse_rule(RISK_RULE), RuleOrigin.MINED, "agent",
                    critical_context=False, strength=0.8)
    gov.reject(p.record.rule_id, by="reviewer-jd", reason="precision too low")
    gov.store.close()

    reopened = open_governor(db, vocab, policy)
    restored = reopened.rulebase.get(p.record.rule_id)
    assert restored.status is RuleStatus.REJECTED
    assert restored.rejected_reason == "precision too low"
    again = reopened.propose(parse_rule(RISK_RULE), RuleOrigin.MINED, "agent",
                             critical_context=False)
    assert again.outcome.value == "already_rejected"
    reopened.store.close()


# --------------------------------------------------------------------------
# Identity is enforced by the schema
# --------------------------------------------------------------------------


def test_canonical_key_is_the_primary_key(db, vocab) -> None:
    """Approve-once is a schema constraint, not only an application check."""
    store = Store(db)
    store.save_vocabulary(vocab)
    from logicdb.program import RuleRecord

    first = RuleRecord(rule=parse_rule(RISK_RULE), rule_id="r0001", strength=0.8)
    store.save_rule(first)

    # Same logic, different spelling and different rule_id.
    second = RuleRecord(
        rule=parse_rule("high_risk(Q) <- thin_file(Q), overdrawn(Q)."),
        rule_id="r0002", strength=0.9,
    )
    store.save_rule(second)

    assert store.counts()["rules"] == 1, "one logical rule, one row"
    stored = store.find_by_canonical_key(first.key)
    assert stored["rule_id"] == "r0001", "the original identity is kept"
    store.close()


def test_load_detects_a_canonicalisation_change(db, vocab) -> None:
    """If canonicalisation ever changes, past approvals can no longer be matched.

    That would silently reopen settled questions, so it must fail loudly at load
    rather than quietly at review time.
    """
    store = Store(db)
    store.save_vocabulary(vocab)
    from logicdb.program import RuleRecord

    store.save_rule(RuleRecord(rule=parse_rule(RISK_RULE), rule_id="r0001", strength=0.8))
    store.conn.execute("UPDATE rules SET canonical_key = 'stale_key'")
    store.conn.commit()

    with pytest.raises(ValueError, match="canonical key"):
        store.load_rulebase(vocab)
    store.close()


# --------------------------------------------------------------------------
# Text export / import
# --------------------------------------------------------------------------


def test_export_round_trips_through_import(db, tmp_path, vocab, policy) -> None:
    gov = open_governor(db, vocab, policy)
    p = gov.propose(parse_rule(RISK_RULE), RuleOrigin.MINED, "agent",
                    critical_context=True, strength=0.8)
    gov.approve(p.record.rule_id, by="reviewer-jd")
    exported = gov.store.export_text()
    gov.store.close()

    assert "high_risk(A) <- overdrawn(A), thin_file(A)." in exported
    assert "status=approved" in exported

    fresh = Store(tmp_path / "fresh.sqlite")
    fresh.save_vocabulary(vocab)
    added = fresh.import_text(exported, vocab)
    assert len(added) == 1
    assert added[0].status is RuleStatus.APPROVED
    assert added[0].approved_by == "reviewer-jd"
    fresh.close()


def test_import_does_not_duplicate_existing_logic(db, vocab, policy) -> None:
    """Re-importing an edited file must not resurrect settled questions."""
    gov = open_governor(db, vocab, policy)
    p = gov.propose(parse_rule(RISK_RULE), RuleOrigin.MINED, "agent",
                    critical_context=True, strength=0.8)
    gov.approve(p.record.rule_id, by="jd")
    exported = gov.store.export_text()

    added = gov.store.import_text(exported, vocab)
    assert added == [], "already present under its canonical key"
    assert gov.store.counts()["rules"] == 1
    gov.store.close()


def test_import_rejects_undeclared_predicates(db, vocab) -> None:
    store = Store(db)
    store.save_vocabulary(vocab)
    with pytest.raises(ValueError, match="undeclared predicate"):
        store.import_text("rule r1 [p=0.5]\n  high_risk(A) <- invented(A).", vocab)
    store.close()


# --------------------------------------------------------------------------
# Vocabulary
# --------------------------------------------------------------------------


def test_vocabulary_round_trips(db, vocab) -> None:
    store = Store(db)
    store.save_vocabulary(vocab)
    loaded = store.load_vocabulary()
    assert len(loaded) == len(vocab)
    decl = loaded.get("overdrawn/1")
    assert decl.kind == "observable"
    assert decl.phrase == "{0} is overdrawn"
    store.close()


def test_database_opens_without_the_original_pack(db, vocab, policy) -> None:
    """An auditor with only the .db file must be able to read the rule base."""
    gov = open_governor(db, vocab, policy)
    p = gov.propose(parse_rule(RISK_RULE), RuleOrigin.MINED, "agent",
                    critical_context=True, strength=0.8)
    gov.approve(p.record.rule_id, by="jd")
    gov.store.close()

    store = Store(db)
    rulebase = store.load_rulebase()  # no vocabulary supplied
    assert len(rulebase) == 1
    assert rulebase.get(p.record.rule_id).approved_by == "jd"
    store.close()
