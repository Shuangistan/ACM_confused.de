"""Tests for domain packs and the scripted agent.

The rule-recovery tests are the ones with teeth. On real data you can only ask
whether a mined rule predicts well; the synthetic pack is generated from a known
rule set that is then withheld, so here you can ask whether the mined rule is
*the right rule* -- which is the question a reviewer is actually being asked when
they approve one.

That evaluation also quantifies why review is needed at all: the miner surfaces
the true rules at the top of its ranking, and several plausible near-misses below
them. A reviewer reading the ranking gets the truth; a reviewer rubber-stamping
the queue gets the near-misses too.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from logicdb.agents.base import LabelledCase, RuleProposal, validate_proposals
from logicdb.agents.scripted import MiningConfig, ScriptedAgent
from logicdb.facts import Truth
from logicdb.packs import FeatureMapping, available_packs, load_pack
from logicdb.parser import parse_atom, parse_rule
from logicdb.program import RuleOrigin

PACKS = Path(__file__).parent.parent / "packs"
CASES = PACKS / "benefits_eligibility" / "cases.json"


@pytest.fixture(scope="module")
def pack():
    return load_pack("benefits_eligibility")


@pytest.fixture(scope="module")
def cases(pack) -> list[LabelledCase]:
    if not CASES.exists():
        pytest.skip("run packs/benefits_eligibility/generate.py first")
    raw = json.loads(CASES.read_text())
    return [
        LabelledCase(r["id"], pack.facts_from(r, r["id"]), pack.label_of(r), r)
        for r in raw
    ]


@pytest.fixture(scope="module")
def true_rule_keys() -> dict[str, str]:
    sys.path.insert(0, str(PACKS / "benefits_eligibility"))
    from generate import TRUE_RULES

    keys = {}
    for name, required, forbidden in TRUE_RULES:
        body = ", ".join(
            [f"{p}(A)" for p in required] + [f"not {f}(A)" for f in forbidden]
        )
        keys[parse_rule(f"eligible(A) <- {body}.").canonical_key()] = name
    return keys


# --------------------------------------------------------------------------
# Packs
# --------------------------------------------------------------------------


def test_packs_are_discoverable() -> None:
    assert "benefits_eligibility" in available_packs()


def test_pack_declares_its_vocabulary(pack) -> None:
    assert pack.vocabulary.get("low_income/1").kind == "observable"
    assert pack.vocabulary.get("eligible/1").kind == "derived"


def test_extraction_records_the_source_field(pack) -> None:
    """A proof must be traceable past the logic into the source data."""
    facts = pack.facts_from({"id": "c1", "low_income": True}, "c1")
    record = facts.record_of(parse_atom("low_income(c1)"))
    assert record.truth is Truth.TRUE
    assert "low_income" in record.source


def test_missing_field_becomes_unknown_not_false(pack) -> None:
    """The distinction the whole three-valued design exists to preserve."""
    facts = pack.facts_from({"id": "c1", "low_income": True}, "c1")
    assert facts.truth_of(parse_atom("savings_over_limit(c1)")) is Truth.UNKNOWN


def test_explicit_none_is_also_unknown(pack) -> None:
    facts = pack.facts_from({"id": "c1", "savings_over_limit": None}, "c1")
    assert facts.truth_of(parse_atom("savings_over_limit(c1)")) is Truth.UNKNOWN


def test_feature_mapping_operators() -> None:
    gte = FeatureMapping(predicate="p", source="amount", op="gte", value=1000)
    assert gte.evaluate({"amount": 5000}) is Truth.TRUE
    assert gte.evaluate({"amount": 100}) is Truth.FALSE
    assert gte.evaluate({}) is Truth.UNKNOWN

    member = FeatureMapping(predicate="p", source="code", op="in", value=["A11", "A12"])
    assert member.evaluate({"code": "A11"}) is Truth.TRUE
    assert member.evaluate({"code": "A14"}) is Truth.FALSE


def test_mismatched_type_falls_back_to_missing() -> None:
    """A malformed field must not crash extraction nor read as a confident false."""
    mapping = FeatureMapping(predicate="p", source="amount", op="gte", value=1000)
    assert mapping.evaluate({"amount": "not a number"}) is Truth.UNKNOWN


def test_pack_validation_rejects_undeclared_feature_target(tmp_path) -> None:
    import yaml

    bad = tmp_path / "broken"
    bad.mkdir()
    (bad / "pack.yaml").write_text(
        yaml.safe_dump(
            {
                "domain_id": "broken",
                "decision": {"goal_predicate": "g", "outcomes": ["g"]},
                "predicates": [{"name": "g", "arity": 1, "kind": "derived"}],
                "features": [{"predicate": "nonexistent", "from": "x"}],
            }
        )
    )
    with pytest.raises(ValueError, match="undeclared predicate"):
        load_pack("broken", tmp_path)


def test_criticality_comes_from_the_pack(pack) -> None:
    critical, reason = pack.criticality.assess({"award_amount": 3000})
    assert critical and "large_award" in reason

    routine, _ = pack.criticality.assess({"award_amount": 250, "appeal_waived": False})
    assert not routine

    no_appeal, reason = pack.criticality.assess(
        {"award_amount": 250, "appeal_waived": True}
    )
    assert no_appeal and "no_appeal_route" in reason


# --------------------------------------------------------------------------
# Rule recovery -- the evaluation a real dataset cannot provide
# --------------------------------------------------------------------------


def test_miner_recovers_every_hidden_rule(pack, cases, true_rule_keys) -> None:
    """All three generating rules must be recovered exactly, not approximately."""
    proposals = ScriptedAgent(pack).mine_rules(cases, target="eligible")
    found = {p.rule.canonical_key() for p in proposals} & set(true_rule_keys)
    missing = [true_rule_keys[k] for k in set(true_rule_keys) - found]
    assert not missing, f"failed to recover: {missing}"


def test_true_rules_rank_at_the_top(pack, cases, true_rule_keys) -> None:
    """Ranking is what a reviewer relies on, so it has to put truth first."""
    proposals = ScriptedAgent(pack).mine_rules(cases, target="eligible")
    top_three = {p.rule.canonical_key() for p in proposals[:3]}
    assert top_three == set(true_rule_keys)


def test_miner_does_not_propose_rules_over_noise_attributes(pack, cases) -> None:
    """`owns_bicycle` and `urban_postcode` bear on nothing by construction."""
    proposals = ScriptedAgent(pack).mine_rules(cases, target="eligible")
    noise = {"owns_bicycle", "urban_postcode"}
    for proposal in proposals:
        used = {lit.atom.predicate for lit in proposal.rule.body}
        assert not (used & noise), f"proposed a rule over noise: {proposal.rule}"


def test_negated_literals_are_searched(pack, cases) -> None:
    """Without negation the capital-limit condition cannot be expressed at all.

    A miner restricted to positive conjunctions approximates such a rule with
    whatever correlates instead, handing the reviewer something wrong in a way
    the evidence cannot reveal.
    """
    proposals = ScriptedAgent(pack).mine_rules(cases, target="eligible")
    assert any(
        any(lit.negated for lit in p.rule.body) for p in proposals
    ), "no proposal uses negation"


def test_proposals_carry_reviewable_evidence(pack, cases) -> None:
    """A reviewer shown only the clause has no basis to vouch for it."""
    proposals = ScriptedAgent(pack).mine_rules(cases, target="eligible")
    for proposal in proposals:
        assert proposal.support is not None and proposal.support > 0
        assert 0.0 < proposal.precision <= 1.0
        assert proposal.examples, "must show cases the rule fires on"
        assert proposal.origin is RuleOrigin.MINED
        assert "base rate" in proposal.rationale


def test_mining_is_deterministic(pack, cases) -> None:
    agent = ScriptedAgent(pack)
    first = [str(p.rule) for p in agent.mine_rules(cases, target="eligible")]
    second = [str(p.rule) for p in agent.mine_rules(cases, target="eligible")]
    assert first == second


def test_already_known_rules_are_not_re_proposed(pack, cases) -> None:
    """Approve-once reaches into mining: settled logic is never re-queued."""
    agent = ScriptedAgent(pack)
    first = agent.mine_rules(cases, target="eligible")
    known = {p.rule.canonical_key() for p in first}
    again = agent.mine_rules(cases, target="eligible", existing=known)
    assert not ({p.rule.canonical_key() for p in again} & known)


def test_thresholds_suppress_weak_rules(pack, cases) -> None:
    strict = ScriptedAgent(pack, MiningConfig(min_precision=0.93, min_support=50))
    proposals = strict.mine_rules(cases, target="eligible")
    assert all(p.precision >= 0.93 and p.support >= 50 for p in proposals)


def test_mining_returns_nothing_without_cases(pack) -> None:
    assert ScriptedAgent(pack).mine_rules([], target="eligible") == []


def test_unknown_premises_do_not_count_against_a_rule(pack) -> None:
    """A rule must not be penalised for the data being incompletely recorded.

    Counting unknowns as misses would bias the search toward rules over
    well-recorded fields regardless of whether those are the right fields.
    """
    def build(prefix: str, n: int, label: str, **fields) -> list[LabelledCase]:
        return [
            LabelledCase(
                f"{prefix}{i}",
                pack.facts_from({"id": f"{prefix}{i}", **fields}, f"{prefix}{i}"),
                label,
            )
            for i in range(n)
        ]

    # Every predicate in the target rule has to vary, or it carries no
    # information and is excluded from the search before scoring even happens.
    dataset = (
        # The rule holds: all three premises known and satisfied.
        build("hit", 40, "eligible",
              low_income=True, resident=True, savings_over_limit=False)
        # The capital limit is unrecorded. These must be excluded from the
        # rule's support, not counted as failures.
        + build("gap", 25, "refer_to_officer", low_income=True, resident=True)
        + build("poor", 30, "refer_to_officer",
                low_income=False, resident=True, savings_over_limit=False)
        + build("rich", 20, "refer_to_officer",
                low_income=True, resident=True, savings_over_limit=True)
        + build("away", 15, "refer_to_officer",
                low_income=True, resident=False, savings_over_limit=False)
    )

    agent = ScriptedAgent(pack, MiningConfig(min_support=10, min_lift=0.0))
    proposals = agent.mine_rules(dataset, target="eligible")
    target = parse_rule(
        "eligible(A) <- low_income(A), resident(A), not savings_over_limit(A)."
    ).canonical_key()
    match = next((p for p in proposals if p.rule.canonical_key() == target), None)
    assert match is not None
    assert match.support == 40, "the 25 cases with an unknown premise must be excluded"
    assert match.precision == pytest.approx(1.0), (
        "counting the unknowns as failures would give 40/65 = 0.62 and bias the "
        "search toward rules over well-recorded fields"
    )


# --------------------------------------------------------------------------
# Document extraction
# --------------------------------------------------------------------------


def test_document_extraction_recovers_the_policy_rules(pack, true_rule_keys) -> None:
    """Rules traceable to written authority should be exactly right."""
    agent = ScriptedAgent(pack)
    proposals = []
    for name, text in pack.document_text().items():
        proposals.extend(agent.extract_from_document(name, text, pack.vocabulary))
    found = {p.rule.canonical_key() for p in proposals}
    assert set(true_rule_keys).issubset(found)


def test_extracted_rules_cite_their_clause(pack) -> None:
    """The reviewer's job on a document rule is verification, which needs a pointer."""
    agent = ScriptedAgent(pack)
    for name, text in pack.document_text().items():
        for proposal in agent.extract_from_document(name, text, pack.vocabulary):
            assert proposal.origin is RuleOrigin.DOCUMENT
            assert proposal.source_citation
            assert name in proposal.source_citation
            assert "s" in proposal.source_citation


def test_malformed_annotation_is_skipped_not_fatal(pack) -> None:
    """One bad clause must not block extraction of the rest of the document."""
    text = (
        "## s1 Good\n<!-- rule: eligible(A) <- low_income(A). -->\n"
        "## s2 Broken\n<!-- rule: eligible(A) <- ( -->\n"
        "## s3 Also good\n<!-- rule: refer_to_officer(A) <- recent_award(A). -->\n"
    )
    proposals = ScriptedAgent(pack).extract_from_document("t.md", text, pack.vocabulary)
    assert len(proposals) == 2


# --------------------------------------------------------------------------
# The proposal gate
# --------------------------------------------------------------------------


def test_invented_predicates_are_filtered_before_review(pack) -> None:
    """The load-bearing filter for an LLM backend: hallucinations never reach a human."""
    good = RuleProposal(
        rule=parse_rule("eligible(A) <- low_income(A)."),
        origin=RuleOrigin.MINED, strength=0.8, rationale="ok",
    )
    bad = RuleProposal(
        rule=parse_rule("eligible(A) <- vibes(A)."),
        origin=RuleOrigin.MINED, strength=0.8, rationale="hallucinated",
    )
    accepted, rejected = validate_proposals([good, bad], pack.vocabulary)
    assert [p.rule for p in accepted] == [good.rule]
    assert rejected and "undeclared predicate" in rejected[0][1][0]


def test_agent_cannot_conclude_an_observable(pack) -> None:
    """Otherwise 'measured' and 'inferred' blur, and provenance with them."""
    proposal = RuleProposal(
        rule=parse_rule("low_income(A) <- arrears(A)."),
        origin=RuleOrigin.MINED, strength=0.8, rationale="invalid",
    )
    accepted, rejected = validate_proposals([proposal], pack.vocabulary)
    assert not accepted and rejected
