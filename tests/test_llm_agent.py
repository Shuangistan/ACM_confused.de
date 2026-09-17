"""Tests for the LLM agent backend.

These run without an API key by injecting a stub client. That is not a
convenience -- it is the point. What matters about this backend is not what
Claude says, it is that *whatever* it says is filtered before it can affect a
decision. Those filters are deterministic, so they can and should be tested
against adversarial output: invented predicates, malformed syntax, unsafe
rules, facts attributed to the wrong entity, overconfident strengths.

The scripted agent must also keep working with no key at all, so a missing
`ANTHROPIC_API_KEY` is tested as a clean error rather than an import-time crash.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from logicdb.agents.base import LabelledCase, validate_proposals
from logicdb.agents.llm import DEFAULT_MODEL, LLMAgent, LLMConfig, RefusalError
from logicdb.facts import Truth
from logicdb.packs import load_pack
from logicdb.parser import parse_atom
from logicdb.program import RuleOrigin


class StubClient:
    """Returns a canned payload and records the request that produced it."""

    def __init__(self, payload=None, stop_reason="end_turn", stop_details=None):
        self._payload = payload if payload is not None else {"proposals": []}
        self._stop_reason = stop_reason
        self._stop_details = stop_details
        self.calls = []
        self.beta = SimpleNamespace(messages=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(
            stop_reason=self._stop_reason,
            stop_details=self._stop_details,
            content=[SimpleNamespace(type="text", text=json.dumps(self._payload))],
        )


@pytest.fixture(scope="module")
def pack():
    return load_pack("consumer_credit")


def agent_with(pack, payload, config=None, **kw):
    client = StubClient(payload, **kw)
    return LLMAgent(pack, config, client=client), client


def _a_case(pack):
    """One labelled case -- `mine_rules` returns early on an empty list."""
    return LabelledCase("c1", pack.facts_from({"id": "c1"}, "c1"), "decline")


# --------------------------------------------------------------------------
# Request shape
# --------------------------------------------------------------------------


def test_uses_current_model_and_no_sampling_params(pack) -> None:
    """Sampling parameters are rejected on this model; thinking is adaptive."""
    agent, client = agent_with(pack, {"proposals": []})
    agent.mine_rules([LabelledCase("c1", pack.facts_from({"id": "c1"}, "c1"), "decline")],
                     target="decline")
    (call,) = client.calls
    assert call["model"] == DEFAULT_MODEL == "claude-opus-5"
    assert call["thinking"] == {"type": "adaptive"}
    for banned in ("temperature", "top_p", "top_k", "budget_tokens"):
        assert banned not in call


def test_output_is_schema_constrained(pack) -> None:
    """A schema failure should happen at the API boundary, not in our parser."""
    agent, client = agent_with(pack, {"proposals": []})
    agent.extract_from_document("p.md", "Some policy text.", pack.vocabulary)
    (call,) = client.calls
    fmt = call["output_config"]["format"]
    assert fmt["type"] == "json_schema"
    assert "proposals" in fmt["schema"]["properties"]
    assert fmt["schema"]["additionalProperties"] is False


def test_prompt_carries_the_declared_vocabulary(pack) -> None:
    """The model cannot be expected to stay in vocabulary it was never shown."""
    agent, client = agent_with(pack, {"proposals": []})
    agent.extract_from_document("p.md", "text", pack.vocabulary)
    prompt = client.calls[0]["messages"][0]["content"]
    assert "overdrawn/1" in prompt
    assert "observable" in prompt and "derived" in prompt


# --------------------------------------------------------------------------
# Filtering adversarial output -- the load-bearing tests
# --------------------------------------------------------------------------


def test_invented_predicates_never_reach_a_reviewer(pack) -> None:
    """The failure mode a language model is famous for, handled structurally."""
    agent, _ = agent_with(pack, {"proposals": [
        {"rule": "decline(A) <- vibes_bad(A).", "strength": 0.9,
         "rationale": "hallucinated", "citation": "s1"},
        {"rule": "decline(A) <- overdrawn(A), thin_savings(A).", "strength": 0.8,
         "rationale": "real", "citation": "s4.1"},
    ]})
    proposals = agent.extract_from_document("p.md", "text", pack.vocabulary)
    accepted, rejected = validate_proposals(proposals, pack.vocabulary)

    assert [str(p.rule) for p in accepted] == [
        "decline(A) <- overdrawn(A), thin_savings(A)."
    ]
    assert rejected and "undeclared predicate" in rejected[0][1][0]


def test_malformed_syntax_is_dropped_not_raised(pack) -> None:
    """One bad draft must not discard the good ones alongside it."""
    agent, _ = agent_with(pack, {"proposals": [
        {"rule": "decline(A) <- ((((", "strength": 0.8, "rationale": "x", "citation": ""},
        {"rule": "decline(A) <- overdrawn(A).", "strength": 0.8,
         "rationale": "ok", "citation": ""},
    ]})
    proposals = agent.extract_from_document("p.md", "text", pack.vocabulary)
    assert len(proposals) == 1


def test_unsafe_rules_are_dropped(pack) -> None:
    """An unbound head variable ranges over nothing and must not enter the base."""
    agent, _ = agent_with(pack, {"proposals": [
        {"rule": "decline(A) <- overdrawn(B).", "strength": 0.8,
         "rationale": "unsafe", "citation": ""},
    ]})
    assert agent.extract_from_document("p.md", "t", pack.vocabulary) == []


def test_rule_concluding_an_observable_is_rejected(pack) -> None:
    """Otherwise 'measured' and 'inferred' blur, and provenance with them."""
    agent, _ = agent_with(pack, {"proposals": [
        {"rule": "overdrawn(A) <- thin_savings(A).", "strength": 0.8,
         "rationale": "invalid", "citation": ""},
    ]})
    proposals = agent.extract_from_document("p.md", "t", pack.vocabulary)
    accepted, rejected = validate_proposals(proposals, pack.vocabulary)
    assert not accepted and rejected


def test_strength_is_clamped(pack) -> None:
    """A model-supplied number must not be able to break the rule invariant."""
    agent, _ = agent_with(pack, {"proposals": [
        {"rule": "decline(A) <- overdrawn(A).", "strength": 4.2,
         "rationale": "overconfident", "citation": ""},
    ]})
    (p,) = agent.extract_from_document("p.md", "t", pack.vocabulary)
    assert 0.0 < p.strength <= 1.0


def test_missing_trailing_period_is_tolerated(pack) -> None:
    agent, _ = agent_with(pack, {"proposals": [
        {"rule": "decline(A) <- overdrawn(A)", "strength": 0.7,
         "rationale": "no full stop", "citation": ""},
    ]})
    assert len(agent.extract_from_document("p.md", "t", pack.vocabulary)) == 1


def test_already_known_rules_are_not_re_proposed(pack) -> None:
    """Approve-once reaches the LLM path too."""
    from logicdb.parser import parse_rule

    known = {parse_rule("decline(A) <- overdrawn(A).").canonical_key()}
    agent, _ = agent_with(pack, {"proposals": [
        {"rule": "decline(Z) <- overdrawn(Z).", "strength": 0.8,
         "rationale": "duplicate", "citation": ""},
    ]})
    cases = [LabelledCase("c1", pack.facts_from({"id": "c1"}, "c1"), "decline")]
    assert agent.mine_rules(cases, target="decline", existing=known) == []


# --------------------------------------------------------------------------
# Provenance
# --------------------------------------------------------------------------


def test_document_proposals_keep_their_citation(pack) -> None:
    agent, _ = agent_with(pack, {"proposals": [
        {"rule": "decline(A) <- overdrawn(A).", "strength": 0.8,
         "rationale": "encodes s4.1", "citation": "lending_policy.md s4.1"},
    ]})
    (p,) = agent.extract_from_document("lending_policy.md", "t", pack.vocabulary)
    assert p.origin is RuleOrigin.DOCUMENT
    assert p.source_citation == "lending_policy.md s4.1"


def test_mined_proposals_carry_no_measured_evidence(pack) -> None:
    """The model did not count anything, and the proposal must not imply it did.

    The scripted miner reports support and precision because it measured them.
    This backend reasons instead, so those fields stay empty and a reviewer can
    see at a glance which kind of claim they are being asked to approve.
    """
    agent, _ = agent_with(pack, {"proposals": [
        {"rule": "decline(A) <- overdrawn(A).", "strength": 0.8,
         "rationale": "pattern in the sample", "citation": "n/a"},
    ]})
    cases = [LabelledCase("c1", pack.facts_from({"id": "c1"}, "c1"), "decline")]
    (p,) = agent.mine_rules(cases, target="decline")
    assert p.origin is RuleOrigin.MINED
    assert p.support is None and p.precision is None
    assert p.source_citation is None


# --------------------------------------------------------------------------
# Fact extraction from text
# --------------------------------------------------------------------------


def test_extracts_facts_with_unknown_preserved(pack) -> None:
    agent, _ = agent_with(pack, {"facts": [
        {"atom": "overdrawn(app_1)", "truth": "true", "source": "account is overdrawn"},
        {"atom": "guarantor(app_1)", "truth": "unknown", "source": "not mentioned"},
    ]})
    store = agent.extract_facts_from_text("...", "app_1")
    assert store.truth_of(parse_atom("overdrawn(app_1)")) is Truth.TRUE
    assert store.truth_of(parse_atom("guarantor(app_1)")) is Truth.UNKNOWN


def test_facts_about_the_wrong_entity_are_dropped(pack) -> None:
    """A fact silently attributed to the wrong applicant is worse than a gap."""
    agent, _ = agent_with(pack, {"facts": [
        {"atom": "overdrawn(someone_else)", "truth": "true", "source": "x"},
    ]})
    store = agent.extract_facts_from_text("...", "app_1")
    assert len(store) == 0


def test_undeclared_fact_predicates_are_dropped(pack) -> None:
    agent, _ = agent_with(pack, {"facts": [
        {"atom": "invented(app_1)", "truth": "true", "source": "x"},
    ]})
    assert len(agent.extract_facts_from_text("...", "app_1")) == 0


def test_structured_records_bypass_the_model(pack) -> None:
    """Where a deterministic table exists, using a model would lose auditability."""
    agent, client = agent_with(pack, {"facts": []})
    store = agent.extract_facts({"id": "app_1", "overdrawn": True}, "app_1")
    assert client.calls == [], "extract_facts must not call the API"
    assert store.truth_of(parse_atom("overdrawn(app_1)")) is Truth.TRUE


# --------------------------------------------------------------------------
# Refusals and configuration
# --------------------------------------------------------------------------


def test_refusal_is_raised_not_silently_empty(pack) -> None:
    """An empty proposal list and a refusal are very different states."""
    agent, _ = agent_with(
        pack, {"proposals": []}, stop_reason="refusal",
        stop_details=SimpleNamespace(category="cyber"),
    )
    with pytest.raises(RefusalError, match="declined"):
        agent.extract_from_document("p.md", "t", pack.vocabulary)


def test_fallback_is_opt_in(pack) -> None:
    agent, client = agent_with(pack, {"proposals": []})
    agent.extract_from_document("p.md", "t", pack.vocabulary)
    assert "extra_body" not in client.calls[0]

    client2 = StubClient({"proposals": []})
    opted = LLMAgent(pack, LLMConfig(server_side_fallback=True), client=client2)
    opted.extract_from_document("p.md", "t", pack.vocabulary)
    assert client2.calls[0]["extra_body"] == {"fallbacks": "default"}


def test_missing_api_key_is_a_clean_error(pack, monkeypatch) -> None:
    """The scripted agent is the default and must work with no key at all."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
        _ = LLMAgent(pack).client


def test_both_backends_satisfy_one_protocol(pack) -> None:
    """Swapping backends changes where proposals come from, nothing else."""
    from logicdb.agents.base import Agent
    from logicdb.agents.scripted import ScriptedAgent

    assert isinstance(ScriptedAgent(pack), Agent)
    assert isinstance(LLMAgent(pack), Agent)


# ---------------------------------------------------------------------------
# Citations
#
# Observed on a live run against claude-haiku-4-5: four of six drafted rules
# cited "Implicit in screening practice" -- text appearing nowhere in the
# source document. A fabricated citation is worse than a missing one, because
# `document` is the strongest provenance tier precisely on the grounds that a
# reviewer can check it, and an uncheckable citation spends that trust without
# earning it.
# ---------------------------------------------------------------------------
DOC = """## s1.1 Experience
An applicant who does not meet the experience requirement is unsuitable.

## s1.2 Right to work
An applicant without the right to work is unsuitable.
"""


def _doc_agent(pack, proposals):
    return LLMAgent(pack, client=StubClient({"proposals": proposals}))


def test_a_citation_naming_real_text_is_kept(pack) -> None:
    agent = _doc_agent(pack, [{
        "rule": "requirements_unmet(A) <- applicant(A), not right_to_work(A).",
        "strength": 0.9, "citation": "s1.2", "rationale": "as written",
    }])
    got = agent.extract_from_document("policy.md", DOC, pack.vocabulary)
    assert len(got) == 1
    assert got[0].citation_verified is True
    assert got[0].source_citation == "s1.2"
    assert agent.unverified_citations == []


def test_a_trailing_gloss_still_counts_as_locatable(pack) -> None:
    agent = _doc_agent(pack, [{
        "rule": "requirements_unmet(A) <- applicant(A), not right_to_work(A).",
        "strength": 0.9, "citation": "s1.2 (contrapositive)",
        "rationale": "restated",
    }])
    got = agent.extract_from_document("policy.md", DOC, pack.vocabulary)
    assert got[0].citation_verified is True


def test_a_fabricated_citation_is_flagged_not_trusted(pack) -> None:
    agent = _doc_agent(pack, [{
        "rule": "requirements_unmet(A) <- applicant(A), not right_to_work(A).",
        "strength": 0.9, "citation": "Implicit in screening practice",
        "rationale": "seemed reasonable",
    }])
    got = agent.extract_from_document("policy.md", DOC, pack.vocabulary)
    assert len(got) == 1, "the rule survives; only its provenance is downgraded"
    assert got[0].citation_verified is False
    assert agent.unverified_citations, "the reviewer must be told"
    assert "NOT FOUND" in got[0].describe()


def test_mined_rules_are_not_subject_to_citation_checks(pack) -> None:
    """There is no document to check against, so nothing is flagged."""
    agent = LLMAgent(pack, client=StubClient({"proposals": [{
        "rule": "requirements_unmet(A) <- applicant(A), not right_to_work(A).",
        "strength": 0.8, "citation": "from the data", "rationale": "mined",
    }]}))
    got = agent.mine_rules([_a_case(pack)], target="decline")
    assert got[0].citation_verified is True
    assert got[0].source_citation is None


# ---------------------------------------------------------------------------
# Request shape follows the model
#
# `effort` and adaptive thinking are 400 errors on pre-4.6 models rather than
# ignored fields, so sending the newest shape everywhere makes the cheap
# backends unusable.
# ---------------------------------------------------------------------------
def test_legacy_models_drop_effort_and_thinking(pack) -> None:
    agent, client = agent_with(pack, {"proposals": []},
                               config=LLMConfig(model="claude-haiku-4-5"))
    agent.mine_rules([_a_case(pack)], target="decline")
    (sent,) = client.calls
    assert "thinking" not in sent
    assert "effort" not in sent["output_config"]


def test_current_models_keep_effort_and_thinking(pack) -> None:
    agent, client = agent_with(pack, {"proposals": []},
                               config=LLMConfig(model="claude-opus-5"))
    agent.mine_rules([_a_case(pack)], target="decline")
    (sent,) = client.calls
    assert sent["thinking"] == {"type": "adaptive"}
    assert sent["output_config"]["effort"] == "high"


def test_agent_id_names_the_model_actually_used(pack) -> None:
    agent = LLMAgent(pack, LLMConfig(model="claude-haiku-4-5"))
    assert "haiku" in agent.agent_id
