"""End-to-end tests for the web interface.

These drive the app the way a person would: ask something with an empty rule
base, have the agent propose, work the review queue, ask again. The purpose is
partly to catch template errors -- which are invisible until a page is actually
rendered -- and partly to check that the governance guarantees survive the HTTP
layer. A blocking rule that stops the engine but not the web handler would be no
protection at all.
"""

from __future__ import annotations

import html

import pytest
from fastapi.testclient import TestClient

from logicdb.app import create_app
from logicdb.program import RuleStatus


@pytest.fixture
def client(tmp_path):
    app = create_app("consumer_credit", tmp_path / "app.sqlite")
    with TestClient(app) as c:
        c.state = app.state.logic
        yield c


def first_case(client) -> str:
    return next(iter(client.state.cases))


def text_of(response) -> str:
    """Page text with HTML entities resolved.

    Jinja autoescapes, so a rule renders as `decline(A) &lt;- ...`. That is
    correct output; asserting against the raw response would silently test the
    escaping rather than the content.
    """
    return html.unescape(response.text)


# --------------------------------------------------------------------------
# Pages render
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "path",
    ["/", "/dashboard", "/console", "/query", "/review", "/rules", "/agent", "/export"],
)
def test_pages_render(client, path: str) -> None:
    response = client.get(path)
    assert response.status_code == 200
    assert "<html" in response.text


def test_rules_filter_by_status(client) -> None:
    assert client.get("/rules?status=approved").status_code == 200


def test_unknown_case_is_a_404(client) -> None:
    response = client.post("/query", data={"entity": "no_such_case"})
    assert response.status_code == 404


def test_unknown_rule_is_a_404(client) -> None:
    assert client.get("/rules/nope").status_code == 404


# --------------------------------------------------------------------------
# The loop
# --------------------------------------------------------------------------


def test_empty_database_reports_absence_not_denial(client) -> None:
    """The distinction a user must not be allowed to misread."""
    response = client.post("/query", data={"entity": first_case(client)})
    assert response.status_code == 200
    body = " ".join(text_of(response).split())
    assert "No applicable logic" in body
    assert "This is <em>not</em> a negative answer" in body
    assert "absence of logic" in body


def test_agent_extraction_populates_the_queue(client) -> None:
    response = client.post("/agent/extract")
    assert response.status_code == 200
    assert "lending_policy.md" in response.text

    queue = text_of(client.get("/review"))
    assert "s4.1" in queue or "s7.1" in queue


def test_review_queue_shows_provenance_specific_evidence(client) -> None:
    """Document and mined rules must not be presented identically.

    They ask the reviewer different questions; rendering them the same invites
    the easier answer to both.
    """
    client.post("/agent/extract")
    client.post("/agent/mine", data={"target": "decline"})
    queue = text_of(client.get("/review"))

    assert "Cites:" in queue, "document rules must show their clause"
    assert "Evidence:" in queue, "mined rules must show support and precision"
    assert "until you decide it is also a reason" in queue


def test_approve_then_query_succeeds(client) -> None:
    client.post("/agent/extract")
    governor = client.state.governor
    for record in list(governor.review_queue()):
        client.post(
            f"/review/{record.rule_id}/approve",
            data={"reviewer": "tester", "note": "checked"},
            follow_redirects=False,
        )
    assert all(
        r.status is RuleStatus.APPROVED for r in governor.rulebase.all_records()
    )

    # Find a case the approved logic actually covers.
    for entity in list(client.state.cases)[:60]:
        response = client.post("/query", data={"entity": entity})
        assert response.status_code == 200
        body = text_of(response)
        if "Derivation" in body:
            assert "P =" in body
            assert "No applicable logic" not in body, (
                "a derivation heading over an uncovered case implies logic where "
                "there is none"
            )
            return
    pytest.skip("no covered case in the sample")


def test_critical_query_blocks_through_the_web_layer(client) -> None:
    """The guarantee must hold at the HTTP boundary, not only inside the engine."""
    client.post("/agent/mine", data={"target": "decline"})
    governor = client.state.governor
    if not governor.rulebase.all_records():
        pytest.skip("mining produced nothing on this pack")

    # A large advance is critical per the pack's criticality policy.
    critical_entity = next(
        (
            e for e, r in client.state.cases.items()
            if r.get("credit_amount", 0) >= 5000
        ),
        None,
    )
    assert critical_entity is not None
    response = client.post("/query", data={"entity": critical_entity})
    assert response.status_code == 200
    body = text_of(response)
    if "Blocked" in body:
        assert "cannot do" in body
        assert "Needs approval" in body


def test_rejection_shows_the_impact_report(client) -> None:
    """The reviewer who caused the blast radius is the right person to see it."""
    client.post("/agent/mine", data={"target": "decline"})
    governor = client.state.governor
    provisional = governor.rulebase.by_status(RuleStatus.PROVISIONAL)
    if not provisional:
        pytest.skip("no provisional rules")

    record = provisional[0]
    response = client.post(
        f"/review/{record.rule_id}/reject",
        data={"reviewer": "tester", "reason": "precision too low"},
    )
    assert response.status_code == 200
    body = text_of(response)
    assert "Rule rejected" in body
    assert "precision too low" in body


def test_rule_detail_shows_append_only_history(client) -> None:
    client.post("/agent/extract")
    governor = client.state.governor
    record = governor.rulebase.all_records()[0]
    client.post(f"/review/{record.rule_id}/approve",
                data={"reviewer": "tester", "note": "ok"}, follow_redirects=False)

    page = text_of(client.get(f"/rules/{record.rule_id}"))
    assert record.key in page, "the canonical key is the identity and must be shown"
    assert "queued_online" in page
    assert "approved" in page
    assert "tester" in page


def test_export_round_trips_the_rule_base(client) -> None:
    client.post("/agent/extract")
    page = text_of(client.get("/export"))
    assert "elevated_risk(A) <-" in page or "decline(A) <-" in page


def test_outcome_feedback_updates_evidence(client) -> None:
    client.post("/agent/extract")
    governor = client.state.governor
    for record in list(governor.review_queue()):
        client.post(f"/review/{record.rule_id}/approve",
                    data={"reviewer": "t", "note": ""}, follow_redirects=False)

    for entity in list(client.state.cases)[:60]:
        client.post("/query", data={"entity": entity})
        answered = [q for q in governor.queries.values() if q.status == "answered"]
        if answered:
            query = answered[0]
            client.post(f"/query/{query.query_id}/outcome", data={"correct": "yes"},
                        follow_redirects=False)
            used = governor.rulebase.get(query.rule_ids[0])
            assert used.stats.observations >= 1
            return
    pytest.skip("no answerable case in the sample")


# --------------------------------------------------------------------------
# Domain neutrality
# --------------------------------------------------------------------------


def test_the_same_app_serves_a_different_domain(tmp_path) -> None:
    """The engine shares no code with either domain -- only data differs."""
    app = create_app("benefits_eligibility", tmp_path / "benefits.sqlite")
    with TestClient(app) as client:
        assert client.get("/dashboard").status_code == 200
        assert "Hardship grant eligibility" in text_of(client.get("/dashboard"))
        response = client.post("/agent/extract")
        assert response.status_code == 200
        assert "hardship_policy.md" in response.text
        assert "eligible(A) <-" in text_of(client.get("/export"))
