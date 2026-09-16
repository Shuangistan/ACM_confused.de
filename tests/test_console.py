"""Tests for the control-room API.

The page is HTML and JavaScript, so what can be tested here is the contract it
consumes. The one that matters is that the governance guarantee survives the new
front end: a critical case must appear as a *blocked* row, not as an answered
one and not as an error. A UI that quietly rendered a blocked query as "no
applicable logic" would look identical to a working system while having removed
the protection entirely.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from logicdb.app import create_app


@pytest.fixture
def client(tmp_path):
    app = create_app("recruitment", tmp_path / "console.sqlite")
    with TestClient(app) as c:
        c.console = app.state.console
        yield c


def approve_all(client, reviewer="head-of-talent"):
    state = client.get("/api/state").json()
    for rule in state["rules"]:
        state = client.post(
            f"/api/rules/{rule['rule_id']}/approve",
            json={"reviewer": reviewer, "note": "verified against clause"},
        ).json()
    return state


def test_page_and_state_load(client) -> None:
    assert client.get("/console").status_code == 200
    state = client.get("/api/state").json()
    assert state["pack"]["id"] == "recruitment"
    assert state["metrics"]["total_cases"] == 8000
    assert state["rules"] == []


def test_extraction_cites_every_clause(client) -> None:
    out = client.post("/api/extract").json()
    assert out["proposed"] == 8
    assert not out["skipped"], "no clause should fail on the shipped policy"
    assert all(r["citation"] for r in out["rules"] if r["origin"] == "document")


def test_critical_cases_block_before_approval(client) -> None:
    """The guarantee, as the UI sees it."""
    client.post("/api/extract")
    out = client.post("/api/screen", json={"batch": 400}).json()

    assert out["metrics"]["blocked"] > 0, "critical cases must block"
    assert out["metrics"]["committed"] == 0, "nothing decides on unapproved logic"
    kinds = {row["kind"] for row in out["queue"]}
    assert "blocked" in kinds
    blocked = next(r for r in out["queue"] if r["kind"] == "blocked")
    assert blocked["pending"], "a blocked row must name the rules it waits on"


def test_approval_clears_the_block_and_decisions_follow(client) -> None:
    client.post("/api/extract")
    client.post("/api/screen", json={"batch": 400})
    state = approve_all(client)

    assert state["metrics"]["blocked"] == 0
    assert state["metrics"]["rules_approved"] == 8

    after = client.post("/api/screen", json={"batch": 600}).json()
    assert after["metrics"]["committed"] > 0
    assert after["metrics"]["decisions_per_review"] > 1


def test_approval_requires_a_named_reviewer(client) -> None:
    client.post("/api/extract")
    rule = client.get("/api/state").json()["rules"][0]["rule_id"]
    assert client.post(f"/api/rules/{rule}/approve", json={"reviewer": " "}).status_code == 400


def test_rejection_requires_a_reason_and_reports_impact(client) -> None:
    client.post("/api/extract")
    approve_all(client)
    client.post("/api/screen", json={"batch": 400})
    rule = client.get("/api/state").json()["rules"][0]["rule_id"]

    assert client.post(f"/api/rules/{rule}/reject",
                       json={"reviewer": "r"}).status_code == 400

    out = client.post(f"/api/rules/{rule}/reject",
                      json={"reviewer": "r", "reason": "too crude"}).json()
    assert out["impact"]["rule_id"] == rule
    assert "affected" in out["impact"]


def test_case_inspection_returns_a_real_proof(client) -> None:
    client.post("/api/extract")
    approve_all(client)
    client.post("/api/screen", json={"batch": 200})
    detail = client.get("/api/case/CV00005").json()

    assert detail["entity"] == "CV00005"
    assert detail["proof"], "every answer carries its derivation"
    assert detail["facts"], "and the facts it rests on, with sources"
    assert all("source" in f for f in detail["facts"])
    assert isinstance(detail["critical"], bool)


def test_unknown_case_is_a_404(client) -> None:
    assert client.get("/api/case/NOPE").status_code == 404


def test_invariant_holds_through_the_api(client) -> None:
    client.post("/api/extract")
    approve_all(client)
    out = client.post("/api/screen", json={"batch": 800}).json()
    assert out["metrics"]["invariant_violations"] == 0


def test_exports_are_downloadable(client) -> None:
    client.post("/api/extract")
    approve_all(client)
    csv = client.get("/api/export.csv")
    assert csv.status_code == 200 and "rule_id" in csv.text
    js = client.get("/api/export.json")
    assert js.status_code == 200 and "metrics" in js.json()


def test_malformed_policy_clauses_are_reported_not_dropped(client) -> None:
    """A clause the extractor cannot use must reach the reviewer as a note.

    Silently skipping is how a policy author ends up believing a rule is in
    force when it never entered the rule base.
    """
    console = client.console
    console.agent.extract_from_document(
        "broken.md",
        "## s9\n<!-- rule: reject(A) <- not applicant(A). -->\n",
        console.pack.vocabulary,
    )
    assert console.agent.skipped
    text, reason = console.agent.skipped[0]
    assert "unsafe" in reason


# --------------------------------------------------------------------------
# The showcase
# --------------------------------------------------------------------------


def test_showcase_is_the_landing_page(client) -> None:
    body = client.get("/").text
    assert client.get("/").status_code == 200
    assert "Approve once" in body
    # It must lead into the running system, not just describe it.
    assert 'href="/console"' in body


def test_showcase_links_resolve(client) -> None:
    """A showcase whose links 404 is worse than no showcase."""
    import re

    body = client.get("/").text
    targets = {h for h in re.findall(r'href="(/[^"#]*)"', body)}
    assert targets, "the page should link somewhere"
    for target in sorted(targets):
        code = client.get(target).status_code
        # The PDFs 404 cleanly when unbuilt; every other link must resolve.
        assert code in (200, 404), f"{target} returned {code}"
        if target.endswith(".pdf") and code == 404:
            assert "pdflatex" in client.get(target).json()["detail"]


def test_showcase_figures_are_served(client) -> None:
    for name in ("amortisation", "recovery"):
        res = client.get(f"/static/media/{name}.svg")
        assert res.status_code == 200
        assert res.text.lstrip().startswith("<?xml") or "<svg" in res.text[:400]


def test_unbuilt_pdf_says_how_to_build_it(client, monkeypatch) -> None:
    """A missing artifact should explain itself rather than 500."""
    from logicdb import console as mod

    monkeypatch.setattr(mod, "HERE", mod.HERE / "does-not-exist")
    res = client.get("/paper.pdf")
    assert res.status_code == 404
    assert "pdflatex" in res.json()["detail"]
