"""Serves the page and one endpoint.

    GET  /            the console
    POST /api/turn    one exchange

Run:  uvicorn server:app --reload --port 8000   (from prototyping/)
"""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

import json

from agent import graph, ladder, scales
from agent.db import store as logic_store

#: One store for the whole process. Two instances were opening the same file on
#: two connections — one of them by a relative path, so it also depended on the
#: working directory the server happened to start in.
DB = logic_store()
from agent.state import DEFAULT_INPUTS

FRONTEND = Path(__file__).parent / "frontend"

#: The pages are edited constantly during development. Without this the browser
#: is free to serve a cached copy without revalidating, and a change that is
#: live on the server appears not to have happened at all. `no-cache` still
#: allows the ETag round trip, so nothing is re-downloaded needlessly.
NO_CACHE = {"Cache-Control": "no-cache, must-revalidate"}


def page(name: str) -> FileResponse:
    return FileResponse(FRONTEND / name, headers=NO_CACHE)

app = FastAPI(title="confused.de")


class Inputs(BaseModel):
    harm: int = Field(DEFAULT_INPUTS["harm"], ge=1, le=10)
    reversibility: int = Field(DEFAULT_INPUTS["reversibility"], ge=1, le=10)
    rights: bool = DEFAULT_INPUTS["rights"]
    confidence: int = Field(DEFAULT_INPUTS["confidence"], ge=0, le=100)


class Turn(BaseModel):
    session: str = "default"
    message: str
    model: str = "haiku"
    mode: str = "auto"
    inputs: Inputs = Field(default_factory=Inputs)


@app.get("/")
def index() -> FileResponse:
    return page("index.html")


@app.get("/api/tiers")
def tiers() -> JSONResponse:
    """The four tiers and their wording, so the page need not hard-code them."""
    return JSONResponse({"tiers": ladder.TIERS,
                         "restrictiveness": ladder.RESTRICTIVENESS})


@app.get("/database")
def database_page() -> FileResponse:
    return page("database.html")


@app.get("/api/db/stats")
def db_stats() -> JSONResponse:
    return JSONResponse(DB.stats())


@app.get("/api/db/rules")
def db_rules(status: str | None = None) -> JSONResponse:
    return JSONResponse({"rules": [r.as_dict() for r in DB.rules(status)]})


@app.get("/api/db/rules/{key}")
def db_rule(key: str) -> JSONResponse:
    row = DB.get_rule(key)
    if row is None:
        return JSONResponse({"error": "no such rule"}, status_code=404)
    return JSONResponse({
        "rule": row.as_dict(),
        "cases": DB.cases_using(key),
        "events": DB.events(key),
    })


# --------------------------------------------------------------------------
# The expert side
# --------------------------------------------------------------------------
class Verdict(BaseModel):
    verdict: str = "approved"          # approved | rejected
    actor: str = "expert"
    note: str = ""
    #: The answer as the expert wants it sent. Empty means the draft stands.
    #: Editing it is the point: approving something you may not alter is a
    #: rubber stamp with extra steps.
    reply: str = ""


class RuleVerdict(BaseModel):
    status: str = "approved"           # approved | rejected
    actor: str = "expert"
    note: str = ""


@app.get("/api/expert/queue")
def expert_queue() -> JSONResponse:
    """Inquiries parked for a person, and rules awaiting review.

    Two lists, never merged. Releasing one inquiry is not the same act as
    approving the logic behind it, and an interface that blurs them invites the
    cheaper click.
    """
    return JSONResponse({
        "decisions": DB.waiting(),
        "rules": [r.as_dict() for r in DB.rules("proposed")],
    })


@app.get("/api/expert/decision/{case_id}")
def expert_decision(case_id: str) -> JSONResponse:
    case = DB.get_case(case_id)
    if case is None:
        return JSONResponse({"error": "no such decision"}, status_code=404)
    return JSONResponse(case)


@app.post("/api/expert/decision/{case_id}")
def settle_decision(case_id: str, body: Verdict) -> JSONResponse:
    """Release or refuse one parked inquiry, and write the answer it was owed."""
    case = DB.get_case(case_id)
    if case is None:
        return JSONResponse({"error": "no such decision"}, status_code=404)
    if case.get("status") != "waiting":
        return JSONResponse({"error": f"already {case.get('status')}"},
                            status_code=409)

    reply = graph.reply_for_settled(case, body.verdict, body.actor, body.note,
                                    edited=body.reply)
    ok = DB.settle_case(case_id, body.verdict, body.actor, body.note, reply)
    if not ok:
        return JSONResponse({"error": "could not settle"}, status_code=409)

    # Their decision becomes candidate logic, queued for review like any other.
    # Deciding a case and licensing the logic behind it stay separate acts.
    from agent.nodes.learn import rule_from_decision
    learned = rule_from_decision(case, body.verdict, body.actor,
                                 body.reply or reply)
    return JSONResponse({"case_id": case_id, "status": body.verdict,
                         "reply": reply, "learned": learned})


@app.get("/api/db/generalisations")
def db_generalisations() -> JSONResponse:
    """Rules that could be merged, and what each merge would have done.

    Never applied automatically. A generalisation fires on strictly more cases
    than its sources, so it always risks deciding something nobody intended.
    """
    from logic.generalize import suggestions
    from logic.parse import ParseError, parse_rule
    from logic.syntax import UnsafeRule

    rules = []
    for row in DB.active_rules():
        try:
            rules.append((row.canonical_key, parse_rule(row.text)))
        except (ParseError, UnsafeRule, ValueError):
            continue
    out = []
    for c in suggestions(rules, DB.cases(limit=2000)):
        out.append({
            "text": str(c.rule), "how": c.how, "sources": list(c.sources),
            "source_texts": [r.text for r in
                             (DB.get_rule(k) for k in c.sources) if r],
            "keeps": len(c.keeps), "gains": len(c.gains),
            "conflicts": len(c.conflicts), "evidence": c.evidence,
            "reach": round(c.reach, 2), "caution": c.caution(),
            "conflict_cases": [
                {"case_id": cid,
                 "question": (DB.get_case(cid) or {}).get("question", "")}
                for cid in c.conflicts[:5]],
        })
    return JSONResponse({"candidates": out})


class Generalisation(BaseModel):
    text: str
    sources: list[str] = []
    actor: str = "expert"
    note: str = ""
    retire_sources: bool = True


@app.post("/api/db/generalisations")
def accept_generalisation(body: Generalisation) -> JSONResponse:
    """Approve a merge, and retire what it replaces."""
    from logic.parse import ParseError, parse_rule
    from logic.store import RuleRow
    from logic.syntax import UnsafeRule
    try:
        rule = parse_rule(body.text)
    except (ParseError, UnsafeRule, ValueError) as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)

    DB.upsert_rule(RuleRow(rule.canonical_key(), str(rule), rule.head.predicate,
                           origin="generalisation", proposed_by=body.actor,
                           note=body.note))
    DB.set_status(rule.canonical_key(), "approved", body.actor,
                  body.note or "approved as a generalisation")
    retired = []
    if body.retire_sources:
        for key in body.sources:
            if DB.get_rule(key) is not None:
                DB.retire_rule(key, body.actor,
                               f"replaced by {rule.canonical_key()[:8]}")
                retired.append(key)
    return JSONResponse({"canonical_key": rule.canonical_key(),
                         "retired": retired})


@app.get("/api/expert/audits")
def audit_queue() -> JSONResponse:
    """Decisions pulled for audit that nobody has ruled on yet."""
    return JSONResponse({"audits": DB.audit_queue(),
                         "rereview": [r.as_dict() for r in DB.needs_rereview()]})


@app.post("/api/expert/audit/{case_id}")
def audit_decision(case_id: str, body: Verdict) -> JSONResponse:
    """A verdict on a decision that already took effect.

    Not an approval — the decision has happened. What this changes is the
    record of the rules behind it.
    """
    ok = DB.audit_case(case_id, body.verdict == "approved", body.actor, body.note)
    if not ok:
        return JSONResponse({"error": "no such decision"}, status_code=404)
    return JSONResponse({"case_id": case_id, "verdict": body.verdict})


@app.get("/api/expert/rule/{key}/impact")
def rule_impact(key: str) -> JSONResponse:
    """What retiring this rule would touch, without touching it."""
    if DB.get_rule(key) is None:
        return JSONResponse({"error": "no such rule"}, status_code=404)
    return JSONResponse(DB.impact_of(key))


@app.post("/api/expert/rule/{key}/retire")
def retire_rule(key: str, body: RuleVerdict) -> JSONResponse:
    """Withdraw an approved rule and enumerate what it already decided."""
    if DB.get_rule(key) is None:
        return JSONResponse({"error": "no such rule"}, status_code=404)
    return JSONResponse(DB.retire_rule(key, body.actor, body.note))


@app.post("/api/expert/rule/{key}")
def settle_rule(key: str, body: RuleVerdict) -> JSONResponse:
    """Approve or reject a rule. This one governs every future matching case."""
    if DB.get_rule(key) is None:
        return JSONResponse({"error": "no such rule"}, status_code=404)
    if body.status not in ("approved", "rejected"):
        return JSONResponse({"error": "unknown status"}, status_code=400)
    DB.set_status(key, body.status, body.actor, body.note)
    return JSONResponse({"canonical_key": key, "status": body.status})


class Reset(BaseModel):
    #: Required, so a stray request cannot empty the database by arriving.
    confirm: bool = False


@app.post("/api/db/reset")
def db_reset(body: Reset) -> JSONResponse:
    """Empty the database. Everything: rules, decisions, history."""
    if not body.confirm:
        return JSONResponse({"error": "confirm must be true"}, status_code=400)
    return JSONResponse({"removed": DB.reset()})


@app.get("/api/db/cases")
def db_cases(limit: int = 200) -> JSONResponse:
    return JSONResponse({"cases": DB.cases(limit)})


@app.get("/api/scales")
def api_scales() -> JSONResponse:
    """The input definitions. Served rather than duplicated in the page, so the
    help text a person reads is the same string the estimator was given."""
    return JSONResponse({"scales": scales.SCALES, "order": list(scales.ORDER)})


@app.post("/api/turn/stream")
def api_turn_stream(body: Turn) -> StreamingResponse:
    """Server-sent events. One JSON object per `data:` line."""

    def events():
        try:
            for event in graph.turn_stream(
                session=body.session,
                message=body.message,
                model=body.model,
                mode=body.mode,
                user_inputs=body.inputs.model_dump(),
            ):
                yield f"data: {json.dumps(event)}\n\n"
        except Exception as exc:  # noqa: BLE001
            # The page cannot see a traceback, so failures are sent as events
            # rather than left as a stream that simply stops.
            yield f"data: {json.dumps({'kind': 'error', 'message': str(exc)})}\n\n"

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/api/turn")
def api_turn(body: Turn) -> JSONResponse:
    result = graph.turn(
        session=body.session,
        message=body.message,
        model=body.model,
        mode=body.mode,
        user_inputs=body.inputs.model_dump(),
    )
    return JSONResponse(result)
