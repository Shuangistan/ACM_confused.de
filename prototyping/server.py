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

    reply = graph.reply_for_settled(case, body.verdict, body.actor, body.note)
    ok = DB.settle_case(case_id, body.verdict, body.actor, body.note, reply)
    if not ok:
        return JSONResponse({"error": "could not settle"}, status_code=409)
    return JSONResponse({"case_id": case_id, "status": body.verdict,
                         "reply": reply})


@app.post("/api/expert/rule/{key}")
def settle_rule(key: str, body: RuleVerdict) -> JSONResponse:
    """Approve or reject a rule. This one governs every future matching case."""
    if DB.get_rule(key) is None:
        return JSONResponse({"error": "no such rule"}, status_code=404)
    if body.status not in ("approved", "rejected"):
        return JSONResponse({"error": "unknown status"}, status_code=400)
    DB.set_status(key, body.status, body.actor, body.note)
    return JSONResponse({"canonical_key": key, "status": body.status})


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
