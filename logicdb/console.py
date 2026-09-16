"""The control room: a JSON API behind a single operational page.

The existing `app.py` renders one page per action, which is fine for reading a
proof and wrong for watching a campaign run. This module serves the same
governed engine as state a page can poll, so screening progress, the review
queue and the accountability record update in place.

Nothing here decides anything. Every route delegates to `Governor`, so the
guarantees hold identically whichever front end is attached -- in particular a
critical query still raises `BlockedForApproval`, and the page renders that as a
blocked row rather than as an error.

The shape of this interface is borrowed deliberately from the group's earlier
client-side prototype: a fleet, a queue, an inspector, a record. The difference
is what sits behind it. There, 100 agents were a simulation and every screened
CV became a human review. Here the fleet is the *rule base*, a rule is reviewed
once, and the engine that decides is the one that also writes the proof.
"""

from __future__ import annotations

import csv
import io
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Body, HTTPException, Request
from fastapi.responses import (
    FileResponse,
    HTMLResponse,
    JSONResponse,
    StreamingResponse,
)

from .agents.base import validate_proposals
from .agents.scripted import ScriptedAgent
from .explain import Explainer
from .governance import BlockedForApproval, Governor, open_governor
from .packs import DomainPack, load_pack
from .program import RuleOrigin, RuleStatus

HERE = Path(__file__).parent


@dataclass
class Console:
    """One campaign: a pack, a governor, and the case stream it screens."""

    pack: DomainPack
    governor: Governor
    agent: ScriptedAgent
    explainer: Explainer
    cases: list[dict[str, Any]] = field(default_factory=list)
    cursor: int = 0
    #: Cases the engine could not decide, awaiting a person. Held separately
    #: from the query log because this is a worklist, not a history.
    human_queue: list[str] = field(default_factory=list)
    blocked: list[dict[str, Any]] = field(default_factory=list)
    outcomes: dict[str, str] = field(default_factory=dict)
    skipped_clauses: list[tuple[str, str]] = field(default_factory=list)

    # -- screening -----------------------------------------------------
    def screen(self, batch: int) -> dict[str, Any]:
        """Push the next `batch` cases through the governed engine."""
        decided = 0
        for _ in range(batch):
            if self.cursor >= len(self.cases):
                break
            rec = self.cases[self.cursor]
            self.cursor += 1
            entity = str(rec[self.pack.entity_field])
            goal = self.pack.goal_for(entity)
            facts = self.pack.facts_from(rec, entity)
            try:
                answer, query = self.governor.ask(
                    goal, facts, entity, "screening-console",
                    context=self.pack.context_from(rec),
                    competing=self.pack.competing_goals(entity),
                )
            except BlockedForApproval as blocked:
                # Not an error. The governance guarantee, visible.
                self.blocked.append({
                    "entity": entity,
                    "reason": blocked.query.criticality_reason,
                    "pending": [r.rule_id for r in blocked.pending],
                })
                continue
            decided += 1
            committed = None
            for atom in self.pack.competing_goals(entity):
                bounds = answer.outcomes.get(str(atom))
                if bounds and bounds.lower > 0.5:
                    committed = atom.predicate
                    break
            if committed:
                self.outcomes[entity] = committed
            else:
                self.human_queue.append(entity)
        return {"decided": decided, "cursor": self.cursor}

    # -- views ---------------------------------------------------------
    def metrics(self) -> dict[str, Any]:
        gov = self.governor
        approved = gov.rulebase.by_status(RuleStatus.APPROVED)
        answered = [q for q in gov.queries.values() if q.status == "answered"]
        committed = len(self.outcomes)
        return {
            "total_cases": len(self.cases),
            "screened": self.cursor,
            "committed": committed,
            "human_queue": len(self.human_queue),
            "blocked": len(self.blocked),
            "rules_approved": len(approved),
            "rules_pending": len(gov.review_queue()),
            "decisions_per_review": (
                len(answered) / len(approved) if approved else 0.0
            ),
            "invariant_violations": len(gov.verify_invariant()),
            "outcome_counts": {
                name: sum(1 for v in self.outcomes.values() if v == name)
                for name in self.pack.outcomes
            },
            "complete": self.cursor >= len(self.cases),
        }

    def rule_rows(self) -> list[dict[str, Any]]:
        rows = []
        for r in self.governor.rulebase.all_records():
            rows.append({
                "rule_id": r.rule_id,
                "text": str(r.rule),
                "status": r.status.value,
                "origin": r.origin.value,
                "strength": round(r.strength, 2),
                "citation": r.source_citation,
                "support": r.mining_support,
                "precision": r.mining_precision,
                "used": r.stats.reuse_count,
                "successes": r.stats.successes,
                "observations": r.stats.observations,
                "approved_by": r.approved_by,
                "rejected_reason": r.rejected_reason,
                "notes": r.notes,
            })
        rows.sort(key=lambda x: (x["status"] != "pending_online",
                                 x["status"] != "provisional",
                                 -x["used"], x["rule_id"]))
        return rows

    def queue_rows(self, limit: int = 60) -> list[dict[str, Any]]:
        rows = []
        for entry in self.blocked[:limit]:
            rows.append({
                "kind": "blocked", "entity": entry["entity"],
                "detail": entry["reason"], "pending": entry["pending"],
            })
        for entity in self.human_queue[: max(0, limit - len(rows))]:
            rows.append({
                "kind": "human", "entity": entity,
                "detail": "No applicable logic — needs a person",
                "pending": [],
            })
        return rows

    def inspect(self, entity: str) -> dict[str, Any]:
        rec = next(
            (c for c in self.cases
             if str(c[self.pack.entity_field]) == entity), None)
        if rec is None:
            raise HTTPException(404, f"no case {entity}")
        goal = self.pack.goal_for(entity)
        facts = self.pack.facts_from(rec, entity)
        critical, reason = self.pack.criticality.assess(self.pack.context_from(rec))
        answer = self.governor.solver.answer(
            goal, facts, critical=critical,
            competing=self.pack.competing_goals(entity),
        )
        uncertain = self.explainer.uncertain_atoms(facts, critical)
        return {
            "entity": entity,
            "critical": critical,
            "criticality_reason": reason,
            "bounds": [round(answer.bounds.lower, 3), round(answer.bounds.upper, 3)],
            "uncovered": answer.is_uncovered,
            "outcomes": {
                k: [round(v.lower, 3), round(v.upper, 3)]
                for k, v in answer.outcomes.items()
            },
            "proof": self.explainer.render_proof(goal, answer.evaluation, uncertain),
            "provenance": self.explainer.provenance_report(goal, answer.evaluation),
            "ask": [
                {"atom": str(c.atom), "decisive": c.decisive,
                 "if_true": str(c.if_true), "if_false": str(c.if_false)}
                for c in answer.ask
            ],
            "assumptions": answer.assumptions,
            "facts": [
                {"atom": str(f.atom), "truth": f.truth.value, "source": f.source}
                for f in sorted(facts, key=lambda r: str(r.atom))
            ],
            "rules": sorted(answer.evaluation.rules_used),
        }

    def log_rows(self, limit: int = 40) -> list[dict[str, Any]]:
        rows = []
        for r in self.governor.rulebase.all_records():
            for h in self.governor.store.rule_history(r.rule_id):
                rows.append({"time": h["at"], "actor": h["actor"],
                             "event": h["event"], "subject": r.rule_id,
                             "detail": h["detail"]})
        rows.sort(key=lambda x: x["time"], reverse=True)
        return rows[:limit]


def make_router(console: Console) -> APIRouter:
    router = APIRouter()

    def state() -> dict[str, Any]:
        return {
            "pack": {
                "id": console.pack.domain_id,
                "name": console.pack.name,
                "question": console.pack.decision_question
                if hasattr(console.pack, "decision_question") else "",
                "goal": console.pack.goal_predicate,
                "outcomes": console.pack.outcomes,
            },
            "metrics": console.metrics(),
            "rules": console.rule_rows(),
            "queue": console.queue_rows(),
            "log": console.log_rows(),
            "skipped": [{"text": t, "reason": r}
                        for t, r in console.skipped_clauses],
        }

    @router.get("/api/state")
    def get_state() -> JSONResponse:
        return JSONResponse(state())

    @router.post("/api/screen")
    def screen(body: dict = Body(default={})) -> JSONResponse:
        batch = int(body.get("batch", 250))
        result = console.screen(max(1, min(batch, 2000)))
        return JSONResponse({**result, **state()})

    @router.post("/api/extract")
    def extract() -> JSONResponse:
        proposals = []
        for name, text in console.pack.document_text().items():
            proposals.extend(
                console.agent.extract_from_document(
                    name, text, console.pack.vocabulary))
        console.skipped_clauses = list(console.agent.skipped)
        accepted, rejected = validate_proposals(proposals, console.pack.vocabulary)
        for proposal in accepted:
            console.governor.propose(
                proposal.rule, origin=proposal.origin,
                proposed_by=console.agent.agent_id, critical_context=True,
                strength=proposal.strength,
                source_citation=proposal.source_citation)
        console.skipped_clauses += [
            (str(p.rule), "; ".join(probs)) for p, probs in rejected]
        return JSONResponse({"proposed": len(accepted), **state()})

    @router.post("/api/rules/{rule_id}/approve")
    def approve(rule_id: str, body: dict = Body(default={})) -> JSONResponse:
        reviewer = (body.get("reviewer") or "").strip()
        if not reviewer:
            raise HTTPException(400, "Name the reviewer accountable for this approval.")
        console.governor.approve(rule_id, by=reviewer,
                                 note=(body.get("note") or "").strip())
        # A rule becoming approved can unblock queries that were stalled on it.
        console.blocked = [
            b for b in console.blocked if rule_id not in b["pending"]]
        return JSONResponse(state())

    @router.post("/api/rules/{rule_id}/reject")
    def reject(rule_id: str, body: dict = Body(default={})) -> JSONResponse:
        reviewer = (body.get("reviewer") or "").strip()
        reason = (body.get("reason") or "").strip()
        if not reviewer or not reason:
            raise HTTPException(400, "A rejection needs a named reviewer and a reason.")
        report = console.governor.reject(rule_id, by=reviewer, reason=reason)
        return JSONResponse({
            "impact": {
                "rule_id": report.rule_id,
                "affected": [q.query_id for q in report.affected_queries],
            },
            **state(),
        })

    @router.get("/api/case/{entity}")
    def case(entity: str) -> JSONResponse:
        return JSONResponse(console.inspect(entity))

    @router.get("/api/export.json")
    def export_json() -> StreamingResponse:
        payload = json.dumps({
            "pack": console.pack.domain_id,
            "metrics": console.metrics(),
            "rules": console.rule_rows(),
            "log": console.log_rows(limit=100000),
        }, indent=2)
        return StreamingResponse(
            io.BytesIO(payload.encode()), media_type="application/json",
            headers={"Content-Disposition":
                     'attachment; filename="decision_record.json"'})

    @router.get("/api/export.csv")
    def export_csv() -> StreamingResponse:
        buf = io.StringIO()
        w = csv.writer(buf)
        w.writerow(["rule_id", "status", "origin", "rule", "citation",
                    "strength", "used", "right", "observations", "approved_by"])
        for r in console.rule_rows():
            w.writerow([r["rule_id"], r["status"], r["origin"], r["text"],
                        r["citation"] or "", r["strength"], r["used"],
                        r["successes"], r["observations"], r["approved_by"] or ""])
        return StreamingResponse(
            io.BytesIO(buf.getvalue().encode()), media_type="text/csv",
            headers={"Content-Disposition":
                     'attachment; filename="rule_base.csv"'})

    @router.get("/console", response_class=HTMLResponse)
    def page() -> HTMLResponse:
        return HTMLResponse((HERE / "templates" / "console.html").read_text())

    @router.get("/", response_class=HTMLResponse)
    def showcase() -> HTMLResponse:
        """The landing page: what the work is, and a door into the running system.

        Served from the same process as the control room on purpose. A showcase
        that links to a screenshot is a claim; one that links to `/console` is
        an invitation to check.
        """
        return HTMLResponse((HERE / "templates" / "showcase.html").read_text())

    @router.get("/paper.pdf")
    def paper() -> FileResponse:
        return _artifact("paper.pdf", "application/pdf")

    @router.get("/slides.pdf")
    def slides() -> FileResponse:
        return _artifact("slides.pdf", "application/pdf")

    return router


def _artifact(name: str, media_type: str) -> FileResponse:
    """Serve a built PDF, or say plainly that it has not been built yet."""
    path = HERE.parent / "paper" / name
    if not path.exists():
        raise HTTPException(
            404,
            f"{name} has not been built. Run: cd paper && pdflatex {Path(name).stem}.tex",
        )
    return FileResponse(path, media_type=media_type, filename=name)


__all__ = ["Console", "build_console", "make_router"]
