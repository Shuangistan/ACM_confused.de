"""Web interface: query console, proof view, review queue, rule browser.

The review queue is the screen that matters. Everything else displays state; the
queue is where a human actually exercises control, and its design carries the
argument. A reviewer is shown *different evidence depending on provenance* --
a document rule beside the clause it cites, a mined rule beside its support,
precision and counterexamples -- because those call for different judgements. On
a document rule the question is "does this faithfully encode the clause?"; on a
mined rule it is "is this correlation also a reason?". A queue that rendered
both identically would invite the reviewer to answer the easier question in both
cases, which is how rule-level oversight would decay into the same
rubber-stamping the design exists to avoid.

The app holds one `Governor` for the process. That is fine for a demo and a
seminar room, and wrong for anything larger: writes are serialised through one
SQLite connection.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from .agents.base import LabelledCase, RuleProposal, validate_proposals
from .agents.scripted import ScriptedAgent
from .explain import Explainer
from .governance import BlockedForApproval, Governor, open_governor
from .packs import DomainPack, load_pack
from .program import RuleStatus

HERE = Path(__file__).parent


@dataclass
class AppState:
    """Everything the request handlers share."""

    pack: DomainPack
    governor: Governor
    agent: ScriptedAgent
    explainer: Explainer
    cases: dict[str, dict[str, Any]]

    def case(self, entity: str) -> dict[str, Any]:
        record = self.cases.get(entity)
        if record is None:
            raise HTTPException(404, f"no case {entity}")
        return record

    def labelled_cases(self, limit: int | None = None) -> list[LabelledCase]:
        """Cases with a known outcome, for mining."""
        out = []
        for entity, record in self.cases.items():
            label = self.pack.label_of(record)
            if label is None:
                continue
            out.append(
                LabelledCase(entity, self.pack.facts_from(record, entity), label, record)
            )
            if limit and len(out) >= limit:
                break
        return out


def create_app(
    pack_id: str = "consumer_credit",
    db_path: str | Path = "data/logicdb.sqlite",
) -> FastAPI:
    pack = load_pack(pack_id)
    governor = open_governor(db_path, pack.vocabulary, pack.criticality)
    agent = ScriptedAgent(pack)
    explainer = Explainer(governor.rulebase, pack.vocabulary)

    cases: dict[str, dict[str, Any]] = {}
    cases_file = pack.root / "cases.json" if pack.root else None
    if cases_file and cases_file.exists():
        for record in json.loads(cases_file.read_text()):
            cases[str(record[pack.entity_field])] = record

    state = AppState(pack, governor, agent, explainer, cases)

    app = FastAPI(title=f"Logic database — {pack.name}")
    app.state.logic = state

    templates = Jinja2Templates(directory=str(HERE / "templates"))
    templates.env.globals["pack"] = pack
    app.mount("/static", StaticFiles(directory=str(HERE / "static")), name="static")

    def page(request: Request, name: str, **context: Any) -> HTMLResponse:
        return templates.TemplateResponse(
            request=request, name=name, context={"pack": pack, **context}
        )

    # ------------------------------------------------------------------
    # Dashboard
    # ------------------------------------------------------------------
    @app.get("/dashboard", response_class=HTMLResponse)
    def dashboard(request: Request) -> HTMLResponse:
        summary = governor.governance_summary()
        return page(
            request,
            "dashboard.html",
            summary=summary,
            amortisation=governor.store.amortisation(),
            counts=governor.store.counts(),
            queue_size=len(governor.review_queue()),
            rereview=governor.rules_needing_rereview(),
            violations=governor.verify_invariant(),
            recent=list(governor.queries.values())[-12:][::-1],
            n_cases=len(cases),
        )

    # ------------------------------------------------------------------
    # Querying
    # ------------------------------------------------------------------
    @app.get("/query", response_class=HTMLResponse)
    def query_form(request: Request) -> HTMLResponse:
        sample = list(cases)[:40]
        return page(request, "query.html", sample=sample, result=None)

    @app.post("/query", response_class=HTMLResponse)
    def run_query(
        request: Request, entity: str = Form(...), asked_by: str = Form("officer")
    ) -> HTMLResponse:
        record = state.case(entity.strip())
        facts = pack.facts_from(record, entity.strip())
        goal = pack.goal_for(entity.strip())

        try:
            answer, query = governor.ask(
                goal, facts, entity.strip(), asked_by,
                context=pack.context_from(record),
                competing=pack.competing_goals(entity.strip()),
            )
        except BlockedForApproval as blocked:
            return page(
                request,
                "blocked.html",
                query=blocked.query,
                pending=blocked.pending,
                entity=entity.strip(),
            )

        proof = explainer.render_proof(
            goal, answer.evaluation, explainer.uncertain_atoms(facts, query.critical)
        )
        return page(
            request,
            "result.html",
            answer=answer,
            query=query,
            proof=proof,
            facts=sorted(facts, key=lambda r: str(r.atom)),
            provenance=explainer.provenance_report(goal, answer.evaluation),
            rules=[governor.rulebase.get(r) for r in query.rule_ids],
            entity=entity.strip(),
        )

    @app.post("/query/{query_id}/outcome")
    def record_outcome(query_id: str, correct: str = Form(...)) -> RedirectResponse:
        governor.record_outcome(query_id, correct=correct == "yes")
        return RedirectResponse("/", status_code=303)

    # ------------------------------------------------------------------
    # Review queue
    # ------------------------------------------------------------------
    @app.get("/review", response_class=HTMLResponse)
    def review(request: Request) -> HTMLResponse:
        queue = governor.review_queue()
        # Blocking reviews first: a query is stalled on each of those, while the
        # provisional lane can wait. Sorting them together would hide that.
        blocking = [r for r in queue if r.status is RuleStatus.PENDING_ONLINE]
        offline = [r for r in queue if r.status is RuleStatus.PROVISIONAL]
        return page(
            request,
            "review.html",
            blocking=blocking,
            offline=offline,
            rereview=governor.rules_needing_rereview(),
            impact_preview={
                r.rule_id: governor.store.queries_using_rule(r.rule_id, True)
                for r in offline
            },
        )

    @app.post("/review/{rule_id}/approve")
    def approve(
        rule_id: str, reviewer: str = Form("reviewer"), note: str = Form("")
    ) -> RedirectResponse:
        governor.approve(rule_id, by=reviewer, note=note)
        return RedirectResponse("/review", status_code=303)

    @app.post("/review/{rule_id}/reject", response_class=HTMLResponse)
    def reject(
        request: Request,
        rule_id: str,
        reviewer: str = Form("reviewer"),
        reason: str = Form("no reason given"),
    ) -> HTMLResponse:
        report = governor.reject(rule_id, by=reviewer, reason=reason)
        # The impact report is shown rather than redirected past. A rejection may
        # have a blast radius, and the reviewer who caused it is the right person
        # to see it.
        return page(request, "impact.html", report=report)

    # ------------------------------------------------------------------
    # Rule browser
    # ------------------------------------------------------------------
    @app.get("/rules", response_class=HTMLResponse)
    def rules(request: Request, status: str = "") -> HTMLResponse:
        records = governor.rulebase.all_records()
        if status:
            records = [r for r in records if r.status.value == status]
        records.sort(key=lambda r: (-r.stats.reuse_count, r.rule_id))
        return page(
            request, "rules.html", records=records, status=status,
            statuses=[s.value for s in RuleStatus],
        )

    @app.get("/rules/{rule_id}", response_class=HTMLResponse)
    def rule_detail(request: Request, rule_id: str) -> HTMLResponse:
        record = governor.rulebase.get(rule_id)
        if record is None:
            raise HTTPException(404, f"no rule {rule_id}")
        return page(
            request,
            "rule_detail.html",
            record=record,
            history=governor.store.rule_history(rule_id),
            used_in=governor.store.queries_using_rule(rule_id)[:50],
        )

    @app.get("/export", response_class=HTMLResponse)
    def export(request: Request) -> HTMLResponse:
        return page(request, "export.html", text=governor.store.export_text())

    # ------------------------------------------------------------------
    # Agent
    # ------------------------------------------------------------------
    @app.get("/agent", response_class=HTMLResponse)
    def agent_page(request: Request) -> HTMLResponse:
        return page(request, "agent.html", proposals=None, source=None, rejected=[])

    @app.post("/agent/extract", response_class=HTMLResponse)
    def agent_extract(request: Request) -> HTMLResponse:
        proposals: list[RuleProposal] = []
        for name, text in pack.document_text().items():
            proposals.extend(agent.extract_from_document(name, text, pack.vocabulary))
        return _submit(request, proposals, "documents", critical_context=True)

    @app.post("/agent/mine", response_class=HTMLResponse)
    def agent_mine(request: Request, target: str = Form("")) -> HTMLResponse:
        known = {r.key for r in governor.rulebase.all_records()}
        proposals = agent.mine_rules(
            state.labelled_cases(), target=target or pack.goal_predicate, existing=known
        )
        return _submit(request, proposals, "labelled cases", critical_context=False)

    def _submit(
        request: Request,
        proposals: list[RuleProposal],
        source: str,
        critical_context: bool,
    ) -> HTMLResponse:
        # Validation before routing: a proposal naming an undeclared predicate is
        # discarded mechanically and never consumes reviewer attention. With an
        # LLM backend this is the filter that keeps hallucinations out.
        accepted, rejected = validate_proposals(proposals, pack.vocabulary)
        results = []
        for proposal in accepted:
            outcome = governor.propose(
                proposal.rule,
                origin=proposal.origin,
                proposed_by=agent.agent_id,
                critical_context=critical_context,
                strength=proposal.strength,
                source_citation=proposal.source_citation,
                mining_support=proposal.support,
                mining_precision=proposal.precision,
                notes=proposal.rationale,
            )
            results.append((proposal, outcome))
        return page(
            request, "agent.html", proposals=results, source=source, rejected=rejected
        )

    # The control room: a JSON API plus one operational page, sharing this
    # process's governor so both front ends see the same governed state.
    from .console import Console, make_router

    console = Console(pack=pack, governor=governor, agent=agent,
                      explainer=explainer, cases=list(cases.values()))
    app.include_router(make_router(console))
    app.state.console = console

    return app


import os

#: Configured by run.sh so the same module serves any pack without editing code.
app = create_app(
    pack_id=os.environ.get("LOGICDB_PACK", "consumer_credit"),
    db_path=os.environ.get("LOGICDB_DB", "data/logicdb.sqlite"),
)
