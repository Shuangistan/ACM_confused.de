"""The agent loop: an empty database learns its own logic, under review.

Run:  python demo_agent.py [--pack consumer_credit] [--db path.sqlite]

Starts from nothing and shows the loop the project is about:

  1. The database is empty, so the first query is uncovered -- which is reported
     as an absence of applicable logic, not as a negative answer.
  2. The agent extracts rules from the written policy, each citing its clause.
  3. The agent mines rules from labelled cases, each carrying support and
     precision.
  4. A reviewer works the queue, seeing what evidence stands behind each rule.
  5. The approved logic answers hundreds of cases with no further review.
  6. Outcomes feed back, and rules whose evidence turned against them are
     flagged -- the one case where an approved rule returns.

The comparison worth watching is step 2 against step 3. Document rules and
mined rules arrive through the same gate but are not equally trustworthy, and
the queue shows why.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from logicdb.agents.base import LabelledCase, validate_proposals
from logicdb.agents.scripted import ScriptedAgent
from logicdb.explain import Explainer
from logicdb.governance import BlockedForApproval, open_governor
from logicdb.packs import load_pack

RULE = "─" * 78


def banner(n: int, title: str) -> None:
    print(f"\n{RULE}\n {n}. {title}\n{RULE}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pack", default="consumer_credit")
    ap.add_argument("--db", default="data/demo_agent.sqlite")
    ap.add_argument("--train", type=int, default=600)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    db = Path(args.db)
    if db.exists():
        db.unlink()  # a fresh start, so the demo always tells the same story

    pack = load_pack(args.pack)
    gov = open_governor(db, pack.vocabulary, pack.criticality)
    agent = ScriptedAgent(pack)
    explainer = Explainer(gov.rulebase, pack.vocabulary)

    raw = json.loads((pack.root / "cases.json").read_text())
    rng = random.Random(args.seed)
    rng.shuffle(raw)
    train_raw, live_raw = raw[: args.train], raw[args.train :]
    train = [
        LabelledCase(r["id"], pack.facts_from(r, r["id"]), pack.label_of(r), r)
        for r in train_raw
    ]

    print(f"pack: {pack.name}   {len(pack.vocabulary)} predicates, "
          f"{len(raw)} cases ({len(train)} for evidence, {len(live_raw)} live)")

    # ------------------------------------------------------------------
    banner(1, "An empty database cannot answer, and says so")
    first = live_raw[0]
    facts = pack.facts_from(first, first["id"])
    answer, query = gov.ask(
        pack.goal_for(first["id"]), facts, first["id"], "officer-1",
        context=pack.context_from(first), competing=pack.competing_goals(first["id"]),
    )
    print(f"query {query.query_id} on {first['id']}: {answer.bounds}")
    print(f"  uncovered: {answer.is_uncovered}")
    print("  -> 'no applicable logic' is not 'no'. The difference is the whole "
          "reason\n     to ask the agent for a rule rather than to decline.")

    # ------------------------------------------------------------------
    banner(2, "The agent extracts rules from the written policy")
    doc_proposals = []
    for name, text in pack.document_text().items():
        doc_proposals.extend(agent.extract_from_document(name, text, pack.vocabulary))
    good, bad = validate_proposals(doc_proposals, pack.vocabulary)
    print(f"{len(good)} valid, {len(bad)} rejected before review\n")
    for proposal in good:
        result = gov.propose(
            proposal.rule, origin=proposal.origin, proposed_by=agent.agent_id,
            critical_context=True, strength=proposal.strength,
            source_citation=proposal.source_citation,
        )
        print(f"  {proposal.rule}")
        print(f"     cites {proposal.source_citation}  -> {result.outcome.value}")

    # ------------------------------------------------------------------
    banner(3, "The agent mines rules from labelled cases")
    known = {r.key for r in gov.rulebase.all_records()}
    mined = agent.mine_rules(train, target=pack.goal_predicate, existing=known)
    print(f"{len(mined)} proposals from {len(train)} cases\n")
    for proposal in mined:
        result = gov.propose(
            proposal.rule, origin=proposal.origin, proposed_by=agent.agent_id,
            critical_context=False, strength=proposal.strength,
            mining_support=proposal.support, mining_precision=proposal.precision,
            notes=proposal.rationale,
        )
        print(f"  {proposal.rule}")
        print(f"     {proposal.rationale}")
        print(f"     -> {result.outcome.value}")

    # ------------------------------------------------------------------
    banner(4, "The reviewer works the queue")
    queue = gov.review_queue()
    print(f"{len(queue)} rules awaiting a human.\n")
    print("Note what the reviewer is looking at: a document rule can be checked")
    print("against its clause; a mined rule can only be judged on its evidence.\n")

    approved = rejected = 0
    for record in queue:
        if record.origin.value == "document":
            gov.approve(record.rule_id, by="reviewer-jd",
                        note=f"verified against {record.source_citation}")
            approved += 1
            verdict = "APPROVE (checked against the clause)"
        elif (record.mining_precision or 0) >= 0.75:
            gov.approve(record.rule_id, by="reviewer-jd",
                        note="evidence adequate; consistent with policy intent")
            approved += 1
            verdict = f"APPROVE (precision {record.mining_precision:.2f})"
        else:
            gov.reject(record.rule_id, by="reviewer-jd",
                       reason=f"precision {record.mining_precision:.2f} too close to "
                              f"chance to be defensible as a reason")
            rejected += 1
            verdict = f"REJECT  (precision {record.mining_precision:.2f})"
        print(f"  {record.rule_id} [{record.origin.value:8s}] {verdict}")
        print(f"      {record.rule}")
    print(f"\n{approved} approved, {rejected} rejected.")

    # ------------------------------------------------------------------
    banner(5, "The approved logic now runs, with no further review")
    answered = uncovered = blocked = 0
    correct = 0
    scored = 0
    for record in live_raw:
        entity = record["id"]
        facts = pack.facts_from(record, entity)
        try:
            answer, query = gov.ask(
                pack.goal_for(entity), facts, entity, "officer-2",
                context=pack.context_from(record),
                competing=pack.competing_goals(entity),
            )
        except BlockedForApproval:
            blocked += 1
            continue
        if answer.is_uncovered:
            uncovered += 1
            continue
        answered += 1
        # Score only where the system committed: a wide interval is a refusal to
        # call it, and grading refusals as errors would reward overconfidence.
        if answer.bounds.lower > 0.5 or answer.bounds.upper < 0.5:
            scored += 1
            predicted = "decline" if answer.bounds.lower > 0.5 else "approve"
            was_right = predicted == pack.label_of(record)
            correct += was_right
            gov.record_outcome(query.query_id, correct=was_right, label=predicted)

    print(f"  {len(live_raw)} live cases")
    print(f"    answered:             {answered}")
    print(f"    no applicable logic:  {uncovered}")
    print(f"    blocked for approval: {blocked}")
    print(f"    committed to a call:  {scored}  ({correct} correct, "
          f"{correct / max(1, scored):.0%})")
    print(f"    reviews required:     0")

    # ------------------------------------------------------------------
    banner(6, "What one act of review bought")
    print(gov.amortisation_report())

    flagged = gov.rules_needing_rereview()
    print(f"\nrules whose evidence has turned against them: {len(flagged)}")
    for record in flagged:
        print(f"  {record.rule_id}  claimed {record.strength:.2f}, observed "
              f"{record.stats.posterior_mean():.2f} over "
              f"{record.stats.observations} outcomes")
        print(f"    {record.rule}")
    if flagged:
        print("  -> the one case where an approved rule comes back: the reviewer is")
        print("     being shown new evidence, not asked to repeat themselves.")

    # ------------------------------------------------------------------
    banner(7, "A decision, explained")
    example = next(
        (q for q in gov.queries.values() if q.status == "answered" and q.rule_ids),
        None,
    )
    if example is not None:
        facts = gov.store.replay_facts(example.query_id, pack.vocabulary)
        answer = gov.solver.answer(
            example.goal, facts, critical=example.critical,
            competing=pack.competing_goals(example.entity),
        )
        print(explainer.explain_answer(answer, facts, critical=example.critical))

    violations = gov.verify_invariant()
    print(f"\ngovernance invariant violations: {len(violations)}")
    print(f"database: {db}  {gov.store.counts()}")
    gov.store.close()


if __name__ == "__main__":
    main()
