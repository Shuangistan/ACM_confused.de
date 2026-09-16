"""End-to-end walkthrough of the governed logic database.

Run:  python demo.py

Shows the full loop the design is for:

  1. A routine query runs on provisional logic and keeps working.
  2. A critical query on the same logic BLOCKS -- it cannot proceed on rules no
     human has approved.
  3. A human approves the rule once.
  4. The blocked query now goes through.
  5. Four more similar cases reuse that approval with no further review. This is
     the amortisation the whole design turns on.
  6. A case with a missing fact yields an interval, not a guess, plus the one
     question worth asking.
  7. The proof is rendered from the actual derivation, with provenance.
  8. A provisional rule is rejected, and every decision that used it is named.
  9. The governance invariant is audited over the whole run.
"""

from __future__ import annotations

from logicdb.explain import Explainer
from logicdb.facts import FactStore, Truth
from logicdb.governance import (
    BlockedForApproval,
    CriticalityPolicy,
    CriticalityRule,
    Governor,
)
from logicdb.parser import parse_atom, parse_rule
from logicdb.program import RuleBase, RuleOrigin
from logicdb.syntax import PredicateDecl, Vocabulary

RULE = "─" * 78


def banner(n: int, title: str) -> None:
    print(f"\n{RULE}\n {n}. {title}\n{RULE}")


def build_vocabulary() -> Vocabulary:
    return Vocabulary(
        [
            PredicateDecl("overdrawn", 1, "observable",
                          phrase="{0} is overdrawn on its current account"),
            PredicateDecl("thin_file", 1, "observable",
                          phrase="{0} has a thin credit file"),
            PredicateDecl("guarantor", 1, "observable",
                          phrase="{0} has a guarantor"),
            PredicateDecl("employed_long", 1, "observable",
                          phrase="{0} has been in employment 4+ years"),
            PredicateDecl("large_amount", 1, "observable",
                          phrase="{0} requests a large amount"),
            PredicateDecl("high_risk", 1, "derived", phrase="{0} is high risk"),
            PredicateDecl("mitigated", 1, "derived",
                          phrase="{0} has mitigating circumstances"),
            PredicateDecl("decline", 1, "derived", phrase="{0} should be declined"),
        ]
    )


def facts_for(vocab: Vocabulary, entity: str, **kw: str) -> FactStore:
    store = FactStore(vocab)
    sources = {
        "overdrawn": "application form, field 3",
        "thin_file": "credit bureau report",
        "guarantor": "application form, section 7",
        "employed_long": "employer letter",
        "large_amount": "application form, field 1",
    }
    for predicate, state in kw.items():
        store.assert_fact(
            parse_atom(f"{predicate}({entity})"),
            Truth(state),
            source=sources.get(predicate, "unspecified")
            if state != "unknown"
            else "not supplied by the applicant",
        )
    return store


def main() -> None:
    vocab = build_vocabulary()
    policy = CriticalityPolicy(
        rules=[
            CriticalityRule(
                rule_id="large_sums",
                description="loans of 10,000 or more are reviewed online",
                field="amount", op="gte", value=10_000,
            )
        ],
        default_critical=False,
    )
    gov = Governor(RuleBase(vocab), criticality=policy, vocabulary=vocab)
    explainer = Explainer(gov.rulebase, vocab)

    # Seed logic that a human already signed off, so the demo is not starting
    # from nothing.
    for text, strength in [
        ("mitigated(A) <- guarantor(A).", 0.90),
        ("decline(A) <- high_risk(A), not mitigated(A).", 0.88),
    ]:
        seeded = gov.propose(
            parse_rule(text), origin=RuleOrigin.SEED, proposed_by="pack",
            critical_context=True, strength=strength,
        )
        gov.approve(seeded.record.rule_id, by="policy-team")

    banner(1, "A routine query runs on logic the agent just proposed")
    proposal = gov.propose(
        parse_rule("high_risk(A) <- overdrawn(A), thin_file(A)."),
        origin=RuleOrigin.MINED, proposed_by="agent",
        critical_context=False, strength=0.80,
        mining_support=214, mining_precision=0.79,
    )
    print(f"agent proposes:  {proposal.record.rule}")
    print(f"routing:         {proposal.message}")

    facts = facts_for(vocab, "app_small", overdrawn="true", thin_file="true",
                      guarantor="false", employed_long="false", large_amount="false")
    answer, query = gov.ask(
        parse_atom("decline(app_small)"), facts, "app_small", "officer-1",
        context={"amount": 800},
    )
    print(f"\nquery {query.query_id} (amount 800, routine): P = {answer.bounds}")
    print(f"  used rules: {', '.join(query.rule_ids)}")
    print(f"  provisional: {', '.join(query.provisional_rule_ids) or 'none'}")
    print("  -> routine work is not stalled waiting for a reviewer.")

    banner(2, "The same logic, a critical query: it blocks")
    big_facts = facts_for(vocab, "app_big", overdrawn="true", thin_file="true",
                          guarantor="false", employed_long="false", large_amount="true")
    try:
        gov.ask(parse_atom("decline(app_big)"), big_facts, "app_big", "officer-1",
                context={"amount": 45_000})
        print("  ERROR: this should not have been reachable")
    except BlockedForApproval as blocked:
        print(f"BLOCKED: {blocked}")
        print("  -> a critical decision cannot rest on unreviewed logic.")

    banner(3, "A human reviews the rule, once")
    pending = gov.review_queue()
    for record in pending:
        print(record.describe())
    gov.approve(proposal.record.rule_id, by="reviewer-jd",
                note="Checked against lending policy s4.2. Sound.")
    print(f"\napproved {proposal.record.rule_id} by reviewer-jd")

    banner(4, "The blocked query now proceeds")
    answer, query = gov.ask(
        parse_atom("decline(app_big)"), big_facts, "app_big", "officer-1",
        context={"amount": 45_000},
    )
    print(f"query {query.query_id} (amount 45,000, critical): P = {answer.bounds}")
    print(f"  criticality: {query.criticality_reason}")

    banner(5, "Four more critical cases -- no further approval required")
    for i in range(4):
        entity = f"app_{i}"
        f = facts_for(vocab, entity, overdrawn="true", thin_file="true",
                      guarantor="false", employed_long="false", large_amount="true")
        a, q = gov.ask(parse_atom(f"decline({entity})"), f, entity, "officer-2",
                       context={"amount": 20_000 + i})
        print(f"  {q.query_id}  {entity:9s}  P = {a.bounds}  reviews required: 0")
    print(f"\nreview queue is {'empty' if not gov.review_queue() else 'NOT empty'}")
    print(f"{proposal.record.rule_id} has now served "
          f"{proposal.record.stats.reuse_count} queries on one approval.")

    banner(6, "A case where a critical fact is missing")
    unknown_facts = facts_for(vocab, "app_gap", overdrawn="true", thin_file="true",
                              guarantor="unknown", employed_long="false",
                              large_amount="true")
    answer, query = gov.ask(
        parse_atom("decline(app_gap)"), unknown_facts, "app_gap", "officer-2",
        context={"amount": 30_000},
    )
    print(f"query {query.query_id}: P = {answer.bounds}   (width {answer.bounds.width:.2f})")
    print("\nThe system does not guess. It says what it would need:")
    for candidate in answer.ask:
        tag = "DECISIVE" if candidate.decisive else f"gain {candidate.information_gain:.2f}"
        print(f"  ask: {candidate.atom}   [{tag}]")
        print(f"       if true  -> {candidate.if_true}")
        print(f"       if false -> {candidate.if_false}")

    banner(7, "The explanation is the derivation, not a story about it")
    print(explainer.explain_answer(answer, unknown_facts, critical=True))
    print()
    print(explainer.provenance_report(parse_atom("high_risk(app_gap)"), answer.evaluation))

    banner(8, "A provisional rule is rejected: what did it already touch?")
    weak = gov.propose(
        parse_rule("high_risk(A) <- large_amount(A), not employed_long(A)."),
        origin=RuleOrigin.MINED, proposed_by="agent",
        critical_context=False, strength=0.6,
        mining_support=41, mining_precision=0.58,
    )
    print(f"agent proposes:  {weak.record.rule}")
    print(f"routing:         {weak.message}\n")
    for i in range(3):
        entity = f"routine_{i}"
        f = facts_for(vocab, entity, overdrawn="false", thin_file="false",
                      guarantor="false", employed_long="false", large_amount="true")
        gov.ask(parse_atom(f"decline({entity})"), f, entity, "officer-3",
                context={"amount": 900})
    report = gov.reject(weak.record.rule_id, by="reviewer-jd",
                        reason="precision 0.58 is barely above chance; not a reason")
    print(report.render())

    banner(9, "Audit")
    print(gov.amortisation_report())
    violations = gov.verify_invariant()
    print(f"\ngovernance invariant violations: {len(violations)}")
    if violations:
        for v in violations:
            print(f"  {v}")
    else:
        print("  No critical decision rested on unapproved logic.")
    print()
    summary = gov.governance_summary()
    width = max(len(k) for k in summary)
    for key, value in summary.items():
        print(f"  {key:<{width}}  {value}")


if __name__ == "__main__":
    main()
