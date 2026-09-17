"""The graph, end to end: route, decide, allocate oversight, audit, close gaps.

Run with `python demo_loop.py`. No API key needed -- the scripted agent is the
default, because a demonstration of determinism that cannot be re-run is not
one.

What to watch, in order:

1. Oversight is allocated by *rules*, and each allocation names the rule that
   produced it. Not a branch in this file.
2. An empty rule base decides nothing and says so. `uncovered` is not `no`.
3. The gap provokes proposals; the checker filters what a machine can filter;
   the rest stops at `interrupt()` for a person.
4. The cycle after that proposes nothing, which is what ends the run.
5. Audit sampling starts near 100% on fresh rules and decays as they earn
   evidence -- never to zero, and it climbs back when they start failing.
6. The checker exhausts the input space and reports on the base as a whole.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from logicdb.agents.base import LabelledCase
from logicdb.graph import run_graph
from logicdb.oversight import Mode
from logicdb.program import RuleStatus
from logicdb.system import System

RULE = "=" * 74


def banner(n: int, title: str) -> None:
    print(f"\n{RULE}\n {n}. {title}\n{RULE}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pack", default="recruitment")
    ap.add_argument("--db", default="data/demo_loop.sqlite")
    ap.add_argument("--train", type=int, default=400)
    ap.add_argument("--live", type=int, default=600)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    db = Path(args.db)
    if db.exists():
        db.unlink()  # a fresh start, so the demo always tells the same story

    system = System.build(args.pack, db)
    pack = system.pack
    raw = json.loads((pack.root / "cases.json").read_text())
    random.Random(args.seed).shuffle(raw)
    train_raw = raw[: args.train]
    live_raw = raw[args.train : args.train + args.live]
    labelled = [
        LabelledCase(r["id"], pack.facts_from(r, r["id"]), pack.label_of(r), r)
        for r in train_raw
    ]
    truth = {r["id"]: pack.label_of(r) for r in raw}

    print(f"pack:      {pack.name}  ({len(pack.vocabulary)} predicates)")
    print(f"oversight: {system.oversight.pack.name}  "
          f"({len(list(system.oversight.rulebase))} rules, seeded approved)")
    print(f"caseload:  {len(live_raw)} live, {len(labelled)} labelled for evidence")
    print(f"declared:  harm={pack.oversight['harm']}, "
          f"reversibility={pack.oversight['reversibility']}")

    # ------------------------------------------------------------------
    banner(1, "The ladder is rules, and it names the rule that fired")
    for label, h, r, rights, width in [
        ("routine screening", pack.oversight["harm"],
         pack.oversight["reversibility"], False, 0.05),
        ("right-to-work in question", pack.oversight["harm"],
         pack.oversight["reversibility"], True, 0.05),
        ("evidence too thin", 2, 9, False, 0.55),
        ("low harm, reversible", 2, 9, False, 0.05),
    ]:
        mode, used = system.oversight.assess_mode(label, h, r, rights, width)
        print(f"  {label:26} -> {mode.value:18} by {', '.join(sorted(used))}")
    print("\n  A proof, not a branch. These rules carry an approver and a citation")
    print("  like any other.")

    # ------------------------------------------------------------------
    banner(2, "The graph")

    def reviewer(record) -> tuple[bool, str]:
        """Stands in for a person at the gate."""
        if record.source_citation:
            return True, f"checked against {record.source_citation}"
        if (record.mining_support or 0) >= 20 and (record.mining_precision or 0) >= 0.80:
            return True, (
                f"support {record.mining_support}, "
                f"precision {record.mining_precision:.0%}"
            )
        return False, "not enough evidence to vouch for this"

    def auditor(decision) -> tuple[bool, str | None]:
        """Stands in for a person checking one decision against what happened."""
        said = max(
            decision.answer.outcomes, key=lambda k: decision.answer.outcomes[k].lower
        )
        # outcome keys are rendered atoms -- reject("CV01733") -- not predicates
        return said.split("(")[0] == truth.get(decision.entity), truth.get(decision.entity)

    run = run_graph(
        system, live_raw, labelled, auditor=auditor, reviewer=reviewer, max_cycles=4
    )
    print(run.describe())
    print(f"\n  stopped after {len(run.reports)} cycles -- the last changed nothing,")
    print("  which is the graph's own stop test, not a fixed iteration count.")

    approved = [r for r in system.governor.rulebase if r.status is RuleStatus.APPROVED]
    rejected = [r for r in system.governor.rulebase if r.status is RuleStatus.REJECTED]
    print(f"  rule base now: {len(approved)} approved, {len(rejected)} rejected")

    # ------------------------------------------------------------------
    banner(3, "Where the oversight went")
    modes: dict[str, int] = {}
    audited = automatic = 0
    example = None
    for record in live_raw:
        decision = system.node.decide(record)
        if decision.assessment is None:
            continue
        key = decision.assessment.mode.value
        modes[key] = modes.get(key, 0) + 1
        if decision.assessment.sampled:
            audited += 1
            example = example or decision
        if decision.took_effect:
            automatic += 1
    total = sum(modes.values()) or 1
    for mode, count in sorted(modes.items(), key=lambda kv: -kv[1]):
        print(f"  {mode:20} {count:>5}  ({count / total:.0%})")
    print(f"\n  decided without a person: {automatic} of {total}")
    print(f"  pulled for outcome audit: {audited}")
    if example is not None:
        print("\n  a sampled case, with the reason the rate came out where it did:\n")
        for line in example.assessment.describe().splitlines():
            print(f"    {line}")

    # ------------------------------------------------------------------
    banner(4, "Sampling decays as a rule earns evidence")
    target = next(
        (r for r in approved if r.rule.head.predicate in pack.outcomes), None
    )
    if target is not None:
        print(f"  rule {target.rule_id} ({target.origin.value}), "
              f"as outcomes accumulate:\n")
        print(f"    {'right/seen':>12}  {'audit rate':>10}")
        for successes, failures in [
            (0, 0), (5, 0), (20, 0), (50, 0), (100, 0), (100, 10), (100, 30)
        ]:
            target.stats.successes, target.stats.failures = successes, failures
            rate, _ = system.oversight.audit_rate(Mode.AI_WITH_OVERSIGHT, [target])
            print(f"    {successes:>4}/{successes + failures:<7}  {rate:>9.1%}  "
                  f"{'#' * max(1, round(rate * 40))}")
        target.stats.successes = target.stats.failures = 0
        print("\n  Nothing was tuned. That is the Beta posterior the rule already")
        print("  maintained, read as P(accuracy below target). Failures push the")
        print("  rate back up by themselves, and it never reaches zero.")

    # ------------------------------------------------------------------
    banner(5, "The rule base as a whole")
    report = system.checker.check(system.governor.rulebase, deep=True)
    for line in report.describe().splitlines():
        print(f"  {line}")
    print("\n  Every input, not every case seen so far. A property that holds on")
    print("  600 cases may still fail on the 601st; this one cannot.")
    print(f"\n  governance invariant violations: {system.summary()['invariant_violations']}")


if __name__ == "__main__":
    main()
