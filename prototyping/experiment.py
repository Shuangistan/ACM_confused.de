"""The coverage experiment the paper reports.

Runs a caseload past an empty database, approving every rule the agent proposes,
and records how the share of decisions made deterministically moves. Everything
the paper claims about coverage comes from here, so it is a script rather than a
session someone remembers.

    python experiment.py --cases 12

It is deliberately small. The point is not a benchmark — there is no ground
truth to benchmark against — but a measurement of whether the database
accumulates logic that answers later cases, and of how much of that logic turns
out to be one clause per case shape.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from agent import db, graph  # noqa: E402
from logic.store import APPROVED  # noqa: E402

#: Four shapes, three phrasings each. If the database is generalising, the
#: second and third phrasing of a shape should be decided by the rule the first
#: one produced. If it is only memorising, every one arrives uncovered.
CASELOAD = [
    # -- consumables, low stakes, reversible
    "Should we reorder toner now? Stock is low but we have not run out.",
    "Should we reorder printer paper now? Stock is low, not yet out.",
    "Do we order more envelopes? We are low but not out of them.",
    # -- scheduling, trivial
    "Do we move the standup to Thursday this week?",
    "Should we shift the team meeting to Friday morning?",
    "Do we reschedule the review to next Tuesday?",
    # -- tenancy, consequential
    "We need to decide whether to evict a tenant three months in arrears who "
    "disputes the debt and was given no formal notice.",
    "We need to decide whether to evict another tenant in arrears who disputes "
    "the amount, with no formal notice sent.",
    "We must decide whether to end a tenancy over arrears the tenant disputes; "
    "no formal notice was given.",
    # -- support withdrawal, rights-engaging
    "We need to decide whether to withdraw a disabled student's funded support "
    "after one disputed attendance report.",
    "Should we stop a student's disability funding over a single contested "
    "attendance record?",
    "We need to decide whether to end funded support for a disabled student on "
    "one disputed report.",
]

INPUTS = {"harm": 5, "reversibility": 5, "rights": False, "confidence": 50}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cases", type=int, default=len(CASELOAD))
    ap.add_argument("--model", default="haiku")
    ap.add_argument("--out", default="experiment.json")
    ap.add_argument("--keep", action="store_true",
                    help="do not reset the database first")
    args = ap.parse_args()

    if not args.keep:
        # Never while the server is running: it holds the file open, and
        # deleting it underneath leaves SQLite reporting a readonly database.
        db.DB_PATH.unlink(missing_ok=True)
        db._store = None

    store = db.store()
    rows = []
    print(f"{'#':>3}  {'tier':18} {'covered':8} {'parked':7} {'rules':6} coverage")
    print("-" * 66)

    for index, question in enumerate(CASELOAD[: args.cases], start=1):
        out = graph.turn(f"exp-{index}", question, args.model, "auto", INPUTS)

        # An expert approves whatever was proposed. Generous on purpose: the
        # question here is whether approved logic gets reused, not whether a
        # reviewer would have accepted it.
        approved_now = 0
        for rule in store.rules("proposed"):
            store.set_status(rule.canonical_key, APPROVED, "experiment",
                             "approved for the coverage measurement")
            approved_now += 1

        stats = store.stats()
        rows.append({
            "n": index,
            "question": question,
            "tier": out.get("tier", ""),
            "covered": bool(out.get("covered")),
            "blocked": bool(out.get("blocked")),
            "conclusions": out.get("conclusions", []),
            "approved_after": stats["rules"].get("approved", 0),
            "coverage": stats["coverage"],
        })
        print(f"{index:>3}  {out.get('tier') or '—':18} "
              f"{str(bool(out.get('covered'))):8} "
              f"{str(bool(out.get('blocked'))):7} "
              f"{stats['rules'].get('approved', 0):<6} {stats['coverage']:.0%}")

    stats = store.stats()
    vocab = store.vocabulary()
    summary = {
        "cases": len(rows),
        "coverage": stats["coverage"],
        "rules": stats["rules"],
        "predicates": len(vocab),
        "vocabulary": vocab,
        "rows": rows,
    }
    Path(args.out).write_text(json.dumps(summary, indent=2))

    print("-" * 66)
    print(f"  coverage        {stats['coverage']:.0%}")
    print(f"  approved rules  {stats['rules'].get('approved', 0)}")
    print(f"  distinct terms  {len(vocab)}")
    print(f"  written to      {args.out}")


if __name__ == "__main__":
    main()
