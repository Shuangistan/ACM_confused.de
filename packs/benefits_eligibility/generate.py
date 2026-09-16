"""Generate synthetic eligibility cases from a known rule set.

Run:  python packs/benefits_eligibility/generate.py [--n 800] [--noise 0.05]
                                                    [--missing 0.08]

The generating rules in `TRUE_RULES` are the ground truth the mining evaluation
scores against. Nothing else in the system reads them -- they exist so that
`tests/test_mining.py` can ask the question a real dataset cannot answer: not
"does this mined rule predict well" but "is it the right rule".

Three properties are built in deliberately, because a generator that produced
clean separable data would make the miner look better than it is:

* **Irrelevant attributes.** `owns_bicycle` and `urban_postcode` bear on nothing.
  A miner that proposes rules over them is finding noise, and the evaluation
  will say so.
* **Label noise.** A fraction of outcomes are flipped, so no rule achieves
  precision 1.0 and the thresholds have to do real work.
* **Missing values.** Some fields are absent, which exercises the distinction
  between unknown and false through the whole pipeline rather than only in tests.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
DEFAULT_SEED = 20260915

#: The hidden logic. Each entry is (name, required-true attributes,
#: required-false attributes). An applicant is eligible if any rule is satisfied.
TRUE_RULES: list[tuple[str, tuple[str, ...], tuple[str, ...]]] = [
    ("hardship_core", ("low_income", "resident"), ("savings_over_limit",)),
    ("dependants_route", ("has_dependants", "arrears", "resident"), ()),
    ("disability_route", ("disability_premium", "resident"), ("savings_over_limit",)),
]

ATTRIBUTES = [
    "low_income", "has_dependants", "recent_award", "resident",
    "savings_over_limit", "disability_premium", "employed", "arrears",
    "owns_bicycle", "urban_postcode",
]

#: Marginal probabilities. `resident` is common because it gates two of the
#: three routes; if it were rare, most cases would be trivially ineligible and
#: the miner would have almost nothing to work with.
PRIORS = {
    "low_income": 0.45,
    "has_dependants": 0.40,
    "recent_award": 0.20,
    "resident": 0.85,
    "savings_over_limit": 0.25,
    "disability_premium": 0.18,
    "employed": 0.55,
    "arrears": 0.30,
    "owns_bicycle": 0.35,
    "urban_postcode": 0.60,
}

#: Fields allowed to go missing, and how often. `savings_over_limit` is the
#: interesting one: it appears negated in two true rules, so an unknown value
#: genuinely blocks a conclusion rather than merely weakening it.
MISSING_PRONE = ("savings_over_limit", "disability_premium", "arrears")


def satisfies(record: dict, required: tuple[str, ...], forbidden: tuple[str, ...]) -> bool:
    return all(record.get(a) for a in required) and not any(
        record.get(f) for f in forbidden
    )


def true_label(record: dict) -> tuple[str, list[str]]:
    """Apply the hidden rules. Returns the outcome and which routes fired."""
    fired = [
        name for name, required, forbidden in TRUE_RULES
        if satisfies(record, required, forbidden)
    ]
    return ("eligible" if fired else "not_eligible"), fired


def generate(n: int, noise: float, missing: float, seed: int) -> list[dict]:
    rng = np.random.default_rng(seed)
    cases: list[dict] = []

    for i in range(n):
        record = {a: bool(rng.random() < PRIORS[a]) for a in ATTRIBUTES}

        # Label from the complete record, before anything is hidden. The truth
        # does not depend on what was written down, which is exactly why an
        # unknown field has to widen the answer rather than change it.
        label, fired = true_label(record)

        if rng.random() < noise:
            label = "not_eligible" if label == "eligible" else "eligible"
            flipped = True
        else:
            flipped = False

        for attribute in MISSING_PRONE:
            if rng.random() < missing:
                record[attribute] = None

        award = int(rng.choice([250, 500, 900, 1500, 2500, 4000],
                               p=[0.25, 0.25, 0.2, 0.15, 0.1, 0.05]))
        cases.append(
            {
                "id": f"case_{i:04d}",
                **record,
                "award_amount": award,
                "appeal_waived": bool(rng.random() < 0.08),
                "outcome": label,
                # Kept for analysis only; never fed to the agent.
                "_routes_fired": fired,
                "_label_flipped": flipped,
            }
        )
    return cases


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=800)
    ap.add_argument("--noise", type=float, default=0.05)
    ap.add_argument("--missing", type=float, default=0.08)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = ap.parse_args()

    cases = generate(args.n, args.noise, args.missing, args.seed)
    out = HERE / "cases.json"
    out.write_text(json.dumps(cases, indent=2), encoding="utf-8")

    eligible = sum(1 for c in cases if c["outcome"] == "eligible")
    flipped = sum(1 for c in cases if c["_label_flipped"])
    missing_counts = {
        a: sum(1 for c in cases if c[a] is None) for a in MISSING_PRONE
    }
    route_counts: dict[str, int] = {}
    for case in cases:
        for route in case["_routes_fired"]:
            route_counts[route] = route_counts.get(route, 0) + 1

    print(f"wrote {len(cases)} cases to {out}")
    print(f"  eligible:      {eligible} ({eligible / len(cases):.1%})")
    print(f"  labels flipped:{flipped} ({flipped / len(cases):.1%})")
    print(f"  missing values:{missing_counts}")
    print(f"  true routes:   {route_counts}")
    print("\nHidden rules the miner will be scored against:")
    for name, required, forbidden in TRUE_RULES:
        body = " AND ".join(list(required) + [f"NOT {f}" for f in forbidden])
        print(f"  {name:18s} eligible <- {body}")


if __name__ == "__main__":
    main()
