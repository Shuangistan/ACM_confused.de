"""Decode the UCI Statlog German Credit data into records for the pack.

Run:  python packs/consumer_credit/build.py [--missing 0.10]

Two jobs beyond decoding:

* **Human-readable fields.** The raw attributes are codes (A11, A34, ...). A
  reviewer asked to approve `decline(A) <- checking_A11(A)` has no way to judge
  it. Decoded into `overdrawn`, they can.

* **Realistic gaps.** Real credit files have holes, and the distinction between
  "no guarantor" and "nobody recorded a guarantor" is the one this whole system
  is built to preserve. A dataset with no missing values would let that go
  untested on the one pack drawn from real data, so gaps are introduced
  deliberately, concentrated on the fields where absence is plausible.

Sex is decoded and carried for subgroup analysis but is *not* mapped to a
predicate in `pack.yaml`. It is therefore absent from the vocabulary, which
means the agent cannot mine a rule over it -- a constraint enforced by the
closed vocabulary rather than by anyone remembering not to.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
DEFAULT_SEED = 20260915

COLUMNS = [
    "checking_status", "duration_months", "credit_history", "purpose",
    "credit_amount", "savings_status", "employment_since", "installment_rate",
    "personal_status_sex", "other_debtors", "residence_since", "property",
    "age_years", "other_installment_plans", "housing", "existing_credits",
    "job", "dependants", "telephone", "foreign_worker", "target",
]

#: Fields where a gap is plausible in a real file. Guarantor status leads the
#: list on purpose: it appears negated in the lending policy, so an unknown
#: value genuinely blocks a conclusion rather than merely weakening one.
MISSING_PRONE = ("other_debtors", "savings_status", "employment_since", "property")


def decode(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    out["id"] = [f"app_{i:04d}" for i in range(len(df))]

    out["overdrawn"] = df["checking_status"].eq("A11")
    out["no_checking_account"] = df["checking_status"].eq("A14")
    out["healthy_balance"] = df["checking_status"].eq("A13")

    out["adverse_history"] = df["credit_history"].isin(["A33", "A34"])
    out["clean_history"] = df["credit_history"].isin(["A30", "A31"])

    out["thin_savings"] = df["savings_status"].isin(["A61", "A65"])
    out["substantial_savings"] = df["savings_status"].isin(["A63", "A64"])

    out["employed_long"] = df["employment_since"].isin(["A74", "A75"])
    out["unemployed"] = df["employment_since"].eq("A71")

    out["guarantor"] = df["other_debtors"].isin(["A102", "A103"])
    out["owns_property"] = df["property"].eq("A121")
    out["no_property"] = df["property"].eq("A124")

    out["large_amount"] = df["credit_amount"] >= 5000
    out["long_term"] = df["duration_months"] >= 24
    out["high_instalment_burden"] = df["installment_rate"] >= 4
    out["multiple_credits"] = df["existing_credits"] >= 2
    out["other_plans"] = df["other_installment_plans"].isin(["A141", "A142"])
    out["young_applicant"] = df["age_years"] < 26
    out["foreign_worker"] = df["foreign_worker"].eq("A201")

    # Carried for analysis only; deliberately not a declared predicate.
    out["sex"] = df["personal_status_sex"].map(
        lambda c: "female" if c in ("A92", "A95") else "male"
    )
    out["credit_amount"] = df["credit_amount"].astype(int)
    out["duration_months"] = df["duration_months"].astype(int)
    out["age_years"] = df["age_years"].astype(int)

    # 1 = good risk, 2 = bad. The decision predicate is `decline`, so the
    # positive outcome for mining is the bad-risk case.
    out["outcome"] = np.where(df["target"] == 2, "decline", "approve")
    return out


#: Which decoded columns a missing raw field should blank out.
BLANKS: dict[str, tuple[str, ...]] = {
    "other_debtors": ("guarantor",),
    "savings_status": ("thin_savings", "substantial_savings"),
    "employment_since": ("employed_long", "unemployed"),
    "property": ("owns_property", "no_property"),
}


def introduce_gaps(records: list[dict], rate: float, seed: int) -> dict[str, int]:
    rng = np.random.default_rng(seed)
    counts = {field: 0 for field in MISSING_PRONE}
    for record in records:
        for field in MISSING_PRONE:
            if rng.random() < rate:
                for column in BLANKS[field]:
                    record[column] = None
                counts[field] += 1
    return counts


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--missing", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = ap.parse_args()

    raw = HERE / "german.data"
    if not raw.exists():
        raise SystemExit(
            f"{raw} not found. Download from "
            f"https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data"
        )

    df = pd.read_csv(raw, sep=r"\s+", header=None, names=COLUMNS)
    decoded = decode(df)
    records = decoded.to_dict(orient="records")
    for record in records:
        for key, value in list(record.items()):
            if isinstance(value, (np.bool_, bool)):
                record[key] = bool(value)
            elif isinstance(value, (np.integer,)):
                record[key] = int(value)

    gaps = introduce_gaps(records, args.missing, args.seed)

    out = HERE / "cases.json"
    out.write_text(json.dumps(records, indent=2), encoding="utf-8")

    declines = sum(1 for r in records if r["outcome"] == "decline")
    by_sex: dict[str, int] = {}
    for record in records:
        by_sex[record["sex"]] = by_sex.get(record["sex"], 0) + 1

    print(f"wrote {len(records)} cases to {out}")
    print(f"  decline (bad risk): {declines} ({declines / len(records):.1%})")
    print(f"  subgroups:          {by_sex}")
    print(f"  fields blanked:     {gaps}")
    print(
        "\nSex is carried on each record for subgroup analysis but is not a "
        "declared predicate, so the agent cannot mine a rule over it."
    )


if __name__ == "__main__":
    main()
