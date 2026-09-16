"""Generate the recruitment campaign: 8,000 applicants, four branches.

Mirrors the scenario in the group's earlier client-side prototype so the two
can be compared directly. The distributions are deliberately the same shape:
most applications are clean, a minority carry an explicit failure, a minority
have a gap in the evidence, and a few have a timeline conflict.

Two things differ from the earlier prototype, and both matter for what can be
measured:

* **There is a ground truth.** Each applicant carries a hidden `suitable` flag
  set from the full record before any evidence is hidden. So a rejection can be
  *wrong* -- an applicant who was in fact suitable but whose evidence was
  incomplete or mis-stated. Without that, an auditor has nothing to find and
  contestability can be shown as an affordance but never exercised.

* **Protected attributes exist in the data and not in the vocabulary.** `age_band`
  and `sex` are recorded for subgroup analysis and are not declared predicates,
  so no rule can be written or mined over them. One is correlated with a
  legitimate-looking signal on purpose, so the subgroup gap is detectable.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
DEFAULT_SEED = 20260916

BRANCHES = [
    ("North", "Customer Support Specialist", "junior"),
    ("South", "Operations Coordinator", "senior"),
    ("East", "Sales Associate", "junior"),
    ("West", "IT Support Analyst", "specialist"),
]


def generate(n: int, seed: int) -> list[dict]:
    rng = np.random.default_rng(seed)
    out: list[dict] = []

    for i in range(n):
        b = i % 4
        branch, role, seniority = BRANCHES[b]

        # Ground truth first, from the complete picture. Everything below either
        # reveals or hides parts of it -- which is what makes a rejection able to
        # be wrong rather than correct by construction.
        suitable = bool(rng.random() < 0.58)

        rec = {
            "id": f"CV{i + 1:05d}",
            "applicant": True,
            "branch": branch,
            "branch_role": role,
            "seniority": seniority,
            # No second reviewer on roughly a third of screens: that is the
            # condition under which a rejection cannot be caught later.
            "sole_screener": bool(rng.random() < 0.34),
            "right_to_work": bool(rng.random() < 0.97),
            "applied_in_window": bool(rng.random() < 0.95),
            "holds_credential": bool(rng.random() < 0.7),
            # Protected attributes: recorded, never declared as predicates.
            "sex": "female" if rng.random() < 0.46 else "male",
            "age_band": str(rng.choice(["under_25", "25_39", "40_plus"],
                                       p=[0.34, 0.45, 0.21])),
            "_suitable": suitable,
        }

        # A suitable applicant usually evidences the requirements; an unsuitable
        # one usually does not. The overlap is the interesting region.
        rec["meets_experience"] = bool(rng.random() < (0.88 if suitable else 0.32))
        rec["has_core_skill"] = bool(rng.random() < (0.9 if suitable else 0.3))
        rec["has_second_skill"] = bool(rng.random() < (0.82 if suitable else 0.4))

        rec["missing_experience_evidence"] = False
        rec["skill_evidence_absent"] = False
        rec["timeline_conflict"] = bool(rng.random() < 0.08)

        # Evidence gaps. A gap on a suitable applicant is where a careless
        # screen produces a wrong rejection, so gaps are deliberately not rare.
        if rng.random() < 0.11:
            rec["missing_experience_evidence"] = True
            rec["meets_experience"] = None
        if rng.random() < 0.10:
            rec["skill_evidence_absent"] = True
            rec["has_core_skill"] = None

        # Older applicants are likelier to have a career-break timeline that
        # reads as a conflict. A legitimate-looking signal that tracks a
        # protected attribute -- exactly the proxy a reviewer cannot reliably
        # spot, and the reason s4.1 is enforced by the vocabulary instead.
        if rec["age_band"] == "40_plus" and rng.random() < 0.22:
            rec["timeline_conflict"] = True

        rec["outcome"] = "advance" if suitable else "reject"
        out.append(rec)

    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=8000)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = ap.parse_args()

    cases = generate(args.n, args.seed)
    (HERE / "cases.json").write_text(json.dumps(cases, indent=2), encoding="utf-8")

    suitable = sum(1 for c in cases if c["_suitable"])
    gaps = sum(1 for c in cases
               if c["missing_experience_evidence"] or c["skill_evidence_absent"])
    conflicts = sum(1 for c in cases if c["timeline_conflict"])
    critical = sum(1 for c in cases
                   if c["seniority"] in ("senior", "specialist") or c["sole_screener"])

    print(f"wrote {len(cases)} applicants to {HERE / 'cases.json'}")
    print(f"  suitable (hidden ground truth): {suitable} ({suitable/len(cases):.0%})")
    print(f"  evidence gaps:                  {gaps} ({gaps/len(cases):.0%})")
    print(f"  timeline conflicts:             {conflicts} ({conflicts/len(cases):.0%})")
    print(f"  critical screens:               {critical} ({critical/len(cases):.0%})")
    by_age = {}
    for c in cases:
        if c["timeline_conflict"]:
            by_age[c["age_band"]] = by_age.get(c["age_band"], 0) + 1
    print(f"  conflicts by age band:          {by_age}")
    print("\n  age_band and sex are recorded but are NOT declared predicates,")
    print("  so no rule can be written or mined over either.")


if __name__ == "__main__":
    main()
