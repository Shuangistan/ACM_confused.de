# Demo runbook

Seven beats, ~8 minutes. Every step below was executed against the live app and
the outputs are what it actually returns — no step is aspirational.

There are two runnable demos. Pick by audience:

| | surface | best for |
|---|---|---|
| **A. The policy gap** | classic pages, `consumer_credit` | the strongest single moment (beat 7) |
| **B. The control room** | `/console`, `recruitment` | the most visual, and the head-to-head against our own earlier prototype |

Both are tested. B is the better opener for a jury; A has the better punchline.

---

# A. The policy gap — classic pages

## Setup (do this before the room fills)

```bash
cd /tools/ACM_SummerSchool
conda activate iese
rm -f data/logicdb.sqlite      # the story depends on starting empty
./run.sh consumer_credit
```

Have two tabs open: **`/query`** and **`/review`**. Keep
`packs/consumer_credit/lending_policy.md` open in an editor — beat 3 needs it.

---

## Beat 1 — The empty database refuses to guess (45s)

`/query` → **`app_0001`**

> **No applicable logic.** The rule base has nothing that bears on this case.
> This is *not* a negative answer — it is an absence of logic.

**Say:** "Most systems would return a score here. This one distinguishes *I
have nothing to say* from *no*. That distinction is the whole reason the next
step exists."

## Beat 2 — The agent drafts logic from written policy (60s)

`/agent` → **Extract from documents** → 8 rules, each citing a clause.

**Say:** "The agent proposed; it did not decide. Each rule cites the section it
claims to encode, so a reviewer can open the policy and compare — that's the
difference between a rule you can check and one you can only trust."

## Beat 3 — A critical query blocks (90s — the centrepiece)

`/query` → **`app_0003`** (5,234 DM — above the 5,000 threshold, so critical)

> **Blocked — awaiting approval.** Needs approval of: `r0001`, `r0005`.
> A critical decision cannot rest on unreviewed logic.

**Say:** "This isn't the system declining to answer. It *cannot* answer. The
engine is handed only approved rules, so an unapproved rule can't contribute
even by accident — and `verify_invariant()` re-audits that after the fact."

*If asked why routine queries aren't blocked:* criticality attaches to the
query, approval attaches to the rule. Small loans run on provisional logic; the
exposure that buys is enumerable, which beat 7 shows.

## Beat 4 — One human, one pass (75s)

`/review` → point at the two kinds of evidence side by side:

- **document rules** show `Cites: lending_policy.md, s4.1` — *does this encode
  the clause?*
- **mined rules** show support, precision, counterexamples — *is this
  correlation also a reason?*

**Say:** "Different provenance, different question, so different evidence. If
both rendered identically, a reviewer would answer the easier one twice."

Approve all 8. Note the wording on the button: **approve permanently**.

## Beat 5 — The same query now goes through (45s)

`/query` → **`app_0001`** → `P = 0.67`, full derivation, every fact traced to a
source field and every rule to its approver and clause.

Then **`app_0009`** → `P = [0.64, 0.81]`.

**Say:** "That interval isn't model uncertainty. Facts are missing, and it spans
every way they could resolve. Below it, the system names the single fact that
would settle it — and stays silent about the unknowns that would change
nothing."

## Beat 6 — The payoff (45s)

Run 5–10 more cases from `/query`, then open **`/`**.

> reviews performed **8** · rule applications **65** · decisions per review **8.1**

**Say:** "This ratio is the argument. Reviewing decisions is linear work.
Reviewing rules amortises — and it keeps climbing as the base matures. In the
full run it reaches 36.8."

## Beat 7 — The system finds a hole in the human policy (60s)

`/query` → **`app_0003`** → *No applicable logic*, even though 8 rules are
approved.

Open `lending_policy.md`:
- **s7.1** covers elevated risk **without** security
- **s7.2** covers security **without** elevated risk
- **Neither covers both at once** — and this applicant has both.

**Say:** "I wrote that policy and didn't notice the gap. The system did, on
**11% of cases**, by refusing to guess. A scoring model would have produced a
confident number for every one of them."

---

## Closing line

> "The contribution isn't the rule engine — Datalog is from 1977. It's moving
> the oversight. A reviewer here reads one rule and governs every case it
> matches, and a critical decision is *structurally* incapable of resting on
> logic nobody approved."

---

## Backup: if the web app fails

```bash
python demo.py          # governance loop, terminal, ~3s
python demo_agent.py    # full agent loop on 1000 real cases, ~20s
```

`demo.py` covers beats 1–6 and prints the amortisation report. `demo_agent.py`
covers the agent loop and ends with a rendered proof.

---

# B. The control room — recruitment

```bash
rm -f data/logicdb.sqlite
./run.sh recruitment          # http://127.0.0.1:8000/console
```

Type a name into **Accountable reviewer** and tick **authorise** first — screening
refuses without both, which is itself worth pointing at.

| beat | action | what they see |
|---|---|---|
| 1 | Load `/console` | 8,000 cases, **0 rules**. Nothing can be decided. |
| 2 | **Extract rules from policy** | 8 rules appear, each citing its clause, all amber — *blocking*. |
| 3 | **Screen 250** | **338 blocked** of 500, 0 decided. Click a blocked row: it names the rule it waits on. |
| 4 | Click a rule → **Approve** | Card turns green. Blocked count drops to **0**. |
| 5 | **Run to end** | Metrics climb live; decisions-per-review passes 100. |
| 6 | Click any case | The proof, every fact with its source, and what would change the answer. |
| 7 | Scroll to the record | Every proposal and approval, with its actor. Exportable. |

**The line for beat 3:** "It isn't declining to answer. It *cannot* answer —
the engine is handed only approved rules."

**The line for beat 5:** "Those 8 reviews are the entire human cost of the logic.
Our earlier prototype needed 5,858 case reviews for the same campaign."

---

## Live numbers, for reference

| Figure | Value | Where |
|---|---|---|
| Rules extracted from policy | 8, all cited | beat 2 |
| Decisions per review (demo) | 8.1 | beat 6 |
| Decisions per review (full run) | 36.8 | `paper/figures/` |
| Policy gap rate | 11% of 300 cases | beat 7 |
| Rule recovery, synthetic pack | 3/3, ranked top-3 | not shown live |
| Governance invariant violations | 0 | `/` dashboard |
| Tests | 285 passing | `python -m pytest` |
| Control room, blocked before approval | 338 of 500 | demo B beat 3 |
| Human case reviews, recruitment | 2,558 vs 5,858 | `-56%` against our own prototype |
| Wrong rejections, recruitment | 26.4% | measurable only because ground truth exists |
