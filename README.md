# A governed logic database for agentic decision support

Take the decision logic out of the model and put it in an inspectable database;
then put human oversight on the logic rather than on individual decisions.

Built for the ACM Summer School project *Human-Centred, Agentic,
Decision-Support AI*. The original brief asked what meaningful human control
looks like when agents participate in consequential decisions. This is an answer
you can run.

## Why rule-level oversight

Per-decision oversight does not scale, and it degrades into rubber-stamping
exactly where it matters most — when the reviewer cannot independently verify
the output. Rule-level oversight amortises instead: a rule reviewed once governs
every future case whose facts match it.

In the credit run, **16 reviews governed 589 decisions** — 36.8 decisions per
act of review, with the ratio still climbing as the rule base matures.

(An earlier draft reported 66.7 here. That figure divided *rule applications* by
reviews, and several rules fire per decision, so it overstated the claim by that
factor. Both ratios are now reported separately.)

### 1. The logic database

- **Determinism** — a terminating fixpoint with no function symbols. Each round
  matches a snapshot, so a rule's result cannot depend on where in the round it
  ran, and insertion order cannot change the answer. Stratification is re-checked
  on *every* insertion, because rules arrive from an agent over time.
- **Explainability** — forward chaining, so the whole derivation exists. The
  explanation renders it rather than narrating alongside it.
- **Evidence that stays honest** — three-valued facts, and uncertainty that
  survives negation. Missing evidence becomes an interval and a question, never
  a silent false.
- **Exact bounds cheaply** — polarity analysis replaces 2^k evaluations with two
  plus a small enumeration, checked against exhaustive enumeration on random
  programs including negation.

117 of the 289 tests exist to establish these — 40% of the suite, because
approval means nothing without them.

### 2. Governance over it

- **The agent proposes; only the engine infers.** An LLM drafts candidate facts
  and rules and never produces a decision.
- **Oversight that binds** — a critical decision is structurally incapable of
  resting on a rule no human approved. Enforced in the engine, and audited after
  the fact by `verify_invariant()`.

## Running it

```bash
conda activate iese          # note: lowercase
./run.sh                     # everything, at http://127.0.0.1:8000
python -m pytest             # 285 tests
```

`./run.sh` serves four things from one process:

| | |
|---|---|
| **`/`** | the showcase — the argument, the results, links to everything |
| **`/console`** | the control room — screen a campaign, review rules, read proofs |
| **`/dashboard`** | the classic view — one page per action |
| **`/paper.pdf`**, **`/slides.pdf`** | the built artifacts |

Pass a pack to change domain with no code change:
`./run.sh consumer_credit` · `./run.sh benefits_eligibility`

Terminal demos, if you prefer them:

```bash
python demo.py               # governance loop: block, approve, reuse, reject
python demo_agent.py         # agent loop: empty database learns its own logic
```

No installs required — the `iese` env already has everything, which is also why
the logic engine is built rather than imported.

## What a decision looks like

```
Answer: P = [0.00, 0.70]
  The interval is 0.70 wide because facts are missing. It is not a confidence
  interval over a model — it spans every way the unknowns could resolve.

Derivation:
[rule] app_gap should be declined  p=0.70
  [rule] by rule r0002: decline(A) <- high_risk(A), not mitigated(A).  p=0.88
    (origin document; approved by policy-team; cites s7.1; right 70/191 in use)
    [rule] app_gap is high risk  p=0.80
      [fact] app_gap is overdrawn      (source: application form, field 3)
      [fact] app_gap has a thin file   (source: credit bureau report)
    [UNKNOWN] not app_gap has mitigating circumstances
      (ASSUMED, not established — this is unknown, and this end of the
       interval is what follows if it does not hold)

Missing information, ranked by whether it would change anything:
  app_gap has a guarantor   [DECISIVE]
      if true -> 0.00, if false -> 0.70
```

Three things worth noticing. The system **refuses to guess** — a missing fact
produces an interval, not a number. It **names the one question** that would
settle it, and stays silent about unknowns that would change nothing. And it
distinguishes *assumed* from *established*, so an interval endpoint can't be
misread as a finding.

## The governance model

| State | Meaning |
|---|---|
| `PROPOSED` | Agent created it; usable nowhere |
| `PENDING_ONLINE` | A critical query is **blocked** on it |
| `PROVISIONAL` | Usable for routine queries now, queued for offline review |
| `APPROVED` | Permanent — never re-reviewed |
| `REJECTED` | Never usable; triggers impact analysis over past use |
| `RETIRED` | Superseded; past decisions keep the version they used |

**Criticality attaches to the query; approval attaches to the rule.** Routine
work is never stalled waiting for a reviewer, and the exposure that buys is
enumerable: rejecting a provisional rule names every decision that used it,
across sessions.

**Approve once, never again.** Rules are keyed by canonical hash — variables
renumbered, body sorted — so `decline(A) <- overdrawn(A), thin_file(A)` and
`decline(Z) <- thin_file(Z), overdrawn(Z)` are one rule with one approval.
Without canonicalisation, "approve once" quietly degrades into approving
variants forever.

The one exception to permanence: a rule whose observed accuracy falls well below
its claimed strength is flagged for re-review. The reviewer is being shown new
evidence, not asked to repeat themselves.

## Provenance decides how hard to look

A reviewer is shown different evidence depending on where a rule came from,
because the two ask different questions.

| Origin | The reviewer's question | What they're shown |
|---|---|---|
| `document` | Does this faithfully encode the clause? | The citation — open it and compare |
| `mined` | Is this correlation also a *reason*? | Support, precision, examples, counterexamples |
| `human` | — | Author and notes |

On the synthetic pack, where the generating rules are known and hidden, the
miner **recovered all three exactly**, ranked as its top three proposals, with
zero proposals over the two attributes that bear on nothing by construction.
That evaluation — *is it the right rule*, not *does it predict well* — is the
one real data cannot provide.

## Layout

```
logicdb/
  syntax.py      AST + canonical hashing (the identity approve-once rests on)
  parser.py      surface syntax, with error messages an agent can act on
  facts.py       three-valued facts: TRUE / FALSE / UNKNOWN, each with a source
  program.py     rule lifecycle + stratification checking
  engine.py      semi-naive forward chaining, proof DAG
  probability.py noisy-OR, bounds under unknowns, value-of-information ranking
  explain.py     proof → readable text, counterfactuals, provenance report
  governance.py  criticality routing, approve-once, impact analysis
  store.py       SQLite; identity is the primary key, history is append-only
  app.py         FastAPI: classic pages — dashboard, review queue, rule browser
  console.py     JSON API + the showcase and control room; serves the PDFs
  templates/     showcase.html · console.html · the classic pages
  static/media/  figures, generated from live runs by paper/figures/
  agents/        one protocol, two backends (scripted, Claude)
packs/
  consumer_credit/      UCI German Credit + a written lending policy
  benefits_eligibility/ synthetic, generated from known rules that are withheld
  recruitment/          8,000-applicant campaign, with a withheld suitability
                        label so a rejection can be measurably wrong
```

Nothing in `logicdb/` knows what a guarantor is. Three packs share the engine
and no code.

Three front ends share one governed engine, so the showcase links into a
running system rather than a screenshot.

## Design notes worth knowing

**Unknown survives negation.** A derived atom blocked by a missing fact is
itself unknown, so `not mitigated(app)` cannot silently succeed because nobody
recorded a guarantor. This was a real bug, caught by a test written for it.

**Outcomes are never normalised.** `approve` and `decline` are computed
independently. If both are low, the rule base has nothing to say — and
normalising two near-zero numbers would disguise that as a confident 50/50.

**The independence assumption is stated.** Noisy-OR treats supporting rules as
independent evidence, which is frequently wrong. It appears in every
explanation rather than hiding inside the arithmetic.

**Bounds are verified against the definition.** The fast path skips 2^k
enumeration by reasoning about polarity; the tests check it against exhaustive
enumeration on random programs including negation. A shortcut nobody checks is
a guess.

**Structural constraints beat reviewed ones.** The lending policy forbids
decisions resting on sex. That clause is deliberately *not* a rule — a rule can
state a condition but not the absence of a hidden correlation. Instead `sex` is
absent from the vocabulary, so no rule can be written or mined over it. Review
cannot reliably spot a proxy and should not be asked to.

## The head-to-head

The `recruitment` pack reproduces the scenario of the group's earlier
client-side prototype (`decision_partner_mvp.html`, `mvp+`) so the two can be
compared on identical inputs rather than rhetorically.

| | agent fleet | rule base |
|---|---|---|
| one-off rule reviews | — | 8 |
| human case reviews | 5,858 | **2,558** |
| decided without a person | 0 | 5,442 |
| screening logic inspectable | no | yes |
| wrong rejections measurable | no | **yes** |

The last row is the one that matters. In the agent fleet a rejection fires only
on an explicit unmet requirement, so it is correct by construction and an
auditor has nothing to find. The recruitment pack carries a withheld
suitability label: against it, **26.4% of automatic rejections are of genuinely
suitable applicants.**

## Known limitations

- **Credit assignment is crude.** `record_outcome` marks every rule in a
  derivation right or wrong together, so a determination rule is blamed for
  failures caused by its premises. Proper attribution needs something like a
  Shapley value over the derivation.
- **Rejection flags but does not reverse.** The system names the decisions a
  withdrawn rule touched and stops there. Whether that invalidates them is a
  judgement it should not make alone.
- **One writer.** Both front ends share a single governor and SQLite connection
  — fine for a seminar room, wrong for anything larger.
- **The head-to-head is not like-for-like.** The baseline is our own earlier
  prototype, and the two systems partition work differently.
- **The demo's accuracy figure is not a benchmark.** It scores an invented
  lending policy against real data. The system detecting that its own policy is
  wrong is the machinery working, not a capability claim.
