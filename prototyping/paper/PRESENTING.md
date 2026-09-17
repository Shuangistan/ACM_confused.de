# Presenting this

29 slides, 17 numbered stops. Aim for **12 minutes** of talk and leave the rest
for the demo and questions. The demo is the argument; the slides set it up.

---

## Before you stand up

```bash
cd prototyping
./stop.sh && rm -f data/logic.sqlite && ./start.sh
```

A clean database is the point: the demo's first case must park, and its second
must not. If the database already holds a tenancy rule, step 7 loses its force.

Have **three windows** ready: the slides, the console at `localhost:8000`, and
the Logic Database in a second tab. Open the Logic Database tab *before* you
start talking, so you are not fumbling for it in front of people.

Check the key works: ask "Hi" and make sure you get a reply. That costs about
600 tokens and proves the whole chain.

---

## The line to hold

Everything here follows from one sentence, and it is worth saying almost
verbatim early:

> **Human-centred does not mean a person in every loop. At volume that produces
> ceremony. It means choosing the unit of human judgement so that judgement
> scales — and the unit we chose is the rule, not the case.**

If you only land one idea, land that.

---

## Walking the slides

**1–4 · The problem** (2 min)
The title names both contributions in order — the executable matrix, then the
logic database — and the subtitle names the formalism. Say that before you
advance; it is the shape of the whole talk.

Open with the thing everybody in the room already believes — a person reviews
each consequential action — and then take it apart. Two failures, not one:
unaffordable at volume, *and* useless without the reasoning. Most audiences have
felt the first and not articulated the second. Slide 4 is where you say the line
above.

**5–11 · Contribution one: the matrix and the loop** (5 min)
This is the half they should remember, so give it the time.

Say that the matrix is **our group's artefact**, not ours alone, and that most
such artefacts live in a document. Ours decides. Nine lines, in a module that
imports nothing — *a module that cannot reach a network cannot be talked into
anything*.

The parity slide is a credibility slide: we run the same rule twice, in two
languages, and we test that they agree over all 1,000 inputs. Mention that we
mutated a threshold to check the test bites. People who build systems will
notice.

Then the gate. Spend time on **"has a person already ruled on this logic?"** and
on why approved logic proceeds even at `human_only`. Expect pushback here — see
the questions below. Say plainly that it is the most contestable claim in the
design and that you made it deliberately.

Slide 10 is the human-centred slide: the expert reviews a **prepared, editable**
answer. *An expert who may only accept or refuse is not reviewing, they are
signing.*

**12–15 · Contribution two: the substrate** (3 min)
Frame it as subordinate and necessary: *capturing a person's judgement as a rule
is only honest if the rule behaves.* Determinism, explainability, transparency,
formal footing — then the umbrella slide, which is the one people remember.

On the umbrella: do not rush it. Recorded-absent fires, **unmentioned does not**.
Under a closed world, silence becomes evidence against a person. Say that we
shipped that defect twice — it buys more credibility than any claim of rigour.

**16–18 · Evidence** (2 min)
Lead with what broke. An audience that has heard three polished talks will lean
forward at *"a gate that overfitted its own prompt"* and at Haiku 0/4 versus
Sonnet 4/4.

Then coverage: **18–30 per cent**, and say you are reporting the range rather
than the better number. Then why it is not higher — the `reorder_toner` /
`reorder_paper` slide is the sharpest result in the deck. *Generalisation can
widen a rule; it cannot repair a vocabulary.*

**19 · What we have not measured** (1 min)
Do not hurry this and do not apologise through it. "No expert has used this" is
the honest headline, and stating it before anyone asks is worth more than
defending it afterwards.

---

## The demo (5–6 min)

Nine steps, verified from an empty database. Type them; do not paste from a
script, because watching you type makes it obviously live.

| | What you type | What to say |
|---|---|---|
| 1 | `Hi` | "Nothing is classified. A greeting is not a decision." |
| 2 | `What are the rules on evicting a tenant?` | "A question of fact. Still nothing." |
| 3 | `Should we reorder toner? Stock is low but we have not run out.` | "Low harm, reversible. Decided, nobody waits." |
| 4 | `We need to decide whether to evict a tenant three months in arrears who disputes the debt and was given no formal notice.` | "Human only. It parks. **Nothing is said to the person.**" |
| 5 | Open **Expert review** | "Here is the prepared answer — and I can change it." |
| 6 | Replace with: *Do not evict. Serve formal notice, allow 14 days, and resolve the disputed amount first.* → Approve | "The answer is attributed to me and says I revised it." |
| 7 | The clause appears → Approve it | "My decision, as logic. One more click, with the clause in front of me." |
| 8 | `Another tenant is in arrears and disputes the amount; no formal notice was sent. Do we evict?` | "**Decided. Nobody waits.** That is the whole argument." |
| 9 | Switch to the Logic Database tab | Coverage, the rule with my name on it, the decision it made, and the merge candidate. |

**Step 4 and step 8 are the demo.** Everything else is scaffolding. If you are
running short, cut 2 and 3, never 4 or 8.

If something fails live: the honest recovery is to say what you expected and
what happened, and move on. You have a slide deck full of measured failures —
one more will not undermine you.

---

## Questions you should expect

**"Isn't approving a rule just a blanket authorisation?"**
No, and the distinction matters. A blanket authorisation approves unseen future
decisions. Approving a clause approves a stated condition — the body says
exactly which cases it claims — and the system still audits a proportion of what
it decides, retires it on request, and enumerates what it already did.

**"What if the expert approves a bad rule?"**
Then it decides badly until someone notices, and the mechanisms for noticing are
the audit sample, the per-rule tally, and retirement with impact analysis. We do
not claim to prevent it. We claim to make it visible and reversible.

**"Why does approved logic proceed at `human_only`?"**
Because the alternative is asking the same person the same question forever,
which is how oversight becomes ceremony. Say it is the most contestable choice
in the design, that a rule an expert approved *is* their decision, and that a
group could reasonably decide otherwise — the gate is one line.

**"Only 18 per cent?"**
Yes, and we know precisely why, which is more useful than a better number we
could not explain. Then the vocabulary slide.

**"Why not just use an LLM for all of it?"**
Because a model cannot promise that the same case gets the same answer tomorrow,
and cannot show you the derivation that produced it. Those two properties are
what an approval is worth.

**"Has anyone actually used it?"**
No. Say so immediately. Then say what the study would be: five people, ten
proposed clauses, measure whether they can tell a good rule from a bad one and
how long it takes. It is half an hour of work and we have not done it.

**"Is the matrix yours?"**
No — it is the group's. Credit it. And mention that we found two defects in it
which we deliberately did not fix: reversibility does nothing at harm 4–7, and
confidence overrides harm entirely. Reporting a colleague's artefact honestly is
better received than quietly amending it.

---

## Things not to claim

- Do not say the system is deterministic full stop. **The decision path is
  deterministic once the logic is approved**; extraction and estimation are a
  model, and vary.
- Do not call the `human_only` conduct a guarantee. It is a prompt, it held in
  every trial, and three trials is an anecdote.
- Do not present coverage as a trend. Two runs.
- Do not imply the vocabulary transfer is validated. A tenancy rule firing on a
  library fine was sound in the case we saw and is unexamined in general.
