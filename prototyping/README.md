# confused.de — prototyping

A decision console with two halves that stay apart on purpose.

**The ladder** decides how much human oversight a decision needs — harm,
reversibility, rights, evidence, in, one of four control tiers out. It says
nothing about what the answer is.

**The logic database** decides the case itself. Facts extracted from the
conversation, approved rules run over them, and the conclusion comes with the
derivation that produced it. Same facts and same rules, same answer, every time.

Between them sits a gate. It asks one question — *does an approved rule cover
this case?* — and if not, and the consequences warrant it, the inquiry parks for
a person. A rule an expert has approved **is** their decision; applying it
executes a call they already made, so matching cases afterwards do not come back
to them. That is the whole argument.

## Running it

```bash
pip install -r requirements.txt
./start.sh          # serves on :8000 and opens Chrome
./stop.sh
```

Needs `ANTHROPIC_API_KEY`, or `API_KEY=` in `../.env`. Model defaults to Haiku.

For the tests, `pip install -r requirements-dev.txt` and `python -m pytest`.
`node` is needed too — one test extracts the classification from the shipped
page and runs it under Node to check it against the Python. Without node that
test skips rather than fails, so a green run on a machine without it has not
checked the thing it exists for.

```
/            the console
/database    the rule base, the cases, and coverage
```

## Layout

```
agent/            the graph and the model-facing parts
  graph.py          intake → suggest → classify → extract → derive → gate → …
  nodes/            one module per stage, grouped as the graph is
    intake.py         is there a decision on the table?
    assess.py         four properties estimated, a tier derived
    logic.py          the case as facts; what the approved rules make of them
    gate.py           proceed, or park for a person
    reply.py          the answer, under the conduct its tier imposes
    learn.py          what the database takes away from the decision
  ladder.py         the classification. Imports nothing, on purpose
  scales.py         what the four inputs mean — shared with the page's help text
  llm.py            the client, and the request shape older models need
logic/            the deterministic core
  syntax.py         rules, canonical identity, safety
  parse.py          surface syntax both ways
  facts.py          three-valued facts; UNKNOWN is a value, not an absence
  engine.py         stratified fixpoint, proofs, blocked heads
  store.py          rules, cases, events
frontend/         two self-contained pages
tests/
```

## Two things worth knowing before changing anything

**`ladder.py` imports nothing.** It decides how much human oversight applies,
and a module that cannot reach a network cannot be talked into anything. Keep it
that way.

**The ladder exists twice** — here and in JavaScript inside `index.html`, so the
sliders respond without a round trip. `tests/test_ladder_parity.py` extracts the
JavaScript from the shipped page, runs it under `node`, and compares all 1,000
input combinations. If you change one, that test tells you.

## Not built yet

Nothing generalizes. Rules arrive one per case shape and never merge, so
coverage creeps rather than climbs — `reorder_toner` and `reorder_paper` sit
side by side where `reorder(C, X)` would do. Anti-unification over approved
rules, replayed against the stored cases to show a reviewer where it would have
disagreed, is the missing piece and the one that decides whether any of this
works.
