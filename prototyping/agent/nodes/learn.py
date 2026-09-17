"""What the database takes away from a decision.

`propose` drafts the logic of the case so a person can review it once and every
matching case afterwards is decided without a model. `record_case` writes the
decision down — not a history table, but the evidence every future
generalization is judged against."""

from __future__ import annotations

from typing import Any

from logic.engine import Engine
from logic.parse import ParseError, parse_rule
from logic.store import RuleRow
from logic.syntax import UnsafeRule

from .. import db, llm
from ..state import ConsoleState
from .common import emit
from .logic import _case_id, _fact_store

RULE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "rules": {"type": "array", "items": {"type": "string"}},
        "why": {"type": "string"},
    },
    "required": ["rules", "why"],
    "additionalProperties": False,
}

PROPOSE_SYSTEM = """\
Draft the logic of this case, so a person reviews it once and matching cases
afterwards are decided without a model.

    HEAD(c) <- CONDITION(c), not OTHER_CONDITION(c).      # schematic

Enforced after you reply; a clause breaking one is discarded unseen:
* lowercase predicates, `c` for this case, full stop at the end
* at least one positive condition
* use the case's own fact terms; coin no synonym for one already in use
* **it must fire on the facts given** — checked, and discarded if not

Write the general rule, not a transcription: drop the particulars that did not
matter, keep the conditions that did. A clause firing only on the case that
produced it adds nothing, because the next case will differ somewhere.

Two or three clauses at most. `why` is one sentence for the reviewer.

{already}\
"""


def propose(state: ConsoleState) -> dict[str, Any]:
    """Draft rules for a case the base could not answer, and queue them.

    Nothing proposed here takes effect. It enters the database as `proposed`
    and waits for a person; the gate is the point of the whole arrangement.
    """
    facts = state.get("facts") or []
    if not facts:
        return {}

    # What the base can already do, so the model adds to it instead of
    # restating it. Duplicates are harmless — they collide on the canonical key
    # and change nothing — but they cost tokens and clutter the queue.
    existing = [r.text for r in db.store().active_rules()][:40]
    concluded = state.get("conclusions") or []
    if concluded:
        already = (
            "The approved rules already decide this case, concluding:\n"
            + "\n".join(f"    {c}" for c in concluded)
            + "\n\nSo propose only logic that is **missing** — a conclusion "
              "nothing yet reaches, or a condition the existing clauses do not "
              "account for. If the case is fully captured already, return an "
              "empty list. A restatement of a rule already in force is worse "
              "than nothing: it spends a reviewer's attention on a decision "
              "they have already made."
        )
    else:
        already = ("Nothing in the approved base fires on this case, so the "
                   "logic that would decide it is missing entirely.")
    if existing:
        already += ("\n\nAlready approved and in force:\n"
                    + "\n".join(f"    {t}" for t in existing))

    prompt = [{
        "role": "user",
        "content": ("Facts extracted from this case:\n"
                    + "\n".join(f"  {f}" for f in facts)
                    + f"\n\nThe decision: {state.get('problem', '')}"),
    }]
    try:
        reply = llm.ask(llm.at_least(state.get("model", llm.DEFAULT_MODEL),
                                     llm.EXTRACTION_FLOOR),
                        PROPOSE_SYSTEM.format(already=already),
                        prompt, schema=RULE_SCHEMA, max_tokens=500)
        got = reply.data
    except Exception:  # noqa: BLE001
        return {}
    used = {"input_tokens": reply.input_tokens,
            "output_tokens": reply.output_tokens}

    store = db.store()
    case_facts, _ = _fact_store(facts)
    queued, rejected = [], []
    for text in got.get("rules", [])[:4]:
        text = str(text).strip()
        if not text:
            continue
        try:
            rule = parse_rule(text)
        except (ParseError, UnsafeRule, ValueError) as exc:
            rejected.append((text, str(exc)[:90]))
            continue
        if not rule.body:
            rejected.append((text, "a bare fact, not a rule"))
            continue
        # Does it actually fire on the case that prompted it? A rule proposed
        # for this case that cannot decide this case is worthless, and asking
        # the model nicely is not enough — a run produced a clause requiring
        # the exact opposite of the facts in front of it. Checked, not trusted.
        try:
            probe = Engine().evaluate([("probe", rule)], case_facts)
            fires = bool(probe.derived)
        except Exception:  # noqa: BLE001
            fires = False
        if not fires:
            rejected.append((text, "does not fire on the case that prompted it"))
            continue
        store.upsert_rule(RuleRow(
            canonical_key=rule.canonical_key(),
            text=str(rule),
            head=rule.head.predicate,
            origin="agent",
            proposed_by=f"llm-agent-{llm.MODELS.get(state.get('model','haiku'))}",
            note=str(got.get("why", ""))[:240],
        ))
        queued.append(rule.canonical_key())

    # Handed on in state rather than written here: `propose` runs before the
    # case row exists — it is created by `park` or `record` further down the
    # graph — so an UPDATE at this point silently matches nothing.
    used["drafts"] = {
        "kept": [store.get_rule(k).text for k in queued if store.get_rule(k)],
        "rejected": [{"text": t, "why": w} for t, w in rejected],
    }
    if queued or rejected:
        emit({"kind": "proposed", "queued": len(queued),
               "rejected": len(rejected), "why": got.get("why", "")})
    return used


def _last_assistant(state: ConsoleState) -> str:
    for message in reversed(state.get("messages") or []):
        if message.get("role") == "assistant":
            return message.get("content", "")
    return ""


def record_case(state: ConsoleState, status: str) -> str:
    """Write one decision down. Returns its id.

    Not a history table. This is the evidence every future generalization is
    judged against, and — when `status` is `waiting` — the durable home of an
    inquiry parked for an expert.
    """
    facts = state.get("facts") or []
    question = ""
    for message in reversed(state.get("messages") or []):
        if message.get("role") == "user":
            question = message.get("content", "")[:300]
            break

    case_id = state.get("case_id") or _case_id(state)
    db.store().record_case(
        case_id=case_id,
        question=question,
        # The atom is stored bare and its polarity in the truth column. Storing
        # "not x(c)" *and* truth=false meant anything rebuilding the facts
        # prepended a second negation — `not not x(c)` — so every replay of a
        # negative fact was quietly wrong.
        facts=[[f[4:].strip() if f.lower().startswith("not ") else f,
                "false" if f.lower().startswith("not ") else "true",
                "extracted"] for f in facts],
        conclusions=list(state.get("conclusions") or []),
        rules_used=list(state.get("rules_used") or []),
        decided_by="rules" if state.get("covered") else "uncovered",
        tier=state.get("tier") or "",
        covered=bool(state.get("covered")),
        session=state.get("session") or "",
        status=status,
        sampled=bool(state.get("sampled")),
        drafts=state.get("drafts") or {},
        # A parked case carries the draft the expert will review; a settled one
        # carries what was actually said.
        reply=(state.get("draft", "") if status == "waiting"
               else _last_assistant(state)),
    )
    return case_id


def record(state: ConsoleState) -> dict[str, Any]:
    """The ordinary path: a decision that proceeded."""
    if not (state.get("facts") or []):
        return {}
    record_case(state, status="settled")
    return {}


FROM_DECISION_SYSTEM = """\
An expert has decided a case the rule base could not. Write the logic of
*their* decision, so the next case like it is decided without them.

    HEAD(c) <- CONDITION(c), not OTHER_CONDITION(c).      # schematic

Enforced after you reply; a clause breaking one is discarded unseen:
* lowercase predicates, `c` for this case, full stop at the end
* at least one positive condition
* conditions only from the facts listed — invent none, and name no conclusion
* **it must fire on those facts** — checked

Capture what the expert decided, not what was drafted for them; where the two
differ, theirs is the decision. Generalise: keep the conditions their reasoning
turned on.

One or two clauses. `why` is one sentence for the next reviewer.\
"""


def _as_literal(row: list) -> str:
    """One stored fact as rule syntax, whichever way the row was written.

    Rows written before the polarity fix keep `not ` inside the atom text;
    prepending another would give `not not x(c)`.
    """
    atom, truth = str(row[0]).strip(), (row[1] if len(row) > 1 else "true")
    bare = atom[4:].strip() if atom.lower().startswith("not ") else atom
    negated = truth == "false" or atom.lower().startswith("not ")
    return f"not {bare}" if negated else bare


def rule_from_decision(case: dict, verdict: str, actor: str, answer: str,
                       model: str = llm.DEFAULT_MODEL) -> dict[str, Any]:
    """Turn an expert's ruling into candidate logic.

    Until now an expert's decision taught the database nothing: they released
    an inquiry, the person got their answer, and the next identical case came
    straight back to them. The whole argument is that a human decision made
    once should not have to be made again, which is only true if the decision
    is written down as logic somebody can approve.

    It is queued as `proposed`, never approved. Deciding a case and licensing
    the logic behind it remain separate acts — this just stops the second one
    requiring an expert to draft a clause by hand.
    """
    facts = [_as_literal(f) for f in (case.get("facts") or [])]
    if not facts or verdict != "approved":
        return {"queued": 0, "rejected": 0, "rules": []}

    prompt = [{"role": "user", "content":
        "Facts of the case:\n" + "\n".join(f"  {f}" for f in facts)
        + f"\n\nThe question: {case.get('question','')}"
        + f"\n\nWhat {actor} decided:\n{answer.strip()[:1200]}"}]
    try:
        # Same floor as extraction, for the same measured reason: Haiku drafted
        # clauses that did not fire on the very facts they were drawn from.
        got = llm.ask(llm.at_least(model, llm.EXTRACTION_FLOOR),
                      FROM_DECISION_SYSTEM, prompt,
                      schema=RULE_SCHEMA, max_tokens=500).data
    except Exception:  # noqa: BLE001
        return {"queued": 0, "rejected": 0, "rules": []}

    store = db.store()
    case_facts, _ = _fact_store(facts)
    queued = rejected = 0
    made: list[dict[str, Any]] = []
    for text in got.get("rules", [])[:3]:
        try:
            rule = parse_rule(str(text).strip())
        except (ParseError, UnsafeRule, ValueError):
            rejected += 1
            continue
        if not rule.body:
            rejected += 1
            continue
        try:
            fires = bool(Engine().evaluate([("probe", rule)], case_facts).derived)
        except Exception:  # noqa: BLE001
            fires = False
        if not fires:
            rejected += 1
            continue
        store.upsert_rule(RuleRow(
            canonical_key=rule.canonical_key(), text=str(rule),
            head=rule.head.predicate, origin="expert-decision",
            proposed_by=f"decision by {actor}",
            note=str(got.get("why", ""))[:240]))
        made.append({"key": rule.canonical_key(), "text": str(rule),
                     "why": str(got.get("why", ""))[:240]})
        queued += 1
    return {"queued": queued, "rejected": rejected, "rules": made}


__all__ = ["FROM_DECISION_SYSTEM", "PROPOSE_SYSTEM", "RULE_SCHEMA", "propose",
           "record", "record_case", "rule_from_decision"]