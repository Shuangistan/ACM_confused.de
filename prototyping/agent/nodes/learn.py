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
Draft the logic of this case, so a person can review it once and every matching \
case afterwards is decided without a model.

Syntax, function-free Horn clauses:

    conclusion(c) <- condition(c), other(c), not absent_thing(c).

Rules, all enforced after you reply — a clause that breaks one is discarded \
before any person sees it:

* Lowercase predicates, `c` for this case, end each clause with a full stop.
* Every clause needs at least one positive condition; a body of only negations \
is rejected.
* Use the fact terms from this case as they are written. Do not coin a synonym \
for a term already in use.

**The rule must fire on the facts below.** A clause whose body cannot be
satisfied by this case is not logic for this case, whatever else it may be; it
is checked after you reply and discarded if it does not fire.

Write the **general** rule, not a transcription of this case. A clause that \
only ever fires on the case that produced it adds nothing: the next case will \
differ in some detail, nothing will match, and a model will be asked again. \
Drop the particulars that did not matter and keep the conditions that did.

Two or three clauses at most. Intermediate conclusions are welcome when they \
name something worth naming.

`why` is one sentence a reviewer reads before approving.

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
        reply = llm.ask(state.get("model", llm.DEFAULT_MODEL),
                        PROPOSE_SYSTEM.format(already=already),
                        prompt, schema=RULE_SCHEMA, max_tokens=800)
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
        facts=[[f, "false" if f.lower().startswith("not ") else "true",
                "extracted"] for f in facts],
        conclusions=list(state.get("conclusions") or []),
        rules_used=list(state.get("rules_used") or []),
        decided_by="rules" if state.get("covered") else "uncovered",
        tier=state.get("tier") or "",
        covered=bool(state.get("covered")),
        session=state.get("session") or "",
        status=status,
        sampled=bool(state.get("sampled")),
        reply=_last_assistant(state) if status == "settled" else "",
    )
    return case_id


def record(state: ConsoleState) -> dict[str, Any]:
    """The ordinary path: a decision that proceeded."""
    if not (state.get("facts") or []):
        return {}
    record_case(state, status="settled")
    return {}


__all__ = ["PROPOSE_SYSTEM", "RULE_SCHEMA", "propose", "record", "record_case"]
