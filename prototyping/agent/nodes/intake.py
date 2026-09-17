"""Taking in a decision. Nothing is classified here.

A greeting is not a decision, and a system that scores one has confused having
a conversation with having a problem."""

from __future__ import annotations

from typing import Any

from .. import llm
from ..state import ConsoleState
from .common import emit

#: The model is asked the *narrow* question — is this purely conversational? —
#: rather than the broad one. Four attempts at "is this a decision?" produced
#: 0/6 on one ordinary case and 6/6 on a near-identical one, because the model
#: matched the examples in the prompt instead of applying the criterion. Asking
#: it to recognise a greeting is a judgement it makes reliably; asking it to
#: recognise a decision is not, and the gate fails open so that being unsure
#: costs a turn of assessment rather than a decision that never arrives.
INTAKE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "reply": {"type": "string"},
        "conversational": {"type": "boolean"},
        "problem": {"type": "string"},
    },
    "required": ["reply", "conversational", "problem"],
    "additionalProperties": False,
}

INTAKE_SYSTEM = """\
Someone is bringing you a decision. You are not deciding or assessing anything.

Each turn: reply, and say whether their message was **purely conversational** \
— a greeting, thanks, small talk, a question about you, or a request for \
general information, with no choice of their own in it.

    "Hi"                                  conversational
    "What can you do?"                    conversational
    "What are the rules on eviction?"     conversational
    "I'm dealing with a difficult tenant" conversational — a topic, no choice

Anything else is not. If they have named something they might or might not do, \
however ordinary and however thinly described, it is not conversational — and \
when you are unsure, it is not. Being wrong that way costs a moment of \
needless assessment; being wrong the other way means their decision is never \
assessed at all.

`problem` is one sentence naming what they are deciding, whenever you can see \
one. Leave it empty only for genuinely conversational turns.

`reply` is what the person sees. Brief and plain. If something important is \
missing you may ask for it, but ask for one thing, and never hold back a \
decision in order to collect more first. Never mention phases, readiness or \
any of this machinery; to them it is a conversation.\
"""


def intake(state: ConsoleState) -> dict[str, Any]:
    """Converse until there is a decision on the table. Classify nothing."""
    messages = [
        {"role": m["role"], "content": m["content"]}
        for m in state.get("messages", [])
        if m.get("role") in ("user", "assistant")
    ]
    if not messages:
        return {}

    try:
        reply_obj = llm.ask(state.get("model", llm.DEFAULT_MODEL), INTAKE_SYSTEM,
                            messages, schema=INTAKE_SCHEMA, max_tokens=700)
        got = reply_obj.data
    except Exception as exc:  # noqa: BLE001
        text = f"[the model could not be reached: {type(exc).__name__}]"
        emit({"kind": "text", "text": text})
        return {"messages": [{"role": "assistant", "content": text}],
                "phase": "intake"}

    ready = not bool(got.get("conversational", True))
    reply = str(got.get("reply", "")).strip()
    out: dict[str, Any] = {
        "phase": "deciding" if ready else "intake",
        "problem": str(got.get("problem", "")).strip() if ready else "",
        # Intake calls cost money like any other. Omitting them made the meter
        # read zero for every greeting, which is precisely the turn a person
        # would not think to suspect.
        "input_tokens": reply_obj.input_tokens,
        "output_tokens": reply_obj.output_tokens,
    }
    # On handover the decision side writes the reply, under the conduct its
    # tier imposes. Emitting an intake reply too would put two answers on
    # screen, the first of them written before the control level was known.
    if not ready:
        emit({"kind": "text", "text": reply})
        out["messages"] = [{"role": "assistant", "content": reply}]
    else:
        emit({"kind": "phase", "phase": "deciding", "problem": out["problem"]})
    return out


def ready_to_decide(state: ConsoleState) -> str:
    return "suggest" if state.get("phase") == "deciding" else "__end__"


__all__ = ["INTAKE_SCHEMA", "INTAKE_SYSTEM", "intake", "ready_to_decide"]
