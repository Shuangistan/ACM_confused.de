"""The answer, written under the conduct its tier imposes.

Runs after classification on purpose: a reply written before the control level
is known cannot be constrained by it, and the constraint is the product."""

from __future__ import annotations

from typing import Any

from .. import ladder, llm
from ..state import ConsoleState
from .common import emit

_CONDUCT = {
    ladder.HUMAN_ONLY:
        "This decision is for a person alone. Lay out what is known and what is "
        "missing. Do not recommend an outcome, and do not imply one.",
    ladder.HUMAN_APPROVAL:
        "A person must approve before anything takes effect. You may set out "
        "options and their consequences; say plainly that the choice is theirs.",
    ladder.AI_WITH_OVERSIGHT:
        "Bounded action is permitted here. You may state what you would do, "
        "while making clear it is logged and reversible.",
    ladder.AI_RECOMMENDS:
        "You may analyse and recommend. You may not act on your own "
        "recommendation.",
}

RESPOND_SYSTEM = """\
You help a person think through a decision. You do not make it.

Two habits matter more than fluency:

Missing information is not a negative finding. If a document, a fact or an \
answer is absent, say it is absent. Never let a gap quietly become a reason \
against someone.

Name what would change things. When the decision turns on something nobody \
has established, say which fact it is and what it would settle.

Be concise and concrete. No preamble, no restating the question back.

Control level for this decision: {tier_name}
{conduct}
{derived}\
"""


def respond(state: ConsoleState) -> dict[str, Any]:
    """Write the reply, under the conduct the tier imposes."""
    tier = state.get("tier", ladder.HUMAN_APPROVAL)
    messages = [
        {"role": m["role"], "content": m["content"]}
        for m in state.get("messages", [])
        if m.get("role") in ("user", "assistant")
    ]
    if not messages:
        return {}

    conclusions = state.get("conclusions") or []
    if conclusions:
        derived = (
            "\nThe approved rule base already decides this case. It concluded:\n"
            + "\n".join(f"  {c}" for c in conclusions)
            + "\n\nBuild your reply on that conclusion and say it follows from "
              "approved logic rather than from your own judgement. Do not "
              "contradict it; if you think it is wrong, say so plainly and say "
              "which rule should be revisited."
        )
    else:
        derived = (
            "\nThe rule base has nothing that bears on this case. That is an "
            "absence of applicable logic, not a negative answer, and you should "
            "not present it as one."
        )
    system = RESPOND_SYSTEM.format(
        tier_name=ladder.TIERS[tier]["name"],
        conduct=_CONDUCT.get(tier, ""),
        derived=derived,
    )
    used = {"input_tokens": 0, "output_tokens": 0}
    collected: list[str] = []

    def fragment(text: str) -> None:
        collected.append(text)
        emit({"kind": "text", "text": text})

    try:
        reply = llm.stream(
            state.get("model", llm.DEFAULT_MODEL), system, messages,
            on_text=fragment, max_tokens=900,
        )
        text = "".join(collected)
        used = {"input_tokens": reply.input_tokens,
                "output_tokens": reply.output_tokens}
    except llm.RefusalError:
        text = ("I can't help with that request. If you rephrase what you are "
                "deciding, I'll pick it up from there.")
        emit({"kind": "text", "text": text})
    except Exception as exc:  # noqa: BLE001
        text = f"[the model could not be reached: {type(exc).__name__}]"
        emit({"kind": "text", "text": text})

    return {"messages": [{"role": "assistant", "content": text}], **used}


__all__ = ["RESPOND_SYSTEM", "respond"]
