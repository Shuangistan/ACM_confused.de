"""How much human oversight this decision needs.

The model estimates four properties of the case; the ladder turns them into a
tier. It may judge how harmful something is, because that is a judgement about
the world. It may not decide what oversight follows, because that is a policy,
and a policy has to be the same on Tuesday as it was on Monday."""

from __future__ import annotations

from typing import Any

from .. import ladder, llm, scales
from ..state import ConsoleState, DEFAULT_INPUTS
from .common import clamp, emit

#: Ranges are stated in the prompt and enforced by `_clamp`, not by the schema:
#: structured outputs reject `minimum`/`maximum` on integers with a 400. Range
#: checking belongs in code regardless -- a model's numbers are an estimate, and
#: an estimate outside the scale should be corrected rather than trusted.
ESTIMATE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "harm": {"type": "integer"},
        "reversibility": {"type": "integer"},
        "rights": {"type": "boolean"},
        "confidence": {"type": "integer"},
        "rationale": {"type": "string"},
    },
    "required": ["harm", "reversibility", "rights", "confidence", "rationale"],
    "additionalProperties": False,
}

SUGGEST_SYSTEM = """You estimate four properties of a decision someone is describing. You do not decide anything, and you do not say what level of human oversight applies — a separate rule does that, from your numbers.

{scales}

rationale      One sentence naming what in the conversation drove your estimate. A person reads this next to your numbers.

Estimate from what has actually been said. Where the conversation is thin, that is what `confidence` is for — do not compensate by inventing detail.""".format(scales=scales.prompt_block())


def suggest(state: ConsoleState) -> dict[str, Any]:
    """Read the conversation; return the four inputs. Never a tier."""
    transcript = [
        {"role": m["role"], "content": m["content"]}
        for m in state.get("messages", [])
        if m.get("role") in ("user", "assistant")
    ]
    if not transcript:
        return {"suggested": dict(DEFAULT_INPUTS), "rationale": "nothing said yet"}

    prompt = list(transcript)

    try:
        reply = llm.ask(
            state.get("model", llm.DEFAULT_MODEL),
            SUGGEST_SYSTEM,
            prompt,
            schema=ESTIMATE_SCHEMA,
            max_tokens=700,
        )
        got = reply.data
    except Exception as exc:  # noqa: BLE001
        # A failed estimate must not silently become a permissive one. Falling
        # back to the defaults keeps the tier where it was rather than opening
        # it up because a network call went wrong.
        return {
            "suggested": dict(state.get("suggested") or DEFAULT_INPUTS),
            "rationale": f"estimate unavailable ({type(exc).__name__}); "
                         f"previous values kept",
        }

    return {
        "suggested": {
            "harm": clamp(got.get("harm"), 1, 10, DEFAULT_INPUTS["harm"]),
            "reversibility": clamp(
                got.get("reversibility"), 1, 10, DEFAULT_INPUTS["reversibility"]
            ),
            "rights": bool(got.get("rights", False)),
            "confidence": clamp(
                got.get("confidence"), 0, 100, DEFAULT_INPUTS["confidence"]
            ),
        },
        "rationale": str(got.get("rationale", "")).strip()[:240],
        "input_tokens": reply.input_tokens,
        "output_tokens": reply.output_tokens,
        "assessed": True,
    }


def classify(state: ConsoleState) -> dict[str, Any]:
    """Pick the effective inputs, run the ladder, compute the warning.

    No model, no network, no randomness. Same inputs, same tier, every time.
    """
    suggested = state.get("suggested") or dict(DEFAULT_INPUTS)
    user = state.get("user") or dict(DEFAULT_INPUTS)
    effective = suggested if state.get("mode", "auto") == "auto" else user

    tier = ladder.classify_inputs(effective)
    suggested_tier = ladder.classify_inputs(suggested)
    out = {
        "effective": effective,
        "tier": tier,
        "suggested_tier": suggested_tier,
        "warning": ladder.warning_for(tier, suggested_tier),
    }
    # Emitted before the reply is written, so the panel settles while the
    # answer is still arriving. The control level is the part a person is
    # waiting on; making them read a paragraph first would be backwards.
    emit({
        "kind": "estimate",
        "suggested": suggested,
        "rationale": state.get("rationale", ""),
        "tier": tier,
        "suggested_tier": suggested_tier,
        "warning": out["warning"],
    })
    return out


__all__ = ["ESTIMATE_SCHEMA", "SUGGEST_SYSTEM", "classify", "suggest"]
