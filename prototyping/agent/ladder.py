"""The classification. Four numbers in, one control tier out.

This module imports nothing — not LangGraph, not the Anthropic SDK, not even
the rest of this package. That is deliberate and worth defending: it is the one
piece that decides how much human oversight a decision gets, and it must stay
deterministic, cheap to test, and impossible to accidentally make dependent on
a model's output. A module that cannot reach a network cannot be talked into
anything.

The ladder is transcribed from `decision_partner_mvp.html:76-79`. A second copy
lives in the page itself, because the sliders have to respond without a round
trip. Two copies of the rule that allocates oversight is exactly the kind of
thing that silently diverges, so `tests/test_ladder_parity.py` extracts the
JavaScript from the shipped page and checks both against each other over every
possible input.
"""

from __future__ import annotations

# --------------------------------------------------------------------------
# The four tiers
# --------------------------------------------------------------------------
HUMAN_ONLY = "human_only"
HUMAN_APPROVAL = "human_approval"
AI_WITH_OVERSIGHT = "ai_with_oversight"
AI_RECOMMENDS = "ai_recommends"

TIERS: dict[str, dict[str, str]] = {
    HUMAN_ONLY: {
        "risk": "Critical",
        "name": "Human only — AI may not decide",
        "why": "Potential harm is very high and the decision is hard to reverse. "
               "The AI may inform a person; it may not make or execute the decision.",
    },
    HUMAN_APPROVAL: {
        "risk": "High consequence",
        "name": "Human approval required",
        "why": "Material harm, a rights impact or thin evidence requires an "
               "accountable person to review and explicitly approve the action.",
    },
    AI_WITH_OVERSIGHT: {
        "risk": "Lower consequence",
        "name": "AI acts, with oversight",
        "why": "Low harm and high reversibility permit bounded autonomy, with "
               "logging, monitoring and a human able to pause or reverse.",
    },
    AI_RECOMMENDS: {
        "risk": "Moderate",
        "name": "AI recommends; the human decides",
        "why": "The AI may analyse and propose. It may not act on its own proposal.",
    },
}

#: How much control a person keeps, ascending. NOT the ladder's evaluation
#: order. `ai_recommends` is the ladder's residual `else` branch, so it looks
#: like the bottom of the scale — but it grants the AI *less* authority than
#: `ai_with_oversight`, which lets it act. Ranking by evaluation order makes the
#: override warning fire backwards in exactly the case that matters.
RESTRICTIVENESS: dict[str, int] = {
    AI_WITH_OVERSIGHT: 0,
    AI_RECOMMENDS: 1,
    HUMAN_APPROVAL: 2,
    HUMAN_ONLY: 3,
}


# --------------------------------------------------------------------------
# Classification
# --------------------------------------------------------------------------
def classify(harm: int, reversibility: int, rights: bool, confidence: int) -> str:
    """The ladder. Evaluated top down; the first tier that matches wins."""
    if (harm >= 9 and rights) or (harm >= 8 and reversibility <= 3):
        return HUMAN_ONLY
    if harm >= 6 or rights or confidence < 70:
        return HUMAN_APPROVAL
    if harm <= 3 and reversibility >= 7:
        return AI_WITH_OVERSIGHT
    return AI_RECOMMENDS


def classify_inputs(inputs: dict) -> str:
    """`classify` over the dict shape the API and the graph pass around."""
    return classify(
        int(inputs["harm"]),
        int(inputs["reversibility"]),
        bool(inputs["rights"]),
        int(inputs["confidence"]),
    )


def is_downgrade(effective: str, suggested: str) -> bool:
    """Whether the effective tier gives the AI more authority than suggested.

    Only this direction warrants a warning. A person who takes *more* control
    than the system proposed needs no telling; a person who takes less has
    granted the AI authority the conversation did not support, and that should
    not be possible to do without noticing.
    """
    return RESTRICTIVENESS[effective] < RESTRICTIVENESS[suggested]


def warning_for(effective: str, suggested: str) -> str | None:
    """The override notice, or None when there is nothing to say."""
    if not is_downgrade(effective, suggested):
        return None
    return (
        f"The conversation suggests {TIERS[suggested]['name']}. You have selected "
        f"{TIERS[effective]['name']}, which gives the AI more authority. Your "
        f"choice stands, and is recorded against your name."
    )


__all__ = [
    "AI_RECOMMENDS", "AI_WITH_OVERSIGHT", "HUMAN_APPROVAL", "HUMAN_ONLY",
    "RESTRICTIVENESS", "TIERS", "classify", "classify_inputs", "is_downgrade",
    "warning_for",
]
