"""What travels along the graph's edges.

Kept to primitives. Anything in graph state is checkpointed, so it has to
survive a round trip; a rich object that serialises subtly wrong would corrupt
the record of what was decided rather than raise.
"""

from __future__ import annotations

import operator
from typing import Annotated, Any, TypedDict

#: The four MVP inputs, in the shape everything passes around.
Inputs = dict[str, Any]

#: Fallbacks only, for the case where an estimate could not be produced. Not
#: defaults in any meaningful sense: the console shows no tier until a real
#: estimate exists, so these are never presented to anyone as a judgement.
#: Deliberately mid-scale rather than the MVP's showcase numbers, which were
#: chosen to make a demo land on a particular tier.
DEFAULT_INPUTS: Inputs = {
    "harm": 5,
    "reversibility": 5,
    "rights": False,
    "confidence": 50,
}


class ConsoleState(TypedDict, total=False):
    """One session's worth of conversation and control allocation."""

    #: Plain dicts of {role, content}. Not LangChain message objects: the API
    #: boundary is JSON either way, and a second representation to convert
    #: between is one more place for a turn to be lost.
    messages: Annotated[list[dict[str, str]], operator.add]

    model: str            # haiku | sonnet | opus
    mode: str             # auto | manual

    #: "intake" until a decision has actually been described, then "deciding".
    #: Nothing is classified during intake: a greeting is not a decision, and a
    #: system that scores one has confused having a conversation with having a
    #: problem.
    phase: str
    #: The decision, as the intake node came to understand it. Written once at
    #: handover; it is what the decision side is answering.
    problem: str

    #: What the person set. Trusted, and always allowed to win.
    user: Inputs
    #: What `suggest` read out of the conversation.
    suggested: Inputs
    #: Whichever of the two was actually used.
    effective: Inputs

    #: Accumulated over the session by the checkpointer, so the figure the page
    #: shows is the conversation's real cost rather than the last turn's.
    input_tokens: Annotated[int, operator.add]
    output_tokens: Annotated[int, operator.add]
    #: True once a turn has completed. Before that there is no decision to
    #: classify, and the panel must not imply otherwise.
    assessed: bool

    #: Facts the extractor read out of the conversation, as rule-syntax atoms.
    facts: list[str]
    #: What the rule base concluded from them, and which rules did it.
    conclusions: list[str]
    proof: list[str]
    #: Rules that came within a literal or two of deciding this case.
    near: list[dict]
    #: What abstraction produced for this case, kept and rejected alike.
    drafts: dict
    rules_used: list[str]
    #: True when approved rules answered the case; false when nothing fired and
    #: the model had to. The share of turns where this is true is the coverage
    #: figure, and the only honest measure of whether the database is working.
    covered: bool

    #: Set by `gate`. Blocked means no approved rule covers a case whose
    #: consequences require a person, so it is parked rather than answered.
    #: The answer the agent prepared for a blocked case. Shown to the expert,
    #: never to the person waiting, until somebody has ruled on it.
    draft: str
    blocked: bool
    block_reason: str
    sampled: bool
    case_id: str
    session: str

    tier: str
    suggested_tier: str
    warning: str | None
    #: The suggester's one-line account of why it estimated what it did.
    rationale: str


__all__ = ["ConsoleState", "DEFAULT_INPUTS", "Inputs"]
