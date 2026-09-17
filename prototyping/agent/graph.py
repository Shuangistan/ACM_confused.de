"""The topology.

    START -> suggest -> classify -> respond -> END

`suggest` runs first so that `respond` knows the tier it is operating under.
The order is the mechanism, not a detail: a reply written before the control
level is known cannot be constrained by it, and the constraint is the product.

`classify` sits between them and touches no model. Everything either side of it
is an estimate; it alone is a decision, and it is the same decision every time
for the same numbers.
"""

from __future__ import annotations

from typing import Any

from langgraph.graph import END, START, StateGraph

from . import nodes
from .state import ConsoleState

_compiled: Any = None


def build():
    graph = StateGraph(ConsoleState)
    graph.add_node("intake", nodes.intake)
    graph.add_node("suggest", nodes.suggest)
    graph.add_node("classify", nodes.classify)
    graph.add_node("extract", nodes.extract)
    graph.add_node("derive", nodes.derive)
    graph.add_node("gate", nodes.gate)
    graph.add_node("park", nodes.park)
    graph.add_node("respond", nodes.respond)
    graph.add_node("propose", nodes.propose)
    graph.add_node("record", nodes.record)

    # Which side of the system a turn enters. Once a decision has been
    # described the intake node is out of the path entirely, so later turns
    # cost one call rather than two.
    graph.set_conditional_entry_point(
        lambda s: "suggest" if s.get("phase") == "deciding" else "intake",
        ["intake", "suggest"],
    )
    graph.add_conditional_edges("intake", nodes.ready_to_decide,
                                {"suggest": "suggest", "__end__": END})
    graph.add_edge("suggest", "classify")
    graph.add_edge("classify", "extract")
    graph.add_edge("extract", "derive")
    # Abstraction runs before the gate, so a parked inquiry still contributes
    # candidate logic. Those are the most consequential cases in the system;
    # having them wait for a person *and* leave the database none the wiser
    # would mean the logic of hard decisions is the logic never written down.
    #
    # A case the rules already decided is skipped, though. Its logic is by
    # definition written down, so a proposal could only restate it — and the
    # cost of a turn is then supposed to *fall* as coverage rises. Paying for a
    # proposal on every covered case makes that claim false.
    graph.add_conditional_edges(
        "derive",
        lambda s: "gate" if s.get("covered") else "propose",
        {"gate": "gate", "propose": "propose"},
    )
    graph.add_edge("propose", "gate")
    # The only question the gate asks is whether approved logic covers the
    # case. Nothing is said to the person while it waits.
    # The gate decides before the answer is written, but the answer is written
    # either way. An expert asked to approve a bare question has nothing to
    # approve: the matrix says the agent *prepares* the action and a person
    # rules on it, so the draft is what they are there to check. What the gate
    # controls is who sees it — the person waiting, or the expert first.
    graph.add_edge("gate", "respond")
    graph.add_conditional_edges(
        "respond",
        lambda s: "park" if s.get("blocked") else "record",
        {"park": "park", "record": "record"},
    )
    graph.add_edge("park", END)
    # Every decision is abstracted, not only the ones the rule base missed. A
    # case it already decided can still be carrying logic nobody has written
    # down, and a routine decision is exactly where that logic hides — nobody
    # reviews it, so nobody notices it was never captured. The third call is
    # the price of the database learning from ordinary work rather than only
    # from its failures.
    graph.add_edge("record", END)
    return graph


def app():
    """Compiled once and reused; the checkpointer holds the sessions."""
    global _compiled
    if _compiled is None:
        from langgraph.checkpoint.memory import InMemorySaver

        _compiled = build().compile(checkpointer=InMemorySaver())
    return _compiled


def turn(
    session: str,
    message: str,
    model: str,
    mode: str,
    user_inputs: dict,
) -> dict[str, Any]:
    """One exchange. History comes from the checkpointer, keyed by session."""
    result = app().invoke(
        {
            "messages": [{"role": "user", "content": message}],
            "model": model,
            "mode": mode,
            "user": user_inputs,
            "input_tokens": 0,
            "output_tokens": 0,
        },
        {"configurable": {"thread_id": session}},
    )
    reply = ""
    for entry in reversed(result.get("messages", [])):
        if entry.get("role") == "assistant":
            reply = entry.get("content", "")
            break
    return {
        "reply": reply,
        "suggested": result.get("suggested", {}),
        "rationale": result.get("rationale", ""),
        "tier": result.get("tier", ""),
        "suggested_tier": result.get("suggested_tier", ""),
        "warning": result.get("warning"),
        "assessed": bool(result.get("assessed")),
        "covered": bool(result.get("covered")),
        "blocked": bool(result.get("blocked")),
        "block_reason": result.get("block_reason", ""),
        "case_id": result.get("case_id", ""),
        "conclusions": result.get("conclusions", []),
        "phase": result.get("phase", "intake"),
        "problem": result.get("problem", ""),
        "usage": {
            "input_tokens": result.get("input_tokens", 0),
            "output_tokens": result.get("output_tokens", 0),
        },
    }


def turn_stream(
    session: str,
    message: str,
    model: str,
    mode: str,
    user_inputs: dict,
):
    """The same turn, yielding events as they happen.

    `estimate` arrives once `classify` has run, then `text` fragments as the
    reply is written, then `done` with the session's token totals. The order is
    the point: the control level lands first, because that is what the person
    is waiting to know.
    """
    payload = {
        "messages": [{"role": "user", "content": message}],
        "model": model,
        "mode": mode,
        "user": user_inputs,
        "session": session,
        "input_tokens": 0,
        "output_tokens": 0,
    }
    config = {"configurable": {"thread_id": session}}
    for event in app().stream(payload, config, stream_mode="custom"):
        yield event

    # Totals are read from the settled state rather than accumulated here, so
    # the figure is the session's, not this turn's.
    final = app().get_state(config).values
    yield {
        "kind": "done",
        "phase": final.get("phase", "intake"),
        "covered": bool(final.get("covered")),
        "blocked": bool(final.get("blocked")),
        "case_id": final.get("case_id", ""),
        "problem": final.get("problem", ""),
        "usage": {
            "input_tokens": final.get("input_tokens", 0),
            "output_tokens": final.get("output_tokens", 0),
        },
    }


def reply_for_settled(case: dict, verdict: str, actor: str, note: str,
                      edited: str = "") -> str:
    """What the person waiting finally sees.

    The agent prepared a draft; an expert has approved it, changed it, or
    refused it. The text sent is theirs either way, and it says so — the point
    of the arrangement is that a named person stands behind the answer, which
    is only true if the answer is attributed to them.
    """
    from agent import ladder

    who = actor or "an expert"
    body = (edited or "").strip() or (case.get("reply") or "").strip()

    if verdict != "approved":
        head = f"{who} reviewed this and did not approve it."
        parts = [head]
        if note:
            parts += ["", note]
        return "\n".join(parts)

    changed = bool(edited.strip()) and edited.strip() != (case.get("reply") or "").strip()
    head = (f"Reviewed and approved by {who}"
            + (", who revised it." if changed else "."))

    parts = [head, "", body or "(no answer was prepared)"]
    conclusions = case.get("conclusions") or []
    if conclusions:
        parts += ["", "Derived from approved logic:"]
        parts += [f"  {c}" for c in conclusions]
    tier = ladder.TIERS.get(case.get("tier", ""), {}).get("name")
    if tier:
        parts += ["", f"Control level: {tier}."]
    if note:
        parts += ["", f"Note from {who}: {note}"]
    return "\n".join(parts)


__all__ = ["app", "build", "reply_for_settled", "turn", "turn_stream"]
