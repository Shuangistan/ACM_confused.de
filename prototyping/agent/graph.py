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
    graph.add_edge("derive", "propose")
    graph.add_edge("propose", "gate")
    # The only question the gate asks is whether approved logic covers the
    # case. Nothing is said to the person while it waits.
    graph.add_conditional_edges("gate", nodes.may_proceed,
                                {"park": "park", "respond": "respond"})
    graph.add_edge("park", END)
    # Every decision is abstracted, not only the ones the rule base missed. A
    # case it already decided can still be carrying logic nobody has written
    # down, and a routine decision is exactly where that logic hides — nobody
    # reviews it, so nobody notices it was never captured. The third call is
    # the price of the database learning from ordinary work rather than only
    # from its failures.
    graph.add_edge("respond", "record")
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


def reply_for_settled(case: dict, verdict: str, actor: str, note: str) -> str:
    """The answer a parked inquiry was owed, written once a person has ruled.

    Composed here rather than by a model. The expert has just made the
    decision; asking a model to narrate it would put a generated account
    between their judgement and the person waiting for it, which is the thing
    this arrangement exists to prevent.
    """
    from agent import ladder

    tier = ladder.TIERS.get(case.get("tier", ""), {}).get("name", "reviewed")
    who = actor or "an expert"
    if verdict != "approved":
        head = (f"{who} has reviewed this and declined to approve it.")
    else:
        head = f"{who} has reviewed this and approved it to proceed."

    lines = [head, ""]
    conclusions = case.get("conclusions") or []
    if conclusions:
        lines.append("Conclusion, from the approved rule base:")
        lines += [f"  {c}" for c in conclusions]
    else:
        lines.append("No rule in the database bears on this case, so there is "
                     "no derived conclusion — only this person's judgement.")
    lines += ["", f"Control level: {tier}."]
    if note:
        lines += ["", f"Their note: {note}"]
    return "\n".join(lines)


__all__ = ["app", "build", "reply_for_settled", "turn", "turn_stream"]
