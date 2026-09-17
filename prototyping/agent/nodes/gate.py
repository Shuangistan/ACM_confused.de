"""May this decision proceed without a person?

The tier sets what is at stake, not whether a person is asked. What decides that
is whether the logic has been approved already: a rule an expert signed off *is*
their decision, and applying it executes a call they have made rather than
making a new one."""

from __future__ import annotations

import hashlib
from typing import Any

from .. import ladder
from ..state import ConsoleState
from .common import emit
from .learn import record_case

#: How often a decision that proceeded is pulled for an audit afterwards.
#: Approval happens once per rule; this is the part that never stops. The floor
#: is highest where a wrong decision hurts most, so the logic with the gravest
#: consequences is the logic that stays watched, and it is never zero — a rule
#: right a thousand times can still start being wrong when the population
#: shifts underneath it.
AUDIT_RATE = {
    ladder.HUMAN_ONLY: 0.50,
    ladder.HUMAN_APPROVAL: 0.20,
    ladder.AI_RECOMMENDS: 0.10,
    ladder.AI_WITH_OVERSIGHT: 0.05,
}


def _sampled(case_id: str, rate: float) -> bool:
    """Whether this decision is pulled, decided reproducibly.

    Not `random`. Asked why a particular case was or was not audited, "the
    random number generator" is not an answer anybody should accept; hashing
    the case id gives a uniform draw an auditor can recompute.
    """
    if rate >= 1.0:
        return True
    if rate <= 0.0:
        return False
    draw = int.from_bytes(
        hashlib.sha256(f"audit:{case_id}".encode()).digest()[:8], "big") / 2**64
    return draw < rate


def gate(state: ConsoleState) -> dict[str, Any]:
    """Block only when no approved rule covers the case.

    The tier sets how much is at stake, not whether a person is asked. What
    decides that is whether the logic has been approved already: a rule an
    expert signed off *is* their decision, and applying it is executing a call
    they have made, not making a new one. Asking them to confirm it again for
    every matching case is how oversight turns into rubber-stamping — the
    failure this whole design exists to avoid.
    """
    tier = state.get("tier") or ladder.HUMAN_APPROVAL
    covered = bool(state.get("covered"))
    case_id = state.get("case_id") or ""

    if covered:
        blocked, why = False, ""
    elif tier in (ladder.HUMAN_ONLY, ladder.HUMAN_APPROVAL):
        blocked = True
        why = ("No approved rule covers this case, and its consequences put it "
               f"at {ladder.TIERS[tier]['name'].lower()}.")
    else:
        blocked, why = False, ""

    sampled = (not blocked) and _sampled(case_id, AUDIT_RATE.get(tier, 0.1))
    out = {"blocked": blocked, "block_reason": why, "sampled": sampled}
    if blocked:
        emit({"kind": "waiting", "case_id": case_id, "tier": tier, "why": why})
    return out


def may_proceed(state: ConsoleState) -> str:
    return "park" if state.get("blocked") else "respond"


def park(state: ConsoleState) -> dict[str, Any]:
    """Write the parked inquiry down and stop. No reply is generated.

    Deliberately nothing is said about the outcome while it waits. A holding
    answer that gestured at the likely decision would be the decision, made
    without the person whose job it was.
    """
    record_case(state, status="waiting")
    return {}


__all__ = ["AUDIT_RATE", "gate", "may_proceed", "park"]
