"""The logic database, doing its job.

`extract` turns the conversation into ground facts; `derive` runs the approved
rules over them. Only `derive` can produce a decision deterministically: what it
concludes is a function of the facts and the approved rule base, so the same
case put twice gives the same answer and the same proof."""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from typing import Any

from logic.engine import Engine
from logic.facts import FactStore, Truth
from logic.parse import ParseError, parse_atom, parse_rule
from logic.syntax import UnsafeRule

from .. import db, llm
from ..state import ConsoleState
from .common import emit

FACTS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"facts": {"type": "array", "items": {"type": "string"}}},
    "required": ["facts"],
    "additionalProperties": False,
}

EXTRACT_SYSTEM = """\
Rewrite a described case as logical atoms for a rule engine. Atoms only — not \
sentences, not notes, not assessments.

Form: `predicate(c)`. Lowercase predicate, underscores between words, the \
constant `c` for this case. Two places where a relation needs them: \
`owes(c, landlord)`.

    in_arrears(c)
    debt_disputed(c)
    not formal_notice_given(c)

`not ` marks something stated to be **absent**. Say nothing about what was \
never mentioned: silence is not a negative, and a fact invented here becomes a \
reason against a person later.

{vocabulary}

Eight atoms at most. Record what is the case, never what should follow from \
it — no atom whose name contains should, must, recommend, suggest or defer.\
"""


def _vocabulary_block() -> str:
    terms = db.store().vocabulary()
    if not terms:
        return ("No terms are in use yet, so you are choosing the first ones. "
                "Prefer plain, general names.")
    return (
        "Terms already in the database. Reuse these exactly wherever one fits, "
        "and coin a new one only when none does — a synonym is worse than a "
        "clumsy match, because the rules written against the existing term "
        "will not fire on yours.\n    " + ", ".join(terms)
    )


def extract(state: ConsoleState) -> dict[str, Any]:
    """The conversation, as ground facts. Its own call and its own prompt.

    This was briefly folded into the estimator to save a call. One system
    prompt doing two jobs blended them: the model stayed in assessment voice
    and returned English sentences, every one dropped at parse time, so the
    rule base could never fire at all. Cheaper, and useless.
    """
    messages = [
        {"role": m["role"], "content": m["content"]}
        for m in state.get("messages", []) if m.get("role") in ("user", "assistant")
    ]
    if not messages:
        return {"facts": []}
    try:
        reply = llm.ask(
            state.get("model", llm.DEFAULT_MODEL),
            EXTRACT_SYSTEM.format(vocabulary=_vocabulary_block()),
            messages, schema=FACTS_SCHEMA, max_tokens=600,
        )
    except Exception:  # noqa: BLE001
        return {"facts": []}
    return {
        "facts": [str(f).strip() for f in reply.data.get("facts", [])
                  if str(f).strip()][:20],
        "input_tokens": reply.input_tokens,
        "output_tokens": reply.output_tokens,
    }


def _fact_store(facts: list[str]) -> tuple[FactStore, list[str]]:
    """Turn extracted strings into ground facts, dropping what will not parse.

    A malformed atom is discarded rather than guessed at. The model is writing
    into a database that other decisions will rest on, and a fact nobody can
    parse is better lost than approximated.
    """
    store, dropped = FactStore(), []
    for text in facts:
        body, negated = text.strip(), False
        if body.lower().startswith("not "):
            body, negated = body[4:].strip(), True
        try:
            atom = parse_atom(body)
        except (ParseError, ValueError):
            dropped.append(text)
            continue
        if not atom.is_ground():
            dropped.append(text)
            continue
        store.assert_fact(atom, Truth.FALSE if negated else Truth.TRUE,
                          source="extracted from the conversation")
    return store, dropped


def derive(state: ConsoleState) -> dict[str, Any]:
    """Run the approved rules over the extracted facts. No model involved.

    This is the only place a decision can come from deterministically. What it
    concludes is a function of the facts and the approved rule base, so the
    same case put twice produces the same answer and the same proof.
    """
    facts, _dropped = _fact_store(state.get("facts") or [])
    rows = db.store().active_rules()

    parsed: list[tuple[str, Any]] = []
    for row in rows:
        try:
            parsed.append((row.canonical_key, parse_rule(row.text)))
        except (ParseError, UnsafeRule, ValueError):
            continue        # a stored rule that no longer parses is skipped,
                            # never silently repaired

    if not parsed or len(facts) == 0:
        return {"conclusions": [], "proof": [], "rules_used": [],
                "covered": False, "case_id": _case_id(state)}

    try:
        result = Engine().evaluate(parsed, facts)
    except Exception:  # noqa: BLE001
        return {"conclusions": [], "proof": [], "rules_used": [],
                "covered": False, "case_id": _case_id(state)}

    conclusions = [str(a) for a in result.conclusions()]
    proof: list[str] = []
    for atom in result.conclusions():
        for support in result.derived[atom].supports:
            premises = ", ".join(str(p) for p in support.premises)
            proof.append(f"{atom} <- {premises}  [{support.rule_id[:8]}]")

    covered = bool(conclusions)
    out = {
        "conclusions": conclusions,
        "proof": proof,
        "rules_used": sorted(result.rules_used),
        "covered": covered,
        # Minted here so the gate can sample on it and a parked inquiry has an
        # identity to be approved against.
        "case_id": _case_id(state),
    }
    emit({"kind": "derived", **out})
    return out


def _case_id(state: ConsoleState) -> str:
    """Unique per inquiry, not per content.

    An earlier version hashed the question and its facts, so asking the same
    thing twice overwrote the first answer. Convenient for replay and useless
    as an audit trail — and impossible once a parked decision needs its own
    identity to be approved against.
    """
    seed = f"{state.get('session','')}|{len(state.get('messages') or [])}|" \
           f"{'|'.join(state.get('facts') or [])}|{_now_seed()}"
    return hashlib.sha256(seed.encode()).hexdigest()[:16]


def _now_seed() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()


__all__ = ["EXTRACT_SYSTEM", "FACTS_SCHEMA", "derive", "extract"]
