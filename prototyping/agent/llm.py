"""The Anthropic client, and the one thing about it that is not obvious.

`output_config.effort` and adaptive thinking arrived with the 4.6 generation.
Older models do not ignore them — they reject the request with a 400 — so the
request shape has to follow the model rather than assume the newest one. Haiku
4.5 is the default here precisely because it is cheap, which means this is the
common path, not an edge case.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

#: What the page's selector offers. Haiku is the default: a summer-school
#: prototype does not need Opus to estimate a harm score, and the cost
#: difference over a demo is the difference between free and not.
MODELS: dict[str, str] = {
    "haiku": "claude-haiku-4-5",
    "sonnet": "claude-sonnet-5",
    "opus": "claude-opus-5",
}
DEFAULT_MODEL = "haiku"

#: Extraction gets a stronger model than the rest, whatever the page selects.
#: Measured on four ordinary decisions, Haiku 4.5 extracted facts from 0 of 4
#: and Sonnet from 4 of 4, on an identical prompt. Four rounds of rewording had
#: not moved it, because the limit was not the wording. Everything downstream --
#: derivation, near misses, rule proposals, learning from an expert's decision
#: -- is dead when extraction returns nothing, so this is the one call worth
#: paying more for. It is a floor, not an override: pick Opus and extraction
#: uses Opus.
EXTRACTION_FLOOR = "sonnet"
_TIER = {"haiku": 0, "sonnet": 1, "opus": 2}


def at_least(model: str, floor: str) -> str:
    """The stronger of the two, by capability rather than by name."""
    return model if _TIER.get(model, 0) >= _TIER.get(floor, 0) else floor

#: Models that reject `effort` and adaptive thinking rather than ignoring them.
LEGACY_PREFIXES = ("claude-haiku-4-5", "claude-haiku-3", "claude-3")


class RefusalError(RuntimeError):
    """The model declined. Surfaced, never swallowed into an empty result."""


@dataclass
class Reply:
    """What a call produced, and what it cost.

    Usage travels with the result rather than being logged aside, because a
    prototype run on a budget needs the cost visible at the point of use, not
    discoverable afterwards in a dashboard nobody opens.
    """

    data: Any
    input_tokens: int = 0
    output_tokens: int = 0


def supports_effort(model_id: str) -> bool:
    return not model_id.startswith(LEGACY_PREFIXES)


def load_api_key() -> str:
    """From the environment, falling back to the project's .env."""
    key = os.environ.get("ANTHROPIC_API_KEY")
    if key:
        return key
    for candidate in (
        Path(__file__).resolve().parents[2] / ".env",
        Path(__file__).resolve().parents[1] / ".env",
    ):
        if not candidate.exists():
            continue
        for line in candidate.read_text().splitlines():
            name, _, value = line.partition("=")
            if name.strip() in ("ANTHROPIC_API_KEY", "API_KEY"):
                return value.strip().strip('"').strip("'")
    raise RuntimeError(
        "No API key. Set ANTHROPIC_API_KEY, or put API_KEY=... in .env"
    )


_client: Any = None


def client() -> Any:
    global _client
    if _client is None:
        import anthropic

        _client = anthropic.Anthropic(api_key=load_api_key())
    return _client


def ask(
    model: str,
    system: str,
    messages: list[dict[str, str]],
    schema: dict[str, Any] | None = None,
    max_tokens: int = 1024,
) -> Reply:
    """One call. With `schema`, the reply is forced to match it and parsed."""
    model_id = MODELS.get(model, MODELS[DEFAULT_MODEL])
    kwargs: dict[str, Any] = {
        "model": model_id,
        "max_tokens": max_tokens,
        "system": system,
        "messages": messages,
    }
    if schema is not None:
        kwargs["output_config"] = {
            "format": {"type": "json_schema", "schema": schema}
        }
        kwargs["betas"] = ["structured-outputs-2025-11-13"]
    if supports_effort(model_id):
        kwargs["thinking"] = {"type": "adaptive"}
        if schema is not None:
            kwargs["output_config"]["effort"] = "low"

    response = client().beta.messages.create(**kwargs)

    # Checked before touching `content`: a refusal can carry an empty content
    # list, and indexing it would raise something unrelated to the cause.
    if getattr(response, "stop_reason", None) == "refusal":
        raise RefusalError("the model declined this request")

    text = next(
        (b.text for b in response.content if getattr(b, "type", None) == "text"), ""
    )
    if not text:
        raise RuntimeError(
            f"no text in response (stop_reason="
            f"{getattr(response, 'stop_reason', None)})"
        )
    usage = getattr(response, "usage", None)
    return Reply(
        data=json.loads(text) if schema is not None else text,
        input_tokens=int(getattr(usage, "input_tokens", 0) or 0),
        output_tokens=int(getattr(usage, "output_tokens", 0) or 0),
    )


def stream(
    model: str,
    system: str,
    messages: list[dict[str, str]],
    on_text,
    max_tokens: int = 1024,
) -> Reply:
    """Like `ask`, but hands each fragment to `on_text` as it arrives.

    Only the reply is streamed. The estimate is not: it is a small structured
    object that is useless in fragments, and the panel needs all of it or none.
    """
    model_id = MODELS.get(model, MODELS[DEFAULT_MODEL])
    kwargs: dict[str, Any] = {
        "model": model_id,
        "max_tokens": max_tokens,
        "system": system,
        "messages": messages,
    }
    if supports_effort(model_id):
        kwargs["thinking"] = {"type": "adaptive"}

    with client().beta.messages.stream(**kwargs) as streamed:
        for fragment in streamed.text_stream:
            on_text(fragment)
        final = streamed.get_final_message()

    if getattr(final, "stop_reason", None) == "refusal":
        raise RefusalError("the model declined this request")
    usage = getattr(final, "usage", None)
    return Reply(
        data="",
        input_tokens=int(getattr(usage, "input_tokens", 0) or 0),
        output_tokens=int(getattr(usage, "output_tokens", 0) or 0),
    )


__all__ = [
    "DEFAULT_MODEL", "EXTRACTION_FLOOR", "MODELS", "at_least", "RefusalError", "Reply", "ask", "client",
    "load_api_key", "stream", "supports_effort",
]
