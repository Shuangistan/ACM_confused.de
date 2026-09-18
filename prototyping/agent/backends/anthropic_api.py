"""The Anthropic client, and the one thing about it that is not obvious.

`output_config.effort` and adaptive thinking arrived with the 4.6 generation.
Older models do not ignore them -- they reject the request with a 400 -- so the
request shape has to follow the model rather than assume the newest one. Haiku
4.5 is the default here precisely because it is cheap, which means this is the
common path, not an edge case.
"""

from __future__ import annotations

import json
from typing import Any

from .common import Reply, RefusalError, env_or_dotenv

NAME = "anthropic"

#: What the page's selector offers. Haiku is the default: a summer-school
#: prototype does not need Opus to estimate a harm score, and the cost
#: difference over a demo is the difference between free and not.
MODELS: dict[str, str] = {
    "haiku": "claude-haiku-4-5",
    "sonnet": "claude-sonnet-5",
    "opus": "claude-opus-5",
}
LABELS: dict[str, str] = {
    "haiku": "Claude Haiku 4.5",
    "sonnet": "Claude Sonnet 5",
    "opus": "Claude Opus 5",
}

#: Models that reject `effort` and adaptive thinking rather than ignoring them.
LEGACY_PREFIXES = ("claude-haiku-4-5", "claude-haiku-3", "claude-3")


def supports_effort(model_id: str) -> bool:
    return not model_id.startswith(LEGACY_PREFIXES)


def load_api_key() -> str:
    key = env_or_dotenv("ANTHROPIC_API_KEY", "API_KEY")
    if key:
        return key
    raise RuntimeError(
        "No Anthropic API key. Set ANTHROPIC_API_KEY in the environment, or "
        "put API_KEY=sk-ant-... in prototyping/.env"
    )


_client: Any = None


def client() -> Any:
    global _client
    if _client is None:
        import anthropic

        _client = anthropic.Anthropic(api_key=load_api_key())
    return _client


def ask(model_id: str, system: str, messages: list[dict[str, str]],
        schema: dict[str, Any] | None = None, max_tokens: int = 1024) -> Reply:
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


def stream(model_id: str, system: str, messages: list[dict[str, str]],
           on_text, max_tokens: int = 1024) -> Reply:
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


__all__ = ["NAME", "MODELS", "LABELS", "ask", "client", "load_api_key",
           "stream", "supports_effort"]
