"""The OpenAI client, written against Chat Completions on purpose.

The Responses API is the newer surface, but Chat Completions is what every
OpenAI-compatible server speaks -- Azure OpenAI, vLLM, Ollama, OpenRouter, LM
Studio. A summer-school prototype that a student can point at a local model by
setting one environment variable is worth more here than one that can only talk
to api.openai.com, and nothing in this system needs what Responses adds.

Three things about the request shape are not obvious, and each of them is a 400
rather than a silently ignored field:

  * reasoning models (gpt-5, o1, o3, o4) take `max_completion_tokens`;
    everything else takes `max_tokens`, and each rejects the other's name.
  * those same models take `reasoning_effort` and reject `temperature`;
    the rest are the other way round. So this sends neither unless it is sure.
  * `strict: true` structured outputs require every object in the schema to set
    `additionalProperties: false` and to list every property in `required`.
    The four schemas in this system already do, but `_strictify` enforces it
    rather than trusting that, because the failure is a 400 at the far end of a
    graph rather than an error where the schema was written.
"""

from __future__ import annotations

import copy
import json
import os
from typing import Any

from .common import Reply, RefusalError, env_or_dotenv

NAME = "openai"

#: Tier -> model id. Overridable because OpenAI's catalogue moves faster than
#: this file does, and because an account that has no access to one of these
#: should not need a code change. `check_setup.py` reports what is reachable.
MODELS: dict[str, str] = {
    "haiku":  os.environ.get("OPENAI_MODEL_SMALL")  or "gpt-4o-mini",
    "sonnet": os.environ.get("OPENAI_MODEL_MEDIUM") or "gpt-4o",
    "opus":   os.environ.get("OPENAI_MODEL_LARGE")  or "gpt-4.1",
}
LABELS: dict[str, str] = {k: v for k, v in MODELS.items()}

#: Prefixes whose models reason internally, take `reasoning_effort`, and want
#: `max_completion_tokens`.
REASONING_PREFIXES = ("gpt-5", "o1", "o3", "o4")


def is_reasoning(model_id: str) -> bool:
    return model_id.startswith(REASONING_PREFIXES)


def supports_effort(model_id: str) -> bool:
    """Kept for interface parity with the Anthropic backend."""
    return is_reasoning(model_id)


def load_api_key() -> str:
    key = env_or_dotenv("OPENAI_API_KEY", "API_KEY")
    if key:
        return key
    raise RuntimeError(
        "No OpenAI API key. Set OPENAI_API_KEY in the environment, or put "
        "OPENAI_API_KEY=sk-... in prototyping\\.env"
    )


_client: Any = None


def client() -> Any:
    global _client
    if _client is None:
        from openai import OpenAI

        kwargs: dict[str, Any] = {"api_key": load_api_key()}
        # Set OPENAI_BASE_URL to point at Azure, a local server, or a proxy.
        base = env_or_dotenv("OPENAI_BASE_URL")
        if base:
            kwargs["base_url"] = base
        _client = OpenAI(**kwargs)
    return _client


def _strictify(schema: dict[str, Any]) -> dict[str, Any]:
    """Every object closed, every property required. Recursively."""
    out = copy.deepcopy(schema)

    def walk(node: Any) -> None:
        if not isinstance(node, dict):
            return
        if node.get("type") == "object":
            node["additionalProperties"] = False
            props = node.get("properties") or {}
            node["required"] = list(props.keys())
            for child in props.values():
                walk(child)
        for key in ("items", "prefixItems"):
            child = node.get(key)
            if isinstance(child, list):
                for c in child:
                    walk(c)
            elif child is not None:
                walk(child)
        for key in ("anyOf", "oneOf", "allOf"):
            for c in node.get(key) or []:
                walk(c)

    walk(out)
    return out


def _prompt(system: str, messages: list[dict[str, str]]) -> list[dict[str, str]]:
    """Chat Completions takes the system prompt as the first message.

    `system` rather than `developer`: the newer role name is not understood by
    every compatible server, and on OpenAI's own models the two behave the same
    for a prompt this simple.
    """
    return [{"role": "system", "content": system}] + [
        {"role": m["role"], "content": m["content"]}
        for m in messages
        if m.get("role") in ("user", "assistant")
    ]


def _budget(model_id: str, max_tokens: int) -> dict[str, Any]:
    key = "max_completion_tokens" if is_reasoning(model_id) else "max_tokens"
    return {key: max_tokens}


def _call(kwargs: dict[str, Any]) -> Any:
    """One request, with a single retry if the token-budget name was wrong.

    The prefix list above is a guess about a catalogue that changes. When it
    guesses wrong the server says so precisely, so the cheap correct move is to
    believe it and send the other name -- rather than make the operator read a
    stack trace to learn that `max_tokens` should have been
    `max_completion_tokens`.
    """
    try:
        return client().chat.completions.create(**kwargs)
    except Exception as exc:  # noqa: BLE001 -- narrowed by inspecting the text
        text = str(exc)
        swap = {"max_tokens": "max_completion_tokens",
                "max_completion_tokens": "max_tokens"}
        for old, new in swap.items():
            if old in kwargs and old in text and "unsupported" in text.lower():
                kwargs[new] = kwargs.pop(old)
                return client().chat.completions.create(**kwargs)
        raise


def _usage(response: Any) -> tuple[int, int]:
    u = getattr(response, "usage", None)
    return (int(getattr(u, "prompt_tokens", 0) or 0),
            int(getattr(u, "completion_tokens", 0) or 0))


def ask(model_id: str, system: str, messages: list[dict[str, str]],
        schema: dict[str, Any] | None = None, max_tokens: int = 1024) -> Reply:
    kwargs: dict[str, Any] = {
        "model": model_id,
        "messages": _prompt(system, messages),
        **_budget(model_id, max_tokens),
    }
    if schema is not None:
        kwargs["response_format"] = {
            "type": "json_schema",
            "json_schema": {
                "name": "reply",
                "schema": _strictify(schema),
                "strict": True,
            },
        }
    if is_reasoning(model_id):
        # Low, not default: everything this system asks a model to do in a
        # structured call is extraction, not deliberation, and reasoning tokens
        # are billed as output.
        kwargs["reasoning_effort"] = "low"

    response = _call(kwargs)
    choice = response.choices[0]

    # Checked before reading `content`: a refusal leaves content null, and the
    # resulting TypeError would name the wrong cause.
    if getattr(choice.message, "refusal", None):
        raise RefusalError(str(choice.message.refusal))
    if choice.finish_reason == "length":
        raise RuntimeError(
            f"the reply hit the {max_tokens}-token budget before finishing"
        )

    text = choice.message.content or ""
    if not text:
        raise RuntimeError(
            f"no text in response (finish_reason={choice.finish_reason})"
        )
    inp, out = _usage(response)
    return Reply(
        data=json.loads(text) if schema is not None else text,
        input_tokens=inp,
        output_tokens=out,
    )


def stream(model_id: str, system: str, messages: list[dict[str, str]],
           on_text, max_tokens: int = 1024) -> Reply:
    kwargs: dict[str, Any] = {
        "model": model_id,
        "messages": _prompt(system, messages),
        "stream": True,
        # Without this the final chunk carries no usage and the page's cost
        # meter reads zero for every streamed reply -- which is worse than no
        # meter, because it looks like an answer.
        "stream_options": {"include_usage": True},
        **_budget(model_id, max_tokens),
    }
    if is_reasoning(model_id):
        kwargs["reasoning_effort"] = "low"

    inp = out = 0
    refusal = ""
    for chunk in _call(kwargs):
        if getattr(chunk, "usage", None):
            inp, out = _usage(chunk)
        for choice in chunk.choices or []:
            delta = choice.delta
            if getattr(delta, "refusal", None):
                refusal += delta.refusal
            fragment = getattr(delta, "content", None)
            if fragment:
                on_text(fragment)

    if refusal:
        raise RefusalError(refusal)
    return Reply(data="", input_tokens=inp, output_tokens=out)


__all__ = ["NAME", "MODELS", "LABELS", "ask", "client", "is_reasoning",
           "load_api_key", "stream", "supports_effort"]
