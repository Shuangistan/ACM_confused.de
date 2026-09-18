"""Which provider answers, and how strong a model it uses.

The system talks to one provider at a time, chosen by `LLM_PROVIDER` at import.
Everything above this module -- the graph, the nodes, the server -- is written
against the six names below and never learns which backend is behind them.

The three tier keys are `haiku`, `sonnet` and `opus` under either provider.
That naming is a wart. It is kept because those strings are already written
into checkpointed conversation state, into the `proposed_by` column of every
rule the agent has drafted, and into every recorded case; renaming them would
make an existing database misreport its own history in order to settle a
cosmetic complaint. Read them as cheap / middle / strong. The page never shows
them: it calls /api/models and displays the provider's real model names.
"""

from __future__ import annotations

import os

from .backends import Reply, RefusalError  # noqa: F401 -- re-exported

#: Set LLM_PROVIDER=openai to run against OpenAI or any compatible server.
PROVIDER = (os.environ.get("LLM_PROVIDER") or "anthropic").strip().lower()

if PROVIDER in ("openai", "oai"):
    from .backends import openai_api as _backend
elif PROVIDER in ("anthropic", "claude"):
    from .backends import anthropic_api as _backend
else:
    raise RuntimeError(
        f"Unknown LLM_PROVIDER={PROVIDER!r}. Use 'anthropic' or 'openai'."
    )

MODELS: dict[str, str] = _backend.MODELS
LABELS: dict[str, str] = _backend.LABELS
DEFAULT_MODEL = "haiku"

#: Extraction gets a stronger model than the rest, whatever the page selects.
#: Measured on four ordinary decisions against Anthropic, Haiku 4.5 extracted
#: facts from 0 of 4 and Sonnet from 4 of 4, on an identical prompt. Four rounds
#: of rewording had not moved it, because the limit was not the wording.
#: Everything downstream -- derivation, near misses, rule proposals, learning
#: from an expert's decision -- is dead when extraction returns nothing, so this
#: is the one call worth paying more for. It is a floor, not an override: pick
#: the strongest tier and extraction uses it.
#:
#: The measurement is Anthropic's. Whether the same cliff sits between the two
#: cheapest OpenAI models is untested here, so the floor is applied to both
#: providers -- the cautious reading of an unmeasured case, not a finding.
EXTRACTION_FLOOR = "sonnet"

_TIER = {"haiku": 0, "sonnet": 1, "opus": 2}
#: Neutral names, accepted on the wire so a future page need not say "haiku".
_ALIASES = {"small": "haiku", "medium": "sonnet", "large": "opus",
            "cheap": "haiku", "strong": "opus"}


def tier(name: str | None) -> str:
    """A wire value, normalised to a tier key this backend knows."""
    key = (name or DEFAULT_MODEL).strip().lower()
    key = _ALIASES.get(key, key)
    return key if key in MODELS else DEFAULT_MODEL


def at_least(model: str, floor: str) -> str:
    """The stronger of the two, by capability rather than by name."""
    model, floor = tier(model), tier(floor)
    return model if _TIER.get(model, 0) >= _TIER.get(floor, 0) else floor


def model_id(name: str | None) -> str:
    """The provider's own identifier for a tier."""
    return MODELS[tier(name)]


def catalogue() -> list[dict[str, object]]:
    """What the page should put in its selector, in tier order."""
    return [
        {"value": key, "label": LABELS.get(key, MODELS[key]),
         "model": MODELS[key], "default": key == DEFAULT_MODEL}
        for key in sorted(MODELS, key=lambda k: _TIER.get(k, 0))
    ]


def supports_effort(name: str) -> bool:
    return _backend.supports_effort(model_id(name) if name in MODELS else name)


def load_api_key() -> str:
    return _backend.load_api_key()


def client():
    return _backend.client()


def ask(model, system, messages, schema=None, max_tokens: int = 1024) -> Reply:
    """One call. With `schema`, the reply is forced to match it and parsed."""
    return _backend.ask(model_id(model), system, messages,
                        schema=schema, max_tokens=max_tokens)


def stream(model, system, messages, on_text, max_tokens: int = 1024) -> Reply:
    """Like `ask`, but hands each fragment to `on_text` as it arrives.

    Only the reply is streamed. The estimate is not: it is a small structured
    object that is useless in fragments, and the panel needs all of it or none.
    """
    return _backend.stream(model_id(model), system, messages, on_text,
                           max_tokens=max_tokens)


__all__ = [
    "DEFAULT_MODEL", "EXTRACTION_FLOOR", "LABELS", "MODELS", "PROVIDER",
    "RefusalError", "Reply", "ask", "at_least", "catalogue", "client",
    "load_api_key", "model_id", "stream", "supports_effort", "tier",
]
