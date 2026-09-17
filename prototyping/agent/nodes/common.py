"""Shared by the node modules: the stream writer, and nothing else.

`emit` is a no-op outside a streaming invocation, so every node works
unchanged under the non-streaming endpoint and under the tests."""

from __future__ import annotations

from typing import Any

from langgraph.config import get_stream_writer

def emit(event: dict[str, Any]) -> None:
    """Push an event to whoever is streaming this run, if anyone is."""
    try:
        writer = get_stream_writer()
    except Exception:  # noqa: BLE001
        return
    if writer is not None:
        writer(event)


def clamp(value: Any, low: int, high: int, fallback: int) -> int:
    """Keep a model's number inside its scale.

    Range checking belongs in code rather than in the schema: structured
    outputs reject `minimum`/`maximum` on integers, and an estimate outside the
    scale should be corrected rather than trusted in any case.
    """
    try:
        return max(low, min(high, int(value)))
    except (TypeError, ValueError):
        return fallback


__all__ = ["clamp", "emit"]
