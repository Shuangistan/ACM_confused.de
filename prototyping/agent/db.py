"""One store, shared by the graph and the API.

A module-level singleton rather than something passed through graph state: the
store owns a database connection, and a connection in checkpointed state would
be serialised on every turn.
"""

from __future__ import annotations

from pathlib import Path

from logic.store import Store

_store: Store | None = None
DB_PATH = Path(__file__).resolve().parents[1] / "data" / "logic.sqlite"


def store() -> Store:
    global _store
    if _store is None:
        _store = Store(DB_PATH)
    return _store


__all__ = ["DB_PATH", "store"]
