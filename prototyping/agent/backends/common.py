"""What both backends need, and neither owns.

`Reply` and `RefusalError` live here rather than in `llm.py` because the
backends are imported *by* `llm.py`; putting the shared types there would make
the import circular.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any


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


def env_or_dotenv(*names: str) -> str | None:
    """The first of `names` set in the environment, or found in a .env file.

    Two .env locations are searched: the repository root and `prototyping/`.
    Windows users who set the variable through System Properties get it from
    the environment; everyone else gets it from the file. Neither is preferred
    on principle -- the environment simply wins because it is cheaper to check.
    """
    for name in names:
        value = os.environ.get(name)
        if value:
            return value.strip()

    wanted = set(names)
    for candidate in (
        Path(__file__).resolve().parents[3] / ".env",
        Path(__file__).resolve().parents[2] / ".env",
    ):
        if not candidate.exists():
            continue
        # utf-8-sig: Windows tools (Notepad, some PowerShell redirections)
        # write a byte-order mark, and it would otherwise become part of the
        # first key's name, producing a "key not found" that looks like a typo.
        try:
            text = candidate.read_text(encoding="utf-8-sig")
        except OSError:
            continue
        for line in text.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            name, _, value = line.partition("=")
            if name.strip() in wanted:
                return value.strip().strip('"').strip("'")
    return None


__all__ = ["Reply", "RefusalError", "env_or_dotenv"]
