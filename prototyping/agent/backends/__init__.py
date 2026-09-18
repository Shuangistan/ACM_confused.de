"""Provider backends. One module each, one interface between them.

Each backend exports the same six names, and `agent/llm.py` picks one:

    NAME            a string, for error messages and /api/models
    MODELS          tier key -> the provider's model id
    LABELS          tier key -> what the page should call it
    load_api_key()  -> str, raising with an actionable message if absent
    ask(...)        -> Reply
    stream(...)     -> Reply

The tier keys are `haiku`, `sonnet` and `opus` whichever provider is running.
That is a wart, and a deliberate one: they are the values already stored in
checkpointed state, in recorded cases and in the `proposed_by` column of every
rule the agent has ever drafted. Renaming them would make the existing database
unreadable to answer a naming complaint. Read them as cheap / middle / strong;
the page shows the provider's real model names, served from /api/models.
"""

from __future__ import annotations

from .common import Reply, RefusalError, env_or_dotenv

__all__ = ["Reply", "RefusalError", "env_or_dotenv"]
