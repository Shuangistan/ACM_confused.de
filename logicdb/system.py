"""Assembly: the pieces, wired together and handed over.

This module builds the object graph and owns nothing else. Control flow lives in
`graph.py`, where the edges are the topology rather than being implied by
statements spread across files; the components themselves live in `engine`,
`governance`, `oversight`, `checker` and `maintenance`.

Keeping assembly separate from orchestration is what lets the same components be
driven three ways without duplicating them: by the graph, by the FastAPI console
one case at a time, and by a test that constructs a `System` and calls a single
node directly.
"""

from __future__ import annotations

from typing import Any

from .agents.base import Agent
from .checker import Checker
from .explain import Explainer
from .governance import Governor, open_governor
from .maintenance import LogicMaintainer
from .node import DecisionNode
from .oversight import OversightPolicy
from .packs import DomainPack, load_pack


class System:
    """Everything, assembled. Construct with `System.build`."""

    def __init__(
        self,
        pack: DomainPack,
        governor: Governor,
        agent: Agent,
        oversight: OversightPolicy,
    ) -> None:
        self.pack = pack
        self.governor = governor
        self.agent = agent
        self.oversight = oversight
        self.explainer = Explainer(governor.rulebase, pack.vocabulary)
        self.checker = Checker(pack.vocabulary, pack.outcomes)
        self.node = DecisionNode(pack, governor, agent, oversight, self.explainer)
        self.maintainer = LogicMaintainer(pack, governor, self.checker)

    @classmethod
    def build(
        cls,
        pack_id: str,
        db_path,
        agent: Agent | None = None,
        oversight_pack: str = "oversight",
    ) -> "System":
        """The ordinary way in.

        The scripted agent is the default because the property being
        demonstrated is determinism, and a demonstration of determinism that
        needs an API key is not one.
        """
        from .agents.scripted import ScriptedAgent

        pack = load_pack(pack_id)
        governor = open_governor(db_path, pack.vocabulary, pack.criticality)
        return cls(
            pack=pack,
            governor=governor,
            agent=agent or ScriptedAgent(pack),
            oversight=OversightPolicy(load_pack(oversight_pack)),
        )

    def summary(self) -> dict[str, Any]:
        """Rule counts, queue depth, and whether the governance invariant holds."""
        counts: dict[str, int] = {}
        for record in self.governor.rulebase:
            counts[record.status.value] = counts.get(record.status.value, 0) + 1
        return {
            "rules": counts,
            "queue": len(self.governor.review_queue()),
            "invariant_violations": self.governor.verify_invariant(),
        }


__all__ = ["System"]
