"""The Agent interface: what an agent may do, and what it may not.

The single architectural commitment of this project lives in this file. An agent
**proposes**; only the engine **infers**. An agent may extract candidate facts
from a record and draft candidate rules from data or documents. It may not
compute an answer, and nothing it produces reaches a decision without passing
through the approval path in `governance.py`.

This is what makes the determinism and explainability claims survive contact
with an LLM. The LLM is nondeterministic and cannot be made otherwise, so
instead of pretending, its nondeterminism is quarantined upstream of a human
gate. Everything downstream of that gate -- the inference, the probability, the
proof -- is reproducible, and a hallucinated predicate is rejected mechanically
before a reviewer ever sees it.

Both implementations honour the same contract, so the scripted agent is a
genuine substitute rather than a mock: swapping `ScriptedAgent` for `LLMAgent`
changes where proposals come from and nothing about how they are governed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from ..facts import FactStore
from ..program import RuleOrigin
from ..syntax import Rule, Vocabulary


@dataclass
class RuleProposal:
    """A candidate rule, with the evidence a reviewer needs to judge it.

    The evidence fields are not decoration. A mined rule is a correlation until
    somebody vouches for it, and a reviewer shown only the clause has no basis
    to vouch. Support and precision let them see how much data stands behind it;
    a citation lets them check it against written authority.
    """

    rule: Rule
    origin: RuleOrigin
    strength: float
    rationale: str
    proposed_by: str = "agent"
    source_citation: str | None = None
    support: int | None = None
    precision: float | None = None
    #: Entities the rule would have fired on, so a reviewer can spot-check.
    examples: tuple[str, ...] = ()
    counterexamples: tuple[str, ...] = ()

    def describe(self) -> str:
        bits = [f"{self.rule}", f"  origin: {self.origin.value}",
                f"  strength: {self.strength:.2f}", f"  because: {self.rationale}"]
        if self.support is not None:
            bits.append(
                f"  evidence: fires on {self.support} cases, correct on "
                f"{self.precision:.0%} of them"
            )
        if self.source_citation:
            bits.append(f"  cites: {self.source_citation}")
        if self.examples:
            bits.append(f"  examples: {', '.join(self.examples[:5])}")
        if self.counterexamples:
            bits.append(f"  counterexamples: {', '.join(self.counterexamples[:5])}")
        return "\n".join(bits)


@dataclass
class LabelledCase:
    """One record with a known outcome, used as evidence for mining."""

    entity: str
    facts: FactStore
    label: str
    raw: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class Agent(Protocol):
    """Proposes facts and rules. Never decides anything."""

    @property
    def agent_id(self) -> str: ...

    def extract_facts(
        self, record: dict[str, Any], entity: str
    ) -> FactStore:
        """Turn a raw record into ground facts."""
        ...

    def mine_rules(
        self, cases: list[LabelledCase], target: str, existing: set[str] | None = None
    ) -> list[RuleProposal]:
        """Induce candidate rules from cases whose outcome is known."""
        ...

    def extract_from_document(
        self, name: str, text: str, vocabulary: Vocabulary
    ) -> list[RuleProposal]:
        """Draft candidate rules from a written policy or regulation."""
        ...


def validate_proposals(
    proposals: list[RuleProposal], vocabulary: Vocabulary
) -> tuple[list[RuleProposal], list[tuple[RuleProposal, list[str]]]]:
    """Split proposals into acceptable and malformed.

    Run before anything is queued, so a proposal referencing an invented
    predicate is discarded mechanically rather than consuming a reviewer's
    attention. With an LLM backend this is the load-bearing filter: it is the
    reason a hallucinated predicate cannot enter the rule base.
    """
    good: list[RuleProposal] = []
    bad: list[tuple[RuleProposal, list[str]]] = []
    for proposal in proposals:
        problems = vocabulary.validate_rule(proposal.rule)
        if problems:
            bad.append((proposal, problems))
        else:
            good.append(proposal)
    return good, bad


__all__ = ["Agent", "LabelledCase", "RuleProposal", "validate_proposals"]
