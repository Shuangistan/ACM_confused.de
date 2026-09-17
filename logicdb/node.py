"""The decision node: it routes, and the engine decides.

In a deployed system this position is held by many agents -- retrieval, document
parsing, tool calls, sub-task decomposition, cross-checking. This node abstracts
all of them, and the abstraction is faithful in the respect that matters:
everything they do is *upstream* work that produces facts and candidate logic.
None of it produces an answer.

That is the whole constraint, and it is worth stating as a table, because the
node is not a passive pipe. It makes plenty of decisions:

===========================================  ==========================================
the node decides                             the node never decides
===========================================  ==========================================
which pack and which query applies           what the answer is
how a raw record becomes candidate facts     which competing outcome wins
whether coverage is adequate                 what the probability is
whether to ask the agent for new logic       why the answer came out as it did
what oversight applies, and what to audit    whether the gate can be skipped
===========================================  ==========================================

The reason for the split is that an explanation is only worth having if it is
the *cause* of the decision rather than an account produced alongside it. A
model emitting a decision and a justification together produces two artefacts
with no mechanism binding them; the proof here is a transcript of the
computation that actually ran. The test is replay: keep the recorded facts and
the approved rules, unplug the model, and the same answer and the same proof
come back. Nothing in this module can fail that test, because nothing in it
computes an answer.

Iteration lives in `graph.py`. This module decides what happens to *one* case,
which is the unit the constraint above is about.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .agents.base import Agent
from .explain import Explainer
from .facts import FactStore
from .governance import BlockedForApproval, Governor
from .oversight import Assessment, OversightPolicy
from .packs import DomainPack
from .probability import Answer
from .program import RuleRecord


@dataclass
class Decision:
    """One case, decided or explicitly not decided."""

    entity: str
    answer: Answer | None
    assessment: Assessment | None
    facts: FactStore
    #: Set when the query was stopped because a rule it needs is unapproved.
    blocked_on: tuple[str, ...] = ()
    #: Set when the rule base has nothing bearing on the case.
    uncovered: bool = False
    rules_used: tuple[str, ...] = ()
    #: The recorded query. An audit verdict has to attach to something, and
    #: this is what `record_outcome` takes.
    query_id: str | None = None

    @property
    def took_effect(self) -> bool:
        """Whether this decision stands without a person having seen it."""
        return (
            self.answer is not None
            and not self.blocked_on
            and not self.uncovered
            and self.assessment is not None
            and self.assessment.mode.decides_automatically
            and not self.assessment.sampled
        )

    @property
    def needs_a_person(self) -> bool:
        return not self.took_effect


@dataclass


class DecisionNode:
    """Routes cases; never decides them."""

    def __init__(
        self,
        pack: DomainPack,
        governor: Governor,
        agent: Agent,
        oversight: OversightPolicy,
        explainer: Explainer | None = None,
        asked_by: str = "decision-node",
    ) -> None:
        self.pack = pack
        self.governor = governor
        self.agent = agent
        self.oversight = oversight
        self.explainer = explainer
        self.asked_by = asked_by

    # -- one case ----------------------------------------------------------
    def decide(self, record: dict[str, Any]) -> Decision:
        """Route one case through the engine and allocate oversight over it."""
        entity = str(record[self.pack.entity_field])
        facts = self.pack.facts_from(record, entity)
        try:
            answer, query = self.governor.ask(
                self.pack.goal_for(entity),
                facts,
                entity,
                self.asked_by,
                context=self.pack.context_from(record),
                competing=self.pack.competing_goals(entity),
            )
        except BlockedForApproval as blocked:
            return Decision(
                entity=entity, answer=None, assessment=None, facts=facts,
                blocked_on=tuple(r.rule_id for r in blocked.pending),
            )

        rules_used = tuple(sorted(answer.evaluation.rules_used))
        if answer.is_uncovered:
            return Decision(
                entity=entity, answer=answer, assessment=None, facts=facts,
                uncovered=True, rules_used=rules_used, query_id=query.query_id,
            )

        assessment = self.oversight.assess(
            entity=entity,
            answer=answer,
            decision_rules=self._records_for(rules_used),
            harm=int(self.pack.oversight.get("harm", 5)),
            reversibility=int(self.pack.oversight.get("reversibility", 5)),
            rights_impacted=self._rights_for(record),
        )
        return Decision(
            entity=entity, answer=answer, assessment=assessment,
            facts=facts, rules_used=rules_used, query_id=query.query_id,
        )

    def _rights_for(self, record: dict[str, Any]) -> bool:
        field_name = self.pack.oversight.get("rights_field")
        if not field_name:
            return bool(self.pack.oversight.get("rights_impacted", False))
        return bool(record.get(field_name, False))

    def _records_for(self, rule_ids: tuple[str, ...]) -> list[RuleRecord]:
        out = []
        for rule_id in rule_ids:
            record = self.governor.rulebase.get(rule_id)
            if record is not None:
                out.append(record)
        return out

__all__ = ["Decision", "DecisionNode"]
