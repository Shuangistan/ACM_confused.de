"""The rule base: rule records, their lifecycle, and stratification.

Two jobs.

**Lifecycle.** A rule is not just a clause, it is a clause with a governance
history: who proposed it, where it came from, whether a human has approved it,
how often it has been used since, how often it was right. The lifecycle states
implement the user's requirement directly -- critical questions checked online,
non-critical checked later offline, and once approved never re-reviewed.

**Stratification.** Negation in Datalog is only well-defined if the program is
stratified: no predicate may depend negatively on itself, directly or through a
cycle. An unstratified program has no unique minimal model, so "deterministic"
would be a lie. Because rules arrive from an agent over time, this must be
checked on every insertion rather than once at startup, and the rejection
message has to name the offending cycle or nobody can fix it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Iterable, Iterator

from .syntax import Rule, Vocabulary


class RuleStatus(str, Enum):
    """Where a rule sits in the approval lifecycle.

    The states are not decorative -- `RuleBase.active_for` refuses to hand a
    non-approved rule to a critical query, which is what makes the governance
    invariant hold in the engine rather than by convention.
    """

    PROPOSED = "proposed"
    """Created by an agent, usable nowhere yet."""

    PENDING_ONLINE = "pending_online"
    """Needed by a critical query, which is blocked until a human rules on it."""

    PROVISIONAL = "provisional"
    """Usable for non-critical queries now; queued for offline review.
    Every decision made with it is tagged, so the exposure is enumerable."""

    APPROVED = "approved"
    """Signed off by a named human. Permanent -- never re-reviewed."""

    REJECTED = "rejected"
    """Refused. Never usable. Triggers impact analysis over past provisional use."""

    RETIRED = "retired"
    """Superseded. Past decisions keep the version they actually used."""

    @property
    def usable_non_critical(self) -> bool:
        return self in (RuleStatus.PROVISIONAL, RuleStatus.APPROVED)

    @property
    def usable_critical(self) -> bool:
        return self is RuleStatus.APPROVED


class RuleOrigin(str, Enum):
    """Where a rule came from. Drives how hard a reviewer should look at it."""

    DOCUMENT = "document"
    """Extracted from a regulation or policy text, citing its clause. Checkable
    against written authority, so the reviewer's job is verification."""

    MINED = "mined"
    """Induced from labelled cases. A correlation until a human vouches for it;
    the reviewer's job is to decide whether it is also a reason."""

    HUMAN = "human"
    """Hand-authored. Trusted, but still versioned and attributable."""

    SEED = "seed"
    """Shipped with the pack as a starting point."""


@dataclass
class RuleStats:
    """Usage and outcome tallies.

    Kept as raw counts rather than a fitted score so a reviewer can see the
    evidence -- "right 41 of 47 times" is auditable in a way that "strength
    0.87" is not.
    """

    #: Distinct queries this rule has contributed to since approval. This is the
    #: amortisation number: one review, this many decisions governed.
    reuse_count: int = 0
    #: Queries where the rule fired and the outcome was later confirmed correct.
    successes: int = 0
    #: ... and incorrect.
    failures: int = 0
    last_used_at: str | None = None

    @property
    def observations(self) -> int:
        return self.successes + self.failures

    def posterior_mean(self, prior_alpha: float = 1.0, prior_beta: float = 1.0) -> float:
        """Beta-Binomial posterior mean for this rule's strength.

        Uniform prior by default, so a rule with no observations sits at 0.5 and
        earns its way up or down. Deliberately not the raw success ratio, which
        would read 1.0 after a single lucky case.
        """
        a = prior_alpha + self.successes
        b = prior_beta + self.failures
        return a / (a + b)


@dataclass
class RuleRecord:
    """A rule plus everything governance needs to know about it."""

    rule: Rule
    rule_id: str
    #: Author-supplied or learned strength: P(head | body). Distinct from
    #: `stats.posterior_mean`, which is what the evidence says. Both are shown
    #: to reviewers; divergence between them is itself informative.
    strength: float = 0.8
    status: RuleStatus = RuleStatus.PROPOSED
    origin: RuleOrigin = RuleOrigin.HUMAN
    proposed_by: str = "unknown"
    proposed_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    approved_by: str | None = None
    approved_at: str | None = None
    rejected_reason: str | None = None
    #: For DOCUMENT origin: the clause this encodes, quoted.
    source_citation: str | None = None
    #: For MINED origin: the evidence behind the proposal, so a reviewer sees
    #: support and precision rather than a bare assertion.
    mining_support: int | None = None
    mining_precision: float | None = None
    supersedes: str | None = None
    version: int = 1
    stats: RuleStats = field(default_factory=RuleStats)
    notes: str = ""

    def __post_init__(self) -> None:
        if not 0.0 < self.strength <= 1.0:
            raise ValueError(
                f"rule {self.rule_id}: strength must be in (0, 1], got "
                f"{self.strength}. A strength of 0 is not a weak rule, it is the "
                f"absence of one."
            )

    @property
    def key(self) -> str:
        """Canonical identity. Approving this approves the logic, not the text."""
        return self.rule.canonical_key()

    @property
    def head_signature(self) -> str:
        return self.rule.head.signature

    def effective_strength(self, use_posterior: bool = False) -> float:
        if use_posterior and self.stats.observations > 0:
            return self.stats.posterior_mean()
        return self.strength

    def approve(self, by: str, at: str | None = None) -> None:
        if self.status is RuleStatus.REJECTED:
            raise ValueError(
                f"rule {self.rule_id} was rejected; propose it again rather than "
                f"approving a refusal in place, so the history stays legible"
            )
        self.status = RuleStatus.APPROVED
        self.approved_by = by
        self.approved_at = at or datetime.now(timezone.utc).isoformat()

    def reject(self, by: str, reason: str, at: str | None = None) -> None:
        self.status = RuleStatus.REJECTED
        self.approved_by = by
        self.approved_at = at or datetime.now(timezone.utc).isoformat()
        self.rejected_reason = reason

    def describe(self) -> str:
        bits = [f"{self.rule_id} [{self.status.value}, {self.origin.value}]"]
        bits.append(f"  {self.rule}")
        bits.append(f"  strength {self.strength:.2f}")
        if self.stats.observations:
            bits[-1] += (
                f"  (evidence: right {self.stats.successes}/"
                f"{self.stats.observations}, posterior "
                f"{self.stats.posterior_mean():.2f})"
            )
        if self.approved_by:
            bits.append(f"  approved by {self.approved_by} at {self.approved_at}")
        if self.source_citation:
            bits.append(f"  cites: {self.source_citation}")
        if self.mining_support is not None:
            bits.append(
                f"  mined: support {self.mining_support}, precision "
                f"{self.mining_precision:.2f}"
            )
        if self.stats.reuse_count:
            bits.append(f"  reused across {self.stats.reuse_count} queries")
        return "\n".join(bits)


class StratificationError(ValueError):
    """The rule base cannot be stratified, so it has no unique model."""


class RuleBase:
    """The logic database: rules, indexed, versioned and stratified."""

    def __init__(self, vocabulary: Vocabulary | None = None) -> None:
        self.vocabulary = vocabulary
        self._records: dict[str, RuleRecord] = {}        # rule_id -> record
        self._by_key: dict[str, str] = {}                # canonical key -> rule_id
        self._by_head: dict[str, list[str]] = {}         # head signature -> rule_ids
        self._strata: dict[str, int] | None = None
        self._next_id = 1

    # -- insertion ---------------------------------------------------------
    def next_rule_id(self) -> str:
        while f"r{self._next_id:04d}" in self._records:
            self._next_id += 1
        return f"r{self._next_id:04d}"

    def find_by_key(self, key: str) -> RuleRecord | None:
        """The approve-once lookup.

        Called before anything is queued for review: if this logic has already
        been ruled on, in any syntactic form, the answer stands and no human is
        asked twice.
        """
        rule_id = self._by_key.get(key)
        return self._records.get(rule_id) if rule_id else None

    def add(self, record: RuleRecord, check_stratification: bool = True) -> RuleRecord:
        """Insert a rule, rejecting it if it breaks vocabulary or stratification.

        Returns the *existing* record when the same logic is already present, so
        callers get approve-once behaviour without having to remember to check.
        """
        if self.vocabulary is not None:
            problems = self.vocabulary.validate_rule(record.rule)
            if problems:
                raise ValueError(
                    f"rule {record.rule_id} rejected:\n  "
                    + "\n  ".join(problems)
                )

        existing = self.find_by_key(record.key)
        if existing is not None:
            return existing

        if check_stratification:
            self._assert_stratifiable_with(record)

        self._records[record.rule_id] = record
        self._by_key[record.key] = record.rule_id
        self._by_head.setdefault(record.head_signature, []).append(record.rule_id)
        self._strata = None
        return record

    def remove(self, rule_id: str) -> None:
        record = self._records.pop(rule_id, None)
        if record is None:
            return
        self._by_key.pop(record.key, None)
        heads = self._by_head.get(record.head_signature, [])
        if rule_id in heads:
            heads.remove(rule_id)
        self._strata = None

    # -- lookup ------------------------------------------------------------
    def get(self, rule_id: str) -> RuleRecord | None:
        return self._records.get(rule_id)

    def __len__(self) -> int:
        return len(self._records)

    def __iter__(self) -> Iterator[RuleRecord]:
        return iter(self._records.values())

    def all_records(self) -> list[RuleRecord]:
        return list(self._records.values())

    def by_status(self, *statuses: RuleStatus) -> list[RuleRecord]:
        wanted = set(statuses)
        return [r for r in self._records.values() if r.status in wanted]

    def review_queue(self) -> list[RuleRecord]:
        """What a human still owes a decision on, most urgent first.

        Blocking reviews come first because a query is stalled on each one; the
        provisional queue can wait, which is the whole point of splitting them.
        """
        pending = self.by_status(RuleStatus.PENDING_ONLINE)
        provisional = self.by_status(RuleStatus.PROVISIONAL)
        pending.sort(key=lambda r: r.proposed_at)
        provisional.sort(key=lambda r: -r.stats.reuse_count)
        return pending + provisional

    def active_for(self, critical: bool) -> list[RuleRecord]:
        """Rules usable for a query of this criticality.

        The one place the governance guarantee is enforced. A critical query
        sees only approved rules, so a critical decision resting on unreviewed
        logic is not something the system declines to do -- it is something it
        cannot do.
        """
        if critical:
            return [r for r in self._records.values() if r.status.usable_critical]
        return [r for r in self._records.values() if r.status.usable_non_critical]


    # -- stratification ----------------------------------------------------
    def dependency_edges(
        self, records: Iterable[RuleRecord] | None = None
    ) -> list[tuple[str, str, bool]]:
        """(body_predicate, head_predicate, is_negative) over the given rules."""
        edges: list[tuple[str, str, bool]] = []
        for rec in (records if records is not None else self._records.values()):
            head = rec.head_signature
            for lit in rec.rule.body:
                edges.append((lit.signature, head, lit.negated))
        return edges

    def strata(self, records: Iterable[RuleRecord] | None = None) -> dict[str, int]:
        """Assign each predicate a stratum number.

        Standard iterative assignment: a positive dependency requires the body
        predicate's stratum to be no higher than the head's, a negative one
        requires it to be strictly lower. Iterating to a fixpoint either
        converges or proves no stratification exists.
        """
        if records is None and self._strata is not None:
            return self._strata

        pool = list(records) if records is not None else list(self._records.values())
        edges = self.dependency_edges(pool)
        predicates = {e[0] for e in edges} | {e[1] for e in edges}
        for rec in pool:
            predicates.add(rec.head_signature)

        level = {p: 0 for p in predicates}
        # A stratum cannot exceed the predicate count; exceeding it means a
        # negative cycle is pumping levels forever.
        for _ in range(len(predicates) + 1):
            changed = False
            for body, head, negated in edges:
                required = level[body] + 1 if negated else level[body]
                if required > level[head]:
                    level[head] = required
                    changed = True
            if not changed:
                if records is None:
                    self._strata = level
                return level

        raise StratificationError(
            "rule base is not stratifiable: "
            + _describe_negative_cycle(edges)
            + "\nA predicate that depends negatively on itself has no unique "
              "minimal model, so the answer would depend on evaluation order "
              "rather than on the logic."
        )

    def _assert_stratifiable_with(self, record: RuleRecord) -> None:
        candidate = list(self._records.values()) + [record]
        try:
            self.strata(candidate)
        except StratificationError as exc:
            raise StratificationError(
                f"adding rule {record.rule_id} ({record.rule}) would break "
                f"stratification.\n{exc}"
            ) from None


    # -- reporting ---------------------------------------------------------
def _describe_negative_cycle(edges: list[tuple[str, str, bool]]) -> str:
    """Find and render a cycle containing a negative edge, for the error message.

    Worth the effort: "not stratifiable" is unactionable, whereas naming the
    loop tells whoever proposed the rule exactly what to change.
    """
    adjacency: dict[str, list[tuple[str, bool]]] = {}
    for body, head, negated in edges:
        adjacency.setdefault(body, []).append((head, negated))

    for start in adjacency:
        stack = [(start, [start], False)]
        seen: set[tuple[str, bool]] = set()
        while stack:
            node, path, saw_negative = stack.pop()
            for nxt, negated in adjacency.get(node, ()):
                negative_now = saw_negative or negated
                if nxt == start and negative_now:
                    loop = " -> ".join(path + [start])
                    return f"negative cycle through {loop}"
                state = (nxt, negative_now)
                if state in seen or len(path) > len(adjacency) + 1:
                    continue
                seen.add(state)
                stack.append((nxt, path + [nxt], negative_now))
    return "a predicate depends negatively on itself"


__all__ = [
    "RuleBase",
    "RuleOrigin",
    "RuleRecord",
    "RuleStats",
    "RuleStatus",
    "StratificationError",
]
