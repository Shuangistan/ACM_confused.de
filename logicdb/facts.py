"""The fact store: ground atoms with three-valued truth and per-fact provenance.

`UNKNOWN` is the reason this module exists rather than being a dict of atoms.

Most rule systems treat a missing fact as false -- the closed-world assumption.
For a decision system that is a quiet disaster: "no guarantor was recorded"
becomes "there is no guarantor", and a decision gets made on an assumption
nobody ever stated. The user's requirement was explicit that critical
information may be missing, so absence and falsity are kept apart:

* `TRUE`    -- asserted, with a source.
* `FALSE`   -- asserted as not holding, with a source. A real claim.
* `UNKNOWN` -- explicitly recorded as not known. Also a real claim, and the one
               that makes the answer an interval rather than a number.
* absent    -- never considered at all. Treated as UNKNOWN when the predicate is
               declared askable, so a field nobody filled in cannot silently
               become a negative.

Every fact carries provenance, because an explanation that bottoms out in
`overdrawn(app_17)` with no account of where that came from has not actually
explained anything.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Iterable, Iterator

from .syntax import Atom, Const, Term, Vocabulary


class Truth(str, Enum):
    TRUE = "true"
    FALSE = "false"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class FactRecord:
    """One ground atom and what is known about it."""

    atom: Atom
    truth: Truth = Truth.TRUE
    #: Where this came from: a form field, a document clause, an agent
    #: extraction, a human correction. Free text, but never empty in practice.
    source: str = "unspecified"
    #: Who or what asserted it.
    asserted_by: str = "system"
    #: Confidence in the assertion itself, distinct from rule strength. An
    #: extraction the agent is unsure of is not the same as a rule that is
    #: probabilistic.
    confidence: float = 1.0

    def __post_init__(self) -> None:
        if not self.atom.is_ground():
            raise ValueError(
                f"facts must be ground; {self.atom} contains variables. "
                f"A non-ground assertion is a rule, not a fact."
            )
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"confidence must be in [0, 1], got {self.confidence}")

    def __str__(self) -> str:
        return f"{self.atom} = {self.truth.value}  [{self.source}]"


class FactStore:
    """Ground facts for one query, indexed for matching."""

    def __init__(self, vocabulary: Vocabulary | None = None) -> None:
        self.vocabulary = vocabulary
        self._facts: dict[Atom, FactRecord] = {}
        #: predicate signature -> atoms, so the engine can enumerate candidate
        #: bindings without scanning everything.
        self._by_signature: dict[str, set[Atom]] = {}

    # -- writing -----------------------------------------------------------
    def assert_fact(
        self,
        atom: Atom,
        truth: Truth = Truth.TRUE,
        source: str = "unspecified",
        asserted_by: str = "system",
        confidence: float = 1.0,
    ) -> FactRecord:
        if self.vocabulary is not None:
            decl = self.vocabulary.get(atom.signature)
            if decl is None:
                raise ValueError(
                    f"undeclared predicate {atom.signature}; the vocabulary is "
                    f"fixed so that a proposing agent cannot invent one"
                )
            if decl.kind == "derived":
                raise ValueError(
                    f"{atom.signature} is declared derived, so it is concluded by "
                    f"rules and cannot be asserted as a fact. Asserting it would "
                    f"erase the distinction between what was observed and what "
                    f"was inferred."
                )
        record = FactRecord(atom, truth, source, asserted_by, confidence)
        self._facts[atom] = record
        self._by_signature.setdefault(atom.signature, set()).add(atom)
        return record

    def add(self, record: FactRecord) -> None:
        self._facts[record.atom] = record
        self._by_signature.setdefault(record.atom.signature, set()).add(record.atom)

    def extend(self, records: Iterable[FactRecord]) -> None:
        for r in records:
            self.add(r)


    # -- reading -----------------------------------------------------------
    def truth_of(self, atom: Atom) -> Truth:
        """Truth of a ground atom, with absence handled deliberately.

        An atom nobody recorded is UNKNOWN when its predicate is askable, and
        FALSE otherwise. The distinction matters: "we never asked about the
        guarantor" is genuinely unknown and worth asking about, whereas a
        derived bookkeeping predicate that simply did not fire is false.
        """
        record = self._facts.get(atom)
        if record is not None:
            return record.truth
        if self.vocabulary is not None:
            decl = self.vocabulary.get(atom.signature)
            if decl is not None and decl.kind == "observable" and decl.askable:
                return Truth.UNKNOWN
        return Truth.FALSE

    def record_of(self, atom: Atom) -> FactRecord | None:
        return self._facts.get(atom)

    def atoms_for(self, signature: str) -> set[Atom]:
        return set(self._by_signature.get(signature, ()))

    def unknowns(self) -> list[Atom]:
        """Atoms explicitly recorded as unknown.

        These are what the value-of-information ranking considers asking about.
        Deliberately excludes atoms that were never mentioned: the universe of
        atoms nobody has named is unbounded, and a system that offered to ask
        about all of them would be useless.
        """
        return [a for a, r in self._facts.items() if r.truth is Truth.UNKNOWN]

    def with_assignment(self, assignment: dict[Atom, Truth]) -> "FactStore":
        """A copy with some atoms forced to given truth values.

        This is how the probability layer computes bounds: evaluate the same
        program with the unknowns pinned one way, then the other. Copy rather
        than mutate, so a bounds computation can never corrupt the real facts.
        """
        clone = FactStore(self.vocabulary)
        clone._facts = dict(self._facts)
        clone._by_signature = {k: set(v) for k, v in self._by_signature.items()}
        for atom, truth in assignment.items():
            existing = clone._facts.get(atom)
            clone.add(
                FactRecord(
                    atom=atom,
                    truth=truth,
                    source=f"assumed {truth.value}"
                    + (f" (was {existing.truth.value})" if existing else ""),
                    asserted_by="bounds-analysis",
                    confidence=existing.confidence if existing else 1.0,
                )
            )
        return clone


    # -- dunders -----------------------------------------------------------
    def __len__(self) -> int:
        return len(self._facts)

    def __contains__(self, atom: Atom) -> bool:
        return atom in self._facts

    def __iter__(self) -> Iterator[FactRecord]:
        return iter(self._facts.values())

    def __repr__(self) -> str:
        counts: dict[str, int] = {}
        for r in self._facts.values():
            counts[r.truth.value] = counts.get(r.truth.value, 0) + 1
        return f"FactStore({len(self._facts)} facts: {counts})"


__all__ = ["FactRecord", "FactStore", "Truth"]
