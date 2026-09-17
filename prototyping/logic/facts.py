"""Three-valued facts. The third value is the whole point.

A fact is TRUE, FALSE, or UNKNOWN, and **UNKNOWN is not absence**. It is a
recorded claim that nobody knows, which is a different thing from a claim that
the answer is no.

The reason this matters more here than in a textbook is that facts arrive from a
model reading free text. If the extractor did not mention `medical_evidence`,
that does not mean there is none — it means nobody said. Under a closed world
that silence becomes `false`, `not medical_evidence` succeeds, and a rule fires
against a person on evidence that was never gathered. We shipped exactly that
defect twice in the previous build.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Iterator

from .syntax import Atom


class Truth(str, Enum):
    TRUE = "true"
    FALSE = "false"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class FactRecord:
    atom: Atom
    truth: Truth = Truth.TRUE
    #: Where it came from — a quoted span, a form field, a person's name. What
    #: an explanation points at when someone asks why this was believed.
    source: str = "unstated"

    def __post_init__(self) -> None:
        if not self.atom.is_ground():
            raise ValueError(f"facts must be ground: {self.atom}")


class FactStore:
    """The ground facts of one case."""

    def __init__(self, records: list[FactRecord] | None = None) -> None:
        self._by_atom: dict[Atom, FactRecord] = {}
        for record in records or []:
            self.assert_fact(record.atom, record.truth, record.source)

    def assert_fact(self, atom: Atom, truth: Truth = Truth.TRUE,
                    source: str = "unstated") -> FactRecord:
        record = FactRecord(atom, truth, source)
        self._by_atom[atom] = record
        return record

    def truth_of(self, atom: Atom) -> Truth:
        """Absent means UNKNOWN. Never FALSE."""
        record = self._by_atom.get(atom)
        return record.truth if record else Truth.UNKNOWN

    def get(self, atom: Atom) -> FactRecord | None:
        return self._by_atom.get(atom)

    def atoms(self, truth: Truth | None = None) -> list[Atom]:
        return [a for a, r in self._by_atom.items()
                if truth is None or r.truth is truth]

    def __iter__(self) -> Iterator[FactRecord]:
        return iter(self._by_atom.values())

    def __len__(self) -> int:
        return len(self._by_atom)

    def __contains__(self, atom: Atom) -> bool:
        return atom in self._by_atom


__all__ = ["FactRecord", "FactStore", "Truth"]
