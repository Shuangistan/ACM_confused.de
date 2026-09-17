"""Terms, atoms, literals, rules — and when two rules are the same rule.

Function-free Horn clauses with negated body literals. First-order: predicates
carry arity, because a real case has structure inside it. A rule that must say
"not allergic to the drug about to be prescribed" relates two entities within
one case, and no propositional encoding of that survives contact with a second
drug.

Two things here are load-bearing.

**Canonical identity.** A rule approved once must never be re-reviewed, and that
is only possible if "the same rule" survives variable renaming and body
reordering. Without it an agent proposing `p(X) <- q(X), r(X)` today and
`p(A) <- r(A), q(A)` tomorrow costs two reviews for one idea.

**Safety.** A variable in the head, or inside a negation, that never appears in
a positive body literal ranges over nothing. Rejected at construction, because
an agent writing rules will produce them and the failure is far more confusing
three layers down.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Iterator


@dataclass(frozen=True)
class Var:
    name: str

    def __str__(self) -> str:
        return self.name


@dataclass(frozen=True)
class Const:
    value: str | int | float | bool

    def __str__(self) -> str:
        return f'"{self.value}"' if isinstance(self.value, str) else str(self.value)


Term = Var | Const
Binding = dict[str, Term]


@dataclass(frozen=True)
class Atom:
    predicate: str
    terms: tuple[Term, ...] = ()

    @property
    def arity(self) -> int:
        return len(self.terms)

    @property
    def signature(self) -> tuple[str, int]:
        return (self.predicate, self.arity)

    def is_ground(self) -> bool:
        return all(isinstance(t, Const) for t in self.terms)

    def variables(self) -> Iterator[Var]:
        for t in self.terms:
            if isinstance(t, Var):
                yield t

    def substitute(self, binding: Binding) -> "Atom":
        return Atom(self.predicate, tuple(
            binding.get(t.name, t) if isinstance(t, Var) else t for t in self.terms
        ))

    def __str__(self) -> str:
        if not self.terms:
            return self.predicate
        return f"{self.predicate}({', '.join(str(t) for t in self.terms)})"


@dataclass(frozen=True)
class Literal:
    atom: Atom
    negated: bool = False

    @property
    def predicate(self) -> str:
        return self.atom.predicate

    @property
    def signature(self) -> tuple[str, int]:
        return self.atom.signature

    def variables(self) -> Iterator[Var]:
        return self.atom.variables()

    def substitute(self, binding: Binding) -> "Literal":
        return Literal(self.atom.substitute(binding), self.negated)

    def __str__(self) -> str:
        return f"not {self.atom}" if self.negated else str(self.atom)


class UnsafeRule(ValueError):
    """A rule with a variable that ranges over nothing."""


@dataclass(frozen=True)
class Rule:
    head: Atom
    body: tuple[Literal, ...] = ()

    def __post_init__(self) -> None:
        positive = {v.name for l in self.body if not l.negated for v in l.variables()}
        unbound = {v.name for v in self.head.variables()} - positive
        if unbound:
            raise UnsafeRule(
                f"unsafe rule: head variable(s) {sorted(unbound)} in '{self}' "
                f"never appear in a positive body literal"
            )
        for lit in self.body:
            if not lit.negated:
                continue
            loose = {v.name for v in lit.variables()} - positive
            if loose:
                raise UnsafeRule(
                    f"unsafe rule: variable(s) {sorted(loose)} appear only inside "
                    f"the negated literal '{lit}' in '{self}'"
                )

    @property
    def positive_body(self) -> tuple[Literal, ...]:
        return tuple(l for l in self.body if not l.negated)

    @property
    def negative_body(self) -> tuple[Literal, ...]:
        return tuple(l for l in self.body if l.negated)

    def variables(self) -> set[str]:
        names = {v.name for v in self.head.variables()}
        for lit in self.body:
            names |= {v.name for v in lit.variables()}
        return names

    def substitute(self, binding: Binding) -> "Rule":
        return Rule(self.head.substitute(binding),
                    tuple(l.substitute(binding) for l in self.body))

    # -- identity ----------------------------------------------------------
    def canonical_form(self) -> str:
        """A normal form identical for every renaming and body ordering.

        Three passes, to break a circularity: ordering the literals wants the
        variable names settled, and naming the variables wants the literals
        ordered. So sort on a variable-blind sketch, assign indices walking that
        order, then re-sort with the real names in place — otherwise two
        literals of identical shape get ordered by chance.
        """
        def sketch(lit: Literal) -> tuple:
            slots = tuple(("v",) if isinstance(t, Var) else ("c", str(t))
                          for t in lit.atom.terms)
            return (lit.predicate, lit.negated, len(lit.atom.terms), slots)

        ordered = sorted(self.body, key=lambda l: (sketch(l), str(l)))
        renaming: dict[str, str] = {}

        def assign(atom: Atom) -> None:
            for t in atom.terms:
                if isinstance(t, Var) and t.name not in renaming:
                    renaming[t.name] = f"V{len(renaming)}"

        assign(self.head)                      # head first: readable indices
        for lit in ordered:
            assign(lit.atom)

        binding: Binding = {k: Var(v) for k, v in renaming.items()}
        head = self.head.substitute(binding)
        renamed = sorted((l.substitute(binding) for l in ordered), key=str)
        if not renamed:
            return f"{head}."
        return f"{head} <- {', '.join(str(l) for l in renamed)}."

    def canonical_key(self) -> str:
        """Stable identity for this rule's logical content, across processes."""
        return hashlib.sha256(self.canonical_form().encode()).hexdigest()[:16]

    def __str__(self) -> str:
        if not self.body:
            return f"{self.head}."
        return f"{self.head} <- {', '.join(str(l) for l in self.body)}."


def same_rule(a: Rule, b: Rule) -> bool:
    return a.canonical_key() == b.canonical_key()


__all__ = ["Atom", "Binding", "Const", "Literal", "Rule", "Term", "UnsafeRule",
           "Var", "same_rule"]
