"""AST for the rule language, plus the canonical hashing that makes approve-once work.

The language is function-free Horn clauses with stratified negation -- Datalog.
Chosen because it is decidable and guaranteed to terminate, which is what
actually delivers the determinism requirement, and because a Datalog derivation
*is* an explanation rather than something an explanation has to be reconstructed
from.

The subtle part of this module is `canonical_key`. The requirement is that once
a human approves a rule, the same rule never comes back for approval. Taken
naively -- hashing the printed text -- that fails immediately: the agent
proposes `decline(A) <- overdrawn(A), thin_file(A)`, a human approves it, and
next week the agent proposes `decline(X) <- thin_file(X), overdrawn(X)`, which
is the same rule wearing different clothes. Without canonicalisation "approve
once" quietly degrades into approving trivial variants forever, and the whole
amortisation argument collapses.

So two rules hash equal exactly when they are equal up to variable renaming and
body-literal order. Everything else -- probability, provenance, who approved it
-- is deliberately excluded from the key: those are facts *about* a rule, not
part of what the rule says. Re-mining the same logic with a slightly different
strength estimate must not re-open a settled question.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Iterable, Iterator, Union

# --------------------------------------------------------------------------
# Terms
# --------------------------------------------------------------------------

_VAR_RE = re.compile(r"^[A-Z_][A-Za-z0-9_]*$")
_NAME_RE = re.compile(r"^[a-z][A-Za-z0-9_]*$")


@dataclass(frozen=True, order=True)
class Var:
    """A universally quantified variable. Written with a leading capital."""

    name: str

    def __post_init__(self) -> None:
        if not _VAR_RE.match(self.name):
            raise ValueError(
                f"variable {self.name!r} must start with an uppercase letter or "
                f"underscore"
            )

    def __str__(self) -> str:
        return self.name


@dataclass(frozen=True, order=True)
class Const:
    """A ground value: an identifier, a number, or a quoted string."""

    value: Union[str, int, float, bool]

    def __str__(self) -> str:
        v = self.value
        if isinstance(v, bool):
            return "true" if v else "false"
        if isinstance(v, str) and not _NAME_RE.match(v):
            escaped = v.replace('"', '\\"')
            return f'"{escaped}"'
        return str(v)


Term = Union[Var, Const]


def is_ground(term: Term) -> bool:
    return isinstance(term, Const)


# --------------------------------------------------------------------------
# Atoms and literals
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Atom:
    """A predicate applied to terms, e.g. `overdrawn(A)` or `amount(A, 5000)`."""

    predicate: str
    terms: tuple[Term, ...] = ()

    def __post_init__(self) -> None:
        if not _NAME_RE.match(self.predicate):
            raise ValueError(
                f"predicate {self.predicate!r} must start with a lowercase letter"
            )

    @property
    def arity(self) -> int:
        return len(self.terms)

    @property
    def signature(self) -> str:
        return f"{self.predicate}/{self.arity}"

    def is_ground(self) -> bool:
        return all(is_ground(t) for t in self.terms)

    def variables(self) -> list[Var]:
        seen: dict[str, Var] = {}
        for t in self.terms:
            if isinstance(t, Var) and t.name not in seen:
                seen[t.name] = t
        return list(seen.values())

    def substitute(self, binding: dict[str, Term]) -> "Atom":
        return Atom(
            self.predicate,
            tuple(
                binding.get(t.name, t) if isinstance(t, Var) else t for t in self.terms
            ),
        )

    def __str__(self) -> str:
        if not self.terms:
            return self.predicate
        return f"{self.predicate}({', '.join(str(t) for t in self.terms)})"


@dataclass(frozen=True)
class Literal:
    """An atom, possibly negated. Negation is `not` in the surface syntax."""

    atom: Atom
    negated: bool = False

    @property
    def predicate(self) -> str:
        return self.atom.predicate

    @property
    def signature(self) -> str:
        return self.atom.signature

    def is_ground(self) -> bool:
        return self.atom.is_ground()

    def variables(self) -> list[Var]:
        return self.atom.variables()

    def substitute(self, binding: dict[str, Term]) -> "Literal":
        return Literal(self.atom.substitute(binding), self.negated)

    def __str__(self) -> str:
        return f"not {self.atom}" if self.negated else str(self.atom)


# --------------------------------------------------------------------------
# Rules
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Rule:
    """A Horn clause: `head <- body`, universally quantified over its variables.

    Universal quantification is not a technicality here -- it is the property the
    whole design rests on. One approved rule governs every case whose facts match
    its body, which is what makes reviewing rules cheaper than reviewing
    decisions, and what the reuse-factor metric counts.
    """

    head: Atom
    body: tuple[Literal, ...] = ()

    def __post_init__(self) -> None:
        self._check_safety()

    def _check_safety(self) -> None:
        """Reject unsafe rules -- the classic Datalog footgun.

        A variable in the head, or inside a negated body literal, that never
        appears in a positive body literal is unbound: there is no finite set of
        values to range over, so the rule would either derive infinitely many
        facts or silently derive none. Caught at construction because an agent
        proposing rules will produce these, and the failure is far more
        confusing later.
        """
        positive_vars = {
            v.name for lit in self.body if not lit.negated for v in lit.variables()
        }
        head_vars = {v.name for v in self.head.variables()}
        unbound_head = head_vars - positive_vars
        if unbound_head:
            raise ValueError(
                f"unsafe rule: head variable(s) {sorted(unbound_head)} in "
                f"'{self}' never appear in a positive body literal, so they range "
                f"over nothing"
            )
        for lit in self.body:
            if not lit.negated:
                continue
            unbound = {v.name for v in lit.variables()} - positive_vars
            if unbound:
                raise ValueError(
                    f"unsafe rule: variable(s) {sorted(unbound)} appear only inside "
                    f"the negated literal '{lit}' in '{self}'; negation needs a "
                    f"bound range to be meaningful"
                )

    @property
    def is_fact(self) -> bool:
        return not self.body

    def variables(self) -> list[Var]:
        seen: dict[str, Var] = {}
        for v in self.head.variables():
            seen.setdefault(v.name, v)
        for lit in self.body:
            for v in lit.variables():
                seen.setdefault(v.name, v)
        return list(seen.values())

    def substitute(self, binding: dict[str, Term]) -> "Rule":
        return Rule(
            self.head.substitute(binding),
            tuple(lit.substitute(binding) for lit in self.body),
        )

    def __str__(self) -> str:
        if self.is_fact:
            return f"{self.head}."
        return f"{self.head} <- {', '.join(str(l) for l in self.body)}."

    # -- canonicalisation ---------------------------------------------------
    def canonical_form(self) -> str:
        """A normal form identical for all variable renamings and body orders.

        The procedure has to break a circularity: to order the literals
        canonically we want variable names settled, and to name the variables
        canonically we want the literals ordered. It is resolved by first sorting
        literals on a variable-blind sketch (predicate, negation, and the shape
        of each argument slot), then walking that order to assign variable
        indices in order of first appearance, then re-sorting with the real
        names now in place.

        Rules that are genuinely alpha-equivalent converge on the same string.
        Rules that merely look similar do not.
        """

        def sketch(lit: Literal) -> tuple:
            slots = tuple(
                ("v",) if isinstance(t, Var) else ("c", str(t)) for t in lit.atom.terms
            )
            return (lit.predicate, lit.negated, len(lit.atom.terms), slots)

        ordered = sorted(self.body, key=lambda l: (sketch(l), str(l)))

        # Head first so the conclusion's variables get the lowest indices; that
        # keeps the canonical form readable when it is shown to a reviewer.
        renaming: dict[str, str] = {}

        def assign(atom: Atom) -> None:
            for t in atom.terms:
                if isinstance(t, Var) and t.name not in renaming:
                    renaming[t.name] = f"V{len(renaming)}"

        assign(self.head)
        for lit in ordered:
            assign(lit.atom)

        binding: dict[str, Term] = {k: Var(v) for k, v in renaming.items()}
        head = self.head.substitute(binding)
        renamed = [lit.substitute(binding) for lit in ordered]
        # Re-sort now that names are canonical, so two rules whose sketches tie
        # still land in the same order.
        renamed.sort(key=str)

        if not renamed:
            return f"{head}."
        return f"{head} <- {', '.join(str(l) for l in renamed)}."

    def canonical_key(self) -> str:
        """Stable 16-hex-char identity for this rule's logical content.

        Stable across processes and runs, unlike `hash()`. Used as the primary
        key for approval: approving this key approves the rule forever, in any
        syntactic dress it turns up wearing later.
        """
        return hashlib.sha256(self.canonical_form().encode("utf-8")).hexdigest()[:16]


def rules_equivalent(a: Rule, b: Rule) -> bool:
    """True when two rules say the same thing up to renaming and ordering."""
    return a.canonical_key() == b.canonical_key()


# --------------------------------------------------------------------------
# Predicate declarations -- the vocabulary an agent is allowed to use
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class PredicateDecl:
    """Declared signature for a predicate.

    Declaring the vocabulary up front is what stops a proposing agent -- an LLM
    especially -- from inventing predicates. A rule mentioning something
    undeclared is rejected at the door rather than entering the rule base and
    quietly never firing, which is the failure mode that would otherwise be
    nearly invisible.
    """

    name: str
    arity: int
    kind: str = "observable"   # "observable" | "derived"
    arg_types: tuple[str, ...] = ()
    description: str = ""
    #: Display template, e.g. "{0} has no guarantor". Used by the explanation
    #: layer so proofs read as sentences rather than as predicate soup.
    phrase: str = ""
    #: For observables: whether this fact can plausibly be missing in the real
    #: world. Drives which unknowns the value-of-information ranking bothers to
    #: consider asking about.
    askable: bool = True

    @property
    def signature(self) -> str:
        return f"{self.name}/{self.arity}"

    def __post_init__(self) -> None:
        if self.kind not in ("observable", "derived"):
            raise ValueError(
                f"predicate {self.name}: kind must be 'observable' or 'derived', "
                f"got {self.kind!r}"
            )
        if self.arg_types and len(self.arg_types) != self.arity:
            raise ValueError(
                f"predicate {self.name}: {len(self.arg_types)} arg_types declared "
                f"for arity {self.arity}"
            )


class Vocabulary:
    """The set of declared predicates, and validation against it."""

    def __init__(self, decls: Iterable[PredicateDecl]) -> None:
        self._by_sig: dict[str, PredicateDecl] = {}
        for d in decls:
            if d.signature in self._by_sig:
                raise ValueError(f"duplicate predicate declaration {d.signature}")
            self._by_sig[d.signature] = d

    def __contains__(self, signature: str) -> bool:
        return signature in self._by_sig

    def __iter__(self) -> Iterator[PredicateDecl]:
        return iter(self._by_sig.values())

    def __len__(self) -> int:
        return len(self._by_sig)

    def get(self, signature: str) -> PredicateDecl | None:
        return self._by_sig.get(signature)

    def observables(self) -> list[PredicateDecl]:
        return [d for d in self._by_sig.values() if d.kind == "observable"]

    def derived(self) -> list[PredicateDecl]:
        return [d for d in self._by_sig.values() if d.kind == "derived"]

    def validate_rule(self, rule: Rule) -> list[str]:
        """Return a list of problems, empty when the rule is acceptable."""
        problems: list[str] = []

        head_decl = self.get(rule.head.signature)
        if head_decl is None:
            problems.append(
                f"undeclared predicate in head: {rule.head.signature}. "
                f"Known: {sorted(self._by_sig)}"
            )
        elif head_decl.kind == "observable" and rule.body:
            # Observables come from the world; letting rules conclude them would
            # blur the line between what was measured and what was inferred, and
            # the provenance of every downstream fact with it.
            problems.append(
                f"{rule.head.signature} is declared observable, so it may be "
                f"asserted as a fact but not concluded by a rule"
            )

        for lit in rule.body:
            if lit.signature not in self._by_sig:
                problems.append(
                    f"undeclared predicate in body: {lit.signature}. "
                    f"Known: {sorted(self._by_sig)}"
                )
        return problems


__all__ = [
    "Atom",
    "Const",
    "Literal",
    "PredicateDecl",
    "Rule",
    "Term",
    "Var",
    "Vocabulary",
    "is_ground",
    "rules_equivalent",
]
