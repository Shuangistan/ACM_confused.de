"""Rules that almost fired, and what would have made them fire.

`derive` answers all-or-nothing: a rule's body is satisfied or the rule is
invisible. That throws away the most useful thing the rule base knows. A clause
needing three conditions with two already met is not a miss — it is a question,
and a precise one: *establish this one fact and the case is decided.*

Two uses follow from that.

To the person, it is the only question worth asking. Rather than a list of
everything unknown, one literal whose truth settles the matter.

To an expert, it is evidence about the rule itself. A clause that keeps coming
within one literal of firing, on a condition nobody can ever establish, is a
clause written against evidence that does not exist — and that is a reason to
revise it, visible only if near misses are counted.

Nothing here calls a model. It is a search over bindings, which is why it can
be trusted to say the same thing twice.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product

from .facts import FactStore, Truth
from .syntax import Atom, Binding, Const, Literal, Rule


@dataclass
class NearMiss:
    """A rule one or two literals short of deciding this case."""

    rule_id: str
    rule: Rule
    head: Atom
    #: What is satisfied already, and what is not.
    satisfied: tuple[Literal, ...]
    missing: tuple[Literal, ...]
    binding: Binding

    @property
    def distance(self) -> int:
        return len(self.missing)

    def question(self) -> str:
        """The missing literal, as something to go and find out."""
        if len(self.missing) != 1:
            return ""
        lit = self.missing[0]
        atom = lit.atom.substitute(self.binding)
        if lit.negated:
            return f"is it established that {atom} does NOT hold?"
        return f"is {atom} the case?"


def _constants(facts: FactStore) -> list[Const]:
    seen: dict[str, Const] = {}
    for record in facts:
        for term in record.atom.terms:
            if isinstance(term, Const):
                seen.setdefault(str(term), term)
    return list(seen.values())


def _satisfied(lit: Literal, binding: Binding, facts: FactStore,
               derived: set[Atom]) -> bool:
    """Whether this literal holds, under the three-valued reading.

    A negated literal needs the atom *recorded* false. Unmentioned is unknown,
    and unknown does not satisfy a negation — the same rule the engine follows,
    for the same reason.
    """
    atom = lit.atom.substitute(binding)
    if not atom.is_ground():
        return False
    truth = Truth.TRUE if atom in derived else facts.truth_of(atom)
    return truth is (Truth.FALSE if lit.negated else Truth.TRUE)


def near_misses(rules: list[tuple[str, Rule]], facts: FactStore,
                derived: set[Atom] | None = None, max_missing: int = 2,
                ) -> list[NearMiss]:
    """Every rule within `max_missing` literals of firing, best binding first.

    Bindings are enumerated over the constants the case mentions. That is a
    small set — one, in most cases here — so the search is cheap and complete
    rather than heuristic.
    """
    derived = derived or set()
    constants = _constants(facts)
    out: list[NearMiss] = []

    for rule_id, rule in rules:
        if rule.head in derived:
            continue                     # already concluded; not a miss
        variables = sorted(rule.variables())
        candidates = (
            [dict(zip(variables, combo))
             for combo in product(constants, repeat=len(variables))]
            if variables else [{}]
        )
        best: NearMiss | None = None
        for binding in candidates:
            if rule.head.substitute(binding) in derived:
                continue
            sat, miss = [], []
            for lit in rule.body:
                (sat if _satisfied(lit, binding, facts, derived) else miss).append(lit)
            if not miss or len(miss) > max_missing:
                continue
            # A binding that satisfies nothing is not a near miss, it is an
            # unrelated rule sharing a variable.
            if not sat:
                continue
            candidate = NearMiss(rule_id, rule, rule.head.substitute(binding),
                                 tuple(sat), tuple(miss), binding)
            if best is None or candidate.distance < best.distance:
                best = candidate
        if best is not None:
            out.append(best)

    out.sort(key=lambda n: (n.distance, -len(n.satisfied), str(n.head)))
    return out


__all__ = ["NearMiss", "near_misses"]
