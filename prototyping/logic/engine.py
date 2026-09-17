"""Forward chaining to a fixpoint, with the derivation kept.

The engine is deliberately the least clever part of the system. It has no
function symbols, so the set of derivable atoms is finite and termination is
structural rather than a timeout. Rules are matched against a snapshot taken at
the start of each round, so a rule's result cannot depend on where in the round
it ran, and inserting the same rules in a different order cannot change the
answer.

Forward rather than backward, for one reason: the whole derivation exists when
it finishes, so the explanation is a transcript of what actually happened rather
than an account reconstructed afterwards. That distinction is the difference
between a proof and a story.

**Unknowns survive negation.** If a body literal is UNKNOWN, the rule does not
fire — and its head is recorded as *blocked* rather than left absent, so any
rule reading `not head` sees UNKNOWN too. Without that, ignorance silently
becomes evidence one round later. This was the defect we shipped twice.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .facts import FactStore, Truth
from .syntax import Atom, Binding, Rule, Var


class StratificationError(ValueError):
    """Negation inside a recursive cycle: the program has no single meaning."""


@dataclass(frozen=True)
class Support:
    """One reason to believe an atom: a rule, and the premises it used."""

    rule_id: str
    premises: tuple[Atom, ...]
    negated_premises: tuple[Atom, ...] = ()


@dataclass
class Derived:
    atom: Atom
    supports: list[Support] = field(default_factory=list)


@dataclass
class Derivation:
    """What the engine concluded, and how."""

    derived: dict[Atom, Derived]
    #: Heads that a rule would have derived but for an UNKNOWN premise. Their
    #: truth is UNKNOWN, not FALSE — the distinction the whole file exists for.
    blocked: set[Atom]
    facts: FactStore
    rounds: int
    rules_used: set[str]
    #: Signatures that some rule concludes. An atom of such a predicate that was
    #: not derived is false by exhaustion; an atom of any other predicate was
    #: simply never mentioned, and is unknown.
    head_signatures: frozenset = frozenset()

    def holds(self, atom: Atom) -> bool:
        return atom in self.derived or self.facts.truth_of(atom) is Truth.TRUE

    def truth_of(self, atom: Atom) -> Truth:
        if atom in self.derived:
            return Truth.TRUE
        if atom in self.blocked:
            return Truth.UNKNOWN
        fact = self.facts.truth_of(atom)
        if fact is not Truth.UNKNOWN:
            return fact
        # An underivable atom of a derived predicate is false by exhaustion;
        # an unmentioned observable is simply unknown.
        return (Truth.FALSE if atom.signature in self.head_signatures
                else Truth.UNKNOWN)

    def conclusions(self, predicates: set[str] | None = None) -> list[Atom]:
        return sorted(
            (a for a in self.derived
             if predicates is None or a.predicate in predicates),
            key=str,
        )


# --------------------------------------------------------------------------
# Stratification
# --------------------------------------------------------------------------
def stratify(rules: list[tuple[str, Rule]]) -> list[list[tuple[str, Rule]]]:
    """Order the rules so every negated premise is settled before it is read.

    Re-run on every insertion rather than once at startup, because rules arrive
    from an agent over time and a recursive rule it proposes must not be able to
    give the program two meanings.
    """
    heads = {rule.head.signature for _, rule in rules}
    level: dict[tuple[str, int], int] = {sig: 0 for sig in heads}

    for _ in range(len(heads) + 1):
        changed = False
        for _, rule in rules:
            head = rule.head.signature
            for lit in rule.body:
                if lit.signature not in heads:
                    continue                      # an input, already settled
                need = level[lit.signature] + (1 if lit.negated else 0)
                if need > level[head]:
                    level[head] = need
                    changed = True
        if not changed:
            break
    else:
        cycle = sorted(f"{n}/{a}" for n, a in heads if level[(n, a)] > len(heads))
        raise StratificationError(
            f"negation inside a recursive cycle involving {', '.join(cycle)}; "
            f"the program has no single meaning"
        )

    strata: dict[int, list[tuple[str, Rule]]] = {}
    for rule_id, rule in rules:
        strata.setdefault(level[rule.head.signature], []).append((rule_id, rule))
    return [strata[k] for k in sorted(strata)]


# --------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------
class Engine:
    def __init__(self, max_rounds: int = 200) -> None:
        #: A backstop, not the termination argument — termination is structural.
        #: Reaching it means something is wrong, so it raises rather than
        #: returning a partial answer that would look like a real one.
        self.max_rounds = max_rounds

    def evaluate(self, rules: list[tuple[str, Rule]], facts: FactStore) -> Derivation:
        derived: dict[Atom, Derived] = {}
        blocked: set[Atom] = set()
        used: set[str] = set()
        rounds = 0
        heads = frozenset(r.head.signature for _, r in rules)

        for stratum in stratify(rules):
            while True:
                rounds += 1
                if rounds > self.max_rounds:
                    raise RuntimeError("fixpoint not reached; rule base is wrong")
                # A snapshot, so a rule's result cannot depend on where in the
                # round it ran. Also correct semi-naive semantics.
                known = frozenset(derived) | frozenset(facts.atoms(Truth.TRUE))
                blocked_now = frozenset(blocked)
                new = False

                for rule_id, rule in stratum:
                    for binding, premises, negs, uncertain in self._match(
                        rule, known, facts, blocked_now, heads
                    ):
                        head = rule.head.substitute(binding)
                        if uncertain:
                            if head not in derived and head not in blocked:
                                blocked.add(head)
                                new = True
                            continue
                        support = Support(rule_id, premises, negs)
                        node = derived.get(head)
                        if node is None:
                            derived[head] = Derived(head, [support])
                            blocked.discard(head)
                            used.add(rule_id)
                            new = True
                        elif support not in node.supports:
                            node.supports.append(support)
                            used.add(rule_id)
                            new = True
                if not new:
                    break

        return Derivation(derived, blocked, facts, rounds, used, heads)

    # -- matching ----------------------------------------------------------
    def _match(self, rule: Rule, known: frozenset, facts: FactStore,
               blocked: frozenset, heads: frozenset):
        """Every way this rule's body is satisfied against the snapshot.

        Yields (binding, positive premises, negated premises, uncertain).
        `uncertain` is set when a premise is UNKNOWN — the rule cannot fire, but
        its head must not be treated as false either.
        """
        positives = rule.positive_body
        results = []

        def walk(i: int, binding: Binding, premises: list[Atom]) -> None:
            if i == len(positives):
                negs: list[Atom] = []
                uncertain = False
                for lit in rule.negative_body:
                    atom = lit.atom.substitute(binding)
                    if atom in blocked:
                        uncertain = True          # ignorance survives negation
                        break
                    if atom in known:
                        return                    # the negation plainly fails
                    truth = facts.truth_of(atom)
                    if truth is Truth.FALSE:
                        negs.append(atom)         # recorded absent: negation holds
                        continue
                    if atom.signature in heads:
                        negs.append(atom)         # derivable but not derived: false
                        continue
                    # An observable nobody mentioned. Not false — unknown. Under
                    # a closed world this is where silence becomes evidence.
                    uncertain = True
                    break
                results.append((dict(binding), tuple(premises), tuple(negs), uncertain))
                return

            lit = positives[i]
            for candidate in known:
                if candidate.signature != lit.signature:
                    continue
                extended = self._unify(lit.atom, candidate, binding)
                if extended is not None:
                    walk(i + 1, extended, premises + [candidate])

        walk(0, {}, [])
        return results

    @staticmethod
    def _unify(pattern: Atom, ground: Atom, binding: Binding) -> Binding | None:
        out = dict(binding)
        for p, g in zip(pattern.terms, ground.terms):
            if isinstance(p, Var):
                bound = out.get(p.name)
                if bound is None:
                    out[p.name] = g
                elif bound != g:
                    return None
            elif p != g:
                return None
        return out


__all__ = ["Derivation", "Derived", "Engine", "StratificationError", "Support",
           "stratify"]
