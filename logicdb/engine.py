"""Inference: semi-naive forward chaining over strata, producing a proof DAG.

Forward chaining rather than backward: the goal is not only to answer the query
but to know *everything the rule base concluded and why*. A backward chainer
answers the question asked; a forward chainer hands you the whole derivation,
which is what the explanation, the counterfactuals and the value-of-information
ranking all consume.

Termination is structural rather than something to be careful about. Datalog has
no function symbols, so every derivable atom is built from constants already
present in the facts and rules. That set is finite, derivation only adds atoms,
so the fixpoint is reached in finitely many rounds. A recursive rule an agent
adds cannot hang the system -- which is exactly the property Prolog could not
have offered.

Three-valued evaluation. A body literal can be TRUE, FALSE or UNKNOWN, and a
rule only fires when every literal is definitely TRUE. Atoms that would fire
*if* some unknown resolved favourably are tracked separately as `blocked_by`,
and that record is what turns "we can't tell" into "ask about the guarantor".
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Iterator

from .facts import FactStore, Truth
from .program import RuleBase, RuleRecord
from .syntax import Atom, Literal, Rule, Term, Var


# --------------------------------------------------------------------------
# Proof structures
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Support:
    """One rule application that derived an atom.

    An atom can have several supports -- several independent reasons to believe
    it -- which is what the noisy-OR combination in `probability.py` consumes.
    """

    rule_id: str
    rule: Rule
    strength: float
    #: Ground body literals, in the order they appear in the rule, so the
    #: explanation reads the way the rule was written.
    premises: tuple[tuple[Atom, bool], ...]
    #: Round of the fixpoint at which this support became available. Used to
    #: break recursive cycles when computing probability: a support may only
    #: count toward an atom whose own depth is strictly greater.
    depth: int

    def positive_premises(self) -> tuple[Atom, ...]:
        return tuple(a for a, neg in self.premises if not neg)


@dataclass
class DerivedAtom:
    """A ground atom the rule base concluded, with every reason it did."""

    atom: Atom
    depth: int
    supports: list[Support] = field(default_factory=list)

    def rule_ids(self) -> set[str]:
        return {s.rule_id for s in self.supports}


@dataclass(frozen=True)
class BlockedDerivation:
    """A rule that would have fired but for one or more unknown premises.

    The most useful thing the engine produces when it cannot answer. Each of
    these names a specific missing fact and the conclusion it is standing in the
    way of, which is what makes "critical information may be missing" into an
    actionable question rather than a disclaimer.
    """

    rule_id: str
    rule: Rule
    would_derive: Atom
    unknown_premises: tuple[Atom, ...]
    strength: float


@dataclass
class Evaluation:
    """The result of running the rule base to fixpoint."""

    derived: dict[Atom, DerivedAtom]
    blocked: list[BlockedDerivation]
    facts: FactStore
    rounds: int
    rules_used: set[str]

    def holds(self, atom: Atom) -> bool:
        return atom in self.derived or self.facts.truth_of(atom) is Truth.TRUE

    def get(self, atom: Atom) -> DerivedAtom | None:
        return self.derived.get(atom)

    def atoms_for(self, predicate: str) -> list[Atom]:
        return [a for a in self.derived if a.predicate == predicate]

    def unknowns_blocking(self, goal: Atom) -> set[Atom]:
        """Unknown atoms standing between the current facts and `goal`.

        Walks transitively: a premise may itself be a derived atom whose own
        derivation is blocked further down, and the fact worth asking about is
        the one at the bottom.
        """
        out: set[Atom] = set()
        seen: set[Atom] = set()
        frontier = [goal]
        while frontier:
            current = frontier.pop()
            if current in seen:
                continue
            seen.add(current)
            for block in self.blocked:
                if block.would_derive != current:
                    continue
                for premise in block.unknown_premises:
                    if self.facts.truth_of(premise) is Truth.UNKNOWN:
                        out.add(premise)
                    frontier.append(premise)
            node = self.derived.get(current)
            if node is not None:
                for support in node.supports:
                    frontier.extend(support.positive_premises())
        return out


# --------------------------------------------------------------------------
# Unification
# --------------------------------------------------------------------------


def match_atom(pattern: Atom, ground: Atom, binding: dict[str, Term]) -> dict[str, Term] | None:
    """One-way match of a rule literal against a ground atom.

    One-way rather than full unification because Datalog has no function
    symbols: the pattern may contain variables, the target never does. This is
    both simpler and enough.
    """
    if pattern.predicate != ground.predicate or pattern.arity != ground.arity:
        return None
    result = dict(binding)
    for p, g in zip(pattern.terms, ground.terms):
        if isinstance(p, Var):
            bound = result.get(p.name)
            if bound is None:
                result[p.name] = g
            elif bound != g:
                return None
        elif p != g:
            return None
    return result


# --------------------------------------------------------------------------
# The evaluator
# --------------------------------------------------------------------------


class Engine:
    """Evaluates a rule base against facts to a fixpoint."""

    def __init__(self, max_rounds: int = 200, use_posterior_strength: bool = False) -> None:
        #: A backstop, not the termination argument -- termination is structural.
        #: Hitting this means something is wrong, so it raises rather than
        #: returning a partial answer that would look like a real one.
        self.max_rounds = max_rounds
        self.use_posterior_strength = use_posterior_strength

    def evaluate(
        self,
        rulebase: RuleBase,
        facts: FactStore,
        critical: bool = False,
        rules: Iterable[RuleRecord] | None = None,
    ) -> Evaluation:
        """Run to fixpoint.

        `critical` selects which rules are eligible. This is where the
        governance guarantee becomes mechanical: for a critical query the engine
        is handed only approved rules, so an unapproved rule cannot contribute
        to the conclusion even by accident.
        """
        active = list(rules) if rules is not None else rulebase.active_for(critical)
        if not active:
            return Evaluation({}, [], facts, 0, set())

        strata = rulebase.strata(active)
        by_stratum: dict[int, list[RuleRecord]] = {}
        for record in active:
            by_stratum.setdefault(strata.get(record.head_signature, 0), []).append(record)

        derived: dict[Atom, DerivedAtom] = {}
        blocked: list[BlockedDerivation] = []
        #: Heads that some rule would have concluded but for an unknown premise.
        #: These atoms are genuinely UNKNOWN rather than false, and keeping them
        #: separate is what stops unknown-ness from being lost at a negation --
        #: see `_truth_of`.
        blocked_heads: set[Atom] = set()
        rules_used: set[str] = set()
        total_rounds = 0
        depth = 0

        for level in sorted(by_stratum):
            # Strata are evaluated in order so that by the time a negated
            # literal is tested, the predicate it negates is already complete.
            # Testing negation against a half-built relation is the classic way
            # to get an answer that depends on evaluation order.
            stratum_rules = by_stratum[level]
            frontier_changed = True
            while frontier_changed:
                if total_rounds >= self.max_rounds:
                    raise RuntimeError(
                        f"fixpoint not reached within {self.max_rounds} rounds at "
                        f"stratum {level}. Datalog guarantees termination, so this "
                        f"indicates a bug in evaluation rather than in the rules."
                    )
                total_rounds += 1
                depth += 1
                frontier_changed = False

                # Snapshot what is derived at the start of the round. Matching
                # against the live dictionary would mutate it mid-iteration, and
                # would also make a rule's result depend on where in the round it
                # happened to run -- the opposite of the determinism this whole
                # design is for. Atoms derived during this round are picked up by
                # the next one, which is what the fixpoint loop is for.
                snapshot = _index_by_signature(derived)
                blocked_snapshot = _index_by_signature_set(blocked_heads)

                for record in stratum_rules:
                    for binding, premises, unknowns in self._match_body(
                        record.rule, facts, derived, snapshot,
                        blocked_heads, blocked_snapshot,
                    ):
                        head = record.rule.head.substitute(binding)
                        if not head.is_ground():
                            continue

                        if unknowns:
                            block = BlockedDerivation(
                                rule_id=record.rule_id,
                                rule=record.rule,
                                would_derive=head,
                                unknown_premises=tuple(unknowns),
                                strength=record.effective_strength(
                                    self.use_posterior_strength
                                ),
                            )
                            if block not in blocked:
                                blocked.append(block)
                            if head not in derived and head not in blocked_heads:
                                blocked_heads.add(head)
                                # A newly uncertain head can change how a
                                # negation elsewhere evaluates, so the fixpoint
                                # has not settled yet.
                                frontier_changed = True
                            continue

                        support = Support(
                            rule_id=record.rule_id,
                            rule=record.rule,
                            strength=record.effective_strength(self.use_posterior_strength),
                            premises=tuple(premises),
                            depth=depth,
                        )
                        node = derived.get(head)
                        if node is None:
                            derived[head] = DerivedAtom(head, depth, [support])
                            rules_used.add(record.rule_id)
                            frontier_changed = True
                        elif support.rule_id not in node.rule_ids() or not any(
                            s.premises == support.premises for s in node.supports
                        ):
                            # A genuinely new reason for an already-known
                            # conclusion. Recorded because it raises the
                            # probability and because a reviewer wants to see
                            # that two independent rules agree.
                            node.supports.append(support)
                            rules_used.add(record.rule_id)
                            frontier_changed = True

        return Evaluation(derived, blocked, facts, total_rounds, rules_used)

    # -- body matching -----------------------------------------------------
    def _match_body(
        self,
        rule: Rule,
        facts: FactStore,
        derived: dict[Atom, DerivedAtom],
        snapshot: dict[str, set[Atom]],
        blocked_heads: set[Atom],
        blocked_snapshot: dict[str, set[Atom]],
    ) -> Iterator[tuple[dict[str, Term], list[tuple[Atom, bool]], list[Atom]]]:
        """Enumerate ground instances of a rule body.

        Yields `(binding, premises, unknown_premises)`. A yielded instance with
        a non-empty unknown list did not fire but would have if those atoms
        resolved favourably -- the raw material for value-of-information.

        Positive literals are matched first regardless of the order they were
        written in, because they are what bind the variables. A negated literal
        is only meaningful once its variables are ground, and requiring the
        author to order the body correctly would be a trap for a proposing agent.
        """
        positives = [l for l in rule.body if not l.negated]
        negatives = [l for l in rule.body if l.negated]

        for binding, premises, unknowns in self._match_positives(
            positives, 0, {}, [], [], facts, derived, snapshot, blocked_snapshot
        ):
            ok = True
            all_unknowns = list(unknowns)
            for lit in negatives:
                ground = lit.atom.substitute(binding)
                if not ground.is_ground():
                    # Safety checking at Rule construction should prevent this.
                    ok = False
                    break
                truth = self._truth_of(ground, facts, derived, blocked_heads)
                if truth is Truth.TRUE:
                    ok = False
                    break
                if truth is Truth.UNKNOWN:
                    all_unknowns.append(ground)
                premises.append((ground, True))
            if ok:
                yield binding, premises, all_unknowns

    def _match_positives(
        self,
        literals: list[Literal],
        index: int,
        binding: dict[str, Term],
        premises: list[tuple[Atom, bool]],
        unknowns: list[Atom],
        facts: FactStore,
        derived: dict[Atom, DerivedAtom],
        snapshot: dict[str, set[Atom]],
        blocked_snapshot: dict[str, set[Atom]],
    ) -> Iterator[tuple[dict[str, Term], list[tuple[Atom, bool]], list[Atom]]]:
        if index >= len(literals):
            yield dict(binding), list(premises), list(unknowns)
            return

        literal = literals[index]
        pattern = literal.atom.substitute(binding)

        for candidate, truth in self._candidates(
            pattern, facts, snapshot, blocked_snapshot
        ):
            extended = match_atom(pattern, candidate, binding)
            if extended is None:
                continue
            if truth is Truth.FALSE:
                continue
            next_unknowns = unknowns + [candidate] if truth is Truth.UNKNOWN else unknowns
            yield from self._match_positives(
                literals,
                index + 1,
                extended,
                premises + [(candidate, False)],
                next_unknowns,
                facts,
                derived,
                snapshot,
                blocked_snapshot,
            )

    def _candidates(
        self,
        pattern: Atom,
        facts: FactStore,
        snapshot: dict[str, set[Atom]],
        blocked_snapshot: dict[str, set[Atom]],
    ) -> Iterator[tuple[Atom, Truth]]:
        """Ground atoms that could match `pattern`, with their truth.

        Drawn from the facts and from the start-of-round snapshot of derived
        atoms, indexed by signature. The finiteness of this set is what makes
        the whole thing terminate.
        """
        seen: set[Atom] = set()
        for atom in facts.atoms_for(pattern.signature):
            if atom not in seen:
                seen.add(atom)
                yield atom, facts.truth_of(atom)
        for atom in snapshot.get(pattern.signature, ()):
            if atom not in seen:
                seen.add(atom)
                yield atom, Truth.TRUE
        # A derived atom whose own derivation is blocked is unknown, not absent.
        # Offering it here is what lets uncertainty propagate up a chain instead
        # of stopping at the first derived predicate.
        for atom in blocked_snapshot.get(pattern.signature, ()):
            if atom not in seen:
                seen.add(atom)
                yield atom, Truth.UNKNOWN
        # A fully ground pattern may name an atom nobody has recorded. Offer it
        # so that `truth_of` can class it as UNKNOWN for an askable predicate --
        # otherwise an unrecorded field would silently read as false, which is
        # the closed-world failure this system exists to avoid.
        if pattern.is_ground() and pattern not in seen:
            yield pattern, facts.truth_of(pattern)

    @staticmethod
    def _truth_of(
        atom: Atom,
        facts: FactStore,
        derived: dict[Atom, DerivedAtom],
        blocked_heads: set[Atom],
    ) -> Truth:
        """Truth of a ground atom, with blocked derivations treated as unknown.

        The `blocked_heads` check is the one that matters. Without it, a derived
        atom whose own derivation was stopped by a missing fact reads as FALSE,
        so `not mitigated(app)` succeeds and the applicant is declined -- on the
        strength of a guarantor nobody ever asked about. The unknown has to
        survive the negation, or three-valued logic buys nothing beyond the
        first level.
        """
        if atom in derived:
            return Truth.TRUE
        if atom in blocked_heads:
            return Truth.UNKNOWN
        return facts.truth_of(atom)


def _index_by_signature(derived: dict[Atom, DerivedAtom]) -> dict[str, set[Atom]]:
    index: dict[str, set[Atom]] = {}
    for atom in derived:
        index.setdefault(atom.signature, set()).add(atom)
    return index


def _index_by_signature_set(atoms: set[Atom]) -> dict[str, set[Atom]]:
    index: dict[str, set[Atom]] = {}
    for atom in atoms:
        index.setdefault(atom.signature, set()).add(atom)
    return index


def evaluate(
    rulebase: RuleBase,
    facts: FactStore,
    critical: bool = False,
    use_posterior_strength: bool = False,
) -> Evaluation:
    """Convenience wrapper for a single evaluation."""
    return Engine(use_posterior_strength=use_posterior_strength).evaluate(
        rulebase, facts, critical
    )


__all__ = [
    "BlockedDerivation",
    "DerivedAtom",
    "Engine",
    "Evaluation",
    "Support",
    "evaluate",
    "match_atom",
]
