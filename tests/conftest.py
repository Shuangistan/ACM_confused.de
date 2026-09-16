"""Shared fixtures and generators for the logic database tests."""

from __future__ import annotations

import random

import pytest

from logicdb.facts import FactStore, Truth
from logicdb.parser import parse_atom, parse_rule
from logicdb.program import RuleBase, RuleOrigin, RuleRecord, RuleStatus
from logicdb.syntax import PredicateDecl, Vocabulary


@pytest.fixture
def lending_vocab() -> Vocabulary:
    return Vocabulary(
        [
            PredicateDecl("overdrawn", 1, "observable", phrase="{0} is overdrawn"),
            PredicateDecl("guarantor", 1, "observable", phrase="{0} has a guarantor"),
            PredicateDecl("thin_file", 1, "observable",
                          phrase="{0} has a thin credit file"),
            PredicateDecl("employed_long", 1, "observable",
                          phrase="{0} has been employed 4+ years"),
            PredicateDecl("owns_property", 1, "observable",
                          phrase="{0} owns property"),
            PredicateDecl("large_amount", 1, "observable",
                          phrase="{0} requests a large amount"),
            PredicateDecl("high_risk", 1, "derived", phrase="{0} is high risk"),
            PredicateDecl("mitigated", 1, "derived", phrase="{0} has mitigation"),
            PredicateDecl("decline", 1, "derived", phrase="{0} should be declined"),
            PredicateDecl("approve", 1, "derived", phrase="{0} should be approved"),
        ]
    )


@pytest.fixture
def lending_rules(lending_vocab: Vocabulary) -> RuleBase:
    rb = RuleBase(lending_vocab)
    specs = [
        ("high_risk(A) <- overdrawn(A), thin_file(A).", 0.80),
        ("high_risk(A) <- large_amount(A), not employed_long(A).", 0.70),
        ("mitigated(A) <- guarantor(A).", 0.90),
        ("mitigated(A) <- owns_property(A), employed_long(A).", 0.75),
        ("decline(A) <- high_risk(A), not mitigated(A).", 0.88),
        ("approve(A) <- mitigated(A), not high_risk(A).", 0.85),
    ]
    for i, (text, strength) in enumerate(specs, start=1):
        rb.add(
            RuleRecord(
                rule=parse_rule(text),
                rule_id=f"r{i:04d}",
                strength=strength,
                status=RuleStatus.APPROVED,
                origin=RuleOrigin.SEED,
                approved_by="fixture",
            )
        )
    return rb


def make_facts(vocab: Vocabulary, entity: str = "app1", **assignments: str) -> FactStore:
    """Build a fact store from `predicate="true"|"false"|"unknown"` kwargs."""
    store = FactStore(vocab)
    for predicate, state in assignments.items():
        store.assert_fact(
            parse_atom(f"{predicate}({entity})"),
            Truth(state),
            source=f"test fixture ({state})",
        )
    return store


def random_program(
    seed: int, n_observables: int = 6, n_rules: int = 5, allow_negation: bool = True
) -> tuple[RuleBase, Vocabulary, list[str]]:
    """A random stratified program, for property testing.

    Random programs are how the bounds fast path gets checked against the
    definition. Hand-written examples only ever cover the cases the author
    thought of, and the interesting failures here are the ones nobody would
    think to write down.
    """
    rng = random.Random(seed)
    observables = [f"o{i}" for i in range(n_observables)]
    derived = ["m1", "m2", "goal"]
    vocab = Vocabulary(
        [PredicateDecl(name, 1, "observable") for name in observables]
        + [PredicateDecl(name, 1, "derived") for name in derived]
    )

    rb = RuleBase(vocab)
    made, attempts = 0, 0
    while made < n_rules and attempts < 80:
        attempts += 1
        head = rng.choice(derived)
        # Only `goal` may depend on the intermediates, which keeps the generated
        # programs stratifiable without rejecting most of what is generated.
        pool = observables + (["m1", "m2"] if head == "goal" else [])
        body = rng.sample(pool, min(rng.randint(1, 3), len(pool)))
        literals = [
            ("not " if (allow_negation and rng.random() < 0.3 and b in observables) else "")
            + f"{b}(A)"
            for b in body
        ]
        text = f"{head}(A) <- " + ", ".join(literals) + "."
        try:
            rb.add(
                RuleRecord(
                    rule=parse_rule(text),
                    rule_id=f"r{made:03d}",
                    strength=round(rng.uniform(0.4, 0.95), 2),
                    status=RuleStatus.APPROVED,
                    origin=RuleOrigin.SEED,
                )
            )
            made += 1
        except Exception:
            continue
    return rb, vocab, observables


def random_facts(
    vocab: Vocabulary, observables: list[str], seed: int, entity: str = "e1"
) -> tuple[FactStore, list]:
    rng = random.Random(seed)
    store = FactStore(vocab)
    unknowns = []
    for name in observables:
        atom = parse_atom(f"{name}({entity})")
        truth = rng.choice([Truth.TRUE, Truth.FALSE, Truth.UNKNOWN])
        store.assert_fact(atom, truth, source="random fixture")
        if truth is Truth.UNKNOWN:
            unknowns.append(atom)
    return store, unknowns
