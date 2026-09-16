"""Tests for inference: termination, determinism, stratification, three-valued truth.

Determinism and termination are the properties claimed on the tin, so they are
tested as properties over generated programs rather than demonstrated on one
example. The three-valued tests exist because the single most dangerous thing
this system could do is treat a fact nobody recorded as false and then act
confidently on that assumption.
"""

from __future__ import annotations

import pytest

from logicdb.engine import Engine, evaluate
from logicdb.facts import FactStore, Truth
from logicdb.parser import parse_atom, parse_rule
from logicdb.program import (
    RuleBase,
    RuleOrigin,
    RuleRecord,
    RuleStatus,
    StratificationError,
)
from logicdb.syntax import PredicateDecl, Vocabulary

from .conftest import make_facts, random_facts, random_program


# --------------------------------------------------------------------------
# Basic derivation
# --------------------------------------------------------------------------


def test_derives_conclusion_from_facts(lending_vocab, lending_rules) -> None:
    facts = make_facts(
        lending_vocab, overdrawn="true", thin_file="true", guarantor="false",
        owns_property="false", employed_long="false", large_amount="false",
    )
    ev = evaluate(lending_rules, facts, critical=True)
    assert ev.holds(parse_atom("high_risk(app1)"))
    assert ev.holds(parse_atom("decline(app1)"))


def test_negated_literal_blocks_a_rule(lending_vocab, lending_rules) -> None:
    """A guarantor mitigates, so the decline rule must not fire."""
    facts = make_facts(
        lending_vocab, overdrawn="true", thin_file="true", guarantor="true",
        owns_property="false", employed_long="false", large_amount="false",
    )
    ev = evaluate(lending_rules, facts, critical=True)
    assert ev.holds(parse_atom("high_risk(app1)"))
    assert ev.holds(parse_atom("mitigated(app1)"))
    assert not ev.holds(parse_atom("decline(app1)"))


def test_multiple_supports_recorded_for_one_conclusion(lending_vocab, lending_rules) -> None:
    """Two independent routes to the same conclusion are both kept.

    Needed by the probability layer, and a reviewer wants to see that two rules
    agree rather than only that the conclusion held.
    """
    facts = make_facts(
        lending_vocab, guarantor="true", owns_property="true", employed_long="true",
        overdrawn="false", thin_file="false", large_amount="false",
    )
    ev = evaluate(lending_rules, facts, critical=True)
    node = ev.get(parse_atom("mitigated(app1)"))
    assert node is not None
    assert len(node.rule_ids()) == 2


# --------------------------------------------------------------------------
# Three-valued truth
# --------------------------------------------------------------------------


def test_unknown_premise_blocks_rather_than_defaults_to_false(
    lending_vocab, lending_rules
) -> None:
    """The failure this system exists to prevent.

    Under a closed-world reading, an unrecorded guarantor would read as "no
    guarantor" and the application would be declined on an assumption nobody
    made. Here the rule is blocked instead, and the missing fact is named.
    """
    facts = make_facts(
        lending_vocab, overdrawn="true", thin_file="true", guarantor="unknown",
        owns_property="false", employed_long="false", large_amount="false",
    )
    ev = evaluate(lending_rules, facts, critical=True)

    assert ev.holds(parse_atom("high_risk(app1)"))
    assert not ev.holds(parse_atom("decline(app1)")), (
        "an unknown guarantor must not silently become 'no guarantor'"
    )
    blocking = ev.unknowns_blocking(parse_atom("decline(app1)"))
    assert parse_atom("guarantor(app1)") in blocking


def test_unrecorded_askable_fact_is_unknown_not_false(lending_vocab) -> None:
    facts = FactStore(lending_vocab)
    assert facts.truth_of(parse_atom("guarantor(app1)")) is Truth.UNKNOWN


def test_non_askable_predicate_defaults_to_false() -> None:
    vocab = Vocabulary([PredicateDecl("bookkeeping", 1, "observable", askable=False)])
    facts = FactStore(vocab)
    assert facts.truth_of(parse_atom("bookkeeping(x)")) is Truth.FALSE


def test_blocked_derivation_names_the_rule_and_the_missing_fact(
    lending_vocab, lending_rules
) -> None:
    facts = make_facts(
        lending_vocab, overdrawn="true", thin_file="true", guarantor="unknown",
        owns_property="false", employed_long="false", large_amount="false",
    )
    ev = evaluate(lending_rules, facts, critical=True)

    # The immediate obstacle to `decline` is the derived atom `mitigated`, whose
    # own derivation is stuck. The fact actually worth asking a human about sits
    # one level further down.
    decline_blocks = [
        b for b in ev.blocked if b.would_derive == parse_atom("decline(app1)")
    ]
    assert decline_blocks
    assert parse_atom("mitigated(app1)") in decline_blocks[0].unknown_premises

    mitigated_blocks = [
        b for b in ev.blocked if b.would_derive == parse_atom("mitigated(app1)")
    ]
    assert mitigated_blocks
    assert parse_atom("guarantor(app1)") in mitigated_blocks[0].unknown_premises

    # ...and the transitive walk is what the question-ranking actually consumes.
    assert parse_atom("guarantor(app1)") in ev.unknowns_blocking(
        parse_atom("decline(app1)")
    )


# --------------------------------------------------------------------------
# Facts cannot be forged
# --------------------------------------------------------------------------


def test_cannot_assert_a_derived_predicate(lending_vocab) -> None:
    facts = FactStore(lending_vocab)
    with pytest.raises(ValueError, match="declared derived"):
        facts.assert_fact(parse_atom("decline(app1)"), Truth.TRUE)


def test_cannot_assert_an_undeclared_predicate(lending_vocab) -> None:
    facts = FactStore(lending_vocab)
    with pytest.raises(ValueError, match="undeclared predicate"):
        facts.assert_fact(parse_atom("invented(app1)"), Truth.TRUE)


def test_facts_must_be_ground(lending_vocab) -> None:
    from logicdb.facts import FactRecord
    from logicdb.syntax import Atom, Var

    with pytest.raises(ValueError, match="must be ground"):
        FactRecord(Atom("overdrawn", (Var("A"),)))


# --------------------------------------------------------------------------
# Stratification
# --------------------------------------------------------------------------


def test_direct_negative_self_dependency_rejected() -> None:
    vocab = Vocabulary(
        [PredicateDecl("p", 1, "observable"), PredicateDecl("q", 1, "derived")]
    )
    rb = RuleBase(vocab)
    with pytest.raises(StratificationError, match="not stratifiable|negative cycle"):
        rb.add(
            RuleRecord(rule=parse_rule("q(A) <- p(A), not q(A)."), rule_id="bad")
        )


def test_negative_cycle_through_two_predicates_rejected() -> None:
    vocab = Vocabulary(
        [
            PredicateDecl("p", 1, "observable"),
            PredicateDecl("q", 1, "derived"),
            PredicateDecl("r", 1, "derived"),
        ]
    )
    rb = RuleBase(vocab)
    rb.add(RuleRecord(rule=parse_rule("q(A) <- p(A), not r(A)."), rule_id="r1"))
    with pytest.raises(StratificationError) as exc:
        rb.add(RuleRecord(rule=parse_rule("r(A) <- p(A), not q(A)."), rule_id="r2"))
    assert "r2" in str(exc.value), "the message must name the rule being rejected"


def test_positive_recursion_is_allowed_and_terminates() -> None:
    """Recursion is fine; it is negative recursion that has no unique model."""
    vocab = Vocabulary(
        [
            PredicateDecl("edge", 2, "observable"),
            PredicateDecl("path", 2, "derived"),
        ]
    )
    rb = RuleBase(vocab)
    rb.add(RuleRecord(rule=parse_rule("path(A, B) <- edge(A, B)."), rule_id="r1",
                      status=RuleStatus.APPROVED))
    rb.add(RuleRecord(rule=parse_rule("path(A, C) <- path(A, B), edge(B, C)."),
                      rule_id="r2", status=RuleStatus.APPROVED))

    facts = FactStore(vocab)
    for a, b in [("n1", "n2"), ("n2", "n3"), ("n3", "n4")]:
        facts.assert_fact(parse_atom(f"edge({a}, {b})"), Truth.TRUE, source="test")

    ev = evaluate(rb, facts, critical=True)
    assert ev.holds(parse_atom("path(n1, n4)"))
    assert ev.rounds < 20, "a transitive closure over 4 nodes must converge quickly"


def test_strata_order_negated_predicates_first(lending_rules) -> None:
    strata = lending_rules.strata()
    assert strata["mitigated/1"] < strata["decline/1"], (
        "mitigated must be fully computed before decline tests 'not mitigated'"
    )


# --------------------------------------------------------------------------
# Determinism and termination as properties
# --------------------------------------------------------------------------


@pytest.mark.parametrize("seed", range(20))
def test_evaluation_is_deterministic(seed: int) -> None:
    rb, vocab, observables = random_program(seed)
    facts, _ = random_facts(vocab, observables, seed + 500)

    runs = []
    for _ in range(3):
        ev = evaluate(rb, facts, critical=True)
        runs.append(sorted(str(a) for a in ev.derived))
    assert runs[0] == runs[1] == runs[2]


@pytest.mark.parametrize("seed", range(20))
def test_evaluation_terminates(seed: int) -> None:
    """Datalog's finiteness is structural, not a timeout. No program may hang."""
    rb, vocab, observables = random_program(seed)
    facts, _ = random_facts(vocab, observables, seed + 700)
    ev = Engine(max_rounds=100).evaluate(rb, facts, critical=True)
    assert ev.rounds <= 100


def test_rule_order_does_not_change_the_result() -> None:
    """Insertion order must not matter; otherwise 'deterministic' is hollow."""
    vocab = Vocabulary(
        [
            PredicateDecl("a", 1, "observable"),
            PredicateDecl("b", 1, "observable"),
            PredicateDecl("m", 1, "derived"),
            PredicateDecl("g", 1, "derived"),
        ]
    )
    texts = [
        ("g(A) <- m(A), b(A).", "r1"),
        ("m(A) <- a(A).", "r2"),
        ("g(A) <- a(A), b(A).", "r3"),
    ]

    results = []
    for order in (texts, list(reversed(texts))):
        rb = RuleBase(vocab)
        for text, rid in order:
            rb.add(RuleRecord(rule=parse_rule(text), rule_id=rid,
                              status=RuleStatus.APPROVED))
        facts = FactStore(vocab)
        facts.assert_fact(parse_atom("a(x)"), Truth.TRUE, source="t")
        facts.assert_fact(parse_atom("b(x)"), Truth.TRUE, source="t")
        ev = evaluate(rb, facts, critical=True)
        results.append(sorted(str(a) for a in ev.derived))
    assert results[0] == results[1]


# --------------------------------------------------------------------------
# Provenance
# --------------------------------------------------------------------------


def test_supports_carry_their_premises(lending_vocab, lending_rules) -> None:
    """Every conclusion must know what it rests on, or explanation is impossible."""
    facts = make_facts(
        lending_vocab, overdrawn="true", thin_file="true", guarantor="false",
        owns_property="false", employed_long="false", large_amount="false",
    )
    ev = evaluate(lending_rules, facts, critical=True)
    node = ev.get(parse_atom("high_risk(app1)"))
    premises = {str(a) for support in node.supports for a, _ in support.premises}
    assert premises == {"overdrawn(app1)", "thin_file(app1)"}


def test_facts_retain_their_source(lending_vocab) -> None:
    facts = FactStore(lending_vocab)
    facts.assert_fact(
        parse_atom("overdrawn(app1)"), Truth.TRUE, source="application form, field 3"
    )
    assert facts.record_of(parse_atom("overdrawn(app1)")).source == (
        "application form, field 3"
    )
