"""Tests for probability, bounds under missing information, and what to ask.

`test_bounds_match_brute_force` is the most important test in the suite. The
fast path skips enumerating 2^k assignments by reasoning about polarity, and a
shortcut that nobody checks against the definition is a guess. Here it is
checked against exhaustive enumeration on randomly generated programs, including
ones with negation, where the polarity reasoning is exactly what could go wrong.
"""

from __future__ import annotations

import pytest

from logicdb.facts import Truth
from logicdb.parser import parse_atom, parse_rule
from logicdb.probability import (
    ProbabilisticSolver,
    ProbabilityComputer,
    noisy_or,
    predicate_polarities,
)
from logicdb.program import RuleBase, RuleRecord, RuleStatus
from logicdb.engine import evaluate
from logicdb.facts import FactStore
from logicdb.syntax import PredicateDecl, Vocabulary

from .conftest import make_facts, random_facts, random_program


# --------------------------------------------------------------------------
# Combination
# --------------------------------------------------------------------------


def test_noisy_or_of_nothing_is_zero() -> None:
    assert noisy_or([]) == 0.0


def test_noisy_or_of_one_is_itself() -> None:
    assert noisy_or([0.7]) == pytest.approx(0.7)


def test_noisy_or_increases_with_more_support() -> None:
    """Two independent reasons must beat one, without ever exceeding 1."""
    assert noisy_or([0.7, 0.5]) == pytest.approx(1 - 0.3 * 0.5)
    assert noisy_or([0.7, 0.5]) > 0.7
    assert noisy_or([0.9, 0.9, 0.9]) <= 1.0


def test_certainty_saturates() -> None:
    assert noisy_or([1.0, 0.5]) == pytest.approx(1.0)


def test_probability_chains_through_derived_atoms(lending_vocab, lending_rules) -> None:
    facts = make_facts(
        lending_vocab, overdrawn="true", thin_file="true", guarantor="false",
        owns_property="false", employed_long="false", large_amount="false",
    )
    ev = evaluate(lending_rules, facts, critical=True)
    computer = ProbabilityComputer(ev)
    # high_risk from r0001 at 0.80, then decline from r0005 at 0.88.
    assert computer.probability_of(parse_atom("high_risk(app1)")) == pytest.approx(0.80)
    assert computer.probability_of(parse_atom("decline(app1)")) == pytest.approx(
        0.88 * 0.80
    )


def test_circular_support_contributes_nothing() -> None:
    """A rule base must not be able to inflate its confidence by citing itself."""
    vocab = Vocabulary(
        [PredicateDecl("seed", 1, "observable"), PredicateDecl("p", 1, "derived")]
    )
    rb = RuleBase(vocab)
    rb.add(RuleRecord(rule=parse_rule("p(A) <- seed(A)."), rule_id="r1",
                      strength=0.6, status=RuleStatus.APPROVED))
    rb.add(RuleRecord(rule=parse_rule("p(A) <- p(A)."), rule_id="r2",
                      strength=0.99, status=RuleStatus.APPROVED))

    facts = FactStore(vocab)
    facts.assert_fact(parse_atom("seed(x)"), Truth.TRUE, source="t")
    ev = evaluate(rb, facts, critical=True)
    p = ProbabilityComputer(ev).probability_of(parse_atom("p(x)"))
    assert p == pytest.approx(0.6), "self-support must add nothing"


# --------------------------------------------------------------------------
# Polarity
# --------------------------------------------------------------------------


def test_polarity_of_positive_dependency(lending_rules) -> None:
    pol = predicate_polarities(lending_rules.all_records(), "decline/1")
    assert pol["high_risk/1"] == {1}


def test_polarity_of_negated_dependency(lending_rules) -> None:
    """`decline <- high_risk, not mitigated` makes mitigated negative."""
    pol = predicate_polarities(lending_rules.all_records(), "decline/1")
    assert pol["mitigated/1"] == {-1}
    assert pol["guarantor/1"] == {-1}, "a guarantor can only reduce P(decline)"


def test_polarity_composes_through_double_negation(lending_rules) -> None:
    """employed_long reaches decline by two routes, and both lower it.

    It suppresses high_risk (which raises decline) and supports mitigated (which
    suppresses decline). Two different paths, same sign -- so it is unambiguously
    negative and the bounds code can set it directly instead of searching.
    """
    pol = predicate_polarities(lending_rules.all_records(), "decline/1")
    assert pol["employed_long/1"] == {-1}


def test_polarity_can_be_both() -> None:
    """A predicate that raises the goal on one path and lowers it on another.

    These are the only unknowns the bounds computation has to enumerate, because
    no single assignment is known in advance to minimise or maximise the answer.
    """
    vocab = Vocabulary(
        [
            PredicateDecl("flagged", 1, "observable"),
            PredicateDecl("other", 1, "observable"),
            PredicateDecl("goal", 1, "derived"),
        ]
    )
    rb = RuleBase(vocab)
    rb.add(RuleRecord(rule=parse_rule("goal(A) <- flagged(A)."), rule_id="r1",
                      strength=0.7, status=RuleStatus.APPROVED))
    rb.add(RuleRecord(rule=parse_rule("goal(A) <- other(A), not flagged(A)."),
                      rule_id="r2", strength=0.6, status=RuleStatus.APPROVED))

    pol = predicate_polarities(rb.all_records(), "goal/1")
    assert pol["flagged/1"] == {1, -1}


# --------------------------------------------------------------------------
# Bounds
# --------------------------------------------------------------------------


def test_no_unknowns_gives_a_point(lending_vocab, lending_rules) -> None:
    facts = make_facts(
        lending_vocab, overdrawn="true", thin_file="true", guarantor="false",
        owns_property="false", employed_long="false", large_amount="false",
    )
    bounds = ProbabilisticSolver(lending_rules).bounds_for(
        parse_atom("decline(app1)"), facts, critical=True
    )
    assert bounds.width == pytest.approx(0.0)


def test_unknown_widens_the_interval(lending_vocab, lending_rules) -> None:
    facts = make_facts(
        lending_vocab, overdrawn="true", thin_file="true", guarantor="unknown",
        owns_property="false", employed_long="false", large_amount="false",
    )
    bounds = ProbabilisticSolver(lending_rules).bounds_for(
        parse_atom("decline(app1)"), facts, critical=True
    )
    assert bounds.width > 0.5, "a decisive unknown should produce a wide interval"
    assert bounds.lower == pytest.approx(0.0)


@pytest.mark.parametrize("seed", range(40))
def test_bounds_match_brute_force(seed: int) -> None:
    """The fast path must agree with exhaustive enumeration, exactly.

    The fast path avoids 2^k evaluations by reasoning that an unknown appearing
    only positively is minimised at FALSE and maximised at TRUE, enumerating
    only the unknowns that appear in both polarities. That reasoning is the part
    that could be wrong, so it is checked against the definition rather than
    asserted. Random programs include negation for exactly this reason.
    """
    rb, vocab, observables = random_program(seed)
    facts, _ = random_facts(vocab, observables, seed + 1000)
    solver = ProbabilisticSolver(rb)
    goal = parse_atom("goal(e1)")

    fast = solver.bounds_for(goal, facts, critical=True)
    brute = solver.bounds_brute_force(goal, facts, critical=True)

    assert fast.lower == pytest.approx(brute.lower, abs=1e-9)
    assert fast.upper == pytest.approx(brute.upper, abs=1e-9)


@pytest.mark.parametrize("seed", range(15))
def test_bounds_match_brute_force_without_negation(seed: int) -> None:
    """The purely monotone case, where no enumeration should be needed at all."""
    rb, vocab, observables = random_program(seed, allow_negation=False)
    facts, _ = random_facts(vocab, observables, seed + 2000)
    solver = ProbabilisticSolver(rb)
    goal = parse_atom("goal(e1)")
    fast = solver.bounds_for(goal, facts, critical=True)
    brute = solver.bounds_brute_force(goal, facts, critical=True)
    assert fast.lower == pytest.approx(brute.lower, abs=1e-9)
    assert fast.upper == pytest.approx(brute.upper, abs=1e-9)


def test_bounds_always_contain_the_point_estimate(lending_vocab, lending_rules) -> None:
    facts = make_facts(
        lending_vocab, overdrawn="true", thin_file="unknown", guarantor="unknown",
        owns_property="false", employed_long="false", large_amount="true",
    )
    solver = ProbabilisticSolver(lending_rules)
    bounds = solver.bounds_for(parse_atom("decline(app1)"), facts, critical=True)
    assert 0.0 <= bounds.lower <= bounds.upper <= 1.0


# --------------------------------------------------------------------------
# Value of information
# --------------------------------------------------------------------------


def test_names_the_decisive_unknown(lending_vocab, lending_rules) -> None:
    """The point of the whole exercise: turn 'something is missing' into a question."""
    facts = make_facts(
        lending_vocab, overdrawn="true", thin_file="true", guarantor="unknown",
        owns_property="false", employed_long="false", large_amount="false",
    )
    solver = ProbabilisticSolver(lending_rules)
    answer = solver.answer(parse_atom("decline(app1)"), facts, critical=True)

    best = answer.best_question()
    assert best is not None
    assert str(best.atom) == "guarantor(app1)"
    assert best.decisive


def test_does_not_ask_about_irrelevant_unknowns() -> None:
    """A system that asks for information it will not use trains people to ignore it."""
    vocab = Vocabulary(
        [
            PredicateDecl("relevant", 1, "observable"),
            PredicateDecl("irrelevant", 1, "observable"),
            PredicateDecl("goal", 1, "derived"),
        ]
    )
    rb = RuleBase(vocab)
    rb.add(RuleRecord(rule=parse_rule("goal(A) <- relevant(A)."), rule_id="r1",
                      strength=0.9, status=RuleStatus.APPROVED))

    facts = FactStore(vocab)
    facts.assert_fact(parse_atom("relevant(x)"), Truth.UNKNOWN, source="t")
    facts.assert_fact(parse_atom("irrelevant(x)"), Truth.UNKNOWN, source="t")

    answer = ProbabilisticSolver(rb).answer(parse_atom("goal(x)"), facts, critical=True)
    asked = {str(c.atom) for c in answer.ask}
    assert "relevant(x)" in asked
    assert "irrelevant(x)" not in asked


def test_resolving_the_decisive_unknown_collapses_the_interval(
    lending_vocab, lending_rules
) -> None:
    facts = make_facts(
        lending_vocab, overdrawn="true", thin_file="true", guarantor="unknown",
        owns_property="false", employed_long="false", large_amount="false",
    )
    solver = ProbabilisticSolver(lending_rules)
    answer = solver.answer(parse_atom("decline(app1)"), facts, critical=True)
    best = answer.best_question()
    assert best.if_true.width == pytest.approx(0.0)
    assert best.if_false.width == pytest.approx(0.0)
    assert best.if_true.upper < best.if_false.lower, (
        "a guarantor should make declining less likely, not more"
    )


# --------------------------------------------------------------------------
# Answer semantics
# --------------------------------------------------------------------------


def test_uncovered_case_is_flagged_not_disguised() -> None:
    """Outcomes are never normalised, so ignorance stays visible.

    Normalising two near-zero outcome probabilities would turn "the rule base has
    nothing to say" into a confident-looking 50/50. That is the single most
    misleading thing this system could report, so it is tested.
    """
    vocab = Vocabulary(
        [
            PredicateDecl("fact_a", 1, "observable"),
            PredicateDecl("approve", 1, "derived"),
            PredicateDecl("decline", 1, "derived"),
        ]
    )
    rb = RuleBase(vocab)
    rb.add(RuleRecord(rule=parse_rule("approve(A) <- fact_a(A)."), rule_id="r1",
                      strength=0.9, status=RuleStatus.APPROVED))

    facts = FactStore(vocab)
    facts.assert_fact(parse_atom("fact_a(x)"), Truth.FALSE, source="t")

    answer = ProbabilisticSolver(rb).answer(
        parse_atom("approve(x)"), facts, critical=True,
        competing=[parse_atom("decline(x)")],
    )
    assert answer.is_uncovered
    assert all(b.upper < 0.2 for b in answer.outcomes.values())


def test_independence_assumption_is_stated(lending_vocab, lending_rules) -> None:
    """The main modelling lie must be visible, not buried in the arithmetic."""
    facts = make_facts(
        lending_vocab, guarantor="true", owns_property="true", employed_long="true",
        overdrawn="false", thin_file="false", large_amount="false",
    )
    answer = ProbabilisticSolver(lending_rules).answer(
        parse_atom("approve(app1)"), facts, critical=True
    )
    assert any("independent" in a for a in answer.assumptions)


def test_unknown_count_is_reported(lending_vocab, lending_rules) -> None:
    facts = make_facts(
        lending_vocab, overdrawn="true", thin_file="unknown", guarantor="unknown",
        owns_property="false", employed_long="false", large_amount="false",
    )
    answer = ProbabilisticSolver(lending_rules).answer(
        parse_atom("decline(app1)"), facts, critical=True
    )
    assert any("unknown" in a for a in answer.assumptions)
    assert len(answer.unknowns_considered) == 2
