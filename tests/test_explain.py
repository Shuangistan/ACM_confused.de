"""Tests for explanation rendering.

The point of these is narrow but important: the explanation must not overstate
what is known. An interval endpoint is produced by assuming the unknowns resolve
a particular way, and a reviewer reading the derivation for that endpoint must
be able to tell an assumption from a finding. Getting this wrong would put the
closed-world error back into the one artifact people actually read.
"""

from __future__ import annotations

import pytest

from logicdb.engine import evaluate
from logicdb.explain import Explainer, find_counterfactuals, phrase_atom
from logicdb.facts import FactStore, Truth
from logicdb.parser import parse_atom, parse_rule
from logicdb.probability import ProbabilisticSolver
from logicdb.program import RuleBase, RuleOrigin, RuleRecord, RuleStatus
from logicdb.syntax import PredicateDecl, Vocabulary

from .conftest import make_facts


# --------------------------------------------------------------------------
# Phrasing
# --------------------------------------------------------------------------


def test_phrase_template_is_used(lending_vocab) -> None:
    assert phrase_atom(parse_atom("overdrawn(app1)"), lending_vocab) == (
        "app1 is overdrawn"
    )


def test_falls_back_to_raw_atom_without_a_template() -> None:
    vocab = Vocabulary([PredicateDecl("bare", 1, "observable")])
    assert phrase_atom(parse_atom("bare(x)"), vocab) == "bare(x)"


# --------------------------------------------------------------------------
# Proof structure
# --------------------------------------------------------------------------


def test_proof_reaches_the_facts(lending_vocab, lending_rules) -> None:
    facts = make_facts(
        lending_vocab, overdrawn="true", thin_file="true", guarantor="false",
        owns_property="false", employed_long="false", large_amount="false",
    )
    ev = evaluate(lending_rules, facts, critical=True)
    tree = Explainer(lending_rules, lending_vocab).proof_tree(
        parse_atom("decline(app1)"), ev
    )
    leaves = {n.text for n in tree.walk() if n.kind == "fact"}
    assert "app1 is overdrawn" in leaves
    assert "app1 has a thin credit file" in leaves


def test_proof_cites_the_rules_and_who_approved_them(lending_vocab, lending_rules) -> None:
    facts = make_facts(
        lending_vocab, overdrawn="true", thin_file="true", guarantor="false",
        owns_property="false", employed_long="false", large_amount="false",
    )
    ev = evaluate(lending_rules, facts, critical=True)
    rendered = Explainer(lending_rules, lending_vocab).render_proof(
        parse_atom("decline(app1)"), ev
    )
    assert "by rule r0005" in rendered
    assert "approved by fixture" in rendered


def test_facts_carry_their_source_into_the_proof(lending_vocab, lending_rules) -> None:
    facts = FactStore(lending_vocab)
    facts.assert_fact(parse_atom("overdrawn(app1)"), Truth.TRUE,
                      source="application form, field 3")
    facts.assert_fact(parse_atom("thin_file(app1)"), Truth.TRUE,
                      source="credit bureau report")
    ev = evaluate(lending_rules, facts, critical=True)
    rendered = Explainer(lending_rules, lending_vocab).render_proof(
        parse_atom("high_risk(app1)"), ev
    )
    assert "application form, field 3" in rendered


# --------------------------------------------------------------------------
# Assumed vs established -- the important one
# --------------------------------------------------------------------------


def test_assumed_negation_is_not_reported_as_established(
    lending_vocab, lending_rules
) -> None:
    """An unknown fact assumed false must never read as 'confirmed not to hold'.

    The evaluation attached to an answer has its unknowns pinned to produce one
    end of the interval. Rendering that naively would tell a reviewer a fact was
    established when nobody ever supplied it -- turning an interval endpoint into
    an apparent finding.
    """
    facts = make_facts(
        lending_vocab, overdrawn="true", thin_file="true", guarantor="unknown",
        owns_property="false", employed_long="false", large_amount="false",
    )
    explainer = Explainer(lending_rules, lending_vocab)
    solver = ProbabilisticSolver(lending_rules)
    answer = solver.answer(parse_atom("decline(app1)"), facts, critical=True)

    rendered = explainer.explain_answer(answer, facts, critical=True)
    assert "ASSUMED, not established" in rendered
    assert "confirmed not to hold" not in rendered


def test_established_negation_is_reported_as_established(
    lending_vocab, lending_rules
) -> None:
    """The converse: a genuinely false fact must not be hedged into an assumption."""
    facts = make_facts(
        lending_vocab, overdrawn="true", thin_file="true", guarantor="false",
        owns_property="false", employed_long="false", large_amount="false",
    )
    explainer = Explainer(lending_rules, lending_vocab)
    answer = ProbabilisticSolver(lending_rules).answer(
        parse_atom("decline(app1)"), facts, critical=True
    )
    rendered = explainer.explain_answer(answer, facts, critical=True)
    assert "confirmed not to hold" in rendered
    assert "ASSUMED, not established" not in rendered


def test_uncertain_atoms_include_blocked_derivations(lending_vocab, lending_rules) -> None:
    """Uncertainty must be tracked through derived atoms, not just raw facts."""
    facts = make_facts(
        lending_vocab, overdrawn="true", thin_file="true", guarantor="unknown",
        owns_property="false", employed_long="false", large_amount="false",
    )
    uncertain = Explainer(lending_rules, lending_vocab).uncertain_atoms(
        facts, critical=True
    )
    assert parse_atom("guarantor(app1)") in uncertain
    assert parse_atom("mitigated(app1)") in uncertain, (
        "a derived atom blocked by an unknown is itself uncertain"
    )


# --------------------------------------------------------------------------
# Counterfactuals
# --------------------------------------------------------------------------


def test_counterfactual_finds_the_flipping_fact(lending_vocab, lending_rules) -> None:
    facts = make_facts(
        lending_vocab, overdrawn="true", thin_file="true", guarantor="false",
        owns_property="false", employed_long="false", large_amount="false",
    )
    cfs = find_counterfactuals(
        parse_atom("decline(app1)"), facts, lending_rules, critical=True
    )
    flipped = {str(c.atom) for c in cfs}
    assert "guarantor(app1)" in flipped or "thin_file(app1)" in flipped
    for cf in cfs:
        assert cf.conclusion_holds_now != cf.conclusion_holds_then


def test_counterfactuals_skip_unknowns(lending_vocab, lending_rules) -> None:
    """Unknowns belong to the question ranking, not the counterfactual list."""
    facts = make_facts(
        lending_vocab, overdrawn="true", thin_file="true", guarantor="unknown",
        owns_property="false", employed_long="false", large_amount="false",
    )
    cfs = find_counterfactuals(
        parse_atom("decline(app1)"), facts, lending_rules, critical=True
    )
    assert all(str(c.atom) != "guarantor(app1)" for c in cfs)


# --------------------------------------------------------------------------
# Provenance and governance disclosure
# --------------------------------------------------------------------------


def test_provenance_report_lists_facts_and_rules(lending_vocab, lending_rules) -> None:
    facts = make_facts(
        lending_vocab, overdrawn="true", thin_file="true", guarantor="false",
        owns_property="false", employed_long="false", large_amount="false",
    )
    ev = evaluate(lending_rules, facts, critical=True)
    report = Explainer(lending_rules, lending_vocab).provenance_report(
        parse_atom("high_risk(app1)"), ev
    )
    assert "overdrawn(app1)" in report
    assert "r0001" in report
    assert "approved by fixture" in report


def test_provenance_flags_an_unattributed_rule() -> None:
    """A rule nobody approved is a traceability gap and must be named as one."""
    vocab = Vocabulary(
        [PredicateDecl("a", 1, "observable"), PredicateDecl("g", 1, "derived")]
    )
    rb = RuleBase(vocab)
    rb.add(
        RuleRecord(
            rule=parse_rule("g(X) <- a(X)."), rule_id="r1", strength=0.8,
            status=RuleStatus.PROVISIONAL, origin=RuleOrigin.MINED,
        )
    )
    facts = FactStore(vocab)
    facts.assert_fact(parse_atom("a(x)"), Truth.TRUE, source="t")
    ev = evaluate(rb, facts, critical=False)
    report = Explainer(rb, vocab).provenance_report(parse_atom("g(x)"), ev)
    assert "GAP" in report
    assert "NOBODY" in report


def test_explanation_discloses_unapproved_rules() -> None:
    vocab = Vocabulary(
        [PredicateDecl("a", 1, "observable"), PredicateDecl("g", 1, "derived")]
    )
    rb = RuleBase(vocab)
    rb.add(
        RuleRecord(
            rule=parse_rule("g(X) <- a(X)."), rule_id="r1", strength=0.8,
            status=RuleStatus.PROVISIONAL, origin=RuleOrigin.MINED,
        )
    )
    facts = FactStore(vocab)
    facts.assert_fact(parse_atom("a(x)"), Truth.TRUE, source="t")
    answer = ProbabilisticSolver(rb).answer(parse_atom("g(x)"), facts, critical=False)
    rendered = Explainer(rb, vocab).explain_answer(answer, facts, critical=False)
    assert "provisional" in rendered.lower()


def test_uncovered_case_is_explained_as_absence_not_denial() -> None:
    """'No applicable logic' and 'no' are different answers and must read differently."""
    vocab = Vocabulary(
        [PredicateDecl("a", 1, "observable"), PredicateDecl("g", 1, "derived")]
    )
    rb = RuleBase(vocab)
    rb.add(
        RuleRecord(
            rule=parse_rule("g(X) <- a(X)."), rule_id="r1", strength=0.8,
            status=RuleStatus.APPROVED, origin=RuleOrigin.SEED, approved_by="x",
        )
    )
    facts = FactStore(vocab)
    facts.assert_fact(parse_atom("a(x)"), Truth.FALSE, source="t")
    answer = ProbabilisticSolver(rb).answer(parse_atom("g(x)"), facts, critical=True)
    rendered = Explainer(rb, vocab).explain_answer(answer, facts, critical=True)
    assert "absence of applicable logic" in rendered
