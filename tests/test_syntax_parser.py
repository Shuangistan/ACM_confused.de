"""Tests for the rule AST, canonical hashing and the parser.

The canonical-hashing tests are the important ones. Approve-once is the property
that makes rule-level oversight cheaper than decision-level oversight, and it
rests entirely on two logically identical rules hashing to the same key. If that
breaks, the system keeps asking humans to re-approve things they already
approved, and the whole argument for the design collapses -- quietly, because
everything still appears to work.
"""

from __future__ import annotations

import pytest

from logicdb.parser import (
    ParseError,
    format_rule,
    parse_atom,
    parse_program,
    parse_rule,
)
from logicdb.syntax import (
    Atom,
    Const,
    Literal,
    PredicateDecl,
    Rule,
    Var,
    Vocabulary,
    rules_equivalent,
)


# --------------------------------------------------------------------------
# Parsing
# --------------------------------------------------------------------------


def test_parses_rule_with_metadata() -> None:
    rules, facts = parse_program(
        'rule r0012 [p=0.86, origin=mined, status=approved, by=JD, at=2026-03-04]\n'
        "  decline(A) <- overdrawn(A), no_guarantor(A), not waived(A)."
    )
    assert not facts
    (parsed,) = rules
    assert parsed.rule_id == "r0012"
    assert parsed.metadata == {
        "p": 0.86, "origin": "mined", "status": "approved",
        "by": "JD", "at": "2026-03-04",
    }
    assert len(parsed.rule.body) == 3
    assert parsed.rule.body[2].negated


def test_parses_three_fact_states() -> None:
    _, facts = parse_program(
        'fact overdrawn(app17) [source="form field 3"].\n'
        "fact guarantor(app17) = unknown.\n"
        "fact waived(app17) = false."
    )
    assert [f.state for f in facts] == ["true", "unknown", "false"]
    assert facts[0].metadata["source"] == "form field 3"


def test_bare_clauses_need_no_keyword() -> None:
    rules, facts = parse_program("decline(A) <- overdrawn(A).\noverdrawn(app1).")
    assert len(rules) == 1 and len(facts) == 1


def test_numbers_and_strings_as_terms() -> None:
    rule = parse_rule('flag(A) <- amount(A, 5000), band(A, "high risk").')
    assert rule.body[0].atom.terms[1] == Const(5000)
    assert rule.body[1].atom.terms[1] == Const("high risk")


def test_round_trips_through_formatter() -> None:
    source = "decline(A) <- overdrawn(A), not waived(A)."
    assert parse_rule(format_rule(parse_rule(source))) == parse_rule(source)


@pytest.mark.parametrize(
    "source, fragment",
    [
        ("decline(A <- overdrawn(A).", "expected ')'"),
        ("Decline(A) <- overdrawn(A).", "must start with a lowercase letter"),
        ("decline(A) <- .", "expected NAME"),
        ("decline(A) overdrawn(A).", "expected '<-'"),
    ],
)
def test_syntax_errors_are_actionable(source: str, fragment: str) -> None:
    """Error text is the only feedback a proposing agent gets; it must be usable."""
    with pytest.raises(ParseError) as exc:
        parse_rule(source)
    assert fragment in str(exc.value)
    assert "line 1" in str(exc.value)


def test_error_points_at_the_offending_column() -> None:
    with pytest.raises(ParseError) as exc:
        parse_rule("decline(A) <- overdrawn(A, .")
    rendered = str(exc.value)
    assert "^" in rendered, "the caret line locates the problem for the reader"


# --------------------------------------------------------------------------
# Safety
# --------------------------------------------------------------------------


def test_unsafe_head_variable_rejected() -> None:
    with pytest.raises(ValueError, match="unsafe rule.*head variable"):
        parse_rule("decline(A) <- overdrawn(B).")


def test_unsafe_negated_variable_rejected() -> None:
    """A variable appearing only under negation ranges over nothing."""
    with pytest.raises(ValueError, match="unsafe rule"):
        parse_rule("decline(A) <- overdrawn(A), not waived(B).")


def test_safe_rule_with_negation_accepted() -> None:
    rule = parse_rule("decline(A) <- overdrawn(A), not waived(A).")
    assert len(rule.body) == 2


# --------------------------------------------------------------------------
# Canonical hashing -- the approve-once property
# --------------------------------------------------------------------------


def test_variable_renaming_does_not_change_identity() -> None:
    a = parse_rule("decline(A) <- overdrawn(A), thin_file(A).")
    b = parse_rule("decline(X) <- overdrawn(X), thin_file(X).")
    assert rules_equivalent(a, b)


def test_body_reordering_does_not_change_identity() -> None:
    a = parse_rule("decline(A) <- overdrawn(A), thin_file(A).")
    b = parse_rule("decline(A) <- thin_file(A), overdrawn(A).")
    assert rules_equivalent(a, b)


def test_renaming_and_reordering_together() -> None:
    """The case that actually happens: an agent re-proposes the same logic.

    Without this, the same rule returns to the review queue wearing different
    clothes, a human approves it again, and the amortisation argument -- one
    review, many decisions -- silently becomes one review per proposal.
    """
    a = parse_rule("decline(A) <- overdrawn(A), thin_file(A), not waived(A).")
    b = parse_rule("decline(Z) <- not waived(Z), thin_file(Z), overdrawn(Z).")
    assert rules_equivalent(a, b)
    assert a.canonical_key() == b.canonical_key()


def test_different_variable_structure_is_a_different_rule() -> None:
    """Sharing a variable is a real difference, not cosmetic."""
    shared = parse_rule("linked(A, A) <- knows(A, A).")
    distinct = parse_rule("linked(A, B) <- knows(A, B).")
    assert not rules_equivalent(shared, distinct)


def test_negation_changes_identity() -> None:
    a = parse_rule("decline(A) <- overdrawn(A), waived(A).")
    b = parse_rule("decline(A) <- overdrawn(A), not waived(A).")
    assert not rules_equivalent(a, b)


def test_different_constants_are_different_rules() -> None:
    a = parse_rule("flag(A) <- amount(A, 5000).")
    b = parse_rule("flag(A) <- amount(A, 9000).")
    assert not rules_equivalent(a, b)


def test_canonical_key_is_stable_across_calls() -> None:
    """Must not depend on `hash()`, which is randomised per process."""
    rule = parse_rule("decline(A) <- overdrawn(A), thin_file(A).")
    assert len({rule.canonical_key() for _ in range(50)}) == 1


def test_canonical_keys_do_not_collide_across_many_rules() -> None:
    rules = [
        parse_rule(f"decline(A) <- p{i}(A), q{j}(A).")
        for i in range(12)
        for j in range(12)
    ]
    assert len({r.canonical_key() for r in rules}) == len(rules)


# --------------------------------------------------------------------------
# Vocabulary
# --------------------------------------------------------------------------


@pytest.fixture
def vocab() -> Vocabulary:
    return Vocabulary(
        [
            PredicateDecl("overdrawn", 1, "observable"),
            PredicateDecl("waived", 1, "observable"),
            PredicateDecl("decline", 1, "derived"),
        ]
    )


def test_undeclared_predicate_is_reported(vocab: Vocabulary) -> None:
    problems = vocab.validate_rule(parse_rule("decline(A) <- hallucinated(A)."))
    assert any("undeclared predicate in body" in p for p in problems)


def test_rule_may_not_conclude_an_observable(vocab: Vocabulary) -> None:
    """Otherwise 'observed' and 'inferred' blur, and provenance with them."""
    problems = vocab.validate_rule(parse_rule("overdrawn(A) <- waived(A)."))
    assert any("declared observable" in p for p in problems)


def test_valid_rule_has_no_problems(vocab: Vocabulary) -> None:
    assert vocab.validate_rule(parse_rule("decline(A) <- overdrawn(A).")) == []


def test_duplicate_declaration_rejected() -> None:
    with pytest.raises(ValueError, match="duplicate predicate"):
        Vocabulary([PredicateDecl("p", 1), PredicateDecl("p", 1)])


def test_same_name_different_arity_is_a_different_predicate() -> None:
    v = Vocabulary([PredicateDecl("amount", 1), PredicateDecl("amount", 2)])
    assert len(v) == 2
