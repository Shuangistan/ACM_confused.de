"""The properties the decisions rest on.

Not coverage. Each of these is something the rest of the system assumes, and
most of them exist because it once broke. Where that is the case the comment
says so, because a test whose reason is forgotten gets deleted the first time it
is inconvenient.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from logic.engine import Engine, StratificationError  # noqa: E402
from logic.facts import FactStore, Truth  # noqa: E402
from logic.parse import ParseError, parse_atom, parse_program, parse_rule  # noqa: E402
from logic.syntax import UnsafeRule, same_rule  # noqa: E402


def rules(text: str):
    return [(f"r{i}", r) for i, r in enumerate(parse_program(text), start=1)]


def facts(**truths) -> FactStore:
    store = FactStore()
    for name, value in truths.items():
        store.assert_fact(parse_atom(f"{name}(c)"),
                          Truth.TRUE if value is True else
                          Truth.FALSE if value is False else Truth.UNKNOWN)
    return store


UMBRELLA = """
  will_get_wet(C)       <- going_out(C), raining(C), not has_umbrella(C).
  suggest_umbrella(C)   <- will_get_wet(C).
  suggest_stay_home(C)  <- will_get_wet(C), storm_warning(C).
"""


# ---------------------------------------------------------------------------
# Unknowns survive negation
#
# The defect shipped twice in the previous build. Facts come from a model
# reading free text: if it did not mention an umbrella, that does not mean
# there is none. Under a closed world the silence becomes `false`, the negation
# succeeds, and a rule fires against a person on evidence nobody gathered.
# ---------------------------------------------------------------------------
def test_an_explicit_absence_lets_the_negation_through() -> None:
    d = Engine().evaluate(rules(UMBRELLA),
                          facts(going_out=True, raining=True, has_umbrella=False))
    assert d.holds(parse_atom("will_get_wet(c)"))


def test_an_unmentioned_fact_blocks_the_rule_instead() -> None:
    d = Engine().evaluate(rules(UMBRELLA), facts(going_out=True, raining=True))
    assert d.truth_of(parse_atom("will_get_wet(c)")) is Truth.UNKNOWN
    assert not d.conclusions()


def test_ignorance_propagates_one_step_further() -> None:
    """A blocked head must read UNKNOWN to whatever reads it, not false."""
    d = Engine().evaluate(
        rules(UMBRELLA + "  fine(C) <- going_out(C), not will_get_wet(C)."),
        facts(going_out=True, raining=True))
    assert d.truth_of(parse_atom("fine(c)")) is Truth.UNKNOWN


def test_a_derived_predicate_that_cannot_hold_is_false_not_unknown() -> None:
    """Exhaustion is knowledge; silence is not."""
    d = Engine().evaluate(rules(UMBRELLA),
                          facts(going_out=True, raining=False, has_umbrella=False))
    assert d.truth_of(parse_atom("will_get_wet(c)")) is Truth.FALSE


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------
def test_rule_order_cannot_change_the_answer() -> None:
    base = rules(UMBRELLA)
    given = facts(going_out=True, raining=True, has_umbrella=False,
                  storm_warning=True)
    first = {str(a) for a in Engine().evaluate(base, given).conclusions()}
    second = {str(a) for a in
              Engine().evaluate(list(reversed(base)), given).conclusions()}
    assert first == second


def test_the_same_inputs_give_the_same_answer_every_time() -> None:
    base, given = rules(UMBRELLA), facts(going_out=True, raining=True,
                                         has_umbrella=False)
    runs = {tuple(str(a) for a in Engine().evaluate(base, given).conclusions())
            for _ in range(20)}
    assert len(runs) == 1


def test_two_conclusions_may_hold_at_once() -> None:
    """Not a conflict. Reporting both is honest; picking one silently is not."""
    d = Engine().evaluate(rules(UMBRELLA),
                          facts(going_out=True, raining=True, has_umbrella=False,
                                storm_warning=True))
    assert {a.predicate for a in d.conclusions()} == {
        "will_get_wet", "suggest_umbrella", "suggest_stay_home"}


# ---------------------------------------------------------------------------
# Termination and stratification
# ---------------------------------------------------------------------------
def test_recursion_reaches_a_fixpoint() -> None:
    d = Engine().evaluate(
        rules("reaches(X, Y) <- edge(X, Y).\n"
              "reaches(X, Z) <- reaches(X, Y), edge(Y, Z)."),
        FactStore([]))
    store = FactStore()
    for a, b in (("a", "b"), ("b", "c"), ("c", "d")):
        store.assert_fact(parse_atom(f'edge("{a}", "{b}")'))
    d = Engine().evaluate(
        rules("reaches(X, Y) <- edge(X, Y).\n"
              "reaches(X, Z) <- reaches(X, Y), edge(Y, Z)."), store)
    assert d.holds(parse_atom('reaches("a", "d")'))


def test_negation_inside_a_cycle_is_refused() -> None:
    """Such a program has two meanings, so it must not have one by accident."""
    with pytest.raises(StratificationError):
        Engine().evaluate(rules("p(C) <- q(C), not r(C).\n"
                                "r(C) <- q(C), not p(C)."),
                          facts(q=True))


# ---------------------------------------------------------------------------
# Identity — what approve-once rests on
# ---------------------------------------------------------------------------
def test_renaming_and_reordering_is_the_same_rule() -> None:
    a = parse_rule("decline(X) <- overdrawn(X), not guarantor(X).")
    b = parse_rule("decline(A) <- not guarantor(A), overdrawn(A).")
    assert same_rule(a, b)


def test_negation_is_part_of_a_rule_s_content() -> None:
    assert not same_rule(parse_rule("d(A) <- o(A), not g(A)."),
                         parse_rule("d(A) <- o(A), g(A)."))


def test_argument_order_is_part_of_it_too() -> None:
    assert not same_rule(parse_rule("knows(X, Y) <- met(X, Y)."),
                         parse_rule("knows(X, Y) <- met(Y, X)."))


def test_identity_survives_a_round_trip_through_text() -> None:
    rule = parse_rule("decline(A) <- overdrawn(A), not guarantor(A).")
    assert parse_rule(str(rule)).canonical_key() == rule.canonical_key()


# ---------------------------------------------------------------------------
# Safety — an agent writing rules will produce these
# ---------------------------------------------------------------------------
def test_an_unbound_head_variable_is_refused() -> None:
    with pytest.raises(UnsafeRule):
        parse_rule("decline(A) <- overdrawn(B).")


def test_a_variable_only_inside_a_negation_is_refused() -> None:
    with pytest.raises(UnsafeRule):
        parse_rule("decline(A) <- overdrawn(A), not guarantor(B).")


def test_an_all_negative_body_is_refused() -> None:
    with pytest.raises(UnsafeRule):
        parse_rule("decline(A) <- not guarantor(A).")


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------
def test_a_date_is_one_token_not_three_numbers() -> None:
    """A real bug: `2024-03-04` lexed as 2024, -03, -04."""
    atom = parse_atom("opened(c, 2024-03-04)")
    assert atom.terms[1].value == "2024-03-04"


def test_a_missing_final_period_is_tolerated() -> None:
    """The commonest thing a model gets wrong, and it changes no meaning."""
    assert parse_rule("p(X) <- q(X)") == parse_rule("p(X) <- q(X).")


def test_trailing_junk_is_an_error_not_silently_dropped() -> None:
    with pytest.raises(ParseError):
        parse_rule("p(X) <- q(X). and then some")
