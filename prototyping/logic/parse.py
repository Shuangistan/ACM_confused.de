"""Surface syntax, both directions.

    suggest_umbrella(C) <- going_out(C), raining(C), not has_umbrella(C).

Rules are read from three places — a file, an agent's JSON, and a person typing
into a review queue — so the parser's errors are read by people at least as
often as its output is read by the engine.

One tokenizer detail, because it was a real bug: DATE must be tried before
NUMBER, or `2024-03-04` lexes as three integers and a rule that looked fine
quietly means something else.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .syntax import Atom, Const, Literal, Rule, Term, Var


class ParseError(ValueError):
    """Malformed input, with the position that caused it."""


@dataclass(frozen=True)
class Token:
    kind: str
    text: str
    pos: int


_SPEC = [
    ("WS", r"[ \t\r\n]+"),
    ("COMMENT", r"\#[^\n]*"),
    ("ARROW", r"<-|:-"),
    ("DATE", r"\d{4}-\d{2}-\d{2}"),
    ("NUMBER", r"-?\d+\.\d+|-?\d+"),
    ("STRING", r'"[^"]*"|\'[^\']*\''),
    ("NOT", r"\bnot\b"),
    ("NAME", r"[A-Za-z_][A-Za-z0-9_]*"),
    ("LPAREN", r"\("), ("RPAREN", r"\)"), ("COMMA", r","), ("DOT", r"\."),
]
_MASTER = re.compile("|".join(f"(?P<{k}>{v})" for k, v in _SPEC))


def tokenize(text: str) -> list[Token]:
    out, pos = [], 0
    while pos < len(text):
        m = _MASTER.match(text, pos)
        if m is None:
            raise ParseError(f"unexpected character {text[pos]!r} at position {pos}")
        if m.lastgroup not in ("WS", "COMMENT"):
            out.append(Token(m.lastgroup or "", m.group(), pos))
        pos = m.end()
    return out


class _Parser:
    def __init__(self, tokens: list[Token], source: str) -> None:
        self.tokens, self.source, self.i = tokens, source, 0

    def peek(self):
        return self.tokens[self.i] if self.i < len(self.tokens) else None

    def at(self, kind: str) -> bool:
        t = self.peek()
        return t is not None and t.kind == kind

    def take(self, kind: str, what: str) -> Token:
        t = self.peek()
        if t is None:
            raise ParseError(f"expected {what}; input ended: {self.source!r}")
        if t.kind != kind:
            raise ParseError(
                f"expected {what} at position {t.pos}, got {t.text!r} in {self.source!r}")
        self.i += 1
        return t

    def term(self) -> Term:
        t = self.peek()
        if t is None:
            raise ParseError(f"expected a term; input ended: {self.source!r}")
        self.i += 1
        if t.kind == "STRING":
            return Const(t.text[1:-1])
        if t.kind == "DATE":
            return Const(t.text)
        if t.kind == "NUMBER":
            return Const(float(t.text) if "." in t.text else int(t.text))
        if t.kind == "NAME":
            if t.text[0].isupper():
                return Var(t.text)
            if t.text in ("true", "false"):
                return Const(t.text == "true")
            return Const(t.text)
        raise ParseError(f"expected a term at position {t.pos}, got {t.text!r}")

    def atom(self) -> Atom:
        name = self.take("NAME", "a predicate name")
        if name.text[0].isupper():
            raise ParseError(
                f"predicate {name.text!r} at position {name.pos} starts with a "
                f"capital, which marks a variable; predicates are lowercase")
        if not self.at("LPAREN"):
            return Atom(name.text, ())
        self.take("LPAREN", "'('")
        terms = []
        if not self.at("RPAREN"):
            terms.append(self.term())
            while self.at("COMMA"):
                self.take("COMMA", "','")
                terms.append(self.term())
        self.take("RPAREN", "')'")
        return Atom(name.text, tuple(terms))

    def literal(self) -> Literal:
        if self.at("NOT"):
            self.take("NOT", "'not'")
            return Literal(self.atom(), negated=True)
        return Literal(self.atom(), negated=False)

    def rule(self) -> Rule:
        head = self.atom()
        body = []
        if self.at("ARROW"):
            self.take("ARROW", "'<-'")
            body.append(self.literal())
            while self.at("COMMA"):
                self.take("COMMA", "','")
                body.append(self.literal())
        # A missing final period is the commonest thing a model gets wrong and
        # changes no meaning, so it is tolerated rather than costing a proposal.
        if self.at("DOT"):
            self.take("DOT", "'.'")
        return Rule(head, tuple(body))


def parse_rule(text: str) -> Rule:
    p = _Parser(tokenize(text), text)
    rule = p.rule()
    if p.peek() is not None:
        t = p.peek()
        raise ParseError(f"unexpected {t.text!r} at position {t.pos} after the rule")
    return rule


def parse_atom(text: str) -> Atom:
    p = _Parser(tokenize(text), text)
    atom = p.atom()
    if p.at("DOT"):
        p.take("DOT", "'.'")
    if p.peek() is not None:
        t = p.peek()
        raise ParseError(f"unexpected {t.text!r} at position {t.pos} after the atom")
    return atom


def parse_program(text: str) -> list[Rule]:
    rules, tokens, start = [], tokenize(text), 0
    for i, t in enumerate(tokens):
        if t.kind == "DOT":
            chunk = tokens[start:i + 1]
            if chunk:
                rules.append(_Parser(chunk, text).rule())
            start = i + 1
    if tokens[start:]:
        raise ParseError("the last rule is missing its closing '.'")
    return rules


__all__ = ["ParseError", "Token", "parse_atom", "parse_program", "parse_rule",
           "tokenize"]
