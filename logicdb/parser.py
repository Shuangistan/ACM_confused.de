"""Parser for the rule surface syntax.

A hand-written recursive-descent parser rather than a grammar library, for one
reason: error messages. A proposing agent gets rules wrong constantly, and the
message it receives back is its only chance to correct itself. A parser that
says `line 4, col 23: expected ')' after argument 2 of overdrawn/2` gives the
agent -- and the human reviewing the queue -- something to act on. A generic
"parse error" does not.

Surface syntax:

    # comments run to end of line

    rule r0012 [p=0.86, origin=mined, status=approved, by=JD, at=2026-03-04]
      decline(A) <- overdrawn(A), no_guarantor(A), not waived(A).

    fact overdrawn(app_17) [source="application form, field 3"].
    fact guarantor(app_17) = unknown [source="not supplied"].

Metadata in brackets is optional and free-form; it is attached to the rule but
excluded from its canonical identity, so re-mining the same logic with a
different strength estimate does not re-open a settled approval.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from .syntax import Atom, Const, Literal, Rule, Term, Var


class ParseError(ValueError):
    """A syntax error with a location, and where possible a suggested fix."""

    def __init__(self, message: str, line: int, col: int, source_line: str = "") -> None:
        self.line, self.col = line, col
        detail = f"line {line}, col {col}: {message}"
        if source_line:
            caret = " " * (col - 1) + "^"
            detail = f"{detail}\n  {source_line}\n  {caret}"
        super().__init__(detail)


@dataclass
class ParsedRule:
    """A rule plus the metadata that travelled with it in the source text."""

    rule: Rule
    rule_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ParsedFact:
    atom: Atom
    state: str = "true"           # "true" | "false" | "unknown"
    metadata: dict[str, Any] = field(default_factory=dict)


# --------------------------------------------------------------------------
# Tokenizer
# --------------------------------------------------------------------------

_TOKEN_SPEC = [
    ("WS",       r"[ \t\r]+"),
    ("COMMENT",  r"#[^\n]*"),
    ("NEWLINE",  r"\n"),
    ("ARROW",    r"<-|:-"),
    # Dates must be matched before NUMBER, which would otherwise take 2026-03-04
    # as three separate numeric tokens.
    ("DATE",     r"\d{4}-\d{2}-\d{2}"),
    ("NUMBER",   r"-?\d+\.\d+|-?\d+"),
    ("STRING",   r'"(?:[^"\\]|\\.)*"'),
    ("NAME",     r"[A-Za-z_][A-Za-z0-9_]*"),
    ("LPAREN",   r"\("),
    ("RPAREN",   r"\)"),
    ("LBRACK",   r"\["),
    ("RBRACK",   r"\]"),
    ("COMMA",    r","),
    ("DOT",      r"\."),
    ("EQUALS",   r"="),
]
_MASTER = re.compile("|".join(f"(?P<{n}>{p})" for n, p in _TOKEN_SPEC))


@dataclass(frozen=True)
class Token:
    kind: str
    text: str
    line: int
    col: int


def tokenize(source: str) -> list[Token]:
    tokens: list[Token] = []
    line, line_start = 1, 0
    pos = 0
    while pos < len(source):
        m = _MASTER.match(source, pos)
        if m is None:
            col = pos - line_start + 1
            raise ParseError(
                f"unexpected character {source[pos]!r}", line, col,
                source.splitlines()[line - 1] if line <= len(source.splitlines()) else "",
            )
        kind = m.lastgroup
        text = m.group()
        if kind == "NEWLINE":
            line += 1
            line_start = m.end()
        elif kind not in ("WS", "COMMENT"):
            tokens.append(Token(kind, text, line, m.start() - line_start + 1))
        pos = m.end()
    tokens.append(Token("EOF", "", line, 1))
    return tokens


# --------------------------------------------------------------------------
# Parser
# --------------------------------------------------------------------------

_KEYWORDS = {"rule", "fact", "not", "unknown", "true", "false"}


class Parser:
    def __init__(self, source: str) -> None:
        self.source = source
        self.lines = source.splitlines()
        self.tokens = tokenize(source)
        self.pos = 0

    # -- token helpers -----------------------------------------------------
    @property
    def current(self) -> Token:
        return self.tokens[self.pos]

    def _source_line(self, tok: Token) -> str:
        return self.lines[tok.line - 1] if 0 < tok.line <= len(self.lines) else ""

    def error(self, message: str, tok: Token | None = None) -> ParseError:
        tok = tok or self.current
        return ParseError(message, tok.line, tok.col, self._source_line(tok))

    def accept(self, kind: str) -> Token | None:
        if self.current.kind == kind:
            tok = self.current
            self.pos += 1
            return tok
        return None

    def expect(self, kind: str, context: str = "") -> Token:
        tok = self.accept(kind)
        if tok is None:
            suffix = f" {context}" if context else ""
            got = "end of input" if self.current.kind == "EOF" else repr(self.current.text)
            raise self.error(f"expected {kind}{suffix}, got {got}")
        return tok

    # -- entry points ------------------------------------------------------
    def parse_program(self) -> tuple[list[ParsedRule], list[ParsedFact]]:
        rules: list[ParsedRule] = []
        facts: list[ParsedFact] = []
        while self.current.kind != "EOF":
            if self.current.kind == "NAME" and self.current.text == "rule":
                rules.append(self.parse_rule_statement())
            elif self.current.kind == "NAME" and self.current.text == "fact":
                facts.append(self.parse_fact_statement())
            else:
                # Bare clauses are allowed; `rule`/`fact` keywords are optional
                # sugar. An implication is a rule, a lone atom is a fact.
                start = self.pos
                parsed = self.parse_bare_clause()
                if isinstance(parsed, ParsedRule):
                    rules.append(parsed)
                else:
                    facts.append(parsed)
                if self.pos == start:  # defensive: never spin
                    raise self.error("could not make progress parsing statement")
        return rules, facts

    def parse_rule_statement(self) -> ParsedRule:
        self.expect("NAME")  # 'rule'
        rule_id = None
        if self.current.kind == "NAME" and self.current.text not in _KEYWORDS:
            nxt = self.tokens[self.pos + 1]
            # `rule r12 [..]` or `rule r12 head(..)` -- an id is a NAME not
            # immediately followed by '(' , which would make it the head atom.
            if nxt.kind != "LPAREN":
                rule_id = self.expect("NAME").text
        metadata = self.parse_metadata()
        head, body = self.parse_clause()
        return ParsedRule(rule=Rule(head, tuple(body)), rule_id=rule_id, metadata=metadata)

    def parse_fact_statement(self) -> ParsedFact:
        self.expect("NAME")  # 'fact'
        atom = self.parse_atom()
        state = "true"
        if self.accept("EQUALS"):
            tok = self.expect("NAME", "after '=' in a fact")
            if tok.text not in ("true", "false", "unknown"):
                raise self.error(
                    f"fact state must be true, false or unknown, got {tok.text!r}", tok
                )
            state = tok.text
        metadata = self.parse_metadata()
        self.expect("DOT", "to end the fact")
        return ParsedFact(atom=atom, state=state, metadata=metadata)

    def parse_bare_clause(self) -> ParsedRule | ParsedFact:
        head = self.parse_atom()
        if self.current.kind == "ARROW":
            self.pos += 1
            body = self.parse_body()
            metadata = self.parse_metadata()
            self.expect("DOT", "to end the rule")
            return ParsedRule(rule=Rule(head, tuple(body)), metadata=metadata)
        metadata = self.parse_metadata()
        if self.current.kind == "NAME":
            # `decline(A) overdrawn(A).` -- almost certainly a rule with the
            # arrow left out. Saying so is far more useful to a proposing agent
            # than complaining about a missing full stop.
            raise self.error(
                f"expected '<-' before {self.current.text!r}; a clause with a body "
                f"needs an arrow, as in 'head(A) <- {self.current.text}(A).'"
            )
        self.expect("DOT", "to end the fact")
        return ParsedFact(atom=head, metadata=metadata)

    # -- pieces ------------------------------------------------------------
    def parse_clause(self) -> tuple[Atom, list[Literal]]:
        head = self.parse_atom()
        if self.current.kind != "ARROW":
            metadata_hint = ""
            if self.current.kind == "DOT":
                metadata_hint = " (a rule needs a body; write it as a fact instead)"
            raise self.error(f"expected '<-' after the rule head{metadata_hint}")
        self.pos += 1
        body = self.parse_body()
        self.expect("DOT", "to end the rule")
        return head, body

    def parse_body(self) -> list[Literal]:
        literals = [self.parse_literal()]
        while self.accept("COMMA"):
            literals.append(self.parse_literal())
        return literals

    def parse_literal(self) -> Literal:
        negated = False
        if self.current.kind == "NAME" and self.current.text == "not":
            self.pos += 1
            negated = True
        return Literal(self.parse_atom(), negated)

    def parse_atom(self) -> Atom:
        tok = self.expect("NAME", "as a predicate name")
        if not tok.text[0].islower():
            raise self.error(
                f"predicate {tok.text!r} must start with a lowercase letter "
                f"(uppercase is reserved for variables)",
                tok,
            )
        if not self.accept("LPAREN"):
            return Atom(tok.text, ())
        terms: list[Term] = []
        if self.current.kind != "RPAREN":
            terms.append(self.parse_term(tok.text, len(terms) + 1))
            while self.accept("COMMA"):
                terms.append(self.parse_term(tok.text, len(terms) + 1))
        if self.current.kind != "RPAREN":
            raise self.error(
                f"expected ')' after argument {len(terms)} of {tok.text}/{len(terms)}"
            )
        self.pos += 1
        return Atom(tok.text, tuple(terms))

    def parse_term(self, predicate: str, index: int) -> Term:
        tok = self.current
        if tok.kind == "NAME":
            self.pos += 1
            if tok.text == "true":
                return Const(True)
            if tok.text == "false":
                return Const(False)
            return Var(tok.text) if tok.text[0].isupper() or tok.text[0] == "_" else Const(tok.text)
        if tok.kind == "DATE":
            self.pos += 1
            return Const(tok.text)
        if tok.kind == "NUMBER":
            self.pos += 1
            return Const(float(tok.text) if "." in tok.text else int(tok.text))
        if tok.kind == "STRING":
            self.pos += 1
            return Const(_unescape(tok.text[1:-1]))
        raise self.error(
            f"expected a term as argument {index} of {predicate}, got "
            f"{tok.text!r}"
        )

    def parse_metadata(self) -> dict[str, Any]:
        if not self.accept("LBRACK"):
            return {}
        meta: dict[str, Any] = {}
        while self.current.kind != "RBRACK":
            key = self.expect("NAME", "as a metadata key").text
            self.expect("EQUALS", f"after metadata key {key!r}")
            meta[key] = self.parse_metadata_value(key)
            if not self.accept("COMMA"):
                break
        self.expect("RBRACK", "to close the metadata block")
        return meta

    def parse_metadata_value(self, key: str) -> Any:
        tok = self.current
        if tok.kind == "STRING":
            self.pos += 1
            return _unescape(tok.text[1:-1])
        if tok.kind == "NUMBER":
            self.pos += 1
            return float(tok.text) if "." in tok.text else int(tok.text)
        if tok.kind == "DATE":
            self.pos += 1
            return tok.text
        if tok.kind == "NAME":
            self.pos += 1
            if tok.text == "true":
                return True
            if tok.text == "false":
                return False
            return tok.text
        raise self.error(f"expected a value for metadata key {key!r}", tok)


def _unescape(text: str) -> str:
    return text.replace('\\"', '"').replace("\\\\", "\\")


# --------------------------------------------------------------------------
# Convenience
# --------------------------------------------------------------------------


def parse_program(source: str) -> tuple[list[ParsedRule], list[ParsedFact]]:
    return Parser(source).parse_program()


def parse_rule(source: str) -> Rule:
    """Parse exactly one rule. Raises if the text holds anything else."""
    rules, facts = parse_program(source)
    if facts or len(rules) != 1:
        raise ParseError(
            f"expected exactly one rule, found {len(rules)} rule(s) and "
            f"{len(facts)} fact(s)",
            1, 1,
        )
    return rules[0].rule


def parse_atom(source: str) -> Atom:
    """Parse a single atom, e.g. a query goal."""
    parser = Parser(source.rstrip(". \n"))
    atom = parser.parse_atom()
    if parser.current.kind != "EOF":
        raise parser.error("unexpected trailing input after the atom")
    return atom


def format_rule(rule: Rule, metadata: dict[str, Any] | None = None,
                rule_id: str | None = None) -> str:
    """Render a rule back to surface syntax. Round-trips through the parser."""
    parts = ["rule"]
    if rule_id:
        parts.append(rule_id)
    if metadata:
        rendered = ", ".join(f"{k}={_format_value(v)}" for k, v in metadata.items())
        parts.append(f"[{rendered}]")
    header = " ".join(parts)
    return f"{header}\n  {rule}"


def _format_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    text = str(value)
    if re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", text):
        return text
    return '"' + text.replace("\\", "\\\\").replace('"', '\\"') + '"'


__all__ = [
    "ParseError",
    "ParsedFact",
    "ParsedRule",
    "Parser",
    "format_rule",
    "parse_atom",
    "parse_program",
    "parse_rule",
    "tokenize",
]
