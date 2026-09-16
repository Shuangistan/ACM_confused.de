"""A governed logic database for agentic decision support.

The idea in one line: take the decision logic out of the model and put it in an
inspectable database, then put human oversight on the logic rather than on
individual decisions.

Three properties follow, and they are the reasons for every design choice here:

* **Determinism** -- inference is a terminating Datalog fixpoint. Same facts,
  same rules, same answer, every time.
* **Explainability** -- an answer arrives with the proof that produced it, not a
  narrative written afterwards to match it.
* **Oversight that binds** -- a critical decision is structurally incapable of
  resting on a rule no human has approved.

The architectural commitment underneath all three: **the agent may propose, but
only the engine may infer.** An LLM extracts candidate facts and drafts candidate
rules; it never produces a decision. Nondeterminism is quarantined upstream of a
human approval gate, and everything downstream of that gate is reproducible.
"""

from .engine import Engine, Evaluation, evaluate
from .facts import FactRecord, FactStore, Truth
from .parser import ParseError, format_rule, parse_atom, parse_program, parse_rule
from .program import (
    RuleBase,
    RuleOrigin,
    RuleRecord,
    RuleStatus,
    StratificationError,
)
from .syntax import Atom, Const, Literal, PredicateDecl, Rule, Var, Vocabulary

__all__ = [
    "Atom",
    "Const",
    "Engine",
    "Evaluation",
    "FactRecord",
    "FactStore",
    "Literal",
    "ParseError",
    "PredicateDecl",
    "Rule",
    "RuleBase",
    "RuleOrigin",
    "RuleRecord",
    "RuleStatus",
    "StratificationError",
    "Truth",
    "Var",
    "Vocabulary",
    "evaluate",
    "format_rule",
    "parse_atom",
    "parse_program",
    "parse_rule",
]

__version__ = "0.1.0"
