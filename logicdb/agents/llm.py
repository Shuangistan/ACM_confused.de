"""The LLM agent: Claude proposes rules and facts, and never decides anything.

This is where the project's central commitment is enforced in code rather than
asserted in prose. A language model is not deterministic and cannot be made so.
Rather than pretend otherwise, its nondeterminism is quarantined: the model may
draft candidate rules and extract candidate facts, and everything it produces
passes through schema validation, vocabulary validation, safety checking,
stratification checking, and a human approval gate before it can affect any
decision. Downstream of that gate the inference is pure Datalog -- reproducible,
explainable, and identical whether the rule was written by a person or drafted
by a model.

So the failure mode a language model is famous for is handled structurally. A
hallucinated predicate does not become a bad decision; it becomes a rejected
proposal that no reviewer ever sees, because `validate_proposals` discards it
before the queue. The worst a confident, wrong model can do here is waste a
reviewer's attention -- and the review queue shows the evidence behind every
proposal precisely so that attention is spent well.

Requires `ANTHROPIC_API_KEY`. The scripted agent is the default and must keep
working without one; this backend is opt-in.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any

from ..facts import FactStore, Truth
from ..packs import DomainPack
from ..parser import ParseError, parse_atom, parse_rule
from ..program import RuleOrigin
from ..syntax import Vocabulary
from .base import LabelledCase, RuleProposal

#: Claude Opus 5. Thinking is on by default on this model, and the raw chain of
#: thought is never returned -- which suits the design: the model's reasoning is
#: not the audit trail here. The proof produced by the engine is.
DEFAULT_MODEL = "claude-opus-5"

#: Schema-constrained output. The model cannot return prose where a rule is
#: expected, so a malformed proposal fails at the API boundary rather than
#: halfway through our parser.
RULE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "proposals": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "rule": {
                        "type": "string",
                        "description": (
                            "One rule in the surface syntax, e.g. "
                            "'decline(A) <- overdrawn(A), not guarantor(A).' "
                            "Must end with a full stop."
                        ),
                    },
                    "strength": {"type": "number"},
                    "rationale": {"type": "string"},
                    "citation": {
                        "type": "string",
                        "description": "Clause or section this encodes, if any.",
                    },
                },
                "required": ["rule", "strength", "rationale", "citation"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["proposals"],
    "additionalProperties": False,
}

FACT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "facts": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "atom": {"type": "string"},
                    "truth": {"type": "string", "enum": ["true", "false", "unknown"]},
                    "source": {"type": "string"},
                },
                "required": ["atom", "truth", "source"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["facts"],
    "additionalProperties": False,
}


class RefusalError(RuntimeError):
    """The model declined the request.

    Surfaced rather than swallowed: a silently empty proposal list would look
    identical to "no rules found", and those are very different states.
    """


#: Adaptive thinking and the `effort` parameter arrived with the 4.6
#: generation. Older models do not ignore them -- they reject the request with
#: a 400 -- so the request shape has to follow the model rather than assume the
#: newest one. Listed as prefixes because point releases share the behaviour.
LEGACY_MODEL_PREFIXES = ("claude-haiku-4-5", "claude-haiku-3", "claude-3")


def supports_effort(model: str) -> bool:
    """Whether `output_config.effort` and adaptive thinking may be sent."""
    return not model.startswith(LEGACY_MODEL_PREFIXES)


@dataclass
class LLMConfig:
    model: str = DEFAULT_MODEL
    max_tokens: int = 16000
    #: `high` is the sensible floor for work where a wrong rule costs a
    #: reviewer's time. Rule drafting is judgement, not transcription. Set to
    #: None on models that do not accept it; see `supports_effort`.
    effort: str | None = "high"
    #: Adaptive thinking, where the model supports it.
    thinking: bool = True
    max_proposals: int = 8
    #: Re-run a declined request on Anthropic's recommended fallback model.
    #: Off by default because the installed SDK does not type the parameter, so
    #: it has to be passed through `extra_body`.
    server_side_fallback: bool = False

    def __post_init__(self) -> None:
        if not supports_effort(self.model):
            self.effort = None
            self.thinking = False


SYSTEM_PROMPT = """\
You draft candidate logic for a governed rule database. You do not make \
decisions, and nothing you produce takes effect until a human approves it.

Write rules in this syntax:

    head(A) <- body_one(A), body_two(A), not body_three(A).

Constraints, all enforced mechanically after you reply -- a rule that breaks one \
is discarded before any human sees it:

* Use only the predicates listed below. There is no mechanism for introducing a \
new one, so a rule naming an undeclared predicate is simply dropped.
* Only `derived` predicates may appear as a head. `observable` predicates come \
from the world and cannot be concluded.
* Every variable in the head, and every variable inside a `not`, must also \
appear in a positive body literal.
* Variables are uppercase; predicates are lowercase; end each rule with a full \
stop.

Set `strength` to the probability the head holds when the body does. Be honest \
rather than confident: a reviewer sees this number next to the evidence, and an \
overstated strength is worse than a modest one because it survives review and \
then underperforms in use.

Prefer few, general, readable rules over many narrow ones. The output is a queue \
for a human reviewer, and a long queue of near-duplicates does not get read \
carefully -- which defeats the purpose of having a human review rules at all.\
"""


class LLMAgent:
    """Drafts rules and facts with Claude. Proposes only; never infers."""

    def __init__(
        self,
        pack: DomainPack,
        config: LLMConfig | None = None,
        agent_id: str | None = None,
        client: Any = None,
    ) -> None:
        self.pack = pack
        self.config = config or LLMConfig()
        agent_id = agent_id or f"llm-agent-{self.config.model}"
        #: Drafts that never became proposals, as (text, reason). A silent drop
        #: is indistinguishable from a model that drafted nothing, and the two
        #: call for opposite responses -- so what is discarded is recorded.
        self.skipped: list[tuple[str, str]] = []
        #: Drafts whose citation names nothing in the source, as (rule,
        #: claimed citation).
        self.unverified_citations: list[tuple[str, str]] = []
        self._agent_id = agent_id
        self._client = client

    @property
    def agent_id(self) -> str:
        return self._agent_id

    @property
    def client(self) -> Any:
        if self._client is None:
            import anthropic

            if not os.environ.get("ANTHROPIC_API_KEY"):
                raise RuntimeError(
                    "ANTHROPIC_API_KEY is not set. The scripted agent is the "
                    "default and needs no key; this backend is opt-in."
                )
            self._client = anthropic.Anthropic()
        return self._client

    # -- vocabulary description -------------------------------------------
    def _vocabulary_block(self) -> str:
        lines = ["Declared predicates:", ""]
        for decl in self.pack.vocabulary:
            kind = decl.kind
            phrase = f" -- {decl.phrase}" if decl.phrase else ""
            lines.append(f"  {decl.name}/{decl.arity}  [{kind}]{phrase}")
        lines += [
            "",
            f"Decision predicate: {self.pack.goal_predicate}",
            f"Competing outcomes: {', '.join(self.pack.outcomes)}",
        ]
        return "\n".join(lines)

    # -- the call ---------------------------------------------------------
    def _ask(self, prompt: str, schema: dict[str, Any]) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "model": self.config.model,
            "max_tokens": self.config.max_tokens,
            "system": SYSTEM_PROMPT,
            "messages": [{"role": "user", "content": prompt}],
            "output_config": {
                "format": {"type": "json_schema", "schema": schema},
            },
            "betas": ["structured-outputs-2025-11-13"],
        }
        # Both are 400 errors rather than ignored fields on pre-4.6 models, so
        # they are added only where they are accepted.
        if self.config.effort is not None:
            kwargs["output_config"]["effort"] = self.config.effort
        if self.config.thinking:
            kwargs["thinking"] = {"type": "adaptive"}
        if self.config.server_side_fallback:
            kwargs["betas"] = kwargs["betas"] + ["server-side-fallback-2026-07-01"]
            kwargs["extra_body"] = {"fallbacks": "default"}

        response = self.client.beta.messages.create(**kwargs)

        # Checked before touching `content`: a refusal can carry an empty
        # content list, and indexing it would raise something unrelated to the
        # actual cause.
        if getattr(response, "stop_reason", None) == "refusal":
            details = getattr(response, "stop_details", None)
            category = getattr(details, "category", None) if details else None
            raise RefusalError(
                f"the model declined this request (category: {category}). "
                f"No proposals were produced."
            )

        text = next(
            (b.text for b in response.content if getattr(b, "type", None) == "text"),
            "",
        )
        if not text:
            raise RuntimeError(
                f"no text content in response (stop_reason="
                f"{getattr(response, 'stop_reason', None)})"
            )
        return json.loads(text)

    # -- facts ------------------------------------------------------------
    def extract_facts(self, record: dict[str, Any], entity: str) -> FactStore:
        """Structured records go through the pack's declared mapping.

        Deliberately not delegated to the model. Where a deterministic table
        exists, using a model instead would trade an auditable mapping for an
        unauditable one and gain nothing.
        """
        return self.pack.facts_from(record, entity)

    def extract_facts_from_text(self, text: str, entity: str) -> FactStore:
        """Extract facts from unstructured text -- the case the table cannot handle.

        Anything that fails to parse, names an undeclared predicate, or targets
        the wrong entity is dropped. A fact silently attributed to the wrong
        applicant would be far worse than a missing one.
        """
        prompt = (
            f"{self._vocabulary_block()}\n\n"
            f"Extract facts about entity `{entity}` from the text below. Record a "
            f"fact as `unknown` when the text does not settle it -- do not infer "
            f"absence from silence. Cite the phrase each fact came from.\n\n"
            f"---\n{text}\n---"
        )
        payload = self._ask(prompt, FACT_SCHEMA)

        store = FactStore(self.pack.vocabulary)
        for item in payload.get("facts", []):
            try:
                atom = parse_atom(item["atom"])
            except (ParseError, ValueError):
                continue
            if self.pack.vocabulary.get(atom.signature) is None:
                continue
            if not atom.is_ground() or str(atom.terms[0]) != entity:
                continue
            try:
                store.assert_fact(
                    atom,
                    Truth(item["truth"]),
                    source=f"extracted by {self._agent_id}: {item['source']}",
                    asserted_by=self._agent_id,
                )
            except ValueError:
                continue
        return store

    # -- mining -----------------------------------------------------------
    def mine_rules(
        self,
        cases: list[LabelledCase],
        target: str,
        existing: set[str] | None = None,
    ) -> list[RuleProposal]:
        """Draft rules from labelled cases.

        The model is shown a sample and its outcomes and asked for hypotheses.
        Unlike the scripted miner, it cannot report support and precision --
        it has not counted anything. That gap is deliberate and visible: the
        proposals arrive without the evidence fields, so a reviewer can see they
        are reasoned hypotheses rather than measured regularities.
        """
        existing = existing or set()
        if not cases:
            return []

        sample = cases[:60]
        rendered = []
        for case in sample:
            true_facts = sorted(
                r.atom.predicate for r in case.facts if r.truth is Truth.TRUE
            )
            unknown = sorted(
                r.atom.predicate for r in case.facts if r.truth is Truth.UNKNOWN
            )
            line = f"  {case.entity}: {', '.join(true_facts) or '(none)'}"
            if unknown:
                line += f"  [unknown: {', '.join(unknown)}]"
            line += f"  -> {case.label}"
            rendered.append(line)

        prompt = (
            f"{self._vocabulary_block()}\n\n"
            f"Below are {len(sample)} cases. Each line lists the predicates that "
            f"hold, any that are unknown, and the recorded outcome.\n\n"
            + "\n".join(rendered)
            + f"\n\nPropose at most {self.config.max_proposals} rules concluding "
            f"`{target}`. Say in each rationale what pattern you think you are "
            f"seeing and how confident you are that it is a reason rather than a "
            f"coincidence. A reviewer will check these against the data, so an "
            f"honest 'this may be confounded' is more useful than a confident "
            f"claim."
        )
        payload = self._ask(prompt, RULE_SCHEMA)
        return self._to_proposals(payload, RuleOrigin.MINED, existing)

    # -- documents --------------------------------------------------------
    def extract_from_document(
        self, name: str, text: str, vocabulary: Vocabulary
    ) -> list[RuleProposal]:
        """Draft rules from policy prose, citing the clause each encodes.

        This is the case the scripted agent genuinely cannot do -- it reads
        machine-readable annotations, while this reads the document as written.
        The citation is the reviewer's lever: it points at a clause they can
        open and compare, which is what makes a drafted rule checkable rather
        than merely plausible.
        """
        prompt = (
            f"{self._vocabulary_block()}\n\n"
            f"The policy document `{name}` follows. Encode as rules only those "
            f"clauses that state a condition and a consequence. Cite the section "
            f"for each.\n\n"
            f"Leave a clause unencoded when it states a principle a rule cannot "
            f"capture -- for instance a prohibition on relying on a protected "
            f"characteristic, which no rule can express, since a rule can state a "
            f"condition but not the absence of a hidden correlation. Say which "
            f"clauses you left out and why; that list is as useful to a reviewer "
            f"as the rules themselves.\n\n"
            f"---\n{text}\n---"
        )
        payload = self._ask(prompt, RULE_SCHEMA)
        return self._to_proposals(
            payload, RuleOrigin.DOCUMENT, set(), source_text=text
        )

    # -- shared ------------------------------------------------------------
    @staticmethod
    def _citation_is_real(citation: str, source_text: str) -> bool:
        """Whether a drafted citation actually points at the source.

        Deliberately generous -- a trailing gloss such as `s4.1
        (contrapositive)` still counts, because the reviewer can find the
        clause. What it rejects is a citation naming no locatable text at all,
        which is the failure mode observed in practice: rules attributed to
        "implicit in screening practice" rather than to anything written.
        """
        anchor = citation.split("(")[0].strip().strip(".,;:")
        if not anchor:
            return False
        return anchor.lower() in source_text.lower()

    def _to_proposals(
        self,
        payload: dict[str, Any],
        origin: RuleOrigin,
        existing: set[str],
        source_text: str | None = None,
    ) -> list[RuleProposal]:
        """Parse drafted rules, discarding anything malformed or unsafe.

        Parse and safety failures are dropped here; vocabulary failures are left
        to `validate_proposals`, which reports them so a reviewer can see what
        the model tried to invent.
        """
        self.skipped = []
        self.unverified_citations = []
        out: list[RuleProposal] = []
        for item in payload.get("proposals", []):
            text = str(item.get("rule", "")).strip()
            if not text.endswith("."):
                text += "."
            try:
                rule = parse_rule(text)
            except (ParseError, ValueError) as exc:
                self.skipped.append((text, f"unparseable or unsafe: {exc}"))
                continue
            if rule.canonical_key() in existing:
                self.skipped.append((text, "already in the rule base"))
                continue

            strength = float(item.get("strength", 0.7))
            strength = min(max(strength, 0.01), 1.0)
            citation = str(item.get("citation", "")).strip() or None

            verified = True
            if citation and source_text is not None:
                verified = self._citation_is_real(citation, source_text)
                if not verified:
                    self.unverified_citations.append((text, citation))

            out.append(
                RuleProposal(
                    rule=rule,
                    origin=origin,
                    strength=round(strength, 3),
                    rationale=str(item.get("rationale", "")).strip()
                    or "no rationale given",
                    proposed_by=self._agent_id,
                    source_citation=(
                        citation if origin is RuleOrigin.DOCUMENT else None
                    ),
                    citation_verified=verified,
                )
            )
            if len(out) >= self.config.max_proposals:
                break
        return out


__all__ = ["DEFAULT_MODEL", "LEGACY_MODEL_PREFIXES", "LLMAgent", "LLMConfig",
           "RefusalError", "supports_effort"]
