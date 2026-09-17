"""Domain packs: the only channel through which domain knowledge enters.

Everything specific to lending, benefits or anything else arrives here as data.
If code in `logicdb/` ever needs to know what `overdrawn` means, the abstraction
has failed and the fix belongs in this file rather than as a special case
upstream.

A pack declares five things:

* **Vocabulary** -- the predicates that exist. Closed, so a proposing agent
  cannot invent one.
* **Decision** -- the goal predicate and its competing outcomes.
* **Criticality** -- when a query is consequential enough to need online review.
  This is itself data, and reviewable, which it should be since it decides what
  gets scrutinised.
* **Feature mappings** -- how a raw record becomes ground facts. Declarative so
  extraction is auditable: you can read off exactly which field produced which
  fact, and what happens when the field is absent.
* **Seed rules and documents** -- starting logic, and the written sources an
  agent may extract further rules from.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .facts import FactStore, Truth
from .governance import CriticalityPolicy, CriticalityRule
from .parser import parse_rule
from .syntax import Atom, Const, PredicateDecl, Rule, Vocabulary

PACK_DIR = Path(__file__).parent.parent / "packs"


@dataclass(frozen=True)
class FeatureMapping:
    """How one raw field becomes a ground fact.

    `missing` is the field that matters. A record with no value for
    `guarantor` must produce `unknown`, not `false` -- the difference between
    "we asked and there is none" and "nobody filled this in" is the difference
    the whole three-valued design exists to preserve, and it is lost here if a
    pack author is careless. So it is explicit per mapping rather than defaulted
    globally.
    """

    predicate: str
    source: str
    op: str = "truthy"
    value: Any = None
    #: Truth to record when the source field is absent or None.
    missing: str = "unknown"
    description: str = ""

    def evaluate(self, record: dict[str, Any]) -> Truth:
        if self.source not in record or record[self.source] is None:
            return Truth(self.missing)
        actual = record[self.source]
        try:
            result = self._compare(actual)
        except TypeError:
            return Truth(self.missing)
        return Truth.TRUE if result else Truth.FALSE

    def _compare(self, actual: Any) -> bool:
        op, expected = self.op, self.value
        if op == "truthy":
            return bool(actual)
        if op == "eq":
            return actual == expected
        if op == "ne":
            return actual != expected
        if op == "in":
            return actual in expected
        if op == "not_in":
            return actual not in expected
        if op == "gt":
            return actual > expected
        if op == "gte":
            return actual >= expected
        if op == "lt":
            return actual < expected
        if op == "lte":
            return actual <= expected
        if op == "contains":
            return expected in actual
        raise ValueError(f"unknown feature op {op!r} for predicate {self.predicate}")


@dataclass
class SeedRule:
    text: str
    strength: float = 0.8
    origin: str = "seed"
    cites: str | None = None
    approved_by: str | None = None

    def rule(self) -> Rule:
        return parse_rule(self.text)


@dataclass
class DomainPack:
    """Everything the framework needs to run a domain, and nothing more."""

    domain_id: str
    name: str
    description: str
    goal_predicate: str
    outcomes: list[str]
    vocabulary: Vocabulary
    criticality: CriticalityPolicy
    features: list[FeatureMapping] = field(default_factory=list)
    seed_rules: list[SeedRule] = field(default_factory=list)
    documents: list[Path] = field(default_factory=list)
    entity_field: str = "id"
    context_fields: list[str] = field(default_factory=list)
    label_field: str | None = None
    #: Maps a raw label value onto one of `outcomes`, when they differ.
    label_map: dict[str, str] = field(default_factory=dict)
    source: str | None = None
    licence: str | None = None
    root: Path | None = None
    #: Harm, reversibility and rights-impact for decisions of this kind. They
    #: are properties of the decision *type*, not of the individual case --
    #: nothing about one applicant makes a screening rejection more or less
    #: reversible -- so they are declared once by the pack author rather than
    #: guessed per case. `rights_field` names a context field where the rights
    #: question genuinely does vary case by case.
    oversight: dict[str, Any] = field(default_factory=dict)

    # -- extraction --------------------------------------------------------
    def facts_from(self, record: dict[str, Any], entity: str | None = None) -> FactStore:
        """Turn one raw record into ground facts.

        Deterministic and auditable: each fact records the field it came from, so
        a proof can be traced back past the logic into the source data.
        """
        entity = entity or str(record.get(self.entity_field, "entity"))
        store = FactStore(self.vocabulary)
        for mapping in self.features:
            truth = mapping.evaluate(record)
            raw = record.get(mapping.source, "<absent>")
            source = (
                f"{mapping.source} = {raw!r}"
                if truth is not Truth.UNKNOWN
                else f"{mapping.source} not supplied"
            )
            store.assert_fact(
                Atom(mapping.predicate, (Const(entity),)),
                truth,
                source=source,
                asserted_by="extraction",
            )
        return store

    def context_from(self, record: dict[str, Any]) -> dict[str, Any]:
        """The fields the criticality policy reads."""
        context = {f: record.get(f) for f in self.context_fields}
        context["goal"] = self.goal_predicate
        return context

    def label_of(self, record: dict[str, Any]) -> str | None:
        if self.label_field is None:
            return None
        raw = record.get(self.label_field)
        if raw is None:
            return None
        return self.label_map.get(str(raw), str(raw))

    def goal_for(self, entity: str, outcome: str | None = None) -> Atom:
        return Atom(outcome or self.goal_predicate, (Const(entity),))

    def competing_goals(self, entity: str) -> list[Atom]:
        return [Atom(o, (Const(entity),)) for o in self.outcomes]

    def observable_predicates(self) -> list[str]:
        return [d.name for d in self.vocabulary.observables()]

    def document_text(self) -> dict[str, str]:
        out: dict[str, str] = {}
        for path in self.documents:
            if path.exists():
                out[path.name] = path.read_text(encoding="utf-8")
        return out


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------


def load_pack(domain_id: str, directory: str | Path | None = None) -> DomainPack:
    root = Path(directory) if directory else PACK_DIR
    pack_dir = root / domain_id
    pack_file = pack_dir / "pack.yaml"
    if not pack_file.exists():
        available = sorted(
            p.name for p in root.iterdir() if p.is_dir() and (p / "pack.yaml").exists()
        ) if root.exists() else []
        raise FileNotFoundError(
            f"no domain pack {domain_id!r} in {root}; available: {available}"
        )

    data = yaml.safe_load(pack_file.read_text(encoding="utf-8")) or {}
    if data.get("domain_id") != domain_id:
        raise ValueError(
            f"{pack_file} declares domain_id {data.get('domain_id')!r} but is "
            f"filed under {domain_id!r}"
        )

    vocabulary = Vocabulary(
        PredicateDecl(
            name=d["name"],
            arity=int(d.get("arity", 1)),
            kind=d.get("kind", "observable"),
            arg_types=tuple(d.get("arg_types", ())),
            description=d.get("description", ""),
            phrase=d.get("phrase", ""),
            askable=bool(d.get("askable", True)),
        )
        for d in data.get("predicates", [])
    )

    crit = data.get("criticality", {}) or {}
    criticality = CriticalityPolicy(
        rules=[
            CriticalityRule(
                rule_id=r["rule_id"],
                description=r.get("description", ""),
                field=r["field"],
                op=r.get("op", "eq"),
                value=r.get("value"),
            )
            for r in crit.get("rules", [])
        ],
        default_critical=bool(crit.get("default_critical", True)),
    )

    features = [
        FeatureMapping(
            predicate=f["predicate"],
            source=f["from"],
            op=f.get("op", "truthy"),
            value=f.get("value"),
            # YAML reads a bare `false` as a boolean, so `missing: false` arrives
            # as False rather than "false". Coerced here so a pack author can
            # write it the obvious way.
            missing=str(f.get("missing", "unknown")).lower(),
            description=f.get("description", ""),
        )
        for f in data.get("features", [])
    ]

    decision = data.get("decision", {}) or {}
    pack = DomainPack(
        domain_id=domain_id,
        name=data.get("name", domain_id),
        description=data.get("description", ""),
        goal_predicate=decision.get("goal_predicate", "decision"),
        outcomes=list(decision.get("outcomes", [])),
        vocabulary=vocabulary,
        criticality=criticality,
        features=features,
        seed_rules=[
            SeedRule(
                text=s["text"],
                strength=float(s.get("strength", 0.8)),
                origin=s.get("origin", "seed"),
                cites=s.get("cites"),
                approved_by=s.get("approved_by"),
            )
            for s in data.get("seed_rules", [])
        ],
        documents=[pack_dir / d for d in data.get("documents", [])],
        entity_field=data.get("entity_field", "id"),
        context_fields=list(data.get("context_fields", [])),
        label_field=decision.get("label_field"),
        label_map={str(k): str(v) for k, v in (decision.get("label_map") or {}).items()},
        oversight=dict(data.get("oversight", {}) or {}),
        source=data.get("source"),
        licence=data.get("licence"),
        root=pack_dir,
    )

    _validate(pack)
    return pack


def _validate(pack: DomainPack) -> None:
    """Catch pack errors at load, where they are cheap to diagnose."""
    problems: list[str] = []

    for mapping in pack.features:
        sig = f"{mapping.predicate}/1"
        decl = pack.vocabulary.get(sig)
        if decl is None:
            problems.append(f"feature maps to undeclared predicate {sig}")
        elif decl.kind != "observable":
            problems.append(
                f"feature maps to {sig}, which is declared derived; only "
                f"observables can come from data"
            )
        if mapping.missing not in ("unknown", "false", "true"):
            problems.append(
                f"feature {mapping.predicate}: missing must be unknown/false/true"
            )

    for outcome in pack.outcomes:
        if pack.vocabulary.get(f"{outcome}/1") is None:
            problems.append(f"outcome {outcome!r} is not a declared predicate")

    for seed in pack.seed_rules:
        try:
            issues = pack.vocabulary.validate_rule(seed.rule())
        except Exception as exc:
            problems.append(f"seed rule {seed.text!r} does not parse: {exc}")
            continue
        problems.extend(f"seed rule {seed.text!r}: {i}" for i in issues)

    if problems:
        raise ValueError(
            f"domain pack {pack.domain_id} is invalid:\n  " + "\n  ".join(problems)
        )


def available_packs(directory: str | Path | None = None) -> list[str]:
    root = Path(directory) if directory else PACK_DIR
    if not root.exists():
        return []
    return sorted(
        p.name for p in root.iterdir() if p.is_dir() and (p / "pack.yaml").exists()
    )


__all__ = [
    "DomainPack",
    "FeatureMapping",
    "SeedRule",
    "available_packs",
    "load_pack",
]
