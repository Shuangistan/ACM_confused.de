"""How much human oversight a decision gets, and which decisions get audited.

Two mechanisms live here, and they are deliberately separate because they catch
different failures.

**Allocation** answers *does this decision need a person, and how urgently?* It
is the harm/reversibility/rights ladder from `decision_partner_mvp.html`,
expressed as ordinary rules in `packs/oversight` and evaluated by the same
engine as everything else. A system that routes some decisions past a human owes
an account of why those ones; running the ladder through the engine means that
account is a proof, with an approver and an audit trail, rather than a branch in
a script.

**Sampling** answers *which of the automated decisions get checked anyway?*
Rule approval asks whether the logic is legitimate, and cannot detect a
fact-extraction error, a case the rule was never meant for, or drift in the
incoming population. Outcome sampling asks whether legitimate logic is being
applied to cases it fits, and cannot tell you whether the rule should exist.
Neither substitutes for the other.

A sampled review is an **outcome audit, not a re-approval**. The reviewer is
never asked to bless the rule again -- only to say whether this decision was
right. That keeps the approve-once guarantee intact while still producing the
ground truth the Beta-Binomial posterior needs, which is the loop that was
otherwise left open: something has to decide which decisions get a label, and
this is that something.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from enum import Enum

from scipy import stats

from .engine import Engine
from .packs import DomainPack
from .probability import Answer
from .program import RuleBase, RuleOrigin, RuleRecord, RuleStatus
from .syntax import Atom, Const

#: Below this, a rule's accuracy is not good enough to stop auditing. Provenance
#: sets the bar and evidence clears it: a mined rule is a correlation until it
#: has earned trust, so it is held to a stiffer standard than a rule traceable
#: to a written clause.
ACCURACY_TARGET = {
    RuleOrigin.MINED: 0.85,
    RuleOrigin.DOCUMENT: 0.80,
    RuleOrigin.HUMAN: 0.80,
}
DEFAULT_TARGET = 0.80

#: Sampling never reaches zero. A rule that has been right a thousand times can
#: still start being wrong when the population shifts underneath it, and a floor
#: is the only thing that would notice.
AUDIT_FLOOR = 0.02

#: Bounds wider than this mean the case rests on evidence too incomplete to act
#: on. Mirrors the MVP's `c < 70` test, but over an interval width rather than a
#: self-reported confidence.
UNCERTAIN_WIDTH = 0.30


class Mode(str, Enum):
    """The four oversight modes, in descending order of human involvement."""

    HUMAN_ONLY = "human_only"
    HUMAN_APPROVAL = "human_approval"
    AI_WITH_OVERSIGHT = "ai_with_oversight"
    AI_RECOMMENDS = "ai_recommends"

    @property
    def headline(self) -> str:
        return {
            Mode.HUMAN_ONLY: "Human only -- AI may not decide",
            Mode.HUMAN_APPROVAL: "Human approval required",
            Mode.AI_WITH_OVERSIGHT: "AI acts, with oversight",
            Mode.AI_RECOMMENDS: "AI recommends; the human decides",
        }[self]

    @property
    def decides_automatically(self) -> bool:
        """Whether the engine's answer may take effect without a person."""
        return self is Mode.AI_WITH_OVERSIGHT


#: The ladder is ordered, so a case satisfying several tiers takes the highest.
#: The rules exclude one another by negation as well; this is the tie-break of
#: last resort, and disagreement between the two is a bug in the pack.
_PRECEDENCE = [
    Mode.HUMAN_ONLY,
    Mode.HUMAN_APPROVAL,
    Mode.AI_WITH_OVERSIGHT,
    Mode.AI_RECOMMENDS,
]


@dataclass
class Assessment:
    """What oversight this decision gets, and why."""

    entity: str
    mode: Mode
    #: Probability that a decision in this mode is pulled for an outcome audit.
    audit_rate: float
    #: Whether *this* decision was pulled. Deterministic -- see `_sampled`.
    sampled: bool
    #: The oversight rules that fired, so the allocation is itself explainable.
    rules_used: set[str] = field(default_factory=set)
    #: Why the rate came out where it did, in a reviewer's words.
    rate_reasons: list[str] = field(default_factory=list)
    interval_width: float = 0.0

    @property
    def needs_a_person(self) -> bool:
        return self.mode in (Mode.HUMAN_ONLY, Mode.HUMAN_APPROVAL) or self.sampled

    def describe(self) -> str:
        lines = [
            f"{self.entity}: {self.mode.headline}",
            f"  audit rate {self.audit_rate:.0%}"
            f"  ({'PULLED for audit' if self.sampled else 'not pulled'})",
        ]
        for reason in self.rate_reasons:
            lines.append(f"    {reason}")
        if self.rules_used:
            lines.append(f"  by oversight rules: {', '.join(sorted(self.rules_used))}")
        return "\n".join(lines)


def _sampled(entity: str, rate: float, salt: str = "") -> bool:
    """Whether this decision is pulled, decided reproducibly.

    Not `random`. An audit sample that cannot be reproduced cannot be defended:
    asked why a particular case was or was not checked, "the random number
    generator" is not an answer anybody should accept. Hashing the entity gives
    a uniform draw that any auditor can recompute from the case id alone.
    """
    if rate >= 1.0:
        return True
    if rate <= 0.0:
        return False
    digest = hashlib.sha256(f"{salt}:{entity}".encode()).digest()
    draw = int.from_bytes(digest[:8], "big") / 2**64
    return draw < rate


def _doubt(record: RuleRecord) -> float:
    """P(this rule's accuracy is below its target | the evidence so far).

    The Beta-Binomial posterior the rule already maintains, read as an audit
    signal rather than as a strength. A rule with no track record is almost
    entirely doubt, so it is audited nearly always; as successes accumulate the
    posterior concentrates above the target and the doubt -- and with it the
    sampling rate -- decays toward the floor. Failures push it back up without
    anybody having to intervene.
    """
    target = ACCURACY_TARGET.get(record.origin, DEFAULT_TARGET)
    alpha = 1.0 + record.stats.successes
    beta = 1.0 + record.stats.failures
    return float(stats.beta.cdf(target, alpha, beta))


class OversightPolicy:
    """Allocates oversight, and decides what to audit.

    The allocation half owns a second rule base -- the oversight pack -- and
    runs it through the ordinary engine. The audit half reads the posteriors of
    whichever rules produced the decision under review.
    """

    def __init__(
        self,
        pack: DomainPack,
        rulebase: RuleBase | None = None,
        uncertain_width: float = UNCERTAIN_WIDTH,
        floor: float = AUDIT_FLOOR,
        salt: str = "audit-v1",
    ) -> None:
        self.pack = pack
        self.uncertain_width = uncertain_width
        self.floor = floor
        self.salt = salt
        self.engine = Engine()
        self.rulebase = rulebase if rulebase is not None else self._seeded_rulebase()

    def _seeded_rulebase(self) -> RuleBase:
        """The ladder, approved.

        Seeded as APPROVED rather than PROPOSED because a system with no
        oversight policy at all cannot safely be asked to decide which of its
        decisions need one. The seed is attributed and cited like any other
        rule, so it is auditable and can be superseded -- it is a starting
        position, not an exemption.
        """
        from .parser import parse_rule

        rb = RuleBase(self.pack.vocabulary)
        for i, seed in enumerate(self.pack.seed_rules, start=1):
            rb.add(
                RuleRecord(
                    rule=parse_rule(seed.text),
                    rule_id=f"ov{i:04d}",
                    strength=seed.strength,
                    status=RuleStatus.APPROVED,
                    origin=RuleOrigin.DOCUMENT,
                    proposed_by="oversight-pack",
                    approved_by=seed.approved_by or "seed",
                    source_citation=seed.cites,
                )
            )
        return rb

    # -- allocation --------------------------------------------------------
    def assess_mode(
        self, entity: str, harm: int, reversibility: int,
        rights_impacted: bool, interval_width: float,
    ) -> tuple[Mode, set[str]]:
        """Run the ladder. Returns the mode and the rules that produced it."""
        record = {
            "id": entity,
            "harm": harm,
            "reversibility": reversibility,
            "rights_impacted": rights_impacted,
            "interval_width": interval_width,
        }
        facts = self.pack.facts_from(record, entity)
        evaluation = self.engine.evaluate(self.rulebase, facts)
        for mode in _PRECEDENCE:
            if evaluation.holds(Atom(mode.value, (Const(entity),))):
                return mode, set(evaluation.rules_used)
        # The residual rule makes this unreachable on a complete pack; if the
        # ladder is ever edited into a gap, fail closed rather than silently
        # automating.
        return Mode.HUMAN_APPROVAL, set(evaluation.rules_used)

    # -- audit sampling ----------------------------------------------------
    def audit_rate(
        self, mode: Mode, decision_rules: list[RuleRecord]
    ) -> tuple[float, list[str]]:
        """How often decisions like this one should be checked after the fact."""
        reasons: list[str] = []
        if not mode.decides_automatically:
            return 1.0, ["a person sees this decision regardless of sampling"]
        if not decision_rules:
            return 1.0, ["no rule produced this; it cannot be left unchecked"]

        provisional = [r for r in decision_rules if r.status is RuleStatus.PROVISIONAL]
        if provisional:
            return 1.0, [
                f"{len(provisional)} rule(s) still provisional -- every decision "
                f"they touch is audited until they are reviewed"
            ]

        # The weakest link: a derivation is only as trustworthy as the rule in
        # it with the least evidence behind it.
        worst = max(decision_rules, key=_doubt)
        doubt = _doubt(worst)
        rate = self.floor + (1.0 - self.floor) * doubt
        target = ACCURACY_TARGET.get(worst.origin, DEFAULT_TARGET)
        reasons.append(
            f"weakest rule {worst.rule_id} ({worst.origin.value}): right "
            f"{worst.stats.successes}/{worst.stats.observations}, "
            f"P(accuracy < {target:.0%}) = {doubt:.0%}"
        )
        if rate <= self.floor + 1e-9:
            reasons.append(
                f"at the {self.floor:.0%} floor -- kept non-zero so drift is "
                f"still detectable"
            )
        return min(max(rate, self.floor), 1.0), reasons

    # -- both, for one decision -------------------------------------------
    def assess(
        self,
        entity: str,
        answer: Answer,
        decision_rules: list[RuleRecord],
        harm: int,
        reversibility: int,
        rights_impacted: bool,
    ) -> Assessment:
        width = answer.bounds.upper - answer.bounds.lower
        mode, ov_rules = self.assess_mode(
            entity, harm, reversibility, rights_impacted, width
        )
        rate, reasons = self.audit_rate(mode, decision_rules)
        if width >= self.uncertain_width and mode.decides_automatically:
            reasons.append(f"bounds {width:.2f} wide -- evidence is incomplete")
        return Assessment(
            entity=entity,
            mode=mode,
            audit_rate=rate,
            sampled=_sampled(entity, rate, self.salt),
            rules_used=ov_rules,
            rate_reasons=reasons,
            interval_width=width,
        )


__all__ = [
    "ACCURACY_TARGET", "AUDIT_FLOOR", "UNCERTAIN_WIDTH",
    "Assessment", "Mode", "OversightPolicy",
]
