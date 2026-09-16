"""Generate the paper's figures from real runs — no hand-entered numbers.

Palette validated with the data-viz validator (light mode, surface #fcfcfb):
lightness band PASS, chroma floor PASS, CVD separation PASS (protan dE 9.4,
above the 8.0 floor), normal-vision floor PASS (dE 20.2), contrast PASS.
Both series are also direct-labelled, so identity never rests on colour alone.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "packs" / "benefits_eligibility"))

GREEN, AMBER = "#1C9970", "#C4761A"
INK, MUTED, GRID = "#1D2321", "#5C6663", "#DFE3DE"

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 9,
    "axes.edgecolor": GRID,
    "axes.labelcolor": INK,
    "axes.titlesize": 9.5,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "xtick.labelsize": 8, "ytick.labelsize": 8,
    "figure.facecolor": "white", "axes.facecolor": "white",
    "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})


def recessive(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_linewidth(0.6)
    ax.grid(True, color=GRID, linewidth=0.5, alpha=0.9)
    ax.set_axisbelow(True)


# ---------------------------------------------------------------- figure 1
def amortisation():
    """Oversight cost per decision, measured over a real run.

    The comparison is honest by construction: per-decision oversight costs
    exactly one review per decision by definition, so that series is flat at
    1.0. The rule-level series is measured -- cumulative approved rules over
    cumulative decisions as the run proceeds.
    """
    from logicdb.agents.base import LabelledCase
    from logicdb.agents.scripted import ScriptedAgent
    from logicdb.governance import BlockedForApproval, Governor
    from logicdb.packs import load_pack
    from logicdb.program import RuleBase, RuleStatus

    pack = load_pack("consumer_credit")
    gov = Governor(RuleBase(pack.vocabulary), criticality=pack.criticality,
                   vocabulary=pack.vocabulary)
    agent = ScriptedAgent(pack)
    cases = json.loads((pack.root / "cases.json").read_text())

    for name, text in pack.document_text().items():
        for p in agent.extract_from_document(name, text, pack.vocabulary):
            r = gov.propose(p.rule, p.origin, "agent", True, strength=p.strength,
                            source_citation=p.source_citation)
            if r.record:
                gov.approve(r.record.rule_id, by="reviewer")

    train = [LabelledCase(c["id"], pack.facts_from(c, c["id"]), pack.label_of(c))
             for c in cases[:600]]
    known = {r.key for r in gov.rulebase.all_records()}
    for p in agent.mine_rules(train, target=pack.goal_predicate, existing=known):
        r = gov.propose(p.rule, p.origin, "agent", False, strength=p.strength,
                        mining_support=p.support, mining_precision=p.precision)
        if r.record and (p.precision or 0) >= 0.75:
            gov.approve(r.record.rule_id, by="reviewer")

    decisions, cost = [], []
    n = 0
    for rec in cases:
        try:
            answer, _ = gov.ask(pack.goal_for(rec["id"]),
                                pack.facts_from(rec, rec["id"]), rec["id"], "officer",
                                context=pack.context_from(rec))
        except BlockedForApproval:
            continue
        if answer.is_uncovered:
            continue
        n += 1
        reviews = len(gov.rulebase.by_status(RuleStatus.APPROVED))
        decisions.append(n)
        cost.append(reviews / n)

    fig, ax = plt.subplots(figsize=(5.4, 2.9))
    recessive(ax)
    ax.plot(decisions, [1.0] * len(decisions), color=AMBER, linewidth=2,
            solid_capstyle="round")
    ax.plot(decisions, cost, color=GREEN, linewidth=2, solid_capstyle="round")

    # The break-even point. Early on, rule-level oversight costs MORE per
    # decision -- the reviews are paid up front, before any decision is made.
    # Marking where the curve crosses 1.0 turns a clipped line into the most
    # useful fact on the chart: how long before the approach pays for itself.
    crossover = next((d for d, c in zip(decisions, cost) if c <= 1.0), None)

    ax.annotate("per-decision oversight", xy=(decisions[len(decisions) // 2], 1.0),
                xytext=(0, 7), textcoords="offset points", color=INK, fontsize=8.5,
                ha="center")
    ax.annotate(f"rule-level oversight\n{cost[-1]:.3f} at n={decisions[-1]}",
                xy=(decisions[-1], cost[-1]), xytext=(-8, 26),
                textcoords="offset points", color=INK, fontsize=8.5, ha="right")

    if crossover is not None:
        ax.plot([crossover], [1.0], marker="o", markersize=5, color=GREEN,
                markeredgecolor="white", markeredgewidth=1.4, zorder=5)
        ax.annotate(f"breaks even at {crossover} decisions",
                    xy=(crossover, 1.0), xytext=(16, -20),
                    textcoords="offset points", color=INK, fontsize=8.5,
                    ha="left",
                    arrowprops=dict(arrowstyle="-", color=GRID, linewidth=0.8))

    ax.set_xlabel("decisions made")
    ax.set_ylabel("reviews per decision")
    ax.set_ylim(0, 1.18)
    ax.set_xlim(0, decisions[-1])
    fig.savefig(Path(__file__).parent / "amortisation.pdf")
    print(f"  amortisation.pdf  break-even at {crossover} decisions; "
          f"{decisions[-1]} total, "
          f"final cost {cost[-1]:.4f} reviews/decision "
          f"({1/cost[-1]:.1f} decisions per review)")


# ---------------------------------------------------------------- figure 2
def recovery():
    """Mined proposals ranked by precision, against the withheld true rules."""
    from logicdb.agents.base import LabelledCase
    from logicdb.agents.scripted import ScriptedAgent
    from logicdb.packs import load_pack
    from logicdb.parser import parse_rule
    from generate import TRUE_RULES

    pack = load_pack("benefits_eligibility")
    raw = json.loads((pack.root / "cases.json").read_text())
    cases = [LabelledCase(r["id"], pack.facts_from(r, r["id"]), pack.label_of(r))
             for r in raw]
    props = ScriptedAgent(pack).mine_rules(cases, target="eligible")

    truth = set()
    for _, req, forb in TRUE_RULES:
        body = ", ".join([f"{p}(A)" for p in req] + [f"not {f}(A)" for f in forb])
        truth.add(parse_rule(f"eligible(A) <- {body}.").canonical_key())

    labels, values, colors, is_true = [], [], [], []
    for i, p in enumerate(props, 1):
        hit = p.rule.canonical_key() in truth
        labels.append(f"{i}")
        values.append(p.precision)
        colors.append(GREEN if hit else AMBER)
        is_true.append(hit)

    fig, ax = plt.subplots(figsize=(5.4, 2.7))
    recessive(ax)
    ax.grid(axis="y", visible=False)
    y = range(len(values))
    ax.barh(list(y), values, height=0.62, color=colors, linewidth=0)
    ax.invert_yaxis()

    for i, (v, hit, p) in enumerate(zip(values, is_true, props)):
        tag = "recovered true rule" if hit else "near-miss"
        ax.text(v + 0.012, i, f"{v:.2f}   {tag}  (n={p.support})",
                va="center", fontsize=8, color=INK)

    ax.set_yticks(list(y))
    ax.set_yticklabels([f"#{l}" for l in labels])
    ax.set_xlabel("precision on cases where the rule fires")
    ax.set_xlim(0, 1.32)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    fig.savefig(Path(__file__).parent / "recovery.pdf")
    print(f"  recovery.pdf      {sum(is_true)}/{len(truth)} true rules recovered, "
          f"ranked {[i+1 for i,h in enumerate(is_true) if h]}")


if __name__ == "__main__":
    print("generating figures from live runs:")
    amortisation()
    recovery()
