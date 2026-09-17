"""How to assess the four inputs.

Each scale carries anchors: what a 2 looks like, what an 8 looks like. A
definition tells you what "harm" means; anchors tell you whether the case in
front of you is a 3 or a 7, which is the question a person actually has.

These are used twice — written into the estimator's prompt, and served to the
page as the help behind each control. That is the point of keeping them here
rather than in both places. A person and a model scoring the same case against
different anchors produce two numbers that look comparable and are not, and the
whole panel rests on comparing them.

The anchors encode judgements that are not neutral, and should be argued with
rather than inherited: harm is measured against the person the decision lands
on, not the cost to the organisation; and something reversible only after
months of appeal is not reversible.
"""

from __future__ import annotations

SCALES: dict[str, dict] = {
    "harm": {
        "label": "Potential harm if wrong",
        "range": "1-10",
        "short": "How badly a wrong decision hurts the person it lands on — "
                 "not what it costs the organisation.",
        "bands": [
            ("1-2", "Trivial. They would shrug it off; nothing lasting."),
            ("3-5", "Real but absorbable. Cost, delay or inconvenience they "
                    "can recover from."),
            ("6-7", "Material. An opportunity, entitlement or service is lost "
                    "to them."),
            ("8-10", "Severe. Livelihood, housing, health, liberty or standing "
                     "is damaged."),
        ],
        "tip": "If you are torn between two bands, ask what the person would "
               "say the harm was — not what the file records.",
        "threshold": "6 or above requires human approval. 8 with low "
                     "reversibility, or 9 with a right engaged, rules out an "
                     "automated decision entirely.",
    },
    "reversibility": {
        "label": "Reversibility",
        "range": "1-10",
        "short": "How readily a wrong decision can actually be undone.",
        "bands": [
            ("1-3", "Not undoable. The opportunity has passed, the money is "
                    "spent, the harm has landed."),
            ("4-6", "Undoable with effort. An appeal or review, taking weeks "
                    "or months."),
            ("7-10", "Immediate. Corrected on request, leaving no trace."),
        ],
        "tip": "Score it from the affected person's side. A decision that is "
               "reversible on paper but takes nine months of appeals is not "
               "reversible.",
        "threshold": "3 or below, together with high harm, rules out an "
                     "automated decision.",
    },
    "confidence": {
        "label": "AI confidence",
        "range": "0-100%",
        "short": "How complete the evidence is — not how certain the model "
                 "sounds.",
        "bands": [
            ("0-39", "Key facts missing or disputed. Largely assertion."),
            ("40-69", "Partial. Important gaps remain open."),
            ("70-89", "Main facts established; minor gaps."),
            ("90-100", "Fully evidenced, nothing contested."),
        ],
        "tip": "Absent evidence lowers this score. It is never a reason "
               "against the person — that is what this control is for.",
        "threshold": "Below 70% requires human approval regardless of harm. "
                     "Above 70% it has no further effect: the scale is finer "
                     "than the rule that reads it.",
    },
    "rights": {
        "label": "Sensitive / rights-affecting",
        "range": "yes / no",
        "short": "Whether the decision bears on something the person is "
                 "entitled to.",
        "bands": [
            ("yes", "Employment, benefits, housing, health care, education, "
                    "immigration status, liberty, non-discrimination, due "
                    "process."),
            ("no", "Operational or administrative matters that touch no "
                   "entitlement."),
        ],
        "tip": "If a wrong decision would deny someone something they are "
               "entitled to, answer yes. When genuinely unsure, answer yes: "
               "the cost is a review that was not needed.",
        "threshold": "Yes always requires at least human approval, whatever "
                     "the harm score.",
    },
}

#: The order the page shows them in.
ORDER = ("harm", "confidence", "reversibility", "rights")


def prompt_block() -> str:
    """The same anchors, formatted for the estimator's system prompt.

    The model is given the bands rather than a bare range, for the same reason
    the person is: without anchors, "7" means whatever the scorer felt.
    """
    out: list[str] = []
    for key in ORDER:
        s = SCALES[key]
        out.append(f"{key} ({s['range']}) — {s['short']}")
        for band, text in s["bands"]:
            out.append(f"    {band:<7} {text}")
        out.append(f"    note: {s['tip']}")
        out.append("")
    return "\n".join(out).rstrip()


__all__ = ["ORDER", "SCALES", "prompt_block"]
