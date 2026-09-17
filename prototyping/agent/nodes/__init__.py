"""The graph's nodes, grouped the way the graph is.

    intake   -> a decision is put on the table, or the turn is conversation
    assess   -> four properties estimated, a tier derived
    logic    -> the case as facts, and what the approved rules make of them
    gate     -> proceed, or park for a person
    reply    -> the answer, under the conduct its tier imposes
    learn    -> what the database takes away from it

Split out of one 800-line module. The seams follow the graph rather than the
order things were written, so a change to how oversight is allocated touches
`assess`, and a change to how logic is stored touches `logic` — and neither
touches the other.
"""

from .assess import classify, suggest
from .gate import gate, may_proceed, park
from .intake import intake, ready_to_decide
from .learn import propose, record
from .logic import derive, extract
from .reply import respond

__all__ = [
    "classify", "derive", "extract", "gate", "intake", "may_proceed", "park",
    "propose", "ready_to_decide", "record", "respond", "suggest",
]
