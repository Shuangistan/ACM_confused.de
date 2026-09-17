"""SQLite persistence for the logic database.

Without this the system's central claim is empty. "Approve once and never again"
is a promise about the future, so it means nothing if approvals evaporate when
the process exits; and impact analysis -- "this rejected rule already affected
these fourteen decisions" -- can only reach back as far as the record goes.

Three choices here are load-bearing rather than incidental:

**Identity is the primary key.** The `rules` table is keyed on `canonical_key`,
not on `rule_id`. So approve-once stops being a convention the application layer
remembers to honour and becomes something the database will not let you violate:
inserting the same logic twice is a constraint failure, however it is spelled.

**Approval history is append-only.** `rule_events` is never updated or deleted.
The `rules` table holds current state for querying; `rule_events` holds how it
got there. If the two ever disagree, the event log is what happened. A governance
record that can be quietly rewritten is not evidence of anything.

**Decisions store their inputs.** Every query persists the facts it saw and the
rule versions it used, so a decision from six months ago can be replayed exactly
rather than re-derived against a rule base that has since moved on. Re-deriving
would silently rewrite history, which is the failure that makes audits useless.

stdlib `sqlite3` rather than an ORM: the schema is six tables, and the file needs
to be readable by anyone with a `.db` and no Python environment, long after the
summer school is over.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from .facts import FactRecord, FactStore, Truth
from .parser import parse_atom, parse_rule
from .program import RuleBase, RuleOrigin, RuleRecord, RuleStats, RuleStatus
from .syntax import PredicateDecl, Vocabulary

DEFAULT_DB = Path(__file__).parent.parent / "data" / "logicdb.sqlite"

SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

-- The declared vocabulary. Persisted so a database can be opened and audited
-- without the pack that created it, and so a changed declaration is detectable
-- rather than silently invalidating stored rules.
CREATE TABLE IF NOT EXISTS vocabulary (
    signature   TEXT PRIMARY KEY,
    name        TEXT NOT NULL,
    arity       INTEGER NOT NULL,
    kind        TEXT NOT NULL,
    phrase      TEXT NOT NULL DEFAULT '',
    description TEXT NOT NULL DEFAULT '',
    askable     INTEGER NOT NULL DEFAULT 1,
    arg_types   TEXT NOT NULL DEFAULT '[]'
);

-- The rules. Keyed by canonical identity, so the same logic cannot be stored
-- twice under different spellings -- approve-once enforced by the database
-- rather than by the application remembering to check.
CREATE TABLE IF NOT EXISTS rules (
    canonical_key    TEXT PRIMARY KEY,
    rule_id          TEXT NOT NULL UNIQUE,
    rule_text        TEXT NOT NULL,
    canonical_form   TEXT NOT NULL,
    head_signature   TEXT NOT NULL,
    strength         REAL NOT NULL,
    status           TEXT NOT NULL,
    origin           TEXT NOT NULL,
    proposed_by      TEXT NOT NULL DEFAULT 'unknown',
    proposed_at      TEXT NOT NULL,
    approved_by      TEXT,
    approved_at      TEXT,
    rejected_reason  TEXT,
    source_citation  TEXT,
    mining_support   INTEGER,
    mining_precision REAL,
    supersedes       TEXT,
    version          INTEGER NOT NULL DEFAULT 1,
    reuse_count      INTEGER NOT NULL DEFAULT 0,
    successes        INTEGER NOT NULL DEFAULT 0,
    failures         INTEGER NOT NULL DEFAULT 0,
    last_used_at     TEXT,
    notes            TEXT NOT NULL DEFAULT ''
);

CREATE INDEX IF NOT EXISTS idx_rules_status ON rules(status);
CREATE INDEX IF NOT EXISTS idx_rules_head   ON rules(head_signature);

-- Append-only. Never UPDATEd, never DELETEd. This is the governance record;
-- `rules` is a materialised view of where it ended up.
CREATE TABLE IF NOT EXISTS rule_events (
    seq           INTEGER PRIMARY KEY AUTOINCREMENT,
    canonical_key TEXT NOT NULL,
    rule_id       TEXT NOT NULL,
    event         TEXT NOT NULL,
    actor         TEXT NOT NULL,
    at            TEXT NOT NULL,
    detail        TEXT NOT NULL DEFAULT ''
);

CREATE INDEX IF NOT EXISTS idx_rule_events_key ON rule_events(canonical_key);

CREATE TABLE IF NOT EXISTS queries (
    query_id           TEXT PRIMARY KEY,
    goal               TEXT NOT NULL,
    entity             TEXT NOT NULL,
    critical           INTEGER NOT NULL,
    criticality_reason TEXT NOT NULL DEFAULT '',
    asked_by           TEXT NOT NULL,
    asked_at           TEXT NOT NULL,
    lower              REAL NOT NULL DEFAULT 0,
    upper              REAL NOT NULL DEFAULT 0,
    status             TEXT NOT NULL,
    blocked_on         TEXT NOT NULL DEFAULT '[]',
    outcome_label      TEXT,
    confirmed_correct  INTEGER
);

CREATE INDEX IF NOT EXISTS idx_queries_status ON queries(status);

-- Which rules produced which decision. Impact analysis is a join over this.
CREATE TABLE IF NOT EXISTS query_rules (
    query_id    TEXT NOT NULL,
    rule_id     TEXT NOT NULL,
    provisional INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (query_id, rule_id),
    FOREIGN KEY (query_id) REFERENCES queries(query_id)
);

CREATE INDEX IF NOT EXISTS idx_query_rules_rule ON query_rules(rule_id);

-- The facts each decision actually saw, so it can be replayed rather than
-- re-derived against a rule base that has moved on since.
CREATE TABLE IF NOT EXISTS query_facts (
    query_id   TEXT NOT NULL,
    atom       TEXT NOT NULL,
    truth      TEXT NOT NULL,
    source     TEXT NOT NULL DEFAULT '',
    confidence REAL NOT NULL DEFAULT 1.0,
    PRIMARY KEY (query_id, atom),
    FOREIGN KEY (query_id) REFERENCES queries(query_id)
);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class Store:
    """Durable storage for rules, their approval history, and decisions made."""

    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path) if path else DEFAULT_DB
        if str(self.path) != ":memory:":
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.path), timeout=10.0, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        #: One connection is shared across threads, so transactions have to be
        #: serialised. Without this, two concurrent writers interleave inside
        #: the same connection and a child row can be inserted before its
        #: parent commits -- which surfaces as a foreign-key failure on
        #: `query_rules`, not as anything resembling a race. Reentrant because
        #: a few writes nest.
        self._write_lock = threading.RLock()
        self.conn.executescript(SCHEMA)
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> "Store":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    @contextmanager
    def _tx(self) -> Iterator[sqlite3.Connection]:
        with self._write_lock:
            try:
                yield self.conn
                self.conn.commit()
            except Exception:
                self.conn.rollback()
                raise

    # ----------------------------------------------------------------------
    # Vocabulary
    # ----------------------------------------------------------------------
    def save_vocabulary(self, vocabulary: Vocabulary) -> None:
        with self._tx() as conn:
            for decl in vocabulary:
                conn.execute(
                    "INSERT INTO vocabulary (signature, name, arity, kind, phrase,"
                    " description, askable, arg_types) VALUES (?,?,?,?,?,?,?,?)"
                    " ON CONFLICT(signature) DO UPDATE SET"
                    " name=excluded.name, arity=excluded.arity, kind=excluded.kind,"
                    " phrase=excluded.phrase, description=excluded.description,"
                    " askable=excluded.askable, arg_types=excluded.arg_types",
                    (
                        decl.signature, decl.name, decl.arity, decl.kind,
                        decl.phrase, decl.description, int(decl.askable),
                        json.dumps(list(decl.arg_types)),
                    ),
                )

    def load_vocabulary(self) -> Vocabulary:
        rows = self.conn.execute("SELECT * FROM vocabulary ORDER BY signature").fetchall()
        return Vocabulary(
            PredicateDecl(
                name=r["name"], arity=r["arity"], kind=r["kind"],
                arg_types=tuple(json.loads(r["arg_types"])),
                description=r["description"], phrase=r["phrase"],
                askable=bool(r["askable"]),
            )
            for r in rows
        )

    # ----------------------------------------------------------------------
    # Rules
    # ----------------------------------------------------------------------
    def save_rule(self, record: RuleRecord) -> None:
        """Insert or update a rule, keyed on its canonical identity."""
        with self._tx() as conn:
            conn.execute(
                "INSERT INTO rules (canonical_key, rule_id, rule_text, canonical_form,"
                " head_signature, strength, status, origin, proposed_by, proposed_at,"
                " approved_by, approved_at, rejected_reason, source_citation,"
                " mining_support, mining_precision, supersedes, version, reuse_count,"
                " successes, failures, last_used_at, notes)"
                " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)"
                " ON CONFLICT(canonical_key) DO UPDATE SET"
                " strength=excluded.strength, status=excluded.status,"
                " approved_by=excluded.approved_by, approved_at=excluded.approved_at,"
                " rejected_reason=excluded.rejected_reason,"
                " source_citation=excluded.source_citation,"
                " mining_support=excluded.mining_support,"
                " mining_precision=excluded.mining_precision,"
                " supersedes=excluded.supersedes, version=excluded.version,"
                " reuse_count=excluded.reuse_count, successes=excluded.successes,"
                " failures=excluded.failures, last_used_at=excluded.last_used_at,"
                " notes=excluded.notes",
                (
                    record.key, record.rule_id, str(record.rule),
                    record.rule.canonical_form(), record.head_signature,
                    record.strength, record.status.value, record.origin.value,
                    record.proposed_by, record.proposed_at, record.approved_by,
                    record.approved_at, record.rejected_reason, record.source_citation,
                    record.mining_support, record.mining_precision, record.supersedes,
                    record.version, record.stats.reuse_count, record.stats.successes,
                    record.stats.failures, record.stats.last_used_at, record.notes,
                ),
            )

    def log_rule_event(
        self, record: RuleRecord, event: str, actor: str, detail: str = ""
    ) -> None:
        """Append to the governance record. Never modifies an existing row."""
        with self._tx() as conn:
            conn.execute(
                "INSERT INTO rule_events (canonical_key, rule_id, event, actor, at,"
                " detail) VALUES (?,?,?,?,?,?)",
                (record.key, record.rule_id, event, actor, _now(), detail),
            )

    def rule_history(self, rule_id: str) -> list[dict[str, Any]]:
        """Everything that ever happened to a rule, oldest first."""
        rows = self.conn.execute(
            "SELECT * FROM rule_events WHERE rule_id = ? ORDER BY seq", (rule_id,)
        ).fetchall()
        return [dict(r) for r in rows]

    def find_by_canonical_key(self, key: str) -> dict[str, Any] | None:
        row = self.conn.execute(
            "SELECT * FROM rules WHERE canonical_key = ?", (key,)
        ).fetchone()
        return dict(row) if row else None

    def load_rulebase(self, vocabulary: Vocabulary | None = None) -> RuleBase:
        """Rebuild the in-memory rule base from disk.

        Stratification is re-checked on load. A database that accumulated rules
        which are individually fine but jointly unstratifiable should fail loudly
        at startup, not produce order-dependent answers at query time.
        """
        vocab = vocabulary if vocabulary is not None else self.load_vocabulary()
        rb = RuleBase(vocab if len(vocab) else None)
        rows = self.conn.execute("SELECT * FROM rules ORDER BY rule_id").fetchall()
        for row in rows:
            record = RuleRecord(
                rule=parse_rule(row["rule_text"]),
                rule_id=row["rule_id"],
                strength=row["strength"],
                status=RuleStatus(row["status"]),
                origin=RuleOrigin(row["origin"]),
                proposed_by=row["proposed_by"],
                proposed_at=row["proposed_at"],
                approved_by=row["approved_by"],
                approved_at=row["approved_at"],
                rejected_reason=row["rejected_reason"],
                source_citation=row["source_citation"],
                mining_support=row["mining_support"],
                mining_precision=row["mining_precision"],
                supersedes=row["supersedes"],
                version=row["version"],
                stats=RuleStats(
                    reuse_count=row["reuse_count"],
                    successes=row["successes"],
                    failures=row["failures"],
                    last_used_at=row["last_used_at"],
                ),
                notes=row["notes"],
            )
            if record.key != row["canonical_key"]:
                raise ValueError(
                    f"stored rule {row['rule_id']} has canonical key "
                    f"{record.key} but was filed under {row['canonical_key']}. "
                    f"The canonicalisation changed, so previously-granted "
                    f"approvals can no longer be matched to re-proposed rules."
                )
            rb.add(record, check_stratification=False)

        # One stratification check over the whole loaded set, rather than
        # incrementally per insert, so load order cannot reject a valid base.
        rb.strata()
        return rb

    # ----------------------------------------------------------------------
    # Queries
    # ----------------------------------------------------------------------
    def save_query(self, query: Any, facts: FactStore | None = None) -> None:
        with self._tx() as conn:
            conn.execute(
                "INSERT INTO queries (query_id, goal, entity, critical,"
                " criticality_reason, asked_by, asked_at, lower, upper, status,"
                " blocked_on, outcome_label, confirmed_correct)"
                " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)"
                " ON CONFLICT(query_id) DO UPDATE SET"
                " lower=excluded.lower, upper=excluded.upper, status=excluded.status,"
                " blocked_on=excluded.blocked_on,"
                " outcome_label=excluded.outcome_label,"
                " confirmed_correct=excluded.confirmed_correct",
                (
                    query.query_id, str(query.goal), query.entity,
                    int(query.critical), query.criticality_reason, query.asked_by,
                    query.asked_at, query.lower, query.upper, query.status,
                    json.dumps(list(query.blocked_on)), query.outcome_label,
                    None if query.confirmed_correct is None
                    else int(query.confirmed_correct),
                ),
            )
            provisional = set(query.provisional_rule_ids)
            for rule_id in query.rule_ids:
                conn.execute(
                    "INSERT INTO query_rules (query_id, rule_id, provisional)"
                    " VALUES (?,?,?) ON CONFLICT(query_id, rule_id) DO UPDATE SET"
                    " provisional=excluded.provisional",
                    (query.query_id, rule_id, int(rule_id in provisional)),
                )
            if facts is not None:
                for record in facts:
                    conn.execute(
                        "INSERT INTO query_facts (query_id, atom, truth, source,"
                        " confidence) VALUES (?,?,?,?,?)"
                        " ON CONFLICT(query_id, atom) DO UPDATE SET"
                        " truth=excluded.truth, source=excluded.source,"
                        " confidence=excluded.confidence",
                        (
                            query.query_id, str(record.atom), record.truth.value,
                            record.source, record.confidence,
                        ),
                    )

    def load_queries(self) -> dict[str, Any]:
        """Rebuild the query log, so impact analysis reaches back across restarts."""
        from .governance import QueryRecord

        out: dict[str, QueryRecord] = {}
        for row in self.conn.execute("SELECT * FROM queries ORDER BY asked_at"):
            rules = self.conn.execute(
                "SELECT rule_id, provisional FROM query_rules WHERE query_id = ?"
                " ORDER BY rule_id",
                (row["query_id"],),
            ).fetchall()
            out[row["query_id"]] = QueryRecord(
                query_id=row["query_id"],
                goal=parse_atom(row["goal"]),
                entity=row["entity"],
                critical=bool(row["critical"]),
                criticality_reason=row["criticality_reason"],
                asked_by=row["asked_by"],
                asked_at=row["asked_at"],
                rule_ids=tuple(r["rule_id"] for r in rules),
                provisional_rule_ids=tuple(
                    r["rule_id"] for r in rules if r["provisional"]
                ),
                lower=row["lower"],
                upper=row["upper"],
                blocked_on=tuple(json.loads(row["blocked_on"])),
                status=row["status"],
                outcome_label=row["outcome_label"],
                confirmed_correct=None if row["confirmed_correct"] is None
                else bool(row["confirmed_correct"]),
            )
        return out

    def replay_facts(self, query_id: str, vocabulary: Vocabulary | None = None) -> FactStore:
        """The exact facts a past decision saw.

        Replaying from stored inputs rather than re-deriving is the difference
        between auditing a decision and auditing what the system would do today.
        """
        store = FactStore(vocabulary)
        rows = self.conn.execute(
            "SELECT * FROM query_facts WHERE query_id = ? ORDER BY atom", (query_id,)
        ).fetchall()
        for row in rows:
            store.add(
                FactRecord(
                    atom=parse_atom(row["atom"]),
                    truth=Truth(row["truth"]),
                    source=row["source"],
                    confidence=row["confidence"],
                )
            )
        return store

    def queries_using_rule(self, rule_id: str, provisional_only: bool = False) -> list[str]:
        """Impact analysis as a single join, over all history rather than this session."""
        sql = "SELECT query_id FROM query_rules WHERE rule_id = ?"
        if provisional_only:
            sql += " AND provisional = 1"
        return [r["query_id"] for r in self.conn.execute(sql + " ORDER BY query_id", (rule_id,))]

    # ----------------------------------------------------------------------
    # Reporting
    # ----------------------------------------------------------------------
    def counts(self) -> dict[str, int]:
        tables = ("vocabulary", "rules", "rule_events", "queries", "query_rules",
                  "query_facts")
        return {
            t: self.conn.execute(f"SELECT COUNT(*) AS n FROM {t}").fetchone()["n"]
            for t in tables
        }

    def amortisation(self) -> dict[str, Any]:
        """What one act of review bought, over the whole recorded history.

        Two different ratios, kept apart because conflating them overstates the
        result. `applications_per_rule` counts how often a rule fired, and
        several rules fire per decision -- so reporting it as "decisions per
        review" inflates the headline by exactly that factor.
        `decisions_per_review` counts answered decisions, which is the honest
        claim and the smaller number.
        """
        row = self.conn.execute(
            "SELECT COUNT(*) AS approved, COALESCE(SUM(reuse_count), 0) AS uses"
            " FROM rules WHERE status = 'approved'"
        ).fetchone()
        approved, uses = row["approved"], row["uses"]
        answered = self.conn.execute(
            "SELECT COUNT(*) AS n FROM queries WHERE status = 'answered'"
        ).fetchone()["n"]
        return {
            "approved_rules": approved,
            "rule_applications": uses,
            "decisions_answered": answered,
            "applications_per_rule": (uses / approved) if approved else 0.0,
            "decisions_per_review": (answered / approved) if approved else 0.0,
        }

    # ----------------------------------------------------------------------
    # Text export / import
    # ----------------------------------------------------------------------
    def export_text(self) -> str:
        """The rule base as reviewable, diffable source text.

        Worth having because a rule base is the kind of artifact that belongs
        under version control and in front of a reviewer, and neither works well
        against a binary file.
        """
        lines = [
            "# Logic database export",
            f"# generated {_now()}",
            "",
        ]
        for row in self.conn.execute("SELECT * FROM rules ORDER BY rule_id"):
            meta = [f"p={row['strength']}", f"origin={row['origin']}",
                    f"status={row['status']}"]
            if row["approved_by"]:
                meta.append(f'by="{row["approved_by"]}"')
            if row["mining_support"] is not None:
                meta.append(f"support={row['mining_support']}")
                meta.append(f"precision={row['mining_precision']}")
            if row["reuse_count"]:
                meta.append(f"used={row['reuse_count']}")
            lines.append(f"rule {row['rule_id']} [{', '.join(meta)}]")
            lines.append(f"  {row['rule_text']}")
            lines.append("")
        return "\n".join(lines)

    def import_text(
        self, source: str, vocabulary: Vocabulary | None = None, actor: str = "import"
    ) -> list[RuleRecord]:
        """Load rules from surface syntax.

        Approve-once applies here too: a rule already present under its canonical
        key is skipped rather than reinserted, so re-importing an edited file
        cannot resurrect a settled question or duplicate an approval.
        """
        from .parser import parse_program

        parsed_rules, _ = parse_program(source)
        added: list[RuleRecord] = []
        for parsed in parsed_rules:
            key = parsed.rule.canonical_key()
            if self.find_by_canonical_key(key) is not None:
                continue
            meta = parsed.metadata
            record = RuleRecord(
                rule=parsed.rule,
                rule_id=parsed.rule_id or f"i{len(added):04d}",
                strength=float(meta.get("p", 0.7)),
                status=RuleStatus(meta.get("status", "proposed")),
                origin=RuleOrigin(meta.get("origin", "human")),
                proposed_by=str(meta.get("by", actor)),
                approved_by=str(meta["by"]) if meta.get("status") == "approved"
                and meta.get("by") else None,
                approved_at=str(meta["at"]) if meta.get("at") else None,
                source_citation=meta.get("cites"),
            )
            if vocabulary is not None:
                problems = vocabulary.validate_rule(record.rule)
                if problems:
                    raise ValueError(
                        f"imported rule {record.rule_id} is invalid:\n  "
                        + "\n  ".join(problems)
                    )
            self.save_rule(record)
            self.log_rule_event(record, "imported", actor, f"from text by {actor}")
            added.append(record)
        return added


__all__ = ["DEFAULT_DB", "SCHEMA", "Store"]
