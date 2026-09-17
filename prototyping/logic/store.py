"""Persistence: the rules, and the cases they were used on.

Two tables carry the weight, and the second is the one people underestimate.

`rules` is keyed by **canonical key**, not by an autoincrement id. That is what
makes approve-once true rather than aspirational: the same rule proposed again
in different syntax collides with the row that is already approved, instead of
arriving as a fresh proposal and costing a second review.

`cases` is not a history table. It is the evidence every future generalization
is judged against — propose dropping a literal, replay it over the stored cases,
and show the expert where it would have disagreed. Deleting old cases therefore
weakens every approval made afterwards, which is worth knowing before anyone
adds a retention policy.

`rule_events` is append-only. Rows are never updated or removed, because the
question an audit asks is not what the rule base says now but who changed it and
when.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCHEMA = """
CREATE TABLE IF NOT EXISTS rules (
    canonical_key TEXT PRIMARY KEY,
    text          TEXT NOT NULL,
    head          TEXT NOT NULL,
    status        TEXT NOT NULL DEFAULT 'proposed',
    origin        TEXT NOT NULL DEFAULT 'agent',
    proposed_by   TEXT NOT NULL DEFAULT 'unknown',
    proposed_at   TEXT NOT NULL,
    approved_by   TEXT,
    approved_at   TEXT,
    note          TEXT NOT NULL DEFAULT '',
    uses          INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS cases (
    case_id     TEXT PRIMARY KEY,
    created_at  TEXT NOT NULL,
    question    TEXT NOT NULL DEFAULT '',
    facts       TEXT NOT NULL,   -- [[atom, truth, source], ...]
    conclusions TEXT NOT NULL,   -- [atom, ...]
    rules_used  TEXT NOT NULL,   -- [canonical_key, ...]
    decided_by  TEXT NOT NULL    -- 'rules' | 'model' | 'uncovered'
);

CREATE TABLE IF NOT EXISTS rule_events (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    canonical_key TEXT NOT NULL,
    at            TEXT NOT NULL,
    event         TEXT NOT NULL,
    actor         TEXT NOT NULL,
    detail        TEXT NOT NULL DEFAULT ''
);

CREATE INDEX IF NOT EXISTS idx_rules_status ON rules(status);
CREATE INDEX IF NOT EXISTS idx_cases_created ON cases(created_at);
"""

APPROVED, PROPOSED, REJECTED, RETIRED = "approved", "proposed", "rejected", "retired"


def _drafts(raw: Any) -> dict[str, Any]:
    """Always {kept, rejected}, whatever the row happens to hold.

    Rows written before this column existed default to an empty list, and a
    caller reaching for `.get("kept")` on a list gets an AttributeError rather
    than an empty result — a schema default leaking out as a crash.
    """
    try:
        value = json.loads(raw) if raw else {}
    except (TypeError, ValueError):
        value = {}
    if not isinstance(value, dict):
        value = {}
    value.setdefault("kept", [])
    value.setdefault("rejected", [])
    return value


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass
class RuleRow:
    canonical_key: str
    text: str
    head: str
    status: str = PROPOSED
    origin: str = "agent"
    proposed_by: str = "unknown"
    proposed_at: str = field(default_factory=_now)
    approved_by: str | None = None
    approved_at: str | None = None
    note: str = ""
    uses: int = 0
    #: Verdicts returned by sampled audits of decisions this rule took part in.
    audit_right: int = 0
    audit_wrong: int = 0

    def as_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


class Store:
    def __init__(self, path: str | Path = "data/logic.sqlite") -> None:
        self.path = Path(path)
        if str(self.path) != ":memory:":
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.path), timeout=10.0,
                                    check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)
        self._migrate()
        self.conn.commit()
        #: One connection shared across threads, so transactions are serialised.
        #: Without this two concurrent writers interleave inside the same
        #: connection and a child row can land before its parent commits — which
        #: surfaces as a foreign-key failure, not as anything resembling a race.
        self._lock = threading.RLock()

    #: Added after the first schema shipped. Applied on open rather than by a
    #: migration tool: the database is small, and an expert's approval must not
    #: be lost because a file predates the column that records it.
    _LATER = (
        ("cases", "tier", "TEXT NOT NULL DEFAULT ''"),
        ("cases", "covered", "INTEGER NOT NULL DEFAULT 0"),
        ("cases", "session", "TEXT NOT NULL DEFAULT ''"),
        ("cases", "status", "TEXT NOT NULL DEFAULT 'settled'"),
        ("cases", "sampled", "INTEGER NOT NULL DEFAULT 0"),
        ("cases", "reply", "TEXT NOT NULL DEFAULT ''"),
        ("cases", "approved_by", "TEXT"),
        ("cases", "approved_at", "TEXT"),
        ("cases", "approval_note", "TEXT NOT NULL DEFAULT ''"),
        ("cases", "drafts", "TEXT NOT NULL DEFAULT '{}'"),
        ("rules", "audit_right", "INTEGER NOT NULL DEFAULT 0"),
        ("rules", "audit_wrong", "INTEGER NOT NULL DEFAULT 0"),
    )

    def _migrate(self) -> None:
        for table, column, decl in self._LATER:
            have = {r["name"] for r in self.conn.execute(f"PRAGMA table_info({table})")}
            if column not in have:
                self.conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {decl}")

    def close(self) -> None:
        self.conn.close()

    # -- rules -------------------------------------------------------------
    def upsert_rule(self, row: RuleRow) -> str:
        """Insert, or leave an existing row alone.

        Deliberately not an update: a rule already approved must not be quietly
        reset to `proposed` because an agent proposed it again. That is the
        whole promise of approve-once.
        """
        with self._lock:
            existing = self.get_rule(row.canonical_key)
            if existing is not None:
                return existing.status
            self.conn.execute(
                "INSERT INTO rules (canonical_key, text, head, status, origin,"
                " proposed_by, proposed_at, note) VALUES (?,?,?,?,?,?,?,?)",
                (row.canonical_key, row.text, row.head, row.status, row.origin,
                 row.proposed_by, row.proposed_at, row.note),
            )
            self._log(row.canonical_key, "proposed", row.proposed_by, row.note)
            self.conn.commit()
            return row.status

    def set_status(self, key: str, status: str, actor: str, detail: str = "") -> None:
        with self._lock:
            stamp = _now() if status == APPROVED else None
            self.conn.execute(
                "UPDATE rules SET status=?, approved_by=?, approved_at=?"
                " WHERE canonical_key=?",
                (status, actor if status == APPROVED else None, stamp, key),
            )
            self._log(key, status, actor, detail)
            self.conn.commit()

    def retire_rule(self, key: str, actor: str, reason: str = "") -> dict[str, Any]:
        """Withdraw an approved rule, and say what it already decided.

        Approve-once means an approved rule keeps deciding without coming back
        to anyone, so withdrawing one has to be possible and has to be
        accountable. The decisions it already made are not reversed here --
        whether they should be is a judgement this system must not make alone
        -- but they are enumerated, because the alternative is discovering them
        one complaint at a time.
        """
        touched = self.cases_using(key)
        with self._lock:
            self.conn.execute(
                "UPDATE rules SET status=? WHERE canonical_key=?",
                (RETIRED, key))
            self._log(key, RETIRED, actor,
                      f"{reason} (decided {len(touched)} case(s) while in force)")
            self.conn.commit()
        return {
            "canonical_key": key,
            "cases_affected": len(touched),
            "cases": [{"case_id": c["case_id"], "question": c["question"],
                       "created_at": c["created_at"],
                       "conclusions": c["conclusions"]} for c in touched],
        }

    def impact_of(self, key: str) -> dict[str, Any]:
        """What retiring this rule would touch, without touching it."""
        touched = self.cases_using(key)
        return {
            "canonical_key": key,
            "cases_affected": len(touched),
            "cases": [{"case_id": c["case_id"], "question": c["question"],
                       "created_at": c["created_at"]} for c in touched],
        }

    # -- what a rule's record says about it --------------------------------
    def note_outcome(self, keys: list[str], correct: bool) -> None:
        """An audit verdict, attributed to every rule in the derivation.

        Crude: a rule is credited or blamed for a decision it merely took part
        in. Proper attribution needs something like a Shapley value over the
        derivation, and until then a rule with a poor record may be carrying
        someone else's mistake -- which is why this flags for re-review rather
        than retiring anything by itself.
        """
        if not keys:
            return
        column = "audit_right" if correct else "audit_wrong"
        with self._lock:
            self.conn.executemany(
                f"UPDATE rules SET {column} = {column} + 1 WHERE canonical_key = ?",
                [(k,) for k in keys])
            self.conn.commit()

    def needs_rereview(self, floor: int = 3, threshold: float = 0.7
                       ) -> list[RuleRow]:
        """Approved rules whose record has turned against them.

        The one circumstance in which an approved rule comes back to a person,
        and it is justified because the evidence changed -- not because the
        reviewer is being asked to repeat themselves.
        """
        out = []
        for row in self.rules(APPROVED):
            seen = row.audit_right + row.audit_wrong
            if seen >= floor and row.audit_right / seen < threshold:
                out.append(row)
        return out

    def note_use(self, keys: list[str]) -> None:
        if not keys:
            return
        with self._lock:
            self.conn.executemany(
                "UPDATE rules SET uses = uses + 1 WHERE canonical_key = ?",
                [(k,) for k in keys],
            )
            self.conn.commit()

    def get_rule(self, key: str) -> RuleRow | None:
        row = self.conn.execute(
            "SELECT * FROM rules WHERE canonical_key=?", (key,)).fetchone()
        return RuleRow(**dict(row)) if row else None

    def rules(self, status: str | None = None) -> list[RuleRow]:
        sql = "SELECT * FROM rules"
        args: tuple = ()
        if status:
            sql += " WHERE status=?"
            args = (status,)
        sql += " ORDER BY uses DESC, proposed_at DESC"
        return [RuleRow(**dict(r)) for r in self.conn.execute(sql, args)]

    def active_rules(self) -> list[RuleRow]:
        """What the engine is allowed to run: approved only."""
        return self.rules(APPROVED)

    # -- cases -------------------------------------------------------------
    def record_case(self, case_id: str, question: str, facts: list[tuple],
                    conclusions: list[str], rules_used: list[str],
                    decided_by: str, tier: str = "", covered: bool = False,
                    session: str = "", status: str = "settled",
                    sampled: bool = False, reply: str = "",
                    drafts: dict | None = None) -> None:
        """Every decision, whatever its tier, appended not merged.

        A row with `status='waiting'` is an inquiry parked for an expert. It
        lives here rather than in the graph's checkpointer on purpose: an
        in-memory checkpointer loses every pending approval when the process
        restarts, and the person who was waiting is left waiting for something
        that no longer exists.
        """
        with self._lock:
            self.conn.execute(
                "INSERT OR REPLACE INTO cases (case_id, created_at, question,"
                " facts, conclusions, rules_used, decided_by, tier, covered,"
                " session, status, sampled, reply, drafts)"
                " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (case_id, _now(), question, json.dumps(facts),
                 json.dumps(conclusions), json.dumps(rules_used), decided_by,
                 tier, int(covered), session, status, int(sampled), reply,
                 json.dumps(drafts or {})),
            )
            self.conn.commit()
        self.note_use(rules_used)

    def get_case(self, case_id: str) -> dict[str, Any] | None:
        row = self.conn.execute(
            "SELECT * FROM cases WHERE case_id=?", (case_id,)).fetchone()
        if row is None:
            return None
        d = dict(row)
        for key in ("facts", "conclusions", "rules_used"):
            d[key] = json.loads(d[key])
        d["drafts"] = _drafts(d.get("drafts"))
        return d

    def audit_queue(self) -> list[dict[str, Any]]:
        """Decisions pulled for audit that nobody has ruled on yet.

        Sampling that produces no queue is a number in a column. The rate is
        the promise; this is where it is kept.
        """
        return [c for c in self.cases(limit=10_000)
                if c.get("sampled") and c.get("status") == "settled"
                and not c.get("approved_by")]

    def waiting(self) -> list[dict[str, Any]]:
        """The expert queue: inquiries that cannot proceed without a person."""
        return [c for c in self.cases(limit=10_000) if c.get("status") == "waiting"]

    def audit_case(self, case_id: str, correct: bool, actor: str,
                   note: str = "") -> bool:
        """An auditor's verdict on a decision that already took effect.

        Not an approval. The decision stands either way -- it has happened --
        and what this changes is the record of the rules behind it.
        """
        case = self.get_case(case_id)
        if case is None:
            return False
        with self._lock:
            self.conn.execute(
                "UPDATE cases SET approved_by=?, approved_at=?, approval_note=?,"
                " status=? WHERE case_id=?",
                (actor, _now(), note,
                 "audited_ok" if correct else "audited_wrong", case_id))
            self.conn.commit()
        self.note_outcome(case.get("rules_used") or [], correct)
        return True

    def settle_case(self, case_id: str, status: str, actor: str,
                    note: str = "", reply: str = "") -> bool:
        """An expert's verdict on one parked decision.

        Distinct from approving a rule, and the two must never be merged. This
        releases a single inquiry; approving a rule governs every case that
        matches it from now on. Collapsing them would let one case quietly
        re-license the logic behind it.
        """
        if status not in ("approved", "rejected"):
            raise ValueError(f"unknown status {status!r}")
        with self._lock:
            cur = self.conn.execute(
                "UPDATE cases SET status=?, approved_by=?, approved_at=?,"
                " approval_note=?, reply=COALESCE(NULLIF(?,''), reply)"
                " WHERE case_id=? AND status='waiting'",
                (status, actor, _now(), note, reply, case_id))
            self.conn.commit()
        return cur.rowcount > 0

    def cases(self, limit: int = 200) -> list[dict[str, Any]]:
        out = []
        for r in self.conn.execute(
            "SELECT * FROM cases ORDER BY created_at DESC LIMIT ?", (limit,)
        ):
            d = dict(r)
            for key in ("facts", "conclusions", "rules_used"):
                d[key] = json.loads(d[key])
            d["drafts"] = _drafts(d.get("drafts"))
            out.append(d)
        return out

    def cases_using(self, key: str) -> list[dict[str, Any]]:
        return [c for c in self.cases(limit=10_000) if key in c["rules_used"]]

    # -- events ------------------------------------------------------------
    def _log(self, key: str, event: str, actor: str, detail: str = "") -> None:
        self.conn.execute(
            "INSERT INTO rule_events (canonical_key, at, event, actor, detail)"
            " VALUES (?,?,?,?,?)", (key, _now(), event, actor, detail))

    def events(self, key: str | None = None, limit: int = 200) -> list[dict]:
        sql = "SELECT * FROM rule_events"
        args: tuple = ()
        if key:
            sql += " WHERE canonical_key=?"
            args = (key,)
        sql += " ORDER BY id DESC LIMIT ?"
        return [dict(r) for r in self.conn.execute(sql, args + (limit,))]

    def vocabulary(self, limit: int = 24) -> list[str]:
        """Predicates the extractor may use — conditions only, never conclusions.

        Fed to the extractor so it reuses terms instead of coining synonyms: a
        base holding both `debt_disputed` and `disputes_arrears` matches neither
        reliably, and the coverage it reports is an illusion.

        Rule *heads* are excluded, and that exclusion is the point. Offered
        `needs_supervisor_review` as vocabulary, the extractor duly wrote it
        down as a fact about the case — asserting a conclusion as evidence, so
        that rules concluding it fire on a premise nobody established. A term
        that some rule concludes is never something the world reports.
        """
        import re as _re
        rows = self.rules()                 # already ordered by uses, desc
        heads = {row.head for row in rows}
        seen: list[str] = []
        for row in rows:
            body = row.text.split("<-", 1)
            if len(body) < 2:
                continue        # a bare fact states no conditions
            for term in _re.findall(r"([a-z_][a-z0-9_]*)\s*\(", body[1]):
                if term not in heads and term not in seen:
                    seen.append(term)
        # Capped, and the cap is not cosmetic: this list is sent on every
        # extraction, so an uncapped vocabulary is a bill that grows with the
        # database forever. Ordered by use, so what is kept is what actually
        # gets matched.
        return seen[:limit]

    def reset(self) -> dict[str, int]:
        """Empty every table. Returns what was destroyed.

        Irreversible, and it takes the case log with it — which is not merely
        history but the evidence any future generalisation would be judged
        against. A reset is therefore not a tidy-up; it is starting the
        database's life over, and the counts are returned so a caller can say
        what was lost rather than report a cheerful success.
        """
        before = {
            "rules": len(self.rules()),
            "cases": len(self.cases(limit=10_000)),
            "events": len(self.events(limit=10_000)),
        }
        with self._lock:
            for table in ("rule_events", "cases", "rules"):
                self.conn.execute(f"DELETE FROM {table}")
            self.conn.execute("DELETE FROM sqlite_sequence WHERE name='rule_events'")
            self.conn.commit()
            self.conn.execute("VACUUM")
        return before

    # -- the number that says whether any of this is working ---------------
    def stats(self) -> dict[str, Any]:
        """Coverage is the headline: the share of cases the rules decided.

        It starts at zero and every approved generalization should move it. A
        plateau means the rules are not generalizing and the database has become
        a cache of past answers — a failure worth seeing rather than discovering.
        """
        counts = {r["status"]: r["n"] for r in self.conn.execute(
            "SELECT status, COUNT(*) n FROM rules GROUP BY status")}
        decided = {r["decided_by"]: r["n"] for r in self.conn.execute(
            "SELECT decided_by, COUNT(*) n FROM cases GROUP BY decided_by")}
        statuses = {r["status"]: r["n"] for r in self.conn.execute(
            "SELECT status, COUNT(*) n FROM cases GROUP BY status")}
        sampled = self.conn.execute(
            "SELECT COUNT(*) n FROM cases WHERE sampled=1").fetchone()["n"]
        total = sum(decided.values())
        return {
            "rules": counts,
            "rules_total": sum(counts.values()),
            "cases": decided,
            "cases_total": total,
            "coverage": (decided.get("rules", 0) / total) if total else 0.0,
            "statuses": statuses,
            "waiting": statuses.get("waiting", 0),
            "sampled": sampled,
        }


__all__ = ["APPROVED", "PROPOSED", "REJECTED", "RETIRED", "RuleRow", "Store"]
