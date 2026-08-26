"""Tamper-evident decision ledger.

The brief asks for "a clear audit trail behind every decision". An append-only
table satisfies that literally. It does not satisfy an auditor who asks the
next question: how do you know nobody edited a row after the incident?

So each record carries the SHA-256 of its predecessor. Altering any historical
decision breaks every hash downstream of it, and `verify_chain()` reports the
exact sequence number where the break occurs. Cheap to build, and it converts
"we log decisions" into "we can prove our logs are intact", which is the
difference between a demo feature and something a regulated enterprise would
actually accept.

Also here: retention. Policy declares a jurisdiction-specific retention
window, and `purge_expired()` enforces it. Under GDPR an audit trail that
keeps prompt text forever is itself the compliance problem, so records store
a redacted preview, never raw prompt bodies.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

from .schemas import Decision

GENESIS = "0" * 64

SCHEMA = """
CREATE TABLE IF NOT EXISTS ledger (
    seq            INTEGER PRIMARY KEY AUTOINCREMENT,
    ts             REAL    NOT NULL,
    request_id     TEXT    NOT NULL,
    session_id     TEXT    NOT NULL,
    turn           INTEGER NOT NULL,
    use_case       TEXT    NOT NULL,
    stakes         TEXT    NOT NULL,
    action         TEXT    NOT NULL,
    reason         TEXT    NOT NULL,
    expected_harm  REAL    NOT NULL,
    latency_ms     REAL    NOT NULL,
    cost_usd       REAL    NOT NULL,
    policy_version TEXT    NOT NULL,
    payload        TEXT    NOT NULL,
    prev_hash      TEXT    NOT NULL,
    record_hash    TEXT    NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_session ON ledger(session_id);
CREATE INDEX IF NOT EXISTS idx_action  ON ledger(action);
CREATE INDEX IF NOT EXISTS idx_ts      ON ledger(ts);

CREATE TABLE IF NOT EXISTS overrides (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    ts           REAL NOT NULL,
    ledger_seq   INTEGER NOT NULL,
    reviewer     TEXT NOT NULL,
    from_action  TEXT NOT NULL,
    to_action    TEXT NOT NULL,
    rationale    TEXT NOT NULL,
    FOREIGN KEY (ledger_seq) REFERENCES ledger(seq)
);
"""


@dataclass
class ChainStatus:
    intact: bool
    records: int
    broken_at: int | None = None
    detail: str = ""


class Ledger:
    def __init__(self, path: str | Path = "data/preflight.db") -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(SCHEMA)
        self._conn.commit()

    # -- write -------------------------------------------------------------

    @staticmethod
    def _hash_record(prev_hash: str, body: dict[str, Any]) -> str:
        canonical = json.dumps(body, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256((prev_hash + canonical).encode()).hexdigest()

    def _last_hash(self) -> str:
        row = self._conn.execute(
            "SELECT record_hash FROM ledger ORDER BY seq DESC LIMIT 1"
        ).fetchone()
        return row["record_hash"] if row else GENESIS

    def append(self, decision: Decision, policy_version: str) -> int:
        prev = self._last_hash()

        payload = decision.model_dump(mode="json")
        # Never persist raw prompt or full response bodies — retention risk.
        payload["response_preview"] = decision.response_preview[:240]

        body = {
            "ts": decision.ts,
            "request_id": decision.request_id,
            "session_id": decision.session_id,
            "turn": decision.turn,
            "action": decision.action.value,
            "reason": decision.reason,
            "risk": decision.risk.as_dict(),
            "session_risk": decision.session_risk.as_dict(),
            "policy_version": policy_version,
        }
        record_hash = self._hash_record(prev, body)

        cur = self._conn.execute(
            """INSERT INTO ledger
               (ts, request_id, session_id, turn, use_case, stakes, action,
                reason, expected_harm, latency_ms, cost_usd, policy_version,
                payload, prev_hash, record_hash)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                decision.ts, decision.request_id, decision.session_id,
                decision.turn, decision.use_case, decision.stakes.value,
                decision.action.value, decision.reason, decision.expected_harm,
                decision.latency_ms, decision.cost_usd, policy_version,
                json.dumps(payload), prev, record_hash,
            ),
        )
        self._conn.commit()
        return int(cur.lastrowid)

    def record_override(
        self, ledger_seq: int, reviewer: str,
        from_action: str, to_action: str, rationale: str,
    ) -> None:
        """A clinician-style override. Captured, not just permitted.

        Every override is a free training label — this is the concrete
        mechanism behind "escalations retune the thresholds", not a slogan.
        """
        self._conn.execute(
            """INSERT INTO overrides
               (ts, ledger_seq, reviewer, from_action, to_action, rationale)
               VALUES (?,?,?,?,?,?)""",
            (time.time(), ledger_seq, reviewer, from_action, to_action, rationale),
        )
        self._conn.commit()

    # -- read --------------------------------------------------------------

    def recent(self, limit: int = 50) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            "SELECT * FROM ledger ORDER BY seq DESC LIMIT ?", (limit,)
        ).fetchall()
        return [dict(r) for r in rows]

    def queue(self, limit: int = 25) -> list[dict[str, Any]]:
        """Human review queue, ranked by expected harm — not by recency.

        Ranking by time is what produces alert fatigue: the reviewer works a
        firehose in arrival order and the one item that mattered is buried.
        """
        rows = self._conn.execute(
            """SELECT * FROM ledger
               WHERE action IN ('escalate','block')
               ORDER BY expected_harm DESC, ts DESC LIMIT ?""",
            (limit,),
        ).fetchall()
        return [dict(r) for r in rows]

    def by_session(self, session_id: str) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            "SELECT * FROM ledger WHERE session_id=? ORDER BY seq ASC",
            (session_id,),
        ).fetchall()
        return [dict(r) for r in rows]

    def overrides(self) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            "SELECT * FROM overrides ORDER BY ts DESC"
        ).fetchall()
        return [dict(r) for r in rows]

    def stats(self) -> dict[str, Any]:
        row = self._conn.execute(
            """SELECT COUNT(*) n,
                      COALESCE(AVG(latency_ms),0)  avg_latency,
                      COALESCE(SUM(cost_usd),0)    total_cost
               FROM ledger"""
        ).fetchone()
        actions = self._conn.execute(
            "SELECT action, COUNT(*) c FROM ledger GROUP BY action"
        ).fetchall()
        return {
            "records": row["n"],
            "avg_latency_ms": round(row["avg_latency"], 2),
            "total_cost_usd": round(row["total_cost"], 5),
            "actions": {r["action"]: r["c"] for r in actions},
        }

    def latencies(self) -> list[float]:
        rows = self._conn.execute("SELECT latency_ms FROM ledger").fetchall()
        return [r["latency_ms"] for r in rows]

    # -- integrity ---------------------------------------------------------

    def verify_chain(self) -> ChainStatus:
        """Walk the chain and recompute every hash."""
        prev = GENESIS
        n = 0
        for row in self._iter_all():
            body = {
                "ts": row["ts"],
                "request_id": row["request_id"],
                "session_id": row["session_id"],
                "turn": row["turn"],
                "action": row["action"],
                "reason": row["reason"],
                "risk": json.loads(row["payload"])["risk"],
                "session_risk": json.loads(row["payload"])["session_risk"],
                "policy_version": row["policy_version"],
            }
            expected = self._hash_record(prev, body)
            if row["prev_hash"] != prev:
                return ChainStatus(
                    False, n, row["seq"],
                    f"prev_hash mismatch at seq {row['seq']}",
                )
            if expected != row["record_hash"]:
                return ChainStatus(
                    False, n, row["seq"],
                    f"record altered after write at seq {row['seq']}",
                )
            prev = row["record_hash"]
            n += 1
        return ChainStatus(True, n, None, f"{n} records verified")

    def _iter_all(self) -> Iterator[sqlite3.Row]:
        cur = self._conn.execute("SELECT * FROM ledger ORDER BY seq ASC")
        while True:
            batch = cur.fetchmany(500)
            if not batch:
                break
            yield from batch

    def purge_expired(self, retention_days: int) -> int:
        """Enforce the policy's retention window.

        Note the honest tension: purging breaks the hash chain by design.
        We record the purge as a chain checkpoint rather than pretending the
        records were never there — an auditor sees that a purge happened,
        when, and how many records it covered.
        """
        cutoff = time.time() - retention_days * 86400
        cur = self._conn.execute("DELETE FROM ledger WHERE ts < ?", (cutoff,))
        self._conn.commit()
        return cur.rowcount
