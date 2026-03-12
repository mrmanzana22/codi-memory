"""health_alerts.py — Persist, deduplicate, and escalate health alerts.

Receives alert candidates (plain dicts) from health_monitor.py and
upserts them into the health_alerts table with dedup + escalation.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
from typing import Any


def fetch_open_alerts(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    """Get all non-resolved alerts."""
    rows = conn.execute(
        """SELECT id, alert_key, subsystem, status, severity, title, description,
                  evidence_json, recommended_action, first_seen_at, last_seen_at,
                  resolved_at, occurrence_count, source_run_id, owner, dedupe_hash,
                  created_at, updated_at
           FROM health_alerts
           WHERE status IN ('open', 'diagnosing', 'actioned')"""
    ).fetchall()
    return [_row_to_dict(row) for row in rows]


def upsert_health_alerts(
    conn: sqlite3.Connection,
    alerts: list[dict[str, Any]],
    run_id: str,
    now_iso: str,
) -> list[int]:
    """Upsert alert candidates with dedup. Returns list of alert IDs."""
    ids: list[int] = []
    for alert in alerts:
        dedupe = _build_dedupe_hash(
            alert["alert_key"], alert["subsystem"], alert.get("dedupe_suffix")
        )
        existing = _find_open(conn, dedupe)
        if existing:
            ids.append(_update(conn, existing, alert, run_id, now_iso))
        else:
            ids.append(_insert(conn, alert, dedupe, run_id, now_iso))
    conn.commit()
    return ids


def _build_dedupe_hash(alert_key: str, subsystem: str, suffix: str | None = None) -> str:
    raw = f"{alert_key}|{subsystem}|{suffix or ''}"
    return hashlib.sha256(raw.encode()).hexdigest()


def _find_open(conn: sqlite3.Connection, dedupe_hash: str) -> dict | None:
    row = conn.execute(
        """SELECT id, alert_key, subsystem, status, severity, title, description,
                  evidence_json, recommended_action, first_seen_at, last_seen_at,
                  resolved_at, occurrence_count, source_run_id, owner, dedupe_hash,
                  created_at, updated_at
           FROM health_alerts
           WHERE dedupe_hash = ? AND status IN ('open', 'diagnosing', 'actioned')
           LIMIT 1""",
        (dedupe_hash,),
    ).fetchone()
    return _row_to_dict(row) if row else None


def _insert(conn: sqlite3.Connection, alert: dict, dedupe: str, run_id: str, now: str) -> int:
    cur = conn.execute(
        """INSERT INTO health_alerts (
             alert_key, subsystem, status, severity, title, description,
             evidence_json, recommended_action, first_seen_at, last_seen_at,
             resolved_at, occurrence_count, source_run_id, owner, dedupe_hash,
             created_at, updated_at
           ) VALUES (?, ?, 'open', ?, ?, ?, ?, ?, ?, ?, NULL, 1, ?, NULL, ?, ?, ?)""",
        (
            alert["alert_key"], alert["subsystem"], alert["severity"],
            alert["title"], alert["description"],
            json.dumps(alert["evidence"], sort_keys=True),
            alert.get("recommended_action"),
            now, now, run_id, dedupe, now, now,
        ),
    )
    return int(cur.lastrowid)


def _build_evidence_json(existing_json: str | None, new_evidence: dict) -> str:
    """Store 1-deep evidence: latest + previous-latest only (no unbounded nesting)."""
    if existing_json:
        existing = json.loads(existing_json)
        prev = existing.get("latest", existing)
    else:
        prev = {}
    return json.dumps({"latest": new_evidence, "previous": prev}, sort_keys=True)


def _update(conn: sqlite3.Connection, existing: dict, alert: dict, run_id: str, now: str) -> int:
    count = int(existing["occurrence_count"]) + 1
    severity = "critical" if existing["severity"] == "warning" and count >= 3 else existing["severity"]
    conn.execute(
        """UPDATE health_alerts
           SET severity = ?, last_seen_at = ?, occurrence_count = ?,
               evidence_json = ?, recommended_action = ?,
               source_run_id = ?, updated_at = ?
           WHERE id = ?""",
        (
            severity, now, count,
            _build_evidence_json(existing.get("evidence_json"), alert["evidence"]),
            alert.get("recommended_action"),
            run_id, now, existing["id"],
        ),
    )
    return int(existing["id"])


def auto_resolve_cleared(
    conn: sqlite3.Connection,
    active_candidates: list[dict[str, Any]],
    now_iso: str,
) -> int:
    """Resolve open alerts whose conditions no longer fire.

    Any open alert whose alert_key is NOT in the current candidate set
    is considered cleared and gets auto-resolved.

    Returns number of alerts resolved.
    """
    active_keys = {a["alert_key"] for a in active_candidates}
    open_alerts = fetch_open_alerts(conn)
    resolved = 0
    for alert in open_alerts:
        if alert["alert_key"] not in active_keys:
            conn.execute(
                """UPDATE health_alerts
                   SET status = 'resolved', resolved_at = ?, updated_at = ?
                   WHERE id = ?""",
                (now_iso, now_iso, alert["id"]),
            )
            resolved += 1
    if resolved:
        conn.commit()
    return resolved


_KEYS = (
    "id", "alert_key", "subsystem", "status", "severity", "title", "description",
    "evidence_json", "recommended_action", "first_seen_at", "last_seen_at",
    "resolved_at", "occurrence_count", "source_run_id", "owner", "dedupe_hash",
    "created_at", "updated_at",
)


def _row_to_dict(row: tuple | None) -> dict[str, Any]:
    if row is None:
        return {}
    return dict(zip(_KEYS, row))
