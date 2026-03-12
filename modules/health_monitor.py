"""health_monitor.py — Operational health monitoring for sleep loop.

Collects metrics from SQLite tables, evaluates 3 rules, and produces
alert candidates for health_alerts.py to persist.

v1: tick_dead, ai_disconnected, wm_stalled. No remediation.
"""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timedelta
from typing import Any

from modules.config import TZ_COL

# ---------------------------------------------------------------------------
# Expected tick intervals (in sleep-loop cycles, ~30min each)
# ---------------------------------------------------------------------------
CYCLE_MINUTES = 30

EXPECTED_TICK_INTERVALS = {
    "prospective": 1,
    "health": 1,
    "proactive_contact": 1,
    "self_model": 3,
    "reconsolidation": 3,
    "homeostasis": 3,
    "consolidation": 6,
    "curiosity": 6,
    "backup": 6,
    "sharpe_insights": 6,
    "causal_discovery": 12,
    "curiosity_resolve": 12,
}


# ---------------------------------------------------------------------------
# Metric collection (all SQLite, no PG in v1)
# ---------------------------------------------------------------------------
def collect_metrics(conn: sqlite3.Connection, now_iso: str) -> dict[str, Any]:
    """Collect all health metrics from SQLite. Returns flat dict."""
    since_24h = _hours_ago_iso(now_iso, 24)
    return {
        "wm": _collect_wm(conn, since_24h),
        "sleep_loop": _collect_sleep_loop(conn),
        "global": _collect_global(conn, since_24h),
        "active_inference": _collect_ai(conn, since_24h),
    }


def _collect_wm(conn: sqlite3.Connection, since_24h: str) -> dict:
    row = conn.execute(
        """SELECT COUNT(*), MAX(created_at) FROM write_queue_log
           WHERE kind = 'remember' AND status = 'done' AND created_at > ?""",
        (since_24h,),
    ).fetchone()
    return {"writes_24h": int(row[0] or 0), "latest_write_at": row[1]}


def _collect_sleep_loop(conn: sqlite3.Connection) -> dict:
    rows = conn.execute(
        "SELECT key, value FROM sleep_loop_state WHERE key LIKE 'last_%'"
    ).fetchall()
    return {key: value for key, value in rows}


def _collect_global(conn: sqlite3.Connection, since_24h: str) -> dict:
    write_row = conn.execute(
        """SELECT COUNT(*), MAX(created_at) FROM write_queue_log
           WHERE status = 'done' AND created_at > ?""",
        (since_24h,),
    ).fetchone()
    event_row = conn.execute(
        """SELECT COALESCE(SUM(count), 0), MAX(last_seen) FROM event_counts
           WHERE last_seen > ?""",
        (since_24h,),
    ).fetchone()
    return {
        "events_24h": int(event_row[0] or 0),
        "writes_24h": int(write_row[0] or 0),
        "latest_write_job_at": write_row[1],
    }


def _collect_ai(conn: sqlite3.Connection, since_24h: str) -> dict:
    if not _table_exists(conn, "generative_model_transitions"):
        return {"total_observations": 0, "table_exists": False}
    total = conn.execute(
        "SELECT COALESCE(SUM(count), 0) FROM generative_model_transitions WHERE updated_at > ?",
        (since_24h,),
    ).fetchone()[0]
    return {"total_observations": int(total or 0), "table_exists": True}


# ---------------------------------------------------------------------------
# Rules (3 for v1)
# ---------------------------------------------------------------------------
def evaluate_rules(
    metrics: dict[str, Any],
    now_iso: str,
) -> list[dict[str, Any]]:
    """Evaluate health rules and return alert candidates."""
    alerts: list[dict[str, Any]] = []

    wm = _rule_wm_stalled(metrics, now_iso)
    if wm:
        alerts.append(wm)

    ai = _rule_ai_disconnected(metrics, now_iso)
    if ai:
        alerts.append(ai)

    alerts.extend(_rule_tick_dead(metrics, now_iso))
    return alerts


def _rule_wm_stalled(metrics: dict, now_iso: str) -> dict | None:
    global_writes = metrics["global"]["writes_24h"]
    wm_writes = metrics["wm"]["writes_24h"]
    if global_writes >= 20 and wm_writes == 0:
        return {
            "alert_key": "wm_stalled",
            "subsystem": "working_memory",
            "severity": "critical",
            "title": "Working memory stalled",
            "description": "No remember jobs completed despite ongoing write activity.",
            "recommended_action": "Verify write-worker liveness and datastore reachability.",
            "evidence": {
                "global_writes_24h": global_writes,
                "wm_writes_24h": wm_writes,
                "latest_write_at": metrics["wm"]["latest_write_at"],
            },
        }
    return None


_AI_MIN_WRITES_TO_CHECK = 20  # same as _PERSIST_INTERVAL; AI won't persist below this


def _rule_ai_disconnected(metrics: dict, now_iso: str) -> dict | None:
    writes = metrics["global"]["writes_24h"]
    events = metrics["global"]["events_24h"]
    observations = metrics["active_inference"]["total_observations"]
    enough_activity = writes >= _AI_MIN_WRITES_TO_CHECK
    if enough_activity and observations == 0:
        return {
            "alert_key": "ai_disconnected",
            "subsystem": "active_inference",
            "severity": "critical",
            "title": "Active inference disconnected",
            "description": "No persisted observations despite sufficient upstream activity.",
            "recommended_action": "Verify observation handler wiring and transition persistence.",
            "evidence": {
                "events_24h": events,
                "writes_24h": writes,
                "total_observations": observations,
                "table_exists": metrics["active_inference"]["table_exists"],
            },
        }
    return None


def _rule_tick_dead(metrics: dict, now_iso: str) -> list[dict]:
    alerts = []
    sleep = metrics["sleep_loop"]
    for tick_name, expected_cycles in EXPECTED_TICK_INTERVALS.items():
        last_at = sleep.get(f"last_{tick_name}")
        if not last_at:
            continue
        age_min = _minutes_since(last_at, now_iso)
        max_age = expected_cycles * CYCLE_MINUTES * 2
        if age_min > max_age:
            alerts.append({
                "alert_key": "tick_dead",
                "subsystem": "sleep_loop",
                "severity": "critical",
                "title": f"Sleep loop tick overdue: {tick_name}",
                "description": "Tick has not executed within its expected cycle window.",
                "recommended_action": "Retry tick on next eligible cycle.",
                "evidence": {
                    "tick_name": tick_name,
                    "last_at": last_at,
                    "age_minutes": round(age_min),
                    "max_age_minutes": max_age,
                },
                "dedupe_suffix": tick_name,
            })
    return alerts


# ---------------------------------------------------------------------------
# Entry point (called from _tick_health in sleep_loop.py)
# ---------------------------------------------------------------------------
def run_health_check(conn: sqlite3.Connection, now_iso: str) -> dict[str, Any]:
    """Collect metrics, evaluate rules, persist alerts. Returns summary."""
    from modules.health_alerts import fetch_open_alerts, upsert_health_alerts, auto_resolve_cleared

    metrics = collect_metrics(conn, now_iso)
    candidates = evaluate_rules(metrics, now_iso)
    run_id = f"health_{now_iso.replace(':', '').replace('-', '')}"

    open_alerts = fetch_open_alerts(conn)
    alert_ids = upsert_health_alerts(conn, candidates, run_id, now_iso)
    resolved = auto_resolve_cleared(conn, candidates, now_iso)

    return {
        "metrics_collected": len(metrics),
        "alerts_fired": len(candidates),
        "alert_ids": alert_ids,
        "open_alerts": len(open_alerts) - resolved,
        "auto_resolved": resolved,
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _minutes_since(older_iso: str, newer_iso: str) -> float:
    older = _parse_iso(older_iso)
    newer = _parse_iso(newer_iso)
    return max(0.0, (newer - older).total_seconds() / 60.0)


def _hours_ago_iso(now_iso: str, hours: int) -> str:
    return (_parse_iso(now_iso) - timedelta(hours=hours)).isoformat()


def _parse_iso(value: str) -> datetime:
    dt = datetime.fromisoformat(value)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=TZ_COL)
    return dt


def _table_exists(conn: sqlite3.Connection, name: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=? LIMIT 1",
        (name,),
    ).fetchone()
    return row is not None
