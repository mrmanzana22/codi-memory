"""
P1 Evaluation Harness — Tick Health
====================================
Measures: sleep loop tick success rate over a rolling window.

Data source: tool_calls table in memories_fts.db
  - tool_name LIKE 'sleep_tick_%'
  - success column (1=ok, 0=failed)

Target: >= 95% success rate over 7 days.
"""

import json
import os
import sqlite3
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))


def _get_db_path() -> str:
    """Get FTS DB path from config or env."""
    path = os.environ.get("FTS_DB_PATH")
    if path:
        return path
    try:
        from modules.config import FTS_DB_PATH
        return FTS_DB_PATH
    except ImportError:
        return os.path.expanduser("~/codi-memory/memories_fts.db")


def run_tick_health_test(days: int = 7) -> dict:
    """Measure tick health over rolling window.

    Returns dict with overall success rate and per-tick breakdown.
    """
    db_path = _get_db_path()
    if not os.path.exists(db_path):
        return {"test": "tick_health", "error": f"DB not found: {db_path}"}

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    # Check if tool_calls table exists
    tables = [r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='tool_calls'"
    ).fetchall()]
    if "tool_calls" not in tables:
        conn.close()
        return {"test": "tick_health", "error": "tool_calls table not found"}

    # Overall tick health
    row = conn.execute("""
        SELECT
            COUNT(*) as total,
            SUM(success) as successes
        FROM tool_calls
        WHERE tool_name LIKE 'sleep_tick_%'
          AND started_at >= datetime('now', ? || ' days')
    """, (f"-{days}",)).fetchone()

    total = row["total"] or 0
    successes = row["successes"] or 0
    health_pct = round(successes / total, 3) if total else 0

    # Per-tick breakdown
    per_tick = {}
    rows = conn.execute("""
        SELECT
            tool_name,
            COUNT(*) as attempts,
            SUM(success) as successes,
            ROUND(AVG(duration_ms), 0) as avg_duration_ms,
            MAX(duration_ms) as max_duration_ms
        FROM tool_calls
        WHERE tool_name LIKE 'sleep_tick_%'
          AND started_at >= datetime('now', ? || ' days')
        GROUP BY tool_name
        ORDER BY (1.0 * SUM(success) / COUNT(*)) ASC
    """, (f"-{days}",)).fetchall()

    for r in rows:
        attempts = r["attempts"]
        succ = r["successes"] or 0
        per_tick[r["tool_name"]] = {
            "attempts": attempts,
            "successes": succ,
            "success_pct": round(succ / attempts, 3) if attempts else 0,
            "avg_duration_ms": r["avg_duration_ms"],
            "max_duration_ms": r["max_duration_ms"],
        }

    # Daily breakdown
    daily = []
    day_rows = conn.execute("""
        SELECT
            DATE(started_at) as day,
            COUNT(*) as total,
            SUM(success) as successes
        FROM tool_calls
        WHERE tool_name LIKE 'sleep_tick_%'
          AND started_at >= datetime('now', ? || ' days')
        GROUP BY DATE(started_at)
        ORDER BY day DESC
    """, (f"-{days}",)).fetchall()

    for r in day_rows:
        t = r["total"]
        s = r["successes"] or 0
        daily.append({
            "date": r["day"],
            "total": t,
            "successes": s,
            "health_pct": round(s / t, 3) if t else 0,
        })

    conn.close()

    return {
        "test": "tick_health",
        "window_days": days,
        "total_ticks": total,
        "successful_ticks": successes,
        "health_pct": health_pct,
        "target": 0.95,
        "pass": health_pct >= 0.95,
        "per_tick": per_tick,
        "daily": daily,
    }


if __name__ == "__main__":
    result = run_tick_health_test()
    print(json.dumps(result, indent=2, ensure_ascii=False))
