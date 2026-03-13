"""
P1 Evaluation Harness — Prediction Accuracy
=============================================
Measures: topic prediction accuracy over rolling window.

Data source: prediction_results table in memories_fts.db
  - predicted_topic vs actual_topic
  - hit column (1=correct, 0=miss)
  - source column (filter to 'interactive' only — exclude sleep_loop)

Target: >= 40% accuracy on interactive predictions.
"""

import json
import os
import sqlite3
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))


def _get_db_path() -> str:
    path = os.environ.get("FTS_DB_PATH")
    if path:
        return path
    try:
        from modules.config import FTS_DB_PATH
        return FTS_DB_PATH
    except ImportError:
        return os.path.expanduser("~/codi-memory/memories_fts.db")


def run_prediction_test(days: int = 7) -> dict:
    """Measure prediction accuracy over rolling window."""
    db_path = _get_db_path()
    if not os.path.exists(db_path):
        return {"test": "prediction_accuracy", "error": f"DB not found: {db_path}"}

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    # Check table exists
    tables = [r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='prediction_results'"
    ).fetchall()]
    if "prediction_results" not in tables:
        conn.close()
        return {"test": "prediction_accuracy", "error": "prediction_results table not found"}

    # Overall accuracy (interactive only — PCI fix)
    row = conn.execute("""
        SELECT
            COUNT(*) as total,
            SUM(hit) as hits,
            ROUND(AVG(surprise_score), 3) as avg_surprise
        FROM prediction_results
        WHERE source = 'interactive'
          AND created_at > datetime('now', ? || ' days')
    """, (f"-{days}",)).fetchone()

    total = row["total"] or 0
    hits = row["hits"] or 0
    accuracy = round(hits / total, 3) if total else 0
    avg_surprise = row["avg_surprise"] or 0

    # Per-topic breakdown
    per_topic = {}
    topic_rows = conn.execute("""
        SELECT
            predicted_topic,
            COUNT(*) as predictions,
            SUM(hit) as hits,
            ROUND(AVG(surprise_score), 3) as avg_surprise
        FROM prediction_results
        WHERE source = 'interactive'
          AND created_at > datetime('now', ? || ' days')
        GROUP BY predicted_topic
        ORDER BY COUNT(*) DESC
    """, (f"-{days}",)).fetchall()

    for r in topic_rows:
        p = r["predictions"]
        h = r["hits"] or 0
        per_topic[r["predicted_topic"]] = {
            "predictions": p,
            "hits": h,
            "accuracy": round(h / p, 3) if p else 0,
            "avg_surprise": r["avg_surprise"],
        }

    # Sleep loop contamination check
    sleep_row = conn.execute("""
        SELECT COUNT(*) as sleep_count
        FROM prediction_results
        WHERE source = 'sleep_loop'
          AND created_at > datetime('now', ? || ' days')
    """, (f"-{days}",)).fetchone()
    sleep_count = sleep_row["sleep_count"] or 0

    # Daily trend
    daily = []
    day_rows = conn.execute("""
        SELECT
            DATE(created_at) as day,
            COUNT(*) as total,
            SUM(hit) as hits
        FROM prediction_results
        WHERE source = 'interactive'
          AND created_at > datetime('now', ? || ' days')
        GROUP BY DATE(created_at)
        ORDER BY day DESC
    """, (f"-{days}",)).fetchall()

    for r in day_rows:
        t = r["total"]
        h = r["hits"] or 0
        daily.append({
            "date": r["day"],
            "total": t,
            "hits": h,
            "accuracy": round(h / t, 3) if t else 0,
        })

    conn.close()

    return {
        "test": "prediction_accuracy",
        "window_days": days,
        "total_predictions": total,
        "hits": hits,
        "accuracy": accuracy,
        "avg_surprise": avg_surprise,
        "target": 0.40,
        "pass": accuracy >= 0.40,
        "sleep_loop_records": sleep_count,
        "pci_clean": sleep_count == 0 or (total > 0 and sleep_count / (total + sleep_count) < 0.1),
        "per_topic": per_topic,
        "daily": daily,
    }


if __name__ == "__main__":
    result = run_prediction_test()
    print(json.dumps(result, indent=2, ensure_ascii=False))
