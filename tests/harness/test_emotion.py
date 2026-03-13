"""
P1 Evaluation Harness — Emotion Variance
==========================================
Measures: PAD (Pleasure-Arousal-Dominance) variance over rolling window.

Data source (priority order):
  1. PostgreSQL memories table — emotion_p, emotion_a, emotion_d columns
  2. SQLite emotional_state table (legacy)

The system encodes PAD at memory-creation time via affective charge.
Target: range (max-min) > 0.1 on at least one PAD dimension over 30 days.
  (Range measures responsiveness better than std when many memories encode PAD=0.)
"""

import json
import math
import os
import sys
from datetime import datetime, timedelta

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))


def _std(values: list) -> float:
    """Population standard deviation."""
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    variance = sum((x - mean) ** 2 for x in values) / len(values)
    return math.sqrt(variance)


def run_emotion_test(days: int = 30) -> dict:
    """Measure PAD variance over rolling window.

    Queries PG memories table directly using emotion_p/a/d columns.
    Falls back to SQLite emotional_state if PG unavailable.
    """
    total_in_window = 0

    # --- Strategy 1: PostgreSQL memories (emotion_p/a/d columns) ---
    try:
        from modules.config_pg import get_conn

        with get_conn() as conn:
            with conn.cursor() as cur:
                cutoff = (datetime.utcnow() - timedelta(days=days)).isoformat()

                # Get all non-zero emotion values in window
                cur.execute("""
                    SELECT emotion_p, emotion_a, emotion_d
                    FROM memories
                    WHERE emotion_p IS NOT NULL
                      AND (emotion_p != 0 OR emotion_a != 0 OR emotion_d != 0)
                      AND created_at > %s
                    ORDER BY created_at
                """, (cutoff,))
                rows = cur.fetchall()

                # Also count total for reporting
                cur.execute("""
                    SELECT COUNT(*) FROM memories
                    WHERE emotion_p IS NOT NULL AND created_at > %s
                """, (cutoff,))
                total_in_window = cur.fetchone()[0]

                if len(rows) >= 5:
                    p_vals = [float(r[0]) for r in rows]
                    a_vals = [float(r[1]) for r in rows]
                    d_vals = [float(r[2]) for r in rows]

                    p_std = round(_std(p_vals), 4)
                    a_std = round(_std(a_vals), 4)
                    d_std = round(_std(d_vals), 4)
                    p_range = round(max(p_vals) - min(p_vals), 4)
                    a_range = round(max(a_vals) - min(a_vals), 4)
                    d_range = round(max(d_vals) - min(d_vals), 4)
                    max_range = max(p_range, a_range, d_range)

                    return {
                        "test": "emotion_variance",
                        "source": "postgresql",
                        "window_days": days,
                        "samples": len(rows),
                        "total_in_window": total_in_window,
                        "pleasure_std": p_std,
                        "arousal_std": a_std,
                        "dominance_std": d_std,
                        "max_std": max(p_std, a_std, d_std),
                        "pleasure_range": p_range,
                        "arousal_range": a_range,
                        "dominance_range": d_range,
                        "max_range": max_range,
                        "target": 0.1,
                        "pass": max_range >= 0.1,
                    }

    except Exception as e:
        return {"test": "emotion_variance", "error": f"PG query failed: {e}"}

    # --- Strategy 2: SQLite emotional_state table (legacy) ---
    try:
        db_path = os.environ.get("FTS_DB_PATH")
        if not db_path:
            from modules.config import FTS_DB_PATH
            db_path = FTS_DB_PATH
    except ImportError:
        db_path = os.path.expanduser("~/codi-memory/memories_fts.db")

    if os.path.exists(db_path):
        import sqlite3
        sconn = sqlite3.connect(db_path)
        sconn.row_factory = sqlite3.Row
        tables = [r[0] for r in sconn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()]
        for candidate in ["emotional_state", "emotion_history", "pad_history"]:
            if candidate in tables:
                rows = sconn.execute(f"""
                    SELECT pleasure, arousal, dominance
                    FROM {candidate}
                    WHERE created_at > datetime('now', ? || ' days')
                    ORDER BY created_at
                """, (f"-{days}",)).fetchall()
                sconn.close()
                if rows and len(rows) >= 5:
                    p_vals = [r["pleasure"] for r in rows]
                    a_vals = [r["arousal"] for r in rows]
                    d_vals = [r["dominance"] for r in rows]
                    p_std = round(_std(p_vals), 4)
                    a_std = round(_std(a_vals), 4)
                    d_std = round(_std(d_vals), 4)
                    p_range = round(max(p_vals) - min(p_vals), 4)
                    a_range = round(max(a_vals) - min(a_vals), 4)
                    d_range = round(max(d_vals) - min(d_vals), 4)
                    max_range = max(p_range, a_range, d_range)
                    return {
                        "test": "emotion_variance",
                        "source": candidate,
                        "window_days": days,
                        "samples": len(rows),
                        "pleasure_std": p_std,
                        "arousal_std": a_std,
                        "dominance_std": d_std,
                        "max_std": max(p_std, a_std, d_std),
                        "pleasure_range": p_range,
                        "arousal_range": a_range,
                        "dominance_range": d_range,
                        "max_range": max_range,
                        "target": 0.1,
                        "pass": max_range >= 0.1,
                    }
                break
        else:
            sconn.close()

    return {
        "test": "emotion_variance",
        "error": (
            f"Insufficient non-zero PAD data in last {days} days. "
            f"Total memories: {total_in_window}. "
            f"Emotion encoding may not be writing PAD values to new memories."
        ),
        "samples": 0,
    }


if __name__ == "__main__":
    result = run_emotion_test()
    print(json.dumps(result, indent=2, ensure_ascii=False))
