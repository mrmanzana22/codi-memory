"""
P1 Evaluation Harness — Emotion Variance
==========================================
Measures: PAD (Pleasure-Arousal-Dominance) variance over rolling window.

Data source: emotional_state table in memories_fts.db
  - pleasure, arousal, dominance columns
  - std dev > 0.1 on at least one dimension means the system is responsive

Target: std > 0.1 on at least one PAD dimension over 7 days.
"""

import json
import math
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


def _std(values: list) -> float:
    """Population standard deviation."""
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    variance = sum((x - mean) ** 2 for x in values) / len(values)
    return math.sqrt(variance)


def run_emotion_test(days: int = 7) -> dict:
    """Measure PAD variance over rolling window."""
    db_path = _get_db_path()
    if not os.path.exists(db_path):
        return {"test": "emotion_variance", "error": f"DB not found: {db_path}"}

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    # Check table exists
    tables = [r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    ).fetchall()]

    # Try emotional_state or emotion_history
    table_name = None
    for candidate in ["emotional_state", "emotion_history", "pad_history"]:
        if candidate in tables:
            table_name = candidate
            break

    if not table_name:
        # PAD history is in-memory (resets on restart). Measure from
        # memory payloads that have pad_pleasure/pad_arousal/pad_dominance tags.
        conn.close()
        try:
            from modules.pg_store import pg
            # Scroll recent memories and extract PAD values from payloads
            p_vals, a_vals, d_vals = [], [], []
            offset = None
            for _ in range(5):  # max 5 pages
                batch, next_offset = pg.scroll(
                    filters={}, limit=200, is_semantic=False, offset=offset
                )
                if not batch:
                    break
                for pt in batch:
                    p = pt.payload.get("pad_pleasure")
                    a = pt.payload.get("pad_arousal")
                    d = pt.payload.get("pad_dominance")
                    if p is not None and a is not None and d is not None:
                        p_vals.append(float(p))
                        a_vals.append(float(a))
                        d_vals.append(float(d))
                if not next_offset:
                    break
                offset = next_offset

            if len(p_vals) < 5:
                return {
                    "test": "emotion_variance",
                    "error": f"Only {len(p_vals)} memories with PAD tags",
                    "samples": len(p_vals),
                }

            p_std = round(_std(p_vals), 4)
            a_std = round(_std(a_vals), 4)
            d_std = round(_std(d_vals), 4)
            max_std = max(p_std, a_std, d_std)

            return {
                "test": "emotion_variance",
                "source": "memory_payloads",
                "window_days": days,
                "samples": len(p_vals),
                "pleasure_std": p_std,
                "arousal_std": a_std,
                "dominance_std": d_std,
                "max_std": max_std,
                "target": 0.1,
                "pass": max_std >= 0.1,
                "pleasure_range": [round(min(p_vals), 3), round(max(p_vals), 3)],
                "arousal_range": [round(min(a_vals), 3), round(max(a_vals), 3)],
                "dominance_range": [round(min(d_vals), 3), round(max(d_vals), 3)],
            }
        except Exception as e:
            return {"test": "emotion_variance", "error": str(e)}

    # Query from SQLite table
    rows = conn.execute(f"""
        SELECT pleasure, arousal, dominance
        FROM {table_name}
        WHERE created_at > datetime('now', ? || ' days')
        ORDER BY created_at
    """, (f"-{days}",)).fetchall()

    conn.close()

    if not rows:
        return {
            "test": "emotion_variance",
            "error": f"No emotion data in last {days} days",
            "table": table_name,
        }

    p_vals = [r["pleasure"] for r in rows]
    a_vals = [r["arousal"] for r in rows]
    d_vals = [r["dominance"] for r in rows]

    p_std = round(_std(p_vals), 4)
    a_std = round(_std(a_vals), 4)
    d_std = round(_std(d_vals), 4)
    max_std = max(p_std, a_std, d_std)

    return {
        "test": "emotion_variance",
        "source": table_name,
        "window_days": days,
        "samples": len(rows),
        "pleasure_std": p_std,
        "arousal_std": a_std,
        "dominance_std": d_std,
        "max_std": max_std,
        "target": 0.1,
        "pass": max_std >= 0.1,
        "pleasure_range": [round(min(p_vals), 3), round(max(p_vals), 3)],
        "arousal_range": [round(min(a_vals), 3), round(max(a_vals), 3)],
        "dominance_range": [round(min(d_vals), 3), round(max(d_vals), 3)],
    }


if __name__ == "__main__":
    result = run_emotion_test()
    print(json.dumps(result, indent=2, ensure_ascii=False))
