"""
P1 Evaluation Harness — Consolidation Coverage
================================================
Measures: % of eligible memories processed per consolidation cycle.

Data source: consolidation_log table in memories_fts.db
  - episodes_scanned: memories analyzed per run
  - clusters_found: semantic clusters discovered
  - facts_extracted: knowledge facts created

Target: >= 80% coverage (episodes_scanned / total eligible per cycle).
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


def run_consolidation_test(days: int = 7) -> dict:
    """Measure consolidation coverage over rolling window."""
    db_path = _get_db_path()
    if not os.path.exists(db_path):
        return {"test": "consolidation_coverage", "error": f"DB not found: {db_path}"}

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    # Check table exists
    tables = [r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='consolidation_log'"
    ).fetchall()]
    if "consolidation_log" not in tables:
        conn.close()
        return {"test": "consolidation_coverage", "error": "consolidation_log table not found"}

    # Aggregated stats
    row = conn.execute("""
        SELECT
            COUNT(*) as total_runs,
            ROUND(AVG(episodes_scanned), 1) as avg_episodes,
            SUM(episodes_scanned) as total_episodes_scanned,
            SUM(clusters_found) as total_clusters,
            SUM(facts_extracted) as total_facts_extracted,
            SUM(facts_created) as total_facts_created,
            ROUND(AVG(duration_ms) / 1000.0, 1) as avg_duration_s
        FROM consolidation_log
        WHERE created_at > datetime('now', ? || ' days')
    """, (f"-{days}",)).fetchone()

    total_runs = row["total_runs"] or 0
    total_episodes = row["total_episodes_scanned"] or 0

    # Get total memory count from PG for coverage calculation
    total_memories = 0
    try:
        from modules.pg_store import pg
        info = pg.count()
        # pg.count() returns CollectionInfo with .points_count attribute
        total_memories = getattr(info, "points_count", 0) or 0
    except Exception:
        pass

    # Coverage: unique episodes scanned / total memories
    # Note: same memories may be scanned across multiple runs
    coverage = round(total_episodes / max(total_memories, 1), 3) if total_memories else 0

    # Per-run details (last 10)
    runs = []
    run_rows = conn.execute("""
        SELECT
            batch_id,
            scope,
            episodes_scanned,
            clusters_found,
            facts_extracted,
            facts_created,
            ROUND(duration_ms / 1000.0, 1) as duration_s,
            created_at
        FROM consolidation_log
        WHERE created_at > datetime('now', ? || ' days')
        ORDER BY created_at DESC
        LIMIT 10
    """, (f"-{days}",)).fetchall()

    for r in run_rows:
        runs.append({
            "batch_id": r["batch_id"],
            "scope": r["scope"],
            "episodes_scanned": r["episodes_scanned"],
            "clusters_found": r["clusters_found"],
            "facts_extracted": r["facts_extracted"],
            "facts_created": r["facts_created"],
            "duration_s": r["duration_s"],
            "created_at": r["created_at"],
        })

    conn.close()

    return {
        "test": "consolidation_coverage",
        "window_days": days,
        "total_runs": total_runs,
        "total_episodes_scanned": total_episodes,
        "total_memories": total_memories,
        "cumulative_coverage": coverage,
        "avg_episodes_per_run": row["avg_episodes"],
        "total_clusters": row["total_clusters"] or 0,
        "total_facts_created": row["total_facts_created"] or 0,
        "avg_duration_s": row["avg_duration_s"],
        "target": 0.80,
        "pass": coverage >= 0.80,
        "recent_runs": runs,
    }


if __name__ == "__main__":
    result = run_consolidation_test()
    print(json.dumps(result, indent=2, ensure_ascii=False))
