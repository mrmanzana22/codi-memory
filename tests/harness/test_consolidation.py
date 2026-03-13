"""
P1 Evaluation Harness — Consolidation Coverage
================================================
Measures: % of eligible memories processed per consolidation cycle.

Data source: PostgreSQL consolidation_log table (primary)
  - episodes_scanned: memories analyzed per run
  - clusters_found: semantic clusters discovered
  - facts_extracted / facts_created: knowledge facts produced

Note: SQLite consolidation_log stopped receiving writes on 2026-03-03
when _log_consolidation_run migrated to PG-only. Always query PG.

Target: >= 80% cumulative coverage over 7 days
  (total_episodes_scanned / total_memories across all runs)
"""

import json
import os
import sys
from datetime import datetime, timedelta

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))


def run_consolidation_test(days: int = 7) -> dict:
    """Measure consolidation coverage over rolling window using PG."""
    try:
        from modules.config_pg import get_conn
        from modules.pg_store import pg
    except ImportError as e:
        return {
            "test": "consolidation_coverage",
            "skipped": True,
            "skip_reason": f"psycopg unavailable — run with venv: {e}",
            "pass": None,
        }

    # Total memory count
    total_memories = 0
    try:
        info = pg.count()
        total_memories = getattr(info, "points_count", 0) or 0
    except Exception:
        pass

    # Query PG consolidation_log
    try:
        with get_conn() as conn:
          with conn.cursor() as cur:
            cutoff = (datetime.utcnow() - timedelta(days=days)).isoformat()

            # Aggregated stats
            cur.execute("""
                SELECT
                    COUNT(*) as total_runs,
                    ROUND(AVG(episodes_scanned)::numeric, 1) as avg_episodes,
                    COALESCE(SUM(episodes_scanned), 0) as total_episodes_scanned,
                    COALESCE(SUM(clusters_found), 0) as total_clusters,
                    COALESCE(SUM(facts_extracted), 0) as total_facts_extracted,
                    COALESCE(SUM(facts_created), 0) as total_facts_created,
                    ROUND((AVG(duration_ms) / 1000.0)::numeric, 1) as avg_duration_s
                FROM consolidation_log
                WHERE created_at > %s
            """, (cutoff,))
            row = cur.fetchone()

            total_runs = row[0] or 0
            avg_episodes = float(row[1]) if row[1] else 0
            total_episodes = row[2] or 0
            total_clusters = row[3] or 0
            total_facts_extracted = row[4] or 0
            total_facts_created = row[5] or 0
            avg_duration_s = float(row[6]) if row[6] else 0

            # Per-run details (last 10)
            cur.execute("""
                SELECT
                    batch_id, scope, episodes_scanned, clusters_found,
                    facts_extracted, facts_created,
                    ROUND((duration_ms / 1000.0)::numeric, 1) as duration_s,
                    created_at
                FROM consolidation_log
                WHERE created_at > %s
                ORDER BY created_at DESC
                LIMIT 10
            """, (cutoff,))
            run_rows = cur.fetchall()

            runs = []
            for r in run_rows:
                runs.append({
                    "batch_id": r[0],
                    "scope": r[1],
                    "episodes_scanned": r[2],
                    "clusters_found": r[3],
                    "facts_extracted": r[4],
                    "facts_created": r[5],
                    "duration_s": float(r[6]) if r[6] else 0,
                    "created_at": r[7].isoformat() if r[7] else None,
                })

            # Scope breakdown
            cur.execute("""
                SELECT scope, COUNT(*), SUM(episodes_scanned),
                       SUM(facts_created), ROUND(AVG(duration_ms)::numeric, 0)
                FROM consolidation_log
                WHERE created_at > %s
                GROUP BY scope
            """, (cutoff,))
            scope_rows = cur.fetchall()
            by_scope = {}
            for s in scope_rows:
                by_scope[s[0]] = {
                    "runs": s[1],
                    "episodes": s[2] or 0,
                    "facts_created": s[3] or 0,
                    "avg_duration_ms": int(s[4]) if s[4] else 0,
                }

    except Exception as e:
        return {"test": "consolidation_coverage", "error": f"PG query failed: {e}"}

    # Coverage: cumulative episodes scanned / total memories
    coverage = round(total_episodes / max(total_memories, 1), 3) if total_memories else 0

    return {
        "test": "consolidation_coverage",
        "source": "postgresql",
        "window_days": days,
        "total_runs": total_runs,
        "total_episodes_scanned": total_episodes,
        "total_memories": total_memories,
        "cumulative_coverage": coverage,
        "avg_episodes_per_run": avg_episodes,
        "total_clusters": total_clusters,
        "total_facts_extracted": total_facts_extracted,
        "total_facts_created": total_facts_created,
        "avg_duration_s": avg_duration_s,
        "target": 0.80,
        "pass": coverage >= 0.80,
        "by_scope": by_scope,
        "recent_runs": runs,
    }


if __name__ == "__main__":
    result = run_consolidation_test()
    print(json.dumps(result, indent=2, ensure_ascii=False))
