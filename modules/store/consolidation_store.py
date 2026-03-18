"""Consolidation store — consolidation_log, reconsolidation_log, labile_memories."""

from __future__ import annotations

from typing import Dict, List, Optional

from modules.store._base import SQLiteConnProvider, default_sqlite_provider
from modules.store import store_traced


class ConsolidationStore:
    """Repository for consolidation pipeline data.

    Owns: consolidation_log, reconsolidation_log, labile_memories tables (SQLite).
    Note: consolidation.py writes to consolidation_log via PostgreSQL.
    This store reads from the SQLite copy (synced by write worker).
    Invariant: methods return [] or {} on empty data, never None.
    """

    def __init__(self, conn_provider: Optional[SQLiteConnProvider] = None):
        self._get_conn = conn_provider or default_sqlite_provider

    def _fetch_runs(self, limit: int = 20) -> List[dict]:
        """Internal: fetch consolidation runs (no tracing — avoids double-trace)."""
        conn = self._get_conn()
        rows = conn.execute(
            """SELECT episodes_scanned, facts_extracted, facts_created, contradictions_found
               FROM consolidation_log
               ORDER BY created_at DESC LIMIT ?""",
            (limit,),
        ).fetchall()
        return [
            {
                "episodes_scanned": r[0] or 0,
                "facts_extracted": r[1] or 0,
                "facts_created": r[2] or 0,
                "contradictions_found": r[3] or 0,
            }
            for r in rows
        ]

    @store_traced
    def get_recent_runs(self, limit: int = 20) -> List[dict]:
        """Get recent consolidation runs.

        Returns list of dicts with: episodes_scanned, facts_extracted,
        facts_created, contradictions_found.
        """
        return self._fetch_runs(limit=limit)

    @store_traced
    def get_consolidation_metrics(self, limit: int = 20) -> Dict[str, Optional[float]]:
        """Compute consolidation metrics for contract evaluation.

        Replaces collect_consolidation_metrics() in cognitive_contracts.py.
        Returns dict with: semantic_extraction_rate, false_memory_rate, consolidation_coverage.
        """
        runs = self._fetch_runs(limit=limit)
        if not runs:
            return {}

        total_episodes = sum(r["episodes_scanned"] for r in runs)
        total_facts = sum(r["facts_extracted"] for r in runs)
        total_contradictions = sum(r["contradictions_found"] for r in runs)
        n_runs = len(runs)

        metrics: Dict[str, Optional[float]] = {
            "semantic_extraction_rate": total_facts / n_runs if n_runs else 0,
        }

        if total_facts > 0:
            metrics["false_memory_rate"] = total_contradictions / total_facts

        # Coverage estimate: episodes consolidated / estimated 50/day over 7 days
        daily_episodes_est = 50
        daily_consolidated = total_episodes / 7.0
        metrics["consolidation_coverage"] = min(1.0, daily_consolidated / daily_episodes_est)

        return metrics

    @store_traced
    def get_reconsolidation_stats(self) -> Dict[str, Optional[float]]:
        """Get reconsolidation statistics.

        Replaces collect_reconsolidation_metrics() in cognitive_contracts.py.
        Returns dict with: correction_success_rate, reconsolidation_trigger_rate.
        """
        conn = self._get_conn()

        total_row = conn.execute("SELECT COUNT(*) FROM reconsolidation_log").fetchone()
        total = total_row[0] if total_row else 0

        success_row = conn.execute(
            "SELECT COUNT(*) FROM reconsolidation_log WHERE new_content IS NOT NULL AND new_content != ''"
        ).fetchone()
        success = success_row[0] if success_row else 0

        metrics: Dict[str, Optional[float]] = {}
        if total > 0:
            metrics["correction_success_rate"] = success / total

        # Trigger rate needs prediction store data (high PE events)
        # Import lazily to avoid circular deps
        pe_row = conn.execute(
            "SELECT COUNT(*) FROM prediction_results WHERE surprise_score > 0.6"
        ).fetchone()
        pe_events = pe_row[0] if pe_row else 0
        if pe_events > 0:
            metrics["reconsolidation_trigger_rate"] = total / pe_events

        return metrics
