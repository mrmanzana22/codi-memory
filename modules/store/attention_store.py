"""Attention store — attention_transitions table."""

from __future__ import annotations

from typing import Dict, List, Optional

from modules.store._base import SQLiteConnProvider, default_sqlite_provider
from modules.store import store_traced


class AttentionStore:
    """Repository for attention transition data.

    Owns: attention_transitions table (SQLite).
    Invariant: methods return [] or {} on empty data, never None.
    """

    def __init__(self, conn_provider: Optional[SQLiteConnProvider] = None):
        self._get_conn = conn_provider or default_sqlite_provider

    @store_traced
    def get_policy_diversity(self, days: int = 7) -> int:
        """Count distinct attention drivers in recent transitions.

        Replaces collect_active_inference_metrics() in cognitive_contracts.py.
        Returns count of distinct drivers (capped at 4 per Options Framework).
        """
        conn = self._get_conn()
        rows = conn.execute(
            """SELECT DISTINCT driver FROM attention_transitions
               WHERE created_at > datetime('now', ? || ' days')""",
            (str(-days),),
        ).fetchall()
        return min(4, len(rows)) if rows else 0

    @store_traced
    def get_topic_transitions(self, limit: int = 20) -> List[dict]:
        """Get aggregated topic transition patterns.

        Returns list of dicts with: hour, to_topic, count.
        """
        conn = self._get_conn()
        rows = conn.execute(
            """SELECT strftime('%H', created_at) as hour, to_topic, COUNT(*) as cnt
               FROM attention_transitions
               GROUP BY hour, to_topic
               ORDER BY cnt DESC LIMIT ?""",
            (limit,),
        ).fetchall()
        return [{"hour": r[0], "to_topic": r[1], "count": r[2]} for r in rows]

    @store_traced
    def get_transition_count(self, since: Optional[str] = None) -> int:
        """Count total attention transitions, optionally since a date."""
        conn = self._get_conn()
        if since:
            row = conn.execute(
                "SELECT COUNT(*) FROM attention_transitions WHERE created_at > ?",
                (since,),
            ).fetchone()
        else:
            row = conn.execute("SELECT COUNT(*) FROM attention_transitions").fetchone()
        return row[0] if row else 0
