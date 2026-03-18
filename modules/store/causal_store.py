"""Causal store — causal_discovery_state, spreading_edges, causal_chains."""

from __future__ import annotations

import json
from typing import Dict, List, Optional

from modules.store._base import SQLiteConnProvider, default_sqlite_provider
from modules.store import store_traced


class CausalStore:
    """Repository for causal discovery data.

    Owns: causal_discovery_state, spreading_edges, causal_chains tables (SQLite).
    Invariant: methods return [] or {} on empty data, never None.
    """

    def __init__(self, conn_provider: Optional[SQLiteConnProvider] = None):
        self._get_conn = conn_provider or default_sqlite_provider

    @store_traced
    def get_latest_dag_state(self) -> Dict[str, Optional[float]]:
        """Get the latest NOTEARS DAG discovery state.

        Replaces collect_causal_metrics() in cognitive_contracts.py.
        Returns dict with: dag_edge_count, causal_density.
        """
        conn = self._get_conn()
        row = conn.execute(
            "SELECT n_edges, topics FROM causal_discovery_state ORDER BY created_at DESC LIMIT 1"
        ).fetchone()

        metrics: Dict[str, Optional[float]] = {}
        if not row:
            return metrics

        n_edges = row[0] or 0
        try:
            topics = json.loads(row[1]) if row[1] else []
        except (json.JSONDecodeError, TypeError):
            topics = []

        metrics["dag_edge_count"] = n_edges
        total_possible = len(topics) ** 2
        if total_possible > 0:
            metrics["causal_density"] = n_edges / total_possible

        return metrics

    @store_traced
    def get_edge_count(self, edge_types: Optional[List[str]] = None) -> int:
        """Count edges in spreading_edges table."""
        conn = self._get_conn()
        if edge_types:
            placeholders = ",".join("?" for _ in edge_types)
            row = conn.execute(
                f"SELECT COUNT(*) FROM spreading_edges WHERE edge_type IN ({placeholders})",
                tuple(edge_types),
            ).fetchone()
        else:
            row = conn.execute("SELECT COUNT(*) FROM spreading_edges").fetchone()
        return row[0] if row else 0

    @store_traced
    def get_chain_count(self) -> int:
        """Count entries in causal_chains table."""
        conn = self._get_conn()
        # Table might not exist yet
        try:
            row = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='causal_chains'"
            ).fetchone()
            if not row:
                return 0
            count_row = conn.execute("SELECT COUNT(*) FROM causal_chains").fetchone()
            return count_row[0] if count_row else 0
        except Exception:
            return 0
