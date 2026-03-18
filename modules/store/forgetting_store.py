"""Forgetting store — strength_log table."""

from __future__ import annotations

from typing import Dict, List, Optional

from modules.store._base import SQLiteConnProvider, default_sqlite_provider
from modules.store import store_traced


class ForgettingStore:
    """Repository for memory strength and decay data.

    Owns: strength_log table (SQLite).
    Invariant: methods return [] or {} on empty data, never None.
    """

    def __init__(self, conn_provider: Optional[SQLiteConnProvider] = None):
        self._get_conn = conn_provider or default_sqlite_provider

    @store_traced
    def get_forgetting_metrics(self) -> Dict[str, Optional[float]]:
        """Compute forgetting metrics for contract evaluation.

        Replaces collect_forgetting_metrics() in cognitive_contracts.py.
        Returns dict with: decay_curve_exponent, rif_suppression_rate.
        """
        metrics: Dict[str, Optional[float]] = {}

        # Decay exponent from code constant
        try:
            from modules.activation import ACTR_DECAY_DEFAULT
            metrics["decay_curve_exponent"] = ACTR_DECAY_DEFAULT
        except ImportError:
            pass

        # RIF suppression rate from strength_log
        conn = self._get_conn()
        try:
            rif_row = conn.execute(
                "SELECT COUNT(*) FROM strength_log WHERE event_type = 'rif_suppression'"
            ).fetchone()
            total_row = conn.execute("SELECT COUNT(*) FROM strength_log").fetchone()

            rif_count = rif_row[0] if rif_row else 0
            total = total_row[0] if total_row else 0

            if total > 0:
                metrics["rif_suppression_rate"] = rif_count / total
        except Exception:
            pass

        return metrics
