"""Prediction store — prediction_results, prediction_state tables."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from modules.store._base import SQLiteConnProvider, default_sqlite_provider
from modules.store import store_traced


class PredictionStore:
    """Repository for prediction loop data.

    Owns: prediction_results, prediction_state tables (SQLite).
    Invariant: methods return [] or {} on empty data, never None.
    """

    def __init__(self, conn_provider: Optional[SQLiteConnProvider] = None):
        self._get_conn = conn_provider or default_sqlite_provider

    @store_traced
    def get_accuracy(
        self, sources: Optional[List[str]] = None, limit: int = 100
    ) -> Dict[str, Optional[float]]:
        """Compute prediction accuracy from prediction_results.

        Replaces collect_prediction_metrics() in cognitive_contracts.py.
        Returns dict with: prediction_accuracy, precision_adaptation_speed.
        """
        if sources is None:
            sources = ["interactive", "sleep_loop", "preturn"]

        conn = self._get_conn()
        placeholders = ",".join("?" for _ in sources)
        rows = conn.execute(
            f"""SELECT hit, surprise_score FROM prediction_results
                WHERE source IN ({placeholders})
                ORDER BY created_at DESC LIMIT ?""",
            (*sources, limit),
        ).fetchall()

        metrics: Dict[str, Optional[float]] = {}
        if not rows:
            return metrics

        hits = [r[0] for r in rows if r[0] is not None]
        if hits:
            metrics["prediction_accuracy"] = sum(hits) / len(hits)

        surprises = [r[1] for r in rows if r[1] is not None]
        if len(surprises) > 10:
            mean_s = sum(surprises) / len(surprises)
            variance = sum((s - mean_s) ** 2 for s in surprises) / len(surprises)
            metrics["precision_adaptation_speed"] = max(1, min(20, 10 * (1 - variance)))

        return metrics

    @store_traced
    def get_surprise_scores(self, days: int = 7, limit: int = 100) -> List[float]:
        """Get recent surprise scores for PE analysis.

        Replaces the prediction_results part of collect_pe_metrics().
        Returns list of floats (surprise_score values).
        """
        from datetime import timedelta
        from modules.config import now_col
        _cutoff = (now_col() - timedelta(days=days)).isoformat()
        conn = self._get_conn()
        rows = conn.execute(
            """SELECT surprise_score FROM prediction_results
               WHERE created_at > ?
               ORDER BY created_at DESC LIMIT ?""",
            (_cutoff, limit),
        ).fetchall()
        return [r[0] for r in rows if r[0] is not None]

    @store_traced
    def get_high_pe_count(self, threshold: float = 0.6) -> int:
        """Count prediction results with surprise_score > threshold.

        Used by reconsolidation metrics (trigger_rate = recon_count / high_pe_count).
        """
        conn = self._get_conn()
        row = conn.execute(
            "SELECT COUNT(*) FROM prediction_results WHERE surprise_score > ?",
            (threshold,),
        ).fetchone()
        return row[0] if row else 0

    @store_traced
    def get_recent_results(
        self, limit: int = 100, sources: Optional[List[str]] = None
    ) -> List[dict]:
        """Get recent prediction results as list of dicts.

        Returns list of dicts with keys: hit, surprise_score, source, topic, created_at.
        """
        conn = self._get_conn()
        if sources:
            placeholders = ",".join("?" for _ in sources)
            rows = conn.execute(
                f"""SELECT hit, surprise_score, source, topic, created_at
                    FROM prediction_results
                    WHERE source IN ({placeholders})
                    ORDER BY created_at DESC LIMIT ?""",
                (*sources, limit),
            ).fetchall()
        else:
            rows = conn.execute(
                """SELECT hit, surprise_score, source, topic, created_at
                   FROM prediction_results
                   ORDER BY created_at DESC LIMIT ?""",
                (limit,),
            ).fetchall()

        return [
            {
                "hit": r[0],
                "surprise_score": r[1],
                "source": r[2],
                "topic": r[3],
                "created_at": r[4],
            }
            for r in rows
        ]
