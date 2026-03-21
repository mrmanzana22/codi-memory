"""Metacognition store — fok_calibration_log, metacognition_traces."""

from __future__ import annotations

from typing import Dict, List, Optional

from modules.store._base import SQLiteConnProvider, default_sqlite_provider
from modules.store import store_traced

# FOK coverage mapping: categorical → numeric
_COVERAGE_MAP = {"sparse": 0.2, "partial": 0.5, "comprehensive": 0.8}


class MetacognitionStore:
    """Repository for metacognitive monitoring data.

    Owns: fok_calibration_log, metacognition_traces tables (SQLite).
    Invariant: methods return [] or {} on empty data, never None.
    """

    def __init__(self, conn_provider: Optional[SQLiteConnProvider] = None):
        self._get_conn = conn_provider or default_sqlite_provider

    @store_traced
    def get_calibration_metrics(self, limit: int = 200) -> Dict[str, Optional[float]]:
        """Compute metacognitive calibration metrics.

        Replaces collect_metacognition_metrics() in cognitive_contracts.py.
        Returns dict with: calibration_error, confidence_range, monitoring_control_loop.
        """
        conn = self._get_conn()
        metrics: Dict[str, Optional[float]] = {}

        # FOK calibration
        rows = conn.execute(
            """SELECT fok_predicted, actual_coverage FROM fok_calibration_log
               ORDER BY created_at DESC LIMIT ?""",
            (limit,),
        ).fetchall()

        if rows:
            errors = []
            confidences = []
            for r in rows:
                if r[0] is not None and r[1] is not None:
                    actual = _COVERAGE_MAP.get(r[1], 0.5)
                    errors.append(abs(r[0] - actual))
                    confidences.append(r[0])
            if errors:
                metrics["calibration_error"] = sum(errors) / len(errors)
            if confidences:
                metrics["confidence_range"] = max(confidences) - min(confidences)

        # Monitoring → control loop check
        from datetime import timedelta
        from modules.config import now_col
        _mc_cutoff = (now_col() - timedelta(days=30)).isoformat()
        mc_row = conn.execute(
            """SELECT COUNT(DISTINCT strategy_chosen) FROM metacognition_traces
               WHERE created_at > ?""",
            (_mc_cutoff,)
        ).fetchone()
        if mc_row and mc_row[0] > 1:
            metrics["monitoring_control_loop"] = 1.0
        elif mc_row:
            metrics["monitoring_control_loop"] = 0.0

        return metrics

    @store_traced
    def get_recent_traces(self, limit: int = 50) -> List[dict]:
        """Get recent metacognition traces.

        Returns list of dicts with: query, strategy_chosen, fok_score, created_at.
        """
        conn = self._get_conn()
        rows = conn.execute(
            """SELECT query, strategy_chosen, strategy_reason, fok_score, created_at
               FROM metacognition_traces
               ORDER BY id DESC LIMIT ?""",
            (limit,),
        ).fetchall()
        return [
            {
                "query": r[0],
                "strategy_chosen": r[1],
                "strategy_reason": r[2],
                "fok_score": r[3],
                "created_at": r[4],
            }
            for r in rows
        ]

    @store_traced
    def get_fok_calibration_data(self, limit: int = 200) -> List[dict]:
        """Get raw FOK calibration data for external analysis.

        Returns list of dicts with: fok_predicted, actual_coverage, actual_numeric.
        """
        conn = self._get_conn()
        rows = conn.execute(
            """SELECT fok_predicted, actual_coverage FROM fok_calibration_log
               ORDER BY created_at DESC LIMIT ?""",
            (limit,),
        ).fetchall()
        return [
            {
                "fok_predicted": r[0],
                "actual_coverage": r[1],
                "actual_numeric": _COVERAGE_MAP.get(r[1], 0.5) if r[1] else None,
            }
            for r in rows
        ]
