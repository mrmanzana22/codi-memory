"""CX (Cross-Loop Integration) store — cx_snapshots table."""

from __future__ import annotations

import json
import math
from typing import Dict, List, Optional

from modules.store._base import SQLiteConnProvider, default_sqlite_provider
from modules.store import store_traced


class CXStore:
    """Repository for CX observability data.

    Owns: cx_snapshots table (SQLite).
    Invariant: methods return [] or {} on empty data, never None.
    """

    def __init__(self, conn_provider: Optional[SQLiteConnProvider] = None):
        self._get_conn = conn_provider or default_sqlite_provider

    def _fetch_snapshots(self, hours: int = 24, limit: int = 0) -> List[dict]:
        """Internal: fetch and parse snapshots (no tracing — avoids double-trace)."""
        from datetime import timedelta
        from modules.config import now_col
        _cutoff = (now_col() - timedelta(hours=hours)).isoformat()
        conn = self._get_conn()
        if limit > 0:
            rows = conn.execute(
                """SELECT ts, payload, anomalies FROM cx_snapshots
                   WHERE ts > ?
                   ORDER BY ts DESC LIMIT ?""",
                (_cutoff, limit),
            ).fetchall()
        else:
            rows = conn.execute(
                """SELECT ts, payload, anomalies FROM cx_snapshots
                   WHERE ts > ?
                   ORDER BY ts DESC""",
                (_cutoff,),
            ).fetchall()

        if not rows:
            rows = conn.execute(
                "SELECT ts, payload, anomalies FROM cx_snapshots ORDER BY ts DESC LIMIT 10"
            ).fetchall()

        results = []
        for r in rows:
            try:
                payload = json.loads(r[1]) if r[1] else {}
                anomalies = json.loads(r[2]) if r[2] else []
                results.append({
                    "ts": r[0],
                    "fire_counts": payload.get("fire_counts", {}),
                    "cascade": payload.get("cascade", {}),
                    "anomalies": anomalies,
                    "raw_payload": payload,
                })
            except (json.JSONDecodeError, TypeError):
                continue
        return results

    @store_traced
    def get_recent_snapshots(self, hours: int = 24, limit: int = 0) -> List[dict]:
        """Get CX snapshots from the last N hours.

        Args:
            hours: Window to look back.
            limit: Max rows. 0 = no limit (fetch all in window, matching legacy behavior).

        Returns list of dicts with keys: ts, fire_counts, cascade, anomalies, raw_payload.
        """
        return self._fetch_snapshots(hours=hours, limit=limit)

    @store_traced
    def get_aggregated_cx_metrics(self, hours: int = 24) -> Dict[str, Optional[float]]:
        """Compute aggregated CX metrics for contract evaluation.

        Returns dict with: active_cx_ratio, cascade_depth, cx_diversity_index.
        Replaces collect_cx_metrics() in cognitive_contracts.py.
        """
        snapshots = self._fetch_snapshots(hours=hours)
        if not snapshots:
            return {}

        aggregated_fires: Dict[str, int] = {}
        total_chain = 0
        snap_count = len(snapshots)

        for snap in snapshots:
            for cx_id, count in snap["fire_counts"].items():
                aggregated_fires[cx_id] = aggregated_fires.get(cx_id, 0) + (count or 0)
            total_chain += snap["cascade"].get("chain_count", 0)

        total_cx = 34
        active = sum(1 for v in aggregated_fires.values() if v > 0)

        metrics: Dict[str, Optional[float]] = {
            "active_cx_ratio": active / total_cx,
            "cascade_depth": total_chain / snap_count if snap_count > 0 else 0,
        }

        # Shannon entropy of fire distribution
        total_all = sum(v for v in aggregated_fires.values() if v > 0)
        if total_all > 0:
            entropy = 0.0
            for count in aggregated_fires.values():
                if count > 0:
                    p = count / total_all
                    entropy -= p * math.log2(p)
            metrics["cx_diversity_index"] = entropy
        else:
            metrics["cx_diversity_index"] = 0.0

        return metrics

    @store_traced
    def get_gnw_metrics(self, limit: int = 50) -> Dict[str, Optional[float]]:
        """Compute GNW ignition metrics from CX snapshots.

        Replaces collect_gnw_metrics() in cognitive_contracts.py.
        Returns dict with: ignition_ratio, recurrent_pass_count, coalition_size.
        """
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT payload FROM cx_snapshots ORDER BY ts DESC LIMIT ?",
            (limit,),
        ).fetchall()

        metrics: Dict[str, Optional[float]] = {
            "recurrent_pass_count": 3,  # Architectural constant (competition.py)
            "coalition_size": None,     # Needs runtime tracking — TODO
        }

        if not rows:
            return metrics

        gnw_cx = {"CX-5", "CX-9", "CX-16", "CX-25"}
        total_fires = 0
        total_gnw_fires = 0

        for r in rows:
            try:
                snap = json.loads(r[0])
                fires = snap.get("fire_counts", {})
                for cx_id, count in fires.items():
                    c = count or 0
                    total_fires += c
                    if cx_id in gnw_cx:
                        total_gnw_fires += c
            except (json.JSONDecodeError, TypeError):
                continue

        if total_fires > 0:
            metrics["ignition_ratio"] = total_gnw_fires / total_fires

        return metrics

    @store_traced
    def get_pe_cx_diversity(self, hours: int = 24) -> int:
        """Count distinct CX that fired in recent snapshots.

        Replaces the CX part of collect_pe_metrics() in cognitive_contracts.py.
        """
        from datetime import timedelta
        from modules.config import now_col
        _cutoff = (now_col() - timedelta(hours=hours)).isoformat()
        conn = self._get_conn()
        rows = conn.execute(
            """SELECT payload FROM cx_snapshots
               WHERE ts > ?
               ORDER BY ts DESC LIMIT 5""",
            (_cutoff,),
        ).fetchall()

        all_cx_ids: set = set()
        for r in rows:
            try:
                snap = json.loads(r[0])
                fires = snap.get("fire_counts", {})
                for cx_id, count in fires.items():
                    if count and count > 0:
                        all_cx_ids.add(cx_id)
            except (json.JSONDecodeError, TypeError):
                continue
        return len(all_cx_ids)
