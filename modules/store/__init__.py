"""
CognitiveStore — Unified data access layer for codi-memory.
=============================================================
Phase 2 of architectural restructuring.

Provides domain-specific store classes that encapsulate raw SQL queries.
Each store method is traced via @store_traced (reuses metrics.log_tool_call).

Usage:
    from modules.store import get_store

    store = get_store()
    cx_metrics = store.cx.get_aggregated_cx_metrics(hours=24)
    accuracy = store.prediction.get_accuracy(sources=['interactive'])
"""

from __future__ import annotations

import functools
import time
from typing import Optional

from modules.store._base import SQLiteConnProvider, default_sqlite_provider


# ---------------------------------------------------------------------------
# @store_traced — observability decorator
# ---------------------------------------------------------------------------

def store_traced(func):
    """Trace store method calls to tool_calls table.

    Reuses metrics.log_tool_call for consistency with existing MCP
    instrumentation. Tag prefix 'store:' distinguishes store calls.
    Fail-open: tracing errors never break store operations.
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        cls_name = ""
        if args and hasattr(args[0], "__class__"):
            cls_name = args[0].__class__.__name__

        tool_name = f"store:{cls_name}.{func.__name__}" if cls_name else f"store:{func.__name__}"

        # Capture started_at BEFORE execution (Fix: auditor issue 5.2)
        try:
            from modules.config import now_iso
            started_at = now_iso()
        except Exception:
            started_at = None

        t0 = time.perf_counter()

        try:
            result = func(*args, **kwargs)
            dur_ms = int((time.perf_counter() - t0) * 1000)
            _log_safe(
                tool_name=tool_name,
                started_at=started_at,
                duration_ms=dur_ms,
                success=True,
                error_type=None,
                tag=f"store:{cls_name}" if cls_name else "store",
            )
            return result
        except Exception as e:
            dur_ms = int((time.perf_counter() - t0) * 1000)
            _log_safe(
                tool_name=tool_name,
                started_at=started_at,
                duration_ms=dur_ms,
                success=False,
                error_type=e.__class__.__name__,
                tag=f"store:{cls_name}" if cls_name else "store",
            )
            raise

    return wrapper


def _log_safe(
    tool_name: str,
    started_at: Optional[str],
    duration_ms: int,
    success: bool,
    error_type: Optional[str],
    tag: str,
) -> None:
    """Fail-open: metrics must never break store operations."""
    try:
        from modules.metrics import log_tool_call

        if started_at is None:
            from modules.config import now_iso
            started_at = now_iso()

        log_tool_call(
            tool_name=tool_name,
            started_at=started_at,
            duration_ms=duration_ms,
            success=success,
            error_type=error_type,
            args_size=0,
            result_size=0,
            session_id=None,
            tag=tag,
        )
    except Exception:
        pass


# ---------------------------------------------------------------------------
# CognitiveStore facade
# ---------------------------------------------------------------------------

class CognitiveStore:
    """Facade providing access to all domain-specific stores.

    Not a singleton — instantiated with connection providers.
    Default providers use the existing db_pool thread-local pool.

    Usage:
        store = CognitiveStore()              # production
        store = CognitiveStore(               # testing
            sqlite_provider=my_test_conn_factory
        )

        metrics = store.cx.get_recent_snapshots(hours=24)
        accuracy = store.prediction.get_accuracy(sources=['interactive'])
    """

    def __init__(self, sqlite_provider: Optional[SQLiteConnProvider] = None):
        prov = sqlite_provider or default_sqlite_provider

        # Lazy imports to avoid loading all stores at import time
        from modules.store.cx_store import CXStore
        from modules.store.prediction_store import PredictionStore
        from modules.store.consolidation_store import ConsolidationStore
        from modules.store.metacognition_store import MetacognitionStore
        from modules.store.causal_store import CausalStore
        from modules.store.attention_store import AttentionStore
        from modules.store.forgetting_store import ForgettingStore
        from modules.store.sleep_store import SleepStore

        self.cx = CXStore(prov)
        self.prediction = PredictionStore(prov)
        self.consolidation = ConsolidationStore(prov)
        self.metacognition = MetacognitionStore(prov)
        self.causal = CausalStore(prov)
        self.attention = AttentionStore(prov)
        self.forgetting = ForgettingStore(prov)
        self.sleep = SleepStore(prov)


def get_store(sqlite_provider: Optional[SQLiteConnProvider] = None) -> CognitiveStore:
    """Create a CognitiveStore with default or custom providers."""
    return CognitiveStore(sqlite_provider=sqlite_provider)
