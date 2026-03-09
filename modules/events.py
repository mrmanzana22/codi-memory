"""
Codi Memory - Event Bus Architecture (Phase 0)
===============================================
Simple publish-subscribe event system for cross-module communication.

Enables subsystems to react to events without tight coupling.
Based on architectural recommendation #5 from NEURO_ANALYSIS_REPORT.

Events:
  MEMORY_STORED     - fired after a memory is added to any store
  MEMORY_RETRIEVED  - fired after a memory is retrieved/searched
  EMOTION_CHANGED   - fired when PAD emotional state changes
  WORKSPACE_BROADCAST - fired when global workspace broadcasts
  CONSOLIDATION_COMPLETE - fired after consolidation cycle (future)
  PREDICTION_ERROR  - fired when prediction doesn't match reality (future)

Usage:
  from modules.events import event_bus, Events

  # Subscribe
  event_bus.on(Events.MEMORY_STORED, my_handler)

  # Publish
  event_bus.emit(Events.MEMORY_STORED, {'memory_id': '...', 'content': '...'})
"""

import atexit
import logging
import os
import sqlite3
import threading
import time
from collections import defaultdict
from datetime import datetime

_logger = logging.getLogger(__name__)

from modules.tracing import get_trace_id


class Events:
    """Event name constants."""
    MEMORY_STORED = 'memory_stored'
    MEMORY_RETRIEVED = 'memory_retrieved'       # Payload: query, result_count, episodic_count, semantic_count, top_activation, retrieved_ids, source (search|timeline|ownership|experiences|critical|theme|emotion|focus_attention)
    EMOTION_CHANGED = 'emotion_changed'
    WORKSPACE_BROADCAST = 'workspace_broadcast'
    CONSOLIDATION_COMPLETE = 'consolidation_complete'
    PREDICTION_ERROR = 'prediction_error'
    WORKSPACE_COMPETITION_COMPLETE = 'workspace_competition_complete'  # WIRING-6
    WORKSPACE_RECRUITMENT = 'workspace_recruitment'                    # GWT-5: modules react to broadcast
    RETRIEVAL_QUALITY = 'retrieval_quality'                            # WIRING-7
    RECONSOLIDATION_TRIGGERED = 'reconsolidation_triggered'            # Phase 4
    CONTRADICTION_DETECTED = 'contradiction_detected'                  # Phase 5
    METACOGNITIVE_CONTROL_APPLIED = 'metacognitive_control_applied'    # Phase 4.5 HOT-3 runtime evidence
    ATTENTION_PREDICTION_ERROR = 'attention_prediction_error'          # AST-1 closed loop (Graziano 2013)
    SELF_MODEL_REFRESHED = 'self_model_refreshed'                      # HOT-1 auto-trigger (Rosenthal 2005)
    PERF_BUDGET_VIOLATION = 'perf_budget_violation'                     # Performance contract violation
    SESSION_CLOSE = 'session_close'                                       # Session bridge: checkpoint saved
    SESSION_OPEN = 'session_open'                                         # Session bridge: bridge loaded at wake
    SLEEP_LOOP_COMPLETE = 'sleep_loop_complete'                             # Sleep loop: background maintenance done
    EMOTION_GATING_APPLIED = 'emotion_gating_applied'                        # HOT-4 runtime evidence (Bower 1981)
    AFFECTIVE_CHARGE_COMPUTED = 'affective_charge_computed'                     # Sprint 4: AC from policy shift (Joffily & Coricelli 2013)
    CURIOSITY_RESOLVED = 'curiosity_resolved'                                    # Curiosity discovery saved to long-term memory
    MEMORY_VAULTED = 'memory_vaulted'
    MEMORY_REACTIVATED = 'memory_reactivated'


class EventBus:
    """
    Simple thread-safe publish-subscribe event bus.

    Handlers are called synchronously in the order they were registered.
    Exceptions in handlers are caught and logged (never block the emitter).
    """

    _FLUSH_THRESHOLD = 20  # Flush to SQLite every N dirty events

    def __init__(self):
        self._handlers = defaultdict(list)
        self._lock = threading.Lock()
        self._history = []  # Last N events for debugging
        self._history_max = 50
        # Persistent counters (batched to SQLite)
        self._dirty_counts = defaultdict(int)  # event -> pending increment
        self._dirty_total = 0
        self._last_flush = time.monotonic()
        atexit.register(self._flush_counts)

    def on(self, event_name: str, handler: callable):
        """Register a handler for an event.

        Args:
            event_name: Event to listen for (use Events constants)
            handler: Callable that receives (event_name, data_dict)
        """
        with self._lock:
            if handler not in self._handlers[event_name]:
                self._handlers[event_name].append(handler)

    def off(self, event_name: str, handler: callable):
        """Unregister a handler."""
        with self._lock:
            try:
                self._handlers[event_name].remove(handler)
            except ValueError:
                pass

    def emit(self, event_name: str, data: dict = None):
        """Fire an event, calling all registered handlers.

        Args:
            event_name: Event being fired
            data: Dict of event-specific data
        """
        if data is None:
            data = {}

        # Auto-inject trace_id from contextvars (transparent to callers)
        tid = get_trace_id()
        if tid and "trace_id" not in data:
            data["trace_id"] = tid

        # Record in history
        entry = {
            'event': event_name,
            'timestamp': datetime.now().isoformat(),
            'data_keys': list(data.keys())
        }

        with self._lock:
            self._history.append(entry)
            if len(self._history) > self._history_max:
                self._history = self._history[-self._history_max:]
            handlers = list(self._handlers.get(event_name, []))

        # Accumulate persistent counter (batched)
        self._dirty_counts[event_name] += 1
        self._dirty_total += 1
        if self._dirty_total >= self._FLUSH_THRESHOLD or (time.monotonic() - self._last_flush) > 1.0:
            self._flush_counts()

        # Call handlers outside the lock
        for handler in handlers:
            try:
                handler(event_name, data)
            except Exception as e:
                _logger.error("Error in handler for %s: %s", event_name, e)

    def get_history(self, limit: int = 20) -> list:
        """Get recent event history for debugging."""
        with self._lock:
            return list(self._history[-limit:])

    def get_stats(self) -> dict:
        """Get subscriber counts per event."""
        with self._lock:
            return {
                event: len(handlers)
                for event, handlers in self._handlers.items()
                if handlers
            }

    # ============================================================
    # PERSISTENT EVIDENCE (Phase 5.5 - Block 1995)
    # ============================================================

    @staticmethod
    def _get_db_path() -> str:
        """Get SQLite path for event_counts (reuses FTS DB)."""
        from modules.config import FTS_DB_PATH
        return os.environ.get("FTS_DB_PATH", FTS_DB_PATH)

    _event_tables_validated = False

    def _flush_counts(self):
        """Batch-write dirty counters to SQLite. Safe: never blocks emit on failure."""
        if not self._dirty_counts:
            return
        try:
            from modules.db_pool import get_conn
            conn = get_conn(self._get_db_path())
            if not EventBus._event_tables_validated:
                conn.execute("SELECT 1 FROM event_counts LIMIT 0")
                EventBus._event_tables_validated = True
            now = datetime.now().isoformat()
            for event_name, increment in self._dirty_counts.items():
                conn.execute("""
                    INSERT INTO event_counts (event, count, last_seen)
                    VALUES (?, ?, ?)
                    ON CONFLICT(event) DO UPDATE SET
                        count = count + excluded.count,
                        last_seen = excluded.last_seen
                """, (event_name, increment, now))
            conn.commit()
            self._dirty_counts.clear()
            self._dirty_total = 0
            self._last_flush = time.monotonic()
        except Exception:
            pass  # Never block emit

    def get_persistent_count(self, event_name: str) -> int:
        """Read total historical count for an event type from SQLite."""
        # Flush pending first
        self._flush_counts()
        try:
            db_path = self._get_db_path()
            if not os.path.exists(db_path):
                return 0
            from modules.db_pool import get_conn
            conn = get_conn(db_path)
            cursor = conn.execute(
                "SELECT count FROM event_counts WHERE event = ?", (event_name,)
            )
            row = cursor.fetchone()
            return row[0] if row else 0
        except Exception:
            return 0


# Singleton instance - all modules share this
event_bus = EventBus()
