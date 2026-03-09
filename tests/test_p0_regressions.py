"""
P0 Regression Tests — Prerequisite bugs for Sebastian prototype.
================================================================
Tests for bugs #2, #4, #5, #40.
Bugs #3 and #6 were resolved by prior work (Memory Vault + threshold fix).

Exit criteria: all tests pass → `pytest tests/test_p0_regressions.py` exits 0.
"""

import json
import os
import sqlite3
import threading

import pytest


# ============================================================
# Bug #2: WM duplication in async write path
# remember() pushes to WM, then write_worker triggers MEMORY_STORED
# which pushes again. Fix: _remember_ctx.wm_pushed flag in write_worker.
# ============================================================

class TestBug2WMDuplication:
    """Verify _remember_ctx.wm_pushed prevents double WM push in write_worker."""

    def test_write_worker_sets_wm_pushed_flag(self):
        """_execute_remember must set _remember_ctx.wm_pushed = True
        before calling add_memory_smart, so _on_memory_stored skips."""
        from modules.interface import _remember_ctx
        from unittest.mock import patch, MagicMock

        flag_during_call = []

        def mock_add_memory_smart(**kwargs):
            # Capture flag state DURING the call
            flag_during_call.append(getattr(_remember_ctx, 'wm_pushed', False))
            return json.dumps({"action": "added", "id": "test-123"})

        # add_memory_smart is imported locally inside _execute_remember,
        # so patch at the source module level
        with patch("modules.memory_smart.add_memory_smart", mock_add_memory_smart):
            from modules.write_worker import _execute_remember
            _execute_remember({"content": "test memory", "category": "test"})

        assert flag_during_call == [True], \
            "wm_pushed must be True during add_memory_smart call"

    def test_wm_pushed_flag_reset_after_write(self):
        """Flag must be reset to False after execution (even on error)."""
        from modules.interface import _remember_ctx
        from unittest.mock import patch

        def mock_add_memory_smart(**kwargs):
            raise RuntimeError("simulated failure")

        with patch("modules.memory_smart.add_memory_smart", mock_add_memory_smart):
            from modules.write_worker import _execute_remember
            try:
                _execute_remember({"content": "test", "category": "test"})
            except Exception:
                pass

        assert getattr(_remember_ctx, 'wm_pushed', False) is False, \
            "wm_pushed must be reset to False even after error"

    def test_on_memory_stored_skips_when_flag_set(self):
        """_on_memory_stored handler must return early when wm_pushed is True."""
        from modules.interface import _remember_ctx
        from modules.wiring import _on_memory_stored
        from unittest.mock import patch, call

        _remember_ctx.wm_pushed = True
        try:
            # push_to_working_memory is imported locally inside _on_memory_stored
            with patch("modules.working_memory.push_to_working_memory") as mock_push:
                _on_memory_stored("memory_stored", {
                    "content": "important thing",
                    "importance": "high",
                    "category": "test",
                })
                mock_push.assert_not_called()
        finally:
            _remember_ctx.wm_pushed = False


# ============================================================
# Bug #4: apply_salience_decay pagination cap
# MAX_SCROLL was 500, only covering ~14% of memories.
# Fix: MAX_SCROLL = 10000.
# ============================================================

class TestBug4PaginationCap:
    """Verify MAX_SCROLL is sufficient for full memory coverage."""

    def test_max_scroll_at_least_10000(self):
        """MAX_SCROLL must be >= 10000 to cover realistic memory stores."""
        import modules.workspace as ws
        # Read from source — the constant is local to apply_salience_decay,
        # so we verify by reading the source file.
        import inspect
        source = inspect.getsource(ws.apply_salience_decay)
        assert "MAX_SCROLL = 10000" in source or "MAX_SCROLL = 1" in source, \
            "MAX_SCROLL must be at least 10000"
        # Strict check
        assert "MAX_SCROLL = 10000" in source, \
            f"MAX_SCROLL should be exactly 10000, got something else"


# ============================================================
# Bug #5: PCI (Prediction-Consolidation Interference)
# Sleep loop predictions contaminate transition_stats (Markov model).
# Fix: only record transitions when source == 'interactive'.
# ============================================================

class TestBug5PCI:
    """Verify sleep_loop predictions don't contaminate transition_stats."""

    def _setup_prediction_tables(self, db_path):
        """Create prediction tables for testing."""
        conn = sqlite3.connect(db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS prediction_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                predicted_topic TEXT,
                actual_topic TEXT,
                predicted_keywords TEXT DEFAULT '',
                actual_keywords TEXT DEFAULT '[]',
                surprise_score REAL DEFAULT 0,
                precision_weight REAL DEFAULT 1.0,
                weighted_surprise REAL DEFAULT 0,
                hit INTEGER DEFAULT 0,
                source TEXT DEFAULT 'interactive',
                created_at TEXT
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS transition_stats (
                from_topic TEXT,
                to_topic TEXT,
                count INTEGER DEFAULT 0,
                last_seen TEXT,
                PRIMARY KEY (from_topic, to_topic)
            )
        """)
        conn.commit()
        return conn

    def test_sleep_loop_source_excluded_from_transitions(self):
        """When source='sleep_loop', no INSERT into transition_stats."""
        import inspect
        import hooks.preturn_inject as pi
        source = inspect.getsource(pi)

        # The transition_stats block must be inside `if source == 'interactive':`
        # Find the transition_stats INSERT section
        idx_transition = source.find("INSERT INTO transition_stats")
        assert idx_transition > 0, "transition_stats INSERT must exist"

        # Find the nearest preceding `if source ==` guard
        preceding = source[:idx_transition]
        guard_idx = preceding.rfind("if source == 'interactive'")
        assert guard_idx > 0, \
            "transition_stats INSERT must be guarded by `if source == 'interactive'`"

    def test_prev_topic_query_filters_sleep_loop(self):
        """The prev_actual_topic query must exclude sleep_loop records."""
        import inspect
        import hooks.preturn_inject as pi
        source = inspect.getsource(pi)

        # Find the SELECT for prev_actual_topic near transition_stats
        idx_transition = source.find("INSERT INTO transition_stats")
        # Look backward for the SELECT query
        preceding = source[:idx_transition]
        select_idx = preceding.rfind("SELECT actual_topic FROM prediction_results")
        assert select_idx > 0, "prev_actual_topic SELECT must exist"

        select_block = preceding[select_idx:idx_transition]
        assert "sleep_loop" in select_block, \
            "prev_actual_topic query must filter out sleep_loop records"

    def test_curiosity_prediction_results_filters_sleep_loop(self):
        """curiosity.py must filter sleep_loop from prediction_results queries."""
        import inspect
        import modules.curiosity as cur
        source = inspect.getsource(cur)

        assert "sleep_loop" in source, \
            "curiosity.py must filter sleep_loop from prediction queries"

    def test_precision_computation_filters_sleep_loop(self):
        """_compute_precision must use WHERE source = 'interactive'."""
        import inspect
        import hooks.preturn_inject as pi
        source = inspect.getsource(pi._compute_precision)

        assert "interactive" in source, \
            "_compute_precision must filter for source='interactive'"


# ============================================================
# Bug #40: EventBus DB path resolution
# EventBus._get_db_path() must resolve to absolute path.
# Fix: uses FTS_DB_PATH from config (already absolute).
# ============================================================

class TestBug40EventBusPath:
    """Verify EventBus uses absolute DB path."""

    def test_db_path_is_absolute(self):
        """EventBus._get_db_path() must return an absolute path."""
        from modules.events import EventBus
        path = EventBus._get_db_path()
        assert os.path.isabs(path), \
            f"EventBus DB path must be absolute, got: {path}"

    def test_db_path_matches_fts_config(self):
        """EventBus must use the same DB as FTS index."""
        from modules.events import EventBus
        from modules.config import FTS_DB_PATH
        # They should resolve to the same path (env override or config)
        eb_path = EventBus._get_db_path()
        assert eb_path == FTS_DB_PATH or os.environ.get("FTS_DB_PATH") == eb_path, \
            f"EventBus path ({eb_path}) must match FTS_DB_PATH ({FTS_DB_PATH})"
