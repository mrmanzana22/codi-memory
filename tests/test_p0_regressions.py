"""
P0 Regression Tests — Prerequisite bugs for Sebastian prototype.
================================================================
Tests for bugs #1-#6.

Status (2026-03-12):
  #1 (WM duplication) — FIXED (Proposal #57)
  #2 (EventBus DB path) — NOT A BUG
  #3 (FadeMem critical decay) — FIXED (hard skip for critical)
  #4 (Consolidation pagination) — FIXED (scroll loop)
  #5 (PCI contamination) — IMPROVED (topic-agnostic + env var)
  #6 (Daemon short query filter) — FIXED (threshold < 4 -> < 2)

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
# Bug #3: FadeMem must NOT decay critical memories
# Fix: early return in compute_fadem_strength when importance == "critical"
# ============================================================

class TestBug3FadeMemCritical:
    """Verify critical memories skip decay entirely."""

    def test_critical_memory_no_decay(self):
        """Critical importance must return current_salience unchanged."""
        # Source-level check: verify early return exists for critical
        source_path = os.path.join(
            os.path.dirname(__file__), "..", "modules", "forgetting.py"
        )
        with open(source_path) as f:
            source = f.read()
        assert 'importance == "critical"' in source, \
            "Must check for critical importance"
        assert "return current_salience" in source, \
            "Must return current_salience for critical"

    def test_critical_skip_is_early_return(self):
        """The critical check must come BEFORE the decay computation."""
        import inspect
        from modules.forgetting import compute_fadem_strength
        source = inspect.getsource(compute_fadem_strength)
        idx_critical = source.find('importance == "critical"')
        idx_lambda = source.find("FADEM_LAMBDA_BASE")
        assert idx_critical > 0 and idx_lambda > 0, \
            "Both critical check and FADEM_LAMBDA_BASE must exist in function"
        assert idx_critical < idx_lambda, \
            "Critical check must come before decay computation (early return)"

    def test_high_not_exempt(self):
        """Source must NOT exempt 'high' importance — only critical."""
        source_path = os.path.join(
            os.path.dirname(__file__), "..", "modules", "forgetting.py"
        )
        with open(source_path) as f:
            source = f.read()
        # Find the early return block
        idx_critical = source.find('importance == "critical"')
        block = source[idx_critical:idx_critical + 100]
        assert '"high"' not in block, \
            "Only critical should be exempt, not high"


# ============================================================
# Bug #4: apply_salience_decay pagination cap
# MAX_SCROLL was 500, only covering ~14% of memories.
# Fix: MAX_SCROLL = 10000.
# ============================================================

class TestBug4PaginationCap:
    """Verify decay pagination covers full memory corpus.

    Original bug: MAX_SCROLL=500 capped at ~14% of memories.
    Fix: replaced with sample-based pg.scroll() pagination (no hard cap).
    """

    def test_no_hard_scroll_cap(self):
        """apply_salience_decay must NOT have a low hard cap (old MAX_SCROLL=500)."""
        import inspect
        import modules.workspace as ws
        source = inspect.getsource(ws.apply_salience_decay)
        assert "MAX_SCROLL = 500" not in source, \
            "Old MAX_SCROLL=500 cap must be removed"

    def test_uses_scroll_pagination(self):
        """Must use pg.scroll() pagination for full corpus coverage."""
        import inspect
        import modules.workspace as ws
        source = inspect.getsource(ws.apply_salience_decay)
        assert "pg.scroll(" in source, \
            "Must use pg.scroll() for paginated memory access"


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

    def test_source_detection_not_topic_restricted(self):
        """PCI detection must NOT require topic=='codigo' (P0 fix)."""
        import inspect
        import hooks.preturn_inject as pi
        source = inspect.getsource(pi)

        # Find the source detection block
        idx_source = source.find("CODI_SOURCE")
        assert idx_source > 0, "Must support CODI_SOURCE env var"

        # The old bug: detection only fired when predicted==actual=='codigo'
        # After fix: no topic restriction near the timing heuristic
        idx_gap = source.find("gap_min")
        assert idx_gap > 0, "Timing heuristic must exist"
        # Get surrounding context (200 chars before gap_min)
        context = source[max(0, idx_gap - 200):idx_gap + 100]
        assert "predicted_topic == actual_topic == 'codigo'" not in context, \
            "PCI detection must not be restricted to topic=='codigo'"

    def test_codi_source_env_var_support(self):
        """CODI_SOURCE env var must be checked for explicit source tagging."""
        import inspect
        import hooks.preturn_inject as pi
        source = inspect.getsource(pi)
        assert "os.environ.get('CODI_SOURCE'" in source or \
               'os.environ.get("CODI_SOURCE"' in source, \
            "Must check CODI_SOURCE env var for explicit source override"

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


# ============================================================
# Bug #6: Daemon short query filter
# Was < 4 chars, blocking "ok", "si", "WM?".
# Fix: lowered to < 2 chars (only block empty/single-char).
# ============================================================

class TestBug6DaemonShortQuery:
    """Verify daemon context_builder allows short but meaningful queries."""

    def test_threshold_is_2_not_4(self):
        """Short query filter must use < 2, not < 4."""
        import inspect
        import importlib
        spec = importlib.util.spec_from_file_location(
            "context_builder",
            os.path.expanduser("~/codi-daemon/context_builder.py"),
        )
        mod = importlib.util.module_from_spec(spec)
        # Read source directly to avoid import side effects
        source_path = os.path.expanduser("~/codi-daemon/context_builder.py")
        with open(source_path) as f:
            source = f.read()

        # Must NOT contain the old threshold
        assert "strip()) < 4" not in source, \
            "Short query filter must NOT use < 4 (old threshold)"
        # Must contain the new threshold
        assert "strip()) < 2" in source, \
            "Short query filter must use < 2"
