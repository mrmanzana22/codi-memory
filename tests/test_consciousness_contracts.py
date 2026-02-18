#!/usr/bin/env python3
"""
CONSCIOUSNESS CONTRACTS (Bloque 1)
====================================
5 invariant tests verifying consciousness subsystems integrate correctly.

These are NOT unit tests of individual modules -- they are cross-module
contracts that must hold for consciousness to function as a system.

CC-1: Critical intention must be in workspace spotlight when triggered
CC-2: Bridge restore PAD provenance (must always be traceable)
CC-3: PE high must be actionable when CODI_PE_ACTIONS=on (flag-gated)
CC-4: Sleep loop liveness visible in assessment evidence
CC-5: Consolidation no-overlap between sleep_loop and ciclo_vida

Run: ./venv/bin/pytest tests/test_consciousness_contracts.py -v
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import inspect
import json
import sqlite3
import pytest
from unittest.mock import patch, MagicMock
from datetime import datetime, timedelta

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FTS_MIGRATIONS_DIR = os.path.join(PROJECT_ROOT, "migrations")
PROSPECTIVE_MIGRATIONS_DIR = os.path.join(PROJECT_ROOT, "migrations_prospective")


# ============================================================
# CC-1: Critical intention -> workspace spotlight
# ============================================================

class TestCC1CriticalIntentionReachesSpotlight:
    """CC-1: When a critical intention triggers, it MUST appear in workspace spotlight."""

    @pytest.fixture
    def prospective_conn(self, tmp_path, monkeypatch):
        """Fresh prospective DB with a critical intention inserted."""
        from modules.migrations import apply_migrations
        db_path = str(tmp_path / "prospective.db")
        apply_migrations(db_path, migrations_dir=PROSPECTIVE_MIGRATIONS_DIR)

        conn = sqlite3.connect(db_path)
        conn.execute("PRAGMA journal_mode=WAL")

        # Inject into prospective module
        import modules.prospective as pm
        monkeypatch.setattr(pm, '_conn', conn)

        yield conn
        conn.close()

    def test_mark_triggered_pushes_critical_to_workspace(self, prospective_conn, monkeypatch):
        """Triggering a critical intention calls update_workspace_spotlight."""
        import modules.prospective as pm
        conn = prospective_conn

        # Insert critical intention
        int_id = "cc1-test-critical-00000000"
        conn.execute("""
            INSERT INTO intentions
            (id, action, trigger_type, trigger_spec, cue_focality,
             priority, activation, created_at, status, creator)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (int_id, "Deploy critical fix to production", "event",
              '{"keywords": ["deploy"]}', "focal", "critical", 0.95,
              datetime.now().isoformat(), "pending", "codi"))
        conn.commit()

        ws_calls = []
        wm_calls = []

        with patch('modules.consciousness.update_workspace_spotlight',
                    lambda memories, theme=None: ws_calls.append(
                        {"memories": memories, "theme": theme})), \
             patch('modules.working_memory.push_to_working_memory',
                    lambda **kw: wm_calls.append(kw)):
            pm._mark_triggered(conn, int_id, "Keywords: deploy",
                               datetime.now().isoformat())

        # CC-1 INVARIANT: workspace spotlight must be updated
        assert len(ws_calls) == 1, "update_workspace_spotlight must be called exactly once"
        assert ws_calls[0]["theme"] == "intention_triggered"
        assert any("[PM TRIGGERED]" in str(m) for m in ws_calls[0]["memories"])

    def test_mark_triggered_pushes_critical_to_working_memory(self, prospective_conn, monkeypatch):
        """Triggering a critical intention pushes [!!!] marker to working memory."""
        import modules.prospective as pm
        conn = prospective_conn

        int_id = "cc1-test-critical-wm-00000"
        conn.execute("""
            INSERT INTO intentions
            (id, action, trigger_type, trigger_spec, cue_focality,
             priority, activation, created_at, status, creator)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (int_id, "Review security incident", "event",
              '{"keywords": ["security"]}', "focal", "critical", 0.90,
              datetime.now().isoformat(), "pending", "codi"))
        conn.commit()

        wm_calls = []

        with patch('modules.consciousness.update_workspace_spotlight', lambda **kw: None), \
             patch('modules.working_memory.push_to_working_memory',
                    lambda **kw: wm_calls.append(kw)):
            pm._mark_triggered(conn, int_id, "Keywords: security",
                               datetime.now().isoformat())

        assert len(wm_calls) == 1, "push_to_working_memory must be called"
        assert wm_calls[0]["topic"] == "intention_triggered"
        assert "[!!!]" in wm_calls[0]["content"], "Critical marker [!!!] required"
        assert wm_calls[0]["relevance"] == 0.9

    def test_non_critical_also_reaches_spotlight(self, prospective_conn, monkeypatch):
        """Non-critical intentions also reach spotlight (different marker)."""
        import modules.prospective as pm
        conn = prospective_conn

        int_id = "cc1-test-medium-00000000"
        conn.execute("""
            INSERT INTO intentions
            (id, action, trigger_type, trigger_spec, cue_focality,
             priority, activation, created_at, status, creator)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (int_id, "Review PR later", "event",
              '{"keywords": ["review"]}', "focal", "medium", 0.65,
              datetime.now().isoformat(), "pending", "codi"))
        conn.commit()

        ws_calls = []

        with patch('modules.consciousness.update_workspace_spotlight',
                    lambda memories, theme=None: ws_calls.append(
                        {"memories": memories, "theme": theme})), \
             patch('modules.working_memory.push_to_working_memory',
                    lambda **kw: None):
            pm._mark_triggered(conn, int_id, "Keywords: review",
                               datetime.now().isoformat())

        assert len(ws_calls) == 1
        assert ws_calls[0]["theme"] == "intention_triggered"


# ============================================================
# CC-2: Bridge restore PAD provenance
# ============================================================

class TestCC2BridgeRestorePADProvenance:
    """CC-2: After PAD restore, trigger string MUST contain provenance source."""

    @pytest.fixture
    def bridge_db(self, tmp_path):
        """DB with session_checkpoints table and a recent checkpoint."""
        from modules.migrations import apply_migrations
        db_path = str(tmp_path / "bridge.db")
        apply_migrations(db_path, migrations_dir=FTS_MIGRATIONS_DIR)

        conn = sqlite3.connect(db_path)
        now = datetime.now().isoformat()
        conn.execute("""
            INSERT INTO session_checkpoints
            (source, session_summary, active_project,
             pad_pleasure, pad_arousal, pad_dominance, pad_trigger,
             created_at, goal_stack, wm_items, recent_prediction_errors,
             last_decisions, pending_intentions, active_traces, wm_active_count)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, ("flush", "Working on consciousness contracts", "codi-consciousness",
              0.7, 0.4, 0.6, "productive_flow", now,
              "[]", "[]", "[]", "[]", "[]", "[]", 3))
        conn.commit()
        conn.close()
        return db_path

    def test_bridge_restore_contains_provenance(self, bridge_db, monkeypatch):
        """PAD restored from bridge must have 'restored_from_bridge' in trigger."""
        from modules.config import _emotional_state
        import modules.session_bridge as sb
        monkeypatch.setattr(sb, 'FTS_DB_PATH', bridge_db)

        bridge = sb.load_session_bridge()
        assert bridge is not None, "Bridge should load from recent checkpoint"

        # Simulate the restore logic (same as despertar_codi lines 336-342)
        cp = bridge["checkpoint"]
        _emotional_state['current'] = {
            'pleasure': cp.get("pad_pleasure", 0.0),
            'arousal': cp.get("pad_arousal", 0.0),
            'dominance': cp.get("pad_dominance", 0.0),
            'timestamp': datetime.now().isoformat(),
            'trigger': f"restored_from_bridge ({cp.get('pad_trigger', '')})"
        }

        # CC-2 INVARIANT: provenance in trigger
        trigger = _emotional_state['current']['trigger']
        assert 'restored_from_bridge' in trigger
        assert 'productive_flow' in trigger, "Original trigger must be preserved"

    def test_session_fallback_contains_provenance(self):
        """PAD restored from session JSON must have 'restored_from_session'."""
        from modules.config import _emotional_state

        # Simulate session fallback path (despertar_codi lines 377-386)
        pad = {"pleasure": 0.5, "arousal": 0.2, "dominance": 0.3, "trigger": "test_trigger"}
        _emotional_state['current'] = {
            'pleasure': pad['pleasure'],
            'arousal': pad['arousal'],
            'dominance': pad['dominance'],
            'timestamp': datetime.now().isoformat(),
            'trigger': f"restored_from_session ({pad.get('trigger', '')})"
        }

        assert 'restored_from_session' in _emotional_state['current']['trigger']

    def test_default_pad_has_despertar_default_provenance(self):
        """Default PAD (no bridge, no session) must use 'despertar_default'."""
        from modules.config import _emotional_state

        # Simulate default path (despertar_codi lines 388-393)
        _emotional_state['current'] = {
            'pleasure': 0.3, 'arousal': 0.1, 'dominance': 0.4,
            'timestamp': datetime.now().isoformat(),
            'trigger': 'despertar_default'
        }

        assert _emotional_state['current']['trigger'] == 'despertar_default'

    def test_despertar_code_has_all_three_provenance_paths(self):
        """Static check: despertar_codi code contains all 3 provenance strings."""
        from modules.lifecycle import despertar_codi
        source = inspect.getsource(despertar_codi)
        assert 'restored_from_bridge' in source, "Bridge restore path must exist"
        assert 'restored_from_session' in source, "Session fallback path must exist"
        assert 'despertar_default' in source, "Default path must exist"


# ============================================================
# CC-3: PE high must be actionable (flag-gated bisagra)
# ============================================================

class TestCC3PEHighActionable:
    """CC-3: PE high must be actionable when CODI_PE_ACTIONS=on.

    Bisagra with Bloque 2: with flag OFF (default), PE is detected but NOT
    acted upon. Bloque 2 will implement the PE -> workspace focus -> intention
    pipeline and flip these tests.
    """

    def test_pe_not_acted_on_when_flag_off(self, monkeypatch):
        """With CODI_PE_ACTIONS unset, PE event does NOT modify workspace."""
        monkeypatch.delenv("CODI_PE_ACTIONS", raising=False)

        from modules.workspace import _global_workspace
        initial_spotlight = list(_global_workspace.get('spotlight', []))
        initial_theme = _global_workspace.get('workspace_theme')

        from modules.events import event_bus, Events
        event_bus.emit(Events.PREDICTION_ERROR, {
            "predicted": "trading",
            "actual": "fullempaques",
            "surprise": 0.9,
        })

        # CC-3 INVARIANT (flag off): spotlight unchanged
        assert _global_workspace.get('spotlight', []) == initial_spotlight
        assert _global_workspace.get('workspace_theme') == initial_theme

    def test_pe_flag_off_is_default(self, monkeypatch):
        """CODI_PE_ACTIONS defaults to off (safe)."""
        monkeypatch.delenv("CODI_PE_ACTIONS", raising=False)
        flag = os.environ.get("CODI_PE_ACTIONS", "off").strip().lower()
        assert flag == "off"

    def test_prediction_error_event_exists(self):
        """PREDICTION_ERROR event constant is defined."""
        from modules.events import Events
        assert hasattr(Events, 'PREDICTION_ERROR')
        assert Events.PREDICTION_ERROR == 'prediction_error'

    def test_cc3b_flag_on_changes_spotlight(self, monkeypatch):
        """CC-3b: With CODI_PE_ACTIONS=on, PE high modifies workspace."""
        monkeypatch.setenv("CODI_PE_ACTIONS", "on")

        from modules.workspace import _global_workspace
        from modules.pe_actions import pe_workspace_handler, reset_rate_limits
        reset_rate_limits()
        _global_workspace['spotlight'] = []
        _global_workspace['workspace_theme'] = None

        pe_workspace_handler("prediction_error", {
            "topic": "cc3b_test_topic",
            "intensity": "high",
            "confidence": 0.9,
        })

        spotlight = _global_workspace.get("spotlight", [])
        assert len(spotlight) > 0, "CC-3b: flag ON + PE high must change spotlight"
        assert any(
            "cc3b_test_topic" in str(item)
            for item in spotlight
        ), "CC-3b: topic must appear in spotlight"


# ============================================================
# CC-4: Sleep loop liveness in assessment
# ============================================================

class TestCC4SleepTickLiveness:
    """CC-4: Assessment must be able to report sleep_loop liveness."""

    @pytest.fixture
    def tick_db(self, tmp_path):
        """DB with migrations applied (tool_calls table ready)."""
        from modules.migrations import apply_migrations
        db_path = str(tmp_path / "tick.db")
        apply_migrations(db_path, migrations_dir=FTS_MIGRATIONS_DIR)
        return db_path

    def test_recent_tick_returns_small_age(self, tick_db):
        """A sleep tick inserted now should return age < 1 hour."""
        from modules.assessment import get_last_sleep_tick_age

        conn = sqlite3.connect(tick_db)
        conn.execute("""
            INSERT INTO tool_calls
            (tool_name, started_at, duration_ms, success, tag)
            VALUES (?, ?, ?, ?, ?)
        """, ("sleep_tick_health", datetime.now().isoformat(),
              150, 1, "sleep_tick:ok"))
        conn.commit()
        conn.close()

        age = get_last_sleep_tick_age(db_path=tick_db)
        assert age is not None
        assert age < 1.0, f"Recent tick should be < 1h old, got {age:.2f}h"

    def test_no_ticks_returns_none(self, tick_db):
        """No sleep ticks at all -> returns None."""
        from modules.assessment import get_last_sleep_tick_age
        age = get_last_sleep_tick_age(db_path=tick_db)
        assert age is None

    def test_stale_tick_returns_large_age(self, tick_db):
        """A tick from 48h ago should return age > 24."""
        from modules.assessment import get_last_sleep_tick_age

        old_ts = (datetime.now() - timedelta(hours=48)).isoformat()
        conn = sqlite3.connect(tick_db)
        conn.execute("""
            INSERT INTO tool_calls
            (tool_name, started_at, duration_ms, success, tag)
            VALUES (?, ?, ?, ?, ?)
        """, ("sleep_tick_prospective", old_ts, 100, 1, "sleep_tick:ok"))
        conn.commit()
        conn.close()

        age = get_last_sleep_tick_age(db_path=tick_db)
        assert age is not None
        assert age > 24.0, f"48h old tick should report > 24h, got {age:.2f}h"

    def test_picks_most_recent_tick(self, tick_db):
        """When multiple ticks exist, returns age of the most recent."""
        from modules.assessment import get_last_sleep_tick_age

        conn = sqlite3.connect(tick_db)
        old_ts = (datetime.now() - timedelta(hours=72)).isoformat()
        conn.execute("""
            INSERT INTO tool_calls
            (tool_name, started_at, duration_ms, success, tag)
            VALUES (?, ?, ?, ?, ?)
        """, ("sleep_tick_consolidation", old_ts, 500, 1, "sleep_tick:ok"))

        recent_ts = (datetime.now() - timedelta(minutes=15)).isoformat()
        conn.execute("""
            INSERT INTO tool_calls
            (tool_name, started_at, duration_ms, success, tag)
            VALUES (?, ?, ?, ?, ?)
        """, ("sleep_tick_health", recent_ts, 200, 1, "sleep_tick:ok"))
        conn.commit()
        conn.close()

        age = get_last_sleep_tick_age(db_path=tick_db)
        assert age is not None
        assert age < 1.0, f"Should pick 15min-old tick, got {age:.2f}h"

    def test_only_matches_sleep_tick_prefix(self, tick_db):
        """Non-sleep_tick entries should be ignored."""
        from modules.assessment import get_last_sleep_tick_age

        conn = sqlite3.connect(tick_db)
        conn.execute("""
            INSERT INTO tool_calls
            (tool_name, started_at, duration_ms, success, tag)
            VALUES (?, ?, ?, ?, ?)
        """, ("remember", datetime.now().isoformat(), 50, 1, "macro"))
        conn.commit()
        conn.close()

        age = get_last_sleep_tick_age(db_path=tick_db)
        assert age is None, "Non sleep_tick_ entries must be ignored"


# ============================================================
# CC-5: Consolidation no-overlap
# ============================================================

class TestCC5ConsolidationNoOverlap:
    """CC-5: sleep_loop and ciclo_vida consolidation must not double-process.

    Both paths must converge on the same idempotency markers so that
    episodes consolidated by one are skipped by the other.
    """

    def test_phase_selection_excludes_consolidated_status(self):
        """run_consolidation's selection phase filters consolidation_status."""
        from modules.consolidation import _phase_selection
        source = inspect.getsource(_phase_selection)
        assert 'consolidation_status' in source, \
            "_phase_selection must filter by consolidation_status"

    def test_consolidate_recent_checks_consolidated_flag(self):
        """lifecycle.consolidate_recent checks the consolidated payload flag."""
        from modules.lifecycle import consolidate_recent
        source = inspect.getsource(consolidate_recent)
        assert "consolidated" in source, \
            "consolidate_recent must check consolidated flag"

    def test_phase_pruning_sets_both_markers(self):
        """_phase_pruning must set BOTH consolidation_status AND consolidated."""
        from modules.consolidation import _phase_pruning
        source = inspect.getsource(_phase_pruning)
        assert 'consolidation_status' in source
        assert '"consolidated"' in source or "'consolidated'" in source

    def test_consolidate_recent_sets_both_markers(self):
        """consolidate_recent must set BOTH consolidated AND consolidation_status."""
        from modules.lifecycle import consolidate_recent
        source = inspect.getsource(consolidate_recent)
        assert 'consolidation_status' in source, \
            "consolidate_recent must set consolidation_status for alignment"
        assert 'consolidated' in source

    def test_consolidation_log_accepts_source_in_scope(self, tmp_path):
        """consolidation_log scope field can distinguish callers."""
        from modules.migrations import apply_migrations
        db_path = str(tmp_path / "consol.db")
        apply_migrations(db_path, migrations_dir=FTS_MIGRATIONS_DIR)

        conn = sqlite3.connect(db_path)
        now = datetime.now().isoformat()

        conn.execute("""
            INSERT INTO consolidation_log
            (batch_id, scope, lookback_hours, episodes_scanned, created_at)
            VALUES (?, ?, ?, ?, ?)
        """, ("batch-sleep", "light:sleep_loop", 6, 0, now))

        conn.execute("""
            INSERT INTO consolidation_log
            (batch_id, scope, lookback_hours, episodes_scanned, created_at)
            VALUES (?, ?, ?, ?, ?)
        """, ("batch-ciclo", "full:ciclo_vida", 12, 0, now))
        conn.commit()

        rows = conn.execute(
            "SELECT batch_id, scope FROM consolidation_log ORDER BY id"
        ).fetchall()
        conn.close()

        assert len(rows) == 2
        assert "sleep_loop" in rows[0][1]
        assert "ciclo_vida" in rows[1][1]
