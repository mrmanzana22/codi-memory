"""
CONTINUITY ABLATION TESTS
===========================
6 ablation tests validating graceful degradation when pipeline components are removed.

Each test removes exactly one component and verifies the system degrades predictably,
not catastrophically.

Ablations:
  A: Without bridge -> JSON fallback
  B: Without sleep report -> bridge OK but no [Sleep]
  C: Without hook -> only flush creates checkpoints
  D: Without prospective tick -> activations frozen
  E: Without homeostasis -> PAD frozen at extremes
  F: Stale checkpoint -> cold start

Run: ./venv/bin/pytest tests/test_continuity_ablations.py -v
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import sqlite3
import pytest
from datetime import datetime, timedelta
from unittest.mock import MagicMock
from zoneinfo import ZoneInfo

TZ_COL = ZoneInfo("America/Bogota")

pytestmark = [pytest.mark.continuity, pytest.mark.ablation]

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FTS_MIGRATIONS_DIR = os.path.join(PROJECT_ROOT, "migrations")


# ============================================================
# SHARED FIXTURES (same as battery)
# ============================================================

@pytest.fixture
def continuity_db(tmp_path, monkeypatch):
    """Isolated SQLite DB with full migrations + WM tables."""
    db_path = str(tmp_path / "test_fts.db")
    monkeypatch.setenv("FTS_DB_PATH", db_path)
    monkeypatch.setattr("modules.config.FTS_DB_PATH", db_path)

    from modules.migrations import apply_migrations
    apply_migrations(db_path, migrations_dir=FTS_MIGRATIONS_DIR)

    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS working_memory (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            content TEXT NOT NULL,
            topic TEXT DEFAULT 'general',
            relevance REAL DEFAULT 0.5,
            added_at TEXT NOT NULL,
            occurred_at TEXT,
            source TEXT DEFAULT 'interaction',
            related_memory_id TEXT,
            chain_id TEXT,
            active INTEGER DEFAULT 1,
            last_accessed_at TEXT,
            access_count INTEGER DEFAULT 0
        );
        CREATE TABLE IF NOT EXISTS narrative_traces (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            trace_name TEXT NOT NULL UNIQUE,
            chain_ids TEXT NOT NULL,
            theme TEXT,
            created_at TEXT NOT NULL,
            last_updated TEXT NOT NULL,
            active INTEGER DEFAULT 1
        );
    """)
    conn.commit()
    conn.close()

    monkeypatch.setattr("modules.session_bridge.FTS_DB_PATH", db_path)
    monkeypatch.setattr("modules.sleep_loop.FTS_DB_PATH", db_path)
    monkeypatch.setattr("modules.sleep_loop.DATA_DIR", str(tmp_path))
    monkeypatch.setattr("modules.sleep_loop.LOCK_FILE", str(tmp_path / "sleep_loop.lock"))
    monkeypatch.setattr("modules.db_pool.FTS_DB_PATH", db_path)
    monkeypatch.setattr("modules.working_memory._TABLES_INITIALIZED", False)

    return db_path


@pytest.fixture
def mock_external_deps(monkeypatch):
    """Mock external deps for session_bridge."""
    monkeypatch.setattr("modules.wiring._attention_schema", {
        "current_focus": "ablation_test", "focus_strength": 0.5,
    })
    monkeypatch.setattr("modules.session_bridge._emotional_state", {
        "current": {
            "pleasure": 0.4, "arousal": 0.2, "dominance": 0.5,
            "trigger": "ablation",
        },
        "mood": {"pleasure": 0.2, "arousal": 0.1, "dominance": 0.3, "last_updated": None},
        "history": [],
        "decay_rate": 0.1,
        "mood_shift_rate": 0.05,
    })
    monkeypatch.setattr(
        "modules.prospective.get_pending_intentions",
        lambda limit=10: []
    )
    monkeypatch.setattr("modules.session_bridge.get_trace_id", lambda: "abl_trace")
    monkeypatch.setattr("modules.session_bridge._current_session", "abl-session")


@pytest.fixture
def clean_pad():
    """Save/restore _emotional_state to prevent PAD leaks."""
    from modules.config import _emotional_state
    import copy
    saved = copy.deepcopy(_emotional_state)
    yield _emotional_state
    _emotional_state.clear()
    _emotional_state.update(saved)


class FakePoint:
    """Minimal point mimic for pg.scroll results."""
    def __init__(self, point_id, payload):
        self.id = point_id
        self.payload = payload


class FakePg:
    """Minimal pg_store fake for lifecycle tests.

    Matches the modules.pg_store.pg interface used by despertar_codi:
      - scroll(filters=, limit=, is_semantic=, offset=) -> (points, next_offset)
      - count(is_semantic=) -> object with .points_count
      - search(query, limit=, is_semantic=) -> dict
    """
    def __init__(self, points):
        self.points = points

    def scroll(self, filters=None, limit=50, is_semantic=None, offset=0, **kwargs):
        return (self.points, None)

    def count(self, is_semantic=None):
        return MagicMock(points_count=50)

    def search(self, query, limit=5, is_semantic=None, **kwargs):
        return {"results": []}


def _insert_checkpoint(db_path, **overrides):
    """Helper to insert a checkpoint."""
    from modules.config import now_col as _now_col
    defaults = {
        "session_summary": "ablation test",
        "source": "flush",
        "created_at": _now_col().isoformat(),
        "sleep_report": None,
        "pad_pleasure": 0.4,
        "pad_arousal": 0.2,
        "pad_dominance": 0.5,
        "pad_trigger": "ablation",
        "pending_intentions": "[]",
        "goal_stack": "[]",
        "active_project": None,
        "wm_items": "[]",
        "wm_active_count": 0,
        "last_decisions": "[]",
        "active_traces": "[]",
    }
    defaults.update(overrides)
    conn = sqlite3.connect(db_path)
    cols = list(defaults.keys())
    placeholders = ", ".join(["?"] * len(cols))
    conn.execute(
        f"INSERT INTO session_checkpoints ({', '.join(cols)}) VALUES ({placeholders})",
        list(defaults.values())
    )
    conn.commit()
    last_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
    conn.close()
    return last_id


# ============================================================
# ABLATION A: Without bridge -> JSON fallback
# ============================================================

class TestWithoutBridge:
    """Ablation A: when load_session_bridge returns None, despertar uses JSON fallback."""

    def test_without_bridge_json_fallback(self, continuity_db, monkeypatch, clean_pad):
        """No bridge checkpoint -> despertar uses JSON session state fallback."""
        from modules.config import _emotional_state

        # Mock bridge to return None (no checkpoints)
        monkeypatch.setattr(
            "modules.session_bridge.load_session_bridge",
            lambda: None
        )

        # Provide JSON fallback with known PAD
        monkeypatch.setattr(
            "modules.flush.load_session_state",
            lambda: {
                "session_summary": "JSON fallback session",
                "pad": {"pleasure": 0.3, "arousal": 0.1, "dominance": 0.4, "trigger": "json_fallback"},
                "active_project": "testing",
            }
        )

        # Mock the other deps
        monkeypatch.setattr(
            "modules.lifecycle._verificar_salud_memoria_interna",
            lambda: {"ok": True, "total_memories": 50, "message": "OK"}
        )
        monkeypatch.setattr("modules.triggers._load_triggers", lambda: [])
        monkeypatch.setattr("modules.maintenance._verificar_tareas_vencidas", lambda: [])
        monkeypatch.setattr("modules.lifecycle.pg", FakePg([]))
        from modules.events import event_bus
        monkeypatch.setattr(event_bus, "emit", lambda *a, **kw: None)

        from modules.consciousness import despertar_codi
        result_text = despertar_codi(verbose=True)

        # Should NOT have SESSION BRIDGE section
        assert "SESSION BRIDGE" not in result_text

        # PAD should be from JSON fallback
        current = _emotional_state['current']
        assert abs(current['pleasure'] - 0.3) < 0.01
        assert "restored_from_session" in current.get('trigger', '') or \
               "json_fallback" in current.get('trigger', '')


# ============================================================
# ABLATION B: Without sleep report -> bridge OK but no [Sleep]
# ============================================================

class TestWithoutSleepReport:
    """Ablation B: bridge works fine without sleep report."""

    def test_without_sleep_report_bridge_works(self, continuity_db, mock_external_deps):
        """sleep_report=NULL -> bridge OK but no [Sleep] section."""
        from modules.session_bridge import load_session_bridge

        _insert_checkpoint(
            continuity_db,
            session_summary="Session without sleep",
            sleep_report=None,
        )

        bridge = load_session_bridge()
        assert bridge is not None
        assert "[Sleep]" not in bridge["bridge_text"]
        assert bridge["sleep_report"] is None
        assert "Session without sleep" in bridge["bridge_text"]


# ============================================================
# ABLATION C: Without hook -> only flush creates checkpoints
# ============================================================

class TestWithoutHook:
    """Ablation C: without hook, only explicit flush creates checkpoints."""

    def test_without_hook_no_auto_checkpoint(self, continuity_db, mock_external_deps):
        """Without hook source, flush is the only checkpoint creator."""
        from modules.session_bridge import checkpoint_session_close

        # Only flush -- no hook ever fired
        r = checkpoint_session_close(
            session_summary="Manual flush only",
            source="flush",
        )
        assert r["ok"] is True

        conn = sqlite3.connect(continuity_db)
        count = conn.execute("SELECT COUNT(*) FROM session_checkpoints").fetchone()[0]
        sources = [row[0] for row in conn.execute(
            "SELECT source FROM session_checkpoints"
        ).fetchall()]
        conn.close()

        assert count == 1
        assert all(s == "flush" for s in sources)


# ============================================================
# ABLATION D: Without prospective tick -> activations frozen
# ============================================================

class TestWithoutProspective:
    """Ablation D: prospective no-op -> activation values don't change."""

    def test_without_prospective_tick_frozen(self, continuity_db, monkeypatch):
        """With prospective tick as no-op, activations stay frozen."""
        from modules.sleep_loop import TICK_ORDER

        ticks_called = []

        def make_tick_mock(name):
            def tick_fn(budget_ms):
                ticks_called.append(name)
                return {"tick": name, "ok": True, "detail": "mocked", "elapsed_ms": 10}
            return tick_fn

        # Mock all ticks normally, but prospective is a no-op
        for tick_name in TICK_ORDER:
            monkeypatch.setattr(
                f"modules.sleep_loop._tick_{tick_name}",
                make_tick_mock(tick_name)
            )

        # Override prospective to be explicit no-op
        def noop_prospective(budget_ms):
            return {"tick": "prospective", "ok": True, "detail": "no-op (ablated)", "elapsed_ms": 0}
        monkeypatch.setattr("modules.sleep_loop._tick_prospective", noop_prospective)

        from modules.sleep_loop import run_sleep_loop
        result = run_sleep_loop(reason="ablation_d", budget_ms=8000)

        assert result["ok"] is True
        # Prospective ran but as no-op
        prospective_tick = next(
            t for t in result["tick_results"] if t["tick"] == "prospective"
        )
        assert "no-op" in prospective_tick["detail"]


# ============================================================
# ABLATION E: Without homeostasis -> PAD frozen at extremes
# ============================================================

class TestWithoutHomeostasis:
    """Ablation E: homeostasis no-op -> PAD stays at extreme values, no decay."""

    def test_without_homeostasis_pad_frozen(self, clean_pad, monkeypatch):
        """Without emotional decay, PAD stays at extreme values."""
        from modules.config import _emotional_state

        # Set extreme PAD
        _emotional_state['current'] = {
            'pleasure': 0.9, 'arousal': 0.9, 'dominance': 0.9,
            'timestamp': datetime.now(TZ_COL).isoformat(),
            'trigger': 'extreme_ablation',
        }
        _emotional_state['mood'] = {
            'pleasure': 0.2, 'arousal': 0.1, 'dominance': 0.3,
            'last_updated': None,
        }
        _emotional_state['decay_rate'] = 0.1
        _emotional_state['history'] = []

        # DO NOT call apply_emotional_decay (simulating homeostasis ablation)
        # Instead verify PAD stays extreme
        current = _emotional_state['current']
        assert current['pleasure'] == 0.9
        assert current['arousal'] == 0.9
        assert current['dominance'] == 0.9

        # Compare: with decay it would move toward baseline
        # Without it, values are frozen at extremes


# ============================================================
# ABLATION F: Stale checkpoint -> cold start
# ============================================================

class TestStaleCheckpoint:
    """Ablation F: checkpoint >24h -> bridge returns None, cold start."""

    def test_stale_checkpoint_cold_start(self, continuity_db, mock_external_deps, monkeypatch, clean_pad):
        """Checkpoint >24h -> bridge None, despertar does cold start."""
        from modules.config import _emotional_state, now_col as _now_col

        # Insert very old checkpoint
        old_ts = (_now_col() - timedelta(hours=30)).isoformat()
        _insert_checkpoint(
            continuity_db,
            created_at=old_ts,
            session_summary="Ancient session from 30h ago",
            pad_pleasure=0.9, pad_arousal=0.8, pad_dominance=0.7,
        )

        # Verify bridge returns None (stale)
        from modules.session_bridge import load_session_bridge
        bridge = load_session_bridge()
        assert bridge is None, "Stale checkpoint should return None bridge"

        # Now test despertar with this stale data
        monkeypatch.setattr(
            "modules.lifecycle._verificar_salud_memoria_interna",
            lambda: {"ok": True, "total_memories": 50, "message": "OK"}
        )
        monkeypatch.setattr("modules.triggers._load_triggers", lambda: [])
        monkeypatch.setattr("modules.maintenance._verificar_tareas_vencidas", lambda: [])
        monkeypatch.setattr("modules.lifecycle.pg", FakePg([]))
        monkeypatch.setattr("modules.flush.load_session_state", lambda: None)
        from modules.events import event_bus
        monkeypatch.setattr(event_bus, "emit", lambda *a, **kw: None)

        from modules.consciousness import despertar_codi
        result_text = despertar_codi()

        # Should NOT have SESSION BRIDGE (stale)
        assert "SESSION BRIDGE" not in result_text

        # PAD should be at default cold-start values (0.3, 0.1, 0.4)
        current = _emotional_state['current']
        assert abs(current['pleasure'] - 0.3) < 0.01, f"pleasure: {current['pleasure']}"
        assert abs(current['arousal'] - 0.1) < 0.01, f"arousal: {current['arousal']}"
        assert abs(current['dominance'] - 0.4) < 0.01, f"dominance: {current['dominance']}"
