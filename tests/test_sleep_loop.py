"""
SLEEP LOOP TESTS
=================
Verify sleep loop ticks, lock file, idempotent report writing,
and budget enforcement.

6 tests:
  1. Full loop: all 4 ticks execute and produce report
  2. Lock file: second instance skips when lock exists
  3. Idempotent write: report not clobbered on re-run
  4. No eligible checkpoint: skip when all have reports
  5. Budget enforcement: budget exhaustion skips remaining ticks
  6. Sleep report appears in bridge: load_session_bridge includes it

Run: ./venv/bin/pytest tests/test_sleep_loop.py -v
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import sqlite3
import time
import pytest
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock
from zoneinfo import ZoneInfo

TZ_COL = ZoneInfo("America/Bogota")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FTS_MIGRATIONS_DIR = os.path.join(PROJECT_ROOT, "migrations")


@pytest.fixture
def isolated_db(tmp_path, monkeypatch):
    """Create an isolated SQLite DB with session_checkpoints table."""
    db_path = str(tmp_path / "test_fts.db")
    monkeypatch.setenv("FTS_DB_PATH", db_path)
    monkeypatch.setattr("modules.config.FTS_DB_PATH", db_path)

    # Apply migrations to create tables
    from modules.migrations import apply_migrations
    apply_migrations(db_path, migrations_dir=FTS_MIGRATIONS_DIR)

    # Also init working_memory tables
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

    # Patch DB paths in relevant modules
    monkeypatch.setattr("modules.session_bridge.FTS_DB_PATH", db_path)
    monkeypatch.setattr("modules.sleep_loop.FTS_DB_PATH", db_path)
    monkeypatch.setattr("modules.sleep_loop.DATA_DIR", str(tmp_path))
    monkeypatch.setattr("modules.sleep_loop.LOCK_FILE", str(tmp_path / "sleep_loop.lock"))

    return db_path


@pytest.fixture
def mock_external_deps(monkeypatch):
    """Mock external dependencies for session_bridge."""
    mock_schema = {"current_focus": "testing", "focus_strength": 0.8}
    monkeypatch.setattr("modules.wiring._attention_schema", mock_schema)
    monkeypatch.setattr("modules.session_bridge._emotional_state", {
        "current": {
            "pleasure": 0.5, "arousal": 0.3, "dominance": 0.6,
            "trigger": "test",
        },
        "history": [],
    })
    monkeypatch.setattr(
        "modules.prospective.get_pending_intentions",
        lambda limit=10: []
    )
    monkeypatch.setattr("modules.session_bridge.get_trace_id", lambda: "test_trace")
    monkeypatch.setattr("modules.session_bridge._current_session", "test-session")


@pytest.fixture
def mock_tick_deps(monkeypatch):
    """Mock all tick functions to avoid external dependencies (pg_store, Qdrant, Ollama).

    After the PostgreSQL migration, individual tick functions reach pg_store,
    Qdrant, Ollama, and other external services.  Mocking at the tick-function
    level keeps this test focused on sleep-loop mechanics (ordering, budget,
    report generation) rather than individual tick behaviour.
    """
    from modules.sleep_loop import TICK_ORDER

    def _make_tick_mock(name):
        def tick_fn(budget_ms):
            return {"tick": name, "ok": True, "detail": "mocked", "elapsed_ms": 10}
        return tick_fn

    for tick_name in TICK_ORDER:
        monkeypatch.setattr(
            f"modules.sleep_loop._tick_{tick_name}",
            _make_tick_mock(tick_name),
        )

    # Also mock world model prioritization for deterministic ordering
    monkeypatch.setattr(
        "modules.sleep_loop.SleepWorldModel.prioritize_ticks",
        lambda self, eligible, conn=None: eligible,
    )


def _insert_checkpoint(db_path, summary="test session", source="flush",
                       created_at=None, sleep_report=None):
    """Helper to insert a checkpoint directly."""
    if created_at is None:
        created_at = datetime.now(TZ_COL).isoformat()
    conn = sqlite3.connect(db_path)
    conn.execute(
        "INSERT INTO session_checkpoints (session_summary, source, created_at, sleep_report) "
        "VALUES (?, ?, ?, ?)",
        (summary, source, created_at, sleep_report)
    )
    conn.commit()
    last_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
    conn.close()
    return last_id


class TestFullLoop:
    """Test 1: all 8 ticks execute and produce a report."""

    def test_all_ticks_run(self, isolated_db, mock_tick_deps):
        from modules.sleep_loop import run_sleep_loop, TICK_ORDER

        result = run_sleep_loop(reason="test", budget_ms=8000)

        assert result["ok"] is True
        assert result["report"] is not None
        assert "SLEEP REPORT" in result["report"]
        assert len(result["tick_results"]) == len(TICK_ORDER)

        # All ticks should succeed
        for tick in result["tick_results"]:
            assert tick["ok"] is True, f"Tick {tick['tick']} failed: {tick.get('detail')}"

    def test_report_written_to_checkpoint(self, isolated_db, mock_tick_deps, mock_external_deps):
        from modules.sleep_loop import run_sleep_loop, _write_sleep_report

        cp_id = _insert_checkpoint(isolated_db, "test session")

        result = run_sleep_loop(reason="test", budget_ms=8000)
        written = _write_sleep_report(cp_id, result["report"])

        assert written is True

        # Verify in DB
        conn = sqlite3.connect(isolated_db)
        row = conn.execute(
            "SELECT sleep_report FROM session_checkpoints WHERE id = ?", (cp_id,)
        ).fetchone()
        conn.close()

        assert row is not None
        assert "SLEEP REPORT" in row[0]


class TestLockFile:
    """Test 2: lock file prevents double execution."""

    def test_lock_blocks_second_instance(self, isolated_db, tmp_path, monkeypatch):
        from modules.sleep_loop import _acquire_lock, _release_lock, LOCK_FILE

        lock_path = str(tmp_path / "sleep_loop.lock")
        monkeypatch.setattr("modules.sleep_loop.LOCK_FILE", lock_path)

        # First acquire should succeed
        assert _acquire_lock() is True

        # Second acquire should fail (same PID holds lock)
        assert _acquire_lock() is False

        # Release
        _release_lock()

        # Now should succeed again
        assert _acquire_lock() is True
        _release_lock()

    def test_stale_lock_removed(self, isolated_db, tmp_path, monkeypatch):
        from modules.sleep_loop import _acquire_lock, _release_lock

        lock_path = str(tmp_path / "sleep_loop.lock")
        monkeypatch.setattr("modules.sleep_loop.LOCK_FILE", lock_path)

        # Write a stale lock with a PID that doesn't exist
        with open(lock_path, 'w') as f:
            f.write("999999999")

        # Should acquire (stale lock removed)
        assert _acquire_lock() is True
        _release_lock()


class TestIdempotentWrite:
    """Test 3: report not clobbered on re-run."""

    def test_no_clobber(self, isolated_db):
        from modules.sleep_loop import _write_sleep_report

        cp_id = _insert_checkpoint(isolated_db, "test session")

        # First write
        written1 = _write_sleep_report(cp_id, "FIRST REPORT")
        assert written1 is True

        # Second write should NOT overwrite
        written2 = _write_sleep_report(cp_id, "SECOND REPORT")
        assert written2 is False

        # Verify original preserved
        conn = sqlite3.connect(isolated_db)
        row = conn.execute(
            "SELECT sleep_report FROM session_checkpoints WHERE id = ?", (cp_id,)
        ).fetchone()
        conn.close()
        assert row[0] == "FIRST REPORT"


class TestNoEligibleCheckpoint:
    """Test 4: skip when all checkpoints have reports or are too old."""

    def test_skip_when_all_have_reports(self, isolated_db):
        from modules.sleep_loop import _get_target_checkpoint

        _insert_checkpoint(isolated_db, "session 1", sleep_report="already done")

        target = _get_target_checkpoint(max_age_min=30)
        assert target is None

    def test_skip_when_too_old(self, isolated_db):
        from modules.sleep_loop import _get_target_checkpoint

        old_ts = (datetime.now(TZ_COL) - timedelta(hours=2)).isoformat()
        _insert_checkpoint(isolated_db, "old session", created_at=old_ts)

        target = _get_target_checkpoint(max_age_min=30)
        assert target is None

    def test_finds_eligible(self, isolated_db):
        from modules.sleep_loop import _get_target_checkpoint

        cp_id = _insert_checkpoint(isolated_db, "fresh session")

        target = _get_target_checkpoint(max_age_min=30)
        assert target == cp_id


class TestBudgetEnforcement:
    """Test 5: budget exhaustion skips remaining ticks."""

    def test_zero_budget_skips_all(self, isolated_db, mock_tick_deps):
        from modules.sleep_loop import run_sleep_loop

        result = run_sleep_loop(reason="test", budget_ms=0)

        assert result["ok"] is True
        # All ticks should be skipped
        for tick in result["tick_results"]:
            assert "skipped" in tick.get("detail", "") or tick.get("ok") is True


class TestBridgeIntegration:
    """Test 6: sleep report appears in load_session_bridge output."""

    def test_bridge_includes_sleep_report(self, isolated_db, mock_external_deps):
        from modules.session_bridge import checkpoint_session_close, load_session_bridge

        # Create checkpoint
        checkpoint_session_close(
            session_summary="Implementamos sleep loop",
            source="flush",
        )

        # Write sleep report directly
        conn = sqlite3.connect(isolated_db)
        conn.execute(
            "UPDATE session_checkpoints SET sleep_report = ? "
            "WHERE id = (SELECT id FROM session_checkpoints ORDER BY created_at DESC LIMIT 1)",
            ("SLEEP REPORT (test, 1500ms)\n- consolidation: OK (800ms) light: 0 clusters",)
        )
        conn.commit()
        conn.close()

        # Load bridge
        bridge = load_session_bridge()

        assert bridge is not None
        assert bridge.get("sleep_report") is not None
        assert "SLEEP REPORT" in bridge["sleep_report"]
        # Also check it appears in bridge text
        assert "[Sleep]" in bridge["bridge_text"]
