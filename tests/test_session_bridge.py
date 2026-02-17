"""
SESSION BRIDGE TESTS
=====================
Verify checkpoint_session_close() and load_session_bridge() behavior.

5 tests:
  1. Insert + Fetch: checkpoint creates row, bridge loads it
  2. Anti-hallucination: no intentions/events -> zero PEs
  3. Freshness: checkpoint >24h -> bridge returns None
  4. Dedupe: two calls <120s -> only 1 row (or second deduped)
  5. Retention: 55 inserts -> only 50 remain

Run: ./venv/bin/pytest tests/test_session_bridge.py -v
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

# Fixtures need to set up the session_checkpoints table before importing module
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

    # Also init working_memory tables (needed for data collection)
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

    # Patch FTS_DB_PATH in session_bridge module
    monkeypatch.setattr("modules.session_bridge.FTS_DB_PATH", db_path)

    return db_path


@pytest.fixture
def mock_external_deps(monkeypatch):
    """Mock external dependencies that session_bridge collects from."""
    # Mock attention schema
    mock_schema = {
        "current_focus": "testing",
        "focus_strength": 0.8,
    }
    monkeypatch.setattr("modules.wiring._attention_schema", mock_schema)

    # Mock emotional state
    monkeypatch.setattr("modules.session_bridge._emotional_state", {
        "current": {
            "pleasure": 0.5, "arousal": 0.3, "dominance": 0.6,
            "trigger": "test_trigger",
        },
        "history": [],
    })

    # Mock prospective.get_pending_intentions
    monkeypatch.setattr(
        "modules.prospective.get_pending_intentions",
        lambda limit=10: []
    )

    # Mock tracing
    monkeypatch.setattr("modules.session_bridge.get_trace_id", lambda: "test_trace_123")
    monkeypatch.setattr("modules.session_bridge._current_session", "2026-02-15-001")


class TestInsertAndFetch:
    """Test 1: checkpoint creates row, bridge loads it."""

    def test_insert_and_fetch(self, isolated_db, mock_external_deps):
        from modules.session_bridge import checkpoint_session_close, load_session_bridge

        result = checkpoint_session_close(
            session_summary="Implementamos session bridge",
            decisions="Usar SQLite para persistencia",
            goal_stack=[{"goal": "test", "status": "done", "next_step": "verify"}],
            source="flush",
        )

        assert result["ok"] is True
        assert result["checkpoint_id"] is not None

        bridge = load_session_bridge()
        assert bridge is not None
        assert "Implementamos session bridge" in bridge["bridge_text"]
        assert bridge["hours_since_last"] < 1.0
        assert isinstance(bridge["prediction_errors"], list)


class TestAntiHallucination:
    """Test 2: no intentions/events -> zero PEs, no invented content."""

    def test_no_pes_without_data(self, isolated_db, mock_external_deps):
        from modules.session_bridge import checkpoint_session_close, load_session_bridge

        checkpoint_session_close(
            session_summary="Clean session",
            decisions="",
            goal_stack=[],
        )

        bridge = load_session_bridge()
        assert bridge is not None
        assert bridge["prediction_errors"] == []
        assert "eventos" not in bridge["bridge_text"].lower()
        assert "paso algo" not in bridge["bridge_text"].lower()


class TestFreshness:
    """Test 3: checkpoint >24h -> bridge returns None."""

    def test_stale_checkpoint_returns_none(self, isolated_db, mock_external_deps):
        from modules.session_bridge import checkpoint_session_close, load_session_bridge

        # Insert checkpoint
        checkpoint_session_close(session_summary="old session")

        # Manually backdate the created_at to 25 hours ago
        old_ts = (datetime.now(TZ_COL) - timedelta(hours=25)).isoformat()
        conn = sqlite3.connect(isolated_db)
        conn.execute("UPDATE session_checkpoints SET created_at = ?", (old_ts,))
        conn.commit()
        conn.close()

        bridge = load_session_bridge()
        assert bridge is None


class TestDedupe:
    """Test 4: two calls <120s -> dedupe (only 1 row or second deduped)."""

    def test_dedupe_same_source(self, isolated_db, mock_external_deps):
        from modules.session_bridge import checkpoint_session_close

        r1 = checkpoint_session_close(session_summary="first", source="flush")
        r2 = checkpoint_session_close(session_summary="second", source="flush")

        assert r1["ok"] is True
        assert r2.get("deduped") is True

        # Verify only 1 row
        conn = sqlite3.connect(isolated_db)
        count = conn.execute("SELECT COUNT(*) FROM session_checkpoints").fetchone()[0]
        conn.close()
        assert count == 1

    def test_flush_overwrites_hook(self, isolated_db, mock_external_deps):
        from modules.session_bridge import checkpoint_session_close

        r1 = checkpoint_session_close(session_summary="hook data", source="hook")
        assert r1["ok"] is True

        r2 = checkpoint_session_close(session_summary="flush data", source="flush")
        assert r2["ok"] is True  # Should upgrade, not dedupe

        # Verify still 1 row but with flush data
        conn = sqlite3.connect(isolated_db)
        row = conn.execute(
            "SELECT session_summary, source FROM session_checkpoints LIMIT 1"
        ).fetchone()
        conn.close()
        assert row[0] == "flush data"
        assert row[1] == "flush"


class TestRetention:
    """Test 5: 55 inserts -> only 50 remain."""

    def test_retention_caps_at_50(self, isolated_db, mock_external_deps, monkeypatch):
        from modules.session_bridge import checkpoint_session_close, DEDUPE_WINDOW_SECONDS

        # Disable dedupe by setting window to 0
        monkeypatch.setattr("modules.session_bridge.DEDUPE_WINDOW_SECONDS", 0)

        for i in range(55):
            # Need unique timestamps to avoid dedupe
            ts = (datetime.now(TZ_COL) - timedelta(hours=55 - i)).isoformat()
            conn = sqlite3.connect(isolated_db)
            conn.execute(
                "INSERT INTO session_checkpoints (session_summary, source, created_at) "
                "VALUES (?, 'flush', ?)",
                (f"session {i}", ts)
            )
            conn.commit()
            conn.close()

        # Now call checkpoint which triggers retention cleanup
        checkpoint_session_close(session_summary="final session")

        conn = sqlite3.connect(isolated_db)
        count = conn.execute("SELECT COUNT(*) FROM session_checkpoints").fetchone()[0]
        conn.close()
        assert count <= 50
