#!/usr/bin/env python3
"""
PE -> ACTION PIPELINE TESTS (Bloque 2)
=======================================
Verify flag gating, workspace focus, prospective dedupe,
rate limiting, and handler resilience.

6 tests:
  1. PE high focuses workspace when flag ON
  2. PE does not change spotlight when flag OFF
  3. PE creates intention once, dedupes second call
  4. PE low does not create intention
  5. PE rate limited per topic (same topic in <6h -> skip)
  6. Handler failure does not break emit

Run: ./venv/bin/pytest tests/test_pe_actions.py -v
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import time
import pytest

from modules.events import event_bus, Events


# ============================================================
# Fixtures
# ============================================================

@pytest.fixture(autouse=True)
def clean_pe_state(monkeypatch):
    """Reset PE action state before each test."""
    from modules import pe_actions
    pe_actions.reset_rate_limits()

    # Reset workspace to clean slate
    from modules.workspace import _global_workspace
    _global_workspace['spotlight'] = []
    _global_workspace['workspace_theme'] = None
    _global_workspace['recent_context'] = []

    yield

    # Clean up env var
    monkeypatch.delenv("CODI_PE_ACTIONS", raising=False)


@pytest.fixture
def prospective_db(tmp_path, monkeypatch):
    """Set up a temp prospective DB for intention tests."""
    import sqlite3
    db_path = str(tmp_path / "prospective_test.db")
    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS intentions (
            id TEXT PRIMARY KEY,
            action TEXT NOT NULL,
            trigger_type TEXT DEFAULT 'event',
            trigger_spec TEXT DEFAULT '{}',
            cue_focality TEXT DEFAULT 'focal',
            priority TEXT DEFAULT 'medium',
            status TEXT DEFAULT 'pending',
            activation REAL DEFAULT 0.65,
            created_at TEXT,
            completed_at TEXT,
            expiry TEXT,
            context_at_creation TEXT DEFAULT '',
            creator TEXT DEFAULT 'codi',
            recurrence TEXT
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS intention_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            intention_id TEXT NOT NULL,
            event TEXT NOT NULL,
            detail TEXT,
            created_at TEXT
        )
    """)
    conn.commit()
    conn.close()

    # Monkeypatch prospective module to use this DB
    from modules import prospective
    monkeypatch.setattr(prospective, "PROSPECTIVE_DB_PATH", db_path)
    # Reset cached connection
    monkeypatch.setattr(prospective, "_conn", None)

    return db_path


# ============================================================
# Test 1: PE high focuses workspace when enabled
# ============================================================

class TestPEWorkspaceFocus:
    """H1: PE high -> spotlight when flag ON."""

    def test_pe_high_focuses_workspace_when_enabled(self, monkeypatch):
        """Flag ON + intensity=high -> topic appears in spotlight."""
        monkeypatch.setenv("CODI_PE_ACTIONS", "on")

        from modules.workspace import _global_workspace
        from modules.pe_actions import pe_workspace_handler

        pe_workspace_handler(Events.PREDICTION_ERROR, {
            "topic": "trading_strategy",
            "intensity": "high",
            "confidence": 0.9,
            "source_tool": "record_surprise",
        })

        spotlight = _global_workspace.get("spotlight", [])
        assert len(spotlight) > 0
        # First item should be the PE entry
        pe_item = spotlight[0]
        assert isinstance(pe_item, dict)
        assert "trading_strategy" in pe_item.get("content", "")
        assert pe_item.get("source") == "prediction_error"
        assert _global_workspace.get("workspace_theme") == "prediction_error:trading_strategy"


# ============================================================
# Test 2: PE does NOT change spotlight when flag OFF
# ============================================================

class TestPEFlagOff:
    """Flag OFF -> no workspace changes."""

    def test_pe_off_does_not_change_spotlight(self, monkeypatch):
        """Flag OFF + high PE -> spotlight unchanged."""
        monkeypatch.delenv("CODI_PE_ACTIONS", raising=False)

        from modules.workspace import _global_workspace
        from modules.pe_actions import pe_workspace_handler

        initial_spotlight = list(_global_workspace.get("spotlight", []))

        pe_workspace_handler(Events.PREDICTION_ERROR, {
            "topic": "trading",
            "intensity": "high",
            "confidence": 0.9,
        })

        assert _global_workspace.get("spotlight", []) == initial_spotlight


# ============================================================
# Test 3: PE creates intention once, dedupes second call
# ============================================================

class TestPEIntentionDedupe:
    """H2: intention created once, second PE same day -> dedupe."""

    def test_pe_creates_intention_once_deduped(self, monkeypatch, prospective_db):
        """First PE -> intention created. Second same topic -> skipped."""
        monkeypatch.setenv("CODI_PE_ACTIONS", "on")

        import sqlite3
        from modules.pe_actions import pe_prospective_handler

        # First call: should create intention
        pe_prospective_handler(Events.PREDICTION_ERROR, {
            "topic": "fullempaques_bug",
            "intensity": "high",
            "confidence": 0.8,
        })

        conn = sqlite3.connect(prospective_db)
        count1 = conn.execute("SELECT COUNT(*) FROM intentions").fetchone()[0]
        conn.close()
        assert count1 == 1

        # Second call: same topic, same day -> dedupe
        pe_prospective_handler(Events.PREDICTION_ERROR, {
            "topic": "fullempaques_bug",
            "intensity": "high",
            "confidence": 0.8,
        })

        conn = sqlite3.connect(prospective_db)
        count2 = conn.execute("SELECT COUNT(*) FROM intentions").fetchone()[0]
        conn.close()
        assert count2 == 1  # Still 1 — deduped


# ============================================================
# Test 4: PE low does NOT create intention
# ============================================================

class TestPELowIntensity:
    """Low intensity PE -> no intention, no workspace change."""

    def test_pe_low_does_not_create_intention(self, monkeypatch, prospective_db):
        """intensity=low -> H2 skips."""
        monkeypatch.setenv("CODI_PE_ACTIONS", "on")

        import sqlite3
        from modules.pe_actions import pe_prospective_handler

        pe_prospective_handler(Events.PREDICTION_ERROR, {
            "topic": "minor_thing",
            "intensity": "low",
            "confidence": 0.3,
        })

        conn = sqlite3.connect(prospective_db)
        count = conn.execute("SELECT COUNT(*) FROM intentions").fetchone()[0]
        conn.close()
        assert count == 0

    def test_pe_medium_does_not_create_intention(self, monkeypatch, prospective_db):
        """intensity=medium -> H2 skips (only high triggers action)."""
        monkeypatch.setenv("CODI_PE_ACTIONS", "on")

        import sqlite3
        from modules.pe_actions import pe_prospective_handler

        pe_prospective_handler(Events.PREDICTION_ERROR, {
            "topic": "medium_thing",
            "intensity": "medium",
            "confidence": 0.6,
        })

        conn = sqlite3.connect(prospective_db)
        count = conn.execute("SELECT COUNT(*) FROM intentions").fetchone()[0]
        conn.close()
        assert count == 0


# ============================================================
# Test 5: Rate limited per topic
# ============================================================

class TestPERateLimit:
    """Same topic within cooldown -> second action skipped."""

    def test_pe_rate_limited_per_topic(self, monkeypatch):
        """First PE on topic -> spotlight. Second within 6h -> no-op."""
        monkeypatch.setenv("CODI_PE_ACTIONS", "on")

        from modules.workspace import _global_workspace
        from modules.pe_actions import pe_workspace_handler

        # First call: should focus
        pe_workspace_handler(Events.PREDICTION_ERROR, {
            "topic": "rate_test_topic",
            "intensity": "high",
        })
        spotlight_after_first = list(_global_workspace.get("spotlight", []))
        assert len(spotlight_after_first) > 0

        # Reset spotlight to see if second call changes it
        _global_workspace['spotlight'] = []
        _global_workspace['workspace_theme'] = None

        # Second call: same topic, should be rate limited
        pe_workspace_handler(Events.PREDICTION_ERROR, {
            "topic": "rate_test_topic",
            "intensity": "high",
        })
        assert _global_workspace.get("spotlight", []) == []  # Rate limited

    def test_pe_rate_limit_expires(self, monkeypatch):
        """After cooldown expires, topic can trigger again."""
        monkeypatch.setenv("CODI_PE_ACTIONS", "on")

        from modules import pe_actions
        from modules.workspace import _global_workspace

        # Simulate: set clock to "past" for first action
        fake_time = [time.time()]
        monkeypatch.setattr(pe_actions, "_now", lambda: fake_time[0])

        pe_actions.pe_workspace_handler(Events.PREDICTION_ERROR, {
            "topic": "expire_test",
            "intensity": "high",
        })
        assert len(_global_workspace.get("spotlight", [])) > 0

        # Reset spotlight
        _global_workspace['spotlight'] = []
        _global_workspace['workspace_theme'] = None

        # Advance clock past cooldown (6h + 1s)
        fake_time[0] += pe_actions.COOLDOWN_SECONDS + 1

        pe_actions.pe_workspace_handler(Events.PREDICTION_ERROR, {
            "topic": "expire_test",
            "intensity": "high",
        })
        # Should fire again since cooldown expired
        assert len(_global_workspace.get("spotlight", [])) > 0


# ============================================================
# Test 6: Handler failure does not break emit
# ============================================================

class TestHandlerResilience:
    """A broken handler doesn't crash event_bus.emit."""

    def test_handler_failure_does_not_break_emit(self, monkeypatch):
        """EventBus catches exceptions in handlers (verified by events.py L133-136)."""
        errors_caught = []

        def broken_handler(event_name, data):
            raise RuntimeError("intentional test failure")

        def working_handler(event_name, data):
            errors_caught.append("survived")

        # Register broken handler first, then working handler
        event_bus.on("test_resilience_event", broken_handler)
        event_bus.on("test_resilience_event", working_handler)

        try:
            event_bus.emit("test_resilience_event", {"test": True})
        finally:
            event_bus.off("test_resilience_event", broken_handler)
            event_bus.off("test_resilience_event", working_handler)

        # Working handler must have run despite broken one
        assert "survived" in errors_caught
