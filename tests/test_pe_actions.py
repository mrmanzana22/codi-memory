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
    """Mock PG-based prospective module for PE action tests.

    Post PG-migration: prospective.py uses PostgreSQL, so we mock
    create_intention and check_intention_exists instead of SQLite.
    Tracks created intentions in a list for assertions.
    """
    created_intentions = []

    def _mock_create_intention(**kwargs):
        created_intentions.append(kwargs)
        return {"id": f"mock-{len(created_intentions)}", "ok": True}

    def _mock_check_intention_exists(context_pattern):
        # Search created intentions for matching context
        for intent in created_intentions:
            ctx = intent.get("context", "")
            if context_pattern in ctx:
                return True
        return False

    from modules import prospective
    monkeypatch.setattr(prospective, "create_intention", _mock_create_intention)
    monkeypatch.setattr(prospective, "check_intention_exists", _mock_check_intention_exists)

    return created_intentions


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

        from modules.pe_actions import pe_prospective_handler

        # First call: should create intention
        pe_prospective_handler(Events.PREDICTION_ERROR, {
            "topic": "fullempaques_bug",
            "intensity": "high",
            "confidence": 0.8,
        })

        # prospective_db is the list of created intentions (mocked)
        assert len(prospective_db) == 1

        # Second call: same topic, same day -> dedupe (check_intention_exists finds it)
        pe_prospective_handler(Events.PREDICTION_ERROR, {
            "topic": "fullempaques_bug",
            "intensity": "high",
            "confidence": 0.8,
        })

        assert len(prospective_db) == 1  # Still 1 — deduped


# ============================================================
# Test 4: PE low does NOT create intention
# ============================================================

class TestPELowIntensity:
    """Low intensity PE -> no intention, no workspace change."""

    def test_pe_low_does_not_create_intention(self, monkeypatch, prospective_db):
        """intensity=low -> H2 skips."""
        monkeypatch.setenv("CODI_PE_ACTIONS", "on")

        from modules.pe_actions import pe_prospective_handler

        pe_prospective_handler(Events.PREDICTION_ERROR, {
            "topic": "minor_thing",
            "intensity": "low",
            "confidence": 0.3,
        })

        # prospective_db is the list of created intentions (mocked)
        assert len(prospective_db) == 0

    def test_pe_medium_does_not_create_intention(self, monkeypatch, prospective_db):
        """intensity=medium -> H2 skips (only high triggers action)."""
        monkeypatch.setenv("CODI_PE_ACTIONS", "on")

        from modules.pe_actions import pe_prospective_handler

        pe_prospective_handler(Events.PREDICTION_ERROR, {
            "topic": "medium_thing",
            "intensity": "medium",
            "confidence": 0.6,
        })

        # prospective_db is the list of created intentions (mocked)
        assert len(prospective_db) == 0


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


# ============================================================
# Test 7: H1 and H2 both fire on same PREDICTION_ERROR event
# ============================================================

class TestH1H2Combined:
    """Both handlers should fire on the same PE event emission."""

    def test_h1_and_h2_both_fire_on_same_event(self, monkeypatch, prospective_db):
        """H1 focuses spotlight AND H2 creates intention on single PE event."""
        monkeypatch.setenv("CODI_PE_ACTIONS", "on")

        from modules.workspace import _global_workspace
        from modules.pe_actions import pe_workspace_handler, pe_prospective_handler

        # Register both handlers (production order: H1 first, H2 second)
        event_bus.on(Events.PREDICTION_ERROR, pe_workspace_handler)
        event_bus.on(Events.PREDICTION_ERROR, pe_prospective_handler)

        try:
            # Emit via event bus (production path)
            event_bus.emit(Events.PREDICTION_ERROR, {
                "topic": "combined_test_topic",
                "intensity": "high",
                "confidence": 0.9,
            })

            # H1: spotlight should be updated
            spotlight = _global_workspace.get("spotlight", [])
            assert any("combined_test_topic" in str(item) for item in spotlight), \
                "H1 should have focused spotlight"

            # H2: intention should be created (was blocked by H1 rate limit before fix)
            assert len(prospective_db) == 1, \
                f"H2 should have created intention, got {len(prospective_db)}"
            assert "combined_test_topic" in prospective_db[0].get("action", "")
        finally:
            event_bus.off(Events.PREDICTION_ERROR, pe_workspace_handler)
            event_bus.off(Events.PREDICTION_ERROR, pe_prospective_handler)
