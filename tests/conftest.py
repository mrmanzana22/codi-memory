"""
Shared test fixtures for codi-memory test suite.
Ensures test isolation: no test touches production DBs or leaks event state.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from modules.events import event_bus


@pytest.fixture(autouse=True)
def _isolate_sqlite(tmp_path, monkeypatch):
    """Force all SQLite writes (FTS, event_counts) to a temp directory.

    Prevents tests from writing to the real memories_fts.db in the repo root.
    Scoped per-test via monkeypatch (auto-restored after each test).
    """
    monkeypatch.setenv("FTS_DB_PATH", str(tmp_path / "memories_fts.db"))
    yield


@pytest.fixture
def clean_event_bus():
    """Provide a clean event bus for tests that inspect event history.

    Saves and restores _history + _dirty_counts even if the test fails.
    Usage: def test_x(self, clean_event_bus): ...
    """
    old_history = event_bus._history[:]
    old_dirty = dict(event_bus._dirty_counts)
    old_total = event_bus._dirty_total
    event_bus._history.clear()
    event_bus._dirty_counts.clear()
    event_bus._dirty_total = 0
    yield
    event_bus._history = old_history
    event_bus._dirty_counts.update(old_dirty)
    event_bus._dirty_total = old_total
