"""
Shared test fixtures for codi-memory test suite.
Ensures test isolation: no test touches production DBs or leaks event state.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

import pytest
from modules.events import event_bus


@pytest.fixture(autouse=True)
def _isolate_sqlite(tmp_path, monkeypatch):
    """Force all SQLite writes (FTS, event_counts, prospective) to a temp directory.

    Prevents tests from writing to the real DBs in the repo root.
    Scoped per-test via monkeypatch (auto-restored after each test).
    Runs migrations on isolated DBs so all tables exist.
    """
    db_path = str(tmp_path / "memories_fts.db")
    prosp_path = str(tmp_path / "prospective.db")

    # Env vars
    monkeypatch.setenv("FTS_DB_PATH", db_path)
    monkeypatch.setenv("PROSPECTIVE_DB_PATH", prosp_path)

    # Bypass Qdrant auth guardrail in tests (tests mock Qdrant, never connect)
    monkeypatch.setenv("CODI_ALLOW_INSECURE_QDRANT", "1")

    # Module-level config patches
    monkeypatch.setattr("modules.config.FTS_DB_PATH", db_path, raising=False)
    monkeypatch.setattr("modules.config.PROSPECTIVE_DB_PATH", prosp_path, raising=False)

    # Run migrations on isolated DBs
    from modules.migrations import apply_migrations
    apply_migrations(db_path, migrations_dir=os.path.join(PROJECT_ROOT, "migrations"))
    apply_migrations(prosp_path, migrations_dir=os.path.join(PROJECT_ROOT, "migrations_prospective"))

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
