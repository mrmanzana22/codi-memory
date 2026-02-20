"""
Tests for backfill.py (E5.2)
================================
Covers: dry-run, live mode, batch processing, idempotency, registry.
"""

import os
import sqlite3

import pytest

from modules.migrations import apply_migrations


# ============================================================
# FIXTURES
# ============================================================

@pytest.fixture
def backfill_db(tmp_path):
    """Isolated SQLite DB with all migrations applied."""
    db_path = str(tmp_path / "test_fts.db")
    migrations_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "migrations"
    )
    apply_migrations(db_path, migrations_dir)
    return db_path


@pytest.fixture
def backfill_db_with_data(backfill_db):
    """DB with sample tool_calls data for testing backfills."""
    conn = sqlite3.connect(backfill_db)
    # Insert some rows with NULL tag (for example backfill)
    for i in range(25):
        conn.execute(
            "INSERT INTO tool_calls (tool_name, started_at, duration_ms, success, tag) "
            "VALUES (?, ?, ?, 1, ?)",
            (f"test_tool_{i}", f"2026-02-16T12:{i:02d}:00", i * 100,
             None if i < 15 else f"existing_tag_{i}")
        )
    conn.commit()
    conn.close()
    return backfill_db


# ============================================================
# REGISTRY
# ============================================================

class TestRegistry:
    def test_registry_has_entries(self):
        # Import triggers registration
        import importlib
        import scripts.backfill as bf
        importlib.reload(bf)

        assert len(bf.BACKFILL_REGISTRY) >= 2
        assert "003_write_queue_noop" in bf.BACKFILL_REGISTRY
        assert "example_add_column" in bf.BACKFILL_REGISTRY

    def test_registry_entries_have_required_fields(self):
        import scripts.backfill as bf

        for name, entry in bf.BACKFILL_REGISTRY.items():
            assert "func" in entry, f"{name} missing func"
            assert "description" in entry, f"{name} missing description"
            assert "db_key" in entry, f"{name} missing db_key"
            assert callable(entry["func"]), f"{name} func not callable"


# ============================================================
# NOOP BACKFILL (003)
# ============================================================

class TestNoopBackfill:
    def test_003_noop_dry_run(self, backfill_db):
        from scripts.backfill import run_backfill

        result = run_backfill("003_write_queue_noop", dry_run=True, db_path=backfill_db)
        assert result["status"] == "noop"
        assert "write_queue_rows" in result

    def test_003_noop_live(self, backfill_db):
        from scripts.backfill import run_backfill

        result = run_backfill("003_write_queue_noop", dry_run=False, db_path=backfill_db)
        assert result["status"] == "noop"

    def test_003_detects_missing_table(self, tmp_path):
        """If migration not applied, noop backfill reports error."""
        from scripts.backfill import run_backfill

        # Create DB without migration 003
        db_path = str(tmp_path / "bare.db")
        conn = sqlite3.connect(db_path)
        conn.execute("CREATE TABLE dummy (id INTEGER)")
        conn.close()

        result = run_backfill("003_write_queue_noop", dry_run=True, db_path=db_path)
        assert result["status"] == "error"
        assert "missing" in result["message"].lower()


# ============================================================
# EXAMPLE BACKFILL (batched)
# ============================================================

class TestExampleBackfill:
    def test_dry_run_shows_count(self, backfill_db_with_data):
        from scripts.backfill import run_backfill

        result = run_backfill("example_add_column", dry_run=True, db_path=backfill_db_with_data)
        assert result["status"] == "dry_run"
        assert result["rows_needing_backfill"] == 15  # 15 rows with NULL tag

    def test_live_processes_all_rows(self, backfill_db_with_data):
        from scripts.backfill import run_backfill

        result = run_backfill("example_add_column", dry_run=False, batch_size=5,
                              db_path=backfill_db_with_data)
        assert result["status"] == "done"
        assert result["processed"] == 15

        # Verify all NULL tags are now filled
        conn = sqlite3.connect(backfill_db_with_data)
        null_count = conn.execute("SELECT count(*) FROM tool_calls WHERE tag IS NULL").fetchone()[0]
        conn.close()
        assert null_count == 0

    def test_idempotent_second_run(self, backfill_db_with_data):
        """Running backfill twice should be safe (second run = noop)."""
        from scripts.backfill import run_backfill

        run_backfill("example_add_column", dry_run=False, db_path=backfill_db_with_data)
        result2 = run_backfill("example_add_column", dry_run=False, db_path=backfill_db_with_data)

        assert result2["status"] == "noop"
        assert result2["processed"] == 0

    def test_batch_size_respected(self, backfill_db_with_data):
        """With batch_size=5, processes 5 rows per iteration."""
        from scripts.backfill import run_backfill

        result = run_backfill("example_add_column", dry_run=False, batch_size=5,
                              db_path=backfill_db_with_data)
        assert result["status"] == "done"
        # 15 rows / 5 per batch = 3 batches, all processed
        assert result["processed"] == 15

    def test_noop_when_nothing_to_do(self, backfill_db):
        """Empty table -> nothing to backfill."""
        from scripts.backfill import run_backfill

        result = run_backfill("example_add_column", dry_run=False, db_path=backfill_db)
        assert result["status"] == "noop"


# ============================================================
# ERROR HANDLING
# ============================================================

class TestErrors:
    def test_unknown_backfill(self, backfill_db):
        from scripts.backfill import run_backfill

        result = run_backfill("nonexistent", db_path=backfill_db)
        assert result["status"] == "error"
        assert "Unknown" in result["message"]

    def test_missing_db(self, tmp_path):
        from scripts.backfill import run_backfill

        result = run_backfill("003_write_queue_noop", db_path=str(tmp_path / "nope.db"))
        assert result["status"] == "error"
