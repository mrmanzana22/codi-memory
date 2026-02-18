#!/usr/bin/env python3
"""
DUAL SHADOW WIRE TESTS
======================
Verify that shadow mode creates dual_compare_log rows for each tool kind.

3 tests:
  1. remember shadow -> dual_compare_log row with kind=remember
  2. add_memory shadow -> dual_compare_log row with kind=add_memory
  3. checkpoint_memoria shadow -> dual_compare_log row with kind=checkpoint_memoria

Each test verifies: trace_id set, async_job_id set, compare_status=pending.
These protect the wire: if someone reorders enqueue/record, the link breaks.

Run: ./venv/bin/pytest tests/test_dual_shadow_wire.py -v
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import sqlite3
import pytest

from modules.migrations import apply_migrations

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FTS_MIGRATIONS_DIR = os.path.join(PROJECT_ROOT, "migrations")


@pytest.fixture
def shadow_db(tmp_path, monkeypatch):
    """Create a DB with all migrations, set shadow mode, mock externals."""
    db_path = str(tmp_path / "shadow_test.db")
    apply_migrations(db_path, migrations_dir=FTS_MIGRATIONS_DIR)

    # Set shadow mode
    monkeypatch.setenv("CODI_WRITE_MODE", "shadow")

    # Point config to test DB
    monkeypatch.setattr("modules.config.FTS_DB_PATH", db_path)

    # FTS_DB_PATH is imported by value at module level in write_queue and
    # dual_compare.  If those modules are already loaded, their local copy
    # still points to the original path.  Patch them too.
    import modules.write_queue
    monkeypatch.setattr(modules.write_queue, "FTS_DB_PATH", db_path, raising=False)
    import modules.dual_compare
    monkeypatch.setattr(modules.dual_compare, "FTS_DB_PATH", db_path)

    # write_queue._get_conn() delegates to db_pool; patch pool path too
    import modules.db_pool
    monkeypatch.setattr(modules.db_pool, "FTS_DB_PATH", db_path)
    modules.db_pool.close_thread_connections()  # force pool to pick up new path

    # Set a deterministic trace_id
    from modules.tracing import set_trace_id
    set_trace_id("test_trace_shadow")

    return db_path


def _get_dual_rows(db_path: str) -> list:
    """Read all rows from dual_compare_log."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    rows = conn.execute("SELECT * FROM dual_compare_log").fetchall()
    conn.close()
    return [dict(r) for r in rows]


def _get_queue_rows(db_path: str) -> list:
    """Read all rows from write_queue."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    rows = conn.execute("SELECT * FROM write_queue").fetchall()
    conn.close()
    return [dict(r) for r in rows]


# ============================================================
# Test 1: remember shadow creates dual_compare_log row
# ============================================================

class TestRememberShadowWire:
    """remember in shadow mode -> dual_compare_log row."""

    def test_remember_shadow_creates_dual_row(self, shadow_db, monkeypatch):
        """Shadow remember: sync executes + enqueues + records dual compare."""
        # Mock add_memory_smart to avoid real Qdrant/mem0 calls
        mock_result = json.dumps({
            "action": "saved_new",
            "new_id": "mock-mem-001",
            "message": "Mock saved"
        })
        monkeypatch.setattr(
            "modules.interface.add_memory_smart",
            lambda **kwargs: mock_result,
        )
        # Mock working memory to avoid its DB infrastructure
        monkeypatch.setattr(
            "modules.interface.push_to_working_memory",
            lambda **kwargs: json.dumps({"pretty": "mocked", "items": []}),
        )
        # remember() calls new_trace_id() at the top, overwriting our fixture.
        # Neutralize it so the deterministic trace_id survives.
        monkeypatch.setattr(
            "modules.interface.new_trace_id",
            lambda: "test_trace_shadow",
        )

        from modules.interface import remember
        result = remember(
            content="test shadow remember content",
            importance="medium",
            topic="test",
            source="interaction",
            long_term=True,
        )

        # Verify dual_compare_log has a row
        dual_rows = _get_dual_rows(shadow_db)
        assert len(dual_rows) == 1, f"Expected 1 dual row, got {len(dual_rows)}"

        row = dual_rows[0]
        assert row["kind"] == "remember"
        assert row["trace_id"] == "test_trace_shadow"
        assert row["async_job_id"] is not None
        assert row["compare_status"] == "pending"
        assert row["sync_result_json"] is not None

        # Verify write_queue also has a row
        queue_rows = _get_queue_rows(shadow_db)
        assert len(queue_rows) == 1
        assert queue_rows[0]["kind"] == "remember"

        # The job_ids must match
        assert row["async_job_id"] == queue_rows[0]["job_id"]


# ============================================================
# Test 2: add_memory shadow creates dual_compare_log row
# ============================================================

class TestAddMemoryShadowWire:
    """add_memory in shadow mode -> dual_compare_log row."""

    def test_add_memory_shadow_creates_dual_row(self, shadow_db, monkeypatch):
        """Shadow add_memory: sync + enqueue + dual record."""
        mock_result = "Memoria guardada: test content"
        monkeypatch.setattr(
            "modules.memory_core._add_memory_sync",
            lambda *args, **kwargs: mock_result,
        )

        from modules.memory_core import add_memory
        result = add_memory(
            content="test shadow add_memory content",
            category="test",
            source="experienced",
            importance="medium",
        )

        dual_rows = _get_dual_rows(shadow_db)
        assert len(dual_rows) == 1

        row = dual_rows[0]
        assert row["kind"] == "add_memory"
        assert row["trace_id"] == "test_trace_shadow"
        assert row["async_job_id"] is not None
        assert row["compare_status"] == "pending"

        queue_rows = _get_queue_rows(shadow_db)
        assert len(queue_rows) == 1
        assert row["async_job_id"] == queue_rows[0]["job_id"]


# ============================================================
# Test 3: checkpoint_memoria shadow creates dual_compare_log row
# ============================================================

class TestCheckpointShadowWire:
    """checkpoint_memoria in shadow mode -> dual_compare_log row."""

    def test_checkpoint_shadow_creates_dual_row(self, shadow_db, monkeypatch):
        """Shadow checkpoint: sync + enqueue + dual record."""
        mock_result = json.dumps({
            "message": "Checkpoint guardado",
            "memory_id": "mock-cp-001",
        })
        monkeypatch.setattr(
            "modules.flush._checkpoint_memoria_sync",
            lambda *args, **kwargs: mock_result,
        )

        from modules.flush import _checkpoint_memoria
        result = _checkpoint_memoria(
            momento="test_moment",
            que_paso="test que paso",
            por_que_importa="test importancia",
        )

        dual_rows = _get_dual_rows(shadow_db)
        assert len(dual_rows) == 1

        row = dual_rows[0]
        assert row["kind"] == "checkpoint_memoria"
        assert row["trace_id"] == "test_trace_shadow"
        assert row["async_job_id"] is not None
        assert row["compare_status"] == "pending"

        queue_rows = _get_queue_rows(shadow_db)
        assert len(queue_rows) == 1
        assert row["async_job_id"] == queue_rows[0]["job_id"]


# ============================================================
# Test 4+5: trace_id fallback (no trace set before shadow)
# ============================================================

class TestTraceIdFallback:
    """Shadow paths generate a trace_id when none exists."""

    def test_add_memory_generates_trace_when_empty(self, shadow_db, monkeypatch):
        """add_memory shadow: creates trace_id via new_trace_id() fallback."""
        from modules.tracing import set_trace_id
        set_trace_id("")  # clear any existing trace

        monkeypatch.setattr(
            "modules.memory_core._add_memory_sync",
            lambda *args, **kwargs: "Memoria guardada: fallback test",
        )

        from modules.memory_core import add_memory
        add_memory(content="trace fallback test", category="test",
                   source="experienced", importance="medium")

        dual_rows = _get_dual_rows(shadow_db)
        assert len(dual_rows) == 1
        row = dual_rows[0]
        assert row["trace_id"] != ""
        assert len(row["trace_id"]) >= 8  # new_trace_id format

    def test_checkpoint_generates_trace_when_empty(self, shadow_db, monkeypatch):
        """checkpoint shadow: creates trace_id via new_trace_id() fallback."""
        from modules.tracing import set_trace_id
        set_trace_id("")  # clear any existing trace

        monkeypatch.setattr(
            "modules.flush._checkpoint_memoria_sync",
            lambda *args, **kwargs: '{"message": "ok"}',
        )

        from modules.flush import _checkpoint_memoria
        _checkpoint_memoria(momento="test", que_paso="fallback test",
                            por_que_importa="trace coverage")

        dual_rows = _get_dual_rows(shadow_db)
        assert len(dual_rows) == 1
        row = dual_rows[0]
        assert row["trace_id"] != ""
        assert len(row["trace_id"]) >= 8
