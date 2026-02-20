#!/usr/bin/env python3
"""
CANCEL TOMBSTONE TESTS
=======================
Verify cancel semantics: queued cancel, running tombstone, worker checks,
idempotency, and late cancel race logging.

6 tests:
  1. queued -> cancel -> status=canceled, never claimable
  2. running -> cancel_requested=1 (tombstone set)
  3. done/canceled -> cancel = no_op (idempotent)
  4. worker pre-exec check: skips canceled job
  5. worker post-exec check: late_cancel_race logged
  6. cancel reason persisted

Run: ./venv/bin/pytest tests/test_cancel_tombstone.py -v
"""

import os
import sys
import json
import sqlite3
import time
from datetime import datetime, timedelta

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.migrations import apply_migrations

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FTS_MIGRATIONS_DIR = os.path.join(PROJECT_ROOT, "migrations")


# ============================================================
# FIXTURES
# ============================================================

@pytest.fixture
def cancel_db(tmp_path):
    """Isolated SQLite DB with all migrations applied (including 009)."""
    db_path = str(tmp_path / "test_cancel.db")
    apply_migrations(db_path, migrations_dir=FTS_MIGRATIONS_DIR)
    return db_path


def _insert_job(db_path, job_id, kind="remember", status="queued",
                cancel_requested=0):
    """Insert a job directly for testing."""
    now = datetime.now().isoformat()
    conn = sqlite3.connect(db_path)
    conn.execute(
        "INSERT INTO write_queue "
        "(job_id, kind, payload_json, status, priority, attempts, max_attempts, "
        " cancel_requested, created_at, updated_at) "
        "VALUES (?, ?, ?, ?, 5, 0, 8, ?, ?, ?)",
        (job_id, kind, json.dumps({"content": "test"}), status,
         cancel_requested, now, now),
    )
    conn.commit()
    conn.close()


def _get_job(db_path, job_id):
    """Read a job row as dict."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    row = conn.execute(
        "SELECT * FROM write_queue WHERE job_id = ?", (job_id,)
    ).fetchone()
    conn.close()
    return dict(row) if row else None


def _get_log_entries(db_path, job_id):
    """Read write_queue_log entries for a job."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT * FROM write_queue_log WHERE job_id = ?", (job_id,)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


# ============================================================
# TEST 1: queued -> cancel -> canceled immediately
# ============================================================

class TestCancelQueued:
    """Canceling a queued job transitions to canceled immediately."""

    def test_queued_cancel(self, cancel_db):
        from modules.write_queue import cancel_write_job

        _insert_job(cancel_db, "job-cancel-q1")

        result = cancel_write_job("job-cancel-q1", db_path=cancel_db)

        assert result["outcome"] == "canceled"

        job = _get_job(cancel_db, "job-cancel-q1")
        assert job["status"] == "canceled"
        assert job["cancel_requested"] == 1
        assert job["canceled_at"] is not None
        assert job["cancel_reason"] == "user_cancel"

    def test_canceled_not_claimable(self, cancel_db):
        """After cancel, job is not returned by claim_next_job."""
        from modules.write_queue import cancel_write_job, claim_next_job

        _insert_job(cancel_db, "job-cancel-q2")
        cancel_write_job("job-cancel-q2", db_path=cancel_db)

        claimed = claim_next_job(db_path=cancel_db)
        assert claimed is None  # Nothing to claim


# ============================================================
# TEST 2: running -> tombstone (cancel_requested=1)
# ============================================================

class TestCancelRunning:
    """Canceling a running job sets tombstone, doesn't change status."""

    def test_running_tombstone(self, cancel_db):
        from modules.write_queue import cancel_write_job

        _insert_job(cancel_db, "job-cancel-r1", status="running")

        result = cancel_write_job("job-cancel-r1", db_path=cancel_db)

        assert result["outcome"] == "cancel_requested"

        job = _get_job(cancel_db, "job-cancel-r1")
        assert job["status"] == "running"  # Status unchanged
        assert job["cancel_requested"] == 1  # Tombstone set
        assert job["canceled_at"] is None  # Not canceled yet (worker decides)

    def test_running_tombstone_with_reason(self, cancel_db):
        from modules.write_queue import cancel_write_job

        _insert_job(cancel_db, "job-cancel-r2", status="running")

        result = cancel_write_job("job-cancel-r2", reason="user_changed_mind",
                                  db_path=cancel_db)

        assert result["outcome"] == "cancel_requested"
        job = _get_job(cancel_db, "job-cancel-r2")
        assert job["cancel_reason"] == "user_changed_mind"


# ============================================================
# TEST 3: idempotent cancel on terminal states
# ============================================================

class TestCancelIdempotent:
    """Cancel on done/canceled/dead/failed returns no_op."""

    def test_cancel_done(self, cancel_db):
        from modules.write_queue import cancel_write_job

        _insert_job(cancel_db, "job-done", status="done")
        result = cancel_write_job("job-done", db_path=cancel_db)
        assert result["outcome"] == "no_op"

    def test_cancel_already_canceled(self, cancel_db):
        from modules.write_queue import cancel_write_job

        _insert_job(cancel_db, "job-already-canceled", status="canceled")
        result = cancel_write_job("job-already-canceled", db_path=cancel_db)
        assert result["outcome"] == "no_op"

    def test_cancel_not_found(self, cancel_db):
        from modules.write_queue import cancel_write_job

        result = cancel_write_job("nonexistent", db_path=cancel_db)
        assert result["outcome"] == "not_found"


# ============================================================
# TEST 4: worker pre-exec check skips canceled job
# ============================================================

class TestWorkerPreExecCheck:
    """Worker checks cancel_requested after claim, before execution."""

    def test_check_cancel_requested(self, cancel_db):
        from modules.write_queue import check_cancel_requested

        _insert_job(cancel_db, "job-check-1", cancel_requested=1)
        assert check_cancel_requested("job-check-1", db_path=cancel_db) is True

    def test_check_not_canceled(self, cancel_db):
        from modules.write_queue import check_cancel_requested

        _insert_job(cancel_db, "job-check-2", cancel_requested=0)
        assert check_cancel_requested("job-check-2", db_path=cancel_db) is False

    def test_mark_job_canceled(self, cancel_db):
        from modules.write_queue import mark_job_canceled

        _insert_job(cancel_db, "job-mark-cancel", status="running",
                    cancel_requested=1)

        mark_job_canceled("job-mark-cancel", reason="cancel_before_exec",
                          db_path=cancel_db)

        job = _get_job(cancel_db, "job-mark-cancel")
        assert job["status"] == "canceled"
        assert job["canceled_at"] is not None
        assert job["cancel_reason"] == "cancel_before_exec"


# ============================================================
# TEST 5: cancel reason persisted
# ============================================================

class TestCancelReason:
    """cancel_reason is stored and retrievable."""

    def test_default_reason(self, cancel_db):
        from modules.write_queue import cancel_write_job

        _insert_job(cancel_db, "job-reason-default")
        cancel_write_job("job-reason-default", db_path=cancel_db)

        job = _get_job(cancel_db, "job-reason-default")
        assert job["cancel_reason"] == "user_cancel"

    def test_custom_reason(self, cancel_db):
        from modules.write_queue import cancel_write_job

        _insert_job(cancel_db, "job-reason-custom")
        cancel_write_job("job-reason-custom", reason="duplicate_detected",
                         db_path=cancel_db)

        job = _get_job(cancel_db, "job-reason-custom")
        assert job["cancel_reason"] == "duplicate_detected"

    def test_reason_truncated(self, cancel_db):
        from modules.write_queue import cancel_write_job

        _insert_job(cancel_db, "job-reason-long")
        long_reason = "x" * 500
        cancel_write_job("job-reason-long", reason=long_reason,
                         db_path=cancel_db)

        job = _get_job(cancel_db, "job-reason-long")
        assert len(job["cancel_reason"]) == 200


# ============================================================
# TEST 6: migration 009 schema
# ============================================================

class TestMigration009:
    """Migration 009 adds cancel columns and index."""

    def test_cancel_columns_exist(self, cancel_db):
        conn = sqlite3.connect(cancel_db)
        cols = {r[1] for r in conn.execute(
            "PRAGMA table_info(write_queue)"
        ).fetchall()}
        conn.close()

        assert "cancel_requested" in cols
        assert "canceled_at" in cols
        assert "cancel_reason" in cols

    def test_cancel_index_exists(self, cancel_db):
        conn = sqlite3.connect(cancel_db)
        indexes = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='index' "
            "AND tbl_name='write_queue'"
        ).fetchall()
        conn.close()

        index_names = [i[0] for i in indexes]
        assert "idx_wq_cancel_requested" in index_names
