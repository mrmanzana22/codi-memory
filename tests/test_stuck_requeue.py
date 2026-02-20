"""
Tests for reap_stuck_jobs() (stuck job reaper)
================================================
Covers: requeue under max_attempts, dead on exhaustion,
ignore recent, started_at vs claimed_at distinction.

4 contracts per Hare's spec:
  1. test_reap_requeues_under_max_attempts
  2. test_reap_marks_dead_on_max_attempts
  3. test_reap_ignores_recent_processing
  4. test_reap_uses_started_at_not_claimed_at
"""

import os
import sqlite3
from datetime import datetime, timedelta

import pytest

from modules.migrations import apply_migrations


# ============================================================
# FIXTURES
# ============================================================

@pytest.fixture
def queue_db(tmp_path):
    """Isolated SQLite DB with all migrations applied."""
    db_path = str(tmp_path / "test_stuck.db")
    migrations_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "migrations"
    )
    apply_migrations(db_path, migrations_dir)
    return db_path


def _insert_running_job(
    db_path: str,
    job_id: str = "stuck-job-001",
    kind: str = "remember",
    attempts: int = 2,
    max_attempts: int = 8,
    claimed_at: str = None,
    started_at: str = None,
    lease_until: str = None,
):
    """Insert a job directly in 'running' state for testing."""
    now = datetime.now()
    conn = sqlite3.connect(db_path)
    conn.execute(
        "INSERT INTO write_queue "
        "(job_id, kind, payload_json, status, priority, attempts, max_attempts, "
        " claimed_at, started_at, lease_until, created_at, updated_at) "
        "VALUES (?, ?, '{}', 'running', 5, ?, ?, ?, ?, ?, ?, ?)",
        (
            job_id, kind, attempts, max_attempts,
            claimed_at or (now - timedelta(seconds=200)).isoformat(),
            started_at,
            lease_until or (now + timedelta(seconds=100)).isoformat(),
            (now - timedelta(seconds=300)).isoformat(),
            now.isoformat(),
        ),
    )
    conn.commit()
    conn.close()


def _get_job(db_path: str, job_id: str) -> dict:
    """Read a job row as dict."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    row = conn.execute(
        "SELECT * FROM write_queue WHERE job_id = ?", (job_id,)
    ).fetchone()
    conn.close()
    return dict(row) if row else None


def _get_log_entries(db_path: str, job_id: str) -> list:
    """Read write_queue_log entries for a job."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT * FROM write_queue_log WHERE job_id = ?", (job_id,)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


# ============================================================
# TESTS
# ============================================================

class TestReapStuckJobs:

    def test_reap_requeues_under_max_attempts(self, queue_db):
        """Job stuck >180s with attempts < max_attempts -> requeued."""
        from modules.write_queue import reap_stuck_jobs

        old_started = (datetime.now() - timedelta(seconds=240)).isoformat()
        _insert_running_job(
            queue_db, job_id="stuck-requeue",
            attempts=2, max_attempts=8,
            started_at=old_started,
        )

        result = reap_stuck_jobs(max_exec_seconds=180, db_path=queue_db)

        assert result["requeued"] == 1
        assert result["dead"] == 0

        job = _get_job(queue_db, "stuck-requeue")
        assert job["status"] == "queued"
        assert job["failure_reason"] == "timeout_requeued"
        assert job["claimed_at"] is None
        assert job["started_at"] is None

        logs = _get_log_entries(queue_db, "stuck-requeue")
        assert len(logs) == 1
        assert logs[0]["status"] == "requeued"
        assert logs[0]["error_class"] == "StuckReaper"

    def test_reap_marks_dead_on_max_attempts(self, queue_db):
        """Job stuck >180s with attempts >= max_attempts -> dead."""
        from modules.write_queue import reap_stuck_jobs

        old_started = (datetime.now() - timedelta(seconds=300)).isoformat()
        _insert_running_job(
            queue_db, job_id="stuck-dead",
            attempts=8, max_attempts=8,
            started_at=old_started,
        )

        result = reap_stuck_jobs(max_exec_seconds=180, db_path=queue_db)

        assert result["requeued"] == 0
        assert result["dead"] == 1

        job = _get_job(queue_db, "stuck-dead")
        assert job["status"] == "dead"
        assert job["failure_reason"] == "timeout_dead"

        logs = _get_log_entries(queue_db, "stuck-dead")
        assert len(logs) == 1
        assert logs[0]["status"] == "dead"

    def test_reap_ignores_recent_processing(self, queue_db):
        """Job started <180s ago -> NOT touched."""
        from modules.write_queue import reap_stuck_jobs

        recent_started = (datetime.now() - timedelta(seconds=60)).isoformat()
        _insert_running_job(
            queue_db, job_id="recent-job",
            attempts=1, max_attempts=8,
            started_at=recent_started,
        )

        result = reap_stuck_jobs(max_exec_seconds=180, db_path=queue_db)

        assert result["requeued"] == 0
        assert result["dead"] == 0

        job = _get_job(queue_db, "recent-job")
        assert job["status"] == "running"

    def test_reap_uses_started_at_not_claimed_at(self, queue_db):
        """Job with old claimed_at but recent started_at -> NOT reaped.

        This is the key distinction: reap uses started_at (execution time),
        not claimed_at (lease time). A job can be claimed long ago but
        started recently and still be legitimately executing.
        """
        from modules.write_queue import reap_stuck_jobs

        old_claimed = (datetime.now() - timedelta(seconds=600)).isoformat()
        recent_started = (datetime.now() - timedelta(seconds=30)).isoformat()

        _insert_running_job(
            queue_db, job_id="old-claim-new-start",
            attempts=3, max_attempts=8,
            claimed_at=old_claimed,
            started_at=recent_started,
        )

        result = reap_stuck_jobs(max_exec_seconds=180, db_path=queue_db)

        assert result["requeued"] == 0
        assert result["dead"] == 0

        job = _get_job(queue_db, "old-claim-new-start")
        assert job["status"] == "running"  # untouched
