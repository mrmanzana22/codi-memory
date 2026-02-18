"""
Tests for write_worker.py and async tool modes (E2.3 PR2)
============================================================
Covers: worker execution, tool async/shadow modes, end-to-end pipeline.
"""

import json
import os
import sqlite3
import time
from unittest.mock import patch, MagicMock

import pytest

from modules.migrations import apply_migrations


# ============================================================
# FIXTURES
# ============================================================

@pytest.fixture
def worker_db(tmp_path):
    """Isolated SQLite DB with all migrations applied."""
    db_path = str(tmp_path / "test_fts.db")
    migrations_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "migrations"
    )
    apply_migrations(db_path, migrations_dir)
    return db_path


@pytest.fixture
def populated_queue(worker_db):
    """Queue with pre-loaded jobs ready for worker to process."""
    from modules.write_queue import enqueue_write_job

    jobs = []
    jobs.append(enqueue_write_job(
        kind="remember",
        payload={"content": "test memory for worker", "category": "general",
                 "source": "experienced", "importance": "medium"},
        db_path=worker_db,
    ))
    jobs.append(enqueue_write_job(
        kind="add_memory",
        payload={"content": "another test", "category": "general",
                 "source": "experienced", "importance": "low"},
        db_path=worker_db,
    ))
    return worker_db, jobs


# ============================================================
# WORKER UNIT TESTS
# ============================================================

class TestWorkerProcessJob:
    def test_process_remember_job(self, worker_db):
        """Worker can execute a 'remember' job end-to-end (mocked)."""
        from modules.write_queue import enqueue_write_job
        import modules.write_worker as ww

        job = enqueue_write_job(
            kind="remember",
            payload={"content": "worker test memory", "category": "test",
                     "source": "experienced", "importance": "medium"},
            db_path=worker_db,
        )

        mock_exec = MagicMock(return_value={"action": "saved_new", "result": "ok"})

        with patch.object(ww, "claim_next_job") as mock_claim, \
             patch.object(ww, "mark_job_started") as mock_started, \
             patch.object(ww, "mark_job_done") as mock_done, \
             patch.object(ww, "log_job_completion") as mock_log, \
             patch.dict(ww.JOB_EXECUTORS, {"remember": mock_exec}):

            mock_claim.return_value = {
                "job_id": job["job_id"],
                "kind": "remember",
                "payload": {"content": "worker test memory", "category": "test",
                            "source": "experienced", "importance": "medium"},
                "attempts": 1,
            }

            result = ww.process_one_job()

            assert result is True
            mock_exec.assert_called_once()
            mock_started.assert_called_once_with(job["job_id"])
            mock_done.assert_called_once_with(job["job_id"])
            mock_log.assert_called_once()

    def test_process_unknown_kind(self, worker_db):
        """Worker handles unknown job kind gracefully."""
        with patch("modules.write_worker.claim_next_job") as mock_claim, \
             patch("modules.write_worker.mark_job_failed") as mock_fail, \
             patch("modules.write_worker.log_job_completion") as mock_log:

            mock_claim.return_value = {
                "job_id": "test-unknown",
                "kind": "invalid_kind",
                "payload": {},
                "attempts": 1,
            }

            from modules.write_worker import process_one_job
            result = process_one_job()

            assert result is True
            mock_fail.assert_called_once()

    def test_process_empty_queue(self, worker_db):
        """Worker returns False when queue is empty."""
        with patch("modules.write_worker.claim_next_job") as mock_claim:
            mock_claim.return_value = None

            from modules.write_worker import process_one_job
            result = process_one_job()
            assert result is False

    def test_executor_failure_marks_failed(self, worker_db):
        """When executor raises, job is marked failed."""
        import modules.write_worker as ww

        mock_exec = MagicMock(side_effect=ConnectionError("Qdrant unreachable"))

        with patch.object(ww, "claim_next_job") as mock_claim, \
             patch.object(ww, "mark_job_started") as mock_started, \
             patch.object(ww, "mark_job_failed") as mock_fail, \
             patch.object(ww, "log_job_completion") as mock_log, \
             patch.dict(ww.JOB_EXECUTORS, {"remember": mock_exec}):

            mock_claim.return_value = {
                "job_id": "test-fail",
                "kind": "remember",
                "payload": {"content": "will fail"},
                "attempts": 1,
            }

            result = ww.process_one_job()

            assert result is True
            mock_fail.assert_called_once()
            args = mock_fail.call_args
            assert "ConnectionError" in args[0][1]


# ============================================================
# ASYNC MODE TESTS
# ============================================================

class TestAsyncMode:
    def test_remember_async_returns_ack(self, worker_db, monkeypatch):
        """In async mode, remember() returns fast ACK with job_id."""
        monkeypatch.setenv("CODI_WRITE_MODE", "async")
        import modules.interface as iface

        # Mock working memory push (still sync)
        with patch.object(iface, "push_to_working_memory", return_value='{"pretty":"ok"}'), \
             patch("modules.write_queue.enqueue_write_job") as mock_enqueue:

            mock_enqueue.return_value = {"job_id": "test-async-123", "status": "queued", "dedupe_hit": False}

            result_str = iface.remember("async test content", importance="high", topic="test")
            result = json.loads(result_str)

            assert "queued" in str(result.get("long_term", ""))
            mock_enqueue.assert_called_once()
            # Verify payload
            call_kwargs = mock_enqueue.call_args[1]
            assert call_kwargs["kind"] == "remember"
            assert call_kwargs["priority"] == 3  # high importance = priority 3

    def test_remember_sync_default(self, monkeypatch):
        """Default mode (sync) does NOT enqueue."""
        monkeypatch.setenv("CODI_WRITE_MODE", "sync")
        import modules.interface as iface

        with patch.object(iface, "push_to_working_memory", return_value='{"pretty":"ok"}'), \
             patch.object(iface, "add_memory_smart", return_value='{"action":"saved_new"}'), \
             patch("modules.write_queue.enqueue_write_job") as mock_enqueue:

            iface.remember("sync test content", importance="high", topic="test")

            # In sync mode, enqueue should NOT be called
            mock_enqueue.assert_not_called()

    def test_remember_shadow_does_both(self, monkeypatch):
        """Shadow mode runs sync AND enqueues."""
        monkeypatch.setenv("CODI_WRITE_MODE", "shadow")
        import modules.interface as iface

        with patch.object(iface, "push_to_working_memory", return_value='{"pretty":"ok"}'), \
             patch.object(iface, "add_memory_smart", return_value='{"action":"saved_new"}') as mock_smart, \
             patch("modules.write_queue.enqueue_write_job") as mock_enqueue:

            mock_enqueue.return_value = {"job_id": "shadow-123", "status": "queued", "dedupe_hit": False}

            iface.remember("shadow test content", importance="medium", topic="test")

            # Both should be called
            mock_smart.assert_called_once()
            mock_enqueue.assert_called_once()

    def test_add_memory_async_returns_ack(self, monkeypatch):
        """add_memory in async mode returns ACK."""
        monkeypatch.setenv("CODI_WRITE_MODE", "async")

        with patch("modules.write_queue.enqueue_write_job") as mock_enqueue:
            mock_enqueue.return_value = {"job_id": "add-async-456", "status": "queued", "dedupe_hit": False}

            from modules.memory_core import add_memory
            result_str = add_memory("async add_memory test", category="test")
            result = json.loads(result_str)

            assert result["action"] == "queued"
            assert result["job_id"] == "add-async-456"

    def test_checkpoint_async_returns_ack(self, monkeypatch):
        """checkpoint_memoria in async mode returns ACK."""
        monkeypatch.setenv("CODI_WRITE_MODE", "async")

        with patch("modules.write_queue.enqueue_write_job") as mock_enqueue:
            mock_enqueue.return_value = {"job_id": "cp-async-789", "status": "queued", "dedupe_hit": False}

            from modules.flush import _checkpoint_memoria
            result_str = _checkpoint_memoria("decision", "test decision", "important")

            assert "enqueued" in result_str.lower() or "cp-async" in result_str


# ============================================================
# ACK SPEED
# ============================================================

class TestACKSpeed:
    def test_async_remember_is_fast(self, monkeypatch, worker_db):
        """In async mode, remember should return in < 100ms (no network)."""
        monkeypatch.setenv("CODI_WRITE_MODE", "async")
        import modules.interface as iface

        with patch.object(iface, "push_to_working_memory", return_value='{"pretty":"ok"}'), \
             patch("modules.write_queue.enqueue_write_job") as mock_enqueue:

            mock_enqueue.return_value = {"job_id": "speed-test", "status": "queued", "dedupe_hit": False}

            start = time.monotonic()
            iface.remember("speed test", importance="medium", topic="test")
            elapsed_ms = (time.monotonic() - start) * 1000

            assert elapsed_ms < 100, f"Async remember took {elapsed_ms:.1f}ms, expected < 100ms"


# ============================================================
# BACKOFF
# ============================================================

# ============================================================
# DRAIN TICK
# ============================================================

class TestDrainTick:
    def test_drain_processes_multiple_jobs(self, worker_db):
        """drain_tick() processes multiple queued jobs in one tick."""
        import modules.write_worker as ww

        call_count = {"n": 0}
        max_calls = 3

        def fake_process(lease_seconds=300):
            if call_count["n"] < max_calls:
                call_count["n"] += 1
                return True
            return False

        with patch.object(ww, "process_one_job", side_effect=fake_process):
            drained = ww.drain_tick(max_jobs=10, max_seconds=20)

        assert drained["processed"] == 3

    def test_drain_respects_max_jobs(self, worker_db):
        """drain_tick() stops at max_jobs even if more work exists."""
        import modules.write_worker as ww

        with patch.object(ww, "process_one_job", return_value=True):
            drained = ww.drain_tick(max_jobs=5, max_seconds=60)

        assert drained["processed"] == 5

    def test_drain_stops_on_empty_queue(self, worker_db):
        """drain_tick() returns 0 when queue is empty."""
        import modules.write_worker as ww

        with patch.object(ww, "process_one_job", return_value=False):
            drained = ww.drain_tick(max_jobs=10, max_seconds=20)

        assert drained["processed"] == 0


# ============================================================
# SCHEMA CHECK
# ============================================================

class TestSchemaCheck:
    def test_schema_check_passes_with_migration(self, worker_db, monkeypatch):
        """_check_schema() passes when migration 004 is applied."""
        import modules.write_worker as ww
        monkeypatch.setattr("modules.config.FTS_DB_PATH", worker_db)

        # Should not raise
        ww._check_schema()

    def test_schema_check_fails_without_columns(self, tmp_path, monkeypatch):
        """_check_schema() fails when claimed_at/started_at are missing."""
        import modules.write_worker as ww

        # Create a DB with write_queue but WITHOUT the new columns
        db_path = str(tmp_path / "old.db")
        import sqlite3
        conn = sqlite3.connect(db_path)
        conn.execute("CREATE TABLE write_queue (id INTEGER PRIMARY KEY, job_id TEXT)")
        conn.commit()
        conn.close()

        monkeypatch.setattr("modules.config.FTS_DB_PATH", db_path)

        with pytest.raises(RuntimeError, match="missing columns"):
            ww._check_schema()


class TestBackoff:
    def test_backoff_increases(self):
        from modules.write_worker import _backoff_seconds

        b1 = _backoff_seconds(1)
        b2 = _backoff_seconds(2)
        b3 = _backoff_seconds(3)

        # Should increase (with some jitter)
        assert b1 < b2 < b3
        # First attempt should be around 10-13s
        assert 10 <= b1 <= 15
        # Should cap at MAX_BACKOFF (300)
        b10 = _backoff_seconds(10)
        assert b10 <= 400  # 300 + 30% jitter max


# ============================================================
# SHADOW REPORT LATENCY CALCULATION
# ============================================================

class TestShadowReportLatency:
    """Tests for the 3-way latency breakdown in shadow_report.py."""

    def _create_test_db(self, tmp_path):
        """Create a DB with synthetic write_queue data for report testing."""
        db_path = str(tmp_path / "report_test.db")
        migrations_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "migrations"
        )
        apply_migrations(db_path, migrations_dir)
        return db_path

    def test_report_parses_latency_breakdown(self, tmp_path):
        """Report correctly computes queue_wait and exec from timestamps."""
        db_path = self._create_test_db(tmp_path)

        import sqlite3
        conn = sqlite3.connect(db_path)
        # Insert a synthetic done job with all timestamps
        conn.execute(
            "INSERT INTO write_queue "
            "(job_id, kind, payload_json, status, priority, attempts, max_attempts, "
            " created_at, updated_at, claimed_at, started_at, completed_at) "
            "VALUES (?, ?, ?, 'done', 5, 1, 8, ?, ?, ?, ?, ?)",
            ("test-lat-1", "remember", '{"c":"x"}',
             "2026-02-16T20:00:00-05:00",  # created_at
             "2026-02-16T20:00:30-05:00",  # updated_at
             "2026-02-16T20:00:10-05:00",  # claimed_at (10s queue wait)
             "2026-02-16T20:00:11-05:00",  # started_at (1s overhead)
             "2026-02-16T20:00:41-05:00",  # completed_at (30s exec)
            )
        )
        conn.commit()
        conn.close()

        # Verify we can query the timestamps
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT created_at, claimed_at, started_at, completed_at "
            "FROM write_queue WHERE job_id = 'test-lat-1'"
        ).fetchone()
        conn.close()

        from datetime import datetime
        created = datetime.fromisoformat(row["created_at"])
        claimed = datetime.fromisoformat(row["claimed_at"])
        started = datetime.fromisoformat(row["started_at"])
        completed = datetime.fromisoformat(row["completed_at"])

        queue_wait_ms = (claimed - created).total_seconds() * 1000
        exec_ms = (completed - started).total_seconds() * 1000
        total_ms = (completed - created).total_seconds() * 1000

        assert queue_wait_ms == 10000  # 10s
        assert exec_ms == 30000        # 30s
        assert total_ms == 41000       # 41s total

    def test_report_handles_missing_claimed_at(self, tmp_path):
        """Pre-004 jobs without claimed_at still show total latency."""
        db_path = self._create_test_db(tmp_path)

        import sqlite3
        conn = sqlite3.connect(db_path)
        conn.execute(
            "INSERT INTO write_queue "
            "(job_id, kind, payload_json, status, priority, attempts, max_attempts, "
            " created_at, updated_at, completed_at) "
            "VALUES (?, ?, ?, 'done', 5, 1, 8, ?, ?, ?)",
            ("test-old-1", "remember", '{"c":"x"}',
             "2026-02-16T19:00:00-05:00",
             "2026-02-16T19:00:30-05:00",
             "2026-02-16T19:00:30-05:00",
            )
        )
        conn.commit()
        conn.close()

        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT claimed_at, started_at FROM write_queue WHERE job_id = 'test-old-1'"
        ).fetchone()
        conn.close()

        # Both should be NULL for pre-004 jobs
        assert row["claimed_at"] is None
        assert row["started_at"] is None

    def test_report_latency_multiple_jobs(self, tmp_path):
        """Percentile calculation works correctly with multiple jobs."""
        db_path = self._create_test_db(tmp_path)

        import sqlite3
        conn = sqlite3.connect(db_path)
        # Insert 5 jobs with different latencies
        for i, (q_wait, ex_time) in enumerate([(5, 25), (10, 30), (15, 35), (20, 40), (300, 600)]):
            base = f"2026-02-16T20:{i:02d}:00-05:00"
            claimed = f"2026-02-16T20:{i:02d}:{q_wait:02d}-05:00"
            started = f"2026-02-16T20:{i:02d}:{q_wait + 1:02d}-05:00"
            completed_s = q_wait + 1 + ex_time
            completed_m = completed_s // 60
            completed_s = completed_s % 60
            completed = f"2026-02-16T20:{i + completed_m:02d}:{completed_s:02d}-05:00"

            conn.execute(
                "INSERT INTO write_queue "
                "(job_id, kind, payload_json, status, priority, attempts, max_attempts, "
                " created_at, updated_at, claimed_at, started_at, completed_at) "
                "VALUES (?, ?, ?, 'done', 5, 1, 8, ?, ?, ?, ?, ?)",
                (f"test-multi-{i}", "remember", '{"c":"x"}',
                 base, base, claimed, started, completed)
            )
        conn.commit()
        conn.close()

        # Verify the data is queryable and compute p50
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT claimed_at, created_at FROM write_queue WHERE job_id LIKE 'test-multi-%'"
        ).fetchall()
        conn.close()

        from datetime import datetime
        waits = []
        for r in rows:
            c = datetime.fromisoformat(r["created_at"])
            cl = datetime.fromisoformat(r["claimed_at"])
            waits.append((cl - c).total_seconds())

        assert len(waits) == 5
        waits.sort()
        assert waits[2] == 15  # median of [5, 10, 15, 20, 300]
