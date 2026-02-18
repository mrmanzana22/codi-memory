"""
Tests for write_queue.py (E2.3 PR1)
=====================================
Covers: enqueue, dedupe, claim, retry, dead letter, status, observability.

Based on Hare's spec (E2.3 v1):
  1. test_enqueue_dedupe_returns_same_job_id
  2. test_worker_claims_one_job_with_lease
  3. test_worker_retries_and_marks_dead
  4. test_idempotent_write_no_duplicate_memory (PR2 scope, placeholder)
  5. test_async_tool_returns_ack_fast (PR2 scope, placeholder)
"""

import json
import os
import sqlite3
import time

import pytest

from modules.migrations import apply_migrations


# ============================================================
# FIXTURES
# ============================================================

@pytest.fixture
def queue_db(tmp_path, monkeypatch):
    """Isolated SQLite DB with write_queue tables applied."""
    db_path = str(tmp_path / "test_fts.db")

    # Apply all migrations (001, 002, 003)
    migrations_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "migrations"
    )
    result = apply_migrations(db_path, migrations_dir)
    assert "003" in result["applied"] or "003" in result["skipped"]

    # Verify tables exist
    conn = sqlite3.connect(db_path)
    tables = [r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    ).fetchall()]
    assert "write_queue" in tables
    assert "write_queue_log" in tables
    conn.close()

    return db_path


# ============================================================
# ENQUEUE
# ============================================================

class TestEnqueue:
    def test_enqueue_basic(self, queue_db):
        from modules.write_queue import enqueue_write_job

        result = enqueue_write_job(
            kind="remember",
            payload={"content": "test memory", "topic": "general"},
            db_path=queue_db,
        )

        assert result["status"] == "queued"
        assert result["dedupe_hit"] is False
        assert len(result["job_id"]) == 36  # UUID format

    def test_enqueue_with_priority(self, queue_db):
        from modules.write_queue import enqueue_write_job, get_write_job_status

        result = enqueue_write_job(
            kind="checkpoint_memoria",
            payload={"momento": "decision", "que_paso": "test"},
            priority=1,
            db_path=queue_db,
        )

        status = get_write_job_status(result["job_id"], db_path=queue_db)
        assert status["kind"] == "checkpoint_memoria"
        assert status["status"] == "queued"
        assert status["attempts"] == 0

    def test_enqueue_stores_payload(self, queue_db):
        from modules.write_queue import enqueue_write_job

        payload = {"content": "important stuff", "topic": "consciencia", "importance": "high"}
        result = enqueue_write_job(
            kind="remember",
            payload=payload,
            db_path=queue_db,
        )

        # Verify payload stored correctly
        conn = sqlite3.connect(queue_db)
        row = conn.execute(
            "SELECT payload_json FROM write_queue WHERE job_id = ?",
            (result["job_id"],)
        ).fetchone()
        conn.close()

        stored = json.loads(row[0])
        assert stored["content"] == "important stuff"
        assert stored["topic"] == "consciencia"


# ============================================================
# DEDUPE
# ============================================================

class TestDedupe:
    def test_dedupe_returns_same_job_id(self, queue_db):
        """Spec test #1: identical writes return same job_id."""
        from modules.write_queue import enqueue_write_job, compute_dedupe_key

        key = compute_dedupe_key("remember", "test memory content")

        r1 = enqueue_write_job(
            kind="remember",
            payload={"content": "test memory content"},
            dedupe_key=key,
            db_path=queue_db,
        )
        r2 = enqueue_write_job(
            kind="remember",
            payload={"content": "test memory content"},
            dedupe_key=key,
            db_path=queue_db,
        )

        assert r1["job_id"] == r2["job_id"]
        assert r1["dedupe_hit"] is False
        assert r2["dedupe_hit"] is True

    def test_dedupe_different_kinds_no_collision(self, queue_db):
        from modules.write_queue import compute_dedupe_key

        key1 = compute_dedupe_key("remember", "same content")
        key2 = compute_dedupe_key("add_memory", "same content")
        assert key1 != key2

    def test_no_dedupe_when_key_absent(self, queue_db):
        from modules.write_queue import enqueue_write_job

        r1 = enqueue_write_job(kind="remember", payload={"c": "x"}, db_path=queue_db)
        r2 = enqueue_write_job(kind="remember", payload={"c": "x"}, db_path=queue_db)

        assert r1["job_id"] != r2["job_id"]
        assert r1["dedupe_hit"] is False
        assert r2["dedupe_hit"] is False

    def test_dedupe_ignores_failed_jobs(self, queue_db):
        """Failed (back to queued) jobs should still dedupe."""
        from modules.write_queue import enqueue_write_job, compute_dedupe_key

        key = compute_dedupe_key("remember", "will fail")

        r1 = enqueue_write_job(
            kind="remember",
            payload={"content": "will fail"},
            dedupe_key=key,
            db_path=queue_db,
        )

        # Manually set to failed->queued (simulating retry)
        conn = sqlite3.connect(queue_db)
        conn.execute(
            "UPDATE write_queue SET status = 'queued' WHERE job_id = ?",
            (r1["job_id"],)
        )
        conn.commit()
        conn.close()

        r2 = enqueue_write_job(
            kind="remember",
            payload={"content": "will fail"},
            dedupe_key=key,
            db_path=queue_db,
        )

        # Should still dedupe against the queued job
        assert r2["job_id"] == r1["job_id"]
        assert r2["dedupe_hit"] is True


# ============================================================
# CLAIM / LEASE
# ============================================================

class TestClaim:
    def test_claim_one_job(self, queue_db):
        """Spec test #2: worker claims one job with lease."""
        from modules.write_queue import enqueue_write_job, claim_next_job

        enqueue_write_job(kind="remember", payload={"c": "a"}, db_path=queue_db)
        enqueue_write_job(kind="remember", payload={"c": "b"}, db_path=queue_db)

        claimed = claim_next_job(db_path=queue_db)
        assert claimed is not None
        assert claimed["kind"] == "remember"
        assert claimed["attempts"] == 1

    def test_claim_respects_priority(self, queue_db):
        from modules.write_queue import enqueue_write_job, claim_next_job

        enqueue_write_job(kind="remember", payload={"c": "low"}, priority=10, db_path=queue_db)
        enqueue_write_job(kind="checkpoint_memoria", payload={"c": "high"}, priority=1, db_path=queue_db)

        claimed = claim_next_job(db_path=queue_db)
        assert claimed["kind"] == "checkpoint_memoria"

    def test_claim_sets_running_status(self, queue_db):
        from modules.write_queue import enqueue_write_job, claim_next_job, get_write_job_status

        r = enqueue_write_job(kind="remember", payload={"c": "x"}, db_path=queue_db)
        claim_next_job(db_path=queue_db)

        status = get_write_job_status(r["job_id"], db_path=queue_db)
        assert status["status"] == "running"
        assert status["attempts"] == 1

    def test_claim_skips_running_jobs(self, queue_db):
        from modules.write_queue import enqueue_write_job, claim_next_job

        enqueue_write_job(kind="remember", payload={"c": "a"}, db_path=queue_db)

        c1 = claim_next_job(db_path=queue_db)
        c2 = claim_next_job(db_path=queue_db)

        assert c1 is not None
        assert c2 is None

    def test_claim_returns_none_on_empty_queue(self, queue_db):
        from modules.write_queue import claim_next_job

        claimed = claim_next_job(db_path=queue_db)
        assert claimed is None

    def test_stale_lease_reclaimed(self, queue_db):
        """Jobs with expired leases should be reclaimed."""
        from modules.write_queue import enqueue_write_job, claim_next_job

        r = enqueue_write_job(kind="remember", payload={"c": "stale"}, db_path=queue_db)

        # Claim with very short lease
        claim_next_job(lease_seconds=1, db_path=queue_db)

        # Manually expire the lease
        conn = sqlite3.connect(queue_db)
        conn.execute(
            "UPDATE write_queue SET lease_until = '2020-01-01T00:00:00' WHERE job_id = ?",
            (r["job_id"],)
        )
        conn.commit()
        conn.close()

        # Should be reclaimed
        c2 = claim_next_job(db_path=queue_db)
        assert c2 is not None
        assert c2["job_id"] == r["job_id"]
        assert c2["attempts"] == 2  # incremented again


# ============================================================
# RETRY + DEAD LETTER
# ============================================================

class TestRetry:
    def test_mark_failed_returns_to_queue(self, queue_db):
        from modules.write_queue import enqueue_write_job, claim_next_job, mark_job_failed, get_write_job_status

        r = enqueue_write_job(kind="remember", payload={"c": "x"}, max_attempts=3, db_path=queue_db)
        claim_next_job(db_path=queue_db)

        result = mark_job_failed(r["job_id"], "timeout error", db_path=queue_db)
        assert result == "failed"

        status = get_write_job_status(r["job_id"], db_path=queue_db)
        assert status["status"] == "queued"  # back in queue for retry
        assert "timeout error" in status["last_error"]

    def test_marks_dead_after_max_attempts(self, queue_db):
        """Spec test #3: worker retries and marks dead."""
        from modules.write_queue import enqueue_write_job, claim_next_job, mark_job_failed, get_write_job_status

        r = enqueue_write_job(kind="remember", payload={"c": "x"}, max_attempts=2, db_path=queue_db)

        # Attempt 1
        claim_next_job(db_path=queue_db)
        mark_job_failed(r["job_id"], "error 1", db_path=queue_db)

        # Attempt 2 (should reach max_attempts)
        claim_next_job(db_path=queue_db)
        result = mark_job_failed(r["job_id"], "error 2", db_path=queue_db)

        assert result == "dead"
        status = get_write_job_status(r["job_id"], db_path=queue_db)
        assert status["status"] == "dead"

    def test_mark_done_succeeds(self, queue_db):
        from modules.write_queue import enqueue_write_job, claim_next_job, mark_job_done, get_write_job_status

        r = enqueue_write_job(kind="remember", payload={"c": "x"}, db_path=queue_db)
        claim_next_job(db_path=queue_db)

        success = mark_job_done(r["job_id"], db_path=queue_db)
        assert success is True

        status = get_write_job_status(r["job_id"], db_path=queue_db)
        assert status["status"] == "done"
        assert status["timestamps"]["completed_at"] is not None


# ============================================================
# OBSERVABILITY
# ============================================================

class TestObservability:
    def test_log_job_completion(self, queue_db):
        from modules.write_queue import log_job_completion

        log_job_completion(
            job_id="test-123",
            kind="remember",
            status="done",
            attempts=1,
            duration_ms=42500,
            db_path=queue_db,
        )

        conn = sqlite3.connect(queue_db)
        row = conn.execute(
            "SELECT * FROM write_queue_log WHERE job_id = 'test-123'"
        ).fetchone()
        conn.close()

        assert row is not None
        assert row[3] == "done"  # status
        assert row[5] == 42500  # duration_ms

    def test_queue_stats(self, queue_db):
        from modules.write_queue import enqueue_write_job, claim_next_job, mark_job_done, get_queue_stats

        enqueue_write_job(kind="remember", payload={"c": "a"}, db_path=queue_db)
        enqueue_write_job(kind="remember", payload={"c": "b"}, db_path=queue_db)
        r3 = enqueue_write_job(kind="remember", payload={"c": "c"}, db_path=queue_db)

        claim_next_job(db_path=queue_db)

        stats = get_queue_stats(db_path=queue_db)
        assert stats["total"] == 3
        assert stats["by_status"].get("queued", 0) == 2
        assert stats["by_status"].get("running", 0) == 1


# ============================================================
# STATUS
# ============================================================

class TestStatus:
    def test_status_not_found(self, queue_db):
        from modules.write_queue import get_write_job_status

        result = get_write_job_status("nonexistent-id", db_path=queue_db)
        assert result is None

    def test_status_full_lifecycle(self, queue_db):
        from modules.write_queue import (
            enqueue_write_job, claim_next_job, mark_job_done,
            get_write_job_status,
        )

        r = enqueue_write_job(kind="add_memory", payload={"content": "lifecycle test"}, db_path=queue_db)
        job_id = r["job_id"]

        # queued
        s1 = get_write_job_status(job_id, db_path=queue_db)
        assert s1["status"] == "queued"

        # running
        claim_next_job(db_path=queue_db)
        s2 = get_write_job_status(job_id, db_path=queue_db)
        assert s2["status"] == "running"

        # done
        mark_job_done(job_id, db_path=queue_db)
        s3 = get_write_job_status(job_id, db_path=queue_db)
        assert s3["status"] == "done"
        assert s3["timestamps"]["completed_at"] is not None


# ============================================================
# ACK SPEED (will be more meaningful in PR2 with tool changes)
# ============================================================

# ============================================================
# LATENCY INSTRUMENTATION (migration 004)
# ============================================================

class TestLatencyInstrumentation:
    def test_claim_sets_claimed_at(self, queue_db):
        """claim_next_job() records claimed_at timestamp."""
        from modules.write_queue import enqueue_write_job, claim_next_job

        r = enqueue_write_job(kind="remember", payload={"c": "x"}, db_path=queue_db)
        claim_next_job(db_path=queue_db)

        conn = sqlite3.connect(queue_db)
        row = conn.execute(
            "SELECT claimed_at FROM write_queue WHERE job_id = ?",
            (r["job_id"],)
        ).fetchone()
        conn.close()

        assert row[0] is not None, "claimed_at should be set after claim"

    def test_mark_job_started_sets_started_at(self, queue_db):
        """mark_job_started() records started_at timestamp."""
        from modules.write_queue import enqueue_write_job, claim_next_job, mark_job_started

        r = enqueue_write_job(kind="remember", payload={"c": "x"}, db_path=queue_db)
        claim_next_job(db_path=queue_db)
        success = mark_job_started(r["job_id"], db_path=queue_db)

        assert success is True

        conn = sqlite3.connect(queue_db)
        row = conn.execute(
            "SELECT started_at FROM write_queue WHERE job_id = ?",
            (r["job_id"],)
        ).fetchone()
        conn.close()

        assert row[0] is not None, "started_at should be set after mark_job_started"

    def test_started_at_after_claimed_at(self, queue_db):
        """started_at >= claimed_at (execution starts after claim)."""
        from modules.write_queue import enqueue_write_job, claim_next_job, mark_job_started

        r = enqueue_write_job(kind="remember", payload={"c": "x"}, db_path=queue_db)
        claim_next_job(db_path=queue_db)
        mark_job_started(r["job_id"], db_path=queue_db)

        conn = sqlite3.connect(queue_db)
        row = conn.execute(
            "SELECT claimed_at, started_at FROM write_queue WHERE job_id = ?",
            (r["job_id"],)
        ).fetchone()
        conn.close()

        assert row[0] is not None
        assert row[1] is not None
        assert row[1] >= row[0], "started_at should be >= claimed_at"

    def test_mark_job_started_only_running(self, queue_db):
        """mark_job_started() only updates jobs in 'running' status."""
        from modules.write_queue import enqueue_write_job, mark_job_started

        r = enqueue_write_job(kind="remember", payload={"c": "x"}, db_path=queue_db)
        # Job is 'queued', not 'running'
        success = mark_job_started(r["job_id"], db_path=queue_db)

        assert success is False


class TestACKSpeed:
    def test_enqueue_is_fast(self, queue_db):
        """Enqueue should complete in < 50ms (local SQLite, no network)."""
        from modules.write_queue import enqueue_write_job

        start = time.monotonic()
        for i in range(10):
            enqueue_write_job(
                kind="remember",
                payload={"content": f"speed test {i}"},
                db_path=queue_db,
            )
        elapsed = (time.monotonic() - start) * 1000

        avg_ms = elapsed / 10
        assert avg_ms < 50, f"Enqueue avg {avg_ms:.1f}ms exceeds 50ms target"
