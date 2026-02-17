#!/usr/bin/env python3
"""
DUAL-MODE COMPARE LAYER TESTS
==============================
Verify normalize, compare, record/update, recovery hook, and integrity.

10 test classes:
  1. normalize_sync_result maps each kind correctly
  2. normalize_async_result returns None when status != done
  3. normalize_async_result returns dict when status == done
  4. compare_results: matching case (same action)
  5. compare_results: divergence case (action mismatch)
  6. update_async_result: dual_link_missing recovery hook
  7. Full round-trip: record_sync -> update_async -> compute_comparison
  8. Idempotency: double record_sync_result => 1 row (Fix 1)
  9. Mode-per-row: compare uses write_mode from DB, not env (Fix 2)
  10. Migration 008: unique index + write_mode column

Run: ./venv/bin/pytest tests/test_dual_compare.py -v
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
def dual_db(tmp_path):
    """Create a DB with all migrations applied (including 007)."""
    db_path = str(tmp_path / "test_dual.db")
    apply_migrations(db_path, migrations_dir=FTS_MIGRATIONS_DIR)
    return db_path


# ============================================================
# Test 1: normalize_sync_result maps each kind correctly
# ============================================================

class TestNormalizeSyncResult:
    """normalize_sync_result maps diverse outputs to closed action set."""

    def test_remember_saved_new(self):
        from modules.dual_compare import normalize_sync_result
        raw = json.dumps({
            "action": "saved_new",
            "new_id": "abc-123",
            "message": "Memoria guardada"
        })
        result = normalize_sync_result("remember", raw)
        assert result["action"] == "saved_new"
        assert result["memory_id"] == "abc-123"

    def test_remember_skipped_duplicate(self):
        from modules.dual_compare import normalize_sync_result
        raw = json.dumps({
            "action": "skipped_duplicate",
            "existing_id": "dup-456",
            "score": 0.95
        })
        result = normalize_sync_result("remember", raw)
        assert result["action"] == "skipped_duplicate"
        assert result["memory_id"] == "dup-456"

    def test_remember_saved_with_relation(self):
        from modules.dual_compare import normalize_sync_result
        raw = json.dumps({
            "action": "saved_with_relation",
            "new_id": "rel-789",
            "related_to": "old-111",
            "score": 0.82
        })
        result = normalize_sync_result("remember", raw)
        assert result["action"] == "saved_new"  # mapped to saved_new

    def test_checkpoint_kind(self):
        from modules.dual_compare import normalize_sync_result
        raw = json.dumps({
            "message": "Checkpoint guardado",
            "memory_id": "cp-001"
        })
        result = normalize_sync_result("checkpoint_memoria", raw)
        assert result["action"] == "checkpoint_saved"
        assert result["memory_id"] == "cp-001"

    def test_add_memory_generic(self):
        from modules.dual_compare import normalize_sync_result
        raw = json.dumps({"result": "Memoria guardada: test content..."})
        result = normalize_sync_result("add_memory", raw)
        assert result["action"] == "completed"

    def test_error_case(self):
        from modules.dual_compare import normalize_sync_result
        raw = json.dumps({"action": "error", "detail": "connection refused"})
        result = normalize_sync_result("remember", raw)
        assert result["action"] == "error"


# ============================================================
# Test 2: normalize_async_result returns None when not done
# ============================================================

class TestNormalizeAsyncNotDone:
    """normalize_async_result returns None for non-done statuses."""

    def test_status_running(self):
        from modules.dual_compare import normalize_async_result
        result = normalize_async_result("remember", {"status": "running"})
        assert result is None

    def test_status_queued(self):
        from modules.dual_compare import normalize_async_result
        result = normalize_async_result("remember", {"status": "queued"})
        assert result is None

    def test_empty_dict(self):
        from modules.dual_compare import normalize_async_result
        result = normalize_async_result("remember", {})
        assert result is None

    def test_none_input(self):
        from modules.dual_compare import normalize_async_result
        result = normalize_async_result("remember", None)
        assert result is None


# ============================================================
# Test 3: normalize_async_result returns dict when done
# ============================================================

class TestNormalizeAsyncDone:
    """normalize_async_result returns normalized dict for done jobs."""

    def test_remember_done(self):
        from modules.dual_compare import normalize_async_result
        result = normalize_async_result("remember", {
            "status": "done",
            "action": "saved_new",
            "result": '{"action":"saved_new","new_id":"x"}'
        })
        assert result is not None
        assert result["action"] == "saved_new"

    def test_checkpoint_done(self):
        from modules.dual_compare import normalize_async_result
        result = normalize_async_result("checkpoint_memoria", {
            "status": "done",
            "result": "Checkpoint guardado OK"
        })
        assert result is not None
        assert result["action"] == "checkpoint_saved"

    def test_add_memory_done(self):
        from modules.dual_compare import normalize_async_result
        result = normalize_async_result("add_memory", {
            "status": "done",
            "result": "Guardado"
        })
        assert result is not None
        assert result["action"] == "completed"


# ============================================================
# Test 4: compare_results matching case
# ============================================================

class TestCompareMatch:
    """compare_results returns match=True for same actions."""

    def test_both_saved_new(self):
        from modules.dual_compare import compare_results
        sync = {"action": "saved_new", "memory_id": "aaa"}
        async_ = {"action": "saved_new", "memory_id": "bbb"}
        result = compare_results("remember", sync, async_)
        assert result["match"] is True
        assert result["divergence_code"] is None

    def test_both_skipped_duplicate(self):
        from modules.dual_compare import compare_results
        sync = {"action": "skipped_duplicate", "memory_id": "dup1"}
        async_ = {"action": "skipped_duplicate", "memory_id": "dup1"}
        result = compare_results("remember", sync, async_)
        assert result["match"] is True

    def test_both_checkpoint_saved(self):
        from modules.dual_compare import compare_results
        sync = {"action": "checkpoint_saved", "detail": "v1"}
        async_ = {"action": "checkpoint_saved", "detail": "v2"}
        result = compare_results("checkpoint_memoria", sync, async_)
        assert result["match"] is True


# ============================================================
# Test 5: compare_results divergence case
# ============================================================

class TestCompareDivergence:
    """compare_results detects action mismatches."""

    def test_action_mismatch_not_shadow(self):
        """Outside shadow mode, saved_new vs skipped_duplicate is a real mismatch."""
        from modules.dual_compare import compare_results
        sync = {"action": "saved_new", "memory_id": "aaa"}
        async_ = {"action": "skipped_duplicate", "memory_id": "bbb"}
        result = compare_results("remember", sync, async_, write_mode="sync")
        assert result["match"] is False
        assert result["divergence_code"] == "action_mismatch"

    def test_expected_shadow_dedup(self):
        """In shadow mode, saved_new vs skipped_duplicate is expected (same request)."""
        from modules.dual_compare import compare_results
        sync = {"action": "saved_new", "memory_id": "aaa"}
        async_ = {"action": "skipped_duplicate", "memory_id": "bbb"}
        result = compare_results("remember", sync, async_, write_mode="shadow")
        assert result["match"] is True
        assert result["divergence_code"] == "expected_shadow_dedup"

    def test_shadow_dedup_not_for_other_mismatches(self):
        """Shadow dedup only applies to saved_new vs skipped_duplicate, not other combos."""
        from modules.dual_compare import compare_results
        sync = {"action": "saved_new"}
        async_ = {"action": "noop"}
        result = compare_results("remember", sync, async_, write_mode="shadow")
        assert result["match"] is False
        assert result["divergence_code"] == "action_mismatch"

    def test_async_failed(self):
        from modules.dual_compare import compare_results
        sync = {"action": "saved_new"}
        async_ = {"action": "error", "detail": "timeout"}
        result = compare_results("remember", sync, async_)
        assert result["match"] is False
        assert result["divergence_code"] == "async_failed"

    def test_memory_id_presence_diff(self):
        from modules.dual_compare import compare_results
        # Both sides must explicitly include memory_id key for this check
        sync = {"action": "saved_new", "memory_id": "aaa"}
        async_ = {"action": "saved_new", "memory_id": None}  # key present but None
        result = compare_results("remember", sync, async_)
        assert result["match"] is False
        assert result["divergence_code"] == "memory_id_diff"


# ============================================================
# Test 6: update_async_result recovery hook (dual_link_missing)
# ============================================================

class TestRecoveryHook:
    """update_async_result returns dual_link_missing if no dual row exists."""

    def test_missing_dual_row(self, dual_db):
        from modules.dual_compare import update_async_result
        outcome = update_async_result(
            job_id="nonexistent-job-id",
            result_dict={"action": "saved_new"},
            db_path=dual_db,
        )
        assert outcome == "dual_link_missing"

    def test_update_existing_row(self, dual_db):
        """When dual row exists, update succeeds."""
        from modules.dual_compare import update_async_result, record_sync_result

        # First, record the sync side
        record_sync_result(
            fingerprint="fp_test_update",
            trace_id="trace_001",
            kind="remember",
            raw_output=json.dumps({"action": "saved_new", "new_id": "sync-id"}),
            async_job_id="job-update-test",
            async_status="queued",
            db_path=dual_db,
        )

        # Now update the async side
        outcome = update_async_result(
            job_id="job-update-test",
            result_dict={"status": "done", "action": "saved_new", "result": "ok"},
            db_path=dual_db,
        )
        assert outcome == "updated"


# ============================================================
# Test 7: Full round-trip
# ============================================================

class TestFullRoundTrip:
    """End-to-end: record sync -> update async -> compute comparison."""

    def test_round_trip_match(self, dual_db):
        from modules.dual_compare import (
            record_sync_result,
            update_async_result,
            compute_comparison_for_job,
            compute_request_fingerprint,
        )

        fp = compute_request_fingerprint("remember", "test content", "trace_rt")
        job_id = "job-round-trip-001"

        # 1. Sync path records its result
        record_sync_result(
            fingerprint=fp,
            trace_id="trace_rt",
            kind="remember",
            raw_output=json.dumps({"action": "saved_new", "new_id": "mem-sync"}),
            async_job_id=job_id,
            async_status="queued",
            db_path=dual_db,
        )

        # 2. Worker updates async side
        outcome = update_async_result(
            job_id=job_id,
            result_dict={"status": "done", "action": "saved_new", "result": '{"action":"saved_new"}'},
            db_path=dual_db,
        )
        assert outcome == "updated"

        # 3. Compute comparison
        result = compute_comparison_for_job(job_id, db_path=dual_db)
        assert result is not None
        assert result["match"] is True
        assert result["divergence_code"] is None

        # 4. Verify DB state
        conn = sqlite3.connect(dual_db)
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT * FROM dual_compare_log WHERE async_job_id = ?",
            (job_id,)
        ).fetchone()
        conn.close()

        assert row["compare_status"] == "computed"
        assert row["match"] == 1
        assert row["compare_computed_at"] is not None

    def test_round_trip_divergence(self, dual_db):
        from modules.dual_compare import (
            record_sync_result,
            update_async_result,
            compute_comparison_for_job,
            compute_request_fingerprint,
        )

        fp = compute_request_fingerprint("remember", "diverge content", "trace_div")
        job_id = "job-diverge-001"

        # Sync says saved_new
        record_sync_result(
            fingerprint=fp,
            trace_id="trace_div",
            kind="remember",
            raw_output=json.dumps({"action": "saved_new", "new_id": "mem-s"}),
            async_job_id=job_id,
            async_status="queued",
            db_path=dual_db,
        )

        # Async says skipped_duplicate (divergence!)
        outcome = update_async_result(
            job_id=job_id,
            result_dict={"status": "done", "action": "skipped_duplicate", "result": "dup"},
            db_path=dual_db,
        )
        assert outcome == "updated"

        result = compute_comparison_for_job(job_id, db_path=dual_db)
        assert result is not None
        assert result["match"] is False
        assert result["divergence_code"] == "action_mismatch"

        # Verify DB
        conn = sqlite3.connect(dual_db)
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT * FROM dual_compare_log WHERE async_job_id = ?",
            (job_id,)
        ).fetchone()
        conn.close()

        assert row["compare_status"] == "computed"
        assert row["match"] == 0
        assert row["divergence_code"] == "action_mismatch"

    def test_round_trip_async_error(self, dual_db):
        from modules.dual_compare import (
            record_sync_result,
            update_async_result,
            compute_comparison_for_job,
            compute_request_fingerprint,
        )

        fp = compute_request_fingerprint("remember", "error content", "trace_err")
        job_id = "job-error-001"

        # Sync succeeds
        record_sync_result(
            fingerprint=fp,
            trace_id="trace_err",
            kind="remember",
            raw_output=json.dumps({"action": "saved_new", "new_id": "mem-ok"}),
            async_job_id=job_id,
            async_status="queued",
            db_path=dual_db,
        )

        # Async fails
        outcome = update_async_result(
            job_id=job_id,
            error="TimeoutError: connection timed out",
            failure_reason="timeout",
            db_path=dual_db,
        )
        assert outcome == "updated"

        result = compute_comparison_for_job(job_id, db_path=dual_db)
        assert result is not None
        assert result["match"] is False
        assert result["divergence_code"] == "async_failed"


# ============================================================
# Test 8: Idempotency — double record_sync_result => 1 row
# ============================================================

class TestIdempotency:
    """record_sync_result is idempotent on async_job_id (Fix 1)."""

    def test_double_record_same_job_id(self, dual_db):
        """Calling record_sync_result twice with same async_job_id => 1 row."""
        from modules.dual_compare import record_sync_result

        record_sync_result(
            fingerprint="fp_first",
            trace_id="trace_idem_1",
            kind="remember",
            raw_output=json.dumps({"action": "saved_new", "new_id": "id-1"}),
            async_job_id="job-idempotent-001",
            async_status="queued",
            db_path=dual_db,
        )

        # Second call — different fingerprint, same async_job_id
        record_sync_result(
            fingerprint="fp_second",
            trace_id="trace_idem_2",
            kind="remember",
            raw_output=json.dumps({"action": "saved_new", "new_id": "id-1"}),
            async_job_id="job-idempotent-001",
            async_status="queued",
            db_path=dual_db,
        )

        conn = sqlite3.connect(dual_db)
        count = conn.execute(
            "SELECT COUNT(*) FROM dual_compare_log WHERE async_job_id = ?",
            ("job-idempotent-001",)
        ).fetchone()[0]
        conn.close()

        assert count == 1

    def test_double_record_preserves_computed(self, dual_db):
        """If row is already computed, second record_sync_result doesn't overwrite."""
        from modules.dual_compare import (
            record_sync_result, update_async_result,
            compute_comparison_for_job,
        )

        job_id = "job-idem-computed"

        # Record sync + async + compute
        record_sync_result(
            fingerprint="fp_comp_1",
            trace_id="t1",
            kind="remember",
            raw_output=json.dumps({"action": "saved_new", "new_id": "x"}),
            async_job_id=job_id,
            async_status="queued",
            db_path=dual_db,
        )
        update_async_result(
            job_id=job_id,
            result_dict={"status": "done", "action": "saved_new", "result": "ok"},
            db_path=dual_db,
        )
        compute_comparison_for_job(job_id, db_path=dual_db)

        # Now try to record again — should NOT overwrite
        record_sync_result(
            fingerprint="fp_comp_2",
            trace_id="t2",
            kind="remember",
            raw_output=json.dumps({"action": "error"}),
            async_job_id=job_id,
            async_status="queued",
            db_path=dual_db,
        )

        conn = sqlite3.connect(dual_db)
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT * FROM dual_compare_log WHERE async_job_id = ?",
            (job_id,)
        ).fetchone()
        conn.close()

        # Original fingerprint preserved, not overwritten
        assert row["request_fingerprint"] == "fp_comp_1"
        assert row["compare_status"] == "computed"
        assert row["match"] == 1


# ============================================================
# Test 9: Mode-per-row — compare uses write_mode from DB
# ============================================================

class TestModePerRow:
    """compare_results uses write_mode from row, not env (Fix 2)."""

    def test_shadow_dedup_via_write_mode_param(self):
        """write_mode='shadow' triggers expected_shadow_dedup regardless of env."""
        from modules.dual_compare import compare_results
        sync = {"action": "saved_new", "memory_id": "aaa"}
        async_ = {"action": "skipped_duplicate", "memory_id": "bbb"}

        # Even if env says 'sync', write_mode param overrides
        result = compare_results("remember", sync, async_, write_mode="shadow")
        assert result["match"] is True
        assert result["divergence_code"] == "expected_shadow_dedup"

    def test_sync_mode_via_write_mode_param(self):
        """write_mode='sync' does NOT trigger expected_shadow_dedup."""
        from modules.dual_compare import compare_results
        sync = {"action": "saved_new"}
        async_ = {"action": "skipped_duplicate"}

        result = compare_results("remember", sync, async_, write_mode="sync")
        assert result["match"] is False
        assert result["divergence_code"] == "action_mismatch"

    def test_compute_reads_write_mode_from_db(self, dual_db, monkeypatch):
        """compute_comparison_for_job reads write_mode from DB row, not env.

        Simulates worker scenario: env has no CODI_WRITE_MODE,
        but DB row has write_mode='shadow'.
        """
        from modules.dual_compare import (
            record_sync_result, update_async_result,
            compute_comparison_for_job,
        )

        # Record sync side IN shadow mode (sets write_mode='shadow' in DB)
        monkeypatch.setenv("CODI_WRITE_MODE", "shadow")
        job_id = "job-mode-per-row"

        record_sync_result(
            fingerprint="fp_mpr",
            trace_id="trace_mpr",
            kind="remember",
            raw_output=json.dumps({"action": "saved_new", "new_id": "m1"}),
            async_job_id=job_id,
            async_status="queued",
            db_path=dual_db,
        )

        # Async returns skipped_duplicate (expected in shadow)
        update_async_result(
            job_id=job_id,
            result_dict={"status": "done", "action": "skipped_duplicate", "result": "dup"},
            db_path=dual_db,
        )

        # Now compute comparison AS IF WE'RE THE WORKER (env = sync)
        monkeypatch.setenv("CODI_WRITE_MODE", "sync")

        result = compute_comparison_for_job(job_id, db_path=dual_db)

        # Should still be match because DB row has write_mode='shadow'
        assert result is not None
        assert result["match"] is True
        assert result["divergence_code"] == "expected_shadow_dedup"


# ============================================================
# Test 10: Migration 008 — unique index + write_mode column
# ============================================================

class TestMigration008:
    """Migration 008 creates unique index and write_mode column."""

    def test_unique_index_exists(self, dual_db):
        """UNIQUE index on async_job_id exists after migration."""
        conn = sqlite3.connect(dual_db)
        indexes = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='index' "
            "AND tbl_name='dual_compare_log'"
        ).fetchall()
        conn.close()

        index_names = [i[0] for i in indexes]
        assert "idx_dcl_async_job_id_unique" in index_names

    def test_write_mode_column_exists(self, dual_db):
        """write_mode column exists after migration."""
        conn = sqlite3.connect(dual_db)
        cols = {r[1] for r in conn.execute(
            "PRAGMA table_info(dual_compare_log)"
        ).fetchall()}
        conn.close()

        assert "write_mode" in cols

    def test_unique_constraint_enforced(self, dual_db):
        """Inserting duplicate async_job_id raises IntegrityError."""
        conn = sqlite3.connect(dual_db)
        conn.execute(
            "INSERT INTO dual_compare_log "
            "(request_fingerprint, trace_id, kind, sync_completed_at, "
            " async_job_id, write_mode) "
            "VALUES ('fp1', 't1', 'remember', '2026-01-01', 'dup-job', 'shadow')"
        )
        conn.commit()

        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                "INSERT INTO dual_compare_log "
                "(request_fingerprint, trace_id, kind, sync_completed_at, "
                " async_job_id, write_mode) "
                "VALUES ('fp2', 't2', 'remember', '2026-01-01', 'dup-job', 'shadow')"
            )
        conn.close()


# ============================================================
# Test 11: expected_dual_dedup (dual_ack mode)
# ============================================================

class TestExpectedDualDedup:
    """compare_results handles dual_ack expected dedup pattern."""

    def test_dual_ack_async_saved_sync_deduped_is_match(self):
        """In dual_ack: async=saved_new + sync=skipped_duplicate → match=True."""
        from modules.dual_compare import compare_results
        sync = {"action": "skipped_duplicate", "memory_id": "dup-1"}
        async_ = {"action": "saved_new", "memory_id": "new-1"}
        result = compare_results("remember", sync, async_, write_mode="dual_ack")
        assert result["match"] is True
        assert result["divergence_code"] == "expected_dual_dedup"

    def test_dual_ack_mismatch(self):
        """In dual_ack: async=error + sync=saved_new → match=False."""
        from modules.dual_compare import compare_results
        sync = {"action": "saved_new", "memory_id": "s1"}
        async_ = {"action": "error", "detail": "timeout"}
        result = compare_results("remember", sync, async_, write_mode="dual_ack")
        assert result["match"] is False
        assert result["divergence_code"] == "async_failed"

    def test_dual_ack_both_saved_is_match(self):
        """In dual_ack: both saved_new → match=True (no dedup expected)."""
        from modules.dual_compare import compare_results
        sync = {"action": "saved_new", "memory_id": "s1"}
        async_ = {"action": "saved_new", "memory_id": "a1"}
        result = compare_results("remember", sync, async_, write_mode="dual_ack")
        assert result["match"] is True
        assert result["divergence_code"] is None

    def test_dual_dedup_not_for_shadow(self):
        """Shadow mode does NOT trigger expected_dual_dedup even with inverted pattern."""
        from modules.dual_compare import compare_results
        # Inverted: sync=skipped, async=saved — in shadow this is NOT expected
        sync = {"action": "skipped_duplicate"}
        async_ = {"action": "saved_new"}
        result = compare_results("remember", sync, async_, write_mode="shadow")
        assert result["match"] is False
        assert result["divergence_code"] == "action_mismatch"


# ============================================================
# Test 12: record_async_intent creates row
# ============================================================

class TestRecordAsyncIntent:
    """record_async_intent pre-creates dual_compare_log row."""

    def test_creates_row(self, dual_db):
        from modules.dual_compare import record_async_intent
        record_async_intent(
            fingerprint="fp_intent_test",
            trace_id="trace_intent",
            kind="remember",
            async_job_id="job-intent-001",
            write_mode="dual_ack",
            db_path=dual_db,
        )

        conn = sqlite3.connect(dual_db)
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT * FROM dual_compare_log WHERE async_job_id = ?",
            ("job-intent-001",)
        ).fetchone()
        conn.close()

        assert row is not None
        assert row["kind"] == "remember"
        assert row["write_mode"] == "dual_ack"
        assert row["compare_status"] == "pending"
        assert row["request_fingerprint"] == "fp_intent_test"

    def test_idempotent_on_same_job_id(self, dual_db):
        """Calling record_async_intent twice with same job_id inserts only 1 row."""
        from modules.dual_compare import record_async_intent
        record_async_intent(
            fingerprint="fp_1",
            trace_id="t1",
            kind="remember",
            async_job_id="job-idem-intent",
            db_path=dual_db,
        )
        record_async_intent(
            fingerprint="fp_2",
            trace_id="t2",
            kind="remember",
            async_job_id="job-idem-intent",
            db_path=dual_db,
        )

        conn = sqlite3.connect(dual_db)
        count = conn.execute(
            "SELECT COUNT(*) FROM dual_compare_log WHERE async_job_id = ?",
            ("job-idem-intent",)
        ).fetchone()[0]
        conn.close()

        assert count == 1

    def test_sync_fills_in_after_intent(self, dual_db):
        """record_async_intent + record_sync_result fills both sides."""
        from modules.dual_compare import record_async_intent, record_sync_result

        job_id = "job-intent-then-sync"
        record_async_intent(
            fingerprint="fp_its",
            trace_id="trace_its",
            kind="remember",
            async_job_id=job_id,
            db_path=dual_db,
        )
        record_sync_result(
            fingerprint="fp_its",
            trace_id="trace_its",
            kind="remember",
            raw_output=json.dumps({"action": "skipped_duplicate", "existing_id": "dup-x"}),
            async_job_id=job_id,
            async_status="queued",
            db_path=dual_db,
        )

        conn = sqlite3.connect(dual_db)
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT * FROM dual_compare_log WHERE async_job_id = ?",
            (job_id,)
        ).fetchone()
        conn.close()

        assert row["sync_result_json"] is not None
        sync_data = json.loads(row["sync_result_json"])
        assert sync_data["action"] == "skipped_duplicate"

    def test_full_dual_ack_round_trip(self, dual_db):
        """Full dual_ack round-trip: intent → sync fills → async fills → compute."""
        from modules.dual_compare import (
            record_async_intent, record_sync_result,
            update_async_result, compute_comparison_for_job,
        )

        job_id = "job-dual-ack-rt"

        # 1. Async intent
        record_async_intent(
            fingerprint="fp_dart",
            trace_id="trace_dart",
            kind="remember",
            async_job_id=job_id,
            write_mode="dual_ack",
            db_path=dual_db,
        )

        # 2. Background sync (saw duplicate because async already saved)
        record_sync_result(
            fingerprint="fp_dart",
            trace_id="trace_dart",
            kind="remember",
            raw_output=json.dumps({"action": "skipped_duplicate", "existing_id": "e1"}),
            async_job_id=job_id,
            async_status="queued",
            db_path=dual_db,
        )

        # 3. Worker finishes async job
        update_async_result(
            job_id=job_id,
            result_dict={"status": "done", "action": "saved_new", "result": "ok"},
            db_path=dual_db,
        )

        # 4. Compute comparison
        result = compute_comparison_for_job(job_id, db_path=dual_db)
        assert result is not None
        assert result["match"] is True
        assert result["divergence_code"] == "expected_dual_dedup"


# ============================================================
# Test 13: update_sync_compare_status
# ============================================================

class TestUpdateSyncCompareStatus:
    """update_sync_compare_status updates status column by async_job_id."""

    def test_update_status_ok(self, dual_db):
        """Setting status='ok' updates column and clears sync_last_error."""
        from modules.dual_compare import record_async_intent, update_sync_compare_status

        job_id = "job-sync-status-ok"
        record_async_intent(
            fingerprint="fp_sso",
            trace_id="trace_sso",
            kind="remember",
            async_job_id=job_id,
            db_path=dual_db,
        )

        update_sync_compare_status(job_id, "ok", db_path=dual_db)

        conn = sqlite3.connect(dual_db)
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT sync_compare_status, sync_last_error FROM dual_compare_log "
            "WHERE async_job_id = ?", (job_id,)
        ).fetchone()
        conn.close()

        assert row["sync_compare_status"] == "ok"
        assert row["sync_last_error"] is None

    def test_update_status_error_with_message(self, dual_db):
        """Setting status='error' stores truncated error message."""
        from modules.dual_compare import record_async_intent, update_sync_compare_status

        job_id = "job-sync-status-err"
        record_async_intent(
            fingerprint="fp_sse",
            trace_id="trace_sse",
            kind="remember",
            async_job_id=job_id,
            db_path=dual_db,
        )

        long_error = "x" * 600
        update_sync_compare_status(job_id, "error", error=long_error, db_path=dual_db)

        conn = sqlite3.connect(dual_db)
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT sync_compare_status, sync_last_error FROM dual_compare_log "
            "WHERE async_job_id = ?", (job_id,)
        ).fetchone()
        conn.close()

        assert row["sync_compare_status"] == "error"
        assert len(row["sync_last_error"]) == 500  # truncated

    def test_update_status_missing_row_no_crash(self, dual_db):
        """Calling with nonexistent job_id is a no-op (no crash)."""
        from modules.dual_compare import update_sync_compare_status

        # Should not raise
        update_sync_compare_status("nonexistent-job", "error", error="test", db_path=dual_db)

    def test_update_status_timeout(self, dual_db):
        """Setting status='timeout' with error message works."""
        from modules.dual_compare import record_async_intent, update_sync_compare_status

        job_id = "job-sync-status-timeout"
        record_async_intent(
            fingerprint="fp_sst",
            trace_id="trace_sst",
            kind="remember",
            async_job_id=job_id,
            db_path=dual_db,
        )

        update_sync_compare_status(job_id, "timeout", error="timeout after 60s", db_path=dual_db)

        conn = sqlite3.connect(dual_db)
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT sync_compare_status, sync_last_error FROM dual_compare_log "
            "WHERE async_job_id = ?", (job_id,)
        ).fetchone()
        conn.close()

        assert row["sync_compare_status"] == "timeout"
        assert row["sync_last_error"] == "timeout after 60s"

    def test_update_status_skipped_capacity(self, dual_db):
        """Setting status='skipped_capacity' works."""
        from modules.dual_compare import record_async_intent, update_sync_compare_status

        job_id = "job-sync-status-skip"
        record_async_intent(
            fingerprint="fp_ssk",
            trace_id="trace_ssk",
            kind="remember",
            async_job_id=job_id,
            db_path=dual_db,
        )

        update_sync_compare_status(job_id, "skipped_capacity", error="semaphore full", db_path=dual_db)

        conn = sqlite3.connect(dual_db)
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT sync_compare_status, sync_last_error FROM dual_compare_log "
            "WHERE async_job_id = ?", (job_id,)
        ).fetchone()
        conn.close()

        assert row["sync_compare_status"] == "skipped_capacity"
        assert row["sync_last_error"] == "semaphore full"

    def test_ok_clears_previous_error(self, dual_db):
        """Setting status='ok' after 'error' clears sync_last_error."""
        from modules.dual_compare import record_async_intent, update_sync_compare_status

        job_id = "job-sync-status-clear"
        record_async_intent(
            fingerprint="fp_ssc",
            trace_id="trace_ssc",
            kind="remember",
            async_job_id=job_id,
            db_path=dual_db,
        )

        # First set error
        update_sync_compare_status(job_id, "error", error="some error", db_path=dual_db)
        # Then set ok (should clear error)
        update_sync_compare_status(job_id, "ok", db_path=dual_db)

        conn = sqlite3.connect(dual_db)
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT sync_compare_status, sync_last_error FROM dual_compare_log "
            "WHERE async_job_id = ?", (job_id,)
        ).fetchone()
        conn.close()

        assert row["sync_compare_status"] == "ok"
        assert row["sync_last_error"] is None


# ============================================================
# Test 14: created_at column
# ============================================================

class TestCreatedAtColumn:
    """created_at is set on row creation."""

    def test_record_async_intent_sets_created_at(self, dual_db):
        from modules.dual_compare import record_async_intent

        job_id = "job-created-at-test"
        record_async_intent(
            fingerprint="fp_cat",
            trace_id="trace_cat",
            kind="remember",
            async_job_id=job_id,
            db_path=dual_db,
        )

        conn = sqlite3.connect(dual_db)
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT created_at, sync_completed_at FROM dual_compare_log "
            "WHERE async_job_id = ?", (job_id,)
        ).fetchone()
        conn.close()

        assert row["created_at"] != ""
        assert row["created_at"] is not None

    def test_record_sync_result_sets_created_at(self, dual_db):
        from modules.dual_compare import record_sync_result

        record_sync_result(
            fingerprint="fp_cat_sync",
            trace_id="trace_cat_sync",
            kind="remember",
            raw_output='{"action": "saved_new", "new_id": "x"}',
            async_job_id="job-created-at-sync",
            async_status="queued",
            db_path=dual_db,
        )

        conn = sqlite3.connect(dual_db)
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT created_at FROM dual_compare_log "
            "WHERE async_job_id = ?", ("job-created-at-sync",)
        ).fetchone()
        conn.close()

        assert row["created_at"] != ""
        assert row["created_at"] is not None
