#!/usr/bin/env python3
"""
DUAL REPORT GATE TESTS
======================
Verify the automated PASS/FAIL gate logic of scripts/dual_report.py.

Tests use temp DBs with schema applied via apply_migrations.
No real network calls, no long sleeps.

Run: ./venv/bin/pytest tests/test_dual_report.py -v
"""

import sys
import os
import sqlite3
from datetime import datetime, timezone

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.migrations import apply_migrations

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FTS_MIGRATIONS_DIR = os.path.join(PROJECT_ROOT, "migrations")


@pytest.fixture
def report_db(tmp_path):
    """Create a DB with all migrations applied."""
    db_path = str(tmp_path / "test_report.db")
    apply_migrations(db_path, migrations_dir=FTS_MIGRATIONS_DIR)
    return db_path


def _now_iso():
    return datetime.now(timezone.utc).isoformat()


def _insert_dual_row(db_path, **overrides):
    """Insert a dual_compare_log row with sensible defaults."""
    now = _now_iso()
    defaults = {
        "request_fingerprint": f"fp_{id(overrides)}_{now[-6:]}",
        "trace_id": "trace_test",
        "kind": "remember",
        "sync_completed_at": now,
        "sync_result_json": '{"action": "saved_new"}',
        "sync_preview": "test",
        "async_job_id": f"job_{id(overrides)}_{now[-6:]}",
        "async_status_at_record": "queued",
        "write_mode": "dual_ack",
        "compare_status": "computed",
        "match": 1,
        "divergence_code": None,
        "created_at": now,
        "sync_compare_status": "ok",
        "sync_last_error": None,
    }
    defaults.update(overrides)

    conn = sqlite3.connect(db_path)
    conn.execute(
        "INSERT INTO dual_compare_log "
        "(request_fingerprint, trace_id, kind, sync_completed_at, "
        " sync_result_json, sync_preview, async_job_id, "
        " async_status_at_record, write_mode, compare_status, "
        " match, divergence_code, created_at, sync_compare_status, sync_last_error) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (defaults["request_fingerprint"], defaults["trace_id"],
         defaults["kind"], defaults["sync_completed_at"],
         defaults["sync_result_json"], defaults["sync_preview"],
         defaults["async_job_id"], defaults["async_status_at_record"],
         defaults["write_mode"], defaults["compare_status"],
         defaults["match"], defaults["divergence_code"],
         defaults["created_at"], defaults["sync_compare_status"],
         defaults["sync_last_error"]),
    )
    conn.commit()
    conn.close()


class TestGatePass:
    """Gate returns PASS with clean data."""

    def test_gate_pass_with_matches(self, report_db, capsys):
        from scripts.dual_report import main

        # Insert clean data (all matches)
        for i in range(5):
            _insert_dual_row(
                report_db,
                request_fingerprint=f"fp_pass_{i}",
                async_job_id=f"job_pass_{i}",
                match=1,
                divergence_code=None,
                sync_compare_status="ok",
            )

        exit_code = main(["--db", report_db, "--hours", "1"])
        captured = capsys.readouterr()
        assert "VERDICT: PASS" in captured.out
        assert exit_code == 0

    def test_gate_pass_empty(self, report_db, capsys):
        from scripts.dual_report import main

        exit_code = main(["--db", report_db, "--hours", "1"])
        captured = capsys.readouterr()
        assert "VERDICT: PASS" in captured.out
        assert exit_code == 0


class TestGateFailMismatch:
    """Gate returns FAIL on real mismatches."""

    def test_gate_fail_on_real_mismatch(self, report_db, capsys):
        from scripts.dual_report import main

        # Insert a real mismatch
        _insert_dual_row(
            report_db,
            request_fingerprint="fp_mismatch",
            async_job_id="job_mismatch",
            match=0,
            divergence_code="action_mismatch",
        )

        exit_code = main(["--db", report_db, "--hours", "1"])
        captured = capsys.readouterr()
        assert "VERDICT: FAIL" in captured.out
        assert "real_mismatch" in captured.out
        assert exit_code == 1

    def test_gate_fail_on_async_failed(self, report_db, capsys):
        from scripts.dual_report import main

        _insert_dual_row(
            report_db,
            request_fingerprint="fp_async_fail",
            async_job_id="job_async_fail",
            match=0,
            divergence_code="async_failed",
        )

        exit_code = main(["--db", report_db, "--hours", "1"])
        captured = capsys.readouterr()
        assert "VERDICT: FAIL" in captured.out
        assert exit_code == 1


class TestGateExcludesExpectedDedup:
    """Expected dedup rows don't trigger gate failure."""

    def test_expected_shadow_dedup_excluded(self, report_db, capsys):
        from scripts.dual_report import main

        _insert_dual_row(
            report_db,
            request_fingerprint="fp_shadow_dedup",
            async_job_id="job_shadow_dedup",
            match=1,
            divergence_code="expected_shadow_dedup",
            write_mode="shadow",
        )

        exit_code = main(["--db", report_db, "--hours", "1"])
        captured = capsys.readouterr()
        assert "VERDICT: PASS" in captured.out
        assert exit_code == 0

    def test_expected_dual_dedup_excluded(self, report_db, capsys):
        from scripts.dual_report import main

        _insert_dual_row(
            report_db,
            request_fingerprint="fp_dual_dedup",
            async_job_id="job_dual_dedup",
            match=1,
            divergence_code="expected_dual_dedup",
        )

        exit_code = main(["--db", report_db, "--hours", "1"])
        captured = capsys.readouterr()
        assert "VERDICT: PASS" in captured.out
        assert exit_code == 0

    def test_mixed_expected_and_real_fails(self, report_db, capsys):
        """Expected dedup passes, but a real mismatch still fails."""
        from scripts.dual_report import main

        _insert_dual_row(
            report_db,
            request_fingerprint="fp_expected",
            async_job_id="job_expected",
            match=1,
            divergence_code="expected_dual_dedup",
        )
        _insert_dual_row(
            report_db,
            request_fingerprint="fp_real",
            async_job_id="job_real",
            match=0,
            divergence_code="result_mismatch",
        )

        exit_code = main(["--db", report_db, "--hours", "1"])
        captured = capsys.readouterr()
        assert "VERDICT: FAIL" in captured.out
        assert exit_code == 1


class TestGatePendingAge:
    """Gate fails on old pending dual_ack rows."""

    def test_gate_fail_on_stale_pending(self, report_db, capsys):
        from scripts.dual_report import main

        # Insert a pending row with old created_at (20 minutes ago)
        from datetime import timedelta
        old_time = (datetime.now(timezone.utc) - timedelta(minutes=20)).isoformat()

        _insert_dual_row(
            report_db,
            request_fingerprint="fp_stale",
            async_job_id="job_stale",
            compare_status="pending",
            match=None,
            divergence_code=None,
            created_at=old_time,
            sync_completed_at=old_time,
        )

        exit_code = main(["--db", report_db, "--hours", "1"])
        captured = capsys.readouterr()
        assert "VERDICT: FAIL" in captured.out
        assert "pending_age_max" in captured.out
        assert exit_code == 1


class TestDbNotFound:
    """Report handles missing DB gracefully."""

    def test_missing_db(self, capsys):
        from scripts.dual_report import main

        exit_code = main(["--db", "/nonexistent/path.db"])
        captured = capsys.readouterr()
        assert exit_code == 1
        assert "not found" in captured.err.lower()
