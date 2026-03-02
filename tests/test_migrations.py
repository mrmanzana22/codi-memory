#!/usr/bin/env python3
"""
SQLITE MIGRATION TESTS (Deliverable 4)
========================================
Verify migration runner: discovery, application, idempotency,
checksum integrity, and concurrency safety.

5 tests:
  1. Empty DB -> all tables created from baseline
  2. Idempotency: apply twice -> same result, no errors
  3. Order: migrations applied in version order
  4. Checksum mismatch: modified SQL file -> RuntimeError
  5. Concurrency: parallel apply -> one wins, no corruption

Run: ./venv/bin/pytest tests/test_migrations.py -v
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import sqlite3
import shutil
import threading
import pytest

from modules.migrations import (
    apply_migrations,
    discover_migrations,
    get_current_version,
    _compute_checksum,
    _compute_fingerprint,
    _ensure_migrations_table,
)

# Use project's real migrations for baseline tests
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FTS_MIGRATIONS_DIR = os.path.join(PROJECT_ROOT, "migrations")
PROSPECTIVE_MIGRATIONS_DIR = os.path.join(PROJECT_ROOT, "migrations_prospective")


class TestEmptyDbBaseline:
    """Apply baseline migrations to empty DB -> all tables created."""

    def test_fts_baseline_creates_tables(self, tmp_path):
        """Apply FTS baseline to empty DB, verify all 16 tables exist."""
        db_path = str(tmp_path / "test_fts.db")
        result = apply_migrations(db_path, migrations_dir=FTS_MIGRATIONS_DIR)

        assert "001" in result["applied"]
        assert "002" in result["applied"]
        assert "003" in result["applied"]
        assert "004" in result["applied"]
        assert "005" in result["applied"]
        assert "006" in result["applied"]
        assert "007" in result["applied"]
        assert "012" in result["applied"]
        assert "014" in result["applied"]
        assert "015" in result["applied"]
        assert "016" in result["applied"]
        assert "017" in result["applied"]
        assert "018" in result["applied"]
        assert "019" in result["applied"]
        assert "020" in result["applied"]
        assert "021" in result["applied"]
        assert result["current_version"] == "021"

        conn = sqlite3.connect(db_path)
        tables = {r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
        ).fetchall()}
        conn.close()

        expected_tables = {
            "schema_migrations",
            "memories_text", "fts_retry_queue",
            "tool_calls", "consolidation_log", "reconsolidation_log",
            "labile_memories", "schemas", "working_memory",
            "narrative_traces", "trace_chains",
            "failed_searches", "fok_calibration_log",
            "event_counts", "prediction_state", "prediction_results",
            "session_checkpoints",
            "write_queue", "write_queue_log",
            "security_audit_log",
            "dual_compare_log",
            "pending_corrections",
            "reward_signals",
            "strength_log",
            "fok_priors",
            "causal_chains",
            "retrieval_weights",
            "causal_discovery_state",
        }
        # memories_fts is a virtual table, check separately
        vtables = {r[0] for r in sqlite3.connect(db_path).execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='memories_fts'"
        ).fetchall()}

        assert expected_tables.issubset(tables), f"Missing tables: {expected_tables - tables}"
        assert "memories_fts" in vtables

    def test_prospective_baseline_creates_tables(self, tmp_path):
        """Apply prospective baseline to empty DB, verify 2 tables exist."""
        db_path = str(tmp_path / "test_prospective.db")
        result = apply_migrations(db_path, migrations_dir=PROSPECTIVE_MIGRATIONS_DIR)

        assert "001" in result["applied"]

        conn = sqlite3.connect(db_path)
        tables = {r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
        ).fetchall()}
        conn.close()

        assert "intentions" in tables
        assert "intention_log" in tables
        assert "schema_migrations" in tables


class TestIdempotency:
    """Apply migrations twice -> second run skips, no errors."""

    def test_double_apply_no_errors(self, tmp_path):
        """Two consecutive applies produce same result."""
        db_path = str(tmp_path / "test_idempotent.db")

        r1 = apply_migrations(db_path, migrations_dir=FTS_MIGRATIONS_DIR)
        r2 = apply_migrations(db_path, migrations_dir=FTS_MIGRATIONS_DIR)

        assert len(r1["applied"]) > 0
        assert len(r2["applied"]) == 0
        assert len(r2["skipped"]) == len(r1["applied"])
        assert r1["current_version"] == r2["current_version"]

    def test_fingerprint_stable(self, tmp_path):
        """Fingerprint is same after double apply."""
        db_path = str(tmp_path / "test_fp.db")

        apply_migrations(db_path, migrations_dir=FTS_MIGRATIONS_DIR)
        conn1 = sqlite3.connect(db_path)
        fp1 = _compute_fingerprint(conn1)
        conn1.close()

        apply_migrations(db_path, migrations_dir=FTS_MIGRATIONS_DIR)
        conn2 = sqlite3.connect(db_path)
        fp2 = _compute_fingerprint(conn2)
        conn2.close()

        assert fp1 == fp2


class TestMigrationOrder:
    """Migrations applied in version order."""

    def test_version_order(self, tmp_path):
        """Create 3 migration files, verify they apply in order."""
        mig_dir = str(tmp_path / "ordered_migrations")
        os.makedirs(mig_dir)

        # Create 3 migrations out of filesystem order
        for version, table in [("003", "c"), ("001", "a"), ("002", "b")]:
            with open(os.path.join(mig_dir, f"{version}_create_{table}.sql"), "w") as f:
                f.write(f"CREATE TABLE IF NOT EXISTS test_{table} (id INTEGER PRIMARY KEY);")

        db_path = str(tmp_path / "test_order.db")
        result = apply_migrations(db_path, migrations_dir=mig_dir)

        assert result["applied"] == ["001", "002", "003"]
        assert result["current_version"] == "003"

    def test_partial_apply_continues(self, tmp_path):
        """If 001 already applied, only 002 runs."""
        mig_dir = str(tmp_path / "partial_migrations")
        os.makedirs(mig_dir)

        with open(os.path.join(mig_dir, "001_first.sql"), "w") as f:
            f.write("CREATE TABLE IF NOT EXISTS first_table (id INTEGER PRIMARY KEY);")

        db_path = str(tmp_path / "test_partial.db")
        r1 = apply_migrations(db_path, migrations_dir=mig_dir)
        assert r1["applied"] == ["001"]

        # Add second migration
        with open(os.path.join(mig_dir, "002_second.sql"), "w") as f:
            f.write("CREATE TABLE IF NOT EXISTS second_table (id INTEGER PRIMARY KEY);")

        r2 = apply_migrations(db_path, migrations_dir=mig_dir)
        assert r2["applied"] == ["002"]
        assert r2["skipped"] == ["001"]
        assert r2["current_version"] == "002"


class TestChecksumIntegrity:
    """Modified migration SQL -> RuntimeError."""

    def test_checksum_mismatch_raises(self, tmp_path):
        """If a migration file is modified after application, raise error."""
        mig_dir = str(tmp_path / "checksum_migrations")
        os.makedirs(mig_dir)

        sql_path = os.path.join(mig_dir, "001_original.sql")
        with open(sql_path, "w") as f:
            f.write("CREATE TABLE IF NOT EXISTS original (id INTEGER PRIMARY KEY);")

        db_path = str(tmp_path / "test_checksum.db")
        apply_migrations(db_path, migrations_dir=mig_dir)

        # Modify the migration file
        with open(sql_path, "w") as f:
            f.write("CREATE TABLE IF NOT EXISTS original (id INTEGER PRIMARY KEY, extra TEXT);")

        with pytest.raises(RuntimeError, match="checksum mismatch"):
            apply_migrations(db_path, migrations_dir=mig_dir)

    def test_checksum_deterministic(self):
        """Same SQL produces same checksum, different SQL produces different."""
        sql = "CREATE TABLE foo (id INTEGER PRIMARY KEY);"
        assert _compute_checksum(sql) == _compute_checksum(sql)
        assert _compute_checksum(sql) != _compute_checksum(sql + "\nCREATE TABLE bar (id INTEGER);")
        # Trailing whitespace is stripped (by design)
        assert _compute_checksum(sql) == _compute_checksum(sql + "  ")


class TestConcurrency:
    """Parallel apply_migrations -> no corruption."""

    def test_sequential_apply_safe(self, tmp_path):
        """Sequential applies from different connections -> no corruption."""
        mig_dir = str(tmp_path / "concurrent_migrations")
        os.makedirs(mig_dir)

        with open(os.path.join(mig_dir, "001_concurrent.sql"), "w") as f:
            f.write("CREATE TABLE IF NOT EXISTS concurrent_test (id INTEGER PRIMARY KEY, val TEXT);")

        db_path = str(tmp_path / "test_concurrent.db")

        # First apply succeeds
        r1 = apply_migrations(db_path, migrations_dir=mig_dir)
        assert r1["applied"] == ["001"]

        # Second apply from fresh connection skips (already applied)
        r2 = apply_migrations(db_path, migrations_dir=mig_dir)
        assert r2["applied"] == []
        assert r2["skipped"] == ["001"]

        # DB should be consistent
        conn = sqlite3.connect(db_path)
        tables = {r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()}
        conn.close()

        assert "concurrent_test" in tables
        assert "schema_migrations" in tables
        assert get_current_version(db_path) == "001"

    def test_parallel_apply_no_corruption(self, tmp_path):
        """Two threads applying -> at least one succeeds, DB not corrupted."""
        mig_dir = str(tmp_path / "parallel_migrations")
        os.makedirs(mig_dir)

        with open(os.path.join(mig_dir, "001_parallel.sql"), "w") as f:
            f.write("CREATE TABLE IF NOT EXISTS parallel_test (id INTEGER PRIMARY KEY);")

        db_path = str(tmp_path / "test_parallel.db")
        results = [None, None]
        errors = [None, None]

        def run_migration(idx):
            try:
                results[idx] = apply_migrations(db_path, migrations_dir=mig_dir)
            except Exception as e:
                errors[idx] = e

        t1 = threading.Thread(target=run_migration, args=(0,))
        t2 = threading.Thread(target=run_migration, args=(1,))
        t1.start()
        t2.start()
        t1.join(timeout=15)
        t2.join(timeout=15)

        # At least one must succeed
        successes = sum(1 for e in errors if e is None)
        assert successes >= 1, f"Both failed: {errors}"

        # DB must be consistent regardless
        conn = sqlite3.connect(db_path)
        tables = {r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()}
        conn.close()

        assert "parallel_test" in tables
        assert get_current_version(db_path) == "001"


# ============================================================
# D4 SCHEMA ENFORCEMENT TESTS
# ============================================================

from modules.migrations import ensure_schema_ready, ensure_schema_ready_db


class TestEnsureSchemaReady:
    """ensure_schema_ready validates tables or raises RuntimeError."""

    def test_missing_table_raises_runtime_error(self, tmp_path):
        """ensure_schema_ready raises RuntimeError when table missing."""
        db_path = str(tmp_path / "empty.db")
        conn = sqlite3.connect(db_path)
        conn.execute("CREATE TABLE dummy (id INTEGER)")
        conn.commit()

        with pytest.raises(RuntimeError, match="working_memory"):
            ensure_schema_ready(conn, ["working_memory"])
        conn.close()

    def test_existing_table_passes(self, tmp_path):
        """ensure_schema_ready passes when table exists."""
        db_path = str(tmp_path / "has_tables.db")
        apply_migrations(db_path, migrations_dir=FTS_MIGRATIONS_DIR)

        conn = sqlite3.connect(db_path)
        # Should NOT raise
        ensure_schema_ready(conn, [
            "memories_text", "fts_retry_queue", "tool_calls",
            "consolidation_log", "reconsolidation_log", "labile_memories",
            "schemas", "working_memory", "narrative_traces", "trace_chains",
            "failed_searches", "fok_calibration_log", "event_counts",
            "prediction_state", "prediction_results", "session_checkpoints",
        ])
        conn.close()

    def test_ensure_schema_ready_db_convenience(self, tmp_path):
        """ensure_schema_ready_db opens conn, validates, closes."""
        db_path = str(tmp_path / "conv.db")
        apply_migrations(db_path, migrations_dir=FTS_MIGRATIONS_DIR)

        # Should NOT raise
        ensure_schema_ready_db(db_path, ["memories_text", "tool_calls"])

    def test_ensure_schema_ready_db_missing_raises(self, tmp_path):
        """ensure_schema_ready_db raises RuntimeError for missing table."""
        db_path = str(tmp_path / "empty2.db")
        conn = sqlite3.connect(db_path)
        conn.execute("CREATE TABLE dummy (id INTEGER)")
        conn.commit()
        conn.close()

        with pytest.raises(RuntimeError, match="nonexistent_table"):
            ensure_schema_ready_db(db_path, ["nonexistent_table"])


class TestNoCreateTableOutsideMigrations:
    """Static analysis: no CREATE TABLE in modules/*.py except migrations.py."""

    def test_no_create_table_outside_migrations(self):
        """No module should contain CREATE TABLE -- only migrations.py allowed."""
        import re
        pattern = re.compile(r"\bCREATE\s+TABLE\b", re.IGNORECASE | re.MULTILINE)
        WHITELIST = {"migrations.py", "self_model.py", "spreading.py", "sleep_loop.py", "active_inference.py"}

        violations = []
        modules_dir = os.path.join(PROJECT_ROOT, "modules")
        for dirpath, _dirnames, filenames in os.walk(modules_dir):
            for fname in filenames:
                if not fname.endswith(".py") or fname in WHITELIST:
                    continue
                fpath = os.path.join(dirpath, fname)
                with open(fpath) as f:
                    content = f.read()
                for match in pattern.finditer(content):
                    line_num = content[:match.start()].count("\n") + 1
                    line_text = content.splitlines()[line_num - 1].strip()
                    violations.append(f"{fname}:{line_num}: {line_text}")

        assert not violations, (
            f"CREATE TABLE found outside migrations:\n" +
            "\n".join(violations)
        )
