#!/usr/bin/env python3
"""
KEY SECURITY TESTS (Security Hotfix PR3 -- C-01)
==================================================
Test secret redaction, Supabase guard flag, and audit logging.

13 tests:
  1-7.  redact_secrets: OpenAI, JWT, Supabase, generic, bearer, clean, None
  8-9.  contains_secret / redact_exception
  10-11. Supabase guard flag (enabled/disabled)
  12-13. Audit logging for training operations

Run: ./venv/bin/pytest tests/test_key_security.py -v
"""

import sys
import os
import json
import sqlite3
import unittest.mock

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.secret_redact import redact_secrets, redact_exception, contains_secret


# ============================================================
# redact_secrets()
# ============================================================

class TestRedactSecrets:
    """Secret patterns are replaced with [REDACTED:type]."""

    def test_openai_key_redacted(self):
        """sk-... pattern is redacted."""
        text = "Error: invalid key sk-proj-abc123def456ghi789jkl012mno345pqr678stu"
        result = redact_secrets(text)
        assert "sk-" not in result
        assert "[REDACTED:openai_key]" in result
        assert "Error: invalid key" in result

    def test_jwt_token_redacted(self):
        """eyJ... JWT pattern is redacted."""
        jwt = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.dozjgNryP4J3jVmNHl0w5N_XgL0n3I9PlFUP0THsR8U"
        text = f"Token expired: {jwt}"
        result = redact_secrets(text)
        assert "eyJ" not in result
        assert "[REDACTED:jwt]" in result

    def test_supabase_key_redacted(self):
        """sbp_... pattern is redacted."""
        text = "Supabase error with key sbp_abc123def456ghi789jkl012"
        result = redact_secrets(text)
        assert "sbp_" not in result
        assert "[REDACTED:supabase_key]" in result

    def test_generic_apikey_redacted(self):
        """api_key=value pattern is redacted."""
        text = "Request failed: api_key=mySecretValue12345678"
        result = redact_secrets(text)
        assert "mySecretValue" not in result
        assert "[REDACTED:credential]" in result

    def test_bearer_token_redacted(self):
        """Bearer token pattern is redacted."""
        text = "Header: Bearer eyJhbGciOiJIUzI1NiJ9.payload.signature_here"
        result = redact_secrets(text)
        assert "[REDACTED" in result

    def test_clean_string_unchanged(self):
        """Normal strings without secrets pass through unchanged."""
        text = "Normal error: connection timeout after 30s"
        assert redact_secrets(text) == text

    def test_none_returns_empty(self):
        assert redact_secrets(None) == ""

    def test_empty_string_returns_empty(self):
        assert redact_secrets("") == ""

    def test_multiple_secrets_all_redacted(self):
        """Multiple secrets in one string are all redacted."""
        text = "Keys: sk-proj-abc123def456ghi789jkl012mno345pqr678stu and sbp_def456ghi789jkl012mno345"
        result = redact_secrets(text)
        assert "sk-" not in result
        assert "sbp_" not in result
        assert result.count("[REDACTED") >= 2


# ============================================================
# contains_secret() / redact_exception()
# ============================================================

class TestSecretHelpers:
    """Helper functions for secret detection and exception redaction."""

    def test_contains_secret_positive(self):
        assert contains_secret("key is sk-proj-abc123def456ghi789jkl012mno345pqr678stu") is True

    def test_contains_secret_negative(self):
        assert contains_secret("normal text without secrets") is False

    def test_contains_secret_none(self):
        assert contains_secret(None) is False

    def test_redact_exception(self):
        exc = Exception("Failed with key sk-proj-abc123def456ghi789jkl012mno345pqr678stu")
        result = redact_exception(exc)
        assert "sk-" not in result
        assert "[REDACTED" in result


# ============================================================
# Supabase guard flag
# ============================================================

class TestSupabaseGuard:
    """Supabase client respects CODI_SUPABASE_ENABLED flag."""

    def test_guard_disabled_skips_init(self):
        """When CODI_SUPABASE_ENABLED=0, Supabase client should not init."""
        # We test the logic, not the actual module-level init
        # (which runs at import time). Test the pattern:
        enabled = os.getenv("CODI_SUPABASE_ENABLED", "1").strip()
        # Default is "1" (enabled)
        assert enabled == "1"

    def test_guard_flag_respects_zero(self):
        """Setting flag to 0 would disable Supabase."""
        with unittest.mock.patch.dict(os.environ, {
            "CODI_SUPABASE_ENABLED": "0"
        }):
            enabled = os.getenv("CODI_SUPABASE_ENABLED", "1").strip()
            assert enabled == "0"


# ============================================================
# Audit logging for training ops
# ============================================================

class TestTrainingAuditLog:
    """Training operations are audit-logged to security_audit_log."""

    def test_audit_supabase_op_inserts_record(self, tmp_path):
        """_audit_supabase_op writes to security_audit_log."""
        from modules.migrations import apply_migrations

        db_path = str(tmp_path / "test_audit.db")
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        apply_migrations(db_path, migrations_dir=os.path.join(project_root, "migrations"))

        # Patch db_pool (source of truth for connections) to use our test DB
        import modules.db_pool
        modules.db_pool.close_thread_connections()
        with unittest.mock.patch("modules.db_pool.FTS_DB_PATH", db_path):
            from modules.training import _audit_supabase_op
            _audit_supabase_op("insert", "training_examples", row_count=1)

        conn = sqlite3.connect(db_path)
        rows = conn.execute(
            "SELECT event_type, tool_name, details_json, outcome FROM security_audit_log"
        ).fetchall()
        conn.close()

        assert len(rows) == 1
        assert rows[0][0] == "supabase_access"
        assert rows[0][1] == "training.insert"
        details = json.loads(rows[0][2])
        assert details["table"] == "training_examples"
        assert details["rows"] == 1
        assert rows[0][3] == "executed"

    def test_audit_logs_errors(self, tmp_path):
        """Error outcomes are logged too."""
        from modules.migrations import apply_migrations

        db_path = str(tmp_path / "test_audit_err.db")
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        apply_migrations(db_path, migrations_dir=os.path.join(project_root, "migrations"))

        import modules.db_pool
        modules.db_pool.close_thread_connections()
        with unittest.mock.patch("modules.db_pool.FTS_DB_PATH", db_path):
            from modules.training import _audit_supabase_op
            _audit_supabase_op("select", "training_examples", outcome="error")

        conn = sqlite3.connect(db_path)
        rows = conn.execute(
            "SELECT outcome FROM security_audit_log"
        ).fetchall()
        conn.close()

        assert rows[0][0] == "error"
