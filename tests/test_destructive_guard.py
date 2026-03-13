"""
Tests for the destructive guard (G-03 remediation).

Validates:
  - Two-step token flow (request -> confirm)
  - dry_run mode
  - Guard bypass via CODI_DESTRUCTIVE_GUARD=off
  - Token expiry (TTL)
  - Single-use tokens
  - Token-tool mismatch rejection
  - Token-payload mismatch rejection
  - Legacy bridge gated by CODI_DESTRUCTIVE_LEGACY
"""

import time
from unittest.mock import patch, MagicMock

import pytest

from modules.destructive_guard import (
    is_guard_enabled,
    is_legacy_enabled,
    compute_fingerprint,
    request_confirmation,
    validate_and_consume,
    _pending_tokens,
    _lock,
    TokenInfo,
    TOKEN_TTL_SECONDS,
)


@pytest.fixture(autouse=True)
def _clean_tokens():
    """Ensure token store is empty before/after each test."""
    with _lock:
        _pending_tokens.clear()
    yield
    with _lock:
        _pending_tokens.clear()


class TestGuardConfig:
    def test_guard_enabled_by_default(self, monkeypatch):
        monkeypatch.delenv("CODI_DESTRUCTIVE_GUARD", raising=False)
        # No .destructive_guard file -> default ON
        with patch("modules.destructive_guard.open", side_effect=FileNotFoundError):
            assert is_guard_enabled() is True

    def test_guard_disabled_by_env(self, monkeypatch):
        monkeypatch.setenv("CODI_DESTRUCTIVE_GUARD", "off")
        assert is_guard_enabled() is False

    def test_guard_enabled_by_env(self, monkeypatch):
        monkeypatch.setenv("CODI_DESTRUCTIVE_GUARD", "on")
        assert is_guard_enabled() is True

    def test_legacy_disabled_by_default(self, monkeypatch):
        monkeypatch.delenv("CODI_DESTRUCTIVE_LEGACY", raising=False)
        assert is_legacy_enabled() is False

    def test_legacy_enabled_by_env(self, monkeypatch):
        monkeypatch.setenv("CODI_DESTRUCTIVE_LEGACY", "on")
        assert is_legacy_enabled() is True


class TestFingerprint:
    def test_deterministic(self):
        fp1 = compute_fingerprint("delete_memory", memory_id="abc123")
        fp2 = compute_fingerprint("delete_memory", memory_id="abc123")
        assert fp1 == fp2

    def test_different_args_different_fingerprint(self):
        fp1 = compute_fingerprint("delete_memory", memory_id="abc")
        fp2 = compute_fingerprint("delete_memory", memory_id="xyz")
        assert fp1 != fp2

    def test_different_tools_different_fingerprint(self):
        fp1 = compute_fingerprint("delete_memory", memory_id="abc")
        fp2 = compute_fingerprint("clear_all_memories", memory_id="abc")
        assert fp1 != fp2


class TestTokenFlow:
    """Core two-step flow tests."""

    def test_blocked_without_token(self):
        """T1: Tool call without token issues a token and blocks execution."""
        fp = compute_fingerprint("delete_memory", memory_id="test-id")
        token = request_confirmation("delete_memory", fp)
        assert token is not None
        assert len(token) == 16
        # Token is stored
        assert token in _pending_tokens

    def test_allows_with_valid_token(self):
        """T3: Valid token allows execution."""
        fp = compute_fingerprint("delete_memory", memory_id="test-id")
        token = request_confirmation("delete_memory", fp)
        err = validate_and_consume(token, "delete_memory", fp)
        assert err is None  # success
        # Token consumed
        assert token not in _pending_tokens

    def test_token_single_use(self):
        """T6: Token cannot be reused."""
        fp = compute_fingerprint("delete_memory", memory_id="test-id")
        token = request_confirmation("delete_memory", fp)
        # First use: success
        err1 = validate_and_consume(token, "delete_memory", fp)
        assert err1 is None
        # Second use: rejected
        err2 = validate_and_consume(token, "delete_memory", fp)
        assert err2 is not None
        assert "invalid or expired" in err2.lower()

    def test_token_expires_after_ttl(self):
        """T5: Expired token is rejected."""
        from datetime import datetime, timedelta

        fp = compute_fingerprint("delete_memory", memory_id="test-id")
        token = request_confirmation("delete_memory", fp)
        # Manually expire it
        with _lock:
            _pending_tokens[token].expires_at = datetime.now() - timedelta(seconds=1)
        err = validate_and_consume(token, "delete_memory", fp)
        assert err is not None
        assert "invalid or expired" in err.lower()

    def test_token_tool_mismatch_rejected(self):
        """T7: Token for one tool rejected on a different tool."""
        fp = compute_fingerprint("delete_memory", memory_id="test-id")
        token = request_confirmation("delete_memory", fp)
        # Try to use on clear_all_memories
        fp2 = compute_fingerprint("clear_all_memories")
        err = validate_and_consume(token, "clear_all_memories", fp2)
        assert err is not None
        assert "delete_memory" in err

    def test_token_payload_mismatch_rejected(self):
        """T8: Token for one payload rejected with different payload."""
        fp_foo = compute_fingerprint("delete_by_content", search_query="foo")
        token = request_confirmation("delete_by_content", fp_foo)
        # Try with different args
        fp_bar = compute_fingerprint("delete_by_content", search_query="bar")
        err = validate_and_consume(token, "delete_by_content", fp_bar)
        assert err is not None
        assert "mismatch" in err.lower()

    def test_empty_token_rejected(self):
        """Empty string token is rejected with clear message."""
        fp = compute_fingerprint("delete_memory", memory_id="x")
        err = validate_and_consume("", "delete_memory", fp)
        assert err is not None
        assert "no confirm_token" in err.lower()


class TestDeleteMemoryGuarded:
    """Integration tests: delete_memory with guard."""

    def test_delete_memory_blocked_returns_token(self, monkeypatch):
        monkeypatch.setenv("CODI_DESTRUCTIVE_GUARD", "on")
        from modules.memory_core import delete_memory
        result = delete_memory("some-id")
        assert "GUARD" in result
        assert "confirm_token=" in result

    def test_delete_memory_dry_run(self, monkeypatch):
        monkeypatch.setenv("CODI_DESTRUCTIVE_GUARD", "on")
        from modules.memory_core import delete_memory
        result = delete_memory("some-id", dry_run=True)
        assert "DRY RUN" in result
        assert "some-id" in result

    def test_delete_memory_guard_disabled_executes(self, monkeypatch):
        """T4: Guard off -> executes directly."""
        monkeypatch.setenv("CODI_DESTRUCTIVE_GUARD", "off")
        mock_pg = MagicMock()
        mock_fts = MagicMock()
        with patch("modules.memory_core.pg", mock_pg), \
             patch("modules.memory_core.delete_memory_fts", mock_fts):
            from modules.memory_core import delete_memory
            result = delete_memory("test-id")
        assert "eliminado" in result.lower()
        mock_pg.delete.assert_called_once_with("test-id")

    def test_delete_memory_valid_token_executes(self, monkeypatch):
        monkeypatch.setenv("CODI_DESTRUCTIVE_GUARD", "on")
        from modules.memory_core import delete_memory
        from modules.destructive_guard import compute_fingerprint, request_confirmation

        fp = compute_fingerprint("delete_memory", memory_id="test-id")
        token = request_confirmation("delete_memory", fp)

        mock_pg = MagicMock()
        mock_fts = MagicMock()
        with patch("modules.memory_core.pg", mock_pg), \
             patch("modules.memory_core.delete_memory_fts", mock_fts):
            result = delete_memory("test-id", confirm_token=token)
        assert "eliminado" in result.lower()
        mock_pg.delete.assert_called_once()


class TestClearAllMemoriesGuarded:
    """Integration tests: clear_all_memories is permanently blocked."""

    def test_clear_permanently_blocked(self, monkeypatch):
        monkeypatch.setenv("CODI_DESTRUCTIVE_GUARD", "on")
        from modules.memory_core import clear_all_memories
        result = clear_all_memories()
        assert "BLOQUEADO" in result

    def test_clear_blocked_even_with_dry_run(self, monkeypatch):
        monkeypatch.setenv("CODI_DESTRUCTIVE_GUARD", "on")
        from modules.memory_core import clear_all_memories
        result = clear_all_memories(dry_run=True)
        assert "BLOQUEADO" in result

    def test_clear_blocked_with_legacy_code(self, monkeypatch):
        monkeypatch.setenv("CODI_DESTRUCTIVE_GUARD", "on")
        monkeypatch.delenv("CODI_DESTRUCTIVE_LEGACY", raising=False)
        from modules.memory_core import clear_all_memories
        result = clear_all_memories(confirm_code="DELETE_ALL_MEMORIES")
        assert "BLOQUEADO" in result

    def test_clear_blocked_even_with_legacy_enabled(self, monkeypatch):
        monkeypatch.setenv("CODI_DESTRUCTIVE_GUARD", "on")
        monkeypatch.setenv("CODI_DESTRUCTIVE_LEGACY", "on")
        from modules.memory_core import clear_all_memories
        result = clear_all_memories(confirm_code="DELETE_ALL_MEMORIES")
        assert "BLOQUEADO" in result


class TestDeleteByContentGuarded:
    """Integration tests: delete_by_content with guard."""

    def test_delete_by_content_preview_with_token(self, monkeypatch):
        monkeypatch.setenv("CODI_DESTRUCTIVE_GUARD", "on")
        mock_pg = MagicMock()
        mock_pg.search.return_value = {
            "results": [
                {"id": "mem-1", "memory": "test memory content", "score": 0.95}
            ]
        }
        with patch("modules.memory_core.pg", mock_pg):
            from modules.memory_core import delete_by_content
            result = delete_by_content("test query")
        assert "PREVIEW" in result
        assert "confirm_token=" in result

    def test_delete_by_content_legacy_confirm_blocked(self, monkeypatch):
        monkeypatch.setenv("CODI_DESTRUCTIVE_GUARD", "on")
        monkeypatch.delenv("CODI_DESTRUCTIVE_LEGACY", raising=False)
        mock_pg = MagicMock()
        mock_pg.search.return_value = {
            "results": [{"id": "m1", "memory": "x", "score": 0.9}]
        }
        with patch("modules.memory_core.pg", mock_pg):
            from modules.memory_core import delete_by_content
            result = delete_by_content("test", confirm=True)
        assert "ya no es suficiente" in result
