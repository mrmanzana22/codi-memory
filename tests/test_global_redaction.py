"""Tests for PR-S1: Global secret redaction in metrics wrapper + HTTP endpoints."""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.secret_redact import redact_secrets, redact_exception, contains_secret


class TestRedactSecrets:
    """Unit tests for the redaction function itself."""

    def test_openai_key_redacted(self):
        text = "Error connecting with key sk-proj-abc123def456ghi789jkl012mno345pqr678stu901vwx234"
        result = redact_secrets(text)
        assert "sk-proj-" not in result
        assert "[REDACTED:openai_key]" in result

    def test_jwt_redacted(self):
        text = "Auth failed: eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0In0.dBjftJeZ4CVP-mB92K27uhbUJU1p1r_wW1gFWFOEjXk"
        result = redact_secrets(text)
        assert "eyJ" not in result
        assert "[REDACTED:jwt]" in result

    def test_bearer_redacted(self):
        text = "Header: Bearer eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9.payload.signature"
        result = redact_secrets(text)
        assert "[REDACTED:" in result

    def test_clean_text_unchanged(self):
        text = "Normal error: connection timed out after 30s"
        assert redact_secrets(text) == text

    def test_none_returns_empty(self):
        assert redact_secrets(None) == ""

    def test_empty_string_returns_empty(self):
        assert redact_secrets("") == ""

    def test_non_string_passthrough(self):
        assert redact_secrets(123) == 123

    def test_redact_exception(self):
        exc = Exception("Failed with key sk-proj-abc123def456ghi789jkl012mno345pqr678stu901vwx234")
        result = redact_exception(exc)
        assert "sk-proj-" not in result
        assert "[REDACTED:openai_key]" in result

    def test_contains_secret_positive(self):
        assert contains_secret("key is sk-proj-abc123def456ghi789jkl012mno345pqr678stu901vwx234")

    def test_contains_secret_negative(self):
        assert not contains_secret("normal text without secrets")


class TestMetricsRedactionWrapper:
    """Test that the metrics instrument_mcp wrapper redacts tool responses."""

    def test_sync_tool_response_redacted(self, monkeypatch):
        """A sync MCP tool returning a string with a key gets redacted."""
        from unittest.mock import MagicMock

        # Create a fake MCP server
        fake_mcp = MagicMock()
        fake_mcp._metrics_instrumented = False

        # Capture the wrapped tool function
        registered_fn = None

        def fake_tool_decorator(*args, **kwargs):
            def decorator(fn):
                nonlocal registered_fn
                registered_fn = fn
                return fn
            return decorator

        fake_mcp.tool = fake_tool_decorator

        # Instrument it
        from modules.metrics import instrument_mcp
        instrument_mcp(fake_mcp)

        # Now register a tool that returns a secret
        LEAKED_KEY = "sk-proj-abc123def456ghi789jkl012mno345pqr678stu901vwx234"

        @fake_mcp.tool()
        def leaky_tool() -> str:
            return f"Error: API key was {LEAKED_KEY}"

        # The registered_fn is the wrapped version
        # Monkeypatch log_tool_call to no-op (avoid DB access)
        monkeypatch.setattr("modules.metrics.log_tool_call", lambda **kw: None)
        monkeypatch.setattr("modules.metrics.check_budget_violation", lambda *a: None)

        result = registered_fn()
        assert "sk-proj-" not in result
        assert "[REDACTED:openai_key]" in result

    def test_clean_response_unchanged(self, monkeypatch):
        """A tool returning clean text passes through unchanged."""
        from unittest.mock import MagicMock

        fake_mcp = MagicMock()
        fake_mcp._metrics_instrumented = False

        registered_fn = None

        def fake_tool_decorator(*args, **kwargs):
            def decorator(fn):
                nonlocal registered_fn
                registered_fn = fn
                return fn
            return decorator

        fake_mcp.tool = fake_tool_decorator

        from modules.metrics import instrument_mcp
        instrument_mcp(fake_mcp)

        @fake_mcp.tool()
        def clean_tool() -> str:
            return "All good, 5 memories found"

        monkeypatch.setattr("modules.metrics.log_tool_call", lambda **kw: None)
        monkeypatch.setattr("modules.metrics.check_budget_violation", lambda *a: None)

        result = registered_fn()
        assert result == "All good, 5 memories found"
