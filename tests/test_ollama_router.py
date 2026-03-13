"""
Tests for ollama_router.py — quality gates, model selection, logging.

These tests are self-contained: they don't need mem0, qdrant, or SQLite.
The autouse _isolate_sqlite fixture from conftest.py is disabled here.
"""

import json
import os
import sys
import pytest
from unittest.mock import patch, MagicMock

# Ensure CODI_USE_OLLAMA is set for tests
os.environ["CODI_USE_OLLAMA"] = "true"

# The ollama_router module only uses lazy imports from config_pg,
# so it can be imported without mem0/qdrant. But the conftest.py
# autouse fixture _isolate_sqlite triggers modules.config import.
# We override it here.
pytestmark = pytest.mark.usefixtures()  # clear autouse fixtures

from modules.ollama_router import (
    OllamaClient,
    OllamaQualityGate,
    OllamaRouter,
    TASK_MODEL_MAP,
    _strip_think_tags,
    _extract_json_from_response,
    ollama_chat_completion,
    get_ollama_stats,
)


@pytest.fixture(autouse=True)
def _isolate_sqlite():
    """Override the conftest autouse fixture — router tests don't need SQLite."""
    yield


# ──────────────────────────────────────────────
# Quality Gate Tests
# ──────────────────────────────────────────────

class TestQualityGateJsonArray:
    """Test JSON array validation."""

    def test_valid_edge_classify(self):
        output = json.dumps([
            {"type": "causes", "confidence": 0.85},
            {"type": "co_occurs", "confidence": 0.6},
        ])
        ok, reason = OllamaQualityGate.validate("edge_classify", output)
        assert ok, reason

    def test_edge_classify_string_items(self):
        """Backward compat: string items are OK."""
        output = json.dumps(["causes", "enables"])
        ok, reason = OllamaQualityGate.validate("edge_classify", output)
        assert ok, reason

    def test_edge_classify_invalid_type(self):
        output = json.dumps([{"type": "loves"}])
        ok, reason = OllamaQualityGate.validate("edge_classify", output)
        assert not ok

    def test_edge_classify_string_invalid_type(self):
        output = json.dumps(["loves"])
        ok, reason = OllamaQualityGate.validate("edge_classify", output)
        assert not ok

    def test_edge_classify_not_array(self):
        output = json.dumps({"type": "causes"})
        ok, reason = OllamaQualityGate.validate("edge_classify", output)
        assert not ok

    def test_edge_classify_empty_array(self):
        output = json.dumps([])
        ok, reason = OllamaQualityGate.validate("edge_classify", output)
        assert not ok  # min_items = 1

    def test_valid_semantic_extract(self):
        output = json.dumps([
            {"fact": "Codi uses prediction error", "confidence": 0.9},
        ])
        ok, reason = OllamaQualityGate.validate("semantic_extract", output)
        assert ok, reason

    def test_semantic_extract_missing_key(self):
        output = json.dumps([{"fact": "something"}])  # missing confidence
        ok, reason = OllamaQualityGate.validate("semantic_extract", output)
        assert not ok

    def test_self_extract_empty_valid(self):
        """Empty array is valid for self_extract (no self-knowledge found)."""
        output = json.dumps([])
        ok, reason = OllamaQualityGate.validate("self_extract", output)
        assert ok, reason

    def test_invalid_json(self):
        ok, reason = OllamaQualityGate.validate("edge_classify", "not json{")
        assert not ok
        assert "invalid JSON" in reason

    def test_markdown_wrapped_json(self):
        """JSON wrapped in ```json ... ``` should still pass."""
        inner = json.dumps([{"type": "causes", "confidence": 0.8}])
        output = f"```json\n{inner}\n```"
        ok, reason = OllamaQualityGate.validate("edge_classify", output)
        assert ok, reason


class TestQualityGatePlainText:
    """Test plain text validation."""

    def test_valid_compression(self):
        output = "This is a summary of the episodes. " * 3
        ok, reason = OllamaQualityGate.validate("compress_episodes", output)
        assert ok, reason

    def test_too_short(self):
        ok, reason = OllamaQualityGate.validate("compress_episodes", "Short")
        assert not ok
        assert "too short" in reason

    def test_checkpoint_too_short(self):
        ok, reason = OllamaQualityGate.validate("compress_checkpoints", "Hi")
        assert not ok

    def test_curiosity_valid(self):
        output = "x" * 150
        ok, reason = OllamaQualityGate.validate("resolve_curiosity", output)
        assert ok, reason

    def test_curiosity_too_short(self):
        output = "x" * 50
        ok, reason = OllamaQualityGate.validate("resolve_curiosity", output)
        assert not ok

    def test_unknown_task_type_passes(self):
        ok, reason = OllamaQualityGate.validate("nonexistent_task", "anything")
        assert ok


# ──────────────────────────────────────────────
# Think Tag Stripping
# ──────────────────────────────────────────────

class TestThinkTags:
    def test_strip_think_tags(self):
        text = "<think>Let me think about this...</think>The actual answer is 42."
        assert _strip_think_tags(text) == "The actual answer is 42."

    def test_no_think_tags(self):
        text = "No thinking here."
        assert _strip_think_tags(text) == "No thinking here."

    def test_multiline_think(self):
        text = "<think>\nStep 1: consider\nStep 2: decide\n</think>\nFinal answer."
        assert _strip_think_tags(text) == "Final answer."

    def test_think_with_json(self):
        """DeepSeek R1 often wraps JSON after think block."""
        inner = json.dumps([{"type": "causes"}])
        text = f"<think>reasoning</think>\n```json\n{inner}\n```"
        result = _strip_think_tags(text)
        assert "causes" in result
        assert "<think>" not in result


class TestExtractJson:
    def test_plain_json(self):
        j = '[{"a": 1}]'
        assert _extract_json_from_response(j) == j

    def test_markdown_json(self):
        j = '[{"a": 1}]'
        text = f"```json\n{j}\n```"
        assert _extract_json_from_response(text) == j

    def test_think_plus_markdown(self):
        j = '[{"a": 1}]'
        text = f"<think>hmm</think>\n```json\n{j}\n```"
        result = _extract_json_from_response(text)
        assert result == j


# ──────────────────────────────────────────────
# Model Selection
# ──────────────────────────────────────────────

class TestModelSelection:
    def test_all_tasks_have_models(self):
        for task in [
            "edge_classify", "semantic_extract", "self_extract",
            "compress_episodes", "compress_checkpoints",
            "resolve_curiosity", "mine_tool_patterns",
            "study_simple", "study_complex",
        ]:
            assert task in TASK_MODEL_MAP, f"Missing model for {task}"

    def test_fast_tasks_use_4b(self):
        assert TASK_MODEL_MAP["edge_classify"] == "qwen3.5:4b"
        assert TASK_MODEL_MAP["compress_checkpoints"] == "qwen3.5:4b"

    def test_reasoning_uses_expected_model(self):
        # All tasks consolidated to qwen3.5:4b (2026-03-09)
        assert TASK_MODEL_MAP["resolve_curiosity"] == "qwen3.5:4b"
        assert TASK_MODEL_MAP["study_complex"] == "qwen3.5:4b"


# ──────────────────────────────────────────────
# Router Integration (mocked Ollama)
# ──────────────────────────────────────────────

class TestRouter:
    @patch.object(OllamaClient, "chat")
    @patch.object(OllamaClient, "is_available", return_value=True)
    def test_route_success(self, mock_avail, mock_chat):
        mock_chat.return_value = json.dumps([
            {"type": "causes", "confidence": 0.9},
        ])
        router = OllamaRouter()
        result = router.route("edge_classify", "test prompt")
        assert result["ok"]
        assert result["model"] == "qwen3.5:4b"
        assert result["attempts"] == 1

    @patch.object(OllamaClient, "chat")
    def test_route_quality_fail_then_retry(self, mock_chat):
        # First call: bad output. Second call: good output.
        mock_chat.side_effect = [
            "not json at all",
            json.dumps([{"type": "causes", "confidence": 0.8}]),
        ]
        router = OllamaRouter()
        result = router.route("edge_classify", "test prompt")
        assert result["ok"]
        assert result["attempts"] == 2

    @patch.object(OllamaClient, "chat")
    def test_route_all_attempts_fail(self, mock_chat):
        mock_chat.return_value = "garbage"
        router = OllamaRouter()
        result = router.route("edge_classify", "test prompt")
        assert not result["ok"]

    @patch.object(OllamaClient, "chat")
    def test_route_runtime_error(self, mock_chat):
        mock_chat.side_effect = RuntimeError("connection refused")
        router = OllamaRouter()
        result = router.route("edge_classify", "test prompt")
        assert not result["ok"]
        assert "connection refused" in result["quality_reason"]

    def test_route_unknown_task(self):
        router = OllamaRouter()
        result = router.route("nonexistent", "test")
        assert not result["ok"]
        assert "unknown task_type" in result["quality_reason"]

    @patch.object(OllamaClient, "chat")
    def test_route_plain_text_task(self, mock_chat):
        mock_chat.return_value = "This is a detailed summary of the episodes covering various topics. " * 2
        router = OllamaRouter()
        result = router.route("compress_episodes", "test prompt")
        assert result["ok"]

    @patch.object(OllamaClient, "chat")
    def test_route_deepseek_think_tags(self, mock_chat):
        """DeepSeek R1 output with <think> tags should be cleaned."""
        inner = json.dumps([{"fact": "Codi uses PE", "confidence": 0.9}])
        mock_chat.return_value = f"<think>Let me analyze...</think>\n{inner}"
        router = OllamaRouter()
        result = router.route("semantic_extract", "test prompt")
        assert result["ok"]
        assert "<think>" not in result["output"]


# ──────────────────────────────────────────────
# Convenience Function
# ──────────────────────────────────────────────

class TestOllamaChatCompletion:
    @patch("modules.ollama_router._use_ollama", return_value=False)
    def test_disabled_returns_none(self, mock_flag):
        result = ollama_chat_completion("edge_classify", "test")
        assert result is None

    @patch("modules.ollama_router._use_ollama", return_value=True)
    @patch.object(OllamaClient, "is_available", return_value=False)
    def test_unavailable_returns_none(self, mock_avail, mock_flag):
        result = ollama_chat_completion("edge_classify", "test")
        assert result is None

    @patch("modules.ollama_router._use_ollama", return_value=True)
    @patch.object(OllamaClient, "is_available", return_value=True)
    @patch.object(OllamaClient, "chat")
    def test_success_returns_output(self, mock_chat, mock_avail, mock_flag):
        mock_chat.return_value = json.dumps([{"type": "causes", "confidence": 0.9}])
        result = ollama_chat_completion("edge_classify", "test")
        assert result is not None
        assert "causes" in result

    @patch("modules.ollama_router._use_ollama", return_value=True)
    @patch.object(OllamaClient, "is_available", return_value=True)
    @patch.object(OllamaClient, "chat")
    def test_quality_fail_returns_none(self, mock_chat, mock_avail, mock_flag):
        mock_chat.return_value = "garbage"
        result = ollama_chat_completion("edge_classify", "test")
        assert result is None


# ──────────────────────────────────────────────
# PG Stats (mocked)
# ──────────────────────────────────────────────

class TestStats:
    def test_stats_returns_error_on_no_pg(self):
        """Stats should return error dict when PG is not available."""
        stats = get_ollama_stats()
        # Without PG, it returns an error dict
        assert "error" in stats or "total_tasks" in stats

    def test_stats_function_exists(self):
        """Verify the stats function is importable and callable."""
        assert callable(get_ollama_stats)
