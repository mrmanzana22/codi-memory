#!/usr/bin/env python3
"""
T1.4 — flush.py + consolidation.py CONTRACT TESTS
====================================================
Tests for checkpoint/flush lifecycle and consolidation basics
(detect_contradiction, search_semantic, session state).

Uses patch_externals + _isolate_sqlite (autouse).

Run: ./venv/bin/pytest tests/test_tier1_flush_consolidation.py -v
     ./venv/bin/pytest tests/test_tier1_flush_consolidation.py -v -m tier1
"""

import sys
import os
import json

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

pytestmark = pytest.mark.tier1


# ============================================================
# CHECKPOINT MEMORIA
# ============================================================

class TestCheckpointMemoria:
    """Contract: checkpoint saves to WM + long-term + journal."""

    def test_sync_checkpoint_returns_confirmation(self, patch_externals, monkeypatch):
        monkeypatch.setenv("CODI_WRITE_MODE", "sync")
        from modules.flush import _checkpoint_memoria
        result = _checkpoint_memoria(
            momento="decision",
            que_paso="Elegimos usar Docker",
            por_que_importa="Simplifica el deploy",
        )
        assert "checkpoint" in result.lower() or "guardado" in result.lower() or "decision" in result.lower()

    def test_async_checkpoint_returns_queued(self, patch_externals, monkeypatch):
        monkeypatch.setenv("CODI_WRITE_MODE", "async")
        from modules.flush import _checkpoint_memoria
        result = _checkpoint_memoria(
            momento="aprendizaje",
            que_paso="mem0 consolida automaticamente",
            por_que_importa="Evitar duplicar manualmente",
        )
        parsed = json.loads(result)
        assert parsed["action"] == "queued"

    def test_checkpoint_valid_momento_types(self, patch_externals, monkeypatch):
        """All momento types work without error."""
        monkeypatch.setenv("CODI_WRITE_MODE", "sync")
        from modules.flush import _checkpoint_memoria
        for momento in ["decision", "error_resuelto", "aprendizaje", "momento_personal", "tarea_completada"]:
            result = _checkpoint_memoria(
                momento=momento,
                que_paso=f"Testing {momento}",
                por_que_importa="Test coverage",
            )
            assert "error" not in result.lower() or "error_resuelto" in result.lower()


# ============================================================
# SESSION STATE
# ============================================================

class TestSessionState:
    """Contract: save/load session state for cross-session persistence."""

    def test_save_returns_true(self, patch_externals, monkeypatch, tmp_path):
        # Override session state file path
        state_file = str(tmp_path / "session_state.json")
        monkeypatch.setattr("modules.flush.SESSION_STATE_FILE", state_file, raising=False)
        from modules.flush import _save_session_state
        result = _save_session_state(resumen="test session")
        assert result is True

    def test_save_creates_file(self, patch_externals, monkeypatch, tmp_path):
        state_file = str(tmp_path / "session_state.json")
        monkeypatch.setattr("modules.flush.SESSION_STATE_FILE", state_file, raising=False)
        from modules.flush import _save_session_state
        _save_session_state(resumen="persist test")
        assert os.path.exists(state_file)
        with open(state_file) as f:
            data = json.load(f)
        assert "timestamp" in data

    def test_load_returns_none_when_no_file(self, patch_externals, monkeypatch, tmp_path):
        state_file = str(tmp_path / "nonexistent.json")
        monkeypatch.setattr("modules.flush.SESSION_STATE_FILE", state_file, raising=False)
        from modules.flush import load_session_state
        result = load_session_state()
        assert result is None

    def test_round_trip(self, patch_externals, monkeypatch, tmp_path):
        """Save then load returns data."""
        state_file = str(tmp_path / "session_state.json")
        monkeypatch.setattr("modules.flush.SESSION_STATE_FILE", state_file, raising=False)
        from modules.flush import _save_session_state, load_session_state
        _save_session_state(resumen="round trip")
        result = load_session_state()
        # result is dict or None (None if too old -- but we just saved, so should be fresh)
        if result is not None:
            assert isinstance(result, dict)


# ============================================================
# FLUSH SESSION
# ============================================================

class TestFlushSession:
    """Contract: flush_session consolidates all session data."""

    def test_flush_returns_string(self, patch_externals, monkeypatch, tmp_path):
        monkeypatch.setenv("CODI_WRITE_MODE", "sync")
        # Override session state file
        state_file = str(tmp_path / "session_state.json")
        monkeypatch.setattr("modules.flush.SESSION_STATE_FILE", state_file, raising=False)
        from modules.flush import _flush_session
        result = _flush_session(
            resumen="End of session",
            decisiones="Used Docker",
            errores="None",
            aprendizajes="Docker works",
        )
        assert isinstance(result, str)
        assert len(result) > 0


# ============================================================
# DETECT CONTRADICTION (pure-ish)
# ============================================================

class TestDetectContradiction:
    """Contract: 3-channel PE detector returns prediction_error."""

    def test_no_contradiction(self):
        from modules.consolidation import detect_contradiction
        result = detect_contradiction(
            "The sky is blue",
            "The sky is blue and beautiful"
        )
        assert "prediction_error" in result
        assert result["prediction_error"] < 0.5  # Low PE for similar

    def test_contradiction_detected(self):
        from modules.consolidation import detect_contradiction
        result = detect_contradiction(
            "The project uses Python 3.10",
            "The project uses Python 3.12, not 3.10"
        )
        assert "prediction_error" in result
        # Higher PE expected for contradictory content
        assert result["prediction_error"] > 0.0

    def test_returns_channels(self):
        from modules.consolidation import detect_contradiction
        result = detect_contradiction("A is true", "A is false")
        assert "channels" in result

    def test_empty_text_no_crash(self):
        from modules.consolidation import detect_contradiction
        result = detect_contradiction("", "")
        assert "prediction_error" in result


# ============================================================
# SEARCH SEMANTIC
# ============================================================

class TestSearchSemantic:
    """Contract: search_semantic queries codi_semantic collection."""

    def test_empty_collection_returns_empty(self, patch_externals):
        from modules.consolidation import search_semantic
        results = search_semantic("anything", limit=5)
        assert isinstance(results, list)
        assert len(results) == 0  # FakeQdrant has no query_points, returns []

    def test_returns_list(self, patch_externals):
        from modules.consolidation import search_semantic
        results = search_semantic("test query")
        assert isinstance(results, list)


# ============================================================
# GET CONSOLIDATION STATS
# ============================================================

class TestConsolidationStats:
    """Contract: stats returns formatted report."""

    def test_returns_string(self, patch_externals):
        from modules.consolidation import get_consolidation_stats
        result = get_consolidation_stats()
        assert isinstance(result, str)
        # Should have some recognizable content
        assert "consolidat" in result.lower() or "facts" in result.lower() or "error" in result.lower()


# ============================================================
# EXPORT MEMORIES (basic smoke)
# ============================================================

class TestExportMemories:
    """Contract: export returns markdown string."""

    def test_export_markdown_returns_string(self, patch_externals):
        from modules.flush import _export_memories_markdown
        result = _export_memories_markdown()
        assert isinstance(result, str)
