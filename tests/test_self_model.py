"""Tests for modules/self_model.py.

Covers:
  - reflect_on_self (with mocked qdrant)
  - assess_confidence (with mocked memory + qdrant)
  - update_self_model (with mocked memory + qdrant)
  - get_self_model_summary (with mocked qdrant)
  - assess_butlin_indicators (delegates to assessment.py)
"""

import json
import pytest
from unittest.mock import patch, MagicMock
from tests.conftest import FakeQdrantPoint


class TestReflectOnSelf:
    """Tests for reflect_on_self."""

    def test_no_memories_returns_message(self, patch_externals, monkeypatch):
        # scroll_all uses qdrant imported in modules.qdrant_utils, which
        # patch_externals doesn't cover. Patch it explicitly.
        monkeypatch.setattr("modules.qdrant_utils.qdrant", patch_externals["qdrant"])
        from modules.self_model import reflect_on_self
        result = reflect_on_self()
        assert "No encontre memorias" in result

    def test_with_self_ref_memories(self, patch_externals):
        from modules.self_model import reflect_on_self

        # Add fake self-referential memories
        fake_qdrant = patch_externals["qdrant"]
        points = [
            FakeQdrantPoint("m1", {
                'data': 'Puedo programar en Python',
                'self_reference': True,
                'ownership_source': 'experienced',
            }),
            FakeQdrantPoint("m2", {
                'data': 'No puedo hacer deploy sin revision',
                'self_reference': True,
                'ownership_source': 'learned',
            }),
        ]

        original_scroll = fake_qdrant.scroll
        def mock_scroll(collection_name, scroll_filter=None, limit=10, **kwargs):
            if scroll_filter:
                return (points, None)
            return ([], None)

        fake_qdrant.scroll = mock_scroll

        result = reflect_on_self()
        assert "REFLEXION" in result


class TestAssessConfidence:
    """Tests for assess_confidence."""

    def test_no_memories_returns_zero(self, patch_externals):
        from modules.self_model import assess_confidence
        result = assess_confidence("quantum computing")
        assert "confianza es 0" in result or "No tengo memorias" in result

    def test_with_memories(self, patch_externals):
        from modules.self_model import assess_confidence
        from modules.config import USER_ID

        fake_mem0 = patch_externals["mem0"]
        fake_qdrant = patch_externals["qdrant"]

        # Add searchable memory (use actual USER_ID from config)
        fake_mem0.add(content="Python es mi lenguaje principal", user_id=USER_ID)

        # Mock qdrant retrieve to return points with ownership data
        points = [
            FakeQdrantPoint("p1", {
                'data': 'Python es mi lenguaje principal',
                'ownership_source': 'experienced',
                'ownership_confidence': 0.9,
            }),
        ]
        fake_qdrant.retrieve = lambda *a, **kw: points

        result = assess_confidence("Python")
        assert "Evaluacion de Confianza" in result or "Score" in result


class TestUpdateSelfModel:
    """Tests for update_self_model."""

    def test_update_capacidad(self, patch_externals):
        from modules.self_model import update_self_model
        result = update_self_model("Puedo analizar patrones de trading", aspect="capacidad")
        assert "Self-model actualizado" in result
        assert "capacidad" in result

    def test_invalid_aspect_defaults_to_general(self, patch_externals):
        from modules.self_model import update_self_model
        result = update_self_model("Something", aspect="invalid")
        assert "Self-model actualizado" in result

    def test_valid_aspects(self, patch_externals):
        from modules.self_model import update_self_model
        for aspect in ['capacidad', 'limitacion', 'valor', 'preferencia', 'general']:
            result = update_self_model(f"Test {aspect}", aspect=aspect)
            assert "error" not in result.lower()


class TestGetSelfModelSummary:
    """Tests for get_self_model_summary."""

    def test_no_points_returns_message(self, patch_externals, monkeypatch):
        # scroll_all uses qdrant imported in modules.qdrant_utils, which
        # patch_externals doesn't cover. Patch it explicitly.
        monkeypatch.setattr("modules.qdrant_utils.qdrant", patch_externals["qdrant"])
        from modules.self_model import get_self_model_summary
        result = get_self_model_summary()
        assert "No tengo un self-model" in result

    def test_with_points(self, patch_externals):
        from modules.self_model import get_self_model_summary
        fake_qdrant = patch_externals["qdrant"]

        points = [
            FakeQdrantPoint("m1", {
                'data': 'Puedo codear rapido',
                'self_reference': True,
                'self_model_aspect': 'capacidad',
                'ownership_source': 'experienced',
                'ownership_confidence': 0.9,
            }),
        ]

        original_scroll = fake_qdrant.scroll
        def mock_scroll(collection_name, scroll_filter=None, **kwargs):
            if scroll_filter:
                return (points, None)
            return original_scroll(collection_name, **kwargs)

        fake_qdrant.scroll = mock_scroll

        result = get_self_model_summary()
        assert "MI SELF-MODEL" in result
        assert "puedo hacer" in result.lower() or "capacidad" in result.lower() or "codear" in result.lower()


class TestAssessButlinIndicators:
    """Tests for assess_butlin_indicators (delegates to assessment.py)."""

    def test_returns_formatted_assessment(self, patch_externals):
        from modules.self_model import assess_butlin_indicators
        result = assess_butlin_indicators()
        assert "Butlin" in result or "Assessment" in result or "Score" in result
