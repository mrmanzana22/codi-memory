"""Tests for modules/self_model.py.

Covers:
  - reflect_on_self (with mocked pg_store)
  - assess_confidence (with mocked pg_store + search_with_fts_content)
  - update_self_model (with mocked pg_store)
  - get_self_model_summary (with mocked pg_store)
  - assess_butlin_indicators (delegates to assessment.py)

Backend: pg_store (PostgreSQL + pgvector).
Migrated from Qdrant/mem0 mocks in Fase 2.
"""

import json
import pytest
from unittest.mock import patch, MagicMock
from modules.pg_store import Point, CollectionInfo


def _make_point(pid, payload, score=0.0):
    """Helper to create a Point object compatible with pg_store."""
    return Point(id=pid, payload=payload, score=score)


class TestReflectOnSelf:
    """Tests for reflect_on_self."""

    @patch("modules.self_model.pg")
    def test_no_memories_returns_message(self, mock_pg):
        # All scroll calls return empty
        mock_pg.scroll.return_value = ([], None)

        from modules.self_model import reflect_on_self
        result = reflect_on_self(update_attention=False)
        assert "No encontre memorias" in result

    @patch("modules.self_model.pg")
    def test_with_self_ref_memories(self, mock_pg):
        points = [
            _make_point("m1", {
                'data': 'Puedo programar en Python',
                'self_reference': True,
                'ownership_source': 'experienced',
            }),
            _make_point("m2", {
                'data': 'No puedo hacer deploy sin revision',
                'self_reference': True,
                'ownership_source': 'learned',
            }),
        ]

        def mock_scroll(filters=None, limit=50, is_semantic=None, **kwargs):
            # Return points for the self_reference filter, empty for others
            if filters and "metadata_key" in filters:
                mk = filters["metadata_key"]
                if mk.get("key") == "self_reference":
                    return (points, None)
            return ([], None)

        mock_pg.scroll.side_effect = mock_scroll

        from modules.self_model import reflect_on_self
        result = reflect_on_self(update_attention=False)
        assert "REFLEXION" in result


class TestAssessConfidence:
    """Tests for assess_confidence."""

    @patch("modules.self_model.search_with_fts_content")
    @patch("modules.self_model.pg")
    def test_no_memories_returns_zero(self, mock_pg, mock_search):
        # search_with_fts_content returns no results
        mock_search.return_value = {"results": []}

        from modules.self_model import assess_confidence
        result = assess_confidence("quantum computing")
        assert "confianza es 0" in result or "No tengo memorias" in result

    @patch("modules.self_model.search_with_fts_content")
    @patch("modules.self_model.pg")
    def test_with_memories(self, mock_pg, mock_search):
        # search returns a memory hit
        mock_search.return_value = {
            "results": [
                {"id": "p1", "memory": "Python es mi lenguaje principal", "score": 0.8}
            ]
        }

        # pg.get_by_ids returns points with ownership data
        points = [
            _make_point("p1", {
                'data': 'Python es mi lenguaje principal',
                'ownership_source': 'experienced',
                'ownership_confidence': 0.9,
                'attention_access_count': 0,
            }),
        ]
        mock_pg.get_by_ids.return_value = points
        mock_pg.update_payload.return_value = True

        from modules.self_model import assess_confidence
        result = assess_confidence("Python")
        assert "Evaluacion de Confianza" in result or "Score" in result


class TestUpdateSelfModel:
    """Tests for update_self_model."""

    @patch("modules.self_model.pg")
    def test_update_capacidad(self, mock_pg):
        mock_pg.add.return_value = {"results": [{"id": "new-1"}]}
        mock_pg.update_payload.return_value = True

        from modules.self_model import update_self_model
        result = update_self_model("Puedo analizar patrones de trading", aspect="capacidad")
        assert "Self-model actualizado" in result
        assert "capacidad" in result

    @patch("modules.self_model.pg")
    def test_invalid_aspect_defaults_to_general(self, mock_pg):
        mock_pg.add.return_value = {"results": [{"id": "new-2"}]}
        mock_pg.update_payload.return_value = True

        from modules.self_model import update_self_model
        result = update_self_model("Something", aspect="invalid")
        assert "Self-model actualizado" in result

    @patch("modules.self_model.pg")
    def test_valid_aspects(self, mock_pg):
        mock_pg.add.return_value = {"results": [{"id": "new-3"}]}
        mock_pg.update_payload.return_value = True

        from modules.self_model import update_self_model
        for aspect in ['capacidad', 'limitacion', 'valor', 'preferencia', 'general']:
            result = update_self_model(f"Test {aspect}", aspect=aspect)
            assert "error" not in result.lower()


class TestGetSelfModelSummary:
    """Tests for get_self_model_summary."""

    @patch("modules.self_model.pg")
    def test_no_points_returns_message(self, mock_pg):
        mock_pg.scroll.return_value = ([], None)

        from modules.self_model import get_self_model_summary
        result = get_self_model_summary()
        assert "No tengo un self-model" in result

    @patch("modules.self_model.pg")
    def test_with_points(self, mock_pg):
        points = [
            _make_point("m1", {
                'data': 'Puedo codear rapido',
                'self_reference': True,
                'self_model_aspect': 'capacidad',
                'ownership_source': 'experienced',
                'ownership_confidence': 0.9,
                'attention_access_count': 0,
            }),
        ]

        mock_pg.scroll.return_value = (points, None)
        mock_pg.update_payload.return_value = True

        from modules.self_model import get_self_model_summary
        result = get_self_model_summary()
        assert "MI SELF-MODEL" in result
        assert "puedo hacer" in result.lower() or "capacidad" in result.lower() or "codear" in result.lower()


class TestAssessButlinIndicators:
    """Tests for assess_butlin_indicators (delegates to assessment.py)."""

    def test_returns_formatted_assessment(self, patch_externals):
        from modules.self_model import assess_butlin_indicators
        result = assess_butlin_indicators()
        assert "Butlin" in result or "Assessment" in result or "Score" in result
