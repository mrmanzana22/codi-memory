"""Tests for modules/reconsolidation.py.

Covers:
  - _extract_key_entities (pure function)
  - detect_contradiction (3-channel detection, pure)
  - check_reconsolidation (with mocked activation)
  - mark_as_labile / clear_expired_labile (SQLite)
"""

import pytest
from unittest.mock import patch, MagicMock
from datetime import datetime, timedelta


class TestExtractKeyEntities:
    """Tests for _extract_key_entities (pure heuristic)."""

    def test_filters_short_words(self):
        from modules.reconsolidation import _extract_key_entities
        result = _extract_key_entities("el sol es grande")
        # "el", "sol", "es" are <= 4 chars, only "grande" passes
        assert "grande" in result
        assert "sol" not in result

    def test_filters_stopwords(self):
        from modules.reconsolidation import _extract_key_entities
        result = _extract_key_entities("para hacer desde hasta entre sobre")
        assert len(result) == 0

    def test_extracts_meaningful_entities(self):
        from modules.reconsolidation import _extract_key_entities
        result = _extract_key_entities("Trading en Kraken con Bitcoin y Ethereum")
        assert "trading" in result
        assert "kraken" in result
        assert "bitcoin" in result
        assert "ethereum" in result

    def test_strips_punctuation(self):
        from modules.reconsolidation import _extract_key_entities
        result = _extract_key_entities("proyecto(fullempaques): cotizador,precios.")
        assert "proyecto" in result or "fullempaques" in result

    def test_filters_verb_suffixes(self):
        from modules.reconsolidation import _extract_key_entities
        result = _extract_key_entities("implementación actualizando corrieron")
        # These end in -ción, -ando, -ieron → filtered
        assert len(result) == 0


class TestDetectContradiction:
    """Tests for detect_contradiction (3-channel PE detection)."""

    def test_no_contradiction_unrelated_texts(self):
        from modules.reconsolidation import detect_contradiction
        result = detect_contradiction(
            "El cielo es azul", "Los gatos son mascotas"
        )
        assert result["prediction_error"] == 0.0
        assert result["detail"] is None

    def test_empty_inputs_return_zero(self):
        from modules.reconsolidation import detect_contradiction
        assert detect_contradiction("", "algo")["prediction_error"] == 0.0
        assert detect_contradiction("algo", "")["prediction_error"] == 0.0
        assert detect_contradiction("", "")["prediction_error"] == 0.0

    def test_canal1_keyword_detection(self):
        from modules.reconsolidation import detect_contradiction
        result = detect_contradiction(
            "Usamos Docker para Qdrant",
            "Correccion: ya no usamos Docker, migramos a cloud"
        )
        # "correccion" and "migramos" are CORRECTION_PATTERNS
        assert result["channels"]["keywords"] > 0

    def test_canal3_negation_detection(self):
        from modules.reconsolidation import detect_contradiction
        # Same entities but one has negation
        result = detect_contradiction(
            "Kraken permite trading automatico",
            "Kraken nunca permite trading automatico"
        )
        assert result["channels"]["negation"] > 0

    def test_returns_channels_dict(self):
        from modules.reconsolidation import detect_contradiction
        result = detect_contradiction("test memory", "test context")
        assert "keywords" in result["channels"]
        assert "topic_confirmation" in result["channels"]
        assert "negation" in result["channels"]
        assert "shared_entities" in result["channels"]

    @patch("modules.reconsolidation._embed_text")
    def test_topic_confirmation_with_shared_entities(self, mock_embed):
        from modules.reconsolidation import detect_contradiction
        # Mock embeddings to return similar vectors
        mock_embed.return_value = [1.0, 0.0, 0.0]
        result = detect_contradiction(
            "El proyecto fullempaques tiene errores graves",
            "Correccion: fullempaques ya no tiene errores graves"
        )
        # Should detect via canal1 (keywords) + canal2 (topic) + canal3 (negation)
        assert result["prediction_error"] > 0


class TestCheckReconsolidation:
    """Tests for check_reconsolidation (integration with activation)."""

    @patch("modules.reconsolidation.compute_unified_activation")
    def test_too_weak_memory_rejected(self, mock_activation):
        from modules.reconsolidation import check_reconsolidation, RECONSOLIDATION_STRENGTH_FLOOR
        mock_activation.return_value = MagicMock(total=RECONSOLIDATION_STRENGTH_FLOOR - 0.01)
        result = check_reconsolidation("mem-1", {}, "some context")
        assert result["should_reconsolidate"] is False
        assert result["reason"] == "too_weak"

    @patch("modules.reconsolidation.compute_unified_activation")
    def test_too_strong_memory_rejected(self, mock_activation):
        from modules.reconsolidation import check_reconsolidation, RECONSOLIDATION_STRENGTH_CEILING
        mock_activation.return_value = MagicMock(total=RECONSOLIDATION_STRENGTH_CEILING + 0.01)
        result = check_reconsolidation("mem-1", {}, "some context")
        assert result["should_reconsolidate"] is False
        assert result["reason"] == "too_strong"

    @patch("modules.reconsolidation.detect_contradiction")
    @patch("modules.reconsolidation.compute_unified_activation")
    def test_no_pe_rejected(self, mock_activation, mock_contradiction):
        from modules.reconsolidation import check_reconsolidation
        mock_activation.return_value = MagicMock(total=0.5)
        mock_contradiction.return_value = {
            "prediction_error": 0.0, "detail": None, "channels": {}
        }
        result = check_reconsolidation("mem-1", {"data": "old"}, "new context")
        assert result["should_reconsolidate"] is False
        assert result["reason"] == "no_prediction_error"


class TestLabileMemory:
    """Tests for mark_as_labile and clear_expired_labile (SQLite)."""

    def test_mark_as_labile(self):
        from modules.reconsolidation import mark_as_labile, _consolidation_conn
        result = mark_as_labile("test-mem-1", prediction_error=0.7, trigger_context="test")
        assert result is True

        # Verify in DB
        conn = _consolidation_conn()
        row = conn.execute(
            "SELECT * FROM labile_memories WHERE memory_id = ?", ("test-mem-1",)
        ).fetchone()
        conn.close()
        assert row is not None

    def test_clear_expired_labile(self):
        from modules.reconsolidation import mark_as_labile, clear_expired_labile, _consolidation_conn

        # Insert a labile memory with expired window
        conn = _consolidation_conn()
        expired_time = (datetime.now() - timedelta(hours=1)).isoformat()
        conn.execute("""
            INSERT OR REPLACE INTO labile_memories
            (memory_id, marked_at, window_expires, prediction_error, trigger_context)
            VALUES (?, ?, ?, ?, ?)
        """, ("expired-mem", datetime.now().isoformat(), expired_time, 0.5, "test"))
        conn.commit()
        conn.close()

        clear_expired_labile()

        conn = _consolidation_conn()
        row = conn.execute(
            "SELECT * FROM labile_memories WHERE memory_id = ?", ("expired-mem",)
        ).fetchone()
        conn.close()
        assert row is None
