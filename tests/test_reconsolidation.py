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
        assert "semantic_divergence" in result["channels"]
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

    @patch("modules.reconsolidation._embed_text")
    def test_canal4_semantic_divergence(self, mock_embed):
        """Canal 4: same entities, low cosine sim -> PE from divergence."""
        from modules.reconsolidation import detect_contradiction
        # Orthogonal vectors = low cosine similarity (different claims)
        mock_embed.side_effect = [
            [1.0, 0.0] + [0.0] * 1534,
            [0.0, 1.0] + [0.0] * 1534,
        ]
        result = detect_contradiction(
            "Docker container runtime works perfectly in production",
            "Podman container runtime works perfectly in production"
        )
        # Canal 4 should fire: same entities (container, runtime, production)
        # but different embeddings
        assert result["channels"]["semantic_divergence"] > 0
        assert result["prediction_error"] > 0

    @patch("modules.reconsolidation._embed_text")
    def test_canal4_zero_when_similar_vectors(self, mock_embed):
        """Canal 4: same entities + high cosine sim -> no divergence (agreement)."""
        from modules.reconsolidation import detect_contradiction
        # Nearly identical vectors = agreement, not contradiction
        mock_embed.return_value = [0.9, 0.1] + [0.0] * 1534
        result = detect_contradiction(
            "Docker container runtime is great for production",
            "Docker container runtime is great for production deployments"
        )
        # Canal 4 should be low: high cosine sim means agreement
        assert result["channels"]["semantic_divergence"] < 0.15


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


class TestEmotionalVulnerability:
    """Tests for emotional reconsolidation vulnerability (Nader & Einarsson 2010)."""

    @patch("modules.reconsolidation.compute_unified_activation")
    @patch("modules.reconsolidation.detect_contradiction")
    def test_emotional_memory_lower_threshold(self, mock_contradiction, mock_activation):
        """High-arousal memory should reconsolidate with lower PE."""
        from modules.reconsolidation import check_reconsolidation
        from modules.config import RECONSOLIDATION_PE_THRESHOLD

        mock_activation.return_value = MagicMock(total=0.5)

        # PE just below normal threshold but above emotional threshold
        pe_value = RECONSOLIDATION_PE_THRESHOLD * 0.85  # 15% below normal threshold
        mock_contradiction.return_value = {
            "prediction_error": pe_value,
            "detail": "test contradiction",
        }

        # Neutral memory: should NOT reconsolidate (PE below threshold)
        neutral_payload = {"data": "neutral fact", "pad_at_encoding": {"P": 0.0, "A": 0.0, "D": 0.0}}
        result_neutral = check_reconsolidation("mem1", neutral_payload, "context")
        assert result_neutral["should_reconsolidate"] is False

        # Emotional memory: SHOULD reconsolidate (lower effective threshold)
        emotional_payload = {"data": "emotional fact", "pad_at_encoding": {"P": -0.5, "A": 0.8, "D": 0.0}}
        result_emotional = check_reconsolidation("mem2", emotional_payload, "context")
        assert result_emotional["should_reconsolidate"] is True
        assert result_emotional.get("emotional_vulnerability") is True

    @patch("modules.reconsolidation.compute_unified_activation")
    @patch("modules.reconsolidation.detect_contradiction")
    def test_low_arousal_uses_normal_threshold(self, mock_contradiction, mock_activation):
        """Low-arousal memory should use the standard PE threshold."""
        from modules.reconsolidation import check_reconsolidation
        from modules.config import RECONSOLIDATION_PE_THRESHOLD

        mock_activation.return_value = MagicMock(total=0.5)
        mock_contradiction.return_value = {
            "prediction_error": RECONSOLIDATION_PE_THRESHOLD * 0.9,
            "detail": "minor",
        }

        payload = {"data": "calm fact", "pad_at_encoding": {"P": 0.2, "A": 0.1, "D": 0.3}}
        result = check_reconsolidation("mem3", payload, "context")
        assert result["should_reconsolidate"] is False


class TestBypassGuard:
    """Test bypass_guard parameter for internal callers (Proposal #179).

    Note: bypass_guard lives on _correct_memory_impl (not the MCP-exposed
    correct_memory wrapper) because FastMCP rejects params starting with '_'.
    """

    @patch("modules.destructive_guard.is_guard_enabled", return_value=True)
    def test_bypass_guard_skips_token_requirement(self, mock_guard):
        """bypass_guard=True should skip two-step confirmation even with guard enabled."""
        from modules.reconsolidation import _correct_memory_impl
        result = _correct_memory_impl(
            memory_id="nonexistent-id",
            correction="test correction",
            force=True,
            bypass_guard=True,
        )
        # Gets past the guard, fails at ID resolution (expected)
        assert "CONFIRMACION REQUERIDA" not in result

    @patch("modules.destructive_guard.request_confirmation",
           return_value={"confirm_token": "test-token-123", "ttl_seconds": 300})
    @patch("modules.destructive_guard.is_guard_enabled", return_value=True)
    def test_without_bypass_requires_token(self, mock_guard, mock_request):
        """Without bypass_guard, guard-enabled correct_memory requires token."""
        from modules.reconsolidation import correct_memory
        result = correct_memory(
            memory_id="test-id",
            correction="test correction",
            force=True,
        )
        assert "CONFIRMACION REQUERIDA" in result
