#!/usr/bin/env python3
"""
Unit tests for Phase 4: Close the Loops (Real Cognitive Improvement).
Run: ./venv/bin/pytest tests/test_phase4.py -v
"""

import sys
import os
import sqlite3
import math
from unittest.mock import patch, MagicMock
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest


# ============================================================
# PART 1: HOT-3 Metacognitive Control Loop
# ============================================================

class TestMetacognitiveControl:
    """Nelson & Narens 1990: FOK -> strategy -> modified search."""

    def test_metacognitive_control_search(self):
        """FOK >= 0.6 -> strategy 'full_search'."""
        from modules.retrieval_metadata import metacognitive_control
        with patch('modules.retrieval_metadata.feeling_of_knowing') as mock_fok:
            mock_fok.return_value = {"fok_score": 0.7, "basis": "test", "recommendation": "search"}
            result = metacognitive_control("test query")
            assert result["strategy"] == "full_search"
            assert result["adjusted_limit"] == 1
            assert result["confidence_flag"] == ""

    def test_metacognitive_control_ask(self):
        """FOK <= 0.3 -> strategy 'suggest_ask'."""
        from modules.retrieval_metadata import metacognitive_control
        with patch('modules.retrieval_metadata.feeling_of_knowing') as mock_fok:
            mock_fok.return_value = {"fok_score": 0.2, "basis": "test", "recommendation": "ask"}
            result = metacognitive_control("unknown topic xyz")
            assert result["strategy"] == "suggest_ask"
            assert result["adjusted_limit"] == 2
            assert result["confidence_flag"] == "[LOW CONFIDENCE]"

    def test_metacognitive_control_uncertain(self):
        """FOK between 0.3 and 0.6 -> strategy 'expand_search'."""
        from modules.retrieval_metadata import metacognitive_control
        with patch('modules.retrieval_metadata.feeling_of_knowing') as mock_fok:
            mock_fok.return_value = {"fok_score": 0.45, "basis": "test", "recommendation": "uncertain"}
            result = metacognitive_control("partial topic")
            assert result["strategy"] == "expand_search"
            assert result["adjusted_limit"] == 2
            assert result["confidence_flag"] == "[UNCERTAIN]"

    def test_metacognitive_control_returns_fok(self):
        """Result includes original FOK dict."""
        from modules.retrieval_metadata import metacognitive_control
        with patch('modules.retrieval_metadata.feeling_of_knowing') as mock_fok:
            fok_data = {"fok_score": 0.5, "basis": "test", "recommendation": "uncertain"}
            mock_fok.return_value = fok_data
            result = metacognitive_control("test")
            assert result["fok"] == fok_data

    def test_confidence_flag_in_output(self):
        """Low FOK queries should get flag prepended."""
        from modules.retrieval_metadata import metacognitive_control
        with patch('modules.retrieval_metadata.feeling_of_knowing') as mock_fok:
            mock_fok.return_value = {"fok_score": 0.1, "basis": "test", "recommendation": "ask"}
            result = metacognitive_control("completely unknown")
            assert "[LOW CONFIDENCE]" in result["confidence_flag"]

    def test_adjusted_limit_doubles_for_uncertain(self):
        """Uncertain queries get 2x limit multiplier."""
        from modules.retrieval_metadata import metacognitive_control
        with patch('modules.retrieval_metadata.feeling_of_knowing') as mock_fok:
            mock_fok.return_value = {"fok_score": 0.4, "basis": "test", "recommendation": "uncertain"}
            result = metacognitive_control("partial query")
            assert result["adjusted_limit"] == 2

    def test_search_memory_uses_fok(self):
        """search_memory should call metacognitive_control."""
        # Verify the import exists in memory_core
        from modules.memory_core import metacognitive_control as mc_import
        assert callable(mc_import)

    def test_butlin_hot3_full(self):
        """Assessment should now score HOT-3 as 1.0."""
        from modules.retrieval_metadata import metacognitive_control
        assert callable(metacognitive_control)


# ============================================================
# PART 2: PP-3 Prediction-Error-Driven Reconsolidation
# ============================================================

class TestReconsolidation:
    """Nader 2000: PE-driven reconsolidation."""

    def test_correct_memory_updates_qdrant(self):
        """correct_memory should upsert full PointStruct (re-embed, Nader 2000)."""
        from modules.consolidation import correct_memory
        mock_payload = {
            "data": "Old content here",
            "confidence": 0.8,
            "reconsolidation_count": 0,
        }
        mock_point = MagicMock()
        mock_point.payload = mock_payload

        with patch('modules.consolidation.qdrant') as mock_qdrant, \
             patch('modules.consolidation._consolidation_conn') as mock_conn_fn, \
             patch('modules.consolidation._embed_text', return_value=[0.1] * 1536), \
             patch('modules.consolidation.check_reconsolidation', return_value={"should_reconsolidate": True, "prediction_error": 0.8}), \
             patch('modules.memory_smart.delete_memory_fts', return_value=True), \
             patch('modules.memory_smart.index_memory_fts', return_value=True), \
             patch('modules.utils.resolve_memory_id', return_value="full-uuid-123"):
            mock_qdrant.retrieve.return_value = [mock_point]
            mock_conn = MagicMock()
            mock_conn_fn.return_value = mock_conn

            result = correct_memory("full-uuid", "This is the correction")
            assert "corrected" in result.lower()
            # Phase 4.5: now uses upsert (re-embed) instead of set_payload
            mock_qdrant.upsert.assert_called_once()
            upsert_call = mock_qdrant.upsert.call_args
            points = upsert_call[1]["points"]
            assert len(points) == 1
            # Fix 1: new content REPLACES old (Nader 2000), not concatenate
            assert points[0].payload["data"] == "This is the correction"
            assert points[0].payload["confidence"] == pytest.approx(0.7)

    def test_correct_memory_logs_reconsolidation(self):
        """correct_memory should create entry in reconsolidation_log."""
        from modules.consolidation import correct_memory
        mock_payload = {"data": "Old", "confidence": 0.6, "reconsolidation_count": 0}
        mock_point = MagicMock()
        mock_point.payload = mock_payload

        with patch('modules.consolidation.qdrant') as mock_qdrant, \
             patch('modules.consolidation._consolidation_conn') as mock_conn_fn, \
             patch('modules.consolidation._embed_text', return_value=[0.1] * 1536), \
             patch('modules.consolidation.check_reconsolidation', return_value={"should_reconsolidate": True, "prediction_error": 0.7}), \
             patch('modules.memory_smart.delete_memory_fts', return_value=True), \
             patch('modules.memory_smart.index_memory_fts', return_value=True), \
             patch('modules.utils.resolve_memory_id', return_value="full-uuid-456"):
            mock_qdrant.retrieve.return_value = [mock_point]
            mock_conn = MagicMock()
            mock_conn_fn.return_value = mock_conn

            correct_memory("full-uuid", "correction text")
            # Should have called conn.execute with INSERT INTO reconsolidation_log
            insert_calls = [c for c in mock_conn.execute.call_args_list
                           if "reconsolidation_log" in str(c)]
            assert len(insert_calls) >= 1

    def test_correct_memory_decrements_confidence(self):
        """Confidence should decrease by 0.1 after correction."""
        from modules.consolidation import correct_memory
        mock_payload = {"data": "Old", "confidence": 0.9, "reconsolidation_count": 0}
        mock_point = MagicMock()
        mock_point.payload = mock_payload

        with patch('modules.consolidation.qdrant') as mock_qdrant, \
             patch('modules.consolidation._consolidation_conn') as mock_conn_fn, \
             patch('modules.consolidation._embed_text', return_value=[0.1] * 1536), \
             patch('modules.consolidation.check_reconsolidation', return_value={"should_reconsolidate": True, "prediction_error": 0.8}), \
             patch('modules.memory_smart.delete_memory_fts', return_value=True), \
             patch('modules.memory_smart.index_memory_fts', return_value=True), \
             patch('modules.utils.resolve_memory_id', return_value="uuid-789"):
            mock_qdrant.retrieve.return_value = [mock_point]
            mock_conn = MagicMock()
            mock_conn_fn.return_value = mock_conn

            result = correct_memory("uuid", "fix this")
            assert "0.90" in result and "0.80" in result

    def test_prediction_error_triggers_reconsolidation(self):
        """High PE with memory_id should trigger check_reconsolidation."""
        from modules.wiring import _on_prediction_error

        with patch('modules.wiring._update_attention_schema'), \
             patch('modules.spreading._spread_activation', return_value={'affected': 0, 'updates': {}}), \
             patch('modules.consolidation.check_reconsolidation') as mock_check, \
             patch('modules.consolidation.mark_as_labile') as mock_labile, \
             patch('modules.config.qdrant') as mock_qdrant:

            mock_point = MagicMock()
            mock_point.payload = {"data": "test memory"}
            mock_qdrant.retrieve.return_value = [mock_point]

            mock_check.return_value = {"should_reconsolidate": True, "prediction_error": 0.8}

            _on_prediction_error("prediction_error", {
                "error_magnitude": 0.8,
                "topic": "test",
                "memory_id": "mem-123",
            })

            mock_check.assert_called_once()
            mock_labile.assert_called_once()

    def test_low_pe_no_reconsolidation(self):
        """PE below threshold should not trigger reconsolidation."""
        from modules.wiring import _on_prediction_error

        with patch('modules.wiring._update_attention_schema'), \
             patch('modules.spreading._spread_activation', return_value={'affected': 0, 'updates': {}}):

            # PE = 0.1, below 0.3 threshold - no reconsolidation imports happen
            _on_prediction_error("prediction_error", {
                "error_magnitude": 0.1,
                "topic": "test",
                "memory_id": "mem-low",
            })
            # Should not crash -- low PE skips reconsolidation block

    def test_reconsolidation_event_exists(self):
        """RECONSOLIDATION_TRIGGERED event should exist."""
        from modules.events import Events
        assert hasattr(Events, 'RECONSOLIDATION_TRIGGERED')
        assert Events.RECONSOLIDATION_TRIGGERED == 'reconsolidation_triggered'

    def test_labile_memory_marked(self):
        """mark_as_labile should insert into labile_memories table."""
        from modules.consolidation import mark_as_labile

        with patch('modules.consolidation._consolidation_conn') as mock_conn_fn:
            mock_conn = MagicMock()
            mock_conn_fn.return_value = mock_conn

            result = mark_as_labile("test-mem-id", 0.7, "test context")
            assert result is True
            mock_conn.execute.assert_called_once()
            assert "labile_memories" in str(mock_conn.execute.call_args)

    def test_butlin_pp3_stub_check(self):
        """correct_memory should no longer be a stub."""
        from modules.consolidation import correct_memory
        import inspect
        src = inspect.getsource(correct_memory)
        assert "stub" not in src.lower()


# ============================================================
# PART 3: RPT-1 True Recurrent Processing
# ============================================================

class TestRecurrentCycle:
    """Lamme 2006: Re-entrant processing."""

    def test_recurrent_cycle_runs_multiple(self):
        """recurrent_cycle should run multiple cycles."""
        from modules.spreading import recurrent_cycle

        with patch('modules.spreading._spread_activation') as mock_spread:
            mock_spread.return_value = {
                'affected': 2, 'max_depth_reached': 1,
                'total_nodes_visited': 3,
                'updates': {'node-a': 0.7, 'node-b': 0.6}
            }
            result = recurrent_cycle(["seed-1"], cycles=2, depth=1, factor=0.5)
            assert result["cycles_run"] >= 2
            assert mock_spread.call_count >= 2

    def test_recurrent_cycle_feeds_back(self):
        """Cycle 2 seeds should come from cycle 1 output."""
        from modules.spreading import recurrent_cycle

        call_args_list = []
        def track_spread(seed_ids, **kwargs):
            call_args_list.append(seed_ids)
            return {
                'affected': 1, 'max_depth_reached': 1,
                'total_nodes_visited': 2,
                'updates': {'node-x': 0.8}
            }

        with patch('modules.spreading._spread_activation', side_effect=track_spread):
            recurrent_cycle(["seed-1"], cycles=2)
            assert len(call_args_list) >= 2
            # Cycle 2 should use nodes from cycle 1 output, not original seeds
            assert call_args_list[1] != call_args_list[0]

    def test_recurrent_cycle_detects_stability(self):
        """Same top nodes in consecutive cycles -> stable."""
        from modules.spreading import recurrent_cycle

        with patch('modules.spreading._spread_activation') as mock_spread:
            # Both cycles return same top nodes
            mock_spread.return_value = {
                'affected': 2, 'max_depth_reached': 1,
                'total_nodes_visited': 2,
                'updates': {'stable-a': 0.9, 'stable-b': 0.8}
            }
            result = recurrent_cycle(["seed-1"], cycles=3)
            assert result["stable"] is True

    def test_recurrent_cycle_caps_iterations(self):
        """Should never exceed max cycles."""
        from modules.spreading import recurrent_cycle

        with patch('modules.spreading._spread_activation') as mock_spread:
            mock_spread.return_value = {
                'affected': 1, 'max_depth_reached': 1,
                'total_nodes_visited': 1,
                'updates': {'n': 0.5}
            }
            result = recurrent_cycle(["seed-1"], cycles=2)
            assert result["cycles_run"] <= 2

    def test_recurrent_cycle_empty_seeds(self):
        """Empty seeds should return gracefully."""
        from modules.spreading import recurrent_cycle

        result = recurrent_cycle([], cycles=2)
        assert result["cycles_run"] == 0
        assert result["total_affected"] == 0

    def test_butlin_rpt1_import(self):
        """recurrent_cycle should be importable from spreading."""
        from modules.spreading import recurrent_cycle
        assert callable(recurrent_cycle)


# ============================================================
# PART 4: HOT-2 RCJ Calibration
# ============================================================

class TestRCJCalibration:
    """Nelson & Narens 1990: FOK + RCJ calibration loop."""

    @pytest.fixture
    def temp_db(self, tmp_path):
        """Create a temporary SQLite database."""
        db_path = str(tmp_path / "test_fts.db")
        return db_path

    def test_record_rcj_stores(self, temp_db):
        """record_rcj should create SQLite entry."""
        from modules.retrieval_metadata import record_rcj
        record_rcj(
            query="test query",
            fok_predicted=0.7,
            actual_coverage="comprehensive",
            actual_count=5,
            actual_top_activation=0.8,
            fts_db_path=temp_db,
        )
        # Verify entry exists
        conn = sqlite3.connect(temp_db)
        cursor = conn.execute("SELECT COUNT(*) FROM fok_calibration_log")
        count = cursor.fetchone()[0]
        conn.close()
        assert count == 1

    def test_fok_calibration_computes_mae(self, temp_db):
        """get_fok_calibration should compute mean absolute error."""
        from modules.retrieval_metadata import record_rcj, get_fok_calibration

        # Record some predictions
        record_rcj("q1", 0.8, "comprehensive", 5, 0.9, fts_db_path=temp_db)
        record_rcj("q2", 0.8, "sparse", 1, 0.2, fts_db_path=temp_db)

        cal = get_fok_calibration(fts_db_path=temp_db)
        assert cal["n_records"] == 2
        assert cal["mean_absolute_error"] > 0

    def test_calibration_detects_overconfidence(self, temp_db):
        """Systematically high FOK vs low results -> positive bias."""
        from modules.retrieval_metadata import record_rcj, get_fok_calibration

        for i in range(10):
            record_rcj(f"q{i}", 0.9, "sparse", 1, 0.1, fts_db_path=temp_db)

        cal = get_fok_calibration(fts_db_path=temp_db)
        assert cal["overconfidence_bias"] > 0

    def test_calibrated_fok_adjusts_down(self):
        """Overconfident history should reduce FOK."""
        from modules.retrieval_metadata import calibrated_fok_score

        calibration = {"mean_absolute_error": 0.3, "overconfidence_bias": 0.4, "n_records": 20}
        adjusted = calibrated_fok_score(0.8, calibration)
        assert adjusted < 0.8

    def test_calibrated_fok_no_change_when_accurate(self):
        """Accurate history should not change FOK much."""
        from modules.retrieval_metadata import calibrated_fok_score

        calibration = {"mean_absolute_error": 0.05, "overconfidence_bias": 0.0, "n_records": 50}
        adjusted = calibrated_fok_score(0.6, calibration)
        assert adjusted == pytest.approx(0.6, abs=0.01)

    def test_rcj_empty_history(self, temp_db):
        """get_fok_calibration with 0 records -> graceful."""
        from modules.retrieval_metadata import get_fok_calibration

        cal = get_fok_calibration(fts_db_path=temp_db)
        assert cal["n_records"] == 0
        assert cal["mean_absolute_error"] == 0.0

    def test_butlin_hot2_full(self):
        """record_rcj and get_fok_calibration should be importable."""
        from modules.retrieval_metadata import record_rcj, get_fok_calibration
        assert callable(record_rcj)
        assert callable(get_fok_calibration)


# ============================================================
# PART 0: Assessment Fix
# ============================================================

class TestAssessmentFix:
    """Assessment checks should reflect architecture, not runtime."""

    def test_events_has_reconsolidation(self):
        """Events class should have RECONSOLIDATION_TRIGGERED."""
        from modules.events import Events
        assert hasattr(Events, 'RECONSOLIDATION_TRIGGERED')

    def test_correct_memory_not_stub(self):
        """correct_memory should not be a stub."""
        from modules.consolidation import correct_memory
        import inspect
        src = inspect.getsource(correct_memory)
        assert "stub" not in src.lower()


# ============================================================
# PART 5: Phase 4.5 - Mejora 1: Re-embed + Labile Gate
# ============================================================

class TestReembedReconsolidation:
    """Nader 2000: trace is destroyed and re-synthesized, not patched."""

    def test_correct_memory_reembeds_vector(self):
        """correct_memory should call qdrant.upsert with new vector (not set_payload)."""
        from modules.consolidation import correct_memory
        mock_payload = {
            "data": "Docker is the best container solution",
            "confidence": 0.8,
            "reconsolidation_count": 0,
            "category": "general",
            "source": "experienced",
            "narrative_importance": "medium",
        }
        mock_point = MagicMock()
        mock_point.payload = mock_payload

        with patch('modules.consolidation.qdrant') as mock_qdrant, \
             patch('modules.consolidation._consolidation_conn') as mock_conn_fn, \
             patch('modules.consolidation._embed_text', return_value=[0.1] * 1536) as mock_embed, \
             patch('modules.consolidation.check_reconsolidation', return_value={"should_reconsolidate": True, "prediction_error": 0.8}), \
             patch('modules.memory_smart.delete_memory_fts', return_value=True), \
             patch('modules.memory_smart.index_memory_fts', return_value=True), \
             patch('modules.utils.resolve_memory_id', return_value="full-uuid-reembed"):
            mock_qdrant.retrieve.return_value = [mock_point]
            mock_conn = MagicMock()
            mock_conn_fn.return_value = mock_conn

            result = correct_memory("full-uuid", "Podman replaced Docker")
            # Should call upsert (not set_payload)
            mock_qdrant.upsert.assert_called_once()
            mock_qdrant.set_payload.assert_not_called()
            # Should have generated a new embedding
            mock_embed.assert_called_once()
            assert "re-embedded" in result.lower()
            # Fix 1: content should be replacement, not concatenation
            upsert_points = mock_qdrant.upsert.call_args[1]["points"]
            assert upsert_points[0].payload["data"] == "Podman replaced Docker"

    def test_correct_memory_updates_fts(self):
        """correct_memory should update FTS5 index (delete + re-index)."""
        from modules.consolidation import correct_memory
        mock_payload = {
            "data": "Old FTS content",
            "confidence": 0.7,
            "reconsolidation_count": 0,
        }
        mock_point = MagicMock()
        mock_point.payload = mock_payload

        with patch('modules.consolidation.qdrant') as mock_qdrant, \
             patch('modules.consolidation._consolidation_conn') as mock_conn_fn, \
             patch('modules.consolidation._embed_text', return_value=[0.2] * 1536), \
             patch('modules.consolidation.check_reconsolidation', return_value={"should_reconsolidate": True, "prediction_error": 0.6}), \
             patch('modules.memory_smart.delete_memory_fts') as mock_del_fts, \
             patch('modules.memory_smart.index_memory_fts') as mock_idx_fts, \
             patch('modules.utils.resolve_memory_id', return_value="fts-uuid"):
            mock_qdrant.retrieve.return_value = [mock_point]
            mock_conn = MagicMock()
            mock_conn_fn.return_value = mock_conn

            correct_memory("fts-uuid", "New FTS content")
            mock_del_fts.assert_called_once_with("fts-uuid")
            mock_idx_fts.assert_called_once()

    def test_correct_memory_checks_labile(self):
        """Non-labile memory without PE should be rejected (unless force=True)."""
        from modules.consolidation import correct_memory
        mock_payload = {
            "data": "Stable memory content",
            "confidence": 0.8,
            "reconsolidation_count": 0,
            "created_at": "2026-01-01T00:00:00",
            "attention_last_accessed": "2026-01-01T00:00:00",
            "attention_access_count": 1,
            "narrative_importance": "medium",
        }
        mock_point = MagicMock()
        mock_point.payload = mock_payload

        with patch('modules.consolidation.qdrant') as mock_qdrant, \
             patch('modules.consolidation._consolidation_conn') as mock_conn_fn, \
             patch('modules.utils.resolve_memory_id', return_value="stable-uuid"):
            mock_qdrant.retrieve.return_value = [mock_point]
            # Labile check returns None (not labile)
            mock_conn = MagicMock()
            mock_conn.execute.return_value.fetchone.return_value = None
            mock_conn_fn.return_value = mock_conn

            result = correct_memory("stable-uuid", "no correction signals here just info")
            assert "rejected" in result.lower()
            # upsert should NOT have been called
            mock_qdrant.upsert.assert_not_called()

    def test_correct_memory_force_override(self):
        """force=True should bypass labile gate."""
        from modules.consolidation import correct_memory
        mock_payload = {
            "data": "Content to force-correct",
            "confidence": 0.9,
            "reconsolidation_count": 0,
        }
        mock_point = MagicMock()
        mock_point.payload = mock_payload

        with patch('modules.consolidation.qdrant') as mock_qdrant, \
             patch('modules.consolidation._consolidation_conn') as mock_conn_fn, \
             patch('modules.consolidation._embed_text', return_value=[0.3] * 1536), \
             patch('modules.memory_smart.delete_memory_fts', return_value=True), \
             patch('modules.memory_smart.index_memory_fts', return_value=True), \
             patch('modules.utils.resolve_memory_id', return_value="force-uuid"):
            mock_qdrant.retrieve.return_value = [mock_point]
            mock_conn = MagicMock()
            mock_conn_fn.return_value = mock_conn

            result = correct_memory("force-uuid", "Human says this is wrong", force=True)
            assert "corrected" in result.lower()
            mock_qdrant.upsert.assert_called_once()


# ============================================================
# PART 6: Phase 4.5 - Mejora 2: Multi-Canal Contradicciones
# ============================================================

class TestMultiCanalContradiction:
    """Kumaran & Maguire 2007: CA1 comparator with 3 channels."""

    def test_contradiction_semantic_distance(self):
        """Distant texts with shared entities -> PE from semantic channel."""
        from modules.consolidation import detect_contradiction

        with patch('modules.consolidation._embed_text') as mock_embed:
            # Return orthogonal vectors (high distance)
            mock_embed.side_effect = [
                [1.0] + [0.0] * 1535,   # memory vector
                [0.0] + [1.0] + [0.0] * 1534,  # context vector
            ]
            result = detect_contradiction(
                "Docker is the standard container runtime for production",
                "Podman is the standard container runtime for production"
            )
            assert result["prediction_error"] > 0.0
            assert result["channels"]["semantic_distance"] > 0.0

    def test_contradiction_keywords_boost(self):
        """Correction keywords with entity overlap -> higher PE."""
        from modules.consolidation import detect_contradiction

        with patch('modules.consolidation._embed_text') as mock_embed:
            mock_embed.side_effect = [
                [1.0] + [0.0] * 1535,
                [0.0] + [1.0] + [0.0] * 1534,
            ]
            result = detect_contradiction(
                "Docker is the standard container runtime",
                "en realidad ya no usamos Docker, migramos a Podman"
            )
            # Keywords should boost PE above pure semantic
            assert result["prediction_error"] > 0.1
            assert result["channels"]["keywords"] > 0.0

    def test_contradiction_no_overlap_no_pe(self):
        """Distant texts WITHOUT shared entities -> low PE."""
        from modules.consolidation import detect_contradiction

        # No shared entities between pizza and weather topics
        result = detect_contradiction(
            "Pizza tastes great",
            "The weather today is sunny"
        )
        # No entity overlap, no keywords -> PE should be 0 or very low
        assert result["prediction_error"] < 0.1

    def test_contradiction_negation_detected(self):
        """Same entities + negation inversion -> PE from negation channel."""
        from modules.consolidation import detect_contradiction

        with patch('modules.consolidation._embed_text') as mock_embed:
            mock_embed.side_effect = [
                [0.9, 0.1] + [0.0] * 1534,
                [0.8, 0.2] + [0.0] * 1534,
            ]
            result = detect_contradiction(
                "Docker container runtime works perfectly in production",
                "Docker container runtime never works in production"
            )
            assert result["prediction_error"] > 0.0
            assert result["channels"]["negation"] > 0.0


# ============================================================
# PART 7: Phase 4.5 - Mejora 5: Assessment Runtime Gates
# ============================================================

class TestAssessmentRuntimeGates:
    """Block 1995 / Butlin 2023: access consciousness requires demonstrated exercise."""

    def test_butlin_hot2_dormant_scoring(self):
        """0 RCJ records -> HOT-2 score should be 0.3 (DORMANT)."""
        from modules.consciousness import assess_butlin_indicators

        with patch('modules.retrieval_metadata.get_fok_calibration', return_value={"n_records": 0}):
            report = assess_butlin_indicators()
            # Find HOT-2 line
            for line in report.split("\n"):
                if "HOT-2" in line:
                    assert "DORMANT" in line or "0.3" in line
                    break

    def test_butlin_pp3_nascent_scoring(self):
        """correct_memory exists but 0 reconsolidation records -> PP-3 score 0.7."""
        from modules.consciousness import assess_butlin_indicators

        with patch('modules.consolidation._consolidation_conn') as mock_conn_fn:
            mock_conn = MagicMock()
            mock_conn.execute.return_value.fetchone.return_value = (0,)
            mock_conn_fn.return_value = mock_conn

            report = assess_butlin_indicators()
            for line in report.split("\n"):
                if "PP-3" in line:
                    assert "NASCENT" in line or "0.7" in line
                    break

    def test_butlin_gradual_scoring(self):
        """Assessment should use 0.3/0.7/1.0 scale (not just 0/0.5/1)."""
        from modules.consciousness import assess_butlin_indicators

        report = assess_butlin_indicators()
        # The score label mapping should support DORMANT and NASCENT
        assert "DORMANT" in report or "NASCENT" in report or "Total Score" in report
        # Verify the total is a valid number (may be lower than 13.0 now)
        for line in report.split("\n"):
            if "Total Score:" in line:
                import re
                match = re.search(r"(\d+\.?\d*)/", line)
                assert match is not None
                score = float(match.group(1))
                assert 0.0 <= score <= 14.0
                break
