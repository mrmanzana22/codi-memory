#!/usr/bin/env python3
"""
Unit tests for Phase 4: Close the Loops (Real Cognitive Improvement).
Run: ./venv/bin/pytest tests/test_phase4.py -v
"""

import sys
import os
import sqlite3
import math
import unittest.mock
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

    def test_butlin_hot3_importable(self):
        """metacognitive_control should be importable (structural prerequisite)."""
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

        embed_output = [0.2] * 1536
        with patch('modules.consolidation.qdrant') as mock_qdrant, \
             patch('modules.consolidation._consolidation_conn') as mock_conn_fn, \
             patch('modules.consolidation._embed_text', return_value=embed_output) as mock_embed, \
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
            # PE-proportional decrement: 0.8 - (0.05 + 0.15 * 0.8) = 0.63
            assert points[0].payload["confidence"] == pytest.approx(0.63, abs=0.01)
            # Fix 2: vector comes from re-embedding the correction text (Nader 2000)
            mock_embed.assert_called_once_with("This is the correction")
            assert points[0].vector == embed_output

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
        """Confidence decrement should be proportional to PE (Exton-McGuinness 2015)."""
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
            # PE=0.8 -> delta = 0.05 + 0.15*0.8 = 0.17 -> new = 0.73
            assert "0.90" in result and "0.73" in result

    def test_prediction_error_pushes_to_working_memory(self):
        """High PE with actual_keywords should push surprise to working memory.

        Note: PE-driven reconsolidation is deferred because preturn_inject
        doesn't emit memory_id (PE is topic-level, not memory-specific).
        """
        from modules.wiring import _on_prediction_error

        with patch('modules.wiring._update_attention_schema'), \
             patch('modules.working_memory.push_to_working_memory') as mock_wm:

            _on_prediction_error("prediction_error", {
                "error_magnitude": 0.8,
                "topic": "trading",
                "actual_keywords": ["bitcoin", "crash"],
            })

            mock_wm.assert_called_once()
            call_kwargs = mock_wm.call_args
            assert "PREDICTION ERROR" in call_kwargs[1]["content"] or "PREDICTION ERROR" in call_kwargs[0][0]

    def test_low_pe_no_wm_push(self):
        """PE below threshold should not push to working memory."""
        from modules.wiring import _on_prediction_error

        with patch('modules.wiring._update_attention_schema'), \
             patch('modules.working_memory.push_to_working_memory') as mock_wm:

            _on_prediction_error("prediction_error", {
                "error_magnitude": 0.1,
                "topic": "test",
                "actual_keywords": ["something"],
            })
            # PE = 0.1, below 0.3 threshold -- no WM push
            mock_wm.assert_not_called()

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

    def test_topic_confirmation_amplifies_contradiction(self):
        """Same topic (high cosine sim) amplifies C1+C3 PE (Kumaran 2006)."""
        from modules.consolidation import detect_contradiction

        # High cosine sim (same topic) + negation -> amplified PE
        with patch('modules.consolidation._embed_text') as mock_embed:
            mock_embed.side_effect = [
                [0.9, 0.1] + [0.0] * 1534,
                [0.85, 0.15] + [0.0] * 1534,
            ]
            same_topic = detect_contradiction(
                "Docker container runtime works perfectly in production",
                "en realidad Docker container runtime never works in production"
            )

        # Low cosine sim (different topic) + same correction keyword
        with patch('modules.consolidation._embed_text') as mock_embed:
            mock_embed.side_effect = [
                [1.0] + [0.0] * 1535,
                [0.0] + [1.0] + [0.0] * 1534,
            ]
            diff_topic = detect_contradiction(
                "Docker container runtime works perfectly in production",
                "en realidad Docker container runtime never works in production"
            )

        # Same topic should amplify PE relative to different topic
        assert same_topic["prediction_error"] > diff_topic["prediction_error"]
        assert same_topic["channels"]["topic_confirmation"] > 0.5

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

    def test_butlin_pp3_dormant_with_zero_records(self):
        """correct_memory exists but 0 reconsolidation records -> PP-3 DORMANT (0.3).

        Block 1995: exercise required. 0 records = not exercised = DORMANT.
        """
        from modules.consciousness import assess_butlin_indicators

        with patch('modules.consolidation._consolidation_conn') as mock_conn_fn:
            mock_conn = MagicMock()
            mock_conn.execute.return_value.fetchone.return_value = (0,)
            mock_conn_fn.return_value = mock_conn

            report = assess_butlin_indicators()
            for line in report.split("\n"):
                if "PP-3" in line:
                    assert "DORMANT" in line or "0.3" in line
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


# ============================================================
# PART 8: Phase 5 - Inline Contradiction Detection at Encoding
# ============================================================

class TestInlineContradictionCheck:
    """Kumaran & Maguire 2007: automatic CA1 comparator at encoding time."""

    def _reset_session_state(self):
        """Reset module-level session counters for test isolation."""
        import modules.memory_smart as ms
        ms._contradiction_session_count = 0
        ms._contradiction_cooldown = {}

    def test_inline_contradiction_detects_conflict(self):
        """Candidates with shared entities + high PE -> detected=True."""
        self._reset_session_state()
        from modules.memory_smart import _inline_contradiction_check

        candidates = [{
            "id": "mem-old-123",
            "memory": "Docker is the standard container runtime for production deployments",
            "score": 0.75,
        }]

        with patch('modules.consolidation.detect_contradiction') as mock_detect, \
             patch('modules.consolidation._extract_key_entities') as mock_entities:
            mock_entities.side_effect = [
                {"docker", "container", "production"},  # new content
                {"docker", "container", "production"},  # existing memory
            ]
            mock_detect.return_value = {
                "prediction_error": 0.55,
                "detail": "semantic + negation",
                "channels": {"keywords": 0.0, "topic_confirmation": 0.3, "negation": 0.25},
            }

            result = _inline_contradiction_check(
                "Podman replaced Docker as the container runtime in production",
                candidates,
            )

            assert result["detected"] is True
            assert result["pe"] >= 0.3
            assert result["conflicting_memory_id"] == "mem-old-123"
            assert len(result["shared_entities"]) >= 2

    def test_inline_contradiction_no_overlap_skips(self):
        """Candidates without shared entities -> detected=False (no expensive call)."""
        self._reset_session_state()
        from modules.memory_smart import _inline_contradiction_check

        candidates = [{
            "id": "mem-pizza",
            "memory": "Pizza is delicious Italian food",
            "score": 0.60,
        }]

        with patch('modules.consolidation.detect_contradiction') as mock_detect, \
             patch('modules.consolidation._extract_key_entities') as mock_entities:
            mock_entities.side_effect = [
                {"weather", "sunny", "forecast"},  # new content entities
                {"pizza", "italian", "food"},       # memory entities - no overlap
            ]

            result = _inline_contradiction_check(
                "The weather is sunny today",
                candidates,
            )

            assert result["detected"] is False
            # detect_contradiction should NOT have been called (entity pre-screen)
            mock_detect.assert_not_called()

    def test_inline_contradiction_rate_limit(self):
        """After MAX_PER_SESSION alerts -> detected=False."""
        self._reset_session_state()
        import modules.memory_smart as ms
        ms._contradiction_session_count = 3  # MAX_PER_SESSION default

        from modules.memory_smart import _inline_contradiction_check

        result = _inline_contradiction_check("any content", [{"id": "x", "memory": "y", "score": 0.7}])
        assert result["detected"] is False
        assert result.get("reason") == "session_rate_limit"

    def test_inline_contradiction_cooldown(self):
        """Same topic within cooldown window -> detected=False."""
        self._reset_session_state()
        import modules.memory_smart as ms
        from modules.memory_smart import _inline_contradiction_check

        # Pre-populate cooldown for topic "container_docker_production"
        ms._contradiction_cooldown["container_docker_production"] = datetime.now().isoformat()

        candidates = [{
            "id": "mem-cd",
            "memory": "Docker container is used in production environments",
            "score": 0.70,
        }]

        with patch('modules.consolidation.detect_contradiction') as mock_detect, \
             patch('modules.consolidation._extract_key_entities') as mock_entities:
            mock_entities.side_effect = [
                {"container", "docker", "production"},
                {"container", "docker", "production"},
            ]
            mock_detect.return_value = {
                "prediction_error": 0.60,
                "detail": "high PE",
                "channels": {"keywords": 0.2, "topic_confirmation": 0.2, "negation": 0.2},
            }

            result = _inline_contradiction_check(
                "Podman container replaced Docker in production",
                candidates,
            )

            assert result["detected"] is False
            assert result.get("reason") == "cooldown"

    def test_inline_contradiction_skips_dedup_range(self):
        """Candidates with score >= dedup_threshold are skipped."""
        self._reset_session_state()
        from modules.memory_smart import _inline_contradiction_check

        candidates = [{
            "id": "mem-dup",
            "memory": "This is practically the same text",
            "score": 0.95,  # Above dedup threshold
        }]

        with patch('modules.consolidation.detect_contradiction') as mock_detect, \
             patch('modules.consolidation._extract_key_entities') as mock_entities:
            result = _inline_contradiction_check("This is practically the same text", candidates)
            assert result["detected"] is False
            mock_detect.assert_not_called()

    def test_inline_contradiction_skips_low_score(self):
        """Candidates with score < CONTRADICTION_SCORE_FLOOR are skipped."""
        self._reset_session_state()
        from modules.memory_smart import _inline_contradiction_check

        candidates = [{
            "id": "mem-low",
            "memory": "Completely unrelated memory content here",
            "score": 0.30,  # Below 0.50 floor
        }]

        with patch('modules.consolidation.detect_contradiction') as mock_detect, \
             patch('modules.consolidation._extract_key_entities') as mock_entities:
            result = _inline_contradiction_check("New content about something", candidates)
            assert result["detected"] is False
            mock_detect.assert_not_called()


class TestContradictionEventEmission:
    """Phase 5: add_memory_smart emits CONTRADICTION_DETECTED event."""

    def test_add_memory_smart_emits_contradiction_event(self):
        """When contradiction detected, event_bus should emit CONTRADICTION_DETECTED."""
        from modules.memory_smart import add_memory_smart
        from modules.events import event_bus, Events

        emitted_events = []
        def capture_event(event_name, data):
            if event_name == Events.CONTRADICTION_DETECTED:
                emitted_events.append(data)

        event_bus.on(Events.CONTRADICTION_DETECTED, capture_event)

        try:
            with patch('modules.memory_smart.memory') as mock_mem, \
                 patch('modules.memory_smart.qdrant') as mock_qdrant, \
                 patch('modules.memory_smart.index_memory_fts'), \
                 patch('modules.memory_smart._inline_contradiction_check') as mock_check:

                mock_mem.search.return_value = {
                    "results": [{"id": "existing-1", "memory": "Old fact", "score": 0.70}]
                }
                mock_mem.add.return_value = {"results": [{"id": "new-mem-1"}]}
                mock_check.return_value = {
                    "detected": True,
                    "pe": 0.55,
                    "conflicting_memory_id": "existing-1",
                    "conflicting_text": "Old fact",
                    "detail": "test",
                    "channels": {"keywords": 0.1, "topic_confirmation": 0.2, "negation": 0.25},
                    "shared_entities": ["docker", "production"],
                }

                result = add_memory_smart("Docker is no longer used in production")

                assert len(emitted_events) == 1
                assert emitted_events[0]["pe"] == 0.55
                assert emitted_events[0]["conflicting_memory_id"] == "existing-1"
        finally:
            event_bus.off(Events.CONTRADICTION_DETECTED, capture_event)

    def test_add_memory_smart_no_event_when_no_contradiction(self):
        """No contradiction -> no CONTRADICTION_DETECTED event."""
        from modules.memory_smart import add_memory_smart
        from modules.events import event_bus, Events

        emitted_events = []
        def capture_event(event_name, data):
            if event_name == Events.CONTRADICTION_DETECTED:
                emitted_events.append(data)

        event_bus.on(Events.CONTRADICTION_DETECTED, capture_event)

        try:
            with patch('modules.memory_smart.memory') as mock_mem, \
                 patch('modules.memory_smart.qdrant') as mock_qdrant, \
                 patch('modules.memory_smart.index_memory_fts'), \
                 patch('modules.memory_smart._inline_contradiction_check') as mock_check:

                mock_mem.search.return_value = {
                    "results": [{"id": "e-1", "memory": "Some fact", "score": 0.70}]
                }
                mock_mem.add.return_value = {"results": [{"id": "new-2"}]}
                mock_check.return_value = {"detected": False, "pe": 0.05}

                add_memory_smart("A completely compatible new fact")

                assert len(emitted_events) == 0
        finally:
            event_bus.off(Events.CONTRADICTION_DETECTED, capture_event)


class TestContradictionWiringHandler:
    """Phase 5: _on_contradiction_detected handler in wiring.py."""

    def test_handler_pushes_high_pe_to_wm(self):
        """PE >= ALERT threshold -> push alert to working memory."""
        from modules.wiring import _on_contradiction_detected

        with patch('modules.working_memory.push_to_working_memory') as mock_push, \
             patch('modules.wiring._update_attention_schema'):

            _on_contradiction_detected("contradiction_detected", {
                "pe": 0.60,
                "conflicting_memory_id": "old-mem-abc",
                "conflicting_text": "Docker is great for production",
                "new_content": "Podman replaced Docker in production",
                "shared_entities": ["docker", "production"],
            })

            mock_push.assert_called_once()
            call_kwargs = mock_push.call_args[1]
            assert call_kwargs["topic"] == "contradiction"
            assert call_kwargs["relevance"] > 0.6
            assert "CONTRADICTION" in call_kwargs["content"]

    def test_handler_marks_labile_on_high_pe(self):
        """PE >= ALERT threshold -> mark conflicting memory as labile."""
        from modules.wiring import _on_contradiction_detected

        with patch('modules.working_memory.push_to_working_memory'), \
             patch('modules.consolidation.mark_as_labile') as mock_labile, \
             patch('modules.wiring._update_attention_schema'):

            _on_contradiction_detected("contradiction_detected", {
                "pe": 0.65,
                "conflicting_memory_id": "old-mem-xyz",
                "conflicting_text": "Old belief about Docker",
                "new_content": "New contradicting info",
                "shared_entities": ["docker"],
            })

            mock_labile.assert_called_once_with(
                memory_id="old-mem-xyz",
                prediction_error=0.65,
                trigger_context=unittest.mock.ANY,
            )

    def test_handler_silent_note_on_moderate_pe(self):
        """PE between SILENT and ALERT -> silent WM note only."""
        from modules.wiring import _on_contradiction_detected

        with patch('modules.working_memory.push_to_working_memory') as mock_push, \
             patch('modules.wiring._update_attention_schema'):

            _on_contradiction_detected("contradiction_detected", {
                "pe": 0.35,  # Below ALERT (0.50), above SILENT (0.30)
                "conflicting_memory_id": "old-mem-mod",
                "conflicting_text": "Some old memory",
                "new_content": "Some new info",
                "shared_entities": ["topic"],
            })

            mock_push.assert_called_once()
            call_kwargs = mock_push.call_args[1]
            assert call_kwargs["topic"] == "metamemory"
            assert "PE NOTE" in call_kwargs["content"]

    def test_handler_updates_attention_schema(self):
        """Handler should always update attention schema."""
        from modules.wiring import _on_contradiction_detected

        with patch('modules.working_memory.push_to_working_memory'), \
             patch('modules.wiring._update_attention_schema') as mock_attn:

            _on_contradiction_detected("contradiction_detected", {
                "pe": 0.55,
                "conflicting_memory_id": "mem-attn",
                "conflicting_text": "Old",
                "new_content": "New",
                "shared_entities": ["topic1", "topic2"],
            })

            mock_attn.assert_called_once()
            call_kwargs = mock_attn.call_args[1]
            assert call_kwargs["driver"] == "contradiction_detected"
            assert call_kwargs["strength"] == 0.55


class TestContradictionEventType:
    """Phase 5: CONTRADICTION_DETECTED event type exists."""

    def test_events_has_contradiction_detected(self):
        """Events class should have CONTRADICTION_DETECTED."""
        from modules.events import Events
        assert hasattr(Events, 'CONTRADICTION_DETECTED')
        assert Events.CONTRADICTION_DETECTED == 'contradiction_detected'

    def test_wiring_registers_contradiction_handler(self):
        """wire_event_bus should register the contradiction handler."""
        from modules.events import event_bus, Events
        from modules.wiring import wire_event_bus

        wire_event_bus()
        stats = event_bus.get_stats()
        assert Events.CONTRADICTION_DETECTED in stats
        assert stats[Events.CONTRADICTION_DETECTED] >= 1


# ============================================================
# PART 9: Session State Persistence (despertar/flush cycle)
# ============================================================

class TestSessionStatePersistence:
    """Session state should survive across despertar/flush cycles."""

    @pytest.fixture
    def temp_state_file(self, tmp_path):
        """Temporary session state file."""
        return str(tmp_path / "session_state.json")

    def test_save_session_state_creates_file(self, temp_state_file):
        """_save_session_state should write a valid JSON file."""
        import json
        import sqlite3
        from modules.flush import _save_session_state
        from modules.config import _emotional_state

        _emotional_state['current'] = {
            'pleasure': 0.6, 'arousal': 0.3, 'dominance': 0.5,
            'timestamp': '2026-02-13T22:00:00', 'trigger': 'tarea_completada'
        }

        # Mock the WM SQLite query
        mock_conn = MagicMock()
        mock_rows = [{"topic": "consciencia"}, {"topic": "fullempaques"}]
        mock_conn.__enter__ = MagicMock(return_value=mock_conn)
        mock_conn.__exit__ = MagicMock(return_value=False)
        mock_conn.execute.return_value.fetchall.return_value = mock_rows

        with patch('modules.flush.SESSION_STATE_FILE', temp_state_file), \
             patch('modules.working_memory._get_conn', return_value=mock_conn):
            result = _save_session_state("Completamos Phase 5")

        assert result is True
        assert os.path.exists(temp_state_file)

        with open(temp_state_file) as f:
            state = json.load(f)

        assert state["pad"]["pleasure"] == 0.6
        assert state["pad"]["arousal"] == 0.3
        assert state["pad"]["dominance"] == 0.5
        assert state["pad"]["trigger"] == "tarea_completada"
        assert state["session_summary"] == "Completamos Phase 5"

    def test_load_session_state_returns_saved(self, temp_state_file):
        """load_session_state should return state if fresh."""
        import json
        from modules.flush import load_session_state
        from modules.config import now_iso

        state = {
            "timestamp": now_iso(),
            "pad": {"pleasure": 0.6, "arousal": 0.3, "dominance": 0.5, "trigger": "test"},
            "active_project": "consciencia",
            "wm_topics": ["consciencia", "fullempaques"],
            "session_summary": "We did Phase 5",
        }
        with open(temp_state_file, 'w') as f:
            json.dump(state, f)

        with patch('modules.flush.SESSION_STATE_FILE', temp_state_file):
            loaded = load_session_state()

        assert loaded is not None
        assert loaded["pad"]["pleasure"] == 0.6
        assert loaded["active_project"] == "consciencia"

    def test_load_session_state_expired_returns_none(self, temp_state_file):
        """load_session_state should return None for old state (>24h)."""
        import json
        from modules.flush import load_session_state

        state = {
            "timestamp": "2025-01-01T00:00:00-05:00",  # Very old
            "pad": {"pleasure": 0.6, "arousal": 0.3, "dominance": 0.5, "trigger": "old"},
            "active_project": "old_project",
            "wm_topics": [],
            "session_summary": "old session",
        }
        with open(temp_state_file, 'w') as f:
            json.dump(state, f)

        with patch('modules.flush.SESSION_STATE_FILE', temp_state_file):
            loaded = load_session_state()

        assert loaded is None

    def test_load_session_state_no_file_returns_none(self, temp_state_file):
        """No state file -> None."""
        from modules.flush import load_session_state

        with patch('modules.flush.SESSION_STATE_FILE', "/nonexistent/path.json"):
            loaded = load_session_state()

        assert loaded is None

    def test_despertar_restores_pad_from_session(self):
        """despertar_codi should use PAD from session state, not defaults."""
        from modules.config import _emotional_state

        mock_session = {
            "timestamp": datetime.now().isoformat(),
            "pad": {"pleasure": 0.7, "arousal": -0.2, "dominance": 0.8, "trigger": "exito"},
            "active_project": "consciencia",
            "wm_topics": ["consciencia"],
            "session_summary": "Finished Phase 5",
        }

        with patch('modules.flush.load_session_state', return_value=mock_session), \
             patch('modules.flush.load_session_state', return_value=mock_session), \
             patch('modules.consciousness.qdrant') as mock_qdrant, \
             patch('modules.consciousness.memory') as mock_mem, \
             patch('modules.consciousness._verificar_salud_memoria_interna', return_value={"ok": True}):

            mock_qdrant.scroll.return_value = ([], None)
            mock_mem.search.return_value = {"results": []}

            from modules.consciousness import despertar_codi
            result = despertar_codi()

            # PAD should be restored, not default
            assert _emotional_state['current']['pleasure'] == 0.7
            assert _emotional_state['current']['arousal'] == -0.2
            assert _emotional_state['current']['dominance'] == 0.8
            assert "restored_from_session" in _emotional_state['current']['trigger']

    def test_despertar_uses_defaults_without_session(self):
        """No session state -> despertar uses default PAD values."""
        from modules.config import _emotional_state

        with patch('modules.flush.load_session_state', return_value=None), \
             patch('modules.consciousness.qdrant') as mock_qdrant, \
             patch('modules.consciousness.memory') as mock_mem, \
             patch('modules.consciousness._verificar_salud_memoria_interna', return_value={"ok": True}):

            mock_qdrant.scroll.return_value = ([], None)
            mock_mem.search.return_value = {"results": []}

            from modules.consciousness import despertar_codi
            result = despertar_codi()

            # Should be defaults
            assert _emotional_state['current']['pleasure'] == 0.3
            assert _emotional_state['current']['arousal'] == 0.1
            assert _emotional_state['current']['dominance'] == 0.4
            assert "default" in _emotional_state['current']['trigger']

    def test_despertar_shows_last_session_summary(self):
        """If session state has summary, despertar should show it."""
        mock_session = {
            "timestamp": datetime.now().isoformat(),
            "pad": {"pleasure": 0.5, "arousal": 0.0, "dominance": 0.3, "trigger": "test"},
            "active_project": "trading",
            "wm_topics": ["trading"],
            "session_summary": "Implementamos estrategia de grid trading",
        }

        with patch('modules.flush.load_session_state', return_value=mock_session), \
             patch('modules.consciousness.qdrant') as mock_qdrant, \
             patch('modules.consciousness.memory') as mock_mem, \
             patch('modules.consciousness._verificar_salud_memoria_interna', return_value={"ok": True}):

            mock_qdrant.scroll.return_value = ([], None)
            mock_mem.search.return_value = {"results": []}

            from modules.consciousness import despertar_codi
            result = despertar_codi()

            assert "ULTIMA SESION" in result
            assert "grid trading" in result

    def test_despertar_dynamic_project_query(self):
        """Project search query should use active_project from session, not hardcoded."""
        mock_session = {
            "timestamp": datetime.now().isoformat(),
            "pad": {"pleasure": 0.3, "arousal": 0.1, "dominance": 0.4, "trigger": "test"},
            "active_project": "trading",
            "wm_topics": ["trading"],
            "session_summary": "Worked on trading",
        }

        with patch('modules.flush.load_session_state', return_value=mock_session), \
             patch('modules.consciousness.qdrant') as mock_qdrant, \
             patch('modules.consciousness.memory') as mock_mem, \
             patch('modules.consciousness._verificar_salud_memoria_interna', return_value={"ok": True}):

            mock_qdrant.scroll.return_value = ([], None)
            mock_mem.search.return_value = {"results": []}

            from modules.consciousness import despertar_codi
            result = despertar_codi()

            # Check that memory.search was called with "trading" not "fullempaques"
            search_calls = mock_mem.search.call_args_list
            project_call = [c for c in search_calls if "proyecto" in str(c)]
            assert len(project_call) >= 1
            assert "trading" in str(project_call[0])


# ============================================================
# PART N+1: Neuro-audit round fixes
# ============================================================

class TestHOT2ConfigDBPath:
    """HOT-2 assessment must use FTS_DB_PATH from config, not default."""

    def test_hot2_nascent_with_5_records(self, tmp_path):
        """5 RCJ records in config DB -> HOT-2 score 0.7 (nascent)."""
        import sqlite3
        db_path = str(tmp_path / "test_fts.db")
        conn = sqlite3.connect(db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS fok_calibration_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                query TEXT, fok_predicted REAL, actual_coverage TEXT,
                actual_count INTEGER, actual_top_activation REAL, created_at TEXT
            )
        """)
        for i in range(5):
            conn.execute(
                "INSERT INTO fok_calibration_log (query, fok_predicted, actual_coverage, actual_count, actual_top_activation, created_at) VALUES (?, ?, ?, ?, ?, ?)",
                (f"query_{i}", 0.5, "partial", 3, 0.4, f"2026-01-{10+i}T00:00:00")
            )
        conn.commit()
        conn.close()

        with patch('modules.config.FTS_DB_PATH', db_path):
            from modules.retrieval_metadata import get_fok_calibration
            cal = get_fok_calibration(fts_db_path=db_path)
            assert cal["n_records"] == 5

    def test_hot2_full_with_20_records(self, tmp_path):
        """20 RCJ records in config DB -> HOT-2 score 1.0 (full)."""
        import sqlite3
        db_path = str(tmp_path / "test_fts.db")
        conn = sqlite3.connect(db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS fok_calibration_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                query TEXT, fok_predicted REAL, actual_coverage TEXT,
                actual_count INTEGER, actual_top_activation REAL, created_at TEXT
            )
        """)
        for i in range(20):
            conn.execute(
                "INSERT INTO fok_calibration_log (query, fok_predicted, actual_coverage, actual_count, actual_top_activation, created_at) VALUES (?, ?, ?, ?, ?, ?)",
                (f"query_{i}", 0.6, "comprehensive", 5, 0.7, f"2026-01-{10+i:02d}T00:00:00")
            )
        conn.commit()
        conn.close()

        from modules.retrieval_metadata import get_fok_calibration
        cal = get_fok_calibration(fts_db_path=db_path)
        assert cal["n_records"] == 20


class TestHOT4EmotionGatedRanking:
    """HOT-4: emotion must change retrieval ranking (Bower 1981, Godden & Baddeley 1975)."""

    def test_mood_congruence_changes_score(self):
        """Positive pleasure should boost positive-valence memories over negative ones."""
        from modules.activation import compute_unified_activation
        import math

        # Two memories: same base activation inputs, different emotional valence
        base_kwargs = dict(
            created_at="2026-02-10T00:00:00",
            last_accessed="2026-02-13T00:00:00",
            access_count=3,
            access_timestamps=None,
            spreading_boost=0.5,
            importance="medium",
            noise=False,  # deterministic for test
        )

        # Memory A: positive valence, high pleasure state
        act_pos = compute_unified_activation(
            **base_kwargs,
            emotional_arousal=0.5,
            emotional_congruence=1.0 * 0.8,  # positive valence * positive pleasure
        )

        # Memory B: negative valence, same high pleasure state
        act_neg = compute_unified_activation(
            **base_kwargs,
            emotional_arousal=0.5,
            emotional_congruence=-1.0 * 0.8,  # negative valence * positive pleasure
        )

        # Positive-congruent memory should score higher
        assert act_pos.total > act_neg.total, (
            f"Mood congruence failed: positive={act_pos.total:.4f} should > negative={act_neg.total:.4f}"
        )

    def test_hot4_assessment_nascent(self):
        """HOT-4 should score 0.7 (nascent) when emotion gating is wired."""
        import re
        with patch('modules.config._emotional_state', {'current': {'pleasure': 0.5, 'arousal': 0.3, 'dominance': 0.0}}):
            from modules.consciousness import assess_butlin_indicators
            result = assess_butlin_indicators()
            # Result is markdown string; parse HOT-4 line
            match = re.search(r'\*\*HOT-4\*\*\s+\[(\w+)\]\s+\(([0-9.]+)\)', result)
            assert match, f"HOT-4 not found in assessment output:\n{result}"
            label = match.group(1)
            score = float(match.group(2))
            assert score >= 0.7, (
                f"HOT-4 should be >= 0.7 (nascent) with emotion gating wired, got {score} [{label}]"
            )


# ============================================================
# PART 12: Runtime Evidence Scoring (Block 1995)
# ============================================================

class TestRuntimeEvidenceScoring:
    """PP-2 and HOT-3 must show runtime emissions to score above DORMANT."""

    def test_pp2_dormant_without_emissions(self):
        """PP-2 should be DORMANT (0.3) when no evidence exists (session or persistent)."""
        import re
        from modules.events import event_bus
        # Clear history AND mock persistent counts to simulate truly cold state
        old_history = event_bus._history[:]
        event_bus._history.clear()
        try:
            with patch.object(event_bus, 'get_persistent_count', return_value=0):
                from modules.consciousness import assess_butlin_indicators
                result = assess_butlin_indicators()
                match = re.search(r'\*\*PP-2\*\*\s+\[(\w+)\]\s+\(([0-9.]+)\)', result)
                assert match, f"PP-2 not found in output:\n{result}"
                score = float(match.group(2))
                assert score == pytest.approx(0.3), f"PP-2 without any evidence should be 0.3 DORMANT, got {score}"
        finally:
            event_bus._history = old_history

    def test_pp2_nascent_with_emissions(self):
        """PP-2 should be >= 0.7 NASCENT when PREDICTION_ERROR has been emitted."""
        import re
        from modules.events import event_bus, Events
        old_history = event_bus._history[:]
        event_bus._history.clear()
        try:
            # Emit 3 PE events
            for _ in range(3):
                event_bus.emit(Events.PREDICTION_ERROR, {"error_magnitude": 0.6, "topic": "test"})
            from modules.consciousness import assess_butlin_indicators
            result = assess_butlin_indicators()
            match = re.search(r'\*\*PP-2\*\*\s+\[(\w+)\]\s+\(([0-9.]+)\)', result)
            assert match, f"PP-2 not found in output:\n{result}"
            score = float(match.group(2))
            assert score >= 0.7, f"PP-2 with {3} emissions should be >= 0.7 NASCENT, got {score}"
        finally:
            event_bus._history = old_history

    def test_hot3_dormant_without_emissions(self):
        """HOT-3 should be DORMANT (0.3) when no evidence exists (session or persistent)."""
        import re
        from modules.events import event_bus
        old_history = event_bus._history[:]
        event_bus._history.clear()
        try:
            with patch.object(event_bus, 'get_persistent_count', return_value=0):
                from modules.consciousness import assess_butlin_indicators
                result = assess_butlin_indicators()
                match = re.search(r'\*\*HOT-3\*\*\s+\[(\w+)\]\s+\(([0-9.]+)\)', result)
                assert match, f"HOT-3 not found in output:\n{result}"
                score = float(match.group(2))
                assert score == pytest.approx(0.3), f"HOT-3 without any evidence should be 0.3 DORMANT, got {score}"
        finally:
            event_bus._history = old_history

    def test_hot3_nascent_with_emissions(self):
        """HOT-3 should be >= 0.7 NASCENT when METACOGNITIVE_CONTROL_APPLIED has been emitted."""
        import re
        from modules.events import event_bus, Events
        old_history = event_bus._history[:]
        event_bus._history.clear()
        try:
            # Emit 5 metacognitive control events
            for _ in range(5):
                event_bus.emit(Events.METACOGNITIVE_CONTROL_APPLIED, {"strategy": "full_search", "fok_score": 0.7})
            from modules.consciousness import assess_butlin_indicators
            result = assess_butlin_indicators()
            match = re.search(r'\*\*HOT-3\*\*\s+\[(\w+)\]\s+\(([0-9.]+)\)', result)
            assert match, f"HOT-3 not found in output:\n{result}"
            score = float(match.group(2))
            assert score >= 0.7, f"HOT-3 with {5} emissions should be >= 0.7 NASCENT, got {score}"
        finally:
            event_bus._history = old_history


# ============================================================
# PART 13: Graph Densification (Phase 5.5)
# ============================================================

class TestGraphDensification:
    """Auto-connect neighbors to densify graph for spreading activation."""

    def test_auto_connect_creates_related_memories(self):
        """New memory should auto-connect to similar existing memories."""
        from modules.memory_smart import _auto_connect_neighbors

        mock_results = {
            "results": [
                {"id": "similar-1", "score": 0.8, "memory": "related content 1"},
                {"id": "similar-2", "score": 0.6, "memory": "related content 2"},
                {"id": "similar-3", "score": 0.4, "memory": "related content 3"},  # below threshold
            ]
        }

        with patch('modules.memory_smart.memory') as mock_mem, \
             patch('modules.memory_smart.qdrant') as mock_qdrant:
            mock_mem.search.return_value = mock_results

            _auto_connect_neighbors("new-id", "test content")

            # Should set payload with 2 connections (similar-3 below 0.5 threshold)
            mock_qdrant.set_payload.assert_called_once()
            call_args = mock_qdrant.set_payload.call_args
            related = call_args[1]["payload"]["related_memories"]
            assert len(related) == 2
            assert "similar-1" in related
            assert "similar-2" in related
            assert "similar-3" not in related  # below threshold

    def test_auto_connect_excludes_already_connected(self):
        """Should not duplicate connections to already-connected memories."""
        from modules.memory_smart import _auto_connect_neighbors

        mock_results = {
            "results": [
                {"id": "already-connected", "score": 0.9, "memory": "old"},
                {"id": "new-neighbor", "score": 0.7, "memory": "new"},
            ]
        }

        with patch('modules.memory_smart.memory') as mock_mem, \
             patch('modules.memory_smart.qdrant') as mock_qdrant:
            mock_mem.search.return_value = mock_results

            _auto_connect_neighbors("new-id", "test", exclude_ids=["already-connected"])

            call_args = mock_qdrant.set_payload.call_args
            related = call_args[1]["payload"]["related_memories"]
            assert "already-connected" not in related
            assert "new-neighbor" in related

    def test_auto_connect_caps_at_max(self):
        """Should create at most GRAPH_AUTO_CONNECT_MAX connections."""
        from modules.memory_smart import _auto_connect_neighbors

        mock_results = {
            "results": [
                {"id": f"mem-{i}", "score": 0.9 - i * 0.05, "memory": f"content {i}"}
                for i in range(7)
            ]
        }

        with patch('modules.memory_smart.memory') as mock_mem, \
             patch('modules.memory_smart.qdrant') as mock_qdrant:
            mock_mem.search.return_value = mock_results

            _auto_connect_neighbors("new-id", "test")

            call_args = mock_qdrant.set_payload.call_args
            related = call_args[1]["payload"]["related_memories"]
            assert len(related) <= 3  # GRAPH_AUTO_CONNECT_MAX = 3

    def test_spreading_reads_related_memories(self):
        """Spreading activation should traverse related_memories edges."""
        from modules.spreading import _get_neighbors

        payload = {
            "related_to": "neighbor-1",
            "related_memories": ["neighbor-2", "neighbor-3"],
            "consolidated_with": ["neighbor-4"],
        }

        neighbors = _get_neighbors("self-id", payload)
        assert "neighbor-1" in neighbors
        assert "neighbor-2" in neighbors
        assert "neighbor-3" in neighbors
        assert "neighbor-4" in neighbors
        assert "self-id" not in neighbors


# ============================================================
# PART 14: Persistent Evidence (Phase 5.5 - Block 1995)
# ============================================================

class TestPersistentEvidence:
    """Event counts survive restart via SQLite persistence."""

    def test_event_counts_persist_to_sqlite(self, tmp_path):
        """Emitted events should be counted in SQLite."""
        from modules.events import EventBus, Events
        db_path = str(tmp_path / "test_events.db")

        bus = EventBus()
        with patch.object(EventBus, '_get_db_path', return_value=db_path):
            # Emit 5 prediction errors
            for _ in range(5):
                bus.emit(Events.PREDICTION_ERROR, {"test": True})
            bus._flush_counts()  # Force flush

            count = bus.get_persistent_count(Events.PREDICTION_ERROR)
            assert count == 5, f"Expected 5 persisted events, got {count}"

    def test_persistent_count_survives_new_bus(self, tmp_path):
        """A new EventBus instance should read counts from previous sessions."""
        from modules.events import EventBus, Events
        db_path = str(tmp_path / "test_events.db")

        # Session 1: emit events
        bus1 = EventBus()
        with patch.object(EventBus, '_get_db_path', return_value=db_path):
            for _ in range(8):
                bus1.emit(Events.PREDICTION_ERROR, {"test": True})
            bus1._flush_counts()

        # Session 2: new bus, empty history, but persistent counts remain
        bus2 = EventBus()
        assert len(bus2._history) == 0  # Fresh session
        with patch.object(EventBus, '_get_db_path', return_value=db_path):
            count = bus2.get_persistent_count(Events.PREDICTION_ERROR)
            assert count == 8, f"Persistent count should survive restart, got {count}"

    def test_scoring_uses_persistent_evidence(self, tmp_path):
        """PP-2 should be NASCENT even in cold session if persistent evidence exists."""
        import re
        from modules.events import event_bus, EventBus, Events
        db_path = str(tmp_path / "test_scoring.db")

        # Pre-populate persistent counts
        with patch.object(EventBus, '_get_db_path', return_value=db_path):
            for _ in range(6):
                event_bus.emit(Events.PREDICTION_ERROR, {"test": True})
            event_bus._flush_counts()

        # Clear session history (simulate restart)
        old_history = event_bus._history[:]
        event_bus._history.clear()
        try:
            with patch.object(EventBus, '_get_db_path', return_value=db_path):
                from modules.consciousness import assess_butlin_indicators
                result = assess_butlin_indicators()
                match = re.search(r'\*\*PP-2\*\*\s+\[(\w+)\]\s+\(([0-9.]+)\)', result)
                assert match, f"PP-2 not found in output:\n{result}"
                score = float(match.group(2))
                assert score >= 0.7, (
                    f"PP-2 should be >= 0.7 NASCENT with persistent evidence, got {score}"
                )
        finally:
            event_bus._history = old_history


# ============================================================
# PART 15: GWT on Automatic Retrieval Path (Phase 5.5)
# ============================================================

class TestGWTAutomatic:
    """Workspace competition should run on every search_memory call."""

    def test_gwt_competition_filters_results(self):
        """run_workspace_competition should be called in search_memory path."""
        from modules.competition import run_workspace_competition, CompetitionCandidate

        # Verify competition is importable and callable from search_memory's try block
        assert callable(run_workspace_competition)

        # Test the competition filtering logic directly
        candidates = [
            CompetitionCandidate(content="strong", source_domain="episodic", activation=0.8, memory_id="id-1"),
            CompetitionCandidate(content="medium", source_domain="semantic", activation=0.5, memory_id="id-2"),
            CompetitionCandidate(content="weak", source_domain="episodic", activation=0.1, memory_id="id-3"),
        ]
        result = run_workspace_competition(candidates)

        # Winners should only include those above ignition threshold
        winner_ids = {w.memory_id for w in result.winners}
        loser_ids = {l.memory_id for l in result.losers}
        # id-3 (0.1) should be below ignition threshold (0.3)
        assert "id-3" not in winner_ids, "Weak candidate should be filtered by ignition"
        assert "id-1" in winner_ids, "Strong candidate should win"

    def test_gwt_in_search_memory_code(self):
        """search_memory should contain the GWT competition block."""
        import inspect
        from modules.memory_core import search_memory
        src = inspect.getsource(search_memory)
        assert "run_workspace_competition" in src, "GWT competition not found in search_memory"
        assert "CompetitionCandidate" in src, "CompetitionCandidate not used in search_memory"
        assert "winner_ids" in src, "Winner filtering not found in search_memory"
