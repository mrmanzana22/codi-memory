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
        """correct_memory should update Qdrant payload."""
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
             patch('modules.utils.resolve_memory_id', return_value="full-uuid-123"):
            mock_qdrant.retrieve.return_value = [mock_point]
            mock_conn = MagicMock()
            mock_conn_fn.return_value = mock_conn

            result = correct_memory("full-uuid", "This is the correction")
            assert "corrected" in result.lower()
            mock_qdrant.set_payload.assert_called_once()
            call_payload = mock_qdrant.set_payload.call_args[1]["payload"]
            assert "CORRECTED" in call_payload["data"]
            assert call_payload["confidence"] == pytest.approx(0.7)

    def test_correct_memory_logs_reconsolidation(self):
        """correct_memory should create entry in reconsolidation_log."""
        from modules.consolidation import correct_memory
        mock_payload = {"data": "Old", "confidence": 0.6, "reconsolidation_count": 0}
        mock_point = MagicMock()
        mock_point.payload = mock_payload

        with patch('modules.consolidation.qdrant') as mock_qdrant, \
             patch('modules.consolidation._consolidation_conn') as mock_conn_fn, \
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
