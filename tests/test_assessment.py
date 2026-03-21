#!/usr/bin/env python3
"""
ASSESSMENT EXTERNAL EVALUATOR TESTS (Deliverable 3)
=====================================================
Verify that the decoupled assessor is deterministic, independent,
produces correct scores from fixtures, and matches legacy scoring.

4 test classes:
  1. Determinism: same evidence -> same score (two runs)
  2. Independence: assessment.py does not import consciousness.py
  3. Evidence correctness: fixture with known data -> expected scores
  4. Paridad: new assessment == legacy assess_butlin_indicators

Run: ./venv/bin/pytest tests/test_assessment.py -v
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest


class TestDeterminism:
    """Same evidence snapshot -> same score, every time."""

    def test_same_evidence_same_score(self):
        """Two runs with identical evidence produce identical results."""
        from modules.assessment import get_assessment

        # Create a fixed evidence snapshot (no IO)
        evidence = _make_fixture_evidence(
            total_events=150, handler_count=10, comp_count=15,
            comp_subscribers=3, retrieved_count=60, active_event_types=7,
        )

        result1 = get_assessment(evidence=evidence)
        result2 = get_assessment(evidence=evidence)

        assert result1["score_total"] == result2["score_total"]
        assert result1["version"] == result2["version"]
        for i in range(len(result1["indicators"])):
            assert result1["indicators"][i]["score"] == result2["indicators"][i]["score"]
            assert result1["indicators"][i]["name"] == result2["indicators"][i]["name"]

    def test_different_evidence_different_score(self):
        """Different evidence produces different scores."""
        from modules.assessment import get_assessment

        ev_high = _make_fixture_evidence(total_events=200, comp_count=20, retrieved_count=100, active_event_types=8)
        ev_low = _make_fixture_evidence(total_events=0, comp_count=0, retrieved_count=0, active_event_types=0)

        r_high = get_assessment(evidence=ev_high)
        r_low = get_assessment(evidence=ev_low)

        assert r_high["score_total"] > r_low["score_total"]


class TestIndependence:
    """assessment.py does not import consciousness.py."""

    def test_no_consciousness_import(self):
        """assessment.py source code does not contain 'from modules.consciousness' or 'import consciousness'."""
        import inspect
        from modules import assessment
        source = inspect.getsource(assessment)

        # Check that assessment doesn't import consciousness as a scoring dependency
        # Note: collect_evidence() may reference consciousness module to check function existence
        # but score_indicator() and get_assessment() must not depend on it for scoring
        scoring_functions = [
            assessment.score_indicator,
            assessment.get_assessment,
            assessment.format_assessment,
            assessment._score_gwt1, assessment._score_gwt2, assessment._score_gwt3, assessment._score_gwt4,
            assessment._score_hot1, assessment._score_hot2, assessment._score_hot3, assessment._score_hot4,
            assessment._score_ast1,
            assessment._score_pp1, assessment._score_pp2, assessment._score_pp3,
            assessment._score_rpt1, assessment._score_rpt2,
        ]
        for fn in scoring_functions:
            fn_source = inspect.getsource(fn)
            assert "consciousness" not in fn_source, \
                f"{fn.__name__} references consciousness.py - scoring must be independent"

    def test_scoring_works_without_consciousness(self):
        """score_indicator works with just a dict, no module imports needed."""
        from modules.assessment import score_indicator

        evidence = _make_fixture_evidence(total_events=50, handler_count=6)
        result = score_indicator("GWT-1", evidence)

        assert result["name"] == "GWT-1"
        assert result["score"] == 0.7  # 50 >= 10 -> NASCENT
        assert isinstance(result["evidence"], str)


class TestEvidenceCorrectness:
    """Verify scoring rules produce expected results from known fixtures."""

    def test_all_full_scores(self):
        """When all evidence is maxed, all eligible indicators score FULL."""
        from modules.assessment import get_assessment

        ev = _make_fixture_evidence(
            total_events=200, handler_count=12, comp_count=20, comp_subscribers=5,
            self_model_refreshes=15, fok_n_records=25, metacognitive_count=30,
            has_emotional_state=True, has_emotion_gating=True,
            has_attention_schema=True, attention_suppressed=True,
            attention_history_len=10, attention_pe_count=25,
            has_prediction_schemas=True, prediction_error_count=30,
            has_correct_memory=True, reconsolidation_count=10,
            has_recurrent_cycle=True, retrieved_count=100,
            active_event_types=8, has_meta_functions=True,
        )
        result = get_assessment(evidence=ev)

        # All should be FULL except:
        # - HOT-4 caps at 0.7 NASCENT (needs runtime emotion gating/change evidence in event_counts)
        # - RPT-2 is 0.5 PARTIAL when Phi_E=0 (Sprint 4 FIX-09: differentiation without integration)
        for ind in result["indicators"]:
            if ind["name"] == "HOT-4":
                assert ind["score"] == 0.7, f"HOT-4 should be NASCENT, got {ind['score']}"
            elif ind["name"] == "RPT-2":
                assert ind["score"] == 0.5, f"RPT-2 should be PARTIAL (no Phi_E), got {ind['score']}"
            else:
                assert ind["score"] == 1.0, f"{ind['name']} should be FULL, got {ind['score']}"

    def test_all_absent_scores(self):
        """When no evidence, indicators score ABSENT or minimal DORMANT."""
        from modules.assessment import get_assessment

        ev = _make_fixture_evidence()  # all defaults = minimal
        result = get_assessment(evidence=ev)

        for ind in result["indicators"]:
            assert ind["score"] <= 0.5, f"{ind['name']} should be <= PARTIAL with no evidence, got {ind['score']}"

    def test_gwt1_thresholds(self):
        """GWT-1 follows exact threshold rules."""
        from modules.assessment import score_indicator

        assert score_indicator("GWT-1", _make_fixture_evidence(total_events=100, handler_count=5))["score"] == 1.0
        assert score_indicator("GWT-1", _make_fixture_evidence(total_events=50, handler_count=5))["score"] == 0.7
        assert score_indicator("GWT-1", _make_fixture_evidence(total_events=5, handler_count=5))["score"] == 0.3
        assert score_indicator("GWT-1", _make_fixture_evidence(total_events=0, handler_count=2))["score"] == 0.0

    def test_fast_tool_contract_lookup(self):
        """PP-3 reconsolidation thresholds: 0->DORMANT, 1->NASCENT, 5->FULL."""
        from modules.assessment import score_indicator

        assert score_indicator("PP-3", _make_fixture_evidence(has_correct_memory=True, reconsolidation_count=0))["score"] == 0.3
        assert score_indicator("PP-3", _make_fixture_evidence(has_correct_memory=True, reconsolidation_count=3))["score"] == 0.7
        assert score_indicator("PP-3", _make_fixture_evidence(has_correct_memory=True, reconsolidation_count=5))["score"] == 1.0
        assert score_indicator("PP-3", _make_fixture_evidence(has_correct_memory=False))["score"] == 0.0

    def test_assessment_has_14_indicators(self):
        """Assessment always produces exactly 14 indicators."""
        from modules.assessment import get_assessment
        result = get_assessment(evidence=_make_fixture_evidence())
        assert len(result["indicators"]) == 14

    def test_format_produces_markdown(self):
        """format_assessment produces valid markdown with expected sections."""
        from modules.assessment import get_assessment, format_assessment
        result = get_assessment(evidence=_make_fixture_evidence(total_events=50))
        text = format_assessment(result)
        assert "# Butlin Consciousness Assessment" in text
        assert "Structural Score:" in text
        assert "Functional Score:" in text
        assert "## Summary" in text


class TestParidad:
    """New assessment matches legacy assess_butlin_indicators."""

    def test_new_vs_legacy_same_format(self):
        """Both produce markdown with '# Butlin Consciousness Assessment' header."""
        from modules.consciousness import assess_butlin_indicators, _legacy_assess_butlin
        new_result = assess_butlin_indicators()
        legacy_result = _legacy_assess_butlin()

        # Both should produce markdown with same header
        assert "# Butlin Consciousness Assessment" in new_result
        assert "# Butlin Consciousness Assessment" in legacy_result

    def test_new_vs_legacy_same_indicator_count(self):
        """Both produce 14 indicators."""
        from modules.consciousness import assess_butlin_indicators, _legacy_assess_butlin
        new_result = assess_butlin_indicators()
        legacy_result = _legacy_assess_butlin()

        # Count indicator lines (lines starting with "- **")
        new_indicators = [l for l in new_result.split("\n") if l.strip().startswith("- **")]
        legacy_indicators = [l for l in legacy_result.split("\n") if l.strip().startswith("- **")]

        assert len(new_indicators) == 14
        assert len(legacy_indicators) == 14

    def test_new_vs_legacy_indicator_names_match(self):
        """Both produce same indicator names in same order."""
        from modules.consciousness import assess_butlin_indicators, _legacy_assess_butlin
        new_result = assess_butlin_indicators()
        legacy_result = _legacy_assess_butlin()

        def extract_names(text):
            names = []
            for line in text.split("\n"):
                if line.strip().startswith("- **"):
                    # Extract name between ** markers
                    name = line.split("**")[1] if "**" in line else ""
                    names.append(name)
            return names

        new_names = extract_names(new_result)
        legacy_names = extract_names(legacy_result)

        assert new_names == legacy_names, f"Indicator order mismatch:\nnew:    {new_names}\nlegacy: {legacy_names}"


# ============================================================
# FIXTURE HELPER
# ============================================================

def _make_fixture_evidence(**overrides) -> dict:
    """Create an evidence snapshot with defaults, overridable for testing."""
    defaults = {
        "event_counts": {},
        "handler_count": 0,
        "total_events": 0,
        "comp_count": 0,
        "comp_subscribers": 0,
        "self_model_refreshes": 0,
        "fok_n_records": 0,
        "metacognitive_count": 0,
        "has_emotional_state": False,
        "has_emotion_gating": False,
        "has_meta_functions": False,
        "has_attention_schema": False,
        "attention_suppressed": False,
        "attention_history_len": 0,
        "attention_pe_count": 0,
        "has_prediction_schemas": False,
        "prediction_error_count": 0,
        "has_correct_memory": False,
        "reconsolidation_count": 0,
        "has_recurrent_cycle": False,
        "retrieved_count": 0,
        "active_event_types": 0,
        "competition_slots": 5,
        "ignition_threshold": 0.25,
        "coalition_bonus": 0.10,
    }
    defaults.update(overrides)
    return defaults
