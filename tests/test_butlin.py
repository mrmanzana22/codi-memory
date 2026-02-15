#!/usr/bin/env python3
"""
Unit tests for Butlin Consciousness Assessment Tool (Phase 3E).
Run: python3 -m pytest tests/test_butlin.py -v
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest


class TestButlinAssessment:
    def test_returns_string(self):
        from modules.consciousness import assess_butlin_indicators
        result = assess_butlin_indicators()
        assert isinstance(result, str)

    def test_contains_all_14_indicators(self):
        from modules.consciousness import assess_butlin_indicators
        result = assess_butlin_indicators()
        for indicator in ["GWT-1", "GWT-2", "GWT-3", "GWT-4",
                         "HOT-1", "HOT-2", "HOT-3", "HOT-4",
                         "AST-1",
                         "PP-1", "PP-2", "PP-3",
                         "RPT-1", "RPT-2"]:
            assert indicator in result, f"Missing indicator {indicator}"

    def test_scores_are_valid(self):
        from modules.consciousness import assess_butlin_indicators
        result = assess_butlin_indicators()
        # Check that scores appear as 0.0, 0.5, or 1.0
        assert any(label in result for label in ["ABSENT", "PARTIAL", "FULL"])

    def test_total_score_present(self):
        from modules.consciousness import assess_butlin_indicators
        result = assess_butlin_indicators()
        assert "Total Score:" in result

    def test_contains_theory_sections(self):
        from modules.consciousness import assess_butlin_indicators
        result = assess_butlin_indicators()
        assert "Global Workspace Theory" in result
        assert "Higher-Order Theories" in result
        assert "Attention Schema Theory" in result
        assert "Predictive Processing" in result
        assert "Recurrent Processing Theory" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
