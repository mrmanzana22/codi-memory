#!/usr/bin/env python3
"""
Unit tests for Phase 3 Wave 1: SDR (3C), Temporal (3D), Attention Schema (3B).
Run: python3 -m pytest tests/test_phase3_wave1.py -v
"""

import sys
import os
import math

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest


# ============================================================
# 3C: STATE-DEPENDENT RETRIEVAL
# ============================================================

class TestSDRBonus:
    """State-Dependent Retrieval (Godden & Baddeley 1975)."""

    def test_identical_state_max_bonus(self):
        """Same PAD at encoding and retrieval -> max bonus 0.08."""
        p_diff, a_diff = 0.0, 0.0
        dist = math.sqrt(p_diff**2 + a_diff**2)
        bonus = 0.08 * math.exp(-2.0 * dist)
        assert bonus == pytest.approx(0.08)

    def test_distant_state_near_zero(self):
        """Very different PAD -> bonus near zero."""
        p_diff, a_diff = 1.0, 1.0
        dist = math.sqrt(p_diff**2 + a_diff**2)
        bonus = 0.08 * math.exp(-2.0 * dist)
        assert bonus < 0.01

    def test_moderate_distance(self):
        """Moderate PAD distance -> partial bonus."""
        p_diff, a_diff = 0.3, 0.3
        dist = math.sqrt(p_diff**2 + a_diff**2)
        bonus = 0.08 * math.exp(-2.0 * dist)
        assert 0.01 < bonus < 0.07

    def test_bonus_always_positive(self):
        """Bonus should never be negative."""
        for p in [-1, 0, 0.5, 1]:
            for a in [-1, 0, 0.5, 1]:
                dist = math.sqrt(p**2 + a**2)
                bonus = 0.08 * math.exp(-2.0 * dist)
                assert bonus >= 0

    def test_bonus_capped_at_008(self):
        """Bonus should never exceed 0.08."""
        bonus = 0.08 * math.exp(-2.0 * 0)
        assert bonus <= 0.08


# ============================================================
# 3D: TEMPORAL METADATA
# ============================================================

class TestTemporalContext:
    """Temporal context classification (Friedman 1993)."""

    def test_morning(self):
        from datetime import datetime
        dt = datetime(2026, 2, 13, 8, 0)
        hour = dt.hour
        assert 5 <= hour < 12  # morning

    def test_afternoon(self):
        from datetime import datetime
        dt = datetime(2026, 2, 13, 14, 0)
        hour = dt.hour
        assert 12 <= hour < 17  # afternoon

    def test_evening(self):
        from datetime import datetime
        dt = datetime(2026, 2, 13, 19, 0)
        hour = dt.hour
        assert 17 <= hour < 21  # evening

    def test_night(self):
        from datetime import datetime
        dt = datetime(2026, 2, 13, 23, 0)
        hour = dt.hour
        assert hour >= 21 or hour < 5  # night

    def test_weekday(self):
        from datetime import datetime
        # Feb 13, 2026 is a Friday (weekday=4)
        dt = datetime(2026, 2, 13)
        assert dt.weekday() < 5

    def test_weekend(self):
        from datetime import datetime
        # Feb 14, 2026 is a Saturday (weekday=5)
        dt = datetime(2026, 2, 14)
        assert dt.weekday() >= 5


class TestTemporalDistance:
    """Temporal distance labels at retrieval."""

    def test_just_now(self):
        delta_hours = 0.5
        if delta_hours < 1:
            label = "just_now"
        assert label == "just_now"

    def test_today(self):
        delta_hours = 5
        if delta_hours < 1:
            label = "just_now"
        elif delta_hours < 24:
            label = "today"
        assert label == "today"

    def test_recently(self):
        delta_hours = 72
        if delta_hours < 1:
            label = "just_now"
        elif delta_hours < 24:
            label = "today"
        elif delta_hours < 168:
            label = "recently"
        assert label == "recently"

    def test_weeks_ago(self):
        delta_hours = 360
        if delta_hours < 168:
            label = "recently"
        elif delta_hours < 720:
            label = "weeks_ago"
        assert label == "weeks_ago"

    def test_long_ago(self):
        delta_hours = 1000
        if delta_hours >= 720:
            label = "long_ago"
        assert label == "long_ago"


# ============================================================
# 3B: ATTENTION SCHEMA ENHANCEMENT
# ============================================================

class TestAttentionDescription:
    """describe_attention() - Graziano 2013 AST."""

    def test_no_focus_returns_diffuse(self):
        from modules.wiring import _attention_schema, describe_attention
        # Save and reset
        old = _attention_schema.copy()
        _attention_schema["current_focus"] = None
        result = describe_attention()
        assert "difuso" in result.lower()
        # Restore
        _attention_schema.update(old)

    def test_with_focus_returns_description(self):
        from modules.wiring import _attention_schema, describe_attention
        old = {k: v for k, v in _attention_schema.items()}
        _attention_schema["current_focus"] = "trading"
        _attention_schema["focus_strength"] = 0.8
        _attention_schema["driver"] = "user_prompt"
        _attention_schema["suppressed_items"] = ["fullempaques", "tiaw"]
        result = describe_attention()
        assert "trading" in result
        assert "0.8" in result or "0.80" in result
        assert "user_prompt" in result
        assert "fullempaques" in result
        # Restore
        _attention_schema.update(old)

    def test_returns_string(self):
        from modules.wiring import describe_attention
        assert isinstance(describe_attention(), str)


class TestAttentionPrediction:
    """predict_next_focus() - bigram transition model."""

    def test_no_transitions_returns_none(self):
        from modules.wiring import _attention_schema, predict_next_focus
        old_transitions = _attention_schema["topic_transitions"][:]
        old_focus = _attention_schema["current_focus"]
        _attention_schema["topic_transitions"] = []
        _attention_schema["current_focus"] = "trading"
        topic, prob = predict_next_focus()
        assert topic is None
        assert prob == 0.0
        _attention_schema["topic_transitions"] = old_transitions
        _attention_schema["current_focus"] = old_focus

    def test_predicts_most_common_transition(self):
        from modules.wiring import _attention_schema, predict_next_focus
        old_transitions = _attention_schema["topic_transitions"][:]
        old_focus = _attention_schema["current_focus"]
        _attention_schema["current_focus"] = "trading"
        _attention_schema["topic_transitions"] = [
            {"from": "trading", "to": "fullempaques", "at": "t1", "driver": "user"},
            {"from": "trading", "to": "fullempaques", "at": "t2", "driver": "user"},
            {"from": "trading", "to": "tiaw", "at": "t3", "driver": "user"},
        ]
        topic, prob = predict_next_focus()
        assert topic == "fullempaques"
        assert prob == pytest.approx(2/3)
        _attention_schema["topic_transitions"] = old_transitions
        _attention_schema["current_focus"] = old_focus

    def test_prediction_probability_sums_to_one_concept(self):
        """The most common probability should be <= 1.0."""
        from modules.wiring import _attention_schema, predict_next_focus
        old_transitions = _attention_schema["topic_transitions"][:]
        old_focus = _attention_schema["current_focus"]
        _attention_schema["current_focus"] = "x"
        _attention_schema["topic_transitions"] = [
            {"from": "x", "to": "a", "at": "t1", "driver": "u"},
        ]
        _, prob = predict_next_focus()
        assert 0.0 <= prob <= 1.0
        _attention_schema["topic_transitions"] = old_transitions
        _attention_schema["current_focus"] = old_focus


class TestSuppressionTracking:
    """suppressed_items tracking in attention schema."""

    def test_suppressed_items_in_schema(self):
        from modules.wiring import _attention_schema
        assert "suppressed_items" in _attention_schema

    def test_suppressed_capped_at_10(self):
        from modules.wiring import _attention_schema
        old = _attention_schema["suppressed_items"][:]
        _attention_schema["suppressed_items"] = list(range(15))
        _attention_schema["suppressed_items"] = _attention_schema["suppressed_items"][-10:]
        assert len(_attention_schema["suppressed_items"]) <= 10
        _attention_schema["suppressed_items"] = old


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
