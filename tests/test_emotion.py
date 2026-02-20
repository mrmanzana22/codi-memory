"""Tests for modules/emotion.py.

Covers:
  - set_emotional_state / get_emotional_state (PAD model)
  - update_mood_baseline
  - apply_emotional_decay
  - get_emotional_expression
  - add_memory_with_emotion (mocked)
  - tag_memory_emotion (mocked)
  - search_by_emotion (mocked)
"""

import json
import pytest
from unittest.mock import patch, MagicMock


@pytest.fixture(autouse=True)
def _reset_emotional_state(monkeypatch):
    """Reset emotional state before each test."""
    import modules.config as cfg
    original = cfg._emotional_state.copy()
    cfg._emotional_state['current'] = {
        'pleasure': 0, 'arousal': 0, 'dominance': 0,
        'timestamp': None, 'trigger': None
    }
    cfg._emotional_state['mood'] = {
        'pleasure': 0.3, 'arousal': 0.0, 'dominance': 0.2,
        'last_updated': None
    }
    cfg._emotional_state['history'] = []
    cfg._emotional_state['decay_rate'] = 0.15
    yield


class TestSetEmotionalState:
    """Tests for set_emotional_state."""

    def test_sets_pad_values(self):
        from modules.emotion import set_emotional_state
        result = json.loads(set_emotional_state(0.8, 0.5, 0.3, trigger="test"))
        state = result["state"]
        assert state["pleasure"] == 0.8
        assert state["arousal"] == 0.5
        assert state["dominance"] == 0.3
        assert state["trigger"] == "test"

    def test_clamps_values(self):
        from modules.emotion import set_emotional_state
        result = json.loads(set_emotional_state(5.0, -5.0, 2.0))
        state = result["state"]
        assert state["pleasure"] == 1.0
        assert state["arousal"] == -1.0
        assert state["dominance"] == 1.0

    def test_classifies_emotion(self):
        from modules.emotion import set_emotional_state
        # High pleasure, high arousal, high dominance = exuberant
        result = json.loads(set_emotional_state(0.8, 0.8, 0.8))
        assert result["state"]["emotion"] == "exuberant"

    def test_calculates_intensity(self):
        from modules.emotion import set_emotional_state
        result = json.loads(set_emotional_state(0.8, 0.5, 0.3))
        assert result["state"]["intensity"] > 0

    def test_pushes_to_history(self):
        from modules.emotion import set_emotional_state
        import modules.config as cfg
        # Set initial state
        set_emotional_state(0.5, 0.5, 0.5)
        # Set new state (should push old to history)
        set_emotional_state(0.8, 0.2, 0.1)
        assert len(cfg._emotional_state['history']) == 1

    def test_history_capped_at_20(self):
        from modules.emotion import set_emotional_state
        import modules.config as cfg
        for i in range(25):
            set_emotional_state(i * 0.04, 0.0, 0.0)
        assert len(cfg._emotional_state['history']) <= 20


class TestGetEmotionalState:
    """Tests for get_emotional_state."""

    def test_no_state_returns_neutral(self):
        from modules.emotion import get_emotional_state
        result = json.loads(get_emotional_state())
        assert result["current"]["emotion"] == "neutral"

    def test_returns_current_and_mood(self):
        from modules.emotion import set_emotional_state, get_emotional_state
        set_emotional_state(0.5, 0.5, 0.5)
        result = json.loads(get_emotional_state())
        assert "current" in result
        assert "mood_baseline" in result
        assert result["current"]["pleasure"] == 0.5

    def test_include_history(self):
        from modules.emotion import set_emotional_state, get_emotional_state
        set_emotional_state(0.3, 0.3, 0.3)
        set_emotional_state(0.6, 0.6, 0.6)
        result = json.loads(get_emotional_state(include_history=True))
        assert "history" in result
        assert len(result["history"]) >= 1

    def test_without_history(self):
        from modules.emotion import get_emotional_state
        result = json.loads(get_emotional_state(include_history=False))
        assert "history" not in result


class TestUpdateMoodBaseline:
    """Tests for update_mood_baseline."""

    def test_updates_pleasure_only(self):
        from modules.emotion import update_mood_baseline
        result = json.loads(update_mood_baseline(pleasure=0.7))
        assert result["mood"]["pleasure"] == 0.7

    def test_partial_update(self):
        from modules.emotion import update_mood_baseline
        result = json.loads(update_mood_baseline(arousal=0.4))
        mood = result["mood"]
        assert mood["arousal"] == 0.4
        # Other values should stay at defaults
        assert mood["pleasure"] == 0.3  # from fixture reset

    def test_clamps_values(self):
        from modules.emotion import update_mood_baseline
        result = json.loads(update_mood_baseline(pleasure=5.0, arousal=-5.0))
        assert result["mood"]["pleasure"] == 1.0
        assert result["mood"]["arousal"] == -1.0


class TestApplyEmotionalDecay:
    """Tests for apply_emotional_decay."""

    def test_no_state_returns_not_applied(self):
        from modules.emotion import apply_emotional_decay
        result = json.loads(apply_emotional_decay())
        assert result["applied"] is False

    def test_decays_toward_baseline(self):
        from modules.emotion import set_emotional_state, apply_emotional_decay
        set_emotional_state(1.0, 1.0, 1.0)
        result = json.loads(apply_emotional_decay())
        current = result["current"]
        # Should be closer to mood baseline (0.3, 0.0, 0.2)
        assert current["pleasure"] < 1.0
        assert current["arousal"] < 1.0

    def test_decay_pushes_history(self):
        from modules.emotion import set_emotional_state, apply_emotional_decay
        import modules.config as cfg
        set_emotional_state(0.8, 0.8, 0.8)
        apply_emotional_decay()
        assert len(cfg._emotional_state['history']) >= 1


class TestGetEmotionalExpression:
    """Tests for get_emotional_expression."""

    def test_no_state_returns_neutral(self):
        from modules.emotion import get_emotional_expression
        result = json.loads(get_emotional_expression())
        assert "neutral" in result["expression"].lower() or result["intensity"] == "none"

    def test_with_state_returns_expression(self):
        from modules.emotion import set_emotional_state, get_emotional_expression
        set_emotional_state(0.8, 0.8, 0.8, trigger="test success")
        result = json.loads(get_emotional_expression())
        assert "expression" in result
        assert "test success" in result["expression"]

    def test_high_dominance_adds_control(self):
        from modules.emotion import set_emotional_state, get_emotional_expression
        set_emotional_state(0.5, 0.5, 0.8)
        result = json.loads(get_emotional_expression())
        assert "control" in result["expression"]

    def test_low_dominance_adds_vulnerable(self):
        from modules.emotion import set_emotional_state, get_emotional_expression
        set_emotional_state(0.5, 0.5, -0.8)
        result = json.loads(get_emotional_expression())
        assert "vulnerable" in result["expression"]


class TestSearchByEmotion:
    """Tests for search_by_emotion (mocked qdrant)."""

    def test_invalid_emotion_returns_error(self, patch_externals):
        from modules.emotion import search_by_emotion
        result = json.loads(search_by_emotion("happiness"))
        assert result["result"] == "error"
        assert "no valida" in result["message"]

    def test_valid_emotion_empty_results(self, patch_externals):
        from modules.emotion import search_by_emotion
        result = json.loads(search_by_emotion("exuberant"))
        assert result["result"] == "Sin resultados"

    def test_valid_emotions_accepted(self, patch_externals):
        from modules.emotion import search_by_emotion
        for emotion in ['exuberant', 'dependent', 'relaxed', 'docile',
                        'hostile', 'anxious', 'disdainful', 'bored']:
            result = json.loads(search_by_emotion(emotion))
            assert result["result"] != "error", f"Failed for {emotion}"
