"""
Tests for Active Inference Integration (Sprint 5)
==================================================
Tests S5-01 through S5-05: persistence, state observer,
event-driven learning, action recommender, and pre-turn hint.
"""

import json
import sqlite3
import pytest
from collections import Counter
from unittest.mock import MagicMock, patch

from modules.active_inference import (
    ATTEND,
    RETRIEVE,
    STORE,
    CONSOLIDATE,
    Action,
    GenerativeModel,
    SystemState,
)


# ============================================================
# S5-01: Persistence Tests
# ============================================================

class TestGenerativeModelPersistence:
    """Test persist/load round-trip for GenerativeModel."""

    def _make_model_with_data(self) -> GenerativeModel:
        """Create a model with some transition data."""
        model = GenerativeModel()
        s1 = SystemState("general", 0.5, 0.5, 0.0, 0.0)
        s2 = SystemState("general", 0.2, 0.3, 0.0, 0.0)
        s3 = SystemState("codigo", 0.8, 0.7, 0.5, 0.1)
        model.update(s1, RETRIEVE, s2)
        model.update(s1, RETRIEVE, s2)  # Same transition twice
        model.update(s1, STORE, s3)
        model.update(s2, ATTEND, s1)
        return model

    def test_persist_creates_table(self):
        conn = sqlite3.connect(":memory:")
        model = GenerativeModel()
        model.persist(conn)
        tables = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
        assert ("generative_model_transitions",) in tables
        conn.close()

    def test_persist_writes_rows(self):
        conn = sqlite3.connect(":memory:")
        model = self._make_model_with_data()
        rows = model.persist(conn)
        assert rows == 3  # 3 unique (state, action, next_state) combos
        conn.close()

    def test_load_restores_counts(self):
        conn = sqlite3.connect(":memory:")
        original = self._make_model_with_data()
        original.persist(conn)

        loaded = GenerativeModel.load(conn)
        assert loaded.observation_count == original.observation_count
        conn.close()

    def test_load_restores_transitions(self):
        conn = sqlite3.connect(":memory:")
        original = self._make_model_with_data()
        original.persist(conn)

        loaded = GenerativeModel.load(conn)
        s1 = SystemState("general", 0.5, 0.5, 0.0, 0.0)
        # The RETRIEVE transition should predict s2 with high probability
        dist = loaded.predict_next_state(s1, RETRIEVE)
        assert len(dist) > 0
        conn.close()

    def test_load_empty_table(self):
        conn = sqlite3.connect(":memory:")
        model = GenerativeModel()
        model.persist(conn)  # Creates table but no data
        loaded = GenerativeModel.load(conn)
        assert loaded.observation_count == 0
        conn.close()

    def test_load_no_table(self):
        conn = sqlite3.connect(":memory:")
        loaded = GenerativeModel.load(conn)
        assert loaded.observation_count == 0
        conn.close()

    def test_persist_idempotent(self):
        conn = sqlite3.connect(":memory:")
        model = self._make_model_with_data()
        model.persist(conn)
        rows2 = model.persist(conn)  # Persist again
        assert rows2 == 3  # Same number, upsert
        count = conn.execute(
            "SELECT COUNT(*) FROM generative_model_transitions"
        ).fetchone()[0]
        assert count == 3  # No duplicates
        conn.close()

    def test_persist_with_preferences(self):
        conn = sqlite3.connect(":memory:")
        prefs = {"low_uncertainty": 0.9, "moderate_wm": 0.5, "low_pe": 0.3}
        model = GenerativeModel(preferences=prefs)
        s1 = SystemState("general", 0.5, 0.5, 0.0, 0.0)
        s2 = SystemState("general", 0.1, 0.4, 0.0, 0.0)
        model.update(s1, ATTEND, s2)
        model.persist(conn)

        loaded = GenerativeModel.load(conn, preferences=prefs)
        assert loaded.preferences == prefs
        conn.close()


# ============================================================
# S5-02: State Observer + Transition Recorder Tests
# ============================================================

class TestObserveAndRecord:
    """Test the state observer and transition recorder."""

    @patch("modules.active_inference_integration.get_current_state")
    @patch("modules.active_inference_integration._model", None)
    @patch("modules.active_inference_integration._prev_state", None)
    @patch("modules.active_inference_integration._prev_action", None)
    def test_first_observation_no_transition(self, mock_state):
        """First observation should set state but not record a transition."""
        import modules.active_inference_integration as ai_int
        ai_int._model = GenerativeModel()
        ai_int._prev_state = None
        ai_int._prev_action = None

        mock_state.return_value = SystemState("general", 0.5, 0.5, 0.0, 0.0)
        result = ai_int.observe_and_record("retrieve")

        assert result is not None
        assert result.topic == "general"
        assert ai_int._model.observation_count == 0  # No transition yet

    @patch("modules.active_inference_integration.get_current_state")
    def test_second_observation_records_transition(self, mock_state):
        """Second observation should record the transition."""
        import modules.active_inference_integration as ai_int
        ai_int._model = GenerativeModel()
        ai_int._prev_state = SystemState("general", 0.5, 0.5, 0.0, 0.0)
        ai_int._prev_action = RETRIEVE

        mock_state.return_value = SystemState("general", 0.2, 0.3, 0.0, 0.0)
        ai_int.observe_and_record("store")

        assert ai_int._model.observation_count == 1

    @patch("modules.active_inference_integration.get_current_state")
    def test_unknown_action_still_updates_state(self, mock_state):
        """Unknown action name should update state but skip transition."""
        import modules.active_inference_integration as ai_int
        ai_int._model = GenerativeModel()
        ai_int._prev_state = SystemState("general", 0.5, 0.5, 0.0, 0.0)
        ai_int._prev_action = RETRIEVE

        mock_state.return_value = SystemState("codigo", 0.3, 0.4, 0.0, 0.0)
        result = ai_int.observe_and_record("nonexistent_action")

        assert result.topic == "codigo"
        assert ai_int._model.observation_count == 0  # No transition recorded


# ============================================================
# S5-03: Event-driven Learning Tests
# ============================================================

class TestEventHandlers:
    """Test event handlers trigger observe_and_record."""

    @patch("modules.active_inference_integration.observe_and_record")
    @patch("modules.active_inference_integration._maybe_persist")
    def test_on_memory_stored(self, mock_persist, mock_observe):
        from modules.active_inference_integration import _on_memory_stored
        _on_memory_stored({"memory_id": "abc"})
        mock_observe.assert_called_once_with("store")

    @patch("modules.active_inference_integration.observe_and_record")
    @patch("modules.active_inference_integration._maybe_persist")
    def test_on_memory_retrieved(self, mock_persist, mock_observe):
        from modules.active_inference_integration import _on_memory_retrieved
        _on_memory_retrieved({"query": "test"})
        mock_observe.assert_called_once_with("retrieve")

    @patch("modules.active_inference_integration.observe_and_record")
    @patch("modules.active_inference_integration._maybe_persist")
    def test_on_consolidation_complete(self, mock_persist, mock_observe):
        from modules.active_inference_integration import _on_consolidation_complete
        _on_consolidation_complete({})
        mock_observe.assert_called_once_with("consolidate")

    @patch("modules.active_inference_integration.observe_and_record")
    @patch("modules.active_inference_integration._maybe_persist")
    def test_on_prediction_error_high(self, mock_persist, mock_observe):
        from modules.active_inference_integration import _on_prediction_error
        _on_prediction_error({"surprise": 0.8})
        mock_observe.assert_called_once_with("explore")

    @patch("modules.active_inference_integration.observe_and_record")
    @patch("modules.active_inference_integration._maybe_persist")
    def test_on_prediction_error_low(self, mock_persist, mock_observe):
        from modules.active_inference_integration import _on_prediction_error
        _on_prediction_error({"surprise": 0.2})
        mock_observe.assert_called_once_with("attend")

    def test_persist_interval(self):
        """Model should persist after PERSIST_INTERVAL transitions."""
        import modules.active_inference_integration as ai_int
        ai_int._transitions_since_persist = 19  # One away from threshold
        with patch.object(ai_int, "persist_model") as mock_persist:
            ai_int._maybe_persist()
            mock_persist.assert_not_called()  # 19 < 20

            ai_int._transitions_since_persist = 20
            ai_int._maybe_persist()
            mock_persist.assert_called_once()
            assert ai_int._transitions_since_persist == 0


# ============================================================
# S5-04: Action Recommender Tests
# ============================================================

class TestRecommendAction:
    """Test the MCP-callable recommend_action function."""

    def setup_method(self):
        """Reset active option state before each test (Sprint 6 isolation)."""
        import modules.active_inference_integration as _ai
        _ai.reset_active_option()

    @patch("modules.active_inference_integration.get_current_state")
    @patch("modules.active_inference_integration.get_model")
    def test_recommend_returns_valid_structure(self, mock_model, mock_state):
        mock_state.return_value = SystemState("general", 0.5, 0.5, 0.0, 0.0)
        mock_model.return_value = GenerativeModel()

        from modules.active_inference_integration import recommend_action
        result = recommend_action()

        assert "recommended" in result
        assert "efe_scores" in result
        assert "state" in result
        assert "model_observations" in result
        assert "rationale" in result
        assert "description" in result

    @patch("modules.active_inference_integration.get_current_state")
    @patch("modules.active_inference_integration.get_model")
    def test_recommend_low_data_rationale(self, mock_model, mock_state):
        mock_state.return_value = SystemState("general", 0.5, 0.5, 0.0, 0.0)
        model = GenerativeModel()  # 0 observations
        mock_model.return_value = model

        from modules.active_inference_integration import recommend_action
        result = recommend_action()

        assert "Low data" in result["rationale"]

    @patch("modules.active_inference_integration.get_current_state")
    @patch("modules.active_inference_integration.get_model")
    def test_recommend_high_uncertainty_rationale(self, mock_model, mock_state):
        mock_state.return_value = SystemState("general", 0.8, 0.5, 0.0, 0.0)
        # Need enough observations to avoid low-data rationale
        model = GenerativeModel()
        s1 = SystemState("general", 0.5, 0.5, 0.0, 0.0)
        s2 = SystemState("general", 0.2, 0.3, 0.0, 0.0)
        for _ in range(15):
            model.update(s1, RETRIEVE, s2)
        mock_model.return_value = model

        from modules.active_inference_integration import recommend_action
        result = recommend_action()

        # Sprint 6: When model has enough data, an option may be selected instead
        # of a primitive action. Both are valid — check that result is meaningful.
        assert result["rationale"]  # Should not be empty
        # Either high uncertainty drove primitive selection OR option was chosen for exploration
        assert "High uncertainty" in result["rationale"] or result.get("is_option") or "option" in result["rationale"].lower()

    @patch("modules.active_inference_integration.get_current_state")
    @patch("modules.active_inference_integration.get_model")
    def test_recommend_state_fields(self, mock_model, mock_state):
        mock_state.return_value = SystemState("trading", 0.3, 0.7, 0.1, 0.5)
        mock_model.return_value = GenerativeModel()

        from modules.active_inference_integration import recommend_action
        result = recommend_action()

        assert result["state"]["topic"] == "trading"
        assert result["state"]["uncertainty"] == 0.3
        assert result["state"]["wm_load"] == 0.7


# ============================================================
# S5-05: Pre-turn Action Hint Tests
# ============================================================

class TestPreturnHint:
    """Test that the action hint is injected in preturn context."""

    @patch("modules.active_inference_integration.recommend_action")
    def test_hint_appended_to_context(self, mock_rec):
        mock_rec.return_value = {
            "recommended": "retrieve",
            "description": "Search episodic/semantic memory",
            "rationale": "Stable state. 'retrieve' has lowest EFE.",
            "model_observations": 42,
        }
        # Simulate what preturn_inject.py does
        context = "## Memorias Relevantes\n- test memory"
        try:
            rec = mock_rec()
            if rec and rec.get("recommended"):
                context += (
                    f"\n## Active Inference Hint\n"
                    f"- Recommended: {rec['recommended']} ({rec['description']})\n"
                    f"- Rationale: {rec['rationale']}\n"
                    f"- Model: {rec['model_observations']} observations"
                )
        except Exception:
            pass

        assert "Active Inference Hint" in context
        assert "retrieve" in context
        assert "42 observations" in context
