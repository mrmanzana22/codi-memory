#!/usr/bin/env python3
"""
Unit tests for Active Inference Foundation (Sprint 4).
Run: python3 -m pytest tests/test_active_inference.py -v
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import math
import random
import pytest
from modules.active_inference import (
    Action,
    SystemState,
    GenerativeModel,
    ALL_ACTIONS,
    ACTION_MAP,
    RETRIEVE, STORE, CONSOLIDATE, FORGET, ATTEND, EXPLORE,
    compute_efe,
    select_action,
    _entropy,
    DEFAULT_PREFERENCES,
)


# ============================================================
# S4-01: ACTION SPACE
# ============================================================

class TestActionSpace:
    def test_six_actions_defined(self):
        """All 6 canonical actions should be defined."""
        assert len(ALL_ACTIONS) == 6

    def test_action_names_unique(self):
        """Action names should be unique."""
        names = [a.name for a in ALL_ACTIONS]
        assert len(names) == len(set(names))

    def test_action_map_lookup(self):
        """ACTION_MAP should map name -> Action."""
        assert ACTION_MAP["retrieve"] == RETRIEVE
        assert ACTION_MAP["consolidate"] == CONSOLIDATE

    def test_action_costs_bounded(self):
        """All action costs should be in [0, 1]."""
        for action in ALL_ACTIONS:
            assert 0 <= action.cost <= 1.0

    def test_action_frozen(self):
        """Actions should be immutable (frozen dataclass)."""
        with pytest.raises(AttributeError):
            RETRIEVE.cost = 0.9


# ============================================================
# S4-02: SYSTEM STATE + GENERATIVE MODEL
# ============================================================

def _make_state(topic="general", uncertainty=0.5, wm_load=0.5,
                pe=0.0, valence=0.0):
    return SystemState(
        topic=topic,
        uncertainty=uncertainty,
        wm_load=wm_load,
        pe_magnitude=pe,
        emotional_valence=valence,
    )


class TestSystemState:
    def test_as_tuple_discretizes(self):
        """State should discretize uncertainty and wm_load into low/mid/high."""
        s = _make_state(uncertainty=0.1, wm_load=0.9)
        t = s.as_tuple()
        assert t == ("general", "low", "high")

    def test_as_tuple_mid_range(self):
        s = _make_state(topic="trading", uncertainty=0.5, wm_load=0.5)
        t = s.as_tuple()
        assert t == ("trading", "mid", "mid")


class TestGenerativeModel:
    def test_empty_model_uniform_prediction(self):
        """With no data, prediction should be uniform over known states."""
        model = GenerativeModel()
        state = _make_state()
        dist = model.predict_next_state(state, RETRIEVE)
        # Only one state known (current)
        assert len(dist) == 1
        assert sum(dist.values()) == pytest.approx(1.0)

    def test_update_learns_transition(self):
        """After observing transitions, model should predict them."""
        model = GenerativeModel()
        s1 = _make_state(topic="general", uncertainty=0.5)
        s2 = _make_state(topic="trading", uncertainty=0.3)

        # Observe: general + retrieve -> trading
        for _ in range(5):
            model.update(s1, RETRIEVE, s2)

        dist = model.predict_next_state(s1, RETRIEVE)
        # Should strongly predict trading
        trading_key = s2.as_tuple()
        assert dist[trading_key] > 0.5

    def test_dirichlet_smoothing(self):
        """Prediction should have Dirichlet smoothing (no zero probabilities)."""
        model = GenerativeModel()
        s1 = _make_state(topic="general")
        s2 = _make_state(topic="trading")
        s3 = _make_state(topic="codigo")

        model.update(s1, RETRIEVE, s2)
        model.update(s1, RETRIEVE, s3)

        dist = model.predict_next_state(s1, RETRIEVE)
        # Both should have non-zero probability
        for p in dist.values():
            assert p > 0

    def test_observation_count(self):
        model = GenerativeModel()
        s1 = _make_state()
        s2 = _make_state(topic="trading")

        model.update(s1, RETRIEVE, s2)
        model.update(s1, STORE, s2)
        assert model.observation_count == 2

    def test_expected_uncertainty(self):
        """Uncertainty should be higher with more diverse transitions."""
        model = GenerativeModel()
        s = _make_state()

        # Single deterministic transition
        s_next = _make_state(topic="trading")
        for _ in range(10):
            model.update(s, RETRIEVE, s_next)
        low_h = model.get_expected_uncertainty(s, RETRIEVE)

        # Add diverse transitions
        model2 = GenerativeModel()
        for topic in ["trading", "codigo", "consciencia", "fullempaques"]:
            s_n = _make_state(topic=topic)
            for _ in range(3):
                model2.update(s, RETRIEVE, s_n)
        high_h = model2.get_expected_uncertainty(s, RETRIEVE)

        assert high_h > low_h


# ============================================================
# S4-03: EXPECTED FREE ENERGY + POLICY SELECTION
# ============================================================

class TestEFE:
    def test_efe_returns_float(self):
        """EFE should return a finite float."""
        model = GenerativeModel()
        state = _make_state()
        g = compute_efe(state, RETRIEVE, model)
        assert isinstance(g, float)
        assert math.isfinite(g)

    def test_cheaper_action_preferred(self):
        """All else equal, cheaper actions should have lower EFE."""
        model = GenerativeModel()
        state = _make_state()
        g_attend = compute_efe(state, ATTEND, model)      # cost 0.2
        g_consolidate = compute_efe(state, CONSOLIDATE, model)  # cost 0.8
        # Attend should be preferred (lower G)
        assert g_attend < g_consolidate

    def test_efe_with_learned_model(self):
        """EFE should differ when model has learned transitions."""
        model = GenerativeModel()
        state = _make_state()

        # Before learning: EFE is based on priors
        g_before = compute_efe(state, RETRIEVE, model)

        # Learn that RETRIEVE reduces uncertainty
        good_state = _make_state(uncertainty=0.1, wm_load=0.5)
        for _ in range(10):
            model.update(state, RETRIEVE, good_state)

        g_after = compute_efe(state, RETRIEVE, model)
        # After learning, RETRIEVE should be more attractive (lower G)
        assert g_after < g_before


class TestPolicySelection:
    def test_select_returns_action(self):
        """select_action should return an Action (or Option) and EFE dict."""
        from modules.active_inference import Option, OPTION_MAP
        model = GenerativeModel()
        state = _make_state()
        action, efes = select_action(state, model)
        # Sprint 6: may return Action or Option when include_options=True (default)
        assert isinstance(action, (Action, Option))
        assert action.name in {**ACTION_MAP, **OPTION_MAP}
        assert len(efes) >= len(ALL_ACTIONS)

    def test_deterministic_with_high_temperature(self):
        """High temperature should select the best action consistently."""
        random.seed(42)
        model = GenerativeModel()
        state = _make_state()

        # Run many selections with high temp
        counts = {}
        for _ in range(100):
            action, _ = select_action(state, model, temperature=50.0)
            counts[action.name] = counts.get(action.name, 0) + 1

        # Most selected action should dominate
        best = max(counts, key=counts.get)
        assert counts[best] >= 50  # Should be selected > 50% of time

    def test_low_temperature_more_exploratory(self):
        """Low temperature should spread selections more evenly."""
        random.seed(42)
        model = GenerativeModel()
        state = _make_state()

        counts = {}
        for _ in range(100):
            action, _ = select_action(state, model, temperature=0.1)
            counts[action.name] = counts.get(action.name, 0) + 1

        # Should select multiple different actions
        assert len(counts) >= 3

    def test_custom_action_subset(self):
        """Should work with a subset of actions."""
        model = GenerativeModel()
        state = _make_state()
        action, efes = select_action(state, model, actions=[RETRIEVE, ATTEND])
        assert action in [RETRIEVE, ATTEND]
        assert len(efes) == 2


class TestEntropy:
    def test_uniform_entropy(self):
        """Uniform distribution should have maximum entropy."""
        dist = {"a": 0.25, "b": 0.25, "c": 0.25, "d": 0.25}
        h = _entropy(dist)
        assert h == pytest.approx(2.0)  # log2(4) = 2

    def test_deterministic_entropy(self):
        """Deterministic distribution should have zero entropy."""
        dist = {"a": 1.0, "b": 0.0}
        h = _entropy(dist)
        assert h == pytest.approx(0.0)

    def test_binary_entropy(self):
        dist = {"a": 0.5, "b": 0.5}
        h = _entropy(dist)
        assert h == pytest.approx(1.0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
