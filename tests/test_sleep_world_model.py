#!/usr/bin/env python3
"""
Unit tests for Sleep World Model (Sprint 4, S4-05).
Run: python3 -m pytest tests/test_sleep_world_model.py -v
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import sqlite3
import pytest
from modules.sleep_loop import (
    SleepWorldModel,
    WORLD_MODEL_DIMS,
    TICK_ORDER,
)


# ============================================================
# FIXTURES
# ============================================================

@pytest.fixture
def model():
    return SleepWorldModel()


@pytest.fixture
def in_memory_db():
    """In-memory SQLite for persistence tests."""
    conn = sqlite3.connect(":memory:")
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


# ============================================================
# S4-05-01: STATE VECTOR
# ============================================================

class TestWorldModelDimensions:
    def test_eight_dimensions(self):
        """World model should have exactly 8 state dimensions."""
        assert len(WORLD_MODEL_DIMS) == 8

    def test_preferred_state_covers_all_dims(self, model):
        """Preferred state should have a value for every dimension."""
        for dim in WORLD_MODEL_DIMS:
            assert dim in model.PREFERRED

    def test_preferred_state_bounded(self, model):
        """All preferred values should be in [0, 1]."""
        for dim, val in model.PREFERRED.items():
            assert 0.0 <= val <= 1.0, f"{dim} = {val} out of bounds"

    def test_prior_effects_cover_all_ticks(self, model):
        """Prior effects should exist for all tick types."""
        for tick_name in TICK_ORDER:
            assert tick_name in model.PRIOR_EFFECTS


# ============================================================
# S4-05-02: FREE ENERGY
# ============================================================

class TestFreeEnergy:
    def test_zero_at_preferred(self, model):
        """Free energy should be zero when state equals preferred."""
        state = dict(model.PREFERRED)
        fe = model.compute_free_energy(state)
        assert fe == pytest.approx(0.0, abs=1e-10)

    def test_positive_away_from_preferred(self, model):
        """Free energy should be positive when state deviates."""
        state = {d: 1.0 for d in WORLD_MODEL_DIMS}
        fe = model.compute_free_energy(state)
        assert fe > 0

    def test_higher_deviation_higher_fe(self, model):
        """More deviation from preferred = higher free energy."""
        state_close = dict(model.PREFERRED)
        state_close["new_memories"] = 0.1

        state_far = dict(model.PREFERRED)
        state_far["new_memories"] = 0.9

        fe_close = model.compute_free_energy(state_close)
        fe_far = model.compute_free_energy(state_far)
        assert fe_far > fe_close

    def test_free_energy_bounded(self, model):
        """FE should be bounded between 0 and 1 (mean of squared diffs)."""
        worst = {d: 1.0 for d in WORLD_MODEL_DIMS}
        fe = model.compute_free_energy(worst)
        assert 0 <= fe <= 1.0


# ============================================================
# S4-05-03: SIMULATION
# ============================================================

class TestSimulation:
    def test_simulate_applies_prior_effects(self, model):
        """Simulation should apply prior effects to state."""
        state = {d: 0.5 for d in WORLD_MODEL_DIMS}
        result = model.simulate_tick(state, "consolidation")

        # Consolidation reduces new_memories and hours_since_consol
        assert result["new_memories"] < state["new_memories"]
        assert result["hours_since_consol"] < state["hours_since_consol"]
        # Other dims unchanged
        assert result["fts_gap"] == state["fts_gap"]

    def test_simulate_clamps_to_bounds(self, model):
        """Simulated state should always be in [0, 1]."""
        state = {d: 0.0 for d in WORLD_MODEL_DIMS}
        result = model.simulate_tick(state, "consolidation")
        for dim in WORLD_MODEL_DIMS:
            assert 0.0 <= result[dim] <= 1.0

    def test_simulate_unknown_tick(self, model):
        """Unknown tick should return state unchanged."""
        state = {d: 0.5 for d in WORLD_MODEL_DIMS}
        result = model.simulate_tick(state, "unknown_tick")
        assert result == state

    def test_simulate_self_model_no_effect(self, model):
        """Self model tick has no direct state effects."""
        state = {d: 0.5 for d in WORLD_MODEL_DIMS}
        result = model.simulate_tick(state, "self_model")
        assert result == state


# ============================================================
# S4-05-04: PRIORITIZATION
# ============================================================

class TestPrioritization:
    def test_returns_all_eligible(self, model):
        """Prioritize should return all eligible ticks."""
        eligible = ["health", "prospective", "consolidation"]
        ordered = model.prioritize_ticks(eligible)
        assert set(ordered) == set(eligible)
        assert len(ordered) == len(eligible)

    def test_high_need_tick_first(self, model):
        """Tick addressing biggest deviation should rank first."""
        # Manually set state: high fts_gap, everything else at preferred
        original_read = model.read_state

        def mock_read(conn=None):
            state = dict(model.PREFERRED)
            state["fts_gap"] = 0.9  # Big gap — health tick should fix
            return state

        model.read_state = mock_read
        try:
            ordered = model.prioritize_ticks(["consolidation", "health", "prospective"])
            assert ordered[0] == "health"
        finally:
            model.read_state = original_read

    def test_empty_eligible(self, model):
        """Empty eligible list should return empty."""
        assert model.prioritize_ticks([]) == []


# ============================================================
# S4-05-05: LEARNING
# ============================================================

class TestLearning:
    def test_learn_records_effects(self, model):
        """Learning should record observed effects."""
        before = {d: 0.5 for d in WORLD_MODEL_DIMS}
        after = dict(before)
        after["new_memories"] = 0.2  # Big change
        after["fts_gap"] = 0.3      # Big change

        model.learn_from_tick("consolidation", before, after)
        assert "consolidation" in model._learned_effects
        assert "new_memories" in model._learned_effects["consolidation"]
        assert model._learn_counts["consolidation"] == 1

    def test_learn_ignores_tiny_changes(self, model):
        """Changes < 0.01 should be ignored (noise)."""
        before = {d: 0.5 for d in WORLD_MODEL_DIMS}
        after = dict(before)
        after["fts_gap"] = 0.505  # Tiny change

        model.learn_from_tick("health", before, after)
        assert "health" not in model._learned_effects

    def test_learned_effects_used_after_threshold(self, model):
        """After MIN_OBS_FOR_LEARNED observations, learned effects override prior."""
        before = {d: 0.5 for d in WORLD_MODEL_DIMS}
        after = dict(before)
        after["fts_gap"] = 0.1  # Observed: health reduces gap by 0.4

        # Learn enough times
        for _ in range(model.MIN_OBS_FOR_LEARNED + 1):
            model.learn_from_tick("health", before, after)

        # Now simulation should use learned effect
        state = {d: 0.5 for d in WORLD_MODEL_DIMS}
        result = model.simulate_tick(state, "health")
        # Learned effect of -0.4 on fts_gap (EMA converges toward it)
        assert result["fts_gap"] < 0.5

    def test_ema_convergence(self, model):
        """EMA should converge toward consistent observations."""
        before = {d: 0.5 for d in WORLD_MODEL_DIMS}
        after = dict(before)
        after["new_memories"] = 0.1  # Consistent -0.4 delta

        for _ in range(20):
            model.learn_from_tick("consolidation", before, after)

        # Should converge close to -0.4
        learned = model._learned_effects["consolidation"]["new_memories"]
        assert -0.5 < learned < -0.3


# ============================================================
# S4-05-06: PERSISTENCE
# ============================================================

class TestPersistence:
    def test_persist_and_load(self, model, in_memory_db):
        """Learned effects should survive persist + load cycle."""
        # Teach model something
        before = {d: 0.5 for d in WORLD_MODEL_DIMS}
        after = dict(before)
        after["fts_gap"] = 0.1

        for _ in range(5):
            model.learn_from_tick("health", before, after)

        # Persist
        model.persist(in_memory_db)

        # Load into fresh model
        model2 = SleepWorldModel()
        model2.load(in_memory_db)

        assert "health" in model2._learned_effects
        assert "fts_gap" in model2._learned_effects["health"]
        assert model2._learn_counts["health"] >= 5

    def test_persist_empty_model(self, model, in_memory_db):
        """Persisting empty model should not error."""
        model.persist(in_memory_db)
        # No rows written
        rows = in_memory_db.execute(
            "SELECT COUNT(*) FROM world_model_effects"
        ).fetchone()[0]
        assert rows == 0

    def test_load_empty_db(self, in_memory_db):
        """Loading from empty DB should return model with defaults."""
        model = SleepWorldModel()
        model.load(in_memory_db)
        assert model._learned_effects == {}
        assert model._learn_counts == {}


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
