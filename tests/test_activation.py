#!/usr/bin/env python3
"""
Unit tests for the Unified Activation Scoring (WIRING-5).
Run: python3 -m pytest tests/test_activation.py -v
"""

import sys
import os
import math
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from modules.activation import (
    compute_unified_activation,
    compute_effective_decay,
    _compute_base_level,
    ActivationResult,
    ACTR_DECAY_DEFAULT,
    DECAY_SEMANTIC,
    DECAY_WORKING_MEMORY,
    DECAY_EPISODIC_CRITICAL,
    DECAY_EPISODIC_HIGH,
    IMPORTANCE_BOOST,
)


# ============================================================
# CALIBRATION REGRESSION TESTS
# ============================================================

class TestCalibrationRegression:
    """Verify calibrated activation values match Phase 0 targets."""

    def test_30day_old_0_accesses(self):
        """30-day-old memory, 0 accesses -> ~0.10"""
        created = (datetime.now() - timedelta(days=30)).isoformat()
        result = compute_unified_activation(
            created_at=created, access_count=0, noise=False
        )
        assert 0.05 < result.total < 0.18, f"30d/0acc should be ~0.10, got {result.total}"

    def test_1day_old_0_accesses(self):
        """1-day-old memory, 0 accesses -> ~0.28"""
        created = (datetime.now() - timedelta(days=1)).isoformat()
        result = compute_unified_activation(
            created_at=created, access_count=0, noise=False
        )
        assert 0.20 < result.total < 0.40, f"1d/0acc should be ~0.28, got {result.total}"

    def test_1day_old_7_accesses(self):
        """1-day-old memory, 7 recent accesses -> ~0.84"""
        now = datetime.now()
        created = (now - timedelta(days=1)).isoformat()
        last = (now - timedelta(hours=1)).isoformat()
        result = compute_unified_activation(
            created_at=created, last_accessed=last,
            access_count=7, noise=False
        )
        assert 0.70 < result.total < 0.95, f"1d/7acc should be ~0.84, got {result.total}"


# ============================================================
# MODULATION EFFECTS
# ============================================================

class TestModulationEffects:
    def _baseline(self):
        """Medium memory, no modulation, deterministic."""
        created = (datetime.now() - timedelta(days=3)).isoformat()
        return compute_unified_activation(
            created_at=created, access_count=2, noise=False
        )

    def test_critical_importance_boosts(self):
        """Critical importance should increase activation vs. medium."""
        created = (datetime.now() - timedelta(days=3)).isoformat()
        base = compute_unified_activation(
            created_at=created, access_count=2, importance="medium", noise=False
        )
        boosted = compute_unified_activation(
            created_at=created, access_count=2, importance="critical", noise=False
        )
        assert boosted.total > base.total, \
            f"Critical ({boosted.total}) should > medium ({base.total})"
        assert boosted.importance_boost > base.importance_boost

    def test_high_arousal_boosts(self):
        """High emotional arousal should increase activation."""
        created = (datetime.now() - timedelta(days=3)).isoformat()
        calm = compute_unified_activation(
            created_at=created, access_count=2,
            emotional_arousal=0.0, noise=False
        )
        aroused = compute_unified_activation(
            created_at=created, access_count=2,
            emotional_arousal=0.8, noise=False
        )
        assert aroused.total > calm.total, \
            f"Aroused ({aroused.total}) should > calm ({calm.total})"

    def test_prediction_error_boosts(self):
        """High prediction error should increase activation."""
        created = (datetime.now() - timedelta(days=3)).isoformat()
        normal = compute_unified_activation(
            created_at=created, access_count=2,
            prediction_error=0.0, noise=False
        )
        surprised = compute_unified_activation(
            created_at=created, access_count=2,
            prediction_error=0.8, noise=False
        )
        assert surprised.total > normal.total, \
            f"Surprised ({surprised.total}) should > normal ({normal.total})"

    def test_spreading_boost(self):
        """Spreading activation should increase total."""
        created = (datetime.now() - timedelta(days=3)).isoformat()
        cold = compute_unified_activation(
            created_at=created, access_count=2,
            spreading_boost=0.0, noise=False
        )
        hot = compute_unified_activation(
            created_at=created, access_count=2,
            spreading_boost=0.9, noise=False
        )
        assert hot.total > cold.total, \
            f"Hot ({hot.total}) should > cold ({cold.total})"

    def test_all_modulations_maxed_bounded(self):
        """All modulations maxed should still be <= 1.0."""
        created = (datetime.now() - timedelta(hours=1)).isoformat()
        result = compute_unified_activation(
            created_at=created, access_count=10,
            spreading_boost=1.0, emotional_arousal=1.0,
            emotional_congruence=1.0, importance="critical",
            prediction_error=1.0, noise=False
        )
        assert 0.0 <= result.total <= 1.0, f"Total out of bounds: {result.total}"
        assert result.total > 0.90, f"All maxed should be near ceiling: {result.total}"

    def test_modulations_dont_overwhelm_base(self):
        """A 30-day-old memory + all modulations should not outrank a 1-hour-old."""
        old = compute_unified_activation(
            created_at=(datetime.now() - timedelta(days=30)).isoformat(),
            access_count=0, importance="critical",
            spreading_boost=1.0, prediction_error=1.0,
            noise=False
        )
        fresh = compute_unified_activation(
            created_at=(datetime.now() - timedelta(hours=1)).isoformat(),
            access_count=5, noise=False
        )
        # Base-level should dominate: fresh memory wins despite old having max modulations
        assert fresh.total > old.total, \
            f"Fresh ({fresh.total}) should outrank old+maxmod ({old.total})"


# ============================================================
# DECAY TYPE DIFFERENTIATION
# ============================================================

class TestDecayTypes:
    def test_semantic_more_durable_than_episodic(self):
        """Semantic memory should retain more activation over 7 days."""
        created = (datetime.now() - timedelta(days=7)).isoformat()
        episodic = compute_unified_activation(
            created_at=created, access_count=1,
            memory_type="episodic", noise=False
        )
        semantic = compute_unified_activation(
            created_at=created, access_count=1,
            memory_type="semantic", noise=False
        )
        assert semantic.total > episodic.total, \
            f"Semantic ({semantic.total}) should > episodic ({episodic.total})"

    def test_wm_decays_faster(self):
        """Working memory should decay faster than episodic."""
        created = (datetime.now() - timedelta(days=3)).isoformat()
        episodic = compute_unified_activation(
            created_at=created, access_count=1,
            memory_type="episodic", noise=False
        )
        wm = compute_unified_activation(
            created_at=created, access_count=1,
            memory_type="working_memory", noise=False
        )
        assert wm.total < episodic.total, \
            f"WM ({wm.total}) should < episodic ({episodic.total})"

    def test_critical_episodic_decays_slower(self):
        """Critical episodic should decay slower (flashbulb effect)."""
        created = (datetime.now() - timedelta(days=7)).isoformat()
        medium = compute_unified_activation(
            created_at=created, access_count=1,
            importance="medium", memory_type="episodic", noise=False
        )
        critical = compute_unified_activation(
            created_at=created, access_count=1,
            importance="critical", memory_type="episodic", noise=False
        )
        assert critical.total > medium.total, \
            f"Critical ({critical.total}) should > medium ({medium.total})"


# ============================================================
# EFFECTIVE DECAY
# ============================================================

class TestEffectiveDecay:
    def test_default_episodic(self):
        d = compute_effective_decay(importance="medium", memory_type="episodic")
        assert d == ACTR_DECAY_DEFAULT

    def test_critical_slower(self):
        d = compute_effective_decay(importance="critical", memory_type="episodic")
        assert d == DECAY_EPISODIC_CRITICAL
        assert d < ACTR_DECAY_DEFAULT

    def test_semantic_slower_than_episodic(self):
        d_e = compute_effective_decay(memory_type="episodic")
        d_s = compute_effective_decay(memory_type="semantic")
        assert d_s < d_e

    def test_semantic_with_evidence_reduces_decay(self):
        d0 = compute_effective_decay(memory_type="semantic", evidence_count=0)
        d5 = compute_effective_decay(memory_type="semantic", evidence_count=5)
        assert d5 < d0, "More evidence should reduce decay"

    def test_semantic_evidence_has_floor(self):
        d = compute_effective_decay(memory_type="semantic", evidence_count=100)
        assert d >= 0.05, f"Decay should not go below floor: {d}"

    def test_wm_fastest(self):
        d = compute_effective_decay(memory_type="working_memory")
        assert d == DECAY_WORKING_MEMORY
        assert d > ACTR_DECAY_DEFAULT

    def test_high_arousal_slows_episodic_decay(self):
        d_calm = compute_effective_decay(
            importance="medium", memory_type="episodic", emotional_arousal=0.0
        )
        d_aroused = compute_effective_decay(
            importance="medium", memory_type="episodic", emotional_arousal=0.8
        )
        assert d_aroused < d_calm, "High arousal should slow decay"

    def test_override_takes_precedence(self):
        d = compute_effective_decay(
            importance="critical", memory_type="episodic",
            decay_override=0.55
        )
        assert d == 0.55


# ============================================================
# NOISE STOCHASTICITY
# ============================================================

class TestNoise:
    def test_noise_produces_variation(self):
        """With noise=True, 100 runs should produce variation."""
        created = (datetime.now() - timedelta(days=3)).isoformat()
        results = []
        for _ in range(100):
            r = compute_unified_activation(
                created_at=created, access_count=2, noise=True
            )
            results.append(r.total)
        std = (sum((x - sum(results)/len(results))**2 for x in results) / len(results)) ** 0.5
        assert std > 0.01, f"Noise should produce variation, std={std}"

    def test_no_noise_deterministic(self):
        """With noise=False, results should be identical."""
        created = (datetime.now() - timedelta(days=3)).isoformat()
        results = set()
        for _ in range(20):
            r = compute_unified_activation(
                created_at=created, access_count=2, noise=False
            )
            results.add(r.total)
        assert len(results) == 1, f"Deterministic should produce 1 value, got {len(results)}"


# ============================================================
# RESULT STRUCTURE
# ============================================================

class TestResultStructure:
    def test_returns_activation_result(self):
        r = compute_unified_activation(
            created_at=datetime.now().isoformat(), noise=False
        )
        assert isinstance(r, ActivationResult)
        assert hasattr(r, 'total')
        assert hasattr(r, 'base_level')
        assert hasattr(r, 'spread')
        assert hasattr(r, 'emotion')
        assert hasattr(r, 'importance_boost')
        assert hasattr(r, 'surprise_boost')
        assert hasattr(r, 'noise_value')

    def test_components_sum_to_raw(self):
        """Components should add up to the pre-sigmoid raw value."""
        created = (datetime.now() - timedelta(days=2)).isoformat()
        r = compute_unified_activation(
            created_at=created, access_count=3,
            spreading_boost=0.5, emotional_arousal=0.4,
            importance="high", prediction_error=0.3,
            noise=False
        )
        raw = r.base_level + r.spread + r.emotion + r.importance_boost + r.surprise_boost + r.noise_value
        # Verify sigmoid of raw matches total
        shifted = (raw - (-0.07)) * 0.91
        expected_total = 1.0 / (1.0 + math.exp(-shifted))
        assert abs(r.total - round(expected_total, 4)) < 0.001, \
            f"Components don't match: total={r.total}, expected={expected_total}"


# ============================================================
# EDGE CASES
# ============================================================

class TestEdgeCases:
    def test_no_created_at_defaults_30_days(self):
        """Missing created_at should default to 30 days ago."""
        r = compute_unified_activation(created_at="", noise=False)
        assert 0.05 < r.total < 0.25, f"Default should be like 30d old: {r.total}"

    def test_empty_access_timestamps(self):
        """Empty list should fallback gracefully."""
        r = compute_unified_activation(
            created_at=(datetime.now() - timedelta(days=1)).isoformat(),
            access_timestamps=[], noise=False
        )
        assert 0.15 < r.total < 0.45, f"Empty timestamps should fallback: {r.total}"

    def test_high_access_count_near_ceiling(self):
        """Very high access count should approach ceiling."""
        r = compute_unified_activation(
            created_at=(datetime.now() - timedelta(hours=2)).isoformat(),
            last_accessed=(datetime.now() - timedelta(minutes=5)).isoformat(),
            access_count=100, noise=False
        )
        assert r.total > 0.85, f"High access should be near ceiling: {r.total}"

    def test_all_zero_modulations(self):
        """Zero modulations should produce only base_level + importance(medium)."""
        created = (datetime.now() - timedelta(days=3)).isoformat()
        r = compute_unified_activation(
            created_at=created, access_count=1,
            spreading_boost=0.0, emotional_arousal=0.0,
            importance="low", prediction_error=0.0,
            noise=False
        )
        assert r.spread == 0.0
        assert r.emotion == 0.0
        assert r.importance_boost == 0.0
        assert r.surprise_boost == 0.0
        assert r.noise_value == 0.0

    def test_negative_congruence_ignored(self):
        """Negative emotional congruence should be clamped at 0."""
        created = (datetime.now() - timedelta(days=3)).isoformat()
        r = compute_unified_activation(
            created_at=created, emotional_congruence=-0.8, noise=False
        )
        # Emotion component should only come from arousal (which is 0)
        assert r.emotion == 0.0


# ============================================================
# BACKWARD COMPATIBILITY
# ============================================================

class TestBackwardCompatibility:
    def test_base_level_matches_utils(self):
        """_compute_base_level should match utils.compute_base_level_activation."""
        from modules.utils import compute_base_level_activation
        created = (datetime.now() - timedelta(days=5)).isoformat()
        last = (datetime.now() - timedelta(hours=6)).isoformat()

        old_val = compute_base_level_activation(
            created_at_str=created,
            last_accessed_str=last,
            access_count=3,
        )
        new_val = _compute_base_level(
            created_at_str=created,
            last_accessed_str=last,
            access_count=3,
        )
        assert abs(old_val - new_val) < 0.001, \
            f"Values should match: old={old_val}, new={new_val}"


# ============================================================
# TRACE CALLBACK (P0-2 Observability)
# ============================================================

class TestTraceCallback:
    def test_callback_receives_result(self):
        """trace_callback should be called with memory_id, result, decay."""
        captured = {}
        def my_trace(memory_id, result, decay):
            captured['memory_id'] = memory_id
            captured['result'] = result
            captured['decay'] = decay

        created = (datetime.now() - timedelta(days=3)).isoformat()
        r = compute_unified_activation(
            memory_id="test-123",
            created_at=created, access_count=2, noise=False,
            trace_callback=my_trace,
        )
        assert captured['memory_id'] == "test-123"
        assert captured['result'].total == r.total
        assert captured['decay'] == ACTR_DECAY_DEFAULT

    def test_callback_none_is_noop(self):
        """trace_callback=None should not raise."""
        r = compute_unified_activation(
            created_at=datetime.now().isoformat(), noise=False,
            trace_callback=None,
        )
        assert r.total > 0

    def test_callback_exception_swallowed(self):
        """A failing callback should not break activation computation."""
        def bad_trace(**kwargs):
            raise RuntimeError("boom")

        r = compute_unified_activation(
            created_at=(datetime.now() - timedelta(days=1)).isoformat(),
            noise=False, trace_callback=bad_trace,
        )
        assert 0.0 <= r.total <= 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
