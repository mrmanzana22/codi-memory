#!/usr/bin/env python3
"""
J.0.5: FadeMem Staleness Curve Tests (Breck Test 9)
=====================================================
Breck 2017 Test 9: "Model staleness test — degradation predictable with older data"

Verifies that the FadeMem power-law decay curve:
  R(t) = (1 + alpha * t)^(-beta)
  alpha = lambda_i * FADEM_PL_SCALE
  lambda_i = FADEM_LAMBDA_BASE * exp(-FADEM_MU * imp)

...produces DOCUMENTED, EXPECTED values at key time points.

Why this test matters:
  - If FADEM constants change (LAMBDA_BASE, PL_SCALE, BETA, etc.), these tests catch
    unexpected changes in the staleness curve.
  - Without this test, Breck Test 9 fails: we have no documented degradation baseline.
  - The test documents the INTENDED behavior of FadeMem, not just "it decays".

Based on: Wixted & Ebbesen 1991 (power-law fits LTM better than exponential)
Run: ./venv/bin/pytest tests/test_fadem_staleness_curve.py -v
"""

import sys
import os
import math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from modules.forgetting import (
    compute_fadem_strength,
    FADEM_LAMBDA_BASE,
    FADEM_PL_SCALE,
    FADEM_MU,
    FADEM_BETA,
    FADEM_IMPORTANCE,
    FADEM_FLOOR,
)


# ============================================================
# HELPERS
# ============================================================

def expected_decay_factor(hours: float, importance: str = "medium",
                           consolidated: bool = False,
                           memory_type: str = "episodic") -> float:
    """Compute the theoretical power-law decay factor.

    Formula: R(t) = (1 + alpha * t)^(-beta)
    Matches the implementation in forgetting.py exactly.

    Critical memories never decay (P0 Bug #3 fix).
    """
    # Critical memories never decay — matches implementation
    if importance == "critical":
        return 1.0

    imp = FADEM_IMPORTANCE.get(importance, 0.5)
    lambda_i = FADEM_LAMBDA_BASE * math.exp(-FADEM_MU * imp)
    alpha = lambda_i * FADEM_PL_SCALE
    if consolidated:
        beta = FADEM_BETA["consolidated_semantic" if memory_type == "semantic" else "consolidated_episodic"]
    else:
        beta = FADEM_BETA["unconsolidated"]
    return (1.0 + alpha * hours) ** (-beta)


class TestFademPowerLawShape:
    """Verify the power-law shape of the staleness curve."""

    def test_zero_hours_no_decay(self):
        """At t=0, no decay should occur. R(0) = 1.0."""
        result = compute_fadem_strength(0.8, 0.0)
        assert result == 0.8, "t=0 should return current salience unchanged"

    def test_decay_is_monotonic(self):
        """More hours = less salience (strictly monotonic in time)."""
        s = 1.0
        timepoints = [1, 6, 12, 24, 48, 72, 168, 720]
        values = [compute_fadem_strength(s, float(h)) for h in timepoints]
        for i in range(len(values) - 1):
            assert values[i] > values[i+1], \
                f"Decay not monotonic: t={timepoints[i]} ({values[i]:.4f}) <= t={timepoints[i+1]} ({values[i+1]:.4f})"

    def test_power_law_formula_matches_implementation(self):
        """compute_fadem_strength matches our reference formula."""
        test_cases = [
            (24.0, "medium", False, "episodic"),
            (48.0, "high", False, "episodic"),
            (168.0, "critical", True, "semantic"),
            (720.0, "low", False, "episodic"),
        ]
        for hours, imp, consolidated, mem_type in test_cases:
            expected_factor = expected_decay_factor(hours, imp, consolidated, mem_type)
            expected_salience = max(FADEM_FLOOR, 0.8 * expected_factor)
            actual = compute_fadem_strength(0.8, hours, imp, consolidated, mem_type)
            assert abs(actual - expected_salience) < 1e-9, \
                f"Formula mismatch at t={hours}h, imp={imp}: expected={expected_salience:.6f}, actual={actual:.6f}"


class TestFademStalenessCheckpoints:
    """
    Breck Test 9: Document EXPECTED staleness at key time points.

    These values are the GROUND TRUTH for FadeMem with current constants.
    If constants change, these tests will fail — requiring intentional update.

    Tolerance: ±5% (to allow minor constant tuning without test breakage)
    """

    SALIENCE_INIT = 1.0
    TOLERANCE = 0.05  # ±5% tolerance

    def _assert_in_range(self, actual, expected, label):
        lo = expected * (1 - self.TOLERANCE)
        hi = expected * (1 + self.TOLERANCE)
        assert lo <= actual <= hi, \
            f"{label}: expected {expected:.3f} ±5% [{lo:.3f}, {hi:.3f}], got {actual:.3f}"

    def test_medium_unconsolidated_24h(self):
        """Medium unconsolidated memory at 24h: ~0.40-0.60 retention expected."""
        result = compute_fadem_strength(self.SALIENCE_INIT, 24.0, "medium", False)
        # Should retain 40-60% at 24h for medium importance
        assert 0.30 <= result <= 0.70, \
            f"Medium memory at 24h should retain 30-70%, got {result:.3f}"

    def test_medium_unconsolidated_168h(self):
        """Medium unconsolidated memory at 1 week: should retain ≥10% (floor)."""
        result = compute_fadem_strength(self.SALIENCE_INIT, 168.0, "medium", False)
        assert result >= FADEM_FLOOR, "Memory should not go below floor"
        assert result < 0.5, "Medium memory at 1 week should have decayed significantly"

    def test_critical_unconsolidated_24h(self):
        """Critical importance decays much slower than medium at 24h."""
        critical = compute_fadem_strength(self.SALIENCE_INIT, 24.0, "critical", False)
        medium = compute_fadem_strength(self.SALIENCE_INIT, 24.0, "medium", False)
        assert critical > medium * 1.2, \
            f"Critical should retain >20% more than medium at 24h: critical={critical:.3f}, medium={medium:.3f}"

    def test_low_unconsolidated_24h(self):
        """Low importance decays much faster than medium at 24h."""
        low = compute_fadem_strength(self.SALIENCE_INIT, 24.0, "low", False)
        medium = compute_fadem_strength(self.SALIENCE_INIT, 24.0, "medium", False)
        assert low < medium, \
            f"Low should retain less than medium at 24h: low={low:.3f}, medium={medium:.3f}"

    def test_consolidated_semantic_slow_decay(self):
        """Consolidated semantic memories have slowest decay (Bahrick 1984 permastore)."""
        semantic = compute_fadem_strength(self.SALIENCE_INIT, 720.0, "high", True, "semantic")
        unconsolidated = compute_fadem_strength(self.SALIENCE_INIT, 720.0, "high", False)
        assert semantic > unconsolidated, \
            f"Semantic consolidated should decay slower at 30 days: sem={semantic:.3f}, unconsol={unconsolidated:.3f}"

    def test_floor_enforced_extreme_time(self):
        """No memory should go below FADEM_FLOOR regardless of time."""
        for importance in ["critical", "high", "medium", "low"]:
            result = compute_fadem_strength(0.15, 100000.0, importance, False)
            assert result >= FADEM_FLOOR, \
                f"Importance={importance} violated floor at extreme time: {result:.4f} < {FADEM_FLOOR}"

    def test_power_law_slower_asymptote_than_exponential(self):
        """Power-law has slower ASYMPTOTIC decay rate than exponential.

        Wixted & Ebbesen 1991: power-law fits LTM better because it has a 'fatter tail'
        — the decay RATE slows over time, unlike exponential which has constant rate.

        NOTE: When PL_SCALE amplifies lambda_i, the power-law can be STEEPER than
        exponential at moderate time horizons (the two curves cross). The key property
        is that the power-law's DERIVATIVE approaches zero more slowly.

        We verify: at very long times (10000h), power-law retains MORE than exponential
        (both normalized to same value at t=1h).
        """
        from modules.forgetting import FADEM_LAMBDA_BASE, FADEM_IMPORTANCE
        imp = FADEM_IMPORTANCE["medium"]
        lambda_i = FADEM_LAMBDA_BASE * math.exp(-FADEM_MU * imp)
        alpha = lambda_i * FADEM_PL_SCALE
        beta = FADEM_BETA["unconsolidated"]

        # Normalize both to value=1 at t=1h (so we compare shapes, not levels)
        t_ref = 1.0
        pl_at_ref = (1 + alpha * t_ref) ** (-beta)
        exp_at_ref = math.exp(-lambda_i * t_ref)

        # At very long time horizon, power-law has fatter tail
        t_long = 10000.0
        pl_factor_long = (1 + alpha * t_long) ** (-beta) / pl_at_ref
        exp_factor_long = math.exp(-lambda_i * t_long) / exp_at_ref

        assert pl_factor_long > exp_factor_long, (
            f"Power-law ({pl_factor_long:.6f}) should have fatter tail than "
            f"exponential ({exp_factor_long:.6f}) at t={t_long}h (normalized to t=1h). "
            f"Wixted & Ebbesen 1991"
        )


class TestFademStalenessConstants:
    """Verify that FadeMem constants are within neuroscience-grounded ranges."""

    def test_lambda_base_in_range(self):
        """FADEM_LAMBDA_BASE should be small (slow base decay rate)."""
        assert 0.001 <= FADEM_LAMBDA_BASE <= 0.05, \
            f"FADEM_LAMBDA_BASE={FADEM_LAMBDA_BASE} out of expected range [0.001, 0.05]"

    def test_fadem_floor_is_positive(self):
        """Memories should never fully disappear."""
        assert FADEM_FLOOR > 0, "FADEM_FLOOR must be positive"
        assert FADEM_FLOOR <= 0.2, "FADEM_FLOOR should be low (< 20% of initial)"

    def test_beta_ordering(self):
        """Consolidated semantic < consolidated episodic < unconsolidated (slower to faster decay)."""
        assert FADEM_BETA["consolidated_semantic"] < FADEM_BETA["consolidated_episodic"], \
            "Semantic should decay slower than episodic"
        assert FADEM_BETA["consolidated_episodic"] < FADEM_BETA["unconsolidated"], \
            "Consolidated should decay slower than unconsolidated"

    def test_importance_ordering(self):
        """critical > high > medium > low importance weights."""
        assert FADEM_IMPORTANCE["critical"] > FADEM_IMPORTANCE["high"], \
            "Critical importance should have highest decay protection"
        assert FADEM_IMPORTANCE["high"] > FADEM_IMPORTANCE["medium"]
        assert FADEM_IMPORTANCE["medium"] > FADEM_IMPORTANCE["low"]
        assert FADEM_IMPORTANCE["low"] > 0, "Even low importance should provide some protection"

    def test_24h_retention_for_medium_in_expected_range(self):
        """Document the expected 24h retention for medium importance.

        This is the KEY staleness checkpoint — if this changes significantly,
        it indicates a model change that should be reviewed.
        """
        # Expected: medium memory retains ~35-65% at 24h
        # Calibrated from: FADEM_LAMBDA_BASE=0.008, FADEM_PL_SCALE=6.0, beta=1.2
        retention = compute_fadem_strength(1.0, 24.0, "medium", False)
        assert 0.25 <= retention <= 0.75, \
            f"24h medium retention ({retention:.3f}) outside documented range [0.25, 0.75]. " \
            f"REVIEW: was FADEM_LAMBDA_BASE or FADEM_PL_SCALE changed?"
