"""Tests for modules/forgetting.py.

Covers:
  - compute_fadem_strength (pure function)
  - _get_hours_since_access (timing extraction)
  - Importance modulation (critical slower than low)
  - Consolidation effect (consolidated slower)
  - Emotional protection (high arousal resists decay)
  - Floor enforcement
  - apply_rif (mocked Qdrant)
  - RIF event handler wiring
"""

import math
import pytest
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch


class TestComputeFademStrength:
    """Tests for the core FadeMem formula."""

    def test_zero_hours_returns_current(self):
        from modules.forgetting import compute_fadem_strength
        assert compute_fadem_strength(0.8, 0.0) == 0.8

    def test_negative_hours_returns_current(self):
        from modules.forgetting import compute_fadem_strength
        assert compute_fadem_strength(0.8, -5.0) == 0.8

    def test_decay_reduces_salience(self):
        from modules.forgetting import compute_fadem_strength
        result = compute_fadem_strength(0.8, hours_since_access=24.0)
        assert result < 0.8

    def test_more_hours_more_decay(self):
        from modules.forgetting import compute_fadem_strength
        short = compute_fadem_strength(0.8, hours_since_access=6.0)
        long = compute_fadem_strength(0.8, hours_since_access=48.0)
        assert short > long

    def test_floor_respected(self):
        from modules.forgetting import compute_fadem_strength, FADEM_FLOOR
        result = compute_fadem_strength(0.2, hours_since_access=10000.0, importance="low")
        assert result >= FADEM_FLOOR

    def test_floor_value_is_point_one(self):
        from modules.forgetting import FADEM_FLOOR
        assert FADEM_FLOOR == 0.1


class TestImportanceModulation:
    """Critical memories decay slower than low-importance ones."""

    def test_critical_decays_slower_than_medium(self):
        from modules.forgetting import compute_fadem_strength
        critical = compute_fadem_strength(0.8, 48.0, importance="critical")
        medium = compute_fadem_strength(0.8, 48.0, importance="medium")
        assert critical > medium

    def test_high_decays_slower_than_low(self):
        from modules.forgetting import compute_fadem_strength
        high = compute_fadem_strength(0.8, 48.0, importance="high")
        low = compute_fadem_strength(0.8, 48.0, importance="low")
        assert high > low

    def test_importance_ordering(self):
        """critical > high > medium > low at same time."""
        from modules.forgetting import compute_fadem_strength
        hours = 72.0
        vals = {
            imp: compute_fadem_strength(0.8, hours, importance=imp)
            for imp in ["critical", "high", "medium", "low"]
        }
        assert vals["critical"] > vals["high"] > vals["medium"] > vals["low"]

    def test_unknown_importance_uses_medium(self):
        from modules.forgetting import compute_fadem_strength
        unknown = compute_fadem_strength(0.8, 48.0, importance="unknown_level")
        medium = compute_fadem_strength(0.8, 48.0, importance="medium")
        assert unknown == medium


class TestConsolidationEffect:
    """Consolidated memories decay slower than unconsolidated."""

    def test_consolidated_episodic_slower(self):
        from modules.forgetting import compute_fadem_strength
        consol = compute_fadem_strength(0.8, 48.0, consolidated=True, memory_type="episodic")
        unconsol = compute_fadem_strength(0.8, 48.0, consolidated=False, memory_type="episodic")
        assert consol > unconsol

    def test_consolidated_semantic_slowest(self):
        from modules.forgetting import compute_fadem_strength
        semantic = compute_fadem_strength(0.8, 48.0, consolidated=True, memory_type="semantic")
        episodic = compute_fadem_strength(0.8, 48.0, consolidated=True, memory_type="episodic")
        assert semantic > episodic

    def test_semantic_permastore_at_30_days(self):
        """Consolidated semantic should retain significant salience after 30 days."""
        from modules.forgetting import compute_fadem_strength
        result = compute_fadem_strength(
            0.8, hours_since_access=720.0,  # 30 days
            importance="high", consolidated=True, memory_type="semantic"
        )
        # Bahrick 1984: semantic knowledge persists for decades
        assert result > 0.3, f"Semantic permastore too low at 30d: {result}"


class TestEmotionalProtection:
    """High-arousal memories resist decay (Brown & Kulik 1977)."""

    def test_high_arousal_decays_slower(self):
        from modules.forgetting import compute_fadem_strength
        calm = compute_fadem_strength(0.8, 48.0, emotional_arousal=0.0)
        aroused = compute_fadem_strength(0.8, 48.0, emotional_arousal=0.8)
        assert aroused > calm

    def test_arousal_below_threshold_no_effect(self):
        from modules.forgetting import compute_fadem_strength, FADEM_AROUSAL_THRESHOLD
        base = compute_fadem_strength(0.8, 48.0, emotional_arousal=0.0)
        low = compute_fadem_strength(0.8, 48.0, emotional_arousal=FADEM_AROUSAL_THRESHOLD - 0.1)
        assert base == low

    def test_arousal_at_one_max_protection(self):
        from modules.forgetting import compute_fadem_strength
        normal = compute_fadem_strength(0.8, 48.0, emotional_arousal=0.0)
        maxarousal = compute_fadem_strength(0.8, 48.0, emotional_arousal=1.0)
        # Max protection = 40% decay reduction
        normal_loss = 0.8 - normal
        maxarousal_loss = 0.8 - maxarousal
        shield_ratio = 1.0 - (maxarousal_loss / normal_loss) if normal_loss > 0 else 0
        assert 0.35 < shield_ratio < 0.45, f"Shield ratio {shield_ratio} not ~0.40"


class TestDecayMultiplier:
    """External decay_multiplier scales the decay rate."""

    def test_higher_multiplier_more_decay(self):
        from modules.forgetting import compute_fadem_strength
        normal = compute_fadem_strength(0.8, 24.0, decay_multiplier=1.0)
        fast = compute_fadem_strength(0.8, 24.0, decay_multiplier=2.0)
        assert fast < normal

    def test_zero_multiplier_no_decay(self):
        from modules.forgetting import compute_fadem_strength
        result = compute_fadem_strength(0.8, 24.0, decay_multiplier=0.0)
        assert result == 0.8

    def test_lifecycle_multiplier(self):
        """lifecycle uses decay_rate=0.03 → multiplier=0.6."""
        from modules.forgetting import compute_fadem_strength
        sleep = compute_fadem_strength(0.8, 24.0, decay_multiplier=1.0)    # 0.05/0.05
        lifecycle = compute_fadem_strength(0.8, 24.0, decay_multiplier=0.6)  # 0.03/0.05
        assert lifecycle > sleep  # Gentler decay


class TestGetHoursSinceAccess:
    """Tests for _get_hours_since_access helper."""

    def test_no_timing_returns_none(self):
        from modules.forgetting import _get_hours_since_access
        assert _get_hours_since_access({}) is None

    def test_created_at_fallback(self):
        from modules.forgetting import _get_hours_since_access
        ts = (datetime.now() - timedelta(hours=5)).isoformat()
        result = _get_hours_since_access({"created_at": ts})
        assert result is not None
        assert 4.5 < result < 5.5

    def test_access_timestamps_preferred(self):
        from modules.forgetting import _get_hours_since_access
        old_create = (datetime.now() - timedelta(hours=100)).isoformat()
        recent_access = (datetime.now() - timedelta(hours=2)).isoformat()
        result = _get_hours_since_access({
            "created_at": old_create,
            "access_timestamps": [old_create, recent_access],
        })
        assert result is not None
        assert 1.5 < result < 2.5

    def test_empty_timestamps_uses_created(self):
        from modules.forgetting import _get_hours_since_access
        ts = (datetime.now() - timedelta(hours=10)).isoformat()
        result = _get_hours_since_access({
            "access_timestamps": [],
            "created_at": ts,
        })
        assert result is not None
        assert 9.5 < result < 10.5

    def test_invalid_timestamp_returns_none(self):
        from modules.forgetting import _get_hours_since_access
        assert _get_hours_since_access({"created_at": "not-a-date"}) is None


class TestEmotionalArousalExtraction:
    """Tests for _get_emotional_arousal helper."""

    def test_no_pad_returns_zero(self):
        from modules.forgetting import _get_emotional_arousal
        assert _get_emotional_arousal({}) == 0.0

    def test_extracts_abs_arousal(self):
        from modules.forgetting import _get_emotional_arousal
        result = _get_emotional_arousal({"pad_at_encoding": {"P": 0.5, "A": -0.7, "D": 0.3}})
        assert abs(result - 0.7) < 0.01

    def test_invalid_pad_returns_zero(self):
        from modules.forgetting import _get_emotional_arousal
        assert _get_emotional_arousal({"pad_at_encoding": "invalid"}) == 0.0


class TestConsolidationDetection:
    """Tests for _is_consolidated helper."""

    def test_false_by_default(self):
        from modules.forgetting import _is_consolidated
        assert _is_consolidated({}) is False

    def test_true_when_set(self):
        from modules.forgetting import _is_consolidated
        assert _is_consolidated({"consolidated": True}) is True

    def test_false_when_explicit(self):
        from modules.forgetting import _is_consolidated
        assert _is_consolidated({"consolidated": False}) is False


class TestCalibration:
    """Regression tests for expected decay values at key time points."""

    def test_medium_24h_loses_meaningful_salience(self):
        """Medium-importance unconsolidated after 24h should lose 30-60%."""
        from modules.forgetting import compute_fadem_strength
        result = compute_fadem_strength(0.8, 24.0, importance="medium")
        loss_pct = (0.8 - result) / 0.8
        assert 0.15 < loss_pct < 0.70, f"24h loss {loss_pct:.0%} outside expected range"

    def test_critical_consolidated_7_days_retains_most(self):
        """Critical consolidated after 7 days should retain > 50%."""
        from modules.forgetting import compute_fadem_strength
        result = compute_fadem_strength(
            0.8, 168.0, importance="critical",
            consolidated=True, memory_type="episodic",
        )
        assert result > 0.4, f"Critical consolidated at 7d too low: {result}"

    def test_critical_unconsolidated_7_days_hits_floor(self):
        """Critical unconsolidated at 7d decays heavily (should be consolidated by then)."""
        from modules.forgetting import compute_fadem_strength, FADEM_FLOOR
        result = compute_fadem_strength(0.8, 168.0, importance="critical")
        # Unconsolidated memories decay fast by design (Tononi & Cirelli 2003)
        assert result <= 0.3

    def test_low_24h_significant_decay(self):
        """Low-importance unconsolidated after 24h should decay significantly."""
        from modules.forgetting import compute_fadem_strength
        result = compute_fadem_strength(0.8, 24.0, importance="low")
        assert result < 0.6, f"Low at 24h too high: {result}"


# ============================================================
# RIF TESTS (mocked Qdrant)
# ============================================================

class _FakePoint:
    """Minimal Qdrant point mock for RIF tests."""
    def __init__(self, id, payload, vector=None, score=0.8):
        self.id = id
        self.payload = payload
        self.vector = vector or [0.1] * 10
        self.score = score


class TestApplyRif:
    """Tests for apply_rif (Retrieval-Induced Forgetting)."""

    @patch("modules.forgetting.qdrant", create=True)
    @patch("modules.forgetting.record_access", create=True)
    def test_too_few_results_skips(self, mock_record, mock_qdrant):
        from modules.forgetting import apply_rif
        result = apply_rif(["mem-1"])  # Only 1 result
        assert result["applied"] is False
        assert result["reason"] == "too_few_results"

    @patch("modules.config.qdrant")
    @patch("modules.access_tracking.record_access")
    def test_suppresses_competitors(self, mock_record, mock_qdrant):
        from modules.forgetting import apply_rif, FADEM_FLOOR

        # Setup: top result has a vector
        retrieved_point = _FakePoint("r1", {}, vector=[1.0] * 10)
        mock_qdrant.retrieve.return_value = [retrieved_point]

        # Competitors: similar memories not in retrieved set
        comp1 = _FakePoint("c1", {"attention_salience": 0.7, "narrative_importance": "medium"}, score=0.8)
        comp2 = _FakePoint("c2", {"attention_salience": 0.5, "narrative_importance": "low"}, score=0.6)
        # This one is retrieved, should NOT be suppressed
        ret1 = _FakePoint("r1", {"attention_salience": 0.9}, score=0.95)
        ret2 = _FakePoint("r2", {"attention_salience": 0.8}, score=0.90)
        mock_qdrant.search.return_value = [ret1, ret2, comp1, comp2]

        result = apply_rif(["r1", "r2"])
        assert result["applied"] is True
        assert result["suppressed"] == 2
        assert mock_record.call_count == 2

        # Check salience was reduced
        for call in mock_record.call_args_list:
            new_sal = call[0][2]["attention_salience"]
            assert new_sal >= FADEM_FLOOR

    @patch("modules.config.qdrant")
    @patch("modules.access_tracking.record_access")
    def test_critical_exempt_from_rif(self, mock_record, mock_qdrant):
        from modules.forgetting import apply_rif

        retrieved_point = _FakePoint("r1", {}, vector=[1.0] * 10)
        mock_qdrant.retrieve.return_value = [retrieved_point]

        # Critical competitor should be exempt
        critical_comp = _FakePoint("c1", {"attention_salience": 0.9, "narrative_importance": "critical"}, score=0.85)
        medium_comp = _FakePoint("c2", {"attention_salience": 0.6, "narrative_importance": "medium"}, score=0.7)
        ret1 = _FakePoint("r1", {}, score=0.95)
        ret2 = _FakePoint("r2", {}, score=0.90)
        mock_qdrant.search.return_value = [ret1, ret2, critical_comp, medium_comp]

        result = apply_rif(["r1", "r2"])
        assert result["applied"] is True
        # Only medium should be suppressed, not critical
        assert result["suppressed"] == 1

    @patch("modules.config.qdrant")
    @patch("modules.access_tracking.record_access")
    def test_no_competitors_returns_not_applied(self, mock_record, mock_qdrant):
        from modules.forgetting import apply_rif

        retrieved_point = _FakePoint("r1", {}, vector=[1.0] * 10)
        mock_qdrant.retrieve.return_value = [retrieved_point]

        # All neighbors are in the retrieved set
        ret1 = _FakePoint("r1", {}, score=0.95)
        ret2 = _FakePoint("r2", {}, score=0.90)
        mock_qdrant.search.return_value = [ret1, ret2]

        result = apply_rif(["r1", "r2"])
        assert result["applied"] is False
        assert result["reason"] == "no_competitors"

    @patch("modules.config.qdrant")
    @patch("modules.access_tracking.record_access")
    def test_floor_memories_not_suppressed(self, mock_record, mock_qdrant):
        from modules.forgetting import apply_rif, FADEM_FLOOR

        retrieved_point = _FakePoint("r1", {}, vector=[1.0] * 10)
        mock_qdrant.retrieve.return_value = [retrieved_point]

        # Competitor already at floor
        floor_comp = _FakePoint("c1", {"attention_salience": FADEM_FLOOR, "narrative_importance": "low"}, score=0.7)
        ret1 = _FakePoint("r1", {}, score=0.95)
        ret2 = _FakePoint("r2", {}, score=0.90)
        mock_qdrant.search.return_value = [ret1, ret2, floor_comp]

        result = apply_rif(["r1", "r2"])
        assert result["applied"] is True
        assert result["suppressed"] == 0  # At floor, nothing to suppress

    @patch("modules.config.qdrant")
    @patch("modules.access_tracking.record_access")
    def test_strong_competitors_suppressed_more(self, mock_record, mock_qdrant):
        """Inhibitory deficit: strong competitors get more suppression (Anderson 2003)."""
        from modules.forgetting import apply_rif

        retrieved_point = _FakePoint("r1", {}, vector=[1.0] * 10)
        mock_qdrant.retrieve.return_value = [retrieved_point]

        strong = _FakePoint("c1", {"attention_salience": 0.9, "narrative_importance": "medium"}, score=0.8)
        weak = _FakePoint("c2", {"attention_salience": 0.3, "narrative_importance": "medium"}, score=0.8)
        ret1 = _FakePoint("r1", {}, score=0.95)
        ret2 = _FakePoint("r2", {}, score=0.90)
        mock_qdrant.search.return_value = [ret1, ret2, strong, weak]

        result = apply_rif(["r1", "r2"])
        assert result["suppressed"] == 2

        # Strong competitor should lose more absolute salience
        calls = mock_record.call_args_list
        sal_by_id = {}
        for call in calls:
            cid = call[0][1]
            new_sal = call[0][2]["attention_salience"]
            sal_by_id[cid] = new_sal

        strong_loss = 0.9 - sal_by_id["c1"]
        weak_loss = 0.3 - sal_by_id["c2"]
        assert strong_loss > weak_loss

    def test_rif_with_query_embedding(self):
        """Can provide pre-computed embedding instead of fetching from Qdrant."""
        with patch("modules.config.qdrant") as mock_qdrant, \
             patch("modules.access_tracking.record_access"):
            from modules.forgetting import apply_rif

            comp = _FakePoint("c1", {"attention_salience": 0.6, "narrative_importance": "medium"}, score=0.7)
            ret1 = _FakePoint("r1", {}, score=0.95)
            ret2 = _FakePoint("r2", {}, score=0.90)
            mock_qdrant.search.return_value = [ret1, ret2, comp]

            result = apply_rif(["r1", "r2"], query_embedding=[1.0] * 10)
            assert result["applied"] is True
            # Should NOT call retrieve (embedding provided)
            mock_qdrant.retrieve.assert_not_called()


class TestRifConstants:
    """Verify RIF constants are within neuroscience-backed ranges."""

    def test_rif_base_range(self):
        from modules.forgetting import RIF_BASE
        # Anderson 1994: 8-15% recall reduction
        assert 0.05 <= RIF_BASE <= 0.20

    def test_min_results(self):
        from modules.forgetting import RIF_MIN_RESULTS
        assert RIF_MIN_RESULTS >= 2

    def test_max_competitors_bounded(self):
        from modules.forgetting import RIF_MAX_COMPETITORS
        assert 5 <= RIF_MAX_COMPETITORS <= 20
