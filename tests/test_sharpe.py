"""
Tests for the Sharpe Cognitive Module.

Regression tests to prevent field-name mismatches and ensure
backward compatibility with legacy data.
"""

import unittest
from unittest.mock import patch
from modules.sharpe import (
    compute_utility,
    compute_uncertainty,
    compute_sharpe,
    compute_sharpe_for_payload,
    sharpe_score_candidates,
    SHARPE_EPSILON,
)


class TestSharpeFieldNames(unittest.TestCase):
    """Regression: ensure we read the correct Qdrant field names."""

    def test_attention_access_count_is_read(self):
        """attention_access_count=9 must produce higher Sharpe than acc=0."""
        high_acc = {
            "attention_access_count": 9,
            "narrative_importance": "high",
            "evidence_count": 1,
            "created_at": "2026-02-10T10:00:00",
        }
        zero_acc = {
            "narrative_importance": "high",
            "evidence_count": 1,
            "created_at": "2026-02-10T10:00:00",
        }
        r_high = compute_sharpe_for_payload(high_acc)
        r_zero = compute_sharpe_for_payload(zero_acc)
        self.assertGreater(
            r_high["sharpe"], r_zero["sharpe"],
            "attention_access_count=9 should produce higher Sharpe"
        )
        self.assertEqual(r_high["components"]["access_count"], 9)
        self.assertEqual(r_zero["components"]["access_count"], 0)

    def test_legacy_access_count_fallback(self):
        """Old memories with 'access_count' should still be read."""
        legacy = {
            "access_count": 5,
            "narrative_importance": "medium",
            "evidence_count": 1,
            "created_at": "2026-01-15T10:00:00",
        }
        result = compute_sharpe_for_payload(legacy)
        self.assertEqual(result["components"]["access_count"], 5)
        self.assertGreater(result["sharpe"], 1.0)

    def test_no_access_field_defaults_zero(self):
        """Missing both fields -> access_count=0, no crash."""
        bare = {
            "narrative_importance": "medium",
            "evidence_count": 1,
            "created_at": "2026-02-01T10:00:00",
        }
        result = compute_sharpe_for_payload(bare)
        self.assertEqual(result["components"]["access_count"], 0)
        self.assertGreater(result["sharpe"], 0)

    def test_attention_takes_priority_over_legacy(self):
        """When both fields exist, attention_access_count wins."""
        both = {
            "attention_access_count": 8,
            "access_count": 2,
            "narrative_importance": "medium",
        }
        result = compute_sharpe_for_payload(both)
        self.assertEqual(result["components"]["access_count"], 8)


class TestSharpeDiscrimination(unittest.TestCase):
    """Ensure Sharpe properly discriminates memory quality."""

    def test_high_quality_beats_low(self):
        high = compute_sharpe_for_payload({
            "attention_access_count": 10,
            "narrative_importance": "critical",
            "evidence_count": 3,
            "narrative_themes": ["consciencia", "trading"],
            "created_at": "2026-02-15T10:00:00",
        })
        low = compute_sharpe_for_payload({
            "attention_access_count": 0,
            "narrative_importance": "low",
            "evidence_count": 1,
            "narrative_themes": [],
            "created_at": "2026-01-05T10:00:00",
        })
        self.assertGreater(high["sharpe"], low["sharpe"] * 2,
                           "High-quality should be >2x low-quality")

    def test_contradicted_memory_lower(self):
        clean = compute_sharpe_for_payload({
            "attention_access_count": 3,
            "narrative_importance": "high",
            "contradiction_count": 0,
        })
        contradicted = compute_sharpe_for_payload({
            "attention_access_count": 3,
            "narrative_importance": "high",
            "contradiction_count": 3,
        })
        self.assertGreater(clean["sharpe"], contradicted["sharpe"])

    def test_cross_domain_boosts_sharpe(self):
        single_topic = compute_sharpe_for_payload({
            "attention_access_count": 3,
            "narrative_importance": "medium",
            "narrative_themes": ["consciencia"],
        })
        multi_topic = compute_sharpe_for_payload({
            "attention_access_count": 3,
            "narrative_importance": "medium",
            "narrative_themes": ["consciencia", "trading", "fullempaques"],
        })
        self.assertGreater(multi_topic["sharpe"], single_topic["sharpe"])


class TestShadowMode(unittest.TestCase):
    """Shadow mode must not modify candidate scores."""

    @patch("modules.sharpe.SHARPE_MODE", "shadow")
    def test_shadow_preserves_scores(self):
        candidates = [
            {"id": "a", "score": 0.9, "payload": {
                "attention_access_count": 5,
                "narrative_importance": "critical",
            }},
            {"id": "b", "score": 0.3, "payload": {
                "narrative_importance": "low",
            }},
        ]
        result = sharpe_score_candidates(candidates)
        self.assertEqual(result[0]["score"], 0.9)
        self.assertEqual(result[1]["score"], 0.3)
        # But sharpe fields should be added
        self.assertIn("sharpe_score", result[0])
        self.assertIn("sharpe_utility", result[0])
        self.assertIn("sharpe_uncertainty", result[0])


class TestOnMode(unittest.TestCase):
    """ON mode blends Sharpe into scores with p95 clamp."""

    def test_on_mode_changes_scores(self):
        import modules.sharpe as sharpe_mod
        original_mode = sharpe_mod.SHARPE_MODE
        try:
            sharpe_mod.SHARPE_MODE = "on"
            candidates = [
                {"id": "a", "score": 0.8, "payload": {
                    "attention_access_count": 10,
                    "narrative_importance": "critical",
                    "evidence_count": 3,
                    "narrative_themes": ["consciencia", "trading"],
                }},
                {"id": "b", "score": 0.8, "payload": {
                    "attention_access_count": 0,
                    "narrative_importance": "low",
                }},
            ]
            result = sharpe_score_candidates(candidates)
            # Scores should now differ (they were both 0.8 before)
            self.assertNotEqual(result[0]["score"], result[1]["score"])
            # High-access should score higher
            self.assertGreater(result[0]["score"], result[1]["score"])
            # Old score preserved for debugging
            self.assertEqual(result[0]["score_old"], 0.8)
        finally:
            sharpe_mod.SHARPE_MODE = original_mode

    def test_on_mode_preserves_order_for_clear_winners(self):
        import modules.sharpe as sharpe_mod
        original_mode = sharpe_mod.SHARPE_MODE
        try:
            sharpe_mod.SHARPE_MODE = "on"
            candidates = [
                {"id": "high", "score": 0.9, "payload": {
                    "attention_access_count": 8,
                    "narrative_importance": "critical",
                }},
                {"id": "mid", "score": 0.6, "payload": {
                    "attention_access_count": 3,
                    "narrative_importance": "high",
                }},
                {"id": "low", "score": 0.2, "payload": {
                    "attention_access_count": 0,
                    "narrative_importance": "low",
                }},
            ]
            result = sharpe_score_candidates(candidates)
            scores = {c["id"]: c["score"] for c in result}
            self.assertGreater(scores["high"], scores["mid"])
            self.assertGreater(scores["mid"], scores["low"])
        finally:
            sharpe_mod.SHARPE_MODE = original_mode


class TestEdgeCases(unittest.TestCase):
    """Edge cases and boundary conditions."""

    def test_epsilon_prevents_division_by_zero(self):
        s = compute_sharpe(utility=0.0, uncertainty=0.0)
        self.assertEqual(s, 1.0)  # epsilon/epsilon = 1

    def test_sharpe_clamped_to_max(self):
        s = compute_sharpe(utility=1.0, uncertainty=0.0)
        self.assertLessEqual(s, 10.0)

    def test_labile_increases_uncertainty(self):
        stable = compute_uncertainty(evidence_count=1, is_labile=False)
        labile = compute_uncertainty(evidence_count=1, is_labile=True)
        self.assertGreater(labile, stable)

    def test_none_values_dont_crash(self):
        payload = {
            "attention_access_count": None,
            "narrative_importance": None,
            "evidence_count": None,
            "narrative_themes": None,
            "created_at": None,
            "attention_last_accessed": None,
        }
        result = compute_sharpe_for_payload(payload)
        self.assertGreater(result["sharpe"], 0)


if __name__ == "__main__":
    unittest.main()
