"""
Tests for Spotlight Policy (GWT) and PAD micro-updates.
Validates the pure functions, entry-point hooks, and PAD clamping.
"""

import unittest
from unittest.mock import patch, MagicMock
from datetime import datetime, timedelta

from modules.spotlight import (
    make_item,
    build_spotlight,
    select_goal,
    select_risk,
    select_next_action,
    update_spotlight_from_text,
    should_update_from_recall,
    get_spotlight,
    set_spotlight,
    clear_spotlight,
    format_spotlight,
    pad_micro_update,
    SPOTLIGHT_MAX_ITEMS,
    SPOTLIGHT_TTL_SECONDS,
    TYPE_ORDER,
)


# ============================================================
# Helpers
# ============================================================

def _make_intention(action, priority="medium", activation=0.5, action_type="execute"):
    return {
        "id": "test-1",
        "action": action,
        "action_type": action_type,
        "trigger_type": "manual",
        "trigger_spec": {},
        "priority": priority,
        "activation": activation,
    }


# ============================================================
# 1. max_3_and_order
# ============================================================

class TestMax3AndOrder(unittest.TestCase):
    """Spotlight enforces max 3 items and TYPE_ORDER ordering."""

    def test_max_3_items(self):
        items = build_spotlight(
            intentions=[_make_intention("do task A")],
            health_signals={"health_ok": False, "health_message": "disk full"},
            checkpoint_text="siguiente paso: deploy v2",
        )
        self.assertLessEqual(len(items), SPOTLIGHT_MAX_ITEMS)

    def test_order_is_goal_risk_next(self):
        """Items must be ordered: goal, risk, next_action."""
        items = build_spotlight(
            intentions=[_make_intention("build feature X")],
            health_signals={"health_ok": False, "health_message": "unhealthy"},
            checkpoint_text="siguiente paso: test it",
        )
        types = [i["type"] for i in items]
        # Filter to only types present and check order matches TYPE_ORDER
        expected_order = [t for t in TYPE_ORDER if t in types]
        self.assertEqual(types, expected_order)

    def test_empty_inputs_returns_empty(self):
        items = build_spotlight()
        self.assertEqual(items, [])

    def test_text_truncated_to_200(self):
        long_text = "x" * 500
        item = make_item("goal", long_text, "test")
        self.assertLessEqual(len(item["text"]), 200)


# ============================================================
# 2. despertar_populates
# ============================================================

class TestDespertarPopulates(unittest.TestCase):
    """build_spotlight produces correct items from intentions + health."""

    def test_intention_creates_goal(self):
        items = build_spotlight(
            intentions=[_make_intention("finish PR review")],
        )
        goals = [i for i in items if i["type"] == "goal"]
        self.assertEqual(len(goals), 1)
        self.assertIn("finish PR review", goals[0]["text"])

    def test_health_fail_creates_risk(self):
        items = build_spotlight(
            health_signals={"health_ok": False, "health_message": "qdrant down"},
        )
        risks = [i for i in items if i["type"] == "risk"]
        self.assertEqual(len(risks), 1)
        self.assertIn("qdrant down", risks[0]["text"])

    def test_intention_creates_next_action(self):
        items = build_spotlight(
            intentions=[_make_intention("deploy to prod", action_type="execute")],
        )
        actions = [i for i in items if i["type"] == "next_action"]
        self.assertEqual(len(actions), 1)


# ============================================================
# 3. checkpoint_overrides
# ============================================================

class TestCheckpointOverrides(unittest.TestCase):
    """update_spotlight_from_text replaces slots based on keywords."""

    def setUp(self):
        clear_spotlight()

    def test_next_action_keyword_updates_slot(self):
        initial = [make_item("next_action", "old action", "test")]
        updated = update_spotlight_from_text(initial, "siguiente paso: migrate DB", "checkpoint")
        actions = [i for i in updated if i["type"] == "next_action"]
        self.assertEqual(len(actions), 1)
        self.assertIn("migrate DB", actions[0]["text"])
        self.assertEqual(actions[0]["source"], "checkpoint")

    def test_risk_keyword_updates_slot(self):
        initial = [make_item("risk", "old risk", "test")]
        updated = update_spotlight_from_text(initial, "hay un bug critico en auth", "checkpoint")
        risks = [i for i in updated if i["type"] == "risk"]
        self.assertEqual(len(risks), 1)
        self.assertIn("bug", risks[0]["text"])

    def test_goal_keyword_updates_slot(self):
        initial = []
        updated = update_spotlight_from_text(initial, "objetivo: terminar sprint", "checkpoint")
        goals = [i for i in updated if i["type"] == "goal"]
        self.assertEqual(len(goals), 1)

    def test_no_keyword_no_change(self):
        initial = [make_item("goal", "keep this", "test")]
        updated = update_spotlight_from_text(initial, "just a regular message", "checkpoint")
        self.assertEqual(len(updated), 1)
        self.assertEqual(updated[0]["text"], "keep this")


# ============================================================
# 4. recall_no_update_low_signal
# ============================================================

class TestRecallNoUpdateLowSignal(unittest.TestCase):
    """should_update_from_recall returns False for weak signals."""

    def test_empty_results_no_update(self):
        self.assertFalse(should_update_from_recall([]))

    def test_one_result_no_critical_no_update(self):
        results = [{"text": "some normal memory", "source": "search"}]
        self.assertFalse(should_update_from_recall(results))

    def test_two_results_no_critical_no_update(self):
        results = [
            {"text": "memory 1", "source": "search"},
            {"text": "memory 2", "source": "search"},
        ]
        self.assertFalse(should_update_from_recall(results))

    def test_high_importance_triggers_update(self):
        results = [
            {"text": "memory 1", "source": "search", "meta": {"importance": "high"}},
        ]
        self.assertTrue(should_update_from_recall(results))

    def test_critical_keyword_triggers_update(self):
        results = [{"text": "this is critical for the system", "source": "search"}]
        self.assertTrue(should_update_from_recall(results))

    def test_bloqueo_keyword_triggers_update(self):
        results = [{"text": "hay un bloqueo en prod", "source": "search"}]
        self.assertTrue(should_update_from_recall(results))


# ============================================================
# 5. risk_priority
# ============================================================

class TestRiskPriority(unittest.TestCase):
    """Risk selection follows priority: dual_gate > write_queue > access_tracking > general."""

    def test_dual_gate_fail_highest_priority(self):
        signals = {
            "dual_gate": "FAIL",
            "dual_detail": "sync/async mismatch",
            "write_queue_dead": 5,
            "health_ok": False,
            "health_message": "everything broken",
        }
        risk = select_risk(signals)
        self.assertIsNotNone(risk)
        self.assertIn("DUAL gate FAIL", risk["text"])

    def test_dual_gate_warning(self):
        signals = {"dual_gate": "WARNING", "dual_detail": "high divergence"}
        risk = select_risk(signals)
        self.assertIsNotNone(risk)
        self.assertIn("WARNING", risk["text"])

    def test_write_queue_dead_second_priority(self):
        signals = {"write_queue_dead": 3, "health_ok": False, "health_message": "unhealthy"}
        risk = select_risk(signals)
        self.assertIsNotNone(risk)
        self.assertIn("dead=3", risk["text"])

    def test_write_queue_failed_ratio(self):
        signals = {"write_queue_failed_ratio": 0.15}
        risk = select_risk(signals)
        self.assertIsNotNone(risk)
        self.assertIn("15.0%", risk["text"])

    def test_access_tracking_errors(self):
        signals = {"access_tracking_errors": 2}
        risk = select_risk(signals)
        self.assertIsNotNone(risk)
        self.assertIn("Access tracking", risk["text"])

    def test_general_health_lowest_priority(self):
        signals = {"health_ok": False, "health_message": "qdrant timeout"}
        risk = select_risk(signals)
        self.assertIsNotNone(risk)
        self.assertIn("qdrant timeout", risk["text"])

    def test_healthy_returns_none(self):
        signals = {"health_ok": True}
        risk = select_risk(signals)
        self.assertIsNone(risk)

    def test_empty_signals_returns_none(self):
        risk = select_risk({})
        self.assertIsNone(risk)


# ============================================================
# 6. pad_clamped
# ============================================================

class TestPadClamped(unittest.TestCase):
    """PAD micro-updates clamp values to [-1, 1]."""

    def test_gate_pass_increases_pleasure(self):
        with patch("modules.config._emotional_state", {
            "current": {"pleasure": 0.5, "arousal": 0.0, "dominance": 0.0,
                        "timestamp": "", "trigger": ""}
        }):
            result = pad_micro_update("gate_pass")
            self.assertIsNotNone(result)
            self.assertAlmostEqual(result["pleasure"], 0.6)

    def test_error_retry_increases_arousal(self):
        with patch("modules.config._emotional_state", {
            "current": {"pleasure": 0.0, "arousal": 0.3, "dominance": 0.0,
                        "timestamp": "", "trigger": ""}
        }):
            result = pad_micro_update("error_retry")
            self.assertIsNotNone(result)
            self.assertAlmostEqual(result["arousal"], 0.4)

    def test_repeated_block_decreases_dominance(self):
        with patch("modules.config._emotional_state", {
            "current": {"pleasure": 0.0, "arousal": 0.0, "dominance": 0.5,
                        "timestamp": "", "trigger": ""}
        }):
            result = pad_micro_update("repeated_block")
            self.assertIsNotNone(result)
            self.assertAlmostEqual(result["dominance"], 0.4)

    def test_pleasure_clamped_at_1(self):
        with patch("modules.config._emotional_state", {
            "current": {"pleasure": 0.95, "arousal": 0.0, "dominance": 0.0,
                        "timestamp": "", "trigger": ""}
        }):
            result = pad_micro_update("gate_pass")
            self.assertLessEqual(result["pleasure"], 1.0)

    def test_dominance_clamped_at_minus_1(self):
        with patch("modules.config._emotional_state", {
            "current": {"pleasure": 0.0, "arousal": 0.0, "dominance": -0.95,
                        "timestamp": "", "trigger": ""}
        }):
            result = pad_micro_update("repeated_block")
            self.assertGreaterEqual(result["dominance"], -1.0)

    def test_arousal_clamped_at_1(self):
        with patch("modules.config._emotional_state", {
            "current": {"pleasure": 0.0, "arousal": 0.95, "dominance": 0.0,
                        "timestamp": "", "trigger": ""}
        }):
            result = pad_micro_update("error_retry")
            self.assertLessEqual(result["arousal"], 1.0)

    def test_unknown_signal_returns_none(self):
        with patch("modules.config._emotional_state", {
            "current": {"pleasure": 0.0, "arousal": 0.0, "dominance": 0.0,
                        "timestamp": "", "trigger": ""}
        }):
            result = pad_micro_update("unknown_signal")
            self.assertIsNone(result)


# ============================================================
# State management
# ============================================================

class TestStateManagement(unittest.TestCase):
    """get/set/clear/format spotlight state."""

    def setUp(self):
        clear_spotlight()

    def test_set_and_get(self):
        items = [make_item("goal", "test goal", "test")]
        set_spotlight(items)
        result = get_spotlight()
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["type"], "goal")

    def test_clear(self):
        set_spotlight([make_item("goal", "test", "test")])
        clear_spotlight()
        self.assertEqual(get_spotlight(), [])

    def test_format_empty(self):
        self.assertEqual(format_spotlight(), "")

    def test_format_with_items(self):
        set_spotlight([
            make_item("goal", "finish deploy", "intention"),
            make_item("risk", "disk full", "health"),
        ])
        output = format_spotlight()
        self.assertIn("SPOTLIGHT", output)
        self.assertIn("TARGET", output)
        self.assertIn("ALERT", output)
        self.assertIn("finish deploy", output)

    def test_dedup_keeps_latest_per_type(self):
        items = [
            make_item("goal", "old goal", "test"),
            make_item("goal", "new goal", "test"),
        ]
        set_spotlight(items)
        result = get_spotlight()
        goals = [i for i in result if i["type"] == "goal"]
        self.assertEqual(len(goals), 1)
        self.assertEqual(goals[0]["text"], "new goal")


if __name__ == "__main__":
    unittest.main()
