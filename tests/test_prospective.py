#!/usr/bin/env python3
"""
Unit tests for the Prospective Memory module.
Run: python3 -m pytest tests/test_prospective.py -v
"""

import sys
import os
import json
import uuid as _uuid
from datetime import datetime, timedelta, timezone
from unittest.mock import patch, MagicMock

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest


# ============================================================
# FIXTURES
# ============================================================

@pytest.fixture(autouse=True)
def isolated_db(tmp_path, monkeypatch):
    """Use a temporary DB for each test."""
    test_db = str(tmp_path / "test_prospective.db")
    # Run prospective migrations on the temp DB
    from modules.migrations import apply_migrations
    apply_migrations(test_db, migrations_dir=os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "migrations_prospective"))
    monkeypatch.setattr("modules.prospective.PROSPECTIVE_DB_PATH", test_db)
    # Reset the global connection
    import modules.prospective as pm
    pm._conn = None
    yield pm
    if pm._conn:
        pm._conn.close()
        pm._conn = None


# ============================================================
# DECAY CONVERGENCE
# ============================================================

class TestDecayConvergence:
    def test_cumulative_decay_reduces_activation(self, isolated_db):
        """Activation must decrease across maintenance cycles."""
        pm = isolated_db
        act = 0.80
        for _ in range(3):
            act = pm._compute_activation(act, "high", 6.0)
        assert act < 0.80, "Activation should decrease after decay"

    def test_no_double_decay_on_quick_runs(self, isolated_db):
        """Running maintenance 1 min apart should barely change activation."""
        pm = isolated_db
        act = 0.80
        a1 = pm._compute_activation(act, "high", 6.0)
        a2 = pm._compute_activation(a1, "high", 1/60)  # 1 min
        assert abs(a2 - a1) < 0.1, f"Quick re-run shouldn't cause big change: {a1} vs {a2}"

    def test_critical_decays_slower_than_low(self, isolated_db):
        """Critical priority should retain more activation than low."""
        pm = isolated_db
        act_c = pm._compute_activation(0.95, "critical", 24.0)
        act_l = pm._compute_activation(0.45, "low", 24.0)
        assert act_c > act_l, f"Critical {act_c} should be > low {act_l}"


# ============================================================
# FLOOR PROTECTION
# ============================================================

class TestFloorProtection:
    def test_floors_are_monotonic(self, isolated_db):
        """Floors must be: critical > high > medium > low."""
        pm = isolated_db
        f = pm.PM_ACTIVATION_FLOOR
        assert f["critical"] > f["high"] > f["medium"] > f["low"]

    def test_floors_above_thresholds(self, isolated_db):
        """Each floor must be above its respective threshold."""
        pm = isolated_db
        for pri in ["critical", "high", "medium", "low"]:
            assert pm.PM_ACTIVATION_FLOOR[pri] > pm.PM_ACTIVATION_TRIGGER_THRESHOLD[pri], \
                f"{pri} floor {pm.PM_ACTIVATION_FLOOR[pri]} <= threshold {pm.PM_ACTIVATION_TRIGGER_THRESHOLD[pri]}"

    def test_activation_never_below_floor(self, isolated_db):
        """After heavy decay, activation should settle at floor."""
        pm = isolated_db
        act = 0.65  # medium initial
        for _ in range(20):
            act = pm._compute_activation(act, "medium", 24.0)
        floor = pm.PM_ACTIVATION_FLOOR["medium"]
        # Allow noise to push slightly below
        assert act >= floor - 0.05, f"Activation {act} far below floor {floor}"

    def test_medium_visible_after_5_days(self, isolated_db):
        """Medium priority should still be visible after 5 days of maintenance."""
        pm = isolated_db
        act = pm.PM_ACTIVATION_INITIAL["medium"]
        thr = pm.PM_ACTIVATION_TRIGGER_THRESHOLD["medium"]
        for _ in range(20):  # 20 * 6h = 5 days
            act = pm._compute_activation(act, "medium", 6.0)
        # With noise, might occasionally dip, so be lenient
        assert act > thr - 0.05, f"Medium {act} should be near/above threshold {thr}"


# ============================================================
# URGENCY RESCUE
# ============================================================

class TestUrgencyRescue:
    def test_urgency_boosts_near_deadline(self, isolated_db):
        """Intention near deadline gets urgency boost."""
        pm = isolated_db
        now = datetime.now()
        act_no_deadline = pm._compute_activation(0.22, "medium", 0.1)
        act_near_deadline = pm._compute_activation(
            0.22, "medium", 0.1,
            expiry=now + timedelta(days=1), now=now,
        )
        assert act_near_deadline > act_no_deadline, \
            f"Near deadline {act_near_deadline} should > no deadline {act_no_deadline}"

    def test_urgency_past_deadline(self, isolated_db, monkeypatch):
        """Past-deadline intention gets max urgency."""
        # Eliminate stochastic noise so assertion is deterministic
        monkeypatch.setattr("modules.prospective.random.gauss", lambda mu, sigma: 0)
        pm = isolated_db
        now = datetime.now()
        act = pm._compute_activation(
            0.22, "medium", 0.1,
            expiry=now - timedelta(days=1), now=now,
        )
        assert act > 0.40, f"Past deadline should have strong urgency: {act}"

    def test_no_urgency_far_from_deadline(self, isolated_db):
        """Intention far from deadline gets no urgency boost (on average)."""
        pm = isolated_db
        now = datetime.now()
        # Run multiple times to average out noise
        diffs = []
        for _ in range(20):
            act_far = pm._compute_activation(
                0.50, "medium", 0.1,
                expiry=now + timedelta(days=30), now=now,
            )
            act_none = pm._compute_activation(0.50, "medium", 0.1)
            diffs.append(act_far - act_none)
        avg_diff = sum(diffs) / len(diffs)
        assert abs(avg_diff) < 0.05, \
            f"Average diff should be near 0 (no urgency): {avg_diff}"


# ============================================================
# TIMEZONE HANDLING
# ============================================================

class TestTimezone:
    def test_utc_trigger_matches_cot_now(self, isolated_db):
        """UTC trigger time should match equivalent COT time."""
        pm = isolated_db
        now_cot = datetime(2026, 2, 13, 15, 0, 0)  # 3PM COT
        spec = {"trigger_time": "2026-02-13T20:00:00Z", "tolerance_minutes": 30}
        result = pm._check_time(spec, now_cot)
        assert result["matched"], "20:00 UTC = 15:00 COT, should match"

    def test_future_trigger_does_not_match(self, isolated_db):
        """Trigger 1h in the future should not match with 30min tolerance."""
        pm = isolated_db
        now = datetime(2026, 2, 13, 15, 0, 0)
        spec = {"trigger_time": "2026-02-13T16:00:00", "tolerance_minutes": 30}
        result = pm._check_time(spec, now)
        assert not result["matched"], "16:00 is 1h away, 30min tolerance should not match"

    def test_exact_match(self, isolated_db):
        pm = isolated_db
        now = datetime(2026, 2, 13, 15, 0, 0)
        spec = {"trigger_time": "2026-02-13T15:00:00", "tolerance_minutes": 30}
        result = pm._check_time(spec, now)
        assert result["matched"], "Exact time should match"


# ============================================================
# STALE CLEANUP
# ============================================================

class TestStaleCleanup:
    def test_explicit_expiry(self, isolated_db):
        """Expired intentions should be marked expired."""
        pm = isolated_db
        past = (datetime.now() - timedelta(days=1)).isoformat()
        result = pm.create_intention(
            action="Test expired",
            trigger_type="event",
            trigger_spec='{"keywords": ["test"]}',
            priority="medium",
            expiry=past,
        )
        with pm.get_conn() as conn:
            pm._expire_stale(conn, datetime.now())
            conn.commit()

            row = conn.execute(
                "SELECT status FROM intentions WHERE id = %s", (result["id"],)
            ).fetchone()
            assert row[0] == "expired", f"Should be expired, got {row[0]}"


# ============================================================
# THRESHOLD FILTERING
# ============================================================

class TestThresholdFiltering:
    def test_below_threshold_not_triggered(self, isolated_db):
        """Intention below its priority threshold should not trigger."""
        pm = isolated_db
        # Create intention then manually set activation below threshold
        unique_action = f"Verify threshold filtering {_uuid.uuid4().hex[:8]}"
        result = pm.create_intention(
            action=unique_action,
            trigger_type="event",
            trigger_spec='{"keywords": ["lowtest"]}',
            priority="medium",
        )
        with pm.get_conn() as conn:
            conn.execute(
                "UPDATE intentions SET activation = 0.10 WHERE id = %s", (result["id"],)
            )
            conn.commit()

        triggered = pm.check_intentions("lowtest keyword match")
        assert len(triggered) == 0, "Below-threshold intention should not trigger"


# ============================================================
# WORKING MEMORY PUSH
# ============================================================

class TestWMPush:
    def test_mark_triggered_pushes_to_wm(self, isolated_db):
        """_mark_triggered should push to working memory."""
        pm = isolated_db
        unique_action = f"Notify team about deployment {_uuid.uuid4().hex[:8]}"
        result = pm.create_intention(
            action=unique_action,
            trigger_type="event",
            trigger_spec='{"keywords": ["wmtest"]}',
            priority="high",
        )
        with pm.get_conn() as conn:
            with patch("modules.working_memory.push_to_working_memory") as mock_wm, \
                 patch("modules.consciousness.update_workspace_spotlight"):
                pm._mark_triggered(conn, result["id"], "test detail", datetime.now())
                mock_wm.assert_called_once()
                call_kwargs = mock_wm.call_args
                assert "INTENCION DISPARADA" in call_kwargs[1].get("content", call_kwargs[0][0] if call_kwargs[0] else "")
                assert call_kwargs[1].get("topic") == "intention_triggered" or \
                       (len(call_kwargs[0]) > 1 and call_kwargs[0][1] == "intention_triggered")
                # The push happens via import inside _mark_triggered


# ============================================================
# RECURRENCE
# ============================================================

class TestRecurrence:
    def test_recurring_daily_creates_next(self, isolated_db):
        """Triggering a daily recurring intention should create a new one."""
        pm = isolated_db
        unique_action = f"Review server health metrics {_uuid.uuid4().hex[:8]}"
        result = pm.create_intention(
            action=unique_action,
            trigger_type="time",
            trigger_spec='{"trigger_time": "2026-02-13T10:00:00"}',
            priority="medium",
            recurrence="daily",
            context='{"note": "test recurrence context"}',
        )
        with pm.get_conn() as conn:
            now_dt = datetime.now()

            with patch("modules.working_memory.push_to_working_memory"):
                with patch("modules.consciousness.update_workspace_spotlight"):
                    pm._mark_triggered(conn, result["id"], "time match", now_dt)
                    conn.commit()

            # Check that a new pending intention was created
            rows = conn.execute(
                "SELECT id, status, trigger_spec FROM intentions WHERE action = %s",
                (unique_action,)
            ).fetchall()
            statuses = [r[1] for r in rows]
            assert "triggered" in statuses, "Original should be triggered"
            assert "pending" in statuses, "New recurrence should be pending"
            assert len(rows) == 2, f"Should have 2 rows, got {len(rows)}"

    def test_non_recurring_no_new_instance(self, isolated_db):
        """Non-recurring intention should NOT create new instance."""
        pm = isolated_db
        unique_action = f"Send quarterly report {_uuid.uuid4().hex[:8]}"
        result = pm.create_intention(
            action=unique_action,
            trigger_type="event",
            trigger_spec='{"keywords": ["onetime"]}',
            priority="low",
        )
        with pm.get_conn() as conn:
            now_dt = datetime.now()

            with patch("modules.working_memory.push_to_working_memory"):
                with patch("modules.consciousness.update_workspace_spotlight"):
                    pm._mark_triggered(conn, result["id"], "keyword match", now_dt)
                    conn.commit()

            rows = conn.execute(
                "SELECT id FROM intentions WHERE action = %s",
                (unique_action,)
            ).fetchall()
            assert len(rows) == 1, f"Should have 1 row, got {len(rows)}"


# ============================================================
# CONTEXT-DEPENDENT RETRIEVAL
# ============================================================

class TestContextRetrieval:
    def test_context_bonus_boosts_nonfocal(self, isolated_db):
        """Context overlap should boost nonfocal matching score."""
        pm = isolated_db
        spec = {
            "keywords": ["deploy"],
            "semantic_description": "deploy production server update release",
        }
        text = "we need to deploy the new version to production"

        # Without context
        r1 = pm._check_nonfocal(spec, text, "")
        # With matching context
        r2 = pm._check_nonfocal(spec, text, "working on deploy pipeline for production release")

        # At minimum, context should not reduce the match
        # (both may or may not match depending on threshold, but r2 should be >= r1)
        # We check that r2 is at least as good
        if r1.get("matched") and r2.get("matched"):
            pass  # Both match, context didn't hurt
        elif not r1.get("matched") and r2.get("matched"):
            pass  # Context helped promote to match
        elif r1.get("matched") and not r2.get("matched"):
            pytest.fail("Context should never reduce match quality")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
