#!/usr/bin/env python3
"""
Sprint 15: Goal System Tests (ICARUS Activation-Based)
=======================================================
Tests for the goal system based on:
  - Altmann & Trafton 2002 (activation model)
  - Cox et al 2017 (6 goal operations)
  - Pink et al 2025 (episodic context)

Run: ./venv/bin/pytest tests/test_goals.py -v
"""

import sys
import os
import json
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from modules import goals


@pytest.fixture(autouse=True)
def _reset_goals_conn(monkeypatch):
    """Reset goals module connection so it uses the isolated DB."""
    from modules.config import PROSPECTIVE_DB_PATH
    monkeypatch.setattr(goals, "_conn", None)
    monkeypatch.setattr("modules.goals.PROSPECTIVE_DB_PATH", PROSPECTIVE_DB_PATH)
    yield
    # Reset again so next test gets fresh conn
    monkeypatch.setattr(goals, "_conn", None)


# ============================================================
# 1. FORMULATE (Cox 2017)
# ============================================================

class TestCreateGoal:
    """Cox 2017 Formulate operation."""

    def test_create_task(self):
        result = goals.create_goal("Fix login bug", level="task", priority="high")
        assert "id" in result
        assert result["title"] == "Fix login bug"
        assert result["level"] == "task"
        assert result["priority"] == "high"
        assert result["status"] == "active"

    def test_create_project_hierarchy(self):
        proj = goals.create_goal("Main Project", level="project", priority="critical")
        phase = goals.create_goal("Phase 1", level="phase", parent_id=proj["id"])
        sprint = goals.create_goal("Sprint 1", level="sprint", parent_id=phase["id"])
        task = goals.create_goal("Task 1", level="task", parent_id=sprint["id"])

        assert phase["parent_id"] == proj["id"]
        assert sprint["parent_id"] == phase["id"]
        assert task["parent_id"] == sprint["id"]

    def test_invalid_level_rejected(self):
        result = goals.create_goal("Bad goal", level="invalid")
        assert "error" in result

    def test_invalid_priority_rejected(self):
        result = goals.create_goal("Bad goal", level="task", priority="ultra")
        assert "error" in result

    def test_parent_not_found_rejected(self):
        result = goals.create_goal("Orphan", level="task", parent_id="nonexistent")
        assert "error" in result

    def test_context_stored(self):
        """Pink 2025: episodic context should be preserved."""
        result = goals.create_goal(
            "Study papers",
            level="task",
            context="Researching goal architectures for codi-memory",
        )
        assert "id" in result
        # Verify context is stored in DB
        conn = goals._get_conn()
        row = conn.execute(
            "SELECT context FROM goals WHERE id = ?", (result["id"],)
        ).fetchone()
        assert "goal architectures" in row[0]


# ============================================================
# 2. SELECT (Cox 2017)
# ============================================================

class TestGetActiveGoals:
    """Cox 2017 Select operation with ACT-R activation ranking."""

    def test_returns_active_goals(self):
        goals.create_goal("Goal A", level="task")
        goals.create_goal("Goal B", level="task")
        active = goals.get_active_goals()
        assert len(active) == 2
        assert all(g["status"] == "active" for g in active)

    def test_ranked_by_activation(self):
        goals.create_goal("Low", level="task", priority="low")
        goals.create_goal("Critical", level="task", priority="critical")
        active = goals.get_active_goals()
        # Critical should have higher activation
        assert active[0]["priority"] == "critical"
        assert active[0]["activation"] >= active[1]["activation"]

    def test_filter_by_level(self):
        goals.create_goal("Project", level="project")
        goals.create_goal("Task", level="task")
        projects = goals.get_active_goals(level="project")
        assert len(projects) == 1
        assert projects[0]["level"] == "project"

    def test_filter_by_status(self):
        g = goals.create_goal("Done", level="task")
        goals.complete_goal(g["id"])
        active = goals.get_active_goals(status="active")
        completed = goals.get_active_goals(status="completed")
        assert all(g["title"] != "Done" for g in active)
        assert len(completed) == 1


# ============================================================
# 3. CHANGE (Cox 2017)
# ============================================================

class TestUpdateGoal:
    """Cox 2017 Change operation."""

    def test_update_status(self):
        g = goals.create_goal("Pausable", level="task")
        result = goals.update_goal(g["id"], status="paused")
        assert "status→paused" in result["changes"]

    def test_update_priority(self):
        g = goals.create_goal("Reprioritize", level="task", priority="low")
        result = goals.update_goal(g["id"], priority="critical")
        assert "priority→critical" in result["changes"]

    def test_update_nonexistent_fails(self):
        result = goals.update_goal("nonexistent", status="paused")
        assert "error" in result

    def test_invalid_status_rejected(self):
        g = goals.create_goal("Test", level="task")
        result = goals.update_goal(g["id"], status="invalid")
        assert "error" in result

    def test_update_touches_access(self):
        g = goals.create_goal("Track access", level="task")
        conn = goals._get_conn()
        before = conn.execute(
            "SELECT access_count FROM goals WHERE id = ?", (g["id"],)
        ).fetchone()[0]
        goals.update_goal(g["id"], priority="high")
        after = conn.execute(
            "SELECT access_count FROM goals WHERE id = ?", (g["id"],)
        ).fetchone()[0]
        assert after == before + 1


# ============================================================
# 4. DELEGATE (Cox 2017)
# ============================================================

class TestAssignGoal:
    """Cox 2017 Delegate operation."""

    def test_assign_to_agent(self):
        g = goals.create_goal("Research task", level="task")
        result = goals.assign_goal(g["id"], "deep-research-agent")
        assert "assigned_to→deep-research-agent" in result["changes"]


# ============================================================
# 5. ACHIEVE (Cox 2017)
# ============================================================

class TestCompleteGoal:
    """Cox 2017 Achieve operation with cascade check."""

    def test_complete_goal(self):
        g = goals.create_goal("Finish this", level="task")
        result = goals.complete_goal(g["id"], outcome="Done!")
        assert result["status"] == "completed"
        assert result["outcome"] == "Done!"

    def test_cascade_suggestion(self):
        """When all children complete, suggest completing parent."""
        parent = goals.create_goal("Sprint", level="sprint")
        t1 = goals.create_goal("Task 1", level="task", parent_id=parent["id"])
        t2 = goals.create_goal("Task 2", level="task", parent_id=parent["id"])

        goals.complete_goal(t1["id"])
        result = goals.complete_goal(t2["id"])

        assert "cascade_suggestion" in result
        assert result["cascade_suggestion"]["parent_id"] == parent["id"]

    def test_no_cascade_when_siblings_incomplete(self):
        parent = goals.create_goal("Sprint", level="sprint")
        t1 = goals.create_goal("Task 1", level="task", parent_id=parent["id"])
        t2 = goals.create_goal("Task 2", level="task", parent_id=parent["id"])

        result = goals.complete_goal(t1["id"])
        assert "cascade_suggestion" not in result

    def test_complete_nonexistent_fails(self):
        result = goals.complete_goal("nonexistent")
        assert "error" in result


# ============================================================
# 6. MONITOR (Cox 2017)
# ============================================================

class TestGoalHygiene:
    """Cox 2017 Monitor operation — staleness detection."""

    def test_stale_task_gets_paused(self):
        """Tasks not accessed in 3+ days should be auto-paused."""
        g = goals.create_goal("Old task", level="task")
        conn = goals._get_conn()
        # Fake old access time
        conn.execute(
            "UPDATE goals SET last_accessed = '2026-01-01T00:00:00' WHERE id = ?",
            (g["id"],),
        )
        conn.commit()

        result = goals.check_goal_hygiene()
        assert result["paused_count"] >= 1
        paused_ids = [p["id"] for p in result["paused"]]
        assert g["id"] in paused_ids

    def test_fresh_task_not_paused(self):
        """Recently accessed tasks should NOT be paused."""
        goals.create_goal("Fresh task", level="task")
        result = goals.check_goal_hygiene()
        assert result["paused_count"] == 0

    def test_hygiene_respects_level_thresholds(self):
        """Projects have longer staleness window than tasks."""
        task = goals.create_goal("Old task", level="task")
        project = goals.create_goal("Old project", level="project")
        conn = goals._get_conn()
        # Set both to 5 days ago (stale for task but not project)
        old_time = "2026-02-25T00:00:00"
        conn.execute(
            "UPDATE goals SET last_accessed = ? WHERE id IN (?, ?)",
            (old_time, task["id"], project["id"]),
        )
        conn.commit()

        result = goals.check_goal_hygiene()
        paused_ids = [p["id"] for p in result["paused"]]
        assert task["id"] in paused_ids, "Task should be paused (>3d)"
        assert project["id"] not in paused_ids, "Project should NOT be paused (<90d)"


# ============================================================
# 7. ACTIVATION (Altmann & Trafton 2002)
# ============================================================

class TestActivation:
    """Verify ACT-R activation dynamics for goals."""

    def test_touch_increases_activation(self):
        g = goals.create_goal("Touchable", level="task", priority="medium")
        before = goals.get_active_goals()
        act_before = [x for x in before if x["id"] == g["id"]][0]["activation"]

        # Touch multiple times
        for _ in range(5):
            goals.touch_goal(g["id"])

        after = goals.get_active_goals()
        act_after = [x for x in after if x["id"] == g["id"]][0]["activation"]

        assert act_after > act_before, (
            f"Activation should increase after touches: {act_before} -> {act_after}"
        )

    def test_critical_higher_than_low(self):
        goals.create_goal("Critical", level="task", priority="critical")
        goals.create_goal("Low", level="task", priority="low")
        active = goals.get_active_goals()
        critical = [g for g in active if g["priority"] == "critical"][0]
        low = [g for g in active if g["priority"] == "low"][0]
        assert critical["activation"] > low["activation"]


# ============================================================
# 8. INTERFERENCE LEVEL (Altmann & Trafton 2002)
# ============================================================

class TestInterferenceLevel:
    """Interference level = AVG(activation) of active goals."""

    def test_interference_level_computed(self):
        goals.create_goal("A", level="task")
        goals.create_goal("B", level="task")
        ctx = goals.get_context_goals()
        assert "interference_level" in ctx
        assert ctx["interference_level"] > 0

    def test_only_above_threshold_returned(self):
        """Only goals with activation > interference should be in context."""
        # Create several goals with same priority
        for i in range(5):
            goals.create_goal(f"Goal {i}", level="task", priority="medium")

        # Touch one heavily to push it above interference
        active = goals.get_active_goals()
        target_id = active[0]["id"]
        for _ in range(10):
            goals.touch_goal(target_id)

        ctx = goals.get_context_goals()
        assert ctx["above_threshold"] <= ctx["total_active"]


# ============================================================
# 9. GOAL TREE
# ============================================================

class TestGoalTree:
    """Hierarchical goal view."""

    def test_tree_structure(self):
        proj = goals.create_goal("Proj", level="project")
        goals.create_goal("Sprint A", level="sprint", parent_id=proj["id"])
        goals.create_goal("Sprint B", level="sprint", parent_id=proj["id"])

        tree = goals.get_goal_tree()
        assert len(tree) == 1
        assert tree[0]["title"] == "Proj"
        assert len(tree[0]["children"]) == 2

    def test_empty_tree(self):
        tree = goals.get_goal_tree()
        assert tree == []


# ============================================================
# 10. AUDIT LOG
# ============================================================

class TestGoalLog:
    """Goal log audit trail."""

    def test_create_logs_event(self):
        g = goals.create_goal("Logged", level="task")
        conn = goals._get_conn()
        logs = conn.execute(
            "SELECT event, detail FROM goal_log WHERE goal_id = ?",
            (g["id"],),
        ).fetchall()
        assert len(logs) >= 1
        assert logs[0][0] == "created"

    def test_complete_logs_event(self):
        g = goals.create_goal("To complete", level="task")
        goals.complete_goal(g["id"], outcome="Done")
        conn = goals._get_conn()
        logs = conn.execute(
            "SELECT event FROM goal_log WHERE goal_id = ? ORDER BY id",
            (g["id"],),
        ).fetchall()
        events = [l[0] for l in logs]
        assert "created" in events
        assert "completed" in events

    def test_hygiene_logs_auto_pause(self):
        g = goals.create_goal("Will be stale", level="task")
        conn = goals._get_conn()
        conn.execute(
            "UPDATE goals SET last_accessed = '2026-01-01T00:00:00' WHERE id = ?",
            (g["id"],),
        )
        conn.commit()
        goals.check_goal_hygiene()
        logs = conn.execute(
            "SELECT event FROM goal_log WHERE goal_id = ? ORDER BY id",
            (g["id"],),
        ).fetchall()
        events = [l[0] for l in logs]
        assert "auto_paused" in events
