#!/usr/bin/env python3
"""
Sprint 15/15.5: Goal System Tests (ICARUS Activation-Based)
============================================================
Tests for the goal system based on:
  - Altmann & Trafton 2002 (activation model)
  - Cox et al 2017 (6 goal operations)
  - Pink et al 2025 (episodic context)
  - Sprint 15.5: Structured context (ACT-R/SOAR/Duncan)

Run: ./venv/bin/pytest tests/test_goals.py -v

Updated: 2026-03-12 (Migrated to PostgreSQL -- tests use production PG via config_pg)
"""

import sys
import os
import json
import time
import uuid

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from modules import goals
from modules.config_pg import get_conn
from psycopg.rows import dict_row

# A valid UUID that will never exist in the DB
NONEXISTENT_UUID = "00000000-0000-0000-0000-000000000000"

# Unique prefix for test goals to isolate from production data
_TEST_PREFIX = f"_TEST_{uuid.uuid4().hex[:8]}_"

# Number of touches to boost test goals above production goal activation
_BOOST_TOUCHES = 50


def _tname(name):
    """Generate a unique test goal title to avoid collisions with production data."""
    return f"{_TEST_PREFIX}{name}"


def _boost_goal(goal_id, touches=_BOOST_TOUCHES):
    """Boost a goal's activation above production goals via direct SQL.

    Uses a single UPDATE instead of N touch_goal() calls to avoid
    exhausting the PG connection pool during tests.
    """
    with get_conn() as conn:
        with conn.transaction():
            conn.execute(
                "UPDATE goals SET access_count = access_count + %s, "
                "last_accessed = NOW() WHERE id = %s",
                (touches, goal_id),
            )


@pytest.fixture(autouse=True)
def _cleanup_test_goals():
    """Clean up test-created goals after each test.

    Uses the _TEST_PREFIX pattern to identify and delete test goals.
    Runs cleanup AFTER the test (yield fixture).
    """
    yield
    # Delete all test goals created during this test
    try:
        with get_conn() as conn:
            with conn.transaction():
                # Delete logs first (FK constraint on goal_log.goal_id)
                conn.execute(
                    "DELETE FROM goal_log WHERE goal_id IN "
                    "(SELECT id FROM goals WHERE title LIKE %s)",
                    (f"{_TEST_PREFIX}%",),
                )
                # Nullify parent references within test goals (self-referential FK)
                conn.execute(
                    "UPDATE goals SET parent_id = NULL WHERE title LIKE %s",
                    (f"{_TEST_PREFIX}%",),
                )
                # Now delete the goals themselves
                conn.execute(
                    "DELETE FROM goals WHERE title LIKE %s",
                    (f"{_TEST_PREFIX}%",),
                )
    except Exception as e:
        import warnings
        warnings.warn(f"Goal cleanup failed: {e}")


def _pg_query(sql, params=()):
    """Execute a PG query and return rows as dicts."""
    with get_conn() as conn:
        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute(sql, params)
            return cur.fetchall()


def _pg_execute(sql, params=()):
    """Execute a PG statement (no return)."""
    with get_conn() as conn:
        with conn.transaction():
            conn.execute(sql, params)


# ============================================================
# 1. FORMULATE (Cox 2017)
# ============================================================

class TestCreateGoal:
    """Cox 2017 Formulate operation."""

    def test_create_task(self):
        result = goals.create_goal(_tname("Fix login bug"), level="task", priority="high")
        assert "id" in result
        assert result["title"] == _tname("Fix login bug")
        assert result["level"] == "task"
        assert result["priority"] == "high"
        assert result["status"] == "active"

    def test_create_project_hierarchy(self):
        proj = goals.create_goal(_tname("Main Project"), level="project", priority="critical")
        phase = goals.create_goal(_tname("Phase 1"), level="phase", parent_id=proj["id"])
        sprint = goals.create_goal(_tname("Sprint 1"), level="sprint", parent_id=phase["id"])
        task = goals.create_goal(_tname("Task 1"), level="task", parent_id=sprint["id"])

        assert phase["parent_id"] == proj["id"]
        assert sprint["parent_id"] == phase["id"]
        assert task["parent_id"] == sprint["id"]

    def test_invalid_level_rejected(self):
        result = goals.create_goal(_tname("Bad goal"), level="invalid")
        assert "error" in result

    def test_invalid_priority_rejected(self):
        result = goals.create_goal(_tname("Bad goal"), level="task", priority="ultra")
        assert "error" in result

    def test_parent_not_found_rejected(self):
        result = goals.create_goal(
            _tname("Orphan"), level="task", parent_id=NONEXISTENT_UUID
        )
        assert "error" in result

    def test_context_stored(self):
        """Pink 2025: episodic context should be preserved."""
        result = goals.create_goal(
            _tname("Study papers"),
            level="task",
            context="Researching goal architectures for codi-memory",
        )
        assert "id" in result
        rows = _pg_query(
            "SELECT context FROM goals WHERE id = %s", (result["id"],)
        )
        assert len(rows) == 1
        assert "goal architectures" in rows[0]["context"]

    def test_structured_context_stored(self):
        """Sprint 15.5: structured fields persist."""
        result = goals.create_goal(
            _tname("Consciencia"),
            level="project",
            goal_what="Sistema cognitivo con 5 loops de integracion",
            goal_why="Hacer que codi-memory sea un sistema cognitivo completo",
            goal_next_step="Implementar auto-context para goals",
        )
        assert "id" in result
        rows = _pg_query(
            "SELECT goal_what, goal_why, goal_next_step, context_updated_at "
            "FROM goals WHERE id = %s", (result["id"],)
        )
        assert len(rows) == 1
        row = rows[0]
        assert "5 loops" in row["goal_what"]
        assert "cognitivo completo" in row["goal_why"]
        assert "auto-context" in row["goal_next_step"]
        assert row["context_updated_at"] is not None

    def test_missing_what_why_warns(self):
        """Strong elicitation: warn when goal_what/why missing."""
        result = goals.create_goal(_tname("No context"), level="task")
        assert "warnings" in result
        assert len(result["warnings"]) == 2
        assert any("goal_what" in w for w in result["warnings"])
        assert any("goal_why" in w for w in result["warnings"])

    def test_partial_what_only_warns_why(self):
        """Only missing field gets warning."""
        result = goals.create_goal(
            _tname("Partial"), level="task",
            goal_what="Something concrete",
        )
        assert "warnings" in result
        assert len(result["warnings"]) == 1
        assert "goal_why" in result["warnings"][0]

    def test_full_context_no_warnings(self):
        """Complete structured context = no warnings."""
        result = goals.create_goal(
            _tname("Complete"), level="task",
            goal_what="What it is",
            goal_why="Why it matters",
        )
        assert "warnings" not in result


# ============================================================
# 2. SELECT (Cox 2017)
# ============================================================

class TestGetActiveGoals:
    """Cox 2017 Select operation with ACT-R activation ranking."""

    def test_returns_active_goals(self):
        g1 = goals.create_goal(_tname("Goal A"), level="task")
        g2 = goals.create_goal(_tname("Goal B"), level="task")
        _boost_goal(g1["id"])
        _boost_goal(g2["id"])
        active = goals.get_active_goals(limit=100)
        test_active = [g for g in active if g["title"].startswith(_TEST_PREFIX)]
        assert len(test_active) == 2
        assert all(g["status"] == "active" for g in test_active)

    def test_ranked_by_activation(self):
        g_low = goals.create_goal(_tname("Low"), level="task", priority="low")
        g_crit = goals.create_goal(_tname("Critical"), level="task", priority="critical")
        _boost_goal(g_low["id"])
        _boost_goal(g_crit["id"])
        active = goals.get_active_goals(limit=100)
        test_goals_list = [g for g in active if g["title"].startswith(_TEST_PREFIX)]
        # Critical should come first (higher activation due to priority boost)
        assert test_goals_list[0]["priority"] == "critical"
        assert test_goals_list[0]["activation"] >= test_goals_list[1]["activation"]

    def test_filter_by_level(self):
        g_proj = goals.create_goal(_tname("Project"), level="project")
        g_task = goals.create_goal(_tname("Task"), level="task")
        _boost_goal(g_proj["id"])
        _boost_goal(g_task["id"])
        projects = goals.get_active_goals(level="project", limit=100)
        test_projects = [g for g in projects if g["title"].startswith(_TEST_PREFIX)]
        assert len(test_projects) == 1
        assert test_projects[0]["level"] == "project"

    def test_filter_by_status(self):
        g = goals.create_goal(_tname("Done"), level="task")
        _boost_goal(g["id"])
        goals.complete_goal(g["id"])
        active = goals.get_active_goals(status="active", limit=100)
        completed = goals.get_active_goals(status="completed", limit=100)
        assert all(
            gx["title"] != _tname("Done")
            for gx in active
            if gx["title"].startswith(_TEST_PREFIX)
        )
        test_completed = [
            gx for gx in completed if gx["title"].startswith(_TEST_PREFIX)
        ]
        assert len(test_completed) == 1


# ============================================================
# 3. CHANGE (Cox 2017)
# ============================================================

class TestUpdateGoal:
    """Cox 2017 Change operation."""

    def test_update_status(self):
        g = goals.create_goal(_tname("Pausable"), level="task")
        result = goals.update_goal(g["id"], status="paused")
        assert "status\u2192paused" in result["changes"]

    def test_update_priority(self):
        g = goals.create_goal(_tname("Change priority"), level="task", priority="low")
        result = goals.update_goal(g["id"], priority="critical")
        assert "priority\u2192critical" in result["changes"]

    def test_update_nonexistent_fails(self):
        result = goals.update_goal(NONEXISTENT_UUID, status="paused")
        assert "error" in result

    def test_invalid_status_rejected(self):
        g = goals.create_goal(_tname("Status test"), level="task")
        result = goals.update_goal(g["id"], status="invalid")
        assert "error" in result

    def test_update_touches_access(self):
        g = goals.create_goal(_tname("Track access"), level="task")
        rows_before = _pg_query(
            "SELECT access_count FROM goals WHERE id = %s", (g["id"],)
        )
        before = rows_before[0]["access_count"]
        goals.update_goal(g["id"], priority="high")
        rows_after = _pg_query(
            "SELECT access_count FROM goals WHERE id = %s", (g["id"],)
        )
        after = rows_after[0]["access_count"]
        assert after == before + 1


# ============================================================
# 4. DELEGATE (Cox 2017)
# ============================================================

class TestAssignGoal:
    """Cox 2017 Delegate operation."""

    def test_assign_to_agent(self):
        g = goals.create_goal(_tname("Research task"), level="task")
        result = goals.assign_goal(g["id"], "deep-research-agent")
        assert any("deep-research-agent" in c for c in result["changes"])


# ============================================================
# 5. ACHIEVE (Cox 2017)
# ============================================================

class TestCompleteGoal:
    """Cox 2017 Achieve operation with cascade check."""

    def test_complete_goal(self):
        g = goals.create_goal(_tname("Finish this"), level="task")
        result = goals.complete_goal(g["id"], outcome="Done!")
        assert result["status"] == "completed"
        assert result["outcome"] == "Done!"

    def test_cascade_suggestion(self):
        """When all children complete, suggest completing parent."""
        parent = goals.create_goal(_tname("Sprint"), level="sprint")
        t1 = goals.create_goal(_tname("Task 1"), level="task", parent_id=parent["id"])
        t2 = goals.create_goal(_tname("Task 2"), level="task", parent_id=parent["id"])

        goals.complete_goal(t1["id"])
        result = goals.complete_goal(t2["id"])

        assert "cascade_suggestion" in result
        assert str(result["cascade_suggestion"]["parent_id"]) == parent["id"]

    def test_no_cascade_when_siblings_incomplete(self):
        parent = goals.create_goal(_tname("Sprint"), level="sprint")
        t1 = goals.create_goal(_tname("Task 1"), level="task", parent_id=parent["id"])
        goals.create_goal(_tname("Task 2"), level="task", parent_id=parent["id"])

        result = goals.complete_goal(t1["id"])
        assert "cascade_suggestion" not in result

    def test_complete_nonexistent_fails(self):
        result = goals.complete_goal(NONEXISTENT_UUID)
        assert "error" in result


# ============================================================
# 6. MONITOR (Cox 2017)
# ============================================================

class TestGoalHygiene:
    """Cox 2017 Monitor operation -- staleness detection."""

    def test_stale_task_gets_paused(self):
        """Tasks not accessed in 3+ days should be auto-paused."""
        g = goals.create_goal(_tname("Old task"), level="task")
        _pg_execute(
            "UPDATE goals SET last_accessed = '2026-01-01T00:00:00+00' WHERE id = %s",
            (g["id"],),
        )

        result = goals.check_goal_hygiene()
        paused_ids = [p["id"] for p in result["paused"]]
        assert g["id"] in paused_ids

    def test_fresh_task_not_paused(self):
        """Recently accessed tasks should NOT be paused."""
        g = goals.create_goal(_tname("Fresh task"), level="task")
        result = goals.check_goal_hygiene()
        paused_ids = [p["id"] for p in result["paused"]]
        assert g["id"] not in paused_ids

    def test_hygiene_respects_level_thresholds(self):
        """Projects have longer staleness window than tasks."""
        task = goals.create_goal(_tname("Old task"), level="task")
        project = goals.create_goal(_tname("Old project"), level="project")
        # Set both to 5 days ago (stale for task but not project)
        old_time = "2026-02-25T00:00:00+00"
        _pg_execute(
            "UPDATE goals SET last_accessed = %s WHERE id IN (%s, %s)",
            (old_time, task["id"], project["id"]),
        )

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
        g = goals.create_goal(_tname("Touchable"), level="task", priority="medium")
        # Boost to make visible in get_active_goals
        _boost_goal(g["id"])
        active = goals.get_active_goals(limit=100)
        act_before = [x for x in active if x["id"] == g["id"]][0]["activation"]

        # Touch more times to increase further
        for _ in range(20):
            goals.touch_goal(g["id"])

        active = goals.get_active_goals(limit=100)
        act_after = [x for x in active if x["id"] == g["id"]][0]["activation"]

        assert act_after > act_before, (
            f"Activation should increase after touches: {act_before} -> {act_after}"
        )

    def test_critical_higher_than_low(self):
        g_crit = goals.create_goal(_tname("Critical"), level="task", priority="critical")
        g_low = goals.create_goal(_tname("Low"), level="task", priority="low")
        # Boost both equally to make them visible
        _boost_goal(g_crit["id"])
        _boost_goal(g_low["id"])
        active = goals.get_active_goals(limit=100)
        critical = [g for g in active if g["id"] == g_crit["id"]][0]
        low = [g for g in active if g["id"] == g_low["id"]][0]
        assert critical["activation"] > low["activation"]


# ============================================================
# 8. INTERFERENCE LEVEL (Altmann & Trafton 2002)
# ============================================================

class TestInterferenceLevel:
    """Interference level = AVG(activation) of active goals."""

    def test_interference_level_computed(self):
        goals.create_goal(_tname("A"), level="task")
        goals.create_goal(_tname("B"), level="task")
        ctx = goals.get_context_goals()
        assert "interference_level" in ctx
        assert ctx["interference_level"] > 0

    def test_only_above_threshold_returned(self):
        """Only goals with activation > interference should be in context."""
        for i in range(5):
            goals.create_goal(_tname(f"Goal {i}"), level="task", priority="medium")

        # Touch one heavily to push it above interference
        active = goals.get_active_goals(limit=100)
        test_active = [g for g in active if g["title"].startswith(_TEST_PREFIX)]
        if test_active:
            target_id = test_active[0]["id"]
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
        proj = goals.create_goal(_tname("Proj"), level="project")
        goals.create_goal(_tname("Sprint A"), level="sprint", parent_id=proj["id"])
        goals.create_goal(_tname("Sprint B"), level="sprint", parent_id=proj["id"])

        # Use root_id to avoid traversing ALL goals in the production DB
        tree = goals.get_goal_tree(root_id=proj["id"])
        assert len(tree) == 1
        assert tree[0]["title"] == _tname("Proj")
        assert len(tree[0]["children"]) == 2

    def test_empty_tree(self):
        # Use a nonexistent root_id to get empty result without scanning all goals
        tree = goals.get_goal_tree(root_id=NONEXISTENT_UUID)
        assert tree == []


# ============================================================
# 10. AUDIT LOG
# ============================================================

class TestGoalLog:
    """Goal log audit trail."""

    def test_create_logs_event(self):
        g = goals.create_goal(_tname("Logged"), level="task")
        rows = _pg_query(
            "SELECT event, detail FROM goal_log WHERE goal_id = %s",
            (g["id"],),
        )
        assert len(rows) >= 1
        assert rows[0]["event"] == "created"

    def test_complete_logs_event(self):
        g = goals.create_goal(_tname("To complete"), level="task")
        goals.complete_goal(g["id"], outcome="Done")
        rows = _pg_query(
            "SELECT event FROM goal_log WHERE goal_id = %s ORDER BY id",
            (g["id"],),
        )
        events = [r["event"] for r in rows]
        assert "created" in events
        assert "completed" in events

    def test_hygiene_logs_auto_pause(self):
        g = goals.create_goal(_tname("Will be stale"), level="task")
        _pg_execute(
            "UPDATE goals SET last_accessed = '2026-01-01T00:00:00+00' WHERE id = %s",
            (g["id"],),
        )
        goals.check_goal_hygiene()
        rows = _pg_query(
            "SELECT event FROM goal_log WHERE goal_id = %s ORDER BY id",
            (g["id"],),
        )
        events = [r["event"] for r in rows]
        assert "auto_paused" in events


# ============================================================
# 11. STRUCTURED CONTEXT (Sprint 15.5)
# ============================================================

class TestStructuredContext:
    """Sprint 15.5: Structured context fields (ACT-R/SOAR/Duncan)."""

    def test_get_active_goals_includes_structured_fields(self):
        """get_active_goals returns structured context fields."""
        g = goals.create_goal(
            _tname("With context"), level="task",
            goal_what="Test what field",
            goal_why="Test why field",
            goal_next_step="Test next step",
        )
        _boost_goal(g["id"])
        active = goals.get_active_goals(limit=100)
        matched = [x for x in active if x["id"] == g["id"]]
        assert len(matched) == 1
        gd = matched[0]
        assert gd["goal_what"] == "Test what field"
        assert gd["goal_why"] == "Test why field"
        assert gd["goal_next_step"] == "Test next step"
        assert gd["context_updated_at"] is not None

    def test_touch_updates_derivable_fields(self):
        """touch_goal() updates only derivable (I-support) fields."""
        g = goals.create_goal(
            _tname("Touchable"), level="task",
            goal_what="Original what",
            goal_why="Original why",
        )
        goals.touch_goal(
            g["id"],
            last_state="Sprint 15 done",
            next_step="Start Sprint 16",
        )
        rows = _pg_query(
            "SELECT goal_what, goal_why, goal_last_state, goal_next_step "
            "FROM goals WHERE id = %s", (g["id"],)
        )
        row = rows[0]
        # Committed fields unchanged
        assert row["goal_what"] == "Original what"
        assert row["goal_why"] == "Original why"
        # Derivable fields updated
        assert row["goal_last_state"] == "Sprint 15 done"
        assert row["goal_next_step"] == "Start Sprint 16"

    def test_touch_refreshes_context_timestamp(self):
        """Touching with derivable data refreshes context_updated_at."""
        g = goals.create_goal(_tname("Timed"), level="task", goal_what="X")
        _pg_execute(
            "UPDATE goals SET context_updated_at = '2026-01-01T00:00:00+00' WHERE id = %s",
            (g["id"],),
        )
        goals.touch_goal(g["id"], next_step="Fresh step")
        rows = _pg_query(
            "SELECT context_updated_at FROM goals WHERE id = %s", (g["id"],)
        )
        ts = rows[0]["context_updated_at"]
        ts_str = ts.isoformat() if hasattr(ts, "isoformat") else str(ts)
        assert "2026-03" in ts_str  # Should be recent

    def test_touch_without_context_no_timestamp_refresh(self):
        """Touch without derivable fields does NOT refresh context_updated_at."""
        g = goals.create_goal(_tname("Plain touch"), level="task", goal_what="X")
        _pg_execute(
            "UPDATE goals SET context_updated_at = '2026-01-15T00:00:00+00' WHERE id = %s",
            (g["id"],),
        )
        goals.touch_goal(g["id"])  # No context update
        rows = _pg_query(
            "SELECT context_updated_at FROM goals WHERE id = %s", (g["id"],)
        )
        ts = rows[0]["context_updated_at"]
        ts_str = ts.isoformat() if hasattr(ts, "isoformat") else str(ts)
        assert "2026-01-15" in ts_str  # Unchanged

    def test_update_goal_derivable_fields(self):
        """update_goal() supports goal_last_state and goal_next_step."""
        g = goals.create_goal(_tname("Updatable"), level="task", goal_what="X")
        result = goals.update_goal(
            g["id"],
            goal_last_state="Tests passing",
            goal_next_step="Deploy to prod",
        )
        assert "last_state updated" in result["changes"]
        assert "next_step updated" in result["changes"]
        rows = _pg_query(
            "SELECT goal_last_state, goal_next_step, context_updated_at "
            "FROM goals WHERE id = %s", (g["id"],)
        )
        row = rows[0]
        assert row["goal_last_state"] == "Tests passing"
        assert row["goal_next_step"] == "Deploy to prod"
        assert row["context_updated_at"] is not None


# ============================================================
# 12. CONTEXT STALENESS (Duncan 2013 + Sprint 15.5)
# ============================================================

class TestContextStaleness:
    """Staleness detection: warn when context is outdated."""

    def test_stale_context_warning(self):
        """Goals with context_updated_at > 7d should get stale warning.

        We boost the stale goal significantly more than the filler so it
        appears above the interference level (mean activation of all active).
        """
        g = goals.create_goal(
            _tname("Stale goal"), level="task", priority="critical",
            goal_what="Something important",
        )
        filler = goals.create_goal(_tname("Filler"), level="task", priority="low")
        # Boost the stale goal A LOT to ensure it is above interference
        _boost_goal(g["id"], touches=100)
        # Filler gets fewer touches so it stays below
        _boost_goal(filler["id"], touches=10)
        # Set stale timestamp AFTER boosting
        _pg_execute(
            "UPDATE goals SET context_updated_at = '2026-01-01T00:00:00+00' WHERE id = %s",
            (g["id"],),
        )
        ctx = goals.get_context_goals(limit=100)
        # The stale goal should be above interference and generate a warning
        goal_in_ctx = any(x["id"] == g["id"] for x in ctx.get("goals", []))
        assert goal_in_ctx, (
            f"Stale goal should be above interference "
            f"(interference={ctx.get('interference_level')}, "
            f"above={ctx.get('above_threshold')}, total={ctx.get('total_active')})"
        )
        assert len(ctx.get("stale_warnings", [])) >= 1
        assert any("stale" in w for w in ctx["stale_warnings"])

    def test_no_context_warning(self):
        """Goals with NO context should get 'NO context set' warning."""
        g = goals.create_goal(_tname("Empty context"), level="task", priority="critical")
        filler = goals.create_goal(_tname("Filler low"), level="task", priority="low")
        # Boost the empty-context goal heavily to put it above interference
        _boost_goal(g["id"], touches=100)
        _boost_goal(filler["id"], touches=10)
        ctx = goals.get_context_goals(limit=100)
        goal_in_ctx = any(x["id"] == g["id"] for x in ctx.get("goals", []))
        assert goal_in_ctx, "Empty-context goal should be above interference"
        assert len(ctx.get("stale_warnings", [])) >= 1
        assert any("NO context set" in w for w in ctx["stale_warnings"])

    def test_fresh_context_no_warning(self):
        """Goals with fresh context should not get warnings."""
        g = goals.create_goal(
            _tname("Fresh"), level="task", priority="critical",
            goal_what="Fresh what",
            goal_why="Fresh why",
            goal_next_step="Next action",
        )
        goals.create_goal(_tname("Filler"), level="task", priority="low")
        ctx = goals.get_context_goals(limit=100)
        stale = [w for w in ctx.get("stale_warnings", []) if _tname("Fresh") in w]
        assert len(stale) == 0

    def test_context_goals_includes_structured_data(self):
        """get_context_goals returns structured fields for cascading priming."""
        g = goals.create_goal(
            _tname("Context goal"), level="task", priority="critical",
            goal_what="The what",
            goal_next_step="The next step",
        )
        goals.create_goal(_tname("Filler"), level="task", priority="low")
        # Boost to ensure it appears above interference level
        _boost_goal(g["id"])
        ctx = goals.get_context_goals(limit=100)
        matched = [x for x in ctx["goals"] if x["id"] == g["id"]]
        assert len(matched) == 1
        gd = matched[0]
        assert gd["goal_what"] == "The what"
        assert gd["goal_next_step"] == "The next step"
