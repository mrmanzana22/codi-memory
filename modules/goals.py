"""
GOAL SYSTEM MODULE - Sprint 15/15.5 (ICARUS Activation-Based)
==============================================================
Activation-based priority agenda for goal management.
NOT a stack -- goals compete by ACT-R activation (Altmann & Trafton 2002).

6 Operations (Cox, Dannenhauer & Kondrakunta 2017):
  1. Formulate: create_goal()
  2. Select: get_active_goals() -- ranked by activation
  3. Change: update_goal()
  4. Delegate: assign_goal()
  5. Achieve: complete_goal() + cascade check
  6. Monitor: check_goal_hygiene() -- staleness detection

Hierarchy: project > phase > sprint > task
Hygiene: tasks 3d, sprints 2w, phases 60d, projects 90d

Sprint 15.5 - Structured Context (research-backed):
  4 fields replace monolithic context blob:
    - goal_what (committed): what this goal IS (permanent)
    - goal_why (committed): why it matters (permanent)
    - goal_last_state (derivable): where we left off (updated on touch/flush)
    - goal_next_step (derivable): next concrete action (updated on touch/flush)
  Based on:
    - ACT-R goal/imaginal buffer separation
    - SOAR O-support (committed) vs I-support (derived)
    - Duncan 2013: chunking prevents goal neglect
    - Altmann & Trafton 2007: cascading priming
    - Goal Drift arXiv:2505.02709: strong elicitation reduces drift

Neuroscience basis:
  - Altmann & Trafton 2002: Memory for Goals (activation model)
  - Cox et al 2017: Goal Operations for Cognitive Systems
  - Pink et al 2025: Episodic Memory for Long-Term LLM Agents
  - ACT-R base-level learning (Anderson 2007)

Created: 2026-03-02 (Sprint 15)
Updated: 2026-03-02 (Sprint 15.5 - Structured Context)
"""

import json
import logging
import sqlite3
import uuid
from datetime import datetime, timedelta
from typing import Optional

from modules.config import PROSPECTIVE_DB_PATH, connect_fts, now_iso
from modules.activation import (
    compute_unified_activation,
    ACTR_DECAY_DEFAULT,
)

_logger = logging.getLogger(__name__)

# ============================================================
# CONSTANTS
# ============================================================

# Goal-specific decay: between episodic (0.40) and semantic (0.15).
# Goals are more persistent than episodes but not permanent.
DECAY_GOAL = 0.35

# Hygiene staleness thresholds (Altmann & Trafton 2002: goals decay)
GOAL_STALENESS_DAYS = {
    "task": 3,
    "sprint": 14,
    "phase": 60,
    "project": 90,
}

# Priority-based initial activation (mirrors prospective model)
GOAL_ACTIVATION_INITIAL = {
    "critical": 0.95,
    "high": 0.80,
    "medium": 0.65,
    "low": 0.45,
}

# Valid levels and statuses
GOAL_LEVELS = ("project", "phase", "sprint", "task")
GOAL_STATUSES = ("active", "paused", "completed", "abandoned")
GOAL_PRIORITIES = ("critical", "high", "medium", "low")

# Max goals to inject into context (budget)
GOAL_CONTEXT_MAX = 5

# Context freshness: warn if derivable fields older than this (days)
CONTEXT_STALE_DAYS = 7


# ============================================================
# SQLITE CONNECTION (reuse prospective.db)
# ============================================================

_conn = None


def _get_conn() -> sqlite3.Connection:
    global _conn
    if _conn is None:
        _conn = connect_fts(PROSPECTIVE_DB_PATH)
        _init_tables(_conn)
    return _conn


def _init_tables(conn: sqlite3.Connection):
    """Validate goals tables exist (created by migration 002)."""
    from modules.migrations import ensure_schema_ready
    ensure_schema_ready(conn, ["goals", "goal_log"], db_label="prospective")
    _logger.info("Goals tables validated OK")


def _log_event(conn: sqlite3.Connection, goal_id: str, event: str, detail: str = ""):
    """Write to goal_log audit trail."""
    conn.execute(
        "INSERT INTO goal_log (goal_id, event, detail, created_at) VALUES (?, ?, ?, ?)",
        (goal_id, event, detail, now_iso()),
    )


# ============================================================
# 1. FORMULATE: create_goal()
# ============================================================

def create_goal(
    title: str,
    level: str = "task",
    parent_id: Optional[str] = None,
    priority: str = "medium",
    deadline: Optional[str] = None,
    assigned_to: Optional[str] = None,
    context: Optional[str] = None,
    metadata: Optional[dict] = None,
    goal_what: Optional[str] = None,
    goal_why: Optional[str] = None,
    goal_next_step: Optional[str] = None,
) -> dict:
    """Create a new goal (Cox 2017: Formulate operation).

    Args:
        title: Goal description.
        level: project|phase|sprint|task.
        parent_id: Parent goal ID for hierarchy.
        priority: critical|high|medium|low.
        deadline: ISO timestamp (optional).
        assigned_to: Agent or person (optional).
        context: Legacy blob (deprecated, use structured fields).
        metadata: Arbitrary JSON metadata.
        goal_what: COMMITTED - What this goal IS (1 line, permanent).
        goal_why: COMMITTED - Why it matters (1 line, permanent).
        goal_next_step: DERIVABLE - Next concrete action.

    Returns:
        dict with goal details + warnings if structured fields missing.
    """
    if level not in GOAL_LEVELS:
        return {"error": f"Invalid level '{level}'. Must be one of {GOAL_LEVELS}"}
    if priority not in GOAL_PRIORITIES:
        return {"error": f"Invalid priority '{priority}'. Must be one of {GOAL_PRIORITIES}"}

    conn = _get_conn()
    goal_id = str(uuid.uuid4())[:8]
    now = now_iso()

    # Validate parent exists if specified
    if parent_id:
        parent = conn.execute(
            "SELECT id, level FROM goals WHERE id = ?", (parent_id,)
        ).fetchone()
        if not parent:
            return {"error": f"Parent goal '{parent_id}' not found"}

    meta_json = json.dumps(metadata, ensure_ascii=False) if metadata else None

    conn.execute("""
        INSERT INTO goals (id, title, parent_id, level, status, priority,
                          access_count, created_at, last_accessed, deadline,
                          assigned_to, context, metadata,
                          goal_what, goal_why, goal_last_state, goal_next_step,
                          context_updated_at)
        VALUES (?, ?, ?, ?, 'active', ?, 0, ?, ?, ?, ?, ?, ?,
                ?, ?, NULL, ?, ?)
    """, (goal_id, title, parent_id, level, priority, now, now,
          deadline, assigned_to, context, meta_json,
          goal_what, goal_why, goal_next_step, now))

    _log_event(conn, goal_id, "created", f"level={level} priority={priority}")
    conn.commit()

    result = {
        "id": goal_id,
        "title": title,
        "level": level,
        "parent_id": parent_id,
        "priority": priority,
        "status": "active",
        "created_at": now,
    }

    # Warn if structured context is missing (strong elicitation)
    warnings = []
    if not goal_what:
        warnings.append("goal_what is empty — provide a 1-line description of WHAT this goal is")
    if not goal_why:
        warnings.append("goal_why is empty — provide a 1-line description of WHY it matters")
    if warnings:
        result["warnings"] = warnings

    return result


# ============================================================
# 2. SELECT: get_active_goals()
# ============================================================

def _compute_goal_activation(row: tuple) -> float:
    """Compute ACT-R activation for a goal row.

    Uses the unified activation scorer from activation.py with
    goal-specific decay (DECAY_GOAL = 0.35).

    Args:
        row: SQL result with columns:
              0:id, 1:title, 2:parent_id, 3:level, 4:status, 5:priority,
              6:access_count, 7:created_at, 8:last_accessed, 9:deadline,
              10:assigned_to, 11:context, 12:metadata,
              13:goal_what, 14:goal_why, 15:goal_last_state, 16:goal_next_step,
              17:context_updated_at
    """
    result = compute_unified_activation(
        memory_id=row[0],
        created_at=row[7],
        last_accessed=row[8],
        access_count=row[6],
        importance=row[5],  # priority maps to importance
        decay_override=DECAY_GOAL,
        memory_type="episodic",
        noise=False,  # Deterministic for goal selection
    )
    return result.total


def get_active_goals(
    status: str = "active",
    level: Optional[str] = None,
    limit: int = 10,
) -> list:
    """Get goals ranked by ACT-R activation (Cox 2017: Select operation).

    Args:
        status: Filter by status (default: active).
        level: Filter by level (optional).
        limit: Max results.

    Returns:
        List of goal dicts with activation scores.
    """
    conn = _get_conn()

    query = ("SELECT id, title, parent_id, level, status, priority, "
             "access_count, created_at, last_accessed, deadline, "
             "assigned_to, context, metadata, "
             "goal_what, goal_why, goal_last_state, goal_next_step, "
             "context_updated_at "
             "FROM goals WHERE status = ?")
    params = [status]

    if level:
        query += " AND level = ?"
        params.append(level)

    rows = conn.execute(query, params).fetchall()

    # Compute activation for each goal
    scored = []
    for row in rows:
        activation = _compute_goal_activation(row)
        scored.append({
            "id": row[0],
            "title": row[1],
            "parent_id": row[2],
            "level": row[3],
            "status": row[4],
            "priority": row[5],
            "access_count": row[6],
            "created_at": row[7],
            "last_accessed": row[8],
            "deadline": row[9],
            "assigned_to": row[10],
            "goal_what": row[13],
            "goal_why": row[14],
            "goal_last_state": row[15],
            "goal_next_step": row[16],
            "context_updated_at": row[17],
            "activation": round(activation, 4),
        })

    # Sort by activation descending
    scored.sort(key=lambda g: g["activation"], reverse=True)
    return scored[:limit]


def get_context_goals(limit: int = GOAL_CONTEXT_MAX) -> dict:
    """Get top goals above interference level for context injection.

    Interference level (Altmann & Trafton 2002): AVG(activation) of
    all active goals. Only goals above this threshold are injected.

    Returns:
        dict with goals (including structured context), interference level,
        and staleness warnings.
    """
    all_active = get_active_goals(status="active", limit=100)
    if not all_active:
        return {}

    # Compute interference level = mean activation of active goals
    activations = [g["activation"] for g in all_active]
    interference_level = sum(activations) / len(activations)

    # Filter: only goals above interference level
    above = [g for g in all_active if g["activation"] > interference_level]
    result = above[:limit]

    # Check context freshness (Duncan 2013: stale chunks cause neglect)
    stale_warnings = []
    now = datetime.now()
    for g in result:
        has_context = bool(g.get("goal_what") or g.get("goal_why"))
        if not has_context:
            stale_warnings.append(f"{g['id']}:{g['title']} — NO context set")
        else:
            ctx_ts = g.get("context_updated_at")
            if ctx_ts:
                try:
                    ctx_dt = datetime.fromisoformat(ctx_ts)
                    days_old = (now - ctx_dt.replace(tzinfo=None)).days
                    if days_old > CONTEXT_STALE_DAYS:
                        stale_warnings.append(
                            f"{g['id']}:{g['title']} — context {days_old}d old (stale)")
                except Exception:
                    pass

    return {
        "goals": result,
        "interference_level": round(interference_level, 4),
        "total_active": len(all_active),
        "above_threshold": len(above),
        "stale_warnings": stale_warnings,
    }


# ============================================================
# 3. CHANGE: update_goal()
# ============================================================

def update_goal(
    goal_id: str,
    status: Optional[str] = None,
    priority: Optional[str] = None,
    deadline: Optional[str] = None,
    title: Optional[str] = None,
    assigned_to: Optional[str] = None,
    goal_last_state: Optional[str] = None,
    goal_next_step: Optional[str] = None,
) -> dict:
    """Update goal fields (Cox 2017: Change operation).

    Committed fields (goal_what, goal_why) are NOT updatable here —
    they are permanent and set at creation (SOAR O-support pattern).

    Args:
        goal_id: Goal ID to update.
        status: New status (optional).
        priority: New priority (optional).
        deadline: New deadline (optional).
        title: New title (optional).
        assigned_to: New assignee (optional).
        goal_last_state: DERIVABLE - Where we left off (optional).
        goal_next_step: DERIVABLE - Next concrete action (optional).

    Returns:
        dict with updated goal or error.
    """
    conn = _get_conn()

    goal = conn.execute("SELECT id, status FROM goals WHERE id = ?", (goal_id,)).fetchone()
    if not goal:
        return {"error": f"Goal '{goal_id}' not found"}

    updates = []
    params = []
    changes = []

    if status is not None:
        if status not in GOAL_STATUSES:
            return {"error": f"Invalid status '{status}'. Must be one of {GOAL_STATUSES}"}
        updates.append("status = ?")
        params.append(status)
        changes.append(f"status→{status}")

    if priority is not None:
        if priority not in GOAL_PRIORITIES:
            return {"error": f"Invalid priority '{priority}'. Must be one of {GOAL_PRIORITIES}"}
        updates.append("priority = ?")
        params.append(priority)
        changes.append(f"priority→{priority}")

    if deadline is not None:
        updates.append("deadline = ?")
        params.append(deadline)
        changes.append(f"deadline→{deadline}")

    if title is not None:
        updates.append("title = ?")
        params.append(title)
        changes.append(f"title updated")

    if assigned_to is not None:
        updates.append("assigned_to = ?")
        params.append(assigned_to)
        changes.append(f"assigned_to→{assigned_to}")

    # Derivable fields (SOAR I-support: recomputable, updated on touch/flush)
    if goal_last_state is not None:
        updates.append("goal_last_state = ?")
        params.append(goal_last_state)
        changes.append("last_state updated")

    if goal_next_step is not None:
        updates.append("goal_next_step = ?")
        params.append(goal_next_step)
        changes.append("next_step updated")

    if not updates:
        return {"error": "No fields to update"}

    # Touch access + refresh context timestamp if derivable fields changed
    now = now_iso()
    updates.append("last_accessed = ?")
    params.append(now)
    updates.append("access_count = access_count + 1")

    if goal_last_state is not None or goal_next_step is not None:
        updates.append("context_updated_at = ?")
        params.append(now)

    params.append(goal_id)
    conn.execute(
        f"UPDATE goals SET {', '.join(updates)} WHERE id = ?",
        params,
    )
    _log_event(conn, goal_id, "updated", ", ".join(changes))
    conn.commit()

    return {"id": goal_id, "changes": changes}


# ============================================================
# 4. DELEGATE: assign_goal()
# ============================================================

def assign_goal(goal_id: str, assigned_to: str) -> dict:
    """Assign goal to an agent or person (Cox 2017: Delegate operation)."""
    return update_goal(goal_id, assigned_to=assigned_to)


# ============================================================
# 5. ACHIEVE: complete_goal()
# ============================================================

def complete_goal(goal_id: str, outcome: str = "") -> dict:
    """Mark goal as completed and check cascade (Cox 2017: Achieve operation).

    If all sibling tasks of a parent are completed, suggests completing parent.

    Args:
        goal_id: Goal to complete.
        outcome: What happened.

    Returns:
        dict with completion info and cascade suggestions.
    """
    conn = _get_conn()

    goal = conn.execute(
        "SELECT id, title, parent_id, level FROM goals WHERE id = ?",
        (goal_id,),
    ).fetchone()
    if not goal:
        return {"error": f"Goal '{goal_id}' not found"}

    now = now_iso()
    conn.execute(
        "UPDATE goals SET status = 'completed', last_accessed = ?, "
        "access_count = access_count + 1 WHERE id = ?",
        (now, goal_id),
    )
    _log_event(conn, goal_id, "completed", outcome)
    conn.commit()

    result = {
        "id": goal_id,
        "title": goal[1],
        "status": "completed",
        "outcome": outcome,
    }

    # Cascade check: are all siblings of parent completed?
    parent_id = goal[2]
    if parent_id:
        siblings = conn.execute(
            "SELECT id, status FROM goals WHERE parent_id = ?",
            (parent_id,),
        ).fetchall()
        all_done = all(s[1] in ("completed", "abandoned") for s in siblings)
        if all_done:
            parent = conn.execute(
                "SELECT id, title FROM goals WHERE id = ?", (parent_id,)
            ).fetchone()
            result["cascade_suggestion"] = {
                "parent_id": parent_id,
                "parent_title": parent[1] if parent else "?",
                "message": f"All children completed. Consider completing parent '{parent[1]}'.",
            }

    return result


# ============================================================
# 6. MONITOR: check_goal_hygiene()
# ============================================================

def check_goal_hygiene() -> dict:
    """Detect and pause stale goals (Cox 2017: Monitor operation).

    Staleness thresholds by level:
      - task: 3 days
      - sprint: 14 days
      - phase: 60 days
      - project: 90 days

    Returns:
        dict with paused goals and summary.
    """
    conn = _get_conn()
    now = datetime.now()
    paused = []

    for level, days in GOAL_STALENESS_DAYS.items():
        threshold = (now - timedelta(days=days)).isoformat()
        stale = conn.execute(
            "SELECT id, title, last_accessed FROM goals "
            "WHERE status = 'active' AND level = ? AND last_accessed < ?",
            (level, threshold),
        ).fetchall()

        for goal_id, title, last_accessed in stale:
            conn.execute(
                "UPDATE goals SET status = 'paused', last_accessed = ? WHERE id = ?",
                (now_iso(), goal_id),
            )
            _log_event(conn, goal_id, "auto_paused",
                       f"Stale: no access in {days}d (level={level})")
            paused.append({
                "id": goal_id,
                "title": title,
                "level": level,
                "last_accessed": last_accessed,
                "days_stale": days,
            })

    if paused:
        conn.commit()

    return {
        "paused_count": len(paused),
        "paused": paused,
    }


# ============================================================
# TOUCH (access tracking for activation)
# ============================================================

def touch_goal(
    goal_id: str,
    last_state: Optional[str] = None,
    next_step: Optional[str] = None,
) -> dict:
    """Record an access to a goal (boosts activation).

    Called when a goal is referenced in conversation or used
    for context injection. Optionally updates derivable context.

    Args:
        goal_id: Goal to touch.
        last_state: Update where we left off (derivable, optional).
        next_step: Update next action (derivable, optional).
    """
    conn = _get_conn()
    goal = conn.execute("SELECT id FROM goals WHERE id = ?", (goal_id,)).fetchone()
    if not goal:
        return {"error": f"Goal '{goal_id}' not found"}

    now = now_iso()
    updates = ["access_count = access_count + 1", "last_accessed = ?"]
    params = [now]

    if last_state is not None:
        updates.append("goal_last_state = ?")
        params.append(last_state)

    if next_step is not None:
        updates.append("goal_next_step = ?")
        params.append(next_step)

    if last_state is not None or next_step is not None:
        updates.append("context_updated_at = ?")
        params.append(now)

    params.append(goal_id)
    conn.execute(
        f"UPDATE goals SET {', '.join(updates)} WHERE id = ?",
        params,
    )
    conn.commit()
    return {"id": goal_id, "touched": True, "context_refreshed": last_state is not None or next_step is not None}


# ============================================================
# GOAL TREE (hierarchy view)
# ============================================================

def get_goal_tree(root_id: Optional[str] = None) -> list:
    """Get hierarchical view of goals.

    Args:
        root_id: Start from specific goal. None = all top-level.

    Returns:
        List of goal dicts with children nested.
    """
    conn = _get_conn()

    if root_id:
        roots = conn.execute(
            "SELECT id, title, level, status, priority, access_count, "
            "created_at, last_accessed FROM goals WHERE id = ?",
            (root_id,),
        ).fetchall()
    else:
        roots = conn.execute(
            "SELECT id, title, level, status, priority, access_count, "
            "created_at, last_accessed FROM goals WHERE parent_id IS NULL "
            "ORDER BY created_at DESC",
        ).fetchall()

    def _build_node(row):
        node = {
            "id": row[0], "title": row[1], "level": row[2],
            "status": row[3], "priority": row[4],
        }
        children = conn.execute(
            "SELECT id, title, level, status, priority, access_count, "
            "created_at, last_accessed FROM goals WHERE parent_id = ? "
            "ORDER BY created_at",
            (row[0],),
        ).fetchall()
        if children:
            node["children"] = [_build_node(c) for c in children]
        return node

    return [_build_node(r) for r in roots]


# ============================================================
# MCP TOOL REGISTRATION
# ============================================================

def register_goal_tools(mcp):
    """Register goal system MCP tools."""

    @mcp.tool()
    def crear_goal(
        title: str,
        level: str = "task",
        parent_id: str = "",
        priority: str = "medium",
        deadline: str = "",
        assigned_to: str = "",
        goal_what: str = "",
        goal_why: str = "",
        goal_next_step: str = "",
    ) -> str:
        """Create a new goal in the activation-based priority agenda.

        Goals are NOT a stack -- they compete by ACT-R activation.
        Hierarchy: project > phase > sprint > task.

        IMPORTANT: Always provide goal_what and goal_why for context persistence.
        These are COMMITTED fields that help future sessions understand the goal.

        Args:
            title: Short goal name (for display)
            level: project|phase|sprint|task
            parent_id: Parent goal ID for hierarchy (optional)
            priority: critical|high|medium|low
            deadline: ISO timestamp deadline (optional)
            assigned_to: Agent or person name (optional)
            goal_what: REQUIRED - What this goal IS (1 line, permanent)
            goal_why: REQUIRED - Why it matters (1 line, permanent)
            goal_next_step: Next concrete action to take (updated over time)
        """
        result = create_goal(
            title=title,
            level=level,
            parent_id=parent_id or None,
            priority=priority,
            deadline=deadline or None,
            assigned_to=assigned_to or None,
            goal_what=goal_what or None,
            goal_why=goal_why or None,
            goal_next_step=goal_next_step or None,
        )
        return json.dumps(result, indent=2, ensure_ascii=False)

    @mcp.tool()
    def ver_goals(
        status: str = "active",
        level: str = "",
        limit: int = 10,
    ) -> str:
        """Show goals ranked by ACT-R activation.

        Args:
            status: Filter by status (active|paused|completed|abandoned)
            level: Filter by level (project|phase|sprint|task), empty=all
            limit: Max results (default 10)
        """
        goals = get_active_goals(
            status=status,
            level=level or None,
            limit=limit,
        )
        return json.dumps(goals, indent=2, ensure_ascii=False)

    @mcp.tool()
    def actualizar_goal(
        goal_id: str,
        status: str = "",
        priority: str = "",
        deadline: str = "",
        title: str = "",
        assigned_to: str = "",
        goal_last_state: str = "",
        goal_next_step: str = "",
    ) -> str:
        """Update a goal's fields. Committed fields (what/why) are permanent.

        Use goal_last_state and goal_next_step to refresh derivable context.
        These should be updated whenever you work on a goal.

        Args:
            goal_id: The goal ID to update
            status: New status (active|paused|completed|abandoned)
            priority: New priority (critical|high|medium|low)
            deadline: New deadline (ISO timestamp)
            title: New title
            assigned_to: New assignee
            goal_last_state: Where we left off (derivable, refreshable)
            goal_next_step: Next concrete action (derivable, refreshable)
        """
        result = update_goal(
            goal_id=goal_id,
            status=status or None,
            priority=priority or None,
            deadline=deadline or None,
            title=title or None,
            assigned_to=assigned_to or None,
            goal_last_state=goal_last_state or None,
            goal_next_step=goal_next_step or None,
        )
        return json.dumps(result, indent=2, ensure_ascii=False)

    @mcp.tool()
    def completar_goal(
        goal_id: str,
        outcome: str = "",
    ) -> str:
        """Mark a goal as completed. Checks if parent can be completed too.

        Args:
            goal_id: The goal ID to complete
            outcome: What happened / result description
        """
        result = complete_goal(goal_id=goal_id, outcome=outcome)
        return json.dumps(result, indent=2, ensure_ascii=False)

    @mcp.tool()
    def contexto_goals(limit: int = 5) -> str:
        """Get top goals above interference level for context injection.

        Uses Altmann & Trafton (2002) interference level: AVG(activation)
        of all active goals. Only goals above this threshold are returned.

        Args:
            limit: Max goals to return (default 5)
        """
        result = get_context_goals(limit=limit)
        return json.dumps(result, indent=2, ensure_ascii=False)

    @mcp.tool()
    def arbol_goals(root_id: str = "") -> str:
        """Show goal hierarchy as a tree.

        Args:
            root_id: Start from specific goal (empty = all top-level)
        """
        tree = get_goal_tree(root_id=root_id or None)
        return json.dumps(tree, indent=2, ensure_ascii=False)
