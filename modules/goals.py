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
Updated: 2026-03-05 (Fase 5 - migrated from SQLite prospective.db to PostgreSQL)
"""

import json
import logging
import uuid
from datetime import datetime, timedelta
from typing import Optional

from psycopg.rows import dict_row

from modules.config import now_iso
from modules.config_pg import get_conn
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
# POSTGRESQL INIT
# ============================================================

_TABLES_INITIALIZED = False


def _ensure_tables():
    """Create PG tables if not exist. Called once per process."""
    global _TABLES_INITIALIZED
    if _TABLES_INITIALIZED:
        return
    try:
        with get_conn() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS goals (
                    id               TEXT PRIMARY KEY,
                    title            TEXT NOT NULL,
                    parent_id        TEXT REFERENCES goals(id) ON DELETE SET NULL,
                    level            TEXT NOT NULL,
                    status           TEXT DEFAULT 'active',
                    priority         TEXT DEFAULT 'medium',
                    access_count     INTEGER DEFAULT 0,
                    created_at       TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    last_accessed    TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    deadline         TIMESTAMPTZ,
                    assigned_to      TEXT,
                    context          TEXT,
                    metadata         JSONB,
                    goal_what        TEXT,
                    goal_why         TEXT,
                    goal_last_state  TEXT,
                    goal_next_step   TEXT,
                    context_updated_at TIMESTAMPTZ
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS goal_log (
                    id         SERIAL PRIMARY KEY,
                    goal_id    TEXT NOT NULL REFERENCES goals(id) ON DELETE CASCADE,
                    event      TEXT NOT NULL,
                    detail     TEXT,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                )
            """)
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_goals_status "
                "ON goals(status)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_goals_parent "
                "ON goals(parent_id)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_goals_level "
                "ON goals(level)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_goals_last_accessed "
                "ON goals(last_accessed)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_goal_log_goal "
                "ON goal_log(goal_id)"
            )
        _TABLES_INITIALIZED = True
        _logger.info("Goals PG tables ready")
    except Exception as e:
        _logger.error("Failed to initialize goals PG tables: %s", e)
        raise


# ============================================================
# HELPERS
# ============================================================

def _ts(val) -> Optional[str]:
    """Convert TIMESTAMPTZ value (datetime or str) to ISO string."""
    if val is None:
        return None
    if hasattr(val, "isoformat"):
        return val.isoformat()
    return str(val)


def _log_event(conn, goal_id: str, event: str, detail: str = ""):
    """Write to goal_log audit trail (call inside a transaction)."""
    conn.execute(
        "INSERT INTO goal_log (goal_id, event, detail) VALUES (%s, %s, %s)",
        (goal_id, event, detail),
    )


def _compute_goal_activation(row: dict) -> float:
    """Compute ACT-R activation for a goal row dict.

    Uses the unified activation scorer from activation.py with
    goal-specific decay (DECAY_GOAL = 0.35).
    """
    result = compute_unified_activation(
        memory_id=row["id"],
        created_at=_ts(row["created_at"]) or "",
        last_accessed=_ts(row["last_accessed"]),
        access_count=row["access_count"],
        importance=row["priority"],  # priority maps to importance
        decay_override=DECAY_GOAL,
        memory_type="episodic",
        noise=False,  # Deterministic for goal selection
    )
    return result.total


def _row_to_dict(row: dict) -> dict:
    """Normalize a goal row dict (convert TIMESTAMPTZ to ISO strings)."""
    return {
        "id": row["id"],
        "title": row["title"],
        "parent_id": row["parent_id"],
        "level": row["level"],
        "status": row["status"],
        "priority": row["priority"],
        "access_count": row["access_count"],
        "created_at": _ts(row["created_at"]),
        "last_accessed": _ts(row["last_accessed"]),
        "deadline": _ts(row["deadline"]),
        "assigned_to": row["assigned_to"],
        "goal_what": row["goal_what"],
        "goal_why": row["goal_why"],
        "goal_last_state": row["goal_last_state"],
        "goal_next_step": row["goal_next_step"],
        "context_updated_at": _ts(row["context_updated_at"]),
    }


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
    _ensure_tables()
    if level not in GOAL_LEVELS:
        return {"error": f"Invalid level '{level}'. Must be one of {GOAL_LEVELS}"}
    if priority not in GOAL_PRIORITIES:
        return {"error": f"Invalid priority '{priority}'. Must be one of {GOAL_PRIORITIES}"}

    goal_id = str(uuid.uuid4())

    with get_conn() as conn:
        with conn.transaction():
            # Validate parent exists if specified
            if parent_id:
                with conn.cursor(row_factory=dict_row) as cur:
                    cur.execute(
                        "SELECT id, level FROM goals WHERE id = %s",
                        (parent_id,),
                    )
                    parent = cur.fetchone()
                if not parent:
                    return {"error": f"Parent goal '{parent_id}' not found"}

            conn.execute("""
                INSERT INTO goals (id, title, parent_id, level, status, priority,
                                  access_count, deadline, assigned_to, context, metadata,
                                  goal_what, goal_why, goal_last_state, goal_next_step,
                                  context_updated_at)
                VALUES (%s, %s, %s, %s, 'active', %s, 0, %s, %s, %s, %s,
                        %s, %s, NULL, %s, NOW())
            """, (goal_id, title, parent_id, level, priority,
                  deadline, assigned_to, context,
                  json.dumps(metadata, ensure_ascii=False) if metadata else None,
                  goal_what, goal_why, goal_next_step))

            _log_event(conn, goal_id, "created", f"level={level} priority={priority}")

    result = {
        "id": goal_id,
        "title": title,
        "level": level,
        "parent_id": parent_id,
        "priority": priority,
        "status": "active",
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
    _ensure_tables()
    query = (
        "SELECT id, title, parent_id, level, status, priority, "
        "access_count, created_at, last_accessed, deadline, "
        "assigned_to, context, metadata, "
        "goal_what, goal_why, goal_last_state, goal_next_step, "
        "context_updated_at "
        "FROM goals WHERE status = %s"
    )
    params = [status]

    if level:
        query += " AND level = %s"
        params.append(level)

    with get_conn() as conn:
        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute(query, params)
            rows = cur.fetchall()

    # Compute activation for each goal
    scored = []
    for row in rows:
        activation = _compute_goal_activation(row)
        d = _row_to_dict(row)
        d["activation"] = round(activation, 4)
        scored.append(d)

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
    _ensure_tables()
    with get_conn() as conn:
        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute("SELECT id, status FROM goals WHERE id = %s", (goal_id,))
            goal = cur.fetchone()
        if not goal:
            return {"error": f"Goal '{goal_id}' not found"}

        updates = []
        params = []
        changes = []

        if status is not None:
            if status not in GOAL_STATUSES:
                return {"error": f"Invalid status '{status}'. Must be one of {GOAL_STATUSES}"}
            updates.append("status = %s")
            params.append(status)
            changes.append(f"status→{status}")

        if priority is not None:
            if priority not in GOAL_PRIORITIES:
                return {"error": f"Invalid priority '{priority}'. Must be one of {GOAL_PRIORITIES}"}
            updates.append("priority = %s")
            params.append(priority)
            changes.append(f"priority→{priority}")

        if deadline is not None:
            updates.append("deadline = %s")
            params.append(deadline)
            changes.append(f"deadline→{deadline}")

        if title is not None:
            updates.append("title = %s")
            params.append(title)
            changes.append("title updated")

        if assigned_to is not None:
            updates.append("assigned_to = %s")
            params.append(assigned_to)
            changes.append(f"assigned_to→{assigned_to}")

        # Derivable fields (SOAR I-support: recomputable, updated on touch/flush)
        if goal_last_state is not None:
            updates.append("goal_last_state = %s")
            params.append(goal_last_state)
            changes.append("last_state updated")

        if goal_next_step is not None:
            updates.append("goal_next_step = %s")
            params.append(goal_next_step)
            changes.append("next_step updated")

        if not updates:
            return {"error": "No fields to update"}

        # Touch access + refresh context timestamp if derivable fields changed
        updates.append("last_accessed = NOW()")
        updates.append("access_count = access_count + 1")

        if goal_last_state is not None or goal_next_step is not None:
            updates.append("context_updated_at = NOW()")

        params.append(goal_id)
        with conn.transaction():
            conn.execute(
                f"UPDATE goals SET {', '.join(updates)} WHERE id = %s",
                params,
            )
            _log_event(conn, goal_id, "updated", ", ".join(changes))

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
    _ensure_tables()
    with get_conn() as conn:
        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                "SELECT id, title, parent_id, level FROM goals WHERE id = %s",
                (goal_id,),
            )
            goal = cur.fetchone()
        if not goal:
            return {"error": f"Goal '{goal_id}' not found"}

        with conn.transaction():
            conn.execute(
                "UPDATE goals SET status = 'completed', last_accessed = NOW(), "
                "access_count = access_count + 1 WHERE id = %s",
                (goal_id,),
            )
            _log_event(conn, goal_id, "completed", outcome)

        result = {
            "id": goal_id,
            "title": goal["title"],
            "status": "completed",
            "outcome": outcome,
        }

        # Cascade check: are all siblings of parent completed?
        parent_id = goal["parent_id"]
        if parent_id:
            with conn.cursor(row_factory=dict_row) as cur:
                cur.execute(
                    "SELECT id, status FROM goals WHERE parent_id = %s",
                    (parent_id,),
                )
                siblings = cur.fetchall()
            all_done = all(s["status"] in ("completed", "abandoned") for s in siblings)
            if all_done:
                with conn.cursor(row_factory=dict_row) as cur:
                    cur.execute(
                        "SELECT id, title FROM goals WHERE id = %s",
                        (parent_id,),
                    )
                    parent = cur.fetchone()
                parent_title = parent["title"] if parent else "?"
                result["cascade_suggestion"] = {
                    "parent_id": parent_id,
                    "parent_title": parent_title,
                    "message": f"All children completed. Consider completing parent '{parent_title}'.",
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
    _ensure_tables()
    now = datetime.now()
    paused = []

    with get_conn() as conn:
        with conn.transaction():
            for level, days in GOAL_STALENESS_DAYS.items():
                threshold = now - timedelta(days=days)
                with conn.cursor(row_factory=dict_row) as cur:
                    cur.execute(
                        "SELECT id, title, last_accessed FROM goals "
                        "WHERE status = 'active' AND level = %s AND last_accessed < %s",
                        (level, threshold),
                    )
                    stale = cur.fetchall()

                for row in stale:
                    conn.execute(
                        "UPDATE goals SET status = 'paused', last_accessed = NOW() "
                        "WHERE id = %s",
                        (row["id"],),
                    )
                    _log_event(conn, row["id"], "auto_paused",
                               f"Stale: no access in {days}d (level={level})")
                    paused.append({
                        "id": row["id"],
                        "title": row["title"],
                        "level": level,
                        "last_accessed": _ts(row["last_accessed"]),
                        "days_stale": days,
                    })

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
    _ensure_tables()
    with get_conn() as conn:
        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute("SELECT id FROM goals WHERE id = %s", (goal_id,))
            goal = cur.fetchone()
        if not goal:
            return {"error": f"Goal '{goal_id}' not found"}

        updates = ["access_count = access_count + 1", "last_accessed = NOW()"]
        params = []

        if last_state is not None:
            updates.append("goal_last_state = %s")
            params.append(last_state)

        if next_step is not None:
            updates.append("goal_next_step = %s")
            params.append(next_step)

        if last_state is not None or next_step is not None:
            updates.append("context_updated_at = NOW()")

        params.append(goal_id)
        with conn.transaction():
            conn.execute(
                f"UPDATE goals SET {', '.join(updates)} WHERE id = %s",
                params,
            )

    return {
        "id": goal_id,
        "touched": True,
        "context_refreshed": last_state is not None or next_step is not None,
    }


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
    _ensure_tables()
    with get_conn() as conn:
        with conn.cursor(row_factory=dict_row) as cur:
            if root_id:
                cur.execute(
                    "SELECT id, title, level, status, priority, access_count, "
                    "created_at, last_accessed FROM goals WHERE id = %s",
                    (root_id,),
                )
            else:
                cur.execute(
                    "SELECT id, title, level, status, priority, access_count, "
                    "created_at, last_accessed FROM goals WHERE parent_id IS NULL "
                    "ORDER BY created_at DESC"
                )
            roots = cur.fetchall()

        def _build_node(row, cur):
            node = {
                "id": row["id"],
                "title": row["title"],
                "level": row["level"],
                "status": row["status"],
                "priority": row["priority"],
            }
            cur.execute(
                "SELECT id, title, level, status, priority, access_count, "
                "created_at, last_accessed FROM goals WHERE parent_id = %s "
                "ORDER BY created_at",
                (row["id"],),
            )
            children = cur.fetchall()
            if children:
                node["children"] = [_build_node(c, cur) for c in children]
            return node

        with conn.cursor(row_factory=dict_row) as cur:
            return [_build_node(r, cur) for r in roots]


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
