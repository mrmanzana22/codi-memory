"""
PROSPECTIVE MEMORY MODULE - Phase 2 of Codi Consciousness Project

Implements future-oriented memory: "remembering to do things."
Two forms (Einstein & McDaniel, 2005):
  - Event-based: "when X happens, do Y"
  - Time-based: "in N days, check Z"

Monitoring uses tiered approach (Multiprocess Theory):
  - Tier 1 (Focal/Spontaneous): Fast keyword matching (~5ms)
  - Tier 2 (Nonfocal/Strategic): Semantic overlap check (~40ms)

Neuroscience basis:
  - Multiprocess Theory (McDaniel & Einstein, 2005)
  - BA10 intention maintenance (Burgess et al., 2011)
  - Deadline monitoring intensification (Hicks, Marsh & Cook, 2005)
  - Temporal monitoring (Cona, Arcara, Tarantino & Bisiacchi, 2012)
  - Reflective Memory Management (ACL 2025)

Created: 2026-02-13 (Phase 2, Sub-phase 2.1)
"""

import logging
import os
import re
import json
import uuid
import sqlite3
import time
import random
import math
from datetime import datetime, timedelta

_logger = logging.getLogger(__name__)

from modules.config import FTS_DB_PATH, PROSPECTIVE_DB_PATH, now_iso, now_col, connect_fts

# ============================================================
# CONSTANTS
# ============================================================

# PROSPECTIVE_DB_PATH imported from modules.config

# Monitoring budget (ms) - must stay within pre-turn hook's 500ms
PM_CHECK_BUDGET_MS = 50

# Activation parameters (priority-based initial + power-law decay)
PM_ACTIVATION_INITIAL = {
    "critical": 0.95,
    "high": 0.80,
    "medium": 0.65,
    "low": 0.45,
}
PM_DECAY_EXPONENT = 0.5              # Power-law decay rate (custom model, NOT ACT-R)
PM_ACTIVATION_PARTIAL_BOOST = 0.05
PM_ACTIVATION_NOISE_SIGMA = 0.03     # Stochastic noise for non-deterministic retrieval
PM_URGENCY_WINDOW_DAYS = 7           # Deadline monitoring window
# Activation floor per priority (BA10 tonic maintenance -- intentions don't
# fully disappear, they maintain a low-level bias until explicitly expired)
PM_ACTIVATION_FLOOR = {
    "critical": 0.30,
    "high": 0.25,
    "medium": 0.22,
    "low": 0.21,
}
PM_ACTIVATION_TRIGGER_THRESHOLD = {
    "critical": 0.10,
    "high": 0.15,
    "medium": 0.20,
    "low": 0.20,
}
PM_ACTIVATION_TRIGGER_THRESHOLD_DEFAULT = 0.20
PM_MAX_ACTIVE_INTENTIONS = 50


# ============================================================
# SQLITE INITIALIZATION
# ============================================================

_conn = None


def _get_conn() -> sqlite3.Connection:
    global _conn
    if _conn is None:
        _conn = connect_fts(PROSPECTIVE_DB_PATH)
        _init_tables(_conn)
    return _conn


def _init_tables(conn: sqlite3.Connection):
    """Validate prospective memory tables exist (created by migrations)."""
    from modules.migrations import ensure_schema_ready
    ensure_schema_ready(conn, ["intentions", "intention_log"], db_label="prospective")
    _logger.info("Tables validated OK")


# ============================================================
# INTENTION CRUD
# ============================================================

def _resolve_id(conn: sqlite3.Connection, prefix: str) -> str:
    """Resolve a partial ID to full ID."""
    if len(prefix) >= 36:
        return prefix
    row = conn.execute(
        "SELECT id FROM intentions WHERE id LIKE ? LIMIT 2", (f"{prefix}%",)
    ).fetchall()
    if len(row) == 1:
        return row[0][0]
    if len(row) == 0:
        raise ValueError(f"No intention found with prefix '{prefix}'")
    raise ValueError(f"Ambiguous prefix '{prefix}', matches {len(row)} intentions")


def create_intention(
    action: str,
    trigger_type: str = "event",
    trigger_spec: str = "{}",
    priority: str = "medium",
    expiry: str = "",
    context: str = "",
    creator: str = "codi",
    recurrence: str = "",
) -> dict:
    """Create a new prospective memory intention."""
    conn = _get_conn()
    int_id = str(uuid.uuid4())

    # Parse and validate trigger_spec
    try:
        spec = json.loads(trigger_spec) if isinstance(trigger_spec, str) else trigger_spec
    except json.JSONDecodeError:
        spec = {"keywords": [trigger_spec]}

    # Determine focality
    cue_focality = "focal"
    if trigger_type == "event" and spec.get("semantic_description"):
        cue_focality = "nonfocal"

    # Set initial activation based on priority
    activation = PM_ACTIVATION_INITIAL.get(priority, 0.65)

    # Emotional modulation: high arousal enhances PM encoding (LaBar & Cabeza, 2006)
    try:
        from modules.utils import _get_emotional_state
        emo = _get_emotional_state()
        arousal = emo.get("current", {}).get("arousal", 0.0)
        if arousal > 0.3:
            activation = min(1.0, activation + arousal * 0.1)  # Up to +0.1 boost
    except Exception:
        pass

    now = now_iso()
    conn.execute("""
        INSERT INTO intentions
        (id, action, trigger_type, trigger_spec, cue_focality, priority,
         activation, created_at, expiry, context_at_creation, creator, recurrence)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        int_id, action, trigger_type, json.dumps(spec), cue_focality,
        priority, activation, now, expiry or None, context, creator,
        recurrence or None,
    ))

    _log_event(conn, int_id, "created", json.dumps({
        "action": action, "trigger_type": trigger_type, "priority": priority,
    }))
    conn.commit()

    return {
        "id": int_id,
        "action": action,
        "trigger_type": trigger_type,
        "priority": priority,
        "activation": activation,
        "cue_focality": cue_focality,
    }


def get_pending_intentions(limit: int = 10) -> list:
    """Get all pending intentions sorted by activation."""
    conn = _get_conn()
    rows = conn.execute("""
        SELECT id, action, action_type, trigger_type, trigger_spec,
               priority, activation, created_at, expiry, status, creator
        FROM intentions
        WHERE status IN ('pending', 'triggered')
        ORDER BY
            CASE priority
                WHEN 'critical' THEN 0
                WHEN 'high' THEN 1
                WHEN 'medium' THEN 2
                WHEN 'low' THEN 3
            END,
            activation DESC
        LIMIT ?
    """, (limit,)).fetchall()

    return [
        {
            "id": r[0][:8],
            "full_id": r[0],
            "action": r[1],
            "action_type": r[2],
            "trigger_type": r[3],
            "trigger_spec": json.loads(r[4]) if r[4] else {},
            "priority": r[5],
            "activation": round(r[6], 2),
            "created_at": r[7],
            "expiry": r[8],
            "status": r[9],
            "creator": r[10],
        }
        for r in rows
    ]


def complete_intention(intention_id: str, outcome: str = "") -> dict:
    """Mark an intention as completed."""
    conn = _get_conn()
    full_id = _resolve_id(conn, intention_id)
    now = now_iso()

    conn.execute("""
        UPDATE intentions SET status = 'completed', completed_at = ?
        WHERE id = ?
    """, (now, full_id))

    _log_event(conn, full_id, "completed", outcome)
    conn.commit()
    return {"id": full_id[:8], "status": "completed", "outcome": outcome}


def cancel_intention(intention_id: str, reason: str = "") -> dict:
    """Cancel a pending intention."""
    conn = _get_conn()
    full_id = _resolve_id(conn, intention_id)

    conn.execute("""
        UPDATE intentions SET status = 'cancelled'
        WHERE id = ?
    """, (full_id,))

    _log_event(conn, full_id, "cancelled", reason)
    conn.commit()
    return {"id": full_id[:8], "status": "cancelled", "reason": reason}


def snooze_intention(intention_id: str, hours: float = 24.0) -> dict:
    """Snooze an intention for N hours."""
    conn = _get_conn()
    full_id = _resolve_id(conn, intention_id)
    snooze_until = (now_col() + timedelta(hours=hours)).isoformat()

    conn.execute("""
        UPDATE intentions SET snooze_until = ?, status = 'pending'
        WHERE id = ?
    """, (snooze_until, full_id))

    _log_event(conn, full_id, "snoozed", f"until {snooze_until}")
    conn.commit()
    return {"id": full_id[:8], "snoozed_until": snooze_until}


# ============================================================
# MONITORING ENGINE (called from pre-turn hook)
# ============================================================

def check_intentions(prompt: str, current_context: str = "") -> list:
    """Main monitoring function. Two-tier approach per Multiprocess Theory.

    Tier 1 (Spontaneous): Keyword matching for focal cues (~5ms)
    Tier 2 (Strategic): Semantic overlap for nonfocal cues (~40ms, budget permitting)
    Also checks time-based intentions.

    Returns list of triggered intentions.
    """
    start_ms = time.time() * 1000
    triggered = []

    conn = _get_conn()
    now = now_col()
    now_str = now.isoformat()

    # Expire stale intentions first
    _expire_stale(conn, now)

    # Load active intentions (use lowest threshold, filter per-priority below)
    rows = conn.execute("""
        SELECT id, action, action_type, trigger_type, trigger_spec,
               cue_focality, priority, activation, context_at_creation
        FROM intentions
        WHERE status = 'pending'
          AND activation > ?
          AND (snooze_until IS NULL OR snooze_until <= ?)
        ORDER BY activation DESC
        LIMIT ?
    """, (min(PM_ACTIVATION_TRIGGER_THRESHOLD.values()), now_str, PM_MAX_ACTIVE_INTENTIONS)).fetchall()

    if not rows:
        return []

    full_text = f"{prompt} {current_context}".lower()

    for row in rows:
        int_id, action, action_type, trigger_type, spec_raw, \
            cue_focality, priority, activation, ctx_at_creation = row

        # Budget check
        elapsed = time.time() * 1000 - start_ms
        if elapsed > PM_CHECK_BUDGET_MS:
            break

        # Priority-aware activation threshold
        threshold = PM_ACTIVATION_TRIGGER_THRESHOLD.get(priority, PM_ACTIVATION_TRIGGER_THRESHOLD_DEFAULT)
        if activation < threshold:
            continue

        try:
            spec = json.loads(spec_raw)
        except Exception:
            continue

        match_result = None

        # === TIME-BASED ===
        if trigger_type == "time":
            match_result = _check_time(spec, now)

        # === EVENT-BASED FOCAL (Tier 1) ===
        elif trigger_type == "event" and cue_focality == "focal":
            match_result = _check_focal(spec, full_text)

        # === EVENT-BASED NONFOCAL (Tier 2) ===
        elif trigger_type == "event" and cue_focality == "nonfocal":
            if elapsed < PM_CHECK_BUDGET_MS * 0.7:
                match_result = _check_nonfocal(spec, full_text, ctx_at_creation)

        # === CONDITION-BASED ===
        elif trigger_type == "condition":
            match_result = _check_condition(spec)

        # Process result
        if match_result and match_result.get("matched"):
            triggered.append({
                "id": int_id[:8],
                "full_id": int_id,
                "action": action,
                "action_type": action_type,
                "priority": priority,
                "activation": activation,
                "trigger_detail": match_result.get("detail", ""),
            })
            _mark_triggered(conn, int_id, match_result.get("detail", ""), now_str)
        elif match_result and match_result.get("partial"):
            _boost_activation(conn, int_id, PM_ACTIVATION_PARTIAL_BOOST)

        # Increment check counter
        conn.execute("""
            UPDATE intentions SET check_count = check_count + 1, last_checked_at = ?
            WHERE id = ?
        """, (now_str, int_id))

    conn.commit()

    # Rank and cap at 3 (workspace capacity per GWT)
    if len(triggered) > 3:
        priority_w = {"critical": 1.0, "high": 0.8, "medium": 0.5, "low": 0.3}
        triggered.sort(key=lambda t: -(priority_w.get(t["priority"], 0.5) * t["activation"]))
        triggered = triggered[:3]

    return triggered


# ============================================================
# TRIGGER CHECKERS
# ============================================================

def _check_focal(spec: dict, full_text: str) -> dict:
    """Tier 1: Fast keyword matching (spontaneous retrieval path)."""
    keywords = spec.get("keywords", [])
    threshold = spec.get("match_threshold", 1)

    if not keywords:
        return {"matched": False}

    matched_kw = [kw for kw in keywords if kw.lower() in full_text]

    if len(matched_kw) >= threshold:
        return {"matched": True, "detail": f"Keywords: {matched_kw}"}
    if matched_kw:
        return {"matched": False, "partial": True}
    return {"matched": False}


def _check_nonfocal(spec: dict, full_text: str, context_at_creation: str = "") -> dict:
    """Tier 2: Semantic overlap (strategic monitoring path). No API calls.

    Uses encoding context (context_at_creation) as additional matching signal
    per encoding specificity (Tulving & Thomson, 1973): PM retrieval is
    enhanced when current context overlaps with encoding context.
    """
    keywords = spec.get("keywords", [])

    # Fast pre-filter
    if keywords and not any(kw.lower() in full_text for kw in keywords):
        return {"matched": False}

    semantic_desc = spec.get("semantic_description", "")
    if not semantic_desc:
        matched_kw = [kw for kw in keywords if kw.lower() in full_text]
        return {
            "matched": len(matched_kw) >= 2,
            "partial": len(matched_kw) == 1,
            "detail": f"Nonfocal keyword: {matched_kw}",
        }

    # Token overlap as cheap semantic proxy
    desc_tokens = set(re.findall(r'\w+', semantic_desc.lower()))
    text_tokens = set(re.findall(r'\w+', full_text))

    if not desc_tokens:
        return {"matched": False}

    overlap = len(desc_tokens & text_tokens) / len(desc_tokens)

    # Context reinstatement bonus (encoding specificity)
    context_bonus = 0.0
    if context_at_creation:
        ctx_tokens = set(re.findall(r'\w+', context_at_creation.lower()))
        if ctx_tokens:
            ctx_overlap = len(ctx_tokens & text_tokens) / len(ctx_tokens)
            context_bonus = ctx_overlap * 0.15  # Up to 15% boost

    effective_overlap = overlap + context_bonus
    threshold = spec.get("semantic_threshold", 0.65)

    if effective_overlap >= threshold:
        detail = f"Semantic overlap: {overlap:.2f}"
        if context_bonus > 0:
            detail += f" + context: {context_bonus:.2f}"
        return {"matched": True, "detail": detail}
    if effective_overlap >= threshold * 0.6:
        return {"matched": False, "partial": True}
    return {"matched": False}


def _check_time(spec: dict, now: datetime) -> dict:
    """Check time-based trigger.

    Normalizes both trigger_time and now to naive-local (Colombia)
    before comparison, handling UTC and offset-aware inputs correctly.
    """
    trigger_time_str = spec.get("trigger_time", "")
    if not trigger_time_str:
        return {"matched": False}

    try:
        trigger_time = datetime.fromisoformat(trigger_time_str.replace("Z", "+00:00"))

        # Normalize to naive-local: if trigger has tzinfo, convert to COT (UTC-5)
        if trigger_time.tzinfo:
            from datetime import timezone
            cot_offset = timezone(timedelta(hours=-5))
            trigger_time = trigger_time.astimezone(cot_offset).replace(tzinfo=None)

        # Same for now: if aware, convert to COT naive
        now_cmp = now
        if now_cmp.tzinfo:
            from datetime import timezone
            cot_offset = timezone(timedelta(hours=-5))
            now_cmp = now_cmp.astimezone(cot_offset).replace(tzinfo=None)
        else:
            now_cmp = now  # Already naive-local (assumed COT)

    except Exception:
        return {"matched": False}

    tolerance = timedelta(minutes=spec.get("tolerance_minutes", 30))

    # Within window or overdue
    if now_cmp >= trigger_time - tolerance:
        return {"matched": True, "detail": f"Time: {trigger_time_str}"}
    return {"matched": False}


def _check_condition(spec: dict) -> dict:
    """Check condition-based triggers."""
    # Extensible: for now, basic count-based conditions
    cond_type = spec.get("condition_type", "")

    if cond_type == "manual":
        # Manual conditions are only triggered by explicit user/codi action
        return {"matched": False}

    return {"matched": False}


# ============================================================
# LIFECYCLE HELPERS
# ============================================================

def _handle_recurrence(conn: sqlite3.Connection, int_id: str, now_str: str):
    """If intention is recurring, create next instance after trigger.

    Supports: daily, weekly, monthly, custom_hours.
    (Einstein, McDaniel, Smith & Shaw, 1998: habitual PM)
    """
    row = conn.execute(
        "SELECT action, trigger_type, trigger_spec, priority, recurrence, "
        "recurrence_spec, context_at_creation, creator FROM intentions WHERE id = ?",
        (int_id,),
    ).fetchone()

    if not row or not row[4]:  # No recurrence
        return

    action, trigger_type, spec_raw, priority, recurrence, rec_spec_raw, ctx, creator = row

    try:
        spec = json.loads(spec_raw)
        rec_spec = json.loads(rec_spec_raw) if rec_spec_raw else {}
    except Exception:
        return

    # Calculate next trigger time based on recurrence type
    now_dt = datetime.fromisoformat(now_str) if isinstance(now_str, str) else now_str

    if recurrence == "daily":
        next_time = now_dt + timedelta(days=1)
    elif recurrence == "weekly":
        next_time = now_dt + timedelta(weeks=1)
    elif recurrence == "monthly":
        next_time = now_dt + timedelta(days=30)
    elif recurrence == "custom_hours":
        hours = rec_spec.get("hours", 24)
        next_time = now_dt + timedelta(hours=hours)
    else:
        return  # Unknown recurrence type

    # Update trigger_spec with next time for time-based intentions
    if trigger_type == "time":
        spec["trigger_time"] = next_time.isoformat()

    # Create new instance
    new_id = str(uuid.uuid4())
    activation = PM_ACTIVATION_INITIAL.get(priority, 0.65)

    conn.execute("""
        INSERT INTO intentions
        (id, action, trigger_type, trigger_spec, priority, activation,
         created_at, context_at_creation, creator, recurrence, recurrence_spec)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        new_id, action, trigger_type, json.dumps(spec), priority,
        activation, now_str, ctx, creator, recurrence, rec_spec_raw,
    ))
    _log_event(conn, new_id, "created_from_recurrence",
               json.dumps({"parent": int_id[:8], "next_time": next_time.isoformat()}))


def _mark_triggered(conn: sqlite3.Connection, int_id: str, detail: str, now_str: str):
    conn.execute("""
        UPDATE intentions SET status = 'triggered', triggered_at = ?
        WHERE id = ?
    """, (now_str, int_id))
    _log_event(conn, int_id, "triggered", detail)

    # Push to working memory so intention survives context compaction
    # (Kliegel, Martin, McDaniel & Einstein, 2002: intention execution phase)
    try:
        row = conn.execute(
            "SELECT action, priority FROM intentions WHERE id = ?", (int_id,)
        ).fetchone()
        if row:
            from modules.working_memory import push_to_working_memory
            marker = {"critical": "[!!!]", "high": "[!!]", "medium": "[!]", "low": "[.]"}.get(row[1], "[?]")
            push_to_working_memory(
                content=f"{marker} INTENCION DISPARADA: {row[0]}. {detail}",
                topic="intention_triggered",
                relevance=0.9,
                source="prospective_memory",
            )
            # Broadcast to global workspace (GWT competition)
            try:
                from modules.consciousness import update_workspace_spotlight
                update_workspace_spotlight(
                    memories=[f"[PM TRIGGERED] {row[0]}. {detail}"],
                    theme="intention_triggered",
                )
            except Exception:
                pass  # Don't block trigger on workspace failure
    except Exception:
        pass  # Don't block trigger on WM failure

    # Handle recurrence: create next instance if recurring
    # (Einstein, McDaniel, Smith & Shaw, 1998: habitual PM)
    _handle_recurrence(conn, int_id, now_str)


def _boost_activation(conn: sqlite3.Connection, int_id: str, amount: float):
    conn.execute("""
        UPDATE intentions
        SET activation = MIN(1.0, activation + ?),
            partial_match_count = partial_match_count + 1
        WHERE id = ?
    """, (amount, int_id))


def _expire_stale(conn: sqlite3.Connection, now: datetime):
    """Expire past-due intentions and Zeigarnik cleanup."""
    now_str = now.isoformat()

    # Explicit expiry
    conn.execute("""
        UPDATE intentions SET status = 'expired'
        WHERE status = 'pending' AND expiry IS NOT NULL AND expiry < ?
    """, (now_str,))

    # Stale cleanup: very old (>30d) + very weak (<lowest threshold) = expired
    cutoff = (now - timedelta(days=30)).isoformat()
    conn.execute("""
        UPDATE intentions SET status = 'expired'
        WHERE status = 'pending'
          AND activation < ?
          AND created_at < ?
    """, (min(PM_ACTIVATION_TRIGGER_THRESHOLD.values()), cutoff))


def _compute_activation(
    stored_activation: float,
    priority: str,
    hours_since_last_maintenance: float,
    partial_match_count: int = 0,
    expiry: datetime = None,
    now: datetime = None,
) -> float:
    """Compute current activation using cumulative power-law decay.

    The model applies decay as a MULTIPLIER on the stored (current) activation,
    not on the initial value. This makes decay cumulative across maintenance
    cycles and prevents the stationarity bug where medium/low intentions
    become permanently invisible.

    Components:
    1. Power-law decay multiplier since last maintenance
    2. Deadline urgency: monitoring intensification near deadline
       (Hicks, Marsh & Cook, 2005; Cona, Arcara, Tarantino & Bisiacchi, 2012)
    3. Rehearsal boost from partial matches (spreading activation)
    4. Stochastic noise for non-deterministic retrieval

    NOT claiming ACT-R compatibility -- purpose-built model for intentions.
    """
    # --- Power-law decay as multiplier on stored value ---
    h = max(0.01, hours_since_last_maintenance)

    # Priority modulates decay: critical decays 3x slower
    priority_d = {
        "critical": PM_DECAY_EXPONENT * 0.3,
        "high": PM_DECAY_EXPONENT * 0.6,
        "medium": PM_DECAY_EXPONENT,
        "low": PM_DECAY_EXPONENT * 1.5,
    }.get(priority, PM_DECAY_EXPONENT)

    # Power-law multiplier: h^(-d), clamped to [0, 1]
    decay_multiplier = min(1.0, h ** (-priority_d))
    decayed = stored_activation * decay_multiplier

    # --- Deadline urgency: monitoring intensification near deadline ---
    urgency_boost = 0.0
    if expiry and now:
        now_naive = now.replace(tzinfo=None) if now.tzinfo else now
        expiry_naive = expiry.replace(tzinfo=None) if expiry.tzinfo else expiry
        days_to_expiry = (expiry_naive - now_naive).total_seconds() / 86400
        if 0 < days_to_expiry < PM_URGENCY_WINDOW_DAYS:
            # Nonlinear ramp (quadratic) per Harris & Wilkins 1982
            ratio = 1.0 - days_to_expiry / PM_URGENCY_WINDOW_DAYS
            urgency_boost = 0.2 * (ratio ** 2)
        elif days_to_expiry <= 0:
            # Past deadline: strong urgency
            urgency_boost = 0.25

    # --- Rehearsal boost from partial matches ---
    rehearsal = 0.02 * min(partial_match_count, 10)

    # --- Stochastic noise ---
    noise = random.gauss(0, PM_ACTIVATION_NOISE_SIGMA)

    # --- Activation floor (BA10 tonic maintenance) ---
    floor = PM_ACTIVATION_FLOOR.get(priority, 0.21)

    result = max(floor, decayed) + urgency_boost + rehearsal + noise
    return max(0.0, min(1.0, result))


def apply_intention_maintenance():
    """Apply cumulative decay to pending intentions.

    Uses STORED activation as base (not initial), so decay compounds
    across maintenance cycles. Updates last_maintained_at to track
    the interval for next cycle's decay computation.
    """
    conn = _get_conn()
    now = now_col()
    now_naive = now.replace(tzinfo=None) if now.tzinfo else now

    rows = conn.execute("""
        SELECT id, activation, priority, created_at, last_maintained_at,
               partial_match_count, expiry
        FROM intentions WHERE status = 'pending'
    """).fetchall()

    for row in rows:
        int_id, activation, priority, created_str, maint_str, \
            partial_count, expiry_str = row

        try:
            created = datetime.fromisoformat(created_str)
            last_maint = datetime.fromisoformat(maint_str) if maint_str else None
            expiry = datetime.fromisoformat(expiry_str) if expiry_str else None
        except Exception:
            continue

        # Compute hours since last maintenance (or creation if never maintained)
        ref_time = last_maint or created
        ref_naive = ref_time.replace(tzinfo=None) if ref_time.tzinfo else ref_time
        hours_since = max(0.01, (now_naive - ref_naive).total_seconds() / 3600)

        new_act = _compute_activation(
            stored_activation=activation,  # Cumulative: use STORED value
            priority=priority,
            hours_since_last_maintenance=hours_since,
            partial_match_count=partial_count or 0,
            expiry=expiry,
            now=now,
        )

        conn.execute("""
            UPDATE intentions SET activation = ?, last_maintained_at = ?
            WHERE id = ?
        """, (round(new_act, 4), now.isoformat(), int_id))

    # Expire stale (using updated activations)
    _expire_stale(conn, now)
    conn.commit()

    stats = conn.execute("""
        SELECT status, COUNT(*) FROM intentions GROUP BY status
    """).fetchall()

    return {s: c for s, c in stats}


def _log_event(conn: sqlite3.Connection, int_id: str, event: str, detail: str = ""):
    conn.execute("""
        INSERT INTO intention_log (intention_id, event, detail, created_at)
        VALUES (?, ?, ?, ?)
    """, (int_id, event, detail, now_iso()))


# ============================================================
# MCP TOOL REGISTRATION
# ============================================================

def register_prospective_tools(mcp):
    """Register prospective memory MCP tools."""

    @mcp.tool()
    def crear_intencion(
        action: str,
        trigger_type: str = "event",
        trigger_spec: str = "{}",
        priority: str = "medium",
        expiry: str = "",
        context: str = "",
    ) -> str:
        """Create a prospective memory intention (remember to do something later).

        Args:
            action: What to do when triggered. Ex: "Recordarle a Hare renovar API key"
            trigger_type: event (when topic comes up) | time (at specific time) | condition
            trigger_spec: JSON string. For event: {"keywords": ["api", "key"], "match_threshold": 1}.
                         For time: {"trigger_time": "2026-02-15T10:00:00", "tolerance_minutes": 30}.
            priority: low|medium|high|critical
            expiry: ISO timestamp when intention expires (empty = no expiry)
            context: What's happening now (for context-dependent retrieval)
        """
        result = create_intention(
            action=action,
            trigger_type=trigger_type,
            trigger_spec=trigger_spec,
            priority=priority,
            expiry=expiry,
            context=context,
            creator="codi",
        )
        return json.dumps(result, indent=2, ensure_ascii=False)

    @mcp.tool()
    def ver_intenciones(limit: int = 10) -> str:
        """Show all pending prospective memory intentions.

        Shows what Codi is remembering to do in the future.
        """
        intentions = get_pending_intentions(limit)
        if not intentions:
            return "No pending intentions."

        lines = [f"## Pending Intentions ({len(intentions)})"]
        for i in intentions:
            priority_marker = {"critical": "[!!!]", "high": "[!!]", "medium": "[!]", "low": "[.]"}.get(i["priority"], "[?]")
            trigger_info = ""
            if i["trigger_type"] == "time":
                spec = i.get("trigger_spec", {})
                trigger_info = f" (at {spec.get('trigger_time', '?')})"
            elif i["trigger_type"] == "event":
                spec = i.get("trigger_spec", {})
                kw = spec.get("keywords", [])
                trigger_info = f" (keywords: {', '.join(kw)})" if kw else ""

            lines.append(
                f"- {priority_marker} [{i['id']}] {i['action']}"
                f" | {i['trigger_type']}{trigger_info}"
                f" | act={i['activation']} | by={i['creator']}"
            )
        return "\n".join(lines)

    @mcp.tool()
    def completar_intencion(intention_id: str, outcome: str = "") -> str:
        """Mark a prospective memory intention as completed.

        Args:
            intention_id: ID or prefix of the intention
            outcome: What happened when it was completed
        """
        result = complete_intention(intention_id, outcome)
        return json.dumps(result, ensure_ascii=False)

    @mcp.tool()
    def cancelar_intencion(intention_id: str, reason: str = "") -> str:
        """Cancel a prospective memory intention.

        Args:
            intention_id: ID or prefix of the intention
            reason: Why it's being cancelled
        """
        result = cancel_intention(intention_id, reason)
        return json.dumps(result, ensure_ascii=False)

    @mcp.tool()
    def posponer_intencion(intention_id: str, hours: float = 24.0) -> str:
        """Snooze a prospective memory intention for N hours.

        Args:
            intention_id: ID or prefix of the intention
            hours: Hours to snooze (default 24)
        """
        result = snooze_intention(intention_id, hours)
        return json.dumps(result, ensure_ascii=False)

    @mcp.tool()
    def mantenimiento_intenciones() -> str:
        """Run maintenance on prospective memory: decay, expire, cleanup."""
        stats = apply_intention_maintenance()
        return json.dumps(stats, ensure_ascii=False)
