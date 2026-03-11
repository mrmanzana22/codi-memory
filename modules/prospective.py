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
Updated: 2026-03-05 (Fase 5 - migrated from SQLite prospective.db to PostgreSQL)
"""

import logging
import re
import json
import uuid
import time
import random
import math
from datetime import datetime, timedelta

_logger = logging.getLogger(__name__)

from psycopg.rows import dict_row

from modules.config import now_iso, now_col
from modules.config_pg import get_conn

# ============================================================
# CONSTANTS
# ============================================================

PM_CHECK_BUDGET_MS = 50

PM_ACTIVATION_INITIAL = {
    "critical": 0.95,
    "high": 0.80,
    "medium": 0.65,
    "low": 0.45,
}
PM_DECAY_EXPONENT = 0.5
PM_ACTIVATION_PARTIAL_BOOST = 0.05
PM_ACTIVATION_NOISE_SIGMA = 0.03
PM_URGENCY_WINDOW_DAYS = 7
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
                CREATE TABLE IF NOT EXISTS intentions (
                    id                  TEXT PRIMARY KEY,
                    action              TEXT NOT NULL,
                    action_type         TEXT DEFAULT 'remind',
                    trigger_type        TEXT NOT NULL,
                    trigger_spec        JSONB NOT NULL DEFAULT '{}',
                    cue_focality        TEXT DEFAULT 'focal',
                    priority            TEXT DEFAULT 'medium',
                    status              TEXT DEFAULT 'pending',
                    activation          REAL DEFAULT 0.7,
                    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    triggered_at        TIMESTAMPTZ,
                    completed_at        TIMESTAMPTZ,
                    expiry              TIMESTAMPTZ,
                    snooze_until        TIMESTAMPTZ,
                    context_at_creation TEXT,
                    creator             TEXT DEFAULT 'codi',
                    recurrence          TEXT,
                    recurrence_spec     JSONB,
                    check_count         INTEGER DEFAULT 0,
                    partial_match_count INTEGER DEFAULT 0,
                    last_checked_at     TIMESTAMPTZ,
                    last_maintained_at  TIMESTAMPTZ,
                    goal_id             TEXT REFERENCES goals(id) ON DELETE SET NULL
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS intention_log (
                    id           SERIAL PRIMARY KEY,
                    intention_id TEXT NOT NULL REFERENCES intentions(id) ON DELETE CASCADE,
                    event        TEXT NOT NULL,
                    detail       TEXT,
                    created_at   TIMESTAMPTZ NOT NULL DEFAULT NOW()
                )
            """)
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_intentions_status "
                "ON intentions(status)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_intentions_activation "
                "ON intentions(activation DESC)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_intentions_expiry "
                "ON intentions(expiry)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_intentions_pending "
                "ON intentions(status, activation DESC) "
                "WHERE status = 'pending'"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_intention_log_intention "
                "ON intention_log(intention_id)"
            )
        _TABLES_INITIALIZED = True
        _logger.info("Intentions PG tables ready")
    except Exception as e:
        _logger.error("Failed to initialize intentions PG tables: %s", e)
        raise


# ============================================================
# HELPERS
# ============================================================

def _ts(val) -> str:
    """Convert TIMESTAMPTZ value (datetime or None) to ISO string."""
    if val is None:
        return None
    if hasattr(val, "isoformat"):
        return val.isoformat()
    return str(val)


def _as_spec(val) -> dict:
    """Normalize trigger_spec/recurrence_spec: JSONB returns dict, TEXT returns parsed."""
    if val is None:
        return {}
    if isinstance(val, dict):
        return val
    try:
        return json.loads(val)
    except Exception:
        return {}


def _log_event(conn, int_id: str, event: str, detail: str = ""):
    """Write to intention_log audit trail (call inside a transaction)."""
    conn.execute(
        "INSERT INTO intention_log (intention_id, event, detail) "
        "VALUES (%s, %s, %s)",
        (int_id, event, detail),
    )


def _resolve_id(conn, prefix: str) -> str:
    """Resolve a partial ID to full ID."""
    if len(prefix) >= 36:
        return prefix
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            "SELECT id FROM intentions WHERE id::text LIKE %s LIMIT 2",
            (f"{prefix}%",),
        )
        rows = cur.fetchall()
    if len(rows) == 1:
        return str(rows[0]["id"])
    if len(rows) == 0:
        raise ValueError(f"No intention found with prefix '{prefix}'")
    raise ValueError(f"Ambiguous prefix '{prefix}', matches {len(rows)} intentions")


# ============================================================
# INTENTION CRUD
# ============================================================

def create_intention(
    action: str,
    trigger_type: str = "event",
    trigger_spec: str = "{}",
    priority: str = "medium",
    expiry: str = "",
    context: str = "",
    creator: str = "codi",
    recurrence: str = "",
    goal_id: str = "",
) -> dict:
    """Create a new prospective memory intention."""
    _ensure_tables()
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
            activation = min(1.0, activation + arousal * 0.1)
    except Exception:
        pass

    with get_conn() as conn:
        with conn.transaction():
            conn.execute("""
                INSERT INTO intentions
                (id, action, trigger_type, trigger_spec, cue_focality, priority,
                 activation, expiry, context_at_creation, creator, recurrence,
                 goal_id)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """, (
                int_id, action, trigger_type, json.dumps(spec), cue_focality,
                priority, activation,
                expiry or None, json.dumps(context) if isinstance(context, str) else context, creator,
                recurrence or None, goal_id or None,
            ))

            _log_event(conn, int_id, "created", json.dumps({
                "action": action, "trigger_type": trigger_type, "priority": priority,
            }))

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
    _ensure_tables()
    with get_conn() as conn:
        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute("""
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
                LIMIT %s
            """, (limit,))
            rows = cur.fetchall()

    return [
        {
            "id": str(r["id"])[:8],
            "full_id": r["id"],
            "action": r["action"],
            "action_type": r["action_type"],
            "trigger_type": r["trigger_type"],
            "trigger_spec": _as_spec(r["trigger_spec"]),
            "priority": r["priority"],
            "activation": round(r["activation"], 2),
            "created_at": _ts(r["created_at"]),
            "expiry": _ts(r["expiry"]),
            "status": r["status"],
            "creator": r["creator"],
        }
        for r in rows
    ]


def complete_intention(intention_id: str, outcome: str = "") -> dict:
    """Mark an intention as completed."""
    _ensure_tables()
    with get_conn() as conn:
        full_id = _resolve_id(conn, intention_id)
        with conn.transaction():
            conn.execute(
                "UPDATE intentions SET status = 'completed', completed_at = NOW() "
                "WHERE id = %s",
                (full_id,),
            )
            _log_event(conn, full_id, "completed", outcome)
    return {"id": full_id[:8], "status": "completed", "outcome": outcome}


def cancel_intention(intention_id: str, reason: str = "") -> dict:
    """Cancel a pending intention."""
    _ensure_tables()
    with get_conn() as conn:
        full_id = _resolve_id(conn, intention_id)
        with conn.transaction():
            conn.execute(
                "UPDATE intentions SET status = 'cancelled' WHERE id = %s",
                (full_id,),
            )
            _log_event(conn, full_id, "cancelled", reason)
    return {"id": full_id[:8], "status": "cancelled", "reason": reason}


def snooze_intention(intention_id: str, hours: float = 24.0) -> dict:
    """Snooze an intention for N hours."""
    _ensure_tables()
    snooze_until = (now_col() + timedelta(hours=hours)).isoformat()
    with get_conn() as conn:
        full_id = _resolve_id(conn, intention_id)
        with conn.transaction():
            conn.execute(
                "UPDATE intentions SET snooze_until = %s, status = 'pending' "
                "WHERE id = %s",
                (snooze_until, full_id),
            )
            _log_event(conn, full_id, "snoozed", f"until {snooze_until}")
    return {"id": full_id[:8], "snoozed_until": snooze_until}


def check_intention_exists(context_pattern: str) -> bool:
    """Check if a pending intention with matching context exists (for dedup).

    Public API used by pe_actions.py and others.
    """
    _ensure_tables()
    try:
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT id FROM intentions "
                    "WHERE status = 'pending' AND context_at_creation::text LIKE %s "
                    "LIMIT 1",
                    (f"%{context_pattern}%",),
                )
                row = cur.fetchone()
        return row is not None
    except Exception:
        return False


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
    _ensure_tables()
    start_ms = time.time() * 1000
    triggered = []

    now = now_col()

    with get_conn() as conn:
        with conn.transaction():
            # Expire stale intentions first
            _expire_stale(conn, now)

            # Load active intentions
            with conn.cursor(row_factory=dict_row) as cur:
                cur.execute("""
                    SELECT id, action, action_type, trigger_type, trigger_spec,
                           cue_focality, priority, activation, context_at_creation
                    FROM intentions
                    WHERE status = 'pending'
                      AND activation > %s
                      AND (snooze_until IS NULL OR snooze_until <= %s)
                    ORDER BY activation DESC
                    LIMIT %s
                """, (min(PM_ACTIVATION_TRIGGER_THRESHOLD.values()), now, PM_MAX_ACTIVE_INTENTIONS))
                rows = cur.fetchall()

            if not rows:
                return []

            full_text = f"{prompt} {current_context}".lower()

            for row in rows:
                int_id = row["id"]
                action = row["action"]
                action_type = row["action_type"]
                trigger_type = row["trigger_type"]
                spec = _as_spec(row["trigger_spec"])
                cue_focality = row["cue_focality"]
                priority = row["priority"]
                activation = row["activation"]
                ctx_at_creation = row["context_at_creation"] or ""

                # Budget check
                elapsed = time.time() * 1000 - start_ms
                if elapsed > PM_CHECK_BUDGET_MS:
                    break

                # Priority-aware activation threshold
                threshold = PM_ACTIVATION_TRIGGER_THRESHOLD.get(
                    priority, PM_ACTIVATION_TRIGGER_THRESHOLD_DEFAULT
                )
                if activation < threshold:
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
                    _mark_triggered(conn, int_id, match_result.get("detail", ""), now)
                elif match_result and match_result.get("partial"):
                    _boost_activation(conn, int_id, PM_ACTIVATION_PARTIAL_BOOST)

                # Increment check counter
                conn.execute(
                    "UPDATE intentions SET check_count = check_count + 1, "
                    "last_checked_at = NOW() WHERE id = %s",
                    (int_id,),
                )

    # Rank and cap at 3 (workspace capacity per GWT)
    if len(triggered) > 3:
        priority_w = {"critical": 1.0, "high": 0.8, "medium": 0.5, "low": 0.3}
        triggered.sort(
            key=lambda t: -(priority_w.get(t["priority"], 0.5) * t["activation"])
        )
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
            context_bonus = ctx_overlap * 0.15

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

        if trigger_time.tzinfo:
            from datetime import timezone
            cot_offset = timezone(timedelta(hours=-5))
            trigger_time = trigger_time.astimezone(cot_offset).replace(tzinfo=None)

        now_cmp = now
        if now_cmp.tzinfo:
            from datetime import timezone
            cot_offset = timezone(timedelta(hours=-5))
            now_cmp = now_cmp.astimezone(cot_offset).replace(tzinfo=None)

    except Exception:
        return {"matched": False}

    tolerance = timedelta(minutes=spec.get("tolerance_minutes", 30))

    if now_cmp >= trigger_time - tolerance:
        return {"matched": True, "detail": f"Time: {trigger_time_str}"}
    return {"matched": False}


def _check_condition(spec: dict) -> dict:
    """Check condition-based triggers."""
    cond_type = spec.get("condition_type", "")

    if cond_type == "manual":
        return {"matched": False}

    return {"matched": False}


# ============================================================
# LIFECYCLE HELPERS
# ============================================================

def _handle_recurrence(conn, int_id: str, now: datetime):
    """If intention is recurring, create next instance after trigger."""
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            "SELECT action, trigger_type, trigger_spec, priority, recurrence, "
            "recurrence_spec, context_at_creation, creator FROM intentions WHERE id = %s",
            (int_id,),
        )
        row = cur.fetchone()

    if not row or not row["recurrence"]:
        return

    action = row["action"]
    trigger_type = row["trigger_type"]
    spec = _as_spec(row["trigger_spec"])
    priority = row["priority"]
    recurrence = row["recurrence"]
    rec_spec = _as_spec(row["recurrence_spec"])
    ctx = row["context_at_creation"]
    creator = row["creator"]

    now_naive = now.replace(tzinfo=None) if now.tzinfo else now

    if recurrence == "daily":
        next_time = now_naive + timedelta(days=1)
    elif recurrence == "weekly":
        next_time = now_naive + timedelta(weeks=1)
    elif recurrence == "monthly":
        next_time = now_naive + timedelta(days=30)
    elif recurrence == "custom_hours":
        hours = rec_spec.get("hours", 24)
        next_time = now_naive + timedelta(hours=hours)
    else:
        return

    if trigger_type == "time":
        spec["trigger_time"] = next_time.isoformat()

    new_id = str(uuid.uuid4())
    activation = PM_ACTIVATION_INITIAL.get(priority, 0.65)

    conn.execute("""
        INSERT INTO intentions
        (id, action, trigger_type, trigger_spec, priority, activation,
         context_at_creation, creator, recurrence, recurrence_spec)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    """, (
        new_id, action, trigger_type, json.dumps(spec), priority,
        activation, ctx, creator, recurrence,
        json.dumps(rec_spec) if rec_spec else None,
    ))
    _log_event(conn, new_id, "created_from_recurrence",
               json.dumps({"parent": int_id[:8], "next_time": next_time.isoformat()}))


def _mark_triggered(conn, int_id: str, detail: str, now: datetime):
    conn.execute(
        "UPDATE intentions SET status = 'triggered', triggered_at = NOW() "
        "WHERE id = %s",
        (int_id,),
    )
    _log_event(conn, int_id, "triggered", detail)

    # Push to working memory (Kliegel et al., 2002)
    try:
        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                "SELECT action, priority FROM intentions WHERE id = %s", (int_id,)
            )
            row = cur.fetchone()
        if row:
            from modules.working_memory import push_to_working_memory
            marker = {"critical": "[!!!]", "high": "[!!]", "medium": "[!]", "low": "[.]"}.get(
                row["priority"], "[?]"
            )
            push_to_working_memory(
                content=f"{marker} INTENCION DISPARADA: {row['action']}. {detail}",
                topic="intention_triggered",
                relevance=0.9,
                source="prospective_memory",
            )
            try:
                from modules.consciousness import update_workspace_spotlight
                update_workspace_spotlight(
                    memories=[f"[PM TRIGGERED] {row['action']}. {detail}"],
                    theme="intention_triggered",
                )
            except Exception:
                pass
    except Exception:
        pass

    _handle_recurrence(conn, int_id, now)


def _boost_activation(conn, int_id: str, amount: float):
    conn.execute(
        "UPDATE intentions "
        "SET activation = LEAST(1.0, activation + %s), "
        "partial_match_count = partial_match_count + 1 "
        "WHERE id = %s",
        (amount, int_id),
    )


def _expire_stale(conn, now: datetime):
    """Expire past-due intentions and Zeigarnik cleanup."""
    # Explicit expiry
    conn.execute(
        "UPDATE intentions SET status = 'expired' "
        "WHERE status = 'pending' AND expiry IS NOT NULL AND expiry < %s",
        (now,),
    )

    # Stale cleanup: very old (>30d) + very weak (<lowest threshold)
    cutoff = now - timedelta(days=30)
    conn.execute(
        "UPDATE intentions SET status = 'expired' "
        "WHERE status = 'pending' "
        "AND activation < %s "
        "AND created_at < %s",
        (min(PM_ACTIVATION_TRIGGER_THRESHOLD.values()), cutoff),
    )


def _compute_activation(
    stored_activation: float,
    priority: str,
    hours_since_last_maintenance: float,
    partial_match_count: int = 0,
    expiry: datetime = None,
    now: datetime = None,
) -> float:
    """Compute current activation using cumulative power-law decay."""
    h = max(0.01, hours_since_last_maintenance)

    priority_d = {
        "critical": PM_DECAY_EXPONENT * 0.3,
        "high": PM_DECAY_EXPONENT * 0.6,
        "medium": PM_DECAY_EXPONENT,
        "low": PM_DECAY_EXPONENT * 1.5,
    }.get(priority, PM_DECAY_EXPONENT)

    decay_multiplier = min(1.0, h ** (-priority_d))
    decayed = stored_activation * decay_multiplier

    urgency_boost = 0.0
    if expiry and now:
        now_naive = now.replace(tzinfo=None) if now.tzinfo else now
        expiry_naive = expiry.replace(tzinfo=None) if expiry.tzinfo else expiry
        days_to_expiry = (expiry_naive - now_naive).total_seconds() / 86400
        if 0 < days_to_expiry < PM_URGENCY_WINDOW_DAYS:
            ratio = 1.0 - days_to_expiry / PM_URGENCY_WINDOW_DAYS
            urgency_boost = 0.2 * (ratio ** 2)
        elif days_to_expiry <= 0:
            urgency_boost = 0.25

    rehearsal = 0.02 * min(partial_match_count, 10)
    noise = random.gauss(0, PM_ACTIVATION_NOISE_SIGMA)
    floor = PM_ACTIVATION_FLOOR.get(priority, 0.21)

    result = max(floor, decayed) + urgency_boost + rehearsal + noise
    return max(0.0, min(1.0, result))


def apply_intention_maintenance():
    """Apply cumulative decay to pending intentions."""
    _ensure_tables()
    now = now_col()
    now_naive = now.replace(tzinfo=None) if now.tzinfo else now

    with get_conn() as conn:
        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute("""
                SELECT id, activation, priority, created_at, last_maintained_at,
                       partial_match_count, expiry
                FROM intentions WHERE status = 'pending'
            """)
            rows = cur.fetchall()

        with conn.transaction():
            updates = []
            for row in rows:
                int_id = row["id"]
                activation = row["activation"]
                priority = row["priority"]
                created = row["created_at"]       # datetime from PG
                last_maint = row["last_maintained_at"]  # datetime or None
                expiry = row["expiry"]            # datetime or None
                partial_count = row["partial_match_count"] or 0

                if created is None:
                    continue

                # Compute hours since last maintenance
                ref_time = last_maint or created
                ref_naive = ref_time.replace(tzinfo=None) if ref_time.tzinfo else ref_time
                hours_since = max(0.01, (now_naive - ref_naive).total_seconds() / 3600)

                new_act = _compute_activation(
                    stored_activation=activation,
                    priority=priority,
                    hours_since_last_maintenance=hours_since,
                    partial_match_count=partial_count,
                    expiry=expiry,
                    now=now,
                )

                updates.append((round(new_act, 4), int_id))

            # Batch UPDATE: 1 round trip instead of N (psycopg3 executemany)
            if updates:
                with conn.cursor() as cur:
                    cur.executemany(
                        "UPDATE intentions SET activation = %s, last_maintained_at = NOW() "
                        "WHERE id = %s",
                        updates,
                    )

            # Expire stale (using updated activations)
            _expire_stale(conn, now)

        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute("SELECT status, COUNT(*) as cnt FROM intentions GROUP BY status")
            stats = cur.fetchall()

    return {r["status"]: r["cnt"] for r in stats}


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
        goal_id: str = "",
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
            goal_id: Link to a goal in the goal system (optional)
        """
        result = create_intention(
            action=action,
            trigger_type=trigger_type,
            trigger_spec=trigger_spec,
            priority=priority,
            expiry=expiry,
            context=context,
            creator="codi",
            goal_id=goal_id,
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
            priority_marker = {
                "critical": "[!!!]", "high": "[!!]", "medium": "[!]", "low": "[.]"
            }.get(i["priority"], "[?]")
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
        return json.dumps(stats, indent=2)
