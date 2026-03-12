"""
Codi Memory - Session Bridge Module
====================================
Cross-session continuity via structured checkpoints.

Two functions:
  checkpoint_session_close() - captures minimal sufficient state at session end
  load_session_bridge()      - loads checkpoint, calculates PEs, generates narrative bridge

Neuroscience basis: hippocampal replay + autobiographical self + pattern completion.
The brain doesn't persist continuity literally -- it fabricates an illusion from
checkpoints, replay, prediction errors, and narrative. We do the same.

Persistence: SQLite (memories_fts.db), table session_checkpoints.
Created: 2026-02-15 (Session Bridge v1)
"""

import json
import logging
import os
import sqlite3
from datetime import datetime

from modules.config import (
    FTS_DB_PATH, _emotional_state, _current_session,
    now_col, now_iso, TZ_COL, connect_fts, parse_timestamp,
)
from modules.tracing import get_trace_id
from modules.events import event_bus, Events
from modules.secret_redact import redact_secrets

_logger = logging.getLogger(__name__)

# ============================================================
# CONSTANTS
# ============================================================

DEDUPE_WINDOW_SECONDS = 120   # No checkpoint if one exists within 2 min
RETENTION_MAX = 50            # Keep only last 50 checkpoints
BRIDGE_MAX_CHARS = 500        # Cap for narrative bridge text
FRESHNESS_MAX_HOURS = 24      # Ignore checkpoints older than this
REMINDERS_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), "recordatorios_pendientes.json")


# ============================================================
# DATABASE HELPERS
# ============================================================

def _get_conn():
    """Get a WAL-mode SQLite connection to FTS DB."""
    conn = connect_fts(FTS_DB_PATH)
    conn.execute("PRAGMA synchronous=NORMAL")
    return conn


def _hours_since(iso_str: str) -> float:
    """Hours elapsed since an ISO timestamp."""
    try:
        dt = datetime.fromisoformat(iso_str)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=TZ_COL)
        delta = now_col() - dt
        return max(0.0, delta.total_seconds() / 3600)
    except Exception:
        return 999.0


# ============================================================
# CHECKPOINT_SESSION_CLOSE
# ============================================================

def checkpoint_session_close(
    session_summary: str = "",
    decisions: str = "",
    goal_stack: list = None,
    source: str = "flush",
) -> dict:
    """Capture minimal sufficient state for cross-session continuity.

    Args:
        session_summary: Short summary of what happened this session (max 500 chars)
        decisions: Key decisions made (free text or JSON)
        goal_stack: List of dicts [{goal, status, next_step}]
        source: 'flush' or 'hook' (flush has higher priority)

    Returns:
        dict with ok=True/False, checkpoint_id, deduped=True/False
    """
    if goal_stack is None:
        goal_stack = []

    now = now_iso()

    try:
        conn = _get_conn()
        # --- DEDUPE WINDOW ---
        existing = conn.execute(
            "SELECT id, source, created_at FROM session_checkpoints "
            "ORDER BY created_at DESC LIMIT 1"
        ).fetchone()

        if existing:
            existing_id, existing_source, existing_ts = existing
            try:
                ex_dt = datetime.fromisoformat(existing_ts)
                if ex_dt.tzinfo is None:
                    ex_dt = ex_dt.replace(tzinfo=TZ_COL)
                age_sec = (now_col() - ex_dt).total_seconds()
            except Exception:
                age_sec = 9999

            if age_sec < DEDUPE_WINDOW_SECONDS:
                # Source priority: flush > hook
                source_priority = {"flush": 2, "hook": 1}
                new_prio = source_priority.get(source, 0)
                old_prio = source_priority.get(existing_source, 0)

                if new_prio > old_prio:
                    # Upgrade: flush overwrites hook -- collect data and UPDATE
                    pass  # Fall through to data collection, will UPDATE instead
                else:
                    # Same or lower priority: skip
                    conn.close()
                    return {"ok": False, "checkpoint_id": existing_id, "deduped": True}

        # --- COLLECT DATA FROM 7+ SOURCES ---

        # 1. Goal stack + active project
        active_project = None
        wm_topics = []
        try:
            from modules.working_memory import wm_get_active_topics
            topics = wm_get_active_topics(limit=10)
            for topic in topics:
                if topic and topic not in wm_topics:
                    wm_topics.append(topic)
                if not active_project and topic not in ('metamemory', 'contradiction', 'general', ''):
                    active_project = topic
        except Exception:
            _logger.debug("failed to collect active topics from working memory", exc_info=True)

        # 2. Attention schema
        attention_focus = None
        attention_strength = 0.0
        try:
            from modules.wiring import _attention_schema
            attention_focus = _attention_schema.get("current_focus")
            attention_strength = _attention_schema.get("focus_strength", 0.0)
        except Exception:
            _logger.debug("failed to read attention schema", exc_info=True)

        # 3. Prediction state (from preturn_inject SQLite tables)
        last_pred_topic = None
        last_pred_confidence = None
        try:
            row = conn.execute(
                "SELECT predicted_topic, confidence FROM prediction_state WHERE id=1"
            ).fetchone()
            if row:
                last_pred_topic = row[0]
                last_pred_confidence = row[1]
        except Exception:
            _logger.debug("failed to read prediction state", exc_info=True)

        # 4. Recent prediction errors (last 3)
        recent_pes = []
        try:
            pe_rows = conn.execute(
                "SELECT actual_topic, predicted_topic, surprise_score, created_at "
                "FROM prediction_results ORDER BY id DESC LIMIT 3"
            ).fetchall()
            for r in pe_rows:
                recent_pes.append({
                    "actual": r[0], "predicted": r[1],
                    "surprise": r[2], "ts": r[3]
                })
        except Exception:
            _logger.debug("failed to read recent prediction errors", exc_info=True)

        # 5. Working memory: top 15 items by effective_score
        wm_items = []
        wm_active_count = 0
        try:
            from modules.working_memory import wm_get_active_count, wm_get_active_items
            wm_active_count = wm_get_active_count()
            wm_items = wm_get_active_items(limit=15)
        except Exception:
            _logger.debug("failed to read working memory items", exc_info=True)

        # 6. PAD emotional state
        pad = _emotional_state.get('current', {})
        pad_p = pad.get('pleasure', 0.0)
        pad_a = pad.get('arousal', 0.0)
        pad_d = pad.get('dominance', 0.0)
        pad_trigger = pad.get('trigger', '')

        # 7. Pending intentions
        pending_intentions = []
        try:
            from modules.prospective import get_pending_intentions
            raw = get_pending_intentions(limit=10)
            for intent in raw:
                pending_intentions.append({
                    "id": intent.get("id", ""),
                    "action": (intent.get("action", ""))[:150],
                    "priority": intent.get("priority", "medium"),
                    "activation": intent.get("activation", 0.0),
                    "expiry": intent.get("expiry"),
                })
        except Exception:
            _logger.debug("failed to collect pending intentions", exc_info=True)

        # 8. Active narrative traces
        active_traces = []
        try:
            from modules.working_memory import wm_get_active_traces
            active_traces = wm_get_active_traces(limit=10)
        except Exception:
            _logger.debug("failed to read active narrative traces", exc_info=True)

        # 9. Decisions: param + WM decision items
        decision_list = []
        if decisions:
            decision_list.append({"decision": decisions[:300], "why": "session_param"})
        try:
            from modules.working_memory import wm_get_decision_items
            decision_list.extend(wm_get_decision_items(limit=5))
        except Exception:
            _logger.debug("failed to collect decision items from working memory", exc_info=True)

        # --- BUILD ROW DATA ---
        trace_id = get_trace_id() or ""
        session_id = _current_session or ""

        row_data = {
            "trace_id": trace_id,
            "session_id": session_id,
            "source": source,
            "goal_stack": json.dumps(goal_stack, ensure_ascii=False),
            "active_project": active_project,
            "session_summary": (session_summary or "")[:500],
            "attention_focus": attention_focus,
            "attention_strength": attention_strength,
            "last_prediction_topic": last_pred_topic,
            "last_prediction_confidence": last_pred_confidence,
            "recent_prediction_errors": json.dumps(recent_pes, ensure_ascii=False),
            "wm_items": json.dumps(wm_items, ensure_ascii=False),
            "wm_active_count": wm_active_count,
            "pad_pleasure": pad_p,
            "pad_arousal": pad_a,
            "pad_dominance": pad_d,
            "pad_trigger": pad_trigger,
            "last_decisions": json.dumps(decision_list, ensure_ascii=False),
            "pending_intentions": json.dumps(pending_intentions, ensure_ascii=False),
            "active_traces": json.dumps(active_traces, ensure_ascii=False),
            "session_duration_minutes": None,
            "created_at": now,
            "sleep_report": None,
        }

        # --- INSERT OR UPDATE ---
        # Check if we should update an existing dedupe-window record
        should_update = (
            existing
            and age_sec < DEDUPE_WINDOW_SECONDS  # type: ignore[possibly-undefined]
            and source == "flush"
            and existing_source == "hook"  # type: ignore[possibly-undefined]
        )

        if should_update:
            _skip_on_update = {"created_at", "sleep_report"}
            cols = [f"{k} = ?" for k in row_data.keys() if k not in _skip_on_update]
            vals = [v for k, v in row_data.items() if k not in _skip_on_update]
            vals.append(existing_id)  # type: ignore[possibly-undefined]
            conn.execute(
                f"UPDATE session_checkpoints SET {', '.join(cols)} WHERE id = ?",
                vals
            )
            conn.commit()
            checkpoint_id = existing_id  # type: ignore[possibly-undefined]
        else:
            cols = list(row_data.keys())
            placeholders = ", ".join(["?"] * len(cols))
            conn.execute(
                f"INSERT INTO session_checkpoints ({', '.join(cols)}) VALUES ({placeholders})",
                list(row_data.values())
            )
            conn.commit()
            checkpoint_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]

        # --- RETENTION: keep only last N ---
        conn.execute(
            "DELETE FROM session_checkpoints WHERE id NOT IN "
            "(SELECT id FROM session_checkpoints ORDER BY created_at DESC LIMIT ?)",
            (RETENTION_MAX,)
        )
        conn.commit()

        # --- FALLBACK: also save JSON state ---
        try:
            from modules.flush import _save_session_state
            _save_session_state(session_summary)
        except Exception:
            _logger.debug("failed to save fallback JSON session state", exc_info=True)

        # --- EMIT EVENT ---
        event_bus.emit(Events.SESSION_CLOSE, {
            "checkpoint_id": checkpoint_id,
            "source": source,
            "reason": source,
            "active_project": active_project,
        })

        conn.close()
        return {"ok": True, "checkpoint_id": checkpoint_id, "deduped": False}

    except Exception as e:
        try:
            conn.close()
        except Exception:
            _logger.debug("failed to close db connection during error cleanup", exc_info=True)
        return {"ok": False, "error": redact_secrets(str(e)), "checkpoint_id": None}


# ============================================================
# LOAD_SESSION_BRIDGE
# ============================================================

def load_session_bridge() -> dict | None:
    """Load last checkpoint and generate narrative bridge for wake-up.

    Returns dict with:
        bridge_text: str (max 500 chars)
        checkpoint: dict (raw row data)
        prediction_errors: list (deterministic, no invention)
        hours_since_last: float

    Returns None if no checkpoint or too stale (>24h).
    """
    try:
        conn = _get_conn()
        row = conn.execute(
            "SELECT * FROM session_checkpoints ORDER BY created_at DESC LIMIT 1"
        ).fetchone()

        if not row:
            conn.close()
            return None

        # Get column names
        col_names = [desc[0] for desc in conn.execute(
            "SELECT * FROM session_checkpoints LIMIT 0"
        ).description]
        conn.close()

        checkpoint = dict(zip(col_names, row))

        # --- FRESHNESS CHECK ---
        hours = _hours_since(checkpoint.get("created_at", ""))
        if hours > FRESHNESS_MAX_HOURS:
            return None

        # --- CALCULATE PREDICTION ERRORS (deterministic, no LLM, no invention) ---
        prediction_errors = []

        # PE1: Expired intentions
        try:
            pending = json.loads(checkpoint.get("pending_intentions") or "[]")
            now = now_col()
            for intent in pending:
                expiry = intent.get("expiry")
                if expiry:
                    try:
                        exp_dt = datetime.fromisoformat(expiry)
                        if exp_dt.tzinfo is None:
                            exp_dt = exp_dt.replace(tzinfo=TZ_COL)
                        if now > exp_dt:
                            prediction_errors.append({
                                "type": "intention_expired",
                                "detail": f"Intencion expirada: {intent.get('action', '?')[:80]}",
                                "severity": "medium",
                            })
                    except Exception:
                        _logger.debug("failed to parse intention expiry timestamp", exc_info=True)
        except Exception:
            _logger.debug("failed to check expired intentions", exc_info=True)

        # PE2: External events (recordatorios_externos.json newer than checkpoint)
        try:
            rec_file = REMINDERS_FILE
            if os.path.exists(rec_file):
                with open(rec_file, 'r', encoding='utf-8') as f:
                    rec_data = json.load(f)
                pendientes = rec_data.get('pendientes', [])
                checkpoint_ts = checkpoint.get("created_at", "")
                for rec in pendientes:
                    rec_ts = rec.get("timestamp", "")
                    if rec_ts and checkpoint_ts:
                        try:
                            rec_dt = parse_timestamp(rec_ts)
                            checkpoint_dt = parse_timestamp(checkpoint_ts)
                            if rec_dt > checkpoint_dt:
                                prediction_errors.append({
                                    "type": "external_event",
                                    "detail": f"Recordatorio: {rec.get('mensaje', '?')[:80]}",
                                    "severity": rec.get("prioridad", "normal"),
                                })
                        except Exception:
                            _logger.debug("failed to parse reminder/checkpoint timestamps for comparison", exc_info=True)
        except Exception:
            _logger.debug("failed to check external reminders file", exc_info=True)

        # --- GENERATE NARRATIVE BRIDGE ---
        parts = []

        summary = checkpoint.get("session_summary", "")
        if summary:
            parts.append(f"Ultima sesion: {summary[:150]}")

        project = checkpoint.get("active_project", "")
        if project:
            parts.append(f"Proyecto activo: {project}")

        parts.append(f"Han pasado {hours:.1f} horas")

        # PEs
        for pe in prediction_errors[:3]:
            parts.append(f"[PE] {pe['detail'][:100]}")

        # Sleep report (if background maintenance ran)
        sleep_report = checkpoint.get("sleep_report", "")
        if sleep_report and sleep_report.strip():
            parts.append(f"[Sleep] {sleep_report.strip()[:120]}")

        # High/critical pending intentions (max 3)
        try:
            pending = json.loads(checkpoint.get("pending_intentions") or "[]")
            high_intents = [
                i for i in pending
                if i.get("priority") in ("high", "critical")
            ][:3]
            for intent in high_intents:
                parts.append(f"Pendiente [{intent['priority']}]: {intent.get('action', '?')[:80]}")
        except Exception:
            _logger.debug("failed to parse high-priority pending intentions", exc_info=True)

        # Next step from goal stack
        try:
            goals = json.loads(checkpoint.get("goal_stack") or "[]")
            if goals and goals[0].get("next_step"):
                parts.append(f"Siguiente paso: {goals[0]['next_step'][:100]}")
        except Exception:
            _logger.debug("failed to parse goal stack for next step", exc_info=True)

        bridge_text = "\n".join(parts)[:BRIDGE_MAX_CHARS]

        return {
            "bridge_text": bridge_text,
            "checkpoint": checkpoint,
            "prediction_errors": prediction_errors,
            "hours_since_last": round(hours, 2),
            "sleep_report": sleep_report if sleep_report and sleep_report.strip() else None,
        }

    except Exception:
        _logger.debug("failed to load session bridge", exc_info=True)
        return None


# ============================================================
# MCP TOOL REGISTRATION
# ============================================================

def register_tools(mcp):
    """Register session bridge diagnostic tool."""

    @mcp.tool()
    def session_bridge_status() -> str:
        """Show last session checkpoint and bridge status for diagnostics."""
        try:
            conn = _get_conn()

            # Count total checkpoints
            total = conn.execute("SELECT COUNT(*) FROM session_checkpoints").fetchone()[0]

            # Last checkpoint
            row = conn.execute(
                "SELECT id, source, active_project, session_summary, created_at "
                "FROM session_checkpoints ORDER BY created_at DESC LIMIT 1"
            ).fetchone()

            if not row:
                conn.close()
                return f"Session Bridge: 0 checkpoints. No data yet."

            cp_id, cp_source, cp_project, cp_summary, cp_ts = row
            hours = _hours_since(cp_ts)

            # Try loading bridge
            bridge = load_session_bridge()

            conn.close()

            lines = [
                "# SESSION BRIDGE STATUS",
                f"- Total checkpoints: {total} (max {RETENTION_MAX})",
                f"- Last checkpoint: id={cp_id}, source={cp_source}",
                f"- Created: {cp_ts} ({hours:.1f}h ago)",
                f"- Project: {cp_project or 'none'}",
                f"- Summary: {(cp_summary or '')[:100]}",
                "",
            ]

            if bridge:
                lines.append("## BRIDGE (would show at wake)")
                lines.append(bridge["bridge_text"])
                if bridge["prediction_errors"]:
                    lines.append(f"\n## PEs: {len(bridge['prediction_errors'])}")
                    for pe in bridge["prediction_errors"]:
                        lines.append(f"- [{pe['type']}] {pe['detail']}")
            else:
                lines.append("## BRIDGE: None (stale or missing)")

            return "\n".join(lines)

        except Exception as e:
            return f"Session Bridge ERROR: {redact_secrets(str(e))}"
