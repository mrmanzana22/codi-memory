"""
pe_actions.py - Prediction Error -> Action Pipeline (Bloque 2)
===============================================================
Flag-gated handlers that convert high-confidence prediction errors
into workspace focus (H1) and prospective intentions (H2).

Gating: CODI_PE_ACTIONS env var (default "off").
When off, these handlers are no-ops. The existing _on_prediction_error
in wiring.py (always active) continues to do WM push + attention schema.

Guardrails:
  - Rate limit: max 1 action per topic per COOLDOWN_HOURS
  - Idempotent: spotlight not re-focused if topic already there
  - Dedupe: intentions deduped by sha256(pe|topic|YYYY-MM-DD)
  - No LLM calls — deterministic rules only
  - try/except per handler — never blocks the emitter

Created: 2026-02-16 (Bloque 2)
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from typing import Callable, Optional

_logger = logging.getLogger(__name__)

from modules.config import now_iso, now_col


# ============================================================
# CLOCK (injectable for testing)
# ============================================================

def _now() -> float:
    """Current time in seconds since epoch. Monkeypatch in tests."""
    return time.time()


# ============================================================
# CONSTANTS
# ============================================================

COOLDOWN_HOURS = 6
COOLDOWN_SECONDS = COOLDOWN_HOURS * 3600


# ============================================================
# RATE LIMITER (in-memory, per-topic)
# ============================================================

_rate_limit_log: dict[str, float] = {}  # topic -> last_action_timestamp


def _is_rate_limited(topic: str) -> bool:
    """Check if topic has been acted on within cooldown window."""
    normalized = topic.strip().lower()
    last = _rate_limit_log.get(normalized)
    if last is None:
        return False
    return (_now() - last) < COOLDOWN_SECONDS


def _record_action(topic: str) -> None:
    """Record that an action was taken for this topic."""
    normalized = topic.strip().lower()
    _rate_limit_log[normalized] = _now()


def reset_rate_limits() -> None:
    """Clear all rate limits (for testing)."""
    _rate_limit_log.clear()


# ============================================================
# FLAG CHECK
# ============================================================

def _is_pe_actions_enabled() -> bool:
    """Check if CODI_PE_ACTIONS is on."""
    return os.environ.get("CODI_PE_ACTIONS", "off").strip().lower() == "on"


# ============================================================
# INTENTION DEDUPE
# ============================================================

def _compute_intention_dedupe_key(topic: str) -> str:
    """Deterministic dedupe key: sha256(pe|topic_norm|YYYY-MM-DD)[:16]."""
    day = now_col().strftime("%Y-%m-%d")
    normalized = topic.strip().lower()
    raw = f"pe|{normalized}|{day}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def _intention_exists_today(topic: str) -> bool:
    """Check if a PE intention for this topic was already created today."""
    dedupe_key = _compute_intention_dedupe_key(topic)
    try:
        from modules.prospective import _get_conn
        conn = _get_conn()
        row = conn.execute(
            "SELECT id FROM intentions "
            "WHERE status = 'pending' AND context_at_creation LIKE ?",
            (f"%pe_dedupe:{dedupe_key}%",)
        ).fetchone()
        return row is not None
    except Exception:
        return False


# ============================================================
# H1: WORKSPACE REACTION (low risk)
# ============================================================

def pe_workspace_handler(event_name: str, data: dict) -> None:
    """Focus workspace spotlight on PE topic when flag is on.

    Only acts on intensity=high. Skips if:
    - Flag is off
    - Intensity is not high
    - Topic already in spotlight
    - Rate limited
    """
    try:
        if not _is_pe_actions_enabled():
            return

        intensity = data.get("intensity", "medium")
        if intensity != "high":
            return

        topic = data.get("topic", "")
        if not topic:
            return

        if _is_rate_limited(topic):
            return

        # Check if topic already in spotlight
        from modules.workspace import _global_workspace, update_workspace_spotlight
        current_spotlight = _global_workspace.get("spotlight", [])
        topic_lower = topic.strip().lower()

        # Check if any spotlight item already references this topic
        for item in current_spotlight:
            if isinstance(item, dict):
                content = str(item.get("content", "") or item.get("topic", "")).lower()
                if topic_lower in content:
                    return  # Already in spotlight, no-op
            elif isinstance(item, str) and topic_lower in item.lower():
                return

        # Focus: add PE marker to spotlight
        pe_entry = {
            "content": f"[PE] Prediction error on: {topic}",
            "topic": topic,
            "source": "prediction_error",
            "intensity": intensity,
            "timestamp": now_iso(),
        }
        update_workspace_spotlight(
            memories=[pe_entry] + current_spotlight[:4],
            theme=f"prediction_error:{topic}",
        )

        _record_action(topic)

    except Exception as e:
        _logger.error("H1 workspace error: %s", e)


# ============================================================
# H2: PROSPECTIVE REACTION (medium risk)
# ============================================================

def pe_prospective_handler(event_name: str, data: dict) -> None:
    """Create a review intention when a high PE fires.

    Only acts on intensity=high. Dedupes by topic + day.
    """
    try:
        if not _is_pe_actions_enabled():
            return

        intensity = data.get("intensity", "medium")
        if intensity != "high":
            return

        topic = data.get("topic", "")
        if not topic:
            return

        # Rate limit is shared with H1 — if H1 already recorded, skip
        # But we check intention-specific dedupe separately
        if _intention_exists_today(topic):
            return

        dedupe_key = _compute_intention_dedupe_key(topic)

        from modules.prospective import create_intention
        create_intention(
            action=f"Revisar tema '{topic}' - prediction error detectado",
            trigger_type="event",
            trigger_spec=json.dumps({"keywords": [topic.lower()], "match_threshold": 1}),
            priority="medium",
            context=f"PE high on '{topic}'. pe_dedupe:{dedupe_key}",
            creator="pe_actions",
        )

    except Exception as e:
        _logger.error("H2 prospective error: %s", e)


# ============================================================
# HANDLER REGISTRATION (called from wiring.py)
# ============================================================

def get_handlers() -> list[tuple[str, Callable]]:
    """Return list of (event_name, handler) pairs to register."""
    from modules.events import Events
    return [
        (Events.PREDICTION_ERROR, pe_workspace_handler),
        (Events.PREDICTION_ERROR, pe_prospective_handler),
    ]
