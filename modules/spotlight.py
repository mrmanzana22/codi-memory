"""
Spotlight Policy for Global Workspace Theory (GWT).

Executive spotlight: max 3 structured items (goal, risk, next_action).
Read-side only — does NOT change what gets written to memory.

Entry points:
  - despertar_codi()    -> full recompute
  - checkpoint_memoria() -> partial update (next_action, risk, goal)
  - post-recall()       -> update only with strong signal

Storage: module-level _spotlight list (in-session only).
"""

import logging
import re
from datetime import datetime, timedelta
from typing import Optional

from modules.config import now_col, now_iso

_logger = logging.getLogger("codi-memory.spotlight")

# ============================================================
# SPOTLIGHT STATE (in-session, not persisted)
# ============================================================

_spotlight = []  # list of SpotlightItem dicts

SPOTLIGHT_MAX_ITEMS = 3
SPOTLIGHT_TTL_SECONDS = 21600  # 6 hours
VALID_TYPES = frozenset({"goal", "risk", "next_action"})
TYPE_ORDER = ["goal", "risk", "next_action"]


# ============================================================
# PURE FUNCTIONS
# ============================================================

def make_item(item_type: str, text: str, source: str) -> dict:
    """Create a spotlight item dict."""
    return {
        "type": item_type,
        "text": text[:200],  # keep concise
        "source": source,
        "ts": now_iso(),
    }


def _apply_ttl(items: list) -> list:
    """Remove expired items based on TTL."""
    now = now_col()
    alive = []
    for item in items:
        try:
            ts = datetime.fromisoformat(item["ts"])
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=now.tzinfo)
            age = (now - ts).total_seconds()
        except (KeyError, TypeError, ValueError):
            _logger.warning("Dropping spotlight item with invalid timestamp: %r", item)
            continue
        if 0 <= age < SPOTLIGHT_TTL_SECONDS:
            alive.append(item)
    return alive


def _dedup_by_type(items: list) -> list:
    """Keep only the latest item per type, ordered by TYPE_ORDER."""
    by_type = {}
    for item in items:
        by_type[item["type"]] = item  # last one wins
    return [by_type[t] for t in TYPE_ORDER if t in by_type]


def select_goal(intentions: list, checkpoint_text: str = "") -> Optional[dict]:
    """Select goal from intentions or checkpoint text.

    Args:
        intentions: list of intention dicts from get_pending_intentions()
        checkpoint_text: combined text from last checkpoint
    """
    # 1. Most important/urgent intention
    if intentions:
        top = intentions[0]  # already sorted by priority+activation
        return make_item("goal", top["action"][:200], "intention")

    # 2. Checkpoint with goal keywords
    if checkpoint_text:
        text_low = checkpoint_text.lower()
        goal_patterns = ["objetivo", "meta", "hoy", "esta semana", "goal", "plan"]
        for pattern in goal_patterns:
            if pattern in text_low:
                # Extract a concise goal from the text
                return make_item("goal", checkpoint_text[:200], "checkpoint")

    return None


def select_risk(health_signals: dict) -> Optional[dict]:
    """Select most actionable risk from health signals.

    Args:
        health_signals: dict with keys like:
            - dual_gate: "PASS" | "FAIL" | "WARNING" | None
            - dual_detail: str
            - write_queue_dead: int
            - write_queue_failed_ratio: float
            - access_tracking_errors: int
            - health_ok: bool
            - health_message: str
    """
    # Priority 1: dual gate failure
    gate = health_signals.get("dual_gate")
    if gate and isinstance(gate, str) and gate.upper() in ("FAIL", "WARNING"):
        detail = health_signals.get("dual_detail", "")
        return make_item("risk", f"DUAL gate {gate}: {detail}"[:200], "health")

    # Priority 2: write queue unhealthy
    dead = health_signals.get("write_queue_dead", 0)
    failed_ratio = health_signals.get("write_queue_failed_ratio", 0.0)
    if dead > 0 or failed_ratio > 0.05:
        return make_item("risk", f"Write queue: dead={dead} failed_ratio={failed_ratio:.1%}", "health")

    # Priority 3: access tracking errors
    at_errors = health_signals.get("access_tracking_errors", 0)
    if at_errors > 0:
        return make_item("risk", f"Access tracking errors: {at_errors}", "health")

    # Priority 4: general health issue
    if not health_signals.get("health_ok", True):
        msg = health_signals.get("health_message", "unknown issue")
        return make_item("risk", f"Health: {msg}"[:200], "health")

    return None


def select_next_action(intentions: list, checkpoint_text: str = "") -> Optional[dict]:
    """Select next concrete action.

    Args:
        intentions: list of intention dicts
        checkpoint_text: combined text from last checkpoint
    """
    # 1. Check checkpoint for explicit next steps
    if checkpoint_text:
        text_low = checkpoint_text.lower()
        action_patterns = [
            "siguiente paso", "next step", "proximo", "next action",
            "pendiente", "falta", "hay que", "toca",
        ]
        for pattern in action_patterns:
            if pattern in text_low:
                return make_item("next_action", checkpoint_text[:200], "checkpoint")

    # 2. Top intention that has a concrete next step
    if intentions:
        for intent in intentions:
            if intent.get("action_type") in ("execute", "remind", "check"):
                return make_item("next_action", intent["action"][:200], "intention")
        # Fallback: use top intention regardless of type
        return make_item("next_action", intentions[0]["action"][:200], "intention")

    return None


def build_spotlight(
    intentions: list = None,
    health_signals: dict = None,
    checkpoint_text: str = "",
) -> list:
    """Build full spotlight from available context. Pure function.

    Args:
        intentions: from get_pending_intentions()
        health_signals: from health/status checks
        checkpoint_text: text from latest checkpoint

    Returns:
        List of max 3 spotlight items (goal/risk/next_action).
    """
    intentions = intentions or []
    health_signals = health_signals or {}
    items = []

    goal = select_goal(intentions, checkpoint_text)
    if goal:
        items.append(goal)

    risk = select_risk(health_signals)
    if risk:
        items.append(risk)

    next_action = select_next_action(intentions, checkpoint_text)
    if next_action:
        items.append(next_action)

    return items[:SPOTLIGHT_MAX_ITEMS]


def update_spotlight_from_text(current: list, text: str, source: str) -> list:
    """Update spotlight items based on free-text input (checkpoint/recall).

    Detects keywords to decide which slot(s) to update.
    Returns new spotlight list.
    """
    text_low = text.lower()

    updated = list(current)  # shallow copy

    # Detect next_action signals
    action_kw = ["siguiente paso", "next step", "proximo", "pendiente", "hay que", "toca"]
    if any(kw in text_low for kw in action_kw):
        new_item = make_item("next_action", text[:200], source)
        updated = [i for i in updated if i["type"] != "next_action"]
        updated.append(new_item)

    # Detect risk signals
    risk_kw = ["riesgo", "bloqueo", "fail", "error", "rompe", "broken"]
    if any(kw in text_low for kw in risk_kw) or re.search(r"\bbug\b", text_low):
        new_item = make_item("risk", text[:200], source)
        updated = [i for i in updated if i["type"] != "risk"]
        updated.append(new_item)

    # Detect goal signals
    goal_kw = ["objetivo", "meta", "hoy vamos", "esta semana", "goal", "plan"]
    if any(kw in text_low for kw in goal_kw):
        new_item = make_item("goal", text[:200], source)
        updated = [i for i in updated if i["type"] != "goal"]
        updated.append(new_item)

    return _dedup_by_type(updated)[:SPOTLIGHT_MAX_ITEMS]


# ============================================================
# RECALL SIGNAL DETECTION
# ============================================================

def should_update_from_recall(results: list) -> bool:
    """Determine if recall results have strong enough signal to update spotlight.

    Args:
        results: list of result dicts from recall

    Returns:
        True if signal is strong enough to warrant spotlight update.
    """
    if not results:
        return False

    # Strong signal: explicit evidence inside a recall result
    for r in results:
        meta = r.get("meta", {}) or {}
        text = r.get("text") or ""
        text_low = text.lower()
        importance = str(meta.get("importance", "")).lower()
        hits = meta.get("hits") or 0
        if importance in ("critical", "high"):
            return True
        if hits >= 3:
            return True
        if "critical" in text_low or "bloqueo" in text_low:
            return True

    return False


# ============================================================
# STATE MANAGEMENT
# ============================================================

def get_spotlight() -> list:
    """Get current spotlight, applying TTL filter."""
    global _spotlight
    _spotlight = _apply_ttl(_spotlight)
    return list(_spotlight)


def set_spotlight(items: list) -> None:
    """Replace spotlight with new items."""
    global _spotlight
    _spotlight = _dedup_by_type(items)[:SPOTLIGHT_MAX_ITEMS]
    _logger.info(
        "[spotlight] updated: %d items, types=%s",
        len(_spotlight),
        [i["type"] for i in _spotlight],
    )


def clear_spotlight() -> None:
    """Clear all spotlight items (used at session start before recompute)."""
    global _spotlight
    _spotlight = []


def format_spotlight() -> str:
    """Format spotlight for display in despertar/context_snapshot."""
    items = get_spotlight()
    if not items:
        return ""

    lines = ["## SPOTLIGHT (Executive Focus)"]
    emoji_map = {"goal": "TARGET", "risk": "ALERT", "next_action": "NEXT"}
    for item in items:
        label = emoji_map.get(item["type"], item["type"].upper())
        lines.append(f"- [{label}] {item['text']}")
        lines.append(f"  (source: {item['source']}, {item['ts'][:16]})")
    return "\n".join(lines)


# ============================================================
# PAD MICRO-UPDATES
# ============================================================

def pad_micro_update(signal: str) -> Optional[dict]:
    """Apply minimal PAD adjustment based on operational signal.

    Args:
        signal: one of "error_retry", "gate_pass", "repeated_block"

    Returns:
        Updated PAD dict or None if signal unknown.
    """
    from modules.config import _emotional_state

    current = _emotional_state.get("current", {})
    p = current.get("pleasure", 0.0)
    a = current.get("arousal", 0.0)
    d = current.get("dominance", 0.0)

    if signal == "error_retry":
        a = min(1.0, a + 0.1)  # arousal up
    elif signal == "gate_pass":
        p = min(1.0, p + 0.1)  # pleasure up
    elif signal == "repeated_block":
        d = max(-1.0, d - 0.1)  # dominance down
    else:
        return None

    _emotional_state["current"]["pleasure"] = round(p, 2)
    _emotional_state["current"]["arousal"] = round(a, 2)
    _emotional_state["current"]["dominance"] = round(d, 2)
    _emotional_state["current"]["timestamp"] = now_iso()
    _emotional_state["current"]["trigger"] = f"pad_micro:{signal}"

    _logger.info("[pad_micro] %s -> P=%.2f A=%.2f D=%.2f", signal, p, a, d)
    return {"pleasure": p, "arousal": a, "dominance": d, "trigger": signal}
