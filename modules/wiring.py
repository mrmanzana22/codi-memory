"""
Codi Memory - Cross-Module Wiring (Thalamocortical Integration)
================================================================
Registers event bus subscribers that create feedback loops between modules.

Without this module, the system is a collection of isolated tools.
With it, modules communicate automatically via events -- the minimum
requirement for GWT-style consciousness.

Architecture (audit v4 finding):
  "The event bus has zero subscribers. It is a highway with no exits."
  This module builds the exits.

Event Flow:
  MEMORY_STORED -> push high-importance to WM, log for consolidation
  MEMORY_RETRIEVED -> trigger spreading activation on top results
  WORKSPACE_BROADCAST -> push to WM, update attention schema
  EMOTION_CHANGED -> log emotional shift, modulate retrieval bias
  CONSOLIDATION_COMPLETE -> update stats, notify if significant
  PREDICTION_ERROR -> boost encoding strength, log surprise

Called once from server.py at startup.

Created: 2026-02-13 (Wiring Phase - audit v4 remediation)
"""

import sys
import os
import math
import time

from modules.events import event_bus, Events
from modules.config import now_iso

# Track wiring state
_wired = False
_last_interaction_time = None
_interaction_count = 0
_session_interaction_count = 0  # Resets on long gaps

# Importance string -> float map (matches memory_core.py)
IMPORTANCE_MAP = {'critical': 1.0, 'high': 0.8, 'medium': 0.5, 'low': 0.2}


# ============================================================
# HANDLER: MEMORY_STORED
# ============================================================

def _on_memory_stored(event_name: str, data: dict):
    """When a memory is stored, push high-importance ones to working memory.

    Creates feedback: important experiences immediately enter WM,
    making them available for the current cognitive cycle.
    """
    try:
        importance_raw = data.get("importance", "medium")
        # P0-2 FIX: importance is a string ("critical","high","medium","low"), not float
        if isinstance(importance_raw, str):
            importance = IMPORTANCE_MAP.get(importance_raw, 0.5)
        else:
            importance = float(importance_raw)

        content = data.get("content", "")
        category = data.get("category", "general")

        # Only push high-importance memories to WM (avoid flooding)
        if importance >= 0.7 and content:
            from modules.working_memory import push_to_working_memory
            push_to_working_memory(
                content=content[:300],
                topic=category,
                relevance=min(1.0, importance),
                source="event_bus",
            )
    except Exception as e:
        print(f"[wiring] _on_memory_stored error: {e}", file=sys.stderr)


# ============================================================
# HANDLER: MEMORY_RETRIEVED
# ============================================================

def _on_memory_retrieved(event_name: str, data: dict):
    """When memories are retrieved, trigger spreading activation on top results.

    Creates feedback: retrieval primes related memories, changing
    what gets retrieved next time (associative priming).
    SAM model (Raaijmakers & Shiffrin 1981): co-activated memories
    contribute to associative context.
    """
    try:
        retrieved_ids = data.get("retrieved_ids", [])
        result_count = data.get("result_count", 0)
        query_topic = data.get("query_topic", "")

        # Only spread on meaningful retrievals (not empty results)
        # Use recurrent_cycle for true re-entrant processing (Lamme 2006)
        if result_count > 0 and retrieved_ids:
            from modules.spreading import recurrent_cycle
            recurrent_cycle(retrieved_ids[:3], cycles=2, depth=1, factor=0.3)

        # Update attention schema with retrieval topic (coherence fix)
        if query_topic:
            _update_attention_schema(
                focus=query_topic,
                driver="memory_retrieval",
                strength=0.5,
            )
    except Exception as e:
        print(f"[wiring] _on_memory_retrieved error: {e}", file=sys.stderr)


# ============================================================
# HANDLER: WORKSPACE_BROADCAST
# ============================================================

def _on_workspace_broadcast(event_name: str, data: dict):
    """When workspace broadcasts, push content to WM and update attention.

    Creates feedback: broadcast content persists in WM for the session,
    and the attention schema tracks what was broadcast and when.
    """
    try:
        content = data.get("content", "")
        themes = data.get("themes", [])
        memory_id = data.get("memory_id", "")

        # Push broadcast to working memory
        if content:
            from modules.working_memory import push_to_working_memory
            push_to_working_memory(
                content=f"[BROADCAST] {content[:200]}",
                topic=themes[0] if themes else "broadcast",
                relevance=0.85,
                source="workspace_broadcast",
            )

        # Update attention schema
        _update_attention_schema(
            focus=themes[0] if themes else "unknown",
            driver="workspace_broadcast",
            strength=0.8,
        )
    except Exception as e:
        print(f"[wiring] _on_workspace_broadcast error: {e}", file=sys.stderr)


# ============================================================
# HANDLER: EMOTION_CHANGED
# ============================================================

def _on_emotion_changed(event_name: str, data: dict):
    """When emotional state changes, log the shift for temporal tracking.

    Creates feedback: emotional changes modulate subsequent memory
    encoding and retrieval via mood-congruent bias.
    """
    try:
        arousal = data.get("arousal", 0.0)
        emotion = data.get("emotion", "neutral")
        trigger = data.get("trigger", "")

        # Update attention schema: high arousal shifts attention
        if abs(arousal) > 0.5:
            _update_attention_schema(
                focus=f"emotion:{emotion}",
                driver=f"emotion_shift:{trigger}",
                strength=min(1.0, abs(arousal)),
            )
    except Exception as e:
        print(f"[wiring] _on_emotion_changed error: {e}", file=sys.stderr)


# ============================================================
# HANDLER: CONSOLIDATION_COMPLETE
# ============================================================

def _on_consolidation_complete(event_name: str, data: dict):
    """When consolidation finishes, update system state."""
    try:
        facts_created = data.get("facts_created", 0)
        contradictions = data.get("contradictions", 0)

        if facts_created > 0 or contradictions > 0:
            from modules.working_memory import push_to_working_memory
            summary = f"Consolidacion: {facts_created} facts"
            if contradictions > 0:
                summary += f", {contradictions} contradicciones"
            push_to_working_memory(
                content=summary,
                topic="consolidation",
                relevance=0.6,
                source="consolidation_complete",
            )
    except Exception as e:
        print(f"[wiring] _on_consolidation_complete error: {e}", file=sys.stderr)


# ============================================================
# ATTENTION SCHEMA (Graziano 2013)
# ============================================================

# Internal model of what the system is attending to
_attention_schema = {
    "current_focus": None,       # Topic currently being attended
    "focus_strength": 0.0,       # How strongly focused (0-1)
    "focus_started": None,       # When current focus began
    "driver": None,              # What caused current focus
    "history": [],               # Last 20 focus states
    "topic_transitions": [],     # Topic A -> Topic B log (last 30)
    "interrupted_topics": [],    # Topics displaced by new focus
    "suppressed_items": [],      # 3B: Last 10 competition losers (Graziano AST)
}


def _update_attention_schema(focus: str, driver: str = "unknown", strength: float = 0.5):
    """Update the attention schema with a new focus.

    Tracks topic transitions and interrupted topics.
    This is NOT full Graziano AST -- it is a minimal self-model
    of the attention process.
    """
    global _attention_schema
    now = now_iso()

    old_focus = _attention_schema["current_focus"]

    # Record transition
    if old_focus and old_focus != focus:
        _attention_schema["topic_transitions"].append({
            "from": old_focus,
            "to": focus,
            "at": now,
            "driver": driver,
        })
        _attention_schema["topic_transitions"] = _attention_schema["topic_transitions"][-30:]

        # Old topic is now "interrupted"
        if strength > _attention_schema["focus_strength"]:
            _attention_schema["interrupted_topics"].append({
                "topic": old_focus,
                "interrupted_at": now,
                "by": focus,
            })
            _attention_schema["interrupted_topics"] = _attention_schema["interrupted_topics"][-10:]

    # Save current state to history before updating
    if _attention_schema["current_focus"]:
        _attention_schema["history"].append({
            "focus": _attention_schema["current_focus"],
            "strength": _attention_schema["focus_strength"],
            "started": _attention_schema["focus_started"],
            "ended": now,
            "driver": _attention_schema["driver"],
        })
        _attention_schema["history"] = _attention_schema["history"][-20:]

    # Update current focus
    _attention_schema["current_focus"] = focus
    _attention_schema["focus_strength"] = strength
    _attention_schema["focus_started"] = now
    _attention_schema["driver"] = driver


def get_attention_schema() -> dict:
    """Return the current attention schema state (for external queries)."""
    return {
        "current_focus": _attention_schema["current_focus"],
        "focus_strength": _attention_schema["focus_strength"],
        "focus_started": _attention_schema["focus_started"],
        "driver": _attention_schema["driver"],
        "recent_transitions": _attention_schema["topic_transitions"][-5:],
        "interrupted_topics": _attention_schema["interrupted_topics"][-3:],
        "suppressed_items": _attention_schema["suppressed_items"][-5:],
        "history_length": len(_attention_schema["history"]),
    }


def describe_attention() -> str:
    """Self-descriptive report of attention state (Graziano 2013 AST, Phase 3B).

    Generates a higher-order representation of the attention process itself.
    This is the HOT (Higher-Order Thought) component -- the system
    describes what it is attending to, why, and what it predicts next.
    """
    focus = _attention_schema["current_focus"]
    strength = _attention_schema["focus_strength"]
    driver = _attention_schema["driver"]
    suppressed = _attention_schema["suppressed_items"]

    if not focus:
        return "No tengo foco de atencion activo. Estoy en estado difuso."

    # Build self-description
    parts = [f"Estoy atendiendo a '{focus}' (fuerza: {strength:.2f})"]

    if driver:
        parts.append(f"porque {driver}")

    if suppressed:
        unique_suppressed = list(dict.fromkeys(suppressed[-5:]))
        parts.append(f"Estoy ignorando: {', '.join(unique_suppressed)}")

    # Predict next focus
    prediction, prob = predict_next_focus()
    if prediction:
        parts.append(f"Predigo que mi proximo foco sera '{prediction}' (prob: {prob:.2f})")

    return ". ".join(parts) + "."


def predict_next_focus() -> tuple:
    """Bigram transition model for attention prediction (Phase 3B).

    Uses topic_transitions history to build transition probabilities.

    Returns:
        (predicted_topic: str | None, probability: float)
    """
    transitions = _attention_schema["topic_transitions"]
    current = _attention_schema["current_focus"]

    if not transitions or not current:
        return (None, 0.0)

    # Count transitions FROM current topic
    from collections import Counter
    current_lower = current.lower()
    next_counts = Counter()
    for t in transitions:
        if t.get("from", "").lower() == current_lower:
            next_counts[t["to"]] += 1

    if not next_counts:
        return (None, 0.0)

    total = sum(next_counts.values())
    most_common = next_counts.most_common(1)[0]
    return (most_common[0], most_common[1] / total)


# ============================================================
# HANDLER: PREDICTION_ERROR (Schultz 1997)
# ============================================================

def _on_prediction_error(event_name: str, data: dict):
    """Prediction errors boost encoding strength and capture attention.

    Neurobiological basis: Dopaminergic prediction error signals
    (Schultz 1997) enhance hippocampal encoding. Surprise increases
    memory formation via locus coeruleus-norepinephrine system.
    """
    error_magnitude = data.get("error_magnitude", 0.5)
    topic = data.get("topic", "unknown")
    memory_id = data.get("memory_id", "")
    actual_keywords = data.get("actual_keywords", [])

    # Effect 1: Boost salience of surprising memory (Schultz 1997)
    if error_magnitude > 0.3:
        try:
            if memory_id:
                from modules.spreading import _spread_activation
                _spread_activation([memory_id], depth=1, factor=min(0.5, error_magnitude))
            elif actual_keywords:
                from modules.working_memory import push_to_working_memory
                push_to_working_memory(
                    content=f"[PREDICTION ERROR] Surprise en topic '{topic}': {', '.join(actual_keywords[:5])}",
                    topic="prediction_surprise",
                    relevance=min(0.95, 0.6 + error_magnitude),
                    source="prediction_error",
                )
        except Exception as e:
            print(f"[wiring] _on_prediction_error spreading error: {e}", file=sys.stderr)

    # Effect 2: Surprise captures attention (Corbetta & Shulman 2002)
    if error_magnitude > 0.3:
        try:
            _update_attention_schema(
                focus=f"surprise:{topic}",
                driver="prediction_error",
                strength=min(1.0, error_magnitude),
            )
        except Exception as e:
            print(f"[wiring] _on_prediction_error attention error: {e}", file=sys.stderr)

    # Effect 3: PE-driven reconsolidation (Nader 2000, Lee 2009)
    # If error involves a specific memory, check for reconsolidation
    if memory_id and error_magnitude > 0.3:
        try:
            from modules.consolidation import check_reconsolidation, mark_as_labile
            from modules.config import qdrant, COLLECTION_NAME
            pts = qdrant.retrieve(collection_name=COLLECTION_NAME, ids=[memory_id], with_payload=True)
            if pts:
                payload = pts[0].payload or {}
                trigger_context = f"PE={error_magnitude:.2f} topic={topic}"
                recon = check_reconsolidation(memory_id, payload, trigger_context)
                if recon.get("should_reconsolidate"):
                    mark_as_labile(memory_id, error_magnitude, trigger_context)
        except Exception as e:
            print(f"[wiring] _on_prediction_error reconsolidation error: {e}", file=sys.stderr)


# ============================================================
# TEMPORAL DYNAMICS (called from pre-turn hook)
# ============================================================

def process_elapsed_time(elapsed_seconds: float):
    """Process temporal dynamics based on time since last interaction.

    A brain does not freeze between inputs. This function applies
    time-proportional decay and maintenance when the user returns
    after a gap.

    Called from preturn_inject.py on every turn.
    """
    global _interaction_count, _session_interaction_count
    _interaction_count += 1

    if elapsed_seconds < 60:
        _session_interaction_count += 1
        return  # Less than 1 min, nothing to do

    elapsed_hours = elapsed_seconds / 3600

    # Reset session counter on long gaps (8+ hours = new session)
    if elapsed_hours >= 8.0:
        _session_interaction_count = 0
    _session_interaction_count += 1

    # --- WM item decay (Ebbinghaus 1885 exponential forgetting) ---
    if elapsed_hours >= 0.5:  # 30+ minutes
        try:
            from modules.working_memory import _get_conn
            # P0-1 FIX: _get_conn is a @contextmanager, must use `with`
            with _get_conn() as conn:
                # Exponential decay: relevance * exp(-lambda * hours)
                # lambda=0.1: ~60% at 5h, ~37% at 10h, ~14% at 20h
                decay_multiplier = math.exp(-0.1 * elapsed_hours)
                conn.execute("""
                    UPDATE working_memory
                    SET relevance = MAX(0.1, relevance * ?)
                    WHERE active = 1
                """, (decay_multiplier,))
                conn.commit()
        except Exception as e:
            print(f"[wiring] WM decay error: {e}", file=sys.stderr)

    # --- Attention focus decay (Mackworth 1948 vigilance decrement) ---
    if elapsed_hours >= 0.25:  # 15+ minutes
        if _attention_schema["focus_strength"] > 0.1:
            decay = min(0.5, elapsed_hours * 0.1)
            _attention_schema["focus_strength"] = max(
                0.1, _attention_schema["focus_strength"] - decay
            )

    # --- PM activation maintenance (every 2+ hours) ---
    if elapsed_hours >= 2.0:
        try:
            from modules.prospective import apply_intention_maintenance
            apply_intention_maintenance()
        except Exception as e:
            print(f"[wiring] PM maintenance error: {e}", file=sys.stderr)

    # --- Emotional decay toward baseline (proportional to gap) ---
    if elapsed_hours >= 1.0:
        try:
            from modules.consciousness import apply_emotional_decay
            # Scale decay by time: 1 step per hour, max 10
            steps = min(int(elapsed_hours), 10)
            for _ in range(steps):
                apply_emotional_decay()
        except Exception as e:
            print(f"[wiring] Emotional decay error: {e}", file=sys.stderr)

    # --- Consolidation check if long gap (8+ hours) ---
    if elapsed_hours >= 8.0 and _session_interaction_count <= 1:
        try:
            from modules.consolidation import count_unconsolidated_episodic
            pending = count_unconsolidated_episodic(lookback_hours=24)
            if pending > 20:
                from modules.working_memory import push_to_working_memory
                push_to_working_memory(
                    content=f"[MAINTENANCE] {pending} episodios sin consolidar. Considerar correr consolidacion.",
                    topic="maintenance",
                    relevance=0.7,
                    source="temporal_dynamics",
                )
        except Exception as e:
            print(f"[wiring] Consolidation check error: {e}", file=sys.stderr)


def get_last_interaction_time() -> str:
    """Get the last interaction time (for hook to compute elapsed)."""
    return _last_interaction_time or ""


def set_last_interaction_time(iso_time: str):
    """Set the last interaction time."""
    global _last_interaction_time
    _last_interaction_time = iso_time


# ============================================================
# WIRING-6: WORKSPACE COMPETITION HANDLER
# ============================================================

def _on_competition_complete(event_name: str, data: dict):
    """React to workspace competition results (WIRING-6.4).

    Updates attention schema with winning topic and applies
    salience penalty to losers (inhibition of return per GWT).
    """
    try:
        winners = data.get('winner_domains', [])
        loser_ids = data.get('loser_ids', [])
        top_activation = data.get('top_activation', 0.0)

        # Update attention schema with dominant winning domain
        if winners:
            _update_attention_schema(
                focus=winners[0],
                driver="workspace_competition",
                strength=min(1.0, top_activation),
            )

        # 3B: Track suppressed items (Graziano AST - what we're NOT attending)
        loser_domains = data.get('loser_domains', [])
        if loser_domains:
            _attention_schema["suppressed_items"] = (
                _attention_schema["suppressed_items"] + loser_domains[:5]
            )[-10:]

        # Salience penalty for losers (inhibition of return) - batched
        if loser_ids:
            try:
                from modules.config import qdrant, COLLECTION_NAME
                batch_ids = loser_ids[:20]
                points = qdrant.retrieve(
                    collection_name=COLLECTION_NAME,
                    ids=batch_ids, with_payload=True
                )
                for p in points:
                    old_sal = p.payload.get('attention_salience', 0.5)
                    new_sal = max(0.1, old_sal - 0.05)
                    qdrant.set_payload(
                        collection_name=COLLECTION_NAME,
                        payload={'attention_salience': new_sal},
                        points=[p.id]
                    )
            except Exception:
                pass
    except Exception as e:
        print(f"[wiring] Competition handler error: {e}", file=sys.stderr)


# ============================================================
# WIRING-7: RETRIEVAL QUALITY HANDLER
# ============================================================

def _on_retrieval_quality(event_name: str, data: dict):
    """React to retrieval quality signals (WIRING-7.4).

    If retrieval was sparse/empty, push knowledge gap note to WM.
    """
    try:
        coverage = data.get("coverage", "")
        query = data.get("query", "")

        if coverage in ("sparse", "empty") and query:
            from modules.working_memory import push_to_working_memory
            push_to_working_memory(
                content=f"[KNOWLEDGE GAP] Search for '{query[:50]}' returned {coverage} results",
                topic="metamemory",
                relevance=0.5,
                source="retrieval_quality",
            )
    except Exception as e:
        print(f"[wiring] Retrieval quality handler error: {e}", file=sys.stderr)


# ============================================================
# WIRING REGISTRATION (called once at startup)
# ============================================================

def wire_event_bus():
    """Register all event handlers. Called from server.py at startup.

    This is the thalamocortical integration layer -- it connects
    the cortical modules (memory, emotion, workspace, PM) via
    the thalamic relay (event bus).
    """
    global _wired
    if _wired:
        return  # Idempotent

    event_bus.on(Events.MEMORY_STORED, _on_memory_stored)
    event_bus.on(Events.MEMORY_RETRIEVED, _on_memory_retrieved)
    event_bus.on(Events.WORKSPACE_BROADCAST, _on_workspace_broadcast)
    event_bus.on(Events.EMOTION_CHANGED, _on_emotion_changed)
    event_bus.on(Events.CONSOLIDATION_COMPLETE, _on_consolidation_complete)
    event_bus.on(Events.PREDICTION_ERROR, _on_prediction_error)
    event_bus.on(Events.WORKSPACE_COMPETITION_COMPLETE, _on_competition_complete)
    event_bus.on(Events.RETRIEVAL_QUALITY, _on_retrieval_quality)

    _wired = True
    stats = event_bus.get_stats()
    print(f"[wiring] Event bus wired: {sum(stats.values())} handlers across {len(stats)} events", file=sys.stderr)


def get_wiring_stats() -> dict:
    """Get wiring diagnostics."""
    return {
        "wired": _wired,
        "event_bus_stats": event_bus.get_stats(),
        "event_bus_history": len(event_bus.get_history()),
        "interaction_count": _interaction_count,
        "last_interaction": _last_interaction_time,
        "attention_schema": get_attention_schema(),
    }
