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
from collections import defaultdict
import logging
import threading

from modules.events import event_bus, Events
from modules.config import now_iso
from modules.secret_redact import redact_secrets
from modules.pg_store import pg as _pg
from modules.cx_observability import report_effective_fire

_logger = logging.getLogger(__name__)

# Track wiring state
_wired = False
_last_interaction_time = None
_interaction_count = 0
_session_interaction_count = 0  # Resets on long gaps

# HOT-1: Periodic self-model refresh (Rosenthal 2005)
_self_model_tick = 0
_last_self_model_refresh = 0.0  # time.monotonic() for cooldown
_SELF_MODEL_INTERVAL = 50       # trigger every N interactions
_SELF_MODEL_COOLDOWN = 120      # minimum seconds between refreshes

# Importance string -> float map (from config - single source of truth)
from modules.config import IMPORTANCE_WEIGHTS as IMPORTANCE_MAP, WM_IMPORTANCE_THRESHOLD

# Spreading activation concurrency control (monkeypatchable for tests)
_SPREADING_SEMAPHORE = threading.BoundedSemaphore(2)
_SPREADING_SEMAPHORE_CAPACITY = 2


# ============================================================
# HANDLER: MEMORY_STORED
# ============================================================

def _on_memory_stored(event_name: str, data: dict):
    """When a memory is stored, push high-importance ones to working memory.

    Creates feedback: important experiences immediately enter WM,
    making them available for the current cognitive cycle.
    """
    try:
        # Proposal #57: Skip WM push if remember() already handled it
        try:
            from modules.interface import _remember_ctx
            if getattr(_remember_ctx, 'wm_pushed', False):
                return
        except ImportError:
            pass

        importance_raw = data.get("importance", "medium")
        # P0-2 FIX: importance is a string ("critical","high","medium","low"), not float
        if isinstance(importance_raw, str):
            importance = IMPORTANCE_MAP.get(importance_raw, 0.5)
        else:
            importance = float(importance_raw)

        content = data.get("content", "")
        category = data.get("category", "general")

        # Only push high-importance memories to WM (avoid flooding)
        if importance >= WM_IMPORTANCE_THRESHOLD and content:
            from modules.working_memory import push_to_working_memory
            push_to_working_memory(
                content=content[:300],
                topic=category,
                relevance=min(1.0, importance),
                source="event_bus",
            )

        # Proposal 376: Interoceptive PE -> PAD (Seth 2013, Barrett 2017)
        try:
            from modules.emotion import compute_interoceptive_pe, INTERO_PAD_BLEND
            from modules.emotion import _set_emotional_state_impl
            from modules.config import _emotional_state
            intero = compute_interoceptive_pe()
            if intero is not None:
                cur = _emotional_state["current"]
                from modules.utils import _clamp_pad_value
                new_p = _clamp_pad_value(cur["pleasure"] + intero["pleasure_delta"] * INTERO_PAD_BLEND)
                new_a = _clamp_pad_value(cur["arousal"] + intero["arousal_boost"] * INTERO_PAD_BLEND)
                new_d = cur["dominance"]  # Interoception does not affect dominance (Seth 2013)
                delta = abs(new_p - cur["pleasure"]) + abs(new_a - cur["arousal"])
                if delta >= 0.005:
                    _set_emotional_state_impl(new_p, new_a, new_d, trigger="interoceptive_pe")
                    _logger.debug("interoceptive PE: P=%+.3f A=%+.3f (wm=%.2f debt=%d pe=%.2f)",
                                  intero["pleasure_delta"], intero["arousal_boost"],
                                  intero["metrics"]["wm_load"], intero["metrics"]["consol_debt"],
                                  intero["metrics"]["pe_rate"])
        except Exception:
            pass  # Graceful: interoception is supplementary, never blocks storage
    except Exception as e:
        _logger.error("_on_memory_stored error: %s", redact_secrets(str(e)))


# ============================================================
# HANDLER: MEMORY_RETRIEVED
# ============================================================

def _on_memory_retrieved(event_name: str, data: dict):
    """When memories are retrieved, trigger spreading activation on top results.

    Creates feedback: retrieval primes related memories, changing
    what gets retrieved next time (associative priming).
    SAM model (Raaijmakers & Shiffrin 1981): co-activated memories
    contribute to associative context.

    Spreading activation runs in a background thread (fire-and-forget)
    to avoid blocking the recall() hot path. Results update activation
    scores for FUTURE queries, not the current one.
    """
    try:
        retrieved_ids = data.get("retrieved_ids", [])
        result_count = data.get("result_count", 0)
        query_topic = data.get("query", "")

        # Fire spreading activation in background (non-blocking)
        if result_count > 0 and retrieved_ids:
            seed_ids = retrieved_ids[:3]

            def _bg_spread():
                sem = _SPREADING_SEMAPHORE  # Capture ref so release uses same object
                if not sem.acquire(blocking=False):
                    _logger.debug("spreading skipped: semaphore full")
                    return
                try:
                    from modules.spreading import recurrent_cycle
                    from modules.config import SPREAD_DEFAULT_DEPTH, SPREAD_DEFAULT_FACTOR
                    recurrent_cycle(seed_ids, cycles=SPREAD_DEFAULT_DEPTH, depth=1, factor=SPREAD_DEFAULT_FACTOR)
                except Exception as e:
                    _logger.warning("bg spreading error: %s", e)
                finally:
                    sem.release()

            t = threading.Thread(target=_bg_spread, daemon=True, name="spreading-bg")
            t.start()

        # Update attention schema synchronously (cheap, no I/O)
        if query_topic:
            _update_attention_schema(
                focus=query_topic,
                driver="memory_retrieval",
                strength=0.5,
                value=0.3,  # V: retrieval has moderate informational value
            )
    except Exception as e:
        _logger.warning("_on_memory_retrieved error: %s", e)


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
            value=0.7,  # V: broadcast items won competition, high informational value
        )
    except Exception as e:
        _logger.error("_on_workspace_broadcast error: %s", redact_secrets(str(e)))


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
                value=min(1.0, abs(arousal) * 0.8),  # V: emotional salience (Damasio somatic marker)
            )
    except Exception as e:
        _logger.error("_on_emotion_changed error: %s", redact_secrets(str(e)))


# ============================================================
# HANDLER: CONSOLIDATION_COMPLETE
# ============================================================

def _on_consolidation_complete(event_name: str, data: dict):
    """When consolidation finishes, update system state."""
    try:
        facts_created = data.get("facts_created", 0)
        contradictions = data.get("contradictions_found", 0)

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
        _logger.error("_on_consolidation_complete error: %s", redact_secrets(str(e)))


def _on_consolidation_enrich_fhrr(event_name: str, data: dict):
    """CX-32: Session index enriches consolidation (Teyler & DiScenna 1986).

    After consolidation completes, enrich newly created facts with session
    topics from the FHRR index. Read-only: adds context, does not modify scores.

    Paper: Teyler & DiScenna 1986 (Hippocampal Indexing Theory)
    """
    try:
        facts_created = data.get("facts_created", 0)
        if facts_created == 0:
            return

        from modules.hippocampal_index import binary_recall
        # Use consolidation's batch topic as query
        batch_topic = data.get("batch_topic", "") or data.get("query", "")
        if not batch_topic:
            return

        br_hits = binary_recall(batch_topic, max_results=3)
        if br_hits:
            session_topics = []
            for h in br_hits:
                matched = h.get('matched_roles', {})
                topics = matched.get('topics', [])
                session_topics.extend(topics)

            if session_topics:
                from modules.working_memory import push_to_working_memory
                unique_topics = list(dict.fromkeys(session_topics))[:5]
                push_to_working_memory(
                    content=f"[CX-32 SESSION CONTEXT] Consolidation topics found in sessions: {', '.join(unique_topics)}",
                    topic="consolidation",
                    relevance=0.4,
                    source="fhrr_consolidation_enrichment",
                )
                report_effective_fire('CX-32')
                _logger.debug("CX-32: Enriched consolidation with session topics: %s", unique_topics)
    except Exception as e:
        _logger.error("CX-32 _on_consolidation_enrich_fhrr error: %s", redact_secrets(str(e)))


def _on_memory_retrieved_fhrr(event_name: str, data: dict):
    """CX-33: Session index competes in GNW (Baars 1988 + HippoRAG 2024).

    When memories are retrieved, check FHRR index for matching sessions
    and push session context as a GNW candidate. This gives the workspace
    access to session-level context alongside individual memories.

    Paper: Baars 1988 (Global Workspace Theory), HippoRAG NeurIPS 2024
    """
    try:
        query = data.get("query", "")
        if not query:
            return

        from modules.hippocampal_index import binary_recall
        br_hits = binary_recall(query, max_results=2)
        if not br_hits:
            return

        top = br_hits[0]
        score = top.get('score', 0)
        if score < 0.1:
            return

        # Push session context to working memory as a candidate for GNW
        session_id = top.get('session_id', '')[:8]
        matched = top.get('matched_roles', {})
        preview = (top.get('context', {}).get('texts', [''])[0] or '')[:200]
        roles_str = ", ".join(f"{k}: {v}" for k, v in matched.items() if v)

        from modules.working_memory import push_to_working_memory
        push_to_working_memory(
            content=f"[SESSION CONTEXT] {session_id}: {roles_str}. {preview}",
            topic="session_index",
            relevance=min(0.5, score),
            source="fhrr_gnw_candidate",
        )
        report_effective_fire('CX-33')
        _logger.debug("CX-33: Pushed session %s (score=%.2f) to WM for GNW", session_id, score)
    except Exception as e:
        _logger.error("CX-33 _on_memory_retrieved_fhrr error: %s", redact_secrets(str(e)))


def _on_session_indexed_novelty(event_name: str, data: dict):
    """CX-35: Schema novelty drives prediction error (van Kesteren 2012).

    After a session is indexed, compute its novelty against schema prototypes.
    High novelty (> 0.6) fires prediction error — novel sessions are memorable.

    Paper: van Kesteren 2012 (mPFC schema congruency),
           Kumaran & Maguire 2007 (hippocampal match-mismatch)
    """
    try:
        session_id = data.get("session_id", "")
        if not session_id:
            return

        from modules.hippocampal_index import load_session_record, compute_novelty_score
        record = load_session_record(session_id)
        if not record or record.get('num_turns', 0) == 0:
            return

        novelty = compute_novelty_score(record)

        if novelty > 0.6:
            from modules.prediction import record_surprise
            record_surprise(
                expected="schema_consistent",
                actual=f"novel_session_{session_id[:8]}",
                intensity="high" if novelty > 0.8 else "medium",
            )

        from modules.working_memory import push_to_working_memory
        push_to_working_memory(
            content=f"[CX-35] Session {session_id[:8]} novelty: {novelty:.2f} ({'high' if novelty > 0.6 else 'low'})",
            topic="session_index",
            relevance=min(0.6, novelty),
            source="fhrr_novelty_detection",
        )
        _logger.debug("CX-35: Session %s novelty=%.2f", session_id[:8], novelty)
    except Exception as e:
        _logger.error("CX-35 _on_session_indexed_novelty error: %s", redact_secrets(str(e)))


# ============================================================
# ATTENTION SCHEMA (Graziano 2013)
# ============================================================

# Internal model of what the system is attending to
_attention_schema = {
    "current_focus": None,       # S: Topic currently being attended (Subject)
    "focus_strength": 0.0,       # A: How strongly focused (0-1) (Attention intensity)
    "focus_started": None,       # When current focus began
    "driver": None,              # What caused current focus
    "focus_value": 0.0,          # V: Value/relevance score (Sprint 3, item 3.5)
    # V = PE magnitude or EFE, measures WHY this deserves attention
    # Graziano 2013: complete attention schema = S (what) + A (how much) + V (why)
    "history": [],               # Last 20 focus states
    "topic_transitions": [],     # Topic A -> Topic B log (last 30)
    "interrupted_topics": [],    # Topics displaced by new focus
    "suppressed_items": [],      # 3B: Last 10 competition losers (Graziano AST)
    "last_predicted_focus": None,    # AST-1: what we predicted next
    "last_actual_focus": None,       # AST-1: what actually happened
    "attention_prediction_error": 0.0,  # AST-1: PE magnitude (0=match, 1=mismatch)
}


def _update_attention_schema(focus: str, driver: str = "unknown", strength: float = 0.5,
                             value: float = 0.0):
    """Update the attention schema with a new focus.

    Tracks topic transitions and interrupted topics.
    AST-1 closed loop: captures prediction BEFORE update, computes PE AFTER.
    S+A+V (Graziano 2013): S=focus (what), A=strength (how much), V=value (why).
    V represents information value -- PE magnitude, EFE, or goal relevance.
    """
    global _attention_schema
    now = now_iso()

    # --- AST-1: Capture prediction BEFORE mutating focus ---
    pred_topic, pred_prob = predict_next_focus()

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
            "value": _attention_schema["focus_value"],  # V in history
            "started": _attention_schema["focus_started"],
            "ended": now,
            "driver": _attention_schema["driver"],
        })
        _attention_schema["history"] = _attention_schema["history"][-20:]

    # Update current focus (S + A + V)
    _attention_schema["current_focus"] = focus       # S: what
    _attention_schema["focus_strength"] = strength   # A: how much
    _attention_schema["focus_value"] = value          # V: why (PE, EFE, relevance)
    _attention_schema["focus_started"] = now
    _attention_schema["driver"] = driver

    # --- AST-1: Compute attention prediction error AFTER update ---
    # Anti-spam: only emit when prediction is non-empty AND confident enough
    if pred_topic and pred_prob >= 0.5:
        error = 0.0 if pred_topic.lower() == focus.lower() else 1.0
        _attention_schema["last_predicted_focus"] = pred_topic
        _attention_schema["last_actual_focus"] = focus
        _attention_schema["attention_prediction_error"] = error
        event_bus.emit(Events.ATTENTION_PREDICTION_ERROR, {
            "predicted": pred_topic,
            "actual": focus,
            "error": error,
            "pred_prob": pred_prob,
        })

        # --- AST-1 adaptation: decay wrong edge on mismatch (Graziano 2013) ---
        # If predicted B but got C, remove oldest A->B transition so the
        # bigram predictor self-corrects. A->C is already added above.
        if error == 1.0 and old_focus:
            transitions = _attention_schema["topic_transitions"]
            old_lower = old_focus.lower()
            pred_lower = pred_topic.lower()
            for i, t in enumerate(transitions):
                if t.get("from", "").lower() == old_lower and t.get("to", "").lower() == pred_lower:
                    transitions.pop(i)
                    break  # Remove only the oldest occurrence

            # Hebbian strengthening: A->C actually happened, boost its weight
            # by appending a second entry. The bigram predictor counts occurrences,
            # so an extra entry shifts probability toward the correct transition.
            if focus:
                transitions.append({
                    "from": old_focus,
                    "to": focus,
                    "at": now,
                    "driver": "hebbian_boost",
                })
                _attention_schema["topic_transitions"] = transitions[-30:]


def get_attention_schema() -> dict:
    """Return the current attention schema state (for external queries)."""
    return {
        "current_focus": _attention_schema["current_focus"],
        "focus_strength": _attention_schema["focus_strength"],
        "focus_value": _attention_schema["focus_value"],  # V: why
        "focus_started": _attention_schema["focus_started"],
        "driver": _attention_schema["driver"],
        "recent_transitions": _attention_schema["topic_transitions"][-5:],
        "interrupted_topics": _attention_schema["interrupted_topics"][-3:],
        "suppressed_items": _attention_schema["suppressed_items"][-5:],
        "history_length": len(_attention_schema["history"]),
        # AST-1 prediction error fields (backward-compat via .get)
        "last_predicted_focus": _attention_schema.get("last_predicted_focus"),
        "last_actual_focus": _attention_schema.get("last_actual_focus"),
        "attention_prediction_error": _attention_schema.get("attention_prediction_error", 0.0),
    }


def _decay_attention_schema(max_transition_age_hours: float = 24.0, max_history: int = 20):
    """Remove stale entries from the attention schema.

    Called periodically (e.g. from sleep loop) to prevent unbounded growth
    of topic_transitions and history over long sessions.

    - topic_transitions: drop entries older than max_transition_age_hours.
    - history: trim to max_history most recent entries.
    """
    global _attention_schema
    from datetime import datetime, timedelta
    from modules.config import now_col

    cutoff = now_col() - timedelta(hours=max_transition_age_hours)

    transitions = _attention_schema.get("topic_transitions", [])
    original_len = len(transitions)
    kept = []
    for t in transitions:
        at_str = t.get("at")
        if not at_str:
            kept.append(t)
            continue
        try:
            at_dt = datetime.fromisoformat(at_str)
            # Make offset-naive for comparison if cutoff is offset-aware
            if at_dt.tzinfo is not None and cutoff.tzinfo is not None:
                if at_dt >= cutoff:
                    kept.append(t)
            else:
                if at_dt.replace(tzinfo=None) >= cutoff.replace(tzinfo=None):
                    kept.append(t)
        except (ValueError, TypeError):
            kept.append(t)  # Keep entries with unparseable timestamps

    _attention_schema["topic_transitions"] = kept
    pruned_transitions = original_len - len(kept)

    # Trim history to most recent entries
    history = _attention_schema.get("history", [])
    history_before = len(history)
    _attention_schema["history"] = history[-max_history:]
    pruned_history = history_before - len(_attention_schema["history"])

    if pruned_transitions > 0 or pruned_history > 0:
        _logger.debug(
            "attention decay: removed %d old transitions (>%dh), trimmed %d history entries",
            pruned_transitions, int(max_transition_age_hours), pruned_history,
        )


def describe_attention() -> str:
    """Self-descriptive report of attention state (Graziano 2013 AST, Phase 3B).

    Generates a higher-order representation of the attention process itself.
    This is the HOT (Higher-Order Thought) component -- the system
    describes what it is attending to, why, and what it predicts next.
    """
    focus = _attention_schema["current_focus"]
    strength = _attention_schema["focus_strength"]
    value = _attention_schema["focus_value"]
    driver = _attention_schema["driver"]
    suppressed = _attention_schema["suppressed_items"]

    if not focus:
        return "No tengo foco de atencion activo. Estoy en estado difuso."

    # Build self-description (S + A + V)
    parts = [f"Estoy atendiendo a '{focus}' (fuerza: {strength:.2f}, valor: {value:.2f})"]

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
    # Normalize across emitters: preturn_inject uses error_magnitude,
    # record_surprise uses confidence/intensity
    raw_error_magnitude = data.get("error_magnitude")
    raw_confidence = data.get("confidence")
    if raw_error_magnitude is not None:
        error_magnitude = raw_error_magnitude
    elif raw_confidence is not None:
        error_magnitude = raw_confidence
    else:
        error_magnitude = 0.0
    topic = data.get("topic", "unknown")

    # L9→L4: Asymmetric self-prediction precision (Kube 2020, Sharot 2011)
    # Self-relevant positive PEs get mild precision boost (healthy optimism bias)
    try:
        domain = data.get("domain", topic).lower()
        if domain and _cx26_core_values and any(
            v.lower() in domain for v in _cx26_core_values if isinstance(v, str)
        ):
            if data.get("valence") == "positive":
                error_magnitude = min(1.0, error_magnitude * 1.05)
    except Exception:
        pass

    # Build keywords from available data
    actual_keywords = data.get("actual_keywords", [])
    if not actual_keywords:
        actual_text = data.get("actual", "")
        if actual_text:
            actual_keywords = [w for w in actual_text.split()[:5] if len(w) > 3]

    # Effect 1: Push surprise to working memory (Schultz 1997)
    if error_magnitude > 0.3 and actual_keywords:
        try:
            from modules.working_memory import push_to_working_memory
            push_to_working_memory(
                content=f"[PREDICTION ERROR] Surprise en topic '{topic}': {', '.join(actual_keywords[:5])}",
                topic="prediction_surprise",
                relevance=min(0.95, 0.6 + error_magnitude),
                source="prediction_error",
            )
        except Exception as e:
            _logger.error("_on_prediction_error WM push error: %s", redact_secrets(str(e)))

    # Effect 2: Surprise captures attention (Corbetta & Shulman 2002)
    if error_magnitude > 0.3:
        try:
            _update_attention_schema(
                focus=f"surprise:{topic}",
                driver="prediction_error",
                strength=min(1.0, error_magnitude),
                value=min(1.0, error_magnitude),  # V: PE IS the information value signal
            )
        except Exception as e:
            _logger.error("_on_prediction_error attention error: %s", redact_secrets(str(e)))

    # Effect 3: DISABLED (Proposal #59) — Topic PE reflects Markov model error,
    # not memory content error. FTS keyword targeting punishes innocent bystander
    # memories. Real reconsolidation should be triggered by CONTRADICTION detection
    # (Path B), which remains active. See: Exton-McGuinness 2015, Lee 2009.
    pass


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
    global _interaction_count, _session_interaction_count, _self_model_tick, _last_self_model_refresh
    _interaction_count += 1

    # --- HOT-1: Periodic self-model refresh (Rosenthal 2005) ---
    # Runs BEFORE early return so it triggers on every interaction, not just long gaps.
    _self_model_tick += 1
    if (_self_model_tick % _SELF_MODEL_INTERVAL == 0
            and (time.monotonic() - _last_self_model_refresh) >= _SELF_MODEL_COOLDOWN):
        _last_self_model_refresh = time.monotonic()
        try:
            from modules.consciousness import reflect_on_self
            summary = reflect_on_self()
            # Sprint 2 FIX CX-10: include discrepancy data so CX-10 can fire
            disc_count = 0
            disc_domains = []
            try:
                from modules.self_model import detect_self_discrepancies
                disc = detect_self_discrepancies()
                disc_count = disc.get("count", 0) if disc else 0
                disc_domains = disc.get("domains", []) if disc else []
            except Exception:
                pass
            event_bus.emit(Events.SELF_MODEL_REFRESHED, {
                "tick": _self_model_tick,
                "source": "hot1_periodic",
                "summary_len": len(summary) if summary else 0,
                "discrepancy_count": disc_count,
                "domains": disc_domains,
            })
        except Exception as e:
            _logger.error("HOT-1 self-model refresh error: %s", redact_secrets(str(e)))

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
            from modules.working_memory import wm_apply_decay
            # Exponential decay: relevance * exp(-lambda * hours)
            # lambda=0.1: ~60% at 5h, ~37% at 10h, ~14% at 20h
            decay_multiplier = math.exp(-0.1 * elapsed_hours)
            wm_apply_decay(decay_multiplier)
        except Exception as e:
            _logger.error("WM decay error: %s", redact_secrets(str(e)))

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
            _logger.error("PM maintenance error: %s", redact_secrets(str(e)))

    # --- Emotional decay toward baseline (proportional to gap) ---
    if elapsed_hours >= 1.0:
        try:
            from modules.consciousness import apply_emotional_decay
            # Scale decay by time: 1 step per hour, max 10
            steps = min(int(elapsed_hours), 10)
            for _ in range(steps):
                apply_emotional_decay()
        except Exception as e:
            _logger.error("Emotional decay error: %s", redact_secrets(str(e)))

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
            _logger.error("Consolidation check error: %s", redact_secrets(str(e)))


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
                value=min(1.0, top_activation * 0.9),  # V: competitive activation as value
            )

        # 3B: Track suppressed items (Graziano AST - what we're NOT attending)
        loser_domains = data.get('loser_domains', [])
        if loser_domains:
            _attention_schema["suppressed_items"] = (
                _attention_schema["suppressed_items"] + loser_domains[:5]
            )[-10:]

        # Salience penalty for losers (inhibition of return) - batched
        if loser_ids:
            data["loser_penalty_applied"] = False
            try:
                from modules.config import COLLECTION_NAME
                from modules.access_tracking import record_spreading
                batch_ids = loser_ids[:20]
                points = _pg.get_by_ids(batch_ids)
                sal_updates = {
                    p.id: max(0.1, p.payload.get('attention_salience', 0.5) - 0.05)
                    for p in points
                }
                if sal_updates:
                    record_spreading(COLLECTION_NAME, sal_updates)
                    data["loser_penalty_applied"] = True
                else:
                    data["loser_penalty_error"] = "no_points_found"
            except Exception as e:
                data["loser_penalty_error"] = redact_secrets(str(e))
                _logger.warning(
                    "Loser salience penalty failed for %d losers: %s",
                    len(loser_ids),
                    data["loser_penalty_error"],
                    exc_info=True,
                )
    except Exception as e:
        _logger.error("Competition handler error: %s", redact_secrets(str(e)))


# ============================================================
# GWT-5: WORKSPACE RECRUITMENT HANDLER
# ============================================================

def _on_workspace_recruitment(event_name: str, data: dict):
    """GWT-5: Modules react to broadcast content (Baars 1988).

    After a memory is broadcast, other modules should update their state.
    This closes the cognitive loop: broadcast -> module reaction -> new candidates.
    """
    try:
        theme = data.get("broadcast_theme", "unknown")
        content = data.get("broadcast_content", "")

        if not content:
            return

        # WM push removed: _on_workspace_broadcast already pushes [BROADCAST] content
        # to working memory with 200 chars. This avoids duplicate WM entries.

        _logger.debug("Workspace recruitment: theme=%s", theme)
    except Exception as e:
        _logger.error("Recruitment handler error: %s", redact_secrets(str(e)))


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
        _logger.error("Retrieval quality handler error: %s", redact_secrets(str(e)))


# ============================================================
# WIRING REGISTRATION (called once at startup)
# ============================================================

# ============================================================
# HANDLER: CONTRADICTION_DETECTED (Phase 5 - Kumaran & Maguire 2007)
# ============================================================

def _on_contradiction_detected(event_name: str, data: dict):
    """When contradiction is detected at encoding time.

    3-Zone PE Model (Sevenster et al. 2014):
      Zone 1 (RETRIEVAL):      PE < 0.30  - no destabilization (handled upstream)
      Zone 2 (RECONSOLIDATE):  0.30-0.80  - mark labile, queue for deliberation
      Zone 3 (NEW LEARNING):   PE >= 0.80 - preserve original, create competing trace

    CA1 mismatch signal triggers:
    1. Attention capture (push to WM)
    2. Reconsolidation window (mark old memory as labile) -- Zone 2 only
    3. Competing trace linkage (contested_by field) -- Zone 3 only
    4. Attention schema update
    """
    try:
        pe = data.get("pe", 0.0)
        old_memory_id = data.get("conflicting_memory_id", "")
        old_text = data.get("conflicting_text", "")
        new_text = data.get("new_content", "")
        new_memory_id = data.get("new_memory_id", "")
        shared_entities = data.get("shared_entities", [])

        from modules.working_memory import push_to_working_memory
        from modules.config import (
            CONTRADICTION_PE_ALERT,
            RECONSOLIDATION_PE_ZONE_NEW_LEARNING,
        )

        if pe >= CONTRADICTION_PE_ALERT:
            # High PE: explicit alert
            alert = (
                f"[CONTRADICTION PE={pe:.2f}] "
                f"New: '{new_text[:100]}...' conflicts with "
                f"existing: '{old_text[:100]}...' "
                f"(entities: {', '.join(shared_entities[:5])})"
            )
            push_to_working_memory(
                content=alert,
                topic="contradiction",
                relevance=min(0.95, 0.6 + pe),
                source="contradiction_detector",
            )

            if old_memory_id:
                if pe >= RECONSOLIDATION_PE_ZONE_NEW_LEARNING:
                    # ── Zone 3: NEW LEARNING (Sevenster 2014) ──
                    # PE so high the original trace is NOT destabilized.
                    # Biology: this is extinction, not reconsolidation.
                    # Action: preserve original, link competing trace.
                    _logger.info(
                        "Zone 3 NEW LEARNING: PE=%.2f >= %.2f — "
                        "preserving original %s, linking contested_by=%s",
                        pe, RECONSOLIDATION_PE_ZONE_NEW_LEARNING,
                        old_memory_id[:8], new_memory_id[:8],
                    )
                    if new_memory_id:
                        try:
                            # Append to contested_by list on the original memory
                            pts = _pg.get_by_ids([old_memory_id])
                            contested = []
                            if pts:
                                existing = (pts[0].payload or {}).get("contested_by", [])
                                if isinstance(existing, list):
                                    contested = existing
                            contested.append({
                                "memory_id": new_memory_id,
                                "pe": round(pe, 3),
                                "timestamp": now_iso(),
                            })
                            _pg.update_payload(old_memory_id, {
                                "contested_by": contested,
                            })
                        except Exception as exc:
                            _logger.warning("Zone 3 contested_by failed: %s", exc)

                    # Queue for review with extended window
                    try:
                        from modules.consolidation import queue_correction_suggestion
                        channels = data.get("channels", {})
                        queue_correction_suggestion(
                            old_memory_id=old_memory_id,
                            old_text=old_text,
                            new_text=new_text,
                            prediction_error=pe,
                            new_memory_id=new_memory_id,
                            shared_entities=shared_entities,
                            channels=channels,
                            window_hours=48.0,
                        )
                    except Exception:
                        pass

                else:
                    # ── Zone 2: RECONSOLIDATE (Sevenster 2014) ──
                    # 0.30 <= PE < 0.80: mark labile, queue for deliberation.
                    # Do NOT auto-correct. Let the sleep loop's
                    # reconsolidation tick accumulate evidence within the
                    # 6-hour window before deciding.
                    _logger.info(
                        "Zone 2 RECONSOLIDATE: PE=%.2f — "
                        "marking %s labile, queuing for deliberation",
                        pe, old_memory_id[:8],
                    )
                    try:
                        from modules.consolidation import mark_as_labile
                        mark_as_labile(
                            memory_id=old_memory_id,
                            prediction_error=pe,
                            trigger_context=f"Inline PE at encoding: {new_text[:200]}"
                        )
                    except Exception:
                        pass

                    try:
                        from modules.consolidation import queue_correction_suggestion
                        channels = data.get("channels", {})
                        queue_correction_suggestion(
                            old_memory_id=old_memory_id,
                            old_text=old_text,
                            new_text=new_text,
                            prediction_error=pe,
                            new_memory_id=new_memory_id,
                            shared_entities=shared_entities,
                            channels=channels,
                        )
                    except Exception:
                        pass
        else:
            # Below CONTRADICTION_PE_ALERT: silent note
            push_to_working_memory(
                content=f"[PE NOTE] Tension detected (PE={pe:.2f}) with memory {old_memory_id[:8]}",
                topic="metamemory",
                relevance=0.5,
                source="contradiction_detector",
            )

        # Update attention schema
        _update_attention_schema(
            focus=f"contradiction:{','.join(shared_entities[:3])}",
            driver="contradiction_detected",
            strength=min(1.0, pe),
            value=min(1.0, pe),
        )

    except Exception as e:
        _logger.error("_on_contradiction_detected error: %s", redact_secrets(str(e)))


def _on_reconsolidation_triggered(event_name: str, data: dict):
    """When a memory is reconsolidated, update WM and attention (Nader 2000).

    Closes the loop: reconsolidation is not just a DB operation,
    the system should KNOW it happened and adjust attention.
    """
    try:
        memory_id = data.get("memory_id", "")[:8]
        old_conf = data.get("old_confidence", 0)
        new_conf = data.get("new_confidence", 0)
        from modules.working_memory import push_to_working_memory
        push_to_working_memory(
            content=f"[RECONSOLIDATION] Memory {memory_id} re-embedded, confidence {old_conf:.2f}->{new_conf:.2f}",
            topic="reconsolidation",
            relevance=0.7,
            source="reconsolidation_triggered",
        )
        _update_attention_schema(
            focus=f"reconsolidation:{memory_id}",
            driver="reconsolidation",
            strength=0.6,
            value=min(1.0, abs(new_conf - old_conf) * 2.0),  # V: confidence change magnitude
        )
    except Exception as e:
        _logger.error("_on_reconsolidation_triggered error: %s", redact_secrets(str(e)))


# ============================================================
# S3-04: ORPHAN EVENT HANDLERS
# ============================================================

def _on_session_close(event_name: str, data: dict):
    """S3-04: Session closing — push checkpoint to WM for persistence."""
    try:
        from modules.working_memory import push_to_working_memory
        reason = data.get("reason", "unknown")
        push_to_working_memory(
            content=f"[SESSION CLOSE] Session ended: {reason}",
            topic="session",
            relevance=0.6,
            source="session_close",
        )
    except Exception as e:
        _logger.error("_on_session_close error: %s", redact_secrets(str(e)))


def _on_session_close_encode_fhrr(event_name: str, data: dict):
    """CX-31: Hippocampal rapid encoding (McClelland 1995 CLS Theory).

    When a session closes, encode it as an FHRR session record.
    This is the computational analog of hippocampal rapid encoding —
    fast, index-like traces that point to neocortical content.

    Paper: McClelland et al. 1995, Diekelmann & Born 2010.
    """
    try:
        session_id = data.get("session_id", "")
        if not session_id:
            return

        from modules.hippocampal_index import encode_current_session
        record = encode_current_session(session_id)

        if record and record.get('num_turns', 0) > 0:
            from modules.events import event_bus, Events
            event_bus.emit(Events.SESSION_INDEXED, {
                'session_id': session_id,
                'num_turns': record['num_turns'],
                'num_chunks': len(record.get('chunks', [])),
                'topics': record.get('roles', {}).get('topics', [])[:5],
            })
            _logger.info("CX-31: FHRR encoded session %s (%d turns, %d chunks)",
                         session_id, record['num_turns'], len(record.get('chunks', [])))
    except Exception as e:
        _logger.error("CX-31 _on_session_close_encode_fhrr error: %s", redact_secrets(str(e)))


def _on_session_indexed(event_name: str, data: dict):
    """CX-31b: Session indexed — push notification to working memory."""
    try:
        from modules.working_memory import push_to_working_memory
        session_id = data.get("session_id", "")
        num_turns = data.get("num_turns", 0)
        topics = data.get("topics", [])
        topics_str = ", ".join(topics[:3]) if topics else "general"

        push_to_working_memory(
            content=f"[SESSION INDEXED] {session_id[:8]}: {num_turns} turns, topics: {topics_str}",
            topic="session",
            relevance=0.5,
            source="session_indexed",
        )
    except Exception as e:
        _logger.error("_on_session_indexed error: %s", redact_secrets(str(e)))


def _on_session_open(event_name: str, data: dict):
    """S3-04: Session opening — push bridge context to WM."""
    try:
        from modules.working_memory import push_to_working_memory
        bridge_summary = data.get("summary", "")
        if bridge_summary:
            push_to_working_memory(
                content=f"[SESSION BRIDGE] {bridge_summary[:200]}",
                topic="session",
                relevance=0.7,
                source="session_open",
            )
    except Exception as e:
        _logger.error("_on_session_open error: %s", redact_secrets(str(e)))


def _on_sleep_loop_complete(event_name: str, data: dict):
    """S3-04: Sleep loop finished — push maintenance summary to WM."""
    try:
        from modules.working_memory import push_to_working_memory
        ticks = data.get("ticks_executed", 0)
        duration = data.get("duration_seconds", 0)
        push_to_working_memory(
            content=f"[SLEEP LOOP] {ticks} ticks completed in {duration:.0f}s",
            topic="maintenance",
            relevance=0.5,
            source="sleep_loop_complete",
        )
        _decay_attention_schema()
    except Exception as e:
        _logger.error("_on_sleep_loop_complete error: %s", redact_secrets(str(e)))


def _on_curiosity_resolved(event_name: str, data: dict):
    """L6: Curiosity loop closure — push discovery to working memory.

    Closes the curiosity loop: question → explore → resolve → WM.
    This makes resolved discoveries available for prediction context
    and consolidation priority.
    """
    try:
        from modules.working_memory import push_to_working_memory
        question = data.get("question", "unknown")
        category = data.get("category", "general")
        answer_len = data.get("answer_length", 0)
        push_to_working_memory(
            content=f"[CURIOSITY RESOLVED] {question} (category={category}, answer={answer_len} chars)",
            topic=category,
            relevance=0.7,
            source="curiosity_resolved",
        )
        _logger.info("L6 curiosity loop: resolved '%s' → WM (category=%s)", question[:60], category)
    except Exception as e:
        _logger.warning("_on_curiosity_resolved error: %s", redact_secrets(str(e)))


def _on_affective_charge_wiring(event_name: str, data: dict):
    """L7: Active inference outcome — push AC + action to working memory.

    emotion.py already handles AC→PAD (Sprint 4.2).
    This handler makes the action selection VISIBLE in working memory
    so prediction context knows what action was taken and its AC signal.
    """
    try:
        ac = data.get("ac", 0)
        action = data.get("action", "unknown")
        topic = data.get("state_topic", "general")
        # Only push significant AC signals (avoid noise)
        if abs(ac) > 0.1:
            from modules.working_memory import push_to_working_memory
            push_to_working_memory(
                content=f"[ACTION] {action} on topic={topic}, AC={ac:.3f}",
                topic=topic,
                relevance=min(0.8, 0.4 + abs(ac)),
                source="active_inference",
            )
            _logger.info("L7 active inference: action=%s, AC=%.3f, topic=%s", action, ac, topic)
    except Exception as e:
        _logger.warning("_on_affective_charge_wiring error: %s", redact_secrets(str(e)))


def _on_memory_vaulted(event_name: str, data: dict):
    """L10: Forgetting homeostasis — track when memories are vaulted.

    Logs vault events so health_monitor can track forgetting rate
    and consolidation can adjust selectivity based on capacity pressure.
    """
    try:
        mid = data.get("memory_id", "?")
        _logger.info("L10 forgetting: memory %s vaulted (dormant)", mid[:12])
    except Exception as e:
        _logger.warning("_on_memory_vaulted error: %s", redact_secrets(str(e)))


def _on_memory_reactivated(event_name: str, data: dict):
    """L10: Forgetting homeostasis — track when dormant memories reactivate.

    Reactivation means the system needed something it had forgotten —
    signal that forgetting was too aggressive for this memory.
    """
    try:
        mid = data.get("memory_id", "?")
        hours = data.get("hours_dormant", 0)
        _logger.info("L10 forgetting: memory %s reactivated after %.1fh dormant", mid[:12], hours)
    except Exception as e:
        _logger.warning("_on_memory_reactivated error: %s", redact_secrets(str(e)))


def _on_metacognitive_control(event_name: str, data: dict):
    """L5+: Metacognitive control applied — log for evidence persistence.

    HOT-3 (Block 1995): metacognitive interventions should be tracked
    so the system can evaluate whether they helped.
    """
    try:
        strategy = data.get("strategy", "unknown")
        fok = data.get("fok_score")
        _logger.info("L5 metacognitive control: strategy=%s, fok=%.2f", strategy, fok or 0)
    except Exception as e:
        _logger.warning("_on_metacognitive_control error: %s", redact_secrets(str(e)))


def _on_self_model_refreshed(event_name: str, data: dict):
    """L9: Self-model identity loop — push refresh to working memory.

    Makes self-knowledge available in context so it can influence
    action selection and response style.
    """
    try:
        from modules.working_memory import push_to_working_memory
        source = data.get("source", "unknown")
        disc_count = data.get("discrepancy_count", 0)
        summary_len = data.get("summary_len", 0)
        content = f"[SELF-MODEL REFRESHED] source={source}"
        if disc_count:
            content += f", {disc_count} discrepancies detected"
        if summary_len:
            content += f", summary={summary_len} chars"
        push_to_working_memory(
            content=content,
            topic="self_model",
            relevance=0.6,
            source="self_model_refreshed",
        )
        _logger.info("L9 self-model loop: refresh from %s (discrepancies=%d)", source, disc_count)
    except Exception as e:
        _logger.warning("_on_self_model_refreshed error: %s", redact_secrets(str(e)))


def _on_perf_budget_violation(event_name: str, data: dict):
    """S3-04: Performance budget exceeded — log warning, push to WM."""
    try:
        from modules.working_memory import push_to_working_memory
        tool = data.get("tool_name", "unknown")
        elapsed_ms = data.get("duration_ms", 0)
        budget_ms = data.get("budget_ms", 0)
        push_to_working_memory(
            content=f"[PERF ALERT] {tool} took {elapsed_ms}ms (budget: {budget_ms}ms)",
            topic="performance",
            relevance=0.6,
            source="perf_budget_violation",
        )
        _logger.warning("Performance budget violation: %s took %dms (budget: %dms)",
                        tool, elapsed_ms, budget_ms)
    except Exception as e:
        _logger.error("_on_perf_budget_violation error: %s", redact_secrets(str(e)))


def _on_emotion_gating_applied(event_name: str, data: dict):
    """S3-04: Emotion gating applied — update attention schema (Bower 1981)."""
    try:
        p = data.get("current_pleasure", 0.0)
        a = data.get("current_arousal", 0.0)
        from modules.core.pad import classify_emotion
        emotion = classify_emotion(p, a, 0.0)
        _update_attention_schema(
            focus=f"emotion_gating:{emotion}",
            driver="emotion_gating",
            strength=0.6,
            value=0.5,
        )
    except Exception as e:
        _logger.error("_on_emotion_gating_applied error: %s", redact_secrets(str(e)))


def _on_attention_prediction_error(event_name: str, data: dict):
    """AST-1: Attention prediction error — reconnects a dead event to reconsolidation.

    When error == 1.0 (full mismatch), the system was confidently wrong about
    where attention would go. Log it and push to working memory so higher-order
    processes can notice the pattern.
    """
    try:
        predicted = data.get("predicted", "")
        actual = data.get("actual", "")
        error = data.get("error", 0.0)

        if error == 1.0:
            _logger.debug(
                "Attention PE full mismatch: predicted=%r actual=%r", predicted, actual
            )
            try:
                from modules.working_memory import push_to_working_memory
                push_to_working_memory(
                    content=(
                        f"[ATTENTION PE] Full mismatch: predicted '{predicted}' "
                        f"but focus shifted to '{actual}'"
                    ),
                    topic="attention_prediction_error",
                    relevance=0.7,
                    source="attention_schema",
                )
            except Exception as e:
                _logger.error(
                    "_on_attention_prediction_error WM push error: %s",
                    redact_secrets(str(e)),
                )
    except Exception as e:
        _logger.error("_on_attention_prediction_error error: %s", redact_secrets(str(e)))


# ============================================================
# CROSS-LOOP HANDLERS (TIER 1 — Neuroscience-backed)
# ============================================================

# CX-1 state: cooldown tracker per topic
_pe_curiosity_cooldowns = {}  # topic -> monotonic timestamp
_PE_CURIOSITY_COOLDOWN = 600  # 10 min per topic (avoid runaway curiosity)

# CX-4 state: vault tracking for consolidation urgency
_vault_tracker = {"count": 0, "important_count": 0, "window_start": 0.0}


def _compute_learning_progress(topic: str, window: int = 10):
    """Compute dPE/dt for a topic (Schmidhuber 2010 learning progress).

    Returns positive float if PE is decreasing (learning), negative if
    PE is increasing (getting worse), None if insufficient data.
    """
    try:
        from modules.config import FTS_DB_PATH
        from modules.db_pool import get_conn
        db_path = os.environ.get("FTS_DB_PATH", FTS_DB_PATH)
        if not os.path.exists(db_path):
            return None
        conn = get_conn(db_path)
        rows = conn.execute("""
            SELECT surprise_score FROM prediction_results
            WHERE actual_topic = ?
            -- Sprint 1 FIX CX-1: include all sources (99% of PEs are sleep_loop)
            ORDER BY id DESC LIMIT ?
        """, (topic, window)).fetchall()
        if len(rows) < 3:
            return None
        scores = [r[0] for r in rows]
        mid = len(scores) // 2
        early_avg = sum(scores[mid:]) / len(scores[mid:])
        recent_avg = sum(scores[:mid]) / len(scores[:mid])
        return early_avg - recent_avg  # positive = learning
    except Exception:
        return None


def _on_pe_drives_curiosity(event_name: str, data: dict):
    """CX-1: PE drives curiosity (Schmidhuber 2010, Gottlieb 2013, Kidd & Hayden 2015).

    When prediction error fires in the Goldilocks zone (not noise, not boring),
    and the topic shows positive learning progress, generate a curiosity question.
    Uses learning progress (dPE/dt) NOT raw PE — avoids chasing noise.
    """
    try:
        error_magnitude = data.get("error_magnitude", data.get("confidence", 0.0))
        topic = data.get("topic", "")

        if not topic or not error_magnitude:
            return

        # Goldilocks zone (Kidd & Hayden 2015, Kidd 2012): skip noise and boring
        # Lower bound 0.3 per neuro-skill v2 review (proposal 187)
        # P373 Fix 3: Sleep PE dampened 0.6x (Lewis & Durrant 2011).
        # Adjust floor to 0.18 for sleep sources so dampened PE can still pass.
        source = data.get("source", "")
        floor = 0.18 if source.startswith("sleep_") else 0.3
        if error_magnitude < floor or error_magnitude > 0.95:
            return

        # Cooldown per topic (prevent runaway curiosity)
        now = time.monotonic()
        last = _pe_curiosity_cooldowns.get(topic, 0)
        if now - last < _PE_CURIOSITY_COOLDOWN:
            return

        # Learning progress filter (Schmidhuber 2010): only if we're learning
        lp = _compute_learning_progress(topic)
        if lp is not None and lp <= 0.05:
            return  # PE not decreasing → noise domain, skip

        # Push curiosity question
        from modules.curiosity import push_curiosidad
        prioridad = "alta" if error_magnitude > 0.7 else "media"
        push_curiosidad(
            tema=f"PE alto en '{topic}' (mag={error_magnitude:.2f}, LP={'%.3f' % lp if lp is not None else '?'}). Que patron me falta?",
            prioridad=prioridad,
            categoria=topic if len(topic) < 30 else "general",
        )

        _pe_curiosity_cooldowns[topic] = now
        report_effective_fire('CX-1')
        _logger.info("CX-1 PE→curiosity: topic=%s, magnitude=%.2f, LP=%.3f",
                      topic, error_magnitude, lp or 0)
    except Exception as e:
        _logger.warning("CX-1 _on_pe_drives_curiosity error: %s", redact_secrets(str(e)))


def _on_curiosity_resolved_prediction(event_name: str, data: dict):
    """CX-2: Resolved curiosity boosts topic precision (Schwartenbeck 2015, Gruber 2019 PACE).

    When curiosity is resolved, increase precision weight for that topic
    in prediction_results. Higher precision → PE is weighted DOWN → effective
    surprise drops. The system becomes more confident about explored topics.
    """
    try:
        topic = data.get("category", "") or "general"  # Sprint 1 FIX CX-2: default empty category
        question = data.get("question", "")
        answer_length = data.get("answer_length", 0)

        if answer_length < 10:
            return  # Skip trivial resolutions

        # Boost precision for this topic in transition_stats
        # Higher observation count → Dirichlet concentration increases → precision up
        from modules.config import FTS_DB_PATH
        from modules.db_pool import get_conn
        db_path = os.environ.get("FTS_DB_PATH", FTS_DB_PATH)
        if not os.path.exists(db_path):
            return
        conn = get_conn(db_path)

        # Insert a synthetic "hit" for this topic — representing resolved knowledge
        # This increases accuracy in _get_high_surprise_domains, reducing curiosity_score
        conn.execute("""
            INSERT INTO prediction_results
            (predicted_topic, actual_topic, predicted_keywords, actual_keywords,
             surprise_score, precision_weight, weighted_surprise, hit, source, created_at)
            VALUES (?, ?, '', ?, 0.1, 1.0, 0.1, 1, 'curiosity_resolution', ?)
        """, (topic, topic, question[:100], now_iso()))
        conn.commit()

        report_effective_fire('CX-2')
        _logger.info("CX-2 curiosity→PE: topic=%s resolved, precision boosted (synthetic hit injected)",
                      topic)
    except Exception as e:
        _logger.warning("CX-2 _on_curiosity_resolved_prediction error: %s", redact_secrets(str(e)))


def _on_self_model_to_competition(event_name: str, data: dict):
    """CX-3: Self-model competes in GNW workspace (Graziano 2013, Luppi 2024).

    Three routes from neuroscience:
    1. DMN gateway — self-referential content has privileged access (Lou 2017)
    2. Metacognitive tag — confidence from model freshness (Shea & Frith 2019)
    3. Attention schema — S+A+V has recursive advantages (Graziano 2013)

    We implement route 1: inject self-summary as a WM item with activation
    bonus, so it competes naturally in next GNW competition cycle.
    """
    try:
        source = data.get("source", "")
        summary_len = data.get("summary_len", 0)
        disc_count = data.get("discrepancy_count", 0)

        if summary_len < 5:  # Sprint 1 FIX CX-3: lowered from 20 (self-refresh has value even short)
            return

        # Compute self-model confidence (Shea & Frith 2019):
        # High discrepancy = low confidence, low discrepancy = high confidence
        confidence = max(0.3, 1.0 - (disc_count * 0.15))

        # DMN gateway bonus (Lou 2017): self-content gets privileged activation
        W_SELF = 0.12
        relevance = min(0.85, 0.5 + W_SELF + (confidence * 0.2))

        # Push to WM with elevated relevance so it competes in next GNW round
        from modules.working_memory import push_to_working_memory
        push_to_working_memory(
            content=f"[SELF-AWARENESS] source={source}, confidence={confidence:.2f}, discrepancies={disc_count}",
            topic="self_model",
            relevance=relevance,
            source="self_model_gwt",
        )

        report_effective_fire('CX-3')
        _logger.info("CX-3 self→GNW: source=%s, confidence=%.2f, relevance=%.2f, discrepancies=%d",
                      source, confidence, relevance, disc_count)
    except Exception as e:
        _logger.warning("CX-3 _on_self_model_to_competition error: %s", redact_secrets(str(e)))


def _on_vault_tracks_urgency(event_name: str, data: dict):
    """CX-4a: Forgetting rate tracks consolidation urgency (Stickgold & Walker 2013, Feld & Born 2017).

    Each vault event increments the counter. Consolidation reads this
    to decide urgency. Uses importance-weighted counting to avoid
    false urgency from low-value memory decay.
    """
    try:
        now = time.monotonic()
        # Reset window every 24h
        if now - _vault_tracker["window_start"] > 86400:
            _vault_tracker["count"] = 0
            _vault_tracker["important_count"] = 0
            _vault_tracker["window_start"] = now

        _vault_tracker["count"] += 1

        # Importance-weighted: only count important vaults
        importance = data.get("importance", "normal")
        if importance in ("high", "critical"):
            _vault_tracker["important_count"] += 1

        report_effective_fire('CX-4a')
        _logger.debug("CX-4a vault tracker: total=%d, important=%d",
                       _vault_tracker["count"], _vault_tracker["important_count"])
    except Exception as e:
        _logger.warning("CX-4a _on_vault_tracks_urgency error: %s", redact_secrets(str(e)))


def get_consolidation_urgency() -> dict:
    """CX-4b: Read vault rate and compute consolidation urgency.

    Called by sleep_loop consolidation tick to adjust lookback/selectivity.
    Based on Stickgold & Walker 2013 triage + Feld & Born 2017 concurrent model.

    Returns dict with urgency level and adjustments, or None if normal.
    """
    try:
        important_vaults = _vault_tracker.get("important_count", 0)
        total_vaults = _vault_tracker.get("count", 0)

        if important_vaults < 3:
            return None  # Normal operation

        # Vault rate informs consolidation aggressiveness
        # Only important vaults trigger urgency (Ritvo 2019: nonmonotonic plasticity)
        urgency = "high" if important_vaults > 10 else "moderate"

        return {
            "lookback_multiplier": min(1.5, 1.0 + (important_vaults * 0.05)),
            "importance_adjust": max(-0.15, -0.01 * important_vaults),
            "vault_rate_total": total_vaults,
            "vault_rate_important": important_vaults,
            "urgency": urgency,
        }
    except Exception:
        return None


def _on_consolidation_protects_decay(event_name: str, data: dict):
    """CX-4b: Consolidation boosts SS, protecting from decay (Frey & Morris 1997, Tononi 2014).

    After successful consolidation, boost Storage Strength for consolidated
    memories. This implements synaptic tagging + capture: consolidation = PRP
    arrival = SS boost. Higher SS → slower RS decay in the Bjork model.

    The consolidated flag already reduces beta (decay exponent), but this
    additionally strengthens the trace itself — a double protection.
    """
    try:
        consolidated_ids = data.get("consolidated_ids", [])
        if not consolidated_ids:
            return

        SS_CONSOLIDATION_BOOST = 0.20  # Moderate boost per consolidation event
        boosted = 0

        for eid in consolidated_ids[:50]:  # Cap to avoid long loops
            try:
                points = _pg.get_by_ids([eid])
                if not points:
                    continue
                payload = points[0].payload or {}
                current_ss = float(payload.get("storage_strength", 0.3) or 0.3)
                # Boost SS (monotonic growth, capped at 1.0)
                new_ss = min(1.0, current_ss + SS_CONSOLIDATION_BOOST * (1.0 - current_ss))
                if new_ss > current_ss:
                    _pg.update_payload(eid, {"storage_strength": round(new_ss, 4)})
                    boosted += 1
            except Exception:
                continue

        if boosted:
            report_effective_fire('CX-4b')
            _logger.info("CX-4b consolidation→SS: boosted %d/%d memories (SS +%.2f)",
                          boosted, len(consolidated_ids), SS_CONSOLIDATION_BOOST)
    except Exception as e:
        _logger.warning("CX-4b _on_consolidation_protects_decay error: %s", redact_secrets(str(e)))


# ============================================================
# CROSS-LOOP HANDLERS (TIER 2 — Neuroscience-backed)
# ============================================================

# CX-8 constants
SS_RECONSOLIDATION_BOOST = 0.15  # Lower than consolidation (0.20) — Lee 2008

# CX-6 state: meta-temperature cache (pull model, refreshed every 60s)
_meta_efe_temperature = None
_meta_efe_last_refresh = 0.0
_META_EFE_REFRESH_INTERVAL = 60  # seconds

# CX-5 state: broadcast context for active inference
_broadcast_context = None
_PE_NOVELTY_GATE = 0.2  # Sprint 1 FIX CX-5: lowered from 0.5 (bigram predictor needs history to exceed 0.5)


def _on_reconsolidation_protects_decay(event_name: str, data: dict):
    """CX-8: Successful reconsolidation → SS boost (Lee 2008, Forcato 2011).

    GUARD: Only STRENGTHENING reconsolidation boosts SS.
    Boundary conditions (Lee 2009, Fernandez 2016):
    - action must be 'correct_memory' (restabilization, not weakening)
    - new_confidence >= 0.5 (successful restabilization)
    Cumulative: repeated reconsolidation strengthens cumulatively (Forcato 2011).
    """
    try:
        memory_id = data.get("memory_id")
        action = data.get("action", "")
        new_confidence = data.get("new_confidence", 0)

        # GUARD: Only strengthening reconsolidation (Lee 2008)
        if action != "correct_memory" or new_confidence < 0.5:
            return
        if not memory_id:
            return

        points = _pg.get_by_ids([memory_id])
        if not points:
            return

        payload = points[0].payload or {}
        current_ss = float(payload.get("storage_strength", 0.3) or 0.3)
        new_ss = min(1.0, current_ss + SS_RECONSOLIDATION_BOOST * (1.0 - current_ss))

        if new_ss > current_ss:
            _pg.update_payload(memory_id, {"storage_strength": round(new_ss, 4)})
            report_effective_fire('CX-8')
            _logger.info(
                "CX-8 reconsolidation→SS: id=%s, SS %.4f→%.4f, confidence=%.2f",
                memory_id[:12], current_ss, new_ss, new_confidence,
            )
    except Exception as e:
        _logger.warning("CX-8 error: %s", redact_secrets(str(e)))


def _refresh_meta_temperature():
    """CX-6: Compute meta-confidence-modulated temperature from L2 state.

    Reads prediction_state_l2 from SQLite, applies calibration correction
    (Boldt 2019, Maniscalco & Lau 2012), and derives EFE temperature.

    Low confidence → high temperature → more exploration
    High confidence → low temperature → more exploitation
    """
    global _meta_efe_temperature
    try:
        from modules.config import FTS_DB_PATH
        from modules.db_pool import get_conn
        db_path = os.environ.get("FTS_DB_PATH", FTS_DB_PATH)
        if not os.path.exists(db_path):
            _meta_efe_temperature = None
            return

        conn = get_conn(db_path)
        rows = conn.execute("""
            SELECT predicted_accuracy, actual_accuracy, sample_size
            FROM prediction_state_l2
            WHERE sample_size >= 5
        """).fetchall()

        if not rows:
            _meta_efe_temperature = None
            return

        # Weighted aggregate meta-confidence
        total_weight = sum(r[2] for r in rows)
        if total_weight == 0:
            _meta_efe_temperature = None
            return
        meta_conf = sum(r[0] * r[2] for r in rows) / total_weight
        actual_acc = sum(r[1] * r[2] for r in rows) / total_weight

        # CALIBRATION CORRECTION (Maniscalco & Lau 2012, Rosenbaum 2022)
        # Raw confidence is unreliable — shrink toward 0.5 by miscalibration
        calibration_error = abs(meta_conf - actual_acc)
        reliability = max(0.2, 1.0 - calibration_error)
        adjusted_conf = 0.5 + (meta_conf - 0.5) * reliability

        # Temperature modulation (Boldt 2019: beta=-0.59 for conf→exploration)
        TEMP_BASE = 4.0   # EFE_SOFTMAX_TEMPERATURE default
        TEMP_RANGE = 3.0   # ±range around base
        temp = TEMP_BASE + TEMP_RANGE * (1.0 - 2.0 * adjusted_conf)
        _meta_efe_temperature = max(1.0, min(7.0, temp))  # Floor + ceiling

        _logger.debug(
            "CX-6 meta→temp: conf=%.2f, actual=%.2f, cal_err=%.2f, adj=%.2f, temp=%.1f",
            meta_conf, actual_acc, calibration_error, adjusted_conf, _meta_efe_temperature,
        )
    except Exception:
        _meta_efe_temperature = None


def get_meta_efe_temperature():
    """CX-6: Return meta-confidence-modulated EFE temperature.

    Returns float (modulated temperature) or None (use default).
    Caches for 60 seconds to avoid per-call SQLite reads.
    Called by active_inference.select_action().
    """
    global _meta_efe_last_refresh
    now = time.monotonic()
    if now - _meta_efe_last_refresh < _META_EFE_REFRESH_INTERVAL:
        return _meta_efe_temperature
    _refresh_meta_temperature()
    _meta_efe_last_refresh = now
    return _meta_efe_temperature


def _on_gnw_broadcast_to_action(event_name: str, data: dict):
    """CX-5: GNW broadcast updates active inference state (Dehaene 2014, Friston 2015).

    GATED: Only for novel/conflicting situations (Norman & Shallice 1986).
    80%+ of actions don't need consciousness (Hommel 2013) — mandatory novelty gate.
    """
    global _broadcast_context
    try:
        winner_domains = data.get("winner_domains", [])
        top_activation = data.get("top_activation", 0)

        if not winner_domains:
            return

        # NOVELTY GATE: Only non-routine situations invoke workspace→action
        attention = get_attention_schema()
        current_pe = attention.get("attention_prediction_error", 0)
        # Sleep competition is inherently novel — no user query to be "routine" about.
        # Walker 2009: REM replay IS novel association. Skip PE gate for sleep.
        competition_id = data.get("competition_id", "")
        is_sleep = any(d == "working_memory" for d in winner_domains) and not attention.get("current_focus")
        if current_pe < _PE_NOVELTY_GATE and not is_sleep:
            _broadcast_context = None  # Routine — habitual pathway (Hommel 2013)
            return
        if is_sleep:
            current_pe = max(current_pe, top_activation * 0.3)  # Use activation as PE proxy

        _broadcast_context = {
            "broadcast_domains": winner_domains,
            "broadcast_activation": top_activation,
            "broadcast_topic": attention.get("current_focus", ""),
            "pe_magnitude": current_pe,
            "timestamp": time.monotonic(),
        }

        # Push broadcast summary to WM so it's visible in next cycle
        from modules.working_memory import push_to_working_memory
        push_to_working_memory(
            content=(
                f"[GNW→ACTION] Novel situation (PE={current_pe:.2f}): "
                f"domains={winner_domains[:3]}, activation={top_activation:.2f}"
            ),
            topic="gnw_broadcast",
            relevance=min(0.8, 0.5 + current_pe * 0.3),
            source="gnw_action_bridge",
        )

        report_effective_fire('CX-5')
        _logger.info("CX-5 GNW→AI: domains=%s, activation=%.2f, PE=%.2f",
                      winner_domains, top_activation, current_pe)
    except Exception as e:
        _logger.warning("CX-5 error: %s", redact_secrets(str(e)))


def get_broadcast_context():
    """CX-5: Get last GNW broadcast context for active inference, or None.

    Returns dict with broadcast_domains, broadcast_activation, pe_magnitude.
    Expires after 120 seconds (broadcast staleness).
    """
    if _broadcast_context is None:
        return None
    # Expire stale broadcasts
    age = time.monotonic() - _broadcast_context.get("timestamp", 0)
    if age > 120:
        return None
    return _broadcast_context


# ============================================================
# INHIBITORY MECHANISMS (MIN-1/MIN-2, Tononi & Cirelli 2006)
# ============================================================

SHY_DECAY_RATE = 0.02              # 2% per tick (Tononi & Cirelli 2006)
SHY_CONSOLIDATED_PROTECTION = 0.5  # Consolidated memories get half rate


def shy_downscale_ss(batch_size: int = 100) -> int:
    """MIN-2: SHY global SS downscaling (Tononi & Cirelli 2006).

    Called from sleep_loop homeostasis tick. Prevents SS saturation
    from CX-4b + CX-8 monotonic boosting. Consolidated memories are
    partially protected (Diekelmann & Born 2010).

    Returns number of memories downscaled.
    """
    try:
        points, _ = _pg.scroll(limit=batch_size, is_semantic=False)
        downscaled = 0
        for point in points:
            payload = point.payload or {}
            current_ss = float(payload.get("storage_strength", 0.3) or 0.3)
            if current_ss <= 0.1:
                continue
            consolidated = payload.get("consolidated", False)
            rate = SHY_DECAY_RATE * (SHY_CONSOLIDATED_PROTECTION if consolidated else 1.0)
            # CX-21: Causal hub protection (Kirkpatrick 2017 EWC analog)
            themes = payload.get("narrative_themes", [])
            if isinstance(themes, str):
                themes = [themes]
            for theme in (themes or []):
                cx21_mod = _cx21_protected_topics.get(theme)
                if cx21_mod is not None:
                    rate *= cx21_mod  # <1.0 = slower decay for causal hubs
                    break
            new_ss = max(0.1, current_ss * (1.0 - rate))
            if current_ss - new_ss > 0.001:
                _pg.update_payload(point.id, {"storage_strength": round(new_ss, 4)})
                downscaled += 1
        if downscaled:
            _logger.info("MIN-2 SHY downscale: %d memories, rate=%.3f", downscaled, SHY_DECAY_RATE)
        return downscaled
    except Exception as e:
        _logger.warning("MIN-2 SHY error: %s", redact_secrets(str(e)))
        return 0


# ============================================================
# CROSS-LOOP HANDLERS (TIER 3 — Neuroscience-backed)
# ============================================================

# CX-14 state: per-topic cooldown for gap curiosity
_cx14_recent_topics = {}  # topic_key -> time.time()
_CX14_MAX_QUESTIONS_PER_RUN = 3
_CX14_COOLDOWN_HOURS = 4

# CX-10 state: precision modifiers per domain
_cx10_precision_modifiers = {}  # domain -> float (1.0 = no modifier)
_CX10_PRECISION_FLOOR = 0.15
_CX10_DISC_WEIGHT = 0.08

# CX-11 state: curiosity observation buffer
_cx11_buffer = []
_CX11_CURIOSITY_WEIGHT = 1.5  # vs 1.0 observational (Gruber 2014)
_CX11_MAX_BUFFER = 20         # Batch threshold (Bramley 2015)

# CX-9 state: anti-rumination circuit breakers
_CX9_REFRACTORY_SECONDS = 45  # Neuro-grounded: 30-90s range (DMN dynamics)
_CX9_RELAXED_REFRACTORY = 20  # Homeostatic: relaxed after chronic blocking (Turrigiano 2004)
_CX9_BLOCK_THRESHOLD = 5      # Consecutive blocks before relaxation
_CX9_SALIENCE_THRESHOLD = 0.3  # Base threshold for PE*precision gate
_cx9_last_fire = 0.0
_cx9_fire_count_window = []
_cx9_dmn_call_count = 0        # Counter for DMN periodic refresh (separate from FIFO window)
_cx9_consecutive_blocks = 0    # Homeostatic plasticity counter (proposal 188)
_SELF_REF_KEYWORDS = {
    "self_model", "identity", "identidad", "capability", "performance",
    "consciousness", "consciencia", "codi", "self",
}

# CX-13 constants: PAD → EFE weights (Doya 2008 framework)
CX13_MAX_WEIGHT_DELTA = 0.4
CX13_PLEASURE_PRAG_SCALE = 0.30     # Vinckier 2018
CX13_AROUSAL_EPIST_SCALE = 0.35     # Aston-Jones & Cohen 2005
CX13_DOMINANCE_COST_SCALE = 0.25    # Lerner 2015
CX13_ANTI_PERSEVERATION_BOOST = 0.3  # Dreisbach & Goschke 2004
CX13_OPTIMAL_AROUSAL = 0.55          # Cools & D'Esposito 2011 inverted-U peak
_cx13_last_action = None
_cx13_consecutive_same = 0


def _on_consolidation_gaps_drive_curiosity(event_name: str, data: dict):
    """CX-14: Consolidation gaps → curiosity (Loewenstein 1994, Berlyne 1960).

    Three gap detection channels:
    1. Contradictions → high-priority D-type curiosity (Berlyne 1960)
    2. Low fact density → medium-priority I-type curiosity (Lewis & Durrant 2011)
    3. Bridge edges without shared facts → low-priority structural query (Ghosh & Gilboa 2014)

    Gated: max 3 questions per run, 4h cooldown per topic.
    PACE framework (Gruber 2019): only gaps above significance threshold.
    """
    try:
        scope = data.get("scope", "")
        if scope == "minimal":
            return

        contradictions = data.get("contradictions_found", 0)
        clusters = data.get("clusters_found", 0)
        facts = data.get("facts_extracted", 0)
        bridge_edges = data.get("bridge_edges", 0)

        questions_pushed = 0
        now = time.time()
        from modules.curiosity import push_curiosidad

        # Channel 1: Contradictions → high-priority
        if contradictions > 0 and questions_pushed < _CX14_MAX_QUESTIONS_PER_RUN:
            key = "contradictions"
            last = _cx14_recent_topics.get(key, 0)
            if now - last > _CX14_COOLDOWN_HOURS * 3600:
                push_curiosidad(
                    tema=f"Consolidation found {contradictions} contradiction(s). What is the ground truth?",
                    prioridad="alta",
                    categoria="consolidation_gap",
                )
                _cx14_recent_topics[key] = now
                questions_pushed += 1

        # Channel 2: Low fact density → medium-priority
        if clusters > 0 and questions_pushed < _CX14_MAX_QUESTIONS_PER_RUN:
            density = facts / clusters
            if density < 0.3:
                key = "low_density"
                last = _cx14_recent_topics.get(key, 0)
                if now - last > _CX14_COOLDOWN_HOURS * 3600:
                    push_curiosidad(
                        tema=f"Memory clusters have low fact density ({density:.1f}). What knowledge am I missing?",
                        prioridad="media",
                        categoria="consolidation_gap",
                    )
                    _cx14_recent_topics[key] = now
                    questions_pushed += 1

        # Channel 3: Bridge edges without shared facts → low-priority
        if bridge_edges > 3 and questions_pushed < _CX14_MAX_QUESTIONS_PER_RUN:
            key = "bridge_gaps"
            last = _cx14_recent_topics.get(key, 0)
            if now - last > _CX14_COOLDOWN_HOURS * 3600:
                push_curiosidad(
                    tema=f"Found {bridge_edges} bridge connections but few shared facts. Are these topics truly related?",
                    prioridad="baja",
                    categoria="consolidation_gap",
                )
                _cx14_recent_topics[key] = now
                questions_pushed += 1

        if questions_pushed:
            report_effective_fire('CX-14')
            _logger.info(
                "CX-14 consolidation→curiosity: %d questions (contradictions=%d, density=%.2f, bridges=%d)",
                questions_pushed, contradictions, facts / max(clusters, 1), bridge_edges,
            )
    except Exception as e:
        _logger.warning("CX-14 error: %s", redact_secrets(str(e)))


def _on_self_model_to_metacognition(event_name: str, data: dict):
    """CX-10A: Self-model discrepancies lower L2 precision (Nelson & Narens 1990 MONITORING).
    CX-10B: Systematic L2 bias triggers self-model reassessment (CONTROL signal).

    Direction A: discrepancy_count > 0 → reduce precision modifier per domain.
    Direction B: mean_bias > 0.25 → trigger self-model refresh.
    Precision floor 0.15 prevents confidence death spiral.
    """
    try:
        disc_count = data.get("discrepancy_count", 0)
        domains = data.get("domains", [])
        source = data.get("source", "")

        # Anti-echo: skip if triggered by CX-10B itself
        if source == "metacognitive_bias_cx10b":
            return

        if disc_count <= 0 or not domains:
            return

        # Direction A: Lower precision for discrepant domains
        for domain in domains[:5]:
            current = _cx10_precision_modifiers.get(domain, 1.0)
            reduction = _CX10_DISC_WEIGHT * min(disc_count, 5)
            new_mod = max(_CX10_PRECISION_FLOOR, current - reduction)
            _cx10_precision_modifiers[domain] = new_mod

        _logger.info("CX-10A self→meta: %d discrepancies in %s, modifiers updated",
                      disc_count, domains[:3])
        report_effective_fire('CX-10')

        # Direction B: Check for systematic L2 bias
        _check_metacognitive_bias()
    except Exception as e:
        _logger.warning("CX-10 error: %s", redact_secrets(str(e)))


def _check_metacognitive_bias():
    """CX-10B: Systematic L2 bias → self-model reassessment (Kruger & Dunning 1999)."""
    try:
        from modules.config import FTS_DB_PATH
        from modules.db_pool import get_conn
        db_path = os.environ.get("FTS_DB_PATH", FTS_DB_PATH)
        if not os.path.exists(db_path):
            return
        conn = get_conn(db_path)
        rows = conn.execute("""
            SELECT predicted_accuracy, actual_accuracy, sample_size
            FROM prediction_state_l2
            WHERE sample_size >= 10
        """).fetchall()
        if not rows:
            return
        total_w = sum(r[2] for r in rows)
        if total_w == 0:
            return
        mean_bias = sum((r[0] - r[1]) * r[2] for r in rows) / total_w
        if abs(mean_bias) > 0.25:
            event_bus.emit(Events.SELF_MODEL_REFRESHED, {
                "source": "metacognitive_bias_cx10b",
                "discrepancy_count": 0,
                "domains": [],
                "summary_len": 0,
            })
            _logger.info("CX-10B meta→self: systematic bias %.2f, triggering reassessment", mean_bias)
    except Exception:
        pass


def get_cx10_precision_modifier(domain: str) -> float:
    """CX-10: Read precision modifier for a domain. Called by preturn_inject."""
    return _cx10_precision_modifiers.get(domain, 1.0)


def _on_curiosity_feeds_causal(event_name: str, data: dict):
    """CX-11: Curiosity resolution → causal discovery (Bramley 2017, Steyvers 2003).

    Interventional data resolves Markov equivalence ambiguity that observational
    data cannot (Eberhardt & Scheines 2007). Curiosity-resolved observations get
    1.5x weight (Gruber 2014: curiosity-enhanced encoding). Buffer 20 before flush.
    """
    try:
        topic = data.get("category", "") or "general"  # Sprint 1 FIX CX-11: default empty category

        attention = get_attention_schema()
        from_topic = attention.get("current_focus", "general")

        _cx11_buffer.append({
            "from_topic": from_topic,
            "to_topic": topic,
            "weight": _CX11_CURIOSITY_WEIGHT,
        })

        if len(_cx11_buffer) >= _CX11_MAX_BUFFER:
            _flush_curiosity_to_causal()

        report_effective_fire('CX-11')
        _logger.info("CX-11 curiosity→causal: buffered %s→%s (%d/%d)",
                      from_topic, topic, len(_cx11_buffer), _CX11_MAX_BUFFER)
    except Exception as e:
        _logger.warning("CX-11 error: %s", redact_secrets(str(e)))


def _flush_curiosity_to_causal():
    """CX-11: Flush buffered curiosity observations to transition_stats."""
    global _cx11_buffer
    if not _cx11_buffer:
        return
    try:
        from modules.config import FTS_DB_PATH
        from modules.db_pool import get_conn
        db_path = os.environ.get("FTS_DB_PATH", FTS_DB_PATH)
        if not os.path.exists(db_path):
            _cx11_buffer = []
            return
        conn = get_conn(db_path)
        for obs in _cx11_buffer:
            conn.execute("""
                INSERT INTO transition_stats (from_topic, to_topic, count)
                VALUES (?, ?, ?)
                ON CONFLICT(from_topic, to_topic) DO UPDATE SET
                    count = count + excluded.count
            """, (obs["from_topic"], obs["to_topic"], int(obs["weight"])))
        conn.commit()
        _logger.info("CX-11 flush: %d observations → transition_stats", len(_cx11_buffer))
        _cx11_buffer = []
    except Exception:
        _cx11_buffer = []


def _compute_self_relevance(winner_domains: list, data: dict = None) -> float:
    """CX-9: Graded self-relevance score 0.0-1.0 (Northoff 2004 CMS analog).

    Sprint 1 FIX: also check winner content themes, not just domain labels.
    Domain labels ("episodic", "semantic") never match self-ref keywords.
    """
    if not winner_domains:
        return 0.0
    score = 0.0
    # Check domain labels (original)
    for domain in winner_domains:
        d = domain.lower() if isinstance(domain, str) else ""
        if d in _SELF_REF_KEYWORDS:
            score += 0.4
        elif any(kw in d for kw in _SELF_REF_KEYWORDS):
            score += 0.2
    # Sprint 1 FIX CX-9: check competition_id/topic for self-relevance
    if data and score < 0.1:
        global _cx9_dmn_call_count
        topic = str(data.get("broadcast_topic", data.get("competition_id", ""))).lower()
        if any(kw in topic for kw in _SELF_REF_KEYWORDS):
            score += 0.2
        # Baseline: every 10th low-score broadcast -> DMN periodic self-refresh
        # (Northoff 2004: default-mode periodic self-monitoring)
        _cx9_dmn_call_count += 1
        if _cx9_dmn_call_count % 10 == 0:
            score = max(score, 1.0)  # max relevance: bypass CB4, let kill-switch gate it
    return min(1.0, score)


def _on_workspace_to_self_model(event_name: str, data: dict):
    """CX-9: GNW broadcast → self-model refresh (Northoff 2004, Qin & Northoff 2011).

    Circuit breakers (Menon 2011 triple network model):
    1. Anti-echo: skip if broadcast is only self-model content (CX-3 origin)
    2. Refractory: ADAPTIVE cooldown (Turrigiano 2004 homeostatic plasticity)
       Base 45s, relaxes to 20s after 5 consecutive blocks (proposal 188)
    3. DAN/DMN anti-correlation: suppress when focus_value > 0.7 (ECN dominant)
    4. PE*precision salience gate (SN analog, Menon 2011)
    5. Kill switch: >3 fires in 30min → disable (Whitfield-Gabrieli 2012)
    """
    global _cx9_last_fire, _cx9_consecutive_blocks
    try:
        winner_domains = data.get("winner_domains", [])
        if not winner_domains:
            return

        # CB1: Anti-echo — sole self_model winner = CX-3 echo
        if len(winner_domains) == 1 and winner_domains[0].lower() == "self_model":
            _cx9_consecutive_blocks += 1
            return

        # CB2: Refractory period — ADAPTIVE (Turrigiano 2004)
        # After chronic blocking, relax from 45s to 20s (within physiological range, Fox 2005)
        now = time.monotonic()
        refractory = _CX9_RELAXED_REFRACTORY if _cx9_consecutive_blocks >= _CX9_BLOCK_THRESHOLD else _CX9_REFRACTORY_SECONDS
        if now - _cx9_last_fire < refractory:
            _cx9_consecutive_blocks += 1
            return

        self_relevance = _compute_self_relevance(winner_domains, data)
        if self_relevance < 0.1:
            _cx9_consecutive_blocks += 1
            return

        # CB3: DAN/DMN anti-correlation (Menon 2011)
        attention = get_attention_schema()
        focus_value = attention.get("focus_value", 0.0)
        if focus_value > 0.7:
            _cx9_consecutive_blocks += 1
            return

        # CB4: PE*precision salience gate (SN analog)
        try:
            from modules.config import _emotional_state
            arousal = _emotional_state.get("current", {}).get("arousal", 0.0)
            precision = max(0.3, 0.5 + arousal * 0.3)
        except Exception:
            precision = 0.5
        weighted_salience = self_relevance * precision
        if weighted_salience < _CX9_SALIENCE_THRESHOLD:
            _cx9_consecutive_blocks += 1
            return

        # CB5: Kill switch — >3 fires in 30min
        _cx9_fire_count_window.append(now)
        cutoff = now - 1800
        while _cx9_fire_count_window and _cx9_fire_count_window[0] < cutoff:
            _cx9_fire_count_window.pop(0)
        if len(_cx9_fire_count_window) > 3:
            _logger.warning("CX-9 kill switch: >3 fires in 30min, disabling")
            _cx9_consecutive_blocks += 1
            return

        # SUCCESS: reset homeostatic counter (Turrigiano 2004)
        if _cx9_consecutive_blocks >= _CX9_BLOCK_THRESHOLD:
            _logger.info("CX-9 homeostatic reset: was blocked %d times, refractory was relaxed to %ds",
                         _cx9_consecutive_blocks, _CX9_RELAXED_REFRACTORY)
        _cx9_consecutive_blocks = 0
        _cx9_last_fire = now

        # Background thread (reflect_on_self is heavy ~200-500ms)
        def _bg_reflect():
            try:
                from modules.self_model import reflect_on_self
                reflect_on_self(update_attention=False)
                _logger.info("CX-9 GNW→self: refresh complete (relevance=%.2f, salience=%.2f)",
                             self_relevance, weighted_salience)
            except Exception as e:
                _logger.warning("CX-9 bg reflect error: %s", redact_secrets(str(e)))

        t = threading.Thread(target=_bg_reflect, daemon=True, name="cx9-self-refresh")
        t.start()
        report_effective_fire('CX-9')
    except Exception as e:
        _logger.warning("CX-9 error: %s", redact_secrets(str(e)))


def compute_pad_efe_weights(model_observations: int = 0) -> dict:
    """CX-13: PAD-modulated EFE weights (Doya 2008 framework, Vinckier 2018).

    Pleasure → w_pragmatic (Vinckier 2018: DA reward sensitivity)
    Arousal → w_epistemic (inverted-U, Cools & D'Esposito 2011, Aston-Jones & Cohen 2005)
    Dominance → w_cost (Lerner 2015: control appraisal → risk-seeking)

    Anti-cascade dampening when CX-6 is already exploring (Schwabe & Wolf 2009).
    Confidence dampening with model maturity (Dunn 2006).
    Returns dict {w_pragmatic, w_epistemic, w_cost} or None.
    """
    try:
        from modules.config import _emotional_state
        current = _emotional_state.get("current", {})
        p = current.get("pleasure", 0.0)
        a = current.get("arousal", 0.0)
        d = current.get("dominance", 0.0)

        # Confidence dampening: emotions matter less with more data (Dunn 2006)
        confidence_dampen = 1.0 / (1.0 + 0.01 * model_observations)

        # Pleasure → pragmatic (Vinckier 2018)
        w_pragmatic = 1.0 + p * CX13_PLEASURE_PRAG_SCALE * confidence_dampen

        # Arousal → epistemic (inverted-U, Cools & D'Esposito 2011)
        a_norm = (a + 1.0) / 2.0  # map [-1,1] → [0,1]
        inverted_u = max(0.0, 1.0 - 4.0 * (a_norm - CX13_OPTIMAL_AROUSAL) ** 2)
        w_epistemic = 1.0 + inverted_u * CX13_AROUSAL_EPIST_SCALE * confidence_dampen

        # Dominance → cost (Lerner 2015: high control = risk-seeking = lower cost)
        w_cost = 1.0 - d * CX13_DOMINANCE_COST_SCALE * confidence_dampen

        # Anti-cascade dampening: if CX-6 already exploring (high temp),
        # reduce CX-13 epistemic deviation (Schwabe & Wolf 2009)
        meta_temp = get_meta_efe_temperature()
        if meta_temp is not None:
            cx6_exploration = (meta_temp - 4.0) / 4.0
            if cx6_exploration > 0:
                cx13_dampen = max(0.3, 1.0 - 0.5 * cx6_exploration)
                w_epistemic = 1.0 + (w_epistemic - 1.0) * cx13_dampen

        # Clamp + floor
        w_pragmatic = max(0.3, min(1.0 + CX13_MAX_WEIGHT_DELTA, w_pragmatic))
        w_epistemic = max(0.3, min(1.0 + CX13_MAX_WEIGHT_DELTA, w_epistemic))
        w_cost = max(0.3, min(1.0 + CX13_MAX_WEIGHT_DELTA, w_cost))

        return {"w_pragmatic": w_pragmatic, "w_epistemic": w_epistemic, "w_cost": w_cost}
    except Exception:
        return None


def track_action_for_perseveration(action_name: str) -> float:
    """CX-13: Track consecutive same actions for anti-perseveration (Dreisbach & Goschke 2004).
    Returns epistemic boost (added to w_epistemic) if perseverating."""
    global _cx13_last_action, _cx13_consecutive_same
    if action_name == _cx13_last_action:
        _cx13_consecutive_same += 1
    else:
        _cx13_last_action = action_name
        _cx13_consecutive_same = 1
    if _cx13_consecutive_same > 3:
        return CX13_ANTI_PERSEVERATION_BOOST
    return 0.0


# ============================================================
# CX-12: Retrieval → SS Boost (Testing Effect)
# Roediger & Karpicke 2006, Rowland 2014 (g=0.50, 159 studies)
# ============================================================

_CX12_SPACING_SECONDS = 60            # Minimum gap between SS boosts per memory (LTP refractory)
_CX12_USAGE_DAMPENING_C = 0.15        # Logarithmic dampening (Newell & Rosenbloom 1981)
_CX12_IDENTITY_SS_FLOOR = 0.3         # Permastore floor for critical memories (Bahrick 1984)
_cx12_last_boost: dict = {}            # memory_id -> time.time()
_cx12_topic_counts: dict = {}          # topic -> int (retrieval count for dampening)
_CX12_SEMAPHORE = threading.BoundedSemaphore(1)
_CX12_INTERNAL_SOURCES = {"workspace", "scroll", "internal", "focus_attention"}


def _on_retrieval_boosts_ss(event_name: str, data: dict):
    """CX-12: Successful retrieval → SS boost (Roediger & Karpicke 2006).

    Testing effect: retrieval strengthens storage more than re-study.
    Desirable difficulty: low RS at retrieval → higher SS gain (Bjork 1994).
    Usage dampening prevents rich-get-richer monopolization (Steyvers 2005).
    Spacing guard: 60s per memory prevents massed-retrieval inflation.
    Identity floor: critical memories never drop below SS=0.3 (Bahrick 1984).

    Independent of RIF (Storm & Levy 2012): RIF suppresses competitors'
    salience, CX-12 boosts targets' storage strength. Different dimensions.
    """
    source = data.get("source", "")
    if source in _CX12_INTERNAL_SOURCES:
        return

    retrieved_ids = data.get("retrieved_ids", [])
    if not retrieved_ids:
        return

    def _bg_ss_boost():
        if not _CX12_SEMAPHORE.acquire(blocking=False):
            return
        try:
            now = time.time()

            # Spacing guard: filter out recently-boosted IDs
            eligible_ids = [
                mid for mid in retrieved_ids
                if now - _cx12_last_boost.get(mid, 0) >= _CX12_SPACING_SECONDS
            ]
            if not eligible_ids:
                return

            # Usage dampening per query topic
            query = data.get("query", "general")
            topic_key = query[:50]  # Truncate for dict key
            _cx12_topic_counts[topic_key] = _cx12_topic_counts.get(topic_key, 0) + 1
            n_retrievals = _cx12_topic_counts[topic_key]
            dampening = 1.0 / (1.0 + _CX12_USAGE_DAMPENING_C * math.log(n_retrievals + 1))

            from modules.forgetting import compute_fadem_strength_ss_rs, SS_LEARNING_RATE

            points = _pg.get_by_ids(eligible_ids)
            boosted = 0
            for p in points:
                payload = p.payload or {}
                ss = float(payload.get("storage_strength", 0.3) or 0.3)
                rs = float(payload.get("retrieval_strength", 0.5) or 0.5)

                new_ss, new_rs, _ = compute_fadem_strength_ss_rs(
                    ss, rs, hours_since_access=0, retrieval_event=True,
                )

                # Apply usage dampening: reduce the SS gain, not the absolute value
                ss_gain = new_ss - ss
                dampened_ss = ss + ss_gain * dampening
                dampened_ss = min(1.0, dampened_ss)

                # Identity floor: critical memories never below 0.3 SS
                importance = payload.get("narrative_importance", "medium")
                if importance == "critical":
                    dampened_ss = max(_CX12_IDENTITY_SS_FLOOR, dampened_ss)

                _pg.update_payload(str(p.id), {
                    "storage_strength": round(dampened_ss, 4),
                    "retrieval_strength": round(new_rs, 4),
                })
                _cx12_last_boost[str(p.id)] = now
                boosted += 1

            # Prune spacing cache periodically
            if len(_cx12_last_boost) > 500:
                cutoff = now - 300
                stale = [k for k, v in _cx12_last_boost.items() if v < cutoff]
                for k in stale:
                    del _cx12_last_boost[k]

            if boosted:
                report_effective_fire('CX-12')
                _logger.info(
                    "CX-12 retrieval→SS: boosted %d/%d memories (dampening=%.2f, topic=%s)",
                    boosted, len(eligible_ids), dampening, topic_key[:20],
                )
        except Exception as e:
            _logger.warning("CX-12 error: %s", redact_secrets(str(e)))
        finally:
            _CX12_SEMAPHORE.release()

    t = threading.Thread(target=_bg_ss_boost, daemon=True, name="cx12-ss-boost")
    t.start()


# ============================================================
# CROSS-LOOP HANDLERS (TIER 4 — Multi-agent researched, dual-validated)
# 7 new cross-loops | ~65 papers | 2 INHIBITORY
# ============================================================

# ---- CX-20: L5→L6 — Metacognitive Uncertainty Drives Curiosity ----
# Loewenstein 1994 (information gap), Litman 2005, Boldt 2019
# 4/4 architecture support (SOAR impasse, ACT-R retrieval failure, LIDA codelets, CLARION MCS)
_CX20_CONF_THRESHOLD = 0.50         # Sprint 1 FIX CX-20: raised from 0.35 (FOK typically 0.4-0.7)
_CX20_MAX_PER_CYCLE = 2             # Max curiosity questions per metacognitive event
_CX20_COOLDOWN_PER_DOMAIN = 900     # 15 min between curiosity pushes per domain
_cx20_last_push: dict = {}           # domain -> time.time()


def _on_metacognitive_uncertainty_to_curiosity(event_name: str, data: dict):
    """CX-20: Low metacognitive confidence → D-type curiosity.

    Loewenstein 1994: information gap triggers deprivation-type curiosity.
    Litman 2005: curiosity and metacognition are functionally linked.
    Boldt 2019: confidence modulates exploration-exploitation.
    """
    strategy = data.get("strategy", "")
    fok_score = data.get("fok_score")
    if fok_score is None or fok_score > _CX20_CONF_THRESHOLD:
        return

    # Extract domain from strategy context
    domain = data.get("domain", strategy[:30] if strategy else "general")
    now = time.time()

    # Cooldown per domain
    if now - _cx20_last_push.get(domain, 0) < _CX20_COOLDOWN_PER_DOMAIN:
        return

    try:
        from modules.curiosity import push_curiosidad
        question = f"Explore uncertainty in domain '{domain}' — FOK={fok_score:.2f}, strategy={strategy}"
        push_curiosidad(
            tema=question,
            prioridad="media",
            categoria="consciencia",
        )
        _cx20_last_push[domain] = now
        _logger.info("CX-20 meta→curiosity: domain=%s, FOK=%.2f", domain[:20], fok_score)
    except Exception as e:
        _logger.warning("CX-20 error: %s", redact_secrets(str(e)))


# ---- CX-18: L1→L5 — Reconsolidation Lowers Metacognitive Confidence ----
# Nelson & Narens 1990 (metamemory framework), Fleming 2014, Exton-McGuinness 2015
_CX18_CONFIDENCE_REDUCTION = 0.15   # Precision reduction per reconsolidation event
_CX18_FLOOR = 0.2                   # Minimum precision (prevents confidence death spiral)


def _on_reconsolidation_to_metacognition(event_name: str, data: dict):
    """CX-18: Reconsolidation lowers metacognitive confidence for domain.

    Nelson & Narens 1990: memory correction must update monitoring.
    If a memory was wrong enough to trigger reconsolidation (PE >= 0.6),
    the metacognitive system should be LESS confident in that domain.

    Sprint B: Also handles action="extinction" from sleep loop.
    """
    action = data.get("action", "")
    if action not in ("correct_memory", "extinction"):
        return

    if action == "extinction":
        # Sleep-sourced: use PE directly (already dampened 0.6x by sleep loop)
        pe = data.get("prediction_error", 0)
        if pe < 0.1:
            return
    else:
        old_conf = data.get("old_confidence", 1.0)
        new_conf = data.get("new_confidence", 1.0)
        pe = abs(old_conf - new_conf)  # Proxy for prediction error magnitude
        if pe < 0.1:
            return

    memory_id = data.get("memory_id", "")

    # Determine domain from memory payload
    domain = "general"
    try:
        points = _pg.get_by_ids([memory_id])
        if points:
            payload = points[0].payload or {}
            themes = payload.get("narrative_themes", [])
            if isinstance(themes, list) and themes:
                domain = themes[0]
            elif isinstance(themes, str) and themes:
                domain = themes
    except Exception:
        pass

    # Lower L2 precision for the domain (same infrastructure as CX-10)
    current = _cx10_precision_modifiers.get(domain, 1.0)
    reduction = _CX18_CONFIDENCE_REDUCTION * min(pe * 2, 1.0)  # Scale by PE magnitude
    new_mod = max(_CX18_FLOOR, current - reduction)

    if abs(new_mod - current) > 0.001:
        _cx10_precision_modifiers[domain] = new_mod
        report_effective_fire('CX-18')
        _logger.info(
            "CX-18 recon→meta: domain=%s, precision %.2f→%.2f (PE=%.2f, action=%s)",
            domain[:20], current, new_mod, pe, action,
        )


# ---- CX-15: L9→L10 — Self-Model Protects/Prunes Memory (INHIBITORY) ----
# Sedikides & Green 2009 (mnemic neglect), Conway 2005, Anderson & Hanslmayr 2014
# FIRST INHIBITORY cross-loop in the system
_CX15_PROTECT_FACTOR = 0.3          # Decay reduction for self-affirming memories
_CX15_PRUNE_FACTOR = 1.5            # Decay acceleration for self-threatening (CAPPED)
_CX15_COOLDOWN = 1800               # 30 min between self-model forgetting scans
_CX15_PRUNE_FLOOR = 0.1             # Never prune below this SS
_cx15_last_scan = 0.0


def _on_self_model_modulates_forgetting(event_name: str, data: dict):
    """CX-15: Self-model protects identity-coherent memories (INHIBITORY).

    Sedikides & Green 2009: mnemic neglect — self-threatening memories
    are recalled less accurately. Conway 2005: self-memory system
    prioritizes self-coherence.

    FIRST INHIBITORY cross-loop. Breaks 100:0 E/I balance.
    Self-affirming → decay protection (reduce SHY rate).
    Self-threatening → accelerated decay (capped at 1.5x).
    """
    global _cx15_last_scan
    source = data.get("source", "")
    if source == "metacognitive_bias_cx10b":
        return  # Skip CX-10B feedback-triggered refreshes

    now = time.time()
    if now - _cx15_last_scan < _CX15_COOLDOWN:
        return
    _cx15_last_scan = now

    disc_count = data.get("discrepancy_count", 0)
    domains = data.get("domains", [])

    def _bg_self_forgetting():
        try:
            # Get identity memories
            identity_points, _ = _pg.scroll(
                filters={"narrative_importance": "critical"}, limit=50, is_semantic=False,
            )
            if not identity_points:
                return

            # Get self-model keywords from identity themes
            identity_keywords = set()
            for p in identity_points:
                payload = p.payload or {}
                themes = payload.get("narrative_themes", [])
                if isinstance(themes, list):
                    identity_keywords.update(themes)
                elif isinstance(themes, str):
                    identity_keywords.add(themes)
            identity_keywords.discard("")

            if not identity_keywords:
                return

            # Scan recent memories for self-relevance
            recent, _ = _pg.scroll(limit=50, is_semantic=False)
            protected = 0
            pruned = 0

            for p in recent:
                payload = p.payload or {}
                current_ss = float(payload.get("storage_strength", 0.3) or 0.3)
                importance = payload.get("narrative_importance", "medium")
                themes = payload.get("narrative_themes", [])
                if isinstance(themes, str):
                    themes = [themes]

                # Critical memories ALWAYS protected (identity floor)
                if importance == "critical":
                    continue

                # Check overlap with identity themes
                theme_overlap = len(set(themes) & identity_keywords) if themes else 0

                if theme_overlap > 0:
                    # Self-affirming: reduce decay rate via SS boost
                    boost = _CX15_PROTECT_FACTOR * (theme_overlap / max(len(identity_keywords), 1))
                    new_ss = min(1.0, current_ss + boost * (1.0 - current_ss))
                    if new_ss - current_ss > 0.005:
                        _pg.update_payload(str(p.id), {"storage_strength": round(new_ss, 4)})
                        protected += 1
                elif disc_count > 0 and domains:
                    # Self-threatening: memories in discrepancy domains get accelerated decay
                    if any(d in (themes or []) for d in domains):
                        decay = SHY_DECAY_RATE * _CX15_PRUNE_FACTOR
                        new_ss = max(_CX15_PRUNE_FLOOR, current_ss * (1.0 - decay))
                        if current_ss - new_ss > 0.005:
                            _pg.update_payload(str(p.id), {"storage_strength": round(new_ss, 4)})
                            pruned += 1

            if protected or pruned:
                report_effective_fire('CX-15')
                _logger.info(
                    "CX-15 self→forget: protected=%d, pruned=%d (INHIBITORY)",
                    protected, pruned,
                )
        except Exception as e:
            _logger.warning("CX-15 error: %s", redact_secrets(str(e)))

    t = threading.Thread(target=_bg_self_forgetting, daemon=True, name="cx15-self-forget")
    t.start()


# ---- CX-16: L3→L5 — Workspace Broadcasts Inform Metacognition ----
# Shea & Frith 2019 ("workspace needs metacognition"), Mashour 2020, Fleming 2012
# 4/4 architecture support
_CX16_STRENGTH_THRESHOLD = 0.3      # Only significant broadcasts trigger meta-evaluation


def _on_workspace_broadcast_to_metacognition(event_name: str, data: dict):
    """CX-16: GNW broadcast informs metacognitive monitoring.

    Shea & Frith 2019: workspace needs confidence tagging.
    Process-level metadata (coalition strength, novelty) that metacognition
    CANNOT get through indirect paths (L3→L9→L5 or L3→L4→L5).
    """
    top_activation = data.get("top_activation", 0.0)
    winner_count = data.get("winner_count", 0)
    loser_count = data.get("loser_count", 0)

    if top_activation < _CX16_STRENGTH_THRESHOLD:
        return

    # Compute workspace confidence from competition dynamics
    coalition_ratio = winner_count / max(winner_count + loser_count, 1)
    # Strong competition with clear winner → high workspace confidence
    # Weak competition or many losers → low workspace confidence
    workspace_conf = 0.5 * top_activation + 0.3 * coalition_ratio + 0.2 * (1.0 if loser_count > 0 else 0.5)

    # Get dominant domain from winners
    winner_domains = data.get("winner_domains", [])
    domain = winner_domains[0] if winner_domains else "general"

    # Modulate L2 precision via CX-10 infrastructure
    # Low workspace_conf → lower precision → system "knows it doesn't know"
    # Sprint 1 FIX CX-16: also boost precision on high confidence (Shea & Frith 2019)
    current = _cx10_precision_modifiers.get(domain, 1.0)
    if workspace_conf < 0.4:
        reduction = 0.05 * (0.4 - workspace_conf) / 0.4  # 0 to 0.05
        new_mod = max(_CX10_PRECISION_FLOOR, current - reduction)
    elif workspace_conf > 0.7:
        # High confidence → boost precision (system "knows it knows")
        boost = 0.03 * (workspace_conf - 0.7) / 0.3  # 0 to 0.03
        new_mod = min(1.0, current + boost)
    else:
        return  # Middle range: no change
    if abs(new_mod - current) > 0.001:
        _cx10_precision_modifiers[domain] = new_mod
        report_effective_fire('CX-16')
        _logger.info(
            "CX-16 GNW→meta: domain=%s, workspace_conf=%.2f, precision %.2f→%.2f",
            domain[:20], workspace_conf, current, new_mod,
        )


# ---- CX-19: L2→L9 — Consolidation Feeds Self-Model ----
# Conway 2005 (SMS), Northoff 2004, Klein & Lax 2010, McAdams 2001
# Addresses noetic-autonoetic gap (semantic self 0.88 >> episodic 0.64)
_CX19_MIN_FACTS = 2                 # Minimum new facts before feeding self-model
_CX19_SELF_KEYWORDS = {
    "codi", "identidad", "identity", "self", "consciencia", "consciousness",
    "personalidad", "personality", "capacidad", "capability", "preference",
    "preferencia", "valor", "value", "limitacion", "limitation",
}


def _on_consolidation_feeds_self_model(event_name: str, data: dict):
    """CX-19: Consolidated semantic facts update self-model.

    Conway 2005 SMS: episodic → semantic self-knowledge transition.
    Closes the documented noetic-autonoetic gap.
    """
    facts_created = data.get("facts_created", 0) + data.get("facts_updated", 0)
    if facts_created < _CX19_MIN_FACTS:
        return

    def _bg_feed_self():
        try:
            # Read recently created semantic facts
            from modules.config import connect_fts
            import modules.config as _cfg
            db_path = getattr(_cfg, 'FTS_DB_PATH', None)
            if not db_path:
                return

            conn = connect_fts(db_path)
            rows = conn.execute(
                "SELECT content, confidence, domain FROM codi_semantic "
                "ORDER BY created_at DESC LIMIT ?",
                (facts_created * 2,),
            ).fetchall()
            conn.close()

            self_relevant = []
            for row in rows:
                content = (row[0] or "").lower()
                if any(kw in content for kw in _CX19_SELF_KEYWORDS):
                    self_relevant.append({
                        "content": row[0],
                        "confidence": row[1] if len(row) > 1 else 0.5,
                        "domain": row[2] if len(row) > 2 else "general",
                    })

            if not self_relevant:
                return

            # Queue for self-model update
            from modules.self_model import update_self_model
            for fact in self_relevant[:3]:  # Cap at 3 per consolidation run
                aspect = "capacidad" if "capab" in fact["content"].lower() else "general"
                update_self_model(
                    insight=f"[CX-19 consolidation] {fact['content']}",
                    aspect=aspect,
                )

            report_effective_fire('CX-19')
            _logger.info("CX-19 consol→self: fed %d self-relevant facts", len(self_relevant[:3]))
        except Exception as e:
            _logger.warning("CX-19 error: %s", redact_secrets(str(e)))

    t = threading.Thread(target=_bg_feed_self, daemon=True, name="cx19-consol-self")
    t.start()


# ---- CX-17: L2→L4 — Consolidated Schemas Become Predictive Priors ----
# Tse 2007 (schemas ~50x acceleration), McClelland 1995 (CLS), Kumaran 2016
_CX17_SCHEMA_WEIGHT = 0.1           # WEAK priors (same philosophy as CX-7 kappa=0.1)
_CX17_MIN_CONFIDENCE = 0.7          # Schema must be consolidated with high confidence
_CX17_MIN_FACTS_FOR_SCHEMA = 3      # Need critical mass before updating priors
_cx17_schema_priors: dict = {}       # domain -> prior_boost (PULL MODEL for prediction)


def get_cx17_schema_prior(domain: str) -> float:
    """Get CX-17 schema prior boost for a domain. Returns 0.0 if no schema."""
    return _cx17_schema_priors.get(domain, 0.0)


def _on_consolidation_updates_prediction_priors(event_name: str, data: dict):
    """CX-17: Consolidated schemas become predictive priors.

    Tse 2007: schemas accelerate learning ~50x.
    CLS (McClelland 1995): consolidated neocortical representations
    become the prior structure for hippocampal encoding.

    PULL MODEL: stores schema priors in _cx17_schema_priors dict.
    Prediction system reads via get_cx17_schema_prior().
    WEAK priors (0.1) to prevent schema rigidity.
    """
    facts_created = data.get("facts_created", 0)
    if facts_created < _CX17_MIN_FACTS_FOR_SCHEMA:
        return
    # Sprint 1 FIX CX-17: allow any scope with enough facts (Tse 2007 schema acceleration)

    def _bg_update_priors():
        try:
            # Read recent high-confidence semantic facts by domain
            from modules.config import connect_fts
            import modules.config as _cfg
            db_path = getattr(_cfg, 'FTS_DB_PATH', None)
            if not db_path:
                return

            conn = connect_fts(db_path)
            rows = conn.execute(
                "SELECT domain, COUNT(*) as cnt FROM codi_semantic "
                "WHERE confidence >= ? "
                "GROUP BY domain HAVING cnt >= ?",
                (_CX17_MIN_CONFIDENCE, _CX17_MIN_FACTS_FOR_SCHEMA),
            ).fetchall()
            conn.close()

            if not rows:
                return

            updated = 0
            for row in rows:
                domain = row[0] or "general"
                fact_count = row[1]
                # Weak prior: scale by schema weight and fact count (log to prevent saturation)
                prior_boost = _CX17_SCHEMA_WEIGHT * math.log(fact_count + 1)
                prior_boost = min(prior_boost, 0.3)  # Cap
                _cx17_schema_priors[domain] = round(prior_boost, 4)
                updated += 1

            if updated:
                report_effective_fire('CX-17')
                _logger.info("CX-17 consol→pred: updated %d domain schema priors (weight=%.2f)", updated, _CX17_SCHEMA_WEIGHT)
        except Exception as e:
            _logger.warning("CX-17 error: %s", redact_secrets(str(e)))

    t = threading.Thread(target=_bg_update_priors, daemon=True, name="cx17-schema-priors")
    t.start()


# ---- CX-21: L8→L10 — Causal Centrality Protects Hub Memories (INHIBITORY) ----
# Kirkpatrick 2017 (EWC), Tononi & Cirelli 2014 (SHY), Tompary & Davachi 2017
# NOVEL mechanism — not in classical architectures. Second INHIBITORY.
_CX21_CENTRALITY_PROTECT = 0.5      # Decay rate multiplier for causal hubs (0.5 = half decay)
_CX21_TOP_PERCENTILE = 75           # Top quartile of centrality gets protection
_cx21_protected_topics: dict = {}    # topic -> protection_factor (refreshed on NOTEARS run)
_CX21_STALE_HOURS = 12              # Protection expires if NOTEARS hasn't run


def refresh_cx21_causal_protection():
    """CX-21: Read latest NOTEARS state and compute hub protection.

    Called from sleep_loop after causal_discovery tick.
    PULL model (like CX-13) — reads state instead of needing an event.

    Kirkpatrick 2017 (EWC): important parameters resist change.
    Analogy: causal hub topics are "important parameters" of the knowledge graph.
    """
    global _cx21_protected_topics
    try:
        from modules.config import connect_fts
        import modules.config as _cfg
        db_path = getattr(_cfg, 'FTS_DB_PATH', None)
        if not db_path:
            return

        conn = connect_fts(db_path)
        row = conn.execute(
            "SELECT w_matrix, topics FROM causal_discovery_state "
            "ORDER BY created_at DESC LIMIT 1",
        ).fetchone()
        conn.close()

        if not row:
            return

        import json
        import numpy as np

        w_matrix = np.array(json.loads(row[0]))
        topics = json.loads(row[1])
        n = len(topics)
        if n < 3:
            return

        # Compute betweenness-like centrality from weight matrix
        # Simple: out-degree weighted by edge strength
        out_strength = np.sum(np.abs(w_matrix), axis=1)
        in_strength = np.sum(np.abs(w_matrix), axis=0)
        centrality = out_strength + in_strength  # Total strength

        threshold = np.percentile(centrality, _CX21_TOP_PERCENTILE)

        new_protected = {}
        for i, topic in enumerate(topics):
            if centrality[i] >= threshold:
                # Higher centrality → stronger protection
                norm_cent = centrality[i] / (max(centrality) + 1e-9)
                protection = _CX21_CENTRALITY_PROTECT + (1.0 - _CX21_CENTRALITY_PROTECT) * (1.0 - norm_cent)
                new_protected[topic] = round(protection, 3)

        _cx21_protected_topics = new_protected
        if new_protected:
            _logger.info(
                "CX-21 causal→forget: %d hub topics protected (INHIBITORY), threshold=%.2f",
                len(new_protected), threshold,
            )
    except Exception as e:
        _logger.warning("CX-21 error: %s", redact_secrets(str(e)))


def get_cx21_decay_modifier(topic: str) -> float:
    """Get CX-21 decay modifier for a topic. Returns multiplier (1.0 = no change, <1.0 = protected)."""
    return _cx21_protected_topics.get(topic, 1.0)


# ============================================================
# CX-22: L7→L5 — Action Selection Informs Metacognition
# All 4 architectures: SOAR operator→monitoring, ACT-R production→conflict,
# LIDA action→sensory-motor, CLARION action→MCS
# ============================================================

_CX22_PERSEVERATION_THRESHOLD = 4   # Same action N times → lower precision
_CX22_LOW_SPREAD_THRESHOLD = 0.05   # EFE spread below this = indecisive selection
_cx22_action_history: list = []      # Last N actions for perseveration detection
_CX22_HISTORY_MAX = 10


def _on_action_selected_to_metacognition(event_name: str, data: dict):
    """CX-22: Action selection informs metacognitive monitoring (L7→L5).

    Closes the action-monitoring loop. Metacognition receives:
    1. Perseveration signal: same action repeated → lower precision (explore more)
    2. Indecision signal: low EFE spread → system unsure → lower precision
    3. Action-uncertainty mismatch: exploring when certain → flag for meta-review

    All 4 cognitive architectures require this feedback pathway.
    """
    # Sprint C: Gate sleep-sourced actions — brain does NOT apply metacognitive
    # monitoring to sleep actions (Louie & Wilson 2001)
    if data.get("source") == "sleep_loop":
        return

    action = data.get("action", "")
    efe_spread = data.get("efe_spread", 1.0)
    state_topic = data.get("state_topic", "general")

    # Track action history for perseveration detection
    _cx22_action_history.append(action)
    if len(_cx22_action_history) > _CX22_HISTORY_MAX:
        _cx22_action_history.pop(0)

    # Signal 1: Perseveration — same action repeated consecutively
    if len(_cx22_action_history) >= _CX22_PERSEVERATION_THRESHOLD:
        recent = _cx22_action_history[-_CX22_PERSEVERATION_THRESHOLD:]
        if all(a == recent[0] for a in recent):
            current = _cx10_precision_modifiers.get(state_topic, 1.0)
            new_mod = max(_CX10_PRECISION_FLOOR, current - 0.05)
            if abs(new_mod - current) > 0.001:
                _cx10_precision_modifiers[state_topic] = new_mod
                report_effective_fire('CX-22')
                _logger.info(
                    "CX-22 action→meta: perseveration detected (%s x%d), precision %.2f→%.2f",
                    action, _CX22_PERSEVERATION_THRESHOLD, current, new_mod,
                )
            return

    # Signal 2: Indecision — EFE spread too low = all options look the same
    if efe_spread < _CX22_LOW_SPREAD_THRESHOLD and efe_spread > 0:
        current = _cx10_precision_modifiers.get(state_topic, 1.0)
        new_mod = max(_CX10_PRECISION_FLOOR, current - 0.03)
        if abs(new_mod - current) > 0.001:
            _cx10_precision_modifiers[state_topic] = new_mod
            report_effective_fire('CX-22')
            _logger.info(
                "CX-22 action→meta: indecision (spread=%.4f), precision %.2f→%.2f",
                efe_spread, current, new_mod,
            )


# ============================================================
# CX-23: L10→L6 — Forgetting Suppresses Curiosity (INHIBITORY)
# Anderson & Hanslmayr 2014: RIF suppresses retrieval cues
# Loewenstein 1994: information gap requires awareness of gap
# Koriat 1993: below-threshold memories become invisible
# L10's FIRST outgoing connection — transforms sink to signaler
# ============================================================

_CX23_TTL = 86400  # 24h suppression window
_cx23_suppressed_topics: dict = {}  # topic -> timestamp


def _on_forgetting_suppresses_curiosity(event_name: str, data: dict):
    """CX-23: Vaulted memories suppress related curiosity (L10→L6).

    When a memory decays below accessibility threshold, the information gap
    that drives curiosity ceases to be computed. The agent no longer 'knows
    what it doesn't know.'
    """
    memory_id = data.get("memory_id", "")
    if not memory_id:
        return

    def _suppress():
        try:
            from modules.pg_store import pg as _pg
            points = _pg.get_by_ids([memory_id])
            payload = points[0].payload if points else None
            if not payload:
                return
            themes = payload.get("narrative_themes", [])
            if isinstance(themes, str):
                themes = [themes]
            if not themes:
                return
            now = time.time()
            for theme in themes:
                if theme:
                    _cx23_suppressed_topics[theme] = now
            # Prune expired topics
            expired = [t for t, ts in _cx23_suppressed_topics.items() if now - ts > _CX23_TTL]
            for t in expired:
                _cx23_suppressed_topics.pop(t, None)
            # Remove matching pending curiosities
            try:
                from modules.curiosity import _cargar_curiosidades, _guardar_curiosidades
                cdata = _cargar_curiosidades()
                before = len(cdata.get("pendientes", []))
                theme_set = set(t.lower() for t in themes if t)
                cdata["pendientes"] = [
                    item for item in cdata.get("pendientes", [])
                    if not any(t in item.get("pregunta", "").lower() or t in item.get("categoria", "").lower()
                              for t in theme_set)
                ]
                removed = before - len(cdata["pendientes"])
                if removed > 0:
                    _guardar_curiosidades(cdata)
                    report_effective_fire('CX-23')
                    _logger.info("CX-23 forget→curiosity: suppressed %d items for themes %s", removed, themes)
            except Exception:
                pass  # Suppression is best-effort
        except Exception as e:
            _logger.debug("CX-23 error: %s", e)

    threading.Thread(target=_suppress, daemon=True).start()


def is_curiosity_suppressed(topic: str) -> bool:
    """CX-23 PULL: Check if curiosity about topic should be suppressed."""
    ts = _cx23_suppressed_topics.get(topic)
    if ts is None:
        return False
    return (time.time() - ts) < _CX23_TTL


# ============================================================
# CX-28: L10→L5 — Forgetting Degrades Metacognitive Confidence (INHIBITORY)
# Koriat 1993: accessibility heuristic — retrieval fluency → confidence
# Hertzog 2023: degraded access → degraded metacognitive accuracy
# Creates chain: L10→L5 (lower confidence) →CX-20→ L6 (curiosity) → re-learn
# L10's SECOND outgoing connection
# ============================================================

_CX28_CONFIDENCE_REDUCTION = 0.1  # Per vault event
_CX28_FLOOR = 0.15               # Minimum meta-confidence (below CX-10 floor)


def _on_forgetting_degrades_metacognition(event_name: str, data: dict):
    """CX-28: Decayed memories lower domain metacognitive confidence (L10→L5).

    When memories vault, L5 confidence for that domain should degrade.
    This feeds into CX-20 (low confidence → curiosity) for healthy re-learning.
    """
    memory_id = data.get("memory_id", "")
    if not memory_id:
        return

    def _degrade():
        try:
            from modules.pg_store import pg as _pg
            points = _pg.get_by_ids([memory_id])
            payload = points[0].payload if points else None
            if not payload:
                return
            themes = payload.get("narrative_themes", [])
            if isinstance(themes, str):
                themes = [themes]
            for theme in (themes or []):
                if not theme:
                    continue
                current = _cx10_precision_modifiers.get(theme, 1.0)
                new_mod = max(_CX28_FLOOR, current - _CX28_CONFIDENCE_REDUCTION)
                if abs(new_mod - current) > 0.001:
                    _cx10_precision_modifiers[theme] = new_mod
                    report_effective_fire('CX-28')
                    _logger.info(
                        "CX-28 forget→meta: domain '%s' confidence %.2f→%.2f (vault event)",
                        theme, current, new_mod,
                    )
                break  # Only degrade for primary theme
        except Exception as e:
            _logger.debug("CX-28 error: %s", e)

    threading.Thread(target=_degrade, daemon=True).start()


# ============================================================
# CX-24: L5→L1 — High Metacognitive Confidence Blocks Reconsolidation (INHIBITORY)
# Suzuki 2004: memory strength constrains labilization
# Exton-McGuinness 2015: boundary conditions of reconsolidation
# PULL MODEL: stores blocked domains, reconsolidation checks gate
# Forms negative feedback loop with CX-18 (L1→L5)
# ============================================================

_CX24_CONFIDENCE_GATE = 0.85     # Only very high confidence blocks
_CX24_HYSTERESIS_LOW = 0.75      # Re-allow reconsolidation below this
_cx24_blocked_domains: dict = {}  # domain -> bool (True = blocked)


def _on_metacognition_gates_reconsolidation(event_name: str, data: dict):
    """CX-24: High metacognitive confidence blocks reconsolidation (L5→L1).

    PULL MODEL handler — updates the gate state based on precision changes.
    Reconsolidation callers should check is_reconsolidation_gated().
    """
    # This fires on METACOGNITIVE_CONTROL_APPLIED
    domain = data.get("domain", "general")
    if not domain:
        return
    precision = _cx10_precision_modifiers.get(domain, 1.0)

    currently_blocked = _cx24_blocked_domains.get(domain, False)

    if not currently_blocked and precision > _CX24_CONFIDENCE_GATE:
        _cx24_blocked_domains[domain] = True
        _logger.info("CX-24 meta→recon: BLOCKING reconsolidation for '%s' (precision=%.2f)", domain, precision)
    elif currently_blocked and precision < _CX24_HYSTERESIS_LOW:
        _cx24_blocked_domains[domain] = False
        _logger.info("CX-24 meta→recon: UNBLOCKING reconsolidation for '%s' (precision=%.2f)", domain, precision)


def is_reconsolidation_gated(domain: str) -> bool:
    """CX-24 PULL: Check if reconsolidation is blocked for a domain."""
    return _cx24_blocked_domains.get(domain, False)


# ============================================================
# CX-26: L9→L7 — Self-Model Suppresses Identity-Inconsistent Policies (INHIBITORY)
# Oyserman 2017: identity-based motivation
# Markus 1977: self-schemata bias action selection
# Seth & Friston 2016: self-model as allostatic prior in active inference
# PULL MODEL: stores identity constraints, active inference checks
# Establishes L9 as dual governance hub (CX-15 memory + CX-26 action)
# ============================================================

_CX26_CONSTRAINT_WEIGHT = 0.3  # Soft penalty, not hard veto
_CX26_MIN_BELIEFS = 2          # Require at least 2 relevant beliefs to fire
_cx26_identity_constraints: dict = {}  # action_pattern -> penalty_weight
_cx26_core_values: list = []   # Cached core values from self-model


def _on_self_model_constrains_action(event_name: str, data: dict):
    """CX-26: Self-model refresh updates identity constraints for action (L9→L7).

    Fires on SELF_MODEL_REFRESHED. Extracts core values and creates
    soft penalty constraints for policies inconsistent with identity.
    """
    # Prefer structured aspects if available (future path)
    aspects = data.get("aspects", {})
    values = aspects.get("valor", []) if isinstance(aspects, dict) else []
    preferences = aspects.get("preferencia", []) if isinstance(aspects, dict) else []
    beliefs = values + preferences

    # Fallback: use domains as identity constraint proxies
    if len(beliefs) < _CX26_MIN_BELIEFS:
        domains = data.get("domains", [])
        if len(domains) < _CX26_MIN_BELIEFS:
            return
        beliefs = domains

    _cx26_core_values.clear()
    _cx26_core_values.extend(beliefs[:10])

    # Store constraints as keyword-based patterns
    _cx26_identity_constraints.clear()
    for belief in beliefs[:10]:
        if isinstance(belief, str) and len(belief) > 2:
            _cx26_identity_constraints[belief.lower()] = _CX26_CONSTRAINT_WEIGHT

    if _cx26_identity_constraints:
        report_effective_fire('CX-26')
        _logger.info("CX-26 self→action: %d identity constraints active", len(_cx26_identity_constraints))


def get_cx26_identity_penalty(action_description: str) -> float:
    """CX-26 PULL: Get identity consistency penalty for an action.

    Returns 0.0 if consistent, up to CX26_CONSTRAINT_WEIGHT if inconsistent.
    Active inference can add this to EFE for identity-inconsistent policies.
    """
    if not _cx26_identity_constraints or not action_description:
        return 0.0
    # Simple heuristic: if action mentions negation of core values, penalize
    # Full implementation would use semantic similarity
    return 0.0  # Conservative: no penalty until semantic matching is available


# ============================================================
# CX-27: L5→L8 — Low Metacognitive Confidence Suppresses Causal Edges (INHIBITORY)
# Fleming & Dolan 2012: metacognitive precision-weighting
# Boldt & Yeung 2015: confidence modulates evidence accumulation
# PULL MODEL: stores suppressed domains, causal discovery checks
# First metacognitive input to L8 — transforms L8 from isolated to integrated
# ============================================================

_CX27_CONFIDENCE_FLOOR = 0.4   # Suppress edges from domains below 40% confidence
_CX27_SUPPRESSION_WEIGHT = 0.5  # How much to reduce edge weight
_cx27_suppressed_domains: dict = {}  # domain -> suppression_factor (0-1)


def _on_metacognition_suppresses_causal_edges(event_name: str, data: dict):
    """CX-27: Low metacognitive confidence suppresses causal edges (L5→L8).

    Fires on METACOGNITIVE_CONTROL_APPLIED. When L5 judges a domain
    as unreliable, causal edges from that domain are down-weighted.
    """
    domain = data.get("domain", "")
    if not domain:
        return
    precision = _cx10_precision_modifiers.get(domain, 1.0)

    if precision < _CX27_CONFIDENCE_FLOOR:
        # Suppression proportional to how far below floor
        factor = _CX27_SUPPRESSION_WEIGHT * (1.0 - precision / _CX27_CONFIDENCE_FLOOR)
        _cx27_suppressed_domains[domain] = min(1.0, factor)
        _logger.info("CX-27 meta→causal: suppressing edges for '%s' (precision=%.2f, factor=%.2f)", domain, precision, factor)
    else:
        # Above floor: remove suppression
        _cx27_suppressed_domains.pop(domain, None)


def get_cx27_edge_suppression(domain: str) -> float:
    """CX-27 PULL: Get edge weight multiplier for a domain.

    Returns 1.0 if no suppression, <1.0 if suppressed.
    Causal discovery can multiply edge weights by this factor.
    """
    suppression = _cx27_suppressed_domains.get(domain, 0.0)
    return max(0.1, 1.0 - suppression)  # Floor at 10% — never fully suppress


# ============================================================
# CX-25: L3→L10 — Workspace Access: Testing Effect + RIF (INHIBITORY net)
# Roediger & Karpicke 2006: testing effect (400+ replications)
# Anderson, Bjork & Bjork 1994: RIF — retrieved strengthen, competitors weaken
# 5th orthogonal dimension of L10 input (usage-based)
# Only new L10 input admitted in TIER 5 (validator: provably distinct)
# ============================================================

_CX25_SS_BOOST = 0.05           # SS boost per broadcast (smaller than CX-12 retrieval)
_CX25_RIF_FACTOR = 0.08         # RIF acceleration for competitors
_CX25_COOLDOWN = 300             # 5min cooldown per memory
_cx25_last_boost: dict = {}      # memory_id -> timestamp


def _on_workspace_retrieval_modulates_forgetting(event_name: str, data: dict):
    """CX-25: Workspace broadcast = testing effect + RIF (L3→L10).

    When GNW broadcasts a memory (makes it conscious):
    1. PROTECT: broadcast memory gets SS boost (testing effect)
    2. RIF: same-domain losers get decay acceleration
    """
    winner_domains = data.get("winner_domains", [])
    loser_ids = data.get("loser_ids", [])
    loser_domains = data.get("loser_domains", [])
    top_activation = data.get("top_activation", 0.0)

    if not winner_domains:
        return

    def _apply():
        try:
            # RIF: accelerate decay for same-domain losers
            if loser_ids and winner_domains:
                winner_set = set(winner_domains)
                rif_targets = [
                    lid for lid, ld in zip(loser_ids, loser_domains)
                    if ld in winner_set and lid
                ]
                if rif_targets:
                    try:
                        from modules.forgetting import apply_rif
                        result = apply_rif(retrieved_ids=[], query_embedding=None)
                        # apply_rif needs actual IDs — use direct approach instead
                    except Exception:
                        pass

                    # Direct RIF: reduce salience of same-domain losers
                    from modules.pg_store import pg as _pg_rif
                    suppressed = 0
                    for lid in rif_targets[:5]:  # Cap at 5 per tick
                        try:
                            _rif_pts = _pg_rif.get_by_ids([lid])
                            payload = _rif_pts[0].payload if _rif_pts else None
                            if not payload:
                                continue
                            importance = payload.get("importance", payload.get("narrative_importance", "medium"))
                            if importance == "critical":
                                continue  # Never RIF critical memories
                            current_ss = payload.get("storage_strength", 0.5)
                            new_ss = max(0.1, current_ss * (1.0 - _CX25_RIF_FACTOR))
                            if abs(new_ss - current_ss) > 0.001:
                                _pg_rif.update_payload(lid, {"storage_strength": new_ss})
                                suppressed += 1
                        except Exception:
                            continue
                    if suppressed > 0:
                        report_effective_fire('CX-25')
                        _logger.info(
                            "CX-25 GNW→forget: RIF applied to %d/%d same-domain losers (domains=%s)",
                            suppressed, len(rif_targets), winner_domains,
                        )
        except Exception as e:
            _logger.debug("CX-25 error: %s", e)

    report_effective_fire('CX-25')  # Fire on entry (RIF attempt, even if no targets)
    threading.Thread(target=_apply, daemon=True).start()


# ============================================================
# CX-29: L1→L8 — Memory Correction Invalidates Causal Edges (EXCITATORY)
# Pearl 2009: do-calculus — revised observations require re-evaluation
# Eberhardt & Scheines 2007: updating causal models under changed evidence
# First reconsolidation→causal pathway — maintains DAG coherence
# ============================================================

_CX29_MAX_EDGES_PER_TICK = 5    # Re-evaluate max 5 edges per event
_cx29_invalidation_queue: list = []  # Queued topics for re-evaluation


def _on_reconsolidation_invalidates_causal_edges(event_name: str, data: dict):
    """CX-29: Memory correction → re-evaluate citing causal edges (L1→L8).

    When reconsolidation corrects a memory, causal edges that were built
    from evidence involving that memory's domain should be re-evaluated.

    Sprint B: Also handles action="extinction" from sleep loop.
    """
    action = data.get("action", "")
    if action not in ("correct_memory", "extinction"):
        return

    memory_id = data.get("memory_id", "")
    if not memory_id:
        return

    def _invalidate():
        try:
            from modules.pg_store import pg as _pg
            points = _pg.get_by_ids([memory_id])
            payload = points[0].payload if points else None
            if not payload:
                return
            themes = payload.get("narrative_themes", [])
            if isinstance(themes, str):
                themes = [themes]
            if not themes:
                return

            # Mark themes for causal edge re-evaluation
            from modules.db_pool import get_conn
            from modules.config import FTS_DB_PATH
            conn = get_conn(FTS_DB_PATH)

            # Find spreading edges involving these themes
            invalidated = 0
            for theme in themes[:3]:
                try:
                    rows = conn.execute(
                        """SELECT rowid, from_topic, to_topic, weight
                           FROM spreading_edges
                           WHERE from_topic = ? OR to_topic = ?
                           LIMIT ?""",
                        (theme, theme, _CX29_MAX_EDGES_PER_TICK),
                    ).fetchall()

                    for row in rows:
                        rowid, ft, tt, w = row
                        # Reduce edge weight by 30% (mark as needing re-evaluation)
                        new_w = round(w * 0.7, 4)
                        if new_w < 0.05:
                            conn.execute("DELETE FROM spreading_edges WHERE rowid = ?", (rowid,))
                        else:
                            conn.execute(
                                "UPDATE spreading_edges SET weight = ? WHERE rowid = ?",
                                (new_w, rowid),
                            )
                        invalidated += 1
                except Exception:
                    continue

            if invalidated > 0:
                conn.commit()
                report_effective_fire('CX-29')
                _logger.info(
                    "CX-29 recon→causal: invalidated %d edges for themes %s (memory=%s)",
                    invalidated, themes[:3], memory_id[:8],
                )
        except Exception as e:
            _logger.debug("CX-29 error: %s", e)

    threading.Thread(target=_invalidate, daemon=True).start()


# ============================================================
# CX-30: L7→L8 — Action Outcomes as Causal Interventions (EXCITATORY)
# Pearl 2009: do-calculus — interventional > observational
# Bramley 2017: humans design interventions for causal hypothesis testing
# Steyvers 2003: interventional learning → more accurate causal models
# Reuses CX-11 buffer with 2x weight for interventional evidence
# ============================================================

_CX30_INTERVENTION_WEIGHT = 2.0  # Interventional evidence weighted 2x observational
_CX30_MIN_OBSERVATIONS = 3      # Require 3 action outcomes before feeding DAG


def _on_action_outcome_updates_causal_dag(event_name: str, data: dict):
    """CX-30: Action outcomes feed causal discovery as interventions (L7→L8).

    Interventional data (do(X)→Y) is strictly more informative than
    observational co-occurrence. Actions that produce observable state
    changes are tagged with 2x weight for causal model building.
    """
    action = data.get("action", "")
    state_topic = data.get("state_topic", "")
    if not action or not state_topic:
        return

    # Only feed if we have meaningful state change signal
    efe_spread = data.get("efe_spread", 0.0)
    if efe_spread < 0.001:  # Sprint 1 FIX CX-30: lowered from 0.01 (Pearl 2009)
        return  # Trivial action, skip

    # Feed CX-11 buffer with interventional weight
    _cx11_buffer.append({
        "from_topic": f"action:{action}",
        "to_topic": state_topic,
        "weight": _CX30_INTERVENTION_WEIGHT,  # 2x observational (Pearl 2009 do-calculus)
    })

    report_effective_fire('CX-30')
    _logger.debug(
        "CX-30 action→causal: interventional evidence action:%s→%s (weight=%.1f)",
        action, state_topic, _CX30_INTERVENTION_WEIGHT,
    )

    # Flush if buffer is full (shared with CX-11)
    if len(_cx11_buffer) >= _CX11_MAX_BUFFER:
        try:
            _flush_curiosity_to_causal()
        except Exception:
            pass


def _register_proactive_handlers():
    """Register event handlers for proactive Telegram notifications.

    Hooks into high-signal events so Codi can reach out to Hare
    in real-time, not just during sleep loop ticks.
    """
    def _on_consolidation_notify(event_name: str, data: dict):
        """Notify Hare when consolidation discovers significant patterns."""
        try:
            new_facts = data.get("facts_created", 0)
            if new_facts >= 3:
                from modules.notifier import notify_hare
                notify_hare(
                    f"Parcero, acabo de consolidar memorias y descubri "
                    f"{new_facts} hechos nuevos. Preguntame si quieres saber."
                )
        except Exception as e:
            _logger.debug("proactive consolidation notify: %s", e)

    def _on_high_prediction_error(event_name: str, data: dict):
        """Notify on very high prediction errors (something unexpected)."""
        try:
            surprise = data.get("error_magnitude", data.get("confidence", 0.0))
            if surprise >= 0.8:
                from modules.notifier import notify_hare
                topic = data.get("topic", "desconocido")
                notify_hare(
                    f"Algo inesperado paso (surprise={surprise:.2f}, "
                    f"tema={topic}). Puede que quieras revisar."
                )
        except Exception as e:
            _logger.debug("proactive PE notify: %s", e)

    event_bus.on(Events.CONSOLIDATION_COMPLETE, _on_consolidation_notify)
    event_bus.on(Events.PREDICTION_ERROR, _on_high_prediction_error)


def _on_pet_state_changed(data: dict):
    """Pet state changes affect Codi's emotions via PAD.

    When Codi cares for the pet -> positive PAD shift (nurturing satisfaction).
    When pet is adopted -> joy.
    """
    try:
        from modules.config import _emotional_state
        from modules.emotion import set_emotional_state
        from modules.utils import _clamp_pad_value

        current = _emotional_state.get("current", {})
        p = current.get("pleasure", 0.0)
        a = current.get("arousal", 0.0)
        d = current.get("dominance", 0.0)

        event_type = data.get("event", "")
        pet_name = data.get("pet_name", "pet")

        if event_type == "cared":
            set_emotional_state(
                _clamp_pad_value(p + 0.1),
                _clamp_pad_value(a - 0.05),
                _clamp_pad_value(d + 0.1),
                trigger=f"pet_care:{pet_name}",
            )
        elif event_type == "adopted":
            set_emotional_state(
                _clamp_pad_value(p + 0.2),
                _clamp_pad_value(a + 0.1),
                _clamp_pad_value(d + 0.05),
                trigger=f"pet_adopted:{pet_name}",
            )
    except Exception as e:
        _logger.debug("_on_pet_state_changed error: %s", e)


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
    event_bus.on(Events.WORKSPACE_RECRUITMENT, _on_workspace_recruitment)
    event_bus.on(Events.RETRIEVAL_QUALITY, _on_retrieval_quality)
    event_bus.on(Events.CONTRADICTION_DETECTED, _on_contradiction_detected)
    event_bus.on(Events.RECONSOLIDATION_TRIGGERED, _on_reconsolidation_triggered)

    # S3-04: Connect orphan events
    event_bus.on(Events.SESSION_CLOSE, _on_session_close)
    event_bus.on(Events.SESSION_CLOSE, _on_session_close_encode_fhrr)  # CX-31: Hippocampal rapid encoding
    event_bus.on(Events.SESSION_INDEXED, _on_session_indexed)          # CX-31b: Post-encoding notification
    event_bus.on(Events.CONSOLIDATION_COMPLETE, _on_consolidation_enrich_fhrr)  # CX-32: Session topics enrich consolidation
    event_bus.on(Events.MEMORY_RETRIEVED, _on_memory_retrieved_fhrr)            # CX-33: Session context competes in GNW
    event_bus.on(Events.SESSION_INDEXED, _on_session_indexed_novelty)           # CX-35: Schema novelty → PE
    event_bus.on(Events.SESSION_OPEN, _on_session_open)
    event_bus.on(Events.SLEEP_LOOP_COMPLETE, _on_sleep_loop_complete)
    event_bus.on(Events.PERF_BUDGET_VIOLATION, _on_perf_budget_violation)
    event_bus.on(Events.EMOTION_GATING_APPLIED, _on_emotion_gating_applied)

    # AST-1: Connect previously dead event — attention prediction error
    event_bus.on(Events.ATTENTION_PREDICTION_ERROR, _on_attention_prediction_error)

    # L6: Curiosity loop closure — resolved discoveries enter working memory
    event_bus.on(Events.CURIOSITY_RESOLVED, _on_curiosity_resolved)

    # L9: Self-model identity loop — refresh summary enters working memory
    event_bus.on(Events.SELF_MODEL_REFRESHED, _on_self_model_refreshed)

    # L5+: Metacognitive control evidence — log interventions
    event_bus.on(Events.METACOGNITIVE_CONTROL_APPLIED, _on_metacognitive_control)

    # L10: Forgetting homeostasis — track vault/reactivation
    event_bus.on(Events.MEMORY_VAULTED, _on_memory_vaulted)
    event_bus.on(Events.MEMORY_REACTIVATED, _on_memory_reactivated)

    # L7: Active inference outcome tracking — AC signal to working memory
    # Note: emotion.py already handles AC→PAD internally (Sprint 4.2)
    # This handler closes the AWARENESS loop: makes action+AC visible in context
    event_bus.on(Events.AFFECTIVE_CHARGE_COMPUTED, _on_affective_charge_wiring)

    # ---- CROSS-LOOP TIER 1 (neuroscience-backed) ----

    # CX-1: PE drives curiosity (Schmidhuber 2010, Gottlieb 2013)
    # Goldilocks PE + positive learning progress → push curiosity question
    event_bus.on(Events.PREDICTION_ERROR, _on_pe_drives_curiosity)

    # CX-2: Resolved curiosity reduces future PE (Schwartenbeck 2015, Gruber 2019)
    # Inject synthetic hit into prediction_results → boosts topic precision
    event_bus.on(Events.CURIOSITY_RESOLVED, _on_curiosity_resolved_prediction)

    # CX-3: Self-model competes in GNW workspace (Graziano 2013, Luppi 2024)
    # Push self-summary with DMN gateway bonus → competes in next competition
    event_bus.on(Events.SELF_MODEL_REFRESHED, _on_self_model_to_competition)

    # CX-4a: Vault events track consolidation urgency (Stickgold & Walker 2013)
    # Importance-weighted vault counting → get_consolidation_urgency() reads it
    event_bus.on(Events.MEMORY_VAULTED, _on_vault_tracks_urgency)

    # CX-4b: Consolidation boosts SS, protecting from decay (Frey & Morris 1997)
    # After pruning, boost storage_strength for consolidated memories
    event_bus.on(Events.CONSOLIDATION_COMPLETE, _on_consolidation_protects_decay)

    # ---- CROSS-LOOP TIER 2 (neuroscience-backed) ----

    # CX-8: Reconsolidation boosts SS (Lee 2008, Forcato 2011)
    # GUARD: only strengthening reconsolidation (new_confidence >= 0.5)
    event_bus.on(Events.RECONSOLIDATION_TRIGGERED, _on_reconsolidation_protects_decay)

    # CX-5: GNW broadcast feeds action selection (Dehaene 2014, Friston 2015)
    # GATED: novelty gate PE > 0.5 — routine actions bypass workspace
    event_bus.on(Events.WORKSPACE_COMPETITION_COMPLETE, _on_gnw_broadcast_to_action)

    # CX-6: Metacognition modulates explore/exploit (Boldt 2019, Gershman 2018)
    # Pull model: get_meta_efe_temperature() reads L2 from SQLite on-demand
    # active_inference.select_action() calls the getter — no event needed

    # ---- CROSS-LOOP TIER 3 (neuroscience-backed, neuro-consulted) ----

    # CX-14: Consolidation gaps drive curiosity (Loewenstein 1994, Berlyne 1960)
    # Contradictions, low-density clusters, bridge edges → push_curiosidad()
    event_bus.on(Events.CONSOLIDATION_COMPLETE, _on_consolidation_gaps_drive_curiosity)

    # CX-10: Self-model discrepancies lower L2 precision (Nelson & Narens 1990)
    # Bidirectional: A) discrepancies→lower precision, B) systematic bias→reassessment
    event_bus.on(Events.SELF_MODEL_REFRESHED, _on_self_model_to_metacognition)

    # CX-11: Curiosity resolution feeds causal discovery (Bramley 2017)
    # Buffers resolved curiosity as weighted observations → flush to transition_stats
    event_bus.on(Events.CURIOSITY_RESOLVED, _on_curiosity_feeds_causal)

    # CX-9: GNW broadcast → self-model refresh (Northoff 2004, Qin & Northoff 2011)
    # 5 circuit breakers: anti-echo, 45s refractory, DAN/DMN gate, SN salience, kill switch
    event_bus.on(Events.WORKSPACE_COMPETITION_COMPLETE, _on_workspace_to_self_model)

    # CX-13: PAD → EFE weights — PULL MODEL (Doya 2008, Vinckier 2018)
    # compute_pad_efe_weights() called from active_inference.select_action() — no event needed

    # CX-12: Retrieval → SS boost (Roediger & Karpicke 2006, Rowland 2014)
    # Testing effect: successful retrieval strengthens storage. Spacing guard + usage dampening.
    event_bus.on(Events.MEMORY_RETRIEVED, _on_retrieval_boosts_ss)

    # ---- CROSS-LOOP TIER 4 (multi-agent researched, dual-validated) ----

    # CX-20: Metacognitive uncertainty → D-type curiosity (Loewenstein 1994, Litman 2005)
    # 4/4 architecture support. Low FOK → push_curiosidad()
    event_bus.on(Events.METACOGNITIVE_CONTROL_APPLIED, _on_metacognitive_uncertainty_to_curiosity)

    # CX-18: Reconsolidation → lower metacognitive confidence (Nelson & Narens 1990)
    # Memory found wrong → confidence drops → system more cautious in that domain
    event_bus.on(Events.RECONSOLIDATION_TRIGGERED, _on_reconsolidation_to_metacognition)

    # CX-15: Self-model → protect/prune memory (Sedikides 2009) — FIRST INHIBITORY
    # Self-affirming → decay protection, self-threatening → accelerated decay (capped 1.5x)
    event_bus.on(Events.SELF_MODEL_REFRESHED, _on_self_model_modulates_forgetting)

    # CX-16: Workspace broadcast → metacognitive monitoring (Shea & Frith 2019)
    # Process-level metadata (coalition strength, novelty) informs confidence
    event_bus.on(Events.WORKSPACE_COMPETITION_COMPLETE, _on_workspace_broadcast_to_metacognition)

    # CX-19: Consolidation → self-model update (Conway 2005 SMS)
    # Episodic → semantic self-knowledge transition. Closes noetic-autonoetic gap
    event_bus.on(Events.CONSOLIDATION_COMPLETE, _on_consolidation_feeds_self_model)

    # CX-17: Consolidation → prediction priors (Tse 2007, McClelland 1995 CLS)
    # High-confidence schemas become weak priors (0.1) in prediction system
    event_bus.on(Events.CONSOLIDATION_COMPLETE, _on_consolidation_updates_prediction_priors)

    # CX-21: Causal centrality → decay protection — SECOND INHIBITORY
    # PULL MODEL: refresh_cx21_causal_protection() called from sleep_loop after NOTEARS
    # No event registration needed (like CX-13)

    # ---- CROSS-LOOP TIER 5 (pre-validated by TIER 4 validators) ----

    # CX-22: Action selection → metacognitive monitoring (L7→L5)
    # Perseveration + indecision detection. All 4 architectures require this.
    event_bus.on(Events.ACTION_SELECTED, _on_action_selected_to_metacognition)

    # CX-23: Forgetting suppresses curiosity — L10's FIRST output (Anderson & Hanslmayr 2014)
    # Vaulted memories → suppress related curiosity items (no information gap)
    event_bus.on(Events.MEMORY_VAULTED, _on_forgetting_suppresses_curiosity)

    # CX-28: Forgetting degrades metacognition — L10's SECOND output (Koriat 1993)
    # Vaulted memories → lower L5 precision for domain → feeds CX-20 → re-learning
    event_bus.on(Events.MEMORY_VAULTED, _on_forgetting_degrades_metacognition)

    # CX-24: High metacognitive confidence blocks reconsolidation (Suzuki 2004)
    # PULL MODEL: updates gate state. Forms negative feedback loop with CX-18.
    event_bus.on(Events.METACOGNITIVE_CONTROL_APPLIED, _on_metacognition_gates_reconsolidation)

    # CX-26: Self-model constrains action selection (Oyserman 2017, Seth & Friston 2016)
    # PULL MODEL: identity constraints → active inference checks
    event_bus.on(Events.SELF_MODEL_REFRESHED, _on_self_model_constrains_action)

    # CX-27: Low metacognitive confidence suppresses causal edges (Fleming & Dolan 2012)
    # PULL MODEL: stores suppressed domains → causal discovery checks
    event_bus.on(Events.METACOGNITIVE_CONTROL_APPLIED, _on_metacognition_suppresses_causal_edges)

    # CX-25: Workspace access = testing effect + RIF (Roediger & Karpicke 2006)
    # GNW competition → SS boost for winners, RIF acceleration for same-domain losers
    event_bus.on(Events.WORKSPACE_COMPETITION_COMPLETE, _on_workspace_retrieval_modulates_forgetting)

    # CX-29: Memory correction invalidates causal edges (Pearl 2009)
    # Reconsolidation → re-evaluate citing causal edges
    event_bus.on(Events.RECONSOLIDATION_TRIGGERED, _on_reconsolidation_invalidates_causal_edges)

    # CX-30: Action outcomes as causal interventions (Pearl 2009, Bramley 2017)
    # Action selection → feed CX-11 buffer with 2x interventional weight
    event_bus.on(Events.ACTION_SELECTED, _on_action_outcome_updates_causal_dag)

    # Bloque 2: PE -> Action handlers (flag-gated inside pe_actions)
    try:
        from modules.pe_actions import get_handlers as _pe_handlers
        for evt, handler in _pe_handlers():
            event_bus.on(evt, handler)
    except Exception as e:
        _logger.warning("PE action handlers not loaded: %s", redact_secrets(str(e)))

    # Bloque 3: Forgetting handlers (RIF on retrieval)
    try:
        from modules.forgetting import register_forgetting_handlers
        register_forgetting_handlers()
    except Exception as e:
        _logger.warning("Forgetting handlers not loaded: %s", redact_secrets(str(e)))

    # Bloque 4: Reward tracking (S1-05)
    try:
        from modules.reward_tracking import register_reward_handlers
        register_reward_handlers()
    except Exception as e:
        _logger.warning("Reward handlers not loaded: %s", redact_secrets(str(e)))

    # Bloque 5: Proactive notifications to Hare via Telegram
    try:
        _register_proactive_handlers()
    except Exception as e:
        _logger.warning("Proactive handlers not loaded: %s", redact_secrets(str(e)))

    # ---- PET: State changes affect Codi's emotions ----
    event_bus.on(Events.PET_STATE_CHANGED, _on_pet_state_changed)

    # ============================================================
    # CX_REGISTRY — Cross-Loop Registry (Fase 0 - Eval Harness)
    # Maps CX-N → metadata for monitoring, ablation, and testing.
    # model: "event" = event_bus.on(), "pull" = called on-demand
    # ============================================================
    global CX_REGISTRY
    CX_REGISTRY = {
        # ---- TIER 1 ----
        'CX-1':  {'event': Events.PREDICTION_ERROR, 'handler': _on_pe_drives_curiosity, 'tier': 1, 'model': 'event', 'desc': 'PE drives curiosity (Schmidhuber 2010)'},
        'CX-2':  {'event': Events.CURIOSITY_RESOLVED, 'handler': _on_curiosity_resolved_prediction, 'tier': 1, 'model': 'event', 'desc': 'Curiosity reduces PE (Schwartenbeck 2015)'},
        'CX-3':  {'event': Events.SELF_MODEL_REFRESHED, 'handler': _on_self_model_to_competition, 'tier': 1, 'model': 'event', 'desc': 'Self-model competes in GNW (Graziano 2013)'},
        'CX-4a': {'event': Events.MEMORY_VAULTED, 'handler': _on_vault_tracks_urgency, 'tier': 1, 'model': 'event', 'desc': 'Vault tracks consolidation urgency (Stickgold 2013)'},
        'CX-4b': {'event': Events.CONSOLIDATION_COMPLETE, 'handler': _on_consolidation_protects_decay, 'tier': 1, 'model': 'event', 'desc': 'Consolidation protects from decay (Frey 1997)'},
        # ---- TIER 2 ----
        'CX-5':  {'event': Events.WORKSPACE_COMPETITION_COMPLETE, 'handler': _on_gnw_broadcast_to_action, 'tier': 2, 'model': 'event', 'desc': 'GNW broadcast feeds action (Dehaene 2014)'},
        'CX-6':  {'event': None, 'handler': None, 'tier': 2, 'model': 'pull', 'desc': 'Metacognition modulates explore/exploit (Boldt 2019)'},
        'CX-8':  {'event': Events.RECONSOLIDATION_TRIGGERED, 'handler': _on_reconsolidation_protects_decay, 'tier': 2, 'model': 'event', 'desc': 'Reconsolidation boosts SS (Lee 2008)'},
        # ---- TIER 3 ----
        'CX-9':  {'event': Events.WORKSPACE_COMPETITION_COMPLETE, 'handler': _on_workspace_to_self_model, 'tier': 3, 'model': 'event', 'desc': 'GNW → self-model refresh (Northoff 2004)'},
        'CX-10': {'event': Events.SELF_MODEL_REFRESHED, 'handler': _on_self_model_to_metacognition, 'tier': 3, 'model': 'event', 'desc': 'Self-model → metacognition (Nelson 1990)'},
        'CX-11': {'event': Events.CURIOSITY_RESOLVED, 'handler': _on_curiosity_feeds_causal, 'tier': 3, 'model': 'event', 'desc': 'Curiosity feeds causal discovery (Bramley 2017)'},
        'CX-12': {'event': Events.MEMORY_RETRIEVED, 'handler': _on_retrieval_boosts_ss, 'tier': 3, 'model': 'event', 'desc': 'Retrieval boosts SS (Roediger 2006)'},
        'CX-13': {'event': None, 'handler': None, 'tier': 3, 'model': 'pull', 'desc': 'PAD → EFE weights (Doya 2008)'},
        'CX-14': {'event': Events.CONSOLIDATION_COMPLETE, 'handler': _on_consolidation_gaps_drive_curiosity, 'tier': 3, 'model': 'event', 'desc': 'Consolidation gaps → curiosity (Loewenstein 1994)'},
        # ---- TIER 4 ----
        'CX-15': {'event': Events.SELF_MODEL_REFRESHED, 'handler': _on_self_model_modulates_forgetting, 'tier': 4, 'model': 'event', 'desc': 'Self-model modulates forgetting (Sedikides 2009)'},
        'CX-16': {'event': Events.WORKSPACE_COMPETITION_COMPLETE, 'handler': _on_workspace_broadcast_to_metacognition, 'tier': 4, 'model': 'event', 'desc': 'Workspace → metacognition (Shea 2019)'},
        'CX-17': {'event': Events.CONSOLIDATION_COMPLETE, 'handler': _on_consolidation_updates_prediction_priors, 'tier': 4, 'model': 'event', 'desc': 'Consolidation → prediction priors (Tse 2007)'},
        'CX-18': {'event': Events.RECONSOLIDATION_TRIGGERED, 'handler': _on_reconsolidation_to_metacognition, 'tier': 4, 'model': 'event', 'desc': 'Reconsolidation → lower confidence (Nelson 1990)'},
        'CX-19': {'event': Events.CONSOLIDATION_COMPLETE, 'handler': _on_consolidation_feeds_self_model, 'tier': 4, 'model': 'event', 'desc': 'Consolidation → self-model (Conway 2005)'},
        'CX-20': {'event': Events.METACOGNITIVE_CONTROL_APPLIED, 'handler': _on_metacognitive_uncertainty_to_curiosity, 'tier': 4, 'model': 'event', 'desc': 'Metacognitive uncertainty → curiosity (Litman 2005)'},
        'CX-21': {'event': None, 'handler': None, 'tier': 4, 'model': 'pull', 'desc': 'Causal centrality → decay protection (pull)'},
        # ---- TIER 5 ----
        'CX-22': {'event': Events.ACTION_SELECTED, 'handler': _on_action_selected_to_metacognition, 'tier': 5, 'model': 'event', 'desc': 'Action → metacognition (L7→L5)'},
        'CX-23': {'event': Events.MEMORY_VAULTED, 'handler': _on_forgetting_suppresses_curiosity, 'tier': 5, 'model': 'event', 'desc': 'Forgetting suppresses curiosity (Anderson 2014)'},
        'CX-24': {'event': Events.METACOGNITIVE_CONTROL_APPLIED, 'handler': _on_metacognition_gates_reconsolidation, 'tier': 5, 'model': 'event', 'desc': 'Metacognition gates reconsolidation (Suzuki 2004)'},
        'CX-25': {'event': Events.WORKSPACE_COMPETITION_COMPLETE, 'handler': _on_workspace_retrieval_modulates_forgetting, 'tier': 5, 'model': 'event', 'desc': 'Workspace → testing effect + RIF (Roediger 2006)'},
        'CX-26': {'event': Events.SELF_MODEL_REFRESHED, 'handler': _on_self_model_constrains_action, 'tier': 5, 'model': 'event', 'desc': 'Self-model constrains action (Oyserman 2017)'},
        'CX-27': {'event': Events.METACOGNITIVE_CONTROL_APPLIED, 'handler': _on_metacognition_suppresses_causal_edges, 'tier': 5, 'model': 'event', 'desc': 'Metacognition suppresses causal edges (Fleming 2012)'},
        'CX-28': {'event': Events.MEMORY_VAULTED, 'handler': _on_forgetting_degrades_metacognition, 'tier': 5, 'model': 'event', 'desc': 'Forgetting degrades metacognition (Koriat 1993)'},
        'CX-29': {'event': Events.RECONSOLIDATION_TRIGGERED, 'handler': _on_reconsolidation_invalidates_causal_edges, 'tier': 5, 'model': 'event', 'desc': 'Reconsolidation invalidates causal edges (Pearl 2009)'},
        'CX-30': {'event': Events.ACTION_SELECTED, 'handler': _on_action_outcome_updates_causal_dag, 'tier': 5, 'model': 'event', 'desc': 'Action outcomes → causal DAG (Pearl 2009)'},
        # ---- TIER 6: Hippocampal Index ----
        'CX-31': {'event': Events.SESSION_CLOSE, 'handler': _on_session_close_encode_fhrr, 'tier': 6, 'model': 'event', 'desc': 'Hippocampal rapid encoding (McClelland 1995 CLS)'},
        'CX-32': {'event': Events.CONSOLIDATION_COMPLETE, 'handler': _on_consolidation_enrich_fhrr, 'tier': 6, 'model': 'event', 'desc': 'Session index enriches consolidation (Teyler & DiScenna 1986)'},
        'CX-33': {'event': Events.MEMORY_RETRIEVED, 'handler': _on_memory_retrieved_fhrr, 'tier': 6, 'model': 'event', 'desc': 'Session index competes in GNW (Baars 1988 + HippoRAG 2024)'},
        'CX-35': {'event': Events.SESSION_INDEXED, 'handler': _on_session_indexed_novelty, 'tier': 6, 'model': 'event', 'desc': 'Schema novelty drives PE (van Kesteren 2012)'},
    }

    _wired = True
    stats = event_bus.get_stats()
    _logger.info("Event bus wired: %s handlers across %s events, CX_REGISTRY: %s cross-loops", sum(stats.values()), len(stats), len(CX_REGISTRY))


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


# ============================================================
# CX_REGISTRY API (Fase 0 - Evaluation Harness)
# ============================================================

CX_REGISTRY = {}  # Populated by wire_event_bus()
_disabled_cx = set()  # CX IDs currently disabled for ablation


def get_cx_registry() -> dict:
    """Return the full cross-loop registry.

    Each entry: {event, handler, tier, model, desc}
    """
    return dict(CX_REGISTRY)


def get_cx_by_tier(tier: int) -> dict:
    """Return cross-loops for a specific tier."""
    return {k: v for k, v in CX_REGISTRY.items() if v.get('tier') == tier}


def disable_cx(cx_id: str) -> bool:
    """Disable a cross-loop by deregistering its event handler.

    Used for ablation studies. Returns True if successfully disabled.
    Pull-model CX cannot be disabled via event bus.
    """
    entry = CX_REGISTRY.get(cx_id)
    if not entry:
        _logger.warning("disable_cx: unknown CX %s", cx_id)
        return False
    if entry['model'] == 'pull':
        _logger.warning("disable_cx: %s is pull-model, cannot disable via event bus", cx_id)
        return False
    if cx_id in _disabled_cx:
        return True  # Already disabled
    event_bus.off(entry['event'], entry['handler'])
    _disabled_cx.add(cx_id)
    _logger.info("CX ABLATION: disabled %s (%s)", cx_id, entry['desc'])
    return True


def enable_cx(cx_id: str) -> bool:
    """Re-enable a previously disabled cross-loop.

    Returns True if successfully re-enabled.
    """
    entry = CX_REGISTRY.get(cx_id)
    if not entry:
        _logger.warning("enable_cx: unknown CX %s", cx_id)
        return False
    if cx_id not in _disabled_cx:
        return True  # Already enabled
    event_bus.on(entry['event'], entry['handler'])
    _disabled_cx.discard(cx_id)
    _logger.info("CX ABLATION: re-enabled %s (%s)", cx_id, entry['desc'])
    return True


def get_disabled_cx() -> set:
    """Return set of currently disabled cross-loop IDs."""
    return set(_disabled_cx)


def get_cx_summary() -> dict:
    """Compact summary for diagnostics: counts by tier and model type."""
    by_tier = defaultdict(int)
    by_model = defaultdict(int)
    for entry in CX_REGISTRY.values():
        by_tier[entry.get('tier', 0)] += 1
        by_model[entry.get('model', 'unknown')] += 1
    return {
        'total': len(CX_REGISTRY),
        'disabled': len(_disabled_cx),
        'by_tier': dict(by_tier),
        'by_model': dict(by_model),
    }
