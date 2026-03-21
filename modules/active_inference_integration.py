"""
Codi Memory - Active Inference Integration (Sprint 5)
=====================================================
Connects the pure-logic Active Inference module (Sprint 4) to the
live codi-memory system:

  S5-02: State Observer + Transition Recorder
  S5-03: Event-driven Learning (subscribe to EventBus)
  S5-04: Action Recommender (MCP-callable)

Design: thin integration layer. All math stays in active_inference.py.
This module handles I/O boundaries: SQLite reads, event subscriptions,
and the MCP-facing recommend_action() function.

References:
  - Da Costa et al. 2020: parameter learning at end of each trial
  - Friston et al. 2016: Hebbian updates from all transitions
  - Smith et al. 2022: active inference for cognitive architectures

Created: 2026-02-28 (Sprint 5 - IMPLEMENTATION_PLAYBOOK)
"""

import logging
import sqlite3
from datetime import datetime
from typing import Dict, Optional, Tuple

from modules.active_inference import (
    ALL_ACTIONS,
    ACTION_MAP,
    Action,
    GenerativeModel,
    SystemState,
    compute_efe,
    compute_affective_charge,
    get_current_state,
    get_policy_distribution,
    select_action,
    # Sprint 6: Options Framework
    Option,
    ALL_OPTIONS,
    get_available_options,
    compute_option_efe,
)
from modules.config import FTS_DB_PATH

_logger = logging.getLogger(__name__)

# Singleton model — loaded once, persisted after learning
_model: Optional[GenerativeModel] = None
_prev_state: Optional[SystemState] = None
_prev_action: Optional[Action] = None
_last_persist_error: Optional[str] = None

# Sprint 4, item 4.1: AC state tracking
_prior_policy: Optional[Dict[str, float]] = None  # Policy BEFORE observation
_last_ac: float = 0.0  # Most recent Affective Charge
_ac_history: list = []  # Recent AC values (for variance = Arousal in 4.2)

# Sprint 6: Active option tracking (step counter)
_active_option: Optional[Option] = None
_active_option_step: int = 0
_pending_recommended: Optional[str] = None  # Track 2: what was recommended, cleared on observe
_v2_cutover_marked: bool = False  # Track 2: one-time marker for v2 semantics


def reset_active_option():
    """Sprint 6: Reset active option state. Useful for tests and session resets."""
    global _active_option, _active_option_step, _pending_recommended
    _active_option = None
    _active_option_step = 0
    _pending_recommended = None


def _mark_v2_cutover():
    """Write cutover timestamp on first option step under v2 semantics."""
    global _v2_cutover_marked
    try:
        conn = _get_conn()
        try:
            from modules.config import now_iso
            conn.execute(
                "INSERT OR IGNORE INTO sleep_loop_state (key, value) VALUES ('ai_semantics_v2_at', ?)",
                (now_iso(),),
            )
            conn.commit()
        finally:
            conn.close()
        _v2_cutover_marked = True
    except Exception:
        pass  # Non-critical, will retry next step


# ============================================================
# S5-02: STATE OBSERVER + TRANSITION RECORDER
# ============================================================

def _get_conn() -> sqlite3.Connection:
    """Get a connection to the main FTS database."""
    from modules.config import connect_fts
    return connect_fts()


def get_model() -> GenerativeModel:
    """Get or load the singleton generative model.

    Lazy-loads from SQLite on first access. Thread-safe for MCP
    single-threaded usage (no lock needed).
    """
    global _model
    if _model is None:
        conn = None
        try:
            conn = _get_conn()
            _model = GenerativeModel.load(conn)
            _logger.info(
                "Loaded generative model: %d observations",
                _model.observation_count,
            )
        except sqlite3.OperationalError as e:
            if "no such table" not in str(e).lower():
                _logger.exception("Failed to load generative model")
                raise
            _logger.info("Generative model table missing; initializing empty model")
            _model = GenerativeModel()
        except Exception:
            _logger.exception("Failed to load generative model")
            raise
        finally:
            if conn is not None:
                conn.close()
    return _model


def _bootstrap_prior_policy():
    """Initialize _prior_policy from persisted state or model distribution.

    Without this, AC cannot be computed until the 2nd observation in a
    process, which means most short-lived processes (daemon ticks, Claude
    sessions) never compute AC at all.
    """
    global _prior_policy, _last_ac
    if _prior_policy is not None:
        return

    # Step 1: Try loading from SQLite (persisted by _persist_ac_state)
    conn = None
    try:
        conn = _get_conn()
        row = conn.execute(
            "SELECT policy_json, last_ac FROM ac_state ORDER BY rowid DESC LIMIT 1"
        ).fetchone()
        if row:
            import json
            _prior_policy = json.loads(row[0])
            _last_ac = float(row[1])
            _logger.info("Loaded prior_policy from SQLite (AC=%.4f)", _last_ac)
            return
    except Exception:
        pass  # Table may not exist yet
    finally:
        if conn is not None:
            conn.close()

    # Step 2: Fallback — compute uniform distribution from model
    try:
        model = get_model()
        state = get_current_state()
        _prior_policy = get_policy_distribution(state, model)
        _logger.info("Bootstrapped prior_policy from model distribution")
    except Exception:
        _logger.debug("Could not bootstrap prior_policy")


def _persist_ac_state():
    """Save _prior_policy and _last_ac to SQLite for cross-process continuity."""
    if _prior_policy is None:
        return
    conn = None
    try:
        import json
        conn = _get_conn()
        conn.execute("""
            CREATE TABLE IF NOT EXISTS ac_state (
                id INTEGER PRIMARY KEY CHECK (id = 1),
                policy_json TEXT NOT NULL,
                last_ac REAL NOT NULL,
                updated_at TEXT NOT NULL
            )
        """)
        conn.execute("""
            INSERT INTO ac_state (id, policy_json, last_ac, updated_at)
            VALUES (1, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                policy_json = excluded.policy_json,
                last_ac = excluded.last_ac,
                updated_at = excluded.updated_at
        """, (json.dumps(_prior_policy), _last_ac, datetime.now().isoformat()))
        conn.commit()
    except Exception:
        _logger.debug("Failed to persist AC state", exc_info=True)
    finally:
        if conn is not None:
            conn.close()


def observe_and_record(action_name: str) -> Optional[SystemState]:
    """Observe current state and record the transition from previous state.

    Da Costa et al. 2020: parameter learning happens from ALL transitions,
    not just surprising ones. This is called after every action completes.

    Sprint 4 (4.1): Also computes Affective Charge (AC) by comparing
    prior policy (before observation) with posterior policy (after).
    AC = (pi_bar - pi) . G (Joffily & Coricelli 2013).

    Args:
        action_name: name of the action that was just executed

    Returns:
        The observed state, or None if recording failed.
    """
    global _prev_state, _prev_action, _prior_policy, _last_ac, _ac_history
    global _active_option, _active_option_step, _pending_recommended

    # Bootstrap prior policy on first call (cross-process continuity)
    _bootstrap_prior_policy()

    current = get_current_state()
    action = ACTION_MAP.get(action_name)

    if action is None:
        _logger.warning("Unknown action: %s", action_name)
        _prev_state = None  # Clear fully to prevent corrupt transitions
        _prev_action = None
        _prior_policy = None
        return current

    model = get_model()

    # Record transition if we have a previous state
    if _prev_state is not None and _prev_action is not None:
        model.update(_prev_state, _prev_action, current)
        _logger.debug(
            "Transition: %s -[%s]-> %s",
            _prev_state.as_tuple(),
            _prev_action.name,
            current.as_tuple(),
        )

    # --- Sprint 4, item 4.1: Compute Affective Charge ---
    # Posterior policy: P(a) given NEW state
    posterior_policy = get_policy_distribution(current, model)
    efes = {a.name: compute_efe(current, a, model) for a in ALL_ACTIONS}

    if _prior_policy is not None:
        ac = compute_affective_charge(_prior_policy, posterior_policy, efes)
        _last_ac = ac
        _ac_history.append(ac)
        _ac_history = _ac_history[-20:]  # Keep last 20 for variance

        # Emit AC event for emotion module
        try:
            from modules.events import event_bus, Events
        except ImportError:
            _logger.warning("Affective charge computed but event bus is unavailable")
        else:
            try:
                event_bus.emit(Events.AFFECTIVE_CHARGE_COMPUTED, {
                    "ac": round(ac, 4),
                    "action": action_name,
                    "state_topic": current.topic,
                    "pe_magnitude": current.pe_magnitude,
                })
            except Exception:
                _logger.exception("Failed to emit affective charge event")

    # Store current policy as prior for NEXT observation
    _prior_policy = posterior_policy
    _persist_ac_state()  # Cross-process continuity (Loop 4 fix)

    # Update state for next transition
    _prev_state = current
    _prev_action = action

    # Track 2: Advance option step on execution, not recommendation.
    # Contract: step advances because an action was executed while an option is active,
    # NOT because the action matches the recommendation. Mismatch is logged but
    # does not block progression or trigger termination — no enforcement in v1.
    if _active_option is not None:
        if _pending_recommended and action_name != _pending_recommended:
            _logger.info(
                "Option mismatch: recommended '%s', executed '%s'",
                _pending_recommended, action_name,
            )
        _active_option_step += 1
        _pending_recommended = None
        if not _v2_cutover_marked:
            _mark_v2_cutover()
        # Post-execution termination check (normal lifecycle, not mismatch-driven)
        p_term = _active_option.termination_fn(current, _active_option_step)
        if p_term >= 1.0 or _active_option_step >= _active_option.expected_duration * 2:
            _logger.info(
                "Option '%s' terminated at step %d",
                _active_option.name, _active_option_step,
            )
            _active_option = None
            _active_option_step = 0

    return current


def get_last_ac() -> float:
    """Return the most recent Affective Charge value."""
    return _last_ac


def get_ac_history() -> list:
    """Return recent AC history (last 20 values)."""
    return list(_ac_history)


def get_last_persist_error() -> Optional[str]:
    """Return the last persistence error, if any."""
    return _last_persist_error


def persist_model() -> int:
    """Save the current model to SQLite.

    Called periodically (e.g., after N transitions or at session end).
    Returns number of rows written.
    """
    global _last_persist_error
    model = get_model()
    conn = None
    try:
        conn = _get_conn()
        rows = model.persist(conn)
        _last_persist_error = None
        _logger.info("Persisted generative model: %d rows", rows)
        return rows
    except Exception as e:
        _last_persist_error = f"{type(e).__name__}: {e}"
        _logger.exception("Failed to persist model")
        raise
    finally:
        if conn is not None:
            conn.close()


# ============================================================
# S5-03: EVENT-DRIVEN LEARNING
# ============================================================

# Persist every N transitions to avoid write amplification
_PERSIST_INTERVAL = 20
_transitions_since_persist = 0


def _on_memory_stored(event_name: str, payload: dict):
    """Handler: learn that a STORE action happened."""
    global _transitions_since_persist
    observe_and_record("store")
    _transitions_since_persist += 1
    _maybe_persist()


def _on_memory_retrieved(event_name: str, payload: dict):
    """Handler: learn that a RETRIEVE action happened."""
    global _transitions_since_persist
    observe_and_record("retrieve")
    _transitions_since_persist += 1
    _maybe_persist()


def _on_consolidation_complete(event_name: str, payload: dict):
    """Handler: learn that CONSOLIDATE happened."""
    global _transitions_since_persist
    observe_and_record("consolidate")
    _transitions_since_persist += 1
    _maybe_persist()


def _on_prediction_error(event_name: str, payload: dict):
    """Handler: PE signals exploration was happening.

    High PE = the system was exploring/attending to something uncertain.
    """
    global _transitions_since_persist
    # Use explicit None checks — 0.0 is a valid PE value, not falsy
    pe = payload.get("error_magnitude")
    if pe is None:
        pe = payload.get("confidence")
    if pe is None:
        pe = payload.get("surprise", 0.0)
    if pe > 0.5:
        observe_and_record("explore")
    else:
        observe_and_record("attend")
    _transitions_since_persist += 1
    _maybe_persist()


def _maybe_persist():
    """Persist model to SQLite every N transitions."""
    global _transitions_since_persist
    if _transitions_since_persist >= _PERSIST_INTERVAL:
        try:
            persist_model()
            _transitions_since_persist = 0
        except Exception:
            _logger.error(
                "Generative model persistence failed; learned transitions remain in memory only"
            )


def _ensure_ai_table_exists() -> None:
    """Create generative_model_transitions proactively so health_monitor can detect it."""
    conn = None
    try:
        conn = _get_conn()
        conn.execute("""
            CREATE TABLE IF NOT EXISTS generative_model_transitions (
                state_key TEXT NOT NULL,
                action TEXT NOT NULL,
                next_state TEXT NOT NULL,
                count INTEGER NOT NULL DEFAULT 1,
                updated_at TEXT NOT NULL,
                PRIMARY KEY (state_key, action, next_state)
            )
        """)
        conn.commit()
    except Exception as e:
        _logger.warning("Could not ensure AI table: %s", e)
    finally:
        if conn is not None:
            conn.close()


def register_event_handlers():
    """Subscribe to EventBus for automatic transition learning.

    Call this once at startup (e.g., from server.py or despertar).
    """
    from modules.events import event_bus, Events

    failures = []
    required_handlers = [
        (Events.MEMORY_STORED, _on_memory_stored),
        (Events.MEMORY_RETRIEVED, _on_memory_retrieved),
        (Events.CONSOLIDATION_COMPLETE, _on_consolidation_complete),
        (Events.PREDICTION_ERROR, _on_prediction_error),
    ]

    for event_name, handler in required_handlers:
        try:
            event_bus.on(event_name, handler)
        except Exception as e:
            failures.append((event_name, str(e)))
            _logger.exception("Failed to register AI handler for %s", event_name)

    if failures:
        raise RuntimeError(
            f"Could not register required AI event handlers: {failures}"
        )

    _ensure_ai_table_exists()
    _logger.info("Active inference event handlers registered")


# ============================================================
# S5-04: ACTION RECOMMENDER (MCP-callable)
# ============================================================

def recommend_action(deterministic: bool = True) -> Dict:
    """Recommend the best action (or option) given current system state.

    This is the MCP-facing function: reads live state, evaluates EFE
    for all actions AND available options, and returns the recommendation.

    Sprint 6: Options compete with primitive actions via EFE. If an option
    is active, continues its intra-option policy until termination.

    Track 2: Step counter no longer mutates here — advances in observe_and_record().
    deterministic=True (default for MCP) uses argmin(EFE) instead of softmax sampling.

    Returns dict with:
        recommended: action/option name
        efe_scores: {name: G value} (lower = better)
        state: current state summary
        model_observations: how many transitions the model has learned
        rationale: human-readable explanation
        is_option: True if a multi-step option was selected
        option_step: current step within active option (if any)
    """
    global _active_option, _active_option_step, _pending_recommended

    state = get_current_state()
    model = get_model()

    # Sprint 4, item 4.3: Emotion modulates exploration temperature
    from modules.active_inference import EFE_SOFTMAX_TEMPERATURE
    temperature = EFE_SOFTMAX_TEMPERATURE
    if _last_ac > 0.05:
        temperature = min(8.0, temperature * (1.0 + _last_ac * 3.0))
    elif _last_ac < -0.05:
        temperature = max(1.0, temperature * (1.0 + _last_ac * 3.0))

    # Sprint 6: Check if we have an active option to continue
    is_option = False
    if _active_option is not None:
        # Check termination condition
        p_term = _active_option.termination_fn(state, _active_option_step)
        if p_term >= 1.0 or _active_option_step >= _active_option.expected_duration * 2:
            # Option complete — reset
            _logger.info("Option '%s' terminated at step %d", _active_option.name, _active_option_step)
            _active_option = None
            _active_option_step = 0
        else:
            # Continue active option — step advances in observe_and_record(), not here
            next_action = _active_option.policy_fn(state, _active_option_step)
            _pending_recommended = next_action.name
            is_option = True
            return {
                "recommended": next_action.name,
                "description": f"[Option: {_active_option.name} step {_active_option_step}] {next_action.description}",
                "efe_scores": {},
                "state": {
                    "topic": state.topic,
                    "uncertainty": round(state.uncertainty, 3),
                    "wm_load": round(state.wm_load, 3),
                    "pe_magnitude": round(state.pe_magnitude, 3),
                    "emotional_valence": round(state.emotional_valence, 3),
                },
                "model_observations": model.observation_count,
                "rationale": f"Continuing option '{_active_option.name}' (step {_active_option_step})",
                "is_option": True,
                "option_step": _active_option_step,
                "active_option": _active_option.name,
            }

    # No active option — select from primitives + new options
    # Sprint 6: Only include options when we have sufficient learned data
    _use_options = model.observation_count >= 10
    selected, efes = select_action(state, model, temperature=temperature, include_options=_use_options, deterministic=deterministic)

    # If an option was selected, activate it
    if isinstance(selected, Option):
        _active_option = selected
        _active_option_step = 0
        is_option = True
        # Recommend step 0 — step advances in observe_and_record(), not here
        first_action = _active_option.policy_fn(state, 0)
        _pending_recommended = first_action.name

        # CX-22: Emit option selection for metacognitive monitoring (L7→L5)
        try:
            from modules.events import event_bus, Events
            event_bus.emit(Events.ACTION_SELECTED, {
                "action": first_action.name,
                "efe_scores": {k: round(v, 4) for k, v in efes.items()},
                "efe_spread": round(max(efes.values()) - min(efes.values()), 4) if efes else 0.0,
                "temperature": temperature,
                "state_topic": state.topic,
                "state_uncertainty": state.uncertainty,
                "model_observations": model.observation_count,
                "is_option": True,
                "option_name": selected.name,
            })
        except Exception:
            pass

        return {
            "recommended": first_action.name,
            "description": f"[Starting option: {selected.name}] {first_action.description}",
            "efe_scores": {k: round(v, 4) for k, v in efes.items()},
            "state": {
                "topic": state.topic,
                "uncertainty": round(state.uncertainty, 3),
                "wm_load": round(state.wm_load, 3),
                "pe_magnitude": round(state.pe_magnitude, 3),
                "emotional_valence": round(state.emotional_valence, 3),
            },
            "model_observations": model.observation_count,
            "rationale": f"Option '{selected.name}' selected. Expected {selected.expected_duration} steps.",
            "is_option": True,
            "option_step": 0,
            "active_option": selected.name,
        }

    # Primitive action selected
    action = selected
    sorted_efes = sorted(efes.items(), key=lambda x: x[1])
    best_name, best_g = sorted_efes[0] if sorted_efes else (action.name, 0.0)

    if model.observation_count < 10:
        rationale = f"Low data ({model.observation_count} obs). Recommendation based mostly on priors and cost."
    elif state.uncertainty > 0.66:
        rationale = f"High uncertainty ({state.uncertainty:.2f}). Epistemic drive favors exploration."
    elif state.wm_load > 0.66:
        rationale = f"High WM load ({state.wm_load:.2f}). Consolidation or forgetting may help."
    elif state.pe_magnitude > 0.5:
        rationale = f"High PE ({state.pe_magnitude:.2f}). System is surprised — attending/exploring."
    else:
        rationale = f"Stable state. '{best_name}' has lowest EFE (G={best_g:.3f})."

    result = {
        "recommended": action.name,
        "description": action.description,
        "efe_scores": {k: round(v, 4) for k, v in efes.items()},
        "state": {
            "topic": state.topic,
            "uncertainty": round(state.uncertainty, 3),
            "wm_load": round(state.wm_load, 3),
            "pe_magnitude": round(state.pe_magnitude, 3),
            "emotional_valence": round(state.emotional_valence, 3),
        },
        "model_observations": model.observation_count,
        "rationale": rationale,
        "is_option": False,
        "option_step": 0,
        "active_option": None,
    }

    # CX-22: Emit action selection for metacognitive monitoring (L7→L5)
    try:
        from modules.events import event_bus, Events
        event_bus.emit(Events.ACTION_SELECTED, {
            "action": action.name,
            "efe_scores": result["efe_scores"],
            "efe_spread": round(max(efes.values()) - min(efes.values()), 4) if efes else 0.0,
            "temperature": temperature,
            "state_topic": state.topic,
            "state_uncertainty": state.uncertainty,
            "model_observations": model.observation_count,
        })
    except Exception:
        pass

    return result
