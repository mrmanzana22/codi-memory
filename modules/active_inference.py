"""
Codi Memory - Active Inference Foundation (Sprint 4)
=====================================================
Implements the core Active Inference framework (Friston 2010, 2017):
  S4-01: Discrete action space
  S4-02: Simple generative model (state transitions)
  S4-03: Expected Free Energy (EFE) computation + policy selection

The system transitions from REACTIVE (respond to inputs) to
POLICY-SELECTING (choose actions that minimize free energy).

Design: Pure logic, no Qdrant/SQLite dependencies. Fully unit-testable.
State reading happens at the boundary (get_current_state()).

Key equations:
  G(a) = -pragmatic(a) - epistemic(a) + cost(a)
  pragmatic = -D_KL[Q(o|a) || P(o)]  (outcomes close to preferences)
  epistemic = H[Q(s|a)] - E[H[Q(s|o,a)]]  (information gain)
  P(a) = softmax(-gamma * G(a))  (policy selection)

References:
  - Friston 2010: The free-energy principle: a unified brain theory?
  - Friston 2017: Active inference, curiosity and insight
  - Parr & Friston 2019: Generalised free energy and active inference
  - Da Costa et al. 2020: Active inference on discrete state-spaces

Created: 2026-02-28 (Sprint 4 - IMPLEMENTATION_PLAYBOOK)
"""

import json
import logging
import math
import random
import sqlite3
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple

_logger = logging.getLogger(__name__)


# ============================================================
# S4-01: ACTION SPACE (Discrete actions)
# ============================================================

@dataclass(frozen=True)
class Action:
    """A discrete action the system can take.

    Friston 2017: Active inference agents select actions (policies)
    that minimize expected free energy.
    """
    name: str
    description: str
    cost: float           # Computational cost 0-1 (penalizes expensive actions)
    expected_ms: int      # Typical execution time in ms


# Canonical action repertoire
RETRIEVE = Action("retrieve", "Search episodic/semantic memory", 0.3, 200)
STORE = Action("store", "Save new memory to long-term store", 0.4, 500)
CONSOLIDATE = Action("consolidate", "Run consolidation pipeline", 0.8, 5000)
FORGET = Action("forget", "Apply salience decay and pruning", 0.5, 1000)
ATTEND = Action("attend", "Focus attention on a topic", 0.2, 100)
EXPLORE = Action("explore", "Curiosity-driven search for knowledge gaps", 0.4, 300)

ALL_ACTIONS = [RETRIEVE, STORE, CONSOLIDATE, FORGET, ATTEND, EXPLORE]
ACTION_MAP = {a.name: a for a in ALL_ACTIONS}


# ============================================================
# SPRINT 6: OPTIONS FRAMEWORK (<I, pi, beta>) — Sutton et al. 1999
# ============================================================

@dataclass
class Option:
    """Semi-MDP option: temporally extended action.

    Sutton, Precup & Singh 1999: Options extend MDPs with multi-step
    macro-actions. Each option = <I, pi, beta>:
      I    — initiation set: states where option can start
      pi   — intra-option policy: primitive actions to take
      beta — termination: when to stop

    Args:
        name: Option identifier
        description: Human-readable purpose
        initiation_fn: (state) -> bool — can we start this option now?
        policy_fn: (state, step) -> Action — next primitive action
        termination_fn: (state, steps) -> float — P(terminate)
        expected_duration: Typical number of steps
        cost: Computational cost overhead (additive to constituent actions)
    """
    name: str
    description: str
    initiation_fn: object      # Callable[[SystemState], bool]
    policy_fn: object          # Callable[[SystemState, int], Action]
    termination_fn: object     # Callable[[SystemState, int], float]
    expected_duration: int = 3
    cost: float = 0.1


# ============================================================
# 6.2: Initiation conditions (I)
# ============================================================

def _deep_research_initiation(state: "SystemState") -> bool:
    """Start when high uncertainty — need to explore to resolve."""
    return state.uncertainty > 0.5


def _consolidation_cycle_initiation(state: "SystemState") -> bool:
    """Start when WM overloaded — need to offload to long-term."""
    return state.wm_load > 0.7


def _topic_exploration_initiation(state: "SystemState") -> bool:
    """Start when low PE for 3+ turns — knowledge feels stale (proxy: low PE)."""
    return state.pe_magnitude < 0.2 and state.uncertainty > 0.3


def _goal_pursuit_initiation(state: "SystemState") -> bool:
    """Start when on-topic with moderate uncertainty — time to resolve."""
    return state.topic != "general" and 0.2 < state.uncertainty < 0.7


# ============================================================
# 6.3: Intra-option policies (pi)
# ============================================================

def _deep_research_policy(state: "SystemState", step: int) -> "Action":
    """retrieve → explore → retrieve pattern."""
    sequence = [RETRIEVE, EXPLORE, RETRIEVE]
    return sequence[min(step, len(sequence) - 1)]


def _consolidation_policy(state: "SystemState", step: int) -> "Action":
    """consolidate → forget → attend pattern."""
    sequence = [CONSOLIDATE, FORGET, ATTEND]
    return sequence[min(step, len(sequence) - 1)]


def _topic_exploration_policy(state: "SystemState", step: int) -> "Action":
    """explore → attend → store pattern."""
    sequence = [EXPLORE, ATTEND, STORE]
    return sequence[min(step, len(sequence) - 1)]


def _goal_pursuit_policy(state: "SystemState", step: int) -> "Action":
    """retrieve → attend → store pattern."""
    sequence = [RETRIEVE, ATTEND, STORE]
    return sequence[min(step, len(sequence) - 1)]


# ============================================================
# 6.4: Termination conditions (beta)
# ============================================================

def _deep_research_termination(state: "SystemState", steps: int) -> float:
    """Terminate when uncertainty resolved or exceeded max steps."""
    if state.uncertainty < 0.3 or steps >= 5:
        return 1.0
    return steps / 10.0  # Gradual: P(term) grows with steps


def _consolidation_termination(state: "SystemState", steps: int) -> float:
    """Terminate when WM lightened or exceeded max steps."""
    if state.wm_load < 0.4 or steps >= 3:
        return 1.0
    return 0.2


def _topic_exploration_termination(state: "SystemState", steps: int) -> float:
    """Terminate when topic shifts (high PE) or exceeded max steps."""
    if state.pe_magnitude > 0.5 or steps >= 4:
        return 1.0
    return 0.15


def _goal_pursuit_termination(state: "SystemState", steps: int) -> float:
    """Terminate when topic changes or exceeded max steps."""
    if steps >= 6:
        return 1.0
    return 0.1


# Canonical option repertoire
DEEP_RESEARCH = Option(
    "deep_research",
    "Multi-search exploration to resolve uncertainty",
    _deep_research_initiation,
    _deep_research_policy,
    _deep_research_termination,
    expected_duration=3,
    cost=0.15,
)

CONSOLIDATION_CYCLE = Option(
    "consolidation_cycle",
    "Full consolidation + forgetting pipeline",
    _consolidation_cycle_initiation,
    _consolidation_policy,
    _consolidation_termination,
    expected_duration=3,
    cost=0.2,
)

TOPIC_EXPLORATION = Option(
    "topic_exploration",
    "Curiosity-driven topic exploration sequence",
    _topic_exploration_initiation,
    _topic_exploration_policy,
    _topic_exploration_termination,
    expected_duration=3,
    cost=0.12,
)

GOAL_PURSUIT = Option(
    "goal_pursuit",
    "Focused retrieval and storage for session goal",
    _goal_pursuit_initiation,
    _goal_pursuit_policy,
    _goal_pursuit_termination,
    expected_duration=4,
    cost=0.1,
)

ALL_OPTIONS = [DEEP_RESEARCH, CONSOLIDATION_CYCLE, TOPIC_EXPLORATION, GOAL_PURSUIT]
OPTION_MAP = {o.name: o for o in ALL_OPTIONS}


# ============================================================
# 6.5: Option-level EFE
# ============================================================

def compute_option_efe(
    state: "SystemState",
    option: Option,
    model: "GenerativeModel",
    temperature: float = None,
) -> float:
    """Compute Expected Free Energy for a temporally extended option.

    G_option = sum(G(a_i) * P(step_i)) + option.cost
    where P(step_i) = P(not terminated at step i-1) (geometric discounting).

    Sutton et al. 1999: options evaluated by aggregated value over duration.

    Returns:
        Option EFE (lower = better), same scale as primitive action EFE.
    """
    if temperature is None:
        temperature = EFE_SOFTMAX_TEMPERATURE

    n = option.expected_duration
    total_efe = 0.0
    survival_prob = 1.0  # P(still running at step i)

    # Simulate option's intra-policy steps with state rollout (#007)
    sim_state = state
    for step in range(n):
        action = option.policy_fn(sim_state, step)
        step_efe = compute_efe(sim_state, action, model)
        total_efe += survival_prob * step_efe

        # P(terminate at this step) — elapsed count = step + 1 (1-based)
        p_term = option.termination_fn(sim_state, step + 1)
        survival_prob *= (1.0 - p_term)
        if survival_prob < 0.01:
            break

        # Advance sim_state to MAP predicted next state
        next_dist = model.predict_next_state(sim_state, action)
        if next_dist:
            map_tuple = max(next_dist, key=next_dist.get)
            _disc_to_val = {"low": 0.17, "mid": 0.5, "high": 0.83}
            sim_state = SystemState(
                topic=map_tuple[0] if len(map_tuple) > 0 else sim_state.topic,
                uncertainty=_disc_to_val.get(map_tuple[1], sim_state.uncertainty) if len(map_tuple) > 1 else sim_state.uncertainty,
                wm_load=_disc_to_val.get(map_tuple[2], sim_state.wm_load) if len(map_tuple) > 2 else sim_state.wm_load,
                pe_magnitude=sim_state.pe_magnitude,
                emotional_valence=sim_state.emotional_valence,
            )

    # Add option overhead cost
    total_efe += option.cost

    return total_efe


def get_available_options(state: "SystemState") -> List[Option]:
    """Return options whose initiation conditions are met in this state."""
    return [opt for opt in ALL_OPTIONS if opt.initiation_fn(state)]


# ============================================================
# S4-02: SYSTEM STATE + GENERATIVE MODEL
# ============================================================

@dataclass
class SystemState:
    """Observable system state at time t.

    Da Costa et al. 2020: discrete state-space with factored dimensions.
    """
    topic: str              # Current attention topic
    uncertainty: float      # Prediction entropy [0, 1]
    wm_load: float          # Working memory utilization [0, 1]
    pe_magnitude: float     # Last prediction error magnitude [0, 1]
    emotional_valence: float  # PAD pleasure [-1, 1]

    def as_tuple(self) -> tuple:
        """Discretized state for transition counting."""
        return (
            self.topic,
            "low" if self.uncertainty < 0.33 else "mid" if self.uncertainty < 0.66 else "high",
            "low" if self.wm_load < 0.33 else "mid" if self.wm_load < 0.66 else "high",
        )


# Default preferred state distribution (goals)
DEFAULT_PREFERENCES = {
    "low_uncertainty": 0.8,     # Prefer certainty (Friston 2010: minimize surprise)
    "moderate_wm": 0.6,         # Prefer moderate WM load (not empty, not overloaded)
    "low_pe": 0.7,              # Prefer low prediction error (model accuracy)
}


class GenerativeModel:
    """Simple generative model: P(s'|s, a) learned from interaction history.

    Friston 2017: The generative model encodes beliefs about how states
    evolve given actions. Active inference selects actions by evaluating
    the expected free energy under this model.

    Transition model: Dirichlet-Multinomial (conjugate Bayesian).
    Observation model: identity (we observe states directly).
    """

    def __init__(self, preferences: Dict[str, float] = None):
        # P(s'|s, a): transition counts (Dirichlet prior alpha=1)
        self._transition_counts: Dict[Tuple, Counter] = defaultdict(Counter)
        self._total_observations = 0
        self.preferences = preferences or DEFAULT_PREFERENCES.copy()

    def update(self, prev_state: SystemState, action: Action, next_state: SystemState):
        """Learn from observed transition (Bayesian update).

        Dirichlet-Multinomial: increment count for observed transition.
        """
        key = (prev_state.as_tuple(), action.name)
        self._transition_counts[key][next_state.as_tuple()] += 1
        self._total_observations += 1

    def predict_next_state(self, state: SystemState, action: Action) -> Dict[tuple, float]:
        """P(s'|s, a) - predicted next state distribution.

        Uses Dirichlet posterior with alpha=1 prior (uniform).
        Falls back to uniform if no observations for this (s, a) pair.
        """
        key = (state.as_tuple(), action.name)
        counts = self._transition_counts.get(key)

        if not counts:
            # No data: uniform prior over known states
            all_states = set()
            for counter in self._transition_counts.values():
                all_states.update(counter.keys())
            if not all_states:
                return {state.as_tuple(): 1.0}  # Stay in current state
            p = 1.0 / len(all_states)
            return {s: p for s in all_states}

        # Dirichlet posterior: (count + alpha) / (total + alpha * K)
        alpha = 1.0
        total = sum(counts.values())
        k = len(counts)
        return {
            s: (c + alpha) / (total + alpha * k)
            for s, c in counts.items()
        }

    def get_expected_uncertainty(self, state: SystemState, action: Action) -> float:
        """Expected uncertainty after taking action (for epistemic value).

        H[P(s'|s,a)] = entropy of predicted next state distribution.
        Higher entropy = more uncertain about what will happen.
        """
        dist = self.predict_next_state(state, action)
        return _entropy(dist)

    @property
    def observation_count(self) -> int:
        return self._total_observations

    # ---- S5-01: Persistence (Da Costa et al. 2020) ----

    def persist(self, conn: sqlite3.Connection) -> int:
        """Save transition counts to SQLite.

        Serializes Dirichlet counts so the generative model survives
        across sessions. Each (state, action) -> next_state count is
        stored as one row.

        Returns number of rows written.
        """
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
        now = datetime.now().astimezone().isoformat()
        rows = 0
        for (state_tuple, action_name), counter in self._transition_counts.items():
            for next_state, count in counter.items():
                conn.execute("""
                    INSERT INTO generative_model_transitions
                        (state_key, action, next_state, count, updated_at)
                    VALUES (?, ?, ?, ?, ?)
                    ON CONFLICT(state_key, action, next_state)
                    DO UPDATE SET count = ?, updated_at = ?
                """, (
                    json.dumps(state_tuple),
                    action_name,
                    json.dumps(next_state),
                    count, now,
                    count, now,
                ))
                rows += 1
        conn.commit()
        return rows

    @classmethod
    def load(cls, conn: sqlite3.Connection, preferences: Dict[str, float] = None) -> "GenerativeModel":
        """Restore generative model from SQLite.

        Reconstructs Dirichlet counts from persisted rows.
        Returns a fresh model if the table doesn't exist yet.
        """
        model = cls(preferences=preferences)
        try:
            rows = conn.execute("""
                SELECT state_key, action, next_state, count
                FROM generative_model_transitions
            """).fetchall()
        except sqlite3.OperationalError as exc:
            if "no such table" in str(exc).lower():
                # Table doesn't exist yet — return empty model
                return model
            raise

        for state_key_json, action_name, next_state_json, count in rows:
            key = (tuple(json.loads(state_key_json)), action_name)
            next_state = tuple(json.loads(next_state_json))
            model._transition_counts[key][next_state] = count
            model._total_observations += count

        return model


# ============================================================
# S4-03: EXPECTED FREE ENERGY + POLICY SELECTION
# ============================================================

# Temperature for policy selection (same scale as S2-03 competition)
EFE_SOFTMAX_TEMPERATURE = 4.0


def compute_efe(
    state: SystemState,
    action: Action,
    model: GenerativeModel,
) -> float:
    """Compute Expected Free Energy G(a) for a single action.

    G(a) = -pragmatic(a) - epistemic(a) + cost(a)

    Lower G = better action. Active inference selects policies that
    minimize expected free energy (Friston 2017, Parr & Friston 2019).

    Components:
        pragmatic: alignment with preferred outcomes (exploit)
        epistemic: H[P(s'|s,a)] — transition entropy drives exploration.
                   Higher entropy = more to learn = lower G (explore).
        cost: computational cost penalty

    Returns:
        G value (lower = better). Not bounded.
    """
    # Pragmatic value: how close predicted outcomes are to preferences
    pragmatic = _compute_pragmatic_value(state, action, model)

    # Epistemic value: transition entropy — higher = more informative action
    epistemic = _compute_epistemic_value(state, action, model)

    # Cost penalty (normalized 0-1)
    cost = action.cost

    # G = -pragmatic - epistemic + cost  (Friston 2017)
    # Lower G = more pragmatic, more exploratory, less costly
    return -pragmatic - epistemic + cost


def select_action(
    state: SystemState,
    model: GenerativeModel,
    actions: List[Action] = None,
    temperature: float = EFE_SOFTMAX_TEMPERATURE,
    include_options: bool = None,  # None = auto: True only when actions=None
    deterministic: bool = False,
) -> Tuple[object, Dict[str, float]]:
    """Select action (or option) via softmax over negative EFE (policy selection).

    Sprint 6: Extends selection to also consider temporally extended options
    alongside primitive actions. Options compete with primitives via EFE.

    P(a) = exp(-gamma * G(a)) / Z

    Friston 2017: Active inference agents select the action (policy)
    with lowest expected free energy. Temperature controls exploration.

    Track 2: deterministic=True returns argmin(EFE) directly (no sampling).

    Returns:
        (selected_action_or_option, {name: efe_value}) — option or Action
    """
    custom_actions_provided = actions is not None
    if actions is None:
        actions = ALL_ACTIONS

    if not actions:
        return ATTEND, {}

    # Auto-detect include_options: only when using full action set (not custom subset)
    if include_options is None:
        include_options = not custom_actions_provided

    # Compute EFE for each primitive action
    efes = {}
    for action in actions:
        efes[action.name] = compute_efe(state, action, model)

    # Sprint 6.5: Include available options in competition
    option_efes = {}
    available_options = []
    if include_options:
        available_options = get_available_options(state)
        for opt in available_options:
            option_efes[opt.name] = compute_option_efe(state, opt, model, temperature)

    # Build combined candidate pool
    all_efes = {**efes, **option_efes}
    all_candidates = list(actions) + available_options  # Action | Option

    # Track 2: Deterministic mode — argmin(EFE), no sampling
    if deterministic:
        best_name = min(all_efes, key=all_efes.get)
        best_candidate = next(c for c in all_candidates if c.name == best_name)
        return best_candidate, all_efes

    # Softmax selection over combined pool
    min_g = min(all_efes.values())
    weights = {}
    for name, g in all_efes.items():
        weights[name] = math.exp(-temperature * (g - min_g))

    total_w = sum(weights.values())
    if total_w <= 0:
        return actions[0], efes

    # Weighted random selection
    r = random.random()
    cumulative = 0.0
    for candidate in all_candidates:
        cname = candidate.name
        cumulative += weights.get(cname, 0) / total_w
        if r <= cumulative:
            return candidate, all_efes

    return all_candidates[-1], all_efes


def _compute_pragmatic_value(
    state: SystemState,
    action: Action,
    model: GenerativeModel,
) -> float:
    """Pragmatic value: expected alignment with preferences.

    Measures how much the predicted outcomes match desired states.
    Higher = more aligned with goals.
    """
    dist = model.predict_next_state(state, action)
    prefs = model.preferences

    value = 0.0
    for next_state, prob in dist.items():
        # Score each predicted state against preferences
        topic, uncertainty_level, wm_level = next_state

        # Preference alignment scoring
        state_score = 0.0
        if uncertainty_level == "low":
            state_score += prefs.get("low_uncertainty", 0.5)
        elif uncertainty_level == "mid":
            state_score += prefs.get("low_uncertainty", 0.5) * 0.5

        if wm_level == "mid":
            state_score += prefs.get("moderate_wm", 0.5)
        elif wm_level == "low":
            state_score += prefs.get("moderate_wm", 0.5) * 0.3

        state_score += prefs.get("low_pe", 0.5) * 0.5  # Base PE preference

        value += prob * state_score

    return value


def _compute_epistemic_value(
    state: SystemState,
    action: Action,
    model: GenerativeModel,
) -> float:
    """Epistemic value: expected information gain from action.

    H[P(s'|s,a)] — entropy of predicted next state distribution.
    Higher entropy = more uncertain transitions = more to learn.

    Subtracted in EFE (Friston 2017): actions with uncertain outcomes
    are preferred for exploration (curiosity-driven behavior).
    """
    return model.get_expected_uncertainty(state, action)


def get_policy_distribution(
    state: SystemState,
    model: GenerativeModel,
    actions: List[Action] = None,
    temperature: float = EFE_SOFTMAX_TEMPERATURE,
) -> Dict[str, float]:
    """Get the full policy distribution P(a) = softmax(-gamma * G(a)).

    Returns {action_name: probability} — the agent's current policy beliefs.
    Used for AC computation (Sprint 4, item 4.1).
    """
    if actions is None:
        actions = ALL_ACTIONS
    if not actions:
        return {}

    efes = {a.name: compute_efe(state, a, model) for a in actions}

    # Softmax: P(a) = exp(-gamma * G(a)) / Z
    min_g = min(efes.values())
    weights = {}
    for a in actions:
        weights[a.name] = math.exp(-temperature * (efes[a.name] - min_g))

    total_w = sum(weights.values())
    if total_w <= 0:
        p = 1.0 / len(actions)
        return {a.name: p for a in actions}

    return {name: w / total_w for name, w in weights.items()}


def compute_affective_charge(
    prior_policy: Dict[str, float],
    posterior_policy: Dict[str, float],
    efes: Dict[str, float],
) -> float:
    """Compute Affective Charge from policy shift and EFE landscape.

    AC = (pi_bar - pi) . G

    Sprint 4, item 4.1 (CC-2). Joffily & Coricelli 2013, Hesp et al. 2021.

    Interpretation:
      AC > 0: observation shifted policy toward BETTER actions (lower G) → positive valence
      AC < 0: observation shifted policy toward WORSE actions (higher G) → negative valence
      AC ≈ 0: observation didn't change policy beliefs much → neutral

    The sign convention: (prior - posterior) dot G. If posterior shifts mass
    toward low-G actions, the dot product is positive because prior had MORE
    mass on high-G actions (which are subtracted).

    Args:
        prior_policy: P(a) before observation {action_name: prob}
        posterior_policy: P(a) after observation {action_name: prob}
        efes: G(a) per action {action_name: efe_value}

    Returns:
        AC scalar. Positive = good news, negative = bad news.
    """
    ac = 0.0
    for action_name in efes:
        pi_bar = prior_policy.get(action_name, 0.0)
        pi = posterior_policy.get(action_name, 0.0)
        g = efes[action_name]
        ac += (pi_bar - pi) * g
    return ac


def _entropy(dist: Dict) -> float:
    """Shannon entropy of a distribution."""
    h = 0.0
    for p in dist.values():
        if p > 0:
            h -= p * math.log2(p)
    return h


# ============================================================
# STATE READER (boundary function — reads live system state)
# ============================================================

def get_current_state() -> SystemState:
    """Read the current system state from live modules.

    This is the ONLY function with external dependencies.
    Everything else is pure logic.
    """
    topic = "general"
    uncertainty = 0.5
    wm_load = 0.5
    pe_magnitude = 0.0
    valence = 0.0

    # Read attention schema
    try:
        from modules.wiring import get_attention_schema
        schema = get_attention_schema()
        topic = schema.get("current_focus") or "general"
        pe_magnitude = schema.get("attention_prediction_error", 0.0)
    except ImportError as exc:
        _logger.warning("Active inference state read: attention schema unavailable: %s", exc)

    # Read WM load
    try:
        from modules.working_memory import wm_get_active_count
        count = wm_get_active_count()
        wm_load = min(1.0, count / 10.0)  # 10 items = full
    except ImportError as exc:
        _logger.warning("Active inference state read: working memory unavailable: %s", exc)

    # Read emotional state
    try:
        from modules.config import _emotional_state
        valence = _emotional_state.get("current", {}).get("pleasure", 0.0)
    except ImportError as exc:
        _logger.warning("Active inference state read: emotional state unavailable: %s", exc)

    return SystemState(
        topic=topic,
        uncertainty=uncertainty,
        wm_load=wm_load,
        pe_magnitude=pe_magnitude,
        emotional_valence=valence,
    )
