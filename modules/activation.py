"""
Codi Memory - Unified Activation Scoring (WIRING-5)
====================================================
Single canonical function for computing memory activation across all modules.

Foundation: ACT-R base-level activation (Anderson 2007)
  B_i = ln(sum(t_k^{-d}))  -- power-law decay over access history

Modulation signals (additive, pre-sigmoid):
  S = Spreading activation boost (0-1)
  E = Emotional modulation: arousal + congruence
  I = Importance boost (critical/high/medium/low)
  P = Prediction error boost (surprise signal)
  epsilon = Gaussian noise for stochastic retrieval

Final: A_final = sigmoid(B_i + W_s*S + W_e*E + W_i*I + W_p*P + epsilon)

Neuroscience references:
  - Anderson & Schooler 1991: Power-law forgetting
  - Anderson 2007: ACT-R 6.0 architecture
  - LaBar & Cabeza 2006: Emotional enhancement of memory
  - Bower 1981: Mood-congruent retrieval
  - Schultz 1997: Dopaminergic prediction error signals
  - Bahrick 1984: Semantic "permastore" durability
  - Baddeley 2000: Working memory faster decay

Created: 2026-02-13 (WIRING-5 - Phase 1)
"""

import logging
import math
import random
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Callable, Optional

logger = logging.getLogger(__name__)


# ============================================================
# CONSTANTS (Phase 0 calibrated - Feb 2026)
# ============================================================

# ACT-R Base-Level
ACTR_DECAY_DEFAULT = 0.40     # Power-law decay exponent (day/week scale)
ACTR_TIME_UNIT = 3600         # Normalize to hours
ACTR_MIN_ACTIVATION = -5.0    # Floor for ln(0) protection

# Sigmoid Normalization (calibrated: 30d/0acc->0.10, 1d/0acc->0.28, 1d/7acc->0.84)
SIGMOID_CENTER = -0.07
SIGMOID_SCALE = 0.91

# Component Weights (additive in pre-sigmoid space)
W_SPREAD = 0.30               # Anderson 2007: W_j typically 0.2-0.5
W_EMOTION = 0.15              # LaBar & Cabeza 2006: moderate effect size
W_IMPORTANCE = 0.20           # Proxy for hippocampal tagging (Redondo & Morris 2011)
W_PREDICTION = 0.15           # Schultz 1997: dopaminergic PE signal

# Noise
NOISE_SIGMA = 0.15            # Anderson 2007: sigma 0.1-0.25

# Decay Overrides by memory type
DECAY_EPISODIC = 0.40         # Default (Anderson & Schooler 1991)
DECAY_EPISODIC_CRITICAL = 0.20  # Flashbulb memory (Brown & Kulik 1977)
DECAY_EPISODIC_HIGH = 0.30
DECAY_EPISODIC_EMOTIONAL = 0.25  # Amygdala-mediated durability
DECAY_SEMANTIC = 0.15         # Bahrick 1984 "permastore"
DECAY_SEMANTIC_MIN = 0.05     # Floor for evidence-boosted semantic
DECAY_WORKING_MEMORY = 0.60   # WM turnover faster (Baddeley 2000)
EVIDENCE_DECAY_REDUCTION = 0.02  # Each evidence point reduces d

# Importance Boost Lookup
IMPORTANCE_BOOST = {
    'critical': 0.80,
    'high': 0.50,
    'medium': 0.20,
    'low': 0.00,
}


# ============================================================
# RESULT TYPE
# ============================================================

@dataclass
class ActivationResult:
    """Structured result from unified activation computation."""
    total: float          # Final activation, 0.0-1.0
    base_level: float     # ACT-R B_i component (pre-sigmoid)
    spread: float         # Spreading contribution (weighted)
    emotion: float        # Emotional contribution (weighted)
    importance_boost: float  # Importance contribution (weighted)
    surprise_boost: float    # Prediction error contribution (weighted)
    noise_value: float    # Stochastic noise applied


# ============================================================
# BASE-LEVEL ACTIVATION (moved from utils.py)
# ============================================================

def _compute_base_level(
    created_at_str: str,
    last_accessed_str: str = None,
    access_count: int = 0,
    access_timestamps: list = None,
    decay: float = ACTR_DECAY_DEFAULT,
) -> float:
    """Compute ACT-R base-level activation B_i = ln(sum(t_k^{-d})).

    If full access_timestamps are available, uses them directly.
    Otherwise, approximates by interpolating access times uniformly
    between creation and last access.

    Returns:
        Base-level activation (float). Higher = more accessible.
    """
    now = datetime.now()

    # Parse creation time (default: 30 days ago if missing)
    default_created = now - timedelta(days=30)
    try:
        if created_at_str:
            created_at = datetime.fromisoformat(str(created_at_str).replace('Z', '+00:00'))
            if created_at.tzinfo:
                created_at = created_at.replace(tzinfo=None)
        else:
            created_at = default_created
    except Exception:
        created_at = default_created

    # Best case: full access timestamps
    if access_timestamps and len(access_timestamps) > 0:
        total = 0.0
        parsed_any = False
        for ts_str in access_timestamps:
            try:
                ts = datetime.fromisoformat(str(ts_str).replace('Z', '+00:00'))
                if ts.tzinfo:
                    ts = ts.replace(tzinfo=None)
                t_hours = max(1.0, (now - ts).total_seconds() / ACTR_TIME_UNIT)
                total += t_hours ** (-decay)
                parsed_any = True
            except Exception:
                continue
        if total > 0:
            return math.log(total)
        if parsed_any:
            return ACTR_MIN_ACTIVATION

    # Approximation: interpolate access times
    t_created = max(1.0, (now - created_at).total_seconds() / ACTR_TIME_UNIT)

    # Parse last_accessed
    t_last = t_created
    if last_accessed_str:
        try:
            last_acc = datetime.fromisoformat(str(last_accessed_str).replace('Z', '+00:00'))
            if last_acc.tzinfo:
                last_acc = last_acc.replace(tzinfo=None)
            t_last = max(1.0, (now - last_acc).total_seconds() / ACTR_TIME_UNIT)
        except Exception:
            pass

    # Total presentations = access_count + 1 (original encoding)
    n_presentations = max(1, (access_count or 0) + 1)

    if n_presentations == 1:
        total = t_created ** (-decay)
    else:
        total = 0.0
        for k in range(n_presentations):
            frac = k / (n_presentations - 1) if n_presentations > 1 else 0
            t_k = t_created + frac * (t_last - t_created)
            t_k = max(1.0, t_k)
            total += t_k ** (-decay)

    if total > 0:
        return math.log(total)
    return ACTR_MIN_ACTIVATION


# ============================================================
# EFFECTIVE DECAY (by memory type)
# ============================================================

def compute_effective_decay(
    importance: str = "medium",
    memory_type: str = "episodic",
    emotional_arousal: float = 0.0,
    evidence_count: int = 0,
    decay_override: float = None,
) -> float:
    """Compute the effective decay parameter d based on memory type.

    Args:
        importance: narrative_importance string
        memory_type: "episodic" or "semantic" or "working_memory"
        emotional_arousal: abs(arousal) from PAD (0-1)
        evidence_count: for semantic facts, number of supporting episodes
        decay_override: explicit override (e.g., from prospective memory)

    Returns:
        decay parameter d (float)
    """
    if decay_override is not None:
        return max(DECAY_SEMANTIC_MIN, decay_override)

    if memory_type == "semantic":
        d = max(DECAY_SEMANTIC_MIN, DECAY_SEMANTIC - EVIDENCE_DECAY_REDUCTION * evidence_count)
        return d

    if memory_type == "working_memory":
        return DECAY_WORKING_MEMORY

    # Episodic: modulated by importance and emotion
    if importance == "critical":
        d = DECAY_EPISODIC_CRITICAL
    elif importance == "high":
        d = DECAY_EPISODIC_HIGH
    else:
        d = DECAY_EPISODIC

    # High emotional arousal slows decay (LaBar & Cabeza 2006)
    if emotional_arousal > 0.5:
        d = max(DECAY_EPISODIC_CRITICAL, d - 0.10)

    return d


# ============================================================
# UNIFIED ACTIVATION
# ============================================================

def compute_unified_activation(
    # Required: Identity
    memory_id: str = "",

    # Required: Temporal (ACT-R base-level)
    created_at: str = "",
    access_timestamps: list = None,
    last_accessed: str = None,
    access_count: int = 0,

    # Optional: Modulation signals
    spreading_boost: float = 0.0,
    emotional_arousal: float = 0.0,
    emotional_congruence: float = 0.0,
    importance: str = "medium",
    prediction_error: float = 0.0,

    # Control
    noise: bool = True,
    decay_override: float = None,
    memory_type: str = "episodic",
    evidence_count: int = 0,
    trace_callback: Optional[Callable] = None,
) -> ActivationResult:
    """Compute unified activation for any memory in the system.

    Combines ACT-R base-level activation with additive modulation signals,
    then normalizes via calibrated sigmoid to [0, 1].

    Args:
        memory_id: UUID of the memory (for logging)
        created_at: ISO timestamp of creation
        access_timestamps: Full access history (preferred)
        last_accessed: Fallback if no timestamps
        access_count: Fallback if no timestamps
        spreading_boost: From spreading activation (0-1)
        emotional_arousal: Current PAD arousal magnitude (0-1)
        emotional_congruence: Mood-memory valence match (-1 to 1)
        importance: "critical"|"high"|"medium"|"low"
        prediction_error: Surprise signal (0-1)
        noise: Stochastic retrieval noise
        decay_override: Override base decay (e.g., for PM intentions)
        memory_type: "episodic"|"semantic"|"working_memory"
        evidence_count: For semantic facts, number of supporting episodes
        trace_callback: Optional fn(memory_id, result, decay) for observability

    Returns:
        ActivationResult with total activation and component breakdown.
    """
    # 1. Compute effective decay
    d = compute_effective_decay(
        importance=importance,
        memory_type=memory_type,
        emotional_arousal=emotional_arousal,
        evidence_count=evidence_count,
        decay_override=decay_override,
    )

    # 2. Base-level activation B_i
    base_level = _compute_base_level(
        created_at_str=created_at,
        last_accessed_str=last_accessed,
        access_count=access_count,
        access_timestamps=access_timestamps,
        decay=d,
    )

    # 3. Spreading activation (clamped 0-1)
    s_val = max(0.0, min(1.0, spreading_boost))
    spread_component = W_SPREAD * s_val

    # 4. Emotional modulation
    # E = abs(arousal) * 0.6 + congruence * 0.4 (Bower 1981, LaBar & Cabeza 2006)
    e_raw = abs(emotional_arousal) * 0.6 + max(0.0, emotional_congruence) * 0.4
    e_val = max(0.0, min(1.0, e_raw))
    emotion_component = W_EMOTION * e_val

    # 5. Importance boost
    i_val = IMPORTANCE_BOOST.get(importance, 0.2)
    importance_component = W_IMPORTANCE * i_val

    # 6. Prediction error boost
    p_val = max(0.0, min(1.0, prediction_error))
    prediction_component = W_PREDICTION * p_val

    # 7. Noise
    epsilon = random.gauss(0, NOISE_SIGMA) if noise else 0.0

    # 8. Raw activation (additive in pre-sigmoid space)
    raw_activation = (
        base_level
        + spread_component
        + emotion_component
        + importance_component
        + prediction_component
        + epsilon
    )

    # 9. Sigmoid normalization to [0, 1]
    shifted = (raw_activation - SIGMOID_CENTER) * SIGMOID_SCALE
    total = 1.0 / (1.0 + math.exp(-max(-500, min(500, shifted))))

    result = ActivationResult(
        total=round(total, 4),
        base_level=round(base_level, 4),
        spread=round(spread_component, 4),
        emotion=round(emotion_component, 4),
        importance_boost=round(importance_component, 4),
        surprise_boost=round(prediction_component, 4),
        noise_value=round(epsilon, 4),
    )

    # Observability hook for metacognitive monitor (WIRING-5 P0-2)
    if trace_callback is not None:
        try:
            trace_callback(memory_id=memory_id, result=result, decay=d)
        except Exception:
            logger.exception(
                "Activation trace callback failed for memory_id=%s callback=%s",
                memory_id,
                getattr(trace_callback, "__qualname__", getattr(trace_callback, "__name__", trace_callback.__class__.__name__)),
            )

    return result
