"""
Codi Memory - Forgetting Module
=================================
Implements neurally-plausible forgetting mechanisms.

Mechanisms:
  1. FadeMem importance-modulated forgetting curves (replaces flat salience decay)
  2. RIF (Retrieval-Induced Forgetting) - suppress competitors on retrieval

Neuroscience references:
  - Ebbinghaus 1885: Exponential forgetting curve
  - Wixted & Ebbesen 1991: Power-law fits better for LTM
  - Anderson & Schooler 1991: Power-law environmental statistics
  - Bahrick 1984: "Permastore" for consolidated semantic
  - Brown & Kulik 1977: Flashbulb memory emotional protection
  - Tononi & Cirelli 2003: Synaptic homeostasis (sleep downscales weak)
  - Anderson, Bjork & Bjork 1994: Retrieval-Induced Forgetting
  - Storm & Levy 2012: RIF is item-specific inhibition
"""

import logging
import math
import threading
from datetime import datetime, timezone
from typing import Optional

from modules.pg_store import pg

_logger = logging.getLogger(__name__)

# ============================================================
# FADEM CONSTANTS
# ============================================================

# Base decay rate. Calibrated so a medium-importance
# unconsolidated memory loses ~40-55% salience in 24h with no access.
FADEM_LAMBDA_BASE = 0.008

# S0-04: Power-law scaling factor. Power-law form (1+alpha*t)^{-d} needs
# higher alpha than exp(-lambda*t^beta) to achieve similar 24h targets.
# Wixted & Ebbesen 1991: power-law fits LTM better than exponential.
FADEM_PL_SCALE = 6.0

# Importance sensitivity: higher = importance protects more.
# S0-04: Increased from 0.5 to 1.0 for power-law form (needs wider spread
# between importance levels since power-law has narrower dynamic range).
FADEM_MU = 1.0

# Minimum salience floor (memories never fully disappear)
FADEM_FLOOR = 0.1

# Shape parameter (beta) by consolidation/type
# Lower beta = slower decay (sub-linear time progression)
FADEM_BETA = {
    "consolidated_semantic": 0.25,   # Bahrick 1984 permastore (power-law fat tail)
    "consolidated_episodic": 0.6,    # Consolidated episodic (lowered from 0.8 for power-law)
    "unconsolidated": 1.2,           # Fast decay (Tononi & Cirelli 2003)
    "default": 1.0,                  # Unknown status
}

# Importance mapping for decay modulation
# Proposal #66 Fix 2: Boost critical protection (safety net for future removal of must_not)
FADEM_IMPORTANCE = {
    "critical": 1.5,
    "high": 0.8,
    "medium": 0.5,
    "low": 0.2,
}

# Emotional protection: high-arousal memories resist decay
# Brown & Kulik 1977 flashbulb effect
FADEM_AROUSAL_THRESHOLD = 0.5
FADEM_AROUSAL_SHIELD_MAX = 0.4  # Up to 40% decay reduction at arousal=1.0


# ============================================================
# CORE FUNCTION
# ============================================================

def compute_fadem_strength(
    current_salience: float,
    hours_since_access: float,
    importance: str = "medium",
    consolidated: bool = False,
    memory_type: str = "episodic",
    emotional_arousal: float = 0.0,
    decay_multiplier: float = 1.0,
) -> float:
    """Compute new salience using FadeMem importance-modulated decay.

    Args:
        current_salience: Current attention_salience (0-1)
        hours_since_access: Hours since last access or creation
        importance: narrative_importance level
        consolidated: Whether the memory has been consolidated
        memory_type: "episodic" or "semantic"
        emotional_arousal: abs(arousal) from PAD at encoding (0-1)
        decay_multiplier: External scaling factor (e.g., 0.6 for lifecycle gentle decay)

    Returns:
        New salience value in [FADEM_FLOOR, current_salience]
    """
    if hours_since_access <= 0:
        return current_salience

    # Critical memories never decay (P0 Bug #3 fix)
    # These are identity/architectural facts that must persist indefinitely.
    if importance == "critical":
        return current_salience

    # Step 1: Importance modulates lambda (higher importance = slower decay)
    imp = FADEM_IMPORTANCE.get(importance, 0.5)
    lambda_i = FADEM_LAMBDA_BASE * math.exp(-FADEM_MU * imp) * decay_multiplier

    # Step 2: Beta by consolidation status and type
    if consolidated:
        if memory_type == "semantic":
            beta = FADEM_BETA["consolidated_semantic"]
        else:
            beta = FADEM_BETA["consolidated_episodic"]
    else:
        beta = FADEM_BETA["unconsolidated"]

    # Step 3: Core FadeMem formula — Power-law (S0-04, G-INV-06)
    # Was: stretched exponential exp(-λ*t^β) — contradicted docstring.
    # Now: power-law R(t) = (1 + α*t)^{-β} (Wixted & Ebbesen 1991)
    # α = lambda_i * PL_SCALE (recalibrated for power-law form)
    # Properties: faster initial decay, fat tail (slower long-term than exp).
    # Anderson & Schooler 1991: environmental statistics are power-law.
    alpha = lambda_i * FADEM_PL_SCALE
    decay_factor = (1.0 + alpha * hours_since_access) ** (-beta)

    # Step 4: Emotional protection (high arousal memories decay slower)
    if emotional_arousal > FADEM_AROUSAL_THRESHOLD:
        # Linear shield scaling from threshold to 1.0
        shield = (emotional_arousal - FADEM_AROUSAL_THRESHOLD) / (1.0 - FADEM_AROUSAL_THRESHOLD)
        shield *= FADEM_AROUSAL_SHIELD_MAX  # 0 to 0.4
        # Reduce the decay amount by shield fraction
        decay_factor = 1.0 - (1.0 - decay_factor) * (1.0 - shield)

    new_salience = current_salience * decay_factor
    return max(FADEM_FLOOR, new_salience)


# ============================================================
# K.1.4: DUAL STRENGTH MODEL (SS/RS)
# Bjork & Bjork 1992: "A New Theory of Disuse"
# ============================================================

# Learning rate for SS growth per retrieval
SS_LEARNING_RATE = 0.15

# Base decay rate for RS (before SS modulation)
RS_BASE_DECAY = 0.01

# RS power-law scale (matches FADEM_PL_SCALE)
RS_PL_SCALE = 6.0

# Vault threshold sits just above FadeMem floor so dormant memories stop
# competing in normal retrieval but remain stored for cue-based reactivation.
VAULT_RS_THRESHOLD = 0.12
VAULT_MIN_AGE_DAYS = 7
VAULT_REACTIVATION_RS = 0.6


def compute_fadem_strength_ss_rs(
    ss: float,
    rs: float,
    hours_since_access: float,
    importance: str = "medium",
    consolidated: bool = False,
    memory_type: str = "episodic",
    emotional_arousal: float = 0.0,
    retrieval_event: bool = False,
) -> tuple:
    """Compute dual-strength decay using SS/RS model (Bjork & Bjork 1992).

    K.1.4: Adds Storage Strength (SS) and Retrieval Strength (RS) as
    independent memory dimensions. SS grows monotonically with retrievals;
    RS decays with power-law but inversely proportional to SS.

    The "spacing effect" emerges naturally: low RS at retrieval time
    means higher difficulty bonus for SS growth.

    Args:
        ss: Storage Strength (0-1, grows monotonically)
        rs: Retrieval Strength (0-1, decays over time)
        hours_since_access: Hours since last retrieval
        importance: narrative_importance level
        consolidated: Whether the memory has been consolidated
        memory_type: "episodic" or "semantic"
        emotional_arousal: abs(arousal) from PAD at encoding (0-1)
        retrieval_event: If True, update SS and reset RS (retrieval just happened)

    Returns:
        Tuple of (new_ss, new_rs, effective_strength)
        effective_strength is the combined accessibility score (0-1).
    """
    # Step 1: Handle retrieval event (SS grows, RS resets)
    if retrieval_event:
        # Desirable difficulty: low RS → harder retrieval → more SS learning
        difficulty_bonus = max(0.5, 1.5 - rs)
        ss_new = ss + (1.0 - ss) * SS_LEARNING_RATE * difficulty_bonus
        rs_new = 1.0  # Full retrieval strength after successful retrieval
        effective = rs_new * (0.7 + 0.3 * ss_new)
        return (min(1.0, ss_new), rs_new, min(1.0, effective))

    if hours_since_access <= 0:
        effective = rs * (0.7 + 0.3 * ss)
        return (ss, rs, min(1.0, effective))

    # Step 2: RS decay — inversely proportional to SS (Bjork & Bjork 1992)
    # High SS → slower RS decay (well-stored memories are easier to re-access)
    imp = FADEM_IMPORTANCE.get(importance, 0.5)
    lambda_rs = RS_BASE_DECAY * math.exp(-FADEM_MU * imp) * max(0.1, ss) ** (-0.5)

    # Beta by consolidation status
    if consolidated:
        beta = FADEM_BETA["consolidated_semantic"] if memory_type == "semantic" else FADEM_BETA["consolidated_episodic"]
    else:
        beta = FADEM_BETA["unconsolidated"]

    # Power-law RS decay: RS(t) = RS_0 * (1 + α*t)^{-β}
    alpha = lambda_rs * RS_PL_SCALE
    decay_factor = (1.0 + alpha * hours_since_access) ** (-beta)

    # Emotional protection (same as FadeMem)
    if emotional_arousal > FADEM_AROUSAL_THRESHOLD:
        shield = (emotional_arousal - FADEM_AROUSAL_THRESHOLD) / (1.0 - FADEM_AROUSAL_THRESHOLD)
        shield *= FADEM_AROUSAL_SHIELD_MAX
        decay_factor = 1.0 - (1.0 - decay_factor) * (1.0 - shield)

    rs_new = max(FADEM_FLOOR, rs * decay_factor)

    # SS is unchanged (only grows on retrieval)
    # Effective strength: RS determines accessibility, SS adds resilience
    effective = rs_new * (0.7 + 0.3 * ss)

    return (ss, rs_new, min(1.0, effective))


def _get_hours_since_access(payload: dict) -> Optional[float]:
    """Extract hours since last access from memory payload.

    Checks access_timestamps (list), then created_at as fallback.
    Returns None if no timing data available.
    """
    now = datetime.now(timezone.utc)

    # Try access_timestamps first (most recent access)
    timestamps = payload.get("access_timestamps")
    if timestamps and isinstance(timestamps, list) and len(timestamps) > 0:
        try:
            last_ts = timestamps[-1]
            if isinstance(last_ts, str):
                last_dt = datetime.fromisoformat(last_ts.replace("Z", "+00:00"))
                if last_dt.tzinfo is None:
                    last_dt = last_dt.replace(tzinfo=timezone.utc)
                return max(0.0, (now - last_dt).total_seconds() / 3600)
        except (ValueError, TypeError):
            pass

    # Fallback to created_at
    created = payload.get("created_at")
    if created and isinstance(created, str):
        try:
            created_dt = datetime.fromisoformat(created.replace("Z", "+00:00"))
            if created_dt.tzinfo is None:
                created_dt = created_dt.replace(tzinfo=timezone.utc)
            return max(0.0, (now - created_dt).total_seconds() / 3600)
        except (ValueError, TypeError):
            pass

    return None


def _get_emotional_arousal(payload: dict) -> float:
    """Extract emotional arousal from pad_at_encoding."""
    pad = payload.get("pad_at_encoding")
    if isinstance(pad, dict):
        try:
            return abs(float(pad.get("A", 0.0)))
        except (ValueError, TypeError):
            pass
    return 0.0


def _is_consolidated(payload: dict) -> bool:
    """Check if memory has been consolidated."""
    return bool(payload.get("consolidated", False))


def _get_memory_type(payload: dict) -> str:
    """Determine memory type from payload."""
    collection = payload.get("collection", "")
    if "semantic" in str(collection).lower():
        return "semantic"
    if payload.get("memory_type") == "semantic":
        return "semantic"
    return "episodic"


def should_enter_vault(payload: dict) -> bool:
    """Return True when a weak, old episodic memory should move to the vault."""
    if not isinstance(payload, dict):
        return False

    if payload.get("is_dormant"):
        return False

    if payload.get("importance", payload.get("narrative_importance", "medium")) == "critical":
        return False

    if payload.get("is_semantic") or _get_memory_type(payload) == "semantic":
        return False

    if payload.get("causal_links") or payload.get("source_episode_ids") or payload.get("_chain_member"):
        return False

    hours_since_access = _get_hours_since_access(payload)
    if hours_since_access is None or hours_since_access < (VAULT_MIN_AGE_DAYS * 24):
        return False

    rs = float(payload.get("retrieval_strength", payload.get("attention_salience", 0.0)) or 0.0)
    return rs <= VAULT_RS_THRESHOLD


def compute_reactivation_boost(
    ss: float,
    hours_dormant: float,
    reactivation_count: int,
) -> dict:
    """Spacing-aware reactivation boost for a dormant memory."""
    safe_ss = max(0.0, min(1.0, float(ss or 0.0)))
    safe_hours = max(0.0, float(hours_dormant or 0.0))
    safe_reactivations = max(0, int(reactivation_count or 0))

    ss_gain = SS_LEARNING_RATE * math.log1p(safe_hours / 168.0) / (1.0 + 0.3 * safe_reactivations)
    new_ss = min(1.0, safe_ss + ((1.0 - safe_ss) * ss_gain))
    new_rs = VAULT_REACTIVATION_RS
    new_activation = min(1.0, new_rs * (0.7 + 0.3 * new_ss))

    return {
        "new_ss": round(new_ss, 4),
        "new_rs": round(new_rs, 4),
        "new_activation": round(new_activation, 4),
        "ss_gain": round(ss_gain, 4),
    }


# ============================================================
# RIF: RETRIEVAL-INDUCED FORGETTING
# Anderson, Bjork & Bjork 1994
# ============================================================

# Suppression base rate: 8-15% of competitor's salience (Anderson 1994)
RIF_BASE = 0.08

# Minimum results for RIF to trigger (no suppression for single-result queries)
RIF_MIN_RESULTS = 2

# Max competitors to suppress per retrieval event (cap computation)
RIF_MAX_COMPETITORS = 10

# Minimum similarity score for a memory to be considered a "competitor"
RIF_SIMILARITY_THRESHOLD = 0.4

# Semaphore to limit concurrent RIF threads
_RIF_SEMAPHORE = threading.BoundedSemaphore(2)


def apply_rif(retrieved_ids: list, query_embedding: list = None) -> dict:
    """Apply Retrieval-Induced Forgetting to competing memories.

    When memories are retrieved, similar memories that were NOT retrieved
    suffer inhibition (salience reduction). This is active suppression,
    not mere failure to strengthen.

    Anderson, Bjork & Bjork 1994: "Remembering Can Cause Forgetting"
    Storm & Levy 2012: RIF is item-specific inhibition

    Args:
        retrieved_ids: IDs of memories that were successfully retrieved
        query_embedding: Optional pre-computed query embedding for neighbor search.
                        If None, uses top retrieved memory's vector.

    Returns:
        dict with applied status and suppression counts
    """
    if len(retrieved_ids) < RIF_MIN_RESULTS:
        return {"applied": False, "reason": "too_few_results"}

    try:
        # Get search vector (from top result or provided embedding)
        search_vector = query_embedding
        if search_vector is None:
            top_id = retrieved_ids[0]
            points = pg.get_by_ids([top_id])
            if not points or not getattr(points[0], "vector", None):
                return {"applied": False, "reason": "no_vector"}
            search_vector = points[0].vector

        # Search for neighbors (potential competitors)
        neighbors = pg.query_vector(
            search_vector,
            limit=len(retrieved_ids) + RIF_MAX_COMPETITORS + 5,
            is_semantic=False,
        )

        # Identify competitors: similar but NOT retrieved, not critical, not chain members
        # Sprint 1 spec: causal chain members resist inhibition (preserve causal structure)
        retrieved_set = set(str(rid) for rid in retrieved_ids)
        try:
            from modules.spreading import get_chain_member_ids
            _chain_ids = get_chain_member_ids()
        except Exception:
            _chain_ids = set()
        competitors = [
            n for n in neighbors
            if str(n.id) not in retrieved_set
            and n.payload.get('importance', n.payload.get('narrative_importance', 'medium')) != 'critical'
            and str(n.id) not in _chain_ids
            and getattr(n, "score", 1.0) >= RIF_SIMILARITY_THRESHOLD
        ]

        if not competitors:
            return {"applied": False, "reason": "no_competitors", "neighbors_found": len(neighbors)}

        # Apply suppression (Anderson 2003: inhibitory deficit hypothesis)
        suppressed = 0
        for comp in competitors[:RIF_MAX_COMPETITORS]:
            current_salience = comp.payload.get('attention_salience', 0.5)
            if current_salience <= FADEM_FLOOR:
                continue

            # Suppression proportional to similarity AND competitor strength
            # Strong competitors suppressed more (inhibitory deficit)
            suppression = RIF_BASE * comp.score * (current_salience / 1.0)
            new_salience = max(FADEM_FLOOR, current_salience - suppression)

            if new_salience < current_salience:
                pg.update_payload(comp.id, {
                    'attention_salience': new_salience,
                })
                suppressed += 1

        return {
            "applied": True,
            "competitors_found": len(competitors),
            "suppressed": suppressed,
        }

    except Exception as e:
        _logger.warning("RIF error: %s", e)
        return {"applied": False, "reason": f"error: {e}"}


def _on_memory_retrieved(event_name: str, data: dict):
    """Event handler: apply RIF after memory retrieval.

    Runs in background thread to avoid blocking recall() hot path.
    Same pattern as spreading activation in wiring.py.
    """
    retrieved_ids = data.get("retrieved_ids", [])
    if len(retrieved_ids) < RIF_MIN_RESULTS:
        return

    def _bg_rif():
        if not _RIF_SEMAPHORE.acquire(blocking=False):
            _logger.debug("RIF skipped: semaphore full")
            return
        try:
            result = apply_rif(retrieved_ids)
            if result.get("applied"):
                _logger.debug(
                    "RIF: suppressed %d/%d competitors",
                    result["suppressed"], result["competitors_found"],
                )
        except Exception as e:
            _logger.warning("bg RIF error: %s", e)
        finally:
            _RIF_SEMAPHORE.release()

    t = threading.Thread(target=_bg_rif, daemon=True, name="rif-bg")
    t.start()


# ============================================================
# EVENT WIRING
# ============================================================

def register_forgetting_handlers():
    """Wire forgetting handlers to the event bus."""
    from modules.events import event_bus, Events
    event_bus.on(Events.MEMORY_RETRIEVED, _on_memory_retrieved)
