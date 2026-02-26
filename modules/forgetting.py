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
from datetime import datetime
from typing import Optional

_logger = logging.getLogger(__name__)

# ============================================================
# FADEM CONSTANTS
# ============================================================

# Base decay rate (per hour^beta). Calibrated so a medium-importance
# unconsolidated memory loses ~50% salience in 24h with no access.
FADEM_LAMBDA_BASE = 0.008

# Importance sensitivity: higher = importance protects more
FADEM_MU = 0.5

# Minimum salience floor (memories never fully disappear)
FADEM_FLOOR = 0.1

# Shape parameter (beta) by consolidation/type
# Lower beta = slower decay (sub-linear time progression)
FADEM_BETA = {
    "consolidated_semantic": 0.5,    # Bahrick 1984 permastore
    "consolidated_episodic": 0.8,    # Consolidated episodic
    "unconsolidated": 1.2,           # Fast decay (Tononi & Cirelli 2003)
    "default": 1.0,                  # Unknown status
}

# Importance mapping for decay modulation
FADEM_IMPORTANCE = {
    "critical": 1.0,
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

    # Step 3: Core FadeMem formula
    decay_factor = math.exp(-lambda_i * (hours_since_access ** beta))

    # Step 4: Emotional protection (high arousal memories decay slower)
    if emotional_arousal > FADEM_AROUSAL_THRESHOLD:
        # Linear shield scaling from threshold to 1.0
        shield = (emotional_arousal - FADEM_AROUSAL_THRESHOLD) / (1.0 - FADEM_AROUSAL_THRESHOLD)
        shield *= FADEM_AROUSAL_SHIELD_MAX  # 0 to 0.4
        # Reduce the decay amount by shield fraction
        decay_factor = 1.0 - (1.0 - decay_factor) * (1.0 - shield)

    new_salience = current_salience * decay_factor
    return max(FADEM_FLOOR, new_salience)


def _get_hours_since_access(payload: dict) -> Optional[float]:
    """Extract hours since last access from memory payload.

    Checks access_timestamps (list), then created_at as fallback.
    Returns None if no timing data available.
    """
    now = datetime.now()

    # Try access_timestamps first (most recent access)
    timestamps = payload.get("access_timestamps")
    if timestamps and isinstance(timestamps, list) and len(timestamps) > 0:
        try:
            last_ts = timestamps[-1]
            if isinstance(last_ts, str):
                last_dt = datetime.fromisoformat(last_ts.replace("Z", "+00:00").replace("+00:00", ""))
                if last_dt.tzinfo:
                    last_dt = last_dt.replace(tzinfo=None)
                return max(0.0, (now - last_dt).total_seconds() / 3600)
        except (ValueError, TypeError):
            pass

    # Fallback to created_at
    created = payload.get("created_at")
    if created and isinstance(created, str):
        try:
            created_dt = datetime.fromisoformat(created.replace("Z", "+00:00").replace("+00:00", ""))
            if created_dt.tzinfo:
                created_dt = created_dt.replace(tzinfo=None)
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
        from modules.config import qdrant, COLLECTION_NAME
        from modules.access_tracking import record_access

        # Get search vector (from top result or provided embedding)
        search_vector = query_embedding
        if search_vector is None:
            top_id = retrieved_ids[0]
            points = qdrant.retrieve(
                collection_name=COLLECTION_NAME,
                ids=[top_id],
                with_vectors=True,
            )
            if not points or not points[0].vector:
                return {"applied": False, "reason": "no_vector"}
            search_vector = points[0].vector

        # Search for neighbors (potential competitors)
        neighbors = qdrant.search(
            collection_name=COLLECTION_NAME,
            query_vector=search_vector,
            limit=len(retrieved_ids) + RIF_MAX_COMPETITORS + 5,
            score_threshold=RIF_SIMILARITY_THRESHOLD,
            with_payload=True,
        )

        # Identify competitors: similar but NOT retrieved, not critical
        retrieved_set = set(str(rid) for rid in retrieved_ids)
        competitors = [
            n for n in neighbors
            if str(n.id) not in retrieved_set
            and n.payload.get('narrative_importance') != 'critical'
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
                record_access(COLLECTION_NAME, comp.id, {
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
