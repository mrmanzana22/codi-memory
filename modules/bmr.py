"""
Bayesian Model Reduction (BMR) for Memory Operations.
Sprint 9 — Canon v4 refs: PN-19, XC-24, M-INV-01, M-INV-08.

Replaces heuristic cosine thresholds with principled Bayesian scoring.
BMR asks: "Is the merged model better than keeping both separate?"

Formula: Delta_F = F(merged) - F(a) - F(b)
where F = Accuracy - Complexity (negative free energy)

For memory pairs:
  Accuracy ~ cosine_similarity (how well merged represents both)
  Complexity ~ KL(merged_metadata || prior_metadata)

If Delta_F > 0: merge is justified (simpler model explains same data)
If Delta_F < 0: keep separate (merger loses information)

References:
  - Friston et al. 2018, "Bayesian model reduction"
  - Friston & Penny 2011, "Post hoc Bayesian model selection"
  - LaBar & Cabeza 2006, "Cognitive neuroscience of emotional memory" (arousal modulation)
"""

import math
import logging

_logger = logging.getLogger(__name__)


def compute_log_bayes_factor(
    cosine_score: float,
    metadata_a: dict = None,
    metadata_b: dict = None,
    importance_a: str = "medium",
    importance_b: str = "medium",
    arousal: float = 0.0,
) -> float:
    """Compute log Bayes factor for merging two memories.

    Positive = merge is better. Negative = keep separate.

    The score combines:
    1. Accuracy term: how similar the memories are (from cosine)
    2. Complexity penalty: metadata divergence penalizes merge
    3. Emotional modulation: high arousal resists merging (LaBar & Cabeza 2006)

    Args:
        cosine_score: Cosine similarity between embeddings [0, 1]
        metadata_a: Metadata dict for memory A (topic, category, source, etc.)
        metadata_b: Metadata dict for memory B
        importance_a: Importance level of memory A
        importance_b: Importance level of memory B
        arousal: Current emotional arousal [-1, 1], abs value used

    Returns:
        Log Bayes factor. > 0 means merge, < 0 means keep separate.
    """
    metadata_a = metadata_a or {}
    metadata_b = metadata_b or {}

    # --- Accuracy term ---
    # Transform cosine to log-odds space for principled combination.
    # cosine=1.0 → high accuracy (merge captures both), cosine=0.5 → low accuracy.
    # Logit transform with slight clamp to avoid infinities.
    clamped = max(0.01, min(0.99, cosine_score))
    accuracy = math.log(clamped / (1.0 - clamped))  # logit

    # --- Complexity penalty ---
    # KL proxy: count metadata field mismatches.
    # Each mismatch = information the merged model can't capture.
    shared_keys = set(metadata_a.keys()) | set(metadata_b.keys())
    mismatches = 0
    for key in shared_keys:
        if key in ('created_at', 'updated_at', 'id', 'score', 'memory'):
            continue  # Skip non-semantic fields
        val_a = metadata_a.get(key)
        val_b = metadata_b.get(key)
        if val_a != val_b and val_a is not None and val_b is not None:
            mismatches += 1

    # Each mismatch penalizes by ~0.5 log-units
    complexity_penalty = mismatches * 0.5

    # --- Importance prior ---
    # Higher importance memories resist merging (preserve distinctness).
    importance_weight = {
        "critical": 2.0,
        "high": 1.0,
        "medium": 0.0,
        "low": -0.5,
    }
    importance_bias = max(
        importance_weight.get(importance_a, 0.0),
        importance_weight.get(importance_b, 0.0),
    )

    # --- Emotional modulation (LaBar & Cabeza 2006) ---
    # High arousal raises the bar for merging (emotional memories stay distinct).
    arousal_penalty = abs(arousal) * 1.5 if abs(arousal) > 0.3 else 0.0

    # --- Log Bayes factor ---
    log_bf = accuracy - complexity_penalty - importance_bias - arousal_penalty

    return log_bf


def should_merge(
    cosine_score: float,
    metadata_a: dict = None,
    metadata_b: dict = None,
    importance_a: str = "medium",
    importance_b: str = "medium",
    arousal: float = 0.0,
) -> tuple:
    """Convenience wrapper: should these two memories be merged?

    Returns:
        (should_merge: bool, log_bf: float, explanation: str)
    """
    log_bf = compute_log_bayes_factor(
        cosine_score=cosine_score,
        metadata_a=metadata_a,
        metadata_b=metadata_b,
        importance_a=importance_a,
        importance_b=importance_b,
        arousal=arousal,
    )

    if log_bf > 0:
        return (True, log_bf, f"BMR merge justified (log_bf={log_bf:.2f})")
    else:
        return (False, log_bf, f"BMR keep separate (log_bf={log_bf:.2f})")


def bmr_dedup_threshold(
    cosine_score: float,
    importance: str = "medium",
    arousal: float = 0.0,
) -> bool:
    """Drop-in replacement for cosine threshold dedup.

    Instead of: `if cosine > threshold: skip`
    Now:        `if bmr_dedup_threshold(cosine, importance, arousal): skip`

    Returns True if the new memory is a duplicate (should skip).
    """
    log_bf = compute_log_bayes_factor(
        cosine_score=cosine_score,
        importance_a=importance,
        importance_b=importance,
        arousal=arousal,
    )
    # Positive log_bf means merger justified → it's a duplicate
    return log_bf > 0


def bmr_should_consolidate(
    cosine_score: float,
    metadata_existing: dict = None,
    metadata_new: dict = None,
) -> bool:
    """For consolidation: should a new fact merge into an existing semantic memory?

    Replaces CONSOLIDATION_SEMANTIC_DEDUP_THRESHOLD (0.85) with BMR scoring.

    Returns True if merge is justified.
    """
    log_bf = compute_log_bayes_factor(
        cosine_score=cosine_score,
        metadata_a=metadata_existing,
        metadata_b=metadata_new,
        importance_a=metadata_existing.get("narrative_importance", "medium") if metadata_existing else "medium",
        importance_b="medium",
    )
    return log_bf > 0


def bmr_should_prune(
    cosine_score: float,
    episode_metadata: dict = None,
    semantic_metadata: dict = None,
) -> bool:
    """For pruning: is this episode's information preserved in the semantic layer?

    If BMR says merge is justified between episode and semantic fact,
    then the episode's information is captured and it can be pruned.

    Returns True if information is preserved (safe to mark consolidated).
    """
    log_bf = compute_log_bayes_factor(
        cosine_score=cosine_score,
        metadata_a=episode_metadata,
        metadata_b=semantic_metadata,
    )
    # For pruning, we want a higher bar (information must truly be captured)
    # Require log_bf > 1.0 (stronger evidence than simple merge)
    return log_bf > 1.0
