"""
Codi Memory - Thompson Sampling for Channel Selection (S1-04)
=============================================================
Replaces fixed channel weights with adaptive Bayesian bandits.

Each retrieval channel (vector, BM25, semantic) has a Beta(a,b) posterior
tracking its reward rate. On each query, we sample from each posterior
and allocate retrieval budget proportionally.

Thompson Sampling achieves 18x less regret than epsilon-greedy for
non-stationary multi-armed bandits (Russo et al. 2018).

Discount gamma=0.995 for non-stationarity (environment changes).

Neuroscience basis:
  - Daw et al. 2006: model-based vs model-free RL in striatum
  - Gershman & Daw 2017: Bayesian RL with approximate inference
"""

import logging
import math
import random
from typing import Dict, Tuple

_logger = logging.getLogger(__name__)

# ============================================================
# CONSTANTS
# ============================================================

# Default prior: Beta(2, 2) — slightly informed (slight belief in ~50% success)
DEFAULT_ALPHA = 2.0
DEFAULT_BETA = 2.0

# Discount factor for non-stationarity
DISCOUNT_GAMMA = 0.995

# Minimum allocation: every channel gets at least this fraction of budget
MIN_ALLOCATION = 0.15

# Channel names
CHANNELS = ("vector", "bm25", "semantic")


# ============================================================
# CORE API
# ============================================================

def get_channel_posteriors(fts_db_path: str = None,
                           topic: str = None) -> Dict[str, Tuple[float, float]]:
    """Get current Beta posteriors for each channel.

    Args:
        fts_db_path: Path to FTS database (unused, kept for compat)
        topic: If provided, compute per-topic posteriors (Canon v2, CC-8).
               Falls back to global if per-topic data is insufficient.

    Returns:
        {channel: (alpha, beta)} for each channel
    """
    from modules.reward_tracking import get_channel_stats

    # Try per-topic stats first, fall back to global
    stats = {}
    if topic:
        stats = get_channel_stats(lookback_hours=72, topic=topic)

    # Fall back to global if per-topic data is insufficient
    total_data = sum(s.get("total", 0) for s in stats.values())
    if total_data < 5:
        stats = get_channel_stats(lookback_hours=72)

    posteriors = {}
    for ch in CHANNELS:
        ch_stats = stats.get(ch, {})
        s = ch_stats.get("successes", 0)
        f = ch_stats.get("failures", 0)

        # Posterior = prior + data, with discount
        alpha = DEFAULT_ALPHA + s * (DISCOUNT_GAMMA ** max(0, s + f - 1))
        beta = DEFAULT_BETA + f * (DISCOUNT_GAMMA ** max(0, s + f - 1))

        posteriors[ch] = (alpha, beta)

    return posteriors


def sample_allocation(base_limit: int = 5,
                      topic: str = None) -> Dict[str, int]:
    """Thompson Sampling: sample from posteriors to allocate retrieval budget.

    Each channel gets budget proportional to its sampled reward probability.
    Minimum allocation ensures every channel gets at least MIN_ALLOCATION * base_limit.

    Args:
        base_limit: Total retrieval budget (e.g., 5 memories)
        topic: Query topic for per-topic posteriors (Canon v2, CC-8)

    Returns:
        {channel: allocated_limit} — how many items to fetch from each channel
    """
    posteriors = get_channel_posteriors(topic=topic)

    # Sample from each channel's Beta posterior
    samples = {}
    for ch, (a, b) in posteriors.items():
        try:
            samples[ch] = random.betavariate(max(0.01, a), max(0.01, b))
        except ValueError:
            samples[ch] = 0.5  # Fallback

    # Convert samples to allocation proportions
    total = sum(samples.values())
    if total <= 0:
        # Uniform allocation
        per_channel = max(1, base_limit // len(CHANNELS))
        return {ch: per_channel for ch in CHANNELS}

    allocations = {}
    min_items = max(1, int(base_limit * MIN_ALLOCATION))

    for ch in CHANNELS:
        proportion = samples[ch] / total
        raw = proportion * base_limit
        allocations[ch] = max(min_items, round(raw))

    return allocations


def get_channel_summary() -> str:
    """Human-readable summary of channel posteriors and expected rewards."""
    posteriors = get_channel_posteriors()

    lines = ["# Thompson Sampling - Channel Status\n"]
    for ch in CHANNELS:
        a, b = posteriors[ch]
        mean = a / (a + b)
        # 95% credible interval
        # Approximate: mean +/- 1.96 * sqrt(ab / ((a+b)^2 * (a+b+1)))
        var = (a * b) / ((a + b) ** 2 * (a + b + 1))
        ci = 1.96 * math.sqrt(var) if var > 0 else 0
        lines.append(
            f"- **{ch}**: Beta({a:.1f}, {b:.1f}) "
            f"E[p]={mean:.3f} CI=[{max(0,mean-ci):.3f}, {min(1,mean+ci):.3f}]"
        )

    # Sample allocation for a typical query
    alloc = sample_allocation(base_limit=5)
    lines.append(f"\nSample allocation (limit=5): {alloc}")

    return "\n".join(lines)
