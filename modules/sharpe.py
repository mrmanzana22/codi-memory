"""
SHARPE COGNITIVE MODULE - Memory Quality Scoring
=================================================
Inspired by the financial Sharpe Ratio: quality = utility / uncertainty.

Replaces "popularity-based" memory persistence (access frequency)
with "value-based" scoring. Memories that helped resolve tasks,
generated decisions, or connected across domains score higher.

Phase 0: Shadow mode (log-only, no behavioral changes).
Phase 1: Integrated into consolidation selection scoring.
Phase 2: Cross-domain insight detection.
Phase 3: Differential decay based on Sharpe.

Design principles:
  - All signals are OBSERVABLE (logs, counters, events) - not self-report
  - Only for episodic memories in MVP (semantic gets its own later)
  - Guardrails: insights are hypotheses, max 3 per cycle, rollback flag

Based on research session 2026-02-18:
  - Trading/cognition isomorphism: Sharpe Ratio applies to memory quality
  - Utility = "did this memory help?" / Uncertainty = "how reliable?"

Created: 2026-02-18 (Curiosity-driven exploration session)
"""

import logging
import math
from datetime import datetime, timedelta
from typing import Optional

_logger = logging.getLogger(__name__)

# ============================================================
# CONSTANTS
# ============================================================

SHARPE_EPSILON = 0.05           # Avoid division by zero
SHARPE_CLAMP_MAX = 10.0         # Max Sharpe score for display
SHARPE_CLAMP_MIN = 0.0          # Min Sharpe score

# Utility weights (sum to 1.0)
W_ACCESS_FREQUENCY = 0.25       # How often retrieved
W_IMPORTANCE_PROXY = 0.25       # Importance as proxy for "generated decision"
W_EVIDENCE_CONFIRMED = 0.25     # Confirmed by multiple episodes
W_CROSS_DOMAIN = 0.25           # Connected to other topics

# Uncertainty weights (sum to 1.0)
W_LOW_EVIDENCE = 0.30           # Single source = high uncertainty
W_CONTRADICTION = 0.30          # Has contradictions
W_OBSOLESCENCE = 0.25           # Old without access
W_SINGLE_SOURCE = 0.15          # Only one interaction

# Access frequency normalization (calibrated to ~2800 memories)
ACCESS_COUNT_P90 = 10           # 90th percentile access count (estimate)

# Obsolescence threshold
OBSOLESCENCE_DAYS = 30          # After 30 days without access, starts contributing


# ============================================================
# UTILITY: "How much did this memory help?"
# ============================================================

def compute_utility(
    access_count: int = 0,
    importance: str = "medium",
    evidence_count: int = 1,
    topic_connections: int = 0,
    last_accessed_str: str = "",
    created_at_str: str = "",
) -> float:
    """Compute utility score U(m) for a memory.

    Uses observable signals only (no self-report).

    Args:
        access_count: Times this memory was retrieved
        importance: narrative_importance string
        evidence_count: How many episodes support this (for semantic)
        topic_connections: Number of distinct topics this memory connects to
        last_accessed_str: ISO timestamp of last access
        created_at_str: ISO timestamp of creation

    Returns:
        Utility score in [0, 1]
    """
    # 1. Access frequency (normalized to 0-1 via tanh saturation)
    access_norm = math.tanh(access_count / max(1, ACCESS_COUNT_P90))

    # 2. Importance as decision proxy
    importance_map = {
        'critical': 1.0,
        'high': 0.75,
        'medium': 0.40,
        'low': 0.10,
    }
    importance_score = importance_map.get(importance, 0.40)

    # 3. Evidence confirmation (more evidence = more utility)
    evidence_score = math.tanh(evidence_count / 5.0)  # saturates around 5

    # 4. Cross-domain connections (exponentially valuable)
    cross_score = min(1.0, topic_connections / 3.0)  # 3+ topics = max

    # Weighted sum
    utility = (
        W_ACCESS_FREQUENCY * access_norm
        + W_IMPORTANCE_PROXY * importance_score
        + W_EVIDENCE_CONFIRMED * evidence_score
        + W_CROSS_DOMAIN * cross_score
    )

    return round(max(0.0, min(1.0, utility)), 4)


# ============================================================
# UNCERTAINTY: "How reliable is this memory?"
# ============================================================

def compute_uncertainty(
    evidence_count: int = 1,
    contradiction_count: int = 0,
    access_count: int = 0,
    last_accessed_str: str = "",
    created_at_str: str = "",
    is_labile: bool = False,
) -> float:
    """Compute uncertainty score sigma(m) for a memory.

    Higher = less reliable.

    Args:
        evidence_count: How many episodes support this
        contradiction_count: Times this memory was contradicted
        access_count: Times retrieved (0 access on old = maybe obsolete)
        last_accessed_str: ISO timestamp of last access
        created_at_str: ISO timestamp of creation
        is_labile: Whether memory is in reconsolidation window

    Returns:
        Uncertainty score in [0, 1]
    """
    # 1. Low evidence (inverse of evidence count)
    low_evidence = 1.0 / (1.0 + evidence_count)

    # 2. Contradiction signal
    contradiction_signal = min(1.0, contradiction_count / 3.0)  # 3+ = max uncertainty

    # 3. Obsolescence (old + no access = might be stale)
    obsolescence = 0.0
    now = datetime.now()

    last_acc = None
    if last_accessed_str:
        try:
            last_acc = datetime.fromisoformat(
                str(last_accessed_str).replace('Z', '+00:00')
            )
            if last_acc.tzinfo:
                last_acc = last_acc.replace(tzinfo=None)
        except (ValueError, TypeError) as exc:
            _logger.warning("compute_uncertainty: bad last_accessed_str=%r: %s", last_accessed_str, exc)

    created = None
    if created_at_str:
        try:
            created = datetime.fromisoformat(
                str(created_at_str).replace('Z', '+00:00')
            )
            if created.tzinfo:
                created = created.replace(tzinfo=None)
        except (ValueError, TypeError) as exc:
            _logger.warning("compute_uncertainty: bad created_at_str=%r: %s", created_at_str, exc)

    ref_time = last_acc or created or (now - timedelta(days=7))
    days_since = (now - ref_time).total_seconds() / 86400
    if days_since > OBSOLESCENCE_DAYS and access_count < 2:
        obsolescence = min(1.0, (days_since - OBSOLESCENCE_DAYS) / 60.0)

    # 4. Single source (no confirmation from other episodes)
    single_source = 1.0 if evidence_count <= 1 and access_count <= 1 else 0.0

    # Labile memories get uncertainty boost
    labile_boost = 0.3 if is_labile else 0.0

    # Weighted sum
    uncertainty = (
        W_LOW_EVIDENCE * low_evidence
        + W_CONTRADICTION * contradiction_signal
        + W_OBSOLESCENCE * obsolescence
        + W_SINGLE_SOURCE * single_source
        + labile_boost
    )

    return round(max(0.0, min(1.0, uncertainty)), 4)


# ============================================================
# SHARPE COGNITIVE RATIO
# ============================================================

def compute_sharpe(
    utility: float = 0.0,
    uncertainty: float = 0.0,
) -> float:
    """Compute Sharpe Cognitive Ratio for a memory.

    Sharpe(m) = (U(m) + epsilon) / (sigma(m) + epsilon)

    Returns:
        Sharpe score clamped to [0, SHARPE_CLAMP_MAX]
    """
    raw = (utility + SHARPE_EPSILON) / (uncertainty + SHARPE_EPSILON)
    return round(max(SHARPE_CLAMP_MIN, min(SHARPE_CLAMP_MAX, raw)), 4)


def compute_sharpe_for_payload(payload: dict) -> dict:
    """Convenience: compute all Sharpe components from a Qdrant payload.

    Args:
        payload: Memory point payload dict

    Returns:
        Dict with utility, uncertainty, sharpe, and component details
    """
    # Extract observable signals from payload (with legacy fallback)
    acc_raw = payload.get("attention_access_count")
    if acc_raw is None:
        acc_raw = payload.get("access_count", 0)
    access_count = int(acc_raw or 0)
    importance = payload.get("narrative_importance") or "medium"
    evidence_count = int(payload.get("evidence_count") or 1)
    contradiction_count = int(payload.get("contradiction_count") or 0)
    last_accessed = (
        payload.get("last_accessed_at")
        or payload.get("attention_last_accessed")
        or payload.get("last_accessed")
        or ""
    )
    created_at = payload.get("created_at") or ""
    is_labile = bool(payload.get("is_labile") or False)

    # Count topic connections (proxy for cross-domain)
    themes = payload.get("narrative_themes") or []
    topic_connections = len(set(themes)) if themes else 0

    u = compute_utility(
        access_count=access_count,
        importance=importance,
        evidence_count=evidence_count,
        topic_connections=topic_connections,
        last_accessed_str=last_accessed,
        created_at_str=created_at,
    )

    sigma = compute_uncertainty(
        evidence_count=evidence_count,
        contradiction_count=contradiction_count,
        access_count=access_count,
        last_accessed_str=last_accessed,
        created_at_str=created_at,
        is_labile=is_labile,
    )

    sharpe = compute_sharpe(utility=u, uncertainty=sigma)

    return {
        "utility": u,
        "uncertainty": sigma,
        "sharpe": sharpe,
        "components": {
            "access_count": access_count,
            "importance": importance,
            "evidence_count": evidence_count,
            "topic_connections": topic_connections,
            "contradiction_count": contradiction_count,
            "is_labile": is_labile,
        },
    }


# ============================================================
# SHARPE REPORT (MCP TOOL)
# ============================================================

def get_sharpe_report(topic: str = "", top_n: int = 20) -> str:
    """Generate a Sharpe Cognitive report across episodic memories.

    Shadow-mode tool: computes Sharpe for memories without changing behavior.

    Args:
        topic: Optional topic filter
        top_n: Number of top/bottom memories to show (default 20)

    Returns:
        Formatted report string
    """
    from modules.pg_store import pg

    scroll_filters = {"narrative_themes": topic} if topic else {}

    all_scores = []
    offset = None

    # Sanity counters for field tracking
    sanity_has_attention = 0   # has attention_access_count
    sanity_has_legacy = 0      # has access_count (legacy fallback)
    sanity_has_none = 0        # no access field at all

    scrolled = 0
    while True:
        pts, next_offset = pg.scroll(
            filters=scroll_filters,
            limit=100,
            is_semantic=False,
            offset=offset,
        )
        if not pts:
            break

        for p in pts:
            payload = p.payload or {}
            data = payload.get("data", "")
            if not data or len(data) < 10:
                continue

            # Track which field is present
            if payload.get("attention_access_count") is not None:
                sanity_has_attention += 1
            elif payload.get("access_count") is not None:
                sanity_has_legacy += 1
            else:
                sanity_has_none += 1

            result = compute_sharpe_for_payload(payload)
            all_scores.append({
                "id": str(p.id),
                "sharpe": result["sharpe"],
                "utility": result["utility"],
                "uncertainty": result["uncertainty"],
                "importance": result["components"]["importance"],
                "access_count": result["components"]["access_count"],
                "preview": data[:80],
            })

        scrolled += len(pts)
        if not next_offset:
            break
        offset = next_offset

    if not all_scores:
        return "[sharpe] No memories found to score."

    # Sort by Sharpe
    all_scores.sort(key=lambda x: -x["sharpe"])

    # Statistics
    sharpes = [s["sharpe"] for s in all_scores]
    avg_sharpe = sum(sharpes) / len(sharpes)
    median_idx = len(sharpes) // 2
    median_sharpe = sharpes[median_idx]
    max_sharpe = sharpes[0]
    min_sharpe = sharpes[-1]

    # Distribution buckets
    buckets = {"0-1": 0, "1-2": 0, "2-3": 0, "3-5": 0, "5+": 0}
    for s in sharpes:
        if s < 1:
            buckets["0-1"] += 1
        elif s < 2:
            buckets["1-2"] += 1
        elif s < 3:
            buckets["2-3"] += 1
        elif s < 5:
            buckets["3-5"] += 1
        else:
            buckets["5+"] += 1

    # Build report
    lines = [
        f"# SHARPE COGNITIVE REPORT {'(topic: ' + topic + ')' if topic else '(all)'}",
        f"**Memories scored:** {len(all_scores)}",
        f"**Avg Sharpe:** {avg_sharpe:.2f} | **Median:** {median_sharpe:.2f}",
        f"**Range:** {min_sharpe:.2f} - {max_sharpe:.2f}",
        "",
        "## Distribution",
    ]
    for bucket, count in buckets.items():
        pct = count / len(all_scores) * 100
        bar = "#" * int(pct / 2)
        lines.append(f"  {bucket}: {count} ({pct:.0f}%) {bar}")

    lines.append("")
    lines.append(f"## Top {min(top_n, len(all_scores))} (highest Sharpe)")
    for item in all_scores[:top_n]:
        lines.append(
            f"  [{item['sharpe']:.2f}] U={item['utility']:.2f} "
            f"sigma={item['uncertainty']:.2f} "
            f"acc={item['access_count']} imp={item['importance']} "
            f"| {item['preview']}..."
        )

    lines.append("")
    lines.append(f"## Bottom {min(top_n, len(all_scores))} (lowest Sharpe)")
    for item in all_scores[-top_n:]:
        lines.append(
            f"  [{item['sharpe']:.2f}] U={item['utility']:.2f} "
            f"sigma={item['uncertainty']:.2f} "
            f"acc={item['access_count']} imp={item['importance']} "
            f"| {item['preview']}..."
        )

    # Sanity check: field tracking health
    total_sanity = sanity_has_attention + sanity_has_legacy + sanity_has_none
    lines.append("")
    lines.append("## Signal Health")
    if total_sanity > 0:
        pct_att = sanity_has_attention / total_sanity * 100
        pct_leg = sanity_has_legacy / total_sanity * 100
        pct_none = sanity_has_none / total_sanity * 100
        lines.append(f"  attention_access_count: {sanity_has_attention} ({pct_att:.0f}%)")
        lines.append(f"  access_count (legacy): {sanity_has_legacy} ({pct_leg:.0f}%)")
        lines.append(f"  no field (default 0): {sanity_has_none} ({pct_none:.0f}%)")
        with_data = [s for s in all_scores if s["access_count"] > 0]
        lines.append(f"  memories with acc>0: {len(with_data)} ({len(with_data)/len(all_scores)*100:.0f}%)")
    else:
        lines.append("  No memories processed")

    lines.append("")
    lines.append(f"*Shadow mode ({SHARPE_MODE}): no behavioral changes applied*")

    return "\n".join(lines)


# ============================================================
# CONSOLIDATION INTEGRATION (Phase 0: log-only)
# ============================================================

# Feature flag (dot-file > env var > default)
import os

_SHARPE_FLAG_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".sharpe_mode")

def _read_sharpe_mode() -> str:
    """Read SHARPE_MODE from dot-file, then env var, then default."""
    # 1. Dot-file (most reliable in MCP)
    try:
        with open(_SHARPE_FLAG_FILE) as f:
            val = f.read().strip().lower()
            if val in ("shadow", "on", "off"):
                return val
    except FileNotFoundError:
        pass
    # 2. Env var
    env = os.environ.get("SHARPE_MODE", "").strip().lower()
    if env in ("shadow", "on", "off"):
        return env
    # 3. Default
    return "shadow"

SHARPE_MODE = _read_sharpe_mode()

# Blend weights for "on" mode
SHARPE_BLEND_OLD = 0.60      # weight for old score (importance + recency)
SHARPE_BLEND_NEW = 0.40      # weight for sharpe_norm

_logger.info("[sharpe] Module loaded. SHARPE_MODE=%s, blend=%.0f/%.0f",
             SHARPE_MODE, SHARPE_BLEND_OLD * 100, SHARPE_BLEND_NEW * 100)


def sharpe_score_candidates(candidates: list) -> list:
    """Score consolidation candidates with Sharpe.

    Called from consolidation._phase_selection after normal scoring.
    In shadow mode, only logs; does not modify candidate scores.
    In 'on' mode, blends Sharpe into the score with p95 clamp.

    Args:
        candidates: List of candidate dicts from Phase 1

    Returns:
        Same candidates, with sharpe_* fields added
    """
    if SHARPE_MODE == "off":
        return candidates

    scored = 0
    for c in candidates:
        payload = c.get("payload", {})
        result = compute_sharpe_for_payload(payload)

        c["sharpe_utility"] = result["utility"]
        c["sharpe_uncertainty"] = result["uncertainty"]
        c["sharpe_score"] = result["sharpe"]
        scored += 1

    if SHARPE_MODE == "on" and scored > 0:
        # Compute p95 clamp from actual candidates to prevent outlier domination
        sharpe_values = sorted(c.get("sharpe_score", 0) for c in candidates if "sharpe_score" in c)
        p95_idx = int(len(sharpe_values) * 0.95)
        p95_val = sharpe_values[min(p95_idx, len(sharpe_values) - 1)] if sharpe_values else 5.0
        clamp_max = max(p95_val, 1.0)  # at least 1.0 to avoid degenerate clamp

        for c in candidates:
            if "sharpe_score" not in c:
                continue
            clamped = min(c["sharpe_score"], clamp_max)
            sharpe_norm = clamped / clamp_max  # normalize to [0, 1]
            old_score = c.get("score", 0.5)
            c["score_old"] = old_score  # preserve for debugging
            c["score"] = old_score * SHARPE_BLEND_OLD + sharpe_norm * SHARPE_BLEND_NEW

        _logger.info(
            "[sharpe:on] Applied blend (%.0f/%.0f) to %d candidates. p95_clamp=%.2f",
            SHARPE_BLEND_OLD * 100, SHARPE_BLEND_NEW * 100, scored, clamp_max
        )

    if scored > 0:
        sharpes = [c.get("sharpe_score", 0) for c in candidates if "sharpe_score" in c]
        if sharpes:
            avg = sum(sharpes) / len(sharpes)
            _logger.info(
                "[sharpe:%s] Scored %d candidates. Avg=%.2f, Min=%.2f, Max=%.2f",
                SHARPE_MODE, scored, avg, min(sharpes), max(sharpes)
            )

    return candidates


# ============================================================
# REGISTRATION
# ============================================================

def register_sharpe_tools(mcp):
    """Register Sharpe MCP tools."""
    mcp.tool()(get_sharpe_report)
