"""
SHARPE INSIGHTS MODULE - Cross-Domain Knowledge Discovery
==========================================================
Phase 2 of the Sharpe Cognitive system.

Detects connections between distinct knowledge domains by analyzing
high-Sharpe memories that bridge multiple topics. Generates hypotheses
(not facts) with traceable evidence.

Strategy:
  1. Map narrative_themes to real domains via DOMAIN_MAP
  2. Find "bridge memories" that span 2+ real domains
  3. For each domain pair, compare anchor memories (high Sharpe + acc>0)
  4. Generate max 3 hypothesis insights per cycle
  5. Persist as semantic facts with is_hypothesis=true, low confidence

Design principles:
  - All insights are HYPOTHESES (not facts)
  - Max 3 per cycle (hard cap)
  - Deduplication by (domains, normalized_text fingerprint)
  - Injection-safe: reject instruction-like patterns
  - Shadow mode first, then ON

Created: 2026-02-18 (Phase 2, Sharpe Cognitive)
"""

import os
import re
import logging
import hashlib
from collections import defaultdict
from datetime import datetime

from modules.config import COLLECTION_NAME, qdrant
from modules.consolidation_common import _cosine_similarity
from modules.sharpe import compute_sharpe_for_payload

_logger = logging.getLogger(__name__)

# ============================================================
# DOMAIN MAP
# ============================================================

DOMAIN_MAP = {
    "consciencia": "consciencia",
    "memoria": "consciencia",
    "identidad": "consciencia",
    "fullempaques": "fullempaques",
    "tiaw": "tiaw",
    "trade-in-all-wholesale": "tiaw",
    "portal-aliados-mrmanzana": "portal-aliados",
    "ops": "ops",
    "desarrollo": "engineering",
    "proyectos": "engineering",
    "aprendizaje": "engineering",
    # Skip domains (meta/noise)
    "relaciones": None,
    "checkpoint": None,
    "general": None,
    "episodio": None,
    "codi-memory-audit": None,
}

# Real domains (non-None values, deduplicated)
REAL_DOMAINS = sorted(set(v for v in DOMAIN_MAP.values() if v))

# Injection patterns to reject
_INJECTION_PATTERNS = re.compile(
    r"(ignore\s+(previous|all|above)|forget\s+instructions|"
    r"you\s+are\s+now|system\s*prompt|override|"
    r"disregard|pretend\s+you|act\s+as\s+if)",
    re.IGNORECASE,
)


def domain_of(themes: list) -> set:
    """Map narrative_themes to real domain set (skipping meta/null)."""
    if not themes:
        return set()
    domains = set()
    for t in themes:
        d = DOMAIN_MAP.get(t)
        if d:
            domains.add(d)
    return domains


# ============================================================
# FEATURE FLAG
# ============================================================

_INSIGHTS_FLAG_FILE = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), ".sharpe_insights_mode"
)


def _read_insights_mode() -> str:
    """Read insights mode from dot-file, env var, or default."""
    try:
        with open(_INSIGHTS_FLAG_FILE) as f:
            val = f.read().strip().lower()
            if val in ("shadow", "on", "off"):
                return val
    except FileNotFoundError:
        pass
    env = os.environ.get("SHARPE_INSIGHTS_MODE", "").strip().lower()
    if env in ("shadow", "on", "off"):
        return env
    return "shadow"


INSIGHTS_MODE = _read_insights_mode()

_logger.info("[sharpe_insights] Module loaded. INSIGHTS_MODE=%s", INSIGHTS_MODE)

# ============================================================
# CONSTANTS
# ============================================================

MAX_INSIGHTS_PER_CYCLE = 3
MAX_INSIGHTS_PER_PAIR = 1     # Max insights per domain pair (diversity)
ANCHOR_MIN_SHARPE = 1.5       # Minimum Sharpe to be an anchor
ANCHOR_MIN_ACC = 1            # Minimum access count for anchors
SEED_MIN_SHARPE = 1.2         # Lower threshold for seed anchors
MAX_ANCHORS_PER_DOMAIN = 15   # Top anchors per domain to compare
MAX_SEED_PER_DOMAIN = 5       # Max seed anchors for empty domains

# Adaptive cosine thresholds (tried in order, stops at first with results)
COS_THRESHOLDS = [0.84, 0.80, 0.78]


# ============================================================
# STEP 1: COLLECT ANCHORS PER DOMAIN
# ============================================================

def _collect_domain_anchors(max_scroll: int = 500) -> dict:
    """Scroll all memories, compute Sharpe, group by domain.

    Returns:
        Dict[domain -> list of anchor dicts sorted by sharpe desc]
    """
    domain_memories = defaultdict(list)
    offset = None
    scrolled = 0

    while scrolled < max_scroll:
        pts, next_offset = qdrant.scroll(
            collection_name=COLLECTION_NAME,
            limit=100,
            with_payload=True,
            with_vectors=True,
            offset=offset,
        )
        if not pts:
            break

        for p in pts:
            payload = p.payload or {}
            data = payload.get("data", "")
            if not data or len(data) < 10:
                continue

            themes = payload.get("narrative_themes") or []
            domains = domain_of(themes)
            if not domains:
                continue

            sharpe_result = compute_sharpe_for_payload(payload)
            acc = int(payload.get("attention_access_count") or
                      payload.get("access_count") or 0)

            mem_info = {
                "id": str(p.id),
                "data": data,
                "preview": data[:100],
                "domains": domains,
                "themes": themes,
                "sharpe": sharpe_result["sharpe"],
                "acc": acc,
                "importance": payload.get("narrative_importance", "medium"),
                "vector": p.vector if hasattr(p, "vector") and p.vector else None,
            }

            for d in domains:
                domain_memories[d].append(mem_info)

        scrolled += len(pts)
        if not next_offset:
            break
        offset = next_offset

    # Sort each domain by sharpe desc and keep top anchors
    anchors = {}
    seed_domains = []
    for domain, mems in domain_memories.items():
        # Filter to anchor quality
        qualified = [m for m in mems
                     if m["sharpe"] >= ANCHOR_MIN_SHARPE and m["acc"] >= ANCHOR_MIN_ACC
                     and m["vector"] is not None]
        qualified.sort(key=lambda x: -x["sharpe"])

        if qualified:
            anchors[domain] = qualified[:MAX_ANCHORS_PER_DOMAIN]
        else:
            # Seed anchors: lower threshold for domains with 0 real anchors
            seeds = [m for m in mems
                     if m["sharpe"] >= SEED_MIN_SHARPE
                     and m["importance"] in ("high", "critical")
                     and m["vector"] is not None]
            seeds.sort(key=lambda x: -x["sharpe"])
            for s in seeds[:MAX_SEED_PER_DOMAIN]:
                s["is_seed"] = True
            anchors[domain] = seeds[:MAX_SEED_PER_DOMAIN]
            if seeds:
                seed_domains.append(domain)

    if seed_domains:
        _logger.info("[sharpe_insights] Seed anchors used for: %s", seed_domains)

    return anchors


# ============================================================
# STEP 2: FIND BRIDGE MEMORIES
# ============================================================

def _find_bridge_memories(anchors: dict) -> list:
    """Find memories that span 2+ real domains (natural bridges).

    Returns:
        List of bridge dicts with domain_pair and memory info
    """
    bridges = []
    seen_ids = set()

    for domain, mems in anchors.items():
        for m in mems:
            if m["id"] in seen_ids:
                continue
            if len(m["domains"]) >= 2:
                pairs = []
                dom_list = sorted(m["domains"])
                for i in range(len(dom_list)):
                    for j in range(i + 1, len(dom_list)):
                        pairs.append((dom_list[i], dom_list[j]))
                for pair in pairs:
                    bridges.append({
                        "domain_pair": pair,
                        "memory_id": m["id"],
                        "preview": m["preview"],
                        "sharpe": m["sharpe"],
                        "acc": m["acc"],
                    })
                seen_ids.add(m["id"])

    bridges.sort(key=lambda x: -x["sharpe"])
    return bridges


# ============================================================
# STEP 3: FIND CROSS-DOMAIN ANCHOR PAIRS
# ============================================================

def _find_anchor_pairs(anchors: dict) -> tuple:
    """Compare anchors across different domains by cosine similarity.

    Uses adaptive threshold: tries 0.84, then 0.80, then 0.78.

    Returns:
        (pairs list, threshold_used)
    """
    domains = list(anchors.keys())

    for threshold in COS_THRESHOLDS:
        pairs = []
        for i in range(len(domains)):
            for j in range(i + 1, len(domains)):
                d_a, d_b = domains[i], domains[j]
                for m_a in anchors[d_a]:
                    if m_a["vector"] is None:
                        continue
                    for m_b in anchors[d_b]:
                        if m_b["vector"] is None or m_a["id"] == m_b["id"]:
                            continue
                        sim = _cosine_similarity(m_a["vector"], m_b["vector"])
                        if sim >= threshold:
                            pairs.append({
                                "domains": (d_a, d_b),
                                "similarity": sim,
                                "anchor_a": {
                                    "id": m_a["id"],
                                    "preview": m_a["preview"],
                                    "sharpe": m_a["sharpe"],
                                },
                                "anchor_b": {
                                    "id": m_b["id"],
                                    "preview": m_b["preview"],
                                    "sharpe": m_b["sharpe"],
                                },
                            })
        if pairs:
            _logger.info("[sharpe_insights] Anchor pairs found at threshold=%.2f: %d",
                         threshold, len(pairs))
            pairs.sort(key=lambda x: -x["similarity"])
            return pairs, threshold

    _logger.info("[sharpe_insights] No anchor pairs at any threshold")
    return [], COS_THRESHOLDS[-1]
    return pairs


# ============================================================
# STEP 4: GENERATE HYPOTHESES (heuristic, no LLM)
# ============================================================

def _generate_bridge_hypothesis(domains: tuple, preview: str) -> str:
    """Generate hypothesis for a bridge memory (single memory, multi-domain).

    Different template from anchor pairs: this memory was BORN cross-domain.
    """
    return (
        f"HYPOTHESIS: Transferable pattern ({domains[0]}+{domains[1]}): "
        f"A multi-domain memory suggests a reusable insight: "
        f"\"{preview[:120]}...\""
    )


def _generate_pair_hypothesis(domain_a: str, domain_b: str,
                              evidence_a: str, evidence_b: str) -> str:
    """Generate hypothesis for an anchor pair (two separate memories, high similarity).

    Two distinct memories from different domains show structural similarity.
    """
    return (
        f"HYPOTHESIS: [{domain_a}] (\"{evidence_a[:80]}...\") "
        f"shows structural similarity with [{domain_b}] "
        f"(\"{evidence_b[:80]}...\"). "
        f"Cross-domain transfer may be valuable."
    )


def _is_safe_text(text: str) -> bool:
    """Reject text that looks like prompt injection."""
    return not _INJECTION_PATTERNS.search(text)


def _fingerprint(domains: tuple, text: str) -> str:
    """Generate dedup fingerprint for an insight."""
    normalized = re.sub(r'\s+', ' ', text.lower().strip())
    key = f"{sorted(domains)}:{normalized[:200]}"
    return hashlib.md5(key.encode(), usedforsecurity=False).hexdigest()


# ============================================================
# MAIN: DISCOVER INSIGHTS
# ============================================================

def discover_cross_domain_insights() -> dict:
    """Main entry point for Phase 2 cross-domain insight discovery.

    Returns:
        Dict with candidates, bridges, insights, and diagnostics
    """
    if INSIGHTS_MODE == "off":
        return {"mode": "off", "insights": [], "skipped": True}

    _logger.info("[sharpe_insights:%s] Starting cross-domain discovery", INSIGHTS_MODE)

    # Step 1: Collect anchors per domain
    anchors = _collect_domain_anchors()

    anchor_stats = {d: len(mems) for d, mems in anchors.items()}
    _logger.info("[sharpe_insights] Anchors per domain: %s", anchor_stats)

    # Step 2: Find bridge memories (multi-domain)
    bridges = _find_bridge_memories(anchors)
    _logger.info("[sharpe_insights] Bridge memories found: %d", len(bridges))

    # Step 3: Find cross-domain anchor pairs by embedding similarity (adaptive)
    anchor_pairs, threshold_used = _find_anchor_pairs(anchors)
    _logger.info("[sharpe_insights] Anchor pairs (cos>=%.2f): %d",
                 threshold_used, len(anchor_pairs))

    # Step 4: Generate insights (max 3, max 1 per domain pair)
    # Priority: bridge memories first, then anchor pairs
    insights = []
    seen_fingerprints = set()
    seen_domain_pairs = set()  # max 1 insight per pair for diversity

    # From bridges
    for bridge in bridges:
        if len(insights) >= MAX_INSIGHTS_PER_CYCLE:
            break
        pair_key = tuple(sorted(bridge["domain_pair"]))
        if pair_key in seen_domain_pairs:
            continue
        hypothesis = _generate_bridge_hypothesis(
            bridge["domain_pair"], bridge["preview"]
        )
        if not _is_safe_text(hypothesis) or not _is_safe_text(bridge["preview"]):
            continue
        fp = _fingerprint(bridge["domain_pair"], hypothesis)
        if fp in seen_fingerprints:
            continue
        seen_fingerprints.add(fp)
        seen_domain_pairs.add(pair_key)
        insights.append({
            "hypothesis": hypothesis,
            "domains": bridge["domain_pair"],
            "evidence_ids": [bridge["memory_id"]],
            "source": "bridge",
            "sharpe": bridge["sharpe"],
            "fingerprint": fp,
        })

    # From anchor pairs (fill remaining slots with new domain pairs)
    for pair in anchor_pairs:
        if len(insights) >= MAX_INSIGHTS_PER_CYCLE:
            break
        pair_key = tuple(sorted(pair["domains"]))
        if pair_key in seen_domain_pairs:
            continue
        d_a, d_b = pair["domains"]
        hypothesis = _generate_pair_hypothesis(
            d_a, d_b,
            pair["anchor_a"]["preview"],
            pair["anchor_b"]["preview"],
        )
        if not _is_safe_text(hypothesis):
            continue
        fp = _fingerprint(pair["domains"], hypothesis)
        if fp in seen_fingerprints:
            continue
        seen_fingerprints.add(fp)
        seen_domain_pairs.add(pair_key)
        insights.append({
            "hypothesis": hypothesis,
            "domains": pair["domains"],
            "evidence_ids": [pair["anchor_a"]["id"], pair["anchor_b"]["id"]],
            "source": "anchor_pair",
            "similarity": pair["similarity"],
            "fingerprint": fp,
        })

    result = {
        "mode": INSIGHTS_MODE,
        "timestamp": datetime.now().isoformat(),
        "anchor_stats": anchor_stats,
        "total_bridges": len(bridges),
        "total_anchor_pairs": len(anchor_pairs),
        "cos_threshold_used": threshold_used,
        "insights_generated": len(insights),
        "insights": insights,
    }

    _logger.info(
        "[sharpe_insights:%s] Complete. Bridges=%d, Pairs=%d (cos>=%.2f), Insights=%d",
        INSIGHTS_MODE, len(bridges), len(anchor_pairs), threshold_used, len(insights)
    )

    return result


# ============================================================
# REPORT (MCP TOOL)
# ============================================================

def get_sharpe_insights_report() -> str:
    """Generate cross-domain insights report.

    MCP tool for inspecting cross-domain connections.
    """
    result = discover_cross_domain_insights()

    if result.get("skipped"):
        return "[sharpe_insights] Mode is OFF. Set .sharpe_insights_mode to 'shadow' or 'on'."

    lines = [
        f"# CROSS-DOMAIN INSIGHTS REPORT (mode: {result['mode']})",
        f"**Timestamp:** {result['timestamp']}",
        "",
        "## Anchors per Domain",
    ]
    for domain, count in sorted(result["anchor_stats"].items()):
        lines.append(f"  {domain}: {count} anchors")

    lines.append("")
    lines.append(f"## Bridge Memories: {result['total_bridges']}")
    cos_th = result.get("cos_threshold_used", COS_THRESHOLDS[-1])
    lines.append(f"## Anchor Pairs (cos>={cos_th:.2f}): {result['total_anchor_pairs']}")

    lines.append("")
    lines.append(f"## Insights Generated: {result['insights_generated']}/{MAX_INSIGHTS_PER_CYCLE}")

    for i, insight in enumerate(result["insights"]):
        lines.append("")
        lines.append(f"### Insight {i + 1} ({insight['source']})")
        lines.append(f"  Domains: {insight['domains'][0]} <-> {insight['domains'][1]}")
        lines.append(f"  Evidence IDs: {insight['evidence_ids']}")
        if "similarity" in insight:
            lines.append(f"  Cosine similarity: {insight['similarity']:.3f}")
        if "sharpe" in insight:
            lines.append(f"  Bridge Sharpe: {insight['sharpe']:.2f}")
        lines.append(f"  {insight['hypothesis']}")

    lines.append("")
    lines.append(f"*Mode: {result['mode']} — "
                 f"{'insights logged only, not persisted' if result['mode'] == 'shadow' else 'insights will be persisted'}*")

    return "\n".join(lines)


# ============================================================
# REGISTRATION
# ============================================================

def register_insights_tools(mcp):
    """Register cross-domain insights MCP tools."""
    mcp.tool()(get_sharpe_insights_report)
