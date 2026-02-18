"""
SEMANTIC STORE MODULE

Operations on the codi_semantic Qdrant collection:
- Vector search for semantic facts
- Browsing/filtering facts by topic
- Consolidation statistics
- Counting unconsolidated episodic memories

Split from consolidation.py (Phase 1, Sub-phase 1.1)
"""

import logging
from datetime import datetime, timedelta

from qdrant_client.models import Filter, FieldCondition, MatchValue

from modules.config import (
    SEMANTIC_COLLECTION,
    COLLECTION_NAME,
    qdrant,
)
from modules.consolidation_common import _embed_text, _consolidation_conn
from modules.secret_redact import redact_secrets

_logger = logging.getLogger(__name__)


# ============================================================
# SEARCH
# ============================================================

def search_semantic(query: str, limit: int = 5) -> list:
    """Search the semantic store (codi_semantic) via vector similarity.

    Returns:
        List of semantic facts with scores
    """
    try:
        info = qdrant.get_collection(SEMANTIC_COLLECTION)
        if info.points_count == 0:
            return []

        query_vector = _embed_text(query)
        results = qdrant.query_points(
            collection_name=SEMANTIC_COLLECTION,
            query=query_vector,
            limit=limit,
            with_payload=True,
        )
        facts = []
        for hit in results.points:
            payload = hit.payload or {}
            facts.append({
                "id": str(hit.id),
                "fact": payload.get("fact_text", payload.get("data", "")),
                "topic": payload.get("topic", ""),
                "confidence": payload.get("confidence", 0),
                "evidence_count": payload.get("evidence_count", 0),
                "score": hit.score,
            })
        return facts
    except Exception as e:
        _logger.error("Semantic search error: %s", redact_secrets(str(e)))
        return []


# ============================================================
# GET FACTS (MCP TOOL)
# ============================================================

def get_semantic_facts(topic: str = "", limit: int = 10) -> str:
    """Get all semantic facts, optionally filtered by topic.

    MCP tool to inspect consolidated knowledge.

    Args:
        topic: Optional topic to filter by (e.g. 'trading', 'fullempaques')
        limit: Max facts to return (default 10)
    """
    try:
        info = qdrant.get_collection(SEMANTIC_COLLECTION)
        count = info.points_count

        if count == 0:
            return "[semantic] Store is empty (0 facts). Run consolidation first."

        scroll_filter = None
        if topic:
            scroll_filter = Filter(must=[
                FieldCondition(key="topic", match=MatchValue(value=topic))
            ])

        pts, _ = qdrant.scroll(
            collection_name=SEMANTIC_COLLECTION,
            scroll_filter=scroll_filter,
            limit=limit,
            with_payload=True,
        )

        if not pts:
            return f"[semantic] {count} total facts, 0 matching topic='{topic}'"

        lines = [f"=== Semantic Facts ({len(pts)}/{count} total) ==="]
        for p in pts:
            pl = p.payload or {}
            fact = pl.get("fact_text", pl.get("data", "?"))
            topic_val = pl.get("topic", "?")
            conf = pl.get("confidence", 0)
            evidence = pl.get("evidence_count", 0)
            lines.append(f"- [{topic_val}] (conf={conf:.2f}, evidence={evidence}) {fact}")

        return "\n".join(lines)
    except Exception as e:
        return f"[semantic] Error: {e}"


# ============================================================
# STATS (MCP TOOL)
# ============================================================

def get_consolidation_stats() -> str:
    """Get statistics about consolidation runs.

    MCP tool for monitoring.
    """
    try:
        conn = _consolidation_conn()
        total_runs = conn.execute(
            "SELECT COUNT(*) FROM consolidation_log"
        ).fetchone()[0]
        total_facts_created = conn.execute(
            "SELECT COALESCE(SUM(facts_created), 0) FROM consolidation_log"
        ).fetchone()[0]
        total_recon = conn.execute(
            "SELECT COUNT(*) FROM reconsolidation_log"
        ).fetchone()[0]
        labile_count = conn.execute(
            "SELECT COUNT(*) FROM labile_memories"
        ).fetchone()[0]
        conn.close()

        semantic_count = 0
        try:
            info = qdrant.get_collection(SEMANTIC_COLLECTION)
            semantic_count = info.points_count
        except Exception:
            pass

        return (
            f"=== Consolidation Stats ===\n"
            f"Total runs: {total_runs}\n"
            f"Total semantic facts created: {total_facts_created}\n"
            f"Semantic store size: {semantic_count}\n"
            f"Total reconsolidation events: {total_recon}\n"
            f"Currently labile memories: {labile_count}"
        )
    except Exception as e:
        return f"Error getting stats: {e}"


# ============================================================
# COUNT UNCONSOLIDATED
# ============================================================

def count_unconsolidated_episodic(lookback_hours: int = 24) -> int:
    """Count unconsolidated episodic memories in the last N hours."""
    cutoff = datetime.now() - timedelta(hours=lookback_hours)
    scroll_filter = Filter(must_not=[
        FieldCondition(key="consolidation_status", match=MatchValue(value="consolidated"))
    ])

    count = 0
    offset = None
    while True:
        pts, next_offset = qdrant.scroll(
            collection_name=COLLECTION_NAME,
            scroll_filter=scroll_filter,
            limit=100,
            with_payload=True,
            offset=offset,
        )
        if not pts:
            break
        for p in pts:
            created_str = (p.payload or {}).get("created_at", "")
            try:
                created = datetime.fromisoformat(str(created_str).replace("Z", "+00:00"))
                if created.tzinfo:
                    created = created.replace(tzinfo=None)
                if created >= cutoff:
                    count += 1
            except Exception:
                pass
        if not next_offset:
            break
        offset = next_offset

    return count
