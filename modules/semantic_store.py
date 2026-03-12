"""
SEMANTIC STORE MODULE

Operations on semantic facts stored in PostgreSQL (pgvector):
- Vector search for semantic facts
- Browsing/filtering facts by topic
- Consolidation statistics
- Counting unconsolidated episodic memories

Backend: pg_store (PostgreSQL + pgvector)
Migrated from Qdrant in Fase 2, Sprint 2.2.
"""

import logging
from datetime import datetime, timedelta

from modules.pg_store import pg
from modules.config_pg import get_conn
from modules.secret_redact import redact_secrets

_logger = logging.getLogger(__name__)


# ============================================================
# SEARCH
# ============================================================

def search_semantic(query: str, limit: int = 5) -> list:
    """Search the semantic store via vector similarity.

    Returns:
        List of semantic facts with scores
    """
    try:
        info = pg.count(is_semantic=True)
        if info.points_count == 0:
            return []

        results = pg.search(query, limit=limit, is_semantic=True)
        facts = []
        for hit in results["results"]:
            facts.append({
                "id": hit["id"],
                "fact": hit.get("memory", ""),
                "topic": hit.get("category", ""),
                "confidence": hit.get("confidence", 0),
                "evidence_count": hit.get("evidence_count", 0),
                "score": hit.get("score", 0),
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
        info = pg.count(is_semantic=True)
        count = info.points_count

        if count == 0:
            return "[semantic] Store is empty (0 facts). Run consolidation first."

        filters = {"category": topic} if topic else None
        pts, _ = pg.scroll(
            filters=filters,
            limit=limit,
            is_semantic=True,
        )

        if not pts:
            return f"[semantic] {count} total facts, 0 matching topic='{topic}'"

        lines = [f"=== Semantic Facts ({len(pts)}/{count} total) ==="]
        for p in pts:
            pl = p.payload
            fact = pl.get("memory", pl.get("fact_text", pl.get("data", "?")))
            topic_val = pl.get("category", "?")
            conf = pl.get("confidence", 0)
            evidence = pl.get("evidence_count", 0)
            lines.append(f"- [{topic_val}] (conf={conf:.2f}, evidence={evidence}) {fact}")

        return "\n".join(lines)
    except Exception as e:
        return f"[semantic] Error: {redact_secrets(str(e))}"


# ============================================================
# STATS (MCP TOOL)
# ============================================================

def get_consolidation_stats() -> str:
    """Get statistics about consolidation runs.

    MCP tool for monitoring.
    """
    try:
        with get_conn() as conn:
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

        semantic_count = 0
        try:
            info = pg.count(is_semantic=True)
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
        return f"Error getting stats: {redact_secrets(str(e))}"


# ============================================================
# COUNT UNCONSOLIDATED
# ============================================================

def count_unconsolidated_episodic(lookback_hours: int = 24) -> int:
    """Count unconsolidated episodic memories in the last N hours."""
    cutoff = datetime.now() - timedelta(hours=lookback_hours)
    with get_conn() as conn:
        row = conn.execute(
            """SELECT COUNT(*) FROM memories
               WHERE is_semantic = FALSE
                 AND COALESCE(metadata->>'consolidation_status', 'new') != 'consolidated'
                 AND created_at >= %s""",
            (cutoff,),
        ).fetchone()
    return row[0] if row else 0
