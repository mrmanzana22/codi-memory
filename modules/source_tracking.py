"""
SOURCE TRACKING MODULE — Source Monitoring Framework (Johnson et al. 1993)
==========================================================================
Tracks WHERE and WHEN memories were created to prevent confabulation.

SMF principle: memories aren't tagged with a single label — the brain
evaluates multiple features (perceptual, contextual, cognitive, affective)
to determine the origin. This module captures those features at encoding time.

Dual-process support:
  - context_summary: fast heuristic path (1 line)
  - context_full: slow deliberative path (recent WM items)

Created: 2026-03-15 (Source Monitoring Implementation)
Neuroscience basis:
  - Johnson, Hashtroudi & Lindsay 1993: Source Monitoring Framework
  - MemOS (arXiv:2512.13564): MemCube provenance tracking
"""

import json
import logging
from typing import Any, Dict, List, Optional

from psycopg.rows import dict_row

from modules.config_pg import get_conn

_logger = logging.getLogger(__name__)

# ============================================================
# POSTGRESQL INIT
# ============================================================

_TABLES_INITIALIZED = False


def _ensure_tables():
    """Create memory_sources table in PG if not exists. Called once per process."""
    global _TABLES_INITIALIZED
    if _TABLES_INITIALIZED:
        return
    try:
        with get_conn() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS memory_sources (
                    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
                    memory_id UUID NOT NULL,

                    -- WHO (Johnson: contextual features)
                    session_id TEXT,

                    -- WHAT TOPIC (Johnson: contextual features)
                    active_topic TEXT,
                    active_project TEXT,

                    -- HOW (Johnson: cognitive operations)
                    source_type TEXT NOT NULL DEFAULT 'conversation',

                    -- CONTEXT (Johnson: perceptual + contextual detail)
                    context_summary TEXT,
                    context_full TEXT NOT NULL,

                    -- EMOTION AT CREATION (Johnson: affective features)
                    emotion_snapshot JSONB,

                    created_at TIMESTAMPTZ DEFAULT now()
                )
            """)
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_ms_memory_id "
                "ON memory_sources(memory_id)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_ms_session_id "
                "ON memory_sources(session_id)"
            )
        _TABLES_INITIALIZED = True
        _logger.info("memory_sources table ensured")
    except Exception as e:
        _logger.error("Failed to ensure memory_sources table: %s", e)


# ============================================================
# CORE FUNCTIONS
# ============================================================


def record_source(
    memory_id: str,
    session_id: str = "",
    source_type: str = "conversation",
    context_full: str = "",
    context_summary: str = "",
    active_topic: str = "",
    active_project: str = "",
    emotion_snapshot: Optional[Dict[str, Any]] = None,
) -> Optional[str]:
    """Record the source context for a memory at creation time.

    Returns the source record UUID, or None on failure.
    """
    _ensure_tables()
    try:
        with get_conn() as conn:
            row = conn.execute(
                """
                INSERT INTO memory_sources
                    (memory_id, session_id, source_type,
                     context_full, context_summary,
                     active_topic, active_project, emotion_snapshot)
                VALUES
                    (%(memory_id)s, %(session_id)s, %(source_type)s,
                     %(context_full)s, %(context_summary)s,
                     %(active_topic)s, %(active_project)s, %(emotion_snapshot)s)
                RETURNING id
                """,
                {
                    "memory_id": memory_id,
                    "session_id": session_id or None,
                    "source_type": source_type,
                    "context_full": context_full,
                    "context_summary": context_summary or context_full[:120],
                    "active_topic": active_topic or None,
                    "active_project": active_project or None,
                    "emotion_snapshot": json.dumps(emotion_snapshot) if emotion_snapshot else None,
                },
            ).fetchone()
            return str(row[0]) if row else None
    except Exception as e:
        _logger.error("record_source failed for memory_id=%s: %s", memory_id, e)
        return None


def get_source(memory_id: str) -> Optional[Dict[str, Any]]:
    """Get the source record for a memory. Returns dict or None."""
    _ensure_tables()
    try:
        with get_conn() as conn:
            with conn.cursor(row_factory=dict_row) as cur:
                row = cur.execute(
                    "SELECT * FROM memory_sources WHERE memory_id = %s ORDER BY created_at DESC LIMIT 1",
                    (memory_id,),
                ).fetchone()
            if row:
                # Convert UUID/datetime to strings for JSON serialization
                result = {}
                for k, v in row.items():
                    if hasattr(v, "isoformat"):
                        result[k] = v.isoformat()
                    elif hasattr(v, "hex"):  # UUID
                        result[k] = str(v)
                    else:
                        result[k] = v
                return result
            return None
    except Exception as e:
        _logger.error("get_source failed for memory_id=%s: %s", memory_id, e)
        return None


def get_sources_batch(memory_ids: List[str]) -> Dict[str, Dict[str, Any]]:
    """Batch query sources for multiple memory IDs.

    Returns dict mapping memory_id -> source record.
    """
    if not memory_ids:
        return {}
    _ensure_tables()
    try:
        with get_conn() as conn:
            with conn.cursor(row_factory=dict_row) as cur:
                rows = cur.execute(
                    """
                    SELECT DISTINCT ON (memory_id) *
                    FROM memory_sources
                    WHERE memory_id = ANY(%s::uuid[])
                    ORDER BY memory_id, created_at DESC
                    """,
                    (memory_ids,),
                ).fetchall()
            result = {}
            for row in rows:
                mid = str(row["memory_id"])
                entry = {}
                for k, v in row.items():
                    if hasattr(v, "isoformat"):
                        entry[k] = v.isoformat()
                    elif hasattr(v, "hex"):
                        entry[k] = str(v)
                    else:
                        entry[k] = v
                result[mid] = entry
            return result
    except Exception as e:
        _logger.error("get_sources_batch failed: %s", e)
        return {}


def has_source(memory_id: str) -> bool:
    """Quick check if a memory has source tracking."""
    _ensure_tables()
    try:
        with get_conn() as conn:
            row = conn.execute(
                "SELECT 1 FROM memory_sources WHERE memory_id = %s LIMIT 1",
                (memory_id,),
            ).fetchone()
            return row is not None
    except Exception:
        return False


# ============================================================
# TOOL REGISTRATION
# ============================================================


def register_tools(mcp):
    """Register source tracking MCP tools."""

    @mcp.tool()
    def get_memory_source(memory_id: str) -> str:
        """Get the original context where a memory was created.

        Returns source type, session, topic, emotion snapshot,
        and the conversation context at creation time.
        Useful for verifying if a memory is real vs confabulated.

        Args:
            memory_id: UUID of the memory to check
        """
        source = get_source(memory_id)
        if not source:
            return json.dumps({
                "found": False,
                "memory_id": memory_id,
                "message": "No source record found. Memory may predate source tracking.",
            }, ensure_ascii=False)

        # Format for readability
        lines = [
            "# Memory Source",
            f"**Memory ID:** {memory_id}",
            f"**Source Type:** {source.get('source_type', '?')}",
            f"**Session:** {source.get('session_id', '?')}",
            f"**Topic:** {source.get('active_topic', '?')}",
            f"**Project:** {source.get('active_project', '?')}",
            f"**Created:** {source.get('created_at', '?')}",
        ]

        emo = source.get("emotion_snapshot")
        if emo and isinstance(emo, dict):
            p = emo.get("p", emo.get("pleasure", 0))
            a = emo.get("a", emo.get("arousal", 0))
            d = emo.get("d", emo.get("dominance", 0))
            lines.append(f"**Emotion (PAD):** P={p:.2f} A={a:.2f} D={d:.2f}")

        summary = source.get("context_summary", "")
        if summary:
            lines.append(f"\n## Summary\n{summary}")

        full = source.get("context_full", "")
        if full:
            lines.append(f"\n## Full Context\n{full}")

        pretty = "\n".join(lines)
        return json.dumps({
            "found": True,
            "pretty": pretty,
            "source": source,
        }, ensure_ascii=False, default=str)
