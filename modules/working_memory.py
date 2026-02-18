"""
Codi Memory - Working Memory module (Fase 2).
Short-term, high-relevance items with temporal chains and narrative traces.
SQLite-backed (memories_fts.db), connection-per-call, WAL mode.
"""

import json
from datetime import datetime, timedelta
from contextlib import contextmanager

from modules.config import (
    WORKING_MEMORY_MAX_ACTIVE,
    now_col, now_iso, TZ_COL,
    get_qdrant, COLLECTION_NAME,
)
from modules.db_pool import get_conn

# ============================================================
# DATABASE INIT & CONNECTION
# ============================================================

_TABLES_INITIALIZED = False


def _init_tables(conn: sqlite3.Connection):
    """Validate working memory tables exist (created by migrations)."""
    from modules.migrations import ensure_schema_ready
    ensure_schema_ready(conn, [
        "working_memory", "narrative_traces", "trace_chains",
    ])


@contextmanager
def _get_conn():
    """Yield a pooled SQLite connection. PRAGMAs set by db_pool, not here."""
    global _TABLES_INITIALIZED
    conn = get_conn()
    if not _TABLES_INITIALIZED:
        _init_tables(conn)
        _TABLES_INITIALIZED = True
    yield conn


# ============================================================
# SCORING
# ============================================================

def _hours_since(iso_str: str) -> float:
    """Hours elapsed since an ISO timestamp (Colombia TZ aware)."""
    try:
        dt = datetime.fromisoformat(iso_str)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=TZ_COL)
        delta = now_col() - dt
        return max(0.0, delta.total_seconds() / 3600)
    except Exception:
        return 168.0  # 7 days fallback


def _effective_score(relevance, last_accessed_at, access_count, added_at):
    """Compute effective score in [0, 1]."""
    relevance = min(1.0, max(0.0, float(relevance or 0)))
    ref_time = last_accessed_at or added_at
    hours = _hours_since(ref_time) if ref_time else 168.0
    recency = min(1.0, max(0.0, 1.0 - (hours / 168.0)))
    frequency = min(1.0, max(0.0, (access_count or 0) / 10.0))
    return min(1.0, 0.5 * relevance + 0.3 * recency + 0.2 * frequency)


# ============================================================
# AUTO-CHAIN LOGIC
# ============================================================

_general_counter = 0

def _resolve_chain_id(conn: sqlite3.Connection, topic: str, occurred_at: str) -> str:
    """Determine chain_id using temporal window logic."""
    global _general_counter
    if topic == "general":
        _general_counter += 1
        ts = now_col().strftime("%Y%m%d%H%M%S")
        return f"general_{ts}_{_general_counter}"

    # Look for active items with same topic within 7 days
    try:
        ref_dt = datetime.fromisoformat(occurred_at)
        if ref_dt.tzinfo is None:
            ref_dt = ref_dt.replace(tzinfo=TZ_COL)
    except Exception:
        ref_dt = now_col()

    window_start = (ref_dt - timedelta(days=7)).isoformat()
    window_end = (ref_dt + timedelta(days=7)).isoformat()

    row = conn.execute(
        """SELECT chain_id FROM working_memory
           WHERE topic = ? AND active = 1
             AND occurred_at >= ? AND occurred_at <= ?
           ORDER BY occurred_at DESC LIMIT 1""",
        (topic, window_start, window_end)
    ).fetchone()

    if row:
        return row["chain_id"]

    # New chain
    ts = ref_dt.strftime("%Y%m%d%H")
    return f"{topic}_{ts}"


# ============================================================
# INTERNAL CURATION
# ============================================================

def _auto_curate_buffer(conn: sqlite3.Connection):
    """If active items exceed MAX, archive lowest-scored ones."""
    rows = conn.execute(
        "SELECT id, relevance, last_accessed_at, access_count, added_at "
        "FROM working_memory WHERE active = 1"
    ).fetchall()

    if len(rows) <= WORKING_MEMORY_MAX_ACTIVE:
        return

    scored = []
    for r in rows:
        s = _effective_score(r["relevance"], r["last_accessed_at"],
                             r["access_count"], r["added_at"])
        scored.append((r["id"], s))

    scored.sort(key=lambda x: x[1], reverse=True)
    to_archive = [item[0] for item in scored[WORKING_MEMORY_MAX_ACTIVE:]]

    if to_archive:
        placeholders = ",".join("?" * len(to_archive))
        conn.execute(
            f"UPDATE working_memory SET active = 0 WHERE id IN ({placeholders})",
            to_archive
        )


# ============================================================
# CONTEXT LOADER (for despertar_codi)
# ============================================================

def _load_working_memory_context() -> str:
    """Top 10 active items by effective_score, truncated to 2000 chars."""
    try:
        with _get_conn() as conn:
            rows = conn.execute(
                "SELECT id, content, topic, chain_id, relevance, "
                "last_accessed_at, access_count, added_at "
                "FROM working_memory WHERE active = 1"
            ).fetchall()

        if not rows:
            return ""

        scored = []
        for r in rows:
            s = _effective_score(r["relevance"], r["last_accessed_at"],
                                 r["access_count"], r["added_at"])
            scored.append((r, s))

        scored.sort(key=lambda x: x[1], reverse=True)
        top10 = scored[:10]

        lines = []
        total_len = 0
        for r, s in top10:
            content = r["content"][:100]
            topic = r["topic"] or "general"
            chain = r["chain_id"] or "-"
            line = f"- [{s:.2f}] ({topic}/{chain}) {content}"
            if total_len + len(line) > 2000:
                break
            lines.append(line)
            total_len += len(line)

        return "\n".join(lines)
    except Exception:
        return ""


# ============================================================
# TOOLS
# ============================================================

def get_working_memory() -> str:
    """
    Retrieves all active working memory items, ordered by effective score.
    Groups by chain_id. Updates access_count and last_accessed_at.
    """
    try:
        with _get_conn() as conn:
            rows = conn.execute(
                "SELECT * FROM working_memory WHERE active = 1"
            ).fetchall()

            if not rows:
                return json.dumps({
                    "items": [], "count": 0,
                    "pretty": "# WORKING MEMORY\nVacia."
                }, ensure_ascii=False)

            now = now_iso()
            ids = [r["id"] for r in rows]

            # Update access in same connection
            placeholders = ",".join("?" * len(ids))
            conn.execute(
                f"UPDATE working_memory SET access_count = access_count + 1, "
                f"last_accessed_at = ? WHERE id IN ({placeholders})",
                [now] + ids
            )
            conn.commit()

            # Score and sort
            items = []
            for r in rows:
                s = _effective_score(r["relevance"], r["last_accessed_at"],
                                     r["access_count"], r["added_at"])
                items.append({
                    "id": r["id"],
                    "content": r["content"],
                    "topic": r["topic"],
                    "relevance": r["relevance"],
                    "chain_id": r["chain_id"],
                    "source": r["source"],
                    "added_at": r["added_at"],
                    "occurred_at": r["occurred_at"],
                    "related_memory_id": r["related_memory_id"],
                    "access_count": r["access_count"],
                    "last_accessed_at": r["last_accessed_at"],
                    "effective_score": round(s, 4),
                })

            items.sort(key=lambda x: x["effective_score"], reverse=True)

            # Group by chain
            chains = {}
            for item in items:
                cid = item["chain_id"] or "ungrouped"
                chains.setdefault(cid, []).append(item)

            # Pretty
            pretty_lines = ["# WORKING MEMORY", f"Active: {len(items)} items\n"]
            for cid, chain_items in chains.items():
                pretty_lines.append(f"## Chain: {cid}")
                for it in chain_items:
                    pretty_lines.append(
                        f"- [{it['effective_score']:.2f}] {it['content'][:80]}"
                    )
                pretty_lines.append("")

            return json.dumps({
                "items": items,
                "count": len(items),
                "pretty": "\n".join(pretty_lines),
            }, ensure_ascii=False)

    except Exception as e:
        return json.dumps({"error": str(e)}, ensure_ascii=False)


def push_to_working_memory(
    content: str,
    topic: str = "general",
    relevance: float = 0.5,
    occurred_at: str = None,
    source: str = "interaction"
) -> str:
    """
    Pushes a new item into working memory.
    Auto-assigns chain_id via temporal window. Auto-curates if buffer exceeds limit.

    Args:
        content: The content to remember (short-term)
        topic: Topic category (e.g., 'trading', 'fullempaques', 'general')
        relevance: Importance score 0.0-1.0
        occurred_at: When the event happened (ISO). Defaults to now.
        source: Origin of info ('interaction', 'system', 'observation')
    """
    try:
        added_at = now_iso()
        if occurred_at is None:
            occurred_at = added_at

        relevance = min(1.0, max(0.0, float(relevance)))

        with _get_conn() as conn:
            conn.execute("BEGIN IMMEDIATE")
            try:
                chain_id = _resolve_chain_id(conn, topic, occurred_at)

                conn.execute(
                    """INSERT INTO working_memory
                       (content, topic, relevance, added_at, occurred_at,
                        source, chain_id, active, access_count)
                       VALUES (?, ?, ?, ?, ?, ?, ?, 1, 0)""",
                    (content, topic, relevance, added_at, occurred_at,
                     source, chain_id)
                )

                new_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]

                _auto_curate_buffer(conn)
                conn.commit()
            except Exception:
                conn.rollback()
                raise

            return json.dumps({
                "id": new_id,
                "chain_id": chain_id,
                "topic": topic,
                "relevance": relevance,
                "added_at": added_at,
                "occurred_at": occurred_at,
                "pretty": f"# WORKING MEMORY\nPushed: {content[:60]}... -> chain {chain_id}",
            }, ensure_ascii=False)

    except Exception as e:
        return json.dumps({"error": str(e)}, ensure_ascii=False)


def update_working_memory(
    item_id: int,
    relevance: float = None,
    active: int = None
) -> str:
    """
    Updates an active working memory item's relevance or active status.
    Only editable if currently active=1. Archived items are historical.

    Args:
        item_id: The ID of the working memory item
        relevance: New relevance score (0.0-1.0)
        active: Set to 0 to archive, 1 to keep active
    """
    try:
        if relevance is None and active is None:
            return json.dumps({"error": "Nothing to update"}, ensure_ascii=False)

        sets = []
        params = []
        if relevance is not None:
            relevance = min(1.0, max(0.0, float(relevance)))
            sets.append("relevance = ?")
            params.append(relevance)
        if active is not None:
            active = 1 if active else 0
            sets.append("active = ?")
            params.append(active)

        params.append(int(item_id))

        with _get_conn() as conn:
            cursor = conn.execute(
                f"UPDATE working_memory SET {', '.join(sets)} "
                f"WHERE id = ? AND active = 1",
                params
            )
            conn.commit()

            if cursor.rowcount == 0:
                return json.dumps({
                    "error": "Item archivado o inexistente, no se puede modificar",
                    "item_id": int(item_id),
                }, ensure_ascii=False)

            return json.dumps({
                "updated": True,
                "item_id": int(item_id),
                "changes": {
                    k: v for k, v in
                    [("relevance", relevance), ("active", active)]
                    if v is not None
                },
                "pretty": f"# WORKING MEMORY\nUpdated item {item_id}",
            }, ensure_ascii=False)

    except Exception as e:
        return json.dumps({"error": str(e)}, ensure_ascii=False)


def get_narrative_chain(
    topic_or_chain_id: str,
    depth: int = 20
) -> str:
    """
    Retrieves a narrative chain by chain_id or topic.
    Shows full timeline (active + archived). Enriches with Qdrant if available.

    Args:
        topic_or_chain_id: A chain_id (e.g., 'trading_2026020808') or topic name
        depth: Max items to return (default 20)
    """
    try:
        depth = min(100, max(1, int(depth)))
        qdrant_included = False

        with _get_conn() as conn:
            # Step 1: Try as chain_id
            rows = conn.execute(
                "SELECT * FROM working_memory WHERE chain_id = ? "
                "ORDER BY occurred_at ASC LIMIT ?",
                (topic_or_chain_id, depth)
            ).fetchall()

            # Step 2: Fallback to topic
            if not rows:
                rows = conn.execute(
                    "SELECT * FROM working_memory WHERE topic = ? "
                    "ORDER BY occurred_at ASC LIMIT ?",
                    (topic_or_chain_id, depth)
                ).fetchall()

        if not rows:
            return json.dumps({
                "items": [], "count": 0, "qdrant_included": False,
                "pretty": f"# NARRATIVE CHAIN\nNo items for '{topic_or_chain_id}'",
            }, ensure_ascii=False)

        items = []
        for r in rows:
            item = {
                "id": r["id"],
                "content": r["content"],
                "topic": r["topic"],
                "relevance": r["relevance"],
                "chain_id": r["chain_id"],
                "source": r["source"],
                "added_at": r["added_at"],
                "occurred_at": r["occurred_at"],
                "related_memory_id": r["related_memory_id"],
                "active": r["active"],
                "access_count": r["access_count"],
            }

            # Enrich with Qdrant if related_memory_id exists
            if r["related_memory_id"]:
                try:
                    qdrant_client = get_qdrant()
                    points = qdrant_client.retrieve(
                        collection_name=COLLECTION_NAME,
                        ids=[r["related_memory_id"]],
                        with_payload=True
                    )
                    if points:
                        p = points[0]
                        item["qdrant_memory"] = p.payload.get("data", "")
                        qdrant_included = True
                except Exception:
                    pass

            items.append(item)

        # Pretty
        pretty_lines = [
            f"# NARRATIVE CHAIN: {topic_or_chain_id}",
            f"Items: {len(items)} | Qdrant: {'yes' if qdrant_included else 'no'}\n",
        ]
        for it in items:
            status = "active" if it.get("active") else "archived"
            pretty_lines.append(
                f"- [{it['occurred_at']}] ({status}) {it['content'][:80]}"
            )

        return json.dumps({
            "items": items,
            "count": len(items),
            "qdrant_included": qdrant_included,
            "pretty": "\n".join(pretty_lines),
        }, ensure_ascii=False)

    except Exception as e:
        return json.dumps({"error": str(e)}, ensure_ascii=False)


def link_narrative_trace(
    trace_name: str,
    chain_ids: list,
    theme: str = None
) -> str:
    """
    Links multiple chains into a narrative trace (meta-narrative).
    If trace_name exists, updates it (replaces chain_ids).

    Args:
        trace_name: Unique name for the trace (e.g., 'proyecto_consciencia')
        chain_ids: List of chain_ids to link together
        theme: Optional thematic label
    """
    try:
        if not chain_ids or not isinstance(chain_ids, list):
            return json.dumps({"error": "chain_ids must be a non-empty list"},
                              ensure_ascii=False)

        now = now_iso()
        chain_ids_json = json.dumps(chain_ids, ensure_ascii=False)

        with _get_conn() as conn:
            conn.execute("BEGIN IMMEDIATE")
            try:
                existing = conn.execute(
                    "SELECT id FROM narrative_traces WHERE trace_name = ?",
                    (trace_name,)
                ).fetchone()

                if existing:
                    trace_id = existing["id"]
                    conn.execute(
                        "UPDATE narrative_traces SET chain_ids = ?, theme = ?, "
                        "last_updated = ? WHERE id = ?",
                        (chain_ids_json, theme, now, trace_id)
                    )
                    # Clean and re-insert trace_chains
                    conn.execute(
                        "DELETE FROM trace_chains WHERE trace_id = ?",
                        (trace_id,)
                    )
                    for cid in chain_ids:
                        conn.execute(
                            "INSERT INTO trace_chains (trace_id, chain_id) "
                            "VALUES (?, ?)",
                            (trace_id, cid)
                        )
                    action = "updated"
                else:
                    conn.execute(
                        "INSERT INTO narrative_traces "
                        "(trace_name, chain_ids, theme, created_at, last_updated, active) "
                        "VALUES (?, ?, ?, ?, ?, 1)",
                        (trace_name, chain_ids_json, theme, now, now)
                    )
                    trace_id = conn.execute(
                        "SELECT last_insert_rowid()"
                    ).fetchone()[0]
                    for cid in chain_ids:
                        conn.execute(
                            "INSERT INTO trace_chains (trace_id, chain_id) "
                            "VALUES (?, ?)",
                            (trace_id, cid)
                        )
                    action = "created"

                conn.commit()
            except Exception:
                conn.rollback()
                raise

            return json.dumps({
                "trace_id": trace_id,
                "trace_name": trace_name,
                "chain_ids": chain_ids,
                "theme": theme,
                "action": action,
                "pretty": f"# NARRATIVE TRACE\n{action.title()}: {trace_name} "
                          f"({len(chain_ids)} chains)",
            }, ensure_ascii=False)

    except Exception as e:
        return json.dumps({"error": str(e)}, ensure_ascii=False)


# ============================================================
# LIFECYCLE HOOKS (called from consciousness.py ciclo_vida)
# ============================================================

def wm_noche_cleanup():
    """Archive low-relevance, old items. NULL last_accessed_at safe (uses added_at)."""
    try:
        cutoff = (now_col() - timedelta(days=7)).isoformat()
        with _get_conn() as conn:
            cursor = conn.execute(
                """UPDATE working_memory SET active = 0
                   WHERE active = 1 AND relevance < 0.2
                     AND COALESCE(last_accessed_at, added_at) < ?""",
                (cutoff,)
            )
            conn.commit()
            return cursor.rowcount
    except Exception:
        return 0


def wm_active_count() -> int:
    """Count active working memory items."""
    try:
        with _get_conn() as conn:
            row = conn.execute(
                "SELECT COUNT(*) as cnt FROM working_memory WHERE active = 1"
            ).fetchone()
            return row["cnt"] if row else 0
    except Exception:
        return 0


# ============================================================
# REGISTER TOOLS
# ============================================================

def register_tools(mcp):
    mcp.tool()(get_working_memory)
    mcp.tool()(push_to_working_memory)
    mcp.tool()(update_working_memory)
    mcp.tool()(get_narrative_chain)
    mcp.tool()(link_narrative_trace)
