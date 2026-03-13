"""
Codi Memory - Working Memory module (Fase 2).
Short-term, high-relevance items with temporal chains and narrative traces.
PostgreSQL-backed (config_pg pool), autocommit=True.
"""

import json
import hashlib
import socket
import time
from datetime import datetime, timedelta
from contextlib import contextmanager

from modules.config import (
    WORKING_MEMORY_MAX_ACTIVE,
    now_col, now_iso, TZ_COL,
)
from modules.config_pg import get_conn
from modules.secret_redact import redact_secrets
from psycopg.rows import dict_row

# Dedup window to prevent identical pushes (Proposal #57: 30→120s)
# Covers async add_memory_smart() latency (observed 15-84s)
_push_dedup = {}  # {content_hash: timestamp}
_DEDUP_WINDOW_S = 120

# Machine identifier for multi-host tracking
_MACHINE_ID = socket.gethostname()

# ============================================================
# DATABASE INIT
# ============================================================

_TABLES_INITIALIZED = False


def _ensure_tables():
    """Create PG tables if not exist. Called once per process."""
    global _TABLES_INITIALIZED
    if _TABLES_INITIALIZED:
        return
    try:
        with get_conn() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS working_memory (
                    id          SERIAL PRIMARY KEY,
                    content     TEXT NOT NULL,
                    topic       TEXT DEFAULT 'general',
                    relevance   REAL DEFAULT 0.5,
                    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    occurred_at TIMESTAMPTZ,
                    source      TEXT DEFAULT 'interaction',
                    related_memory_id TEXT,
                    chain_id    TEXT,
                    active      BOOLEAN DEFAULT TRUE,
                    last_accessed_at TIMESTAMPTZ,
                    access_count INTEGER DEFAULT 0,
                    machine_id  TEXT DEFAULT NULL
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS narrative_traces (
                    id           SERIAL PRIMARY KEY,
                    trace_name   TEXT NOT NULL UNIQUE,
                    chain_ids    JSONB NOT NULL,
                    theme        TEXT,
                    created_at   TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    last_updated TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    active       BOOLEAN DEFAULT TRUE
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS trace_chains (
                    trace_id INTEGER NOT NULL REFERENCES narrative_traces(id) ON DELETE CASCADE,
                    chain_id TEXT NOT NULL,
                    UNIQUE (trace_id, chain_id)
                )
            """)
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_wm_active_relevance "
                "ON working_memory(active, relevance DESC) WHERE active = TRUE"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_wm_chain_active_time "
                "ON working_memory(chain_id, active, occurred_at)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_wm_topic_active_time "
                "ON working_memory(topic, active, occurred_at)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_wm_created_at "
                "ON working_memory(created_at)"
            )
            # Add machine_id column if not exists (Fase 5 multi-host)
            conn.execute(
                "ALTER TABLE working_memory ADD COLUMN IF NOT EXISTS "
                "machine_id TEXT DEFAULT NULL"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_traces_active_updated "
                "ON narrative_traces(active, last_updated)"
            )
            conn.execute(
                "CREATE UNIQUE INDEX IF NOT EXISTS uq_trace_chain "
                "ON trace_chains(trace_id, chain_id)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_trace_chains_chain "
                "ON trace_chains(chain_id)"
            )
        _TABLES_INITIALIZED = True
    except Exception:
        raise


@contextmanager
def _get_conn():
    """Yield a pooled PG connection. Maintains context manager interface
    for backward compatibility with session_bridge, flush, wiring, sleep_loop."""
    _ensure_tables()
    with get_conn() as conn:
        yield conn


# ============================================================
# SCORING
# ============================================================

def _hours_since(ts_val) -> float:
    """Hours elapsed since a timestamp (datetime or ISO string)."""
    try:
        if hasattr(ts_val, 'total_seconds'):
            # timedelta
            return max(0.0, ts_val.total_seconds() / 3600)
        if hasattr(ts_val, 'isoformat'):
            dt = ts_val
        else:
            dt = datetime.fromisoformat(str(ts_val))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=TZ_COL)
        delta = now_col() - dt
        return max(0.0, delta.total_seconds() / 3600)
    except Exception:
        return 168.0  # 7 days fallback


def _effective_score(relevance, last_accessed_at, access_count, created_at):
    """Compute effective score in [0, 1]."""
    relevance = min(1.0, max(0.0, float(relevance or 0)))
    ref_time = created_at or last_accessed_at  # Proposal #116: use creation time, not last access (avoids self-reinforcing loop)
    hours = _hours_since(ref_time) if ref_time else 168.0
    recency = min(1.0, max(0.0, 1.0 - (hours / 168.0)))
    frequency = min(1.0, max(0.0, (access_count or 0) / 10.0))
    return min(1.0, 0.5 * relevance + 0.3 * recency + 0.2 * frequency)


# ============================================================
# AUTO-CHAIN LOGIC
# ============================================================

_general_counter = 0
_MAX_GENERAL_COUNTER = 1_000_000

def _resolve_chain_id(conn, topic: str, occurred_at) -> str:
    """Determine chain_id using temporal window logic."""
    global _general_counter
    if topic == "general":
        _general_counter += 1
        if _general_counter >= _MAX_GENERAL_COUNTER:
            _general_counter = 0
        ts = now_col().strftime("%Y%m%d%H%M%S")
        return f"general_{ts}_{_general_counter}"

    # Look for active items with same topic within 7 days
    try:
        if hasattr(occurred_at, 'isoformat'):
            ref_dt = occurred_at
        else:
            ref_dt = datetime.fromisoformat(str(occurred_at))
        if ref_dt.tzinfo is None:
            ref_dt = ref_dt.replace(tzinfo=TZ_COL)
    except Exception:
        ref_dt = now_col()

    window_start = (ref_dt - timedelta(days=7)).isoformat()
    window_end = (ref_dt + timedelta(days=7)).isoformat()

    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            """SELECT chain_id FROM working_memory
               WHERE topic = %s AND active = TRUE
                 AND occurred_at >= %s AND occurred_at <= %s
               ORDER BY occurred_at DESC LIMIT 1""",
            (topic, window_start, window_end)
        )
        row = cur.fetchone()

    if row and row["chain_id"]:
        return row["chain_id"]

    # New chain
    ts = ref_dt.strftime("%Y%m%d%H")
    return f"{topic}_{ts}"


# ============================================================
# INTERNAL CURATION
# ============================================================

def _auto_curate_buffer(conn):
    """If active items exceed MAX, archive lowest-scored ones."""
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            "SELECT id, relevance, last_accessed_at, access_count, created_at "
            "FROM working_memory WHERE active = TRUE"
        )
        rows = cur.fetchall()

    if len(rows) <= WORKING_MEMORY_MAX_ACTIVE:
        return

    scored = []
    for r in rows:
        s = _effective_score(r["relevance"], r["last_accessed_at"],
                             r["access_count"], r["created_at"])
        scored.append((r["id"], s))

    scored.sort(key=lambda x: x[1], reverse=True)
    to_archive = [item[0] for item in scored[WORKING_MEMORY_MAX_ACTIVE:]]

    if to_archive:
        conn.execute(
            "UPDATE working_memory SET active = FALSE WHERE id = ANY(%s)",
            (to_archive,)
        )


# ============================================================
# CONTEXT LOADER (for despertar_codi)
# ============================================================

# ============================================================
# PUBLIC HELPER FUNCTIONS (for session_bridge, flush, wiring)
# ============================================================

def wm_get_active_topics(limit: int = 10) -> list:
    """Get distinct active topics ordered by relevance (for session_bridge, flush)."""
    try:
        with _get_conn() as conn:
            with conn.cursor(row_factory=dict_row) as cur:
                cur.execute(
                    "SELECT DISTINCT topic FROM working_memory "
                    "WHERE active = TRUE AND topic IS NOT NULL "
                    "ORDER BY topic LIMIT %s",
                    (limit,)
                )
                rows = cur.fetchall()
        return [r["topic"] for r in rows if r["topic"]]
    except Exception:
        return []


def wm_get_active_items(limit: int = 15) -> list:
    """Get active working memory items ordered by relevance (for session_bridge)."""
    try:
        with _get_conn() as conn:
            with conn.cursor(row_factory=dict_row) as cur:
                cur.execute(
                    "SELECT content, topic, relevance FROM working_memory "
                    "WHERE active = TRUE ORDER BY relevance DESC LIMIT %s",
                    (limit,)
                )
                rows = cur.fetchall()
        return [
            {
                "content": (r["content"] or "")[:200],
                "topic": r["topic"],
                "relevance": round(float(r["relevance"] or 0), 2),
            }
            for r in rows
        ]
    except Exception:
        return []


def wm_get_active_count() -> int:
    """Get count of active working memory items (for session_bridge)."""
    try:
        with _get_conn() as conn:
            row = conn.execute(
                "SELECT COUNT(*) FROM working_memory WHERE active = TRUE"
            ).fetchone()
            return row[0] if row else 0
    except Exception:
        return 0


def wm_get_active_traces(limit: int = 10) -> list:
    """Get active narrative traces (for session_bridge)."""
    try:
        with _get_conn() as conn:
            with conn.cursor(row_factory=dict_row) as cur:
                cur.execute(
                    "SELECT trace_name, theme FROM narrative_traces "
                    "WHERE active = TRUE LIMIT %s",
                    (limit,)
                )
                rows = cur.fetchall()
        return [{"trace_name": r["trace_name"], "theme": r["theme"]} for r in rows]
    except Exception:
        return []


def wm_get_decision_items(limit: int = 5) -> list:
    """Get active working memory items with topic='decision' (for session_bridge)."""
    try:
        with _get_conn() as conn:
            with conn.cursor(row_factory=dict_row) as cur:
                cur.execute(
                    "SELECT content FROM working_memory "
                    "WHERE active = TRUE AND topic = 'decision' "
                    "ORDER BY relevance DESC LIMIT %s",
                    (limit,)
                )
                rows = cur.fetchall()
        return [{"decision": (r["content"] or "")[:200], "why": "wm_topic"} for r in rows]
    except Exception:
        return []


def wm_apply_decay(multiplier: float) -> int:
    """Apply exponential decay to active WM items (for wiring.py).

    Sets relevance = GREATEST(0.1, relevance * multiplier) for all active items.
    Returns number of rows updated.
    """
    try:
        with _get_conn() as conn:
            cur = conn.execute(
                "UPDATE working_memory "
                "SET relevance = GREATEST(0.1, relevance * %s) "
                "WHERE active = TRUE",
                (multiplier,)
            )
            return cur.rowcount
    except Exception:
        return 0


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
            with conn.cursor(row_factory=dict_row) as cur:
                cur.execute("SELECT * FROM working_memory WHERE active = TRUE")
                rows = cur.fetchall()

            if not rows:
                return json.dumps({
                    "items": [], "count": 0,
                    "pretty": "# WORKING MEMORY\nVacia."
                }, ensure_ascii=False)

            now = now_iso()
            ids = [r["id"] for r in rows]

            # Update access in same connection (autocommit handles each statement)
            conn.execute(
                "UPDATE working_memory SET access_count = access_count + 1, "
                "last_accessed_at = %s WHERE id = ANY(%s)",
                (now, ids)
            )

            # Score and sort
            items = []
            for r in rows:
                s = _effective_score(r["relevance"], r["last_accessed_at"],
                                     r["access_count"], r["created_at"])
                items.append({
                    "id": r["id"],
                    "content": r["content"],
                    "topic": r["topic"],
                    "relevance": r["relevance"],
                    "chain_id": r["chain_id"],
                    "source": r["source"],
                    "created_at": r["created_at"].isoformat() if hasattr(r["created_at"], "isoformat") else r["created_at"],
                    "occurred_at": r["occurred_at"].isoformat() if r["occurred_at"] and hasattr(r["occurred_at"], "isoformat") else r["occurred_at"],
                    "related_memory_id": r["related_memory_id"],
                    "access_count": (r["access_count"] or 0) + 1,
                    "last_accessed_at": now,
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
        return json.dumps({"error": redact_secrets(str(e))}, ensure_ascii=False)


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
        # Dedup: skip if identical content pushed in last 120 seconds
        content_hash = hashlib.md5(content[:300].encode()).hexdigest()
        _now = time.time()
        if content_hash in _push_dedup and (_now - _push_dedup[content_hash]) < _DEDUP_WINDOW_S:
            return json.dumps({
                "id": None, "deduped": True, "chain_id": None,
                "topic": topic, "relevance": relevance,
                "pretty": f"# WORKING MEMORY\nDeduped: {content[:60]}... (same content pushed <{_DEDUP_WINDOW_S}s ago)",
            }, ensure_ascii=False)
        # Periodic cleanup of stale entries
        if len(_push_dedup) > 100:
            cutoff = _now - _DEDUP_WINDOW_S
            for k in list(_push_dedup):
                if _push_dedup[k] < cutoff:
                    del _push_dedup[k]

        created_at = now_iso()
        if occurred_at is None:
            occurred_at = created_at

        relevance = min(1.0, max(0.0, float(relevance)))
        base_relevance = relevance

        # Arousal modulates WM admission priority (Phelps 2006)
        try:
            from modules.config import _emotional_state
            current_arousal = abs(float(_emotional_state["current"].get("arousal", 0.0)))
            if current_arousal > 0.05:
                relevance = min(1.0, relevance + current_arousal * 0.15)
        except Exception:
            pass

        with _get_conn() as conn:
            with conn.transaction():
                chain_id = _resolve_chain_id(conn, topic, occurred_at)

                with conn.cursor() as cur:
                    cur.execute(
                        """INSERT INTO working_memory
                           (content, topic, relevance, created_at, occurred_at,
                            source, chain_id, active, access_count, machine_id)
                           VALUES (%s, %s, %s, %s, %s, %s, %s, TRUE, 0, %s)
                           RETURNING id""",
                        (content, topic, relevance, created_at, occurred_at,
                         source, chain_id, _MACHINE_ID)
                    )
                    new_id = cur.fetchone()[0]

                _auto_curate_buffer(conn)

            # Cache dedup AFTER successful transaction (Bug #009)
            _push_dedup[content_hash] = _now

            return json.dumps({
                "id": new_id,
                "chain_id": chain_id,
                "topic": topic,
                "relevance": base_relevance,
                "effective_relevance": relevance,
                "created_at": created_at,
                "occurred_at": occurred_at,
                "pretty": f"# WORKING MEMORY\nPushed: {content[:60]}... -> chain {chain_id}",
            }, ensure_ascii=False)

    except Exception as e:
        return json.dumps({"error": redact_secrets(str(e))}, ensure_ascii=False)


def update_working_memory(
    item_id: int,
    relevance: float = None,
    active: int = None
) -> str:
    """
    Updates an active working memory item's relevance or active status.
    Only editable if currently active=TRUE. Archived items are historical.

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
            sets.append("relevance = %s")
            params.append(relevance)
        if active is not None:
            active_bool = bool(active)
            sets.append("active = %s")
            params.append(active_bool)

        params.append(int(item_id))

        with _get_conn() as conn:
            cursor = conn.execute(
                f"UPDATE working_memory SET {', '.join(sets)} "
                f"WHERE id = %s AND active = TRUE",
                params
            )

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
        return json.dumps({"error": redact_secrets(str(e))}, ensure_ascii=False)


def get_narrative_chain(
    topic_or_chain_id: str,
    depth: int = 20
) -> str:
    """
    Retrieves a narrative chain by chain_id or topic.
    Shows full timeline (active + archived).

    Args:
        topic_or_chain_id: A chain_id (e.g., 'trading_2026020808') or topic name
        depth: Max items to return (default 20)
    """
    try:
        depth = min(100, max(1, int(depth)))

        with _get_conn() as conn:
            # Step 1: Try as chain_id
            with conn.cursor(row_factory=dict_row) as cur:
                cur.execute(
                    "SELECT * FROM working_memory WHERE chain_id = %s "
                    "ORDER BY occurred_at ASC LIMIT %s",
                    (topic_or_chain_id, depth)
                )
                rows = cur.fetchall()

            # Step 2: Fallback to topic
            if not rows:
                with conn.cursor(row_factory=dict_row) as cur:
                    cur.execute(
                        "SELECT * FROM working_memory WHERE topic = %s "
                        "ORDER BY occurred_at ASC LIMIT %s",
                        (topic_or_chain_id, depth)
                    )
                    rows = cur.fetchall()

        if not rows:
            return json.dumps({
                "items": [], "count": 0,
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
                "created_at": r["created_at"].isoformat() if r["created_at"] and hasattr(r["created_at"], "isoformat") else r["created_at"],
                "occurred_at": r["occurred_at"].isoformat() if r["occurred_at"] and hasattr(r["occurred_at"], "isoformat") else r["occurred_at"],
                "related_memory_id": r["related_memory_id"],
                "active": r["active"],
                "access_count": r["access_count"],
            }

            items.append(item)

        # Pretty
        pretty_lines = [
            f"# NARRATIVE CHAIN: {topic_or_chain_id}",
            f"Items: {len(items)}\n",
        ]
        for it in items:
            status = "active" if it.get("active") else "archived"
            pretty_lines.append(
                f"- [{it['occurred_at']}] ({status}) {it['content'][:80]}"
            )

        return json.dumps({
            "items": items,
            "count": len(items),
            "pretty": "\n".join(pretty_lines),
        }, ensure_ascii=False)

    except Exception as e:
        return json.dumps({"error": redact_secrets(str(e))}, ensure_ascii=False)


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
            with conn.transaction():
                with conn.cursor(row_factory=dict_row) as cur:
                    cur.execute(
                        "SELECT id FROM narrative_traces WHERE trace_name = %s",
                        (trace_name,)
                    )
                    existing = cur.fetchone()

                if existing:
                    trace_id = existing["id"]
                    conn.execute(
                        "UPDATE narrative_traces SET chain_ids = %s::jsonb, theme = %s, "
                        "last_updated = %s WHERE id = %s",
                        (chain_ids_json, theme, now, trace_id)
                    )
                    # Clean and re-insert trace_chains
                    conn.execute(
                        "DELETE FROM trace_chains WHERE trace_id = %s",
                        (trace_id,)
                    )
                    for cid in chain_ids:
                        conn.execute(
                            "INSERT INTO trace_chains (trace_id, chain_id) "
                            "VALUES (%s, %s) ON CONFLICT DO NOTHING",
                            (trace_id, cid)
                        )
                    action = "updated"
                else:
                    with conn.cursor() as cur:
                        cur.execute(
                            "INSERT INTO narrative_traces "
                            "(trace_name, chain_ids, theme, created_at, last_updated, active) "
                            "VALUES (%s, %s::jsonb, %s, %s, %s, TRUE) "
                            "RETURNING id",
                            (trace_name, chain_ids_json, theme, now, now)
                        )
                        trace_id = cur.fetchone()[0]
                    for cid in chain_ids:
                        conn.execute(
                            "INSERT INTO trace_chains (trace_id, chain_id) "
                            "VALUES (%s, %s) ON CONFLICT DO NOTHING",
                            (trace_id, cid)
                        )
                    action = "created"

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
        return json.dumps({"error": redact_secrets(str(e))}, ensure_ascii=False)


# ============================================================
# LIFECYCLE HOOKS (called from consciousness.py ciclo_vida)
# ============================================================

def wm_noche_cleanup():
    """Archive low-relevance, old items. NULL last_accessed_at safe (uses created_at)."""
    try:
        cutoff = now_col() - timedelta(days=7)
        with _get_conn() as conn:
            cursor = conn.execute(
                """UPDATE working_memory SET active = FALSE
                   WHERE active = TRUE AND relevance < 0.2
                     AND COALESCE(last_accessed_at, created_at) < %s::timestamptz""",
                (cutoff,)
            )
            return cursor.rowcount
    except Exception:
        return 0


def wm_active_count() -> int:
    """Count active working memory items."""
    try:
        with _get_conn() as conn:
            row = conn.execute(
                "SELECT COUNT(*) FROM working_memory WHERE active = TRUE"
            ).fetchone()
            return row[0] if row else 0
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
