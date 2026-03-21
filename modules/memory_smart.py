"""
Codi Memory - Smart Memory (FTS5 + Deduplication)
FTS5 functions and add_memory_smart tool.
"""

import logging
import os
import json


from datetime import datetime

_logger = logging.getLogger(__name__)

from modules.config import (
    USER_ID, COLLECTION_NAME, BACKUP_FILE, FTS_DB_PATH, now_iso,
    CONTRADICTION_SCORE_FLOOR, CONTRADICTION_PE_SILENT, CONTRADICTION_PE_ALERT,
    CONTRADICTION_MIN_ENTITIES, CONTRADICTION_COOLDOWN_MINUTES,
    CONTRADICTION_MAX_PER_SESSION,
)
from modules.pg_store import pg
from modules.config_pg import get_conn as get_pg_conn
from modules.destructive_guard import (
    is_guard_enabled, compute_fingerprint,
    request_confirmation, validate_and_consume,
)
from modules.db_pool import get_conn
from modules.secret_redact import redact_secrets

# Sprint 13: EFE-gated active learning threshold
# Adaptive: high when model confident, low when uncertain
EFE_GATE_BASE_THRESHOLD = 0.3


def _fts_conn():
    """Return a pooled SQLite connection. Callers should NOT close it."""
    return get_conn()

from modules.utils import enrich_with_ownership
from modules.events import event_bus, Events


# ============================================================
# FTS5 FUNCTIONS
# ============================================================

def init_fts_db():
    """Validate FTS-related tables exist (created by migrations).

    Only validates regular tables (memories_text, fts_retry_queue).
    FTS5 virtual table (memories_fts) and triggers are trusted from baseline migration.
    """
    from modules.migrations import ensure_schema_ready_db
    ensure_schema_ready_db(FTS_DB_PATH, ["memories_text", "fts_retry_queue"])
    _logger.info("FTS5 tables validated")


def _index_memory_fts_raw(memory_id: str, content: str, category: str = "general",
                          source: str = "experienced", importance: str = "medium"):
    """Raw FTS insert. Triggers sync FTS5 automatically. Raises on failure."""
    conn = _fts_conn()
    conn.execute("""
        INSERT OR REPLACE INTO memories_text (memory_id, content, category, source, importance, created_at)
        VALUES (?, ?, ?, ?, ?, datetime('now'))
    """, (memory_id, content, category, source, importance))
    conn.commit()


def _delete_memory_fts_raw(memory_id: str):
    """Raw FTS delete. Triggers sync FTS5 automatically. Raises on failure."""
    conn = _fts_conn()
    conn.execute("DELETE FROM memories_text WHERE memory_id = ?", (memory_id,))
    conn.commit()


def index_memory_fts(memory_id: str, content: str, category: str = "general",
                     source: str = "experienced", importance: str = "medium") -> bool:
    """Safe FTS index: tries raw insert, queues retry on failure."""
    try:
        _index_memory_fts_raw(memory_id, content, category, source, importance)
        return True
    except Exception as e:
        queue_fts_op(
            memory_id=memory_id,
            op="upsert",
            payload={"content": content, "category": category, "source": source, "importance": importance},
            error=f"index_memory_fts failed: {e}",
        )
        return False


def delete_memory_fts(memory_id: str) -> bool:
    """Safe FTS delete: tries raw delete, queues retry on failure."""
    try:
        _delete_memory_fts_raw(memory_id)
        return True
    except Exception as e:
        queue_fts_op(
            memory_id=memory_id,
            op="delete",
            payload=None,
            error=f"delete_memory_fts failed: {e}",
        )
        return False


# ============================================================
# FTS RETRY QUEUE (P2A)
# ============================================================

def queue_fts_op(memory_id: str, op: str, payload: dict = None, error: str = None) -> dict:
    """Enqueue a failed FTS operation for retry. Idempotent by (memory_id, op)."""
    op = (op or "").strip().lower()
    if op not in ("upsert", "delete"):
        return {"ok": False, "error": f"invalid op: {op}"}

    now = now_iso()
    payload_json = json.dumps(payload, ensure_ascii=False) if payload else None
    error_short = (str(error) or "")[:500] if error else None

    conn = _fts_conn()
    try:
        conn.execute("""
            INSERT INTO fts_retry_queue
                (memory_id, op, payload_json, status, attempts, last_error, created_at, updated_at)
            VALUES (?, ?, ?, 'pending', 0, ?, ?, ?)
            ON CONFLICT(memory_id, op) DO UPDATE SET
                payload_json = COALESCE(excluded.payload_json, payload_json),
                status = 'pending',
                last_error = COALESCE(excluded.last_error, last_error),
                updated_at = excluded.updated_at
        """, (memory_id, op, payload_json, error_short, now, now))
        conn.commit()
        return {"ok": True, "memory_id": memory_id, "op": op}
    except Exception as e:
        _logger.error("Error enqueuing FTS op: %s", redact_secrets(str(e)))
        return {"ok": False, "error": redact_secrets(str(e))}


def process_fts_queue(limit: int = 50, max_attempts: int = 10) -> dict:
    """Process pending FTS retry queue items. Idempotent and safe."""
    limit = max(1, min(500, int(limit or 50)))
    max_attempts = max(1, min(50, int(max_attempts or 10)))
    now = now_iso()

    conn = _fts_conn()
    try:
        rows = conn.execute("""
            SELECT id, memory_id, op, payload_json, attempts
            FROM fts_retry_queue
            WHERE status = 'pending' AND attempts < ?
            ORDER BY updated_at ASC
            LIMIT ?
        """, (max_attempts, limit)).fetchall()

        processed = 0
        succeeded = 0
        failed = 0

        for row in rows:
            qid, mem_id, op, payload_json, attempts = row
            processed += 1

            try:
                if op == "upsert":
                    payload = json.loads(payload_json) if payload_json else {}
                    content = payload.get("content", "")
                    if not content:
                        raise ValueError("missing content for upsert")
                    _index_memory_fts_raw(
                        mem_id,
                        content=content,
                        category=payload.get("category", "general"),
                        source=payload.get("source", "experienced"),
                        importance=payload.get("importance", "medium"),
                    )
                elif op == "delete":
                    _delete_memory_fts_raw(mem_id)

                conn.execute("""
                    UPDATE fts_retry_queue
                    SET status = 'done', updated_at = ?, last_error = NULL
                    WHERE id = ?
                """, (now, qid))
                conn.commit()
                succeeded += 1

            except Exception as e:
                new_attempts = (attempts or 0) + 1
                new_status = "failed" if new_attempts >= max_attempts else "pending"
                conn.execute("""
                    UPDATE fts_retry_queue
                    SET attempts = ?, last_error = ?, status = ?, updated_at = ?
                    WHERE id = ?
                """, (new_attempts, str(e)[:500], new_status, now, qid))
                conn.commit()
                failed += 1

        pending = conn.execute(
            "SELECT COUNT(*) FROM fts_retry_queue WHERE status = 'pending'"
        ).fetchone()[0]

        return {
            "ok": True,
            "processed": processed,
            "succeeded": succeeded,
            "failed": failed,
            "remaining_pending": pending,
        }
    except Exception as e:
        return {"ok": False, "error": redact_secrets(str(e))}


def fts_queue_stats() -> dict:
    """Get FTS retry queue statistics."""
    conn = _fts_conn()
    try:
        pending = conn.execute("SELECT COUNT(*) FROM fts_retry_queue WHERE status='pending'").fetchone()[0]
        failed = conn.execute("SELECT COUNT(*) FROM fts_retry_queue WHERE status='failed'").fetchone()[0]
        done = conn.execute("SELECT COUNT(*) FROM fts_retry_queue WHERE status='done'").fetchone()[0]
        max_att = conn.execute("SELECT COALESCE(MAX(attempts), 0) FROM fts_retry_queue").fetchone()[0]
        oldest = conn.execute("""
            SELECT created_at FROM fts_retry_queue
            WHERE status = 'pending'
            ORDER BY created_at ASC LIMIT 1
        """).fetchone()
        return {
            "ok": True,
            "pending": pending,
            "failed": failed,
            "done": done,
            "max_attempts": int(max_att or 0),
            "oldest_pending": oldest[0] if oldest else None,
        }
    except Exception as e:
        return {"ok": False, "error": redact_secrets(str(e))}


def search_fts(query: str, limit: int = 20) -> list:
    """Busca usando PG tsvector (BM25-style ranking) con fallback a SQLite FTS5.

    Migrated from SQLite FTS5 to PostgreSQL tsvector in Fase 2.
    Falls back to SQLite FTS5 when PG returns empty (hybrid transition).
    Returns same format: [{memory_id, content, category, source, bm25_rank}]
    """
    if not query or not query.strip():
        return []
    try:
        results = pg.search_fts(query, limit=min(limit, 100))
        if results:
            return results
    except Exception:
        pass
    # Fallback: SQLite FTS5 (for transition period / tests)
    try:
        from modules.fts_safety import sanitize_fts_query
        safe_q = sanitize_fts_query(query)
        if not safe_q:
            return []
        conn = _fts_conn()
        rows = conn.execute(
            "SELECT memory_id, content, category, source, rank "
            "FROM memories_fts WHERE memories_fts MATCH ? "
            "ORDER BY rank LIMIT ?",
            (safe_q, min(limit, 100)),
        ).fetchall()
        return [
            {
                "memory_id": r["memory_id"],
                "content": r["content"],
                "category": r["category"],
                "source": r["source"],
                "bm25_rank": r["rank"],
            }
            for r in rows
        ]
    except Exception:
        return []


def bm25_rank_to_score(rank: float) -> float:
    """Convierte FTS5 BM25 rank a score 0-1.

    FTS5 rank is negative (more negative = better match).
    rank=-10 -> 0.91 (strong), rank=-2 -> 0.67 (weak), rank=0 -> 0.0.
    """
    normalized = abs(rank) if rank is not None else 0.0
    return normalized / (normalized + 1.0)


def get_texts_by_ids(memory_ids: list) -> dict:
    """Bulk fetch original texts from FTS store.

    Returns {memory_id: {content, category, source, importance}}.
    Uses db_pool (no connection churn). Single SELECT ... WHERE IN.
    """
    if not memory_ids:
        return {}
    conn = get_conn()
    placeholders = ",".join("?" for _ in memory_ids)
    rows = conn.execute(
        f"SELECT memory_id, content, category, source, importance "
        f"FROM memories_text WHERE memory_id IN ({placeholders})",
        memory_ids,
    ).fetchall()
    return {
        row["memory_id"]: {
            "content": row["content"],
            "category": row["category"],
            "source": row["source"],
            "importance": row["importance"],
        }
        for row in rows
    }


def get_text_by_id(memory_id: str):
    """Single-ID convenience wrapper. Returns dict or None."""
    result = get_texts_by_ids([memory_id])
    return result.get(memory_id)


def search_with_fts_content(query, user_id=None, limit=5) -> dict:
    """Hybrid search via pg.search (vector + FTS + activation).

    Returns mem0-compatible format: {"results": [{"id", "memory", "score"}, ...]}
    Score is cosine similarity (0-1), used by BMR dedup scoring.
    PG stores original content (no FTS overlay needed).
    """
    try:
        results = pg.search(query, limit=limit, is_semantic=False)
        return results if results else {"results": []}
    except Exception:
        return {"results": []}


def sync_fts_from_backup():
    """Sincroniza todas las memorias existentes al indice FTS5."""
    if not os.path.exists(BACKUP_FILE):
        _logger.warning("No backup found for FTS sync")
        return "No backup found"
    try:
        with open(BACKUP_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)
        memories = data if isinstance(data, list) else data.get("memories", [])
        count = 0
        for mem in memories:
            memory_id = mem.get("id", str(count))
            content = mem.get("memory", "")
            category = mem.get("metadata", {}).get("category", "general") if isinstance(mem.get("metadata"), dict) else "general"
            source = "experienced"
            importance = "medium"
            if content:
                index_memory_fts(memory_id, content, category, source, importance)
                count += 1
        # Rebuild FTS index para asegurar consistencia despues de bulk insert
        try:
            conn = _fts_conn()
            conn.execute("INSERT INTO memories_fts(memories_fts) VALUES('rebuild')")
            conn.commit()
        except Exception:
            pass
        _logger.info("Synced %s memories to FTS index", count)
        return f"Synced {count} memories to FTS index"
    except Exception as e:
        _logger.error("Error syncing FTS: %s", redact_secrets(str(e)))
        return f"Error syncing FTS: {redact_secrets(str(e))}"


# ============================================================
# MCP TOOL FUNCTIONS
# ============================================================

# ============================================================
# PHASE 5: INLINE CONTRADICTION DETECTION (Kumaran & Maguire 2007)
# CA1 hippocampal comparator: check new memory against similar existing
# ones at encoding time. PNAS 2025 confirms mismatch signals are
# episodic-specific, not schematic.
# ============================================================

_contradiction_cooldown: dict = {}  # topic_key -> last_alert_iso
_contradiction_session_count = 0


def reset_contradiction_counter() -> None:
    """Reset the per-session contradiction counter and cooldowns.

    Called at session start (despertar_codi) so the rate limiter
    doesn't carry over from previous sessions. Without this, after
    CONTRADICTION_MAX_PER_SESSION detections the system permanently
    stops detecting contradictions until MCP restart.
    """
    global _contradiction_session_count, _contradiction_cooldown
    _contradiction_session_count = 0
    _contradiction_cooldown.clear()


def _inline_contradiction_check(
    new_content: str,
    candidates: list,
    dedup_threshold: float = 0.90,
) -> dict:
    """Hippocampal comparator: check new memory against similar existing ones.

    Returns:
        {detected: bool, pe: float, conflicting_memory_id, conflicting_text,
         detail, channels, shared_entities} or {detected: False, ...}
    """
    global _contradiction_session_count

    if _contradiction_session_count >= CONTRADICTION_MAX_PER_SESSION:
        return {"detected": False, "reason": "session_rate_limit"}

    from modules.consolidation import detect_contradiction, _extract_key_entities

    best_pe = 0.0
    best_match = None
    candidates_evaluated = 0
    candidates_skipped_score = 0
    candidates_skipped_entities = 0

    for candidate in candidates:
        score = candidate.get("score", 0)
        if score < CONTRADICTION_SCORE_FLOOR or score >= dedup_threshold:
            candidates_skipped_score += 1
            continue

        memory_text = candidate.get("memory", "")
        memory_id = candidate.get("id", "")

        if not memory_text or len(memory_text) < 10:
            continue

        # Quick entity overlap pre-screen (avoid expensive embedding)
        new_entities = _extract_key_entities(new_content)
        mem_entities = _extract_key_entities(memory_text)
        shared = new_entities & mem_entities

        if len(shared) < CONTRADICTION_MIN_ENTITIES:
            candidates_skipped_entities += 1
            continue

        candidates_evaluated += 1

        # Full 4-channel contradiction detection
        result = detect_contradiction(memory_text, new_content)
        pe = result.get("prediction_error", 0.0)
        channels = result.get("channels", {})

        _logger.info(
            "contradiction_scan mem=%s score=%.2f PE=%.3f "
            "c1=%.2f c2=%.2f c3=%.2f c4=%.2f entities=%s",
            memory_id[:8] if memory_id else "?",
            score, pe,
            channels.get("keywords", 0),
            channels.get("topic_confirmation", 0),
            channels.get("negation", 0),
            channels.get("semantic_divergence", 0),
            ",".join(list(shared)[:5]),
        )

        if pe > best_pe and pe >= CONTRADICTION_PE_SILENT:
            best_pe = pe
            best_match = {
                "memory_id": memory_id,
                "memory_text": memory_text[:300],
                "pe": pe,
                "detail": result.get("detail", ""),
                "channels": channels,
                "shared_entities": list(shared)[:10],
            }

    _logger.info(
        "contradiction_check_summary candidates=%d evaluated=%d "
        "skipped_score=%d skipped_entities=%d best_pe=%.3f detected=%s",
        len(candidates), candidates_evaluated,
        candidates_skipped_score, candidates_skipped_entities,
        best_pe, best_match is not None,
    )

    if not best_match or best_pe < CONTRADICTION_PE_SILENT:
        return {"detected": False, "pe": best_pe}

    # Cooldown check by topic
    topic_key = "_".join(sorted(best_match["shared_entities"][:3]))
    now = datetime.now()
    if topic_key in _contradiction_cooldown:
        try:
            last_alert = datetime.fromisoformat(_contradiction_cooldown[topic_key])
            if (now - last_alert).total_seconds() < CONTRADICTION_COOLDOWN_MINUTES * 60:
                _logger.info("contradiction cooldown active for topic=%s", topic_key)
                return {"detected": False, "reason": "cooldown", "pe": best_pe}
        except Exception:
            pass

    # Detected! Update counters
    _contradiction_cooldown[topic_key] = now.isoformat()
    _contradiction_session_count += 1

    _logger.info(
        "CONTRADICTION DETECTED pe=%.3f mem=%s entities=%s session_count=%d",
        best_pe, best_match["memory_id"][:8],
        ",".join(best_match["shared_entities"][:5]),
        _contradiction_session_count,
    )

    return {
        "detected": True,
        "pe": best_pe,
        "conflicting_memory_id": best_match["memory_id"],
        "conflicting_text": best_match["memory_text"],
        "detail": best_match["detail"],
        "channels": best_match["channels"],
        "shared_entities": best_match["shared_entities"],
    }


def _auto_connect_neighbors(new_id: str, content: str, exclude_ids: list = None):
    """Auto-connect new memory to top-K similar neighbors.

    Densifies the graph so spreading activation has edges to traverse.
    Stores connections in 'related_memories' (list) payload field.

    Args:
        new_id: ID of the newly saved memory
        content: Content text (used for similarity search)
        exclude_ids: IDs to skip (e.g. already connected via related_to)
    """
    from modules.config import (
        GRAPH_AUTO_CONNECT_K, GRAPH_AUTO_CONNECT_MAX, GRAPH_AUTO_CONNECT_MIN_SCORE,
    )
    try:
        exclude = set(exclude_ids or [])
        exclude.add(new_id)

        similar = pg.search(content, limit=GRAPH_AUTO_CONNECT_K + 2, is_semantic=False)
        if not similar or not similar.get("results"):
            return

        connections = []
        for r in similar["results"]:
            rid = r.get("id", "")
            score = r.get("score", 0)
            if rid and rid not in exclude and score >= GRAPH_AUTO_CONNECT_MIN_SCORE:
                connections.append(rid)
                exclude.add(rid)
            if len(connections) >= GRAPH_AUTO_CONNECT_MAX:
                break

        if connections:
            pg.update_payload(new_id, {
                'related_memories': connections,
            })
            # S2-05: Record typed edges in SQLite
            try:
                from modules.config import connect_fts
                from modules.utils import now_iso
                edge_conn = connect_fts()
                try:
                    from modules.spreading import _init_edge_table, _record_edges
                    _init_edge_table(edge_conn)
                    _record_edges(edge_conn, new_id, connections, now_iso(),
                                  edge_type="co_occurs")
                finally:
                    edge_conn.close()
            except Exception:
                pass
    except Exception:
        pass


def _compute_efe_informativeness(content: str, category: str, importance: str) -> float:
    """Sprint 13: Compute EFE-based informativeness score for a memory.

    Answers: "Would storing this memory reduce uncertainty in the generative model?"

    Components:
      1. Novelty: how different is this from existing knowledge? (epistemic)
      2. Utility: how likely to be retrieved in future? (pragmatic)
      3. Importance weight: critical/high memories always pass

    Returns:
        Informativeness score [0, 1]. Higher = more informative.

    References:
        Canon v4 PN-19: BMR + EFE for memory operations
        Friston 2017: Expected Free Energy = pragmatic + epistemic
        Maxwell's Demon (Eq.7): only record if it reduces model uncertainty
    """
    import math

    # Component 1: Importance prior (critical always informative)
    importance_score = {
        "critical": 1.0,
        "high": 0.8,
        "medium": 0.5,
        "low": 0.2,
    }.get(importance, 0.5)

    # Component 2: Content richness proxy (longer, more specific = more informative)
    words = content.split()
    word_count = len(words)
    richness = min(1.0, word_count / 50.0)  # Saturates at 50 words

    # Component 3: Category utility (some categories more useful for retrieval)
    category_utility = {
        "identidad": 0.9,
        "aprendizaje": 0.8,
        "proyecto": 0.7,
        "episodio": 0.5,
        "general": 0.4,
    }.get(category, 0.5)

    # Combined informativeness (weighted average)
    informativeness = (
        0.4 * importance_score +
        0.3 * richness +
        0.3 * category_utility
    )

    return min(1.0, max(0.0, informativeness))


def _get_efe_gate_threshold() -> float:
    """Adaptive EFE gate threshold.

    High global precision → high threshold (model confident, be selective).
    Low global precision → low threshold (model uncertain, accept more).
    """
    try:
        from modules.config import connect_fts, FTS_DB_PATH
        import os
        if not os.path.exists(FTS_DB_PATH):
            return EFE_GATE_BASE_THRESHOLD
        conn = connect_fts(FTS_DB_PATH)
        try:
            # Get recent average precision as proxy for model confidence
            row = conn.execute("""
                SELECT AVG(precision_weight) FROM (
                    SELECT precision_weight FROM prediction_results
                    ORDER BY id DESC LIMIT 20
                )
            """).fetchone()
        finally:
            conn.close()
        if row and row[0]:
            precision = row[0]
            # High precision → higher threshold (be more selective)
            # Low precision → lower threshold (accept more data)
            return EFE_GATE_BASE_THRESHOLD * (0.5 + precision)
        return EFE_GATE_BASE_THRESHOLD
    except Exception:
        return EFE_GATE_BASE_THRESHOLD


def _get_arousal() -> float:
    """Get current emotional arousal for BMR modulation."""
    try:
        from modules.config import _emotional_state
        return abs(_emotional_state["current"].get("arousal", 0.0))
    except Exception:
        return 0.0


def _compute_dedup_threshold(importance: str = "medium") -> float:
    """Hippocampal-inspired dynamic dedup threshold.

    LEGACY: Kept as fallback. Primary dedup now uses BMR scoring (Sprint 9).

    Emotional arousal and importance raise the bar for dedup,
    mirroring amygdala-modulated encoding (LaBar & Cabeza 2006).
    """
    base = 0.90
    arousal = _get_arousal()
    if arousal > 0.3:
        base += min(0.05, (arousal - 0.3) * 0.07)
    base += {"critical": 0.05, "high": 0.03}.get(importance, 0.0)
    return min(0.97, max(0.85, round(base, 3)))


def _assess_content_quality(content: str) -> dict:
    """Assess whether content is rich enough to be findable later.

    Nelson & Narens 1990: metacognitive monitoring at encoding.
    Craik & Tulving 1975: depth of processing predicts recall.

    Returns:
        {"pass": bool, "score": float(0-1), "issues": list, "suggestions": list}
    """
    issues = []
    score = 1.0
    words = content.split()

    # Check 1: Minimum length
    if len(words) < 5:
        issues.append("too_short")
        score -= 0.4

    # Check 2: Has at least one specific term (proper noun, tech term, identifier)
    specific_count = 0
    for w in words:
        # snake_case identifiers, file.py, CamelCase, or UPPERCASE acronyms
        if ('_' in w or ('.' in w and not w.endswith('.'))
                or (len(w) > 2 and w[0].isupper() and any(c.islower() for c in w))
                or (len(w) >= 2 and w.isupper() and w.isalpha())):
            specific_count += 1
    if specific_count == 0:
        issues.append("no_specific_terms")
        score -= 0.3

    # Check 3: Not a known generic phrase
    _low = content.lower().strip().rstrip(".")
    _GENERIC = {
        "hicimos cosas hoy", "trabajamos en el proyecto", "implementamos sprints",
        "avanzamos", "todo bien", "seguimos trabajando", "nada nuevo",
        "implementamos cosas", "hicimos varias cosas",
    }
    if _low in _GENERIC:
        issues.append("generic_phrase")
        score -= 0.5

    suggestions = []
    if "no_specific_terms" in issues:
        suggestions.append("Add module names, function names, or file paths")
    if "too_short" in issues:
        suggestions.append("Add more detail: what was done, why, and where")
    if "generic_phrase" in issues:
        suggestions.append("Replace with specific description of what changed")

    return {
        "pass": score >= 0.5,
        "score": max(0.0, round(score, 2)),
        "issues": issues,
        "suggestions": suggestions,
    }


def _auto_enrich_content(content: str, category: str) -> str:
    """Enrich thin memories with contextual keywords at save time.

    Zero LLM calls — uses working memory, session edits, and goals.
    Craik & Lockhart 1972: elaborative encoding at save time.

    Args:
        content: The original memory content.
        category: Memory category.

    Returns:
        Enriched content string (original + context tags).
    """
    import os
    enrichments = []

    # 1. Active working memory topics (PostgreSQL backend)
    try:
        from modules.working_memory import wm_get_active_topics
        topics = wm_get_active_topics(limit=3)
        topics = [t for t in topics if t not in ('', 'general', 'metamemory', 'metamemoria', 'contradiction')]
        if topics:
            enrichments.append(f"[topics: {', '.join(topics)}]")
    except Exception:
        pass

    # 2. Recent session edits (from edit_tracker.py)
    try:
        edits_log = os.path.expanduser("~/.claude/session_edits.jsonl")
        if os.path.exists(edits_log):
            import json as _json
            with open(edits_log, "r", encoding="utf-8") as f:
                lines = f.readlines()[-5:]
            files = set()
            funcs = set()
            for line in lines:
                if not line.strip():
                    continue
                entry = _json.loads(line)
                files.add(entry.get("file", ""))
                funcs.update(entry.get("functions", []))
            files.discard("")
            if files:
                enrichments.append(f"[files: {', '.join(sorted(files)[:5])}]")
            if funcs:
                enrichments.append(f"[functions: {', '.join(sorted(funcs)[:5])}]")
    except Exception:
        pass

    # 3. Active goal title
    try:
        from modules.config import connect_fts, FTS_DB_PATH
        if os.path.exists(FTS_DB_PATH):
            conn = connect_fts(FTS_DB_PATH)
            try:
                row = conn.execute(
                    "SELECT title FROM goals WHERE status = 'active' "
                    "ORDER BY activation DESC LIMIT 1"
                ).fetchone()
            finally:
                conn.close()
            if row and row[0]:
                enrichments.append(f"[goal: {row[0]}]")
    except Exception:
        pass

    if enrichments:
        return content + " " + " ".join(enrichments)
    return content


def add_memory_smart(content: str, category: str = "general",
                     source: str = "experienced", importance: str = "medium",
                     dedup_threshold: float = 0.0,
                     relate_threshold: float = 0.75) -> str:
    """
    Guarda memoria con deduplicacion inteligente.
    Basado en neurociencia: el cerebro consolida, no duplica.

    Args:
        content: El contenido a recordar
        category: Categoria (identidad, aprendizaje, episodio, proyecto, general)
        source: Como obtuve esta memoria (experienced, told, learned, inferred)
        importance: Importancia (critical, high, medium, low)
        dedup_threshold: Umbral para considerar duplicado (0 = auto via PAD+importance)
        relate_threshold: Umbral para marcar como relacionada (default 0.75)

    Returns:
        Resultado de la operacion con explicacion
    """
    try:
        # Sprint 9: BMR-scored dedup (replaces cosine threshold)
        from modules.bmr import compute_log_bayes_factor

        # Resolve dynamic dedup threshold (legacy fallback)
        if dedup_threshold <= 0:
            dedup_threshold = _compute_dedup_threshold(importance)

        arousal = _get_arousal()

        # 1. Buscar memorias similares (FTS-enriched for accurate dedup)
        similar_results = search_with_fts_content(query=content, user_id=USER_ID, limit=3)
        contradiction_result = None  # Phase 5: set by inline check if similar found

        if similar_results and similar_results.get("results"):
            top_result = similar_results["results"][0]
            top_score = top_result.get("score", 0)
            top_id = top_result.get("id", "")
            top_text = top_result.get("memory", "")[:80]

            # 2. BMR-scored dedup decision (Sprint 9, PN-19)
            # Replaces: `if top_score > dedup_threshold`
            # Now: compute log Bayes factor considering importance + arousal
            top_metadata = {k: v for k, v in top_result.items()
                           if k not in ('memory', 'score', 'id')}
            log_bf = compute_log_bayes_factor(
                cosine_score=top_score,
                metadata_a=top_metadata,
                importance_a=top_metadata.get("importance", "medium"),
                importance_b=importance,
                arousal=arousal,
            )

            if log_bf > 0:
                # DUPLICADO — BMR says merged model is better
                return json.dumps({
                    "action": "skipped_duplicate",
                    "score": round(top_score, 3),
                    "bmr_log_bf": round(log_bf, 3),
                    "dedup_method": "bmr",
                    "existing_memory": top_text,
                    "existing_id": top_id,
                    "message": f"BMR dedup: log_bf={log_bf:.2f} (cosine={top_score:.2f}, importance={importance})"
                }, ensure_ascii=False)

            # Phase 5: Inline contradiction check (Kumaran & Maguire 2007)
            contradiction_result = None
            try:
                contradiction_result = _inline_contradiction_check(
                    new_content=content,
                    candidates=similar_results["results"],
                    dedup_threshold=dedup_threshold,
                )
            except Exception as e:
                _logger.warning(
                    "Inline contradiction check failed for content='%s': %s",
                    content[:80], redact_secrets(str(e)),
                )

            if top_score > relate_threshold:
                # SIMILAR - guardar pero relacionar
                _add_result = pg.add(
                    content=content,
                    category=category,
                    source=source,
                    importance=importance,
                )
                new_id = None
                if _add_result and _add_result.get("results"):
                    new_id = _add_result["results"][0].get("id")

                if new_id:
                    # Enriquecer con ownership + relacion
                    enrich_with_ownership(
                        memory_id=new_id,
                        category=category,
                        content=content,
                        source=source,
                        importance=importance
                    )
                    # Agregar metadata de relacion
                    try:
                        pg.update_payload(new_id, {
                            'related_to': top_id,
                            'relation_score': top_score,
                            'relation_type': 'semantic_similar',
                        })
                    except Exception:
                        pass

                    # Mini spreading activation: activar vecinos de la nueva relacion
                    try:
                        from modules.spreading import _spread_activation
                        spread_seeds = [top_id]
                        if new_id:
                            spread_seeds.append(new_id)
                        _spread_activation(spread_seeds, depth=1, factor=0.3, seed_boost=0.05)
                    except Exception:
                        pass

                # P1: backup removed from hot path

                # Indexar en FTS5 (safe: auto-queues on failure)
                if new_id:
                    index_memory_fts(new_id, content, category, source, importance)
                    # Auto-connect to neighbors (graph densification)
                    _auto_connect_neighbors(new_id, content, exclude_ids=[top_id])

                # Emit MEMORY_STORED event
                try:
                    event_bus.emit(Events.MEMORY_STORED, {
                        'memory_id': new_id,
                        'content': content[:300],
                        'category': category,
                        'source': source,
                        'importance': importance,
                        'action': 'saved_with_relation',
                        'related_to': top_id,
                    })
                except Exception:
                    pass

                # Phase 5: emit contradiction event if detected
                if contradiction_result and contradiction_result.get("detected"):
                    try:
                        event_bus.emit(Events.CONTRADICTION_DETECTED, {
                            'pe': contradiction_result['pe'],
                            'conflicting_memory_id': contradiction_result['conflicting_memory_id'],
                            'conflicting_text': contradiction_result['conflicting_text'],
                            'new_content': content[:300],
                            'new_memory_id': new_id,
                            'detail': contradiction_result.get('detail', ''),
                            'channels': contradiction_result.get('channels', {}),
                            'shared_entities': contradiction_result.get('shared_entities', []),
                        })
                    except Exception:
                        pass
                    return json.dumps({
                        "action": "saved_with_contradiction",
                        "new_id": new_id,
                        "related_to": top_id,
                        "score": round(top_score, 3),
                        "contradiction": {
                            "detected": True,
                            "pe": contradiction_result['pe'],
                            "conflicting_memory_id": contradiction_result['conflicting_memory_id'],
                            "conflicting_text": contradiction_result['conflicting_text'][:200],
                        },
                        "message": (
                            f"Memoria guardada. CONTRADICCION DETECTADA (PE={contradiction_result['pe']:.2f}) "
                            f"con memoria {contradiction_result['conflicting_memory_id'][:8]}."
                        )
                    }, ensure_ascii=False)

                return json.dumps({
                    "action": "saved_with_relation",
                    "new_id": new_id,
                    "related_to": top_id,
                    "score": round(top_score, 3),
                    "message": f"Memoria guardada y relacionada con existente (similitud {top_score:.2f})"
                }, ensure_ascii=False)

        # Sprint B: Quality gate — prevent unfindable memories (Nelson & Narens 1990)
        _quality = _assess_content_quality(content)
        if not _quality["pass"]:
            if importance in ("critical", "high"):
                # Auto-enrich instead of rejecting (Craik & Lockhart 1972)
                content = _auto_enrich_content(content, category)
            elif importance == "low":
                # Low importance + low quality = skip
                return json.dumps({
                    "action": "skipped_low_quality",
                    "quality_score": _quality["score"],
                    "issues": _quality["issues"],
                    "suggestions": _quality["suggestions"],
                    "message": "Memory too vague to be findable. " +
                               "; ".join(_quality["suggestions"])
                }, ensure_ascii=False)
            # medium importance: warn but save (enriched if possible)
            elif _quality["score"] < 0.3:
                content = _auto_enrich_content(content, category)

        # Sprint 13: EFE-gated active learning (PN-19)
        # Only record memories that are informative (reduce noise)
        efe_score = _compute_efe_informativeness(content, category, importance)
        efe_threshold = _get_efe_gate_threshold()
        if efe_score < efe_threshold and importance not in ("critical", "high"):
            return json.dumps({
                "action": "skipped_low_informativeness",
                "efe_score": round(efe_score, 3),
                "efe_threshold": round(efe_threshold, 3),
                "message": f"EFE gate: informativeness {efe_score:.2f} < threshold {efe_threshold:.2f}"
            }, ensure_ascii=False)

        # 3. NUEVA - guardar normal
        _add_result = pg.add(
            content=content,
            category=category,
            source=source,
            importance=importance,
        )
        new_id = None
        if _add_result and _add_result.get("results"):
            new_id = _add_result["results"][0].get("id")

        if new_id:
            enrich_with_ownership(
                memory_id=new_id,
                category=category,
                content=content,
                source=source,
                importance=importance
            )

        # P1: backup removed from hot path

        # Indexar en FTS5 (safe: auto-queues on failure)
        if new_id:
            index_memory_fts(new_id, content, category, source, importance)
            # Auto-connect to neighbors (graph densification)
            _auto_connect_neighbors(new_id, content)

        # Emit MEMORY_STORED event
        try:
            event_bus.emit(Events.MEMORY_STORED, {
                'memory_id': new_id,
                'content': content[:300],
                'category': category,
                'source': source,
                'importance': importance,
                'action': 'saved_new',
            })
        except Exception:
            pass

        # Phase 5: emit contradiction event if detected (new memory branch)
        if contradiction_result and contradiction_result.get("detected"):
            try:
                event_bus.emit(Events.CONTRADICTION_DETECTED, {
                    'pe': contradiction_result['pe'],
                    'conflicting_memory_id': contradiction_result['conflicting_memory_id'],
                    'conflicting_text': contradiction_result['conflicting_text'],
                    'new_content': content[:300],
                    'new_memory_id': new_id,
                    'detail': contradiction_result.get('detail', ''),
                    'channels': contradiction_result.get('channels', {}),
                    'shared_entities': contradiction_result.get('shared_entities', []),
                })
            except Exception:
                pass
            return json.dumps({
                "action": "saved_with_contradiction",
                "new_id": new_id,
                "contradiction": {
                    "detected": True,
                    "pe": contradiction_result['pe'],
                    "conflicting_memory_id": contradiction_result['conflicting_memory_id'],
                    "conflicting_text": contradiction_result['conflicting_text'][:200],
                },
                "message": (
                    f"Memoria guardada. CONTRADICCION DETECTADA (PE={contradiction_result['pe']:.2f}) "
                    f"con memoria {contradiction_result['conflicting_memory_id'][:8]}."
                )
            }, ensure_ascii=False)

        return json.dumps({
            "action": "saved_new",
            "new_id": new_id,
            "message": f"Nueva memoria guardada: {content[:50]}..."
        }, ensure_ascii=False)

    except Exception as e:
        return json.dumps({
            "action": "error",
            "message": f"Error: {redact_secrets(str(e))}"
        }, ensure_ascii=False)


def sync_fts_index(dry_run: bool = False, confirm_token: str = "") -> str:
    """Resincroniza el indice FTS5 desde el backup JSON. Requiere confirm_token (two-step)."""
    tool = "sync_fts_index"
    fp = compute_fingerprint(tool)

    if is_guard_enabled():
        # Preview: count docs that would be reindexed
        doc_count = 0
        try:
            conn = _fts_conn()
            row = conn.execute("SELECT COUNT(*) FROM memories_text").fetchone()
            doc_count = row[0] if row else 0
        except Exception:
            pass

        backup_count = 0
        try:
            if os.path.exists(BACKUP_FILE):
                with open(BACKUP_FILE, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                mems = data if isinstance(data, list) else data.get("memories", [])
                backup_count = len(mems)
        except Exception:
            pass

        if dry_run:
            return (
                f"[DRY RUN] sync_fts_index reindexaria {backup_count} memorias "
                f"desde backup. Index actual tiene {doc_count} docs."
            )

        if not confirm_token:
            token = request_confirmation(tool, fp)
            return (
                f"GUARD: sync_fts_index reindexaria {backup_count} memorias "
                f"(index actual: {doc_count} docs). Operacion destructiva de indice.\n"
                f"Para confirmar, llama con confirm_token='{token}'"
            )

        err = validate_and_consume(confirm_token, tool, fp)
        if err:
            return f"GUARD BLOCKED: {err}"

    try:
        init_fts_db()
        result = sync_fts_from_backup()
        return result
    except Exception as e:
        return f"Error sincronizando FTS: {redact_secrets(str(e))}"


# ============================================================
# REGISTER TOOLS
# ============================================================

def _process_fts_queue_now(limit: int = 100) -> str:
    """Procesa la cola de reintentos FTS pendientes. Usar si sospechas desincronizacion FTS/Qdrant.

    Args:
        limit: Maximo de items a procesar (default 100)
    """
    result = process_fts_queue(limit=limit)
    if result.get("ok"):
        lines = ["# FTS Retry Queue - Procesado"]
        lines.append(f"- Procesados: {result['processed']}")
        lines.append(f"- Exitosos: {result['succeeded']}")
        lines.append(f"- Fallidos: {result['failed']}")
        lines.append(f"- Pendientes restantes: {result['remaining_pending']}")
        return "\n".join(lines)
    return f"Error procesando cola FTS: {result.get('error')}"


def _get_fts_queue_stats() -> str:
    """Muestra estadisticas de la cola de reintentos FTS. Util para diagnosticar desincronizacion."""
    s = fts_queue_stats()
    if s.get("ok"):
        lines = ["# FTS Queue Stats"]
        lines.append(f"- Pendientes: {s['pending']}")
        lines.append(f"- Fallidos (max reintentos): {s['failed']}")
        lines.append(f"- Completados: {s['done']}")
        lines.append(f"- Max intentos en cola: {s['max_attempts']}")
        lines.append(f"- Pendiente mas antiguo: {s['oldest_pending'] or 'ninguno'}")
        if s['pending'] == 0 and s['failed'] == 0:
            lines.append("\n*FTS y Qdrant estan sincronizados*")
        elif s['pending'] > 0:
            lines.append(f"\n*Hay {s['pending']} operaciones pendientes. Usa process_fts_queue_now() para procesarlas.*")
        return "\n".join(lines)
    return f"Error obteniendo stats: {s.get('error')}"


# ============================================================
# MCP TRANSPORT — register_tools
# ============================================================

def register_tools(mcp):
    """Registra las herramientas de memoria smart en el servidor MCP."""
    mcp.tool()(add_memory_smart)
    mcp.tool()(sync_fts_index)

    @mcp.tool()
    def process_fts_queue_now(limit: int = 100) -> str:
        """Procesa la cola de reintentos FTS pendientes. Usar si sospechas desincronizacion FTS/Qdrant."""
        return _process_fts_queue_now(limit=limit)

    @mcp.tool()
    def get_fts_queue_stats() -> str:
        """Muestra estadisticas de la cola de reintentos FTS. Util para diagnosticar desincronizacion."""
        return _get_fts_queue_stats()
