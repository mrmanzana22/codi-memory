"""
pg_store.py - Unified PostgreSQL memory store.
================================================
Fase 1, Sprint 1.2 of the pgvector migration.

Drop-in replacement for mem0 + Qdrant + FTS5.
All operations go to a single PostgreSQL database with pgvector.

Replaces:
    - memory.add() / memory.search() / memory.delete()  (mem0)
    - qdrant.scroll() / qdrant.retrieve() / qdrant.query_points()  (Qdrant)
    - qdrant.upsert() / qdrant.delete() / qdrant.get_collection()  (Qdrant)
    - search_fts() / index_memory_fts()  (SQLite FTS5)

Usage:
    from modules.pg_store import pg

    # Add memory (generates embedding automatically)
    result = pg.add("Codi learned something new", category="aprendizaje")

    # Hybrid search (vector + FTS + activation via RRF)
    results = pg.search("what did Codi learn?", limit=5)

    # Vector-only search
    points = pg.query_vector(embedding, limit=10)

    # Scroll with filters
    points, offset = pg.scroll({"category": "identidad"}, limit=50)
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Any, Optional

import psycopg
from psycopg.rows import dict_row

from modules.config_pg import get_conn

_logger = logging.getLogger(__name__)

# Colombia timezone (UTC-5)
_COL_TZ = timezone(timedelta(hours=-5))

# Embedding model dimensions
EMBEDDING_DIMS = 1536

# Default RRF k parameter
RRF_K = 60

# Default search limits per channel
_K_VECTOR = 40
_K_FTS = 40
_K_ACTIVATION = 40


# ---------------------------------------------------------------------------
# Response types (compatible with Qdrant Point format)
# ---------------------------------------------------------------------------
@dataclass
class PointPayload:
    """Mimics Qdrant point payload for backward compatibility."""

    data: str = ""
    category: str = "general"
    source: str = "experienced"
    narrative_importance: str = "medium"
    confidence: float = 0.5
    evidence_count: int = 0
    memory_type: str = "episodic"
    created_at: str = ""
    updated_at: str = ""
    last_accessed_at: str = ""
    attention_salience: float = 0.0
    attention_access_count: int = 0
    consolidation_status: str = "new"
    topics: list = field(default_factory=list)
    related_memories: list = field(default_factory=list)
    storage_strength: float = 1.0
    retrieval_strength: float = 1.0
    emotion_p: float = 0.0
    emotion_a: float = 0.0
    emotion_d: float = 0.0
    _extra: dict = field(default_factory=dict)

    def get(self, key: str, default: Any = None) -> Any:
        """Dict-like access for backward compatibility."""
        if hasattr(self, key):
            return getattr(self, key)
        return self._extra.get(key, default)

    def __getitem__(self, key: str) -> Any:
        if hasattr(self, key):
            return getattr(self, key)
        return self._extra[key]

    def items(self):
        d = {k: v for k, v in self.__dict__.items() if k != "_extra"}
        d.update(self._extra)
        return d.items()

    def keys(self):
        return [k for k in self.__dict__ if k != "_extra"] + list(self._extra.keys())


@dataclass
class Point:
    """Mimics Qdrant Point for backward compatibility."""

    id: str
    payload: dict
    vector: Optional[list] = None
    score: float = 0.0


@dataclass
class CollectionInfo:
    """Mimics Qdrant CollectionInfo."""

    points_count: int = 0


# ---------------------------------------------------------------------------
# Embedding helper
# ---------------------------------------------------------------------------
def _embed_text(text: str) -> list:
    """Generate embedding using OpenAI text-embedding-3-small.

    Imports from consolidation_common to reuse the LRU cache.
    Falls back to direct OpenAI call if import fails.
    """
    try:
        from modules.consolidation_common import _embed_text as _cached_embed
        return _cached_embed(text)
    except ImportError:
        import openai

        client = openai.OpenAI()
        resp = client.embeddings.create(
            model="text-embedding-3-small", input=text
        )
        return list(resp.data[0].embedding)


# ---------------------------------------------------------------------------
# Row → Point conversion
# ---------------------------------------------------------------------------
def _row_to_point(row: dict, with_vectors: bool = False) -> Point:
    """Convert a PostgreSQL row (dict) to a Point object."""
    payload = {
        "data": row.get("content", ""),
        "fact_text": row.get("content", "") if row.get("is_semantic") else None,
        "category": row.get("category", "general"),
        "source": row.get("source", "experienced"),
        "narrative_importance": row.get("importance", "medium"),
        "importance": row.get("importance", "medium"),
        "confidence": row.get("confidence", 0.5),
        "evidence_count": row.get("evidence_count", 0),
        "memory_type": "semantic" if row.get("is_semantic") else "episodic",
        "is_semantic": row.get("is_semantic", False),
        "created_at": row["created_at"].isoformat() if row.get("created_at") else "",
        "updated_at": row["updated_at"].isoformat() if row.get("updated_at") else "",
        "last_accessed_at": row["last_accessed_at"].isoformat() if row.get("last_accessed_at") else "",
        "attention_salience": row.get("activation_score", 0.0),
        "activation_score": row.get("activation_score", 0.0),
        "attention_access_count": row.get("access_count", 0),
        "access_count": row.get("access_count", 0),
        "storage_strength": row.get("storage_strength", 1.0),
        "retrieval_strength": row.get("retrieval_strength", 1.0),
        "emotion_p": row.get("emotion_p", 0.0),
        "emotion_a": row.get("emotion_a", 0.0),
        "emotion_d": row.get("emotion_d", 0.0),
        "user_id": "hare",
    }

    # Merge JSONB metadata into payload
    metadata = row.get("metadata") or {}
    if isinstance(metadata, dict):
        for k, v in metadata.items():
            if k not in payload:
                payload[k] = v
            # Preserve metadata keys that overlap (e.g., topics, related_memories)
            elif k in ("topics", "related_memories", "causal_links", "source_episode_ids"):
                payload[k] = v

    return Point(
        id=str(row.get("id", "")),
        payload=payload,
        vector=list(row["embedding"]) if with_vectors and row.get("embedding") else None,
        score=row.get("score", 0.0),
    )


# ---------------------------------------------------------------------------
# Filter translation (Qdrant-style dict → SQL WHERE)
# ---------------------------------------------------------------------------
def _build_where(filters: Optional[dict], params: list, is_semantic: Optional[bool] = None) -> str:
    """Build WHERE clause from a filter dict.

    Uses %s placeholders (psycopg3 style). Appends values to params list.

    Supported filter format (simplified Qdrant-compatible):
        {"category": "identidad"}                    → category = %s
        {"importance": "critical"}                     → importance = %s
        {"is_semantic": True}                          → is_semantic = TRUE
        {"consolidation_status": "consolidated"}       → metadata->>'consolidation_status' = %s
        {"activation_score_gt": 0.1}                   → activation_score > %s
        {"created_after": "2026-01-01"}                → created_at >= %s
        {"created_before": "2026-03-01"}               → created_at < %s
        {"metadata_key": {"key": "k", "value": "v"}}  → metadata->>'k' = %s

    Returns SQL WHERE clause (without 'WHERE' keyword).
    """
    clauses = []

    if is_semantic is not None:
        clauses.append("is_semantic = %s")
        params.append(is_semantic)

    if not filters:
        return " AND ".join(clauses) if clauses else "TRUE"

    # Direct column mappings
    _COLUMN_MAP = {
        "category": "category",
        "source": "source",
        "importance": "importance",
        "is_semantic": "is_semantic",
    }

    for key, value in filters.items():
        if key in _COLUMN_MAP:
            clauses.append(f"{_COLUMN_MAP[key]} = %s")
            params.append(value)
        elif key == "activation_score_gt":
            clauses.append("activation_score > %s")
            params.append(float(value))
        elif key == "created_after":
            clauses.append("created_at >= %s")
            params.append(value)
        elif key == "created_before":
            clauses.append("created_at < %s")
            params.append(value)
        elif key == "narrative_importance":
            clauses.append("importance = %s")
            params.append(value)
        elif key == "ownership_source":
            clauses.append("source = %s")
            params.append(value)
        elif key == "consolidation_status":
            clauses.append("metadata->>'consolidation_status' = %s")
            params.append(value)
        elif key == "consolidation_status_not":
            clauses.append("(metadata->>'consolidation_status' IS NULL OR metadata->>'consolidation_status' != %s)")
            params.append(value)
        elif key == "memory_type":
            if value == "semantic":
                clauses.append("is_semantic = TRUE")
            else:
                clauses.append("is_semantic = FALSE")
        elif key == "narrative_importance__not":
            clauses.append("importance != %s")
            params.append(value)
        elif key == "narrative_themes":
            # JSONB array containment: metadata->'narrative_themes' @> '["theme"]'
            clauses.append("metadata->'narrative_themes' @> %s::jsonb")
            params.append(f'["{value}"]')
        elif key == "metadata_key":
            # {"metadata_key": {"key": "k", "value": "v"}}
            if isinstance(value, dict):
                mk = value.get("key", "")
                mv = value.get("value", "")
                clauses.append(f"metadata->>'{mk}' = %s")
                params.append(str(mv))

    return " AND ".join(clauses) if clauses else "TRUE"


# ---------------------------------------------------------------------------
# Main Store Class
# ---------------------------------------------------------------------------
class PGMemoryStore:
    """Unified memory store backed by PostgreSQL + pgvector.

    Replaces mem0, Qdrant, and SQLite FTS5.
    """

    # -----------------------------------------------------------------------
    # ADD (replaces mem0.add)
    # -----------------------------------------------------------------------
    def add(
        self,
        content: str,
        category: str = "general",
        source: str = "experienced",
        importance: str = "medium",
        embedding: Optional[list] = None,
        is_semantic: bool = False,
        confidence: float = 0.5,
        evidence_count: int = 0,
        metadata: Optional[dict] = None,
        emotion_p: float = 0.0,
        emotion_a: float = 0.0,
        emotion_d: float = 0.0,
    ) -> dict:
        """Add a memory to PostgreSQL.

        If embedding is None, generates it via OpenAI.
        FTS index is automatic (GENERATED ALWAYS column).

        Returns:
            {"results": [{"id": uuid_str, "created_at": iso_str}]}
            (mem0-compatible format)
        """
        if embedding is None:
            embedding = _embed_text(content)

        meta = metadata or {}
        now = datetime.now(_COL_TZ)

        with get_conn() as conn:
            row = conn.execute(
                """
                INSERT INTO memories
                    (content, category, source, importance, embedding,
                     is_semantic, confidence, evidence_count,
                     metadata, emotion_p, emotion_a, emotion_d,
                     created_at, updated_at, last_accessed_at)
                VALUES
                    (%(content)s, %(category)s, %(source)s, %(importance)s,
                     %(embedding)s::vector,
                     %(is_semantic)s, %(confidence)s, %(evidence_count)s,
                     %(metadata)s::jsonb, %(emotion_p)s, %(emotion_a)s, %(emotion_d)s,
                     %(now)s, %(now)s, %(now)s)
                RETURNING id, created_at
                """,
                {
                    "content": content,
                    "category": category,
                    "source": source,
                    "importance": importance,
                    "embedding": embedding,
                    "is_semantic": is_semantic,
                    "confidence": confidence,
                    "evidence_count": evidence_count,
                    "metadata": psycopg.types.json.Json(meta),
                    "emotion_p": emotion_p,
                    "emotion_a": emotion_a,
                    "emotion_d": emotion_d,
                    "now": now,
                },
            ).fetchone()

        new_id = str(row[0])
        created = row[1].isoformat() if row[1] else now.isoformat()
        return {"results": [{"id": new_id, "created_at": created}]}

    # -----------------------------------------------------------------------
    # SEARCH (replaces mem0.search + FTS + activation — hybrid RRF)
    # -----------------------------------------------------------------------
    def search(
        self,
        query: str,
        limit: int = 5,
        embedding: Optional[list] = None,
        is_semantic: Optional[bool] = None,
        w_vector: float = 0.40,
        w_fts: float = 0.15,
        w_activation: float = 0.45,
    ) -> dict:
        """Hybrid RRF 3-channel search.

        Channels:
            1. Vector similarity (pgvector HNSW halfvec cosine)
            2. Full-text search (tsvector bilingual English+Spanish)
            3. ACT-R activation score

        Weights are customizable (default: 0.40/0.15/0.45 per WIRING-5).

        Returns:
            {"results": [{"id", "memory", "score", "channel_scores", ...}, ...]}
            (mem0-compatible format with extra fields)
        """
        if embedding is None:
            embedding = _embed_text(query)

        # Sanitize FTS query: words joined with &
        fts_terms = [w for w in query.split() if len(w) > 1]
        fts_query = " & ".join(fts_terms) if fts_terms else query

        # Semantic filter
        semantic_clause = ""
        if is_semantic is True:
            semantic_clause = "AND is_semantic = TRUE"
        elif is_semantic is False:
            semantic_clause = "AND is_semantic = FALSE"

        with get_conn() as conn:
            rows = conn.execute(
                f"""
                WITH vec AS (
                    SELECT id, content, category, source, importance,
                           is_semantic, confidence, evidence_count,
                           metadata, activation_score,
                           storage_strength, retrieval_strength,
                           emotion_p, emotion_a, emotion_d,
                           access_count, created_at, updated_at, last_accessed_at,
                           ROW_NUMBER() OVER (
                               ORDER BY embedding::halfvec(1536) <=> %(emb)s::halfvec(1536)
                           ) AS rank
                    FROM memories
                    WHERE embedding IS NOT NULL {semantic_clause}
                    LIMIT %(k_vec)s
                ),
                fts AS (
                    SELECT id,
                           ROW_NUMBER() OVER (
                               ORDER BY ts_rank(fts_combined, to_tsquery('english', %(fts)s)) DESC
                           ) AS rank
                    FROM memories
                    WHERE fts_combined @@ to_tsquery('english', %(fts)s) {semantic_clause}
                    LIMIT %(k_fts)s
                ),
                act AS (
                    SELECT id,
                           ROW_NUMBER() OVER (
                               ORDER BY activation_score DESC
                           ) AS rank
                    FROM memories
                    WHERE activation_score > 0.05 {semantic_clause}
                    LIMIT %(k_act)s
                ),
                combined AS (
                    SELECT
                        COALESCE(v.id, f.id, a.id) AS id,
                        %(w_vec)s * (1.0 / (COALESCE(v.rank, 999) + 60)) AS vec_score,
                        %(w_fts)s * (1.0 / (COALESCE(f.rank, 999) + 60)) AS fts_score,
                        %(w_act)s * (1.0 / (COALESCE(a.rank, 999) + 60)) AS act_score,
                        (%(w_vec)s * (1.0 / (COALESCE(v.rank, 999) + 60)) +
                         %(w_fts)s * (1.0 / (COALESCE(f.rank, 999) + 60)) +
                         %(w_act)s * (1.0 / (COALESCE(a.rank, 999) + 60))) AS rrf_score
                    FROM vec v
                    FULL OUTER JOIN fts f ON v.id = f.id
                    FULL OUTER JOIN act a ON COALESCE(v.id, f.id) = a.id
                )
                SELECT c.id, c.rrf_score, c.vec_score, c.fts_score, c.act_score,
                       m.content, m.category, m.source, m.importance,
                       m.is_semantic, m.confidence, m.evidence_count,
                       m.metadata, m.activation_score,
                       m.storage_strength, m.retrieval_strength,
                       m.emotion_p, m.emotion_a, m.emotion_d,
                       m.access_count, m.created_at, m.updated_at, m.last_accessed_at
                FROM combined c
                JOIN memories m ON c.id = m.id
                ORDER BY c.rrf_score DESC
                LIMIT %(limit)s
                """,
                {
                    "emb": embedding,
                    "fts": fts_query,
                    "k_vec": _K_VECTOR,
                    "k_fts": _K_FTS,
                    "k_act": _K_ACTIVATION,
                    "w_vec": w_vector,
                    "w_fts": w_fts,
                    "w_act": w_activation,
                    "limit": limit,
                },
            ).fetchall()

        results = []
        for row in rows:
            results.append({
                "id": str(row[0]),
                "score": float(row[1]),
                "memory": row[5],  # Original content (no mem0 compression!)
                "category": row[6],
                "source": row[7],
                "importance": row[8] or "medium",
                "is_semantic": row[9],
                "confidence": float(row[10] or 0.5),
                "evidence_count": row[11] or 0,
                "channel_scores": {
                    "vector": float(row[2]),
                    "fts": float(row[3]),
                    "activation": float(row[4]),
                },
                "activation_score": float(row[13] or 0),
                "created_at": row[20].isoformat() if row[20] else "",
            })

        return {"results": results}

    # -----------------------------------------------------------------------
    # DELETE (replaces mem0.delete)
    # -----------------------------------------------------------------------
    def delete(self, memory_id: str) -> bool:
        """Delete a memory by ID.

        Returns True if deleted, False if not found.
        FTS index is automatically updated (GENERATED ALWAYS column).
        """
        with get_conn() as conn:
            result = conn.execute(
                "DELETE FROM memories WHERE id = %s", (memory_id,)
            )
            return result.rowcount > 0

    def delete_many(self, memory_ids: list) -> int:
        """DISABLED: Bulk delete is too dangerous. Use delete() one by one.
        Disabled after 20,859 memories were lost on 2026-03-03."""
        raise RuntimeError(
            "delete_many() is permanently disabled. "
            "Use delete() for single-record deletion with audit trail."
        )

    def delete_by_filter(self, filters: dict) -> int:
        """DISABLED: Filter-based delete is too dangerous (empty filters = delete all).
        Disabled after 20,859 memories were lost on 2026-03-03."""
        raise RuntimeError(
            "delete_by_filter() is permanently disabled. "
            "Use delete() for single-record deletion with audit trail."
        )

    # -----------------------------------------------------------------------
    # GET BY IDS (replaces qdrant.retrieve)
    # -----------------------------------------------------------------------
    def get_by_ids(
        self,
        ids: list,
        with_vectors: bool = False,
    ) -> list[Point]:
        """Retrieve memories by ID list.

        Returns list of Point objects (Qdrant-compatible).
        """
        if not ids:
            return []

        str_ids = [str(i) for i in ids]
        vec_col = ", embedding" if with_vectors else ""

        with get_conn() as conn:
            with conn.cursor(row_factory=dict_row) as cur:
                rows = cur.execute(
                    f"""
                    SELECT id, content, category, source, importance,
                           is_semantic, confidence, evidence_count,
                           metadata, activation_score, access_count,
                           storage_strength, retrieval_strength,
                           emotion_p, emotion_a, emotion_d,
                           created_at, updated_at, last_accessed_at
                           {vec_col}
                    FROM memories
                    WHERE id = ANY(%s)
                    """,
                    (str_ids,),
                ).fetchall()

        return [_row_to_point(r, with_vectors=with_vectors) for r in rows]

    # -----------------------------------------------------------------------
    # SCROLL (replaces qdrant.scroll)
    # -----------------------------------------------------------------------
    def scroll(
        self,
        filters: Optional[dict] = None,
        limit: int = 50,
        offset: int = 0,
        is_semantic: Optional[bool] = None,
        with_vectors: bool = False,
        order_by: str = "created_at DESC",
    ) -> tuple[list[Point], Optional[int]]:
        """Paginated scroll through memories with filters.

        Returns (list[Point], next_offset_or_None).
        """
        if offset is None:
            offset = 0
        params = []
        where = _build_where(filters, params, is_semantic=is_semantic)
        vec_col = ", embedding" if with_vectors else ""

        # Validate order_by to prevent SQL injection
        _ALLOWED_ORDERS = {
            "created_at DESC", "created_at ASC",
            "activation_score DESC", "updated_at DESC",
            "last_accessed_at DESC", "access_count DESC",
        }
        if order_by not in _ALLOWED_ORDERS:
            order_by = "created_at DESC"

        with get_conn() as conn:
            with conn.cursor(row_factory=dict_row) as cur:
                rows = cur.execute(
                    f"""
                    SELECT id, content, category, source, importance,
                           is_semantic, confidence, evidence_count,
                           metadata, activation_score, access_count,
                           storage_strength, retrieval_strength,
                           emotion_p, emotion_a, emotion_d,
                           created_at, updated_at, last_accessed_at
                           {vec_col}
                    FROM memories
                    WHERE {where}
                    ORDER BY {order_by}
                    LIMIT %s OFFSET %s
                    """,
                    params + [limit + 1, offset],  # +1 to detect more pages
                ).fetchall()

        points = [_row_to_point(r, with_vectors=with_vectors) for r in rows[:limit]]
        next_offset = offset + limit if len(rows) > limit else None
        return points, next_offset

    # -----------------------------------------------------------------------
    # QUERY VECTOR (replaces qdrant.query_points)
    # -----------------------------------------------------------------------
    def query_vector(
        self,
        embedding: list,
        limit: int = 10,
        is_semantic: Optional[bool] = None,
        score_threshold: Optional[float] = None,
        with_vectors: bool = False,
    ) -> list[Point]:
        """Vector similarity search using HNSW halfvec index.

        Returns list of Point objects with .score (cosine similarity 0-1).
        """
        semantic_clause = ""
        if is_semantic is True:
            semantic_clause = "AND is_semantic = TRUE"
        elif is_semantic is False:
            semantic_clause = "AND is_semantic = FALSE"

        threshold_clause = ""
        if score_threshold is not None:
            threshold_clause = f"HAVING 1 - (embedding::halfvec(1536) <=> %s::halfvec(1536)) >= {float(score_threshold)}"

        vec_col = ", m.embedding" if with_vectors else ""

        with get_conn() as conn:
            with conn.cursor(row_factory=dict_row) as cur:
                rows = cur.execute(
                    f"""
                    SELECT m.id, m.content, m.category, m.source, m.importance,
                           m.is_semantic, m.confidence, m.evidence_count,
                           m.metadata, m.activation_score, m.access_count,
                           m.storage_strength, m.retrieval_strength,
                           m.emotion_p, m.emotion_a, m.emotion_d,
                           m.created_at, m.updated_at, m.last_accessed_at,
                           1 - (m.embedding::halfvec(1536) <=> %s::halfvec(1536)) AS score
                           {vec_col}
                    FROM memories m
                    WHERE m.embedding IS NOT NULL {semantic_clause}
                    ORDER BY m.embedding::halfvec(1536) <=> %s::halfvec(1536)
                    LIMIT %s
                    """,
                    (embedding, embedding, limit),
                ).fetchall()

        points = []
        for r in rows:
            p = _row_to_point(r, with_vectors=with_vectors)
            p.score = float(r["score"])
            if score_threshold is not None and p.score < score_threshold:
                continue
            points.append(p)

        return points

    # -----------------------------------------------------------------------
    # UPSERT (replaces qdrant.upsert)
    # -----------------------------------------------------------------------
    def upsert(
        self,
        point_id: str,
        embedding: list,
        payload: dict,
    ) -> str:
        """Upsert a memory point (insert or update on conflict).

        Args:
            point_id: UUID string
            embedding: 1536-dim vector
            payload: Dict with content and metadata fields

        Returns:
            The point ID.
        """
        content = payload.get("data") or payload.get("fact_text") or payload.get("content", "")
        category = payload.get("category", "general")
        source_val = payload.get("source", payload.get("ownership_source", "experienced"))
        importance = payload.get("narrative_importance", payload.get("importance", "medium"))
        is_semantic = payload.get("memory_type") == "semantic" or payload.get("is_semantic", False)
        confidence = float(payload.get("confidence", 0.5))
        evidence_count = int(payload.get("evidence_count", 0))
        activation = float(payload.get("attention_salience", payload.get("activation_score", 0.0)))
        storage_s = float(payload.get("storage_strength", 1.0))
        retrieval_s = float(payload.get("retrieval_strength", 1.0))
        emotion_p = float(payload.get("emotion_p", payload.get("pad_pleasure", 0.0)))
        emotion_a = float(payload.get("emotion_a", payload.get("pad_arousal", 0.0)))
        emotion_d = float(payload.get("emotion_d", payload.get("pad_dominance", 0.0)))

        # Remaining payload fields go into metadata JSONB
        _KNOWN_KEYS = {
            "data", "fact_text", "content", "category", "source", "importance",
            "narrative_importance", "ownership_source", "memory_type", "is_semantic",
            "confidence", "evidence_count", "attention_salience", "activation_score",
            "storage_strength", "retrieval_strength",
            "emotion_p", "emotion_a", "emotion_d",
            "pad_pleasure", "pad_arousal", "pad_dominance",
            "created_at", "updated_at", "user_id",
        }
        metadata = {k: v for k, v in payload.items() if k not in _KNOWN_KEYS}

        now = datetime.now(_COL_TZ)
        created = payload.get("created_at", now.isoformat())

        with get_conn() as conn:
            conn.execute(
                """
                INSERT INTO memories
                    (id, content, category, source, importance, embedding,
                     is_semantic, confidence, evidence_count,
                     activation_score, storage_strength, retrieval_strength,
                     emotion_p, emotion_a, emotion_d,
                     metadata, created_at, updated_at, last_accessed_at)
                VALUES
                    (%s, %s, %s, %s, %s, %s::vector,
                     %s, %s, %s,
                     %s, %s, %s,
                     %s, %s, %s,
                     %s::jsonb, %s, %s, %s)
                ON CONFLICT (id) DO UPDATE SET
                    content = EXCLUDED.content,
                    category = EXCLUDED.category,
                    source = EXCLUDED.source,
                    importance = EXCLUDED.importance,
                    embedding = EXCLUDED.embedding,
                    is_semantic = EXCLUDED.is_semantic,
                    confidence = EXCLUDED.confidence,
                    evidence_count = EXCLUDED.evidence_count,
                    activation_score = EXCLUDED.activation_score,
                    storage_strength = EXCLUDED.storage_strength,
                    retrieval_strength = EXCLUDED.retrieval_strength,
                    emotion_p = EXCLUDED.emotion_p,
                    emotion_a = EXCLUDED.emotion_a,
                    emotion_d = EXCLUDED.emotion_d,
                    metadata = EXCLUDED.metadata,
                    updated_at = EXCLUDED.updated_at
                """,
                (
                    point_id, content, category, source_val, importance, embedding,
                    is_semantic, confidence, evidence_count,
                    activation, storage_s, retrieval_s,
                    emotion_p, emotion_a, emotion_d,
                    psycopg.types.json.Json(metadata),
                    created, now, now,
                ),
            )

        return point_id

    # -----------------------------------------------------------------------
    # UPDATE PAYLOAD (replaces Qdrant set_payload / partial updates)
    # -----------------------------------------------------------------------
    def update_payload(self, memory_id: str, updates: dict) -> bool:
        """Update specific fields of a memory.

        Supports both column fields and JSONB metadata.
        Returns True if updated.
        """
        if not updates:
            return False

        _COLUMN_FIELDS = {
            "activation_score", "storage_strength", "retrieval_strength",
            "emotion_p", "emotion_a", "emotion_d",
            "confidence", "evidence_count", "access_count",
            "category", "source", "importance", "content",
            "last_accessed_at", "updated_at",
        }

        set_parts = []
        params = []
        metadata_updates = {}

        for key, value in updates.items():
            if key in _COLUMN_FIELDS:
                set_parts.append(f"{key} = %s")
                params.append(value)
            else:
                metadata_updates[key] = value

        if metadata_updates:
            set_parts.append("metadata = metadata || %s::jsonb")
            params.append(psycopg.types.json.Json(metadata_updates))

        # Always update updated_at
        if "updated_at" not in updates:
            set_parts.append("updated_at = %s")
            params.append(datetime.now(_COL_TZ))

        params.append(memory_id)
        sql = f"UPDATE memories SET {', '.join(set_parts)} WHERE id = %s"

        with get_conn() as conn:
            result = conn.execute(sql, params)
            return result.rowcount > 0

    # -----------------------------------------------------------------------
    # COUNT (replaces qdrant.get_collection / qdrant.count)
    # -----------------------------------------------------------------------
    def count(self, is_semantic: Optional[bool] = None) -> CollectionInfo:
        """Get total memory count.

        Args:
            is_semantic: None=all, True=semantic only, False=episodic only
        """
        clause = ""
        params = []
        if is_semantic is not None:
            clause = "WHERE is_semantic = %s"
            params = [is_semantic]

        with get_conn() as conn:
            row = conn.execute(
                f"SELECT COUNT(*) FROM memories {clause}", params
            ).fetchone()
            return CollectionInfo(points_count=row[0])

    # -----------------------------------------------------------------------
    # FTS SEARCH (replaces search_fts from memory_smart.py)
    # -----------------------------------------------------------------------
    def search_fts(self, query: str, limit: int = 20) -> list[dict]:
        """Full-text search using PostgreSQL tsvector.

        Searches both English and Spanish dictionaries.
        Returns list of dicts: [{memory_id, content, category, source, bm25_rank}, ...]
        """
        # Sanitize: remove special FTS chars, join with &
        terms = [w for w in query.split() if len(w) > 1]
        tsquery = " & ".join(terms) if terms else query

        with get_conn() as conn:
            rows = conn.execute(
                """
                SELECT id AS memory_id, content, category, source, importance,
                       ts_rank(fts_combined, to_tsquery('english', %s)) AS bm25_rank
                FROM memories
                WHERE fts_combined @@ to_tsquery('english', %s)
                   OR fts_combined @@ to_tsquery('spanish', %s)
                ORDER BY bm25_rank DESC
                LIMIT %s
                """,
                (tsquery, tsquery, tsquery, limit),
            ).fetchall()

        return [
            {
                "memory_id": str(row[0]),
                "content": row[1],
                "category": row[2],
                "source": row[3],
                "importance": row[4],
                "bm25_rank": float(row[5]),
            }
            for row in rows
        ]

    # -----------------------------------------------------------------------
    # ACCESS TRACKING
    # -----------------------------------------------------------------------
    def record_access(self, memory_id: str, context: str = "") -> None:
        """Record an access event and update counters."""
        with get_conn() as conn:
            now = datetime.now(_COL_TZ)
            conn.execute(
                """
                UPDATE memories SET
                    access_count = access_count + 1,
                    last_accessed_at = %s
                WHERE id = %s
                """,
                (now, memory_id),
            )
            conn.execute(
                """
                INSERT INTO access_history (memory_id, accessed_at, context)
                VALUES (%s, %s, %s)
                """,
                (memory_id, now, context[:200] if context else None),
            )

    # -----------------------------------------------------------------------
    # BATCH OPERATIONS
    # -----------------------------------------------------------------------
    def get_all(self, limit: int = 500, is_semantic: Optional[bool] = None) -> list[Point]:
        """Get all memories (paginated). Replaces memory.get_all()."""
        points, _ = self.scroll(
            is_semantic=is_semantic,
            limit=limit,
            order_by="created_at DESC",
        )
        return points

    def search_by_similarity(self, content: str, limit: int = 5) -> list[Point]:
        """Search by content similarity (convenience wrapper)."""
        embedding = _embed_text(content)
        return self.query_vector(embedding, limit=limit)


# ---------------------------------------------------------------------------
# Global singleton
# ---------------------------------------------------------------------------
pg = PGMemoryStore()
