#!/usr/bin/env python3
"""
migrate_to_pg.py - Migrate codi-memory from Qdrant+SQLite to PostgreSQL+pgvector.
===================================================================================

Migrates:
  1. Qdrant `codi_memories`  → PostgreSQL `memories` (is_semantic=FALSE)
  2. Qdrant `codi_semantic`  → PostgreSQL `memories` (is_semantic=TRUE)
  3. SQLite memories_fts.db  → PostgreSQL auxiliary tables
  4. SQLite prospective.db   → PostgreSQL goals/intentions tables

Usage:
  python3 scripts/migrate_to_pg.py
  python3 scripts/migrate_to_pg.py --dry-run
  python3 scripts/migrate_to_pg.py --skip-qdrant
  python3 scripts/migrate_to_pg.py --skip-sqlite
  python3 scripts/migrate_to_pg.py --skip-qdrant --skip-sqlite  # validation only
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sqlite3
import sys
import uuid
from datetime import datetime, timezone, timedelta
from typing import Any, Optional

# ---------------------------------------------------------------------------
# Bootstrap: ensure we can import from the project root
# ---------------------------------------------------------------------------
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# Load .env manually (avoid importing config.py which has heavy deps)
_ENV_PATH = os.path.join(_PROJECT_ROOT, ".env")
if os.path.exists(_ENV_PATH):
    with open(_ENV_PATH) as _f:
        for _line in _f:
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _k, _v = _line.split("=", 1)
                os.environ.setdefault(_k.strip(), _v.strip().strip('"').strip("'"))

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("migrate_to_pg")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
COLLECTION_NAME = "codi_memories"
SEMANTIC_COLLECTION = "codi_semantic"

FTS_DB_PATH = os.path.join(_PROJECT_ROOT, "memories_fts.db")
PROSPECTIVE_DB_PATH = os.path.join(_PROJECT_ROOT, "prospective.db")

BATCH_SIZE = 50       # Insert rows per transaction
SCROLL_LIMIT = 100    # Qdrant scroll page size
PROGRESS_EVERY = 100  # Log progress every N rows

# Colombia timezone
_COL_TZ = timezone(timedelta(hours=-5))

# PostgreSQL UUID validity helper
def _is_valid_uuid(val: Any) -> bool:
    try:
        uuid.UUID(str(val))
        return True
    except (ValueError, AttributeError):
        return False


def _safe_uuid(val: Any) -> Optional[str]:
    """Return a UUID string or None if unparseable."""
    try:
        return str(uuid.UUID(str(val)))
    except (ValueError, AttributeError):
        return None


def _parse_ts(val: Any) -> Optional[str]:
    """
    Normalize a timestamp value to ISO 8601 with timezone.
    Handles: None, empty string, datetime objects, ISO strings.
    Returns None if unparseable.
    """
    if val is None or val == "":
        return None
    if isinstance(val, datetime):
        if val.tzinfo is None:
            val = val.replace(tzinfo=_COL_TZ)
        return val.isoformat()
    if isinstance(val, str):
        # Already an ISO string — pass through (PostgreSQL accepts these)
        return val
    return None


def _safe_float(val: Any, default: float = 0.0) -> float:
    try:
        return float(val) if val is not None else default
    except (TypeError, ValueError):
        return default


def _safe_int(val: Any, default: int = 0) -> int:
    try:
        return int(val) if val is not None else default
    except (TypeError, ValueError):
        return default


def _safe_json_loads(val: Any, default: Any = None) -> Any:
    """Parse JSON from SQLite TEXT columns."""
    if val is None:
        return default
    if isinstance(val, (dict, list)):
        return val
    try:
        return json.loads(val)
    except (json.JSONDecodeError, TypeError):
        return default


# ---------------------------------------------------------------------------
# Qdrant connection
# ---------------------------------------------------------------------------
def _get_qdrant():
    from qdrant_client import QdrantClient

    url = os.environ.get("QDRANT_URL", "").strip()
    api_key = os.environ.get("QDRANT_API_KEY", "").strip() or None
    if not url:
        raise RuntimeError(
            "QDRANT_URL not set. Check .env or set QDRANT_URL env var."
        )
    client = QdrantClient(url=url, api_key=api_key, timeout=60)
    logger.info("Qdrant connected: %s", url)
    return client


# ---------------------------------------------------------------------------
# PostgreSQL connection (from config_pg.py)
# ---------------------------------------------------------------------------
def _get_pg_conn():
    from modules.config_pg import get_conn
    return get_conn


# ---------------------------------------------------------------------------
# Helper: batch insert into PostgreSQL with ON CONFLICT DO NOTHING
# ---------------------------------------------------------------------------
def _batch_insert(
    pg_conn_ctx,
    table: str,
    columns: list[str],
    rows: list[tuple],
    conflict_action: str = "DO NOTHING",
    dry_run: bool = False,
) -> int:
    """
    Insert rows into a PostgreSQL table in a single transaction.

    Returns the number of rows inserted (0 for dry_run).
    Uses ON CONFLICT (id) DO NOTHING by default for idempotency.
    """
    if not rows:
        return 0
    if dry_run:
        logger.debug("[DRY-RUN] Would insert %d rows into %s", len(rows), table)
        return 0

    placeholders = ", ".join(["%s"] * len(columns))
    col_list = ", ".join(columns)
    sql = (
        f"INSERT INTO {table} ({col_list}) VALUES ({placeholders})"
        f" ON CONFLICT {conflict_action}"
    )

    inserted = 0
    try:
        with pg_conn_ctx() as conn:
            with conn.transaction():
                for row in rows:
                    try:
                        result = conn.execute(sql, row)
                        inserted += result.rowcount
                    except Exception as e:
                        logger.warning(
                            "Row insert error in %s: %s | row preview: %s",
                            table,
                            e,
                            str(row)[:120],
                        )
    except Exception as e:
        logger.error("Batch insert failed for %s: %s", table, e)

    return inserted


# ===========================================================================
# MIGRATION 1: Qdrant codi_memories → PostgreSQL memories (episodic)
# ===========================================================================

# Known payload fields that map directly to memory columns (excluded from metadata)
_KNOWN_QDRANT_KEYS = {
    "data", "fact_text", "content",
    "category",
    "source", "ownership_source",
    "importance", "narrative_importance",
    "confidence", "memory_confidence",
    "evidence_count",
    "attention_salience", "activation_score",
    "storage_strength", "retrieval_strength",
    "emotion_p", "emotion_a", "emotion_d",
    "pad_at_encoding",
    "memory_type", "is_semantic",
    "user_id", "hash",
    "created_at", "updated_at",
    "_v",
}


def _extract_emotion(payload: dict) -> tuple[float, float, float]:
    """Extract PAD emotion values from Qdrant payload.

    Checks pad_at_encoding dict first (keys 'P','A','D' or 'p','a','d'),
    then falls back to individual emotion_p/emotion_a/emotion_d fields.
    """
    pad = payload.get("pad_at_encoding")
    if isinstance(pad, dict):
        p = _safe_float(pad.get("P", pad.get("p", pad.get("pleasure", 0.0))))
        a = _safe_float(pad.get("A", pad.get("a", pad.get("arousal", 0.0))))
        d = _safe_float(pad.get("D", pad.get("d", pad.get("dominance", 0.0))))
        return p, a, d
    p = _safe_float(payload.get("emotion_p", payload.get("pad_pleasure", 0.0)))
    a = _safe_float(payload.get("emotion_a", payload.get("pad_arousal", 0.0)))
    d = _safe_float(payload.get("emotion_d", payload.get("pad_dominance", 0.0)))
    return p, a, d


def _qdrant_point_to_pg_row(
    pt_id: Any,
    payload: dict,
    vector: Optional[list],
    is_semantic: bool,
) -> tuple:
    """
    Convert a Qdrant point to a PostgreSQL memories row tuple.

    Returns:
        (id, content, category, source, importance, is_semantic,
         confidence, evidence_count, embedding,
         activation_score, storage_strength, retrieval_strength,
         emotion_p, emotion_a, emotion_d,
         metadata, created_at, updated_at, last_accessed_at, access_count)
    """
    import psycopg

    # ID — must be a valid UUID string
    mem_id = _safe_uuid(pt_id)
    if mem_id is None:
        # Generate a deterministic UUID from the original ID
        mem_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, str(pt_id)))

    # Content
    content = (
        payload.get("data")
        or payload.get("fact_text")
        or payload.get("content")
        or ""
    )
    if not content:
        content = "[empty]"

    # Category
    category = str(payload.get("category", "general"))[:50]

    # Source
    source = str(
        payload.get("source")
        or payload.get("ownership_source")
        or "experienced"
    )[:20]

    # Importance
    importance = str(
        payload.get("narrative_importance")
        or payload.get("importance")
        or "medium"
    )[:10]
    if importance not in ("critical", "high", "medium", "low"):
        importance = "medium"

    # Confidence
    confidence = _safe_float(
        payload.get("confidence", payload.get("memory_confidence", 0.5)),
        default=0.5,
    )

    # Evidence count
    evidence_count = _safe_int(payload.get("evidence_count", 0))

    # Embedding — list of floats or None
    embedding = vector  # already list[float] or None

    # Activation score
    activation_score = _safe_float(
        payload.get("attention_salience", payload.get("activation_score", 0.0))
    )

    # Strength
    storage_strength = _safe_float(payload.get("storage_strength", 1.0), default=1.0)
    retrieval_strength = _safe_float(payload.get("retrieval_strength", 1.0), default=1.0)

    # Emotion PAD
    emotion_p, emotion_a, emotion_d = _extract_emotion(payload)

    # Timestamps
    created_at = _parse_ts(payload.get("created_at")) or datetime.now(_COL_TZ).isoformat()
    updated_at = _parse_ts(payload.get("updated_at") or payload.get("consolidated_at")) or created_at
    last_accessed = _parse_ts(payload.get("attention_last_accessed")) or created_at
    access_count = _safe_int(payload.get("attention_access_count", 0))

    # Metadata JSONB — all remaining payload fields
    metadata = {
        k: v for k, v in payload.items()
        if k not in _KNOWN_QDRANT_KEYS
    }

    return (
        mem_id,
        content,
        category,
        source,
        importance,
        is_semantic,
        confidence,
        evidence_count,
        embedding,
        activation_score,
        storage_strength,
        retrieval_strength,
        emotion_p,
        emotion_a,
        emotion_d,
        psycopg.types.json.Json(metadata),
        created_at,
        updated_at,
        last_accessed,
        access_count,
    )


_MEMORIES_COLUMNS = [
    "id", "content", "category", "source", "importance",
    "is_semantic", "confidence", "evidence_count",
    "embedding",
    "activation_score", "storage_strength", "retrieval_strength",
    "emotion_p", "emotion_a", "emotion_d",
    "metadata",
    "created_at", "updated_at", "last_accessed_at", "access_count",
]

_MEMORIES_CONFLICT = "(id) DO NOTHING"


def migrate_qdrant_memories(dry_run: bool = False) -> tuple[int, int]:
    """
    Migrate Qdrant `codi_memories` → PostgreSQL `memories` (is_semantic=FALSE).

    Returns (scanned, inserted).
    """
    logger.info("=== Migrating Qdrant codi_memories (episodic) ===")
    qdrant = _get_qdrant()
    pg_conn_ctx = _get_pg_conn()

    from pgvector.psycopg import register_vector
    import psycopg

    scanned = 0
    inserted = 0
    offset = None
    page_num = 0

    while True:
        try:
            scroll_kwargs = dict(
                collection_name=COLLECTION_NAME,
                limit=SCROLL_LIMIT,
                with_payload=True,
                with_vectors=True,
            )
            if offset is not None:
                scroll_kwargs["offset"] = offset

            points, next_offset = qdrant.scroll(**scroll_kwargs)
        except Exception as e:
            logger.error("Qdrant scroll error at offset %s: %s", offset, e)
            break

        if not points:
            break

        page_num += 1
        batch_rows = []

        for pt in points:
            scanned += 1
            try:
                row = _qdrant_point_to_pg_row(
                    pt_id=pt.id,
                    payload=pt.payload or {},
                    vector=pt.vector,
                    is_semantic=False,
                )
                batch_rows.append(row)
            except Exception as e:
                logger.warning("Error converting episodic point %s: %s", pt.id, e)

            if scanned % PROGRESS_EVERY == 0:
                logger.info("  [episodic] scanned=%d inserted=%d", scanned, inserted)

        # Insert batch
        n = _batch_insert(
            pg_conn_ctx,
            table="memories",
            columns=_MEMORIES_COLUMNS,
            rows=batch_rows,
            conflict_action=_MEMORIES_CONFLICT,
            dry_run=dry_run,
        )
        inserted += n

        if next_offset is None:
            break
        offset = next_offset

    logger.info(
        "[episodic] DONE — scanned=%d, inserted=%d (dry_run=%s)",
        scanned, inserted, dry_run,
    )
    return scanned, inserted


def migrate_qdrant_semantic(dry_run: bool = False) -> tuple[int, int]:
    """
    Migrate Qdrant `codi_semantic` → PostgreSQL `memories` (is_semantic=TRUE).

    Returns (scanned, inserted).
    """
    logger.info("=== Migrating Qdrant codi_semantic (semantic facts) ===")
    qdrant = _get_qdrant()
    pg_conn_ctx = _get_pg_conn()

    import psycopg

    scanned = 0
    inserted = 0
    offset = None

    while True:
        try:
            scroll_kwargs = dict(
                collection_name=SEMANTIC_COLLECTION,
                limit=SCROLL_LIMIT,
                with_payload=True,
                with_vectors=True,
            )
            if offset is not None:
                scroll_kwargs["offset"] = offset

            points, next_offset = qdrant.scroll(**scroll_kwargs)
        except Exception as e:
            logger.error("Qdrant semantic scroll error at offset %s: %s", offset, e)
            break

        if not points:
            break

        batch_rows = []

        for pt in points:
            scanned += 1
            try:
                row = _qdrant_point_to_pg_row(
                    pt_id=pt.id,
                    payload=pt.payload or {},
                    vector=pt.vector,
                    is_semantic=True,
                )
                batch_rows.append(row)
            except Exception as e:
                logger.warning("Error converting semantic point %s: %s", pt.id, e)

            if scanned % PROGRESS_EVERY == 0:
                logger.info("  [semantic] scanned=%d inserted=%d", scanned, inserted)

        n = _batch_insert(
            pg_conn_ctx,
            table="memories",
            columns=_MEMORIES_COLUMNS,
            rows=batch_rows,
            conflict_action=_MEMORIES_CONFLICT,
            dry_run=dry_run,
        )
        inserted += n

        if next_offset is None:
            break
        offset = next_offset

    logger.info(
        "[semantic] DONE — scanned=%d, inserted=%d (dry_run=%s)",
        scanned, inserted, dry_run,
    )
    return scanned, inserted


# ===========================================================================
# MIGRATION 2: SQLite auxiliary tables
# ===========================================================================

# ---------------------------------------------------------------------------
# Table-specific migration functions
# ---------------------------------------------------------------------------

def _migrate_working_memory(sqlite_conn: sqlite3.Connection, pg_conn_ctx, dry_run: bool) -> tuple[int, int]:
    """
    SQLite working_memory → PostgreSQL working_memory.

    Column differences:
      SQLite: id, content, topic, relevance, added_at, occurred_at, source,
              related_memory_id, chain_id, active, last_accessed_at, access_count
      PG:     id (SERIAL), content, topic, relevance, chain_id, active, source,
              related_memory_id (UUID), occurred_at, created_at, last_accessed_at, access_count
    """
    import psycopg

    rows_sql = sqlite_conn.execute(
        "SELECT id, content, topic, relevance, added_at, occurred_at, source, "
        "       related_memory_id, chain_id, active, last_accessed_at, access_count "
        "FROM working_memory"
    ).fetchall()

    pg_columns = [
        "content", "topic", "relevance", "chain_id", "active", "source",
        "related_memory_id", "occurred_at", "created_at", "last_accessed_at", "access_count",
    ]

    batch_rows = []
    for r in rows_sql:
        (row_id, content, topic, relevance, added_at, occurred_at, source,
         related_mem_id, chain_id, active, last_accessed, access_count) = r

        # related_memory_id must be a valid UUID or NULL
        rel_mem_uuid = _safe_uuid(related_mem_id) if related_mem_id else None

        batch_rows.append((
            content or "",
            (topic or "general")[:100],
            _safe_float(relevance, 0.5),
            chain_id[:100] if chain_id else None,
            bool(active) if active is not None else True,
            (source or "interaction")[:50],
            rel_mem_uuid,
            _parse_ts(occurred_at),
            _parse_ts(added_at),
            _parse_ts(last_accessed),
            _safe_int(access_count, 0),
        ))

    # working_memory has SERIAL PK so no ON CONFLICT on id
    inserted = _batch_insert(
        pg_conn_ctx, "working_memory", pg_columns, batch_rows,
        conflict_action="DO NOTHING",
        dry_run=dry_run,
    )
    return len(rows_sql), inserted


def _migrate_narrative_traces(sqlite_conn: sqlite3.Connection, pg_conn_ctx, dry_run: bool) -> tuple[int, int]:
    """
    SQLite narrative_traces → PostgreSQL narrative_traces.

    Both have: trace_name UNIQUE, chain_ids (TEXT/JSONB), theme, active, created_at, last_updated
    """
    import psycopg

    rows_sql = sqlite_conn.execute(
        "SELECT id, trace_name, chain_ids, theme, created_at, last_updated, active "
        "FROM narrative_traces"
    ).fetchall()

    pg_columns = ["trace_name", "chain_ids", "theme", "active", "created_at", "last_updated"]

    batch_rows = []
    for r in rows_sql:
        (row_id, trace_name, chain_ids, theme, created_at, last_updated, active) = r

        chain_ids_json = _safe_json_loads(chain_ids, default=[])

        batch_rows.append((
            trace_name,
            psycopg.types.json.Json(chain_ids_json),
            theme,
            bool(active) if active is not None else True,
            _parse_ts(created_at),
            _parse_ts(last_updated),
        ))

    inserted = _batch_insert(
        pg_conn_ctx, "narrative_traces", pg_columns, batch_rows,
        conflict_action="(trace_name) DO NOTHING",
        dry_run=dry_run,
    )
    return len(rows_sql), inserted


def _migrate_spreading_edges(sqlite_conn: sqlite3.Connection, pg_conn_ctx, dry_run: bool) -> tuple[int, int]:
    """
    SQLite spreading_edges → PostgreSQL spreading_edges.

    SQLite: from_id, to_id, edge_type, last_seen, directed, strength, discovery_source
    PG:     id (SERIAL), from_id UUID, to_id UUID, edge_type, strength, last_seen,
            UNIQUE(from_id, to_id, edge_type)
    """
    rows_sql = sqlite_conn.execute(
        "SELECT from_id, to_id, edge_type, last_seen, strength "
        "FROM spreading_edges"
    ).fetchall()

    pg_columns = ["from_id", "to_id", "edge_type", "strength", "last_seen"]

    batch_rows = []
    skipped = 0
    for r in rows_sql:
        from_id, to_id, edge_type, last_seen, strength = r

        from_uuid = _safe_uuid(from_id)
        to_uuid = _safe_uuid(to_id)
        if from_uuid is None or to_uuid is None:
            skipped += 1
            continue

        batch_rows.append((
            from_uuid,
            to_uuid,
            (edge_type or "co_occurs")[:50],
            _safe_float(strength, 1.0),
            _parse_ts(last_seen),
        ))

    if skipped:
        logger.warning("  spreading_edges: skipped %d rows with invalid UUIDs", skipped)

    inserted = _batch_insert(
        pg_conn_ctx, "spreading_edges", pg_columns, batch_rows,
        conflict_action="(from_id, to_id, edge_type) DO NOTHING",
        dry_run=dry_run,
    )
    return len(rows_sql), inserted


def _migrate_consolidation_log(sqlite_conn: sqlite3.Connection, pg_conn_ctx, dry_run: bool) -> tuple[int, int]:
    rows_sql = sqlite_conn.execute(
        "SELECT batch_id, scope, lookback_hours, episodes_scanned, clusters_found, "
        "       facts_extracted, facts_created, facts_updated, contradictions_found, "
        "       episodes_pruned, duration_ms, created_at "
        "FROM consolidation_log"
    ).fetchall()

    pg_columns = [
        "batch_id", "scope", "lookback_hours", "episodes_scanned", "clusters_found",
        "facts_extracted", "facts_created", "facts_updated", "contradictions_found",
        "episodes_pruned", "duration_ms", "created_at",
    ]

    batch_rows = []
    for r in rows_sql:
        (batch_id, scope, lookback_hours, episodes_scanned, clusters_found,
         facts_extracted, facts_created, facts_updated, contradictions_found,
         episodes_pruned, duration_ms, created_at) = r
        batch_rows.append((
            batch_id, scope[:50] if scope else None, _safe_int(lookback_hours),
            _safe_int(episodes_scanned), _safe_int(clusters_found),
            _safe_int(facts_extracted), _safe_int(facts_created), _safe_int(facts_updated),
            _safe_int(contradictions_found), _safe_int(episodes_pruned),
            _safe_int(duration_ms), _parse_ts(created_at),
        ))

    inserted = _batch_insert(
        pg_conn_ctx, "consolidation_log", pg_columns, batch_rows,
        conflict_action="(batch_id) DO NOTHING",
        dry_run=dry_run,
    )
    return len(rows_sql), inserted


def _migrate_reconsolidation_log(sqlite_conn: sqlite3.Connection, pg_conn_ctx, dry_run: bool) -> tuple[int, int]:
    rows_sql = sqlite_conn.execute(
        "SELECT memory_id, memory_type, action, prediction_error, memory_strength, "
        "       old_content, new_content, blend_weight, trigger_context, created_at "
        "FROM reconsolidation_log"
    ).fetchall()

    pg_columns = [
        "memory_id", "memory_type", "action", "prediction_error", "memory_strength",
        "old_content", "new_content", "blend_weight", "trigger_context", "created_at",
    ]

    batch_rows = []
    for r in rows_sql:
        (memory_id, memory_type, action, prediction_error, memory_strength,
         old_content, new_content, blend_weight, trigger_context, created_at) = r
        batch_rows.append((
            _safe_uuid(memory_id),
            memory_type[:20] if memory_type else None,
            action[:50] if action else None,
            _safe_float(prediction_error),
            _safe_float(memory_strength),
            old_content, new_content,
            _safe_float(blend_weight),
            trigger_context, _parse_ts(created_at),
        ))

    # No unique constraint on reconsolidation_log — use DO NOTHING with no conflict target
    inserted = _batch_insert(
        pg_conn_ctx, "reconsolidation_log", pg_columns, batch_rows,
        conflict_action="DO NOTHING",
        dry_run=dry_run,
    )
    return len(rows_sql), inserted


def _migrate_prediction_state(sqlite_conn: sqlite3.Connection, pg_conn_ctx, dry_run: bool) -> tuple[int, int]:
    """Migrate prediction_state (L0 only — the main level)."""
    import psycopg

    rows_sql = sqlite_conn.execute(
        "SELECT id, predicted_topic, predicted_keywords, predicted_action, "
        "       topic_distribution, confidence, created_at "
        "FROM prediction_state"
    ).fetchall()

    pg_columns = [
        "predicted_topic", "predicted_keywords", "predicted_action",
        "topic_distribution", "confidence", "created_at",
    ]

    batch_rows = []
    for r in rows_sql:
        (row_id, predicted_topic, predicted_keywords, predicted_action,
         topic_distribution, confidence, created_at) = r

        dist_json = _safe_json_loads(topic_distribution, default={})
        batch_rows.append((
            predicted_topic[:100] if predicted_topic else None,
            predicted_keywords,
            predicted_action[:100] if predicted_action else None,
            psycopg.types.json.Json(dist_json),
            _safe_float(confidence, 0.5),
            _parse_ts(created_at),
        ))

    inserted = _batch_insert(
        pg_conn_ctx, "prediction_state", pg_columns, batch_rows,
        conflict_action="DO NOTHING",
        dry_run=dry_run,
    )
    return len(rows_sql), inserted


def _migrate_prediction_results(sqlite_conn: sqlite3.Connection, pg_conn_ctx, dry_run: bool) -> tuple[int, int]:
    """Migrate prediction_results (L0 only)."""
    rows_sql = sqlite_conn.execute(
        "SELECT id, predicted_topic, actual_topic, predicted_keywords, actual_keywords, "
        "       surprise_score, precision_weight, weighted_surprise, hit, created_at "
        "FROM prediction_results"
    ).fetchall()

    pg_columns = [
        "predicted_topic", "actual_topic", "predicted_keywords", "actual_keywords",
        "surprise_score", "precision_weight", "weighted_surprise", "hit", "created_at",
    ]

    batch_rows = []
    for r in rows_sql:
        (row_id, predicted_topic, actual_topic, predicted_keywords, actual_keywords,
         surprise_score, precision_weight, weighted_surprise, hit, created_at) = r
        batch_rows.append((
            predicted_topic[:100] if predicted_topic else None,
            actual_topic[:100] if actual_topic else None,
            predicted_keywords, actual_keywords,
            _safe_float(surprise_score),
            _safe_float(precision_weight),
            _safe_float(weighted_surprise),
            bool(hit) if hit is not None else False,
            _parse_ts(created_at),
        ))

    inserted = _batch_insert(
        pg_conn_ctx, "prediction_results", pg_columns, batch_rows,
        conflict_action="DO NOTHING",
        dry_run=dry_run,
    )
    return len(rows_sql), inserted


def _migrate_write_queue(sqlite_conn: sqlite3.Connection, pg_conn_ctx, dry_run: bool) -> tuple[int, int]:
    rows_sql = sqlite_conn.execute(
        "SELECT job_id, kind, payload_json, status, priority, attempts, max_attempts, "
        "       lease_until, last_error, dedupe_key, cancel_requested, canceled_at, "
        "       cancel_reason, created_at, updated_at, completed_at, claimed_at, started_at "
        "FROM write_queue"
    ).fetchall()

    pg_columns = [
        "job_id", "kind", "payload_json", "status", "priority", "attempts", "max_attempts",
        "lease_until", "last_error", "dedupe_key", "cancel_requested", "canceled_at",
        "cancel_reason", "created_at", "updated_at", "completed_at", "claimed_at", "started_at",
    ]

    batch_rows = []
    for r in rows_sql:
        (job_id, kind, payload_json, status, priority, attempts, max_attempts,
         lease_until, last_error, dedupe_key, cancel_requested, canceled_at,
         cancel_reason, created_at, updated_at, completed_at, claimed_at, started_at) = r
        batch_rows.append((
            job_id,
            kind[:50] if kind else "unknown",
            payload_json,
            (status or "queued")[:20],
            _safe_int(priority, 0),
            _safe_int(attempts, 0),
            _safe_int(max_attempts, 3),
            _parse_ts(lease_until),
            last_error,
            dedupe_key[:200] if dedupe_key else None,
            bool(cancel_requested) if cancel_requested is not None else False,
            _parse_ts(canceled_at),
            cancel_reason,
            _parse_ts(created_at),
            _parse_ts(updated_at),
            _parse_ts(completed_at),
            _parse_ts(claimed_at),
            _parse_ts(started_at),
        ))

    inserted = _batch_insert(
        pg_conn_ctx, "write_queue", pg_columns, batch_rows,
        conflict_action="(job_id) DO NOTHING",
        dry_run=dry_run,
    )
    return len(rows_sql), inserted


def _migrate_event_counts(sqlite_conn: sqlite3.Connection, pg_conn_ctx, dry_run: bool) -> tuple[int, int]:
    rows_sql = sqlite_conn.execute(
        "SELECT event, count, last_seen FROM event_counts"
    ).fetchall()

    pg_columns = ["event", "count", "last_seen"]

    batch_rows = []
    for r in rows_sql:
        event, count, last_seen = r
        batch_rows.append((
            event[:100] if event else "unknown",
            _safe_int(count, 0),
            _parse_ts(last_seen),
        ))

    inserted = _batch_insert(
        pg_conn_ctx, "event_counts", pg_columns, batch_rows,
        conflict_action="(event) DO NOTHING",
        dry_run=dry_run,
    )
    return len(rows_sql), inserted


def _migrate_tool_calls(sqlite_conn: sqlite3.Connection, pg_conn_ctx, dry_run: bool) -> tuple[int, int]:
    """
    tool_calls can be very large (98k+ rows). Migrate in pages to avoid memory issues.
    """
    PAGE = 1000
    scanned = 0
    inserted_total = 0
    offset = 0

    pg_columns = [
        "tool_name", "started_at", "duration_ms", "success",
        "error_type", "args_size", "result_size", "session_id", "tag",
    ]

    while True:
        rows_sql = sqlite_conn.execute(
            "SELECT tool_name, started_at, duration_ms, success, "
            "       error_type, args_size, result_size, session_id, tag "
            "FROM tool_calls "
            "ORDER BY id "
            f"LIMIT {PAGE} OFFSET {offset}"
        ).fetchall()

        if not rows_sql:
            break

        batch_rows = []
        for r in rows_sql:
            (tool_name, started_at, duration_ms, success,
             error_type, args_size, result_size, session_id, tag) = r
            batch_rows.append((
                tool_name[:100] if tool_name else None,
                _parse_ts(started_at),
                _safe_int(duration_ms),
                bool(success) if success is not None else None,
                error_type[:100] if error_type else None,
                _safe_int(args_size),
                _safe_int(result_size),
                session_id[:100] if session_id else None,
                tag[:100] if tag else None,
            ))

        n = _batch_insert(
            pg_conn_ctx, "tool_calls", pg_columns, batch_rows,
            conflict_action="DO NOTHING",
            dry_run=dry_run,
        )
        scanned += len(rows_sql)
        inserted_total += n
        offset += PAGE

        if scanned % (PAGE * 10) == 0:
            logger.info("  [tool_calls] scanned=%d inserted=%d", scanned, inserted_total)

        if len(rows_sql) < PAGE:
            break

    return scanned, inserted_total


def _migrate_session_checkpoints(sqlite_conn: sqlite3.Connection, pg_conn_ctx, dry_run: bool) -> tuple[int, int]:
    import psycopg

    rows_sql = sqlite_conn.execute(
        "SELECT trace_id, session_id, source, goal_stack, active_project, "
        "       session_summary, attention_focus, attention_strength, "
        "       last_prediction_topic, last_prediction_confidence, "
        "       recent_prediction_errors, wm_items, wm_active_count, "
        "       pad_pleasure, pad_arousal, pad_dominance, pad_trigger, "
        "       last_decisions, pending_intentions, active_traces, "
        "       session_duration_minutes, sleep_report, created_at "
        "FROM session_checkpoints"
    ).fetchall()

    pg_columns = [
        "trace_id", "session_id", "source", "goal_stack", "active_project",
        "session_summary", "attention_focus", "attention_strength",
        "last_prediction_topic", "last_prediction_confidence",
        "recent_prediction_errors", "wm_items", "wm_active_count",
        "pad_pleasure", "pad_arousal", "pad_dominance", "pad_trigger",
        "last_decisions", "pending_intentions", "active_traces",
        "session_duration_minutes", "sleep_report", "created_at",
    ]

    batch_rows = []
    for r in rows_sql:
        (trace_id, session_id, source, goal_stack, active_project,
         session_summary, attention_focus, attention_strength,
         last_pred_topic, last_pred_conf,
         recent_pred_errors, wm_items, wm_active_count,
         pad_p, pad_a, pad_d, pad_trigger,
         last_decisions, pending_intentions, active_traces,
         session_duration, sleep_report, created_at) = r

        batch_rows.append((
            trace_id[:100] if trace_id else None,
            session_id[:100] if session_id else None,
            source[:20] if source else None,
            psycopg.types.json.Json(_safe_json_loads(goal_stack, [])),
            active_project[:200] if active_project else None,
            session_summary,
            attention_focus[:100] if attention_focus else None,
            _safe_float(attention_strength),
            last_pred_topic[:100] if last_pred_topic else None,
            _safe_float(last_pred_conf),
            psycopg.types.json.Json(_safe_json_loads(recent_pred_errors, [])),
            psycopg.types.json.Json(_safe_json_loads(wm_items, [])),
            _safe_int(wm_active_count),
            _safe_float(pad_p),
            _safe_float(pad_a),
            _safe_float(pad_d),
            pad_trigger[:200] if pad_trigger else None,
            psycopg.types.json.Json(_safe_json_loads(last_decisions, [])),
            psycopg.types.json.Json(_safe_json_loads(pending_intentions, [])),
            psycopg.types.json.Json(_safe_json_loads(active_traces, [])),
            _safe_float(session_duration),
            sleep_report,
            _parse_ts(created_at),
        ))

    inserted = _batch_insert(
        pg_conn_ctx, "session_checkpoints", pg_columns, batch_rows,
        conflict_action="DO NOTHING",
        dry_run=dry_run,
    )
    return len(rows_sql), inserted


def _migrate_sleep_loop_state(sqlite_conn: sqlite3.Connection, pg_conn_ctx, dry_run: bool) -> tuple[int, int]:
    rows_sql = sqlite_conn.execute(
        "SELECT key, value FROM sleep_loop_state"
    ).fetchall()

    pg_columns = ["key", "value"]

    batch_rows = [
        (r[0][:100], r[1])
        for r in rows_sql
    ]

    inserted = _batch_insert(
        pg_conn_ctx, "sleep_loop_state", pg_columns, batch_rows,
        conflict_action="(key) DO NOTHING",
        dry_run=dry_run,
    )
    return len(rows_sql), inserted


def _migrate_fok_priors(sqlite_conn: sqlite3.Connection, pg_conn_ctx, dry_run: bool) -> tuple[int, int]:
    rows_sql = sqlite_conn.execute(
        "SELECT topic, alpha, beta, n_observations, last_updated FROM fok_priors"
    ).fetchall()

    pg_columns = ["topic", "alpha", "beta", "n_observations", "last_updated"]

    batch_rows = []
    for r in rows_sql:
        topic, alpha, beta, n_observations, last_updated = r
        batch_rows.append((
            topic[:100] if topic else "general",
            _safe_float(alpha, 1.0),
            _safe_float(beta, 1.0),
            _safe_int(n_observations),
            _parse_ts(last_updated),
        ))

    inserted = _batch_insert(
        pg_conn_ctx, "fok_priors", pg_columns, batch_rows,
        conflict_action="(topic) DO NOTHING",
        dry_run=dry_run,
    )
    return len(rows_sql), inserted


def _migrate_security_audit_log(sqlite_conn: sqlite3.Connection, pg_conn_ctx, dry_run: bool) -> tuple[int, int]:
    import psycopg

    rows_sql = sqlite_conn.execute(
        "SELECT event_type, tool_name, actor, payload_fingerprint, "
        "       details_json, outcome, created_at "
        "FROM security_audit_log"
    ).fetchall()

    pg_columns = [
        "event_type", "tool_name", "actor", "payload_fingerprint",
        "details_json", "outcome", "created_at",
    ]

    batch_rows = []
    for r in rows_sql:
        event_type, tool_name, actor, payload_fingerprint, details_json, outcome, created_at = r
        batch_rows.append((
            event_type[:50] if event_type else None,
            tool_name[:100] if tool_name else None,
            actor[:100] if actor else None,
            payload_fingerprint[:200] if payload_fingerprint else None,
            psycopg.types.json.Json(_safe_json_loads(details_json, {})),
            outcome[:20] if outcome else None,
            _parse_ts(created_at),
        ))

    inserted = _batch_insert(
        pg_conn_ctx, "security_audit_log", pg_columns, batch_rows,
        conflict_action="DO NOTHING",
        dry_run=dry_run,
    )
    return len(rows_sql), inserted


# ---------------------------------------------------------------------------
# Generic table migrator dispatcher
# ---------------------------------------------------------------------------

_SQLITE_TABLE_HANDLERS = {
    "working_memory": _migrate_working_memory,
    "narrative_traces": _migrate_narrative_traces,
    "spreading_edges": _migrate_spreading_edges,
    "consolidation_log": _migrate_consolidation_log,
    "reconsolidation_log": _migrate_reconsolidation_log,
    "prediction_state": _migrate_prediction_state,
    "prediction_results": _migrate_prediction_results,
    "write_queue": _migrate_write_queue,
    "event_counts": _migrate_event_counts,
    "tool_calls": _migrate_tool_calls,
    "session_checkpoints": _migrate_session_checkpoints,
    "sleep_loop_state": _migrate_sleep_loop_state,
    "fok_priors": _migrate_fok_priors,
    "security_audit_log": _migrate_security_audit_log,
}


def migrate_sqlite_table(
    table_name: str,
    sqlite_conn: sqlite3.Connection,
    dry_run: bool = False,
) -> tuple[int, int]:
    """
    Migrate a single SQLite table to PostgreSQL.

    Returns (scanned, inserted). Returns (0, 0) if handler not found.
    """
    handler = _SQLITE_TABLE_HANDLERS.get(table_name)
    if handler is None:
        logger.warning("No handler for table %s — skipping", table_name)
        return 0, 0

    pg_conn_ctx = _get_pg_conn()

    try:
        scanned, inserted = handler(sqlite_conn, pg_conn_ctx, dry_run)
        return scanned, inserted
    except Exception as e:
        logger.error("Failed migrating %s: %s", table_name, e)
        return 0, 0


# ---------------------------------------------------------------------------
# SQLite orchestrator
# ---------------------------------------------------------------------------

_FTS_TABLES_ORDER = [
    "working_memory",
    "narrative_traces",
    "spreading_edges",
    "consolidation_log",
    "reconsolidation_log",
    "prediction_state",
    "prediction_results",
    "write_queue",
    "event_counts",
    "tool_calls",
    "session_checkpoints",
    "sleep_loop_state",
    "fok_priors",
    "security_audit_log",
]

# NOTE: retrieval_weights table does not exist in the current SQLite schema
# (verified — the PostgreSQL schema has it but SQLite does not). Skipped.


def migrate_all_sqlite(
    dry_run: bool = False,
    skip_tables: Optional[list[str]] = None,
) -> dict[str, tuple[int, int]]:
    """
    Migrate all SQLite tables from memories_fts.db and prospective.db.

    Returns dict of {table_name: (scanned, inserted)}.
    """
    results: dict[str, tuple[int, int]] = {}
    skip_set = set(skip_tables or [])

    # --- memories_fts.db ---
    logger.info("=== Migrating SQLite memories_fts.db ===")
    if not os.path.exists(FTS_DB_PATH):
        logger.warning("memories_fts.db not found at %s — skipping", FTS_DB_PATH)
    else:
        fts_conn = sqlite3.connect(FTS_DB_PATH, timeout=30)
        fts_conn.execute("PRAGMA busy_timeout=30000")
        fts_conn.row_factory = None  # keep tuples

        # Get actual tables in this DB
        existing = {
            r[0] for r in fts_conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }

        for table in _FTS_TABLES_ORDER:
            if table in skip_set:
                logger.info("  [skip] %s", table)
                continue
            if table not in existing:
                logger.info("  [not found] %s — skipping", table)
                results[table] = (0, 0)
                continue

            logger.info("  Migrating %s ...", table)
            scanned, inserted = migrate_sqlite_table(table, fts_conn, dry_run)
            results[table] = (scanned, inserted)
            logger.info("    %s: scanned=%d inserted=%d", table, scanned, inserted)

        fts_conn.close()

    # --- prospective.db ---
    logger.info("=== Migrating SQLite prospective.db ===")
    if not os.path.exists(PROSPECTIVE_DB_PATH):
        logger.warning("prospective.db not found at %s — skipping", PROSPECTIVE_DB_PATH)
    else:
        pros_conn = sqlite3.connect(PROSPECTIVE_DB_PATH, timeout=30)
        pros_conn.execute("PRAGMA busy_timeout=30000")

        existing_pros = {
            r[0] for r in pros_conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }

        for table in ["goals", "goal_log", "intentions", "intention_log"]:
            if table in skip_set:
                logger.info("  [skip] %s", table)
                continue
            if table not in existing_pros:
                logger.info("  [not found] %s — skipping", table)
                results[table] = (0, 0)
                continue

            logger.info("  Migrating %s ...", table)
            scanned, inserted = _migrate_prospective_table(table, pros_conn, dry_run)
            results[table] = (scanned, inserted)
            logger.info("    %s: scanned=%d inserted=%d", table, scanned, inserted)

        pros_conn.close()

    return results


# ---------------------------------------------------------------------------
# Prospective DB table migrators
# ---------------------------------------------------------------------------

def _migrate_prospective_table(
    table: str,
    sqlite_conn: sqlite3.Connection,
    dry_run: bool,
) -> tuple[int, int]:
    """Dispatch to the correct handler for prospective.db tables."""
    pg_conn_ctx = _get_pg_conn()
    handlers = {
        "goals": _migrate_goals,
        "goal_log": _migrate_goal_log,
        "intentions": _migrate_intentions,
        "intention_log": _migrate_intention_log,
    }
    handler = handlers.get(table)
    if handler is None:
        return 0, 0
    try:
        return handler(sqlite_conn, pg_conn_ctx, dry_run)
    except Exception as e:
        logger.error("Failed migrating prospective.%s: %s", table, e)
        return 0, 0


def _migrate_goals(sqlite_conn: sqlite3.Connection, pg_conn_ctx, dry_run: bool) -> tuple[int, int]:
    import psycopg

    rows_sql = sqlite_conn.execute(
        "SELECT id, title, parent_id, level, status, priority, access_count, "
        "       created_at, last_accessed, deadline, assigned_to, context, "
        "       metadata, goal_what, goal_why, goal_last_state, goal_next_step, "
        "       context_updated_at "
        "FROM goals"
    ).fetchall()

    pg_columns = [
        "id", "title", "parent_id", "level", "status", "priority", "access_count",
        "created_at", "last_accessed", "deadline", "assigned_to", "context",
        "metadata", "goal_what", "goal_why", "goal_last_state", "goal_next_step",
        "context_updated_at",
    ]

    batch_rows = []
    skipped = 0
    for r in rows_sql:
        (row_id, title, parent_id, level, status, priority, access_count,
         created_at, last_accessed, deadline, assigned_to, context,
         metadata, goal_what, goal_why, goal_last_state, goal_next_step,
         context_updated_at) = r

        goal_uuid = _safe_uuid(row_id)
        if goal_uuid is None:
            logger.warning("goals: invalid UUID %s — generating new one", row_id)
            goal_uuid = str(uuid.uuid4())

        parent_uuid = _safe_uuid(parent_id) if parent_id else None
        meta_json = _safe_json_loads(metadata, {})

        batch_rows.append((
            goal_uuid,
            (title or "")[:200],
            parent_uuid,
            (level or "task")[:20],
            (status or "active")[:20],
            (priority or "medium")[:20],
            _safe_int(access_count),
            _parse_ts(created_at),
            _parse_ts(last_accessed),
            _parse_ts(deadline),
            assigned_to[:100] if assigned_to else None,
            context,
            psycopg.types.json.Json(meta_json),
            goal_what,
            goal_why,
            goal_last_state,
            goal_next_step,
            _parse_ts(context_updated_at),
        ))

    inserted = _batch_insert(
        pg_conn_ctx, "goals", pg_columns, batch_rows,
        conflict_action="(id) DO NOTHING",
        dry_run=dry_run,
    )
    return len(rows_sql), inserted


def _migrate_goal_log(sqlite_conn: sqlite3.Connection, pg_conn_ctx, dry_run: bool) -> tuple[int, int]:
    rows_sql = sqlite_conn.execute(
        "SELECT id, goal_id, event, detail, created_at FROM goal_log"
    ).fetchall()

    pg_columns = ["goal_id", "event", "detail", "created_at"]

    batch_rows = []
    for r in rows_sql:
        row_id, goal_id, event, detail, created_at = r
        goal_uuid = _safe_uuid(goal_id)
        if goal_uuid is None:
            continue  # skip orphaned log entries
        batch_rows.append((
            goal_uuid,
            (event or "")[:50],
            detail,
            _parse_ts(created_at),
        ))

    inserted = _batch_insert(
        pg_conn_ctx, "goal_log", pg_columns, batch_rows,
        conflict_action="DO NOTHING",
        dry_run=dry_run,
    )
    return len(rows_sql), inserted


def _migrate_intentions(sqlite_conn: sqlite3.Connection, pg_conn_ctx, dry_run: bool) -> tuple[int, int]:
    import psycopg

    rows_sql = sqlite_conn.execute(
        "SELECT id, action, action_type, trigger_type, trigger_spec, cue_focality, "
        "       priority, status, activation, created_at, triggered_at, completed_at, "
        "       expiry, snooze_until, context_at_creation, creator, recurrence, "
        "       recurrence_spec, check_count, partial_match_count, "
        "       last_checked_at, last_maintained_at, goal_id "
        "FROM intentions"
    ).fetchall()

    pg_columns = [
        "id", "action", "action_type", "trigger_type", "trigger_spec", "cue_focality",
        "priority", "status", "activation", "created_at", "triggered_at", "completed_at",
        "expiry", "snooze_until", "context_at_creation", "creator", "recurrence",
        "recurrence_spec", "check_count", "partial_match_count",
        "last_checked_at", "last_maintained_at", "goal_id",
    ]

    batch_rows = []
    for r in rows_sql:
        (row_id, action, action_type, trigger_type, trigger_spec, cue_focality,
         priority, status, activation, created_at, triggered_at, completed_at,
         expiry, snooze_until, context_at_creation, creator, recurrence,
         recurrence_spec, check_count, partial_match_count,
         last_checked_at, last_maintained_at, goal_id) = r

        int_uuid = _safe_uuid(row_id)
        if int_uuid is None:
            int_uuid = str(uuid.uuid4())

        goal_uuid = _safe_uuid(goal_id) if goal_id else None

        batch_rows.append((
            int_uuid,
            action or "",
            (action_type or "remind")[:20],
            (trigger_type or "event")[:20],
            psycopg.types.json.Json(_safe_json_loads(trigger_spec, {})),
            (cue_focality or "focal")[:20],
            (priority or "medium")[:20],
            (status or "pending")[:20],
            _safe_float(activation, 0.5),
            _parse_ts(created_at),
            _parse_ts(triggered_at),
            _parse_ts(completed_at),
            _parse_ts(expiry),
            _parse_ts(snooze_until),
            psycopg.types.json.Json(_safe_json_loads(context_at_creation, {})),
            creator[:50] if creator else None,
            recurrence[:20] if recurrence else None,
            psycopg.types.json.Json(_safe_json_loads(recurrence_spec, {})),
            _safe_int(check_count),
            _safe_int(partial_match_count),
            _parse_ts(last_checked_at),
            _parse_ts(last_maintained_at),
            goal_uuid,
        ))

    inserted = _batch_insert(
        pg_conn_ctx, "intentions", pg_columns, batch_rows,
        conflict_action="(id) DO NOTHING",
        dry_run=dry_run,
    )
    return len(rows_sql), inserted


def _migrate_intention_log(sqlite_conn: sqlite3.Connection, pg_conn_ctx, dry_run: bool) -> tuple[int, int]:
    rows_sql = sqlite_conn.execute(
        "SELECT id, intention_id, event, detail, created_at FROM intention_log"
    ).fetchall()

    pg_columns = ["intention_id", "event", "detail", "created_at"]

    batch_rows = []
    for r in rows_sql:
        row_id, intention_id, event, detail, created_at = r
        int_uuid = _safe_uuid(intention_id)
        if int_uuid is None:
            continue
        batch_rows.append((
            int_uuid,
            (event or "")[:50],
            detail,
            _parse_ts(created_at),
        ))

    inserted = _batch_insert(
        pg_conn_ctx, "intention_log", pg_columns, batch_rows,
        conflict_action="DO NOTHING",
        dry_run=dry_run,
    )
    return len(rows_sql), inserted


# ===========================================================================
# VALIDATION
# ===========================================================================

def validate_counts() -> dict[str, dict]:
    """
    Compare row counts between source systems and PostgreSQL.

    Returns dict of comparison results. Prints summary to stdout.
    """
    logger.info("=== Validation: comparing counts ===")
    results = {}

    # --- Qdrant vs PostgreSQL memories ---
    try:
        qdrant = _get_qdrant()
        q_episodic = qdrant.get_collection(COLLECTION_NAME).points_count
        q_semantic = qdrant.get_collection(SEMANTIC_COLLECTION).points_count
    except Exception as e:
        logger.error("Qdrant count error: %s", e)
        q_episodic = None
        q_semantic = None

    pg_conn_ctx = _get_pg_conn()

    try:
        with pg_conn_ctx() as conn:
            pg_episodic = conn.execute(
                "SELECT COUNT(*) FROM memories WHERE is_semantic = FALSE"
            ).fetchone()[0]
            pg_semantic = conn.execute(
                "SELECT COUNT(*) FROM memories WHERE is_semantic = TRUE"
            ).fetchone()[0]
    except Exception as e:
        logger.error("PostgreSQL memories count error: %s", e)
        pg_episodic = None
        pg_semantic = None

    def _check(name, source_count, pg_count):
        if source_count is None or pg_count is None:
            status = "ERROR"
        elif pg_count == source_count:
            status = "OK"
        elif pg_count < source_count:
            status = "MISSING"
        else:
            status = "EXTRA"
        results[name] = {"source": source_count, "pg": pg_count, "status": status}
        icon = "✓" if status == "OK" else "✗"
        print(
            f"  {icon} {name}: source={source_count} pg={pg_count} [{status}]"
        )

    print("\n--- Qdrant vs PostgreSQL ---")
    _check("memories (episodic)", q_episodic, pg_episodic)
    _check("memories (semantic)", q_semantic, pg_semantic)

    # --- SQLite vs PostgreSQL auxiliary tables ---
    sqlite_table_map = {
        # (db, table_name): pg_table_name
        (FTS_DB_PATH, "working_memory"): "working_memory",
        (FTS_DB_PATH, "narrative_traces"): "narrative_traces",
        (FTS_DB_PATH, "spreading_edges"): "spreading_edges",
        (FTS_DB_PATH, "consolidation_log"): "consolidation_log",
        (FTS_DB_PATH, "reconsolidation_log"): "reconsolidation_log",
        (FTS_DB_PATH, "prediction_state"): "prediction_state",
        (FTS_DB_PATH, "prediction_results"): "prediction_results",
        (FTS_DB_PATH, "write_queue"): "write_queue",
        (FTS_DB_PATH, "event_counts"): "event_counts",
        (FTS_DB_PATH, "tool_calls"): "tool_calls",
        (FTS_DB_PATH, "session_checkpoints"): "session_checkpoints",
        (FTS_DB_PATH, "sleep_loop_state"): "sleep_loop_state",
        (FTS_DB_PATH, "fok_priors"): "fok_priors",
        (FTS_DB_PATH, "security_audit_log"): "security_audit_log",
        (PROSPECTIVE_DB_PATH, "goals"): "goals",
        (PROSPECTIVE_DB_PATH, "goal_log"): "goal_log",
        (PROSPECTIVE_DB_PATH, "intentions"): "intentions",
        (PROSPECTIVE_DB_PATH, "intention_log"): "intention_log",
    }

    print("\n--- SQLite vs PostgreSQL ---")
    open_dbs: dict[str, Optional[sqlite3.Connection]] = {}

    for (db_path, sqlite_table), pg_table in sqlite_table_map.items():
        sqlite_count = None
        pg_count = None

        # SQLite count
        if db_path not in open_dbs:
            if os.path.exists(db_path):
                open_dbs[db_path] = sqlite3.connect(db_path)
            else:
                open_dbs[db_path] = None

        db_conn = open_dbs[db_path]
        if db_conn is not None:
            try:
                sqlite_count = db_conn.execute(
                    f"SELECT COUNT(*) FROM {sqlite_table}"
                ).fetchone()[0]
            except Exception:
                sqlite_count = None

        # PostgreSQL count
        try:
            with pg_conn_ctx() as conn:
                pg_count = conn.execute(
                    f"SELECT COUNT(*) FROM {pg_table}"
                ).fetchone()[0]
        except Exception as e:
            logger.warning("PG count error for %s: %s", pg_table, e)
            pg_count = None

        label = f"{sqlite_table} → {pg_table}"
        _check(label, sqlite_count, pg_count)

    for db_conn in open_dbs.values():
        if db_conn is not None:
            db_conn.close()

    # Summary
    all_statuses = [v["status"] for v in results.values()]
    ok_count = all_statuses.count("OK")
    mismatch_count = len(all_statuses) - ok_count
    print(f"\n=== Validation Summary: {ok_count}/{len(all_statuses)} OK, {mismatch_count} mismatches ===")

    return results


# ===========================================================================
# MAIN
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Migrate codi-memory from Qdrant+SQLite to PostgreSQL+pgvector"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview migration without writing to PostgreSQL",
    )
    parser.add_argument(
        "--skip-qdrant",
        action="store_true",
        help="Skip Qdrant → PostgreSQL memories migration",
    )
    parser.add_argument(
        "--skip-sqlite",
        action="store_true",
        help="Skip SQLite → PostgreSQL auxiliary tables migration",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Only run validation (count comparison)",
    )
    args = parser.parse_args()

    if args.dry_run:
        logger.info("*** DRY-RUN MODE — no data will be written to PostgreSQL ***")

    # Ensure pgvector is registered on connections
    # (config_pg.py configure callback does this automatically)

    start_time = datetime.now()

    # -----------------------------------------------------------------------
    # Step 1: Qdrant migrations
    # -----------------------------------------------------------------------
    if not args.skip_qdrant and not args.validate_only:
        try:
            ep_scanned, ep_inserted = migrate_qdrant_memories(dry_run=args.dry_run)
        except Exception as e:
            logger.error("Episodic migration failed: %s", e)
            ep_scanned = ep_inserted = 0

        try:
            sem_scanned, sem_inserted = migrate_qdrant_semantic(dry_run=args.dry_run)
        except Exception as e:
            logger.error("Semantic migration failed: %s", e)
            sem_scanned = sem_inserted = 0
    else:
        if args.skip_qdrant:
            logger.info("Skipping Qdrant migration (--skip-qdrant)")
        ep_scanned = ep_inserted = sem_scanned = sem_inserted = 0

    # -----------------------------------------------------------------------
    # Step 2: SQLite migrations
    # -----------------------------------------------------------------------
    sqlite_results: dict[str, tuple[int, int]] = {}
    if not args.skip_sqlite and not args.validate_only:
        try:
            sqlite_results = migrate_all_sqlite(dry_run=args.dry_run)
        except Exception as e:
            logger.error("SQLite migration failed: %s", e)
    else:
        if args.skip_sqlite:
            logger.info("Skipping SQLite migration (--skip-sqlite)")

    # -----------------------------------------------------------------------
    # Step 3: Validation
    # -----------------------------------------------------------------------
    validation = validate_counts()

    # -----------------------------------------------------------------------
    # Final summary
    # -----------------------------------------------------------------------
    elapsed = (datetime.now() - start_time).total_seconds()
    print("\n" + "=" * 60)
    print("MIGRATION COMPLETE")
    print("=" * 60)
    print(f"  Elapsed:          {elapsed:.1f}s")
    if not args.skip_qdrant and not args.validate_only:
        print(f"  Episodic:         scanned={ep_scanned} inserted={ep_inserted}")
        print(f"  Semantic:         scanned={sem_scanned} inserted={sem_inserted}")
    if sqlite_results:
        total_scanned = sum(v[0] for v in sqlite_results.values())
        total_inserted = sum(v[1] for v in sqlite_results.values())
        print(f"  SQLite tables:    scanned={total_scanned} inserted={total_inserted}")
    if args.dry_run:
        print("  NOTE: DRY-RUN — nothing was written to PostgreSQL")
    print("=" * 60)

    # Exit with non-zero if any validation mismatch
    mismatches = [k for k, v in validation.items() if v["status"] not in ("OK", "ERROR")]
    if mismatches:
        logger.warning("Validation mismatches found for: %s", mismatches)
        sys.exit(1)


if __name__ == "__main__":
    main()
