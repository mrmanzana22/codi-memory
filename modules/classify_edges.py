"""
Codi Memory - Edge Classification (Sprint 1, item 1.2)
======================================================
Batch job that classifies co_occurs edges as causal types based on:
  1. Temporal precedence: A.created_at < B.created_at
  2. Topic coherence: A and B share topic
  3. Temporal proximity: within 24h window
  4. Content signal: A has causal_links

Pearl 2009: causal structure requires directed typed edges.
Target: ~20-30% of edges reclassified as 'causes' or 'enables'.

Run as batch job during sleep loop or manually.
"""

import logging
import sqlite3
from datetime import datetime, timedelta, timezone

from modules.pg_store import pg

_logger = logging.getLogger(__name__)

# Thresholds
MAX_CAUSAL_GAP_HOURS = 24   # Max temporal gap for causal candidacy
BATCH_SIZE = 50              # Qdrant batch retrieve size


def classify_edges(fts_db_path: str = None, dry_run: bool = False) -> dict:
    """Batch classify co_occurs edges into causal types.

    Heuristic rules (ordered by strength):
    1. A has causal_links AND A.created_at < B.created_at -> 'causes' (strong)
    2. Same topic AND A.created_at < B.created_at AND gap < 24h -> 'enables'
    3. Everything else stays 'co_occurs'

    Args:
        fts_db_path: Path to FTS database with spreading_edges
        dry_run: If True, don't write changes, just report

    Returns:
        {total_edges, classified_causes, classified_enables, unchanged, errors}
    """
    from modules.config import COLLECTION_NAME, FTS_DB_PATH

    fts_db_path = fts_db_path or FTS_DB_PATH
    result = {
        "total_edges": 0,
        "classified_causes": 0,
        "classified_enables": 0,
        "unchanged": 0,
        "errors": 0,
    }

    try:
        from modules.config import connect_fts
        conn = connect_fts(fts_db_path)
    except Exception as e:
        _logger.error("classify_edges: can't open DB: %s", e)
        return result

    # 1. Fetch all co_occurs edges
    try:
        edges = conn.execute(
            "SELECT from_id, to_id FROM spreading_edges WHERE edge_type = 'co_occurs'"
        ).fetchall()
    except Exception as e:
        _logger.error("classify_edges: can't read edges: %s", e)
        conn.close()
        return result

    result["total_edges"] = len(edges)
    if not edges:
        conn.close()
        return result

    # 2. Collect unique IDs and batch-fetch payloads
    all_ids = set()
    for from_id, to_id in edges:
        all_ids.add(from_id)
        all_ids.add(to_id)

    payload_cache = {}
    id_list = list(all_ids)

    for i in range(0, len(id_list), BATCH_SIZE):
        batch = id_list[i:i + BATCH_SIZE]
        try:
            pts = pg.get_by_ids(batch)
            for p in (pts or []):
                payload_cache[str(p.id)] = p.payload or {}
        except Exception as e:
            _logger.debug("classify_edges batch retrieve error: %s", e)
            result["errors"] += 1

    _logger.info("classify_edges: %d edges, %d/%d payloads fetched",
                 len(edges), len(payload_cache), len(all_ids))

    # 3. Classify each edge
    updates = []  # (edge_type, directed, from_id, to_id)

    for orig_from, orig_to in edges:
        pl_a = payload_cache.get(orig_from, {})
        pl_b = payload_cache.get(orig_to, {})
        from_id, to_id = orig_from, orig_to

        created_a = pl_a.get("created_at", "")
        created_b = pl_b.get("created_at", "")

        if not created_a or not created_b:
            result["unchanged"] += 1
            continue

        # Parse timestamps
        try:
            ts_a = _parse_ts(created_a)
            ts_b = _parse_ts(created_b)
        except Exception:
            result["unchanged"] += 1
            continue

        # Ensure A is earlier, B is later (for causal analysis)
        if ts_a >= ts_b:
            ts_a, ts_b = ts_b, ts_a
            pl_a, pl_b = pl_b, pl_a
            from_id, to_id = orig_to, orig_from

        gap = ts_b - ts_a
        if gap > timedelta(hours=MAX_CAUSAL_GAP_HOURS):
            result["unchanged"] += 1
            continue

        # Check causal signals using available fields
        causal_links_a = pl_a.get("causal_links", [])
        # Use category (most common) or topic as domain signal
        cat_a = pl_a.get("category", "") or pl_a.get("topic", "")
        cat_b = pl_b.get("category", "") or pl_b.get("topic", "")
        data_a = (pl_a.get("data", "") or "").lower()
        data_b = (pl_b.get("data", "") or "").lower()

        # Causal language patterns in content
        has_causal_language = any(w in data_a for w in
            ["porque", "por eso", "causa", "result", "therefore",
             "fixed", "resolved", "implemented", "applied",
             "discovered", "found that", "leads to"])

        if causal_links_a:
            # Strong: explicit causal_links AND temporal precedence
            updates.append(("causes", 1, from_id, to_id))
            result["classified_causes"] += 1
        elif has_causal_language and gap <= timedelta(hours=6):
            # Strong: causal language in content + tight temporal window
            updates.append(("causes", 1, from_id, to_id))
            result["classified_causes"] += 1
        elif (cat_a and cat_a == cat_b and gap <= timedelta(hours=4)
              and cat_a not in ("general", "aprendizaje", "episodio")):
            # Moderate: specific domain match + temporal proximity -> enables
            updates.append(("enables", 1, from_id, to_id))
            result["classified_enables"] += 1
        elif gap <= timedelta(hours=1) and _content_overlap(data_a, data_b):
            # Content-supported temporal proximity -> enables
            updates.append(("enables", 1, from_id, to_id))
            result["classified_enables"] += 1
        else:
            result["unchanged"] += 1

    # 4. Apply updates
    if not dry_run and updates:
        try:
            conn.executemany(
                "UPDATE spreading_edges SET edge_type = ?, directed = ? "
                "WHERE from_id = ? AND to_id = ?",
                updates,
            )
            conn.commit()
            _logger.info("classify_edges: applied %d updates (%d causes, %d enables)",
                         len(updates), result["classified_causes"],
                         result["classified_enables"])
        except Exception as e:
            _logger.error("classify_edges: update error: %s", e)
            result["errors"] += 1

    conn.close()
    return result


def _parse_ts(ts_str: str) -> datetime:
    """Parse ISO timestamp, normalizing to naive UTC for consistent comparison."""
    normalized = ts_str.strip().replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError:
        for fmt in ("%Y-%m-%dT%H:%M:%S.%f", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S"):
            try:
                return datetime.strptime(normalized, fmt)
            except ValueError:
                continue
        raise ValueError(f"Cannot parse timestamp: {ts_str}")
    if parsed.tzinfo is None:
        return parsed
    return parsed.astimezone(timezone.utc).replace(tzinfo=None)


def _content_overlap(text_a: str, text_b: str, min_shared: int = 3) -> bool:
    """Check if two texts share significant content words.

    Filters stopwords, requires at least min_shared unique words in common.
    """
    _STOP = {"the", "a", "an", "is", "are", "was", "were", "be", "been",
             "to", "of", "in", "for", "on", "with", "at", "by", "from",
             "and", "or", "not", "but", "this", "that", "it", "its",
             "de", "en", "el", "la", "los", "las", "un", "una", "del",
             "que", "con", "por", "para", "es", "no", "se", "como", "y"}

    words_a = {w for w in text_a.split() if len(w) > 3 and w not in _STOP}
    words_b = {w for w in text_b.split() if len(w) > 3 and w not in _STOP}

    if not words_a or not words_b:
        return False

    shared = words_a & words_b
    return len(shared) >= min_shared
