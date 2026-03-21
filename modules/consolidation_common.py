"""
Shared utilities for the consolidation subsystem.

Provides embedding generation, vector similarity, and DB connections
used by consolidation.py, reconsolidation.py, and semantic_store.py.

Extracted to avoid circular imports between the split modules.
"""

import math
import sqlite3
import functools
import logging
from typing import Optional

import openai

from modules.config import FTS_DB_PATH
from modules.secret_redact import redact_secrets

_logger = logging.getLogger(__name__)

# OpenAI client (lazy, uses OPENAI_API_KEY from env)
_oai_client = None


def _get_oai():
    global _oai_client
    if _oai_client is None:
        _oai_client = openai.OpenAI()
    return _oai_client


@functools.lru_cache(maxsize=256)
def _embed_text_cached(text: str) -> tuple:
    """Generate embedding (cached). Returns tuple for hashability."""
    resp = _get_oai().embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return tuple(resp.data[0].embedding)


def _embed_text(text: str) -> list:
    """Generate embedding using text-embedding-3-small (1536 dims).

    Results are LRU-cached (256 entries). Identical texts reuse
    previous embeddings without an API call.
    """
    return list(_embed_text_cached(text))


def get_embed_cache_info() -> dict:
    """Return cache stats for _embed_text (hits, misses, size, maxsize)."""
    info = _embed_text_cached.cache_info()
    return {
        "hits": info.hits,
        "misses": info.misses,
        "current_size": info.currsize,
        "max_size": info.maxsize,
        "hit_rate": f"{info.hits / (info.hits + info.misses) * 100:.1f}%" if (info.hits + info.misses) > 0 else "n/a",
    }


def _cosine_similarity(a: list, b: list) -> float:
    """Compute cosine similarity between two vectors."""
    if len(a) != len(b):
        raise ValueError(
            f"Cosine similarity requires vectors of equal length, got {len(a)} and {len(b)}"
        )
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)

# ============================================================
# OLLAMA ROUTING (opt-in via CODI_USE_OLLAMA=true)
# ============================================================

def _try_ollama(task_type: str, prompt: str) -> Optional[str]:
    """Try routing a chat completion through Ollama.

    Returns response text on success, None if disabled/unavailable/failed.
    When None, caller should fall back to OpenAI as before.
    """
    try:
        from modules.ollama_router import ollama_chat_completion
    except ImportError as e:
        _logger.debug("Ollama routing unavailable: %s", e)
        return None

    return ollama_chat_completion(task_type, prompt)


# ============================================================
# SQLITE CONNECTION
# ============================================================

def _consolidation_conn():
    """Get SQLite connection for consolidation tables."""
    from modules.config import connect_fts
    return connect_fts()


def init_consolidation_db():
    """Validate consolidation-related tables exist in memories_fts.db."""
    from modules.migrations import ensure_schema_ready_db
    ensure_schema_ready_db(FTS_DB_PATH, [
        "consolidation_log", "reconsolidation_log", "labile_memories",
    ])
    _logger.info("Tables validated OK")


# Validate on import
try:
    init_consolidation_db()
except Exception as e:
    _logger.error("Could not validate tables: %s", redact_secrets(str(e)))
    raise
