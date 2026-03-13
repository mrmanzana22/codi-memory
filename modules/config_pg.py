"""
config_pg.py - PostgreSQL configuration and connection pool.
=============================================================
Fase 1, Sprint 1.1 of the pgvector migration.

Provides a synchronous connection pool via psycopg3's ConnectionPool.
Replaces: config.py (connect_fts, get_memory, get_qdrant)
          db_pool.py (thread-local SQLite pool)

Usage:
    from modules.config_pg import get_pool, get_conn

    # Context manager (recommended)
    with get_conn() as conn:
        rows = conn.execute("SELECT ...").fetchall()

    # Direct pool access
    pool = get_pool()
"""

from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from typing import Optional

import psycopg
from psycopg_pool import ConnectionPool
from pgvector.psycopg import register_vector

_logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Connection string
# ---------------------------------------------------------------------------
# Priority: InstanceConfig (YAML with env interpolation) > _default_hare() fallback
from modules.instance_config import get_instance as _get_instance
PG_CONNECTION_STRING = _get_instance().pg_url

# ---------------------------------------------------------------------------
# Pool configuration
# ---------------------------------------------------------------------------
PG_POOL_MIN = int(os.getenv("CODI_PG_POOL_MIN", "2"))
PG_POOL_MAX = int(os.getenv("CODI_PG_POOL_MAX", "10"))
PG_CONNECT_TIMEOUT = int(os.getenv("CODI_PG_CONNECT_TIMEOUT", "10"))

# ---------------------------------------------------------------------------
# Pool singleton
# ---------------------------------------------------------------------------
_pool: Optional[ConnectionPool] = None


def _configure_conn(conn: psycopg.Connection) -> None:
    """Configure each new connection from the pool."""
    register_vector(conn)
    conn.execute("SET search_path TO public")
    # Ensure HNSW index is used (SSD-optimized planner)
    conn.execute("SET random_page_cost = 1.1")


def get_pool() -> ConnectionPool:
    """Get or create the global connection pool.

    Thread-safe: ConnectionPool handles its own locking.
    First call initializes the pool (blocks until min_size connections ready).
    """
    global _pool
    if _pool is None:
        _pool = ConnectionPool(
            conninfo=PG_CONNECTION_STRING,
            min_size=PG_POOL_MIN,
            max_size=PG_POOL_MAX,
            timeout=PG_CONNECT_TIMEOUT,
            configure=_configure_conn,
            kwargs={"autocommit": True},
        )
        _logger.info(
            "PG pool created: min=%d max=%d host=%s",
            PG_POOL_MIN,
            PG_POOL_MAX,
            PG_CONNECTION_STRING.split("@")[-1].split("/")[0] if "@" in PG_CONNECTION_STRING else "?",
        )
    return _pool


@contextmanager
def get_conn():
    """Context manager: borrow a connection from the pool.

    Usage:
        with get_conn() as conn:
            conn.execute("INSERT INTO ...")

    The connection is returned to the pool when the block exits.
    Autocommit is ON by default (each statement is its own transaction).
    For explicit transactions, use `with conn.transaction():`.
    """
    pool = get_pool()
    with pool.connection() as conn:
        yield conn


def close_pool() -> None:
    """Close the pool. Call on shutdown."""
    global _pool
    if _pool is not None:
        _pool.close()
        _pool = None
        _logger.info("PG pool closed.")


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------
def pg_health_check() -> dict:
    """Quick health check. Returns {ok, version, pgvector, pool_size}."""
    try:
        with get_conn() as conn:
            ver = conn.execute("SELECT version()").fetchone()[0]
            pgv = conn.execute(
                "SELECT extversion FROM pg_extension WHERE extname='vector'"
            ).fetchone()
            pgv_ver = pgv[0] if pgv else "NOT INSTALLED"
            pool = get_pool()
            return {
                "ok": True,
                "version": ver[:60],
                "pgvector": pgv_ver,
                "pool_size": pool.get_stats().get("pool_size", 0)
                if hasattr(pool, "get_stats")
                else PG_POOL_MIN,
            }
    except Exception as e:
        return {"ok": False, "error": str(e)}
