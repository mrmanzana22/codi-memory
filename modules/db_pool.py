"""
db_pool.py - Thread-local SQLite connection pool.

Each thread gets at most one connection per db_path. Connections are
created lazily on first use and reused for all subsequent calls in the
same thread. PRAGMAs are applied once at creation time, guaranteeing
consistency across the entire codebase.

Usage:
    from modules.db_pool import get_conn, close_thread_connections

    conn = get_conn()           # uses FTS_DB_PATH by default
    conn.execute("SELECT ...")  # reuses same connection in this thread
    # No need to close — pool manages lifecycle

    # In tests / shutdown:
    close_thread_connections()  # close all connections for this thread
"""

from __future__ import annotations

import sqlite3
import threading
from typing import Optional

from modules.config import FTS_DB_PATH

# ---------------------------------------------------------------------------
# Thread-local storage: each thread has a dict {db_path: Connection}
# ---------------------------------------------------------------------------
_local = threading.local()

# Canonical PRAGMA set applied to every new connection
_PRAGMAS = (
    "PRAGMA journal_mode=WAL",
    "PRAGMA synchronous=NORMAL",
    "PRAGMA busy_timeout=10000",
    "PRAGMA foreign_keys=ON",
)


def _is_alive(conn: sqlite3.Connection) -> bool:
    """Check if a connection is still usable."""
    try:
        conn.execute("SELECT 1")
        return True
    except Exception:
        return False


def _create_conn(db_path: str) -> sqlite3.Connection:
    """Create a new SQLite connection with standard PRAGMAs."""
    conn = sqlite3.connect(
        db_path,
        timeout=10,
        detect_types=sqlite3.PARSE_DECLTYPES,
    )
    conn.row_factory = sqlite3.Row
    for pragma in _PRAGMAS:
        conn.execute(pragma)
    return conn


def get_conn(db_path: str = None) -> sqlite3.Connection:
    """Get a thread-local SQLite connection for the given db_path.

    - First call in a thread creates the connection + applies PRAGMAs.
    - Subsequent calls return the same connection (no churn).
    - If the connection is dead (closed externally), it is recreated.

    Args:
        db_path: Path to the SQLite database. Defaults to FTS_DB_PATH.

    Returns:
        A configured sqlite3.Connection with WAL, Row factory, etc.
    """
    if db_path is None:
        db_path = FTS_DB_PATH

    conns: dict = getattr(_local, "conns", None)
    if conns is None:
        conns = {}
        _local.conns = conns

    conn = conns.get(db_path)

    if conn is not None and _is_alive(conn):
        return conn

    # Connection missing or dead — (re)create
    if conn is not None:
        try:
            conn.close()
        except Exception:
            pass

    conn = _create_conn(db_path)
    conns[db_path] = conn
    return conn


def close_thread_connections() -> None:
    """Close all connections held by the current thread.

    Call this in test teardown or at thread exit to prevent leaks.
    """
    conns: dict = getattr(_local, "conns", None)
    if not conns:
        return
    for db_path, conn in list(conns.items()):
        try:
            conn.close()
        except Exception:
            pass
    conns.clear()


def close_all() -> None:
    """Close connections for the current thread.

    Note: This can only close connections owned by the calling thread
    (Python threading.local limitation). For full cleanup at shutdown,
    each thread should call close_thread_connections() before exiting.
    """
    close_thread_connections()
