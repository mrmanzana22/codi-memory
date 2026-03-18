"""
Store base — connection providers and helpers.

ConnProvider protocols allow dependency injection for testability.
Production code uses db_pool.get_conn() (SQLite) and config_pg.get_conn() (PG).
Tests inject tmp_path-based providers.
"""

from __future__ import annotations

import sqlite3
from typing import Callable, Optional, Protocol, runtime_checkable


@runtime_checkable
class SQLiteConnProvider(Protocol):
    """Protocol for providing SQLite connections."""

    def __call__(self, db_path: Optional[str] = None) -> sqlite3.Connection: ...


def default_sqlite_provider(db_path: Optional[str] = None) -> sqlite3.Connection:
    """Default provider — wraps db_pool.get_conn (thread-local pool)."""
    from modules.db_pool import get_conn

    return get_conn(db_path)


def row_to_dict(row: sqlite3.Row) -> dict:
    """Convert sqlite3.Row to a plain dict. Handles both Row and tuple."""
    if row is None:
        return {}
    if isinstance(row, dict):
        return row
    try:
        return dict(row)
    except (TypeError, ValueError):
        return {}
