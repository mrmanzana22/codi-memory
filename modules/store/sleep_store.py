"""Sleep store — sleep_loop_state table."""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from modules.store._base import SQLiteConnProvider, default_sqlite_provider
from modules.store import store_traced


class SleepStore:
    """Repository for sleep loop state data.

    Owns: sleep_loop_state table (SQLite).
    Invariant: methods return None for missing keys, {} for missing dicts.
    """

    def __init__(self, conn_provider: Optional[SQLiteConnProvider] = None):
        self._get_conn = conn_provider or default_sqlite_provider

    @store_traced
    def get_state(self, key: str) -> Optional[Any]:
        """Get a single state value by key. Returns parsed JSON or raw string."""
        conn = self._get_conn()
        row = conn.execute(
            "SELECT value FROM sleep_loop_state WHERE key = ?", (key,)
        ).fetchone()
        if not row or row[0] is None:
            return None
        try:
            return json.loads(row[0])
        except (json.JSONDecodeError, TypeError):
            return row[0]

    @store_traced
    def set_state(self, key: str, value: Any) -> None:
        """Set a state value. Serializes to JSON if not a string."""
        conn = self._get_conn()
        if isinstance(value, str):
            serialized = value
        else:
            serialized = json.dumps(value)
        conn.execute(
            "INSERT OR REPLACE INTO sleep_loop_state (key, value) VALUES (?, ?)",
            (key, serialized),
        )

    @store_traced
    def get_all_state(self) -> Dict[str, Any]:
        """Get all sleep loop state as a dict."""
        conn = self._get_conn()
        rows = conn.execute("SELECT key, value FROM sleep_loop_state").fetchall()
        result = {}
        for r in rows:
            try:
                result[r[0]] = json.loads(r[1]) if r[1] else None
            except (json.JSONDecodeError, TypeError):
                result[r[0]] = r[1]
        return result

