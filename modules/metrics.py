"""
metrics.py - Instrumentación y auditoría de tools (sin contenido sensible).
Guarda metadatos de cada tool-call en SQLite (memories_fts.db).
"""

from __future__ import annotations

import asyncio
import functools
import json
import sqlite3
import time
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Dict, Optional

from modules.config import FTS_DB_PATH, now_iso
from modules.tracing import get_trace_id


@contextmanager
def metrics_conn():
    conn = sqlite3.connect(FTS_DB_PATH, timeout=3.0)
    try:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA busy_timeout=3000")
        yield conn
    finally:
        conn.close()


def _ensure_tool_calls_schema(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS tool_calls (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          tool_name TEXT NOT NULL,
          started_at TEXT NOT NULL,
          duration_ms INTEGER NOT NULL,
          success INTEGER NOT NULL,
          error_type TEXT,
          args_size INTEGER DEFAULT 0,
          result_size INTEGER DEFAULT 0,
          session_id TEXT,
          tag TEXT
        );
        """
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_tool_calls_time ON tool_calls(started_at);")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_tool_calls_tool ON tool_calls(tool_name, started_at);")


def _safe_len_bytes(obj: Any) -> int:
    """
    Calcula tamaño aproximado sin persistir contenido.
    Ojo: serializa en memoria para medir, pero NO se guarda el contenido.
    """
    try:
        if isinstance(obj, (bytes, bytearray)):
            return len(obj)
        if isinstance(obj, str):
            return len(obj.encode("utf-8", errors="ignore"))
        s = json.dumps(obj, ensure_ascii=False, default=str)
        return len(s.encode("utf-8", errors="ignore"))
    except Exception:
        return 0


def log_tool_call(
    tool_name: str,
    started_at: str,
    duration_ms: int,
    success: bool,
    error_type: Optional[str],
    args_size: int,
    result_size: int,
    session_id: Optional[str] = None,
    tag: Optional[str] = None,
) -> None:
    with metrics_conn() as conn:
        _ensure_tool_calls_schema(conn)
        conn.execute("BEGIN IMMEDIATE")
        conn.execute(
            """
            INSERT INTO tool_calls
            (tool_name, started_at, duration_ms, success, error_type, args_size, result_size, session_id, tag)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                tool_name,
                started_at,
                int(duration_ms),
                1 if success else 0,
                error_type,
                int(args_size),
                int(result_size),
                session_id,
                tag,
            ),
        )
        conn.commit()


MACRO_TAGS = {
    "recall": "macro:recall",
    "remember": "macro:remember",
    "context_snapshot": "macro:context",
}


def instrument_mcp(mcp: Any) -> None:
    """
    Monkeypatch a mcp.tool() para envolver TODAS las tools registradas,
    sin tocar cada módulo individual.
    """
    if getattr(mcp, "_metrics_instrumented", False):
        return

    original_tool = mcp.tool

    def tool_wrapper(*dargs, **dkwargs):
        decorator = original_tool(*dargs, **dkwargs)

        def wrap(fn: Callable):
            tool_name = getattr(fn, "__name__", "unknown_tool")
            tag = getattr(fn, "_audit_tag", None) or MACRO_TAGS.get(tool_name)

            if asyncio.iscoroutinefunction(fn):

                @functools.wraps(fn)
                async def async_wrapped(*args, **kwargs):
                    started = now_iso()
                    t0 = time.perf_counter()
                    args_size = _safe_len_bytes({"args": args, "kwargs": kwargs})
                    try:
                        result = await fn(*args, **kwargs)
                        dur_ms = int((time.perf_counter() - t0) * 1000)
                        res_size = _safe_len_bytes(result)
                        log_tool_call(
                            tool_name=tool_name,
                            started_at=started,
                            duration_ms=dur_ms,
                            success=True,
                            error_type=None,
                            args_size=args_size,
                            result_size=res_size,
                            session_id=get_trace_id() or None,
                            tag=tag,
                        )
                        return result
                    except Exception as e:
                        dur_ms = int((time.perf_counter() - t0) * 1000)
                        log_tool_call(
                            tool_name=tool_name,
                            started_at=started,
                            duration_ms=dur_ms,
                            success=False,
                            error_type=e.__class__.__name__,
                            args_size=args_size,
                            result_size=0,
                            session_id=get_trace_id() or None,
                            tag=tag,
                        )
                        raise

                return decorator(async_wrapped)

            @functools.wraps(fn)
            def wrapped(*args, **kwargs):
                started = now_iso()
                t0 = time.perf_counter()
                args_size = _safe_len_bytes({"args": args, "kwargs": kwargs})
                try:
                    result = fn(*args, **kwargs)
                    dur_ms = int((time.perf_counter() - t0) * 1000)
                    res_size = _safe_len_bytes(result)
                    log_tool_call(
                        tool_name=tool_name,
                        started_at=started,
                        duration_ms=dur_ms,
                        success=True,
                        error_type=None,
                        args_size=args_size,
                        result_size=res_size,
                        session_id=get_trace_id() or None,
                        tag=tag,
                    )
                    return result
                except Exception as e:
                    dur_ms = int((time.perf_counter() - t0) * 1000)
                    log_tool_call(
                        tool_name=tool_name,
                        started_at=started,
                        duration_ms=dur_ms,
                        success=False,
                        error_type=e.__class__.__name__,
                        args_size=args_size,
                        result_size=0,
                        session_id=get_trace_id() or None,
                        tag=tag,
                    )
                    raise

            return decorator(wrapped)

        return wrap

    mcp.tool = tool_wrapper
    mcp._metrics_instrumented = True


def _cutoff_iso(days: int) -> str:
    dt = datetime.now(timezone.utc) - timedelta(days=int(days))
    return dt.isoformat()


def tool_usage_summary(days: int = 7) -> Dict[str, Any]:
    cutoff = _cutoff_iso(days)
    with metrics_conn() as conn:
        _ensure_tool_calls_schema(conn)

        rows = conn.execute(
            """
            SELECT tool_name,
                   COUNT(*) AS calls,
                   SUM(success) AS successes,
                   SUM(CASE WHEN success=0 THEN 1 ELSE 0 END) AS failures,
                   AVG(duration_ms) AS avg_ms
            FROM tool_calls
            WHERE started_at >= ?
            GROUP BY tool_name
            ORDER BY calls DESC
            """,
            (cutoff,),
        ).fetchall()

        total = conn.execute(
            "SELECT COUNT(*) FROM tool_calls WHERE started_at >= ?",
            (cutoff,),
        ).fetchone()[0]

        macro = conn.execute(
            """
            SELECT COUNT(*) FROM tool_calls
            WHERE started_at >= ?
              AND tool_name IN ('recall','remember','context_snapshot')
            """,
            (cutoff,),
        ).fetchone()[0]

    tools = []
    for (name, calls, successes, failures, avg_ms) in rows:
        calls = int(calls or 0)
        successes = int(successes or 0)
        failures = int(failures or 0)
        sr = (successes / calls * 100.0) if calls else 0.0
        fr = (failures / calls * 100.0) if calls else 0.0
        tools.append(
            {
                "tool": name,
                "calls": calls,
                "success_rate": sr,
                "fail_rate": fr,
                "avg_ms": float(avg_ms or 0.0),
            }
        )

    return {
        "days": int(days),
        "total_calls": int(total or 0),
        "macro_calls": int(macro or 0),
        "macro_share": (macro / total) if total else 0.0,
        "tools": tools,
    }
