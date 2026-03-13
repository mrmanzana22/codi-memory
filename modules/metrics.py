"""
metrics.py - Instrumentación y auditoría de tools (sin contenido sensible).
Guarda metadatos de cada tool-call en SQLite (memories_fts.db).
"""

from __future__ import annotations

import asyncio
import functools
import json
import logging
import time
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Dict, Optional

from modules.config import now_iso
from modules.tracing import get_trace_id

_logger = logging.getLogger(__name__)
from modules.secret_redact import redact_secrets
from modules.db_pool import get_conn


@contextmanager
def metrics_conn():
    """Yield a pooled SQLite connection. Does NOT close on exit (pool manages lifecycle)."""
    yield get_conn()


def _ensure_tool_calls_schema(conn: sqlite3.Connection) -> None:
    from modules.migrations import ensure_schema_ready
    ensure_schema_ready(conn, ["tool_calls"])


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


def _log_tool_call_safe(**kwargs) -> None:
    """Fail-open wrapper: metrics must never break actual tool execution."""
    try:
        log_tool_call(**kwargs)
    except Exception:
        _logger.exception("log_tool_call failed for tool=%s", kwargs.get("tool_name", "?"))


# ============================================================
# PERFORMANCE CONTRACTS - Budget lookup & violation detection
# ============================================================

def get_tool_contract(tool_name: str) -> dict:
    """Return the performance contract (p95/p99 budgets) for a tool."""
    from modules.config import PERF_CONTRACTS, PERF_TOOL_CONTRACT
    cat = PERF_TOOL_CONTRACT.get(tool_name, "default")
    return PERF_CONTRACTS[cat]


def compute_percentiles(tool_name: str, days: int = 7) -> dict:
    """Compute p50/p95/p99 latency percentiles from SQLite tool_calls."""
    cutoff = _cutoff_iso(days)
    with metrics_conn() as conn:
        _ensure_tool_calls_schema(conn)
        rows = conn.execute(
            "SELECT duration_ms FROM tool_calls WHERE tool_name = ? AND started_at >= ? ORDER BY duration_ms",
            (tool_name, cutoff),
        ).fetchall()
    if not rows:
        return {"count": 0, "p50": 0, "p95": 0, "p99": 0, "max": 0}
    durations = [r[0] for r in rows]
    n = len(durations)
    return {
        "count": n,
        "p50": durations[int(n * 0.50)] if n > 0 else 0,
        "p95": durations[min(int(n * 0.95), n - 1)],
        "p99": durations[min(int(n * 0.99), n - 1)],
        "max": durations[-1],
    }


def check_budget_violation(tool_name: str, duration_ms: int) -> Optional[dict]:
    """Check if a tool call exceeded its latency budget. Returns violation info or None."""
    contract = get_tool_contract(tool_name)
    if duration_ms > contract["p99"]:
        return {"level": "p99", "budget_ms": contract["p99"], "actual_ms": duration_ms}
    if duration_ms > contract["p95"]:
        return {"level": "p95", "budget_ms": contract["p95"], "actual_ms": duration_ms}
    return None


def _emit_violation(tool_name: str, dur_ms: int, violation: dict) -> None:
    """Emit a PERF_BUDGET_VIOLATION event. Fails silently."""
    try:
        from modules.events import event_bus, Events
        event_bus.emit(Events.PERF_BUDGET_VIOLATION, {
            "tool_name": tool_name,
            "duration_ms": dur_ms,
            **violation,
        })
    except Exception:
        _logger.exception("Failed to emit PERF_BUDGET_VIOLATION for tool=%s dur=%dms", tool_name, dur_ms)


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
                        # Redact secrets from string responses before returning to client
                        if isinstance(result, str):
                            result = redact_secrets(result)
                        dur_ms = int((time.perf_counter() - t0) * 1000)
                        res_size = _safe_len_bytes(result)
                        _log_tool_call_safe(
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
                        violation = check_budget_violation(tool_name, dur_ms)
                        if violation:
                            _emit_violation(tool_name, dur_ms, violation)
                        return result
                    except Exception as e:
                        dur_ms = int((time.perf_counter() - t0) * 1000)
                        _log_tool_call_safe(
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
                    # Redact secrets from string responses before returning to client
                    if isinstance(result, str):
                        result = redact_secrets(result)
                    dur_ms = int((time.perf_counter() - t0) * 1000)
                    res_size = _safe_len_bytes(result)
                    _log_tool_call_safe(
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
                    violation = check_budget_violation(tool_name, dur_ms)
                    if violation:
                        _emit_violation(tool_name, dur_ms, violation)
                    return result
                except Exception as e:
                    dur_ms = int((time.perf_counter() - t0) * 1000)
                    _log_tool_call_safe(
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
    dt = datetime.now().astimezone() - timedelta(days=int(days))
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


# ============================================================
# PERF REPORT MCP TOOL
# ============================================================

def perf_report(days: int = 7) -> str:
    """Performance report: p50/p95/p99 per tool vs budget contracts."""
    summary = tool_usage_summary(days=days)
    lines = [f"# Performance Report ({days}d)\n"]

    violations = []
    for tool_info in summary["tools"]:
        name = tool_info["tool"]
        pcts = compute_percentiles(name, days=days)
        contract = get_tool_contract(name)
        status = "OK"
        if pcts["p95"] > contract["p95"]:
            status = "VIOLATION p95"
            violations.append(name)
        elif pcts["p99"] > contract["p99"]:
            status = "VIOLATION p99"
            violations.append(name)
        lines.append(
            f"| {name} | {pcts['count']} calls | "
            f"p50={pcts['p50']}ms p95={pcts['p95']}ms p99={pcts['p99']}ms | "
            f"budget p95<{contract['p95']}ms p99<{contract['p99']}ms | {status} |"
        )

    lines.insert(1, f"Violations: {len(violations)}/{len(summary['tools'])} tools\n")
    return "\n".join(lines)


def register_metrics_tools(mcp_server: Any) -> None:
    """Register performance-related MCP tools."""

    @mcp_server.tool()
    def perf_report_tool(days: int = 7) -> str:
        """Performance report: p50/p95/p99 per tool vs budget contracts.
        Shows which tools are within or violating their latency budgets."""
        return perf_report(days=days)

    @mcp_server.tool()
    def tool_usage_summary_tool(days: int = 7) -> str:
        """Tool usage summary: call counts, success/fail rates, avg latency."""
        import json
        return json.dumps(tool_usage_summary(days=days), indent=2)

    @mcp_server.tool()
    def embed_cache_stats() -> str:
        """Show embedding cache stats (hits, misses, hit rate, size)."""
        import json
        from modules.consolidation import get_embed_cache_info
        return json.dumps(get_embed_cache_info(), indent=2)
