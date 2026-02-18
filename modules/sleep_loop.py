"""
Codi Memory - Sleep Loop Module
=================================
Background process that runs between sessions to perform memory
consolidation, homeostasis decay, prospective memory review,
and health checks.

Writes a sleep_report to the latest session_checkpoints row.

Design:
  - CLI + launchd: runs every 30 min via launchd plist
  - 8s budget: 4 ticks with hard timeouts
  - Lock file: data/sleep_loop.lock prevents double execution
  - Idempotent: only writes report to checkpoints without one

Neuroscience basis:
  - Sleep consolidation (Diekelmann & Born 2010)
  - Synaptic homeostasis hypothesis (Tononi & Cirelli 2003)
  - Prospective memory maintenance (McDaniel & Einstein 2007)

Created: 2026-02-16 (Sleep Loop MVP)
"""

import argparse
import json
import logging
import os
import signal
import sqlite3
import sys
import time
from datetime import datetime, timedelta

_logger = logging.getLogger(__name__)

# Allow imports when run as CLI (-m modules.sleep_loop)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from modules.config import FTS_DB_PATH, DATA_DIR, TZ_COL, now_col, now_iso
from modules.events import event_bus, Events

# ============================================================
# TICK-LEVEL METRICS (E2.2)
# ============================================================

def _log_tick_metric(tick_name: str, elapsed_ms: int, budget_ms: int,
                     remaining_ms: int, status: str, reason: str = None):
    """Log 1 row per tick to tool_calls table for perf analysis.

    Status values: ok, over_budget, skipped, error
    Lightweight: 1 INSERT per tick, compact payload.
    """
    try:
        from modules.metrics import log_tool_call
        tag = f"sleep_tick:{status}"
        if reason:
            tag += f":{reason}"
        log_tool_call(
            tool_name=f"sleep_tick_{tick_name}",
            started_at=now_iso(),
            duration_ms=int(elapsed_ms),
            success=(status == "ok"),
            error_type=reason if status == "error" else None,
            args_size=0,
            result_size=0,
            session_id=None,
            tag=tag,
        )
    except Exception:
        pass  # Never let logging break the loop

# ============================================================
# CONSTANTS
# ============================================================

LOCK_FILE = os.path.join(DATA_DIR, "sleep_loop.lock")
DEFAULT_BUDGET_MS = 8000
DEFAULT_MAX_AGE_MIN = 30   # Only run if checkpoint < 30 min old w/o report

# Tick order: fast first, heavy last (so budget exhaustion doesn't starve fast ticks)
TICK_ORDER = ["prospective", "health", "consolidation", "homeostasis"]

# Minimum ms required to even attempt a tick (below this, skip)
TICK_MIN_MS = {
    "prospective": 200,
    "health": 200,
    "consolidation": 1500,
    "homeostasis": 200,
}


# ============================================================
# LOCK FILE
# ============================================================

def _acquire_lock() -> bool:
    """Acquire PID-based lock file. Returns True if acquired."""
    os.makedirs(DATA_DIR, exist_ok=True)

    if os.path.exists(LOCK_FILE):
        try:
            with open(LOCK_FILE, 'r') as f:
                old_pid = int(f.read().strip())
            # Check if old process is still alive
            os.kill(old_pid, 0)
            # Process alive -- don't run
            return False
        except (ProcessLookupError, ValueError, OSError):
            # Process dead or invalid PID -- stale lock, safe to remove
            os.remove(LOCK_FILE)

    with open(LOCK_FILE, 'w') as f:
        f.write(str(os.getpid()))
    return True


def _release_lock():
    """Release lock file."""
    try:
        if os.path.exists(LOCK_FILE):
            with open(LOCK_FILE, 'r') as f:
                pid = int(f.read().strip())
            if pid == os.getpid():
                os.remove(LOCK_FILE)
    except Exception:
        pass


# ============================================================
# DATABASE HELPERS
# ============================================================

def _get_conn():
    """Get WAL-mode SQLite connection."""
    conn = sqlite3.connect(FTS_DB_PATH, timeout=5)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA busy_timeout=3000")
    return conn


def _get_target_checkpoint(max_age_min: int) -> int | None:
    """Find latest checkpoint that needs a sleep_report.

    Returns checkpoint ID if found, None otherwise.
    Only considers checkpoints within max_age_min that have no report.
    """
    conn = _get_conn()
    try:
        row = conn.execute(
            "SELECT id, created_at, sleep_report FROM session_checkpoints "
            "ORDER BY created_at DESC LIMIT 1"
        ).fetchone()

        if not row:
            return None

        cp_id, created_at, sleep_report = row

        # Already has a report?
        if sleep_report and sleep_report.strip():
            return None

        # Too old?
        try:
            dt = datetime.fromisoformat(created_at)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=TZ_COL)
            age_min = (now_col() - dt).total_seconds() / 60
            if age_min > max_age_min:
                return None
        except Exception:
            return None

        return cp_id
    finally:
        conn.close()


def _write_sleep_report(checkpoint_id: int, report_text: str) -> bool:
    """Idempotent write of sleep_report to checkpoint row.

    Only writes if sleep_report is NULL or empty (no clobber).
    """
    conn = _get_conn()
    try:
        cursor = conn.execute(
            "UPDATE session_checkpoints SET sleep_report = ? "
            "WHERE id = ? AND (sleep_report IS NULL OR sleep_report = '')",
            (report_text, checkpoint_id)
        )
        conn.commit()
        return cursor.rowcount > 0
    finally:
        conn.close()


# ============================================================
# TICK FUNCTIONS
# ============================================================

def _tick_consolidation(budget_ms: int) -> dict:
    """Tick 1: Light consolidation (no LLM, clustering only)."""
    start = time.monotonic()
    result = {"tick": "consolidation", "ok": False, "detail": ""}

    try:
        from modules.consolidation import run_consolidation
        report = run_consolidation(scope="light", lookback_hours=6)
        elapsed = (time.monotonic() - start) * 1000
        result["ok"] = True
        result["detail"] = report[:200] if report else "no output"
        result["elapsed_ms"] = round(elapsed)
    except Exception as e:
        result["detail"] = f"error: {str(e)[:100]}"
        result["elapsed_ms"] = round((time.monotonic() - start) * 1000)

    return result


def _tick_homeostasis(budget_ms: int) -> dict:
    """Tick 2: Synaptic homeostasis -- salience decay + emotional decay."""
    start = time.monotonic()
    result = {"tick": "homeostasis", "ok": False, "detail": ""}

    parts = []

    # Salience decay
    try:
        from modules.consciousness import apply_salience_decay
        sal_report = apply_salience_decay(decay_rate=0.05)
        parts.append(f"salience: {sal_report[:80]}")
    except Exception as e:
        parts.append(f"salience: error {str(e)[:50]}")

    # Emotional decay (PAD toward baseline)
    try:
        from modules.consciousness import apply_emotional_decay
        emo_report = apply_emotional_decay()
        parts.append(f"emotional: {emo_report[:80]}")
    except Exception as e:
        parts.append(f"emotional: error {str(e)[:50]}")

    elapsed = (time.monotonic() - start) * 1000
    result["ok"] = True
    result["detail"] = "; ".join(parts)
    result["elapsed_ms"] = round(elapsed)
    return result


def _tick_prospective(budget_ms: int) -> dict:
    """Tick 3: Prospective memory -- intention decay + maintenance."""
    start = time.monotonic()
    result = {"tick": "prospective", "ok": False, "detail": ""}

    try:
        from modules.prospective import apply_intention_maintenance
        apply_intention_maintenance()
        elapsed = (time.monotonic() - start) * 1000
        result["ok"] = True
        result["detail"] = "intention maintenance applied"
        result["elapsed_ms"] = round(elapsed)
    except Exception as e:
        result["detail"] = f"error: {str(e)[:100]}"
        result["elapsed_ms"] = round((time.monotonic() - start) * 1000)

    return result


def _tick_health(budget_ms: int) -> dict:
    """Tick 4: Health check + FTS queue processing."""
    start = time.monotonic()
    result = {"tick": "health", "ok": False, "detail": ""}

    parts = []

    # FTS retry queue
    try:
        from modules.memory_smart import process_fts_queue
        fts_result = process_fts_queue(limit=50)
        processed = fts_result.get("processed", 0)
        if processed > 0:
            parts.append(f"fts: {fts_result.get('succeeded', 0)} OK, {fts_result.get('failed', 0)} failed")
        else:
            parts.append("fts: queue empty")
    except Exception as e:
        parts.append(f"fts: error {str(e)[:50]}")

    # Health check
    try:
        from modules.consciousness import _verificar_salud_memoria_interna
        health = _verificar_salud_memoria_interna()
        if health.get("ok"):
            parts.append(f"health: OK ({health.get('total_memories', '?')} mems)")
        else:
            parts.append(f"health: {health.get('message', 'unknown')[:60]}")
    except Exception as e:
        parts.append(f"health: error {str(e)[:50]}")

    elapsed = (time.monotonic() - start) * 1000
    result["ok"] = True
    result["detail"] = "; ".join(parts)
    result["elapsed_ms"] = round(elapsed)
    return result


# ============================================================
# FORMAT REPORT
# ============================================================

def format_sleep_report(tick_results: list, total_ms: int, reason: str) -> str:
    """Format tick results into a human-readable sleep report (300-600 chars)."""
    lines = [f"SLEEP REPORT ({reason}, {total_ms}ms)"]

    for r in tick_results:
        status = "OK" if r.get("ok") else "FAIL"
        lines.append(f"- {r['tick']}: {status} ({r.get('elapsed_ms', '?')}ms) {r.get('detail', '')[:80]}")

    report = "\n".join(lines)
    # Cap at 600 chars
    if len(report) > 600:
        report = report[:597] + "..."
    return report


# ============================================================
# MAIN SLEEP LOOP
# ============================================================

def run_sleep_loop(reason: str = "idle", budget_ms: int = DEFAULT_BUDGET_MS) -> dict:
    """Execute the full sleep loop with 4 ticks.

    Args:
        reason: Why this run was triggered ('launchd', 'idle', 'manual')
        budget_ms: Total time budget in milliseconds (default 8000)

    Returns:
        dict with ok, report, checkpoint_id, elapsed_ms
    """
    start = time.monotonic()

    # Tick dispatch table
    tick_dispatch = {
        "prospective": _tick_prospective,
        "health": _tick_health,
        "consolidation": _tick_consolidation,
        "homeostasis": _tick_homeostasis,
    }

    tick_results = []
    for name in TICK_ORDER:
        func = tick_dispatch[name]
        min_required = TICK_MIN_MS.get(name, 200)

        # Budget gating: how much time remains?
        elapsed_total = (time.monotonic() - start) * 1000
        remaining_ms = budget_ms - elapsed_total

        if remaining_ms < min_required:
            tick_results.append({
                "tick": name, "ok": False,
                "detail": f"skipped (need {min_required}ms, only {int(remaining_ms)}ms left)",
                "elapsed_ms": 0,
                "status": "skipped",
            })
            _log_tick_metric(name, 0, budget_ms, int(remaining_ms),
                             "skipped", "budget_exhausted")
            continue

        try:
            result = func(int(remaining_ms))
            tick_elapsed = result.get("elapsed_ms", 0)

            # Detect over_budget: tick took more than its remaining allocation
            if tick_elapsed > remaining_ms:
                result["status"] = "over_budget"
                _log_tick_metric(name, tick_elapsed, budget_ms,
                                 int(remaining_ms), "over_budget")
            else:
                result["status"] = "ok"
                _log_tick_metric(name, tick_elapsed, budget_ms,
                                 int(remaining_ms), "ok")

            tick_results.append(result)
        except Exception as e:
            tick_results.append({
                "tick": name, "ok": False,
                "detail": f"unhandled: {str(e)[:80]}",
                "elapsed_ms": 0,
                "status": "error",
            })
            _log_tick_metric(name, 0, budget_ms, int(remaining_ms),
                             "error", type(e).__name__)

    total_ms = round((time.monotonic() - start) * 1000)
    report_text = format_sleep_report(tick_results, total_ms, reason)

    # Emit event (best-effort, won't fail if event bus unavailable in CLI)
    try:
        event_bus.emit(Events.SLEEP_LOOP_COMPLETE, {
            "reason": reason,
            "elapsed_ms": total_ms,
            "ticks_ok": sum(1 for t in tick_results if t.get("ok")),
            "ticks_total": len(tick_results),
        })
    except Exception:
        pass

    return {
        "ok": True,
        "report": report_text,
        "tick_results": tick_results,
        "elapsed_ms": total_ms,
    }


# ============================================================
# CLI ENTRY POINT
# ============================================================

def cli_main():
    """CLI entry point: python -m modules.sleep_loop [options]"""
    parser = argparse.ArgumentParser(
        description="Codi Sleep Loop - background memory maintenance"
    )
    parser.add_argument(
        "--reason", default="launchd",
        help="Why this run was triggered (launchd|idle|manual)"
    )
    parser.add_argument(
        "--max-age-min", type=int, default=DEFAULT_MAX_AGE_MIN,
        help=f"Only run if latest checkpoint is < N minutes old (default {DEFAULT_MAX_AGE_MIN})"
    )
    parser.add_argument(
        "--budget-ms", type=int, default=DEFAULT_BUDGET_MS,
        help=f"Total time budget in milliseconds (default {DEFAULT_BUDGET_MS})"
    )
    args = parser.parse_args()

    # Pre-checks
    if not os.path.exists(FTS_DB_PATH):
        _logger.error("FTS DB not found: %s", FTS_DB_PATH)
        sys.exit(0)

    # Check if there's a checkpoint that needs a report
    target_id = _get_target_checkpoint(args.max_age_min)
    if target_id is None:
        _logger.info("No eligible checkpoint (all have reports or too old). Skipping.")
        sys.exit(0)

    # Acquire lock
    if not _acquire_lock():
        _logger.warning("Another instance is running. Skipping.")
        sys.exit(0)

    try:
        _logger.info("Starting (reason=%s, budget=%dms, checkpoint=%s)", args.reason, args.budget_ms, target_id)

        result = run_sleep_loop(reason=args.reason, budget_ms=args.budget_ms)

        # Write report to checkpoint
        if result.get("report"):
            written = _write_sleep_report(target_id, result["report"])
            if written:
                _logger.info("Report written to checkpoint %s", target_id)
            else:
                _logger.info("Report NOT written (checkpoint %s already has one)", target_id)

        _logger.info("Done in %sms", result.get("elapsed_ms", "?"))
        _logger.info("Sleep report:\n%s", result.get("report", ""))

    finally:
        _release_lock()


# ============================================================
# MCP TOOL REGISTRATION
# ============================================================

def register_tools(mcp):
    """Register sleep loop diagnostic tool."""

    @mcp.tool()
    def sleep_report_status(days: int = 7) -> str:
        """Show recent sleep reports from session checkpoints. Useful for diagnosing background maintenance."""
        try:
            conn = _get_conn()

            cutoff = (now_col() - timedelta(days=days)).isoformat()
            rows = conn.execute(
                "SELECT id, created_at, source, sleep_report "
                "FROM session_checkpoints "
                "WHERE created_at > ? "
                "ORDER BY created_at DESC LIMIT 20",
                (cutoff,)
            ).fetchall()

            conn.close()

            if not rows:
                return f"No checkpoints in the last {days} days."

            lines = [f"# SLEEP REPORTS (last {days} days)", ""]

            with_report = 0
            without_report = 0

            for r in rows:
                cp_id, ts, source, report = r
                has_report = bool(report and report.strip())
                if has_report:
                    with_report += 1
                else:
                    without_report += 1

                status = "HAS REPORT" if has_report else "NO REPORT"
                ts_short = ts[:16] if ts else "?"
                lines.append(f"- [{cp_id}] {ts_short} ({source}) [{status}]")
                if has_report:
                    # Show first 100 chars of report
                    lines.append(f"  {report[:100]}...")

            lines.insert(1, f"Total: {len(rows)} checkpoints, {with_report} with reports, {without_report} without")

            return "\n".join(lines)

        except Exception as e:
            return f"Sleep report status ERROR: {e}"


# ============================================================
# MODULE ENTRY POINT
# ============================================================

if __name__ == "__main__":
    cli_main()
