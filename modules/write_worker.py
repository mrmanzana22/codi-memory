"""
write_worker.py - Background worker for async write-path (E2.3 PR2)
=====================================================================
Drains the write_queue, executing memory operations with retries.

Design:
  - Single-threaded claim loop (multi-process safe via lease)
  - Exponential backoff with jitter on failure
  - Dead letter after max_attempts
  - Idempotent execution (dedupe_key prevents duplicate memories)
  - Instrumented: every job logged to write_queue_log + tool_calls

Supported job kinds:
  - remember: add_memory_smart() (dedup + embed + Qdrant)
  - add_memory: memory.add() + enrich + FTS index
  - checkpoint_memoria: memory.add() + enrich + backup + journal

Usage:
  python -m modules.write_worker [--once] [--poll-interval 5]

Created: 2026-02-16 (E2.3 PR2)
"""

from __future__ import annotations

import argparse
import json
import os
import random
import signal
import sys
import time
from datetime import datetime

# Allow imports when run as CLI
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import logging

from modules.config import now_iso
from modules.secret_redact import redact_secrets
from modules.write_queue import (
    claim_next_job,
    mark_job_started,
    mark_job_done,
    mark_job_failed,
    mark_job_canceled,
    check_cancel_requested,
    log_job_completion,
    get_queue_stats,
    reap_stuck_jobs,
)


# ============================================================
# CONSTANTS
# ============================================================

DEFAULT_POLL_INTERVAL = 5    # seconds between polls when queue is empty
DEFAULT_LEASE_SECONDS = 300  # 5 min lease (generous for slow LLM calls)
MAX_BACKOFF = 300            # max seconds between retries
MAX_JOBS_PER_TICK = 10       # max jobs to drain per tick
MAX_DRAIN_SECONDS = 20       # max seconds to spend draining per tick

_logger = logging.getLogger(__name__)

_shutdown = False


def _handle_signal(signum, frame):
    global _shutdown
    _shutdown = True
    _logger.info("Received signal %s, shutting down after current job...", signum)


# ============================================================
# SCHEMA CHECK (Guard #2: fail-fast if migration 004 not applied)
# ============================================================

def _check_schema():
    """Verify write_queue has latency columns. Raises RuntimeError if not."""
    import sqlite3
    from modules.config import FTS_DB_PATH

    conn = sqlite3.connect(FTS_DB_PATH, timeout=5)
    try:
        cols = {r[1] for r in conn.execute("PRAGMA table_info(write_queue)").fetchall()}
        missing = {"claimed_at", "started_at"} - cols
        if missing:
            raise RuntimeError(
                f"write_queue missing columns: {missing}. "
                f"Run: apply_migrations(FTS_DB_PATH, 'migrations') to apply 004."
            )
    finally:
        conn.close()


# ============================================================
# FAILURE CLASSIFICATION
# ============================================================

FAILURE_REASONS = {
    "timeout", "rate_limited", "auth", "upstream_5xx",
    "conn_error", "payload_error", "unknown",
}


def classify_exception(e: Exception) -> str:
    """Classify an exception into a failure reason category.

    Returns one of: timeout, rate_limited, auth, upstream_5xx,
    conn_error, payload_error, unknown.
    """
    cls = type(e).__name__.lower()
    msg = str(e).lower()

    # Timeout variants
    if "timeout" in cls or "timeout" in msg or "timed out" in msg:
        return "timeout"

    # Rate limiting
    if "ratelimit" in cls or "rate_limit" in cls or "429" in msg or "rate limit" in msg:
        return "rate_limited"

    # Authentication / authorization
    if "401" in msg or "403" in msg or "unauthorized" in msg or "forbidden" in msg or "auth" in cls:
        return "auth"

    # Upstream server errors (5xx)
    if any(code in msg for code in ("500", "502", "503", "504")) or "server error" in msg:
        return "upstream_5xx"

    # Connection errors
    if any(k in cls for k in ("connection", "socket", "dns", "resolve")):
        return "conn_error"
    if any(k in msg for k in ("connection refused", "connection reset", "name resolution", "unreachable")):
        return "conn_error"

    # Payload / data errors
    if any(k in cls for k in ("key", "value", "type", "json", "validation", "attribute")):
        return "payload_error"
    if isinstance(e, (KeyError, ValueError, TypeError)):
        return "payload_error"

    return "unknown"


# ============================================================
# JOB EXECUTORS
# ============================================================

def _execute_remember(payload: dict) -> dict:
    """Execute a 'remember' job: add_memory_smart (dedup + embed + Qdrant).

    Payload keys: content, category/topic, source, importance
    """
    from modules.memory_smart import add_memory_smart

    content = payload["content"]
    category = payload.get("category") or payload.get("topic", "general")
    source = payload.get("source", "experienced")
    importance = payload.get("importance", "medium")

    # Map source for memory_smart
    ms_source = "experienced"
    if source in ("reflection", "prediction", "consolidation"):
        ms_source = "inferred"

    result_str = add_memory_smart(
        content=content,
        category=category,
        source=ms_source,
        importance=importance,
    )

    result = json.loads(result_str) if isinstance(result_str, str) else result_str
    return {"action": result.get("action", "unknown"), "result": str(result_str)[:200]}


def _execute_add_memory(payload: dict) -> dict:
    """Execute an 'add_memory' job: memory.add + enrich + FTS."""
    from modules.memory_core import add_memory

    content = payload["content"]
    category = payload.get("category", "general")
    source = payload.get("source", "experienced")
    importance = payload.get("importance", "medium")

    result_str = add_memory(
        content=content,
        category=category,
        source=source,
        importance=importance,
    )

    return {"result": str(result_str)[:200]}


def _execute_checkpoint_memoria(payload: dict) -> dict:
    """Execute a 'checkpoint_memoria' job: full checkpoint pipeline."""
    from modules.flush import _checkpoint_memoria

    momento = payload["momento"]
    que_paso = payload["que_paso"]
    por_que_importa = payload["por_que_importa"]

    result_str = _checkpoint_memoria(momento, que_paso, por_que_importa)

    return {"result": str(result_str)[:200]}


# Dispatch table
JOB_EXECUTORS = {
    "remember": _execute_remember,
    "add_memory": _execute_add_memory,
    "checkpoint_memoria": _execute_checkpoint_memoria,
}


# ============================================================
# DUAL-COMPARE HOOK (best-effort, never fails the job)
# ============================================================

def _dual_compare_hook(
    job_id: str,
    result_dict: dict = None,
    error: str = None,
    failure_reason: str = None,
) -> None:
    """Best-effort dual-compare update after job completion.

    If the dual_compare_log row doesn't exist (sync path didn't record it,
    or dual mode is off), logs 'dual_link_missing' and moves on.
    Never raises — the job outcome is already committed.
    """
    try:
        from modules.dual_compare import update_async_result, compute_comparison_for_job

        outcome = update_async_result(
            job_id=job_id,
            result_dict=result_dict,
            error=error,
            failure_reason=failure_reason,
        )

        if outcome == "dual_link_missing":
            # Expected when dual mode is off or sync path didn't record
            return

        if outcome == "updated":
            compute_comparison_for_job(job_id)

    except Exception as e:
        _logger.warning("dual_compare_hook: %s: %s", type(e).__name__, redact_secrets(str(e)[:80]))


# ============================================================
# PROCESS ONE JOB
# ============================================================

def process_one_job(lease_seconds: int = DEFAULT_LEASE_SECONDS) -> bool:
    """Claim and process one job from the queue.

    Returns True if a job was processed (or skipped), False if queue was empty.
    """
    claimed = claim_next_job(lease_seconds=lease_seconds)
    if not claimed:
        return False

    job_id = claimed["job_id"]
    kind = claimed["kind"]
    payload = claimed["payload"]
    attempts = claimed["attempts"]

    # Cancel check #1: after claim, before any work
    if check_cancel_requested(job_id):
        mark_job_canceled(job_id, reason="cancel_before_exec")
        log_job_completion(job_id, kind, "canceled", attempts,
                           error_class="CancelTombstone",
                           error_msg="cancel_requested before execution")
        _logger.info("%s %s canceled (tombstone, pre-exec)", kind, job_id[:8])
        return True

    _logger.info("Processing %s job %s... (attempt %s)", kind, job_id[:8], attempts)

    executor = JOB_EXECUTORS.get(kind)
    if not executor:
        mark_job_failed(job_id, f"Unknown job kind: {kind}")
        log_job_completion(job_id, kind, "dead", attempts, error_class="UnknownKind",
                           error_msg=f"No executor for kind={kind}")
        return True

    # Record started_at for latency breakdown (queue_wait vs exec)
    mark_job_started(job_id)

    # Cancel check #2: after started_at, right before executor
    # (narrows the race window between claim and execution)
    if check_cancel_requested(job_id):
        mark_job_canceled(job_id, reason="cancel_before_exec")
        log_job_completion(job_id, kind, "canceled", attempts,
                           error_class="CancelTombstone",
                           error_msg="cancel_requested before execution (post-start)")
        _logger.info("%s %s canceled (tombstone, post-start)", kind, job_id[:8])
        return True

    start = time.monotonic()
    try:
        result = executor(payload)
        elapsed_ms = round((time.monotonic() - start) * 1000)

        # Cancel check #3: after execution completed
        # If cancel arrived during exec, log as late_cancel_race but don't undo
        # (for remember/add_memory this is tolerable due to dedupe)
        if check_cancel_requested(job_id):
            mark_job_done(job_id)
            log_job_completion(job_id, kind, "done", attempts,
                               duration_ms=elapsed_ms,
                               error_class="LateCancelRace",
                               error_msg="cancel_requested arrived during execution")
            _dual_compare_hook(job_id, result_dict=result)
            _logger.info("%s %s done in %sms (late_cancel_race)", kind, job_id[:8], elapsed_ms)
            return True

        mark_job_done(job_id)
        log_job_completion(job_id, kind, "done", attempts, duration_ms=elapsed_ms)

        # Dual-compare: best-effort update (never fails the job)
        _dual_compare_hook(job_id, result_dict=result)

        _logger.info("%s %s done in %sms", kind, job_id[:8], elapsed_ms)
        return True

    except Exception as e:
        elapsed_ms = round((time.monotonic() - start) * 1000)
        error_class = type(e).__name__
        error_msg = str(e)[:200]
        failure_reason = classify_exception(e)

        new_status = mark_job_failed(job_id, f"{error_class}: {error_msg}",
                                     failure_reason=failure_reason)
        log_job_completion(job_id, kind, new_status, attempts,
                           duration_ms=elapsed_ms,
                           error_class=error_class,
                           error_msg=error_msg,
                           failure_reason=failure_reason)

        # Dual-compare: record async failure (best-effort)
        _dual_compare_hook(job_id, error=error_msg, failure_reason=failure_reason)

        _logger.error("%s %s %s [%s]: %s: %s", kind, job_id[:8], new_status, failure_reason, error_class, redact_secrets(error_msg[:80]))
        return True


# ============================================================
# BACKOFF
# ============================================================

def _backoff_seconds(attempt: int) -> float:
    """Exponential backoff with jitter: 10s, 30s, 2m, 5m..."""
    base = min(10 * (3 ** (attempt - 1)), MAX_BACKOFF)
    jitter = random.uniform(0, base * 0.3)
    return base + jitter


# ============================================================
# DRAIN TICK
# ============================================================

def drain_tick(
    lease_seconds: int = DEFAULT_LEASE_SECONDS,
    max_jobs: int = MAX_JOBS_PER_TICK,
    max_seconds: float = MAX_DRAIN_SECONDS,
) -> dict:
    """Drain up to max_jobs or until max_seconds elapsed per tick.

    Runs reap_stuck_jobs first, then processes queued jobs.
    Cuts on whichever limit hits first. Prevents monopolizing the daemon
    when a large backlog accumulates.

    Returns:
        dict with processed count and reap_result.
    """
    # Reap stuck jobs before processing new ones
    reap_result = reap_stuck_jobs(max_exec_seconds=180)
    if reap_result["requeued"] > 0 or reap_result["dead"] > 0:
        _logger.info("Reaped stuck: %s requeued, %s dead", reap_result['requeued'], reap_result['dead'])

    tick_start = time.monotonic()
    processed = 0

    while processed < max_jobs:
        if time.monotonic() - tick_start >= max_seconds:
            break

        did_work = process_one_job(lease_seconds=lease_seconds)
        if not did_work:
            break  # queue empty

        processed += 1

    return {"processed": processed, "reap": reap_result}


# ============================================================
# MAIN LOOP
# ============================================================

def _emit_worker_tick(tick_result: dict) -> None:
    """Emit a worker heartbeat to tool_calls for liveness monitoring.

    Recorded as tool_name='write_worker_tick' so assessment_report()
    can detect stale workers (no tick in >10 min).
    """
    try:
        from modules.metrics import log_tool_call
        reap = tick_result.get("reap", {})
        tag = json.dumps({
            "processed": tick_result.get("processed", 0),
            "requeued": reap.get("requeued", 0),
            "dead": reap.get("dead", 0),
        })
        log_tool_call(
            tool_name="write_worker_tick",
            started_at=now_iso(),
            duration_ms=0,
            success=True,
            error_type=None,
            args_size=0,
            result_size=len(tag),
            tag=tag,
        )
    except Exception as e:
        _logger.error("heartbeat emit failed: %s", redact_secrets(str(e)))


def run_worker_loop(
    poll_interval: int = DEFAULT_POLL_INTERVAL,
    lease_seconds: int = DEFAULT_LEASE_SECONDS,
    max_jobs: int = 0,
) -> dict:
    """Run the worker loop, processing jobs until shutdown.

    Uses drain_tick() to process multiple jobs per poll cycle,
    reducing queue wait time under load. Emits a heartbeat
    (write_worker_tick) to tool_calls on every iteration.

    Args:
        poll_interval: Seconds between polls when queue is empty
        lease_seconds: Lease duration per job
        max_jobs: Stop after N jobs (0 = unlimited)

    Returns:
        dict with processed, failed, elapsed_s
    """
    global _shutdown

    start = time.monotonic()
    processed = 0
    failed = 0

    while not _shutdown:
        tick_result = drain_tick(lease_seconds=lease_seconds)
        drained = tick_result["processed"]

        # Emit heartbeat every tick (even when idle)
        _emit_worker_tick(tick_result)

        if drained > 0:
            processed += drained
            if drained > 1:
                _logger.info("Drained %s jobs in tick", drained)
        else:
            # Queue empty, wait before polling again
            time.sleep(poll_interval)

        if max_jobs > 0 and processed >= max_jobs:
            break

    elapsed_s = round(time.monotonic() - start)
    return {"processed": processed, "failed": failed, "elapsed_s": elapsed_s}


# ============================================================
# CLI
# ============================================================

def cli_main():
    """CLI entry point: python -m modules.write_worker [options]"""
    parser = argparse.ArgumentParser(
        description="Codi Write Worker - background memory writer"
    )
    parser.add_argument(
        "--once", action="store_true",
        help="Process one job and exit (for testing/cron)"
    )
    parser.add_argument(
        "--drain", action="store_true",
        help="Process all queued jobs and exit"
    )
    parser.add_argument(
        "--poll-interval", type=int, default=DEFAULT_POLL_INTERVAL,
        help=f"Seconds between polls (default {DEFAULT_POLL_INTERVAL})"
    )
    parser.add_argument(
        "--stats", action="store_true",
        help="Show queue stats and exit"
    )
    args = parser.parse_args()

    if args.stats:
        stats = get_queue_stats()
        _logger.info("Queue stats:\n%s", json.dumps(stats, indent=2))
        return

    # Guard #2: fail-fast if migration 004 not applied
    _check_schema()

    # Set up signal handlers for graceful shutdown
    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)

    if args.once:
        did_work = process_one_job()
        _logger.info("%s", "Processed 1 job" if did_work else "Queue empty")
        sys.exit(0 if did_work else 2)

    if args.drain:
        tick_result = drain_tick()
        count = tick_result["processed"]
        _emit_worker_tick(tick_result)
        _logger.info("Drained %s jobs", count)
        sys.exit(0)

    # Long-running loop
    _logger.info("Starting (poll=%ss)", args.poll_interval)
    result = run_worker_loop(poll_interval=args.poll_interval)
    _logger.info("Stopped: %s", result)


if __name__ == "__main__":
    cli_main()
