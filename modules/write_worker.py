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

from modules.config import now_iso
from modules.write_queue import (
    claim_next_job,
    mark_job_started,
    mark_job_done,
    mark_job_failed,
    log_job_completion,
    get_queue_stats,
)


# ============================================================
# CONSTANTS
# ============================================================

DEFAULT_POLL_INTERVAL = 5    # seconds between polls when queue is empty
DEFAULT_LEASE_SECONDS = 300  # 5 min lease (generous for slow LLM calls)
MAX_BACKOFF = 300            # max seconds between retries
MAX_JOBS_PER_TICK = 10       # max jobs to drain per tick
MAX_DRAIN_SECONDS = 20       # max seconds to spend draining per tick

_shutdown = False


def _handle_signal(signum, frame):
    global _shutdown
    _shutdown = True
    print(f"[write_worker] Received signal {signum}, shutting down after current job...")


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
# PROCESS ONE JOB
# ============================================================

def process_one_job(lease_seconds: int = DEFAULT_LEASE_SECONDS) -> bool:
    """Claim and process one job from the queue.

    Returns True if a job was processed, False if queue was empty.
    """
    claimed = claim_next_job(lease_seconds=lease_seconds)
    if not claimed:
        return False

    job_id = claimed["job_id"]
    kind = claimed["kind"]
    payload = claimed["payload"]
    attempts = claimed["attempts"]

    print(f"[write_worker] Processing {kind} job {job_id[:8]}... (attempt {attempts})")

    executor = JOB_EXECUTORS.get(kind)
    if not executor:
        mark_job_failed(job_id, f"Unknown job kind: {kind}")
        log_job_completion(job_id, kind, "dead", attempts, error_class="UnknownKind",
                           error_msg=f"No executor for kind={kind}")
        return True

    # Record started_at for latency breakdown (queue_wait vs exec)
    mark_job_started(job_id)

    start = time.monotonic()
    try:
        result = executor(payload)
        elapsed_ms = round((time.monotonic() - start) * 1000)

        mark_job_done(job_id)
        log_job_completion(job_id, kind, "done", attempts, duration_ms=elapsed_ms)

        print(f"[write_worker] {kind} {job_id[:8]} done in {elapsed_ms}ms")
        return True

    except Exception as e:
        elapsed_ms = round((time.monotonic() - start) * 1000)
        error_class = type(e).__name__
        error_msg = str(e)[:200]

        new_status = mark_job_failed(job_id, f"{error_class}: {error_msg}")
        log_job_completion(job_id, kind, new_status, attempts,
                           duration_ms=elapsed_ms,
                           error_class=error_class,
                           error_msg=error_msg)

        print(f"[write_worker] {kind} {job_id[:8]} {new_status}: {error_class}: {error_msg[:80]}")
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
) -> int:
    """Drain up to max_jobs or until max_seconds elapsed per tick.

    Cuts on whichever limit hits first. Prevents monopolizing the daemon
    when a large backlog accumulates.

    Returns:
        Number of jobs processed in this tick.
    """
    tick_start = time.monotonic()
    processed = 0

    while processed < max_jobs:
        if time.monotonic() - tick_start >= max_seconds:
            break

        did_work = process_one_job(lease_seconds=lease_seconds)
        if not did_work:
            break  # queue empty

        processed += 1

    return processed


# ============================================================
# MAIN LOOP
# ============================================================

def run_worker_loop(
    poll_interval: int = DEFAULT_POLL_INTERVAL,
    lease_seconds: int = DEFAULT_LEASE_SECONDS,
    max_jobs: int = 0,
) -> dict:
    """Run the worker loop, processing jobs until shutdown.

    Uses drain_tick() to process multiple jobs per poll cycle,
    reducing queue wait time under load.

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
        drained = drain_tick(lease_seconds=lease_seconds)

        if drained > 0:
            processed += drained
            if drained > 1:
                print(f"[write_worker] Drained {drained} jobs in tick")
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
        print(json.dumps(stats, indent=2))
        return

    # Guard #2: fail-fast if migration 004 not applied
    _check_schema()

    # Set up signal handlers for graceful shutdown
    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)

    if args.once:
        did_work = process_one_job()
        print(f"[write_worker] {'Processed 1 job' if did_work else 'Queue empty'}")
        sys.exit(0 if did_work else 2)

    if args.drain:
        count = 0
        while True:
            did_work = process_one_job()
            if not did_work:
                break
            count += 1
        print(f"[write_worker] Drained {count} jobs")
        sys.exit(0)

    # Long-running loop
    print(f"[write_worker] Starting (poll={args.poll_interval}s)")
    result = run_worker_loop(poll_interval=args.poll_interval)
    print(f"[write_worker] Stopped: {result}")


if __name__ == "__main__":
    cli_main()
