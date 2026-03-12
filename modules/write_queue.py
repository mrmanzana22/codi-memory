"""
write_queue.py - Async write-path queue (E2.3 PR1)
====================================================
Enqueue/status API for non-blocking memory operations.

Tools (remember, add_memory, checkpoint_memoria) enqueue jobs
here and return an immediate ACK. A background worker (PR2)
processes the queue with retries and observability.

Design:
  - SQLite table `write_queue` in memories_fts.db
  - Dedupe via SHA256(kind + normalized_text + day_bucket)
  - Lease-based claim for multi-process safety
  - Exponential backoff with jitter for retries
  - Dead letter after max_attempts (default 8)

Created: 2026-02-16 (E2.3)
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import uuid
from datetime import datetime, timedelta
from typing import Optional

from modules.config import FTS_DB_PATH, now_iso, now_col
from modules.db_pool import get_conn as _pool_get_conn


# ============================================================
# CONNECTION
# ============================================================

def _get_conn(db_path: str = None) -> sqlite3.Connection:
    """Get connection from thread-local pool (db_pool).

    Delegates to db_pool.get_conn() which manages lifecycle,
    PRAGMAs (WAL, synchronous=NORMAL, busy_timeout=5000), and
    thread-local reuse. Callers MUST NOT call conn.close().
    """
    return _pool_get_conn(db_path)


# ============================================================
# DEDUPE
# ============================================================

def compute_dedupe_key(kind: str, content: str) -> str:
    """Generate a dedupe key from kind + normalized content + day bucket.

    Two identical writes on the same day produce the same key.
    """
    day_bucket = now_col().strftime("%Y-%m-%d")
    normalized = content.strip().lower()[:500]
    raw = f"{kind}|{normalized}|{day_bucket}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:32]


# ============================================================
# ENQUEUE
# ============================================================

def enqueue_write_job(
    kind: str,
    payload: dict,
    priority: int = 5,
    dedupe_key: str = None,
    session_id: str = None,
    max_attempts: int = 8,
    db_path: str = None,
) -> dict:
    """Enqueue a write job for background processing.

    Args:
        kind: 'remember' | 'add_memory' | 'checkpoint_memoria'
        payload: Full args dict needed to execute the write
        priority: 1=highest, 10=lowest (default 5)
        dedupe_key: Optional dedupe key (auto-generated if not provided)
        session_id: Optional session_id for tracing
        max_attempts: Max retry attempts before marking dead
        db_path: Override DB path (for testing)

    Returns:
        dict with job_id, status, dedupe_hit (bool)
    """
    conn = _get_conn(db_path)
    now = now_iso()
    payload_json = json.dumps(payload, ensure_ascii=False, default=str)
    dedupe_source = json.dumps(payload, ensure_ascii=False, default=str, sort_keys=True)
    effective_dedupe_key = dedupe_key or compute_dedupe_key(kind, dedupe_source)

    # Check dedupe: if same key exists in active states, return existing job
    if effective_dedupe_key:
        row = conn.execute(
            "SELECT job_id, status FROM write_queue "
            "WHERE dedupe_key = ? AND status IN ('queued', 'running', 'done') "
            "ORDER BY created_at DESC LIMIT 1",
            (effective_dedupe_key,)
        ).fetchone()

        if row:
            return {
                "job_id": row["job_id"],
                "status": row["status"],
                "dedupe_hit": True,
            }

    job_id = str(uuid.uuid4())

    conn.execute(
        "INSERT INTO write_queue "
        "(job_id, kind, payload_json, status, priority, attempts, max_attempts, "
        " dedupe_key, session_id, created_at, updated_at) "
        "VALUES (?, ?, ?, 'queued', ?, 0, ?, ?, ?, ?, ?)",
        (job_id, kind, payload_json, priority, max_attempts,
         effective_dedupe_key, session_id, now, now),
    )
    conn.commit()

    return {
        "job_id": job_id,
        "status": "queued",
        "dedupe_hit": False,
    }


# ============================================================
# STATUS
# ============================================================

def get_write_job_status(job_id: str, db_path: str = None) -> dict | None:
    """Get detailed status of a write job including latency breakdown.

    Returns:
        dict with status, timestamps, latency breakdown, preview
        or None if job_id not found.
    """
    conn = _get_conn(db_path)
    row = conn.execute(
        "SELECT job_id, kind, status, attempts, max_attempts, "
        "last_error, failure_reason, payload_json, "
        "created_at, updated_at, claimed_at, started_at, completed_at "
        "FROM write_queue WHERE job_id = ?",
        (job_id,)
    ).fetchone()

    if not row:
        return None

    # Latency breakdown (if timestamps available)
    latency = {}
    try:
        if row["created_at"] and row["completed_at"]:
            c = datetime.fromisoformat(row["created_at"])
            f = datetime.fromisoformat(row["completed_at"])
            latency["total_ms"] = round((f - c).total_seconds() * 1000)
        if row["created_at"] and row["claimed_at"]:
            c = datetime.fromisoformat(row["created_at"])
            cl = datetime.fromisoformat(row["claimed_at"])
            latency["queue_wait_ms"] = round((cl - c).total_seconds() * 1000)
        if row["started_at"] and row["completed_at"]:
            s = datetime.fromisoformat(row["started_at"])
            f = datetime.fromisoformat(row["completed_at"])
            latency["exec_ms"] = round((f - s).total_seconds() * 1000)
    except Exception:
        pass

    # Preview: truncated content from payload (no secrets)
    preview = None
    try:
        payload = json.loads(row["payload_json"])
        content = payload.get("content", "")
        preview = content[:120] + "..." if len(content) > 120 else content
    except Exception:
        pass

    return {
        "job_id": row["job_id"],
        "kind": row["kind"],
        "status": row["status"],
        "attempts": row["attempts"],
        "max_attempts": row["max_attempts"],
        "last_error": row["last_error"],
        "failure_reason": row["failure_reason"],
        "preview": preview,
        "timestamps": {
            "created_at": row["created_at"],
            "claimed_at": row["claimed_at"],
            "started_at": row["started_at"],
            "completed_at": row["completed_at"],
        },
        "latency": latency or None,
    }


# ============================================================
# QUEUE STATS
# ============================================================

def get_queue_stats(db_path: str = None) -> dict:
    """Get summary stats of the write queue.

    Returns:
        dict with counts by status, oldest_queued, total.
    """
    conn = _get_conn(db_path)
    rows = conn.execute(
        "SELECT status, count(*) as cnt FROM write_queue GROUP BY status"
    ).fetchall()

    stats = {r["status"]: r["cnt"] for r in rows}
    total = sum(stats.values())

    oldest = conn.execute(
        "SELECT created_at FROM write_queue "
        "WHERE status = 'queued' ORDER BY created_at ASC LIMIT 1"
    ).fetchone()

    return {
        "total": total,
        "by_status": stats,
        "oldest_queued": oldest["created_at"] if oldest else None,
    }


# ============================================================
# CLAIM (for worker, PR2 will use this)
# ============================================================

def claim_next_job(
    lease_seconds: int = 120,
    db_path: str = None,
) -> dict | None:
    """Claim the next queued job with a lease.

    Uses UPDATE ... WHERE for atomic claim (no race conditions).
    Also reclaims jobs with expired leases (stale workers).

    Args:
        lease_seconds: How long the lease lasts (default 120s)
        db_path: Override DB path (for testing)

    Returns:
        dict with job_id, kind, payload (parsed), attempts
        or None if no jobs available.
    """
    conn = _get_conn(db_path)
    now = now_iso()
    lease_until = (now_col() + timedelta(seconds=lease_seconds)).isoformat()

    # First, reclaim stale leases (workers that died)
    conn.execute(
        "UPDATE write_queue SET status = 'queued', lease_until = NULL "
        "WHERE status = 'running' AND lease_until < ?",
        (now,)
    )

    # Find and claim next job atomically
    row = conn.execute(
        "SELECT id, job_id, kind, payload_json, attempts, session_id "
        "FROM write_queue "
        "WHERE status = 'queued' "
        "ORDER BY priority ASC, created_at ASC "
        "LIMIT 1"
    ).fetchone()

    if not row:
        conn.commit()
        return None

    # Claim it (record claimed_at for latency breakdown)
    cursor = conn.execute(
        "UPDATE write_queue "
        "SET status = 'running', lease_until = ?, updated_at = ?, "
        "    claimed_at = ?, attempts = attempts + 1 "
        "WHERE id = ? AND status = 'queued'",
        (lease_until, now, now, row["id"])
    )
    if cursor.rowcount == 0:
        conn.commit()
        return None
    conn.commit()

    payload = json.loads(row["payload_json"])
    return {
        "job_id": row["job_id"],
        "kind": row["kind"],
        "payload": payload,
        "attempts": row["attempts"] + 1,  # already incremented
        "session_id": row["session_id"],
    }


def mark_job_started(job_id: str, db_path: str = None) -> bool:
    """Record started_at timestamp just before job execution begins.

    Separates queue wait time from actual execution time.
    Returns True if the job was updated.
    """
    conn = _get_conn(db_path)
    now = now_iso()
    cursor = conn.execute(
        "UPDATE write_queue SET started_at = ? "
        "WHERE job_id = ? AND status = 'running'",
        (now, job_id)
    )
    conn.commit()
    return cursor.rowcount > 0


def mark_job_done(job_id: str, db_path: str = None) -> bool:
    """Mark a job as successfully completed.

    Returns True if the job was updated.
    """
    conn = _get_conn(db_path)
    now = now_iso()
    cursor = conn.execute(
        "UPDATE write_queue "
        "SET status = 'done', updated_at = ?, completed_at = ?, lease_until = NULL "
        "WHERE job_id = ? AND status = 'running'",
        (now, now, job_id)
    )
    conn.commit()
    return cursor.rowcount > 0


def mark_job_failed(
    job_id: str,
    error: str,
    failure_reason: str = None,
    db_path: str = None,
) -> str:
    """Mark a job as failed. Returns new status ('failed' or 'dead').

    If attempts >= max_attempts, marks as 'dead' (no more retries).
    Otherwise marks as 'failed' (will be retried).
    """
    conn = _get_conn(db_path)
    now = now_iso()
    row = conn.execute(
        "SELECT attempts, max_attempts FROM write_queue "
        "WHERE job_id = ? AND status = 'running'",
        (job_id,)
    ).fetchone()

    if not row:
        return "not_found"

    new_status = "dead" if row["attempts"] >= row["max_attempts"] else "failed"

    # failed jobs go back to queued for retry, dead stays dead
    final_status = "queued" if new_status == "failed" else "dead"

    conn.execute(
        "UPDATE write_queue "
        "SET status = ?, last_error = ?, failure_reason = ?, "
        "    updated_at = ?, lease_until = NULL, "
        "    claimed_at = NULL, started_at = NULL "
        "WHERE job_id = ? AND status = 'running'",
        (final_status, error[:500], failure_reason, now, job_id)
    )
    conn.commit()

    return new_status


def cancel_write_job(job_id: str, reason: str = None, db_path: str = None) -> dict:
    """Cancel a write job with tombstone semantics.

    Semantics:
      - queued   -> canceled immediately (prevented from executing)
      - running  -> tombstone: cancel_requested=1 (worker checks before/during exec)
      - done/failed/dead/canceled -> no_op (idempotent)

    Returns:
        dict with outcome: 'canceled' | 'cancel_requested' | 'no_op' | 'not_found'
    """
    conn = _get_conn(db_path)
    now = now_iso()
    cancel_reason = (reason or "user_cancel")[:200]
    row = conn.execute(
        "SELECT status, cancel_requested FROM write_queue WHERE job_id = ?",
        (job_id,)
    ).fetchone()

    if not row:
        return {"outcome": "not_found", "message": f"Job {job_id} not found."}

    status = row["status"]

    if status == "queued":
        conn.execute(
            "UPDATE write_queue SET status = 'canceled', "
            "cancel_requested = 1, canceled_at = ?, "
            "cancel_reason = ?, updated_at = ? "
            "WHERE job_id = ? AND status = 'queued'",
            (now, cancel_reason, now, job_id),
        )
        conn.commit()
        return {"outcome": "canceled", "message": f"Job {job_id} canceled."}

    if status == "running":
        # Tombstone: mark for worker to check before/during execution
        conn.execute(
            "UPDATE write_queue SET cancel_requested = 1, "
            "cancel_reason = ?, updated_at = ? "
            "WHERE job_id = ?",
            (cancel_reason, now, job_id),
        )
        conn.commit()
        return {
            "outcome": "cancel_requested",
            "message": f"Job {job_id} is running. Tombstone set — worker will check before applying.",
        }

    return {"outcome": "no_op", "message": f"Job {job_id} is {status}, nothing to cancel."}


def check_cancel_requested(job_id: str, db_path: str = None) -> bool:
    """Check if a job has cancel_requested=1. Used by worker pre-exec."""
    conn = _get_conn(db_path)
    row = conn.execute(
        "SELECT cancel_requested FROM write_queue WHERE job_id = ?",
        (job_id,),
    ).fetchone()
    return bool(row and row["cancel_requested"])


def mark_job_canceled(job_id: str, reason: str = "cancel_requested", db_path: str = None) -> None:
    """Mark a job as canceled by the worker (after seeing tombstone)."""
    conn = _get_conn(db_path)
    now = now_iso()
    conn.execute(
        "UPDATE write_queue SET status = 'canceled', "
        "canceled_at = ?, cancel_reason = ?, updated_at = ? "
        "WHERE job_id = ?",
        (now, reason[:200], now, job_id),
    )
    conn.commit()


# ============================================================
# MCP TOOL WRAPPERS (return str for MCP protocol)
# ============================================================

def write_status(job_id: str) -> str:
    """Check status of an async write job.

    Returns job status, timestamps, latency breakdown, and preview.
    Use this to check if a queued memory operation has completed.
    """
    result = get_write_job_status(job_id)
    if result is None:
        return json.dumps({"error": f"Job '{job_id}' not found."}, ensure_ascii=False)
    return json.dumps(result, ensure_ascii=False, default=str)


def cancel_write(job_id: str, reason: str = "") -> str:
    """Cancel a write job (queued or running).

    For queued jobs: cancels immediately.
    For running jobs: sets tombstone — worker will check before applying.
    For terminal states (done/failed/dead/canceled): no-op.
    """
    result = cancel_write_job(job_id, reason=reason or None)
    return json.dumps(result, ensure_ascii=False)


def register_tools(mcp):
    """Register async write-path tools with MCP server."""
    mcp.tool()(write_status)
    mcp.tool()(cancel_write)


# ============================================================
# STUCK JOB REAPER
# ============================================================

def reap_stuck_jobs(
    max_exec_seconds: int = 180,
    db_path: str = None,
) -> dict:
    """Requeue or kill jobs stuck in 'running' for too long.

    Uses started_at (not claimed_at) to detect jobs that actually began
    executing but never finished. This avoids false positives from lease
    reclaim (which uses claimed_at / lease_until in claim_next_job).

    Jobs with status='running' but no started_at are classified separately
    as 'stuck_no_started_at' and requeued with a distinct reason.

    Args:
        max_exec_seconds: Max seconds a job can execute before being reaped (default 180)
        db_path: Override DB path (for testing)

    Returns:
        dict with requeued count, dead count, and details list
    """
    conn = _get_conn(db_path)
    now = now_iso()
    cutoff = (now_col() - timedelta(seconds=max_exec_seconds)).isoformat()

    requeued = 0
    dead = 0
    details = []

    # Case 1: running + started_at is old (stuck in execution)
    rows = conn.execute(
        "SELECT id, job_id, kind, attempts, max_attempts, started_at "
        "FROM write_queue "
        "WHERE status = 'running' AND started_at IS NOT NULL AND started_at < ?",
        (cutoff,)
    ).fetchall()

    for row in rows:
        if row["attempts"] >= row["max_attempts"]:
            # Exhausted: mark dead
            conn.execute(
                "UPDATE write_queue "
                "SET status = 'dead', failure_reason = 'timeout_dead', "
                "    last_error = ?, updated_at = ?, lease_until = NULL "
                "WHERE id = ?",
                (f"running > {max_exec_seconds}s, attempts exhausted", now, row["id"])
            )
            dead += 1
            details.append({"job_id": row["job_id"], "kind": row["kind"],
                            "action": "dead", "attempts": row["attempts"]})
        else:
            # Requeue for retry
            conn.execute(
                "UPDATE write_queue "
                "SET status = 'queued', failure_reason = 'timeout_requeued', "
                "    last_error = ?, updated_at = ?, "
                "    claimed_at = NULL, started_at = NULL, lease_until = NULL "
                "WHERE id = ?",
                (f"running > {max_exec_seconds}s, requeued", now, row["id"])
            )
            requeued += 1
            details.append({"job_id": row["job_id"], "kind": row["kind"],
                            "action": "requeued", "attempts": row["attempts"]})

    # Case 2: running + no started_at (claimed but never started - rare)
    orphans = conn.execute(
        "SELECT id, job_id, kind, attempts, max_attempts "
        "FROM write_queue "
        "WHERE status = 'running' AND started_at IS NULL AND claimed_at < ?",
        (cutoff,)
    ).fetchall()

    for row in orphans:
        if row["attempts"] >= row["max_attempts"]:
            conn.execute(
                "UPDATE write_queue "
                "SET status = 'dead', failure_reason = 'stuck_no_started_at_exhausted', "
                "    last_error = 'claimed but never started, attempts exhausted', "
                "    updated_at = ?, claimed_at = NULL, lease_until = NULL "
                "WHERE id = ?",
                (now, row["id"])
            )
            dead += 1
            details.append({"job_id": row["job_id"], "kind": row["kind"],
                            "action": "dead_no_start", "attempts": row["attempts"]})
        else:
            conn.execute(
                "UPDATE write_queue "
                "SET status = 'queued', failure_reason = 'stuck_no_started_at', "
                "    last_error = 'claimed but never started, requeued', "
                "    updated_at = ?, claimed_at = NULL, lease_until = NULL "
                "WHERE id = ?",
                (now, row["id"])
            )
            requeued += 1
            details.append({"job_id": row["job_id"], "kind": row["kind"],
                            "action": "requeued_no_start", "attempts": row["attempts"]})

    conn.commit()

    # Log each reap action to write_queue_log
    for d in details:
        _log_reap_action(conn, d["job_id"], d["kind"], d["action"], d["attempts"])

    return {"requeued": requeued, "dead": dead, "details": details}


def _log_reap_action(
    conn: sqlite3.Connection,
    job_id: str,
    kind: str,
    action: str,
    attempts: int,
) -> None:
    """Log a reap action to write_queue_log for observability."""
    now = now_iso()
    status = "dead" if action.startswith("dead") else "requeued"
    reason = "timeout_dead" if action == "dead" else "timeout_requeued"
    try:
        conn.execute(
            "INSERT INTO write_queue_log "
            "(job_id, kind, status, attempts, duration_ms, "
            " error_class, error_msg, failure_reason, session_id, created_at) "
            "VALUES (?, ?, ?, ?, NULL, 'StuckReaper', ?, ?, NULL, ?)",
            (job_id, kind, status, attempts, f"reap_{action}", reason, now),
        )
        conn.commit()
    except Exception:
        pass  # best-effort logging


def log_job_completion(
    job_id: str,
    kind: str,
    status: str,
    attempts: int,
    duration_ms: int = None,
    error_class: str = None,
    error_msg: str = None,
    failure_reason: str = None,
    session_id: str = None,
    db_path: str = None,
) -> None:
    """Write a completion entry to write_queue_log for observability."""
    conn = _get_conn(db_path)
    now = now_iso()
    conn.execute(
        "INSERT INTO write_queue_log "
        "(job_id, kind, status, attempts, duration_ms, "
        " error_class, error_msg, failure_reason, session_id, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (job_id, kind, status, attempts, duration_ms,
         error_class, error_msg[:200] if error_msg else None,
         failure_reason, session_id, now),
    )
    conn.commit()
