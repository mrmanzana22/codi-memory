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


# ============================================================
# CONNECTION
# ============================================================

def _get_conn(db_path: str = None) -> sqlite3.Connection:
    """Get WAL-mode connection to FTS DB (where write_queue lives)."""
    conn = sqlite3.connect(db_path or FTS_DB_PATH, timeout=5)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA busy_timeout=5000")
    conn.row_factory = sqlite3.Row
    return conn


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

    try:
        # Check dedupe: if same key exists in active states, return existing job
        if dedupe_key:
            row = conn.execute(
                "SELECT job_id, status FROM write_queue "
                "WHERE dedupe_key = ? AND status IN ('queued', 'running', 'done') "
                "ORDER BY created_at DESC LIMIT 1",
                (dedupe_key,)
            ).fetchone()

            if row:
                return {
                    "job_id": row["job_id"],
                    "status": row["status"],
                    "dedupe_hit": True,
                }

        job_id = str(uuid.uuid4())
        payload_json = json.dumps(payload, ensure_ascii=False, default=str)

        conn.execute(
            "INSERT INTO write_queue "
            "(job_id, kind, payload_json, status, priority, attempts, max_attempts, "
            " dedupe_key, created_at, updated_at) "
            "VALUES (?, ?, ?, 'queued', ?, 0, ?, ?, ?, ?)",
            (job_id, kind, payload_json, priority, max_attempts,
             dedupe_key, now, now),
        )
        conn.commit()

        return {
            "job_id": job_id,
            "status": "queued",
            "dedupe_hit": False,
        }

    finally:
        conn.close()


# ============================================================
# STATUS
# ============================================================

def get_write_job_status(job_id: str, db_path: str = None) -> dict | None:
    """Get status of a write job.

    Returns:
        dict with status, attempts, last_error, created_at, updated_at, completed_at
        or None if job_id not found.
    """
    conn = _get_conn(db_path)
    try:
        row = conn.execute(
            "SELECT job_id, kind, status, attempts, max_attempts, "
            "last_error, created_at, updated_at, completed_at "
            "FROM write_queue WHERE job_id = ?",
            (job_id,)
        ).fetchone()

        if not row:
            return None

        return {
            "job_id": row["job_id"],
            "kind": row["kind"],
            "status": row["status"],
            "attempts": row["attempts"],
            "max_attempts": row["max_attempts"],
            "last_error": row["last_error"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
            "completed_at": row["completed_at"],
        }

    finally:
        conn.close()


# ============================================================
# QUEUE STATS
# ============================================================

def get_queue_stats(db_path: str = None) -> dict:
    """Get summary stats of the write queue.

    Returns:
        dict with counts by status, oldest_queued, total.
    """
    conn = _get_conn(db_path)
    try:
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

    finally:
        conn.close()


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

    try:
        # First, reclaim stale leases (workers that died)
        conn.execute(
            "UPDATE write_queue SET status = 'queued', lease_until = NULL "
            "WHERE status = 'running' AND lease_until < ?",
            (now,)
        )

        # Find and claim next job atomically
        row = conn.execute(
            "SELECT id, job_id, kind, payload_json, attempts "
            "FROM write_queue "
            "WHERE status = 'queued' "
            "ORDER BY priority ASC, created_at ASC "
            "LIMIT 1"
        ).fetchone()

        if not row:
            conn.commit()
            return None

        # Claim it (record claimed_at for latency breakdown)
        conn.execute(
            "UPDATE write_queue "
            "SET status = 'running', lease_until = ?, updated_at = ?, "
            "    claimed_at = ?, attempts = attempts + 1 "
            "WHERE id = ? AND status = 'queued'",
            (lease_until, now, now, row["id"])
        )
        conn.commit()

        payload = json.loads(row["payload_json"])
        return {
            "job_id": row["job_id"],
            "kind": row["kind"],
            "payload": payload,
            "attempts": row["attempts"] + 1,  # already incremented
        }

    finally:
        conn.close()


def mark_job_started(job_id: str, db_path: str = None) -> bool:
    """Record started_at timestamp just before job execution begins.

    Separates queue wait time from actual execution time.
    Returns True if the job was updated.
    """
    conn = _get_conn(db_path)
    now = now_iso()
    try:
        cursor = conn.execute(
            "UPDATE write_queue SET started_at = ? "
            "WHERE job_id = ? AND status = 'running'",
            (now, job_id)
        )
        conn.commit()
        return cursor.rowcount > 0
    finally:
        conn.close()


def mark_job_done(job_id: str, db_path: str = None) -> bool:
    """Mark a job as successfully completed.

    Returns True if the job was updated.
    """
    conn = _get_conn(db_path)
    now = now_iso()
    try:
        cursor = conn.execute(
            "UPDATE write_queue "
            "SET status = 'done', updated_at = ?, completed_at = ?, lease_until = NULL "
            "WHERE job_id = ? AND status = 'running'",
            (now, now, job_id)
        )
        conn.commit()
        return cursor.rowcount > 0
    finally:
        conn.close()


def mark_job_failed(
    job_id: str,
    error: str,
    db_path: str = None,
) -> str:
    """Mark a job as failed. Returns new status ('failed' or 'dead').

    If attempts >= max_attempts, marks as 'dead' (no more retries).
    Otherwise marks as 'failed' (will be retried).
    """
    conn = _get_conn(db_path)
    now = now_iso()
    try:
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
            "SET status = ?, last_error = ?, updated_at = ?, lease_until = NULL "
            "WHERE job_id = ? AND status = 'running'",
            (final_status, error[:500], now, job_id)
        )
        conn.commit()

        return new_status
    finally:
        conn.close()


def log_job_completion(
    job_id: str,
    kind: str,
    status: str,
    attempts: int,
    duration_ms: int = None,
    error_class: str = None,
    error_msg: str = None,
    session_id: str = None,
    db_path: str = None,
) -> None:
    """Write a completion entry to write_queue_log for observability."""
    conn = _get_conn(db_path)
    now = now_iso()
    try:
        conn.execute(
            "INSERT INTO write_queue_log "
            "(job_id, kind, status, attempts, duration_ms, "
            " error_class, error_msg, session_id, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (job_id, kind, status, attempts, duration_ms,
             error_class, error_msg[:200] if error_msg else None,
             session_id, now),
        )
        conn.commit()
    finally:
        conn.close()
