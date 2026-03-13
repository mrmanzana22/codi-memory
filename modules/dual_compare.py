"""
dual_compare.py - Dual-mode compare layer
==========================================
Tracks sync vs async results for shadow comparison.

When dual-mode is active:
  1. Sync path calls record_sync_result() with its result
  2. Async worker calls update_async_result() after job completion
  3. compute_comparison_for_job() compares both sides

Functions:
  - normalize_sync_result(kind, raw_output) -> dict
  - normalize_async_result(kind, job_result_dict) -> dict | None
  - compare_results(kind, sync_dict, async_dict) -> dict
  - record_sync_result(fingerprint, trace_id, kind, raw_output, ...) -> None
  - update_async_result(job_id, result_dict, error, failure_reason) -> str
  - compute_comparison_for_job(job_id) -> dict | None

Created: 2026-02-16
"""

from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
from typing import Optional

from modules.config import FTS_DB_PATH, now_iso, connect_fts

_logger = logging.getLogger(__name__)


# ============================================================
# CONSTANTS
# ============================================================

# Closed set of normalized actions
VALID_ACTIONS = {
    "saved_new", "skipped_duplicate", "noop", "error",
    "checkpoint_saved", "completed",
}

# Truncation caps (match migration 007 design)
CAP_RESULT_JSON = 8192
CAP_PREVIEW = 200
CAP_LAST_ERROR = 500
CAP_DIFF_SUMMARY = 300
CAP_DIFF_JSON = 4096


# ============================================================
# CONNECTION
# ============================================================

def _get_conn(db_path: str = None) -> sqlite3.Connection:
    conn = connect_fts(db_path)
    conn.row_factory = sqlite3.Row
    return conn


# ============================================================
# FINGERPRINT
# ============================================================

def compute_request_fingerprint(kind: str, content: str, trace_id: str) -> str:
    """SHA256(kind | normalized_content | trace_id)[:32].

    Distinct from dedupe_key which uses day_bucket instead of trace_id.
    """
    normalized = content.strip().lower()[:500]
    raw = f"{kind}|{normalized}|{trace_id}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:32]


# ============================================================
# NORMALIZE SYNC RESULT
# ============================================================

def normalize_sync_result(kind: str, raw_output) -> dict:
    """Standardize sync result to a closed action set.

    Maps the diverse return formats of remember/add_memory/checkpoint_memoria
    into a uniform dict with 'action' in VALID_ACTIONS.

    Returns:
        dict with keys: action, memory_id (optional), detail (optional)
    """
    # Parse JSON string if needed
    if isinstance(raw_output, str):
        try:
            data = json.loads(raw_output)
        except (json.JSONDecodeError, TypeError):
            # Detect plain-text errors before normalizing as data (#011)
            lower = raw_output.lower().strip()
            if lower.startswith(("error:", "exception:", "traceback")):
                return {"action": "error", "detail": raw_output[:200]}
            data = {"raw": raw_output}
    elif isinstance(raw_output, dict):
        data = raw_output
    else:
        return {"action": "error", "detail": f"unexpected type: {type(raw_output).__name__}"}

    # Checkpoint kind
    if kind == "checkpoint_memoria":
        # Checkpoint returns a confirmation string or dict
        if isinstance(data, dict) and data.get("action") == "error":
            return {"action": "error", "detail": str(data.get("detail", ""))[:200]}
        return {
            "action": "checkpoint_saved",
            "memory_id": data.get("memory_id") or data.get("new_id"),
            "detail": str(data.get("message", data.get("result", "")))[:200],
        }

    # Remember / add_memory kinds
    raw_action = data.get("action", "")

    # Map to closed set
    if raw_action in ("saved_new", "saved_with_relation", "saved_with_contradiction"):
        return {
            "action": "saved_new",
            "memory_id": data.get("new_id") or data.get("memory_id"),
        }

    if raw_action == "skipped_duplicate":
        return {
            "action": "skipped_duplicate",
            "memory_id": data.get("existing_id"),
        }

    if raw_action == "queued":
        # Async mode -- shouldn't appear in dual sync path, but handle gracefully
        return {"action": "noop", "detail": "async_mode_queued"}

    if raw_action == "noop":
        return {"action": "noop"}

    if raw_action in ("error", ""):
        if "error" in data or raw_action == "error":
            return {"action": "error", "detail": str(data.get("error", data.get("detail", "")))[:200]}

    # Generic success for add_memory (returns result string, not structured action)
    if "result" in data and not raw_action:
        return {"action": "completed", "detail": str(data["result"])[:200]}

    # Fallback: if we have some data but can't classify, call it completed
    if data:
        return {"action": "completed", "detail": str(data)[:200]}

    return {"action": "noop"}


# ============================================================
# NORMALIZE ASYNC RESULT
# ============================================================

def normalize_async_result(kind: str, job_result_dict: dict) -> Optional[dict]:
    """Normalize async worker result. Returns None if job not done.

    Args:
        kind: 'remember' | 'add_memory' | 'checkpoint_memoria'
        job_result_dict: dict from write_worker executor, or status dict with 'status' key

    Returns:
        Normalized dict (same shape as normalize_sync_result), or None if not done.
    """
    if not job_result_dict:
        return None

    # If this is a status dict (from get_write_job_status), check status
    status = job_result_dict.get("status")
    if status and status != "done":
        return None

    # Worker executor results have 'action' and/or 'result' keys
    raw_action = job_result_dict.get("action", "")
    result_str = job_result_dict.get("result", "")

    if kind == "checkpoint_memoria":
        return {
            "action": "checkpoint_saved",
            "detail": str(result_str)[:200],
        }

    # Remember kind: executor returns {"action": "...", "result": "..."}
    if kind == "remember":
        if raw_action in ("saved_new", "saved_with_relation", "saved_with_contradiction"):
            return {"action": "saved_new"}
        if raw_action == "skipped_duplicate":
            return {"action": "skipped_duplicate"}
        if raw_action == "unknown" or not raw_action:
            # Try to parse the result string for more detail
            try:
                parsed = json.loads(result_str) if isinstance(result_str, str) else {}
                return normalize_sync_result(kind, parsed)
            except Exception:
                return {"action": "completed", "detail": str(result_str)[:200]}

        return {"action": raw_action if raw_action in VALID_ACTIONS else "completed"}

    # add_memory kind: executor returns {"result": "..."}
    if kind == "add_memory":
        return {"action": "completed", "detail": str(result_str)[:200]}

    return {"action": "completed"}


# ============================================================
# COMPARE RESULTS
# ============================================================

def _get_current_write_mode() -> str:
    """Get current write mode from env. Used when recording new rows."""
    import os
    return os.environ.get("CODI_WRITE_MODE", "sync")


def compare_results(kind: str, sync_dict: dict, async_dict: dict,
                    write_mode: str = None) -> dict:
    """Compare normalized sync and async results.

    Args:
        kind: 'remember' | 'add_memory' | 'checkpoint_memoria'
        sync_dict: normalized sync result
        async_dict: normalized async result
        write_mode: write mode from the dual row ('shadow', 'sync', 'async').
                    If None, falls back to env var (backward compat).

    Returns:
        dict with: match (bool), divergence_code (str|None),
        diff_summary (str), diff_json (dict)
    """
    if write_mode is None:
        write_mode = _get_current_write_mode()

    sync_action = sync_dict.get("action", "unknown")
    async_action = async_dict.get("action", "unknown")

    diff_details = {}

    # Action comparison
    if sync_action != async_action:
        diff_details["action"] = {"sync": sync_action, "async": async_action}

        # Expected shadow dedup: sync saves first, async sees the duplicate.
        # Only counts as expected if write_mode is shadow AND the pattern is
        # exactly saved_new -> skipped_duplicate (same request confirmed by
        # being in the same dual_compare_log row).
        if (write_mode == "shadow"
                and sync_action == "saved_new"
                and async_action == "skipped_duplicate"):
            return {
                "match": True,
                "divergence_code": "expected_shadow_dedup",
                "diff_summary": "expected in shadow: sync saved then async deduped",
                "diff_json": json.dumps(diff_details, ensure_ascii=False)[:CAP_DIFF_JSON],
            }

        # Expected dual_ack dedup: either direction is valid because
        # the bg sync-compare thread and async worker race.
        # Normal: async saves first, sync dedupes.
        # Race:   sync (bg compare) saves first, async dedupes.
        if write_mode == "dual_ack":
            if (async_action == "saved_new"
                    and sync_action == "skipped_duplicate"):
                return {
                    "match": True,
                    "divergence_code": "expected_dual_dedup",
                    "diff_summary": "expected in dual_ack: async saved then sync deduped",
                    "diff_json": json.dumps(diff_details, ensure_ascii=False)[:CAP_DIFF_JSON],
                }
            if (sync_action == "saved_new"
                    and async_action == "skipped_duplicate"):
                return {
                    "match": True,
                    "divergence_code": "expected_dual_dedup",
                    "diff_summary": "expected in dual_ack: sync saved then async deduped (race)",
                    "diff_json": json.dumps(diff_details, ensure_ascii=False)[:CAP_DIFF_JSON],
                }

        # Expected legacy dedup: pre-DUAL-2 rows (write_mode unknown/empty)
        # where one side saved and the other deduped.  Same dedup pattern,
        # just missing the write_mode tag.
        if write_mode in ("unknown", "", None):
            is_dedup = (
                (sync_action == "saved_new" and async_action == "skipped_duplicate")
                or (async_action == "saved_new" and sync_action == "skipped_duplicate")
            )
            if is_dedup:
                return {
                    "match": True,
                    "divergence_code": "expected_legacy_dedup",
                    "diff_summary": f"expected dedup (legacy row): sync={sync_action}, async={async_action}",
                    "diff_json": json.dumps(diff_details, ensure_ascii=False)[:CAP_DIFF_JSON],
                }

        # Classify the divergence
        if async_action == "error":
            code = "async_failed"
        elif sync_action == "error":
            code = "sync_failed"
        else:
            code = "action_mismatch"

        summary = f"action: sync={sync_action}, async={async_action}"

        return {
            "match": False,
            "divergence_code": code,
            "diff_summary": summary[:CAP_DIFF_SUMMARY],
            "diff_json": json.dumps(diff_details, ensure_ascii=False)[:CAP_DIFF_JSON],
        }

    # Both have same action — check memory_id for saved_new
    # Only flag divergence when one side is None and the other is not,
    # indicating one side failed to produce an ID. Two different non-None
    # IDs from independent writes (dual mode) are expected and not a divergence.
    if sync_action == "saved_new":
        sync_mid = sync_dict.get("memory_id")
        async_mid = async_dict.get("memory_id")
        if "memory_id" in sync_dict and "memory_id" in async_dict:
            if (sync_mid is None) != (async_mid is None):
                diff_details["memory_id"] = {"sync": sync_mid, "async": async_mid}
                return {
                    "match": False,
                    "divergence_code": "memory_id_diff",
                    "diff_summary": f"memory_id differs: sync={sync_mid}, async={async_mid}",
                    "diff_json": json.dumps(diff_details, ensure_ascii=False)[:CAP_DIFF_JSON],
                }

    # Special case: checkpoint_memoria — both checkpoint_saved is a match
    # regardless of detail differences (each run generates unique content)

    # Match
    return {
        "match": True,
        "divergence_code": None,
        "diff_summary": f"match: both={sync_action}",
        "diff_json": "{}",
    }


# ============================================================
# RECORD SYNC RESULT (called by sync path)
# ============================================================

def record_sync_result(
    fingerprint: str,
    trace_id: str,
    kind: str,
    raw_output,
    async_job_id: str = None,
    async_status: str = None,
    db_path: str = None,
) -> None:
    """Record the sync side of a dual-mode comparison.

    Called immediately after sync execution completes.
    Idempotent: if a row for this async_job_id already exists,
    updates sync-side columns only if not yet computed.
    """
    normalized = normalize_sync_result(kind, raw_output)
    now = now_iso()
    write_mode = _get_current_write_mode()

    sync_result_json = json.dumps(normalized, ensure_ascii=False)[:CAP_RESULT_JSON]
    preview = normalized.get("detail", normalized.get("action", ""))[:CAP_PREVIEW]

    conn = _get_conn(db_path)
    try:
        # Idempotent: check if row already exists for this async_job_id
        if async_job_id:
            existing = conn.execute(
                "SELECT id, compare_status FROM dual_compare_log "
                "WHERE async_job_id = ?",
                (async_job_id,),
            ).fetchone()
            if existing:
                # Row exists — only update sync-side if not yet computed
                if existing["compare_status"] != "computed":
                    # Preserve write_mode if row was pre-created by record_async_intent
                    # (dual_ack sets it at intent time; don't overwrite with env default)
                    existing_mode = conn.execute(
                        "SELECT write_mode FROM dual_compare_log WHERE id = ?",
                        (existing["id"],),
                    ).fetchone()
                    effective_mode = write_mode
                    if existing_mode and existing_mode["write_mode"] not in ("unknown", ""):
                        effective_mode = existing_mode["write_mode"]
                    conn.execute(
                        "UPDATE dual_compare_log SET "
                        "request_fingerprint = ?, trace_id = ?, "
                        "sync_completed_at = ?, sync_result_json = ?, "
                        "sync_preview = ?, write_mode = ? "
                        "WHERE id = ?",
                        (fingerprint, trace_id, now, sync_result_json,
                         preview, effective_mode, existing["id"]),
                    )
                    conn.commit()
                return  # Don't create duplicate

        # No existing row — insert new
        conn.execute(
            "INSERT OR IGNORE INTO dual_compare_log "
            "(request_fingerprint, trace_id, kind, sync_completed_at, "
            " sync_result_json, sync_preview, async_job_id, "
            " async_status_at_record, write_mode, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (fingerprint, trace_id, kind, now, sync_result_json, preview,
             async_job_id, async_status, write_mode, now),
        )
        conn.commit()
    finally:
        conn.close()


# ============================================================
# RECORD ASYNC INTENT (called by dual_ack path before sync)
# ============================================================

def record_async_intent(
    fingerprint: str,
    trace_id: str,
    kind: str,
    async_job_id: str,
    write_mode: str = "dual_ack",
    db_path: str = None,
) -> None:
    """Pre-create dual_compare_log row with async intent (dual_ack mode).

    Called before the background sync comparison runs.
    This is the inverse of record_sync_result(): async fires first,
    then sync fills in its side later.

    The row starts with compare_status='pending'. When:
      - The worker finishes the async job, _dual_compare_hook calls
        update_async_result() which fills async_result_json.
      - The background sync thread calls record_sync_result() which
        fills sync_result_json (using the existing idempotent path).
    """
    now = now_iso()
    conn = _get_conn(db_path)
    try:
        conn.execute(
            "INSERT OR IGNORE INTO dual_compare_log "
            "(request_fingerprint, trace_id, kind, sync_completed_at, "
            " async_job_id, async_status_at_record, write_mode, compare_status, "
            " created_at) "
            "VALUES (?, ?, ?, '', ?, 'queued', ?, 'pending', ?)",
            (fingerprint, trace_id, kind, async_job_id, write_mode, now),
        )
        conn.commit()
    finally:
        conn.close()


# ============================================================
# UPDATE ASYNC RESULT (called by worker after job completion)
# ============================================================

def update_async_result(
    job_id: str,
    result_dict: dict = None,
    error: str = None,
    failure_reason: str = None,
    db_path: str = None,
) -> str:
    """Update the async side of a dual-mode comparison row.

    Called by write_worker after job execution (best-effort).
    Uses async_job_id to find the matching dual row.

    Returns:
        'updated' | 'dual_link_missing' | 'error'
    """
    conn = _get_conn(db_path)
    now = now_iso()
    try:
        row = conn.execute(
            "SELECT id, kind FROM dual_compare_log WHERE async_job_id = ?",
            (job_id,)
        ).fetchone()

        if not row:
            return "dual_link_missing"

        kind = row["kind"]
        async_result_json = None
        if result_dict:
            normalized = normalize_async_result(kind, result_dict)
            if normalized:
                async_result_json = json.dumps(normalized, ensure_ascii=False)[:CAP_RESULT_JSON]

        conn.execute(
            "UPDATE dual_compare_log SET "
            "async_completed_at = ?, async_result_json = ?, "
            "async_last_error = ?, failure_reason = ? "
            "WHERE async_job_id = ?",
            (now,
             async_result_json,
             error[:CAP_LAST_ERROR] if error else None,
             failure_reason,
             job_id),
        )
        conn.commit()
        return "updated"

    except Exception as exc:
        _logger.warning("update_async_result failed for job_id=%s: %s", job_id, exc)
        return f"error:{type(exc).__name__}"
    finally:
        conn.close()


# ============================================================
# COMPUTE COMPARISON
# ============================================================

def compute_comparison_for_job(job_id: str, db_path: str = None) -> Optional[dict]:
    """Compute match/divergence for a completed dual-mode pair.

    Called after update_async_result. Reads both sides, compares,
    and writes result back to dual_compare_log.

    Returns:
        dict with match, divergence_code, diff_summary — or None if not ready.
    """
    conn = _get_conn(db_path)
    now = now_iso()
    try:
        row = conn.execute(
            "SELECT id, kind, sync_result_json, async_result_json, "
            "async_last_error, failure_reason, write_mode "
            "FROM dual_compare_log WHERE async_job_id = ?",
            (job_id,)
        ).fetchone()

        if not row:
            return None

        sync_json = row["sync_result_json"]
        async_json = row["async_result_json"]

        if not sync_json:
            return None

        sync_dict = json.loads(sync_json)

        # Cancel tombstone: job was canceled, not a real divergence
        if row["failure_reason"] == "cancel_tombstone":
            result = {
                "match": True,
                "divergence_code": "canceled_tombstone",
                "diff_summary": "async canceled via tombstone (expected)",
            }
        # If async failed, mark as divergence
        elif row["async_last_error"] and not async_json:
            result = {
                "match": False,
                "divergence_code": "async_failed",
                "diff_summary": f"async error: {row['async_last_error'][:100]}",
                "diff_json": json.dumps({
                    "failure_reason": row["failure_reason"],
                    "error": row["async_last_error"][:200],
                }, ensure_ascii=False)[:CAP_DIFF_JSON],
            }
        elif not async_json:
            # Async not yet complete
            return None
        else:
            async_dict = json.loads(async_json)
            kind = row["kind"]
            write_mode = row["write_mode"] if "write_mode" in row.keys() else None
            result = compare_results(kind, sync_dict, async_dict,
                                     write_mode=write_mode)

        # PAD micro-update on gate pass
        if result.get("match"):
            try:
                from modules.spotlight import pad_micro_update
                pad_micro_update("gate_pass")
            except Exception:
                pass

        # Write comparison result
        conn.execute(
            "UPDATE dual_compare_log SET "
            "compare_status = 'computed', compare_computed_at = ?, "
            "match = ?, divergence_code = ?, diff_summary = ?, diff_json = ? "
            "WHERE id = ?",
            (now,
             1 if result["match"] else 0,
             result["divergence_code"],
             result["diff_summary"][:CAP_DIFF_SUMMARY] if result.get("diff_summary") else None,
             result["diff_json"][:CAP_DIFF_JSON] if result.get("diff_json") else None,
             row["id"]),
        )
        conn.commit()

        return result

    except Exception as exc:
        _logger.warning("Comparison failed for job %s: %s", job_id, exc)
        try:
            conn.execute(
                "UPDATE dual_compare_log SET compare_status = 'error', "
                "diff_summary = ? WHERE async_job_id = ?",
                (f"comparison_error: {str(exc)[:200]}", job_id),
            )
            conn.commit()
        except Exception:
            pass
        return None
    finally:
        conn.close()


# ============================================================
# UPDATE SYNC COMPARE STATUS
# ============================================================

def update_sync_compare_status(
    job_id: str,
    status: str,
    error: str = None,
    db_path: str = None,
) -> None:
    """Update sync_compare_status for a dual_compare_log row.

    Called by the background sync-compare thread to record the outcome
    of the sync comparison attempt (ok/timeout/skipped_capacity/error).

    Behavior:
      - Finds row by async_job_id (not fingerprint)
      - Sets sync_compare_status = status
      - If error provided: sets sync_last_error = error[:500]
      - If status == "ok": clears sync_last_error to NULL
      - If row doesn't exist: no-op (no crash)
      - Idempotent, does NOT touch compare_status or write_mode

    Args:
        job_id: async_job_id to look up
        status: "ok" | "timeout" | "skipped_capacity" | "error"
        error: optional error message (truncated to 500 chars)
        db_path: optional DB path override
    """
    conn = _get_conn(db_path)
    try:
        if status == "ok":
            conn.execute(
                "UPDATE dual_compare_log SET "
                "sync_compare_status = ?, sync_last_error = NULL "
                "WHERE async_job_id = ?",
                (status, job_id),
            )
        else:
            conn.execute(
                "UPDATE dual_compare_log SET "
                "sync_compare_status = ?, sync_last_error = ? "
                "WHERE async_job_id = ?",
                (status, error[:CAP_LAST_ERROR] if error else None, job_id),
            )
        conn.commit()
    except Exception:
        pass  # no-op on any error
    finally:
        conn.close()
