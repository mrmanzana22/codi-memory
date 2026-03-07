"""
Access Tracking Batch Aggregator
================================
Collects pg_store update_payload calls and flushes them in batches.
Reduces N individual DB round-trips to chunked pg.update_payload calls.

Hot-path functions (search_memory, spreading) call record_access() /
record_spreading() which enqueue updates.  A background daemon thread
flushes the pending dict every FLUSH_INTERVAL seconds using pg_store's
update_payload (one call per point, grouped in MAX_OPS_PER_BATCH chunks).

Failure policy: best-effort, drop-on-overload.  Access tracking is
observability/heuristic data, not integrity data.
"""

import logging
import os
import threading
import time

from modules.pg_store import pg

_logger = logging.getLogger(__name__)

# ============================================================
# CONFIGURATION (monkeypatchable for tests)
# ============================================================

FLUSH_INTERVAL: float = 0.5       # seconds between flush ticks
MAX_PENDING: int = 1000            # trigger immediate flush if exceeded
MAX_OPS_PER_BATCH: int = 200       # chunk size per batch_update_points call

# Alert thresholds (for check_health)
ALERT_ERRORS_PER_HOUR: int = 5     # FAIL if errors > this in 1h
ALERT_PENDING_MAX: int = 500       # WARN if pending stays above this


def _read_mode() -> str:
    """Read access tracking mode from env var or dot-file."""
    mode = os.environ.get("CODI_ACCESS_TRACKING_MODE", "").strip().lower()
    if mode in ("batched", "legacy"):
        return mode
    flag_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), ".access_tracking_mode"
    )
    try:
        with open(flag_path) as f:
            return f.read().strip().lower()
    except FileNotFoundError:
        return "legacy"


ACCESS_TRACKING_MODE: str = _read_mode()
_logger.info("access_tracking_mode=%s", ACCESS_TRACKING_MODE)

# ============================================================
# INTERNAL STATE
# ============================================================

_lock = threading.Lock()
# Key: point_id_str -> payload dict to merge
_pending: dict[str, dict] = {}

_stats: dict = {
    "enqueued": 0,
    "flushed": 0,
    "flush_count": 0,
    "dropped": 0,
    "errors": 0,
    "last_flush_ms": 0.0,
}

_flush_semaphore = threading.BoundedSemaphore(1)
_flusher_started = False
_stop_event = threading.Event()


# ============================================================
# PUBLIC API
# ============================================================

def record_access(collection: str, point_id: str, payload: dict) -> None:
    """Enqueue an update_payload update for batched flushing.

    Args:
        collection: Ignored (pg_store is collection-agnostic, kept for backward compat)
        point_id: Point UUID string
        payload: Payload dict to merge (caller computes final values)
    """
    if ACCESS_TRACKING_MODE == "legacy":
        _direct_set_payload(collection, point_id, payload)
        return

    needs_overflow = False
    with _lock:
        key = str(point_id)
        existing = _pending.get(key)
        if existing:
            existing.update(payload)
        else:
            _pending[key] = dict(payload)
        _stats["enqueued"] += 1
        needs_overflow = len(_pending) >= MAX_PENDING
        _ensure_flusher()

    if needs_overflow:
        threading.Thread(
            target=_flush_cycle, args=("overflow",), daemon=True
        ).start()


def record_spreading(
    collection: str,
    updates: dict,
    *,
    last_accessed: str | None = None,
) -> None:
    """Enqueue spreading activation updates.

    Args:
        collection: Ignored (pg_store is collection-agnostic, kept for backward compat)
        updates: {point_id: new_salience_value, ...}
        last_accessed: ISO timestamp (defaults to now_iso())
    """
    if ACCESS_TRACKING_MODE == "legacy":
        _direct_spreading(collection, updates, last_accessed)
        return

    if not updates:
        return

    from modules.config import now_iso
    ts = last_accessed or now_iso()

    needs_overflow = False
    with _lock:
        for pid, sal in updates.items():
            key = str(pid)
            payload = {
                "attention_salience": sal,
                "attention_last_accessed": ts,
            }
            existing = _pending.get(key)
            if existing:
                existing.update(payload)
            else:
                _pending[key] = payload
            _stats["enqueued"] += 1
        needs_overflow = len(_pending) >= MAX_PENDING
        _ensure_flusher()

    if needs_overflow:
        threading.Thread(
            target=_flush_cycle, args=("overflow",), daemon=True
        ).start()


def flush_now(reason: str = "manual") -> dict:
    """Synchronous flush.  Returns stats dict."""
    _flush_cycle(reason)
    with _lock:
        return {**_stats, "pending": len(_pending)}


def get_access_tracking_stats() -> dict:
    """Return current stats + pending count."""
    with _lock:
        return {**_stats, "pending": len(_pending)}


def check_health() -> dict:
    """Return health verdict for canary monitoring.

    Returns dict with:
        mode: current tracking mode
        verdict: "PASS" | "WARN" | "FAIL"
        reason: human-readable explanation
        stats: full stats snapshot
    """
    stats = get_access_tracking_stats()
    mode = ACCESS_TRACKING_MODE

    if mode == "legacy":
        return {"mode": mode, "verdict": "PASS", "reason": "legacy mode (no batching)", "stats": stats}

    errors = stats.get("errors", 0)
    dropped = stats.get("dropped", 0)
    pending = stats.get("pending", 0)

    if errors > ALERT_ERRORS_PER_HOUR:
        return {"mode": mode, "verdict": "FAIL", "reason": f"errors={errors} > {ALERT_ERRORS_PER_HOUR}", "stats": stats}
    if dropped > 0:
        return {"mode": mode, "verdict": "WARN", "reason": f"dropped={dropped} (data loss)", "stats": stats}
    if pending > ALERT_PENDING_MAX:
        return {"mode": mode, "verdict": "WARN", "reason": f"pending={pending} > {ALERT_PENDING_MAX}", "stats": stats}

    return {"mode": mode, "verdict": "PASS", "reason": "healthy", "stats": stats}


def shutdown() -> None:
    """Stop the flusher thread and flush remaining."""
    _stop_event.set()
    _flush_cycle("shutdown")


def _reset_for_testing() -> None:
    """Reset all module state.  Only for tests."""
    global _flusher_started, _flush_semaphore
    # Clear pending FIRST so waking flusher finds nothing to flush
    with _lock:
        _pending.clear()
    _stop_event.set()
    time.sleep(0.1)  # let any running flusher exit
    with _lock:
        for k in _stats:
            _stats[k] = 0
        _flusher_started = False
    _stop_event.clear()
    # Recreate semaphore to guarantee clean state
    _flush_semaphore = threading.BoundedSemaphore(1)


# ============================================================
# LEGACY FALLBACK (direct update_payload, no batching)
# ============================================================

def _direct_set_payload(collection: str, point_id: str, payload: dict) -> None:
    """Legacy mode: call pg.update_payload directly."""
    try:
        pg.update_payload(point_id, payload)
    except Exception:
        pass


def _direct_spreading(
    collection: str, updates: dict, last_accessed: str | None
) -> None:
    """Legacy mode: call pg.update_payload per point (old behavior)."""
    from modules.config import now_iso
    ts = last_accessed or now_iso()
    for pid, sal in updates.items():
        try:
            pg.update_payload(
                pid,
                {
                    "attention_salience": sal,
                    "attention_last_accessed": ts,
                },
            )
        except Exception:
            pass


# ============================================================
# FLUSHER BACKGROUND THREAD
# ============================================================

def _ensure_flusher() -> None:
    """Start background flusher if not already running.  Must hold _lock."""
    global _flusher_started
    if _flusher_started:
        return
    _flusher_started = True
    t = threading.Thread(
        target=_flusher_loop, daemon=True, name="access-tracking-flusher"
    )
    t.start()


def _flusher_loop() -> None:
    """Background loop: flush every FLUSH_INTERVAL seconds."""
    while not _stop_event.is_set():
        _stop_event.wait(timeout=FLUSH_INTERVAL)
        if _pending:
            _flush_cycle("timer")


def _flush_cycle(reason: str) -> None:
    """Acquire semaphore, drain pending, send batch to pg_store."""
    if not _flush_semaphore.acquire(blocking=False):
        return  # another flush in progress

    try:
        with _lock:
            if not _pending:
                return
            batch = dict(_pending)
            _pending.clear()

        t0 = time.monotonic()

        items = list(batch.items())  # [(point_id, payload), ...]
        _send_batch(items)

        elapsed_ms = (time.monotonic() - t0) * 1000
        with _lock:
            _stats["flushed"] += len(batch)
            _stats["flush_count"] += 1
            _stats["last_flush_ms"] = round(elapsed_ms, 1)

    except Exception as e:
        with _lock:
            _stats["errors"] += 1
        _logger.warning("access_tracking flush error: %s", type(e).__name__)
    finally:
        _flush_semaphore.release()


def _send_batch(items: list[tuple[str, dict]]) -> None:
    """Send batched update_payload ops to pg_store in chunks."""
    for i in range(0, len(items), MAX_OPS_PER_BATCH):
        chunk = items[i : i + MAX_OPS_PER_BATCH]
        for pid, payload in chunk:
            try:
                pg.update_payload(pid, payload)
            except Exception as e:
                with _lock:
                    _stats["dropped"] += 1
                    _stats["errors"] += 1
                _logger.warning(
                    "update_payload error (id=%s): %s",
                    pid,
                    type(e).__name__,
                )
