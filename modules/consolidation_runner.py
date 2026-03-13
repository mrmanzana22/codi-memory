"""
Background runner for full consolidation.

Launched as subprocess by sleep_loop when info-debt triggers full scope.
Manages its own lockfile lifecycle and writes status on exit.

Usage:
    python3 -m modules.consolidation_runner [--lookback 24]
"""

import json
import logging
import os
import socket
import sys
import time
from datetime import datetime
from pathlib import Path

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)-8s %(message)s",
    datefmt="%m/%d/%y %H:%M:%S",
)
_logger = logging.getLogger(__name__)

LOCK_DIR = Path.home() / ".codi-daemon"
LOCK_FILE = LOCK_DIR / "full_consolidation.lock"
STATUS_FILE = LOCK_DIR / "full_consolidation.status.json"
STALE_THRESHOLD_S = 7200  # 2 hours — if lock older than this, consider stale


class LockReadError(RuntimeError):
    """Raised when an existing lockfile cannot be read safely."""


def _read_lock() -> dict | None:
    """Read lockfile contents. Returns None only when the lockfile is absent."""
    if not LOCK_FILE.exists():
        return None
    try:
        return json.loads(LOCK_FILE.read_text())
    except Exception as e:
        raise LockReadError(f"Failed to read lockfile {LOCK_FILE}: {e}") from e


def _is_process_alive(pid: int, dispatched_at: str) -> bool:
    """Check if PID is alive AND matches expected dispatch time.

    PID recycling guard: also verify the process started around
    the dispatched_at time (within a few seconds).
    """
    try:
        os.kill(pid, 0)  # signal 0 = existence check
    except OSError:
        return False

    # Extra guard: check if lock is stale by timestamp
    try:
        dt = datetime.fromisoformat(dispatched_at)
        age_s = (datetime.now().astimezone() - dt).total_seconds()
        if age_s > STALE_THRESHOLD_S:
            _logger.warning("Lock PID %d alive but stale (%.0fs old)", pid, age_s)
            return False
    except Exception:
        pass

    return True


def _write_lock():
    """Write lockfile with current process info."""
    LOCK_DIR.mkdir(parents=True, exist_ok=True)
    lock_data = {
        "pid": os.getpid(),
        "hostname": socket.gethostname(),
        "started_by": "consolidation_runner",
        "dispatched_at": datetime.now().astimezone().isoformat(),
    }
    LOCK_FILE.write_text(json.dumps(lock_data, indent=2))
    _logger.info("Lock acquired: pid=%d", os.getpid())


def _write_status(status: str, detail: str, duration_ms: int):
    """Write final status file (persists after lock is released)."""
    STATUS_FILE.write_text(json.dumps({
        "status": status,
        "detail": detail[:500],
        "duration_ms": duration_ms,
        "pid": os.getpid(),
        "hostname": socket.gethostname(),
        "finished_at": datetime.now().astimezone().isoformat(),
    }, indent=2))


def _release_lock():
    """Remove lockfile."""
    try:
        LOCK_FILE.unlink(missing_ok=True)
        _logger.info("Lock released")
    except Exception as e:
        _logger.error("Failed to release lock: %s", e)


def acquire_lock() -> bool:
    """Try to acquire the lock. Returns True if acquired."""
    try:
        existing = _read_lock()
    except LockReadError as e:
        _logger.error("Refusing to acquire lock: %s", e)
        _write_status("lock-read-error", str(e), 0)
        return False

    if existing:
        pid = existing.get("pid", -1)
        dispatched = existing.get("dispatched_at", "")
        if _is_process_alive(pid, dispatched):
            _logger.warning(
                "Full consolidation already running (pid=%d, dispatched=%s)",
                pid, dispatched,
            )
            return False
        else:
            _logger.info("Cleaning stale lock (pid=%d)", pid)
            _write_status("stale-cleanup", f"Stale lock from pid={pid}", 0)
            _release_lock()

    _write_lock()
    return True


def run(lookback_hours: int = 24, scope: str = "full"):
    """Run consolidation with lock management.

    Args:
        lookback_hours: How far back to look for unconsolidated episodes.
        scope: "full" (all 7 phases, ~2-30min) or "light" (clustering
               + graph edges only, ~13-76s).  Both share the same lock
               to prevent concurrent access to Qdrant/spreading_edges.
    """
    if not acquire_lock():
        _logger.info("Exiting: could not acquire lock")
        return

    start = time.monotonic()
    try:
        from modules.consolidation import run_consolidation

        _logger.info("Starting %s consolidation (lookback=%dh)", scope, lookback_hours)
        report = run_consolidation(scope=scope, lookback_hours=lookback_hours)
        duration_ms = int((time.monotonic() - start) * 1000)

        _logger.info("Completed %s in %dms: %s", scope, duration_ms, report[:200])
        _write_status("success", f"[{scope}] {report[:480]}", duration_ms)

    except Exception as e:
        duration_ms = int((time.monotonic() - start) * 1000)
        _logger.error("Failed %s after %dms: %s", scope, duration_ms, e)
        _write_status("error", f"[{scope}] {str(e)[:480]}", duration_ms)
        raise

    finally:
        _release_lock()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Consolidation background runner")
    parser.add_argument("--lookback", type=int, default=24, help="Lookback hours")
    parser.add_argument("--scope", choices=["full", "light"], default="full",
                        help="Consolidation scope (default: full)")
    args = parser.parse_args()
    run(lookback_hours=args.lookback, scope=args.scope)
