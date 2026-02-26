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
from modules.secret_redact import redact_secrets

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
TICK_ORDER = ["prospective", "health", "self_model", "reconsolidation", "consolidation", "homeostasis", "backup"]

# Minimum ms required to even attempt a tick (below this, skip)
TICK_MIN_MS = {
    "prospective": 200,
    "health": 200,
    "self_model": 500,
    "reconsolidation": 300,
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

def _should_run_full_consolidation() -> bool:
    """Check if full consolidation should run (once per day, during night hours)."""
    try:
        from modules.config import now_col
        now = now_col()
        # Only during night hours (10pm-6am) to minimize interference
        if not (22 <= now.hour or now.hour < 6):
            return False
        # Check if full ran in last 20 hours
        conn = _get_conn()
        row = conn.execute(
            "SELECT MAX(created_at) FROM consolidation_log WHERE scope='full'"
        ).fetchone()
        conn.close()
        if row and row[0]:
            from datetime import datetime
            last = datetime.fromisoformat(str(row[0]))
            if last.tzinfo:
                last = last.replace(tzinfo=None)
            hours_since = (datetime.now() - last).total_seconds() / 3600
            return hours_since >= 20
        return True  # never ran, should run
    except Exception:
        return False  # fail safe: don't run full on error


def _tick_reconsolidation(budget_ms: int) -> dict:
    """Tick: Process labile memories within reconsolidation window (Nader 2000).

    Memories destabilized by prediction error can be updated during
    a time-limited reconsolidation window. For topic-shift PEs:
    lower confidence slightly (signal that context was ambiguous).
    """
    start = time.monotonic()
    result = {"tick": "reconsolidation", "ok": False, "detail": ""}

    try:
        from modules.config import FTS_DB_PATH, qdrant, COLLECTION_NAME

        conn = sqlite3.connect(FTS_DB_PATH, timeout=5)
        now = datetime.now().isoformat()
        processed = 0
        expired_count = 0

        # Get active (non-expired) labile memories
        rows = conn.execute(
            "SELECT memory_id, prediction_error, trigger_context "
            "FROM labile_memories WHERE window_expires > ? LIMIT 5",
            (now,)
        ).fetchall()

        # Clean up expired
        expired_count = conn.execute(
            "DELETE FROM labile_memories WHERE window_expires <= ?",
            (now,)
        ).rowcount

        for memory_id, pe, context in rows:
            try:
                points = qdrant.retrieve(
                    COLLECTION_NAME, [memory_id], with_payload=True
                )
                if not points:
                    conn.execute(
                        "DELETE FROM labile_memories WHERE memory_id = ?",
                        (memory_id,)
                    )
                    continue

                payload = points[0].payload
                old_confidence = payload.get("confidence", 0.8)

                # Importance guard: skip reconsolidation for protected memories (Alberini 2005)
                importance = payload.get("narrative_importance", "normal")
                access_count = payload.get("attention_access_count", 0)
                if importance in ("critical", "high") or access_count >= 10:
                    conn.execute(
                        "DELETE FROM labile_memories WHERE memory_id = ?",
                        (memory_id,)
                    )
                    continue

                # Lower confidence proportional to PE magnitude
                confidence_reduction = min(0.2, pe * 0.3)
                new_confidence = max(0.1, old_confidence - confidence_reduction)

                # Update payload in Qdrant
                qdrant.set_payload(
                    COLLECTION_NAME,
                    payload={
                        "confidence": new_confidence,
                        "reconsolidated_at": now,
                        "reconsolidation_count": int(payload.get("reconsolidation_count", 0)) + 1,
                        "last_pe_context": context[:200],
                    },
                    points=[memory_id],
                )

                # Log to reconsolidation_log
                conn.execute("""
                    INSERT INTO reconsolidation_log
                    (memory_id, memory_type, action, prediction_error,
                     memory_strength, old_content, new_content,
                     blend_weight, trigger_context, created_at)
                    VALUES (?, 'episodic', ?, ?, ?, ?, ?, 0.0, ?, ?)
                """, (
                    memory_id,
                    "confidence_adjustment",
                    pe,
                    old_confidence,
                    f"confidence={old_confidence:.2f}",
                    f"confidence={new_confidence:.2f}",
                    context[:200],
                    now,
                ))

                # Remove from labile
                conn.execute(
                    "DELETE FROM labile_memories WHERE memory_id = ?",
                    (memory_id,)
                )

                # Emit event (direct SQLite)
                conn.execute("""
                    INSERT INTO event_counts (event, count, last_seen)
                    VALUES ('reconsolidation_triggered', 1, ?)
                    ON CONFLICT(event) DO UPDATE SET
                        count = count + 1,
                        last_seen = excluded.last_seen
                """, (now,))

                processed += 1

            except Exception:
                pass

        conn.commit()
        conn.close()

        elapsed = (time.monotonic() - start) * 1000
        result["ok"] = True
        detail_parts = []
        if processed > 0:
            detail_parts.append(f"{processed} reconsolidated")
        if expired_count > 0:
            detail_parts.append(f"{expired_count} expired")
        if not detail_parts and not rows:
            detail_parts.append("no labiles")
        result["detail"] = ", ".join(detail_parts) if detail_parts else "idle"
        result["elapsed_ms"] = round(elapsed)
    except Exception as e:
        result["detail"] = f"error: {redact_secrets(str(e))[:100]}"
        result["elapsed_ms"] = round((time.monotonic() - start) * 1000)

    return result


def _tick_consolidation(budget_ms: int) -> dict:
    """Tick: Consolidation. Light every tick, full once per day at night."""
    start = time.monotonic()
    result = {"tick": "consolidation", "ok": False, "detail": ""}

    try:
        from modules.consolidation import run_consolidation

        if _should_run_full_consolidation():
            scope = "full"
            lookback = 24
        else:
            scope = "light"
            lookback = 6

        report = run_consolidation(scope=scope, lookback_hours=lookback)
        elapsed = (time.monotonic() - start) * 1000
        result["ok"] = True
        result["scope"] = scope
        result["detail"] = report[:200] if report else "no output"
        result["elapsed_ms"] = round(elapsed)
    except Exception as e:
        result["detail"] = f"error: {redact_secrets(str(e))[:100]}"
        result["elapsed_ms"] = round((time.monotonic() - start) * 1000)

    return result


def _tick_self_model(budget_ms: int) -> dict:
    """Tick: Self-model refresh (HOT-1, Rosenthal 2005).

    Calls reflect_on_self() to generate a meta-representation,
    then emits SELF_MODEL_REFRESHED event so assessment can detect it.
    """
    start = time.monotonic()
    result = {"tick": "self_model", "ok": False, "detail": ""}

    try:
        from modules.self_model import reflect_on_self
        summary = reflect_on_self()

        if summary and "Error" not in summary[:20]:
            # Emit event (EventBus flush works in venv python3)
            event_bus.emit(Events.SELF_MODEL_REFRESHED, {
                "source": "sleep_loop",
                "summary_len": len(summary),
            })
            # Belt-and-suspenders: direct SQLite write to event_counts
            try:
                conn = _get_conn()
                conn.execute("""
                    INSERT INTO event_counts (event, count, last_seen)
                    VALUES ('self_model_refreshed', 1, ?)
                    ON CONFLICT(event) DO UPDATE SET
                        count = count + 1,
                        last_seen = excluded.last_seen
                """, (datetime.now().isoformat(),))
                conn.commit()
                conn.close()
            except Exception:
                pass  # EventBus emit is primary; this is backup
            elapsed = (time.monotonic() - start) * 1000
            result["ok"] = True
            result["detail"] = f"refreshed ({len(summary)} chars)"
            result["elapsed_ms"] = round(elapsed)
        else:
            elapsed = (time.monotonic() - start) * 1000
            result["detail"] = "no identity memories or error"
            result["elapsed_ms"] = round(elapsed)
    except Exception as e:
        result["detail"] = f"error: {redact_secrets(str(e))[:100]}"
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
        parts.append(f"salience: error {redact_secrets(str(e))[:50]}")

    # Emotional decay (PAD toward baseline)
    try:
        from modules.consciousness import apply_emotional_decay
        emo_report = apply_emotional_decay()
        parts.append(f"emotional: {emo_report[:80]}")
    except Exception as e:
        parts.append(f"emotional: error {redact_secrets(str(e))[:50]}")

    elapsed = (time.monotonic() - start) * 1000
    result["ok"] = True
    result["detail"] = "; ".join(parts)
    result["elapsed_ms"] = round(elapsed)
    return result


def _tick_backup(budget_ms: int) -> dict:
    """Tick: Qdrant snapshot backup. Runs 3x/day (morning, afternoon, night).

    Creates snapshots of codi_memories + codi_semantic, downloads to local
    backups/ directory, keeps last 3 snapshots per collection.
    """
    import glob as glob_mod
    import requests
    from datetime import datetime
    import pytz

    start = time.monotonic()
    result = {"tick": "backup", "ok": False, "detail": ""}
    _tz = pytz.timezone("America/Bogota")
    now = datetime.now(_tz)
    hour = now.hour

    # Run 3x/day: 8am, 14pm, 22pm (morning, afternoon, night)
    backup_hours = {8, 14, 22}
    # Only run if current hour matches AND we haven't backed up this window yet
    if hour not in backup_hours:
        result["ok"] = True
        result["detail"] = f"skip (hour={hour}, next at {min(h for h in backup_hours if h > hour) if any(h > hour for h in backup_hours) else min(backup_hours)})"
        result["elapsed_ms"] = 0
        return result

    backup_dir = os.path.join(os.path.dirname(__file__), '..', 'backups', 'qdrant')
    os.makedirs(backup_dir, exist_ok=True)

    # Check if we already backed up in this window (same date + hour bucket)
    date_str = now.strftime("%Y-%m-%d")
    window = f"{date_str}-{hour:02d}"
    marker = os.path.join(backup_dir, f".backup-{window}.done")
    if os.path.exists(marker):
        result["ok"] = True
        result["detail"] = f"already done for {window}"
        result["elapsed_ms"] = 0
        return result

    parts = []
    collections = ["codi_memories", "codi_semantic"]
    qdrant_url = "http://localhost:6333"

    for coll in collections:
        try:
            # Create snapshot
            resp = requests.post(f"{qdrant_url}/collections/{coll}/snapshots", timeout=30)
            resp.raise_for_status()
            snap_info = resp.json()["result"]
            snap_name = snap_info["name"]
            snap_size = snap_info["size"]

            # Download snapshot
            filename = f"{coll}-{window}.snapshot"
            filepath = os.path.join(backup_dir, filename)
            dl_resp = requests.get(
                f"{qdrant_url}/collections/{coll}/snapshots/{snap_name}",
                stream=True, timeout=60
            )
            dl_resp.raise_for_status()
            with open(filepath, 'wb') as f:
                for chunk in dl_resp.iter_content(chunk_size=8192):
                    f.write(chunk)

            size_mb = round(snap_size / 1024 / 1024, 1)
            parts.append(f"{coll}: {size_mb}MB")

            # Cleanup: keep only last 3 snapshots per collection
            existing = sorted(glob_mod.glob(os.path.join(backup_dir, f"{coll}-*.snapshot")))
            while len(existing) > 3:
                os.remove(existing.pop(0))

            # Delete snapshot from Qdrant server (save disk space)
            requests.delete(
                f"{qdrant_url}/collections/{coll}/snapshots/{snap_name}",
                timeout=10
            )

        except Exception as e:
            parts.append(f"{coll}: error {str(e)[:50]}")

    # Backup SQLite databases (consciousness state)
    import shutil
    sqlite_dir = os.path.join(backup_dir, '..', 'sqlite')
    os.makedirs(sqlite_dir, exist_ok=True)
    project_root = os.path.join(os.path.dirname(__file__), '..')
    for db_name in ['memories_fts.db', 'prospective.db']:
        src = os.path.join(project_root, db_name)
        if os.path.exists(src):
            dst = os.path.join(sqlite_dir, f"{db_name.replace('.db', '')}-{window}.db")
            try:
                # Use SQLite backup API for consistency (not just file copy)
                src_conn = sqlite3.connect(src)
                dst_conn = sqlite3.connect(dst)
                src_conn.backup(dst_conn)
                dst_conn.close()
                src_conn.close()
                size_kb = round(os.path.getsize(dst) / 1024)
                parts.append(f"{db_name}: {size_kb}KB")
                # Keep only last 3 backups
                existing = sorted(glob_mod.glob(os.path.join(
                    sqlite_dir, f"{db_name.replace('.db', '')}-*.db")))
                while len(existing) > 3:
                    os.remove(existing.pop(0))
            except Exception as e:
                parts.append(f"{db_name}: error {str(e)[:40]}")

    # Write marker to prevent re-running this window
    with open(marker, 'w') as f:
        f.write(now.isoformat())

    # Cleanup old markers (keep last 7 days)
    for old_marker in glob_mod.glob(os.path.join(backup_dir, ".backup-*.done")):
        try:
            mtime = os.path.getmtime(old_marker)
            age_days = (time.time() - mtime) / 86400
            if age_days > 7:
                os.remove(old_marker)
        except Exception:
            pass

    elapsed = (time.monotonic() - start) * 1000
    result["ok"] = True
    result["detail"] = "; ".join(parts) if parts else "no collections"
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
        result["detail"] = f"error: {redact_secrets(str(e))[:100]}"
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
        parts.append(f"fts: error {redact_secrets(str(e))[:50]}")

    # Health check
    try:
        from modules.consciousness import _verificar_salud_memoria_interna
        health = _verificar_salud_memoria_interna()
        if health.get("ok"):
            parts.append(f"health: OK ({health.get('total_memories', '?')} mems)")
        else:
            parts.append(f"health: {health.get('message', 'unknown')[:60]}")
    except Exception as e:
        parts.append(f"health: error {redact_secrets(str(e))[:50]}")

    # Incremental FTS sync: index Qdrant memories not yet in FTS
    try:
        from modules.qdrant_utils import scroll_all
        from modules.config import qdrant as _qdrant, COLLECTION_NAME as _COLL

        fts_db_path = os.path.join(os.path.dirname(__file__), '..', 'memories_fts.db')
        fts_conn = sqlite3.connect(fts_db_path)

        # Get IDs already in FTS
        fts_ids = set(
            row[0] for row in
            fts_conn.execute("SELECT memory_id FROM memories_text").fetchall()
        )

        # Quick check: skip if already in sync
        qdrant_count = _qdrant.count(collection_name=_COLL).count
        if len(fts_ids) >= qdrant_count:
            parts.append(f"fts_sync: in sync ({len(fts_ids)})")
            fts_conn.close()
        else:
            # Paginated scroll through ALL Qdrant memories
            points = scroll_all(
                max_results=5000,
                with_payload=['data', 'category', 'narrative_importance', 'ownership_source'],
                with_vectors=False,
                batch_size=200,
            )

            synced = 0
            for p in points:
                mid = str(p.id)
                if mid not in fts_ids:
                    content = p.payload.get('data', '')
                    if not content:
                        continue
                    category = p.payload.get('category', 'general')
                    source = p.payload.get('ownership_source', 'experienced')
                    importance = p.payload.get('narrative_importance', 'medium')
                    fts_conn.execute("""
                        INSERT OR REPLACE INTO memories_text
                        (memory_id, content, category, source, importance, created_at)
                        VALUES (?, ?, ?, ?, ?, datetime('now'))
                    """, (mid, content, category, source, importance))
                    synced += 1

            if synced > 0:
                fts_conn.commit()
            fts_conn.close()

            if synced > 0:
                parts.append(f"fts_sync: {synced} new (total: {len(fts_ids) + synced})")
            else:
                parts.append("fts_sync: 0 new")
    except Exception as e:
        parts.append(f"fts_sync: error {str(e)[:50]}")

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
        "self_model": _tick_self_model,
        "reconsolidation": _tick_reconsolidation,
        "consolidation": _tick_consolidation,
        "homeostasis": _tick_homeostasis,
        "backup": _tick_backup,
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
                "detail": f"unhandled: {redact_secrets(str(e))[:80]}",
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

    # Acquire lock
    if not _acquire_lock():
        _logger.warning("Another instance is running. Skipping.")
        sys.exit(0)

    try:
        # Always run the 4 ticks (consolidation, decay, prospective, health)
        target_id = _get_target_checkpoint(args.max_age_min)
        _logger.info("Starting (reason=%s, budget=%dms, checkpoint=%s)", args.reason, args.budget_ms, target_id or "none")

        result = run_sleep_loop(reason=args.reason, budget_ms=args.budget_ms)

        # Write report to checkpoint if one is available
        if target_id and result.get("report"):
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
            return f"Sleep report status ERROR: {redact_secrets(str(e))}"


# ============================================================
# MODULE ENTRY POINT
# ============================================================

if __name__ == "__main__":
    cli_main()
