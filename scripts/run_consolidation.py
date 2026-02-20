#!/usr/bin/env python3
"""
Standalone consolidation runner for codi-memory.
Runs via cron, notifies via n8n webhook, creates reminders in Supabase.

Usage:
    python3 run_consolidation.py [--lookback HOURS] [--scope SCOPE] [--notify] [--dry-run]

Exit codes: 0 = success, 1 = error
"""

import sys
import os
import json
import argparse
import logging
import re
import fcntl
from datetime import datetime, timedelta
from urllib.request import Request, urlopen
from urllib.error import URLError

# Suppress noisy library warnings
logging.disable(logging.INFO)

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

# Load .env if present
env_file = os.path.join(PROJECT_ROOT, ".env")
if os.path.exists(env_file):
    with open(env_file) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, val = line.split("=", 1)
                os.environ.setdefault(key.strip(), val.strip())

# --- Config ---
N8N_WEBHOOK_URL = "https://appn8n-n8n.lx6zon.easypanel.host/webhook/codi-consolidation"
SUPABASE_URL = "https://qslsjoavlpqjwokgaefr.supabase.co"
SUPABASE_ANON_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InFzbHNqb2F2bHBxandva2dhZWZyIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NzA1MDE1NTAsImV4cCI6MjA4NjA3NzU1MH0.YMo-63J09GqKjH6uvGJ5Krbjiz-AbJeRUmgCAr2q8j0"
HARE_USER_ID = "2b7878b2-c51b-4106-baf7-c3c0cf9c9363"

LOG_FILE = os.path.join(PROJECT_ROOT, "data", "consolidation_cron.log")
LOCK_FILE = os.path.join(PROJECT_ROOT, "data", "consolidation.lock")


def _acquire_lock():
    """Acquire exclusive lockfile to prevent concurrent consolidation runs.

    Returns file handle on success, None if another run is active.
    """
    os.makedirs(os.path.dirname(LOCK_FILE), exist_ok=True)
    try:
        lock_fd = open(LOCK_FILE, "w")
        fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        lock_fd.write(f"{os.getpid()} {datetime.now().isoformat()}\n")
        lock_fd.flush()
        return lock_fd
    except (IOError, OSError):
        return None


def _release_lock(lock_fd):
    """Release the lockfile."""
    if lock_fd:
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_UN)
            lock_fd.close()
        except Exception:
            pass


def _log(msg: str):
    """Append to log file."""
    os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(LOG_FILE, "a") as f:
        f.write(f"[{timestamp}] {msg}\n")


def _parse_report(report: str) -> dict:
    """Extract structured fields from the consolidation report string."""
    fields = {}
    patterns = {
        "episodes_scanned": r"Episodes scanned:\s*(\d+)",
        "clusters_found": r"Clusters found:\s*(\d+)",
        "facts_extracted": r"Facts extracted:\s*(\d+)",
        "facts_created": r"Facts created:\s*(\d+)",
        "facts_updated": r"Facts updated:\s*(\d+)",
        "contradictions": r"Contradictions:\s*(\d+)",
        "episodes_pruned": r"Episodes pruned:\s*(\d+)",
        "duration_ms": r"Duration:\s*(\d+)ms",
    }
    for key, pattern in patterns.items():
        match = re.search(pattern, report)
        fields[key] = int(match.group(1)) if match else 0
    return fields


def _notify_webhook(payload: dict):
    """Send results to n8n webhook for Telegram notification."""
    try:
        data = json.dumps(payload).encode("utf-8")
        req = Request(N8N_WEBHOOK_URL, data=data, method="POST")
        req.add_header("Content-Type", "application/json")
        with urlopen(req, timeout=10) as resp:
            _log(f"Webhook notified: {resp.status}")
    except URLError as e:
        _log(f"Webhook failed: {e}")
    except Exception as e:
        _log(f"Webhook error: {e}")


def _create_reminder(titulo: str, descripcion: str, prioridad: str = "media", categoria: str = "consciencia"):
    """Create a reminder in Supabase hare_pendientes."""
    try:
        tomorrow = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")
        payload = {
            "titulo": titulo,
            "descripcion": descripcion,
            "prioridad": prioridad,
            "categoria": categoria,
            "tipo": "recordatorio",
            "user_id": HARE_USER_ID,
            "fecha_limite": tomorrow,
        }
        data = json.dumps(payload).encode("utf-8")
        url = f"{SUPABASE_URL}/rest/v1/hare_pendientes"
        req = Request(url, data=data, method="POST")
        req.add_header("Content-Type", "application/json")
        req.add_header("apikey", SUPABASE_ANON_KEY)
        req.add_header("Authorization", f"Bearer {SUPABASE_ANON_KEY}")
        req.add_header("Prefer", "return=minimal")
        with urlopen(req, timeout=10) as resp:
            _log(f"Reminder created: {titulo} ({resp.status})")
    except Exception as e:
        _log(f"Reminder failed: {e}")


def _evaluate_importance(fields: dict) -> list:
    """Evaluate if consolidation results warrant reminders for Hare."""
    reminders = []

    if fields.get("contradictions", 0) > 0:
        reminders.append({
            "titulo": f"Codi detecto {fields['contradictions']} contradiccion(es) en memorias",
            "descripcion": f"Durante consolidacion automatica se encontraron {fields['contradictions']} contradicciones entre memorias existentes y nuevos hechos. Revisar con Codi.",
            "prioridad": "alta",
        })

    if fields.get("facts_created", 0) >= 10:
        reminders.append({
            "titulo": f"Consolidacion grande: {fields['facts_created']} facts nuevos",
            "descripcion": f"Se crearon {fields['facts_created']} hechos semanticos nuevos de {fields['episodes_scanned']} episodios. Puede valer la pena revisar calidad.",
            "prioridad": "media",
        })

    if fields.get("episodes_scanned", 0) > 100:
        reminders.append({
            "titulo": "Backlog de memorias alto - revisar frecuencia de consolidacion",
            "descripcion": f"{fields['episodes_scanned']} episodios sin consolidar. Considerar aumentar frecuencia del cron.",
            "prioridad": "media",
        })

    return reminders


def _check_time_intentions() -> list:
    """Check time-based prospective memory intentions.

    This runs independently of user interaction, solving the audit finding
    that time-based triggers only fired during pre-turn hook.
    Returns list of triggered intention descriptions.
    """
    triggered_actions = []
    try:
        from modules.prospective import (
            _get_conn, _check_time, _mark_triggered, _expire_stale,
            apply_intention_maintenance,
            PM_ACTIVATION_TRIGGER_THRESHOLD, PM_ACTIVATION_TRIGGER_THRESHOLD_DEFAULT,
        )
        from modules.config import now_col

        # Run maintenance first (decay + update last_maintained_at)
        maint_stats = apply_intention_maintenance()
        _log(f"Intention maintenance: {maint_stats}")

        conn = _get_conn()
        now = now_col()
        now_str = now.isoformat()

        _expire_stale(conn, now)

        # Only check time-based pending intentions (use lowest threshold)
        rows = conn.execute("""
            SELECT id, action, action_type, trigger_spec, priority, activation
            FROM intentions
            WHERE status = 'pending'
              AND trigger_type = 'time'
              AND activation > ?
              AND (snooze_until IS NULL OR snooze_until <= ?)
        """, (min(PM_ACTIVATION_TRIGGER_THRESHOLD.values()), now_str)).fetchall()

        for int_id, action, action_type, spec_raw, priority, activation in rows:
            try:
                spec = json.loads(spec_raw)
            except Exception:
                continue

            result = _check_time(spec, now)
            if result and result.get("matched"):
                _mark_triggered(conn, int_id, result.get("detail", ""), now_str)
                triggered_actions.append({
                    "action": action,
                    "priority": priority,
                    "detail": result.get("detail", ""),
                })
                _log(f"Time intention triggered: {action[:80]}")

        conn.commit()

    except Exception as e:
        _log(f"Time intention check failed: {e}")

    return triggered_actions


def main():
    parser = argparse.ArgumentParser(description="Run codi-memory consolidation pipeline")
    parser.add_argument("--lookback", type=int, default=24, help="Hours to look back (default: 24)")
    parser.add_argument("--scope", choices=["full", "light"], default="full", help="Scope: full (with LLM) or light")
    parser.add_argument("--notify", action="store_true", default=True, help="Send webhook notification (default: true)")
    parser.add_argument("--no-notify", action="store_false", dest="notify", help="Skip webhook notification")
    parser.add_argument("--dry-run", action="store_true", help="Parse and evaluate without running consolidation")
    args = parser.parse_args()

    # Acquire lockfile to prevent overlapping runs
    lock_fd = _acquire_lock()
    if lock_fd is None:
        msg = "Another consolidation run is already active. Skipping."
        _log(msg)
        print(json.dumps({"status": "locked", "message": msg}))
        return

    # Redirect stdout to stderr during module execution
    real_stdout = sys.stdout
    sys.stdout = sys.stderr

    try:
        from modules.consolidation import run_consolidation, count_unconsolidated_episodic

        # --- Prospective Memory: check time-based intentions ---
        # Runs ALWAYS, even when no episodes to consolidate (autonomous clock)
        triggered_intentions = _check_time_intentions()

        pending = count_unconsolidated_episodic(args.lookback)
        _log(f"Starting consolidation: {pending} pending episodes, lookback={args.lookback}h")

        if pending == 0 and not triggered_intentions:
            _log("No unconsolidated episodes and no triggered intentions. Skipping.")
            sys.stdout = real_stdout
            print(json.dumps({"status": "skipped", "pending_episodes": 0, "triggered_intentions": 0}))
            return

        if pending == 0 and triggered_intentions:
            # Only intentions triggered, no consolidation needed
            _log(f"No episodes but {len(triggered_intentions)} intention(s) triggered.")
            payload = {
                "status": "intentions_only",
                "pending_episodes": 0,
                "triggered_intentions": len(triggered_intentions),
                "intentions": [t["action"][:100] for t in triggered_intentions],
            }
            # Create reminders for triggered intentions
            for t in triggered_intentions:
                pri = "alta" if t["priority"] in ("critical", "high") else "media"
                _create_reminder(
                    f"[Codi PM] {t['action'][:80]}",
                    f"Intencion programada disparada: {t['action']}. {t['detail']}",
                    prioridad=pri,
                )
            if args.notify:
                _notify_webhook(payload)
            sys.stdout = real_stdout
            print(json.dumps(payload))
            return

        if args.dry_run:
            _log(f"Dry run: {pending} episodes would be processed")
            sys.stdout = real_stdout
            print(json.dumps({"status": "dry_run", "pending_episodes": pending}))
            return

        report = run_consolidation(scope=args.scope, lookback_hours=args.lookback)
        fields = _parse_report(report)

        _log(f"Consolidation done: {fields}")

        # Evaluate if reminders are needed
        reminders = _evaluate_importance(fields)
        reminder_texts = []
        for r in reminders:
            _create_reminder(r["titulo"], r["descripcion"], r.get("prioridad", "media"))
            reminder_texts.append(r["titulo"])

        # Create reminders for triggered intentions too
        for t in triggered_intentions:
            pri = "alta" if t["priority"] in ("critical", "high") else "media"
            _create_reminder(
                f"[Codi PM] {t['action'][:80]}",
                f"Intencion programada disparada: {t['action']}. {t['detail']}",
                prioridad=pri,
            )
            reminder_texts.append(f"[PM] {t['action'][:60]}")

        # Build webhook payload
        payload = {
            "status": "success",
            **fields,
            "triggered_intentions": len(triggered_intentions),
            "reminders": " | ".join(reminder_texts) if reminder_texts else "",
        }

        # Notify via webhook
        if args.notify and (fields.get("facts_created", 0) > 0 or triggered_intentions):
            _notify_webhook(payload)

        sys.stdout = real_stdout
        print(json.dumps(payload))

    except Exception as e:
        _log(f"ERROR: {type(e).__name__}: {e}")
        error_payload = {
            "status": "error",
            "error": str(e),
            "error_type": type(e).__name__,
        }
        if args.notify:
            _notify_webhook(error_payload)
        sys.stdout = real_stdout
        print(json.dumps(error_payload))
        sys.exit(1)
    finally:
        _release_lock(lock_fd)


if __name__ == "__main__":
    main()
