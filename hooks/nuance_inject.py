#!/usr/bin/env python3
"""
Codi Memory - Nuance Injection Hook
====================================
Claude Code SessionStart hook.

Detects pending transcripts from previous sessions and injects them
as additional context so Codi (Opus) can extract and save nuances
using remember().

Flow:
  1. Scan .pending_nuances/ for files from OTHER sessions (not current)
  2. Pick the most recent one
  3. Inject transcript + instructions as additionalContext
  4. Delete the file after injection
  5. Auto-cleanup files older than 48h

Design: lightweight, no API calls, no heavy imports.
"""

import sys
import json
import os
from datetime import datetime, timedelta

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PENDING_DIR = os.path.join(BASE_DIR, "hooks", ".pending_nuances")
MAX_AGE_HOURS = 48
MIN_TURNS = 3


def cleanup_old_files():
    """Delete transcript files older than MAX_AGE_HOURS."""
    if not os.path.isdir(PENDING_DIR):
        return
    cutoff = datetime.now() - timedelta(hours=MAX_AGE_HOURS)
    for fname in os.listdir(PENDING_DIR):
        if not fname.endswith(".json"):
            continue
        fpath = os.path.join(PENDING_DIR, fname)
        try:
            mtime = datetime.fromtimestamp(os.path.getmtime(fpath))
            if mtime < cutoff:
                os.remove(fpath)
        except Exception:
            pass


def quarantine_bad_file(fpath: str):
    """Rename unreadable transcript so older valid files can be processed."""
    try:
        os.rename(fpath, f"{fpath}.bad")
    except Exception:
        pass


def find_pending_transcript(current_session_id: str):
    """Find the most recent pending transcript from a different session."""
    if not os.path.isdir(PENDING_DIR):
        return None, None

    best_file = None
    best_time = None

    for fname in os.listdir(PENDING_DIR):
        if not fname.endswith(".json"):
            continue
        # Skip current session's file
        session_id = fname.replace(".json", "")
        if session_id == current_session_id:
            continue

        fpath = os.path.join(PENDING_DIR, fname)
        try:
            mtime = os.path.getmtime(fpath)
            if best_time is None or mtime > best_time:
                best_time = mtime
                best_file = fpath
        except Exception:
            pass

    return best_file, best_time


def format_transcript_for_injection(data: dict) -> str:
    """Format transcript turns into a readable string for Codi."""
    turns = data.get("turns", [])
    if len(turns) < MIN_TURNS:
        return ""

    lines = []
    for t in turns:
        role = "Hare" if t.get("role") == "user" else "Codi"
        text = t.get("text", "").strip()
        if text:
            lines.append(f"[{role}] {text}")

    return "\n".join(lines)


def main():
    try:
        input_data = json.loads(sys.stdin.read())
    except Exception:
        return

    session_id = input_data.get("session_id", "")
    source = input_data.get("source", "")

    # Only run on fresh startup, not resume/compact
    if source not in ("startup", "clear"):
        return

    # Cleanup old files first
    cleanup_old_files()

    # Find pending transcript
    pending_file, _ = find_pending_transcript(session_id)
    if not pending_file:
        return

    data = None
    while pending_file:
        try:
            with open(pending_file, "r") as f:
                data = json.load(f)
            break
        except Exception:
            quarantine_bad_file(pending_file)
            pending_file, _ = find_pending_transcript(session_id)

    if not pending_file or data is None:
        return

    transcript_text = format_transcript_for_injection(data)
    if not transcript_text:
        # Too few turns, clean up
        try:
            os.remove(pending_file)
        except Exception:
            pass
        return

    turn_count = len(data.get("turns", []))
    prev_session = data.get("session_id", "desconocida")

    # Build injection context
    injection = (
        f"<nuance-transcript session=\"{prev_session}\" turns=\"{turn_count}\">\n"
        f"{transcript_text}\n"
        f"</nuance-transcript>\n"
        f"<nuance-instructions>\n"
        f"Procesaste {turn_count} matices de la sesion anterior. "
        f"Analiza el transcript y guarda los matices importantes con remember(). "
        f"Busca: preferencias implicitas de Hare, decisiones no documentadas, "
        f"contexto emocional, patrones de trabajo, cosas que Hare menciono "
        f"al pasar pero que son importantes. Menciona brevemente al usuario: "
        f"\"Procese {turn_count} matices de la sesion anterior.\"\n"
        f"</nuance-instructions>"
    )

    # Output as JSON with additionalContext
    output = {
        "hookSpecificOutput": {
            "hookEventName": "SessionStart",
            "additionalContext": injection,
        }
    }
    try:
        print(json.dumps(output), flush=True)
    except Exception:
        return

    # Delete the processed file only after successful emission
    try:
        os.remove(pending_file)
    except Exception:
        pass


if __name__ == "__main__":
    main()
