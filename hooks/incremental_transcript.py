#!/usr/bin/env python3
"""
Codi Memory - Incremental Transcript Capture
=============================================
Claude Code UserPromptSubmit hook.

Appends each user message to .pending_nuances/{session_id}.json so that
if the session dies (Ctrl+C, crash, timeout), we still have a partial
transcript for nuance extraction in the next session.

Design: append-only, no locks, no parsing. Target < 10ms.
"""

import sys
import json
import os
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PENDING_DIR = os.path.join(BASE_DIR, "hooks", ".pending_nuances")
MAX_TRANSCRIPT_BYTES = 25_000
MAX_TURN_CHARS = 500


def truncate_at_sentence(text: str, max_chars: int) -> str:
    """Truncate text at the last sentence boundary before max_chars."""
    if len(text) <= max_chars:
        return text
    truncated = text[:max_chars]
    # Find last sentence boundary
    for sep in [". ", "? ", "! ", "\n"]:
        pos = truncated.rfind(sep)
        if pos > max_chars // 2:  # Don't cut too early
            return truncated[:pos + 1].rstrip()
    # Fallback: cut at last space
    pos = truncated.rfind(" ")
    if pos > max_chars // 2:
        return truncated[:pos]
    return truncated


def main():
    try:
        input_data = json.loads(sys.stdin.read())
    except Exception:
        return

    session_id = input_data.get("session_id", "")
    prompt = input_data.get("prompt", "")
    if not session_id or not prompt:
        return

    # Strip system-reminder tags from prompt
    import re
    prompt = re.sub(r"<system-reminder>.*?</system-reminder>", "", prompt, flags=re.DOTALL).strip()
    if not prompt:
        return

    os.makedirs(PENDING_DIR, exist_ok=True)
    filepath = os.path.join(PENDING_DIR, f"{session_id}.json")

    # Load existing or create new
    try:
        if os.path.exists(filepath):
            with open(filepath, "r") as f:
                data = json.load(f)
        else:
            data = {
                "session_id": session_id,
                "started_at": datetime.now().isoformat(),
                "finalized": False,
                "turns": [],
            }
    except (json.JSONDecodeError, Exception):
        data = {
            "session_id": session_id,
            "started_at": datetime.now().isoformat(),
            "finalized": False,
            "turns": [],
        }

    # Truncate turn at sentence boundary
    condensed = truncate_at_sentence(prompt, MAX_TURN_CHARS)

    data["turns"].append({
        "role": "user",
        "text": condensed,
        "ts": datetime.now().isoformat(),
    })
    data["updated_at"] = datetime.now().isoformat()

    # Enforce 25K limit: drop oldest turns until we fit
    serialized = json.dumps(data, ensure_ascii=False)
    while len(serialized) > MAX_TRANSCRIPT_BYTES and len(data["turns"]) > 1:
        data["turns"].pop(0)
        serialized = json.dumps(data, ensure_ascii=False)

    try:
        with open(filepath, "w") as f:
            f.write(serialized)
    except Exception:
        pass


if __name__ == "__main__":
    main()
