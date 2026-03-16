#!/usr/bin/env python3
"""
Codi Memory - Edit Tracker Hook (PostToolUse)
==============================================
Fires after Edit/Write tools complete on codi-memory modules.

Captures WHAT changed (file, function hints) with WHEN (timestamp)
and WHO (session_id). Appends to session_edits.jsonl.

Does NOT call remember() — that happens at Stop via session_capture.py.
This follows Craik & Lockhart 1972: capture deep encoding data
(not just filename) for better recall later.

Target: <20ms per invocation. Append-only, no locks.
"""

import sys
import json
import os
import re
from datetime import datetime

EDITS_LOG = os.path.expanduser("~/.claude/session_edits.jsonl")
TRACKED_PATHS = [
    os.path.expanduser("~/codi-memory/modules/"),
    os.path.expanduser("~/codi-memory/hooks/"),
    os.path.expanduser("~/codi-memory/server.py"),
    os.path.expanduser("~/codi-daemon/"),
]


def extract_function_hints(content: str, max_hints: int = 3) -> list:
    """Extract function/class names from edited content for deep encoding."""
    hints = []
    # Match def/class declarations
    for match in re.finditer(r'(?:def|class)\s+(\w+)', content):
        hints.append(match.group(1))
        if len(hints) >= max_hints:
            break
    return hints


def is_tracked_path(file_path: str) -> bool:
    """Check if file is in a tracked directory."""
    if not file_path:
        return False
    resolved = os.path.realpath(file_path)
    return any(resolved.startswith(os.path.realpath(p)) for p in TRACKED_PATHS)


def main():
    try:
        input_data = json.loads(sys.stdin.read())
    except (json.JSONDecodeError, Exception):
        return

    tool_name = input_data.get("tool_name", "")
    if tool_name not in ("Edit", "Write", "MultiEdit"):
        return

    tool_input = input_data.get("tool_input", {})
    file_path = tool_input.get("file_path", "")

    if not is_tracked_path(file_path):
        return

    session_id = input_data.get("session_id", "")

    # Extract function hints from the new content (deep encoding)
    new_content = ""
    if tool_name == "Write":
        new_content = tool_input.get("content", "")[:2000]
    elif tool_name == "Edit":
        new_content = tool_input.get("new_string", "")[:2000]
    elif tool_name == "MultiEdit":
        edits = tool_input.get("edits", [])
        new_content = " ".join(e.get("new_string", "")[:500] for e in edits[:4])

    function_hints = extract_function_hints(new_content)

    # Build entry
    entry = {
        "ts": datetime.now().isoformat(),
        "file": os.path.basename(file_path),
        "path": file_path,
        "tool": tool_name,
        "functions": function_hints,
        "session_id": session_id,
    }

    # Append-only write (no locks needed for JSONL append)
    try:
        with open(EDITS_LOG, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception:
        pass

    # Silent success
    sys.exit(0)


if __name__ == "__main__":
    main()
