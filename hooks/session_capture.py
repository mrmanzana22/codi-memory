#!/usr/bin/env python3
"""
Codi Memory - Session Auto-Capture Hook
=========================================
Claude Code Stop hook.

Fires when Claude finishes responding (end of each turn).
Parses the conversation transcript, applies signal extraction
(keywords + trigger patterns), and saves relevant turns to
codi-memory's FTS5 index.

Inspired by supermemory's signal extraction but adapted to
our architecture: local storage, no cloud dependency, integrated
with our trigger system and ownership model.
"""

import sys
import json
import sqlite3
import os
import re
import uuid
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FTS_DB_PATH = os.path.join(BASE_DIR, "memories_fts.db")
TRIGGERS_FILE = os.path.join(BASE_DIR, "triggers.json")
TRACKER_DIR = os.path.join(BASE_DIR, "hooks", ".trackers")

# Signal keywords — adapted from supermemory + our own
SIGNAL_KEYWORDS = [
    # Decisions & architecture
    'decision', 'decidimos', 'elegimos', 'arquitectura', 'diseño',
    'approach', 'tradeoff', 'patron', 'pattern',
    # Problems & solutions
    'bug', 'error', 'fix', 'solucion', 'solved', 'resuelto',
    'problema', 'falla', 'funciono',
    # Learning & memory
    'remember', 'recuerda', 'importante', 'critical', 'aprendizaje',
    'aprendi', 'leccion', 'insight',
    # Implementation
    'implementar', 'refactor', 'migration', 'deploy', 'upgrade',
    'deprecate', 'completado', 'listo', 'terminamos',
    # Codi-specific
    'checkpoint', 'guardar', 'pendiente', 'siguiente paso',
    'plan', 'proyecto', 'fase',
]

# Context window: how many turns before a signal to capture
SIGNAL_TURNS_BEFORE = 2
MIN_CONTENT_LENGTH = 50
MAX_MEMORY_LENGTH = 500
MAX_MEMORIES_PER_SESSION = 10


def get_db_connection():
    conn = sqlite3.connect(FTS_DB_PATH, timeout=5)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA busy_timeout=3000")
    return conn


def ensure_tracker_dir():
    os.makedirs(TRACKER_DIR, exist_ok=True)


def get_last_captured_uuid(session_id):
    """Get the UUID of the last captured transcript entry for this session."""
    tracker_file = os.path.join(TRACKER_DIR, f"{session_id}.txt")
    if os.path.exists(tracker_file):
        with open(tracker_file, 'r') as f:
            return f.read().strip()
    return None


def save_last_captured_uuid(session_id, uuid_val):
    """Save the UUID of the last captured entry."""
    ensure_tracker_dir()
    tracker_file = os.path.join(TRACKER_DIR, f"{session_id}.txt")
    with open(tracker_file, 'w') as f:
        f.write(uuid_val)


def load_trigger_patterns():
    """Load all trigger patterns for signal detection."""
    if not os.path.exists(TRIGGERS_FILE):
        return []
    try:
        with open(TRIGGERS_FILE, 'r') as f:
            data = json.load(f)
        patterns = []
        for _name, config in data.get('triggers', {}).items():
            patterns.extend(config.get('patterns', []))
        return [p.lower() for p in patterns]
    except Exception:
        return []


def parse_transcript(transcript_path, after_uuid=None):
    """Parse JSONL transcript file into structured turns."""
    if not os.path.exists(transcript_path):
        return [], None

    entries = []
    found_marker = after_uuid is None  # If no marker, include all
    last_uuid = None

    with open(transcript_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue

            entry_uuid = entry.get('uuid', '')
            last_uuid = entry_uuid

            if not found_marker:
                if entry_uuid == after_uuid:
                    found_marker = True
                continue

            # Extract role and content
            msg = entry.get('message', {})
            role = msg.get('role', entry.get('type', ''))
            content_blocks = msg.get('content', [])

            # Skip non-conversation entries
            if role not in ('user', 'assistant', 'human'):
                continue

            # Extract text content
            text_parts = []
            for block in content_blocks:
                if isinstance(block, str):
                    text_parts.append(block)
                elif isinstance(block, dict):
                    if block.get('type') == 'text':
                        text = block.get('text', '')
                        # Strip system reminders and injected context
                        text = re.sub(r'<system-reminder>.*?</system-reminder>', '', text, flags=re.DOTALL)
                        text = re.sub(r'<codi-memory-context>.*?</codi-memory-context>', '', text, flags=re.DOTALL)
                        text = re.sub(r'<system-critical>.*?</system-critical>', '', text, flags=re.DOTALL)
                        if text.strip():
                            text_parts.append(text.strip())
                    # Skip tool_use, tool_result, thinking blocks

            if text_parts:
                entries.append({
                    'uuid': entry_uuid,
                    'role': 'user' if role in ('user', 'human') else 'assistant',
                    'text': '\n'.join(text_parts),
                    'timestamp': entry.get('timestamp', ''),
                })

    return entries, last_uuid


def group_into_turns(entries):
    """Group entries into user-assistant turn pairs."""
    turns = []
    current_turn = {'user': '', 'assistant': '', 'timestamp': '', 'uuid': ''}

    for entry in entries:
        if entry['role'] == 'user':
            # Start new turn if we already have user content
            if current_turn['user']:
                turns.append(current_turn)
                current_turn = {'user': '', 'assistant': '', 'timestamp': '', 'uuid': ''}
            current_turn['user'] = entry['text']
            current_turn['timestamp'] = entry['timestamp']
            current_turn['uuid'] = entry['uuid']
        elif entry['role'] == 'assistant':
            # Append to current turn (take last assistant response)
            current_turn['assistant'] = entry['text']
            if not current_turn['uuid']:
                current_turn['uuid'] = entry['uuid']

    # Don't forget last turn
    if current_turn['user'] or current_turn['assistant']:
        turns.append(current_turn)

    return turns


def find_signal_turns(turns, trigger_patterns):
    """Find turns that contain signal keywords or trigger patterns."""
    all_signals = SIGNAL_KEYWORDS + trigger_patterns
    signal_indices = []

    for i, turn in enumerate(turns):
        user_text = turn['user'].lower()
        assistant_text = turn['assistant'].lower()
        combined = user_text + ' ' + assistant_text

        for signal in all_signals:
            if signal in combined:
                signal_indices.append(i)
                break

    return signal_indices


def get_turns_with_context(turns, signal_indices):
    """Get signal turns plus N preceding turns for context."""
    include_set = set()
    for idx in signal_indices:
        start = max(0, idx - SIGNAL_TURNS_BEFORE)
        for i in range(start, idx + 1):
            include_set.add(i)

    return sorted(include_set)


def format_turn_as_memory(turn):
    """Format a turn pair into a memory-worthy string."""
    parts = []
    if turn['user']:
        # Take first 200 chars of user message
        user_short = turn['user'][:200].strip()
        parts.append(f"Hare dijo: {user_short}")
    if turn['assistant']:
        # Take first 300 chars of assistant response
        asst_short = turn['assistant'][:300].strip()
        parts.append(f"Codi respondio: {asst_short}")

    memory = ' | '.join(parts)

    if len(memory) > MAX_MEMORY_LENGTH:
        memory = memory[:MAX_MEMORY_LENGTH]

    return memory


def save_memories(conn, memories):
    """Save extracted memories to FTS5 index."""
    saved = 0
    for mem in memories[:MAX_MEMORIES_PER_SESSION]:
        if len(mem['content']) < MIN_CONTENT_LENGTH:
            continue

        memory_id = f"auto_{uuid.uuid4().hex[:12]}"
        try:
            conn.execute("""
                INSERT OR IGNORE INTO memories_text
                (memory_id, content, category, source, importance, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                memory_id,
                mem['content'],
                'episodio',
                'experienced',
                'medium',
                mem.get('timestamp', datetime.now().isoformat())
            ))
            saved += 1
        except Exception:
            continue

    if saved > 0:
        conn.commit()
    return saved


def main():
    try:
        input_data = json.loads(sys.stdin.read())
        session_id = input_data.get('session_id', '')
        transcript_path = input_data.get('transcript_path', '')

        if not transcript_path or not session_id:
            return

        if not os.path.exists(transcript_path):
            return

        if not os.path.exists(FTS_DB_PATH):
            return

        # Get last captured position
        last_uuid = get_last_captured_uuid(session_id)

        # Parse transcript
        entries, final_uuid = parse_transcript(transcript_path, after_uuid=last_uuid)

        if not entries or not final_uuid:
            return

        # Group into turns
        turns = group_into_turns(entries)
        if not turns:
            save_last_captured_uuid(session_id, final_uuid)
            return

        # Load trigger patterns for signal detection
        trigger_patterns = load_trigger_patterns()

        # Find signal turns
        signal_indices = find_signal_turns(turns, trigger_patterns)

        if not signal_indices:
            # No signals detected — still update tracker
            save_last_captured_uuid(session_id, final_uuid)
            return

        # Get turns with surrounding context
        selected_indices = get_turns_with_context(turns, signal_indices)

        # Format as memories
        memories = []
        for idx in selected_indices:
            turn = turns[idx]
            content = format_turn_as_memory(turn)
            if content:
                memories.append({
                    'content': content,
                    'timestamp': turn.get('timestamp', datetime.now().isoformat()),
                    'is_signal': idx in signal_indices,
                })

        # Save to database
        conn = get_db_connection()
        try:
            saved = save_memories(conn, memories)
        finally:
            conn.close()

        # Update tracker
        save_last_captured_uuid(session_id, final_uuid)

    except Exception:
        # Never block Claude — silent fail
        pass


if __name__ == '__main__':
    main()
