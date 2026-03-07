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
PROSPECTIVE_DB_PATH = os.path.join(BASE_DIR, "prospective.db")

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
    conn = sqlite3.connect(FTS_DB_PATH, timeout=10)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA busy_timeout=10000")
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

    # Phase 2: Session bridge checkpoint (lightweight SQLite-only)
    try:
        _session_bridge_checkpoint(session_id)
    except Exception:
        pass

    # Phase 3: Finalize nuance transcript for next session
    try:
        _finalize_nuance_transcript(session_id, transcript_path)
    except Exception:
        pass


def _session_bridge_checkpoint(session_id: str):
    """Lightweight session bridge checkpoint via direct SQLite.

    Captures working memory, predictions, intentions, and traces
    without importing heavy modules (Qdrant, mem0, OpenAI).
    Respects 120s dedupe window and 50-row retention.
    """
    # WM tables are in the SAME DB as FTS (memories_fts.db)
    if not os.path.exists(FTS_DB_PATH):
        return

    conn = get_db_connection()

    # --- DEDUPE CHECK ---
    try:
        existing = conn.execute(
            "SELECT id, source, created_at FROM session_checkpoints "
            "ORDER BY created_at DESC LIMIT 1"
        ).fetchone()
        if existing:
            try:
                from datetime import timezone, timedelta
                ex_dt = datetime.fromisoformat(existing[2])
                now = datetime.now(timezone(timedelta(hours=-5)))
                if ex_dt.tzinfo is None:
                    ex_dt = ex_dt.replace(tzinfo=timezone(timedelta(hours=-5)))
                age_sec = (now - ex_dt).total_seconds()
            except Exception:
                age_sec = 9999
            if age_sec < 120:
                # Within dedupe window: skip if existing is flush (higher priority)
                if existing[1] == "flush":
                    conn.close()
                    return
    except Exception:
        pass

    now_iso = datetime.now().isoformat()

    # 1. Working memory items + active project (same DB)
    wm_items = []
    wm_active_count = 0
    wm_topics = []
    active_project = None
    active_traces = []
    try:
        wm_active_count = conn.execute(
            "SELECT COUNT(*) FROM working_memory WHERE active = 1"
        ).fetchone()[0]
        rows = conn.execute(
            "SELECT content, topic, relevance FROM working_memory "
            "WHERE active = 1 ORDER BY relevance DESC LIMIT 15"
        ).fetchall()
        for r in rows:
            wm_items.append({
                "content": (r[0] or "")[:200],
                "topic": r[1],
                "relevance": round(float(r[2] or 0), 2),
            })
            if r[1] and r[1] not in wm_topics:
                wm_topics.append(r[1])
            if not active_project and r[1] not in ('metamemory', 'contradiction', 'general', ''):
                active_project = r[1]
    except Exception:
        pass

    # Active traces (same DB)
    try:
        trace_rows = conn.execute(
            "SELECT trace_name, theme FROM narrative_traces WHERE active = 1 LIMIT 10"
        ).fetchall()
        for r in trace_rows:
            active_traces.append({"trace_name": r[0], "theme": r[1]})
    except Exception:
        pass

    # 2. Prediction state + recent PEs
    last_pred_topic = None
    last_pred_confidence = None
    recent_pes = []
    try:
        row = conn.execute(
            "SELECT predicted_topic, confidence FROM prediction_state WHERE id=1"
        ).fetchone()
        if row:
            last_pred_topic = row[0]
            last_pred_confidence = row[1]
    except Exception:
        pass
    try:
        pe_rows = conn.execute(
            "SELECT actual_topic, predicted_topic, surprise_score, created_at "
            "FROM prediction_results ORDER BY id DESC LIMIT 3"
        ).fetchall()
        for r in pe_rows:
            recent_pes.append({
                "actual": r[0], "predicted": r[1],
                "surprise": r[2], "ts": r[3]
            })
    except Exception:
        pass

    # 3. Pending intentions (from prospective.db)
    pending_intentions = []
    if os.path.exists(PROSPECTIVE_DB_PATH):
        try:
            pm_conn = sqlite3.connect(PROSPECTIVE_DB_PATH, timeout=10)
            pm_conn.execute("PRAGMA busy_timeout=10000")
            int_rows = pm_conn.execute(
                "SELECT id, action, priority, activation, expiry FROM intentions "
                "WHERE status = 'pending' ORDER BY activation DESC LIMIT 10"
            ).fetchall()
            for r in int_rows:
                pending_intentions.append({
                    "id": r[0], "action": (r[1] or "")[:150],
                    "priority": r[2], "activation": r[3], "expiry": r[4],
                })
            pm_conn.close()
        except Exception:
            pass

    # 4. Build and insert checkpoint
    try:
        conn.execute(
            "INSERT INTO session_checkpoints "
            "(trace_id, session_id, source, goal_stack, active_project, "
            " session_summary, attention_focus, attention_strength, "
            " last_prediction_topic, last_prediction_confidence, "
            " recent_prediction_errors, wm_items, wm_active_count, "
            " pad_pleasure, pad_arousal, pad_dominance, pad_trigger, "
            " last_decisions, pending_intentions, active_traces, "
            " session_duration_minutes, created_at, sleep_report) "
            "VALUES (?, ?, 'hook', '[]', ?, ?, NULL, 0.0, ?, ?, ?, ?, ?, "
            " 0.0, 0.0, 0.0, '', '[]', ?, ?, NULL, ?, NULL)",
            (
                "",  # trace_id (not available in hook)
                session_id or "",
                active_project,
                "",  # session_summary (auto-captured separately)
                last_pred_topic,
                last_pred_confidence,
                json.dumps(recent_pes, ensure_ascii=False),
                json.dumps(wm_items, ensure_ascii=False),
                wm_active_count,
                json.dumps(pending_intentions, ensure_ascii=False),
                json.dumps(active_traces, ensure_ascii=False),
                now_iso,
            ),
        )
        conn.commit()

        # Retention: keep only last 50
        conn.execute(
            "DELETE FROM session_checkpoints WHERE id NOT IN "
            "(SELECT id FROM session_checkpoints ORDER BY created_at DESC LIMIT 50)"
        )
        conn.commit()
    except Exception:
        pass
    finally:
        conn.close()


PENDING_DIR = os.path.join(BASE_DIR, "hooks", ".pending_nuances")
NUANCE_MAX_BYTES = 25_000
NUANCE_MAX_TURN_CHARS = 500
NUANCE_MIN_TURNS = 3


def _truncate_at_sentence(text: str, max_chars: int) -> str:
    """Truncate text at the last sentence boundary before max_chars."""
    if len(text) <= max_chars:
        return text
    truncated = text[:max_chars]
    for sep in [". ", "? ", "! ", "\n"]:
        pos = truncated.rfind(sep)
        if pos > max_chars // 2:
            return truncated[:pos + 1].rstrip()
    pos = truncated.rfind(" ")
    if pos > max_chars // 2:
        return truncated[:pos]
    return truncated


def _finalize_nuance_transcript(session_id: str, transcript_path: str):
    """Finalize nuance transcript with both Hare and Codi turns.

    Overwrites the incremental file (user-only) with the complete
    version including Codi responses. If < MIN_TURNS, deletes the file.
    """
    if not transcript_path or not os.path.exists(transcript_path):
        return

    # Parse full transcript (all entries, no UUID filter)
    entries, _ = parse_transcript(transcript_path, after_uuid=None)
    if not entries:
        return

    # Build condensed turns
    turns = []
    for entry in entries:
        role = entry["role"]
        text = _truncate_at_sentence(entry["text"], NUANCE_MAX_TURN_CHARS)
        if text.strip():
            turns.append({
                "role": role,
                "text": text.strip(),
                "ts": entry.get("timestamp", ""),
            })

    if len(turns) < NUANCE_MIN_TURNS:
        # Session too short, clean up incremental file
        filepath = os.path.join(PENDING_DIR, f"{session_id}.json")
        if os.path.exists(filepath):
            os.remove(filepath)
        return

    data = {
        "session_id": session_id,
        "started_at": turns[0].get("ts", datetime.now().isoformat()),
        "updated_at": datetime.now().isoformat(),
        "finalized": True,
        "turns": turns,
    }

    # Enforce size limit: drop oldest turns
    serialized = json.dumps(data, ensure_ascii=False)
    while len(serialized) > NUANCE_MAX_BYTES and len(data["turns"]) > 1:
        data["turns"].pop(0)
        serialized = json.dumps(data, ensure_ascii=False)

    os.makedirs(PENDING_DIR, exist_ok=True)
    filepath = os.path.join(PENDING_DIR, f"{session_id}.json")
    with open(filepath, "w") as f:
        f.write(serialized)


if __name__ == '__main__':
    main()
