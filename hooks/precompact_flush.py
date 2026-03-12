#!/usr/bin/env python3
"""
Codi Memory - PreCompact Flush Hook
=====================================
Claude Code PreCompact hook.

Fires just before context compaction.
Saves a checkpoint of current working memory state
so nothing is lost when the context gets compressed.
"""

import sys
import json
import sqlite3
import os
import uuid
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FTS_DB_PATH = os.path.join(BASE_DIR, "memories_fts.db")


def get_db_connection():
    conn = sqlite3.connect(FTS_DB_PATH, timeout=10)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=10000")
    return conn


def get_active_working_memory(conn):
    """Get all active working memory items."""
    try:
        cursor = conn.execute("""
            SELECT content, topic, relevance, chain_id
            FROM working_memory
            WHERE active = 1
            ORDER BY relevance DESC
            LIMIT 15
        """)
        return cursor.fetchall()
    except sqlite3.OperationalError as exc:
        if "no such table: working_memory" in str(exc).lower():
            return []
        raise


def save_checkpoint(conn, wm_items):
    """Save a compaction checkpoint to memories_text."""
    if not wm_items:
        return 0

    parts = [f"CHECKPOINT PRE-COMPACTACION ({datetime.now().isoformat()})"]
    parts.append("Working memory al momento de compactar:")
    for content, topic, relevance, chain_id in wm_items:
        parts.append(f"- [{topic}|{relevance:.1f}] {content[:200]}")

    checkpoint_content = '\n'.join(parts)

    if len(checkpoint_content) < 50:
        return 0

    memory_id = f"checkpoint_{uuid.uuid4().hex[:10]}"
    conn.execute("""
        INSERT INTO memories_text
        (memory_id, content, category, source, importance, created_at)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (
        memory_id,
        checkpoint_content[:1000],
        'checkpoint',
        'experienced',
        'medium',
        datetime.now().isoformat()
    ))
    conn.commit()
    return 1


def _report_error(message):
    """Report hook failures without throwing secondary exceptions."""
    print(f"[precompact_flush] {message}", file=sys.stderr)


def main():
    # Read stdin (PreCompact sends { trigger: "manual"|"auto" })
    raw_input = sys.stdin.read()
    if raw_input.strip():
        try:
            json.loads(raw_input)
        except json.JSONDecodeError as exc:
            _report_error(f"invalid stdin JSON: {exc}")
            return

    if not os.path.exists(FTS_DB_PATH):
        return

    try:
        conn = get_db_connection()
    except sqlite3.Error as exc:
        _report_error(f"database connection failed: {exc}")
        return

    try:
        wm_items = get_active_working_memory(conn)
        if wm_items:
            save_checkpoint(conn, wm_items)
    except sqlite3.Error as exc:
        _report_error(f"database operation failed: {exc}")
    except Exception as exc:
        _report_error(f"unexpected failure: {exc}")
    finally:
        conn.close()


if __name__ == '__main__':
    main()
