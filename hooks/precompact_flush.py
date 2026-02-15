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
    conn = sqlite3.connect(FTS_DB_PATH, timeout=5)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=3000")
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
    except Exception:
        return []


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
    try:
        conn.execute("""
            INSERT OR IGNORE INTO memories_text
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
    except Exception:
        return 0


def main():
    try:
        # Read stdin (PreCompact sends { trigger: "manual"|"auto" })
        _input_data = json.loads(sys.stdin.read())

        if not os.path.exists(FTS_DB_PATH):
            return

        conn = get_db_connection()
        try:
            wm_items = get_active_working_memory(conn)
            if wm_items:
                save_checkpoint(conn, wm_items)
        finally:
            conn.close()

    except Exception:
        pass


if __name__ == '__main__':
    main()
