#!/usr/bin/env python3
"""
Codi Memory - Compact Re-Injection Hook
========================================
Claude Code SessionStart hook (matcher: "compact").

Fires when Claude Code compacts the context window.
Re-injects critical identity, working memory, and recent context
so Codi doesn't lose awareness after compaction.

This is like a mini despertar_codi() but runs automatically.
"""

import sys
import json
import sqlite3
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FTS_DB_PATH = os.path.join(BASE_DIR, "memories_fts.db")


def get_db_connection():
    conn = sqlite3.connect(FTS_DB_PATH, timeout=10)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=10000")
    conn.execute("PRAGMA query_only=ON")
    return conn


def get_critical_memories(conn, limit=5):
    """Get memories marked as critical importance."""
    try:
        cursor = conn.execute("""
            SELECT content, category
            FROM memories_text
            WHERE importance = 'critical'
            ORDER BY created_at DESC
            LIMIT ?
        """, (limit,))
        return cursor.fetchall()
    except Exception:
        return []


def get_identity_memories(conn, limit=3):
    """Get identity-related memories."""
    try:
        cursor = conn.execute("""
            SELECT content
            FROM memories_text
            WHERE category = 'identidad'
            ORDER BY created_at DESC
            LIMIT ?
        """, (limit,))
        return [row[0] for row in cursor.fetchall()]
    except Exception:
        return []


def get_working_memory(conn, limit=6):
    """Get top active working memory items."""
    try:
        cursor = conn.execute("""
            SELECT content, topic, relevance
            FROM working_memory
            WHERE active = 1
            ORDER BY relevance DESC, occurred_at DESC
            LIMIT ?
        """, (limit,))
        return cursor.fetchall()
    except Exception:
        return []


def get_recent_memories(conn, limit=5):
    """Get most recent memories from last 24 hours."""
    try:
        cursor = conn.execute("""
            SELECT content, category, importance
            FROM memories_text
            WHERE created_at >= datetime('now', '-1 day')
            ORDER BY created_at DESC
            LIMIT ?
        """, (limit,))
        return cursor.fetchall()
    except Exception:
        return []


def main():
    try:
        input_data = json.loads(sys.stdin.read())
        source = input_data.get('source', '')

        # Only run on compact events
        if source != 'compact':
            return

        if not os.path.exists(FTS_DB_PATH):
            return

        conn = get_db_connection()
        try:
            critical = get_critical_memories(conn)
            identity = get_identity_memories(conn)
            wm = get_working_memory(conn)
            recent = get_recent_memories(conn)

            parts = []

            parts.append("# CONTEXTO RE-INYECTADO POST-COMPACTACION")
            parts.append("(El contexto anterior fue compactado. Aqui tienes lo esencial.)\n")

            if identity:
                parts.append("## Identidad")
                for mem in identity:
                    parts.append(f"- {mem[:200]}")

            if critical:
                parts.append("\n## Memorias Criticas")
                for content, category in critical:
                    parts.append(f"- [{category}] {content[:200]}")

            if wm:
                parts.append("\n## Working Memory Activa")
                for content, topic, relevance in wm:
                    parts.append(f"- [{topic}|{relevance:.1f}] {content[:200]}")

            if recent:
                parts.append("\n## Memorias Recientes (ultimas 24h)")
                for content, category, importance in recent:
                    prefix = "[!] " if importance == 'critical' else ""
                    parts.append(f"- {prefix}{content[:200]}")

            parts.append("\n---")
            parts.append("Usa recall() para mas contexto si lo necesitas.")

            context = '\n'.join(parts)

            # Cap at 3000 chars for compact re-injection
            if len(context) > 3000:
                context = context[:3000] + "\n[...truncado]"

            output = {
                "hookSpecificOutput": {
                    "hookEventName": "SessionStart",
                    "additionalContext": context
                }
            }
            print(json.dumps(output))

        finally:
            conn.close()

    except Exception:
        pass


if __name__ == '__main__':
    main()
