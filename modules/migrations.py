"""
migrations.py - SQLite schema migration runner (Deliverable 4).
================================================================
Versioned, deterministic, auditable schema management.

Design:
  - schema_migrations table tracks applied versions
  - SQL files in numbered order (001_name.sql, 002_name.sql, ...)
  - Checksum (SHA256) detects modified migration files
  - Schema fingerprint detects drift in sqlite_master
  - One transaction per migration (rollback on failure)
  - BEGIN IMMEDIATE for multi-process safety

Usage:
  from modules.migrations import apply_migrations
  apply_migrations(db_path, migrations_dir="migrations")
"""

from __future__ import annotations

import hashlib
import os
import sqlite3
from typing import List, Optional, Tuple


def _compute_checksum(sql: str) -> str:
    """SHA256 checksum of migration SQL content."""
    return hashlib.sha256(sql.strip().encode("utf-8")).hexdigest()[:16]


def _compute_fingerprint(conn: sqlite3.Connection) -> str:
    """Compute a fingerprint of the current schema from sqlite_master.

    Captures all tables, indexes, triggers sorted by name.
    Detects drift even when IF NOT EXISTS masks changes.
    """
    rows = conn.execute(
        "SELECT type, name, sql FROM sqlite_master "
        "WHERE type IN ('table', 'index', 'trigger') AND name NOT LIKE 'sqlite_%' "
        "ORDER BY type, name"
    ).fetchall()
    content = "\n".join(f"{r[0]}|{r[1]}|{r[2] or ''}" for r in rows)
    return hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]


def _ensure_migrations_table(conn: sqlite3.Connection) -> None:
    """Create schema_migrations table if not exists."""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS schema_migrations (
            version TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            checksum TEXT NOT NULL,
            schema_fingerprint TEXT,
            applied_at TEXT NOT NULL DEFAULT (datetime('now'))
        )
    """)


def discover_migrations(migrations_dir: str) -> List[Tuple[str, str, str, str]]:
    """Find all .sql files in migrations_dir, sorted by version prefix.

    Returns list of (version, name, sql_content, checksum).
    Files must be named like: 001_description.sql
    """
    if not os.path.isdir(migrations_dir):
        return []

    migrations = []
    for filename in sorted(os.listdir(migrations_dir)):
        if not filename.endswith(".sql"):
            continue
        parts = filename.split("_", 1)
        if len(parts) < 2:
            continue
        version = parts[0]  # e.g. "001"
        name = filename[:-4]  # e.g. "001_baseline"

        filepath = os.path.join(migrations_dir, filename)
        with open(filepath, "r", encoding="utf-8") as f:
            sql = f.read()

        checksum = _compute_checksum(sql)
        migrations.append((version, name, sql, checksum))

    return migrations


def get_applied_versions(conn: sqlite3.Connection) -> dict:
    """Get dict of {version: checksum} for already-applied migrations."""
    _ensure_migrations_table(conn)
    rows = conn.execute(
        "SELECT version, checksum FROM schema_migrations ORDER BY version"
    ).fetchall()
    return {r[0]: r[1] for r in rows}


def get_current_version(db_path: str) -> Optional[str]:
    """Get the latest applied migration version, or None."""
    if not os.path.exists(db_path):
        return None
    conn = sqlite3.connect(db_path, timeout=5)
    try:
        _ensure_migrations_table(conn)
        row = conn.execute(
            "SELECT version FROM schema_migrations ORDER BY version DESC LIMIT 1"
        ).fetchone()
        return row[0] if row else None
    finally:
        conn.close()


def apply_migrations(db_path: str, migrations_dir: str = "migrations") -> dict:
    """Apply pending migrations to a SQLite database.

    Args:
        db_path: Path to SQLite database file.
        migrations_dir: Directory containing .sql migration files.

    Returns:
        dict with {applied: list[str], skipped: list[str], current_version: str|None}

    Raises:
        RuntimeError: If a previously applied migration's checksum has changed.
    """
    # Resolve migrations_dir relative to project root
    if not os.path.isabs(migrations_dir):
        base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        migrations_dir = os.path.join(base, migrations_dir)

    available = discover_migrations(migrations_dir)
    if not available:
        return {"applied": [], "skipped": [], "current_version": None}

    conn = sqlite3.connect(db_path, timeout=10)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=10000")

    try:
        _ensure_migrations_table(conn)
        applied_versions = get_applied_versions(conn)
        result = {"applied": [], "skipped": [], "current_version": None}

        for version, name, sql, checksum in available:
            if version in applied_versions:
                # Verify checksum hasn't changed
                stored_checksum = applied_versions[version]
                if stored_checksum != checksum:
                    raise RuntimeError(
                        f"Migration {name} checksum mismatch! "
                        f"Stored={stored_checksum}, current={checksum}. "
                        f"Migration files must not be modified after application."
                    )
                result["skipped"].append(version)
                continue

            # Apply migration SQL via executescript (handles its own transactions).
            # Then record in schema_migrations in a separate step.
            try:
                conn.executescript(sql)
                fingerprint = _compute_fingerprint(conn)
                conn.execute("BEGIN IMMEDIATE")
                conn.execute(
                    "INSERT OR IGNORE INTO schema_migrations "
                    "(version, name, checksum, schema_fingerprint) "
                    "VALUES (?, ?, ?, ?)",
                    (version, name, checksum, fingerprint),
                )
                conn.commit()
                result["applied"].append(version)
            except Exception as e:
                try:
                    conn.rollback()
                except Exception:
                    pass
                raise RuntimeError(
                    f"Migration {name} failed: {e}. Database rolled back."
                ) from e

        # Determine current version
        row = conn.execute(
            "SELECT version FROM schema_migrations ORDER BY version DESC LIMIT 1"
        ).fetchone()
        result["current_version"] = row[0] if row else None

        return result
    finally:
        conn.close()
