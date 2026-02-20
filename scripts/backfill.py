#!/usr/bin/env python3
"""
backfill.py - Generic backfill framework for schema migrations (E5.2)
=====================================================================
CLI tool to backfill data after schema changes.

Each backfill operation is a function registered in BACKFILL_REGISTRY.
When a new migration needs data backfill, add a function here.

Usage:
    python3 scripts/backfill.py --list
    python3 scripts/backfill.py --run <name> --dry-run
    python3 scripts/backfill.py --run <name> --batch-size 100

Design:
  - --dry-run always runs first (shows what would change)
  - --batch-size controls rows per iteration (default 100)
  - Idempotent: safe to run multiple times
  - Progress logging to stdout + optional log file
  - Exit codes: 0=success, 1=error, 2=no-op (nothing to backfill)

Created: 2026-02-16 (E5.2)
"""

from __future__ import annotations

import argparse
import os
import sqlite3
import sys
import time
from datetime import datetime
from typing import Callable

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)


# ============================================================
# BACKFILL REGISTRY
# ============================================================

# Each entry: name -> {func, description, db_key}
# db_key: "fts" | "prospective" (which DB to operate on)
BACKFILL_REGISTRY: dict[str, dict] = {}


def register_backfill(name: str, description: str, db_key: str = "fts"):
    """Decorator to register a backfill function."""
    def decorator(func: Callable):
        BACKFILL_REGISTRY[name] = {
            "func": func,
            "description": description,
            "db_key": db_key,
        }
        return func
    return decorator


# ============================================================
# BACKFILL FUNCTIONS
# ============================================================

@register_backfill(
    "003_write_queue_noop",
    "Migration 003: write_queue tables are new, no backfill needed. "
    "This exists as a template and to validate the dry-run pipeline.",
    db_key="fts",
)
def backfill_003_write_queue_noop(conn: sqlite3.Connection, batch_size: int, dry_run: bool) -> dict:
    """No-op backfill for migration 003 (new tables, no existing data)."""
    # Check tables exist
    tables = {r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    ).fetchall()}

    if "write_queue" not in tables:
        return {"status": "error", "message": "write_queue table missing. Run migrations first."}
    if "write_queue_log" not in tables:
        return {"status": "error", "message": "write_queue_log table missing. Run migrations first."}

    wq_count = conn.execute("SELECT count(*) FROM write_queue").fetchone()[0]
    wql_count = conn.execute("SELECT count(*) FROM write_queue_log").fetchone()[0]

    return {
        "status": "noop",
        "message": "No backfill needed for migration 003 (new tables).",
        "write_queue_rows": wq_count,
        "write_queue_log_rows": wql_count,
    }


@register_backfill(
    "example_add_column",
    "Example: backfill a hypothetical new column with default values. "
    "Template for real backfills.",
    db_key="fts",
)
def backfill_example_add_column(conn: sqlite3.Connection, batch_size: int, dry_run: bool) -> dict:
    """Example backfill: fill NULL values in a hypothetical column.

    This is a template showing the correct pattern for real backfills.
    """
    # Step 1: Count rows that need backfill
    # (In a real backfill, this would target rows with NULL in the new column)
    total_needing = conn.execute(
        "SELECT count(*) FROM tool_calls WHERE tag IS NULL"
    ).fetchone()[0]

    if total_needing == 0:
        return {"status": "noop", "message": "No rows need backfill.", "processed": 0}

    if dry_run:
        return {
            "status": "dry_run",
            "message": f"Would backfill {total_needing} rows in batches of {batch_size}.",
            "rows_needing_backfill": total_needing,
        }

    # Step 2: Process in batches
    processed = 0
    while True:
        # Find batch of rows to update
        rows = conn.execute(
            "SELECT id FROM tool_calls WHERE tag IS NULL LIMIT ?",
            (batch_size,)
        ).fetchall()

        if not rows:
            break

        ids = [r[0] for r in rows]
        placeholders = ",".join("?" * len(ids))

        conn.execute(
            f"UPDATE tool_calls SET tag = 'backfilled:example' "
            f"WHERE id IN ({placeholders})",
            ids,
        )
        conn.commit()

        processed += len(ids)
        _log(f"  Processed batch: {len(ids)} rows (total: {processed}/{total_needing})")

    return {
        "status": "done",
        "message": f"Backfilled {processed} rows.",
        "processed": processed,
        "total_expected": total_needing,
    }


# ============================================================
# LOGGING
# ============================================================

def _log(msg: str):
    """Print with timestamp."""
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}")


# ============================================================
# DB HELPERS
# ============================================================

DB_PATHS = {
    "fts": os.path.join(PROJECT_ROOT, "memories_fts.db"),
    "prospective": os.path.join(PROJECT_ROOT, "prospective.db"),
}


def _get_conn(db_key: str, db_path: str = None) -> sqlite3.Connection:
    """Get connection to the appropriate DB."""
    path = db_path or DB_PATHS.get(db_key)
    if not path or not os.path.exists(path):
        raise FileNotFoundError(f"DB not found: {path}")

    conn = sqlite3.connect(path, timeout=10)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=10000")
    return conn


# ============================================================
# RUNNER
# ============================================================

def run_backfill(name: str, dry_run: bool = True, batch_size: int = 100,
                 db_path: str = None) -> dict:
    """Run a registered backfill by name.

    Args:
        name: Backfill name from BACKFILL_REGISTRY
        dry_run: If True, only show what would change
        batch_size: Rows to process per batch
        db_path: Override DB path (for testing)

    Returns:
        dict with status, message, and operation-specific fields
    """
    if name not in BACKFILL_REGISTRY:
        return {"status": "error", "message": f"Unknown backfill: {name}"}

    entry = BACKFILL_REGISTRY[name]
    func = entry["func"]
    db_key = entry["db_key"]

    _log(f"Running backfill: {name}")
    _log(f"  Description: {entry['description']}")
    _log(f"  Mode: {'DRY RUN' if dry_run else 'LIVE'}")
    _log(f"  Batch size: {batch_size}")

    start = time.monotonic()

    try:
        conn = _get_conn(db_key, db_path)
        result = func(conn, batch_size, dry_run)
        conn.close()
    except Exception as e:
        result = {"status": "error", "message": f"{type(e).__name__}: {e}"}

    elapsed_ms = round((time.monotonic() - start) * 1000)
    result["elapsed_ms"] = elapsed_ms

    _log(f"  Result: {result.get('status', 'unknown')} ({elapsed_ms}ms)")
    if result.get("message"):
        _log(f"  Message: {result['message']}")

    return result


# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Backfill data after schema migrations"
    )
    parser.add_argument("--list", action="store_true", help="List available backfills")
    parser.add_argument("--run", type=str, help="Run a specific backfill by name")
    parser.add_argument("--dry-run", action="store_true", default=True,
                        help="Show what would change without modifying data (default)")
    parser.add_argument("--live", action="store_true",
                        help="Actually modify data (requires explicit flag)")
    parser.add_argument("--batch-size", type=int, default=100,
                        help="Rows per batch (default 100)")
    parser.add_argument("--all", action="store_true",
                        help="Run all registered backfills")
    args = parser.parse_args()

    if args.list:
        print("Available backfills:")
        for name, entry in sorted(BACKFILL_REGISTRY.items()):
            print(f"  {name}")
            print(f"    DB: {entry['db_key']}")
            print(f"    {entry['description']}")
            print()
        return

    dry_run = not args.live

    if args.all:
        results = {}
        for name in sorted(BACKFILL_REGISTRY.keys()):
            results[name] = run_backfill(name, dry_run=dry_run, batch_size=args.batch_size)
            print()
        # Summary
        print("=" * 40)
        print("SUMMARY")
        for name, r in results.items():
            print(f"  {name}: {r.get('status', '?')}")
        return

    if args.run:
        result = run_backfill(args.run, dry_run=dry_run, batch_size=args.batch_size)
        exit_code = 0 if result["status"] in ("done", "noop", "dry_run") else 1
        sys.exit(exit_code)

    parser.print_help()


if __name__ == "__main__":
    main()
