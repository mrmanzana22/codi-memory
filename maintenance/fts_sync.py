#!/usr/bin/env python3
"""
FTS Sync — Backfill + Orphan Cleanup
Proposal #291 — run once to repair FTS coverage

Usage:
    cd ~/codi-memory && python3 maintenance/fts_sync.py

What it does:
  1. Backfills PG memories missing from SQLite FTS (currently ~1,051)
  2. Removes FTS entries orphaned from deleted PG memories (~3,086)

Safe to re-run: uses INSERT OR REPLACE + LIMIT batches.
SQLite triggers (memories_text_ai, memories_text_ad) auto-sync memories_fts.
"""
import sys
import sqlite3
from pathlib import Path

# Ensure modules are importable
sys.path.insert(0, str(Path(__file__).parent.parent))

from modules.config import FTS_DB_PATH
from modules.config_pg import get_conn as pg_conn
from modules.memory_smart import _index_memory_fts_raw, _delete_memory_fts_raw

BATCH_SIZE = 200


def run_fts_sync(dry_run: bool = False):
    print("=== FTS Sync: Backfill + Orphan Cleanup ===")

    # 1. Load all PG memories
    print("Loading PG memories...")
    with pg_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT id::text, content, category, source, importance FROM memories"
            )
            pg_rows = cur.fetchall()
    pg_ids = {r[0] for r in pg_rows}
    pg_data = {r[0]: r for r in pg_rows}
    print(f"  PG total: {len(pg_ids)}")

    # 2. Load all FTS memory_ids
    print("Loading FTS index...")
    fts_conn = sqlite3.connect(FTS_DB_PATH)
    fts_ids = set(
        r[0] for r in fts_conn.execute(
            "SELECT memory_id FROM memories_text"
        ).fetchall()
    )
    fts_conn.close()
    print(f"  FTS total: {len(fts_ids)}")

    missing = pg_ids - fts_ids
    orphaned = fts_ids - pg_ids
    print(f"\nGap analysis:")
    print(f"  Missing from FTS (to backfill): {len(missing)}")
    print(f"  Orphaned in FTS (to clean):     {len(orphaned)}")

    if dry_run:
        print("\n[DRY RUN — no changes made]")
        return

    # 3. Backfill missing memories
    print(f"\nBackfilling {len(missing)} memories...")
    backfilled = 0
    errors = 0
    for i, mid in enumerate(missing, 1):
        row = pg_data[mid]
        _, content, category, source, importance = row
        try:
            _index_memory_fts_raw(
                memory_id=mid,
                content=(content or "")[:2000],
                category=category or "general",
                source=source or "experienced",
                importance=importance or "medium",
            )
            backfilled += 1
        except Exception as e:
            print(f"  [SKIP backfill] {mid[:8]}...: {e}")
            errors += 1
        if i % BATCH_SIZE == 0:
            print(f"  Progress: {i}/{len(missing)} ({backfilled} ok, {errors} skip)")

    print(f"  Backfill done: {backfilled} indexed, {errors} skipped")

    # 4. Clean orphaned FTS entries
    print(f"\nCleaning {len(orphaned)} orphaned FTS entries...")
    cleaned = 0
    errors = 0
    for i, mid in enumerate(orphaned, 1):
        try:
            _delete_memory_fts_raw(mid)
            cleaned += 1
        except Exception as e:
            print(f"  [SKIP cleanup] {mid[:8]}...: {e}")
            errors += 1
        if i % BATCH_SIZE == 0:
            print(f"  Progress: {i}/{len(orphaned)} ({cleaned} ok, {errors} skip)")

    print(f"  Cleanup done: {cleaned} removed, {errors} skipped")

    # 5. Verify
    fts_conn = sqlite3.connect(FTS_DB_PATH)
    final_fts = fts_conn.execute("SELECT COUNT(*) FROM memories_text").fetchone()[0]
    fts_conn.close()
    expected_coverage = len(pg_ids)
    print(f"\n=== Final state ===")
    print(f"  PG memories:  {len(pg_ids)}")
    print(f"  FTS entries:  {final_fts}")
    print(f"  Coverage:     {min(final_fts, len(pg_ids))/len(pg_ids)*100:.1f}%")
    print(f"  Backfilled:   {backfilled}")
    print(f"  Cleaned:      {cleaned}")


if __name__ == "__main__":
    dry_run = "--dry-run" in sys.argv
    run_fts_sync(dry_run=dry_run)
