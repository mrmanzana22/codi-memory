#!/usr/bin/env python3
"""Backfill FHRR session index from historical session_checkpoints.

Encodes sessions that exist in session_checkpoints but not yet in
fhrr_session_index. Uses pgvector memories as source of turn content.

Usage:
    python3 scripts/backfill_fhrr_sessions.py [--limit N] [--dry-run]

Paper: Diekelmann & Born 2010 (offline hippocampal replay)
"""
import argparse
import sys
import time

sys.path.insert(0, ".")

from modules.config import connect_fts, FTS_DB_PATH


def get_unindexed_sessions(limit: int = 200) -> list[dict]:
    """Find sessions in checkpoints but not in FHRR index."""
    db = connect_fts(FTS_DB_PATH)
    rows = db.execute(
        """SELECT sc.session_id, sc.session_summary, sc.created_at
           FROM session_checkpoints sc
           LEFT JOIN fhrr_session_index fi ON sc.session_id = fi.session_id
           WHERE fi.session_id IS NULL
           ORDER BY sc.created_at DESC
           LIMIT ?""",
        (limit,),
    ).fetchall()
    db.close()
    return [{"session_id": r[0], "summary": r[1], "created_at": r[2]} for r in rows]


def backfill(limit: int = 200, dry_run: bool = False) -> dict:
    """Encode un-indexed sessions."""
    sessions = get_unindexed_sessions(limit)
    print(f"Found {len(sessions)} un-indexed sessions")

    if dry_run:
        for s in sessions[:10]:
            print(f"  {s['session_id'][:8]} | {s['created_at']} | {(s['summary'] or '')[:60]}")
        if len(sessions) > 10:
            print(f"  ... and {len(sessions) - 10} more")
        return {"total": len(sessions), "encoded": 0, "errors": 0, "dry_run": True}

    from modules.hippocampal_index import encode_current_session

    encoded = 0
    errors = 0
    t0 = time.monotonic()

    for i, s in enumerate(sessions):
        sid = s["session_id"]
        try:
            encode_current_session(session_id=sid)
            encoded += 1
            if (i + 1) % 10 == 0:
                elapsed = time.monotonic() - t0
                print(f"  [{i+1}/{len(sessions)}] encoded={encoded} errors={errors} elapsed={elapsed:.1f}s")
        except Exception as e:
            errors += 1
            print(f"  ERROR {sid[:8]}: {e}")

    elapsed = time.monotonic() - t0
    print(f"\nDone: encoded={encoded} errors={errors} elapsed={elapsed:.1f}s")
    return {"total": len(sessions), "encoded": encoded, "errors": errors, "elapsed_s": round(elapsed, 1)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Backfill FHRR session index")
    parser.add_argument("--limit", type=int, default=200, help="Max sessions to encode")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be encoded")
    args = parser.parse_args()

    result = backfill(limit=args.limit, dry_run=args.dry_run)
    sys.exit(0 if result["errors"] == 0 else 1)
