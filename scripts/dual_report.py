#!/usr/bin/env python3
"""
Dual Report: validates dual_ack write-path health with automated gate.

Queries dual_compare_log + write_queue to surface:
  - Volume by write_mode and compare_status
  - Match/mismatch breakdown (excluding expected dedup)
  - Sync-compare thread health (ok/timeout/skipped_capacity/error/pending)
  - Pending age for dual_ack rows (measured from created_at)
  - Write queue health (via get_queue_stats)
  - Go/no-go gate verdict

Usage:
  python scripts/dual_report.py
  python scripts/dual_report.py --hours 48
  python scripts/dual_report.py --db /path/to/memories_fts.db
"""

import argparse
import os
import sqlite3
import sys
from datetime import datetime, timedelta


# ── helpers ──────────────────────────────────────────────────────

def _pct(values: list, p: float):
    if not values:
        return None
    vs = sorted(values)
    if len(vs) == 1:
        return vs[0]
    k = (len(vs) - 1) * (p / 100.0)
    lo = int(k)
    hi = min(lo + 1, len(vs) - 1)
    return vs[lo] + (vs[hi] - vs[lo]) * (k - lo)


def _fmt_seconds(x) -> str:
    if x is None:
        return "-"
    if x < 60:
        return f"{x:.1f}s"
    return f"{x / 60:.1f}min"


def _section(title: str) -> None:
    print(f"\n{'=' * 50}")
    print(f"  {title}")
    print(f"{'=' * 50}")


def _table(headers: list, rows: list) -> None:
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            if i < len(widths):
                widths[i] = max(widths[i], len(str(cell)))
    fmt = "  ".join(f"{{:<{w}}}" for w in widths)
    print("  " + fmt.format(*headers))
    print("  " + "  ".join("-" * w for w in widths))
    for row in rows:
        print("  " + fmt.format(*[str(c) for c in row]))


def _resolve_db(cli_db: str) -> str:
    if cli_db:
        return cli_db
    env_db = os.environ.get("FTS_DB_PATH")
    if env_db:
        return env_db
    candidates = [
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "memories_fts.db"),
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "memories_fts.db"),
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return None


# ── main ─────────────────────────────────────────────────────────

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Dual-mode validation report with automated gate")
    ap.add_argument("--db", default=None, help="Path to SQLite DB (default: auto-detect)")
    ap.add_argument("--hours", type=int, default=48, help="Lookback window in hours")
    args = ap.parse_args(argv)

    db_path = _resolve_db(args.db)
    if not db_path or not os.path.exists(db_path):
        print("ERROR: DB not found. Use --db or set FTS_DB_PATH.", file=sys.stderr)
        return 1

    since = (datetime.now().astimezone() - timedelta(hours=args.hours)).isoformat()

    conn = sqlite3.connect(db_path, timeout=5)
    conn.row_factory = sqlite3.Row

    # Check table exists
    tables = {r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    ).fetchall()}
    if "dual_compare_log" not in tables:
        print("ERROR: dual_compare_log table not found. Run migration 007 first.", file=sys.stderr)
        conn.close()
        return 1

    # Check if new columns exist (migration 010)
    cols = {r[1] for r in conn.execute("PRAGMA table_info(dual_compare_log)").fetchall()}
    has_created_at = "created_at" in cols
    has_sync_status = "sync_compare_status" in cols

    # Use created_at if available, otherwise fall back to sync_completed_at
    time_col = "created_at" if has_created_at else "sync_completed_at"

    print(f"\nDUAL REPORT")
    print(f"DB: {db_path}")
    print(f"Window: last {args.hours}h")
    print(f"Time column: {time_col}")
    print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    gate_fail_reasons = []
    warnings = []

    # ── 1. Volume ──
    _section("1. VOLUME")
    rows = conn.execute(
        f"SELECT write_mode, compare_status, COUNT(*) as cnt "
        f"FROM dual_compare_log WHERE {time_col} >= ? "
        f"GROUP BY write_mode, compare_status",
        (since,),
    ).fetchall()
    total = sum(r["cnt"] for r in rows)
    if rows:
        _table(["Write Mode", "Compare Status", "Count"], [
            [r["write_mode"], r["compare_status"], str(r["cnt"])] for r in rows
        ] + [["TOTAL", "", str(total)]])
    else:
        print("  (no rows in window)")

    # Count dual_ack rows for percentage calculations later
    dual_ack_total = sum(r["cnt"] for r in rows if r["write_mode"] == "dual_ack")

    # ── 2. Match/Mismatch ──
    _section("2. MATCH / MISMATCH")
    mismatch_rows = conn.execute(
        f"SELECT divergence_code, COUNT(*) as cnt "
        f"FROM dual_compare_log "
        f"WHERE {time_col} >= ? "
        f"  AND compare_status = 'computed' "
        f"  AND match = 0 "
        f"  AND divergence_code NOT IN ('expected_shadow_dedup', 'expected_dual_dedup', 'expected_legacy_dedup') "
        f"GROUP BY divergence_code",
        (since,),
    ).fetchall()
    real_mismatch = sum(r["cnt"] for r in mismatch_rows)
    if mismatch_rows:
        _table(["Divergence Code", "Count"], [
            [r["divergence_code"], str(r["cnt"])] for r in mismatch_rows
        ] + [["REAL MISMATCH TOTAL", str(real_mismatch)]])
    else:
        print("  No real mismatches (good)")

    # Also show expected dedup for reference
    expected_rows = conn.execute(
        f"SELECT divergence_code, COUNT(*) as cnt "
        f"FROM dual_compare_log "
        f"WHERE {time_col} >= ? "
        f"  AND compare_status = 'computed' "
        f"  AND divergence_code IN ('expected_shadow_dedup', 'expected_dual_dedup', 'expected_legacy_dedup', 'canceled_tombstone') "
        f"GROUP BY divergence_code",
        (since,),
    ).fetchall()
    if expected_rows:
        print("\n  Expected dedup (not counted as mismatch):")
        for r in expected_rows:
            print(f"    {r['divergence_code']}: {r['cnt']}")

    match_count = conn.execute(
        f"SELECT COUNT(*) FROM dual_compare_log "
        f"WHERE {time_col} >= ? AND compare_status = 'computed' AND match = 1",
        (since,),
    ).fetchone()[0]
    computed = conn.execute(
        f"SELECT COUNT(*) FROM dual_compare_log "
        f"WHERE {time_col} >= ? AND compare_status = 'computed'",
        (since,),
    ).fetchone()[0]
    print(f"\n  Computed: {computed}  |  Match: {match_count}  |  Real mismatch: {real_mismatch}")

    if real_mismatch > 0:
        gate_fail_reasons.append(f"real_mismatch={real_mismatch} (must be 0)")

    # ── 3. Sync Compare Health ──
    _section("3. SYNC COMPARE HEALTH")
    if has_sync_status:
        sc_rows = conn.execute(
            f"SELECT sync_compare_status, COUNT(*) as cnt "
            f"FROM dual_compare_log "
            f"WHERE {time_col} >= ? "
            f"GROUP BY sync_compare_status",
            (since,),
        ).fetchall()
        sc_counts = {r["sync_compare_status"]: r["cnt"] for r in sc_rows}
        if sc_rows:
            _table(["Status", "Count"], [
                [r["sync_compare_status"], str(r["cnt"])] for r in sc_rows
            ])
        else:
            print("  (no data)")

        # Warnings for high skipped/timeout rates
        skipped = sc_counts.get("skipped_capacity", 0)
        timeout = sc_counts.get("timeout", 0)
        if dual_ack_total > 0:
            skip_pct = skipped / dual_ack_total * 100
            timeout_pct = timeout / dual_ack_total * 100
            if skip_pct > 5:
                warnings.append(f"skipped_capacity={skipped} ({skip_pct:.1f}% of dual_ack, threshold 5%)")
            if timeout_pct > 10:
                warnings.append(f"timeout={timeout} ({timeout_pct:.1f}% of dual_ack, threshold 10%)")
    else:
        print("  (sync_compare_status column not available - run migration 010)")

    # ── 4. Pending Age ──
    _section("4. PENDING AGE (dual_ack rows)")
    if has_created_at:
        age_rows = conn.execute(
            "SELECT (julianday('now') - julianday(created_at)) * 86400 as age_seconds "
            "FROM dual_compare_log "
            "WHERE created_at >= ? "
            "  AND compare_status = 'pending' "
            "  AND write_mode = 'dual_ack' "
            "  AND sync_compare_status = 'ok' "
            "ORDER BY age_seconds DESC",
            (since,),
        ).fetchall()
        ages = [r["age_seconds"] for r in age_rows if r["age_seconds"] is not None]
        pending_count = len(ages)
        if ages:
            p50 = _pct(ages, 50)
            p95 = _pct(ages, 95)
            age_max = max(ages)
            _table(["Metric", "Value"], [
                ["Pending count", str(pending_count)],
                ["p50", _fmt_seconds(p50)],
                ["p95", _fmt_seconds(p95)],
                ["max", _fmt_seconds(age_max)],
            ])
            if age_max > 600:  # 10 minutes
                gate_fail_reasons.append(
                    f"pending_age_max={_fmt_seconds(age_max)} (threshold 10min)"
                )
        else:
            print("  No pending dual_ack rows (good)")
    else:
        print("  (created_at column not available - run migration 010)")

    # ── 5. Write Queue Health ──
    _section("5. WRITE QUEUE HEALTH")
    wq_dead = 0
    wq_failed = 0
    wq_total = 0
    if "write_queue" in tables:
        try:
            # Import get_queue_stats for read-only access
            sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            from modules.write_queue import get_queue_stats
            stats = get_queue_stats(db_path)
            wq_total = stats.get("total", 0)
            by_status = stats.get("by_status", {})
            wq_dead = by_status.get("dead", 0)
            wq_failed = by_status.get("failed", 0)

            if by_status:
                _table(["Status", "Count"], [
                    [status, str(cnt)] for status, cnt in sorted(by_status.items())
                ] + [["TOTAL", str(wq_total)]])
            else:
                print("  (empty queue)")

            if stats.get("oldest_queued"):
                print(f"\n  Oldest queued: {stats['oldest_queued']}")
        except Exception as exc:
            print(f"  Error reading write_queue: {exc}")
    else:
        print("  (write_queue table not found)")

    if wq_dead > 0:
        gate_fail_reasons.append(f"write_queue dead={wq_dead} (must be 0)")
    if wq_total > 0 and wq_failed / wq_total >= 0.01:
        gate_fail_reasons.append(
            f"write_queue failed/total={wq_failed}/{wq_total} "
            f"({wq_failed / wq_total * 100:.1f}% >= 1%)"
        )

    # ── 6. Gate Verdict ──
    _section("6. GATE VERDICT")

    if warnings:
        print("  Warnings:")
        for w in warnings:
            print(f"    - {w}")
        print()

    if gate_fail_reasons:
        print("  VERDICT: FAIL")
        for reason in gate_fail_reasons:
            print(f"    - {reason}")
        conn.close()
        return 1
    else:
        if total == 0:
            print("  VERDICT: PASS (no data in window)")
        else:
            print("  VERDICT: PASS")
            print(f"    - rows={total}, real_mismatch=0, dead={wq_dead}")

    conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
