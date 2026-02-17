#!/usr/bin/env python3
"""
Shadow Report: validates async write-path parity in shadow mode.

Queries write_queue + write_queue_log to surface:
  - Backlog by status (queued/running/done/failed/dead)
  - Volume by job kind
  - Retry health (avg attempts, dead count)
  - Dedupe effectiveness
  - Latency percentiles per kind (from write_queue_log.duration_ms)
  - Timeline (jobs/hour)
  - Go/no-go gate verdict

Usage:
  python scripts/shadow_report.py
  python scripts/shadow_report.py --hours 48
  python scripts/shadow_report.py --db /path/to/memories_fts.db
"""

import argparse
import os
import sqlite3
import sys
from datetime import datetime, timedelta, timezone


# ── helpers ──────────────────────────────────────────────────────

def _pct(values: list[float], p: float) -> float | None:
    if not values:
        return None
    vs = sorted(values)
    if len(vs) == 1:
        return vs[0]
    k = (len(vs) - 1) * (p / 100.0)
    lo = int(k)
    hi = min(lo + 1, len(vs) - 1)
    return vs[lo] + (vs[hi] - vs[lo]) * (k - lo)


def _fmt_ms(x: float | None) -> str:
    if x is None:
        return "-"
    return f"{x:.0f}ms" if x < 1000 else f"{x / 1000:.2f}s"


def _section(title: str) -> None:
    print(f"\n{'=' * 50}")
    print(f"  {title}")
    print(f"{'=' * 50}")


def _table(headers: list[str], rows: list[list[str]]) -> None:
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            if i < len(widths):
                widths[i] = max(widths[i], len(cell))
    fmt = "  ".join(f"{{:<{w}}}" for w in widths)
    print("  " + fmt.format(*headers))
    print("  " + "  ".join("-" * w for w in widths))
    for row in rows:
        print("  " + fmt.format(*row))


# ── main ─────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser(description="Shadow mode validation report")
    ap.add_argument("--db", default=None, help="Path to SQLite DB (default: auto-detect)")
    ap.add_argument("--hours", type=int, default=48, help="Lookback window in hours")
    args = ap.parse_args()

    # Resolve DB path
    db_path = args.db
    if not db_path:
        db_path = os.environ.get("FTS_DB_PATH")
    if not db_path:
        candidates = [
            os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "memories_fts.db"),
            os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "memories_fts.db"),
        ]
        for c in candidates:
            if os.path.exists(c):
                db_path = c
                break
    if not db_path or not os.path.exists(db_path):
        print("ERROR: DB not found. Use --db or set FTS_DB_PATH.", file=sys.stderr)
        return 1

    # Use local time for since (DB stores timestamps with local offset)
    since = (datetime.now().astimezone() - timedelta(hours=args.hours)).isoformat()

    conn = sqlite3.connect(db_path, timeout=5)
    conn.row_factory = sqlite3.Row

    # Check tables exist
    tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
    if "write_queue" not in tables:
        print("ERROR: write_queue table not found. Run migration 003 first.", file=sys.stderr)
        conn.close()
        return 1

    has_log = "write_queue_log" in tables

    print(f"\nSHADOW REPORT")
    print(f"DB: {db_path}")
    print(f"Window: last {args.hours}h")
    print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # ── 1. Backlog by status ──
    _section("1. BACKLOG BY STATUS")
    rows = conn.execute(
        "SELECT status, COUNT(*) as cnt FROM write_queue GROUP BY status ORDER BY cnt DESC"
    ).fetchall()
    total = sum(r["cnt"] for r in rows)
    if rows:
        _table(["Status", "Count", "%"], [
            [r["status"], str(r["cnt"]), f"{r['cnt'] / total * 100:.1f}%"] for r in rows
        ] + [["TOTAL", str(total), "100%"]])
    else:
        print("  (empty queue)")

    active = conn.execute(
        "SELECT COUNT(*) FROM write_queue WHERE status IN ('queued', 'running')"
    ).fetchone()[0]
    print(f"\n  Active backlog (queued+running): {active}")

    # ── 2. Volume by kind (window) ──
    _section("2. VOLUME BY KIND (window)")
    rows = conn.execute(
        "SELECT kind, COUNT(*) as cnt FROM write_queue WHERE created_at >= ? GROUP BY kind ORDER BY cnt DESC",
        (since,)
    ).fetchall()
    total_window = sum(r["cnt"] for r in rows) if rows else 0
    if rows:
        _table(["Kind", "Count"], [[r["kind"], str(r["cnt"])] for r in rows])
    else:
        print("  (no jobs in window)")

    # ── 3. Retry health ──
    _section("3. RETRY HEALTH")
    retry_rows = conn.execute(
        """SELECT kind,
                  COUNT(*) as total,
                  SUM(CASE WHEN attempts > 1 THEN 1 ELSE 0 END) as retried,
                  AVG(attempts) as avg_attempts,
                  MAX(attempts) as max_attempts
           FROM write_queue
           WHERE created_at >= ?
           GROUP BY kind""",
        (since,)
    ).fetchall()
    if retry_rows:
        _table(["Kind", "Total", "Retried", "Avg Attempts", "Max"], [
            [r["kind"], str(r["total"]), str(r["retried"] or 0),
             f"{r['avg_attempts']:.2f}", str(r["max_attempts"])]
            for r in retry_rows
        ])
    else:
        print("  (no data)")

    dead = conn.execute(
        "SELECT COUNT(*) FROM write_queue WHERE status = 'dead' AND created_at >= ?", (since,)
    ).fetchone()[0]
    failed = conn.execute(
        "SELECT COUNT(*) FROM write_queue WHERE status = 'failed' AND created_at >= ?", (since,)
    ).fetchone()[0]
    print(f"\n  Dead: {dead}  |  Failed (pending retry): {failed}")

    if dead > 0:
        print("\n  Dead jobs (last 5):")
        dead_rows = conn.execute(
            """SELECT job_id, kind, attempts, last_error, created_at
               FROM write_queue WHERE status = 'dead' AND created_at >= ?
               ORDER BY created_at DESC LIMIT 5""",
            (since,)
        ).fetchall()
        for d in dead_rows:
            err = (d["last_error"] or "")[:80]
            print(f"    {d['job_id'][:8]}.. {d['kind']} attempts={d['attempts']} err={err}")

    # ── 4. Dedupe ──
    _section("4. DEDUPE")
    with_dedupe = conn.execute(
        "SELECT COUNT(*) FROM write_queue WHERE dedupe_key IS NOT NULL AND created_at >= ?", (since,)
    ).fetchone()[0]
    # Dedupe collisions: same dedupe_key appearing more than once
    dupes = conn.execute(
        """SELECT dedupe_key, COUNT(*) as cnt
           FROM write_queue
           WHERE dedupe_key IS NOT NULL AND created_at >= ?
           GROUP BY dedupe_key HAVING cnt > 1
           ORDER BY cnt DESC LIMIT 5""",
        (since,)
    ).fetchall()
    print(f"  Jobs with dedupe_key: {with_dedupe}/{total_window}")
    if dupes:
        print(f"  Duplicate dedupe_keys found: {len(dupes)}")
        for d in dupes:
            print(f"    {d['dedupe_key'][:16]}.. count={d['cnt']}")
    else:
        print("  No duplicate dedupe_keys (good)")

    # ── 5. Latency Breakdown ──
    _section("5. LATENCY BREAKDOWN")
    lat_rows = conn.execute(
        """SELECT kind, created_at, claimed_at, started_at, completed_at
           FROM write_queue
           WHERE status = 'done' AND completed_at IS NOT NULL AND created_at >= ?""",
        (since,)
    ).fetchall()

    by_kind_queue: dict[str, list[float]] = {}
    by_kind_exec: dict[str, list[float]] = {}
    by_kind_total: dict[str, list[float]] = {}

    for r in lat_rows:
        try:
            created = datetime.fromisoformat(r["created_at"])
            completed = datetime.fromisoformat(r["completed_at"])
            total_ms = (completed - created).total_seconds() * 1000.0
            by_kind_total.setdefault(r["kind"], []).append(total_ms)

            if r["claimed_at"]:
                claimed = datetime.fromisoformat(r["claimed_at"])
                queue_ms = (claimed - created).total_seconds() * 1000.0
                by_kind_queue.setdefault(r["kind"], []).append(queue_ms)

            if r["started_at"]:
                started = datetime.fromisoformat(r["started_at"])
                exec_ms = (completed - started).total_seconds() * 1000.0
                by_kind_exec.setdefault(r["kind"], []).append(exec_ms)
        except Exception:
            pass

    all_kinds = sorted(set(list(by_kind_total.keys()) + list(by_kind_queue.keys()) + list(by_kind_exec.keys())))

    if all_kinds:
        # Total latency (end-to-end)
        print("\n  -- Total (created_at -> completed_at) --")
        _table(["Kind", "N", "p50", "p95", "max"], [
            [k, str(len(by_kind_total.get(k, []))),
             _fmt_ms(_pct(by_kind_total.get(k, []), 50)),
             _fmt_ms(_pct(by_kind_total.get(k, []), 95)),
             _fmt_ms(max(by_kind_total[k]) if k in by_kind_total else None)]
            for k in all_kinds
        ])

        # Queue wait (created_at -> claimed_at)
        if any(by_kind_queue.values()):
            print("\n  -- Queue Wait (created_at -> claimed_at) --")
            _table(["Kind", "N", "p50", "p95", "max"], [
                [k, str(len(by_kind_queue.get(k, []))),
                 _fmt_ms(_pct(by_kind_queue.get(k, []), 50)),
                 _fmt_ms(_pct(by_kind_queue.get(k, []), 95)),
                 _fmt_ms(max(by_kind_queue[k]) if k in by_kind_queue else None)]
                for k in all_kinds if k in by_kind_queue
            ])
        else:
            print("\n  -- Queue Wait: no data (pre-004 jobs, claimed_at not yet recorded) --")

        # Exec time (started_at -> completed_at)
        if any(by_kind_exec.values()):
            print("\n  -- Exec Time (started_at -> completed_at) --")
            _table(["Kind", "N", "p50", "p95", "max"], [
                [k, str(len(by_kind_exec.get(k, []))),
                 _fmt_ms(_pct(by_kind_exec.get(k, []), 50)),
                 _fmt_ms(_pct(by_kind_exec.get(k, []), 95)),
                 _fmt_ms(max(by_kind_exec[k]) if k in by_kind_exec else None)]
                for k in all_kinds if k in by_kind_exec
            ])
        else:
            print("\n  -- Exec Time: no data (pre-004 jobs, started_at not yet recorded) --")
    else:
        print("  (no completed jobs in window)")

    # ── 6. Timeline ──
    _section("6. TIMELINE (jobs/hour)")
    timeline_rows = conn.execute(
        """SELECT SUBSTR(created_at, 1, 13) || ':00' as hour, COUNT(*) as cnt
           FROM write_queue
           WHERE created_at >= ?
           GROUP BY hour
           ORDER BY hour""",
        (since,)
    ).fetchall()
    if timeline_rows:
        max_cnt = max(r["cnt"] for r in timeline_rows)
        bar_width = 30
        for r in timeline_rows:
            bar_len = int(r["cnt"] / max_cnt * bar_width) if max_cnt > 0 else 0
            bar = "#" * bar_len
            print(f"  {r['hour']}  {str(r['cnt']).rjust(4)}  {bar}")
    else:
        print("  (no jobs in window)")

    # ── 7. Shadow gate ──
    _section("7. SHADOW GATE (go/no-go)")
    issues = []
    if dead > 0:
        issues.append(f"dead={dead} (must be 0 for 24h)")
    if total_window > 0 and failed / total_window > 0.01:
        issues.append(f"failed/total={failed}/{total_window} ({failed / total_window * 100:.1f}%) > 1%")
    if active > 10:
        issues.append(f"active backlog={active} (growing?)")

    if issues:
        print("  VERDICT: NOT READY")
        for iss in issues:
            print(f"    - {iss}")
    else:
        if total_window == 0:
            print("  VERDICT: NO DATA YET (need 24-48h of shadow usage)")
        else:
            print(f"  VERDICT: PASS - ready to consider async rollout")
            print(f"    - dead=0, failed/total={failed}/{total_window}, active_backlog={active}")

    conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
