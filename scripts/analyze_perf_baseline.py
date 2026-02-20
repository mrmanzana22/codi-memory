#!/usr/bin/env python3
"""
Codi Memory - Performance Baseline Analyzer (E0.1)

Reads tool_calls from SQLite and produces:
  1. Per-tool percentile table (p50/p95/p99/max/avg/error_rate)
  2. Outlier analysis (top N slowest calls per critical tool)
  3. JSON summary for downstream use

Usage:
  python scripts/analyze_perf_baseline.py [--days 30] [--output docs/perf/] [--json]

Output:
  - stdout: human-readable table
  - docs/perf/baseline_YYYY-MM-DD.json (if --json)
"""

import argparse
import json
import os
import sqlite3
import sys
from datetime import datetime, timezone, timedelta

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FTS_DB_PATH = os.path.join(BASE_DIR, "memories_fts.db")

# Tools considered "critical" for SLO tracking
CRITICAL_TOOLS = [
    "despertar_codi", "recall", "remember", "search_memory",
    "checkpoint_memoria", "add_memory", "add_memory_smart",
    "assessment_report", "context_snapshot", "flush_session",
]

# SLO contracts for comparison (v0 targets)
SLO_TARGETS = {
    "despertar_codi":     {"p50": 4000, "p95": 8000,  "p99": 15000},
    "recall":             {"p50": 3000, "p95": 7000,  "p99": 12000},
    "remember":           {"p50": 2000, "p95": 5000,  "p99": 10000},  # async ACK target
    "search_memory":      {"p50": 2500, "p95": 5000,  "p99": 8000},
    "checkpoint_memoria": {"p50": 2000, "p95": 5000,  "p99": 10000},  # async ACK target
    "add_memory":         {"p50": 2000, "p95": 5000,  "p99": 10000},  # async ACK target
}


def get_conn():
    if not os.path.exists(FTS_DB_PATH):
        print(f"ERROR: DB not found at {FTS_DB_PATH}", file=sys.stderr)
        sys.exit(1)
    conn = sqlite3.connect(FTS_DB_PATH, timeout=5)
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def compute_percentiles(durations: list[int]) -> dict:
    """Compute p50/p95/p99/max/avg from sorted durations."""
    if not durations:
        return {"count": 0, "p50": 0, "p95": 0, "p99": 0, "max": 0, "avg": 0}
    durations.sort()
    n = len(durations)
    return {
        "count": n,
        "p50": durations[int(n * 0.50)],
        "p95": durations[min(int(n * 0.95), n - 1)],
        "p99": durations[min(int(n * 0.99), n - 1)],
        "max": durations[-1],
        "avg": round(sum(durations) / n),
    }


def analyze_tools(conn, cutoff_iso: str) -> list[dict]:
    """Analyze all tools from tool_calls table."""
    rows = conn.execute(
        "SELECT tool_name, COUNT(*) as cnt "
        "FROM tool_calls WHERE started_at >= ? "
        "GROUP BY tool_name ORDER BY cnt DESC",
        (cutoff_iso,),
    ).fetchall()

    results = []
    for name, cnt in rows:
        durations = [
            r[0] for r in conn.execute(
                "SELECT duration_ms FROM tool_calls "
                "WHERE tool_name = ? AND started_at >= ? "
                "ORDER BY duration_ms",
                (name, cutoff_iso),
            ).fetchall()
        ]
        errors = conn.execute(
            "SELECT COUNT(*) FROM tool_calls "
            "WHERE tool_name = ? AND started_at >= ? AND success = 0",
            (name, cutoff_iso),
        ).fetchone()[0]

        pcts = compute_percentiles(durations)
        pcts["error_rate"] = round((errors / cnt * 100) if cnt else 0, 1)
        pcts["tool"] = name
        pcts["is_critical"] = name in CRITICAL_TOOLS

        # SLO comparison
        slo = SLO_TARGETS.get(name)
        if slo:
            violations = []
            for level in ("p50", "p95", "p99"):
                if pcts[level] > slo[level]:
                    violations.append(f"{level}({pcts[level]}ms > {slo[level]}ms)")
            pcts["slo_status"] = "VIOLATION" if violations else "OK"
            pcts["slo_violations"] = violations
        else:
            pcts["slo_status"] = "no_slo"
            pcts["slo_violations"] = []

        results.append(pcts)
    return results


def analyze_outliers(conn, cutoff_iso: str, top_n: int = 10) -> dict:
    """Get top N slowest calls per critical tool."""
    outliers = {}
    for tool in CRITICAL_TOOLS:
        rows = conn.execute(
            "SELECT started_at, duration_ms, session_id, tag "
            "FROM tool_calls "
            "WHERE tool_name = ? AND started_at >= ? "
            "ORDER BY duration_ms DESC LIMIT ?",
            (tool, cutoff_iso, top_n),
        ).fetchall()
        if rows:
            outliers[tool] = [
                {"started_at": r[0], "duration_ms": r[1],
                 "session_id": r[2], "tag": r[3]}
                for r in rows
            ]
    return outliers


def print_table(results: list[dict]):
    """Print human-readable table to stdout."""
    hdr = f"{'Tool':35s} | {'N':>5s} | {'p50':>7s} | {'p95':>7s} | {'p99':>7s} | {'max':>7s} | {'avg':>7s} | {'err%':>5s} | {'SLO':>10s}"
    sep = "-" * len(hdr)
    print(hdr)
    print(sep)
    for r in results:
        marker = "*" if r["is_critical"] else " "
        slo_str = r.get("slo_status", "-")
        print(
            f"{r['tool']:35s} | {r['count']:5d} | "
            f"{r['p50']:6d}ms | {r['p95']:6d}ms | {r['p99']:6d}ms | "
            f"{r['max']:6d}ms | {r['avg']:6d}ms | {r['error_rate']:4.1f}% | "
            f"{slo_str:>10s} {marker}"
        )
    print(f"\n* = critical tool (SLO tracked)")


def main():
    parser = argparse.ArgumentParser(description="Codi Performance Baseline Analyzer")
    parser.add_argument("--days", type=int, default=30, help="Lookback period in days")
    parser.add_argument("--output", default=None, help="Output directory for JSON")
    parser.add_argument("--json", action="store_true", help="Write JSON summary")
    parser.add_argument("--top-outliers", type=int, default=10, help="Top N outliers per tool")
    args = parser.parse_args()

    conn = get_conn()

    cutoff = (datetime.now(timezone.utc) - timedelta(days=args.days)).isoformat()

    # Date range
    first = conn.execute("SELECT MIN(started_at) FROM tool_calls").fetchone()[0]
    last = conn.execute("SELECT MAX(started_at) FROM tool_calls").fetchone()[0]
    total = conn.execute("SELECT COUNT(*) FROM tool_calls WHERE started_at >= ?", (cutoff,)).fetchone()[0]

    print(f"=== CODI PERFORMANCE BASELINE ===")
    print(f"Period: {args.days} days (cutoff: {cutoff[:10]})")
    print(f"Data range: {first[:10] if first else '?'} to {last[:10] if last else '?'}")
    print(f"Total calls: {total}")
    print()

    # Analyze
    results = analyze_tools(conn, cutoff)
    outliers = analyze_outliers(conn, cutoff, top_n=args.top_outliers)

    # Print table
    print_table(results)

    # Print SLO violations
    violated = [r for r in results if r.get("slo_status") == "VIOLATION"]
    if violated:
        print(f"\n=== SLO VIOLATIONS ({len(violated)} tools) ===")
        for r in violated:
            print(f"  {r['tool']}: {', '.join(r['slo_violations'])}")

    # Print outliers for critical tools
    print(f"\n=== TOP OUTLIERS (critical tools) ===")
    for tool, entries in outliers.items():
        if entries:
            worst = entries[0]
            print(f"  {tool}: worst={worst['duration_ms']}ms at {worst['started_at'][:19]}")

    # JSON output
    if args.json:
        out_dir = args.output or os.path.join(BASE_DIR, "docs", "perf")
        os.makedirs(out_dir, exist_ok=True)
        today = datetime.now().strftime("%Y-%m-%d")
        out_path = os.path.join(out_dir, f"baseline_{today}.json")

        summary = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "period_days": args.days,
            "data_range": {"first": first, "last": last},
            "total_calls": total,
            "tools": results,
            "outliers": outliers,
            "slo_targets": SLO_TARGETS,
        }

        with open(out_path, "w") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False, default=str)
        print(f"\nJSON summary written to: {out_path}")

    conn.close()


if __name__ == "__main__":
    main()
