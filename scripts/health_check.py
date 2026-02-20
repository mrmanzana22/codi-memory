#!/usr/bin/env python3
"""
Unified Health Check: runs all validation scripts and reports a single verdict.

Orchestrates:
  1. Repo Hygiene  (check_repo_hygiene.py) - no sensitive files in git
  2. Memory Hygiene (memory_hygiene.py)    - metadata integrity in Qdrant
  3. Dual Report    (dual_report.py)       - write-path health
  4. Test Suite     (pytest)               - optional, runs if --tests flag

Each sub-check produces PASS/FAIL. Overall verdict is FAIL if any sub-check fails.

Usage:
  python scripts/health_check.py
  python scripts/health_check.py --skip-qdrant    # skip memory hygiene (no Qdrant)
  python scripts/health_check.py --skip-dual      # skip dual report (no DB)
  python scripts/health_check.py --tests          # also run pytest
  python scripts/health_check.py --hours 24       # dual report lookback window
"""

import argparse
import io
import os
import sys
import time
from contextlib import redirect_stdout
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ============================================================
# SUB-CHECK RUNNERS
# ============================================================

class CheckResult:
    """Result of a single sub-check."""

    def __init__(self, name: str, passed: bool, duration_ms: int = 0,
                 summary: str = "", detail: str = ""):
        self.name = name
        self.passed = passed
        self.duration_ms = duration_ms
        self.summary = summary
        self.detail = detail

    @property
    def status(self) -> str:
        return "PASS" if self.passed else "FAIL"


def run_repo_hygiene() -> CheckResult:
    """Run repo hygiene check (no sensitive files in git)."""
    t0 = time.monotonic()
    try:
        from scripts.check_repo_hygiene import check_hygiene
        violations = check_hygiene()
        passed = len(violations) == 0
        summary = "No sensitive files" if passed else f"{len(violations)} violation(s)"
        detail = "\n".join(violations) if violations else ""
        ms = int((time.monotonic() - t0) * 1000)
        return CheckResult("Repo Hygiene", passed, ms, summary, detail)
    except Exception as e:
        ms = int((time.monotonic() - t0) * 1000)
        return CheckResult("Repo Hygiene", False, ms, f"Error: {e}")


def run_memory_hygiene(collection: str = "codi_memories",
                       limit: int = 500) -> CheckResult:
    """Run memory metadata hygiene check."""
    t0 = time.monotonic()
    try:
        from scripts.memory_hygiene import run_hygiene, format_report
        result = run_hygiene(collection_name=collection, limit=limit)

        has_fails = bool(
            result.empty_data or result.bad_confidence or
            any(i["field"] == "data" for i in result.missing_fields)
        )
        passed = not has_fails

        summary = (f"{result.total_points} points, {result.total_issues} issues"
                   if result.total_points > 0 else "No points scanned")
        detail = format_report(result, collection)
        ms = int((time.monotonic() - t0) * 1000)
        return CheckResult("Memory Hygiene", passed, ms, summary, detail)
    except Exception as e:
        ms = int((time.monotonic() - t0) * 1000)
        return CheckResult("Memory Hygiene", False, ms, f"Error: {e}")


def run_dual_report(hours: int = 48) -> CheckResult:
    """Run dual-mode validation report."""
    t0 = time.monotonic()
    try:
        from scripts.dual_report import main as dual_main

        buf = io.StringIO()
        with redirect_stdout(buf):
            exit_code = dual_main(argv=["--hours", str(hours)])

        output = buf.getvalue()
        passed = exit_code == 0

        # Extract verdict line
        for line in output.splitlines():
            if "VERDICT:" in line:
                summary = line.strip()
                break
        else:
            summary = "PASS" if passed else "FAIL"

        ms = int((time.monotonic() - t0) * 1000)
        return CheckResult("Dual Report", passed, ms, summary, output)
    except Exception as e:
        ms = int((time.monotonic() - t0) * 1000)
        return CheckResult("Dual Report", False, ms, f"Error: {e}")


def run_test_suite() -> CheckResult:
    """Run pytest suite (optional, slow)."""
    t0 = time.monotonic()
    try:
        import subprocess
        result = subprocess.run(
            [sys.executable, "-m", "pytest", "tests/", "-q", "--tb=no", "-x"],
            capture_output=True, text=True, timeout=300,
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        )
        output = result.stdout + result.stderr
        passed = result.returncode == 0

        # Extract summary line (e.g., "1158 passed, 1 failed")
        for line in reversed(output.splitlines()):
            if "passed" in line:
                summary = line.strip()
                break
        else:
            summary = "PASS" if passed else "FAIL"

        ms = int((time.monotonic() - t0) * 1000)
        return CheckResult("Test Suite", passed, ms, summary, output)
    except subprocess.TimeoutExpired:
        ms = int((time.monotonic() - t0) * 1000)
        return CheckResult("Test Suite", False, ms, "Timeout (>300s)")
    except Exception as e:
        ms = int((time.monotonic() - t0) * 1000)
        return CheckResult("Test Suite", False, ms, f"Error: {e}")


# ============================================================
# REPORT FORMAT
# ============================================================

def format_health_report(results: list[CheckResult]) -> str:
    """Format all sub-check results into unified report."""
    lines = []
    lines.append("=" * 60)
    lines.append("CODI-MEMORY HEALTH CHECK")
    lines.append(f"Timestamp: {datetime.now().isoformat(timespec='seconds')}")
    lines.append("=" * 60)

    total_ms = sum(r.duration_ms for r in results)
    passed_count = sum(1 for r in results if r.passed)
    total_count = len(results)

    # Summary table
    lines.append("")
    lines.append(f"  {'Check':<20} {'Status':<8} {'Time':>8}  Summary")
    lines.append(f"  {'-'*20} {'-'*6}  {'-'*8}  {'-'*30}")

    for r in results:
        status_marker = "PASS" if r.passed else "FAIL"
        time_str = f"{r.duration_ms}ms"
        lines.append(f"  {r.name:<20} {status_marker:<8} {time_str:>8}  {r.summary}")

    lines.append("")
    lines.append(f"  Total: {passed_count}/{total_count} passed ({total_ms}ms)")

    # Details for failed checks
    failed = [r for r in results if not r.passed]
    if failed:
        lines.append("")
        lines.append("-" * 60)
        lines.append("FAILURE DETAILS")
        lines.append("-" * 60)
        for r in failed:
            lines.append(f"\n[{r.name}] {r.summary}")
            if r.detail:
                for dl in r.detail.splitlines()[:20]:
                    lines.append(f"  {dl}")

    # Final verdict
    lines.append("")
    lines.append("=" * 60)
    all_passed = all(r.passed for r in results)
    if all_passed:
        lines.append(f"VERDICT: PASS ({passed_count}/{total_count} checks green)")
    else:
        fail_names = ", ".join(r.name for r in failed)
        lines.append(f"VERDICT: FAIL ({fail_names})")
    lines.append("=" * 60)

    return "\n".join(lines)


# ============================================================
# CLI
# ============================================================

def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Unified health check for codi-memory")
    parser.add_argument("--skip-qdrant", action="store_true",
                        help="Skip memory hygiene (needs Qdrant)")
    parser.add_argument("--skip-dual", action="store_true",
                        help="Skip dual report (needs SQLite DB)")
    parser.add_argument("--tests", action="store_true",
                        help="Also run pytest suite (slow)")
    parser.add_argument("--hours", type=int, default=48,
                        help="Dual report lookback window")
    args = parser.parse_args(argv)

    results = []

    # Always run repo hygiene (fast, no dependencies)
    results.append(run_repo_hygiene())

    # Memory hygiene (needs Qdrant)
    if not args.skip_qdrant:
        results.append(run_memory_hygiene())

    # Dual report (needs SQLite)
    if not args.skip_dual:
        results.append(run_dual_report(hours=args.hours))

    # Test suite (optional, slow)
    if args.tests:
        results.append(run_test_suite())

    report = format_health_report(results)
    print(report)

    all_passed = all(r.passed for r in results)
    return 0 if all_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
