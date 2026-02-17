#!/usr/bin/env python3
"""
Dual Canary: daily probe + automated gate check.

Inserts 2 canary remember() calls (stable + timestamped) to ensure
the dual_ack pipeline stays exercised, then runs dual_report.py --hours 24.

Exit codes:
  0 = PASS (or no data)
  2 = FAIL (gate failed)
  1 = unexpected error
"""
import os
import sys
import subprocess
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

# Point to prod DBs
os.environ.setdefault("FTS_DB_PATH", str(REPO_ROOT / "memories_fts.db"))
os.environ.setdefault("PROSPECTIVE_DB_PATH", str(REPO_ROOT / "prospective.db"))


def main() -> int:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    from modules.interface import remember

    # 1) Stable canary (constant signal)
    print(f"[canary] Sending stable probe at {ts}")
    remember(
        content="dual_canary: stable signal (do not remove)",
        topic="ops",
        importance="low",
        long_term=True,
    )

    # 2) Timestamped canary (avoids full dedupe)
    print(f"[canary] Sending timestamped probe")
    remember(
        content=f"dual_canary: timestamp={ts}",
        topic="ops",
        importance="low",
        long_term=True,
    )

    # 3) Run dual_report --hours 24
    print("[canary] Running dual_report.py --hours 24")
    report = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "dual_report.py"), "--hours", "24"],
        capture_output=True,
        text=True,
    )
    print(report.stdout)
    if report.stderr:
        # Filter out deprecation warnings
        for line in report.stderr.splitlines():
            if "DeprecationWarning" not in line and "deprecated" not in line.lower():
                print(line, file=sys.stderr)

    if report.returncode != 0:
        print("[canary] GATE FAILED — check logs")
        return 2

    print("[canary] GATE PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
