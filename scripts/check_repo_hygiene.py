#!/usr/bin/env python3
"""
Repo hygiene check: ensures no sensitive files are tracked in git.

Runs as CI step or locally. Exits 0 if clean, 1 if violations found.

Usage:
  python scripts/check_repo_hygiene.py
"""

import fnmatch
import subprocess
import sys

# Patterns that should NEVER appear in the git index
FORBIDDEN_PATTERNS = [
    # Databases
    "*.db",
    "*.sqlite",
    "*.sqlite3",
    "*.db-wal",
    "*.db-shm",
    "*.db-journal",
    # Backups
    "*backup*.json",
    "*backup*.jsonl",
    "*.bak",
    "*.dump",
    "*.sql.gz",
    # Credentials
    ".env",
    ".env.*",
    # Logs
    "*.log",
    # Runtime config
    ".write_mode",
    ".destructive_guard",
]


def get_tracked_files() -> list[str]:
    """Get list of files tracked by git."""
    try:
        result = subprocess.run(
            ["git", "ls-files"],
            capture_output=True, text=True, check=True,
        )
        return [f.strip() for f in result.stdout.splitlines() if f.strip()]
    except (subprocess.CalledProcessError, FileNotFoundError):
        return []


def check_hygiene() -> list[str]:
    """Return list of tracked files that match forbidden patterns."""
    tracked = get_tracked_files()
    if not tracked:
        print("WARNING: No tracked files found (no .git or empty repo).")
        return []

    violations = []
    for filepath in tracked:
        filename = filepath.split("/")[-1]
        for pattern in FORBIDDEN_PATTERNS:
            if fnmatch.fnmatch(filename, pattern) or fnmatch.fnmatch(filepath, pattern):
                violations.append(f"  {filepath}  (matched: {pattern})")
                break

    return violations


def main() -> int:
    print("Repo Hygiene Check")
    print("=" * 40)

    violations = check_hygiene()

    if violations:
        print(f"\nFAILED: {len(violations)} sensitive file(s) tracked in git:\n")
        for v in violations:
            print(v)
        print("\nFix: add to .gitignore and run 'git rm --cached <file>'")
        return 1

    print("\nPASSED: No sensitive files tracked.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
