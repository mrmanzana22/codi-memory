"""
Static test: ensures no sensitive files are tracked in the git index.
Skips gracefully if .git is not available (e.g., some CI containers).
"""

import os
import subprocess

import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

HAS_GIT = os.path.isdir(os.path.join(PROJECT_ROOT, ".git"))


@pytest.mark.skipif(not HAS_GIT, reason="No .git directory found")
class TestRepoHygiene:

    def test_no_sensitive_files_tracked(self):
        """No database, backup, credential, or log files in git index."""
        result = subprocess.run(
            ["git", "ls-files"],
            capture_output=True, text=True, cwd=PROJECT_ROOT,
        )
        tracked = [f.strip() for f in result.stdout.splitlines() if f.strip()]

        sensitive_extensions = {
            ".db", ".sqlite", ".sqlite3",
            ".db-wal", ".db-shm", ".db-journal",
            ".bak", ".dump", ".log",
        }
        sensitive_names = {
            ".env", ".env.local", ".write_mode", ".destructive_guard",
        }
        sensitive_prefixes = ["memories_backup"]

        violations = []
        for f in tracked:
            name = os.path.basename(f)
            _, ext = os.path.splitext(name)
            if ext in sensitive_extensions:
                violations.append(f"{f} (extension: {ext})")
            elif name in sensitive_names:
                violations.append(f"{f} (filename: {name})")
            elif any(name.startswith(p) for p in sensitive_prefixes):
                violations.append(f"{f} (prefix match)")

        assert not violations, (
            f"Sensitive files tracked in git:\n" +
            "\n".join(f"  - {v}" for v in violations)
        )

    def test_gitignore_covers_backup_patterns(self):
        """Verify .gitignore has entries for timestamped backups."""
        gitignore_path = os.path.join(PROJECT_ROOT, ".gitignore")
        assert os.path.exists(gitignore_path), ".gitignore not found"

        content = open(gitignore_path).read()
        assert "memories_backup_*.json" in content, (
            ".gitignore missing pattern for timestamped backups"
        )

    def test_gitignore_covers_db_artifacts(self):
        """Verify .gitignore covers SQLite WAL/SHM/journal files."""
        gitignore_path = os.path.join(PROJECT_ROOT, ".gitignore")
        content = open(gitignore_path).read()

        for pattern in ["*.db-wal", "*.db-shm", "*.db-journal"]:
            assert pattern in content, f".gitignore missing {pattern}"
