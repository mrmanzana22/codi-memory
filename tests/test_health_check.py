"""Tests for scripts/health_check.py.

Covers:
  - CheckResult data class
  - format_health_report (PASS/FAIL verdict logic)
  - Individual sub-check runners (mocked)
  - main() orchestration with skip flags
"""

import pytest
from unittest.mock import patch, MagicMock
from scripts.health_check import (
    CheckResult,
    format_health_report,
    run_repo_hygiene,
    main,
)


# ============================================================
# CheckResult
# ============================================================

class TestCheckResult:

    def test_pass_status(self):
        r = CheckResult("Test", True, 100, "All good")
        assert r.status == "PASS"

    def test_fail_status(self):
        r = CheckResult("Test", False, 50, "Something wrong")
        assert r.status == "FAIL"


# ============================================================
# format_health_report
# ============================================================

class TestFormatReport:

    def test_all_pass_verdict(self):
        results = [
            CheckResult("Check A", True, 10, "OK"),
            CheckResult("Check B", True, 20, "OK"),
        ]
        report = format_health_report(results)
        assert "VERDICT: PASS" in report
        assert "2/2 checks green" in report

    def test_one_fail_verdict(self):
        results = [
            CheckResult("Repo", True, 10, "Clean"),
            CheckResult("Memory", False, 50, "3 issues"),
        ]
        report = format_health_report(results)
        assert "VERDICT: FAIL" in report
        assert "Memory" in report

    def test_report_includes_timing(self):
        results = [
            CheckResult("Fast Check", True, 5, "Quick"),
        ]
        report = format_health_report(results)
        assert "5ms" in report

    def test_report_includes_summary_table(self):
        results = [
            CheckResult("Repo Hygiene", True, 10, "No sensitive files"),
        ]
        report = format_health_report(results)
        assert "Repo Hygiene" in report
        assert "PASS" in report
        assert "No sensitive files" in report

    def test_failure_details_shown(self):
        results = [
            CheckResult("Bad Check", False, 10, "Failed",
                        detail="Line 1: error\nLine 2: details"),
        ]
        report = format_health_report(results)
        assert "FAILURE DETAILS" in report
        assert "Line 1: error" in report


# ============================================================
# Sub-check runners (mocked)
# ============================================================

class TestRunRepoHygiene:

    def test_pass_when_clean(self):
        with patch("scripts.check_repo_hygiene.check_hygiene", return_value=[]):
            result = run_repo_hygiene()
        assert result.passed
        assert "No sensitive" in result.summary

    def test_fail_when_violations(self):
        violations = ["  .env  (matched: .env)"]
        with patch("scripts.check_repo_hygiene.check_hygiene", return_value=violations):
            result = run_repo_hygiene()
        assert not result.passed
        assert "1 violation" in result.summary


# ============================================================
# main() orchestration
# ============================================================

class TestMainOrchestration:

    def test_skip_all_optional_only_repo(self):
        """With --skip-qdrant --skip-dual, only repo hygiene runs."""
        with patch("scripts.health_check.run_repo_hygiene") as mock_repo:
            mock_repo.return_value = CheckResult("Repo", True, 5, "Clean")

            exit_code = main(["--skip-qdrant", "--skip-dual"])

        assert exit_code == 0
        mock_repo.assert_called_once()

    def test_fail_propagates(self):
        """If repo hygiene fails, overall exits 1."""
        with patch("scripts.health_check.run_repo_hygiene") as mock_repo:
            mock_repo.return_value = CheckResult("Repo", False, 5, "Bad")

            exit_code = main(["--skip-qdrant", "--skip-dual"])

        assert exit_code == 1

    def test_all_checks_called_by_default(self):
        """Without skip flags, repo + memory + dual all run."""
        with patch("scripts.health_check.run_repo_hygiene") as mock_repo, \
             patch("scripts.health_check.run_memory_hygiene") as mock_mem, \
             patch("scripts.health_check.run_dual_report") as mock_dual:

            mock_repo.return_value = CheckResult("Repo", True, 5, "OK")
            mock_mem.return_value = CheckResult("Memory", True, 50, "OK")
            mock_dual.return_value = CheckResult("Dual", True, 100, "OK")

            exit_code = main([])

        assert exit_code == 0
        mock_repo.assert_called_once()
        mock_mem.assert_called_once()
        mock_dual.assert_called_once()

    def test_tests_flag_runs_suite(self):
        """With --tests, pytest suite also runs."""
        with patch("scripts.health_check.run_repo_hygiene") as mock_repo, \
             patch("scripts.health_check.run_memory_hygiene") as mock_mem, \
             patch("scripts.health_check.run_dual_report") as mock_dual, \
             patch("scripts.health_check.run_test_suite") as mock_tests:

            mock_repo.return_value = CheckResult("Repo", True, 5, "OK")
            mock_mem.return_value = CheckResult("Memory", True, 50, "OK")
            mock_dual.return_value = CheckResult("Dual", True, 100, "OK")
            mock_tests.return_value = CheckResult("Tests", True, 5000, "1159 passed")

            exit_code = main(["--tests"])

        assert exit_code == 0
        mock_tests.assert_called_once()

    def test_hours_passed_to_dual(self):
        """--hours flag is forwarded to dual report."""
        with patch("scripts.health_check.run_repo_hygiene") as mock_repo, \
             patch("scripts.health_check.run_memory_hygiene") as mock_mem, \
             patch("scripts.health_check.run_dual_report") as mock_dual:

            mock_repo.return_value = CheckResult("Repo", True, 5, "OK")
            mock_mem.return_value = CheckResult("Memory", True, 50, "OK")
            mock_dual.return_value = CheckResult("Dual", True, 100, "OK")

            main(["--hours", "24"])

        mock_dual.assert_called_once_with(hours=24)
