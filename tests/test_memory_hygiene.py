"""Tests for scripts/memory_hygiene.py.

Covers:
  - check_point (unit-level field validation)
  - run_hygiene (with mock Qdrant)
  - format_report + verdict logic
"""

import pytest
from unittest.mock import MagicMock
from scripts.memory_hygiene import (
    check_point,
    run_hygiene,
    format_report,
    HygieneResult,
    REQUIRED_FIELDS,
    VALID_SOURCES,
    VALID_IMPORTANCE,
)


def _make_clean_payload(**overrides):
    """Build a payload that passes all checks."""
    base = {
        "data": "Some memory content",
        "ownership_source": "experienced",
        "narrative_importance": "medium",
        "created_at": "2026-02-15T10:00:00",
        "ownership_confidence": 0.9,
        "_v": 4.0,
    }
    base.update(overrides)
    return base


def _mock_point(pid, payload):
    """Build a mock Qdrant point."""
    p = MagicMock()
    p.id = pid
    p.payload = payload
    return p


# ============================================================
# check_point UNIT TESTS
# ============================================================

class TestCheckPoint:
    """Unit tests for check_point()."""

    def test_clean_payload_no_issues(self):
        payload = _make_clean_payload()
        issues = check_point("m1", payload, {"m1"})
        assert issues == [], f"Clean payload should have no issues, got {issues}"

    def test_missing_data_field(self):
        payload = _make_clean_payload()
        del payload["data"]
        issues = check_point("m1", payload, {"m1"})
        assert any(i["check"] == "missing_field" and i["field"] == "data" for i in issues)

    def test_empty_data(self):
        payload = _make_clean_payload(data="")
        issues = check_point("m1", payload, {"m1"})
        assert any(i["check"] == "empty_data" for i in issues)

    def test_whitespace_only_data(self):
        payload = _make_clean_payload(data="   ")
        issues = check_point("m1", payload, {"m1"})
        assert any(i["check"] == "empty_data" for i in issues)

    def test_invalid_source(self):
        payload = _make_clean_payload(ownership_source="made_up")
        issues = check_point("m1", payload, {"m1"})
        assert any(i["check"] == "invalid_source" for i in issues)

    def test_all_valid_sources_pass(self):
        for source in VALID_SOURCES:
            payload = _make_clean_payload(ownership_source=source)
            issues = check_point("m1", payload, {"m1"})
            source_issues = [i for i in issues if i["check"] == "invalid_source"]
            assert source_issues == [], f"Valid source '{source}' flagged as invalid"

    def test_invalid_importance(self):
        payload = _make_clean_payload(narrative_importance="super_duper")
        issues = check_point("m1", payload, {"m1"})
        assert any(i["check"] == "invalid_importance" for i in issues)

    def test_all_valid_importance_pass(self):
        for imp in VALID_IMPORTANCE:
            payload = _make_clean_payload(narrative_importance=imp)
            issues = check_point("m1", payload, {"m1"})
            imp_issues = [i for i in issues if i["check"] == "invalid_importance"]
            assert imp_issues == [], f"Valid importance '{imp}' flagged as invalid"

    def test_confidence_out_of_range_high(self):
        payload = _make_clean_payload(ownership_confidence=1.5)
        issues = check_point("m1", payload, {"m1"})
        assert any(i["check"] == "bad_confidence" for i in issues)

    def test_confidence_out_of_range_negative(self):
        payload = _make_clean_payload(ownership_confidence=-0.1)
        issues = check_point("m1", payload, {"m1"})
        assert any(i["check"] == "bad_confidence" for i in issues)

    def test_confidence_at_boundaries_ok(self):
        for val in [0, 0.5, 1.0]:
            payload = _make_clean_payload(ownership_confidence=val)
            issues = check_point("m1", payload, {"m1"})
            conf_issues = [i for i in issues if i["check"] == "bad_confidence"]
            assert conf_issues == [], f"Confidence {val} should be valid"

    def test_old_schema_version(self):
        payload = _make_clean_payload(_v=3.0)
        issues = check_point("m1", payload, {"m1"})
        assert any(i["check"] == "old_schema" for i in issues)

    def test_current_schema_ok(self):
        payload = _make_clean_payload(_v=4.0)
        issues = check_point("m1", payload, {"m1"})
        schema_issues = [i for i in issues if i["check"] == "old_schema"]
        assert schema_issues == []

    def test_orphan_relation(self):
        payload = _make_clean_payload(related_to="nonexistent-id")
        all_ids = {"m1"}
        issues = check_point("m1", payload, all_ids)
        assert any(i["check"] == "orphan_relation" for i in issues)

    def test_valid_relation(self):
        payload = _make_clean_payload(related_to="m2")
        all_ids = {"m1", "m2"}
        issues = check_point("m1", payload, all_ids)
        orphan_issues = [i for i in issues if i["check"] == "orphan_relation"]
        assert orphan_issues == []

    def test_bad_created_at(self):
        payload = _make_clean_payload(created_at="not-a-date-T-here")
        issues = check_point("m1", payload, {"m1"})
        assert any(i["check"] == "bad_created_at" for i in issues)

    def test_missing_optional_fields_ok(self):
        """Payload with only required fields should not fail."""
        payload = {
            "data": "Minimal memory",
            "ownership_source": "experienced",
            "narrative_importance": "medium",
            "created_at": "2026-01-01T00:00:00",
        }
        issues = check_point("m1", payload, {"m1"})
        assert issues == []


# ============================================================
# run_hygiene INTEGRATION TESTS (mocked Qdrant)
# ============================================================

class TestRunHygiene:
    """Integration tests with mocked Qdrant client."""

    def test_all_clean_points(self):
        mock_qdrant = MagicMock()
        points = [
            _mock_point("m1", _make_clean_payload()),
            _mock_point("m2", _make_clean_payload(data="Another memory")),
        ]
        mock_qdrant.scroll.return_value = (points, None)

        result = run_hygiene(qdrant_client=mock_qdrant)
        assert result.total_points == 2
        assert result.total_issues == 0

    def test_mixed_issues(self):
        mock_qdrant = MagicMock()
        points = [
            _mock_point("clean", _make_clean_payload()),
            _mock_point("bad-source", _make_clean_payload(ownership_source="fake")),
            _mock_point("empty", _make_clean_payload(data="")),
        ]
        mock_qdrant.scroll.return_value = (points, None)

        result = run_hygiene(qdrant_client=mock_qdrant)
        assert result.total_points == 3
        assert result.total_issues >= 2  # at least bad_source + empty_data

    def test_empty_collection(self):
        mock_qdrant = MagicMock()
        mock_qdrant.scroll.return_value = ([], None)

        result = run_hygiene(qdrant_client=mock_qdrant)
        assert result.total_points == 0
        assert result.total_issues == 0


# ============================================================
# VERDICT TESTS
# ============================================================

class TestVerdict:
    """Tests for PASS/FAIL verdict logic in format_report."""

    def test_pass_verdict_clean(self):
        result = HygieneResult()
        result.total_points = 10
        report = format_report(result, "test_collection")
        assert "VERDICT: PASS" in report

    def test_fail_on_empty_data(self):
        result = HygieneResult()
        result.total_points = 10
        result.empty_data.append({"id": "m1", "check": "empty_data"})
        report = format_report(result, "test_collection")
        assert "VERDICT: FAIL" in report
        assert "empty_data" in report

    def test_fail_on_bad_confidence(self):
        result = HygieneResult()
        result.total_points = 10
        result.bad_confidence.append({"id": "m1", "check": "bad_confidence", "value": 1.5})
        report = format_report(result, "test_collection")
        assert "VERDICT: FAIL" in report

    def test_warning_on_old_schema(self):
        result = HygieneResult()
        result.total_points = 10
        result.old_schema.append({"id": "m1", "check": "old_schema", "value": 3.0})
        report = format_report(result, "test_collection")
        # Old schema is warning only, not FAIL
        assert "VERDICT: PASS" in report
        assert "WARNINGS:" in report
        assert "old_schema" in report

    def test_warning_on_orphan_relations(self):
        result = HygieneResult()
        result.total_points = 10
        result.orphan_relations.append({
            "id": "m1", "check": "orphan_relation", "related_to": "gone"
        })
        report = format_report(result, "test_collection")
        assert "VERDICT: PASS" in report
        assert "orphan_relations" in report

    def test_report_shows_health_percentage(self):
        result = HygieneResult()
        result.total_points = 100
        report = format_report(result, "test")
        assert "Health: 100.0%" in report
