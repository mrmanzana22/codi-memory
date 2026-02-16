#!/usr/bin/env python3
"""
PERFORMANCE CONTRACTS TESTS
============================
Verify latency budget definitions, percentile calculations,
violation detection, and event emission.

4 tests:
  1. Percentile calculation from known data
  2. Budget violation detection (p95 / p99 / none)
  3. Contract lookup (known tool -> category, unknown -> default)
  4. Violation event emission via event bus

Run: ./venv/bin/pytest tests/test_performance.py -v
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import sqlite3
import tempfile
import pytest
from unittest.mock import patch

from modules.config import PERF_CONTRACTS, PERF_TOOL_CONTRACT


class TestPercentileCalculation:
    """Verify compute_percentiles returns correct p50/p95/p99 from known data."""

    def test_percentiles_from_known_data(self, tmp_path):
        """Insert 100 known durations, verify p50/p95/p99."""
        db_path = str(tmp_path / "test_metrics.db")

        # Create table and insert 100 rows: duration_ms = 1, 2, 3, ..., 100
        conn = sqlite3.connect(db_path)
        conn.execute("""
            CREATE TABLE tool_calls (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                tool_name TEXT NOT NULL,
                started_at TEXT NOT NULL,
                duration_ms INTEGER NOT NULL,
                success INTEGER NOT NULL,
                error_type TEXT,
                args_size INTEGER DEFAULT 0,
                result_size INTEGER DEFAULT 0,
                session_id TEXT,
                tag TEXT
            )
        """)
        for i in range(1, 101):
            conn.execute(
                "INSERT INTO tool_calls (tool_name, started_at, duration_ms, success) VALUES (?, ?, ?, ?)",
                ("test_tool", "2026-01-01T00:00:00", i, 1),
            )
        conn.commit()
        conn.close()

        # Patch metrics_conn to use our temp DB
        from modules import metrics
        from contextlib import contextmanager

        @contextmanager
        def mock_conn():
            c = sqlite3.connect(db_path, timeout=3.0)
            try:
                yield c
            finally:
                c.close()

        with patch.object(metrics, "metrics_conn", mock_conn):
            result = metrics.compute_percentiles("test_tool", days=365)

        assert result["count"] == 100
        assert result["p50"] == 51     # index 50 of 0-99
        assert result["p95"] == 96     # index 95 of 0-99
        assert result["p99"] == 100    # index 99 of 0-99
        assert result["max"] == 100

    def test_percentiles_empty(self, tmp_path):
        """No data returns zeros."""
        db_path = str(tmp_path / "test_empty.db")
        conn = sqlite3.connect(db_path)
        conn.execute("""
            CREATE TABLE tool_calls (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                tool_name TEXT NOT NULL,
                started_at TEXT NOT NULL,
                duration_ms INTEGER NOT NULL,
                success INTEGER NOT NULL,
                error_type TEXT,
                args_size INTEGER DEFAULT 0,
                result_size INTEGER DEFAULT 0,
                session_id TEXT,
                tag TEXT
            )
        """)
        conn.commit()
        conn.close()

        from modules import metrics
        from contextlib import contextmanager

        @contextmanager
        def mock_conn():
            c = sqlite3.connect(db_path, timeout=3.0)
            try:
                yield c
            finally:
                c.close()

        with patch.object(metrics, "metrics_conn", mock_conn):
            result = metrics.compute_percentiles("nonexistent_tool", days=365)

        assert result["count"] == 0
        assert result["p50"] == 0
        assert result["p95"] == 0
        assert result["p99"] == 0


class TestBudgetViolation:
    """Verify check_budget_violation returns correct violation info."""

    def test_p95_violation(self):
        """Duration > p95 but <= p99 returns p95 violation."""
        from modules.metrics import check_budget_violation
        # recall is macro: p95=2000, p99=5000
        result = check_budget_violation("recall", 3000)
        assert result is not None
        assert result["level"] == "p95"
        assert result["budget_ms"] == 2000
        assert result["actual_ms"] == 3000

    def test_p99_violation(self):
        """Duration > p99 returns p99 violation (takes precedence)."""
        from modules.metrics import check_budget_violation
        # recall is macro: p95=2000, p99=5000
        result = check_budget_violation("recall", 6000)
        assert result is not None
        assert result["level"] == "p99"
        assert result["budget_ms"] == 5000
        assert result["actual_ms"] == 6000

    def test_no_violation(self):
        """Duration within budget returns None."""
        from modules.metrics import check_budget_violation
        result = check_budget_violation("recall", 100)
        assert result is None

    def test_fast_tool_violation(self):
        """Fast tools have tight budgets (p95=200ms)."""
        from modules.metrics import check_budget_violation
        result = check_budget_violation("get_emotional_state", 300)
        assert result is not None
        assert result["level"] == "p95"
        assert result["budget_ms"] == 200


class TestContractLookup:
    """Verify get_tool_contract returns correct contract for known and unknown tools."""

    def test_known_tool_macro(self):
        """recall maps to macro contract."""
        from modules.metrics import get_tool_contract
        contract = get_tool_contract("recall")
        assert contract["p95"] == 2000
        assert contract["p99"] == 5000

    def test_known_tool_fast(self):
        """get_working_memory maps to fast contract."""
        from modules.metrics import get_tool_contract
        contract = get_tool_contract("get_working_memory")
        assert contract["p95"] == 200
        assert contract["p99"] == 500

    def test_unknown_tool_default(self):
        """Unknown tools fall back to default contract."""
        from modules.metrics import get_tool_contract
        contract = get_tool_contract("some_unknown_tool_xyz")
        assert contract["p95"] == 1000
        assert contract["p99"] == 3000

    def test_reverse_lookup_completeness(self):
        """All tools in PERF_CONTRACTS have entries in PERF_TOOL_CONTRACT."""
        for cat, spec in PERF_CONTRACTS.items():
            if cat == "default":
                continue
            for tool in spec["tools"]:
                assert tool in PERF_TOOL_CONTRACT, f"{tool} missing from reverse lookup"
                assert PERF_TOOL_CONTRACT[tool] == cat


class TestViolationEventEmission:
    """Verify that budget violations emit PERF_BUDGET_VIOLATION events."""

    def test_violation_emits_event(self):
        """When a tool exceeds budget, event is emitted with correct data."""
        from modules.events import event_bus, Events

        captured = []

        def handler(event_name, data):
            captured.append(data.copy())

        event_bus.on(Events.PERF_BUDGET_VIOLATION, handler)
        try:
            from modules.metrics import _emit_violation
            violation = {"level": "p95", "budget_ms": 2000, "actual_ms": 3500}
            _emit_violation("recall", 3500, violation)

            assert len(captured) == 1
            assert captured[0]["tool_name"] == "recall"
            assert captured[0]["duration_ms"] == 3500
            assert captured[0]["level"] == "p95"
            assert captured[0]["budget_ms"] == 2000
            assert captured[0]["actual_ms"] == 3500
        finally:
            event_bus.off(Events.PERF_BUDGET_VIOLATION, handler)

    def test_no_event_when_within_budget(self):
        """No event emitted when duration is within budget."""
        from modules.events import event_bus, Events
        from modules.metrics import check_budget_violation

        captured = []

        def handler(event_name, data):
            captured.append(data.copy())

        event_bus.on(Events.PERF_BUDGET_VIOLATION, handler)
        try:
            violation = check_budget_violation("recall", 100)
            assert violation is None
            assert len(captured) == 0
        finally:
            event_bus.off(Events.PERF_BUDGET_VIOLATION, handler)
