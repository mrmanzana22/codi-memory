"""
Tests for modules/health_monitor.py — Operational health monitoring.

Unit tests with in-memory SQLite. No external DB required.
Covers: collect_metrics, evaluate_rules, _rule_wm_stalled,
        _rule_ai_disconnected, _rule_tick_dead, helpers.
"""

import sqlite3
import pytest
from datetime import datetime, timedelta
from unittest.mock import patch

from modules.health_monitor import (
    collect_metrics,
    evaluate_rules,
    _rule_wm_stalled,
    _rule_ai_disconnected,
    _rule_tick_dead,
    _minutes_since,
    _hours_ago_iso,
    _parse_iso,
    _table_exists,
    CYCLE_MINUTES,
    EXPECTED_TICK_INTERVALS,
    run_health_check,
)


# ============================================================
# FIXTURES
# ============================================================

@pytest.fixture
def health_db():
    """In-memory SQLite with all tables health_monitor expects."""
    conn = sqlite3.connect(":memory:")
    conn.execute("""
        CREATE TABLE write_queue_log (
            id INTEGER PRIMARY KEY,
            kind TEXT,
            status TEXT,
            created_at TEXT
        )
    """)
    conn.execute("""
        CREATE TABLE event_counts (
            event TEXT PRIMARY KEY,
            count INTEGER DEFAULT 0,
            last_seen TEXT
        )
    """)
    conn.execute("""
        CREATE TABLE sleep_loop_state (
            key TEXT PRIMARY KEY,
            value TEXT
        )
    """)
    conn.execute("""
        CREATE TABLE generative_model_transitions (
            id INTEGER PRIMARY KEY,
            count INTEGER DEFAULT 0,
            updated_at TEXT
        )
    """)
    conn.commit()
    yield conn
    conn.close()


NOW = "2026-03-12T12:00:00-05:00"
RECENT = "2026-03-12T10:00:00-05:00"
OLD = "2026-03-10T10:00:00-05:00"


# ============================================================
# HELPERS
# ============================================================

class TestHelpers:

    def test_minutes_since_same(self):
        assert _minutes_since(NOW, NOW) == 0.0

    def test_minutes_since_one_hour(self):
        result = _minutes_since(RECENT, NOW)
        assert abs(result - 120.0) < 1.0

    def test_hours_ago_iso(self):
        result = _hours_ago_iso(NOW, 24)
        dt = _parse_iso(result)
        now_dt = _parse_iso(NOW)
        assert abs((now_dt - dt).total_seconds() - 86400) < 1

    def test_parse_iso_naive(self):
        dt = _parse_iso("2026-03-12T12:00:00")
        assert dt.tzinfo is not None

    def test_parse_iso_aware(self):
        dt = _parse_iso("2026-03-12T12:00:00-05:00")
        assert dt.tzinfo is not None

    def test_table_exists_true(self, health_db):
        assert _table_exists(health_db, "write_queue_log") is True

    def test_table_exists_false(self, health_db):
        assert _table_exists(health_db, "nonexistent_table") is False


# ============================================================
# METRIC COLLECTION
# ============================================================

class TestCollectMetrics:

    def test_empty_db_returns_structure(self, health_db):
        metrics = collect_metrics(health_db, NOW)
        assert "wm" in metrics
        assert "sleep_loop" in metrics
        assert "global" in metrics
        assert "active_inference" in metrics

    def test_wm_writes_counted(self, health_db):
        health_db.execute(
            "INSERT INTO write_queue_log (kind, status, created_at) VALUES (?, ?, ?)",
            ("remember", "done", RECENT)
        )
        health_db.commit()
        metrics = collect_metrics(health_db, NOW)
        assert metrics["wm"]["writes_24h"] == 1

    def test_old_writes_excluded(self, health_db):
        health_db.execute(
            "INSERT INTO write_queue_log (kind, status, created_at) VALUES (?, ?, ?)",
            ("remember", "done", OLD)
        )
        health_db.commit()
        metrics = collect_metrics(health_db, NOW)
        assert metrics["wm"]["writes_24h"] == 0

    def test_global_events_counted(self, health_db):
        health_db.execute(
            "INSERT INTO event_counts (event, count, last_seen) VALUES (?, ?, ?)",
            ("search", 42, RECENT)
        )
        health_db.commit()
        metrics = collect_metrics(health_db, NOW)
        assert metrics["global"]["events_24h"] == 42

    def test_sleep_loop_state_collected(self, health_db):
        health_db.execute(
            "INSERT INTO sleep_loop_state (key, value) VALUES (?, ?)",
            ("last_health", RECENT)
        )
        health_db.commit()
        metrics = collect_metrics(health_db, NOW)
        assert metrics["sleep_loop"]["last_health"] == RECENT

    def test_ai_metrics_with_table(self, health_db):
        health_db.execute(
            "INSERT INTO generative_model_transitions (count, updated_at) VALUES (?, ?)",
            (10, RECENT)
        )
        health_db.commit()
        metrics = collect_metrics(health_db, NOW)
        assert metrics["active_inference"]["total_observations"] == 10
        assert metrics["active_inference"]["table_exists"] is True

    def test_ai_metrics_no_table(self):
        conn = sqlite3.connect(":memory:")
        conn.execute("""
            CREATE TABLE write_queue_log (
                id INTEGER PRIMARY KEY, kind TEXT, status TEXT, created_at TEXT
            )
        """)
        conn.execute("""
            CREATE TABLE event_counts (event TEXT PRIMARY KEY, count INTEGER, last_seen TEXT)
        """)
        conn.execute("""
            CREATE TABLE sleep_loop_state (key TEXT PRIMARY KEY, value TEXT)
        """)
        conn.commit()
        metrics = collect_metrics(conn, NOW)
        assert metrics["active_inference"]["table_exists"] is False
        assert metrics["active_inference"]["total_observations"] == 0
        conn.close()


# ============================================================
# RULES
# ============================================================

class TestRuleWmStalled:

    def test_stalled_when_global_active_but_wm_zero(self):
        metrics = {
            "global": {"writes_24h": 25, "events_24h": 100, "latest_write_job_at": RECENT},
            "wm": {"writes_24h": 0, "latest_write_at": None},
            "sleep_loop": {},
            "active_inference": {"total_observations": 5, "table_exists": True},
        }
        alert = _rule_wm_stalled(metrics, NOW)
        assert alert is not None
        assert alert["alert_key"] == "wm_stalled"
        assert alert["severity"] == "critical"

    def test_not_stalled_when_wm_has_writes(self):
        metrics = {
            "global": {"writes_24h": 25, "events_24h": 100, "latest_write_job_at": RECENT},
            "wm": {"writes_24h": 5, "latest_write_at": RECENT},
            "sleep_loop": {},
            "active_inference": {"total_observations": 5, "table_exists": True},
        }
        assert _rule_wm_stalled(metrics, NOW) is None

    def test_not_stalled_when_global_low(self):
        metrics = {
            "global": {"writes_24h": 5, "events_24h": 10, "latest_write_job_at": RECENT},
            "wm": {"writes_24h": 0, "latest_write_at": None},
            "sleep_loop": {},
            "active_inference": {"total_observations": 0, "table_exists": True},
        }
        assert _rule_wm_stalled(metrics, NOW) is None


class TestRuleAiDisconnected:

    def test_disconnected_when_enough_activity_zero_observations(self):
        metrics = {
            "global": {"writes_24h": 30, "events_24h": 50, "latest_write_job_at": RECENT},
            "wm": {"writes_24h": 10, "latest_write_at": RECENT},
            "sleep_loop": {},
            "active_inference": {"total_observations": 0, "table_exists": True},
        }
        alert = _rule_ai_disconnected(metrics, NOW)
        assert alert is not None
        assert alert["alert_key"] == "ai_disconnected"
        assert alert["severity"] == "critical"

    def test_not_disconnected_with_observations(self):
        metrics = {
            "global": {"writes_24h": 30, "events_24h": 50, "latest_write_job_at": RECENT},
            "wm": {"writes_24h": 10, "latest_write_at": RECENT},
            "sleep_loop": {},
            "active_inference": {"total_observations": 15, "table_exists": True},
        }
        assert _rule_ai_disconnected(metrics, NOW) is None

    def test_not_disconnected_when_low_activity(self):
        metrics = {
            "global": {"writes_24h": 5, "events_24h": 10, "latest_write_job_at": RECENT},
            "wm": {"writes_24h": 2, "latest_write_at": RECENT},
            "sleep_loop": {},
            "active_inference": {"total_observations": 0, "table_exists": True},
        }
        assert _rule_ai_disconnected(metrics, NOW) is None


class TestRuleTickDead:

    def test_overdue_tick_fires_alert(self):
        # health tick expected every 1 cycle (30 min), max_age = 60 min
        # Set last_health to 2 hours ago -> overdue
        two_hours_ago = "2026-03-12T10:00:00-05:00"
        metrics = {
            "global": {"writes_24h": 0, "events_24h": 0, "latest_write_job_at": None},
            "wm": {"writes_24h": 0, "latest_write_at": None},
            "sleep_loop": {"last_health": two_hours_ago},
            "active_inference": {"total_observations": 0, "table_exists": True},
        }
        alerts = _rule_tick_dead(metrics, NOW)
        assert len(alerts) >= 1
        assert alerts[0]["alert_key"] == "tick_dead"
        assert "health" in alerts[0]["evidence"]["tick_name"]

    def test_recent_tick_no_alert(self):
        metrics = {
            "global": {"writes_24h": 0, "events_24h": 0, "latest_write_job_at": None},
            "wm": {"writes_24h": 0, "latest_write_at": None},
            "sleep_loop": {"last_health": RECENT},
            "active_inference": {"total_observations": 0, "table_exists": True},
        }
        # RECENT = 2h ago, health max_age = 60min -> might fire
        # Let's use NOW - 10 min instead
        ten_min_ago = "2026-03-12T11:50:00-05:00"
        metrics["sleep_loop"]["last_health"] = ten_min_ago
        alerts = _rule_tick_dead(metrics, NOW)
        health_alerts = [a for a in alerts if a["evidence"]["tick_name"] == "health"]
        assert len(health_alerts) == 0

    def test_missing_tick_no_alert(self):
        metrics = {
            "global": {"writes_24h": 0, "events_24h": 0, "latest_write_job_at": None},
            "wm": {"writes_24h": 0, "latest_write_at": None},
            "sleep_loop": {},  # No last_* keys
            "active_inference": {"total_observations": 0, "table_exists": True},
        }
        alerts = _rule_tick_dead(metrics, NOW)
        assert len(alerts) == 0


# ============================================================
# EVALUATE_RULES (integration of rules)
# ============================================================

class TestEvaluateRules:

    def test_no_alerts_healthy_system(self):
        metrics = {
            "global": {"writes_24h": 5, "events_24h": 10, "latest_write_job_at": RECENT},
            "wm": {"writes_24h": 3, "latest_write_at": RECENT},
            "sleep_loop": {},
            "active_inference": {"total_observations": 5, "table_exists": True},
        }
        alerts = evaluate_rules(metrics, NOW)
        assert len(alerts) == 0

    def test_multiple_alerts_fire(self):
        two_hours_ago = "2026-03-12T10:00:00-05:00"
        metrics = {
            "global": {"writes_24h": 25, "events_24h": 50, "latest_write_job_at": RECENT},
            "wm": {"writes_24h": 0, "latest_write_at": None},
            "sleep_loop": {"last_health": two_hours_ago},
            "active_inference": {"total_observations": 0, "table_exists": True},
        }
        alerts = evaluate_rules(metrics, NOW)
        keys = {a["alert_key"] for a in alerts}
        assert "wm_stalled" in keys
        assert "ai_disconnected" in keys
        assert "tick_dead" in keys


# ============================================================
# RUN_HEALTH_CHECK (top-level entry)
# ============================================================

class TestRunHealthCheck:

    @patch("modules.health_alerts.auto_resolve_cleared", return_value=0)
    @patch("modules.health_alerts.upsert_health_alerts", return_value=[])
    @patch("modules.health_alerts.fetch_open_alerts", return_value=[])
    def test_returns_summary(self, mock_fetch, mock_upsert, mock_resolve, health_db):
        result = run_health_check(health_db, NOW)
        assert "metrics_collected" in result
        assert "alerts_fired" in result
        assert "alert_ids" in result
        assert "open_alerts" in result
        assert "auto_resolved" in result
        assert isinstance(result["alerts_fired"], int)
