#!/usr/bin/env python3
"""
WORKER LIVENESS & QUEUE BACKLOG TESTS (PR4 — P0 operational)
==============================================================
Test worker health classification, queue backlog counting,
assessment integration, and boot-path liveness warnings.

10 tests:
  1-3. classify_worker_status: missing, healthy, stale, degraded
  4.   get_queue_backlog: counts by status
  5.   get_worker_health: unified output
  6.   assessment includes worker_health fields
  7.   format_assessment shows worker section
  8-10. get_last_worker_tick_age: no ticks, recent tick, old tick

Run: ./venv/bin/pytest tests/test_worker_liveness.py -v
"""

import sys
import os
import json
import sqlite3
from datetime import datetime, timedelta

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.assessment import (
    classify_worker_status,
    get_queue_backlog,
    get_worker_health,
    get_last_worker_tick_age,
    get_assessment,
    format_assessment,
    WORKER_HEALTHY_THRESHOLD,
    WORKER_STALE_THRESHOLD,
    WORKER_DEGRADED_THRESHOLD,
)
from modules.migrations import apply_migrations

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FTS_MIGRATIONS_DIR = os.path.join(PROJECT_ROOT, "migrations")


# ============================================================
# FIXTURES
# ============================================================

@pytest.fixture
def liveness_db(tmp_path):
    """Isolated DB with all migrations applied."""
    db_path = str(tmp_path / "test_liveness.db")
    apply_migrations(db_path, migrations_dir=FTS_MIGRATIONS_DIR)
    return db_path


def _insert_worker_tick(db_path, minutes_ago=0):
    """Insert a write_worker_tick entry at a given time."""
    ts = (datetime.now() - timedelta(minutes=minutes_ago)).isoformat()
    conn = sqlite3.connect(db_path)
    conn.execute(
        "INSERT INTO tool_calls (tool_name, started_at, duration_ms, success, tag) "
        "VALUES (?, ?, ?, ?, ?)",
        ("write_worker_tick", ts, 100, 1, "tick"),
    )
    conn.commit()
    conn.close()


_queue_job_counter = 0

def _insert_queue_job(db_path, status="queued"):
    """Insert a write_queue job with given status."""
    global _queue_job_counter
    _queue_job_counter += 1
    now = datetime.now().isoformat()
    conn = sqlite3.connect(db_path)
    conn.execute(
        "INSERT INTO write_queue "
        "(job_id, kind, payload_json, status, priority, attempts, max_attempts, "
        " cancel_requested, created_at, updated_at) "
        "VALUES (?, ?, ?, ?, 5, 0, 8, 0, ?, ?)",
        (f"job-{status}-{_queue_job_counter}", "remember", '{"test": true}', status, now, now),
    )
    conn.commit()
    conn.close()


# ============================================================
# classify_worker_status (pure function)
# ============================================================

class TestClassifyWorkerStatus:
    """Pure classification from tick age."""

    def test_missing(self):
        assert classify_worker_status(None) == "missing"

    def test_healthy(self):
        assert classify_worker_status(2.0) == "healthy"

    def test_healthy_at_threshold(self):
        assert classify_worker_status(WORKER_HEALTHY_THRESHOLD) == "healthy"

    def test_stale(self):
        assert classify_worker_status(15.0) == "stale"

    def test_degraded(self):
        assert classify_worker_status(65.0) == "degraded"

    def test_stale_at_boundary(self):
        assert classify_worker_status(WORKER_STALE_THRESHOLD) == "stale"

    def test_degraded_at_boundary(self):
        assert classify_worker_status(WORKER_DEGRADED_THRESHOLD + 1) == "degraded"


# ============================================================
# get_last_worker_tick_age (IO)
# ============================================================

class TestGetLastWorkerTickAge:
    """Tick age from DB."""

    def test_no_ticks_returns_none(self, liveness_db):
        assert get_last_worker_tick_age(liveness_db) is None

    def test_recent_tick(self, liveness_db):
        _insert_worker_tick(liveness_db, minutes_ago=3)
        age = get_last_worker_tick_age(liveness_db)
        assert age is not None
        assert 2 <= age <= 5  # ~3 min, with tolerance

    def test_old_tick(self, liveness_db):
        _insert_worker_tick(liveness_db, minutes_ago=120)
        age = get_last_worker_tick_age(liveness_db)
        assert age is not None
        assert age >= 115  # ~120 min, with tolerance


# ============================================================
# get_queue_backlog (IO)
# ============================================================

class TestGetQueueBacklog:
    """Queue status counts."""

    def test_empty_queue(self, liveness_db):
        result = get_queue_backlog(liveness_db)
        assert result == {}

    def test_mixed_statuses(self, liveness_db):
        _insert_queue_job(liveness_db, "queued")
        _insert_queue_job(liveness_db, "queued")
        _insert_queue_job(liveness_db, "running")
        _insert_queue_job(liveness_db, "done")

        result = get_queue_backlog(liveness_db)
        assert result["queued"] == 2
        assert result["running"] == 1
        assert result["done"] == 1


# ============================================================
# get_worker_health (unified)
# ============================================================

class TestGetWorkerHealth:
    """Unified health check."""

    def test_missing_worker(self, liveness_db):
        result = get_worker_health(liveness_db)
        assert result["status"] == "missing"
        assert result["age_minutes"] is None

    def test_healthy_worker(self, liveness_db):
        _insert_worker_tick(liveness_db, minutes_ago=2)
        result = get_worker_health(liveness_db)
        assert result["status"] == "healthy"
        assert result["age_minutes"] is not None

    def test_includes_queue_backlog(self, liveness_db):
        _insert_worker_tick(liveness_db, minutes_ago=2)
        _insert_queue_job(liveness_db, "queued")
        result = get_worker_health(liveness_db)
        assert "queue_backlog" in result
        assert result["queue_backlog"].get("queued", 0) == 1


# ============================================================
# Assessment integration
# ============================================================

class TestAssessmentIntegration:
    """Assessment report includes worker health."""

    def test_assessment_has_worker_health(self):
        """get_assessment() includes worker_health dict."""
        assessment = get_assessment()
        assert "worker_health" in assessment
        wh = assessment["worker_health"]
        assert "status" in wh
        assert "age_minutes" in wh
        assert "queue_backlog" in wh

    def test_format_assessment_shows_worker(self):
        """format_assessment() includes Worker Health section."""
        assessment = get_assessment()
        text = format_assessment(assessment)
        assert "Worker Health" in text
