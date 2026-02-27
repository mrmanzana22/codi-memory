#!/usr/bin/env python3
"""
T1.3 — interface.py E2E TESTS
================================
End-to-end tests for the 3 macro-tools: recall, remember, context_snapshot.
These exercise the full pipeline from interface -> memory_core/memory_smart
with mocked externals.

Uses patch_externals + _isolate_sqlite (autouse).

Run: ./venv/bin/pytest tests/test_tier1_interface.py -v
     ./venv/bin/pytest tests/test_tier1_interface.py -v -m tier1
"""

import sys
import os
import json

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.interface import recall, remember, context_snapshot
from modules.interface import _auto_importance_from_text, _importance_to_relevance
from modules.interface import _get_remember_mode, _WRITE_MODE_FILE

pytestmark = pytest.mark.tier1


# ============================================================
# HELPERS
# ============================================================

def _parse(result: str) -> dict:
    """Parse JSON response from interface tools."""
    return json.loads(result)


# ============================================================
# RECALL
# ============================================================

class TestRecall:
    """Contract: recall returns JSON with pretty + results."""

    def test_auto_mode_returns_json(self, patch_externals):
        result = recall("test query")
        parsed = _parse(result)
        assert "pretty" in parsed
        assert "results" in parsed
        assert "count" in parsed

    def test_auto_mode_searches_memory(self, patch_externals, monkeypatch):
        monkeypatch.setenv("CODI_WRITE_MODE", "sync")
        # Add a memory first via remember, then recall it
        remember("FastAPI is great for building APIs", topic="aprendizaje")
        result = recall("FastAPI")
        parsed = _parse(result)
        assert parsed["count"] >= 1

    def test_invalid_mode_returns_error(self, patch_externals):
        result = recall("anything", mode="invalid_mode")
        parsed = _parse(result)
        assert "invalido" in parsed["pretty"].lower()
        assert parsed["count"] == 0

    def test_timeline_mode(self, patch_externals):
        result = recall("consciencia", mode="timeline")
        parsed = _parse(result)
        assert "pretty" in parsed
        # Should have called get_project_timeline
        assert any(r["source"] == "get_project_timeline" for r in parsed["results"])

    def test_theme_mode(self, patch_externals):
        result = recall("identidad", mode="theme")
        parsed = _parse(result)
        assert any(r["source"] == "search_by_theme" for r in parsed["results"])

    def test_ownership_mode(self, patch_externals):
        result = recall("source=experienced", mode="ownership")
        parsed = _parse(result)
        assert any(r["source"] == "search_by_ownership" for r in parsed["results"])

    def test_limit_clamped(self, patch_externals):
        # limit > 20 gets clamped
        result = recall("test", limit=999)
        parsed = _parse(result)
        assert "pretty" in parsed  # No crash

    def test_has_trace_id(self, patch_externals):
        result = recall("test query")
        parsed = _parse(result)
        assert "trace_id" in parsed


# ============================================================
# REMEMBER
# ============================================================

class TestRemember:
    """Contract: remember stores in WM + long-term, returns JSON."""

    def test_returns_json_with_all_fields(self, patch_externals, monkeypatch):
        monkeypatch.setenv("CODI_WRITE_MODE", "sync")
        result = remember("important meeting tomorrow", importance="high", topic="trabajo")
        parsed = _parse(result)
        assert "pretty" in parsed
        assert "topic" in parsed
        assert parsed["topic"] == "trabajo"
        assert parsed["importance"] == "high"
        assert "working_memory" in parsed
        assert "long_term" in parsed

    def test_auto_importance_resolved(self, patch_externals, monkeypatch):
        monkeypatch.setenv("CODI_WRITE_MODE", "sync")
        result = remember("some content", importance="auto")
        parsed = _parse(result)
        # auto resolves to actual importance level
        assert parsed["importance"] in ("critical", "high", "medium", "low")

    def test_long_term_stores_memory(self, patch_externals, monkeypatch):
        monkeypatch.setenv("CODI_WRITE_MODE", "sync")
        result = remember("long term test content", importance="high")
        parsed = _parse(result)
        assert parsed["long_term_enabled"] is True
        assert parsed["long_term"] is not None
        # Verify it's valid JSON from add_memory_smart
        lt = json.loads(parsed["long_term"])
        assert "action" in lt

    def test_low_importance_short_skips_lt(self, patch_externals, monkeypatch):
        monkeypatch.setenv("CODI_WRITE_MODE", "sync")
        result = remember("ok", importance="low")
        parsed = _parse(result)
        # Short + low importance -> no long-term
        assert parsed["long_term_enabled"] is False

    def test_long_term_false_skips_lt(self, patch_externals, monkeypatch):
        monkeypatch.setenv("CODI_WRITE_MODE", "sync")
        result = remember("should not go to long-term", long_term=False)
        parsed = _parse(result)
        assert parsed["long_term_enabled"] is False

    def test_async_mode_returns_queued(self, patch_externals, monkeypatch):
        monkeypatch.setenv("CODI_WRITE_MODE", "async")
        result = remember("async content", importance="high")
        parsed = _parse(result)
        lt = json.loads(parsed["long_term"])
        assert lt["action"] == "queued"

    def test_has_trace_id(self, patch_externals, monkeypatch):
        monkeypatch.setenv("CODI_WRITE_MODE", "sync")
        result = remember("trace test")
        parsed = _parse(result)
        assert "trace_id" in parsed


# ============================================================
# CONTEXT_SNAPSHOT
# ============================================================

class TestContextSnapshot:
    """Contract: context_snapshot returns structured state."""

    def test_light_returns_json(self, patch_externals):
        result = context_snapshot(level="light")
        parsed = _parse(result)
        assert "pretty" in parsed
        assert parsed.get("level") == "light"
        assert "working_memory" in parsed

    def test_light_includes_sections(self, patch_externals):
        result = context_snapshot(level="light")
        parsed = _parse(result)
        pretty = parsed["pretty"]
        assert "Working Memory" in pretty

    def test_invalid_level_defaults_light(self, patch_externals):
        result = context_snapshot(level="invalid")
        parsed = _parse(result)
        assert parsed.get("level") == "light"

    def test_has_trace_id(self, patch_externals):
        result = context_snapshot()
        parsed = _parse(result)
        assert "trace_id" in parsed


# ============================================================
# PURE HELPERS
# ============================================================

class TestAutoImportance:
    """Pure function: text -> importance heuristic."""

    def test_short_text_low(self):
        # "hello" is <= 50 chars, no urgency keywords -> "low" (Craik & Lockhart 1972)
        assert _auto_importance_from_text("hello") == "low"

    def test_urgent_keyword_high(self):
        assert _auto_importance_from_text("esto es urgente y hay que resolverlo") == "high"

    def test_long_text_high(self):
        assert _auto_importance_from_text("x" * 201) == "high"


class TestImportanceToRelevance:
    """Pure function: importance -> WM relevance score."""

    def test_critical(self):
        assert _importance_to_relevance("critical") == 0.95

    def test_low(self):
        assert _importance_to_relevance("low") == 0.40

    def test_unknown_defaults(self):
        assert _importance_to_relevance("unknown") == 0.60


# ============================================================
# _get_remember_mode OVERRIDE
# ============================================================

class TestGetRememberMode:
    """Contract: CODI_REMEMBER_MODE overrides CODI_WRITE_MODE for remember()."""

    def test_override_dual_ack(self, monkeypatch):
        monkeypatch.setenv("CODI_REMEMBER_MODE", "dual_ack")
        monkeypatch.setenv("CODI_WRITE_MODE", "sync")
        assert _get_remember_mode() == "dual_ack"

    def test_fallback_to_write_mode(self, monkeypatch):
        monkeypatch.delenv("CODI_REMEMBER_MODE", raising=False)
        monkeypatch.setenv("CODI_WRITE_MODE", "async")
        assert _get_remember_mode() == "async"

    def test_invalid_remember_mode_falls_through(self, monkeypatch):
        monkeypatch.setenv("CODI_REMEMBER_MODE", "bogus")
        monkeypatch.setenv("CODI_WRITE_MODE", "shadow")
        assert _get_remember_mode() == "shadow"

    def test_no_envs_defaults_sync(self, monkeypatch, tmp_path):
        monkeypatch.delenv("CODI_REMEMBER_MODE", raising=False)
        monkeypatch.delenv("CODI_WRITE_MODE", raising=False)
        # Point .write_mode file to nonexistent path so FileNotFoundError => "sync"
        monkeypatch.setattr("modules.interface._WRITE_MODE_FILE", str(tmp_path / "nonexistent"))
        assert _get_remember_mode() == "sync"


# ============================================================
# REMEMBER DUAL_ACK
# ============================================================

class TestRememberDualAck:
    """Contract: remember() with dual_ack returns immediate ACK + background sync."""

    def test_dual_ack_returns_immediate_ack(self, patch_externals, monkeypatch):
        monkeypatch.setenv("CODI_REMEMBER_MODE", "dual_ack")
        result = remember("dual ack test content", importance="high")
        parsed = _parse(result)
        lt = json.loads(parsed["long_term"])
        assert lt["action"] == "queued"

    def test_dual_ack_ack_shape(self, patch_externals, monkeypatch):
        monkeypatch.setenv("CODI_REMEMBER_MODE", "dual_ack")
        result = remember("dual ack shape test", importance="high")
        parsed = _parse(result)
        lt = json.loads(parsed["long_term"])
        assert lt["mode"] == "dual_ack"
        assert "job_id" in lt
        assert lt["kind"] == "remember"
        assert "preview" in lt
        assert "tags" in lt
        assert "check" in lt
        assert "cancel" in lt

    def test_dual_ack_e2e(self, patch_externals, monkeypatch):
        """Full E2E: dual_ack returns ACK + creates dual_compare_log row."""
        import time
        import sqlite3
        from modules.config import FTS_DB_PATH

        monkeypatch.setenv("CODI_REMEMBER_MODE", "dual_ack")
        result = remember("dual ack e2e test", importance="high")
        parsed = _parse(result)
        lt = json.loads(parsed["long_term"])
        job_id = lt["job_id"]

        # ACK returned immediately
        assert lt["action"] == "queued"
        assert lt["mode"] == "dual_ack"

        # Wait briefly for background thread to attempt sync
        time.sleep(0.5)

        # Verify dual_compare_log has the intent row
        conn = sqlite3.connect(FTS_DB_PATH)
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT * FROM dual_compare_log WHERE async_job_id = ?",
            (job_id,)
        ).fetchone()
        conn.close()

        assert row is not None
        assert row["kind"] == "remember"
        assert row["write_mode"] == "dual_ack"
        assert row["compare_status"] in ("pending", "computed")
        # sync_result_json may or may not be filled depending on whether
        # the background sync succeeded (it's fire-and-forget)

    def test_dual_ack_skipped_capacity(self, patch_externals, monkeypatch):
        """When semaphore is full, sync compare is skipped with status."""
        import time
        import threading
        import sqlite3
        import modules.interface as iface
        from modules.config import FTS_DB_PATH

        monkeypatch.setenv("CODI_REMEMBER_MODE", "dual_ack")

        # Replace semaphore with capacity=1 and acquire it to fill the slot
        test_sem = threading.BoundedSemaphore(1)
        test_sem.acquire()  # fills the only slot
        monkeypatch.setattr(iface, "_SYNC_COMPARE_SEMAPHORE", test_sem)

        result = remember("capacity test content", importance="high")
        parsed = _parse(result)
        lt = json.loads(parsed["long_term"])
        job_id = lt["job_id"]

        # Brief sleep to let the bg thread attempt acquire and skip
        time.sleep(0.1)

        conn = sqlite3.connect(FTS_DB_PATH)
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT sync_compare_status, sync_last_error FROM dual_compare_log "
            "WHERE async_job_id = ?", (job_id,)
        ).fetchone()
        conn.close()

        assert row is not None
        assert row["sync_compare_status"] == "skipped_capacity"
        assert "semaphore" in (row["sync_last_error"] or "").lower()

        # Cleanup: release the semaphore
        test_sem.release()

    def test_dual_ack_timeout_records_status(self, patch_externals, monkeypatch):
        """When add_memory_smart takes too long, timeout status is recorded."""
        import time
        import threading
        import modules.interface as iface
        import sqlite3
        from modules.config import FTS_DB_PATH

        monkeypatch.setenv("CODI_REMEMBER_MODE", "dual_ack")
        monkeypatch.setattr(iface, "_SYNC_COMPARE_TIMEOUT", 0.01)  # 10ms
        # Fresh semaphore so previous test bg threads don't interfere
        monkeypatch.setattr(iface, "_SYNC_COMPARE_SEMAPHORE", threading.BoundedSemaphore(3))

        # Make add_memory_smart sleep longer than timeout
        def slow_add_memory(*args, **kwargs):
            time.sleep(0.5)
            return '{"action": "saved_new", "new_id": "slow"}'

        monkeypatch.setattr("modules.interface.add_memory_smart", slow_add_memory)

        result = remember("timeout test content", importance="high")
        parsed = _parse(result)
        lt = json.loads(parsed["long_term"])
        job_id = lt["job_id"]

        # Poll for bg thread to record timeout status (up to 1s)
        status = None
        for _ in range(20):
            time.sleep(0.05)
            conn = sqlite3.connect(FTS_DB_PATH)
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT sync_compare_status, sync_last_error FROM dual_compare_log "
                "WHERE async_job_id = ?", (job_id,)
            ).fetchone()
            conn.close()
            if row and row["sync_compare_status"] != "pending":
                status = row["sync_compare_status"]
                break

        assert row is not None
        assert status == "timeout"
        assert "timeout" in (row["sync_last_error"] or "").lower()
