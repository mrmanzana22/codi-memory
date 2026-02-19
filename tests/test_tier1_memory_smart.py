#!/usr/bin/env python3
"""
T1.2 — memory_smart.py CONTRACT TESTS
=======================================
Core invariants: FTS indexing, dedup detection, contradiction check,
BM25 scoring, retry queue mechanics, search_fts safety.

Uses patch_externals + _isolate_sqlite (autouse).

Run: ./venv/bin/pytest tests/test_tier1_memory_smart.py -v
     ./venv/bin/pytest tests/test_tier1_memory_smart.py -v -m tier1
"""

import sys
import os
import json
import sqlite3

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import modules.memory_smart as ms
from modules.events import event_bus, Events

pytestmark = pytest.mark.tier1


# ============================================================
# HELPERS
# ============================================================

def _text_count(db_path: str) -> int:
    conn = sqlite3.connect(db_path)
    try:
        return conn.execute("SELECT COUNT(*) FROM memories_text").fetchone()[0]
    finally:
        conn.close()


def _retry_count(db_path: str, status: str = "pending") -> int:
    conn = sqlite3.connect(db_path)
    try:
        return conn.execute(
            "SELECT COUNT(*) FROM fts_retry_queue WHERE status = ?", (status,)
        ).fetchone()[0]
    finally:
        conn.close()


# ============================================================
# FTS INDEXING
# ============================================================

class TestFtsIndexing:
    """Contract: index/delete/search on FTS5 backing table."""

    def test_index_creates_row(self, patch_externals):
        db = patch_externals["db_path"]
        ms.index_memory_fts("mem-1", "hello world", "general", "experienced", "medium")
        assert _text_count(db) == 1

    def test_index_idempotent(self, patch_externals):
        db = patch_externals["db_path"]
        ms.index_memory_fts("mem-1", "hello", "general", "experienced", "medium")
        ms.index_memory_fts("mem-1", "hello updated", "general", "experienced", "high")
        assert _text_count(db) == 1  # INSERT OR REPLACE

    def test_delete_removes_row(self, patch_externals):
        db = patch_externals["db_path"]
        ms.index_memory_fts("mem-1", "to delete", "general", "experienced", "medium")
        assert _text_count(db) == 1
        ms.delete_memory_fts("mem-1")
        assert _text_count(db) == 0

    def test_delete_nonexistent_ok(self, patch_externals):
        result = ms.delete_memory_fts("nonexistent-id")
        assert result is True  # No error

    def test_search_finds_indexed(self, patch_externals):
        ms.index_memory_fts("mem-1", "FastAPI backend development", "aprendizaje", "learned", "high")
        results = ms.search_fts("FastAPI")
        assert len(results) >= 1
        assert results[0]["memory_id"] == "mem-1"

    def test_search_returns_empty_on_miss(self, patch_externals):
        ms.index_memory_fts("mem-1", "hello world", "general", "experienced", "medium")
        results = ms.search_fts("xyznonexistent12345")
        assert results == []

    def test_search_sanitizes_injection(self, patch_externals):
        """FTS5 injection attempts are neutralized by fts_safety."""
        ms.index_memory_fts("mem-1", "secret data", "general", "experienced", "medium")
        # Wildcard injection attempt
        results = ms.search_fts("* OR 1=1")
        # Should not dump all docs -- sanitizer strips operators
        assert len(results) <= 1

    def test_search_empty_query_returns_empty(self, patch_externals):
        ms.index_memory_fts("mem-1", "hello", "general", "experienced", "medium")
        results = ms.search_fts("")
        assert results == []


# ============================================================
# BM25 SCORING (pure)
# ============================================================

class TestBm25RankToScore:
    """Pure function: FTS5 rank -> 0-1 score."""

    def test_zero_rank(self):
        assert ms.bm25_rank_to_score(0) == 0.0

    def test_negative_rank_positive_score(self):
        score = ms.bm25_rank_to_score(-10)
        assert 0.8 < score < 1.0

    def test_small_negative(self):
        score = ms.bm25_rank_to_score(-2)
        assert 0.5 < score < 0.8

    def test_none_rank(self):
        assert ms.bm25_rank_to_score(None) == 0.0

    def test_monotonic(self):
        """More negative rank = higher score."""
        s1 = ms.bm25_rank_to_score(-2)
        s2 = ms.bm25_rank_to_score(-10)
        assert s2 > s1


# ============================================================
# FTS RETRY QUEUE
# ============================================================

class TestFtsRetryQueue:
    """Contract: queue_fts_op, process, stats."""

    def test_queue_upsert(self, patch_externals):
        db = patch_externals["db_path"]
        result = ms.queue_fts_op("mem-q1", "upsert",
                                 payload={"content": "queued content", "category": "general"})
        assert result["ok"] is True
        assert _retry_count(db) == 1

    def test_queue_invalid_op_rejected(self, patch_externals):
        result = ms.queue_fts_op("mem-q1", "drop_table")
        assert result["ok"] is False

    def test_queue_idempotent(self, patch_externals):
        db = patch_externals["db_path"]
        ms.queue_fts_op("mem-q1", "upsert", payload={"content": "v1"})
        ms.queue_fts_op("mem-q1", "upsert", payload={"content": "v2"})
        assert _retry_count(db) == 1  # ON CONFLICT DO UPDATE

    def test_process_executes_queued(self, patch_externals):
        db = patch_externals["db_path"]
        ms.queue_fts_op("mem-q1", "upsert",
                         payload={"content": "process me", "category": "general"})
        result = ms.process_fts_queue(limit=10)
        assert result["ok"] is True
        assert result["succeeded"] == 1
        assert _text_count(db) == 1

    def test_stats_reflects_queue(self, patch_externals):
        ms.queue_fts_op("mem-q1", "upsert", payload={"content": "a"})
        ms.queue_fts_op("mem-q2", "delete")
        stats = ms.fts_queue_stats()
        assert stats["ok"] is True
        assert stats["pending"] == 2


# ============================================================
# ADD_MEMORY_SMART: Deduplication
# ============================================================

class TestAddMemorySmart:
    """Contract: dedupe, relate, new paths."""

    def test_first_add_saves_new(self, patch_externals):
        result = ms.add_memory_smart("brand new content", category="general")
        parsed = json.loads(result)
        assert parsed["action"] == "saved_new"
        assert "new_id" in parsed

    def test_duplicate_skipped(self, patch_externals):
        """Adding identical content twice triggers dedup."""
        mem0 = patch_externals["mem0"]
        # First add
        ms.add_memory_smart("exact same content", category="general")
        # FakeMem0 uses substring match with score=0.8
        # Default dedup_threshold=0.90, so 0.8 < 0.90 -> won't dedup
        # Lower the threshold to trigger dedup
        result = ms.add_memory_smart("exact same content", category="general",
                                      dedup_threshold=0.70)
        parsed = json.loads(result)
        assert parsed["action"] == "skipped_duplicate"

    def test_similar_saved_with_relation(self, patch_externals):
        """Content above relate_threshold but below dedup creates relation."""
        ms.add_memory_smart("original memory about cats", category="general")
        # FakeMem0 score=0.8, set thresholds: dedup=0.90, relate=0.70
        result = ms.add_memory_smart("another memory about cats", category="general",
                                      dedup_threshold=0.90, relate_threshold=0.70)
        parsed = json.loads(result)
        assert parsed["action"] == "saved_with_relation"
        assert "related_to" in parsed

    def test_add_indexes_fts(self, patch_externals):
        db = patch_externals["db_path"]
        ms.add_memory_smart("fts indexed content", category="aprendizaje")
        assert _text_count(db) >= 1

    def test_add_error_returns_json(self, patch_externals):
        """When mem0 breaks, add_memory_smart returns error JSON."""
        def broken(*a, **kw):
            raise RuntimeError("boom")
        patch_externals["mem0"].search = broken
        result = ms.add_memory_smart("should fail", category="general")
        parsed = json.loads(result)
        assert parsed["action"] == "error"

    def test_add_emits_stored_event(self, patch_externals, clean_event_bus):
        ms.add_memory_smart("event test", category="general")
        stored = [e for e in event_bus._history if e["event"] == Events.MEMORY_STORED]
        assert len(stored) >= 1


# ============================================================
# CONTRADICTION COUNTER (session-level)
# ============================================================

class TestContradictionCounter:
    """Counter mechanics (inline check tested in dedicated test file)."""

    def setup_method(self):
        ms._contradiction_session_count = 0
        ms._contradiction_cooldown.clear()

    def test_rate_limit_kicks_in(self):
        """After MAX_PER_SESSION, _inline_contradiction_check skips."""
        from modules.config import CONTRADICTION_MAX_PER_SESSION
        ms._contradiction_session_count = CONTRADICTION_MAX_PER_SESSION
        result = ms._inline_contradiction_check("new", [])
        assert result["detected"] is False
        assert result.get("reason") == "session_rate_limit"

    def test_counter_resets(self):
        ms._contradiction_session_count = 5
        ms._contradiction_cooldown["x"] = "2026-01-01"
        ms.reset_contradiction_counter()
        assert ms._contradiction_session_count == 0
        assert len(ms._contradiction_cooldown) == 0


# ============================================================
# Dynamic Dedup Threshold (amygdala-modulated encoding)
# ============================================================

class TestDynamicDedupThreshold:
    """_compute_dedup_threshold adjusts based on PAD arousal + importance."""

    def test_default_medium_no_arousal(self):
        """Base case: medium importance, no arousal = 0.90."""
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr("modules.config._emotional_state",
                        {"current": {"arousal": 0.0}})
            assert ms._compute_dedup_threshold("medium") == 0.90

    def test_critical_importance_raises_threshold(self):
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr("modules.config._emotional_state",
                        {"current": {"arousal": 0.0}})
            result = ms._compute_dedup_threshold("critical")
            assert result == 0.95  # 0.90 + 0.05

    def test_high_importance_raises_threshold(self):
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr("modules.config._emotional_state",
                        {"current": {"arousal": 0.0}})
            result = ms._compute_dedup_threshold("high")
            assert result == 0.93  # 0.90 + 0.03

    def test_high_arousal_raises_threshold(self):
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr("modules.config._emotional_state",
                        {"current": {"arousal": 0.7}})
            result = ms._compute_dedup_threshold("medium")
            # 0.90 + min(0.05, (0.7-0.3)*0.07) = 0.90 + 0.028 = 0.928
            assert result == 0.928

    def test_max_arousal_plus_critical_clamped(self):
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr("modules.config._emotional_state",
                        {"current": {"arousal": 1.0}})
            result = ms._compute_dedup_threshold("critical")
            # 0.90 + 0.05(arousal) + 0.05(critical) = 1.0 -> clamped to 0.97
            assert result == 0.97

    def test_negative_arousal_uses_abs(self):
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr("modules.config._emotional_state",
                        {"current": {"arousal": -0.6}})
            result = ms._compute_dedup_threshold("medium")
            # abs(-0.6) = 0.6; 0.90 + min(0.05, (0.6-0.3)*0.07) = 0.90 + 0.021
            assert result == 0.921

    def test_low_arousal_no_boost(self):
        """Arousal <= 0.3 gives no boost."""
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr("modules.config._emotional_state",
                        {"current": {"arousal": 0.2}})
            result = ms._compute_dedup_threshold("medium")
            assert result == 0.90

    def test_low_importance_no_boost(self):
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr("modules.config._emotional_state",
                        {"current": {"arousal": 0.0}})
            result = ms._compute_dedup_threshold("low")
            assert result == 0.90

    def test_auto_threshold_used_when_zero(self, patch_externals):
        """add_memory_smart with dedup_threshold=0 uses dynamic threshold."""
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr("modules.config._emotional_state",
                        {"current": {"arousal": 0.0}})
            result = ms.add_memory_smart("new content", importance="critical")
            parsed = json.loads(result)
            # With critical importance, threshold=0.95
            # FakeMem0 score=0.8 < 0.95, so should save
            assert parsed["action"] in ("saved_new", "saved_with_relation")
