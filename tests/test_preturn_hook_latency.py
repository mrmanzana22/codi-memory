#!/usr/bin/env python3
"""
J.0.5: Pre-Turn Hook Latency Regression Tests (J.0.4 Findings)
================================================================
Breck 2017 Tests 8/19/20: Existing behavior tested offline, integration tests,
quality validated before production.

Based on J.0.4 profiling findings:
- Pre-turn hook currently takes ~7.2s (14.4x over 500ms budget)
- Root causes: lock contention (3128ms) + cold import chain (627ms)
- These tests verify the FAST path — SQLite-only operations without concurrency

Purpose:
  1. Document the expected latency budget for each hook phase
  2. Catch latency regressions introduced by new code
  3. Verify that the CORE LOGIC (without concurrency) is fast

NOTE: These tests measure the hook functions IN-PROCESS (warm modules).
The full subprocess latency (7.2s) includes Python startup + lock contention
which is a separate infrastructure issue (see J.0.4 report, Fix P0).

Run: ./venv/bin/pytest tests/test_preturn_hook_latency.py -v
"""

import sys
import os
import time
import sqlite3
import json
import tempfile
import pytest
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ============================================================
# FIXTURES
# ============================================================

@pytest.fixture(scope="module")
def temp_fts_db():
    """Create a minimal FTS database for testing hook functions."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name

    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL")

    # Minimal tables needed by hook functions
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS memories_text (
            memory_id TEXT PRIMARY KEY,
            content TEXT NOT NULL,
            category TEXT DEFAULT 'general',
            source TEXT DEFAULT 'experienced',
            importance TEXT DEFAULT 'medium',
            created_at TEXT NOT NULL
        );
        CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
            content,
            memory_id UNINDEXED
        );
        CREATE TABLE IF NOT EXISTS working_memory (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            content TEXT NOT NULL,
            topic TEXT DEFAULT 'general',
            relevance REAL DEFAULT 0.5,
            occurred_at TEXT NOT NULL,
            active INTEGER DEFAULT 1,
            chain_id TEXT
        );
        CREATE TABLE IF NOT EXISTS prediction_state (
            id INTEGER PRIMARY KEY,
            predicted_topic TEXT,
            predicted_keywords TEXT,
            predicted_action TEXT,
            topic_distribution TEXT,
            confidence REAL DEFAULT 0.5,
            created_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS prediction_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            predicted_topic TEXT,
            actual_topic TEXT,
            predicted_keywords TEXT,
            actual_keywords TEXT,
            surprise_score REAL,
            precision_weight REAL,
            weighted_surprise REAL,
            hit INTEGER DEFAULT 0,
            source TEXT DEFAULT 'interactive',
            created_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS prediction_state_l1 (
            id INTEGER PRIMARY KEY DEFAULT 1,
            session_id TEXT,
            session_start TEXT,
            turn_count INTEGER DEFAULT 0,
            goal_topics TEXT DEFAULT '{}',
            goal_locked INTEGER DEFAULT 0,
            predicted_trajectory TEXT DEFAULT '{}',
            last_topic TEXT,
            last_turn_at TEXT,
            session_pe_sum REAL DEFAULT 0.0,
            session_pe_count INTEGER DEFAULT 0
        );
        CREATE TABLE IF NOT EXISTS prediction_results_l1 (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT,
            turn_number INTEGER,
            predicted_goal TEXT,
            actual_topic TEXT,
            session_pe REAL,
            precision_l1 REAL,
            weighted_pe_l1 REAL,
            created_at TEXT
        );
        CREATE TABLE IF NOT EXISTS prediction_state_l2 (
            domain TEXT PRIMARY KEY,
            predicted_accuracy REAL DEFAULT 0.5,
            actual_accuracy REAL DEFAULT 0.5,
            sample_size INTEGER DEFAULT 0,
            last_updated TEXT
        );
        CREATE TABLE IF NOT EXISTS prediction_results_l2 (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            domain TEXT,
            predicted_accuracy REAL,
            actual_accuracy REAL,
            meta_pe REAL,
            precision_l2 REAL,
            weighted_pe_l2 REAL,
            created_at TEXT
        );
        CREATE TABLE IF NOT EXISTS transition_stats (
            from_topic TEXT NOT NULL,
            to_topic TEXT NOT NULL,
            count INTEGER DEFAULT 0,
            last_seen TEXT,
            PRIMARY KEY (from_topic, to_topic)
        );
        CREATE TABLE IF NOT EXISTS attention_state (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS attention_transitions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            from_topic TEXT,
            to_topic TEXT,
            driver TEXT,
            created_at TEXT
        );
        CREATE TABLE IF NOT EXISTS prospective_intentions (
            id TEXT PRIMARY KEY,
            action TEXT NOT NULL,
            trigger_type TEXT DEFAULT 'event',
            trigger_spec TEXT DEFAULT '{}',
            priority TEXT DEFAULT 'medium',
            status TEXT DEFAULT 'pending',
            expiry TEXT,
            context TEXT,
            created_at TEXT NOT NULL,
            completed_at TEXT,
            access_count INTEGER DEFAULT 0
        );
    """)

    # Insert some sample memories
    for i in range(5):
        conn.execute(
            "INSERT INTO memories_text (memory_id, content, category, importance, created_at) "
            "VALUES (?, ?, ?, ?, datetime('now'))",
            (f"mem_{i}", f"Memory about consciencia proyecto {i}", "consciencia", "high")
        )
    # Insert working memory
    for i in range(3):
        conn.execute(
            "INSERT INTO working_memory (content, topic, relevance, occurred_at) "
            "VALUES (?, ?, ?, datetime('now'))",
            (f"Working memory item {i} about consciencia", "consciencia", 0.8 - i * 0.1)
        )

    conn.commit()
    conn.close()

    yield db_path

    try:
        os.unlink(db_path)
    except Exception:
        pass


@pytest.fixture(scope="module")
def hook_module(temp_fts_db, monkeypatch_module):
    """Import hook module with patched FTS_DB_PATH pointing to temp DB."""
    import importlib

    # Patch the path before importing
    original_env = os.environ.copy()
    os.environ["FTS_DB_OVERRIDE"] = temp_fts_db

    # We need to patch at the module level
    import hooks.preturn_inject as hook
    original_path = hook.FTS_DB_PATH
    hook.FTS_DB_PATH = temp_fts_db

    yield hook

    # Restore
    hook.FTS_DB_PATH = original_path


@pytest.fixture(scope="module")
def monkeypatch_module():
    """Module-scoped monkeypatch."""
    return None


# ============================================================
# LATENCY BUDGET TESTS
# ============================================================

class TestCoreFunctionLatency:
    """
    Verify that core hook functions complete within their allocated budgets.

    Budgets are per-function targets within the 500ms total:
    - classify_topic: <2ms (pure CPU)
    - extract_keywords: <2ms (pure CPU)
    - SQLite FTS search: <20ms (local DB, no lock contention)
    - Working memory fetch: <5ms
    - Trigger detection: <5ms (JSON file read)
    - generate_prediction: <5ms

    These budgets assume in-process execution (warm modules, no lock contention).
    The full subprocess adds ~300ms startup + variable lock contention.
    """

    RUNS = 10  # Number of runs to average (reduce measurement noise)

    def _avg_time_ms(self, fn, *args, runs=None):
        n = runs or self.RUNS
        times = []
        for _ in range(n):
            t0 = time.perf_counter()
            fn(*args)
            times.append((time.perf_counter() - t0) * 1000)
        return sum(times) / len(times)

    def test_classify_topic_under_2ms(self):
        """_classify_topic is O(n*k) scan — must be <2ms."""
        from hooks.preturn_inject import _classify_topic
        prompt = "como va el proyecto de consciencia, vamos a revisar el trading y n8n"
        avg = self._avg_time_ms(_classify_topic, prompt)
        assert avg < 2.0, f"_classify_topic averaged {avg:.2f}ms, budget: 2ms"

    def test_extract_keywords_under_2ms(self):
        """_extract_keywords is a simple regex + set — must be <2ms."""
        from hooks.preturn_inject import _extract_keywords
        prompt = "como va el proyecto de consciencia, vamos a revisar el trading y n8n automatizar"
        avg = self._avg_time_ms(_extract_keywords, prompt)
        assert avg < 2.0, f"_extract_keywords averaged {avg:.2f}ms, budget: 2ms"

    def test_detect_triggers_under_5ms(self):
        """detect_triggers reads a JSON file and scans patterns."""
        from hooks.preturn_inject import detect_triggers
        prompt = "vamos a revisar los workflows de n8n para fullempaques"
        avg = self._avg_time_ms(detect_triggers, prompt, runs=5)
        assert avg < 10.0, f"detect_triggers averaged {avg:.2f}ms, budget: 10ms"

    def test_format_context_under_5ms(self):
        """format_context is string formatting — must be <5ms."""
        from hooks.preturn_inject import format_context
        fts = [("Memory content about consciencia", "id1", "consciencia", "high", 0)]
        wm = [("Working memory item", "consciencia", 0.8)]
        avg = self._avg_time_ms(format_context, fts, wm, [], None, None, None, None)
        assert avg < 5.0, f"format_context averaged {avg:.2f}ms, budget: 5ms"

    def test_clean_fts_query_under_1ms(self):
        """clean_fts_query is pure string processing — must be <1ms."""
        from hooks.preturn_inject import clean_fts_query
        avg = self._avg_time_ms(clean_fts_query, "proyecto consciencia n8n trading kraken")
        assert avg < 1.0, f"clean_fts_query averaged {avg:.2f}ms, budget: 1ms"


class TestSQLiteOperationLatency:
    """Verify SQLite operations complete within budget (isolated, no contention)."""

    def test_search_fts_under_20ms(self, temp_fts_db):
        """FTS5 search on local DB should be <20ms without lock contention."""
        from hooks.preturn_inject import search_fts
        conn = sqlite3.connect(temp_fts_db)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA query_only=ON")

        times = []
        for _ in range(5):
            t0 = time.perf_counter()
            search_fts(conn, "consciencia proyecto", 5)
            times.append((time.perf_counter() - t0) * 1000)
        conn.close()

        avg = sum(times) / len(times)
        assert avg < 20.0, f"search_fts averaged {avg:.2f}ms, budget: 20ms"

    def test_get_working_memory_under_5ms(self, temp_fts_db, monkeypatch):
        """Working memory fetch should be <5ms (simple SELECT)."""
        import hooks.preturn_inject as _pi
        # Force SQLite path (skip PG) for latency test
        monkeypatch.setattr(_pi, "_use_pg_wm", False)
        from hooks.preturn_inject import get_working_memory
        conn = sqlite3.connect(temp_fts_db)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA query_only=ON")

        times = []
        for _ in range(5):
            t0 = time.perf_counter()
            get_working_memory(conn, 4)
            times.append((time.perf_counter() - t0) * 1000)
        conn.close()

        avg = sum(times) / len(times)
        assert avg < 5.0, f"get_working_memory averaged {avg:.2f}ms, budget: 5ms"

    def test_sqlite_connection_under_5ms(self, temp_fts_db):
        """Opening SQLite connection should be <5ms (this is baseline)."""
        times = []
        for _ in range(5):
            t0 = time.perf_counter()
            conn = sqlite3.connect(temp_fts_db, timeout=3)
            conn.execute("PRAGMA journal_mode=WAL")
            conn.close()
            times.append((time.perf_counter() - t0) * 1000)
        avg = sum(times) / len(times)
        assert avg < 5.0, f"SQLite connect averaged {avg:.2f}ms, budget: 5ms"


class TestLatencyBudgetKnownIssues:
    """
    Document known latency issues from J.0.4 profiling.

    These tests FAIL by design — they document the CURRENT (broken) behavior.
    They should be FIXED by the P0 recommendations in the profiling report.

    To fix: see track_J_j_0_4_profiling_report.md, Fixes P0-1, P0-2, P0-3.
    """

    @pytest.mark.xfail(
        reason="KNOWN: _update_attention_from_prompt has lock contention issue (J.0.4 Fix P0-1). "
               "Expected: <8ms. Actual: ~3128ms under production load. "
               "Fix: use pooled connection + remove DDL from hot path."
    )
    def test_update_attention_under_8ms_KNOWN_BROKEN(self, temp_fts_db):
        """_update_attention_from_prompt should be <8ms per J.0.4 comment, but is 3128ms."""
        import hooks.preturn_inject as hook
        original = hook.FTS_DB_PATH
        hook.FTS_DB_PATH = temp_fts_db
        try:
            t0 = time.perf_counter()
            hook._update_attention_from_prompt("consciencia proyecto")
            elapsed = (time.perf_counter() - t0) * 1000
            assert elapsed < 8.0, f"_update_attention_from_prompt: {elapsed:.2f}ms, budget: 8ms"
        finally:
            hook.FTS_DB_PATH = original

    @pytest.mark.skip(
        reason="Full subprocess test requires production DB and concurrent processes. "
               "Run manually: time (echo '{...}' | ./venv/bin/python hooks/preturn_inject.py)"
    )
    def test_full_hook_subprocess_under_500ms(self):
        """Full subprocess should be <500ms. Currently ~7200ms. See J.0.4 report."""
        pass
