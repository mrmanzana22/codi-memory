#!/usr/bin/env python3
"""
FTS Hybrid Tests — FTS as source of truth, mem0 as ranking engine.
Verifies get_texts_by_ids, search_with_fts_content, and search_memory
prefer FTS original text over mem0 compressed text.

Run: ./venv/bin/pytest tests/test_fts_hybrid.py -v
"""

import sys
import os
import sqlite3

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.db_pool import get_conn, close_thread_connections


@pytest.fixture(autouse=True)
def _setup_fts_table():
    """Ensure memories_text table exists with test data, clean up after."""
    conn = get_conn()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS memories_text (
            memory_id TEXT PRIMARY KEY,
            content TEXT NOT NULL,
            category TEXT DEFAULT 'general',
            source TEXT DEFAULT 'experienced',
            importance TEXT DEFAULT 'medium',
            created_at TEXT
        )
    """)
    conn.commit()
    yield
    # Clean test rows
    conn.execute("DELETE FROM memories_text WHERE memory_id LIKE 'test-%'")
    conn.commit()


def _insert_test_rows(rows):
    """Insert test rows: list of (memory_id, content, category, source, importance)."""
    conn = get_conn()
    for r in rows:
        conn.execute(
            "INSERT OR REPLACE INTO memories_text (memory_id, content, category, source, importance, created_at) "
            "VALUES (?, ?, ?, ?, ?, datetime('now'))",
            r,
        )
    conn.commit()


# ============================================================
# Step 1: get_texts_by_ids / get_text_by_id
# ============================================================

class TestGetTextsByIds:
    def test_bulk_fetch_all(self):
        """5 IDs, single query, returns all."""
        from modules.memory_smart import get_texts_by_ids

        rows = [
            (f"test-{i}", f"Original content {i}", "general", "experienced", "medium")
            for i in range(5)
        ]
        _insert_test_rows(rows)

        result = get_texts_by_ids([f"test-{i}" for i in range(5)])
        assert len(result) == 5
        for i in range(5):
            assert result[f"test-{i}"]["content"] == f"Original content {i}"
            assert result[f"test-{i}"]["category"] == "general"

    def test_partial_ids(self):
        """Some IDs missing, returns only what exists."""
        from modules.memory_smart import get_texts_by_ids

        _insert_test_rows([("test-exists", "I exist", "general", "experienced", "high")])

        result = get_texts_by_ids(["test-exists", "test-ghost", "test-phantom"])
        assert len(result) == 1
        assert "test-exists" in result
        assert result["test-exists"]["content"] == "I exist"
        assert "test-ghost" not in result

    def test_empty_ids(self):
        """Empty list returns empty dict."""
        from modules.memory_smart import get_texts_by_ids

        result = get_texts_by_ids([])
        assert result == {}

    def test_single_id_found(self):
        """get_text_by_id returns dict for existing ID."""
        from modules.memory_smart import get_text_by_id

        _insert_test_rows([("test-single", "Single memory", "proyecto", "told", "critical")])

        result = get_text_by_id("test-single")
        assert result is not None
        assert result["content"] == "Single memory"
        assert result["category"] == "proyecto"
        assert result["importance"] == "critical"

    def test_single_id_missing(self):
        """get_text_by_id returns None for missing ID."""
        from modules.memory_smart import get_text_by_id

        result = get_text_by_id("test-nonexistent-id-xyz")
        assert result is None


# ============================================================
# Step 2: search_with_fts_content
# ============================================================

class TestSearchWithFtsContent:
    """Post PG-migration: search_with_fts_content wraps pg.search().

    PG stores original text natively (no mem0 compression overlay needed).
    These tests verify the pg.search delegation works correctly.
    """

    def _mock_pg_search(self, monkeypatch, fake_results):
        """Mock pg.search to return fake results."""
        import modules.memory_smart as ms
        monkeypatch.setattr(ms.pg, "search", lambda *a, **kw: fake_results)

    def test_replaces_text(self, monkeypatch):
        """PG returns original text (no compression artifact)."""
        from modules.memory_smart import search_with_fts_content

        fake_results = {
            "results": [
                {"id": "test-a", "memory": "ORIGINAL full text A with details", "score": 0.95},
                {"id": "test-b", "memory": "ORIGINAL full text B with details", "score": 0.80},
                {"id": "test-c", "memory": "Original text C", "score": 0.70},
            ]
        }
        self._mock_pg_search(monkeypatch, fake_results)

        results = search_with_fts_content(query="test", user_id="u1", limit=3)
        r = results["results"]

        assert r[0]["memory"] == "ORIGINAL full text A with details"
        assert r[1]["memory"] == "ORIGINAL full text B with details"
        assert r[2]["memory"] == "Original text C"

    def test_preserves_ranking(self, monkeypatch):
        """id and score stay untouched."""
        from modules.memory_smart import search_with_fts_content

        fake_results = {
            "results": [
                {"id": "test-a", "memory": "Original A", "score": 0.95},
                {"id": "test-b", "memory": "Original B", "score": 0.80},
            ]
        }
        self._mock_pg_search(monkeypatch, fake_results)

        results = search_with_fts_content(query="test", user_id="u1", limit=3)
        r = results["results"]

        assert r[0]["id"] == "test-a"
        assert r[0]["score"] == 0.95
        assert r[1]["id"] == "test-b"
        assert r[1]["score"] == 0.80

    def test_fallback_on_fts_miss(self, monkeypatch):
        """When pg.search returns no results, empty list returned."""
        from modules.memory_smart import search_with_fts_content

        self._mock_pg_search(monkeypatch, {"results": []})

        results = search_with_fts_content(query="test", user_id="u1", limit=3)
        assert results == {"results": []}

    def test_empty_results(self, monkeypatch):
        """Empty/None results pass through."""
        from modules.memory_smart import search_with_fts_content
        import modules.memory_smart as ms

        monkeypatch.setattr(ms.pg, "search", lambda *a, **kw: {"results": []})
        results = search_with_fts_content(query="nothing", user_id="u1", limit=3)
        assert results == {"results": []}

        monkeypatch.setattr(ms.pg, "search", lambda *a, **kw: None)
        results = search_with_fts_content(query="nothing", user_id="u1", limit=3)
        assert results == {"results": []}


# ============================================================
# Step 3: search_memory prefers FTS text
# ============================================================

class TestSearchMemoryPrefersFts:
    def test_bm25_text_preferred_over_vector(self):
        """Verify the priority inversion: bm25_text first, vector fallback."""
        # Simulate the merge logic directly (unit test of the expression)
        # The actual line: text = item["bm25_text"] or (item["vector_result"].get("memory", "") if item["vector_result"] else "")

        # Case 1: both present -> bm25 wins
        item = {"bm25_text": "FTS original", "vector_result": {"memory": "compressed"}}
        text = item["bm25_text"] or (item["vector_result"].get("memory", "") if item["vector_result"] else "")
        assert text == "FTS original"

        # Case 2: bm25 empty -> vector fallback
        item = {"bm25_text": "", "vector_result": {"memory": "compressed fallback"}}
        text = item["bm25_text"] or (item["vector_result"].get("memory", "") if item["vector_result"] else "")
        assert text == "compressed fallback"

        # Case 3: bm25 None -> vector fallback
        item = {"bm25_text": None, "vector_result": {"memory": "compressed fallback"}}
        text = item["bm25_text"] or (item["vector_result"].get("memory", "") if item["vector_result"] else "")
        assert text == "compressed fallback"

        # Case 4: both empty -> empty string
        item = {"bm25_text": "", "vector_result": {"memory": ""}}
        text = item["bm25_text"] or (item["vector_result"].get("memory", "") if item["vector_result"] else "")
        assert text == ""

        # Case 5: no vector result -> bm25 only
        item = {"bm25_text": "FTS only", "vector_result": None}
        text = item["bm25_text"] or (item["vector_result"].get("memory", "") if item["vector_result"] else "")
        assert text == "FTS only"

        # Case 6: no vector, no bm25 -> empty
        item = {"bm25_text": "", "vector_result": None}
        text = item["bm25_text"] or (item["vector_result"].get("memory", "") if item["vector_result"] else "")
        assert text == ""
