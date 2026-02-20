#!/usr/bin/env python3
"""
T1.0 SMOKE TEST — Verify test fixtures work correctly
======================================================
Validates that FakeMem0, FakeQdrant, patch_externals, and
fresh_db fixtures behave as expected before building Tier 1 tests.

Run: ./venv/bin/pytest tests/test_tier1_smoke.py -v
"""

import sys
import os
import sqlite3

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ============================================================
# FakeMem0
# ============================================================

class TestFakeMem0:
    """Verify FakeMem0 stub implements mem0 contract."""

    def test_add_returns_result(self, fake_mem0):
        result = fake_mem0.add("hello world", user_id="u1")
        assert "results" in result
        assert result["results"][0]["event"] == "ADD"
        assert result["results"][0]["memory"] == "hello world"

    def test_search_by_substring(self, fake_mem0):
        fake_mem0.add("the quick brown fox", user_id="u1")
        fake_mem0.add("lazy dog sleeps", user_id="u1")
        results = fake_mem0.search("fox", user_id="u1")
        assert len(results["results"]) == 1
        assert "fox" in results["results"][0]["memory"]

    def test_search_filters_by_user(self, fake_mem0):
        fake_mem0.add("shared content", user_id="u1")
        fake_mem0.add("shared content", user_id="u2")
        results = fake_mem0.search("shared", user_id="u1")
        assert len(results["results"]) == 1

    def test_delete(self, fake_mem0):
        result = fake_mem0.add("to delete", user_id="u1")
        mem_id = result["results"][0]["id"]
        fake_mem0.delete(mem_id)
        all_mems = fake_mem0.get_all()
        assert len(all_mems["results"]) == 0

    def test_delete_all(self, fake_mem0):
        fake_mem0.add("one", user_id="u1")
        fake_mem0.add("two", user_id="u1")
        fake_mem0.add("three", user_id="u2")
        fake_mem0.delete_all(user_id="u1")
        remaining = fake_mem0.get_all()
        assert len(remaining["results"]) == 1

    def test_get_all(self, fake_mem0):
        fake_mem0.add("a", user_id="u1")
        fake_mem0.add("b", user_id="u1")
        all_mems = fake_mem0.get_all(user_id="u1")
        assert len(all_mems["results"]) == 2


# ============================================================
# FakeQdrant
# ============================================================

class TestFakeQdrant:
    """Verify FakeQdrant stub implements qdrant contract."""

    def test_add_point_and_scroll(self, fake_qdrant):
        fake_qdrant.add_point("col1", "p1", {"text": "hello"})
        points, _ = fake_qdrant.scroll("col1")
        assert len(points) == 1
        assert points[0].payload["text"] == "hello"

    def test_retrieve_by_ids(self, fake_qdrant):
        fake_qdrant.add_point("col1", "p1", {"a": 1})
        fake_qdrant.add_point("col1", "p2", {"a": 2})
        result = fake_qdrant.retrieve("col1", ids=["p2"])
        assert len(result) == 1
        assert result[0].payload["a"] == 2

    def test_set_payload(self, fake_qdrant):
        fake_qdrant.add_point("col1", "p1", {"x": 1})
        fake_qdrant.set_payload("col1", {"x": 99}, points=["p1"])
        result = fake_qdrant.retrieve("col1", ids=["p1"])
        assert result[0].payload["x"] == 99

    def test_scroll_empty_collection(self, fake_qdrant):
        points, _ = fake_qdrant.scroll("nonexistent")
        assert points == []

    def test_delete_returns_true(self, fake_qdrant):
        assert fake_qdrant.delete("col1") is True


# ============================================================
# patch_externals
# ============================================================

class TestPatchExternals:
    """Verify patch_externals wires fakes into modules.config."""

    def test_config_memory_is_fake(self, patch_externals):
        import modules.config as cfg
        # After patching, cfg.memory should be the FakeMem0 instance
        assert cfg.memory is patch_externals["mem0"]

    def test_config_qdrant_is_fake(self, patch_externals):
        import modules.config as cfg
        assert cfg.qdrant is patch_externals["qdrant"]

    def test_db_path_is_temp(self, patch_externals):
        db_path = patch_externals["db_path"]
        assert "memories_fts.db" in db_path
        # Should NOT be in the repo root (macOS uses /private/var/folders/...)
        assert "pytest-" in db_path


# ============================================================
# fresh_db
# ============================================================

class TestFreshDb:
    """Verify fresh_db provides a migrated temp DB."""

    def test_returns_string_path(self, fresh_db):
        assert isinstance(fresh_db, str)
        assert fresh_db.endswith(".db")

    def test_has_fts_table(self, fresh_db):
        conn = sqlite3.connect(fresh_db)
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='memories_fts'"
        )
        tables = [row[0] for row in cursor.fetchall()]
        conn.close()
        assert "memories_fts" in tables

    def test_is_isolated_per_test(self, fresh_db):
        """Each test gets its own DB — inserting into backing table works."""
        conn = sqlite3.connect(fresh_db)
        # memories_fts is content-synced from memories_text, so insert there
        conn.execute(
            "INSERT INTO memories_text (content, memory_id, category, source) "
            "VALUES ('smoke test', 'smoke-1', 'test', 'smoke')"
        )
        conn.commit()
        cursor = conn.execute("SELECT COUNT(*) FROM memories_text")
        count = cursor.fetchone()[0]
        conn.close()
        assert count >= 1  # This insert exists only in this test's DB
