#!/usr/bin/env python3
"""
T1.1 — memory_core.py CONTRACT TESTS
======================================
Core invariants: add returns confirmation, search returns results,
delete removes entries, FTS stays in sync, events fire.

Uses patch_externals (FakeMem0 + FakeQdrant), _isolate_sqlite (autouse).

Run: ./venv/bin/pytest tests/test_tier1_memory_core.py -v
     ./venv/bin/pytest tests/test_tier1_memory_core.py -v -m tier1
"""

import sys
import os
import json
import sqlite3

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.events import event_bus, Events

pytestmark = pytest.mark.tier1


# ============================================================
# HELPERS
# ============================================================

def _fts_count(db_path: str) -> int:
    """Count rows in backing FTS table."""
    conn = sqlite3.connect(db_path)
    try:
        cursor = conn.execute("SELECT COUNT(*) FROM memories_text")
        return cursor.fetchone()[0]
    finally:
        conn.close()


def _fts_search(db_path: str, query: str) -> list:
    """Raw FTS5 MATCH search."""
    conn = sqlite3.connect(db_path)
    try:
        cursor = conn.execute(
            "SELECT memory_id, content FROM memories_fts WHERE memories_fts MATCH ?",
            (f'"{query}"',)
        )
        return cursor.fetchall()
    finally:
        conn.close()


# ============================================================
# ADD MEMORY (sync mode)
# ============================================================

class TestAddMemory:
    """Contract: add_memory returns confirmation and indexes content."""

    def test_sync_add_returns_confirmation(self, patch_externals, monkeypatch):
        monkeypatch.setenv("CODI_WRITE_MODE", "sync")
        from modules.memory_core import add_memory
        result = add_memory("Hare likes pizza", category="episodio")
        assert "guardada" in result.lower() or "memoria" in result.lower()

    def test_sync_add_stores_in_mem0(self, patch_externals, monkeypatch):
        monkeypatch.setenv("CODI_WRITE_MODE", "sync")
        from modules.memory_core import add_memory
        add_memory("My favorite color is blue", category="identidad")
        all_mems = patch_externals["mem0"].get_all()
        assert len(all_mems["results"]) >= 1

    def test_sync_add_indexes_fts(self, patch_externals, monkeypatch):
        monkeypatch.setenv("CODI_WRITE_MODE", "sync")
        from modules.memory_core import add_memory
        add_memory("unique_fts_test_token_xyz", category="general")
        count = _fts_count(patch_externals["db_path"])
        assert count >= 1

    def test_sync_add_emits_event(self, patch_externals, monkeypatch, clean_event_bus):
        monkeypatch.setenv("CODI_WRITE_MODE", "sync")
        from modules.memory_core import add_memory
        add_memory("event test content", category="general")
        stored_events = [e for e in event_bus._history if e["event"] == Events.MEMORY_STORED]
        assert len(stored_events) >= 1
        # Event bus stores data_keys, not full data
        assert "content" in stored_events[0]["data_keys"]
        assert "category" in stored_events[0]["data_keys"]

    def test_sync_add_with_importance(self, patch_externals, monkeypatch):
        monkeypatch.setenv("CODI_WRITE_MODE", "sync")
        from modules.memory_core import add_memory
        result = add_memory("critical info", category="identidad", importance="critical")
        assert "guardada" in result.lower() or "memoria" in result.lower()

    def test_async_add_returns_queued_json(self, patch_externals, monkeypatch):
        monkeypatch.setenv("CODI_WRITE_MODE", "async")
        from modules.memory_core import add_memory
        result = add_memory("async test content", category="general")
        parsed = json.loads(result)
        assert parsed["action"] == "queued"
        assert parsed["mode"] == "async"
        assert "job_id" in parsed

    def test_add_error_returns_message(self, patch_externals, monkeypatch):
        """When mem0 raises, add_memory returns error string."""
        monkeypatch.setenv("CODI_WRITE_MODE", "sync")
        def broken_add(*args, **kwargs):
            raise RuntimeError("mem0 is down")
        patch_externals["mem0"].add = broken_add
        from modules.memory_core import add_memory
        result = add_memory("should fail", category="general")
        assert "error" in result.lower()


# ============================================================
# SEARCH MEMORY
# ============================================================

class TestSearchMemory:
    """Contract: search_memory returns formatted results or 'no results' msg."""

    def test_search_returns_results(self, patch_externals, monkeypatch):
        monkeypatch.setenv("CODI_WRITE_MODE", "sync")
        from modules.memory_core import add_memory, search_memory
        add_memory("The project uses FastAPI for the backend", category="aprendizaje")
        result = search_memory("FastAPI")
        # Should find something (FakeMem0 does substring match)
        assert "FastAPI" in result or "encontr" in result.lower()

    def test_search_empty_returns_no_results(self, patch_externals):
        from modules.memory_core import search_memory
        result = search_memory("xyznonexistent12345")
        # Either "no encontre" or empty results
        assert "no encontr" in result.lower() or "error" in result.lower() or "encontrados" in result.lower()

    def test_search_emits_event(self, patch_externals, monkeypatch, clean_event_bus):
        monkeypatch.setenv("CODI_WRITE_MODE", "sync")
        from modules.memory_core import add_memory, search_memory
        add_memory("event tracking test", category="general")
        search_memory("event tracking")
        retrieved = [e for e in event_bus._history if e["event"] == Events.MEMORY_RETRIEVED]
        assert len(retrieved) >= 1

    def test_search_returns_multiple(self, patch_externals, monkeypatch):
        """Adding multiple memories and searching returns multiple results."""
        monkeypatch.setenv("CODI_WRITE_MODE", "sync")
        from modules.memory_core import add_memory, search_memory
        for i in range(5):
            add_memory(f"test memory number {i}", category="general")
        result = search_memory("test memory", limit=10)
        # Should return some numbered results
        lines = [l for l in result.split("\n") if l.strip() and l.strip()[0].isdigit()]
        assert len(lines) >= 2  # At least some results come back

    def test_search_error_returns_message(self, patch_externals, monkeypatch):
        """When mem0.search raises, search_memory returns error or no-results."""
        def broken_search(*args, **kwargs):
            raise RuntimeError("search is down")
        patch_externals["mem0"].search = broken_search
        from modules.memory_core import search_memory
        result = search_memory("anything")
        # search_memory catches exceptions -- returns either error msg or no-results
        assert "error" in result.lower() or "no encontr" in result.lower()


# ============================================================
# DELETE MEMORY (guard off)
# ============================================================

class TestDeleteMemory:
    """Contract: delete removes from mem0 + FTS."""

    def test_delete_removes_from_mem0(self, patch_externals, monkeypatch):
        monkeypatch.setenv("CODI_WRITE_MODE", "sync")
        monkeypatch.setenv("CODI_DESTRUCTIVE_GUARD", "off")
        from modules.memory_core import add_memory, delete_memory
        add_memory("to be deleted", category="general")
        # Get the mem_id from FakeMem0
        all_mems = patch_externals["mem0"].get_all()
        mem_id = all_mems["results"][0]["id"]
        result = delete_memory(mem_id)
        assert "eliminado" in result.lower()
        # Verify it's gone
        remaining = patch_externals["mem0"].get_all()
        assert len(remaining["results"]) == 0

    def test_delete_with_guard_requires_token(self, patch_externals, monkeypatch):
        monkeypatch.setenv("CODI_WRITE_MODE", "sync")
        monkeypatch.setenv("CODI_DESTRUCTIVE_GUARD", "on")
        from modules.memory_core import delete_memory
        result = delete_memory("some-id")
        assert "confirm_token" in result.lower() or "guard" in result.lower()

    def test_delete_dry_run(self, patch_externals, monkeypatch):
        monkeypatch.setenv("CODI_DESTRUCTIVE_GUARD", "on")
        from modules.memory_core import delete_memory
        result = delete_memory("some-id", dry_run=True)
        assert "dry run" in result.lower()


# ============================================================
# CLEAR ALL MEMORIES (guard off)
# ============================================================

class TestClearAllMemories:
    """Contract: clear_all removes everything."""

    def test_clear_all_removes_all(self, patch_externals, monkeypatch):
        monkeypatch.setenv("CODI_WRITE_MODE", "sync")
        monkeypatch.setenv("CODI_DESTRUCTIVE_GUARD", "off")
        from modules.memory_core import add_memory, clear_all_memories
        add_memory("mem 1", category="general")
        add_memory("mem 2", category="general")
        result = clear_all_memories()
        assert "eliminado" in result.lower()
        remaining = patch_externals["mem0"].get_all()
        assert len(remaining["results"]) == 0

    def test_clear_all_with_guard_requires_token(self, patch_externals, monkeypatch):
        monkeypatch.setenv("CODI_DESTRUCTIVE_GUARD", "on")
        from modules.memory_core import clear_all_memories
        result = clear_all_memories()
        assert "confirm_token" in result.lower() or "guard" in result.lower()


# ============================================================
# GET ALL MEMORIES
# ============================================================

class TestGetAllMemories:
    """Contract: get_all_memories returns formatted list."""

    def test_empty_returns_message(self, patch_externals):
        from modules.memory_core import get_all_memories
        result = get_all_memories()
        assert "no hay" in result.lower() or "0" in result or "mostrando" in result.lower()

    def test_returns_populated_list(self, patch_externals):
        # Pre-populate qdrant with a point
        patch_externals["qdrant"].add_point(
            "codi_memories", "test-id-1",
            {"data": "hello world", "category": "general"}
        )
        from modules.memory_core import get_all_memories
        result = get_all_memories()
        assert "hello world" in result or "general" in result


# ============================================================
# SEARCH BY OWNERSHIP
# ============================================================

class TestSearchByOwnership:
    """Contract: ownership filters work via qdrant scroll."""

    def test_no_results_returns_message(self, patch_externals):
        from modules.memory_core import search_by_ownership
        result = search_by_ownership(source="experienced")
        assert "no encontr" in result.lower()

    def test_returns_filtered_results(self, patch_externals):
        patch_externals["qdrant"].add_point(
            "codi_memories", "own-1",
            {"data": "test memory", "ownership_source": "experienced",
             "ownership_confidence": 0.9, "narrative_importance": "high"}
        )
        from modules.memory_core import search_by_ownership
        result = search_by_ownership(source="experienced")
        assert "test memory" in result or "encontr" in result.lower()


# ============================================================
# GET CRITICAL MEMORIES
# ============================================================

class TestGetCriticalMemories:
    """Contract: critical memories filter works."""

    def test_no_critical_returns_message(self, patch_externals):
        from modules.memory_core import get_critical_memories
        result = get_critical_memories()
        assert "no hay" in result.lower() or "no encontr" in result.lower()


# ============================================================
# UPDATE IMPORTANCE
# ============================================================

class TestUpdateImportance:
    """Contract: updates payload in qdrant."""

    def test_invalid_importance_rejected(self, patch_externals):
        from modules.memory_core import update_memory_importance
        result = update_memory_importance("some-id", "super_high")
        assert "debe ser" in result.lower()

    def test_valid_update_returns_confirmation(self, patch_externals, monkeypatch):
        # Pre-populate qdrant
        patch_externals["qdrant"].add_point(
            "codi_memories", "upd-full-id",
            {"data": "updatable", "narrative_importance": "low"}
        )
        # resolve_memory_id does a scroll + match, so we need to mock it
        monkeypatch.setattr(
            "modules.memory_core.resolve_memory_id",
            lambda mid: "upd-full-id" if "upd" in mid else None
        )
        from modules.memory_core import update_memory_importance
        result = update_memory_importance("upd", "critical")
        assert "actualizada" in result.lower() or "critical" in result.lower()


# ============================================================
# RESTORE MEMORIES
# ============================================================

class TestRestoreMemories:
    """Contract: restore reads backup file and adds to mem0."""

    def test_no_backup_file(self, patch_externals, monkeypatch):
        monkeypatch.setattr("modules.memory_core.BACKUP_FILE", "/nonexistent/backup.json")
        from modules.memory_core import restore_memories
        result = restore_memories()
        assert "no existe" in result.lower()

    def test_restore_from_backup(self, patch_externals, monkeypatch, tmp_path):
        backup_file = str(tmp_path / "test_backup.json")
        backup_data = {
            "memories": [
                {"memory": "restored memory 1", "metadata": {"category": "general"}},
                {"memory": "restored memory 2", "metadata": {"category": "identidad"}},
            ]
        }
        import json as _json
        with open(backup_file, "w") as f:
            _json.dump(backup_data, f)

        monkeypatch.setattr("modules.memory_core.BACKUP_FILE", backup_file)
        from modules.memory_core import restore_memories
        result = restore_memories()
        assert "restauradas 2" in result.lower()
