"""
Shared test fixtures for codi-memory test suite.
Ensures test isolation: no test touches production DBs or leaks event state.

Fixtures:
  - _isolate_sqlite (autouse): all SQLite writes go to tmp_path
  - clean_event_bus: reset event history for inspection
  - fresh_db: explicit clean DB path (alias for _isolate_sqlite's db)
  - fake_mem0: in-memory mem0 stub (no network, no embeddings)
  - fake_qdrant: in-memory QdrantClient stub (no network)
  - patch_externals: replaces mem0 + qdrant in modules.config with fakes
"""

import sys
import os
from datetime import datetime
from typing import Dict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

import pytest
from modules.events import event_bus


@pytest.fixture(autouse=True)
def _isolate_sqlite(tmp_path, monkeypatch):
    """Force all SQLite writes (FTS, event_counts, prospective) to a temp directory.

    Prevents tests from writing to the real DBs in the repo root.
    Scoped per-test via monkeypatch (auto-restored after each test).
    Runs migrations on isolated DBs so all tables exist.
    """
    db_path = str(tmp_path / "memories_fts.db")
    prosp_path = str(tmp_path / "prospective.db")

    # Env vars
    monkeypatch.setenv("FTS_DB_PATH", db_path)
    monkeypatch.setenv("PROSPECTIVE_DB_PATH", prosp_path)

    # Bypass Qdrant auth guardrail in tests (tests mock Qdrant, never connect)
    monkeypatch.setenv("CODI_ALLOW_INSECURE_QDRANT", "1")

    # Module-level config patches
    monkeypatch.setattr("modules.config.FTS_DB_PATH", db_path, raising=False)
    monkeypatch.setattr("modules.config.PROSPECTIVE_DB_PATH", prosp_path, raising=False)

    # Patch FTS_DB_PATH in modules that import it directly (import aliasing fix)
    for mod in ["modules.memory_smart", "modules.memory_core",
                "modules.dual_compare", "modules.write_queue",
                "modules.db_pool", "modules.consolidation_common"]:
        monkeypatch.setattr(f"{mod}.FTS_DB_PATH", db_path, raising=False)

    # Force legacy access tracking in tests (tests mock qdrant, not batch API)
    monkeypatch.setattr(
        "modules.access_tracking.ACCESS_TRACKING_MODE", "legacy", raising=False
    )

    # Redirect flag files to tmp_path so .remember_mode / .write_mode on disk
    # don't contaminate tests. Tests control mode via env vars instead.
    monkeypatch.setattr(
        "modules.interface._REMEMBER_MODE_FILE",
        str(tmp_path / ".remember_mode"),
        raising=False,
    )
    monkeypatch.setattr(
        "modules.interface._WRITE_MODE_FILE",
        str(tmp_path / ".write_mode"),
        raising=False,
    )

    # Run migrations on isolated DBs
    from modules.migrations import apply_migrations
    apply_migrations(db_path, migrations_dir=os.path.join(PROJECT_ROOT, "migrations"))
    apply_migrations(prosp_path, migrations_dir=os.path.join(PROJECT_ROOT, "migrations_prospective"))

    yield

    # Clean up db_pool connections so next test gets fresh connections
    # pointing to its own tmp_path DB
    try:
        from modules.db_pool import close_thread_connections
        close_thread_connections()
    except Exception:
        pass


@pytest.fixture
def clean_event_bus():
    """Provide a clean event bus for tests that inspect event history.

    Saves and restores _history + _dirty_counts even if the test fails.
    Usage: def test_x(self, clean_event_bus): ...
    """
    old_history = event_bus._history[:]
    old_dirty = dict(event_bus._dirty_counts)
    old_total = event_bus._dirty_total
    event_bus._history.clear()
    event_bus._dirty_counts.clear()
    event_bus._dirty_total = 0
    yield
    event_bus._history = old_history
    event_bus._dirty_counts.update(old_dirty)
    event_bus._dirty_total = old_total


# ============================================================
# FRESH DB (explicit alias)
# ============================================================

@pytest.fixture
def fresh_db(tmp_path):
    """Explicit clean DB with all FTS migrations.

    Note: _isolate_sqlite already patches config.FTS_DB_PATH.
    Use this when you need the path string directly.
    """
    from modules.config import FTS_DB_PATH
    return FTS_DB_PATH


# ============================================================
# FAKE MEM0 (memory client stub — no network)
# ============================================================

class FakeMem0:
    """In-memory stub for mem0 Memory client.

    Implements: add, search, delete, delete_all, get_all.
    Search uses simple substring matching (no embeddings).
    """

    def __init__(self):
        self._store: Dict[str, dict] = {}
        self._counter = 0

    def add(self, content: str = "", user_id: str = "", metadata: dict = None,
            messages: list = None) -> dict:
        # Real mem0 uses messages=[{"role":"user","content":"..."}]
        if messages and not content:
            for msg in messages:
                if isinstance(msg, dict) and msg.get("content"):
                    content = msg["content"]
                    break
        self._counter += 1
        mem_id = f"fake-mem-{self._counter}"
        self._store[mem_id] = {
            "id": mem_id,
            "memory": content,
            "user_id": user_id,
            "metadata": metadata or {},
        }
        return {"results": [{"id": mem_id, "memory": content, "event": "ADD"}]}

    def search(self, query: str, user_id: str = "", limit: int = 10) -> dict:
        results = []
        for mem in self._store.values():
            if user_id and mem.get("user_id") != user_id:
                continue
            query_words = query.lower().split()
            text = mem["memory"].lower()
            if any(w in text for w in query_words):
                results.append({
                    "id": mem["id"],
                    "memory": mem["memory"],
                    "score": 0.8,
                    "metadata": mem.get("metadata", {}),
                })
        return {"results": results[:limit]}

    def delete(self, memory_id: str = "", **kwargs) -> dict:
        # Real mem0 uses memory_id kwarg
        mid = memory_id or kwargs.get("memory_id", "")
        if mid in self._store:
            del self._store[mid]
            return {"status": "deleted"}
        return {"status": "not_found"}

    def delete_all(self, user_id: str = "") -> dict:
        if user_id:
            to_del = [k for k, v in self._store.items() if v.get("user_id") == user_id]
        else:
            to_del = list(self._store.keys())
        for k in to_del:
            del self._store[k]
        return {"status": "deleted", "count": len(to_del)}

    def get_all(self, user_id: str = "") -> dict:
        results = []
        for mem in self._store.values():
            if user_id and mem.get("user_id") != user_id:
                continue
            results.append({
                "id": mem["id"],
                "memory": mem["memory"],
                "metadata": mem.get("metadata", {}),
            })
        return {"results": results}


@pytest.fixture
def fake_mem0():
    """Fresh FakeMem0 instance."""
    return FakeMem0()


# ============================================================
# FAKE QDRANT (vector client stub — no network)
# ============================================================

class FakeQdrantPoint:
    """Mimics qdrant_client Record / ScoredPoint."""
    def __init__(self, id: str, payload: dict = None, score: float = 0.9):
        self.id = id
        self.payload = payload or {}
        self.score = score


class FakeQdrant:
    """In-memory stub for QdrantClient.

    Implements: scroll, retrieve, set_payload, upsert, delete.
    """

    def __init__(self):
        self._collections: Dict[str, Dict[str, FakeQdrantPoint]] = {}

    def _ensure(self, name: str):
        if name not in self._collections:
            self._collections[name] = {}

    def scroll(self, collection_name: str, scroll_filter=None,
               limit: int = 10, with_payload: bool = True, **kwargs):
        self._ensure(collection_name)
        points = list(self._collections[collection_name].values())[:limit]
        return (points, None)

    def retrieve(self, collection_name: str, ids: list,
                 with_payload: bool = True, **kwargs):
        self._ensure(collection_name)
        col = self._collections[collection_name]
        return [col[id_] for id_ in ids if id_ in col]

    def set_payload(self, collection_name: str, payload: dict,
                    points: list, **kwargs):
        self._ensure(collection_name)
        col = self._collections[collection_name]
        for pid in points:
            if pid in col:
                col[pid].payload.update(payload)
        return True

    def upsert(self, collection_name: str, points: list, **kwargs):
        self._ensure(collection_name)
        col = self._collections[collection_name]
        for p in points:
            pid = p.id if hasattr(p, 'id') else str(p)
            col[pid] = p
        return True

    def delete(self, collection_name: str, points_selector=None, **kwargs):
        return True

    def get_collection(self, collection_name: str):
        """Stub for qdrant get_collection (returns points_count)."""
        self._ensure(collection_name)

        class _Info:
            def __init__(self, count):
                self.points_count = count
        return _Info(len(self._collections[collection_name]))

    def add_point(self, collection_name: str, id: str, payload: dict):
        """Test helper: pre-populate a point."""
        self._ensure(collection_name)
        self._collections[collection_name][id] = FakeQdrantPoint(id, payload)


@pytest.fixture
def fake_qdrant():
    """Fresh FakeQdrant instance."""
    return FakeQdrant()


# ============================================================
# COMBINED PATCH: replace externals in modules.config
# ============================================================

@pytest.fixture
def patch_externals(monkeypatch, fake_mem0, fake_qdrant):
    """Patch memory + qdrant in modules.config AND all importing modules.

    Handles Python's import-aliasing: modules that do
    'from modules.config import memory' get a local ref
    that config-level patching won't reach.
    """
    import modules.config as cfg

    # Patch at config level (for anyone using cfg.memory)
    monkeypatch.setattr(cfg, "memory", fake_mem0)
    monkeypatch.setattr(cfg, "qdrant", fake_qdrant)

    # Patch local references in core modules that import directly
    _modules_with_memory = [
        "modules.memory_core",
        "modules.memory_smart",
        "modules.consolidation",
        "modules.reconsolidation",
        "modules.semantic_store",
        "modules.maintenance",
        "modules.triggers",
        "modules.books",
        "modules.self_model",
        "modules.emotion",
        "modules.curiosity",
        "modules.flush",
    ]
    for mod_path in _modules_with_memory:
        try:
            monkeypatch.setattr(f"{mod_path}.memory", fake_mem0, raising=False)
            monkeypatch.setattr(f"{mod_path}.qdrant", fake_qdrant, raising=False)
        except Exception:
            pass

    return {
        "mem0": fake_mem0,
        "qdrant": fake_qdrant,
        "db_path": cfg.FTS_DB_PATH,
    }
