"""
Shared test fixtures for codi-memory test suite.
Ensures test isolation: no test touches production DBs or leaks event state.

Fixtures:
  - _isolate_sqlite (autouse): all SQLite writes go to tmp_path
  - clean_event_bus: reset event history for inspection
  - fresh_db: explicit clean DB path (alias for _isolate_sqlite's db)
  - fake_mem0: in-memory mem0 stub (no network, no embeddings)
  - fake_qdrant: in-memory QdrantClient stub (no network)
  - fake_pg: in-memory PGMemoryStore stub (replaces pg_store.pg)
  - patch_externals: replaces mem0 + qdrant + pg in modules with fakes
"""

import sys
import os
import uuid
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

    # Run migrations BEFORE triggering module imports via monkeypatch.setattr.
    # consolidation_common.py validates tables at import time — tables must exist
    # before any monkeypatch.setattr triggers its import chain.
    from modules.migrations import apply_migrations
    apply_migrations(db_path, migrations_dir=os.path.join(PROJECT_ROOT, "migrations"))
    apply_migrations(prosp_path, migrations_dir=os.path.join(PROJECT_ROOT, "migrations_prospective"))

    # Patch FTS_DB_PATH in modules that import it directly (import aliasing fix)
    for mod in ["modules.memory_smart", "modules.memory_core",
                "modules.dual_compare", "modules.write_queue",
                "modules.db_pool", "modules.consolidation_common",
                "modules.sleep_loop", "modules.active_inference_integration"]:
        monkeypatch.setattr(f"{mod}.FTS_DB_PATH", db_path, raising=False)

    # Patch PROSPECTIVE_DB_PATH aliases (modules that import directly from config)
    for mod in ["modules.prospective", "modules.goals"]:
        monkeypatch.setattr(f"{mod}.PROSPECTIVE_DB_PATH", prosp_path, raising=False)

    # Reset cached DB connections (Tier 1 — prevents wrong-DB reads/writes)
    monkeypatch.setattr("modules.prospective._conn", None, raising=False)
    monkeypatch.setattr("modules.goals._conn", None, raising=False)

    # Reset DDL guards (Tier 2 — ensures table creation on fresh DBs)
    monkeypatch.setattr("hooks.preturn_inject._DDL_DONE", False, raising=False)
    monkeypatch.setattr("hooks.preturn_inject._PREDICTION_DDL_DONE", False, raising=False)

    # Reset wiring state (Tier 4 — accumulated counters/timestamps)
    monkeypatch.setattr("modules.wiring._wired", False, raising=False)
    monkeypatch.setattr("modules.wiring._interaction_count", 0, raising=False)
    monkeypatch.setattr("modules.wiring._session_interaction_count", 0, raising=False)
    monkeypatch.setattr("modules.wiring._last_interaction_time", None, raising=False)
    monkeypatch.setattr("modules.wiring._self_model_tick", 0, raising=False)

    # Reset prediction state in-place (Tier 4 — preserves object identity for
    # tests that check `_predictive_state is ps` across facade re-exports)
    import modules.prediction as _pred_mod
    _saved_pred = {k: list(v) for k, v in _pred_mod._predictive_state.items()}
    _pred_mod._predictive_state.update({
        "predictions": [], "surprises": [], "belief_updates": [], "accuracy_history": []
    })

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

    yield

    # Restore prediction state (in-place mutation, not replacement)
    _pred_mod._predictive_state.update(_saved_pred)

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
# FAKE PG (PostgreSQL store stub -- replaces pg_store.pg)
# ============================================================

class _FakeCollectionInfo:
    """Mimics pg_store.CollectionInfo."""
    def __init__(self, points_count: int = 0):
        self.points_count = points_count


class _FakePGPoint:
    """Mimics pg_store.Point for FakePG results."""
    def __init__(self, id: str, payload: dict = None, score: float = 0.0, vector=None):
        self.id = id
        self.payload = payload or {}
        self.score = score
        self.vector = vector


class FakePG:
    """In-memory stub for pg_store.PGMemoryStore.

    Implements the same interface as PGMemoryStore but stores everything
    in a dict. No network, no PostgreSQL, no embeddings.
    Search uses simple substring matching (no vectors).
    """

    def __init__(self):
        self._store: Dict[str, dict] = {}
        self._counter = 0

    def add(
        self,
        content: str,
        category: str = "general",
        source: str = "experienced",
        importance: str = "medium",
        embedding=None,
        is_semantic: bool = False,
        confidence: float = 0.5,
        evidence_count: int = 0,
        metadata: dict = None,
        emotion_p: float = 0.0,
        emotion_a: float = 0.0,
        emotion_d: float = 0.0,
    ) -> dict:
        self._counter += 1
        mem_id = str(uuid.uuid4())
        now = datetime.now().isoformat()
        self._store[mem_id] = {
            "id": mem_id,
            "content": content,
            "category": category,
            "source": source,
            "importance": importance,
            "is_semantic": is_semantic,
            "confidence": confidence,
            "evidence_count": evidence_count,
            "metadata": metadata or {},
            "emotion_p": emotion_p,
            "emotion_a": emotion_a,
            "emotion_d": emotion_d,
            "created_at": now,
            "updated_at": now,
            "last_accessed_at": now,
            "activation_score": 0.0,
            "access_count": 0,
            "storage_strength": 1.0,
            "retrieval_strength": 1.0,
            "is_dormant": False,
            "dormant_at": "",
            "reactivation_count": 0,
        }
        return {"results": [{"id": mem_id, "created_at": now}]}

    def search(
        self, query: str, limit: int = 5, embedding=None,
        is_semantic=None, w_vector=0.4, w_fts=0.15, w_activation=0.45,
        include_dormant=False,
    ) -> dict:
        results = []
        for mem_id, mem in self._store.items():
            if is_semantic is not None and mem.get("is_semantic") != is_semantic:
                continue
            query_words = query.lower().split()
            text = mem["content"].lower()
            if any(w in text for w in query_words):
                results.append({
                    "id": mem_id,
                    "memory": mem["content"],
                    "score": 0.8,
                    "category": mem.get("category", "general"),
                    "source": mem.get("source", "experienced"),
                    "importance": mem.get("importance", "medium"),
                    "is_semantic": mem.get("is_semantic", False),
                    "confidence": mem.get("confidence", 0.5),
                    "evidence_count": mem.get("evidence_count", 0),
                })
        return {"results": results[:limit]}

    def delete(self, memory_id: str) -> bool:
        if memory_id in self._store:
            del self._store[memory_id]
            return True
        return False

    def get_by_ids(self, ids: list, with_vectors: bool = False) -> list:
        result = []
        for mid in ids:
            mid_str = str(mid)
            if mid_str in self._store:
                mem = self._store[mid_str]
                result.append(_FakePGPoint(
                    id=mid_str,
                    payload=self._make_payload(mem),
                ))
        return result

    def scroll(
        self, filters=None, limit: int = 50, offset: int = 0,
        is_semantic=None, with_vectors: bool = False, order_by: str = "created_at DESC",
    ) -> tuple:
        points = []
        for mem_id, mem in self._store.items():
            if is_semantic is not None and mem.get("is_semantic") != is_semantic:
                continue
            if filters:
                match = True
                for k, v in filters.items():
                    if k in ("category", "source", "importance"):
                        if mem.get(k) != v:
                            match = False
                            break
                    elif k == "ownership_source":
                        if mem.get("source") != v:
                            match = False
                            break
                    elif k == "narrative_importance":
                        if mem.get("importance") != v:
                            match = False
                            break
                if not match:
                    continue
            points.append(_FakePGPoint(
                id=mem_id,
                payload=self._make_payload(mem),
            ))
        return (points[:limit], None)

    def count(self, is_semantic=None) -> _FakeCollectionInfo:
        c = 0
        for mem in self._store.values():
            if is_semantic is not None and mem.get("is_semantic") != is_semantic:
                continue
            c += 1
        return _FakeCollectionInfo(c)

    def update_payload(self, memory_id: str, updates: dict) -> bool:
        if memory_id in self._store:
            mem = self._store[memory_id]
            for k, v in updates.items():
                if k in mem:
                    mem[k] = v
                else:
                    meta = mem.setdefault("metadata", {})
                    meta[k] = v
            return True
        return False

    def query_vector(
        self, embedding, limit: int = 10, is_semantic=None,
        score_threshold=None, with_vectors=False, include_dormant=False,
    ) -> list:
        """Return all stored memories as fake vector results (score=0.8).

        Tests that need empty results should clear the store first.
        """
        results = []
        for mem_id, mem in self._store.items():
            if is_semantic is not None and mem.get("is_semantic") != is_semantic:
                continue
            results.append(_FakePGPoint(
                id=mem_id,
                payload=self._make_payload(mem),
                score=0.8,
            ))
        return results[:limit]

    def search_fts(self, query: str, limit: int = 20) -> list:
        """Substring-based FTS stub on internal store.

        Returns [{memory_id, content, category, source, bm25_rank}].
        Returns empty by default (search_fts in memory_smart falls back to SQLite FTS).
        """
        if not query or not query.strip():
            return []
        results = []
        query_words = query.lower().split()
        for mem_id, mem in self._store.items():
            text = mem.get("content", "").lower()
            if any(w in text for w in query_words):
                results.append({
                    "memory_id": mem_id,
                    "content": mem["content"],
                    "category": mem.get("category", "general"),
                    "source": mem.get("source", "experienced"),
                    "bm25_rank": -5.0,
                })
        return results[:limit]

    def search_vault(self, embedding, limit: int = 5) -> list:
        return []

    def reactivate_memory(self, *args, **kwargs) -> bool:
        return False

    def get_all(self, limit: int = 500, is_semantic=None) -> list:
        points, _ = self.scroll(is_semantic=is_semantic, limit=limit)
        return points

    def _make_payload(self, mem: dict) -> dict:
        """Build a Point-compatible payload dict from internal store."""
        payload = {
            "data": mem.get("content", ""),
            "category": mem.get("category", "general"),
            "source": mem.get("source", "experienced"),
            "narrative_importance": mem.get("importance", "medium"),
            "importance": mem.get("importance", "medium"),
            "confidence": mem.get("confidence", 0.5),
            "evidence_count": mem.get("evidence_count", 0),
            "memory_type": "semantic" if mem.get("is_semantic") else "episodic",
            "is_semantic": mem.get("is_semantic", False),
            "created_at": mem.get("created_at", ""),
            "updated_at": mem.get("updated_at", ""),
            "last_accessed_at": mem.get("last_accessed_at", ""),
            "attention_salience": mem.get("activation_score", 0.0),
            "activation_score": mem.get("activation_score", 0.0),
            "attention_access_count": mem.get("access_count", 0),
            "access_count": mem.get("access_count", 0),
            "storage_strength": mem.get("storage_strength", 1.0),
            "retrieval_strength": mem.get("retrieval_strength", 1.0),
            "is_dormant": mem.get("is_dormant", False),
            "dormant_at": mem.get("dormant_at", ""),
            "reactivation_count": mem.get("reactivation_count", 0),
            "emotion_p": mem.get("emotion_p", 0.0),
            "emotion_a": mem.get("emotion_a", 0.0),
            "emotion_d": mem.get("emotion_d", 0.0),
            "user_id": "hare",
            "ownership_source": mem.get("source", "experienced"),
            "ownership_confidence": 0.9 if mem.get("source") == "experienced" else 0.7,
        }
        # Merge extra metadata
        meta = mem.get("metadata", {})
        if isinstance(meta, dict):
            for k, v in meta.items():
                if k not in payload:
                    payload[k] = v
        return payload

    # Test helper methods (backward-compatible with FakeQdrant.add_point)
    def add_point(self, collection_name: str, id: str, payload: dict):
        """Pre-populate a point for tests (backward compat with FakeQdrant)."""
        self._store[id] = {
            "id": id,
            "content": payload.get("data", payload.get("content", "")),
            "category": payload.get("category", "general"),
            "source": payload.get("ownership_source", payload.get("source", "experienced")),
            "importance": payload.get("narrative_importance", payload.get("importance", "medium")),
            "is_semantic": payload.get("is_semantic", False),
            "confidence": payload.get("confidence", 0.5),
            "evidence_count": payload.get("evidence_count", 0),
            "metadata": payload,
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "last_accessed_at": datetime.now().isoformat(),
            "activation_score": 0.0,
            "access_count": 0,
            "storage_strength": 1.0,
            "retrieval_strength": 1.0,
            "is_dormant": False,
            "dormant_at": "",
            "reactivation_count": 0,
            "emotion_p": 0.0,
            "emotion_a": 0.0,
            "emotion_d": 0.0,
        }

    def get_all_raw(self) -> dict:
        """Return all memories in mem0-compatible format for test assertions."""
        results = []
        for mem_id, mem in self._store.items():
            results.append({
                "id": mem_id,
                "memory": mem["content"],
                "metadata": mem.get("metadata", {}),
            })
        return {"results": results}


@pytest.fixture
def fake_pg():
    """Fresh FakePG instance."""
    return FakePG()


# ============================================================
# COMBINED PATCH: replace externals in modules.config
# ============================================================

@pytest.fixture
def patch_externals(monkeypatch, fake_mem0, fake_qdrant, fake_pg):
    """Patch memory + qdrant + pg in modules.config AND all importing modules.

    Handles Python's import-aliasing: modules that do
    'from modules.config import memory' get a local ref
    that config-level patching won't reach.

    Post PG migration: pg (FakePG) is the primary mock.
    mem0 and qdrant are kept for backward compatibility with older tests.
    """
    import modules.config as cfg

    # Patch at config level (for anyone using cfg.memory / cfg.qdrant)
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

    # Patch pg (PGMemoryStore) in pg_store module and all modules that import it
    monkeypatch.setattr("modules.pg_store.pg", fake_pg)
    _modules_with_pg = [
        "modules.memory_core",
        "modules.memory_smart",
        "modules.consolidation",
        "modules.reconsolidation",
        "modules.semantic_store",
        "modules.maintenance",
        "modules.flush",
        "modules.utils",
        "modules.books",
        "modules.self_model",
        "modules.emotion",
        "modules.curiosity",
        "modules.wiring",
        "modules.workspace",
        "modules.access_tracking",
        "modules.forgetting",
        "modules.spreading",
        "modules.narrative",
        "modules.prediction",
        "modules.retrieval_metadata",
        "modules.sleep_loop",
        "modules.lifecycle",
        "modules.sharpe",
        "modules.sharpe_insights",
        "modules.classify_edges",
        "modules.learning",
    ]
    for mod_path in _modules_with_pg:
        try:
            monkeypatch.setattr(f"{mod_path}.pg", fake_pg, raising=False)
        except Exception:
            pass

    # Mock _embed_text to avoid OpenAI API calls in tests.
    # Functions like search_memory() and search_with_fts_content() call
    # _embed_text directly (not through pg), so we must mock it.
    _dummy_embedding = [0.1] * 1536
    _dummy_embedding_tuple = tuple(_dummy_embedding)
    monkeypatch.setattr(
        "modules.consolidation_common._embed_text",
        lambda text: _dummy_embedding,
        raising=False,
    )
    monkeypatch.setattr(
        "modules.consolidation_common._embed_text_cached",
        lambda text: _dummy_embedding_tuple,
        raising=False,
    )
    monkeypatch.setattr(
        "modules.pg_store._embed_text",
        lambda text: _dummy_embedding,
        raising=False,
    )
    # Also patch in memory_core where it's imported directly
    monkeypatch.setattr(
        "modules.memory_core._embed_text",
        lambda text: _dummy_embedding,
        raising=False,
    )

    return {
        "mem0": fake_mem0,
        "qdrant": fake_qdrant,
        "pg": fake_pg,
        "db_path": cfg.FTS_DB_PATH,
    }
