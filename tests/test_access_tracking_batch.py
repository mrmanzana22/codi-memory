#!/usr/bin/env python3
"""
ACCESS TRACKING BATCH TESTS
============================
Unit tests for modules/access_tracking.py aggregator + flusher.

7 tests:
  1. Aggregation: multiple record_access for same id merges correctly
  2. Aggregation: different ids kept separate
  3. flush_now builds correct batch ops and calls batch_update_points
  4. Chunking: >MAX_OPS_PER_BATCH items split into multiple calls
  5. Failure in batch_update_points does not crash, increments error/dropped
  6. record_spreading enqueues all updates correctly
  7. Legacy mode calls set_payload directly (bypass aggregator)

Run: ./venv/bin/pytest tests/test_access_tracking_batch.py -v
"""

import sys
import os
import unittest.mock

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture(autouse=True)
def _reset_module(monkeypatch):
    """Reset access_tracking state before each test and force batched mode."""
    import modules.access_tracking as at
    monkeypatch.setattr(at, "ACCESS_TRACKING_MODE", "batched")
    monkeypatch.setattr(at, "FLUSH_INTERVAL", 60)  # prevent auto-flush
    monkeypatch.setattr(at, "MAX_PENDING", 99999)   # prevent overflow flush
    at._reset_for_testing()
    yield
    at._reset_for_testing()


class TestAggregation:
    """record_access merges payloads per (collection, point_id)."""

    def test_same_id_merges(self):
        """Multiple writes to same id: last-writer-wins per field."""
        import modules.access_tracking as at

        at.record_access("coll_a", "id1", {
            "attention_access_count": 5,
            "attention_last_accessed": "t1",
        })
        at.record_access("coll_a", "id1", {
            "attention_access_count": 6,
            "attention_last_accessed": "t2",
            "extra_field": "new",
        })

        with at._lock:
            # _pending is keyed by str(point_id), not (collection, point_id)
            entry = at._pending["id1"]

        assert entry["attention_access_count"] == 6
        assert entry["attention_last_accessed"] == "t2"
        assert entry["extra_field"] == "new"
        assert at._stats["enqueued"] == 2

    def test_different_ids_separate(self):
        """Different ids are stored independently."""
        import modules.access_tracking as at

        at.record_access("coll_a", "id1", {"count": 1})
        at.record_access("coll_a", "id2", {"count": 2})
        at.record_access("coll_b", "id1", {"count": 3})

        with at._lock:
            # pg_store is collection-agnostic: keyed by point_id only
            assert len(at._pending) == 2  # id1 (merged), id2
            assert at._pending["id1"]["count"] == 3  # last-writer-wins: coll_b overwrites coll_a
            assert at._pending["id2"]["count"] == 2


class TestFlush:
    """flush_now drains pending and calls batch_update_points."""

    def test_flush_builds_batch_ops(self, monkeypatch):
        """flush_now groups by collection, calls batch_update_points once per collection."""
        import modules.access_tracking as at
        from unittest.mock import MagicMock

        mock_qdrant = MagicMock()
        monkeypatch.setattr("modules.config.qdrant", mock_qdrant)

        at.record_access("codi_memories", "id1", {"count": 1})
        at.record_access("codi_memories", "id2", {"count": 2})
        at.record_access("codi_semantic", "id3", {"obs": "t1"})

        stats = at.flush_now("test")

        assert mock_qdrant.batch_update_points.call_count == 2
        calls = mock_qdrant.batch_update_points.call_args_list

        collections_called = {c.kwargs["collection_name"] for c in calls}
        assert collections_called == {"codi_memories", "codi_semantic"}

        # Find the episodic call
        for call in calls:
            if call.kwargs["collection_name"] == "codi_memories":
                ops = call.kwargs["update_operations"]
                assert len(ops) == 2

        assert stats["flushed"] == 3
        assert stats["flush_count"] == 1
        assert stats["pending"] == 0

    def test_chunking(self, monkeypatch):
        """More than MAX_OPS_PER_BATCH items split into multiple calls."""
        import modules.access_tracking as at
        from unittest.mock import MagicMock

        monkeypatch.setattr(at, "MAX_OPS_PER_BATCH", 5)
        mock_qdrant = MagicMock()
        monkeypatch.setattr("modules.config.qdrant", mock_qdrant)

        for i in range(12):
            at.record_access("coll", f"id{i}", {"v": i})

        at.flush_now("test")

        # 12 items / 5 per chunk = 3 calls
        assert mock_qdrant.batch_update_points.call_count == 3

    def test_failure_does_not_crash(self, monkeypatch):
        """Qdrant error increments dropped/errors but doesn't raise."""
        import modules.access_tracking as at
        from unittest.mock import MagicMock

        mock_qdrant = MagicMock()
        mock_qdrant.batch_update_points.side_effect = RuntimeError("network")
        monkeypatch.setattr("modules.config.qdrant", mock_qdrant)

        at.record_access("coll", "id1", {"v": 1})
        at.record_access("coll", "id2", {"v": 2})

        stats = at.flush_now("test")

        assert stats["errors"] >= 1
        assert stats["dropped"] == 2

    def test_empty_flush_noop(self, monkeypatch):
        """Flushing with nothing pending is a no-op."""
        import modules.access_tracking as at
        from unittest.mock import MagicMock

        mock_qdrant = MagicMock()
        monkeypatch.setattr("modules.config.qdrant", mock_qdrant)

        stats = at.flush_now("test")

        mock_qdrant.batch_update_points.assert_not_called()
        assert stats["flush_count"] == 0


class TestSpreading:
    """record_spreading enqueues bulk updates."""

    def test_spreading_enqueues_all(self, monkeypatch):
        """All spreading updates are enqueued with correct payloads."""
        import modules.access_tracking as at
        from unittest.mock import MagicMock

        mock_qdrant = MagicMock()
        monkeypatch.setattr("modules.config.qdrant", mock_qdrant)

        updates = {"id1": 0.7, "id2": 0.8, "id3": 0.55}
        at.record_spreading("codi_memories", updates, last_accessed="2026-01-01T00:00")

        assert at._stats["enqueued"] == 3

        stats = at.flush_now("test")

        assert mock_qdrant.batch_update_points.call_count == 1
        call = mock_qdrant.batch_update_points.call_args
        ops = call.kwargs["update_operations"]
        assert len(ops) == 3

        # Verify payload structure
        payloads = {}
        for op in ops:
            sp = op.set_payload
            pid = sp.points[0]
            payloads[pid] = sp.payload

        assert payloads["id1"]["attention_salience"] == 0.7
        assert payloads["id2"]["attention_salience"] == 0.8
        assert payloads["id3"]["attention_salience"] == 0.55
        for p in payloads.values():
            assert p["attention_last_accessed"] == "2026-01-01T00:00"


class TestLegacyMode:
    """Legacy mode bypasses aggregator."""

    def test_legacy_calls_set_payload_directly(self, monkeypatch):
        """In legacy mode, record_access calls qdrant.set_payload directly."""
        import modules.access_tracking as at
        from unittest.mock import MagicMock

        monkeypatch.setattr(at, "ACCESS_TRACKING_MODE", "legacy")
        mock_qdrant = MagicMock()
        monkeypatch.setattr("modules.config.qdrant", mock_qdrant)

        at.record_access("coll", "id1", {"count": 1})

        mock_qdrant.set_payload.assert_called_once_with(
            collection_name="coll",
            payload={"count": 1},
            points=["id1"],
        )
        # Aggregator should be empty
        with at._lock:
            assert len(at._pending) == 0
