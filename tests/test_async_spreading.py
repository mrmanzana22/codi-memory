"""Tests for PR-P1b: Async spreading activation (fire-and-forget)."""

import sys
import os
import time
import threading
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestAsyncSpreading:
    """Test that spreading activation runs in background, not blocking recall."""

    def test_spreading_non_blocking(self, monkeypatch):
        """MEMORY_RETRIEVED handler should return immediately, not wait for spreading."""
        import modules.wiring as wiring

        # Mock recurrent_cycle to sleep (simulate slow BFS)
        slow_called = threading.Event()

        def slow_recurrent_cycle(*args, **kwargs):
            slow_called.set()
            time.sleep(2.0)  # Simulate slow Qdrant calls

        monkeypatch.setattr("modules.spreading.recurrent_cycle", slow_recurrent_cycle)

        # Ensure semaphore is available
        monkeypatch.setattr(wiring, "_SPREADING_SEMAPHORE", threading.BoundedSemaphore(2))

        data = {
            "retrieved_ids": ["id1", "id2", "id3"],
            "result_count": 3,
            "query": "test query",
        }

        t0 = time.perf_counter()
        wiring._on_memory_retrieved("MEMORY_RETRIEVED", data)
        elapsed = time.perf_counter() - t0

        # Handler should return in <200ms (not 2s)
        assert elapsed < 0.2, f"Handler took {elapsed:.2f}s, expected <0.2s"

        # Wait a bit for the bg thread to start
        assert slow_called.wait(timeout=1.0), "Background spreading thread never started"

    def test_spreading_semaphore_skip(self, monkeypatch):
        """When semaphore is full, spreading should be skipped (not queued)."""
        import modules.wiring as wiring

        called = threading.Event()

        def mock_recurrent_cycle(*args, **kwargs):
            called.set()

        monkeypatch.setattr("modules.spreading.recurrent_cycle", mock_recurrent_cycle)

        # Create a semaphore with capacity 1 and acquire it (full)
        sem = threading.BoundedSemaphore(1)
        sem.acquire()  # Now full
        monkeypatch.setattr(wiring, "_SPREADING_SEMAPHORE", sem)

        data = {
            "retrieved_ids": ["id1"],
            "result_count": 1,
            "query": "test",
        }

        wiring._on_memory_retrieved("MEMORY_RETRIEVED", data)
        time.sleep(0.1)

        # recurrent_cycle should NOT have been called
        assert not called.is_set(), "Spreading should have been skipped when semaphore full"

        # Cleanup
        sem.release()

    def test_spreading_eventually_runs(self, monkeypatch):
        """Spreading should actually execute in the background thread."""
        import modules.wiring as wiring

        call_args = {}

        def mock_recurrent_cycle(seed_ids, **kwargs):
            call_args["seed_ids"] = seed_ids
            call_args["kwargs"] = kwargs

        monkeypatch.setattr("modules.spreading.recurrent_cycle", mock_recurrent_cycle)
        monkeypatch.setattr(wiring, "_SPREADING_SEMAPHORE", threading.BoundedSemaphore(2))

        data = {
            "retrieved_ids": ["id1", "id2", "id3", "id4"],
            "result_count": 4,
            "query": "test",
        }

        wiring._on_memory_retrieved("MEMORY_RETRIEVED", data)
        time.sleep(0.3)  # Wait for bg thread

        assert "seed_ids" in call_args, "recurrent_cycle was never called"
        assert call_args["seed_ids"] == ["id1", "id2", "id3"]  # Only top 3

    def test_attention_schema_still_synchronous(self, monkeypatch):
        """Attention schema update should still happen synchronously."""
        import modules.wiring as wiring

        schema_updated = False

        original_update = wiring._update_attention_schema

        def mock_update(*args, **kwargs):
            nonlocal schema_updated
            schema_updated = True

        monkeypatch.setattr(wiring, "_update_attention_schema", mock_update)
        monkeypatch.setattr("modules.spreading.recurrent_cycle", lambda *a, **kw: None)
        monkeypatch.setattr(wiring, "_SPREADING_SEMAPHORE", threading.BoundedSemaphore(2))

        data = {
            "retrieved_ids": ["id1"],
            "result_count": 1,
            "query": "important topic",
        }

        wiring._on_memory_retrieved("MEMORY_RETRIEVED", data)

        # Attention schema update should have happened synchronously
        assert schema_updated, "Attention schema was not updated synchronously"

    def test_no_spreading_on_empty_results(self, monkeypatch):
        """No spreading should fire when result_count is 0."""
        import modules.wiring as wiring

        called = threading.Event()

        def mock_recurrent_cycle(*args, **kwargs):
            called.set()

        monkeypatch.setattr("modules.spreading.recurrent_cycle", mock_recurrent_cycle)
        monkeypatch.setattr(wiring, "_SPREADING_SEMAPHORE", threading.BoundedSemaphore(2))

        data = {
            "retrieved_ids": [],
            "result_count": 0,
            "query": "test",
        }

        wiring._on_memory_retrieved("MEMORY_RETRIEVED", data)
        time.sleep(0.1)

        assert not called.is_set(), "Spreading should not fire on empty results"
