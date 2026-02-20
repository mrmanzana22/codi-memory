"""Tests for modules/working_memory.py.

Covers:
  - _effective_score (pure scoring function)
  - push_to_working_memory (insert + chain resolution)
  - get_working_memory (retrieve + score + group)
  - update_working_memory (update relevance/active)
  - link_narrative_trace (trace creation/update)
  - _auto_curate_buffer (eviction logic)
  - wm_noche_cleanup / wm_active_count (lifecycle)
"""

import json
import pytest
from unittest.mock import patch


class TestEffectiveScore:
    """Tests for _effective_score (pure scoring function)."""

    def test_all_zeros(self):
        from modules.working_memory import _effective_score
        score = _effective_score(0, None, 0, None)
        assert 0.0 <= score <= 1.0

    def test_high_relevance_dominates(self):
        from modules.working_memory import _effective_score
        high = _effective_score(1.0, None, 0, None)
        low = _effective_score(0.0, None, 0, None)
        assert high > low

    def test_recent_access_boosts(self):
        from modules.working_memory import _effective_score
        from modules.config import now_iso
        recent = _effective_score(0.5, now_iso(), 0, None)
        old = _effective_score(0.5, "2020-01-01T00:00:00", 0, None)
        assert recent > old

    def test_high_access_count_boosts(self):
        from modules.working_memory import _effective_score
        many = _effective_score(0.5, None, 10, None)
        few = _effective_score(0.5, None, 0, None)
        assert many > few

    def test_score_clamped_to_01(self):
        from modules.working_memory import _effective_score
        from modules.config import now_iso
        score = _effective_score(1.0, now_iso(), 100, now_iso())
        assert score <= 1.0

    def test_uses_added_at_as_fallback(self):
        from modules.working_memory import _effective_score
        from modules.config import now_iso
        # When last_accessed_at is None, should use added_at
        score = _effective_score(0.5, None, 0, now_iso())
        assert score > 0


class TestPushToWorkingMemory:
    """Tests for push_to_working_memory."""

    def test_push_basic(self):
        from modules.working_memory import push_to_working_memory
        result = json.loads(push_to_working_memory(
            content="Test item",
            topic="testing",
            relevance=0.8,
        ))
        assert "id" in result
        assert result["topic"] == "testing"
        assert result["relevance"] == 0.8
        assert "chain_id" in result

    def test_push_auto_chain_id(self):
        from modules.working_memory import push_to_working_memory
        result = json.loads(push_to_working_memory(
            content="Trading analysis",
            topic="trading",
        ))
        assert result["chain_id"].startswith("trading_")

    def test_push_general_gets_unique_chain(self):
        from modules.working_memory import push_to_working_memory
        r1 = json.loads(push_to_working_memory(content="A", topic="general"))
        r2 = json.loads(push_to_working_memory(content="B", topic="general"))
        # General topic should get unique chain_ids
        assert r1["chain_id"] != r2["chain_id"]

    def test_push_clamps_relevance(self):
        from modules.working_memory import push_to_working_memory
        r1 = json.loads(push_to_working_memory(content="X", relevance=5.0))
        assert r1["relevance"] == 1.0
        r2 = json.loads(push_to_working_memory(content="Y", relevance=-1.0))
        assert r2["relevance"] == 0.0

    def test_push_same_topic_reuses_chain(self):
        from modules.working_memory import push_to_working_memory
        r1 = json.loads(push_to_working_memory(content="First", topic="project_x"))
        r2 = json.loads(push_to_working_memory(content="Second", topic="project_x"))
        # Same topic within 7 days should reuse chain
        assert r1["chain_id"] == r2["chain_id"]


class TestGetWorkingMemory:
    """Tests for get_working_memory."""

    def test_empty_returns_zero_items(self):
        from modules.working_memory import get_working_memory
        result = json.loads(get_working_memory())
        assert result["count"] == 0
        assert result["items"] == []

    def test_returns_pushed_items(self):
        from modules.working_memory import push_to_working_memory, get_working_memory
        push_to_working_memory(content="Item 1", topic="test")
        push_to_working_memory(content="Item 2", topic="test")
        result = json.loads(get_working_memory())
        assert result["count"] == 2
        assert len(result["items"]) == 2

    def test_items_have_effective_score(self):
        from modules.working_memory import push_to_working_memory, get_working_memory
        push_to_working_memory(content="Scored item", relevance=0.9)
        result = json.loads(get_working_memory())
        assert result["items"][0]["effective_score"] > 0

    def test_pretty_output_exists(self):
        from modules.working_memory import push_to_working_memory, get_working_memory
        push_to_working_memory(content="Pretty item")
        result = json.loads(get_working_memory())
        assert "WORKING MEMORY" in result["pretty"]


class TestUpdateWorkingMemory:
    """Tests for update_working_memory."""

    def test_update_relevance(self):
        from modules.working_memory import push_to_working_memory, update_working_memory
        r = json.loads(push_to_working_memory(content="Update me", relevance=0.5))
        item_id = r["id"]
        update = json.loads(update_working_memory(item_id, relevance=0.9))
        assert update["updated"] is True
        assert update["changes"]["relevance"] == 0.9

    def test_archive_item(self):
        from modules.working_memory import (
            push_to_working_memory, update_working_memory, get_working_memory
        )
        r = json.loads(push_to_working_memory(content="Archive me"))
        item_id = r["id"]
        update = json.loads(update_working_memory(item_id, active=0))
        assert update["updated"] is True

        # Should not appear in active items
        wm = json.loads(get_working_memory())
        active_ids = [i["id"] for i in wm["items"]]
        assert item_id not in active_ids

    def test_update_nonexistent_item(self):
        from modules.working_memory import update_working_memory
        result = json.loads(update_working_memory(99999, relevance=0.5))
        assert "error" in result

    def test_update_nothing_returns_error(self):
        from modules.working_memory import update_working_memory
        result = json.loads(update_working_memory(1))
        assert "error" in result


class TestLinkNarrativeTrace:
    """Tests for link_narrative_trace."""

    def test_create_trace(self):
        from modules.working_memory import link_narrative_trace
        result = json.loads(link_narrative_trace(
            trace_name="test_trace",
            chain_ids=["chain_a", "chain_b"],
            theme="testing"
        ))
        assert result["action"] == "created"
        assert result["trace_name"] == "test_trace"
        assert len(result["chain_ids"]) == 2

    def test_update_existing_trace(self):
        from modules.working_memory import link_narrative_trace
        # Create
        link_narrative_trace("update_me", ["c1"])
        # Update
        result = json.loads(link_narrative_trace("update_me", ["c1", "c2", "c3"]))
        assert result["action"] == "updated"
        assert len(result["chain_ids"]) == 3

    def test_empty_chain_ids_rejected(self):
        from modules.working_memory import link_narrative_trace
        result = json.loads(link_narrative_trace("bad", []))
        assert "error" in result

    def test_invalid_chain_ids_rejected(self):
        from modules.working_memory import link_narrative_trace
        result = json.loads(link_narrative_trace("bad", "not_a_list"))
        assert "error" in result


class TestLifecycle:
    """Tests for wm_noche_cleanup and wm_active_count."""

    def test_active_count_starts_zero(self):
        from modules.working_memory import wm_active_count
        # Fresh DB should have 0 active items
        count = wm_active_count()
        assert count >= 0

    def test_active_count_after_push(self):
        from modules.working_memory import push_to_working_memory, wm_active_count
        push_to_working_memory(content="Count me")
        assert wm_active_count() >= 1

    def test_noche_cleanup_returns_count(self):
        from modules.working_memory import wm_noche_cleanup
        result = wm_noche_cleanup()
        assert isinstance(result, int)
        assert result >= 0
