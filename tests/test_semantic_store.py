"""Tests for modules/semantic_store.py.

Covers:
  - get_semantic_facts (mocked qdrant)
  - get_consolidation_stats (SQLite + mocked qdrant)
  - count_unconsolidated_episodic (mocked qdrant)
"""

import pytest
from unittest.mock import patch, MagicMock
from tests.conftest import FakeQdrantPoint


class TestGetSemanticFacts:
    """Tests for get_semantic_facts."""

    def test_empty_collection(self, patch_externals):
        from modules.semantic_store import get_semantic_facts
        result = get_semantic_facts()
        assert "empty" in result.lower() or "0 facts" in result

    def test_with_facts(self, patch_externals):
        from modules.semantic_store import get_semantic_facts
        fake_qdrant = patch_externals["qdrant"]

        # Mock get_collection to return non-zero count
        class FakeInfo:
            points_count = 5

        fake_qdrant.get_collection = lambda name: FakeInfo()

        points = [
            FakeQdrantPoint("f1", {
                'fact_text': 'Python is used for scripting',
                'topic': 'programming',
                'confidence': 0.9,
                'evidence_count': 3,
            }),
        ]

        original_scroll = fake_qdrant.scroll
        def mock_scroll(collection_name, scroll_filter=None, **kwargs):
            return (points, None)

        fake_qdrant.scroll = mock_scroll

        result = get_semantic_facts()
        assert "Semantic Facts" in result
        assert "Python" in result

    def test_filter_by_topic(self, patch_externals):
        from modules.semantic_store import get_semantic_facts
        fake_qdrant = patch_externals["qdrant"]

        class FakeInfo:
            points_count = 10

        fake_qdrant.get_collection = lambda name: FakeInfo()
        fake_qdrant.scroll = lambda **kw: ([], None)

        result = get_semantic_facts(topic="trading")
        assert "trading" in result


class TestGetConsolidationStats:
    """Tests for get_consolidation_stats."""

    def test_returns_stats(self, patch_externals):
        from modules.semantic_store import get_consolidation_stats
        result = get_consolidation_stats()
        assert "Consolidation Stats" in result
        assert "Total runs" in result
        assert "labile" in result.lower()

    def test_shows_zero_counts_fresh_db(self, patch_externals):
        from modules.semantic_store import get_consolidation_stats
        result = get_consolidation_stats()
        assert "Total runs: 0" in result


class TestCountUnconsolidatedEpisodic:
    """Tests for count_unconsolidated_episodic."""

    def test_empty_returns_zero(self, patch_externals):
        from modules.semantic_store import count_unconsolidated_episodic
        count = count_unconsolidated_episodic()
        assert count == 0
        assert isinstance(count, int)
