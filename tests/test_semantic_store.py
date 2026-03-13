"""Tests for modules/semantic_store.py.

Covers:
  - get_semantic_facts (mocked pg_store)
  - get_consolidation_stats (mocked pg_store + get_conn)
  - count_unconsolidated_episodic (mocked get_conn)

Backend: pg_store (PostgreSQL + pgvector).
Migrated from Qdrant/mem0 mocks in Fase 2.
"""

import pytest
from unittest.mock import patch, MagicMock
from modules.pg_store import Point, CollectionInfo


def _make_point(pid, payload, score=0.0):
    """Helper to create a Point object compatible with pg_store."""
    return Point(id=pid, payload=payload, score=score)


class TestGetSemanticFacts:
    """Tests for get_semantic_facts."""

    @patch("modules.semantic_store.pg")
    def test_empty_collection(self, mock_pg):
        mock_pg.count.return_value = CollectionInfo(points_count=0)

        from modules.semantic_store import get_semantic_facts
        result = get_semantic_facts()
        assert "empty" in result.lower() or "0 facts" in result

    @patch("modules.semantic_store.pg")
    def test_with_facts(self, mock_pg):
        mock_pg.count.return_value = CollectionInfo(points_count=5)

        points = [
            _make_point("f1", {
                'memory': 'Python is used for scripting',
                'fact_text': 'Python is used for scripting',
                'category': 'programming',
                'confidence': 0.9,
                'evidence_count': 3,
            }),
        ]
        mock_pg.scroll.return_value = (points, None)

        from modules.semantic_store import get_semantic_facts
        result = get_semantic_facts()
        assert "Semantic Facts" in result
        assert "Python" in result

    @patch("modules.semantic_store.pg")
    def test_filter_by_topic(self, mock_pg):
        mock_pg.count.return_value = CollectionInfo(points_count=10)
        mock_pg.scroll.return_value = ([], None)

        from modules.semantic_store import get_semantic_facts
        result = get_semantic_facts(topic="trading")
        assert "trading" in result


class TestGetConsolidationStats:
    """Tests for get_consolidation_stats."""

    @patch("modules.semantic_store.pg")
    @patch("modules.semantic_store.get_conn")
    def test_returns_stats(self, mock_get_conn, mock_pg):
        # Mock pg.count for semantic store size
        mock_pg.count.return_value = CollectionInfo(points_count=0)

        # Mock get_conn context manager with a fake connection
        mock_conn = MagicMock()
        mock_get_conn.return_value.__enter__ = MagicMock(return_value=mock_conn)
        mock_get_conn.return_value.__exit__ = MagicMock(return_value=False)

        # Each conn.execute().fetchone() returns (0,) for the counts
        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = (0,)
        mock_conn.execute.return_value = mock_cursor

        from modules.semantic_store import get_consolidation_stats
        result = get_consolidation_stats()
        assert "Consolidation Stats" in result
        assert "Total runs" in result
        assert "labile" in result.lower()

    @patch("modules.semantic_store.pg")
    @patch("modules.semantic_store.get_conn")
    def test_shows_zero_counts_fresh_db(self, mock_get_conn, mock_pg):
        # Mock pg.count for semantic store size
        mock_pg.count.return_value = CollectionInfo(points_count=0)

        # Mock get_conn context manager
        mock_conn = MagicMock()
        mock_get_conn.return_value.__enter__ = MagicMock(return_value=mock_conn)
        mock_get_conn.return_value.__exit__ = MagicMock(return_value=False)

        # All SQL queries return 0
        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = (0,)
        mock_conn.execute.return_value = mock_cursor

        from modules.semantic_store import get_consolidation_stats
        result = get_consolidation_stats()
        assert "Total runs: 0" in result


class TestCountUnconsolidatedEpisodic:
    """Tests for count_unconsolidated_episodic."""

    @patch("modules.semantic_store.get_conn")
    def test_empty_returns_zero(self, mock_get_conn):
        # Mock get_conn context manager
        mock_conn = MagicMock()
        mock_get_conn.return_value.__enter__ = MagicMock(return_value=mock_conn)
        mock_get_conn.return_value.__exit__ = MagicMock(return_value=False)

        # Query returns 0
        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = (0,)
        mock_conn.execute.return_value = mock_cursor

        from modules.semantic_store import count_unconsolidated_episodic
        count = count_unconsolidated_episodic()
        assert count == 0
        assert isinstance(count, int)
