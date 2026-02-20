"""Tests for modules/consolidation_common.py.

Covers:
  - _cosine_similarity (pure math)
  - get_embed_cache_info (cache stats)
  - _consolidation_conn (DB connection)
  - init_consolidation_db (table validation)
"""

import math
import pytest
from unittest.mock import patch, MagicMock


class TestCosineSimilarity:
    """Tests for _cosine_similarity (pure math function)."""

    def test_identical_vectors(self):
        from modules.consolidation_common import _cosine_similarity
        assert _cosine_similarity([1, 0, 0], [1, 0, 0]) == pytest.approx(1.0)

    def test_orthogonal_vectors(self):
        from modules.consolidation_common import _cosine_similarity
        assert _cosine_similarity([1, 0, 0], [0, 1, 0]) == pytest.approx(0.0)

    def test_opposite_vectors(self):
        from modules.consolidation_common import _cosine_similarity
        assert _cosine_similarity([1, 0, 0], [-1, 0, 0]) == pytest.approx(-1.0)

    def test_zero_vector_returns_zero(self):
        from modules.consolidation_common import _cosine_similarity
        assert _cosine_similarity([0, 0, 0], [1, 2, 3]) == 0.0

    def test_similar_vectors(self):
        from modules.consolidation_common import _cosine_similarity
        # (1,1,0) and (1,0,0) have cos = 1/sqrt(2) ≈ 0.707
        result = _cosine_similarity([1, 1, 0], [1, 0, 0])
        assert result == pytest.approx(1 / math.sqrt(2), abs=0.001)

    def test_high_dimensional(self):
        from modules.consolidation_common import _cosine_similarity
        a = [1.0] * 1536  # same dimension as text-embedding-3-small
        b = [1.0] * 1536
        assert _cosine_similarity(a, b) == pytest.approx(1.0)


class TestEmbedCacheInfo:
    """Tests for get_embed_cache_info."""

    def test_returns_expected_keys(self):
        from modules.consolidation_common import get_embed_cache_info
        info = get_embed_cache_info()
        assert "hits" in info
        assert "misses" in info
        assert "current_size" in info
        assert "max_size" in info
        assert "hit_rate" in info

    def test_max_size_is_256(self):
        from modules.consolidation_common import get_embed_cache_info
        info = get_embed_cache_info()
        assert info["max_size"] == 256


class TestConsolidationConn:
    """Tests for _consolidation_conn (DB connection)."""

    def test_returns_connection(self):
        from modules.consolidation_common import _consolidation_conn
        conn = _consolidation_conn()
        assert conn is not None
        # Should be able to execute queries
        row = conn.execute("SELECT 1").fetchone()
        assert row[0] == 1
        conn.close()

    def test_consolidation_tables_exist(self):
        from modules.consolidation_common import _consolidation_conn
        conn = _consolidation_conn()
        tables = [r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()]
        conn.close()
        assert "consolidation_log" in tables
        assert "reconsolidation_log" in tables
        assert "labile_memories" in tables
