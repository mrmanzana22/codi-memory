#!/usr/bin/env python3
"""
Unit tests for the Metamemory Layer (WIRING-7).
Run: python3 -m pytest tests/test_metamemory.py -v
"""

import sys
import os
import sqlite3
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from modules.retrieval_metadata import (
    RetrievalResult,
    wrap_retrieval_result,
    get_retrieval_buffer,
    _retrieval_buffer,
    _classify_coverage,
    _compute_confidence,
    feeling_of_knowing,
    init_failed_searches_table,
    log_failed_search,
    get_top_failed_topics,
)


# Helper to build fake merged results
def _fake_merged(count, memory_type="episodic", activation=0.5):
    return [
        {"memory_type": memory_type, "activation": activation, "combined_score": activation}
        for _ in range(count)
    ]


# ============================================================
# COVERAGE CLASSIFICATION
# ============================================================

class TestCoverageClassification:
    def test_empty(self):
        assert _classify_coverage(0, 0.0) == "empty"

    def test_sparse(self):
        assert _classify_coverage(1, 0.9) == "sparse"
        assert _classify_coverage(2, 0.9) == "sparse"

    def test_partial(self):
        assert _classify_coverage(3, 0.3) == "partial"
        assert _classify_coverage(4, 0.4) == "partial"

    def test_comprehensive_by_count(self):
        assert _classify_coverage(5, 0.1) == "comprehensive"
        assert _classify_coverage(10, 0.0) == "comprehensive"

    def test_comprehensive_by_activation(self):
        assert _classify_coverage(3, 0.5) == "comprehensive"
        assert _classify_coverage(4, 0.8) == "comprehensive"


# ============================================================
# CONFIDENCE COMPUTATION
# ============================================================

class TestConfidence:
    def test_zero_results(self):
        assert _compute_confidence(0, 0.0) == 0.0

    def test_full_count_full_activation(self):
        # min(1.0, 5/5) * 0.6 + 1.0 * 0.4 = 0.6 + 0.4 = 1.0
        assert _compute_confidence(5, 1.0) == pytest.approx(1.0)

    def test_partial(self):
        # min(1.0, 3/5) * 0.6 + 0.5 * 0.4 = 0.36 + 0.20 = 0.56
        assert _compute_confidence(3, 0.5) == pytest.approx(0.56)

    def test_count_capped_at_five(self):
        # min(1.0, 100/5) * 0.6 + 0.8 * 0.4 = 0.6 + 0.32 = 0.92
        assert _compute_confidence(100, 0.8) == pytest.approx(0.92)


# ============================================================
# RETRIEVAL WRAPPING
# ============================================================

class TestRetrievalWrapping:
    def setup_method(self):
        _retrieval_buffer.clear()

    def test_wrap_basic(self):
        merged = _fake_merged(3, activation=0.6)
        result = wrap_retrieval_result("test query", merged)
        assert isinstance(result, RetrievalResult)
        assert result.query == "test query"
        assert result.result_count == 3
        assert result.episodic_count == 3
        assert result.semantic_count == 0
        assert result.top_activation == pytest.approx(0.6)
        assert result.mean_activation == pytest.approx(0.6)

    def test_wrap_empty(self):
        result = wrap_retrieval_result("nothing", [])
        assert result.coverage == "empty"
        assert result.confidence_estimate == 0.0
        assert result.result_count == 0

    def test_wrap_mixed_types(self):
        merged = _fake_merged(2, "episodic", 0.7) + _fake_merged(3, "semantic", 0.4)
        result = wrap_retrieval_result("mixed", merged)
        assert result.episodic_count == 2
        assert result.semantic_count == 3
        assert result.result_count == 5
        assert result.top_activation == pytest.approx(0.7)

    def test_stored_in_buffer(self):
        wrap_retrieval_result("q1", _fake_merged(1))
        wrap_retrieval_result("q2", _fake_merged(2))
        buf = get_retrieval_buffer()
        assert len(buf) == 2
        assert buf[0].query == "q1"
        assert buf[1].query == "q2"

    def test_buffer_max_size(self):
        for i in range(25):
            wrap_retrieval_result(f"q{i}", _fake_merged(1))
        assert len(get_retrieval_buffer()) == 20


# ============================================================
# FEELING OF KNOWING (FOK)
# ============================================================

class TestFeelingOfKnowing:
    def test_fok_no_history(self):
        """No failed searches, no WM, no buffer = base 0.5"""
        _retrieval_buffer.clear()
        result = feeling_of_knowing("some random query")
        assert result["fok_score"] == 0.5
        assert result["recommendation"] == "uncertain"

    def test_fok_after_failures(self):
        """Failed searches should reduce FOK."""
        _retrieval_buffer.clear()
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        from modules.migrations import apply_migrations
        apply_migrations(db_path, migrations_dir=os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "migrations"))
        try:
            conn = sqlite3.connect(db_path)
            init_failed_searches_table(conn)
            log_failed_search(conn, "kubernetes deployment", 0, 0.0, "kubernetes")
            log_failed_search(conn, "kubernetes pods", 1, 0.2, "kubernetes")
            log_failed_search(conn, "kubernetes service", 0, 0.0, "kubernetes")
            conn.close()

            result = feeling_of_knowing("kubernetes", fts_db_path=db_path)
            assert result["fok_score"] < 0.5
            assert "failed_searches" in result["basis"]
        finally:
            os.unlink(db_path)

    def test_fok_wm_boost(self):
        """Topic in working memory should boost FOK."""
        _retrieval_buffer.clear()
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        try:
            conn = sqlite3.connect(db_path)
            conn.execute("CREATE TABLE working_memory (id INTEGER PRIMARY KEY, content TEXT, relevance REAL)")
            conn.execute("INSERT INTO working_memory VALUES (1, 'trading strategy analysis', 0.8)")
            conn.commit()

            result = feeling_of_knowing("trading", wm_conn=conn)
            assert result["fok_score"] > 0.5
            assert "in_wm" in result["basis"]
            conn.close()
        finally:
            os.unlink(db_path)

    def test_fok_buffer_boost(self):
        """Successful past queries with similar words should boost FOK."""
        _retrieval_buffer.clear()
        # Add a successful past retrieval
        wrap_retrieval_result("fullempaques production", _fake_merged(5, activation=0.7))

        result = feeling_of_knowing("fullempaques prices")
        assert result["fok_score"] > 0.5
        assert "buffer_hits" in result["basis"]

    def test_fok_clamped(self):
        """FOK score should always be 0-1."""
        _retrieval_buffer.clear()
        result = feeling_of_knowing("x")
        assert 0.0 <= result["fok_score"] <= 1.0


# ============================================================
# FAILED SEARCH LOGGING
# ============================================================

class TestFailedSearchLogging:
    def _setup_db(self):
        """Create temp DB with migrations applied."""
        f = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        db_path = f.name
        f.close()
        from modules.migrations import apply_migrations
        apply_migrations(db_path, migrations_dir=os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "migrations"))
        return db_path

    def test_log_and_retrieve(self):
        db_path = self._setup_db()
        try:
            conn = sqlite3.connect(db_path)
            init_failed_searches_table(conn)
            log_failed_search(conn, "kubernetes", 0, 0.0, "devops")
            log_failed_search(conn, "docker compose", 1, 0.1, "devops")
            log_failed_search(conn, "react hooks", 0, 0.0, "frontend")

            tops = get_top_failed_topics(conn)
            assert len(tops) >= 2
            # devops should be first (2 failures)
            assert tops[0][0] == "devops"
            assert tops[0][1] == 2
            conn.close()
        finally:
            os.unlink(db_path)

    def test_fifo_cleanup(self):
        """Should keep max 500 rows."""
        db_path = self._setup_db()
        try:
            conn = sqlite3.connect(db_path)
            init_failed_searches_table(conn)
            # Insert 510 rows
            for i in range(510):
                conn.execute(
                    "INSERT INTO failed_searches (query, result_count, top_activation, topic, created_at) VALUES (?, 0, 0.0, 'test', ?)",
                    (f"query_{i}", f"2026-01-01T{i:05d}")
                )
            conn.commit()
            # Run cleanup via log_failed_search
            log_failed_search(conn, "trigger_cleanup", 0, 0.0, "test")

            cursor = conn.execute("SELECT COUNT(*) FROM failed_searches")
            count = cursor.fetchone()[0]
            assert count <= 500
            conn.close()
        finally:
            os.unlink(db_path)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
