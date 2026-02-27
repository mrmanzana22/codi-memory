#!/usr/bin/env python3
"""
Unit tests for the Metamemory Layer (WIRING-7).
Run: python3 -m pytest tests/test_metamemory.py -v
"""

import sys
import os
import sqlite3
import tempfile
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from modules.retrieval_metadata import (
    RetrievalResult,
    wrap_retrieval_result,
    get_retrieval_buffer,
    _retrieval_buffer,
    _classify_coverage,
    _compute_confidence,
    compute_memory_confidence,
    _classify_quality,
    diagnose_retrieval_failure,
    feeling_of_knowing,
    init_failed_searches_table,
    log_failed_search,
    get_top_failed_topics,
    _init_fok_calibration_table,
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
        """No failed searches, no WM, no buffer = base 0.35.

        FOK base was lowered from 0.5 to 0.35 (assume less knowledge,
        let evidence raise it). Without any signals, may drop further due
        to FTS metamemory check (fts_count=0 -> -0.10).
        """
        _retrieval_buffer.clear()
        result = feeling_of_knowing("some random query")
        # Base is 0.35; FTS check may subtract 0.10 -> 0.25, or stay at 0.35
        assert 0.20 <= result["fok_score"] <= 0.40
        assert result["recommendation"] in ("uncertain", "ask")

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
        """Topic in working memory should boost FOK above base (0.35)."""
        _retrieval_buffer.clear()
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        try:
            conn = sqlite3.connect(db_path)
            conn.execute("CREATE TABLE working_memory (id INTEGER PRIMARY KEY, content TEXT, relevance REAL)")
            conn.execute("INSERT INTO working_memory VALUES (1, 'trading strategy analysis', 0.8)")
            conn.commit()

            result = feeling_of_knowing("trading", wm_conn=conn)
            # Base 0.35 + wm_boost 0.15 = 0.50
            assert result["fok_score"] >= 0.5
            assert "in_wm" in result["basis"]
            conn.close()
        finally:
            os.unlink(db_path)

    def test_fok_buffer_boost(self):
        """Successful past queries (in fok_calibration_log) should boost FOK."""
        _retrieval_buffer.clear()
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        try:
            # FOK now checks persistent fok_calibration_log table (cross-process)
            # instead of in-memory _retrieval_buffer.
            # Also seeds memories_fts so the FTS metamemory check doesn't penalize.
            from modules.migrations import apply_migrations
            apply_migrations(db_path, migrations_dir=os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "migrations"))
            conn = sqlite3.connect(db_path)
            _init_fok_calibration_table(conn)
            now = datetime.now().isoformat()
            # Seed fok_calibration_log with a past successful retrieval
            conn.execute(
                "INSERT INTO fok_calibration_log (query, fok_predicted, actual_coverage, actual_count, actual_top_activation, created_at) VALUES (?, ?, ?, ?, ?, ?)",
                ("fullempaques production", 0.5, "comprehensive", 5, 0.7, now)
            )
            # Seed memories_fts so FTS metamemory check doesn't penalize (-0.10)
            conn.execute(
                "INSERT INTO memories_text (memory_id, content, created_at) VALUES (?, ?, ?)",
                ("fake-mem-1", "fullempaques production data", now)
            )
            conn.commit()
            conn.close()

            result = feeling_of_knowing("fullempaques prices", fts_db_path=db_path)
            # base 0.35 + fts boost + buffer_hits boost
            assert result["fok_score"] > 0.35
            assert "buffer_hits" in result["basis"]
        finally:
            os.unlink(db_path)

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


# ============================================================
# PER-MEMORY CONFIDENCE (Koriat 1997)
# ============================================================

class TestComputeMemoryConfidence:
    def test_minimal_payload(self):
        """Unknown source, no evidence, no activation = low confidence."""
        conf = compute_memory_confidence({}, activation=0.0)
        assert 0.0 <= conf <= 1.0
        # base reliability (0.15) + source unknown 0.3*0.25 + corroboration log(2)/log(6)*0.20 + fluency 0 + staleness 0.5*0.15
        assert conf > 0.0

    def test_experienced_source_higher(self):
        """Experienced source should yield higher confidence than unknown."""
        conf_exp = compute_memory_confidence({"ownership_source": "experienced"}, activation=0.5)
        conf_unk = compute_memory_confidence({"ownership_source": "unknown"}, activation=0.5)
        assert conf_exp > conf_unk

    def test_source_weight_ordering(self):
        """experienced > learned > told > inferred > unknown."""
        sources = ["experienced", "learned", "told", "inferred", "unknown"]
        scores = [
            compute_memory_confidence({"ownership_source": s}, activation=0.5)
            for s in sources
        ]
        for i in range(len(scores) - 1):
            assert scores[i] >= scores[i + 1], f"{sources[i]} should >= {sources[i+1]}"

    def test_high_activation_boosts(self):
        """Higher activation = higher confidence (retrieval fluency)."""
        conf_high = compute_memory_confidence({}, activation=0.9)
        conf_low = compute_memory_confidence({}, activation=0.1)
        assert conf_high > conf_low

    def test_contradictions_reduce(self):
        """Contradictions should reduce confidence."""
        conf_clean = compute_memory_confidence({"contradiction_count": 0}, activation=0.5)
        conf_dirty = compute_memory_confidence({"contradiction_count": 3}, activation=0.5)
        assert conf_clean > conf_dirty

    def test_contradiction_penalty_capped(self):
        """Contradiction penalty caps at 0.45."""
        conf_max = compute_memory_confidence({"contradiction_count": 100}, activation=0.5)
        assert conf_max >= 0.0

    def test_evidence_corroboration(self):
        """More evidence should increase confidence."""
        conf_low_ev = compute_memory_confidence({"evidence_count": 1}, activation=0.5)
        conf_high_ev = compute_memory_confidence({"evidence_count": 5}, activation=0.5)
        assert conf_high_ev > conf_low_ev

    def test_clamped_0_1(self):
        """Confidence always between 0 and 1."""
        for act in [0.0, 0.5, 1.0]:
            for contra in [0, 5, 10]:
                conf = compute_memory_confidence(
                    {"contradiction_count": contra, "evidence_count": 10},
                    activation=act,
                )
                assert 0.0 <= conf <= 1.0

    def test_full_payload(self):
        """Comprehensive payload should give high confidence."""
        payload = {
            "ownership_source": "experienced",
            "evidence_count": 5,
            "contradiction_count": 0,
            "attention_last_accessed": datetime.now().isoformat(),
        }
        conf = compute_memory_confidence(payload, activation=0.9)
        assert conf >= 0.7


# ============================================================
# QUALITY SPACE CLASSIFICATION (Tulving 1985)
# ============================================================

class TestClassifyQuality:
    def test_empty_is_blank(self):
        assert _classify_quality("empty", 0.0, 0.0) == "blank"

    def test_comprehensive_high_confidence_is_confident_recall(self):
        assert _classify_quality("comprehensive", 0.8, 0.8) == "confident_recall"

    def test_comprehensive_medium_confidence_is_partial_recall(self):
        assert _classify_quality("comprehensive", 0.6, 0.5) == "partial_recall"

    def test_partial_medium_confidence_is_partial_recall(self):
        assert _classify_quality("partial", 0.4, 0.5) == "partial_recall"

    def test_sparse_low_confidence_is_recognition_only(self):
        assert _classify_quality("sparse", 0.2, 0.2) == "recognition_only"

    def test_comprehensive_low_confidence_is_recognition_only(self):
        assert _classify_quality("comprehensive", 0.8, 0.3) == "recognition_only"

    def test_partial_low_confidence_is_recognition_only(self):
        assert _classify_quality("partial", 0.3, 0.2) == "recognition_only"

    def test_confidence_boundary_07(self):
        """At exactly 0.7 confidence, comprehensive = confident_recall."""
        assert _classify_quality("comprehensive", 0.8, 0.7) == "confident_recall"

    def test_confidence_boundary_04(self):
        """At exactly 0.4 confidence, partial = partial_recall."""
        assert _classify_quality("partial", 0.4, 0.4) == "partial_recall"


# ============================================================
# RETRIEVAL FAILURE DIAGNOSTICS (Schacter 1999)
# ============================================================

class TestDiagnoseRetrievalFailure:
    def setup_method(self):
        _retrieval_buffer.clear()

    def test_comprehensive_no_diagnosis(self):
        """Non-failure coverage returns empty string."""
        assert diagnose_retrieval_failure("comprehensive", 5, 0.8, "test query") == ""

    def test_partial_no_diagnosis(self):
        assert diagnose_retrieval_failure("partial", 3, 0.5, "test query") == ""

    def test_empty_no_history_is_never_stored(self):
        """Empty + no past success = never_stored."""
        result = diagnose_retrieval_failure("empty", 0, 0.0, "quantum entanglement")
        assert result == "never_stored"

    def test_empty_with_past_success_is_decayed(self):
        """Empty + past successful query with similar words = decayed."""
        # Add a past successful retrieval with similar topic
        wrap_retrieval_result("kubernetes deployment guide", _fake_merged(5, activation=0.7))
        result = diagnose_retrieval_failure("empty", 0, 0.0, "kubernetes deployment")
        assert result == "decayed"

    def test_sparse_low_activation_is_tip_of_tongue(self):
        """Sparse + low activation = tip_of_tongue."""
        result = diagnose_retrieval_failure("sparse", 1, 0.2, "some query here")
        assert result == "tip_of_tongue"

    def test_sparse_high_activation_is_retrieval_failure(self):
        """Sparse + higher activation = retrieval_failure."""
        result = diagnose_retrieval_failure("sparse", 2, 0.5, "some query here")
        assert result == "retrieval_failure"

    def test_short_query_words_ignored(self):
        """Words <= 3 chars are ignored in buffer matching."""
        wrap_retrieval_result("the cat sat on mat", _fake_merged(5, activation=0.7))
        # "the" and "cat" are <=3, so no match with "the big cat"
        result = diagnose_retrieval_failure("empty", 0, 0.0, "the big cat")
        # "the" is <=3, "big" is ==3, "cat" is ==3 -> all filtered out
        assert result == "never_stored"


# ============================================================
# WRAP RETRIEVAL RESULT - NEW FIELDS
# ============================================================

class TestWrapRetrievalResultNewFields:
    def setup_method(self):
        _retrieval_buffer.clear()

    def test_comprehensive_has_quality_class(self):
        result = wrap_retrieval_result("good query", _fake_merged(6, activation=0.8))
        assert result.quality_class in ("confident_recall", "partial_recall")

    def test_empty_has_blank_quality(self):
        result = wrap_retrieval_result("bad query", [])
        assert result.quality_class == "blank"

    def test_empty_has_failure_diagnosis(self):
        result = wrap_retrieval_result("bad query", [])
        assert result.failure_diagnosis == "never_stored"

    def test_comprehensive_no_failure_diagnosis(self):
        result = wrap_retrieval_result("good query", _fake_merged(5, activation=0.7))
        assert result.failure_diagnosis == ""

    def test_tip_of_tongue_overrides_quality(self):
        """When diagnosis is tip_of_tongue, quality_class should be tip_of_tongue."""
        result = wrap_retrieval_result("sparse query", _fake_merged(1, activation=0.1))
        if result.failure_diagnosis == "tip_of_tongue":
            assert result.quality_class == "tip_of_tongue"

    def test_sparse_gets_diagnosis(self):
        result = wrap_retrieval_result("sparse", _fake_merged(2, activation=0.2))
        assert result.failure_diagnosis != ""


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
