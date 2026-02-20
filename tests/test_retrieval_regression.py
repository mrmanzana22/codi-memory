#!/usr/bin/env python3
"""
RETRIEVAL REGRESSION TESTS - Golden queries for search quality
==============================================================
Deterministic tests that verify search_memory scoring doesn't degrade.

Each test sets up a controlled memory universe (mocked qdrant + mem0),
runs a query, and asserts specific ranking/scoring properties.

If these tests break after a code change, the retrieval pipeline has regressed.

Run: ./venv/bin/pytest tests/test_retrieval_regression.py -v
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from types import SimpleNamespace
from unittest.mock import patch, MagicMock
from modules.events import event_bus, Events


# ============================================================
# HELPERS: Build controlled memory universes
# ============================================================

def _make_point(mid, data, **extra):
    """Create a mock Qdrant point with payload."""
    p = MagicMock()
    p.id = mid
    p.payload = {
        "data": data,
        "ownership_source": extra.get("source", "experienced"),
        "narrative_importance": extra.get("importance", "medium"),
        "created_at": extra.get("created_at", "2026-02-15T10:00:00"),
        "attention_access_count": extra.get("access_count", 0),
        "attention_last_accessed": extra.get("last_accessed", None),
        "access_timestamps": extra.get("access_timestamps", []),
        "attention_salience": extra.get("salience", 0.5),
        "experiential_emotional_valence": extra.get("valence", "neutral"),
        "pad_at_encoding": extra.get("pad", {"P": 0, "A": 0, "D": 0}),
        "_v": 4.0,
        **{k: v for k, v in extra.items() if k not in (
            "source", "importance", "created_at", "access_count",
            "last_accessed", "access_timestamps", "salience", "valence", "pad"
        )},
    }
    return p


def _fake_activation(**kwargs):
    """Return a neutral activation (lets vector score dominate)."""
    return SimpleNamespace(total=0.3)


def _patch_search(vector_results, qdrant_points, semantic_results=None,
                  bm25_results=None, activation_fn=None):
    """Context manager that patches all search_memory dependencies.

    Returns the patches dict for use in assertions.
    """
    from contextlib import contextmanager

    @contextmanager
    def _ctx():
        act_fn = activation_fn or _fake_activation
        patches = {}

        with patch("modules.memory_core.memory") as mock_mem, \
             patch("modules.memory_core.qdrant") as mock_qdrant, \
             patch("modules.memory_core.search_fts", return_value=bm25_results or []), \
             patch("modules.memory_core.search_semantic", return_value=semantic_results or []), \
             patch("modules.memory_core.compute_unified_activation", side_effect=act_fn):

            mock_mem.search.return_value = vector_results
            mock_qdrant.retrieve.return_value = qdrant_points
            mock_qdrant.set_payload.return_value = True

            patches["memory"] = mock_mem
            patches["qdrant"] = mock_qdrant
            yield patches

    return _ctx()


# ============================================================
# GOLDEN QUERY 1: High vector score beats low vector score
# ============================================================

class TestRankingByVectorScore:
    """Verify that higher vector similarity produces higher ranking."""

    def test_high_vector_ranked_first(self):
        """Memory with 0.95 vector score should rank above 0.40 vector score.

        Both have identical activation (0.3), zero BM25.
        Expected ranking: Docker > Pizza (vector dominance).
        """
        from modules.memory_core import search_memory

        vector_results = {
            "results": [
                {"id": "docker-1", "score": 0.95, "memory": "Docker is great for deploy"},
                {"id": "pizza-1", "score": 0.40, "memory": "Pizza was delicious"},
            ]
        }
        points = [
            _make_point("docker-1", "Docker is great for deploy"),
            _make_point("pizza-1", "Pizza was delicious"),
        ]

        with _patch_search(vector_results, points):
            out = search_memory("Docker deploy", limit=5)

        # GOLDEN ASSERT: Docker ranked first
        docker_pos = out.find("Docker")
        pizza_pos = out.find("Pizza")
        assert docker_pos < pizza_pos, \
            f"Docker (0.95) should rank before Pizza (0.40). Docker@{docker_pos}, Pizza@{pizza_pos}"

    def test_three_memories_correct_order(self):
        """Three memories with different vector scores should rank correctly.

        Scores: A=0.92, B=0.65, C=0.30 -> expected order: A, B, C.
        """
        from modules.memory_core import search_memory

        vector_results = {
            "results": [
                {"id": "a", "score": 0.92, "memory": "MEMORY_ALPHA"},
                {"id": "b", "score": 0.65, "memory": "MEMORY_BETA"},
                {"id": "c", "score": 0.30, "memory": "MEMORY_GAMMA"},
            ]
        }
        points = [
            _make_point("a", "MEMORY_ALPHA"),
            _make_point("b", "MEMORY_BETA"),
            _make_point("c", "MEMORY_GAMMA"),
        ]

        with _patch_search(vector_results, points):
            out = search_memory("test query", limit=5)

        alpha_pos = out.find("ALPHA")
        beta_pos = out.find("BETA")
        gamma_pos = out.find("GAMMA")
        assert alpha_pos < beta_pos < gamma_pos, \
            f"Expected ALPHA < BETA < GAMMA, got {alpha_pos}, {beta_pos}, {gamma_pos}"


# ============================================================
# GOLDEN QUERY 2: High activation can beat moderate vector
# ============================================================

class TestActivationBoost:
    """Verify that activation (ACT-R) can elevate lower-vector-score memories."""

    def test_high_activation_beats_moderate_vector(self):
        """Memory with moderate vector (0.55) but high activation (0.8)
        should beat memory with high vector (0.75) but low activation (0.1).

        Formula: 0.40*vector + 0.15*bm25 + 0.45*activation
        A: 0.40*0.55 + 0.15*0 + 0.45*0.8 = 0.22 + 0.36 = 0.58
        B: 0.40*0.75 + 0.15*0 + 0.45*0.1 = 0.30 + 0.045 = 0.345
        """
        from modules.memory_core import search_memory

        vector_results = {
            "results": [
                {"id": "frequent", "score": 0.55, "memory": "FREQUENT_MEMORY"},
                {"id": "rare", "score": 0.75, "memory": "RARE_MEMORY"},
            ]
        }
        points = [
            _make_point("frequent", "FREQUENT_MEMORY", access_count=50),
            _make_point("rare", "RARE_MEMORY", access_count=0),
        ]

        call_count = [0]
        def activation_by_access(**kwargs):
            call_count[0] += 1
            count = kwargs.get("access_count", 0)
            if count >= 50:
                return SimpleNamespace(total=0.8)
            return SimpleNamespace(total=0.1)

        with _patch_search(vector_results, points, activation_fn=activation_by_access):
            out = search_memory("test", limit=5)

        frequent_pos = out.find("FREQUENT")
        rare_pos = out.find("RARE")
        assert frequent_pos < rare_pos, \
            f"Frequent memory (high activation) should rank first. F@{frequent_pos}, R@{rare_pos}"


# ============================================================
# GOLDEN QUERY 3: Semantic facts labeled [FACT] in output
# ============================================================

class TestSemanticFactFormat:
    """Verify semantic facts are distinguishable from episodic memories."""

    def test_semantic_fact_labeled(self):
        """Semantic facts must appear with [FACT][topic] prefix."""
        from modules.memory_core import search_memory

        semantic_results = [
            {
                "id": "fact-1",
                "fact": "Python is dynamically typed",
                "topic": "programming",
                "confidence": 0.9,
                "evidence_count": 5,
                "score": 0.8,
            }
        ]

        # No episodic results, only semantic
        vector_results = {"results": []}

        with _patch_search(vector_results, [], semantic_results=semantic_results):
            out = search_memory("Python typing", limit=5)

        assert "[FACT]" in out, "Semantic fact must be labeled [FACT]"
        assert "[programming]" in out, "Topic must appear in output"
        assert "dynamically typed" in out, "Fact text must appear"

    def test_mixed_episodic_and_semantic(self):
        """When both channels return results, both should appear in output."""
        from modules.memory_core import search_memory

        vector_results = {
            "results": [
                {"id": "ep-1", "score": 0.85, "memory": "EPISODIC_DOCKER"},
            ]
        }
        ep_points = [_make_point("ep-1", "EPISODIC_DOCKER")]

        semantic_results = [
            {
                "id": "sem-1",
                "fact": "SEMANTIC_PYTHON",
                "topic": "code",
                "confidence": 0.9,
                "evidence_count": 3,
                "score": 0.7,
            }
        ]

        with _patch_search(vector_results, ep_points, semantic_results=semantic_results):
            out = search_memory("programming", limit=10)

        assert "EPISODIC_DOCKER" in out, "Episodic result must appear"
        assert "SEMANTIC_PYTHON" in out, "Semantic result must appear"
        assert "[FACT]" in out, "Semantic results must have [FACT] tag"


# ============================================================
# GOLDEN QUERY 4: Output metadata format preserved
# ============================================================

class TestOutputMetadataFormat:
    """Verify the output format includes expected metadata fields."""

    def test_episodic_shows_source_and_importance(self):
        """Episodic output must show [source|importance] metadata."""
        from modules.memory_core import search_memory

        vector_results = {
            "results": [
                {"id": "m1", "score": 0.9, "memory": "Trading analysis completed"},
            ]
        }
        points = [
            _make_point("m1", "Trading analysis completed",
                       source="experienced", importance="high"),
        ]

        with _patch_search(vector_results, points):
            out = search_memory("trading", limit=5)

        assert "experienced" in out, "Source should appear in output"
        assert "high" in out, "Importance should appear in output"
        assert "score:" in out, "Score must be shown"

    def test_semantic_shows_confidence_and_evidence(self):
        """Semantic output must show [score:X|conf:Y|ev:Z] format."""
        from modules.memory_core import search_memory

        semantic_results = [
            {
                "id": "f1",
                "fact": "Kraken supports limit orders",
                "topic": "trading",
                "confidence": 0.85,
                "evidence_count": 4,
                "score": 0.75,
            }
        ]

        with _patch_search({"results": []}, [], semantic_results=semantic_results):
            out = search_memory("Kraken orders", limit=5)

        assert "conf:" in out, "Confidence must appear in semantic output"
        assert "ev:" in out, "Evidence count must appear in semantic output"


# ============================================================
# GOLDEN QUERY 5: BM25 contributes to scoring
# ============================================================

class TestBM25Contribution:
    """Verify BM25 (keyword) results influence the ranking."""

    def test_bm25_only_memory_appears(self):
        """Memory found only via BM25 (not vector) should still appear.

        This tests the hybrid nature: pure keyword matches get included.
        """
        from modules.memory_core import search_memory

        # Vector returns nothing useful, but BM25 finds an exact keyword match
        vector_results = {"results": []}
        bm25_results = [
            {
                "memory_id": "bm25-1",
                "content": "BM25_KEYWORD_MATCH here",
                "bm25_rank": 1,
            }
        ]
        points = [_make_point("bm25-1", "BM25_KEYWORD_MATCH here")]

        with _patch_search(vector_results, points, bm25_results=bm25_results):
            out = search_memory("BM25_KEYWORD_MATCH", limit=5)

        assert "BM25_KEYWORD_MATCH" in out, \
            "BM25-only result must appear in hybrid search output"


# ============================================================
# GOLDEN QUERY 6: Empty results handled gracefully
# ============================================================

class TestEdgeCases:
    """Verify edge cases don't crash or produce unexpected output."""

    def test_no_results_returns_message(self):
        """When no memories match, return a helpful message (not crash)."""
        from modules.memory_core import search_memory

        with _patch_search({"results": []}, []):
            out = search_memory("xyzzy_nonexistent_query_12345", limit=5)

        assert "No encontre" in out or len(out) < 100, \
            "Empty results should return a short message, not crash"

    def test_single_result_works(self):
        """Single result should work without errors."""
        from modules.memory_core import search_memory

        vector_results = {
            "results": [
                {"id": "solo", "score": 0.88, "memory": "SOLO_MEMORY"},
            ]
        }
        points = [_make_point("solo", "SOLO_MEMORY")]

        with _patch_search(vector_results, points):
            out = search_memory("solo test", limit=1)

        assert "SOLO_MEMORY" in out


# ============================================================
# GOLDEN QUERY 7: Score formula correctness
# ============================================================

class TestScoreFormula:
    """Verify the 3-channel fusion formula produces expected scores."""

    def test_episodic_score_in_expected_range(self):
        """With known inputs, the combined score should be predictable.

        vector=0.80, bm25=0, activation=0.5
        Expected: 0.40*0.80 + 0.15*0 + 0.45*0.5 = 0.32 + 0.225 = 0.545
        Output should show score:0.5X (within ~0.1 tolerance for SDR bonus etc).
        """
        from modules.memory_core import search_memory

        vector_results = {
            "results": [
                {"id": "scored", "score": 0.80, "memory": "SCORED_MEMORY"},
            ]
        }
        points = [_make_point("scored", "SCORED_MEMORY")]

        def fixed_activation(**kwargs):
            return SimpleNamespace(total=0.5)

        with _patch_search(vector_results, points, activation_fn=fixed_activation):
            out = search_memory("score test", limit=5)

        # Extract score from "score:X.XX"
        import re
        match = re.search(r"score:(\d+\.\d+)", out)
        assert match, f"Score not found in output: {out}"
        score = float(match.group(1))
        # Expected ~0.545 +/- tolerance for SDR bonus and rounding
        assert 0.40 <= score <= 0.70, \
            f"Expected score ~0.545, got {score}"
