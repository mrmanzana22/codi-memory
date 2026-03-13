"""
Tests for modules/counterfactual.py — Pearl's Ladder of Causation Level 3.

Unit tests with mocks. No DB required.
Covers: parse_counterfactual_query, _extract_intervention_target,
        abduction_step, intervention_step, prediction_step,
        _compute_confidence, _summarize_chain, run_counterfactual.
"""

import pytest
from unittest.mock import patch, MagicMock
from modules.counterfactual import (
    parse_counterfactual_query,
    _extract_intervention_target,
    abduction_step,
    intervention_step,
    prediction_step,
    _compute_confidence,
    _summarize_chain,
    run_counterfactual,
)


# ============================================================
# 8.1: PARSER
# ============================================================

class TestParseCounterfactualQuery:

    def test_english_what_if(self):
        result = parse_counterfactual_query("what if we hadn't used Docker?")
        assert result is not None
        assert result["language"] == "en"
        assert result["intervention_target"] == "Docker"
        assert result["negated"] is True

    def test_english_positive_intervention(self):
        result = parse_counterfactual_query("what if we had used Redis?")
        assert result is not None
        assert result["intervention_target"] == "Redis"
        assert result["negated"] is False

    def test_spanish_counterfactual(self):
        result = parse_counterfactual_query(
            "que hubiera pasado si no usabamos Qdrant?"
        )
        assert result is not None
        assert result["language"] == "es"
        assert result["negated"] is True

    def test_spanish_si_no_hubiera(self):
        result = parse_counterfactual_query(
            "si no hubiera implementado Docker, que pasaba?"
        )
        assert result is not None
        assert result["language"] == "es"

    def test_non_counterfactual_returns_none(self):
        assert parse_counterfactual_query("how does Docker work?") is None

    def test_empty_string(self):
        assert parse_counterfactual_query("") is None

    def test_alternative_scenario_en(self):
        result = parse_counterfactual_query("alternative scenario without Redis")
        assert result is not None
        assert result["language"] == "en"

    def test_raw_query_preserved(self):
        q = "what if we never used Python?"
        result = parse_counterfactual_query(q)
        assert result is not None
        assert result["raw_query"] == q


class TestExtractInterventionTarget:

    def test_capitalized_tech_word(self):
        target = _extract_intervention_target(
            "what if we hadn't used Docker?",
            "what if we hadn't used docker?"
        )
        assert target == "Docker"

    def test_negation_after_not_used(self):
        target = _extract_intervention_target(
            "what if we had not used python for this?",
            "what if we had not used python for this?"
        )
        assert target == "python"

    def test_fallback_longest_word(self):
        target = _extract_intervention_target(
            "what if something different happened",
            "what if something different happened"
        )
        assert len(target) >= 4


# ============================================================
# 8.2: ABDUCTION
# ============================================================

class TestAbductionStep:

    def test_empty_target_returns_empty(self):
        parsed = {"intervention_target": "", "negated": True}
        assert abduction_step(parsed, "/nonexistent/path.db") == []

    def test_nonexistent_db_returns_empty(self):
        parsed = {"intervention_target": "Docker", "negated": True}
        assert abduction_step(parsed, "/no/such/file.db") == []

    def test_abduction_with_mock_db(self, tmp_path):
        import sqlite3
        db_path = str(tmp_path / "test_fts.db")
        conn = sqlite3.connect(db_path)
        conn.execute("""
            CREATE TABLE memories_text (
                memory_id TEXT, content TEXT, topic TEXT, created_at TEXT
            )
        """)
        conn.execute(
            "INSERT INTO memories_text VALUES (?, ?, ?, ?)",
            ("mem-1", "We deployed with Docker successfully", "devops", "2026-01-15T10:00:00")
        )
        conn.execute(
            "INSERT INTO memories_text VALUES (?, ?, ?, ?)",
            ("mem-2", "Docker container config updated", "devops", "2026-01-16T10:00:00")
        )
        conn.commit()
        conn.close()

        parsed = {"intervention_target": "Docker", "negated": True}
        # connect_fts is imported inside abduction_step via `from modules.config import connect_fts`
        # patch at the source (modules.config) so the local import picks it up
        with patch("modules.config.connect_fts", return_value=sqlite3.connect(db_path)):
            chain = abduction_step(parsed, db_path)

        assert len(chain) == 2
        assert chain[0]["memory_id"] == "mem-1"
        assert chain[0]["source"] == "text_match"

    def test_abduction_caps_at_10(self, tmp_path):
        import sqlite3
        db_path = str(tmp_path / "big_fts.db")
        conn = sqlite3.connect(db_path)
        conn.execute("""
            CREATE TABLE memories_text (
                memory_id TEXT, content TEXT, topic TEXT, created_at TEXT
            )
        """)
        for i in range(20):
            conn.execute(
                "INSERT INTO memories_text VALUES (?, ?, ?, ?)",
                (f"mem-{i}", f"Redis usage number {i}", "infra", f"2026-01-{i+1:02d}T10:00:00")
            )
        conn.commit()
        conn.close()

        parsed = {"intervention_target": "Redis", "negated": True}
        with patch("modules.config.connect_fts", return_value=sqlite3.connect(db_path)):
            chain = abduction_step(parsed, db_path)

        assert len(chain) <= 10


# ============================================================
# 8.3: INTERVENTION
# ============================================================

class TestInterventionStep:

    def test_empty_chain_passthrough(self):
        parsed = {"intervention_target": "Docker", "negated": True}
        assert intervention_step([], parsed, "/fake.db") == []

    def test_empty_target_passthrough(self):
        chain = [{"memory_id": "m1", "content": "test", "topic": "t"}]
        parsed = {"intervention_target": "", "negated": True}
        result = intervention_step(chain, parsed, "/fake.db")
        assert result == chain

    def test_removes_target_memories(self):
        chain = [
            {"memory_id": "m1", "content": "Used Docker for deploy", "topic": "devops"},
            {"memory_id": "m2", "content": "Configured monitoring", "topic": "devops"},
            {"memory_id": "m3", "content": "Docker logs collected", "topic": "devops"},
        ]
        parsed = {"intervention_target": "Docker", "negated": True}
        result = intervention_step(chain, parsed, "/nonexistent.db")
        # Only m2 should remain (doesn't mention Docker)
        assert len(result) == 1
        assert result[0]["memory_id"] == "m2"

    def test_positive_intervention_no_negation(self):
        chain = [
            {"memory_id": "m1", "content": "Setup infra", "topic": "devops",
             "memory": "We used Kubernetes"},
        ]
        parsed = {"intervention_target": "Kubernetes", "negated": False}
        result = intervention_step(chain, parsed, "/fake.db")
        assert len(result) >= 1

    def test_no_match_returns_kept(self):
        chain = [
            {"memory_id": "m1", "content": "Used Python", "topic": "dev"},
        ]
        parsed = {"intervention_target": "Rust", "negated": True}
        result = intervention_step(chain, parsed, "/fake.db")
        assert len(result) == 1
        assert result[0]["memory_id"] == "m1"


# ============================================================
# 8.4: PREDICTION
# ============================================================

class TestPredictionHelpers:

    def test_summarize_chain_empty(self):
        result = _summarize_chain([], "factual")
        assert "No factual" in result

    def test_summarize_chain_truncates(self):
        chain = [{"content": "x" * 200, "topic": "test"}]
        result = _summarize_chain(chain, "test")
        assert len(result) < 200

    def test_summarize_chain_topics(self):
        chain = [
            {"content": "short", "topic": "devops"},
            {"content": "last item", "topic": "infra"},
        ]
        result = _summarize_chain(chain, "factual")
        assert "last item" in result

    def test_compute_confidence_no_factual(self):
        assert _compute_confidence([], []) == 0.1

    def test_compute_confidence_no_alternatives(self):
        factual = [{"memory_id": "m1"}]
        assert _compute_confidence(factual, []) == 0.2

    def test_compute_confidence_with_both(self):
        factual = [{"memory_id": f"m{i}"} for i in range(5)]
        alt = [{"memory_id": "a1", "source": "alternative"},
               {"memory_id": "a2", "source": "alternative"}]
        conf = _compute_confidence(factual, alt)
        assert 0.0 < conf <= 1.0
        assert conf > 0.5  # Good evidence = high confidence


class TestPredictionStep:

    def test_prediction_step_returns_structure(self):
        factual = [
            {"memory_id": "m1", "content": "Used Docker", "topic": "devops",
             "created_at": "2026-01-15T10:00:00"},
        ]
        alt = [
            {"memory_id": "m2", "content": "Monitoring active", "topic": "devops",
             "created_at": "2026-01-16T10:00:00"},
        ]
        parsed = {
            "intervention_target": "Docker",
            "negated": True,
            "language": "en",
            "raw_query": "what if we hadn't used Docker?",
        }
        result = prediction_step(alt, factual, parsed, "/nonexistent.db")
        assert "factual_outcome" in result
        assert "counterfactual_outcome" in result
        assert "narrative" in result
        assert "confidence" in result
        assert isinstance(result["confidence"], float)
        assert "factual_memories" in result
        assert "alt_memories" in result

    def test_prediction_step_spanish(self):
        parsed = {
            "intervention_target": "Redis",
            "negated": True,
            "language": "es",
            "raw_query": "que hubiera pasado sin Redis?",
        }
        factual = [{"memory_id": "m1", "content": "Redis running", "topic": "infra"}]
        result = prediction_step([], factual, parsed, "/fake.db")
        assert "Análisis Contrafactual" in result["narrative"]

    def test_prediction_empty_chains(self):
        parsed = {
            "intervention_target": "test",
            "negated": True,
            "language": "en",
            "raw_query": "what if test?",
        }
        result = prediction_step([], [], parsed, "/fake.db")
        assert result["confidence"] == 0.1


# ============================================================
# MAIN ENTRY POINT
# ============================================================

class TestRunCounterfactual:

    def test_non_counterfactual_query(self):
        result = run_counterfactual("how does Docker work?", fts_db_path="/fake.db")
        assert result["error"] == "not_counterfactual"

    @patch("modules.counterfactual.abduction_step", return_value=[])
    def test_counterfactual_with_empty_chain(self, mock_abd):
        result = run_counterfactual(
            "what if we hadn't used Docker?",
            fts_db_path="/fake.db"
        )
        assert "parsed" in result
        assert result["parsed"]["intervention_target"] == "Docker"
        assert "prediction" in result
        assert result["prediction"]["confidence"] == 0.1

    @patch("modules.counterfactual.abduction_step")
    def test_full_pipeline_structure(self, mock_abd):
        mock_abd.return_value = [
            {"memory_id": "m1", "content": "Docker deployed",
             "topic": "devops", "created_at": "2026-01-15T10:00:00"},
        ]
        result = run_counterfactual(
            "what if we hadn't used Docker?",
            fts_db_path="/fake.db"
        )
        assert "parsed" in result
        assert "factual_chain" in result
        assert "alternative_chain" in result
        assert "prediction" in result
        assert result["prediction"]["factual_memories"] >= 0
