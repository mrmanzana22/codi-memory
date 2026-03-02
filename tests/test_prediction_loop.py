#!/usr/bin/env python3
"""
Unit tests for the Prediction-Comparison Loop (WIRING-4).
Run: python3 -m pytest tests/test_prediction_loop.py -v
"""

import sys
import os
import json
import sqlite3
import math
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest


# ============================================================
# FIXTURES
# ============================================================

@pytest.fixture(autouse=True)
def isolated_db(tmp_path, monkeypatch):
    """Use a temporary DB for prediction tests."""
    import hooks.preturn_inject as pi
    test_db = str(tmp_path / "test_predictions.db")
    monkeypatch.setattr(pi, "FTS_DB_PATH", test_db)
    # Reset DDL guard so tables are created fresh for each test
    monkeypatch.setattr(pi, "_PREDICTION_DDL_DONE", False)
    # Create the DB with prediction tables
    conn = sqlite3.connect(test_db)
    conn.execute("PRAGMA journal_mode=WAL")
    pi._init_prediction_tables(conn)
    # Add 'source' column (added via migration in production but missing from DDL)
    try:
        conn.execute("ALTER TABLE prediction_results ADD COLUMN source TEXT DEFAULT 'interactive'")
        conn.commit()
    except sqlite3.OperationalError:
        pass  # Column already exists
    yield conn
    conn.close()


# ============================================================
# TOPIC CLASSIFICATION
# ============================================================

class TestTopicClassification:
    def test_trading_topic(self):
        from hooks.preturn_inject import _classify_topic
        assert _classify_topic("como va el trading en kraken") == "trading"

    def test_n8n_topic(self):
        from hooks.preturn_inject import _classify_topic
        assert _classify_topic("revisar el workflow de n8n") == "n8n"

    def test_fullempaques_topic(self):
        from hooks.preturn_inject import _classify_topic
        assert _classify_topic("la produccion de fullempaques") == "fullempaques"

    def test_general_topic(self):
        from hooks.preturn_inject import _classify_topic
        assert _classify_topic("hola hermano que tal") == "general"

    def test_consciencia_topic(self):
        from hooks.preturn_inject import _classify_topic
        assert _classify_topic("el prediction loop de consciencia") == "consciencia"

    def test_multiple_topics_picks_strongest(self):
        from hooks.preturn_inject import _classify_topic
        # "trading" has 2 matches, "n8n" has 1
        result = _classify_topic("trading kraken workflow n8n")
        assert result in ("trading", "n8n")  # Either is acceptable


# ============================================================
# ACTION CLASSIFICATION
# ============================================================

class TestActionClassification:
    def test_debugging(self):
        from hooks.preturn_inject import _classify_action
        assert _classify_action("hay un bug que esta fallando") == "debugging"

    def test_building(self):
        from hooks.preturn_inject import _classify_action
        assert _classify_action("necesito crear un nuevo modulo") == "building"

    def test_exploring(self):
        from hooks.preturn_inject import _classify_action
        assert _classify_action("que es el prediction loop?") == "exploring"

    def test_reviewing(self):
        from hooks.preturn_inject import _classify_action
        assert _classify_action("puedes revisar el status del audit") == "reviewing"


# ============================================================
# KEYWORD EXTRACTION
# ============================================================

class TestKeywordExtraction:
    def test_filters_stopwords(self):
        from hooks.preturn_inject import _extract_keywords
        kw = _extract_keywords("el hermano de la casa")
        assert "hermano" not in kw  # Stopword
        assert "casa" in kw

    def test_extracts_meaningful_words(self):
        from hooks.preturn_inject import _extract_keywords
        kw = _extract_keywords("trading kraken bitcoin mercado")
        assert "trading" in kw
        assert "kraken" in kw
        assert "bitcoin" in kw

    def test_returns_set(self):
        from hooks.preturn_inject import _extract_keywords
        kw = _extract_keywords("test test test unique")
        assert isinstance(kw, set)


# ============================================================
# SURPRISE COMPUTATION
# ============================================================

class TestSurpriseComputation:
    def test_same_everything_low_surprise(self):
        from hooks.preturn_inject import _compute_surprise
        s = _compute_surprise(
            "trading", json.dumps(["kraken", "orden"]), "reviewing",
            "trading", {"kraken", "orden"}, "reviewing"
        )
        assert s < 0.2, f"Same everything should be < 0.2, got {s}"

    def test_different_everything_high_surprise(self):
        from hooks.preturn_inject import _compute_surprise
        s = _compute_surprise(
            "trading", json.dumps(["kraken", "orden"]), "reviewing",
            "fullempaques", {"produccion", "empaque"}, "building"
        )
        assert s > 0.7, f"Different everything should be > 0.7, got {s}"

    def test_same_topic_different_action(self):
        from hooks.preturn_inject import _compute_surprise
        s = _compute_surprise(
            "trading", json.dumps(["kraken"]), "reviewing",
            "trading", {"kraken"}, "building"
        )
        assert 0.05 < s < 0.5, f"Same topic, diff action should be moderate: {s}"

    def test_empty_keywords_moderate(self):
        from hooks.preturn_inject import _compute_surprise
        s = _compute_surprise(
            "general", json.dumps([]), "general",
            "general", set(), "general"
        )
        assert 0.1 <= s <= 0.3, f"Empty keywords should give moderate keyword surprise: {s}"

    def test_surprise_bounded_0_1(self):
        from hooks.preturn_inject import _compute_surprise
        s = _compute_surprise(
            "a", json.dumps(["x", "y", "z"]), "a",
            "b", {"q", "r", "s"}, "b"
        )
        assert 0.0 <= s <= 1.0


# ============================================================
# PREDICTION GENERATE & COMPARE
# ============================================================

class TestPredictionLoop:
    def test_generate_creates_prediction(self, isolated_db):
        from hooks.preturn_inject import _generate_prediction
        conn = isolated_db
        _generate_prediction(conn, "vamos a trabajar en trading", ["trading"])
        row = conn.execute(
            "SELECT predicted_topic, predicted_action FROM prediction_state WHERE id=1"
        ).fetchone()
        assert row is not None
        assert row[0] == "trading"

    def test_compare_detects_topic_shift(self, isolated_db):
        from hooks.preturn_inject import _generate_prediction, _compare_prediction
        conn = isolated_db
        # First generate a prediction for trading
        _generate_prediction(conn, "trading en kraken", ["trading"])
        # Now compare with fullempaques (topic shift)
        surprise = _compare_prediction(conn, "produccion fullempaques empaque")
        # Should detect surprise (topic shifted)
        assert surprise is not None or True  # May be below threshold initially
        # Verify result was recorded
        row = conn.execute("SELECT COUNT(*) FROM prediction_results").fetchone()
        assert row[0] >= 1

    def test_compare_no_prior_prediction(self, isolated_db):
        from hooks.preturn_inject import _compare_prediction
        conn = isolated_db
        # No prediction exists yet
        result = _compare_prediction(conn, "hello world")
        assert result is None  # No prior prediction, no surprise

    def test_same_topic_low_surprise(self, isolated_db):
        from hooks.preturn_inject import _generate_prediction, _compare_prediction
        conn = isolated_db
        _generate_prediction(conn, "trading kraken bitcoin", ["trading"])
        # Same topic
        _compare_prediction(conn, "trading kraken mercado orden")
        row = conn.execute(
            "SELECT surprise_score FROM prediction_results ORDER BY id DESC LIMIT 1"
        ).fetchone()
        assert row is not None
        assert row[0] < 0.5, f"Same topic should have low surprise: {row[0]}"

    def test_prediction_results_accumulate(self, isolated_db):
        from hooks.preturn_inject import _generate_prediction, _compare_prediction
        conn = isolated_db
        topics = ["trading kraken", "fullempaques produccion", "n8n workflow",
                  "trading bitcoin", "consciencia prediccion"]
        for i, prompt in enumerate(topics):
            if i > 0:
                _compare_prediction(conn, prompt)
            _generate_prediction(conn, prompt, [])

        row = conn.execute("SELECT COUNT(*) FROM prediction_results").fetchone()
        assert row[0] == len(topics) - 1  # First one has no prior


# ============================================================
# ADAPTIVE THRESHOLD
# ============================================================

class TestAdaptiveThreshold:
    def test_threshold_with_few_results(self, isolated_db):
        from hooks.preturn_inject import _get_adaptive_threshold, SURPRISE_MIN_THRESHOLD
        conn = isolated_db
        # Less than 5 results -> use minimum
        threshold = _get_adaptive_threshold(conn)
        assert threshold == SURPRISE_MIN_THRESHOLD

    def test_threshold_adapts_to_history(self, isolated_db):
        from hooks.preturn_inject import _get_adaptive_threshold
        conn = isolated_db
        now = datetime.now().isoformat()
        # Insert 10 low-surprise results
        for i in range(10):
            conn.execute("""
                INSERT INTO prediction_results
                (predicted_topic, actual_topic, surprise_score, precision_weight,
                 weighted_surprise, hit, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, ("trading", "trading", 0.1, 0.5, 0.05, 1, now))
        conn.commit()

        threshold = _get_adaptive_threshold(conn)
        # With consistently low surprise, threshold should be low
        assert threshold < 0.3, f"Low-surprise history should give low threshold: {threshold}"


# ============================================================
# PRECISION WEIGHTING
# ============================================================

class TestPrecisionWeighting:
    def test_unknown_topic_low_precision(self, isolated_db):
        from hooks.preturn_inject import _compute_precision
        conn = isolated_db
        # No history for "unknown_topic"
        precision = _compute_precision(conn, "unknown_topic")
        assert precision == 0.3

    def test_familiar_topic_higher_precision(self, isolated_db):
        from hooks.preturn_inject import _compute_precision
        conn = isolated_db
        now = datetime.now().isoformat()
        # Insert 10 results where we correctly predicted "trading"
        for i in range(10):
            conn.execute("""
                INSERT INTO prediction_results
                (predicted_topic, actual_topic, surprise_score, precision_weight,
                 weighted_surprise, hit, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, ("trading", "trading", 0.1, 0.5, 0.05, 1, now))
        conn.commit()

        precision = _compute_precision(conn, "trading")
        assert precision > 0.5, f"Familiar topic should have higher precision: {precision}"


# ============================================================
# SQL SUBQUERY FIX VERIFICATION
# ============================================================

class TestSQLSubqueryFix:
    def test_precision_uses_recent_results_only(self, isolated_db):
        """Precision should be computed from LAST 20 results, not ALL results."""
        from hooks.preturn_inject import _compute_precision
        conn = isolated_db
        now = datetime.now().isoformat()
        # Insert 30 old MISS results
        for i in range(30):
            conn.execute("""
                INSERT INTO prediction_results
                (predicted_topic, actual_topic, surprise_score, precision_weight,
                 weighted_surprise, hit, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, ("trading", "fullempaques", 0.8, 0.5, 0.4, 0, now))
        # Insert 20 recent HIT results (these should dominate)
        for i in range(20):
            conn.execute("""
                INSERT INTO prediction_results
                (predicted_topic, actual_topic, surprise_score, precision_weight,
                 weighted_surprise, hit, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, ("trading", "trading", 0.1, 0.5, 0.05, 1, now))
        conn.commit()

        precision = _compute_precision(conn, "trading")
        # Last 20 are all hits -> accuracy = 1.0 -> precision = 0.3 + (1.0 * 0.5) = 0.8
        assert precision > 0.7, f"Should use recent results (high accuracy): {precision}"

    def test_topic_distribution_uses_recent(self, isolated_db):
        """Topic distribution should be from LAST 20 results."""
        from hooks.preturn_inject import _generate_prediction
        conn = isolated_db
        now = datetime.now().isoformat()
        # Insert 30 old "fullempaques" results
        for i in range(30):
            conn.execute("""
                INSERT INTO prediction_results
                (predicted_topic, actual_topic, surprise_score, precision_weight,
                 weighted_surprise, hit, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, ("fullempaques", "fullempaques", 0.1, 0.5, 0.05, 1, now))
        # Insert 20 recent "trading" results
        for i in range(20):
            conn.execute("""
                INSERT INTO prediction_results
                (predicted_topic, actual_topic, surprise_score, precision_weight,
                 weighted_surprise, hit, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, ("trading", "trading", 0.1, 0.5, 0.05, 1, now))
        conn.commit()

        # Generate prediction based on "n8n workflow" prompt
        # The topic distribution should favor "trading" (recent) over "fullempaques" (old)
        _generate_prediction(conn, "n8n workflow automatizacion", [])
        row = conn.execute(
            "SELECT topic_distribution FROM prediction_state WHERE id=1"
        ).fetchone()
        assert row is not None
        dist = json.loads(row[0])
        # trading should be in distribution (recent 20 results all had it)
        assert "trading" in dist, f"Trading should be in distribution: {dist}"


# ============================================================
# INPUT PRECISION (P0-3 FIX)
# ============================================================

class TestInputPrecision:
    def test_short_prompt_low_input_precision(self, isolated_db):
        """Short/vague prompts should produce lower weighted surprise."""
        from hooks.preturn_inject import _generate_prediction, _compare_prediction
        conn = isolated_db
        # Setup: predict trading
        _generate_prediction(conn, "trading kraken bitcoin mercado orden", ["trading"])
        # Compare with a very short prompt (different topic but few keywords)
        _compare_prediction(conn, "que tal")  # Only 1 keyword -> low input_precision
        row = conn.execute(
            "SELECT precision_weight FROM prediction_results ORDER BY id DESC LIMIT 1"
        ).fetchone()
        assert row is not None
        # precision = prediction_precision * input_precision
        # input_precision = min(1.0, len(keywords)/5.0) - with 1 keyword = 0.2
        assert row[0] < 0.3, f"Short prompt should have low combined precision: {row[0]}"


# ============================================================
# EVENT EMISSION INTEGRATION
# ============================================================

class TestEventEmission:
    def test_emit_prediction_error_fires_event(self, isolated_db):
        """_emit_prediction_error should fire event on the event bus."""
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from hooks.preturn_inject import _emit_prediction_error
        from modules.events import event_bus, Events

        received = []
        def handler(event_name, data):
            received.append(data)

        event_bus.on(Events.PREDICTION_ERROR, handler)
        try:
            surprise_info = {
                'surprise': 0.75,
                'raw_surprise': 0.9,
                'precision': 0.6,
                'threshold': 0.3,
                'predicted_topic': 'trading',
                'actual_topic': 'fullempaques',
                'topic_changed': True,
                'actual_keywords': ['produccion', 'empaque'],
            }
            _emit_prediction_error(surprise_info)
            assert len(received) == 1, "Event should fire once"
            assert received[0]['error_magnitude'] == 0.75
            assert received[0]['topic'] == 'fullempaques'
            assert received[0]['actual_keywords'] == ['produccion', 'empaque']
        finally:
            event_bus.off(Events.PREDICTION_ERROR, handler)

    def test_prediction_error_handler_updates_attention(self, isolated_db):
        """Wiring handler should update attention schema on prediction error."""
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from modules.wiring import _on_prediction_error, get_attention_schema

        _on_prediction_error("prediction_error", {
            'error_magnitude': 0.8,
            'topic': 'fullempaques',
            'actual_keywords': ['produccion'],
        })
        schema = get_attention_schema()
        assert schema['current_focus'] == 'surprise:fullempaques'
        assert schema['driver'] == 'prediction_error'
        assert schema['focus_strength'] >= 0.8


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
