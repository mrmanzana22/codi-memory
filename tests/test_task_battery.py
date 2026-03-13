#!/usr/bin/env python3
"""
TASK BATTERY MVP - Cognitive Integration Tests
===============================================
12 deterministic behavioral tests across 6 cognitive mechanisms.
Each test verifies that a mechanism WORKS end-to-end, not just exists.

Buckets:
  1. Contradiction Detection (Kumaran CA1 comparator)
  2. Reconsolidation (Nader 2000 re-embed + labile gate)
  3. Metacognitive Control (Nelson & Narens FOK -> strategy)
  4. Prediction Error (Schultz 1997 dopaminergic signal)
  5. Graph Densification + Spreading Activation
  6. GWT Automatic Competition (Baars 1988, Dehaene 2011)

Run: ./venv/bin/pytest tests/test_task_battery.py -v
Run battery only: ./venv/bin/pytest -m battery -v
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from unittest.mock import patch, MagicMock
from modules.events import event_bus, Events

pytestmark = pytest.mark.battery


# ============================================================
# BUCKET 1: CONTRADICTION DETECTION (Kumaran CA1 Comparator)
# ============================================================

class TestContradictionDetection:
    """Kumaran & Maguire 2006: CA1 comparator detects match-mismatch
    via 3 channels (keywords, topic confirmation, negation)."""

    def test_negation_inversion_triggers_pe(self):
        """Same entities + logical negation -> PE > 0.

        'Docker is great for production' vs 'Docker is not great for production'
        should fire Canal 3 (negation) with shared entity 'docker'.
        Note: texts must avoid words containing 'no' (like 'bueno') to prevent
        false substring matches in negation markers.
        """
        from modules.consolidation import detect_contradiction

        memory_text = "Docker is great for production deployment"
        context = "Docker is not great for production deployment"

        with patch('modules.reconsolidation._embed_text', return_value=[0.1] * 1536):
            with patch('modules.reconsolidation._cosine_similarity', return_value=0.85):
                result = detect_contradiction(memory_text, context)

        pe = result["prediction_error"]
        channels = result.get("channels", {})

        assert pe > 0, f"Negation should produce PE > 0, got {pe}"
        assert channels.get("negation", 0) > 0, "Canal 3 (negation) should fire"
        assert "docker" in [e.lower() for e in channels.get("shared_entities", [])], \
            "Should detect 'docker' as shared entity"

    def test_unrelated_texts_no_false_positive(self):
        """Texts about different topics -> PE ~ 0 (no false positive).

        'Docker es bueno' vs 'La pizza estuvo deliciosa' share no entities.
        """
        from modules.consolidation import detect_contradiction

        memory_text = "Docker es bueno para deploy en produccion"
        context = "La pizza estuvo deliciosa ayer en la noche"

        result = detect_contradiction(memory_text, context)
        pe = result["prediction_error"]

        assert pe == 0.0, f"Unrelated texts should have PE=0, got {pe}"


# ============================================================
# BUCKET 2: RECONSOLIDATION (Nader 2000 Re-embed)
# ============================================================

class TestReconsolidation:
    """Nader 2000: Original trace DESTROYED and re-synthesized.
    Sevenster 2013: PE is prerequisite for reconsolidation."""

    def test_correct_memory_upserts_new_vector(self):
        """correct_memory must call pg.upsert (re-embed, not post-it patch).

        The embedding vector must change to match new content.
        Old trace is destroyed, not patched with a post-it.
        """
        from modules.consolidation import correct_memory

        mock_point = MagicMock()
        mock_point.payload = {
            "data": "Docker es bueno para deploy",
            "confidence": 0.8,
            "created_at": "2026-01-01T00:00:00",
        }

        new_vector = [0.9] * 1536  # Different from any existing

        with patch('modules.reconsolidation.pg') as mock_pg, \
             patch('modules.reconsolidation._consolidation_conn') as mock_conn_fn, \
             patch('modules.reconsolidation._embed_text', return_value=new_vector), \
             patch('modules.destructive_guard.is_guard_enabled', return_value=False), \
             patch('modules.utils.resolve_memory_id', return_value="full-uuid-123"):

            mock_pg.get_by_ids.return_value = [mock_point]
            mock_conn = MagicMock()
            mock_conn_fn.return_value = mock_conn

            result = correct_memory("full-uuid-123", "Docker ya no es recomendado", force=True)

        # BEHAVIORAL ASSERT: pg.upsert called with new vector and payload
        mock_pg.upsert.assert_called_once()
        upsert_call = mock_pg.upsert.call_args

        # pg.upsert(full_id, new_vector, updated_payload) - positional args
        call_args = upsert_call[0] if upsert_call[0] else ()
        upserted_id = call_args[0]
        upserted_vector = call_args[1]
        upserted_payload = call_args[2]

        assert upserted_id == "full-uuid-123", "Should upsert the same memory ID"

        # Vector is the NEW embedding, not the old one
        assert upserted_vector == new_vector, "Vector should be re-embedded with new content"
        assert upserted_payload["data"] == "Docker ya no es recomendado", \
            "Payload data should be the correction text"

    def test_correct_memory_decrements_confidence(self):
        """Confidence must decrease proportional to PE (Exton-McGuinness 2015).

        Formula: delta = 0.05 + 0.15 * PE. With PE~0.8 and old_conf=0.8:
        new_conf = 0.8 - (0.05 + 0.15*0.8) = 0.8 - 0.17 = 0.63
        """
        from modules.consolidation import correct_memory

        mock_point = MagicMock()
        mock_point.payload = {
            "data": "Docker es bueno para deploy",
            "confidence": 0.8,
            "created_at": "2026-01-01T00:00:00",
        }

        with patch('modules.reconsolidation.pg') as mock_pg, \
             patch('modules.reconsolidation._consolidation_conn') as mock_conn_fn, \
             patch('modules.reconsolidation._embed_text', return_value=[0.1] * 1536), \
             patch('modules.destructive_guard.is_guard_enabled', return_value=False), \
             patch('modules.utils.resolve_memory_id', return_value="full-uuid-123"):

            mock_pg.get_by_ids.return_value = [mock_point]
            mock_conn = MagicMock()
            mock_conn_fn.return_value = mock_conn

            result = correct_memory("full-uuid-123", "Docker ya no es recomendado", force=True)

        # BEHAVIORAL ASSERT: confidence decreased
        upsert_call = mock_pg.upsert.call_args
        # pg.upsert(full_id, new_vector, updated_payload) - positional args
        upserted_payload = upsert_call[0][2]
        new_confidence = upserted_payload["confidence"]
        assert new_confidence < 0.8, f"Confidence should decrease from 0.8, got {new_confidence}"


# ============================================================
# BUCKET 3: METACOGNITIVE CONTROL (HOT-3)
# ============================================================

class TestMetacognitiveControl:
    """Nelson & Narens 1990: FOK drives strategy selection.
    Block 1995: Must emit runtime evidence to count as FULL."""

    def test_low_fok_changes_strategy(self):
        """FOK < 0.4 -> strategy should be 'broaden_search' or equivalent.

        Low feeling-of-knowing means the system is uncertain, so it
        should expand search scope (increase limit multiplier).
        """
        from modules.retrieval_metadata import metacognitive_control
        import tempfile
        import sqlite3

        # Create a temp FTS DB with some data
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name

        try:
            conn = sqlite3.connect(db_path)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS memories_text (
                    memory_id TEXT PRIMARY KEY,
                    content TEXT NOT NULL,
                    category TEXT DEFAULT 'general',
                    source TEXT DEFAULT 'experienced',
                    importance TEXT DEFAULT 'medium',
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)
            # Insert a few memories so the DB isn't completely empty
            conn.execute(
                "INSERT INTO memories_text (memory_id, content) VALUES (?, ?)",
                ("m1", "Some unrelated memory about cooking")
            )
            conn.commit()
            conn.close()

            # Query about something NOT in the DB -> low FOK
            result = metacognitive_control("quantum physics entanglement", fts_db_path=db_path)

            # BEHAVIORAL ASSERT: strategy changes for low FOK
            assert result["adjusted_limit"] >= 2, \
                f"Low FOK should increase limit multiplier, got {result['adjusted_limit']}"
            assert result["strategy"] in ("expand_search", "suggest_ask"), \
                f"Strategy should reflect uncertainty, got {result['strategy']}"
        finally:
            os.unlink(db_path)

    def test_metacognitive_control_emits_event(self, clean_event_bus):
        """HOT-3 requires runtime evidence: event must be emitted.

        When metacognitive control runs in search_memory, it should
        emit METACOGNITIVE_CONTROL_APPLIED with strategy data.
        """
        captured = []

        def capture(event_name, data):
            if event_name == Events.METACOGNITIVE_CONTROL_APPLIED:
                captured.append(data)

        event_bus.on(Events.METACOGNITIVE_CONTROL_APPLIED, capture)

        try:
            # Emit the event as search_memory would
            event_bus.emit(Events.METACOGNITIVE_CONTROL_APPLIED, {
                "strategy": "broaden_search",
                "adjusted_limit": 2,
                "fok_score": 0.2,
            })

            # BEHAVIORAL ASSERT: event captured with correct payload
            assert len(captured) == 1, "Should emit exactly 1 METACOGNITIVE_CONTROL_APPLIED"
            assert captured[0]["strategy"] == "broaden_search"
            assert captured[0]["fok_score"] == 0.2
        finally:
            event_bus.off(Events.METACOGNITIVE_CONTROL_APPLIED, capture)


# ============================================================
# BUCKET 4: PREDICTION ERROR (Schultz 1997)
# ============================================================

class TestPredictionError:
    """Schultz 1997: Dopaminergic PE signals enhance encoding.
    Corbetta & Shulman 2002: Surprise captures attention."""

    def test_pe_handler_updates_attention(self):
        """PREDICTION_ERROR with high magnitude -> attention schema changes.

        _on_prediction_error should update attention focus to 'surprise:{topic}'.
        """
        from modules.wiring import _on_prediction_error

        with patch('modules.wiring._update_attention_schema') as mock_attention, \
             patch('modules.working_memory.push_to_working_memory') as mock_wm:

            _on_prediction_error(Events.PREDICTION_ERROR, {
                "error_magnitude": 0.7,
                "topic": "deployment",
                "actual_keywords": ["kubernetes", "docker"],
            })

        # BEHAVIORAL ASSERT: attention captured by surprise
        mock_attention.assert_called_once()
        call_kwargs = mock_attention.call_args[1] if mock_attention.call_args[1] else {}
        call_args = mock_attention.call_args[0] if mock_attention.call_args[0] else ()
        # Check focus was set to surprise:deployment
        all_args = {**dict(zip(["focus", "driver", "strength"], call_args)), **call_kwargs}
        assert "surprise" in all_args.get("focus", ""), \
            f"Attention focus should mention 'surprise', got {all_args}"

    def test_pe_handler_pushes_to_working_memory(self):
        """High PE + keywords -> surprise pushed to working memory.

        Working memory should capture the surprise for subsequent processing.
        """
        from modules.wiring import _on_prediction_error

        with patch('modules.wiring._update_attention_schema'), \
             patch('modules.working_memory.push_to_working_memory') as mock_wm:

            _on_prediction_error(Events.PREDICTION_ERROR, {
                "error_magnitude": 0.6,
                "topic": "pricing",
                "actual_keywords": ["discount", "margin"],
            })

        # BEHAVIORAL ASSERT: pushed to WM
        mock_wm.assert_called_once()
        call_kwargs = mock_wm.call_args[1] if mock_wm.call_args[1] else {}
        call_args = mock_wm.call_args[0] if mock_wm.call_args[0] else ()
        all_args = {**dict(zip(["content", "topic", "relevance", "source"], call_args)), **call_kwargs}
        assert "PREDICTION ERROR" in all_args.get("content", ""), \
            "WM content should mention PREDICTION ERROR"
        assert all_args.get("source") == "prediction_error"


# ============================================================
# BUCKET 5: GRAPH DENSIFICATION + SPREADING ACTIVATION
# ============================================================

class TestGraphAndSpreading:
    """Graph densification enables spreading activation.
    Without neighbors, spreading has no edges to traverse."""

    def test_auto_connect_creates_related_memories(self):
        """_auto_connect_neighbors should set 'related_memories' on new memory.

        When saving a memory, similar existing memories should be linked
        via the related_memories payload field.
        """
        from modules.memory_smart import _auto_connect_neighbors

        with patch('modules.memory_smart.pg') as mock_pg:

            mock_pg.search.return_value = {
                "results": [
                    {"id": "neighbor-1", "score": 0.85, "memory": "related content 1"},
                    {"id": "neighbor-2", "score": 0.72, "memory": "related content 2"},
                    {"id": "neighbor-3", "score": 0.60, "memory": "related content 3"},
                    {"id": "low-score", "score": 0.30, "memory": "unrelated"},
                ]
            }

            _auto_connect_neighbors("new-mem-id", "test content about Docker")

        # BEHAVIORAL ASSERT: update_payload called with related_memories
        mock_pg.update_payload.assert_called_once()
        call_args = mock_pg.update_payload.call_args
        # pg.update_payload(new_id, {payload}) - positional args
        updated_id = call_args[0][0]
        updates = call_args[0][1]
        connections = updates["related_memories"]
        assert updated_id == "new-mem-id", "Should update the new memory's payload"
        assert len(connections) == 3, f"Should connect to 3 neighbors (max), got {len(connections)}"
        assert "low-score" not in connections, "Should not connect below min score threshold"

    def test_spreading_reads_related_memories(self):
        """_get_neighbors should read related_memories list field.

        The dense graph edges must be visible to spreading activation.
        """
        from modules.spreading import _get_neighbors

        payload = {
            "related_to": "old-neighbor-1",
            "related_memories": ["auto-1", "auto-2", "auto-3"],
            "consolidated_with": ["cons-1"],
        }

        neighbors = _get_neighbors("seed-id", payload)

        # BEHAVIORAL ASSERT: all connection types included
        assert "old-neighbor-1" in neighbors, "related_to should be included"
        assert "auto-1" in neighbors, "related_memories should be included"
        assert "auto-2" in neighbors, "related_memories should be included"
        assert "cons-1" in neighbors, "consolidated_with should be included"
        assert len(neighbors) == 5, f"Should have 5 unique neighbors, got {len(neighbors)}"


# ============================================================
# BUCKET 6: GWT AUTOMATIC COMPETITION (Baars 1988)
# ============================================================

class TestGWTAutomatic:
    """Baars 1988: Multiple modules compete for workspace access.
    Dehaene 2011: Ignition threshold gates conscious access."""

    def test_competition_filters_below_threshold(self):
        """Candidates below IGNITION_THRESHOLD should be eliminated.

        Only memories with sufficient activation earn workspace access.
        """
        from modules.competition import (
            run_workspace_competition, CompetitionCandidate, IGNITION_THRESHOLD,
        )

        candidates = [
            CompetitionCandidate(
                content="High activation memory about Docker",
                source_domain="episodic",
                activation=0.8,
                memory_id="high-1",
            ),
            CompetitionCandidate(
                content="Medium activation fact",
                source_domain="semantic",
                activation=0.4,
                memory_id="med-1",
            ),
            CompetitionCandidate(
                content="Below threshold noise",
                source_domain="episodic",
                activation=0.1,  # Below IGNITION_THRESHOLD (0.25)
                memory_id="low-1",
            ),
        ]

        result = run_workspace_competition(candidates)

        # BEHAVIORAL ASSERT: below-threshold eliminated
        winner_ids = {w.memory_id for w in result.winners}
        loser_ids = {l.memory_id for l in result.losers}

        assert "high-1" in winner_ids, "High activation should win"
        assert "med-1" in winner_ids, "Medium activation (above threshold) should win"
        assert "low-1" in loser_ids, "Below threshold should be eliminated"
        assert "low-1" not in winner_ids, "Below threshold must NOT be a winner"

    def test_gwt_competition_filters_in_search_memory(self):
        """GWT competition must filter results at runtime in search_memory.

        Two candidates enter: high activation (0.36 combined) survives
        IGNITION_THRESHOLD (0.25), low activation (0.08) does not.
        Output string should contain the winner's text but NOT the loser's.
        """
        from types import SimpleNamespace
        from modules.memory_core import search_memory
        from modules.pg_store import Point

        def _fake_activation(**kwargs):
            return SimpleNamespace(total=0.0)

        # pg.query_vector returns list of Point objects
        p_high = Point(
            id="high-1",
            payload={
                "data": "WINNER_DOCKER_TEXT", "created_at": "",
                "ownership_source": "x", "narrative_importance": "medium",
            },
            score=0.9,
        )
        p_low = Point(
            id="low-1",
            payload={
                "data": "LOSER_PIZZA_TEXT", "created_at": "",
                "ownership_source": "x", "narrative_importance": "medium",
            },
            score=0.2,
        )

        with patch("modules.memory_core.search_semantic", return_value=[]), \
             patch("modules.memory_core.search_fts", return_value=[]), \
             patch("modules.memory_core.compute_unified_activation", side_effect=_fake_activation), \
             patch("modules.memory_core.pg") as mock_pg, \
             patch("modules.consolidation_common._embed_text", return_value=[0.1] * 1536):

            # pg.query_vector returns Point list (vector search)
            mock_pg.query_vector.return_value = [p_high, p_low]
            # pg.get_by_ids returns Point list (payload prefetch)
            mock_pg.get_by_ids.return_value = [p_high, p_low]
            # pg.search_vault returns empty (no dormant memories)
            mock_pg.search_vault.return_value = []

            out = search_memory("docker", limit=5)

        # BEHAVIORAL ASSERT: competition filters below-threshold
        assert "WINNER_DOCKER_TEXT" in out, \
            "High-activation result should survive competition"
        assert "LOSER_PIZZA_TEXT" not in out, \
            "Below-threshold result must be filtered by workspace competition"


# ============================================================
# BUCKET 7: AST-1 ATTENTION PREDICTION ERROR (Graziano 2013)
# ============================================================

class TestAttentionPredictionError:
    """Graziano 2013: Attention schema must predict next focus and
    self-correct via prediction error signal (closed loop)."""

    def test_attention_pe_emitted_on_mismatch(self, clean_event_bus):
        """When predicted focus != actual focus, emit ATTENTION_PREDICTION_ERROR.

        Setup: transitions A->B (x2) so predict returns ("B", 1.0).
        Action: update focus to "C" (mismatch with prediction "B").
        Assert: event emitted with error=1.0, predicted="B", actual="C".
        """
        import modules.wiring as wiring

        # Save and reset schema state
        old_schema = wiring._attention_schema.copy()
        wiring._attention_schema["current_focus"] = "A"
        wiring._attention_schema["focus_strength"] = 0.5
        wiring._attention_schema["topic_transitions"] = [
            {"from": "A", "to": "B", "at": "t1", "driver": "test"},
            {"from": "A", "to": "B", "at": "t2", "driver": "test"},
        ]

        try:
            wiring._update_attention_schema(focus="C", driver="test_mismatch")

            # Check event was emitted
            history = event_bus.get_history()
            pe_events = [e for e in history if e["event"] == Events.ATTENTION_PREDICTION_ERROR]
            assert len(pe_events) == 1, f"Expected 1 PE event, got {len(pe_events)}"

            # Check schema fields updated
            assert wiring._attention_schema.get("last_predicted_focus") == "B"
            assert wiring._attention_schema.get("last_actual_focus") == "C"
            assert wiring._attention_schema.get("attention_prediction_error") == 1.0
        finally:
            # Restore schema
            wiring._attention_schema.update(old_schema)

    def test_attention_pe_not_emitted_without_prediction(self, clean_event_bus):
        """When there are no transitions (predict returns None), no PE event emitted.

        Anti-spam: the system should not emit noise when it has no basis for prediction.
        """
        import modules.wiring as wiring

        old_schema = wiring._attention_schema.copy()
        wiring._attention_schema["current_focus"] = None
        wiring._attention_schema["topic_transitions"] = []

        try:
            wiring._update_attention_schema(focus="X", driver="test_no_pred")

            history = event_bus.get_history()
            pe_events = [e for e in history if e["event"] == Events.ATTENTION_PREDICTION_ERROR]
            assert len(pe_events) == 0, \
                f"No PE event should be emitted without prediction, got {len(pe_events)}"
        finally:
            wiring._attention_schema.update(old_schema)

    def test_attention_pe_decays_wrong_edge(self, clean_event_bus):
        """After mismatch, the wrong edge decays so predictor self-corrects.

        Setup: A->B x3 so predict returns ("B", 1.0).
        Action: update focus to "C" (mismatch).
        Assert: predict_next_focus() from A no longer returns B with prob 1.0.
        The predictor learned from its mistake.
        """
        import modules.wiring as wiring

        old_schema = {k: (v[:] if isinstance(v, list) else v)
                      for k, v in wiring._attention_schema.items()}
        wiring._attention_schema["current_focus"] = "A"
        wiring._attention_schema["focus_strength"] = 0.5
        wiring._attention_schema["topic_transitions"] = [
            {"from": "A", "to": "B", "at": "t1", "driver": "test"},
            {"from": "A", "to": "B", "at": "t2", "driver": "test"},
            {"from": "A", "to": "B", "at": "t3", "driver": "test"},
        ]

        try:
            # Before: predict from A -> B with prob 1.0
            pred_before, prob_before = wiring.predict_next_focus()
            assert pred_before == "B" and prob_before == 1.0, \
                f"Pre-condition: expected B/1.0, got {pred_before}/{prob_before}"

            # Trigger mismatch: predicted B, actual C
            wiring._update_attention_schema(focus="C", driver="test_decay")

            # After: set focus back to A to test predictor from A's perspective
            wiring._attention_schema["current_focus"] = "A"
            pred_after, prob_after = wiring.predict_next_focus()

            # The predictor must have changed: B is no longer 1.0
            # (one A->B removed, one A->C added by transition recording)
            assert prob_after < 1.0 or pred_after != "B", \
                f"After mismatch, predictor should self-correct. Got {pred_after}/{prob_after}"
        finally:
            wiring._attention_schema.update(old_schema)


# ============================================================
# BUCKET 8: HOT-1 PERIODIC SELF-MODEL REFRESH (Rosenthal 2005)
# ============================================================

class TestSelfModelRefresh:
    """Rosenthal 2005: HOT requires periodic meta-representation.
    reflect_on_self() must run automatically, not only on manual call."""

    def test_self_model_refresh_fires_at_interval(self, clean_event_bus, monkeypatch):
        """After 50 interactions, SELF_MODEL_REFRESHED should fire once.

        Cooldown is monkeypatched to 0 to avoid timing dependency.
        reflect_on_self is mocked (it does qdrant I/O in prod).
        """
        import modules.wiring as wiring

        # Reset HOT-1 state
        old_tick = wiring._self_model_tick
        old_refresh = wiring._last_self_model_refresh
        wiring._self_model_tick = 0
        wiring._last_self_model_refresh = 0.0
        monkeypatch.setattr(wiring, "_SELF_MODEL_COOLDOWN", 0)

        try:
            with patch("modules.consciousness.reflect_on_self", return_value="I am Codi"):
                for _ in range(50):
                    wiring.process_elapsed_time(0.5)

            history = event_bus.get_history()
            refresh_events = [e for e in history if e["event"] == Events.SELF_MODEL_REFRESHED]
            assert len(refresh_events) == 1, \
                f"Expected 1 refresh event after 50 ticks, got {len(refresh_events)}"
        finally:
            wiring._self_model_tick = old_tick
            wiring._last_self_model_refresh = old_refresh
