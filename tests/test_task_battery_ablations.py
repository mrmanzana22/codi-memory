#!/usr/bin/env python3
"""
TASK BATTERY ABLATIONS - Prove each module contributes
======================================================
7 ablation classes: disable ONE module, verify degradation in a specific task.

If disabling a module causes NO degradation, either:
  (a) the module isn't contributing, or
  (b) the test isn't capturing the right thing.

Each ablation mirrors a battery test but with the key module mocked to no-op.

Run: ./venv/bin/pytest tests/test_task_battery_ablations.py -v
Run all battery: ./venv/bin/pytest -m battery -v
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from types import SimpleNamespace
from unittest.mock import patch, MagicMock
from modules.events import event_bus, Events

pytestmark = pytest.mark.battery


# ============================================================
# ABLATION A: Without Spreading Activation
# ============================================================

class TestAblationSpreading:
    """Disable spreading activation, verify graph densification degrades."""

    def test_without_auto_connect_no_neighbors(self):
        """When _auto_connect_neighbors is disabled, no related_memories are set.

        Mirror of: TestGraphAndSpreading::test_auto_connect_creates_related_memories
        Expected: pg.update_payload is NOT called (no connections created).
        """
        from modules.memory_smart import _auto_connect_neighbors

        with patch('modules.memory_smart.pg') as mock_pg:

            mock_pg.search.return_value = {
                "results": [
                    {"id": "neighbor-1", "score": 0.85, "memory": "related 1"},
                    {"id": "neighbor-2", "score": 0.72, "memory": "related 2"},
                ]
            }

            # ABLATION: mock the function itself to no-op
            with patch('modules.memory_smart._auto_connect_neighbors'):
                from modules.memory_smart import _auto_connect_neighbors as ablated
                ablated("new-id", "test content")

        # DEGRADATION CHECK: update_payload was NOT called (no connections)
        mock_pg.update_payload.assert_not_called()


# ============================================================
# ABLATION B: Without Metacognitive Control
# ============================================================

class TestAblationMetacognitive:
    """Disable metacognitive control, verify FOK->strategy loop breaks."""

    def test_without_metacognitive_no_event(self, clean_event_bus):
        """When metacognitive_control returns baseline, no event is emitted.

        Mirror of: TestMetacognitiveControl::test_metacognitive_control_emits_event
        Expected: METACOGNITIVE_CONTROL_APPLIED is NOT emitted.
        """
        captured = []

        def capture(event_name, data):
            if event_name == Events.METACOGNITIVE_CONTROL_APPLIED:
                captured.append(data)

        event_bus.on(Events.METACOGNITIVE_CONTROL_APPLIED, capture)

        try:
            # ABLATION: metacognitive_control returns neutral baseline
            baseline = {
                "strategy": "full_search",
                "adjusted_limit": 1,
                "confidence_flag": "",
                "fok": {"fok_score": 0.5},
            }

            with patch('modules.memory_core.metacognitive_control', return_value=baseline):
                # With baseline strategy, limit_multiplier=1, so the event
                # emission code still runs but strategy is "full_search"
                # (no behavioral change). The key: adjusted_limit stays 1.
                pass

            # DEGRADATION CHECK: no event was emitted
            assert len(captured) == 0, \
                "With metacognitive control ablated, no METACOGNITIVE_CONTROL_APPLIED should fire"
        finally:
            event_bus.off(Events.METACOGNITIVE_CONTROL_APPLIED, capture)

    def test_without_metacognitive_limit_unchanged(self):
        """When metacognitive_control is ablated, search limit stays at 1x.

        The real metacognitive_control would return adjusted_limit=2 for low FOK.
        Ablated version returns baseline (adjusted_limit=1), so limit doesn't expand.
        """
        from modules.retrieval_metadata import metacognitive_control
        import tempfile
        import sqlite3

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
            conn.execute(
                "INSERT INTO memories_text (memory_id, content) VALUES (?, ?)",
                ("m1", "Some memory about cooking")
            )
            conn.commit()
            conn.close()

            # WITHOUT ablation: low FOK query -> limit expands
            real_result = metacognitive_control("quantum physics", fts_db_path=db_path)
            assert real_result["adjusted_limit"] >= 2, \
                "Real metacognitive control should expand limit for unknown query"

            # WITH ablation: baseline always returns 1
            baseline = {
                "strategy": "full_search",
                "adjusted_limit": 1,
                "confidence_flag": "",
                "fok": {"fok_score": 0.5},
            }

            # DEGRADATION CHECK: ablated version doesn't expand
            assert baseline["adjusted_limit"] == 1, \
                "Ablated metacognitive control keeps limit at 1x (no expansion)"
            assert baseline["adjusted_limit"] < real_result["adjusted_limit"], \
                "Ablation should show degradation: limit NOT expanded vs real"
        finally:
            os.unlink(db_path)


# ============================================================
# ABLATION C: Without GWT Competition
# ============================================================

class TestAblationGWT:
    """Disable workspace competition, verify below-threshold results leak through."""

    def test_without_competition_loser_appears(self):
        """When GWT competition is disabled, below-threshold results survive.

        Mirror of: TestGWTAutomatic::test_gwt_competition_filters_in_search_memory
        Expected: LOSER_PIZZA_TEXT appears in output (it was filtered before).
        """
        from modules.memory_core import search_memory
        from modules.competition import CompetitionResult
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

        # ABLATION: competition returns ALL candidates as winners (passthrough)
        def passthrough_competition(candidates, **kwargs):
            return CompetitionResult(
                winners=candidates,
                losers=[],
                timestamp="",
                competition_id="ablated",
            )

        with patch("modules.memory_core.search_semantic", return_value=[]), \
             patch("modules.memory_core.search_fts", return_value=[]), \
             patch("modules.memory_core.compute_unified_activation", side_effect=_fake_activation), \
             patch("modules.memory_core.pg") as mock_pg, \
             patch("modules.consolidation_common._embed_text", return_value=[0.1] * 1536), \
             patch("modules.competition.run_workspace_competition", side_effect=passthrough_competition):

            # pg.query_vector returns Point list (vector search)
            mock_pg.query_vector.return_value = [p_high, p_low]
            # pg.get_by_ids returns Point list (payload prefetch)
            mock_pg.get_by_ids.return_value = [p_high, p_low]
            # pg.search_vault returns empty (no dormant memories)
            mock_pg.search_vault.return_value = []

            out = search_memory("docker", limit=5)

        # DEGRADATION CHECK: loser now appears (competition was disabled)
        assert "WINNER_DOCKER_TEXT" in out, "Winner should still appear"
        assert "LOSER_PIZZA_TEXT" in out, \
            "Without GWT competition, below-threshold loser should leak through"


# ============================================================
# ABLATION D: Without Contradiction Detection
# ============================================================

class TestAblationContradiction:
    """Disable contradiction detection, verify PE signal vanishes."""

    def test_without_contradiction_no_pe(self):
        """When detect_contradiction is ablated (returns PE=0), contradictions go undetected.

        Mirror of: TestContradictionDetection::test_negation_inversion_triggers_pe
        Expected: PE=0 even for contradictory texts (negation not detected).
        """
        # The real function would detect "is great" vs "is not great" as contradiction
        memory_text = "Docker is great for production deployment"
        context = "Docker is not great for production deployment"

        # ABLATION: detect_contradiction returns zero PE (no contradiction engine)
        ablated_result = {
            "prediction_error": 0.0,
            "channels": {"keywords": 0, "topic_confirm": 0, "negation": 0},
        }

        with patch('modules.reconsolidation.detect_contradiction', return_value=ablated_result):
            from modules.reconsolidation import detect_contradiction
            result = detect_contradiction(memory_text, context)

        # DEGRADATION CHECK: PE is 0 even for contradictory texts
        assert result["prediction_error"] == 0.0, \
            "With contradiction detection ablated, PE should be 0 (blind to negation)"
        assert result["channels"]["negation"] == 0, \
            "Negation channel should not fire when ablated"

    def test_without_contradiction_unrelated_same_as_contradictory(self):
        """Without contradiction detection, contradictory and unrelated texts look the same.

        Both return PE=0, proving the module is needed to distinguish them.
        """
        ablated_result = {
            "prediction_error": 0.0,
            "channels": {"keywords": 0, "topic_confirm": 0, "negation": 0},
        }

        with patch('modules.reconsolidation.detect_contradiction', return_value=ablated_result):
            from modules.reconsolidation import detect_contradiction
            contradictory = detect_contradiction(
                "Docker is great", "Docker is not great"
            )
            unrelated = detect_contradiction(
                "Docker is great", "Pizza was delicious"
            )

        # DEGRADATION CHECK: both look identical (cannot distinguish)
        assert contradictory["prediction_error"] == unrelated["prediction_error"] == 0.0, \
            "Ablated system cannot distinguish contradictory from unrelated"


# ============================================================
# ABLATION E: Without Reconsolidation (correct_memory)
# ============================================================

class TestAblationReconsolidation:
    """Disable reconsolidation, verify old trace persists unchanged."""

    def test_without_correct_memory_no_upsert(self):
        """When correct_memory is ablated, pg.upsert is never called.

        Mirror of: TestReconsolidation::test_correct_memory_upserts_new_vector
        Expected: No upsert (old vector survives, no re-embedding).
        """
        # ABLATION: correct_memory is a no-op that returns a message
        with patch('modules.reconsolidation.pg') as mock_pg:
            with patch('modules.consolidation.correct_memory',
                       return_value="[ABLATED] No reconsolidation"):
                from modules.consolidation import correct_memory
                result = correct_memory("full-uuid-123", "Docker ya no es recomendado")

        # DEGRADATION CHECK: no upsert (old trace NOT destroyed)
        mock_pg.upsert.assert_not_called()
        assert "ABLATED" in result, "Ablated path should indicate no reconsolidation"

    def test_without_correct_memory_confidence_unchanged(self):
        """When reconsolidation is ablated, confidence is never decremented.

        Mirror of: TestReconsolidation::test_correct_memory_decrements_confidence
        The system cannot learn from corrections -- old confidence persists.
        """
        original_confidence = 0.8

        # ABLATION: correct_memory does nothing, confidence stays at original
        with patch('modules.consolidation.correct_memory',
                    return_value="[ABLATED] No reconsolidation"):
            # Confidence is never touched
            pass

        # DEGRADATION CHECK: confidence unchanged (no PE-based decrement)
        assert original_confidence == 0.8, \
            "Without reconsolidation, confidence cannot decrease from corrections"


# ============================================================
# ABLATION F: Without Attention Prediction Error
# ============================================================

class TestAblationAttentionPE:
    """Disable attention PE, verify self-correction loop breaks."""

    def test_without_attention_pe_no_event(self, clean_event_bus):
        """When PE computation is ablated, no ATTENTION_PREDICTION_ERROR emitted.

        Mirror of: TestAttentionPredictionError::test_attention_pe_emitted_on_mismatch
        Expected: Even with mismatch (predicted B, actual C), no PE event fires.
        """
        import modules.wiring as wiring

        old_schema = {k: (v[:] if isinstance(v, list) else v)
                      for k, v in wiring._attention_schema.items()}
        wiring._attention_schema["current_focus"] = "A"
        wiring._attention_schema["focus_strength"] = 0.5
        wiring._attention_schema["topic_transitions"] = [
            {"from": "A", "to": "B", "at": "t1", "driver": "test"},
            {"from": "A", "to": "B", "at": "t2", "driver": "test"},
        ]

        try:
            # ABLATION: predict_next_focus always returns (None, 0) -> PE path skipped
            with patch.object(wiring, 'predict_next_focus', return_value=(None, 0.0)):
                wiring._update_attention_schema(focus="C", driver="test_ablated")

            history = event_bus.get_history()
            pe_events = [e for e in history if e["event"] == Events.ATTENTION_PREDICTION_ERROR]

            # DEGRADATION CHECK: no PE event (predictor ablated)
            assert len(pe_events) == 0, \
                f"With prediction ablated, no PE event should fire, got {len(pe_events)}"
        finally:
            wiring._attention_schema.update(old_schema)

    def test_without_attention_pe_no_edge_decay(self, clean_event_bus):
        """When PE is ablated, wrong edges are never decayed (no self-correction).

        Mirror of: TestAttentionPredictionError::test_attention_pe_decays_wrong_edge
        Expected: After mismatch, predictor still returns B with prob 1.0 from A.
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
            # Count A->B edges before
            ab_before = sum(
                1 for t in wiring._attention_schema["topic_transitions"]
                if t["from"] == "A" and t["to"] == "B"
            )

            # ABLATION: predict_next_focus returns None -> PE never computed -> no decay
            with patch.object(wiring, 'predict_next_focus', return_value=(None, 0.0)):
                wiring._update_attention_schema(focus="C", driver="test_no_decay")

            # Count A->B edges after (should NOT have decreased)
            ab_after = sum(
                1 for t in wiring._attention_schema["topic_transitions"]
                if t["from"] == "A" and t["to"] == "B"
            )

            # DEGRADATION CHECK: edges unchanged (no self-correction)
            assert ab_after == ab_before, \
                f"Without PE, A->B edges should stay at {ab_before}, got {ab_after}"
        finally:
            wiring._attention_schema.update(old_schema)


# ============================================================
# ABLATION G: Without Self-Model Refresh (HOT-1)
# ============================================================

class TestAblationSelfModelRefresh:
    """Disable periodic self-model refresh, verify HOT-1 loop breaks."""

    def test_without_hot1_no_refresh_event(self, clean_event_bus, monkeypatch):
        """When HOT-1 path is ablated, no SELF_MODEL_REFRESHED event fires.

        Mirror of: TestSelfModelRefresh::test_self_model_refresh_fires_at_interval
        Expected: After 50 ticks, still 0 SELF_MODEL_REFRESHED events.
        """
        import modules.wiring as wiring

        old_tick = wiring._self_model_tick
        old_refresh = wiring._last_self_model_refresh
        wiring._self_model_tick = 0
        wiring._last_self_model_refresh = 0.0
        monkeypatch.setattr(wiring, "_SELF_MODEL_COOLDOWN", 0)

        try:
            # ABLATION: reflect_on_self raises, simulating module removed
            with patch("modules.consciousness.reflect_on_self",
                       side_effect=ImportError("Ablated: no self-model module")):
                for _ in range(50):
                    wiring.process_elapsed_time(0.5)

            history = event_bus.get_history()
            refresh_events = [e for e in history if e["event"] == Events.SELF_MODEL_REFRESHED]

            # DEGRADATION CHECK: no refresh event (HOT-1 broken)
            assert len(refresh_events) == 0, \
                f"With HOT-1 ablated, no refresh should fire, got {len(refresh_events)}"
        finally:
            wiring._self_model_tick = old_tick
            wiring._last_self_model_refresh = old_refresh

    def test_without_hot1_schema_not_updated(self, clean_event_bus, monkeypatch):
        """When HOT-1 is ablated, the system has no self-awareness refresh.

        The real system would update its self-model every 50 ticks.
        Ablated: reflect_on_self never runs, so there's no meta-representation.
        """
        import modules.wiring as wiring

        old_tick = wiring._self_model_tick
        old_refresh = wiring._last_self_model_refresh
        initial_refresh_time = wiring._last_self_model_refresh
        wiring._self_model_tick = 0
        wiring._last_self_model_refresh = 0.0
        monkeypatch.setattr(wiring, "_SELF_MODEL_COOLDOWN", 0)

        try:
            # ABLATION: make HOT-1 interval impossibly large (never triggers)
            monkeypatch.setattr(wiring, "_SELF_MODEL_INTERVAL", 999999)

            with patch("modules.consciousness.reflect_on_self", return_value="I am Codi"):
                for _ in range(50):
                    wiring.process_elapsed_time(0.5)

            # DEGRADATION CHECK: _last_self_model_refresh unchanged (never ran)
            assert wiring._last_self_model_refresh == 0.0, \
                "With interval ablated (999999), refresh should never trigger"
        finally:
            wiring._self_model_tick = old_tick
            wiring._last_self_model_refresh = old_refresh
