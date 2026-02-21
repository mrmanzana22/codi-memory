#!/usr/bin/env python3
"""
Unit tests for the Global Workspace Competition Engine (WIRING-6).
Run: python3 -m pytest tests/test_competition.py -v
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from modules.competition import (
    CompetitionCandidate,
    CompetitionResult,
    run_workspace_competition,
    apply_recurrent_amplification,
    DEFAULT_WORKSPACE_SLOTS,
    ATTENTION_FOCUS_BONUS,
    IGNITION_THRESHOLD,
    COALITION_TOPIC_BONUS,
    AMPLIFICATION_GAIN,
    LATERAL_INHIBITION,
)


def _make_candidate(domain="episodic", activation=0.5, content="test", **kwargs):
    return CompetitionCandidate(
        content=content,
        source_domain=domain,
        activation=activation,
        memory_id=kwargs.get("memory_id", ""),
        metadata=kwargs.get("metadata", {}),
    )


# ============================================================
# BASIC COMPETITION
# ============================================================

class TestBasicCompetition:
    def test_winners_are_highest_activation(self):
        """Top N by activation should win (top winner gets amplified)."""
        candidates = [
            _make_candidate(activation=0.3),
            _make_candidate(activation=0.9),
            _make_candidate(activation=0.7),
            _make_candidate(activation=0.1),
            _make_candidate(activation=0.5),
        ]
        result = run_workspace_competition(candidates, slots=3)
        assert len(result.winners) == 3
        # Top winner gets amplified (0.9 + 0.15 = 1.0 capped)
        assert result.winners[0].activation == pytest.approx(min(1.0, 0.9 + AMPLIFICATION_GAIN))
        assert result.winners[1].activation == pytest.approx(0.7)
        assert result.winners[2].activation == pytest.approx(0.5)

    def test_losers_are_remaining(self):
        """Everyone below top N is a loser."""
        candidates = [_make_candidate(activation=i * 0.1) for i in range(10)]
        result = run_workspace_competition(candidates, slots=3)
        assert len(result.winners) == 3
        assert len(result.losers) == 7

    def test_fewer_candidates_than_slots(self):
        """If fewer candidates than slots, all are winners."""
        candidates = [_make_candidate(activation=0.5), _make_candidate(activation=0.8)]
        result = run_workspace_competition(candidates, slots=5)
        assert len(result.winners) == 2
        assert len(result.losers) == 0

    def test_empty_candidates(self):
        """Empty input should return empty result, no crash."""
        result = run_workspace_competition([], slots=5)
        assert len(result.winners) == 0
        assert len(result.losers) == 0
        assert result.competition_id != ""


# ============================================================
# SLOT CONFIGURATION
# ============================================================

class TestSlotConfig:
    def test_default_slots(self):
        """Default should be DEFAULT_WORKSPACE_SLOTS."""
        candidates = [_make_candidate(activation=i * 0.1) for i in range(10)]
        result = run_workspace_competition(candidates)
        assert len(result.winners) == DEFAULT_WORKSPACE_SLOTS

    def test_custom_slots(self):
        """Custom slot count should be respected."""
        candidates = [_make_candidate(activation=i * 0.1) for i in range(10)]
        result = run_workspace_competition(candidates, slots=2)
        assert len(result.winners) == 2

    def test_slots_one(self):
        """Single slot = only the absolute winner (amplified)."""
        candidates = [
            _make_candidate(activation=0.3),
            _make_candidate(activation=0.9),
            _make_candidate(activation=0.6),
        ]
        result = run_workspace_competition(candidates, slots=1)
        assert len(result.winners) == 1
        assert result.winners[0].activation == pytest.approx(min(1.0, 0.9 + AMPLIFICATION_GAIN))


# ============================================================
# CROSS-DOMAIN COMPETITION
# ============================================================

class TestCrossDomain:
    def test_different_domains_compete_fairly(self):
        """Highest activation wins regardless of domain."""
        candidates = [
            _make_candidate(domain="episodic", activation=0.4),
            _make_candidate(domain="prospective", activation=0.9),
            _make_candidate(domain="working_memory", activation=0.6),
            _make_candidate(domain="semantic", activation=0.3),
        ]
        result = run_workspace_competition(candidates, slots=2)
        assert result.winners[0].source_domain == "prospective"
        assert result.winners[1].source_domain == "working_memory"

    def test_same_domain_can_dominate(self):
        """No forced diversity -- if one domain has all the best, it wins."""
        candidates = [
            _make_candidate(domain="episodic", activation=0.9),
            _make_candidate(domain="episodic", activation=0.8),
            _make_candidate(domain="episodic", activation=0.7),
            _make_candidate(domain="prospective", activation=0.1),
        ]
        result = run_workspace_competition(candidates, slots=3)
        domains = [w.source_domain for w in result.winners]
        assert domains == ["episodic", "episodic", "episodic"]


# ============================================================
# ATTENTION FOCUS BONUS
# ============================================================

class TestAttentionBonus:
    def test_focus_bonus_applied(self):
        """Candidate matching current focus gets bonus."""
        c1 = _make_candidate(activation=0.50, metadata={"topic": "trading"})
        c2 = _make_candidate(activation=0.52, metadata={"topic": "fullempaques"})
        # Without focus: c2 wins
        result_no_focus = run_workspace_competition([c1, c2], slots=1)
        assert result_no_focus.winners[0].metadata["topic"] == "fullempaques"

        # Reset activations (they were mutated)
        c1.activation = 0.50
        c2.activation = 0.52

        # With focus on "trading": c1 gets +0.12 bonus, beats c2's 0.52
        result_focus = run_workspace_competition([c1, c2], slots=1, current_focus="trading")
        assert result_focus.winners[0].metadata["topic"] == "trading"

    def test_focus_bonus_capped_at_one(self):
        """Activation should not exceed 1.0 after bonus."""
        c = _make_candidate(activation=0.98, metadata={"topic": "trading"})
        run_workspace_competition([c], slots=1, current_focus="trading")
        assert c.activation <= 1.0

    def test_no_focus_no_bonus(self):
        """Without current_focus, no attention bonus (but amplification still applies)."""
        c = _make_candidate(activation=0.50, metadata={"topic": "trading"})
        run_workspace_competition([c], slots=1, current_focus=None)
        # No attention bonus, but top winner gets amplification
        assert c.activation == pytest.approx(0.50 + AMPLIFICATION_GAIN)


# ============================================================
# RESULT STRUCTURE
# ============================================================

class TestResultStructure:
    def test_result_has_required_fields(self):
        """CompetitionResult should have all fields."""
        result = run_workspace_competition([_make_candidate()], slots=1)
        assert isinstance(result, CompetitionResult)
        assert isinstance(result.winners, list)
        assert isinstance(result.losers, list)
        assert result.timestamp != ""
        assert result.competition_id != ""

    def test_candidate_structure(self):
        """CompetitionCandidate should preserve all fields."""
        c = CompetitionCandidate(
            content="test memory",
            source_domain="episodic",
            activation=0.75,
            memory_id="abc-123",
            metadata={"topic": "trading", "importance": "high"},
        )
        assert c.content == "test memory"
        assert c.source_domain == "episodic"
        assert c.activation == 0.75
        assert c.memory_id == "abc-123"
        assert c.metadata["topic"] == "trading"


# ============================================================
# IGNITION THRESHOLD (P0-COMP-1)
# ============================================================

class TestIgnitionThreshold:
    def test_below_threshold_rejected(self):
        """Candidates below IGNITION_THRESHOLD should never be winners."""
        candidates = [
            _make_candidate(activation=0.1),
            _make_candidate(activation=0.2),
            _make_candidate(activation=0.05),
        ]
        result = run_workspace_competition(candidates, slots=5)
        assert len(result.winners) == 0
        assert len(result.losers) == 3

    def test_threshold_boundary(self):
        """Candidate at exactly IGNITION_THRESHOLD should be a winner (amplified)."""
        candidates = [
            _make_candidate(activation=IGNITION_THRESHOLD),
            _make_candidate(activation=0.1),
        ]
        result = run_workspace_competition(candidates, slots=5)
        assert len(result.winners) == 1
        assert result.winners[0].activation == pytest.approx(IGNITION_THRESHOLD + AMPLIFICATION_GAIN)

    def test_mix_above_below(self):
        """Only above-threshold candidates compete for slots."""
        candidates = [
            _make_candidate(activation=0.1),   # below
            _make_candidate(activation=0.5),   # above
            _make_candidate(activation=0.8),   # above
            _make_candidate(activation=0.2),   # below
        ]
        result = run_workspace_competition(candidates, slots=1)
        assert len(result.winners) == 1
        assert result.winners[0].activation == pytest.approx(0.8 + AMPLIFICATION_GAIN)
        assert len(result.losers) == 3  # 1 above-threshold loser + 2 below

    def test_threshold_value(self):
        """IGNITION_THRESHOLD should be 0.25."""
        assert IGNITION_THRESHOLD == 0.25


# ============================================================
# COALITION FORMATION (P1-COMP-2)
# ============================================================

class TestCoalitionFormation:
    def test_coalition_boosts_shared_topic(self):
        """Candidates from 2+ domains on same topic get COALITION_TOPIC_BONUS."""
        c1 = _make_candidate(domain="episodic", activation=0.40, metadata={"topic": "trading"})
        c2 = _make_candidate(domain="prospective", activation=0.40, metadata={"topic": "trading"})
        c3 = _make_candidate(domain="semantic", activation=0.50, metadata={"topic": "fullempaques"})

        result = run_workspace_competition([c1, c2, c3], slots=2)
        # c1 and c2 share "trading" from 2 domains -> each gets +0.10 -> 0.50
        # c3 is alone on "fullempaques" -> stays 0.50
        # All three at 0.50, but c1 and c2 got boosted. Winners are the coalition members.
        winner_topics = [w.metadata.get("topic") for w in result.winners]
        assert winner_topics.count("trading") == 2

    def test_no_coalition_same_domain(self):
        """Same domain on same topic does NOT form coalition (need 2+ domains)."""
        c1 = _make_candidate(domain="episodic", activation=0.40, metadata={"topic": "trading"})
        c2 = _make_candidate(domain="episodic", activation=0.40, metadata={"topic": "trading"})
        c3 = _make_candidate(domain="semantic", activation=0.45, metadata={"topic": "fullempaques"})

        result = run_workspace_competition([c1, c2, c3], slots=1)
        # c1 and c2 are same domain, no coalition bonus
        # c3 at 0.45 > c1/c2 at 0.40
        assert result.winners[0].metadata["topic"] == "fullempaques"

    def test_coalition_bonus_value(self):
        """COALITION_TOPIC_BONUS should be 0.10."""
        assert COALITION_TOPIC_BONUS == 0.10


# ============================================================
# RECURRENT AMPLIFICATION (GWT-4)
# ============================================================

class TestRecurrentAmplification:
    def test_winner_boosted(self):
        """Top winner gets AMPLIFICATION_GAIN boost."""
        winner = _make_candidate(activation=0.5)
        loser = _make_candidate(activation=0.3)
        apply_recurrent_amplification([winner], [loser])
        assert winner.activation == pytest.approx(0.5 + AMPLIFICATION_GAIN)

    def test_losers_suppressed(self):
        """All losers get LATERAL_INHIBITION penalty."""
        winner = _make_candidate(activation=0.8)
        losers = [_make_candidate(activation=0.4), _make_candidate(activation=0.3)]
        apply_recurrent_amplification([winner], losers)
        assert losers[0].activation == pytest.approx(0.4 - LATERAL_INHIBITION)
        assert losers[1].activation == pytest.approx(0.3 - LATERAL_INHIBITION)

    def test_winner_capped_at_one(self):
        """Winner activation cannot exceed 1.0."""
        winner = _make_candidate(activation=0.95)
        apply_recurrent_amplification([winner], [])
        assert winner.activation <= 1.0

    def test_loser_floor_at_zero(self):
        """Loser activation cannot go below 0.0."""
        loser = _make_candidate(activation=0.05)
        apply_recurrent_amplification([_make_candidate(activation=0.8)], [loser])
        assert loser.activation >= 0.0

    def test_only_top_winner_amplified(self):
        """Only winners[0] gets amplified, not other winners."""
        w1 = _make_candidate(activation=0.8)
        w2 = _make_candidate(activation=0.7)
        apply_recurrent_amplification([w1, w2], [])
        assert w1.activation == pytest.approx(0.8 + AMPLIFICATION_GAIN)
        assert w2.activation == pytest.approx(0.7)  # Unchanged

    def test_empty_winners_no_crash(self):
        """No crash with empty winners."""
        apply_recurrent_amplification([], [_make_candidate()])

    def test_integrated_in_competition(self):
        """run_workspace_competition() applies amplification automatically."""
        c1 = _make_candidate(activation=0.8)
        c2 = _make_candidate(activation=0.5)
        c3 = _make_candidate(activation=0.3)
        result = run_workspace_competition([c1, c2, c3], slots=1)
        # Winner should be boosted above original 0.8
        assert result.winners[0].activation > 0.8
        # Losers should be suppressed below original
        for loser in result.losers:
            if loser.activation > 0:  # Some may have hit floor
                assert loser.activation < 0.5

    def test_amplification_constants(self):
        """Constants should be within expected ranges."""
        assert 0.1 <= AMPLIFICATION_GAIN <= 0.3
        assert 0.05 <= LATERAL_INHIBITION <= 0.25


# ============================================================
# CYCLE TRACKING (GWT-6)
# ============================================================

class TestCycleTracking:
    def setup_method(self):
        """Reset workspace state before each test."""
        from modules.workspace import _global_workspace
        _global_workspace['cycle_count'] = 0
        _global_workspace['cycle_history'] = []
        _global_workspace['last_cycle_timestamp'] = None

    def test_record_increments_count(self):
        from modules.workspace import record_workspace_cycle, _global_workspace
        record_workspace_cycle("test_theme", "test content")
        assert _global_workspace['cycle_count'] == 1
        record_workspace_cycle("theme2", "content2")
        assert _global_workspace['cycle_count'] == 2

    def test_record_stores_history(self):
        from modules.workspace import record_workspace_cycle, _global_workspace
        record_workspace_cycle("trading", "BTC analysis results")
        assert len(_global_workspace['cycle_history']) == 1
        entry = _global_workspace['cycle_history'][0]
        assert entry['theme'] == "trading"
        assert entry['content'] == "BTC analysis results"
        assert entry['cycle'] == 1
        assert entry['timestamp'] is not None

    def test_history_max_10(self):
        from modules.workspace import record_workspace_cycle, _global_workspace
        for i in range(15):
            record_workspace_cycle(f"theme_{i}", f"content_{i}")
        assert len(_global_workspace['cycle_history']) == 10
        # Oldest should be trimmed
        assert _global_workspace['cycle_history'][0]['cycle'] == 6

    def test_content_truncated(self):
        from modules.workspace import record_workspace_cycle, _global_workspace
        long_content = "x" * 200
        record_workspace_cycle("theme", long_content)
        assert len(_global_workspace['cycle_history'][0]['content']) == 100

    def test_timestamp_set(self):
        from modules.workspace import record_workspace_cycle, _global_workspace
        record_workspace_cycle("theme", "content")
        assert _global_workspace['last_cycle_timestamp'] is not None


# ============================================================
# RECRUITMENT EVENT (GWT-5)
# ============================================================

class TestRecruitmentEvent:
    def test_event_constant_exists(self):
        from modules.events import Events
        assert hasattr(Events, 'WORKSPACE_RECRUITMENT')
        assert Events.WORKSPACE_RECRUITMENT == 'workspace_recruitment'

    def test_handler_registered(self):
        """After wiring, recruitment handler should be registered."""
        from modules.events import event_bus, Events
        from modules.wiring import wire_event_bus
        wire_event_bus()
        stats = event_bus.get_stats()
        assert Events.WORKSPACE_RECRUITMENT in stats


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
