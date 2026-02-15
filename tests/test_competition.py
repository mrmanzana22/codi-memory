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
    DEFAULT_WORKSPACE_SLOTS,
    ATTENTION_FOCUS_BONUS,
    IGNITION_THRESHOLD,
    COALITION_TOPIC_BONUS,
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
        """Top N by activation should win."""
        candidates = [
            _make_candidate(activation=0.3),
            _make_candidate(activation=0.9),
            _make_candidate(activation=0.7),
            _make_candidate(activation=0.1),
            _make_candidate(activation=0.5),
        ]
        result = run_workspace_competition(candidates, slots=3)
        assert len(result.winners) == 3
        assert result.winners[0].activation == 0.9
        assert result.winners[1].activation == 0.7
        assert result.winners[2].activation == 0.5

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
        """Single slot = only the absolute winner."""
        candidates = [
            _make_candidate(activation=0.3),
            _make_candidate(activation=0.9),
            _make_candidate(activation=0.6),
        ]
        result = run_workspace_competition(candidates, slots=1)
        assert len(result.winners) == 1
        assert result.winners[0].activation == 0.9


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
        """Without current_focus, no bonus applied."""
        c = _make_candidate(activation=0.50, metadata={"topic": "trading"})
        run_workspace_competition([c], slots=1, current_focus=None)
        assert c.activation == 0.50


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
        """Candidate at exactly IGNITION_THRESHOLD should be a winner."""
        candidates = [
            _make_candidate(activation=IGNITION_THRESHOLD),
            _make_candidate(activation=0.1),
        ]
        result = run_workspace_competition(candidates, slots=5)
        assert len(result.winners) == 1
        assert result.winners[0].activation == IGNITION_THRESHOLD

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
        assert result.winners[0].activation == 0.8
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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
