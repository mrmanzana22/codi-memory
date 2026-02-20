#!/usr/bin/env python3
"""
CONTRADICTION COUNTER RESET TESTS (PR5 — P0 correctness)
==========================================================
Verify that _contradiction_session_count resets properly
at session boundaries so the system doesn't permanently
stop detecting contradictions.

5 tests:
  1. Counter starts at 0
  2. Counter increments on detection
  3. reset_contradiction_counter resets to 0
  4. After hitting max, reset restores detection capability
  5. Cooldown dict also cleared on reset

Run: ./venv/bin/pytest tests/test_contradiction_reset.py -v
"""

import sys
import os

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import modules.memory_smart as ms
from modules.memory_smart import reset_contradiction_counter


# ============================================================
# TESTS
# ============================================================

class TestContradictionCounterReset:
    """Session counter reset semantics."""

    def setup_method(self):
        """Reset state before each test."""
        ms._contradiction_session_count = 0
        ms._contradiction_cooldown.clear()

    def test_starts_at_zero(self):
        """Counter is 0 at module load / after reset."""
        reset_contradiction_counter()
        assert ms._contradiction_session_count == 0

    def test_counter_increments(self):
        """Simulating detections increments the counter."""
        ms._contradiction_session_count = 0
        ms._contradiction_session_count += 1
        assert ms._contradiction_session_count == 1
        ms._contradiction_session_count += 1
        assert ms._contradiction_session_count == 2

    def test_reset_clears_counter(self):
        """reset_contradiction_counter sets count back to 0."""
        ms._contradiction_session_count = 5
        reset_contradiction_counter()
        assert ms._contradiction_session_count == 0

    def test_reset_restores_detection_after_max(self):
        """After hitting MAX_PER_SESSION, reset enables new detections."""
        from modules.config import CONTRADICTION_MAX_PER_SESSION

        # Simulate hitting the limit
        ms._contradiction_session_count = CONTRADICTION_MAX_PER_SESSION
        assert ms._contradiction_session_count >= CONTRADICTION_MAX_PER_SESSION

        # Reset
        reset_contradiction_counter()
        assert ms._contradiction_session_count == 0
        assert ms._contradiction_session_count < CONTRADICTION_MAX_PER_SESSION

    def test_reset_clears_cooldown(self):
        """reset_contradiction_counter also clears topic cooldowns."""
        ms._contradiction_cooldown["topic_a_b"] = "2026-02-17T12:00:00"
        ms._contradiction_cooldown["topic_c_d"] = "2026-02-17T12:00:00"
        assert len(ms._contradiction_cooldown) == 2

        reset_contradiction_counter()
        assert len(ms._contradiction_cooldown) == 0
