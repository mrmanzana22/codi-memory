"""
Tests for PR-1: Consolidation Quality Fixes
=============================================
Covers:
  1. Extraction prompt is in Spanish with anti-basura filter
  2. Contradiction thresholds allow lower-PE detections
  3. Lockfile prevents concurrent consolidation runs

Run: ./venv/bin/pytest tests/test_consolidation_quality.py -v
"""

import os
import sys
import fcntl
import json
import tempfile
from unittest.mock import patch, MagicMock

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ============================================================
# 1. EXTRACTION PROMPT IN SPANISH
# ============================================================

class TestExtractionPromptSpanish:
    """Verify the extraction prompt generates Spanish instructions."""

    def test_prompt_is_in_spanish(self):
        """The extraction prompt must contain Spanish instructions."""
        from modules.consolidation import _build_extraction_prompt

        prompt = _build_extraction_prompt(
            topic="trading",
            episodes_block="- Hare compro BTC\n- El mercado subio",
            num_episodes=2,
        )

        # Key Spanish phrases that must be present
        assert "IDIOMA: Responde SIEMPRE en espanol" in prompt
        assert "HECHOS SEMANTICOS" in prompt
        assert "CATEGORIAS" in prompt
        assert "EJEMPLOS DE BUENOS HECHOS" in prompt
        assert "REGLAS" in prompt

    def test_prompt_contains_anti_basura_filter(self):
        """Prompt must include the anti-implementation-details filter."""
        from modules.consolidation import _build_extraction_prompt

        prompt = _build_extraction_prompt("test", "- episode", 1)

        assert "FILTRO ANTI-BASURA" in prompt
        assert "lineas de codigo" in prompt.lower()
        assert "errores temporales" in prompt.lower()
        assert "estados transitorios" in prompt.lower()

    def test_prompt_does_not_contain_english_instructions(self):
        """Prompt must NOT contain English system instructions."""
        from modules.consolidation import _build_extraction_prompt

        prompt = _build_extraction_prompt("test", "- episode", 1)

        assert "You are a memory consolidation system" not in prompt
        assert "EXAMPLES OF GOOD FACTS" not in prompt
        assert "EXAMPLES OF BAD FACTS" not in prompt

    def test_prompt_contains_spanish_examples(self):
        """Examples in the prompt must be in Spanish."""
        from modules.consolidation import _build_extraction_prompt

        prompt = _build_extraction_prompt("test", "- episode", 1)

        assert "El workflow TIAW-MainSync" in prompt
        assert "Hare prefiere revisar" in prompt

    def test_prompt_includes_topic_and_episodes(self):
        """Prompt must interpolate the topic and episodes."""
        from modules.consolidation import _build_extraction_prompt

        prompt = _build_extraction_prompt(
            topic="fullempaques",
            episodes_block="- Episode 1\n- Episode 2",
            num_episodes=2,
        )

        assert '"fullempaques"' in prompt
        assert "- Episode 1" in prompt
        assert "NUMERO DE EPISODIOS: 2" in prompt


# ============================================================
# 2. CONTRADICTION THRESHOLDS
# ============================================================

class TestContradictionThresholds:
    """Verify lowered contradiction detection thresholds."""

    def test_score_floor_lowered(self):
        """CONTRADICTION_SCORE_FLOOR should be 0.35 (was 0.50)."""
        from modules.config import CONTRADICTION_SCORE_FLOOR
        assert CONTRADICTION_SCORE_FLOOR == 0.35

    def test_pe_silent_lowered(self):
        """CONTRADICTION_PE_SILENT should be 0.20 (was 0.30)."""
        from modules.config import CONTRADICTION_PE_SILENT
        assert CONTRADICTION_PE_SILENT == 0.20

    def test_min_entities_lowered(self):
        """CONTRADICTION_MIN_ENTITIES should be 1 (was 2)."""
        from modules.config import CONTRADICTION_MIN_ENTITIES
        assert CONTRADICTION_MIN_ENTITIES == 1

    def test_pe_alert_unchanged(self):
        """CONTRADICTION_PE_ALERT should remain 0.50."""
        from modules.config import CONTRADICTION_PE_ALERT
        assert CONTRADICTION_PE_ALERT == 0.50

    def test_pe_critical_unchanged(self):
        """CONTRADICTION_PE_CRITICAL should remain 0.70."""
        from modules.config import CONTRADICTION_PE_CRITICAL
        assert CONTRADICTION_PE_CRITICAL == 0.70

    def test_inline_check_accepts_lower_similarity(self):
        """Candidates with similarity >= 0.35 should pass the floor check."""
        from modules.config import CONTRADICTION_SCORE_FLOOR
        # A candidate with score=0.40 should pass (above 0.35)
        assert 0.40 >= CONTRADICTION_SCORE_FLOOR
        # A candidate with score=0.30 should fail (below 0.35)
        assert 0.30 < CONTRADICTION_SCORE_FLOOR

    def test_single_entity_passes_min_check(self):
        """A single shared entity should now pass the min_entities gate."""
        from modules.config import CONTRADICTION_MIN_ENTITIES
        shared_entities = {"qdrant"}
        assert len(shared_entities) >= CONTRADICTION_MIN_ENTITIES


# ============================================================
# 3. LOCKFILE
# ============================================================

class TestConsolidationLockfile:
    """Verify lockfile prevents concurrent consolidation runs."""

    def test_acquire_and_release_lock(self, tmp_path):
        """Lock can be acquired and released cleanly."""
        # Import the functions (they're in the script, not a module)
        script_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "scripts",
        )
        sys.path.insert(0, script_dir)

        import run_consolidation as rc

        # Patch LOCK_FILE to use tmp_path
        original_lock = rc.LOCK_FILE
        rc.LOCK_FILE = str(tmp_path / "test.lock")

        try:
            lock_fd = rc._acquire_lock()
            assert lock_fd is not None

            rc._release_lock(lock_fd)
            assert True  # No exception = success
        finally:
            rc.LOCK_FILE = original_lock

    def test_second_acquire_fails_while_locked(self, tmp_path):
        """Second lock attempt fails while first is held."""
        script_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "scripts",
        )
        if script_dir not in sys.path:
            sys.path.insert(0, script_dir)

        import run_consolidation as rc

        original_lock = rc.LOCK_FILE
        rc.LOCK_FILE = str(tmp_path / "test.lock")

        try:
            lock1 = rc._acquire_lock()
            assert lock1 is not None

            lock2 = rc._acquire_lock()
            assert lock2 is None  # Should fail

            rc._release_lock(lock1)
        finally:
            rc.LOCK_FILE = original_lock

    def test_lock_released_after_first_holder_releases(self, tmp_path):
        """After releasing lock, a new acquire should succeed."""
        script_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "scripts",
        )
        if script_dir not in sys.path:
            sys.path.insert(0, script_dir)

        import run_consolidation as rc

        original_lock = rc.LOCK_FILE
        rc.LOCK_FILE = str(tmp_path / "test.lock")

        try:
            lock1 = rc._acquire_lock()
            assert lock1 is not None

            rc._release_lock(lock1)

            lock2 = rc._acquire_lock()
            assert lock2 is not None

            rc._release_lock(lock2)
        finally:
            rc.LOCK_FILE = original_lock
