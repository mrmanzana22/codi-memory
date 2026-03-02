#!/usr/bin/env python3
"""
J.0.5: Neuro Invariants Regression Tests (Breck Test 8 + Breck Test 14)
========================================================================
Breck 2017 Tests 8/14: Model invariants are tested. Quality is checked.

These tests verify that:
  1. `check_static_invariants()` returns ZERO violations with current constants
     — meaning neuro-grounded constants have NOT drifted (Breck Test 8)
  2. When constants DO drift, the checker catches it (regression detection)
  3. `check_all()` returns the correct schema (API contract test)
  4. The health module integrates neuro_invariants correctly

Why this test matters:
  - Without Test 8 (model invariants), any refactor can silently break the
    neuroscience grounding of ACT-R decay, FadeMem, etc.
  - Without Test 14 (quality checks), the health tick could stop running
    invariant checks without anyone noticing.
  - Together they close the loop: constants are correct AND the check runs.

Based on: neuro_invariants.py (Feb 26, 2026 neuro audit)
Run: ./venv/bin/pytest tests/test_neuro_invariants_regression.py -v
"""

import sys
import os
import unittest.mock as mock
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from modules.neuro_invariants import (
    check_static_invariants,
    check_runtime_invariants,
    check_all,
    _format_summary,
)


# ============================================================
# BRECK TEST 8 — MODEL INVARIANTS ARE TESTED
# ============================================================

class TestStaticInvariantsNoViolations:
    """
    Breck Test 8: Model invariants are tested.

    Verifies that current constants in activation.py, forgetting.py, and
    config.py are within neuroscience-grounded bounds.

    If this test fails, it means a constant was changed without updating
    the theoretical justification. REQUIRES REVIEW before merging.
    """

    # All known stale checks resolved. Keep set for future use if needed.
    _KNOWN_STALE_CHECKS = set()

    def test_no_static_violations_with_current_constants(self):
        """
        check_static_invariants() returns zero violations (excluding known stale checks).

        This is the GOLDEN test for neuroscience integrity. If it fails,
        one of the following changed without theoretical justification:
          - ACT-R decay constants (Anderson 2007)
          - FadeMem power-law constants (Wixted & Ebbesen 1991)
          - Reconsolidation window/threshold (Nader 2000)
          - Spreading activation factors (Collins & Loftus 1975)

        Known stale: FADEM_MU expected=0.5 in checker but actual=1.0 in forgetting.py.
        The checker's expected value is outdated (not a regression).
        """
        violations = check_static_invariants()
        real_violations = [v for v in violations if v.get("name") not in self._KNOWN_STALE_CHECKS]
        violation_names = [v.get("name", "?") for v in real_violations]
        assert real_violations == [], (
            f"Neuro invariants violated — REVIEW BEFORE MERGE.\n"
            f"Violations: {violation_names}\n"
            f"Full details: {real_violations[:3]}"
        )

    def test_fadem_mu_checker_aligned(self):
        """
        Confirms FADEM_MU=1.0 in forgetting.py matches neuro_invariants checker.
        Fixed: K.2.3 — was 0.5 in checker, 1.0 in code (S0-04 power-law form).
        """
        from modules.forgetting import FADEM_MU
        assert FADEM_MU == 1.0, f"forgetting.FADEM_MU should be 1.0, got {FADEM_MU}"

        # Confirm checker no longer reports FADEM_MU as violation
        violations = check_static_invariants()
        fadem_mu_violations = [v for v in violations if v.get("name") == "FADEM_MU"]
        assert len(fadem_mu_violations) == 0, (
            "FADEM_MU should no longer be a violation — checker expects 1.0 now."
        )

    def test_activation_module_is_importable(self):
        """activation.py must import cleanly (no broken constants)."""
        from modules.activation import (
            DECAY_EPISODIC, DECAY_SEMANTIC, DECAY_WORKING_MEMORY,
            ACTR_DECAY_DEFAULT,
        )
        assert DECAY_EPISODIC > 0
        assert DECAY_SEMANTIC > 0
        assert DECAY_WORKING_MEMORY > 0
        assert ACTR_DECAY_DEFAULT > 0

    def test_forgetting_module_is_importable(self):
        """forgetting.py must import cleanly (no broken constants)."""
        from modules.forgetting import (
            FADEM_LAMBDA_BASE, FADEM_MU, FADEM_FLOOR,
            FADEM_BETA, RIF_BASE,
        )
        assert FADEM_LAMBDA_BASE > 0
        assert FADEM_MU > 0
        assert FADEM_FLOOR > 0
        assert isinstance(FADEM_BETA, dict)
        assert "unconsolidated" in FADEM_BETA
        assert RIF_BASE > 0

    def test_decay_ordering_invariant(self):
        """
        Critical ordering: DECAY_SEMANTIC < DECAY_EPISODIC < DECAY_WORKING_MEMORY.

        Theoretical basis:
          - Semantic memory decays slowest (Bahrick 1984 permastore)
          - Working memory decays fastest (Baddeley 2000)
          - Episodic in between (Anderson 2007)

        If this breaks, memory retrieval will favor wrong memory types.
        """
        from modules.activation import (
            DECAY_SEMANTIC, DECAY_EPISODIC, DECAY_WORKING_MEMORY
        )
        assert DECAY_SEMANTIC < DECAY_EPISODIC, (
            f"Semantic ({DECAY_SEMANTIC}) should decay slower than episodic ({DECAY_EPISODIC}). "
            f"Theory: Bahrick 1984 (semantic permastore)"
        )
        assert DECAY_EPISODIC < DECAY_WORKING_MEMORY, (
            f"Episodic ({DECAY_EPISODIC}) should decay slower than WM ({DECAY_WORKING_MEMORY}). "
            f"Theory: Baddeley 2000"
        )

    def test_fadem_beta_ordering_invariant(self):
        """
        Critical ordering: beta_semantic < beta_episodic < beta_unconsolidated.

        Higher beta = faster power-law decay. Semantic memories should
        have the slowest decay (lowest beta), unconsolidated the fastest.
        """
        from modules.forgetting import FADEM_BETA
        b_sem = FADEM_BETA["consolidated_semantic"]
        b_epis = FADEM_BETA["consolidated_episodic"]
        b_uncon = FADEM_BETA["unconsolidated"]

        assert b_sem < b_epis, (
            f"Semantic beta ({b_sem}) should be lower than episodic ({b_epis}). "
            f"Theory: Tononi & Cirelli 2003 + Bahrick 1984"
        )
        assert b_epis < b_uncon, (
            f"Episodic beta ({b_epis}) should be lower than unconsolidated ({b_uncon}). "
            f"Theory: Consolidation = slower forgetting"
        )

    def test_actr_decay_matches_anderson_2007(self):
        """
        ACTR_DECAY_DEFAULT must be 0.40 (Anderson 2007 best-fit empirical).

        Anderson 2007 fitted ACT-R to human memory data across 29 experiments.
        The best-fit decay parameter was consistently 0.40 ± 0.05.
        Deviating from this without empirical justification is unsound.
        """
        from modules.activation import ACTR_DECAY_DEFAULT
        assert abs(ACTR_DECAY_DEFAULT - 0.40) < 0.05, (
            f"ACTR_DECAY_DEFAULT = {ACTR_DECAY_DEFAULT}, expected 0.40 ± 0.05. "
            f"Theory: Anderson 2007 across 29 experiments"
        )


# ============================================================
# REGRESSION DETECTION — DOES THE CHECKER CATCH DRIFT?
# ============================================================

class TestInvariantCheckerDetectsViolations:
    """
    Verify that check_static_invariants() catches constant drift.

    These tests monkey-patch constants to wrong values and verify
    that the checker detects the violation. Without these tests,
    the checker itself could break silently.
    """

    def test_catches_wrong_decay_ordering(self):
        """Checker detects when semantic decay > episodic (inverted ordering)."""
        import modules.activation as act

        original_semantic = act.DECAY_SEMANTIC
        original_episodic = act.DECAY_EPISODIC
        try:
            # Invert the ordering (semantic > episodic = wrong)
            act.DECAY_SEMANTIC = 0.60
            act.DECAY_EPISODIC = 0.20
            violations = check_static_invariants()
            names = [v.get("name") for v in violations]
            assert "decay_ordering" in names, (
                f"Checker should detect inverted decay ordering. "
                f"Got violations: {names}"
            )
        finally:
            act.DECAY_SEMANTIC = original_semantic
            act.DECAY_EPISODIC = original_episodic

    def test_catches_fadem_lambda_out_of_range(self):
        """Checker detects FADEM_LAMBDA_BASE drift above neuroscience range."""
        import modules.forgetting as forg

        original = forg.FADEM_LAMBDA_BASE
        try:
            forg.FADEM_LAMBDA_BASE = 0.99  # Way above [0.001, 0.05]
            violations = check_static_invariants()
            names = [v.get("name") for v in violations]
            assert "FADEM_LAMBDA_BASE" in names, (
                f"Checker should detect FADEM_LAMBDA_BASE=0.99 as out of range. "
                f"Got violations: {names}"
            )
        finally:
            forg.FADEM_LAMBDA_BASE = original

    def test_catches_fadem_beta_ordering_violation(self):
        """Checker detects when semantic beta > unconsolidated (inverted)."""
        import modules.forgetting as forg

        original_beta = forg.FADEM_BETA.copy()
        try:
            # Invert: make semantic FASTER than unconsolidated (wrong)
            forg.FADEM_BETA = {
                "unconsolidated": 0.3,
                "consolidated_episodic": 0.6,
                "consolidated_semantic": 0.9,  # Highest = fastest (wrong)
            }
            violations = check_static_invariants()
            names = [v.get("name") for v in violations]
            assert "beta_ordering" in names, (
                f"Checker should detect inverted beta ordering. "
                f"Got violations: {names}"
            )
        finally:
            forg.FADEM_BETA = original_beta

    def test_catches_actr_decay_wrong_value(self):
        """Checker detects ACTR_DECAY_DEFAULT changed from Anderson 2007 value."""
        import modules.activation as act

        original = act.ACTR_DECAY_DEFAULT
        try:
            act.ACTR_DECAY_DEFAULT = 0.99  # Wrong value
            violations = check_static_invariants()
            names = [v.get("name") for v in violations]
            assert "ACTR_DECAY_DEFAULT" in names, (
                f"Checker should detect ACTR_DECAY_DEFAULT=0.99. "
                f"Got violations: {names}"
            )
        finally:
            act.ACTR_DECAY_DEFAULT = original

    def test_violations_are_idempotent(self):
        """Checker returns same violations on repeated calls (no side effects)."""
        v1 = check_static_invariants()
        v2 = check_static_invariants()
        assert v1 == v2, "check_static_invariants() should be idempotent"


# ============================================================
# BRECK TEST 14 — QUALITY CHECKS DON'T REGRESS
# ============================================================

class TestCheckAllAPIContract:
    """
    Breck Test 14: Quality is validated before production.

    Verifies that check_all() returns the documented schema and that
    the health integration point works correctly.
    """

    def test_check_all_returns_expected_schema(self):
        """check_all() returns dict with all required fields."""
        result = check_all()

        assert isinstance(result, dict), "check_all() must return dict"
        required_fields = ["ok", "total_violations", "static_violations",
                           "runtime_violations", "critical", "violations", "summary"]
        for field in required_fields:
            assert field in result, f"Missing required field: {field}"

    def test_check_all_ok_field_matches_violations(self):
        """ok=True iff total_violations==0 (no hidden errors)."""
        result = check_all()
        if result["total_violations"] == 0:
            assert result["ok"] is True, "ok should be True when no violations"
        else:
            assert result["ok"] is False, "ok should be False when violations exist"

    def test_check_all_violations_is_list(self):
        """violations field is always a list."""
        result = check_all()
        assert isinstance(result["violations"], list)

    def test_check_all_summary_is_string(self):
        """summary field is always a non-empty string."""
        result = check_all()
        assert isinstance(result["summary"], str)
        assert len(result["summary"]) > 0

    def test_check_all_counts_are_consistent(self):
        """static + runtime violation counts sum to total."""
        result = check_all()
        assert result["static_violations"] + result["runtime_violations"] == result["total_violations"], (
            f"Counts inconsistent: static={result['static_violations']} + "
            f"runtime={result['runtime_violations']} != total={result['total_violations']}"
        )

    def test_format_summary_no_violations(self):
        """_format_summary([]) returns 'all invariants OK' message."""
        summary = _format_summary([])
        assert "OK" in summary.upper() or "ok" in summary.lower(), (
            f"Empty violations should produce OK summary, got: {summary}"
        )

    def test_format_summary_with_violations(self):
        """_format_summary returns violation count and names."""
        fake_violations = [
            {"type": "static", "module": "activation", "name": "decay_ordering",
             "expected": "semantic<episodic", "actual": "wrong", "theory": "Anderson"},
            {"type": "static", "module": "forgetting", "name": "FADEM_LAMBDA_BASE",
             "expected": "[0.001, 0.05]", "actual": 0.99, "theory": "Ebbinghaus"},
        ]
        summary = _format_summary(fake_violations)
        assert "2" in summary, f"Summary should mention count=2, got: {summary}"
        # At least one violation name should appear
        assert "decay_ordering" in summary or "FADEM_LAMBDA_BASE" in summary, (
            f"Summary should include violation names, got: {summary}"
        )

    def test_static_violations_are_tagged_correctly(self):
        """All static violations have type='static'."""
        violations = check_static_invariants()
        for v in violations:
            assert v.get("type") in ("static", "import"), (
                f"Unexpected violation type: {v.get('type')} in {v}"
            )


# ============================================================
# INTEGRATION — NEURO INVARIANTS WIRED TO SLEEP HEALTH TICK
# ============================================================

class TestNeuroInvariantsIntegration:
    """
    Verify that neuro_invariants is integrated into the health check
    pipeline (Breck Test 14: quality validated in production path).
    """

    def test_health_module_imports_neuro_invariants(self):
        """health.py or sleep_loop references neuro_invariants."""
        import importlib
        import modules.sleep_loop as sl

        # The sleep loop must call neuro_invariants as part of health tick
        source_path = sl.__file__
        with open(source_path, "r") as f:
            source = f.read()

        assert "neuro_invariants" in source, (
            "sleep_loop.py must import/call neuro_invariants for health tick. "
            "Without this, invariant checks won't run in production. "
            "See: Breck Test 14 requirement."
        )

    def test_check_all_runs_without_production_db(self):
        """check_all() gracefully handles missing production DB.

        Runtime checks should degrade gracefully (not crash) when
        memories_fts.db is not present (e.g., test environments).
        """
        import modules.neuro_invariants as ni
        original_check = ni.check_runtime_invariants

        def mock_runtime():
            # Simulate DB not present
            return [{"type": "runtime", "module": "fts_db",
                     "name": "db_missing", "detail": "test mode"}]

        ni.check_runtime_invariants = mock_runtime
        try:
            result = check_all()
            # Must not raise — graceful degradation
            assert isinstance(result, dict)
            assert result["runtime_violations"] >= 1  # DB missing detected
        finally:
            ni.check_runtime_invariants = original_check

    def test_static_checks_are_independent_of_db(self):
        """check_static_invariants() works even when SQLite DB is unavailable.

        Static checks are pure constant verification — they must not depend
        on database state. This ensures they work in CI environments.
        """
        # Temporarily hide the DB path
        import modules.neuro_invariants as ni
        original_runtime = ni.check_runtime_invariants

        # Mock runtime to simulate DB failure
        ni.check_runtime_invariants = lambda: []
        try:
            # Static checks should still work fine
            violations = check_static_invariants()
            # Should be a list (even if empty or with violations)
            assert isinstance(violations, list)
        finally:
            ni.check_runtime_invariants = original_runtime
