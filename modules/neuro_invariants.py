"""
Codi Memory - Neuro Invariants Checker
=======================================
Validates that neuroscience-grounded constants and runtime state
remain within theoretically correct bounds.

Extracted from full neuro audit (Feb 26, 2026).
Called by sleep_loop health tick to catch regressions.

References:
  - Anderson 2007 (ACT-R base-level activation)
  - Friston 2010 (precision weighting)
  - Nader 2000 (reconsolidation window)
  - Ebbinghaus 1885 / FadeMem (decay curves)
  - Anderson, Bjork & Bjork 1994 (RIF)
  - Tononi & Cirelli 2003 (synaptic homeostasis)
  - Behrens et al. 2007 (volatility estimation)
"""

import logging
import os
import sqlite3

_logger = logging.getLogger(__name__)

# ============================================================
# INVARIANT DEFINITIONS
# Each invariant: (module, constant_name, check_fn, description)
# ============================================================

def _check_range(value, lo, hi):
    """Check value is within [lo, hi]."""
    return lo <= value <= hi


def check_static_invariants() -> list:
    """Check code-level constants haven't drifted from neuro-grounded values.

    Returns list of violation dicts. Empty = all OK.
    """
    violations = []

    def _v(module, name, expected, actual, theory):
        if actual != expected:
            violations.append({
                "type": "static",
                "module": module,
                "name": name,
                "expected": expected,
                "actual": actual,
                "theory": theory,
            })

    def _r(module, name, lo, hi, actual, theory):
        if not _check_range(actual, lo, hi):
            violations.append({
                "type": "static",
                "module": module,
                "name": name,
                "expected": f"[{lo}, {hi}]",
                "actual": actual,
                "theory": theory,
            })

    # --- ACT-R Activation (Anderson 2007) ---
    try:
        from modules.activation import (
            DECAY_EPISODIC, DECAY_SEMANTIC, DECAY_EPISODIC_CRITICAL,
            DECAY_WORKING_MEMORY, DECAY_SEMANTIC_MIN,
            W_SPREAD, W_EMOTION, W_IMPORTANCE, W_PREDICTION,
            NOISE_SIGMA, ACTR_DECAY_DEFAULT,
        )
        _v("activation", "ACTR_DECAY_DEFAULT", 0.40, ACTR_DECAY_DEFAULT, "Anderson 2007")
        _v("activation", "DECAY_EPISODIC", 0.40, DECAY_EPISODIC, "Anderson 2007")
        _v("activation", "DECAY_SEMANTIC", 0.15, DECAY_SEMANTIC, "Bahrick 1984")
        _v("activation", "DECAY_EPISODIC_CRITICAL", 0.20, DECAY_EPISODIC_CRITICAL, "Brown & Kulik 1977")
        _v("activation", "DECAY_WORKING_MEMORY", 0.60, DECAY_WORKING_MEMORY, "Baddeley 2000")
        _v("activation", "DECAY_SEMANTIC_MIN", 0.05, DECAY_SEMANTIC_MIN, "Bahrick 1984 permastore")
        # Weights must sum to ~0.80 (leaving 0.20 for base-level)
        w_sum = W_SPREAD + W_EMOTION + W_IMPORTANCE + W_PREDICTION
        _r("activation", "modulation_weight_sum", 0.70, 0.90, w_sum, "Anderson 2007")
        _r("activation", "NOISE_SIGMA", 0.10, 0.25, NOISE_SIGMA, "Anderson 2007 recommended range")
        # Critical invariant: semantic < episodic < WM decay
        if not (DECAY_SEMANTIC < DECAY_EPISODIC < DECAY_WORKING_MEMORY):
            violations.append({
                "type": "static", "module": "activation",
                "name": "decay_ordering",
                "expected": "semantic < episodic < WM",
                "actual": f"{DECAY_SEMANTIC} < {DECAY_EPISODIC} < {DECAY_WORKING_MEMORY}",
                "theory": "Bahrick 1984 + Baddeley 2000",
            })
    except ImportError:
        violations.append({"type": "import", "module": "activation", "name": "import_failed",
                           "expected": "importable", "actual": "error", "theory": "—"})

    # --- FadeMem Decay (Ebbinghaus 1885 / Wixted & Ebbesen 1991) ---
    try:
        from modules.forgetting import (
            FADEM_LAMBDA_BASE, FADEM_MU, FADEM_FLOOR,
            FADEM_BETA, FADEM_AROUSAL_THRESHOLD, RIF_BASE,
        )
        _v("forgetting", "FADEM_MU", 0.5, FADEM_MU, "FadeMem importance sensitivity")
        _r("forgetting", "FADEM_FLOOR", 0.05, 0.15, FADEM_FLOOR, "Minimum salience floor")
        _r("forgetting", "FADEM_LAMBDA_BASE", 0.001, 0.05, FADEM_LAMBDA_BASE, "Ebbinghaus 1885")
        # Beta ordering: unconsolidated > episodic > semantic (faster decay = higher beta)
        b_uncon = FADEM_BETA.get("unconsolidated", 0)
        b_epis = FADEM_BETA.get("consolidated_episodic", 0)
        b_sem = FADEM_BETA.get("consolidated_semantic", 0)
        if not (b_sem < b_epis < b_uncon):
            violations.append({
                "type": "static", "module": "forgetting",
                "name": "beta_ordering",
                "expected": "semantic < episodic < unconsolidated",
                "actual": f"{b_sem} < {b_epis} < {b_uncon}",
                "theory": "Tononi & Cirelli 2003 + Bahrick 1984",
            })
        # RIF suppression range (Anderson, Bjork & Bjork 1994)
        _r("forgetting", "RIF_BASE", 0.05, 0.20, RIF_BASE, "Anderson et al. 1994: 8-15%")
    except ImportError:
        violations.append({"type": "import", "module": "forgetting", "name": "import_failed",
                           "expected": "importable", "actual": "error", "theory": "—"})

    # --- Reconsolidation (Nader 2000) ---
    try:
        from modules.config import (
            RECONSOLIDATION_WINDOW_HOURS, RECONSOLIDATION_PE_THRESHOLD,
            RECONSOLIDATION_STRENGTH_FLOOR, RECONSOLIDATION_STRENGTH_CEILING,
        )
        _r("config", "RECONSOLIDATION_WINDOW_HOURS", 0.5, 8.0,
           RECONSOLIDATION_WINDOW_HOURS, "Nader 2000: labile window")
        _r("config", "RECONSOLIDATION_PE_THRESHOLD", 0.1, 0.5,
           RECONSOLIDATION_PE_THRESHOLD, "Sevenster et al. 2013")
        _r("config", "RECONSOLIDATION_STRENGTH_FLOOR", 0.10, 0.25,
           RECONSOLIDATION_STRENGTH_FLOOR, "Exton-McGuinness 2015")
    except ImportError:
        pass  # These may be defined differently

    # --- Spreading Activation (Collins & Loftus 1975) ---
    try:
        from modules.config import (
            SPREAD_DEFAULT_FACTOR, SPREAD_DEFAULT_DEPTH,
            SPREAD_SALIENCE_CAP, SPREAD_SALIENCE_FLOOR,
        )
        _r("config", "SPREAD_DEFAULT_FACTOR", 0.3, 0.9,
           SPREAD_DEFAULT_FACTOR, "Collins & Loftus 1975")
        _r("config", "SPREAD_DEFAULT_DEPTH", 1, 4,
           SPREAD_DEFAULT_DEPTH, "Anderson 2007")
        _v("config", "SPREAD_SALIENCE_CAP", 1.0, SPREAD_SALIENCE_CAP, "Normalization bound")
    except ImportError:
        pass

    # --- Sleep Loop Architecture (Diekelmann & Born 2010) ---
    try:
        from modules.sleep_loop import TICK_ORDER
        expected_ticks = {"prospective", "health", "self_model", "reconsolidation",
                          "consolidation", "homeostasis", "backup"}
        actual_ticks = set(TICK_ORDER)
        missing = expected_ticks - actual_ticks
        if missing:
            violations.append({
                "type": "static", "module": "sleep_loop",
                "name": "missing_ticks",
                "expected": sorted(expected_ticks),
                "actual": sorted(actual_ticks),
                "theory": "Diekelmann & Born 2010 + system architecture",
            })
    except ImportError:
        pass

    return violations


def check_runtime_invariants() -> list:
    """Check runtime state in SQLite for neuroscience invariants.

    Returns list of violation dicts. Empty = all OK.
    """
    violations = []
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    fts_db = os.path.join(BASE_DIR, "memories_fts.db")

    if not os.path.exists(fts_db):
        violations.append({
            "type": "runtime", "module": "fts_db",
            "name": "db_missing", "detail": "memories_fts.db not found",
        })
        return violations

    try:
        conn = sqlite3.connect(fts_db, timeout=3)

        # 1. Transition stats must exist and be populated (Friston 2010)
        try:
            row = conn.execute("SELECT COUNT(*) FROM transition_stats").fetchone()
            if row and row[0] < 5:
                violations.append({
                    "type": "runtime", "module": "precision_weighting",
                    "name": "transition_stats_sparse",
                    "detail": f"Only {row[0]} transition records (need >=5 for dampening)",
                    "theory": "Friston 2010",
                })
        except sqlite3.OperationalError:
            violations.append({
                "type": "runtime", "module": "precision_weighting",
                "name": "transition_stats_missing",
                "detail": "Table transition_stats does not exist",
                "theory": "Friston 2010",
            })

        # 2. Prediction results should not be dominated by sleep_loop (PCI fix)
        try:
            total = conn.execute("SELECT COUNT(*) FROM prediction_results").fetchone()
            sleep = conn.execute(
                "SELECT COUNT(*) FROM prediction_results WHERE source = 'sleep_loop'"
            ).fetchone()
            if total and total[0] > 20 and sleep and sleep[0] > 0:
                ratio = sleep[0] / total[0]
                if ratio > 0.3:
                    violations.append({
                        "type": "runtime", "module": "prediction",
                        "name": "sleep_loop_contamination",
                        "detail": f"sleep_loop is {ratio:.0%} of predictions ({sleep[0]}/{total[0]})",
                        "theory": "PCI fix (Proposal #42)",
                    })
        except sqlite3.OperationalError:
            pass  # Table may not exist yet

        # 3. Attention schema should be persisting (Graziano 2013 AST)
        try:
            row = conn.execute(
                "SELECT COUNT(*) FROM attention_transitions"
            ).fetchone()
            if row and row[0] == 0:
                violations.append({
                    "type": "runtime", "module": "attention",
                    "name": "no_attention_transitions",
                    "detail": "attention_transitions table is empty",
                    "theory": "Graziano 2013 AST",
                })
        except sqlite3.OperationalError:
            pass

        # 4. Labile memories should not exceed window (Nader 2000)
        try:
            expired = conn.execute("""
                SELECT COUNT(*) FROM labile_memories
                WHERE window_expires < datetime('now', '-1 hour')
            """).fetchone()
            if expired and expired[0] > 5:
                violations.append({
                    "type": "runtime", "module": "reconsolidation",
                    "name": "stale_labile_memories",
                    "detail": f"{expired[0]} labile memories past expiry window",
                    "theory": "Nader 2000: window must be enforced",
                })
        except sqlite3.OperationalError:
            pass

        # 5. Event counts should show active wiring (GWT broadcast)
        try:
            events = conn.execute("""
                SELECT event, count FROM event_counts
                WHERE event IN ('prediction_error', 'attention_prediction_error',
                                'memory_stored', 'workspace_broadcast')
            """).fetchall()
            event_dict = {e[0]: e[1] for e in events}
            if not event_dict.get('prediction_error', 0):
                violations.append({
                    "type": "runtime", "module": "wiring",
                    "name": "no_prediction_errors",
                    "detail": "Zero prediction_error events recorded",
                    "theory": "Clark 2013: prediction loop must be active",
                })
        except sqlite3.OperationalError:
            pass

        conn.close()
    except Exception as e:
        violations.append({
            "type": "runtime", "module": "sqlite",
            "name": "db_error", "detail": str(e)[:100],
        })

    return violations


def check_all() -> dict:
    """Run all invariant checks. Returns summary dict."""
    static = check_static_invariants()
    runtime = check_runtime_invariants()

    all_violations = static + runtime
    critical = [v for v in all_violations if v.get("type") == "static"]

    return {
        "ok": len(all_violations) == 0,
        "total_violations": len(all_violations),
        "static_violations": len(static),
        "runtime_violations": len(runtime),
        "critical": len(critical),
        "violations": all_violations,
        "summary": _format_summary(all_violations),
    }


def _format_summary(violations: list) -> str:
    """One-line summary for sleep report."""
    if not violations:
        return "neuro: all invariants OK"
    names = [v.get("name", "?") for v in violations[:3]]
    more = f" +{len(violations) - 3}" if len(violations) > 3 else ""
    return f"neuro: {len(violations)} violations [{', '.join(names)}{more}]"
