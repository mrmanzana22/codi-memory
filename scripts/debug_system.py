#!/usr/bin/env python3
"""
Codi Memory - System Diagnostic Tool
=====================================
Diagnoses all Sprint 3-4 implementations in a single report.

Usage:
    python3 scripts/debug_system.py          # Full report
    python3 scripts/debug_system.py --json   # JSON output

Covers:
  S3-01: GNW Bistable Ignition (competition.py)
  S3-02: Phi_E / INT-1 (assessment.py)
  S3-03: 6D Consciousness Profile
  S3-04: Orphan Event Connections (wiring.py)
  S3-05: Cross-Topic Bridging (consolidation.py)
  S4-01/02/03: Active Inference Module
  S4-04: PE-gated Dual Process (preturn_inject.py)
  S4-05: Sleep World Model (sleep_loop.py)

Created: 2026-02-28 (Sprint 4 debug tooling)
"""

import json
import os
import sqlite3
import sys
import time
from datetime import datetime, timedelta

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

FTS_DB_PATH = os.path.join(BASE_DIR, "memories_fts.db")
DATA_DIR = os.path.join(BASE_DIR, "data")


def _get_conn():
    conn = sqlite3.connect(FTS_DB_PATH, timeout=5)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA query_only=ON")
    return conn


# ============================================================
# S4-04: PE-GATED DUAL PROCESS
# ============================================================

def diagnose_dual_process(conn) -> dict:
    """Check PE-gated dual process stats from prediction_results."""
    result = {
        "section": "S4-04: PE-gated Dual Process",
        "status": "unknown",
        "details": {},
    }

    try:
        # Total predictions
        total = conn.execute(
            "SELECT COUNT(*) FROM prediction_results"
        ).fetchone()[0]
        result["details"]["total_predictions"] = total

        if total == 0:
            result["status"] = "NO_DATA"
            result["details"]["note"] = "No prediction results yet. Hook may not have fired."
            return result

        # Recent predictions (last 20)
        rows = conn.execute("""
            SELECT predicted_topic, actual_topic, surprise_score,
                   precision_weight, weighted_surprise, hit, created_at
            FROM prediction_results
            ORDER BY id DESC LIMIT 20
        """).fetchall()

        # Count habit vs deliberate (surprise_info is None when weighted_surprise <= threshold)
        threshold_row = conn.execute("""
            SELECT AVG(weighted_surprise) + 0.75 * (
                CASE WHEN COUNT(*) > 1
                THEN SQRT(SUM((weighted_surprise - (SELECT AVG(weighted_surprise) FROM prediction_results)) *
                               (weighted_surprise - (SELECT AVG(weighted_surprise) FROM prediction_results))) / COUNT(*))
                ELSE 0 END
            ) as threshold
            FROM (SELECT weighted_surprise FROM prediction_results ORDER BY id DESC LIMIT 50)
        """).fetchone()
        threshold = max(0.25, min(0.8, threshold_row[0])) if threshold_row and threshold_row[0] else 0.25

        above_threshold = sum(1 for r in rows if r[4] and r[4] > threshold)
        below_threshold = len(rows) - above_threshold

        result["details"]["last_20"] = {
            "habit_mode_count": below_threshold,
            "deliberate_mode_count": above_threshold,
            "habit_pct": round(below_threshold / max(1, len(rows)) * 100, 1),
            "adaptive_threshold": round(threshold, 4),
        }

        # Hit rate
        hits = conn.execute(
            "SELECT SUM(hit), COUNT(*) FROM prediction_results"
        ).fetchone()
        if hits[1] > 0:
            result["details"]["accuracy"] = round(hits[0] / hits[1] * 100, 1)

        # Last prediction
        last = conn.execute(
            "SELECT predicted_topic, confidence, created_at FROM prediction_state WHERE id = 1"
        ).fetchone()
        if last:
            result["details"]["current_prediction"] = {
                "topic": last[0],
                "confidence": round(last[1], 3),
                "generated_at": last[2][:19],
            }

        result["status"] = "OK"
    except Exception as e:
        result["status"] = "ERROR"
        result["details"]["error"] = str(e)[:200]

    return result


# ============================================================
# S4-05: SLEEP WORLD MODEL
# ============================================================

def diagnose_world_model(conn) -> dict:
    """Check Sleep World Model state and learned effects."""
    result = {
        "section": "S4-05: Sleep World Model",
        "status": "unknown",
        "details": {},
    }

    try:
        # Check if world_model_effects table exists
        table_exists = conn.execute("""
            SELECT name FROM sqlite_master
            WHERE type='table' AND name='world_model_effects'
        """).fetchone()

        if not table_exists:
            result["status"] = "NOT_INITIALIZED"
            result["details"]["note"] = "world_model_effects table not found. Sleep loop hasn't run with S4-05 yet."
            return result

        # Learned effects
        effects = conn.execute(
            "SELECT tick_name, dimension, effect, observations, updated_at "
            "FROM world_model_effects ORDER BY tick_name, dimension"
        ).fetchall()

        result["details"]["learned_effects_count"] = len(effects)

        if effects:
            by_tick = {}
            for tick, dim, effect, obs, updated in effects:
                if tick not in by_tick:
                    by_tick[tick] = {"observations": obs, "effects": {}, "last_updated": updated}
                by_tick[tick]["effects"][dim] = round(effect, 4)
            result["details"]["by_tick"] = by_tick

        # Tick timestamps
        timestamps = conn.execute("""
            SELECT key, value FROM sleep_loop_state
            WHERE key LIKE 'last_%' ORDER BY key
        """).fetchall()

        if timestamps:
            result["details"]["tick_timestamps"] = {
                k.replace("last_", ""): v[:19] for k, v in timestamps
            }

        # Tick counter
        counter = conn.execute(
            "SELECT value FROM sleep_loop_state WHERE key = 'tick_counter'"
        ).fetchone()
        if counter:
            result["details"]["tick_counter"] = int(counter[0])

        result["status"] = "OK" if effects else "EMPTY"
    except Exception as e:
        result["status"] = "ERROR"
        result["details"]["error"] = str(e)[:200]

    return result


# ============================================================
# S3-01: GNW BISTABLE IGNITION
# ============================================================

def diagnose_gwt_bistability(conn) -> dict:
    """Check workspace cycles for bistable ignition behavior."""
    result = {
        "section": "S3-01: GNW Bistable Ignition",
        "status": "unknown",
        "details": {},
    }

    try:
        table_exists = conn.execute("""
            SELECT name FROM sqlite_master
            WHERE type='table' AND name='workspace_cycles'
        """).fetchone()

        if not table_exists:
            result["status"] = "NO_TABLE"
            result["details"]["note"] = (
                "workspace_cycles table not found. Created by focus_attention() on first call. "
                "Needs MCP restart to activate S3-01/S3-02 changes."
            )
            return result

        total = conn.execute("SELECT COUNT(*) FROM workspace_cycles").fetchone()[0]
        result["details"]["total_cycles"] = total

        if total == 0:
            result["status"] = "NO_DATA"
            return result

        # Recent cycles
        recent = conn.execute("""
            SELECT winning_theme, winning_score, total_candidates,
                   competition_ms, created_at
            FROM workspace_cycles
            ORDER BY id DESC LIMIT 10
        """).fetchall()

        if recent:
            scores = [r[1] for r in recent if r[1] is not None]
            result["details"]["last_10_cycles"] = {
                "avg_winning_score": round(sum(scores) / max(1, len(scores)), 3) if scores else None,
                "max_winning_score": round(max(scores), 3) if scores else None,
                "min_winning_score": round(min(scores), 3) if scores else None,
                "avg_candidates": round(sum(r[2] for r in recent if r[2]) / max(1, len(recent)), 1),
            }

            # Check if scores show bistable behavior (winners near 1.0)
            high_ignition = sum(1 for s in scores if s and s >= 0.9)
            result["details"]["bistable_ignition_pct"] = round(high_ignition / max(1, len(scores)) * 100, 1)

        result["status"] = "OK"
    except Exception as e:
        result["status"] = "ERROR"
        result["details"]["error"] = str(e)[:200]

    return result


# ============================================================
# S3-05: CROSS-TOPIC BRIDGING
# ============================================================

def diagnose_bridge_edges(conn) -> dict:
    """Check cross-topic bridge edges from consolidation."""
    result = {
        "section": "S3-05: Cross-Topic Bridging",
        "status": "unknown",
        "details": {},
    }

    try:
        table_exists = conn.execute("""
            SELECT name FROM sqlite_master
            WHERE type='table' AND name='graph_edges'
        """).fetchone()

        if not table_exists:
            result["status"] = "NO_TABLE"
            result["details"]["note"] = (
                "graph_edges table not found. Created by consolidation pipeline. "
                "Needs full consolidation run (nightly) to activate S3-05 bridging."
            )
            return result

        total_edges = conn.execute("SELECT COUNT(*) FROM graph_edges").fetchone()[0]
        bridge_edges = conn.execute(
            "SELECT COUNT(*) FROM graph_edges WHERE type = 'bridge'"
        ).fetchone()[0]

        result["details"]["total_edges"] = total_edges
        result["details"]["bridge_edges"] = bridge_edges
        result["details"]["bridge_pct"] = round(bridge_edges / max(1, total_edges) * 100, 1)

        # Last consolidation with bridges
        try:
            last_full = conn.execute(
                "SELECT created_at, scope FROM consolidation_log "
                "ORDER BY id DESC LIMIT 1"
            ).fetchone()
            if last_full:
                result["details"]["last_consolidation"] = {
                    "at": last_full[0][:19] if last_full[0] else None,
                    "scope": last_full[1],
                }
        except Exception:
            pass

        result["status"] = "OK" if bridge_edges > 0 else "NO_BRIDGES"
        if bridge_edges == 0:
            result["details"]["note"] = "No bridge edges yet. Requires full consolidation to run."
    except Exception as e:
        result["status"] = "ERROR"
        result["details"]["error"] = str(e)[:200]

    return result


# ============================================================
# S3-04: ORPHAN EVENT CONNECTIONS
# ============================================================

def diagnose_event_connections(conn) -> dict:
    """Check if orphan events are now being handled."""
    result = {
        "section": "S3-04: Orphan Event Connections",
        "status": "unknown",
        "details": {},
    }

    try:
        table_exists = conn.execute("""
            SELECT name FROM sqlite_master
            WHERE type='table' AND name='event_counts'
        """).fetchone()

        if not table_exists:
            result["status"] = "NO_TABLE"
            return result

        # Check previously-orphan events
        orphan_events = [
            "session_close", "session_open", "sleep_loop_complete",
            "perf_budget_violation", "emotion_gating_applied",
        ]

        event_data = {}
        for evt in orphan_events:
            row = conn.execute(
                "SELECT count, last_seen FROM event_counts WHERE event = ?",
                (evt,)
            ).fetchone()
            if row:
                event_data[evt] = {"count": row[0], "last_seen": row[1][:19] if row[1] else None}
            else:
                event_data[evt] = {"count": 0, "last_seen": None}

        result["details"]["events"] = event_data
        connected = sum(1 for v in event_data.values() if v["count"] > 0)
        result["details"]["connected_count"] = f"{connected}/{len(orphan_events)}"

        result["status"] = "OK" if connected > 0 else "NO_EVENTS_YET"
    except Exception as e:
        result["status"] = "ERROR"
        result["details"]["error"] = str(e)[:200]

    return result


# ============================================================
# S4-01/02/03: ACTIVE INFERENCE MODULE
# ============================================================

def diagnose_active_inference() -> dict:
    """Check Active Inference module health."""
    result = {
        "section": "S4-01/02/03: Active Inference Module",
        "status": "unknown",
        "details": {},
    }

    try:
        from modules.active_inference import (
            ALL_ACTIONS, GenerativeModel, SystemState,
            compute_efe, select_action, get_current_state,
        )

        result["details"]["actions_defined"] = len(ALL_ACTIONS)
        result["details"]["action_names"] = [a.name for a in ALL_ACTIONS]

        # Test model instantiation
        model = GenerativeModel()
        result["details"]["model_observations"] = model.observation_count

        # Test current state reading
        try:
            state = get_current_state()
            result["details"]["current_state"] = {
                "topic": state.topic,
                "uncertainty": round(state.uncertainty, 3),
                "wm_load": round(state.wm_load, 3),
                "pe_magnitude": round(state.pe_magnitude, 3),
                "emotional_valence": round(state.emotional_valence, 3),
            }
        except Exception as e:
            result["details"]["current_state_error"] = str(e)[:100]

        # Test EFE computation
        state = SystemState(topic="general", uncertainty=0.5, wm_load=0.5,
                            pe_magnitude=0.0, emotional_valence=0.0)
        action, efes = select_action(state, model)
        result["details"]["test_selection"] = {
            "selected_action": action.name,
            "efe_range": [round(min(efes.values()), 4), round(max(efes.values()), 4)],
        }

        result["status"] = "OK"
    except Exception as e:
        result["status"] = "ERROR"
        result["details"]["error"] = str(e)[:200]

    return result


# ============================================================
# S5: ACTIVE INFERENCE INTEGRATION
# ============================================================

def diagnose_ai_integration(conn) -> dict:
    """Check Sprint 5 Active Inference Integration health."""
    result = {
        "section": "S5: Active Inference Integration",
        "status": "unknown",
        "details": {},
    }

    try:
        # Check persisted model
        try:
            rows = conn.execute(
                "SELECT COUNT(*) FROM generative_model_transitions"
            ).fetchone()[0]
            result["details"]["persisted_transitions"] = rows
        except sqlite3.OperationalError:
            result["details"]["persisted_transitions"] = "TABLE_NOT_CREATED"

        # Check if integration module loads
        try:
            from modules.active_inference_integration import recommend_action, get_model
            model = get_model()
            result["details"]["loaded_observations"] = model.observation_count

            # Get live recommendation
            rec = recommend_action()
            result["details"]["recommendation"] = {
                "action": rec["recommended"],
                "rationale": rec["rationale"][:100],
                "state_topic": rec["state"]["topic"],
                "state_uncertainty": rec["state"]["uncertainty"],
            }
        except Exception as e:
            result["details"]["integration_error"] = str(e)[:150]

        # Check event handler registration
        try:
            from modules.events import event_bus, Events
            handlers = event_bus._handlers.get(Events.MEMORY_STORED, [])
            ai_handlers = [h for h in handlers if "active_inference" in getattr(h, "__module__", "")]
            result["details"]["event_handlers_registered"] = len(ai_handlers) > 0
        except Exception:
            result["details"]["event_handlers_registered"] = "unknown"

        result["status"] = "OK"
    except Exception as e:
        result["status"] = "ERROR"
        result["details"]["error"] = str(e)[:200]

    return result


# ============================================================
# S3-02/03: PHI_E + CONSCIOUSNESS PROFILE
# ============================================================

def diagnose_consciousness_metrics(conn) -> dict:
    """Check Phi_E and 6D consciousness profile."""
    result = {
        "section": "S3-02/03: Phi_E + Consciousness Profile",
        "status": "unknown",
        "details": {},
    }

    try:
        from modules.assessment import compute_phi_e, get_consciousness_profile

        # Phi_E
        try:
            from modules.workspace import get_coalition_timeseries
            history = get_coalition_timeseries()
            phi_e = compute_phi_e(history)
            result["details"]["phi_e"] = round(phi_e, 4)
            result["details"]["coalition_history_size"] = len(history)
        except Exception as e:
            result["details"]["phi_e_error"] = str(e)[:100]

        # 6D Profile
        try:
            profile = get_consciousness_profile()
            result["details"]["consciousness_profile"] = {
                k: round(v, 3) for k, v in profile.items()
            }
        except Exception as e:
            result["details"]["profile_error"] = str(e)[:100]

        result["status"] = "OK"
    except Exception as e:
        result["status"] = "ERROR"
        result["details"]["error"] = str(e)[:200]

    return result


# ============================================================
# OVERALL SYSTEM HEALTH
# ============================================================

def diagnose_system_health(conn) -> dict:
    """Quick overall system health check."""
    result = {
        "section": "System Health",
        "status": "unknown",
        "details": {},
    }

    try:
        # Memory counts
        mem_count = conn.execute("SELECT COUNT(*) FROM memories_text").fetchone()[0]
        result["details"]["total_memories_fts"] = mem_count

        # Working memory
        try:
            wm_count = conn.execute(
                "SELECT COUNT(*) FROM working_memory WHERE active = 1"
            ).fetchone()[0]
            result["details"]["active_wm_items"] = wm_count
        except Exception:
            result["details"]["active_wm_items"] = "table missing"

        # Recent sleep reports
        try:
            sleep_count = conn.execute("""
                SELECT COUNT(*) FROM session_checkpoints
                WHERE sleep_report IS NOT NULL AND sleep_report != ''
                AND created_at > datetime('now', '-7 days')
            """).fetchone()[0]
            result["details"]["sleep_reports_7d"] = sleep_count
        except Exception:
            pass

        # Prediction health
        try:
            pred_count = conn.execute("SELECT COUNT(*) FROM prediction_results").fetchone()[0]
            result["details"]["prediction_results"] = pred_count
        except Exception:
            pass

        result["status"] = "OK"
    except Exception as e:
        result["status"] = "ERROR"
        result["details"]["error"] = str(e)[:200]

    return result


# ============================================================
# MAIN REPORT
# ============================================================

def run_full_diagnostic(as_json=False) -> str:
    """Run all diagnostics and return formatted report."""
    start = time.monotonic()

    if not os.path.exists(FTS_DB_PATH):
        return "ERROR: FTS database not found at " + FTS_DB_PATH

    conn = _get_conn()

    sections = [
        diagnose_system_health(conn),
        diagnose_dual_process(conn),
        diagnose_world_model(conn),
        diagnose_gwt_bistability(conn),
        diagnose_bridge_edges(conn),
        diagnose_event_connections(conn),
        diagnose_active_inference(),
        diagnose_ai_integration(conn),
        diagnose_consciousness_metrics(conn),
    ]

    conn.close()

    elapsed_ms = round((time.monotonic() - start) * 1000)

    if as_json:
        return json.dumps({
            "diagnostics": sections,
            "elapsed_ms": elapsed_ms,
            "timestamp": datetime.now().isoformat()[:19],
        }, indent=2, ensure_ascii=False)

    # Format as readable report
    lines = [
        "=" * 60,
        "CODI MEMORY - SYSTEM DIAGNOSTIC REPORT",
        f"Timestamp: {datetime.now().isoformat()[:19]}",
        f"Elapsed: {elapsed_ms}ms",
        "=" * 60,
    ]

    for section in sections:
        status_icon = {
            "OK": "[OK]",
            "ERROR": "[ERROR]",
            "NO_DATA": "[NO DATA]",
            "NO_TABLE": "[MISSING]",
            "NOT_INITIALIZED": "[PENDING]",
            "NO_BRIDGES": "[WAITING]",
            "NO_EVENTS_YET": "[WAITING]",
            "EMPTY": "[EMPTY]",
        }.get(section["status"], "[???]")

        lines.append("")
        lines.append(f"--- {section['section']} {status_icon} ---")

        for key, value in section["details"].items():
            if isinstance(value, dict):
                lines.append(f"  {key}:")
                for k2, v2 in value.items():
                    lines.append(f"    {k2}: {v2}")
            else:
                lines.append(f"  {key}: {value}")

    lines.append("")
    lines.append("=" * 60)

    return "\n".join(lines)


if __name__ == "__main__":
    as_json = "--json" in sys.argv
    print(run_full_diagnostic(as_json=as_json))
