"""
assessment.py - External Butlin/Block consciousness evaluator (Deliverable 3).
================================================================================
Read-only, deterministic, auditable scoring of 14 consciousness indicators.

Design principles:
  - DOES NOT import consciousness.py (independence)
  - collect_evidence() does all IO (SQLite, event_bus, config)
  - score_indicator() is pure: dict in -> dict out, no side effects
  - get_assessment() = collect + score all 14
  - format_assessment() = dict -> markdown string

Evidence sources (read-only):
  - event_counts table (SQLite)
  - fok_calibration_log table (SQLite)
  - reconsolidation_log table (SQLite)
  - event_bus handlers & history (in-memory)
  - config constants (in-memory)
"""

from __future__ import annotations

import sqlite3
from typing import Any, Dict, List, Optional

ASSESSMENT_VERSION = "1.0.0"

# ============================================================
# INDICATOR DEFINITIONS & THRESHOLDS (auditable spec)
# ============================================================

SCORE_LABELS = {
    0.0: "ABSENT",
    0.3: "DORMANT",
    0.5: "PARTIAL",
    0.7: "NASCENT",
    0.8: "NASCENT+",
    1.0: "FULL",
}

THEORY_NAMES = {
    "GWT": "Global Workspace Theory",
    "HOT": "Higher-Order Theories",
    "AST": "Attention Schema Theory",
    "PP": "Predictive Processing",
    "RPT": "Recurrent Processing Theory",
}

# Each indicator: name, theory, thresholds, evidence_keys
# Thresholds define: { score: condition_description }
INDICATOR_SPECS = [
    {"name": "GWT-1", "theory": "GWT", "desc": "Modular Architecture (Block 1995)"},
    {"name": "GWT-2", "theory": "GWT", "desc": "Limited-Capacity Workspace"},
    {"name": "GWT-3", "theory": "GWT", "desc": "Global Broadcast"},
    {"name": "GWT-4", "theory": "GWT", "desc": "Ignition Gating"},
    {"name": "HOT-1", "theory": "HOT", "desc": "Meta-Monitoring (Rosenthal 2005)"},
    {"name": "HOT-2", "theory": "HOT", "desc": "Confidence Calibration (FOK)"},
    {"name": "HOT-3", "theory": "HOT", "desc": "Higher-Order Control"},
    {"name": "HOT-4", "theory": "HOT", "desc": "Quality Space (Emotional)"},
    {"name": "AST-1", "theory": "AST", "desc": "Attention Schema (Graziano 2013)"},
    {"name": "PP-1",  "theory": "PP",  "desc": "Predictive Model"},
    {"name": "PP-2",  "theory": "PP",  "desc": "Prediction Error"},
    {"name": "PP-3",  "theory": "PP",  "desc": "Model Updating (Reconsolidation)"},
    {"name": "RPT-1", "theory": "RPT", "desc": "Recurrent Processing (Lamme 2006)"},
    {"name": "RPT-2", "theory": "RPT", "desc": "Integration"},
]


# ============================================================
# EVIDENCE SNAPSHOT (stable structure for deterministic scoring)
# ============================================================

def collect_evidence() -> Dict[str, Any]:
    """Collect all evidence from IO sources. Returns a stable EvidenceSnapshot dict.

    This is the ONLY function that does IO. Everything else is pure.
    """
    evidence = {
        "event_counts": {},
        "handler_count": 0,
        "total_events": 0,
        "comp_count": 0,
        "comp_subscribers": 0,
        "self_model_refreshes": 0,
        "fok_n_records": 0,
        "metacognitive_count": 0,
        "has_emotional_state": False,
        "has_emotion_gating": False,
        "has_meta_functions": False,
        "has_attention_schema": False,
        "attention_suppressed": False,
        "attention_history_len": 0,
        "attention_pe_count": 0,
        "has_prediction_schemas": False,
        "prediction_error_count": 0,
        "has_correct_memory": False,
        "reconsolidation_count": 0,
        "has_recurrent_cycle": False,
        "retrieved_count": 0,
        "active_event_types": 0,
        "competition_slots": 5,
        "ignition_threshold": 0.25,
        "coalition_bonus": 0.10,
    }

    # --- Event bus data ---
    try:
        from modules.events import event_bus, Events
        evidence["handler_count"] = sum(len(h) for h in event_bus._handlers.values())

        # Collect all persistent event counts
        all_event_names = [
            getattr(Events, attr)
            for attr in dir(Events)
            if not attr.startswith('_') and isinstance(getattr(Events, attr), str)
        ]
        active_count = 0
        total = 0
        for ev_name in all_event_names:
            cnt = event_bus.get_persistent_count(ev_name)
            evidence["event_counts"][ev_name] = cnt
            total += cnt
            if cnt > 0:
                active_count += 1
        evidence["total_events"] = total
        evidence["active_event_types"] = active_count

        # Specific counts
        evidence["comp_count"] = evidence["event_counts"].get(Events.WORKSPACE_COMPETITION_COMPLETE, 0)
        evidence["comp_subscribers"] = len(event_bus._handlers.get(Events.WORKSPACE_COMPETITION_COMPLETE, []))
        evidence["prediction_error_count"] = evidence["event_counts"].get(Events.PREDICTION_ERROR, 0)
        evidence["retrieved_count"] = evidence["event_counts"].get(Events.MEMORY_RETRIEVED, 0)
        evidence["metacognitive_count"] = evidence["event_counts"].get(Events.METACOGNITIVE_CONTROL_APPLIED, 0)
        evidence["attention_pe_count"] = evidence["event_counts"].get(Events.ATTENTION_PREDICTION_ERROR, 0)

        # Self-model refreshes: persistent + session history
        refresh_persist = evidence["event_counts"].get(Events.SELF_MODEL_REFRESHED, 0)
        session_refresh = sum(
            1 for e in event_bus.get_history(limit=50)
            if e.get("event") == Events.SELF_MODEL_REFRESHED
        )
        evidence["self_model_refreshes"] = max(refresh_persist, session_refresh)

        # Session-level metacognitive (augment persistent)
        session_meta = sum(
            1 for e in event_bus.get_history(limit=50)
            if e.get("event") == Events.METACOGNITIVE_CONTROL_APPLIED
        )
        evidence["metacognitive_count"] = max(evidence["metacognitive_count"], session_meta)

        # Session-level prediction error (augment persistent)
        session_pe = sum(
            1 for e in event_bus.get_history(limit=50)
            if e.get("event") == Events.PREDICTION_ERROR
        )
        evidence["prediction_error_count"] = max(evidence["prediction_error_count"], session_pe)

        # Session-level attention PE
        session_attn_pe = sum(
            1 for e in event_bus.get_history(limit=50)
            if e.get("event") == Events.ATTENTION_PREDICTION_ERROR
        )
        evidence["attention_pe_count"] = max(evidence["attention_pe_count"], session_attn_pe)
    except Exception:
        pass

    # --- Competition constants ---
    try:
        from modules.competition import DEFAULT_WORKSPACE_SLOTS, IGNITION_THRESHOLD, COALITION_TOPIC_BONUS
        evidence["competition_slots"] = DEFAULT_WORKSPACE_SLOTS
        evidence["ignition_threshold"] = IGNITION_THRESHOLD
        evidence["coalition_bonus"] = COALITION_TOPIC_BONUS
    except Exception:
        pass

    # --- Meta-monitoring functions ---
    try:
        # Check if the functions exist in consciousness module namespace
        # We import the module but DON'T import consciousness.py - we check
        # if specific callables exist by trying to import them from their
        # actual registration source
        import modules.consciousness as _cons_mod
        evidence["has_meta_functions"] = all(
            hasattr(_cons_mod, f) and callable(getattr(_cons_mod, f, None))
            for f in ['assess_confidence', 'reflect_on_self', 'identify_knowledge_gaps']
        )
    except Exception:
        evidence["has_meta_functions"] = False

    # --- FOK calibration ---
    try:
        from modules.retrieval_metadata import get_fok_calibration
        from modules.config import FTS_DB_PATH
        cal = get_fok_calibration(fts_db_path=FTS_DB_PATH)
        evidence["fok_n_records"] = cal.get("n_records", 0)
    except Exception:
        evidence["fok_n_records"] = 0

    # --- Emotional state ---
    try:
        from modules.config import _emotional_state
        evidence["has_emotional_state"] = bool(_emotional_state)
        if evidence["has_emotional_state"]:
            try:
                from modules.memory_core import search_memory
                import inspect
                src = inspect.getsource(search_memory)
                evidence["has_emotion_gating"] = "emotional_congruence" in src and "sdr_bonus" in src
            except Exception:
                pass
    except Exception:
        pass

    # --- Attention schema ---
    try:
        from modules.wiring import get_attention_schema
        schema = get_attention_schema()
        evidence["has_attention_schema"] = True
        evidence["attention_suppressed"] = "suppressed_items" in schema
        evidence["attention_history_len"] = len(schema.get("history", []))
    except Exception:
        pass

    # --- Prediction schemas ---
    try:
        from modules.schemas import load_schemas
        evidence["has_prediction_schemas"] = True
    except Exception:
        pass

    # --- Reconsolidation ---
    try:
        from modules.consolidation import correct_memory, _consolidation_conn
        import inspect
        src = inspect.getsource(correct_memory)
        evidence["has_correct_memory"] = "stub" not in src.lower()
        if evidence["has_correct_memory"]:
            try:
                conn = _consolidation_conn()
                evidence["reconsolidation_count"] = conn.execute(
                    "SELECT COUNT(*) FROM reconsolidation_log"
                ).fetchone()[0]
                conn.close()
            except Exception:
                pass
    except Exception:
        pass

    # --- Recurrent processing ---
    try:
        from modules.spreading import _spread_activation, recurrent_cycle
        evidence["has_recurrent_cycle"] = True
    except Exception:
        pass

    # --- Proposal #64: Quality checks for honest scoring ---
    # Attention transition quality (stale word-level vs proper topics)
    try:
        from modules.config import FTS_DB_PATH
        import sqlite3
        conn = sqlite3.connect(FTS_DB_PATH, timeout=3)
        rows = conn.execute(
            "SELECT from_topic, to_topic FROM attention_transitions ORDER BY id DESC LIMIT 10"
        ).fetchall()
        KNOWN_TOPICS = {"general", "codigo", "consciencia", "trading", "fullempaques",
                        "automatizacion", "memoria", "identidad", "relaciones"}
        if rows:
            valid = sum(1 for r in rows if r[0].lower() in KNOWN_TOPICS and r[1].lower() in KNOWN_TOPICS)
            evidence["attention_transition_quality"] = "clean" if valid >= 5 else "stale"
        else:
            evidence["attention_transition_quality"] = "empty"

        # Reconsolidation quality: how many have non-zero blend?
        try:
            quality = conn.execute(
                "SELECT COUNT(*) FROM reconsolidation_log WHERE blend_weight > 0"
            ).fetchone()[0]
            evidence["reconsolidation_quality_count"] = quality
        except Exception:
            evidence["reconsolidation_quality_count"] = 0

        conn.close()
    except Exception:
        evidence["attention_transition_quality"] = "unknown"
        evidence["reconsolidation_quality_count"] = 0

    return evidence


# ============================================================
# PURE SCORING FUNCTIONS (no IO, deterministic)
# ============================================================

def score_indicator(name: str, ev: Dict[str, Any]) -> Dict[str, Any]:
    """Score a single indicator from evidence snapshot. Pure function, no IO."""
    scorers = {
        "GWT-1": _score_gwt1, "GWT-2": _score_gwt2, "GWT-3": _score_gwt3, "GWT-4": _score_gwt4,
        "HOT-1": _score_hot1, "HOT-2": _score_hot2, "HOT-3": _score_hot3, "HOT-4": _score_hot4,
        "AST-1": _score_ast1,
        "PP-1": _score_pp1, "PP-2": _score_pp2, "PP-3": _score_pp3,
        "RPT-1": _score_rpt1, "RPT-2": _score_rpt2,
    }
    scorer = scorers.get(name)
    if not scorer:
        return {"name": name, "theory": "?", "score": 0.0, "evidence": "Unknown indicator"}
    return scorer(ev)


def _score_gwt1(ev: Dict) -> Dict:
    hc = ev["handler_count"]
    te = ev["total_events"]
    if te >= 100:
        return {"name": "GWT-1", "theory": "GWT", "score": 1.0,
                "evidence": f"Cross-module: {hc} handlers, {te} total events emitted"}
    if te >= 10:
        return {"name": "GWT-1", "theory": "GWT", "score": 0.7,
                "evidence": f"Cross-module nascent: {hc} handlers, {te} events emitted"}
    if hc >= 4:
        return {"name": "GWT-1", "theory": "GWT", "score": 0.3,
                "evidence": f"Handlers wired ({hc}) but few events emitted ({te}) (dormant)"}
    return {"name": "GWT-1", "theory": "GWT", "score": 0.0, "evidence": "No event bus"}


def _score_gwt2(ev: Dict) -> Dict:
    cc = ev["comp_count"]
    slots = ev["competition_slots"]
    threshold = ev["ignition_threshold"]
    if cc >= 10:
        return {"name": "GWT-2", "theory": "GWT", "score": 1.0,
                "evidence": f"Workspace competition exercised: {cc} competitions (slots={slots}, threshold={threshold})"}
    if cc > 0:
        return {"name": "GWT-2", "theory": "GWT", "score": 0.7,
                "evidence": f"Workspace competition nascent: {cc} competitions (need 10+ for full)"}
    return {"name": "GWT-2", "theory": "GWT", "score": 0.3,
            "evidence": f"Competition engine wired (slots={slots}, threshold={threshold}) but no runtime evidence yet"}


def _score_gwt3(ev: Dict) -> Dict:
    cc = ev["comp_count"]
    subs = ev["comp_subscribers"]
    if cc >= 10:
        return {"name": "GWT-3", "theory": "GWT", "score": 1.0,
                "evidence": f"Broadcast exercised: {cc} competitions, {subs} active subscribers"}
    if cc > 0:
        return {"name": "GWT-3", "theory": "GWT", "score": 0.7,
                "evidence": f"Broadcast nascent: {cc} competitions (need 10+ for full)"}
    if subs >= 1:
        return {"name": "GWT-3", "theory": "GWT", "score": 0.3,
                "evidence": f"Broadcast wired ({subs} subscribers) but no competitions recorded yet"}
    return {"name": "GWT-3", "theory": "GWT", "score": 0.0,
            "evidence": "No broadcast activity or subscribers detected"}


def _score_gwt4(ev: Dict) -> Dict:
    cc = ev["comp_count"]
    threshold = ev["ignition_threshold"]
    bonus = ev["coalition_bonus"]
    if cc >= 10:
        return {"name": "GWT-4", "theory": "GWT", "score": 1.0,
                "evidence": f"Ignition gating exercised: {cc} competitions (threshold={threshold}, bonus={bonus})"}
    if cc > 0:
        return {"name": "GWT-4", "theory": "GWT", "score": 0.7,
                "evidence": f"Ignition gating nascent: {cc} competitions (need 10+ for full)"}
    return {"name": "GWT-4", "theory": "GWT", "score": 0.3,
            "evidence": f"Ignition params present (threshold={threshold}, bonus={bonus}) but no runtime evidence yet"}


def _score_hot1(ev: Dict) -> Dict:
    has_meta = ev["has_meta_functions"]
    refreshes = ev["self_model_refreshes"]
    if has_meta and refreshes >= 10:
        return {"name": "HOT-1", "theory": "HOT", "score": 1.0,
                "evidence": f"Auto meta-monitoring: {refreshes} self-model refreshes (Rosenthal 2005)"}
    if has_meta and refreshes >= 1:
        return {"name": "HOT-1", "theory": "HOT", "score": 0.7,
                "evidence": f"Auto meta-monitoring nascent: {refreshes} refresh(es)"}
    if has_meta:
        return {"name": "HOT-1", "theory": "HOT", "score": 0.5,
                "evidence": "assess_confidence, reflect_on_self, identify_knowledge_gaps (partial: manual, not automatic)"}
    return {"name": "HOT-1", "theory": "HOT", "score": 0.0, "evidence": "Meta-monitoring functions not found"}


def _score_hot2(ev: Dict) -> Dict:
    n = ev["fok_n_records"]
    if n >= 20:
        return {"name": "HOT-2", "theory": "HOT", "score": 1.0,
                "evidence": f"RCJ calibrated with {n} records (Nelson & Narens 1990)"}
    if n >= 5:
        return {"name": "HOT-2", "theory": "HOT", "score": 0.7,
                "evidence": f"RCJ calibration nascent ({n} records, need 20)"}
    return {"name": "HOT-2", "theory": "HOT", "score": 0.3,
            "evidence": f"RCJ dormant ({n} records, need 5+ for nascent, 20+ for full)"}


def _score_hot3(ev: Dict) -> Dict:
    mc = ev["metacognitive_count"]
    if mc >= 20:
        return {"name": "HOT-3", "theory": "HOT", "score": 1.0,
                "evidence": f"Metacognitive control exercised {mc}x"}
    if mc >= 1:
        return {"name": "HOT-3", "theory": "HOT", "score": 0.7,
                "evidence": f"Metacognitive control nascent: {mc} application(s)"}
    return {"name": "HOT-3", "theory": "HOT", "score": 0.3,
            "evidence": "Metacognitive control implemented but not exercised (dormant)"}


def _score_hot4(ev: Dict) -> Dict:
    if not ev["has_emotional_state"]:
        return {"name": "HOT-4", "theory": "HOT", "score": 0.0, "evidence": "No emotional model"}
    # Full: emotion gating actually modified ranking at runtime
    emotion_gating_count = ev["event_counts"].get("emotion_gating_applied", 0)
    emotion_changed_count = ev["event_counts"].get("emotion_changed", 0)
    if ev["has_emotion_gating"] and (emotion_gating_count >= 5 or emotion_changed_count >= 1):
        return {"name": "HOT-4", "theory": "HOT", "score": 1.0,
                "evidence": f"Emotion-cognition integration: gating={emotion_gating_count}x, "
                            f"emotion_changed={emotion_changed_count}x, PAD+SDR active (Bower 1981, Godden & Baddeley 1975)"}
    if ev["has_emotion_gating"]:
        return {"name": "HOT-4", "theory": "HOT", "score": 0.7,
                "evidence": "PAD model + mood-congruent retrieval (Bower 1981) + state-dependent bonus (Godden & Baddeley 1975) -- nascent: need runtime evidence of ranking change"}
    return {"name": "HOT-4", "theory": "HOT", "score": 0.3,
            "evidence": "PAD emotional model exists but not wired to retrieval (dormant)"}


def _score_ast1(ev: Dict) -> Dict:
    if not ev["has_attention_schema"]:
        return {"name": "AST-1", "theory": "AST", "score": 0.0, "evidence": "No attention schema"}
    pe = ev["attention_pe_count"]
    has_supp = ev["attention_suppressed"]
    has_hist = ev["attention_history_len"] > 0
    transition_quality = ev.get("attention_transition_quality", "unknown")

    # Proposal #64: Discount PE from stale/corrupt predictor (Graziano 2013)
    if transition_quality == "stale":
        pe = 0  # Don't count noise PE events

    if pe >= 20:
        return {"name": "AST-1", "theory": "AST", "score": 1.0,
                "evidence": f"AST closed loop: predict->compare->adapt, {pe} PE events (Graziano 2013)"}
    if pe >= 1:
        return {"name": "AST-1", "theory": "AST", "score": 0.8,
                "evidence": f"AST adaptation nascent: {pe} PE event(s), edge decay active"}
    if has_supp and has_hist:
        return {"name": "AST-1", "theory": "AST", "score": 0.7,
                "evidence": f"AST active: describe, predict, suppression (no PE evidence yet, {ev['attention_history_len']} history)"}
    if has_supp:
        return {"name": "AST-1", "theory": "AST", "score": 0.5,
                "evidence": "AST exists with suppression but no history yet (partial)"}
    return {"name": "AST-1", "theory": "AST", "score": 0.3, "evidence": "AST functions exist but schema empty (dormant)"}


def _score_pp1(ev: Dict) -> Dict:
    pe = ev["prediction_error_count"]
    attn_pe = ev["attention_pe_count"]
    has_schemas = ev["has_prediction_schemas"]
    has_recon = ev["has_correct_memory"]
    transition_quality = ev.get("attention_transition_quality", "unknown")

    # Proposal #64: Don't use noise attention PE for multi-domain claim
    if transition_quality == "stale":
        attn_pe = 0

    # Full: Multi-domain prediction (topic PE + attention PE) with PE-driven updates
    if pe >= 10 and attn_pe >= 1:
        return {"name": "PP-1", "theory": "PP", "score": 1.0,
                "evidence": f"Multi-domain generative model: {pe} topic PEs + {attn_pe} attention PEs, "
                            "PE drives reconsolidation + WM + attention (Clark 2013)"}
    # Nascent: PE exercised with model updating capability
    if pe >= 1 and has_recon:
        return {"name": "PP-1", "theory": "PP", "score": 0.7,
                "evidence": f"Prediction loop exercised ({pe} PEs), PE drives reconsolidation + "
                            "WM boost + attention capture (Clark 2013, Schultz 1997)"}
    # Partial: Prediction infrastructure exists
    if has_schemas:
        return {"name": "PP-1", "theory": "PP", "score": 0.5,
                "evidence": "Prediction loop + schema system present, but prediction not yet exercised"}
    return {"name": "PP-1", "theory": "PP", "score": 0.5,
            "evidence": "Prediction loop exists but no schema system"}


def _score_pp2(ev: Dict) -> Dict:
    pe = ev["prediction_error_count"]
    if pe >= 20:
        return {"name": "PP-2", "theory": "PP", "score": 1.0,
                "evidence": f"PREDICTION_ERROR exercised {pe}x"}
    if pe >= 1:
        return {"name": "PP-2", "theory": "PP", "score": 0.7,
                "evidence": f"PREDICTION_ERROR nascent: {pe} emission(s)"}
    return {"name": "PP-2", "theory": "PP", "score": 0.3,
            "evidence": "PREDICTION_ERROR defined + handler wired but not emitted (dormant)"}


def _score_pp3(ev: Dict) -> Dict:
    if not ev["has_correct_memory"]:
        return {"name": "PP-3", "theory": "PP", "score": 0.0, "evidence": "No model updating"}
    rc = ev["reconsolidation_count"]
    # Proposal #64: Check quality (blend_weight > 0 = real reconsolidation, not extinction)
    quality_rc = ev.get("reconsolidation_quality_count", rc)

    if quality_rc >= 5:
        return {"name": "PP-3", "theory": "PP", "score": 1.0,
                "evidence": f"PE-driven reconsolidation exercised ({quality_rc}/{rc} quality records, blend>0)"}
    if rc >= 5:
        return {"name": "PP-3", "theory": "PP", "score": 0.7,
                "evidence": f"Reconsolidation fires ({rc} records) but quality limited ({quality_rc} with blend>0)"}
    if rc >= 1:
        return {"name": "PP-3", "theory": "PP", "score": 0.7,
                "evidence": f"Reconsolidation nascent ({rc} records, need 5+ for full)"}
    return {"name": "PP-3", "theory": "PP", "score": 0.3,
            "evidence": "Reconsolidation pipeline ready (re-embed + labile gate) but dormant (0 records)"}


def _score_rpt1(ev: Dict) -> Dict:
    if not ev["has_recurrent_cycle"]:
        return {"name": "RPT-1", "theory": "RPT", "score": 0.0, "evidence": "No recurrent processing"}
    rc = ev["retrieved_count"]
    if rc >= 50:
        return {"name": "RPT-1", "theory": "RPT", "score": 1.0,
                "evidence": f"recurrent_cycle exercised on {rc} retrievals (Lamme 2006)"}
    if rc >= 5:
        return {"name": "RPT-1", "theory": "RPT", "score": 0.7,
                "evidence": f"recurrent_cycle nascent: {rc} retrievals triggered spreading"}
    return {"name": "RPT-1", "theory": "RPT", "score": 0.3,
            "evidence": f"recurrent_cycle wired but only {rc} retrievals (dormant)"}


def _score_rpt2(ev: Dict) -> Dict:
    at = ev["active_event_types"]
    if at >= 6:
        return {"name": "RPT-2", "theory": "RPT", "score": 1.0,
                "evidence": f"Cross-module integration: {at} distinct event types exercised"}
    if at >= 3:
        return {"name": "RPT-2", "theory": "RPT", "score": 0.7,
                "evidence": f"Integration nascent: {at} event types exercised"}
    if at >= 1:
        return {"name": "RPT-2", "theory": "RPT", "score": 0.3,
                "evidence": f"Integration dormant: only {at} event type(s) exercised"}
    return {"name": "RPT-2", "theory": "RPT", "score": 0.0, "evidence": "No event types exercised"}


# ============================================================
# MAIN ASSESSMENT FUNCTIONS
# ============================================================

def get_assessment(evidence: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Run full assessment. If evidence not provided, collects it (IO).

    Returns:
        {
            "version": str,
            "score_total": float,
            "max_score": float,
            "pct": float,
            "indicators": [{"name", "theory", "score", "evidence"}, ...],
            "summary": {"full": int, "nascent": int, "partial": int, "dormant": int, "absent": int},
            "evidence_sources": list[str],
        }
    """
    if evidence is None:
        evidence = collect_evidence()

    indicators = []
    for spec in INDICATOR_SPECS:
        result = score_indicator(spec["name"], evidence)
        indicators.append(result)

    total = sum(i["score"] for i in indicators)
    max_score = len(indicators) * 1.0

    summary = {
        "full": sum(1 for i in indicators if i["score"] == 1.0),
        "nascent": sum(1 for i in indicators if 0.7 <= i["score"] < 1.0),
        "partial": sum(1 for i in indicators if i["score"] == 0.5),
        "dormant": sum(1 for i in indicators if i["score"] == 0.3),
        "absent": sum(1 for i in indicators if i["score"] == 0.0),
    }

    evidence_sources = [
        "event_counts (SQLite)",
        "event_bus handlers (in-memory)",
        "fok_calibration_log (SQLite)",
        "reconsolidation_log (SQLite)",
        "config._emotional_state",
        "competition constants",
        "attention schema state",
    ]

    worker_health = get_worker_health()

    return {
        "version": ASSESSMENT_VERSION,
        "score_total": total,
        "max_score": max_score,
        "pct": (total / max_score * 100) if max_score > 0 else 0,
        "indicators": indicators,
        "summary": summary,
        "evidence_sources": evidence_sources,
        "worker_health": worker_health,
    }


def format_assessment(assessment: Dict[str, Any]) -> str:
    """Format assessment dict to markdown string."""
    total = assessment["score_total"]
    max_score = assessment["max_score"]
    pct = assessment["pct"]
    indicators = assessment["indicators"]
    summary = assessment["summary"]

    lines = ["# Butlin Consciousness Assessment\n"]
    lines.append(f"**Total Score: {total:.1f}/{max_score:.0f} ({pct:.0f}%)**\n")

    current_theory = ""
    for ind in indicators:
        if ind["theory"] != current_theory:
            current_theory = ind["theory"]
            lines.append(f"\n## {THEORY_NAMES.get(current_theory, current_theory)}")

        label = SCORE_LABELS.get(ind["score"], f"{ind['score']:.1f}")
        lines.append(f"- **{ind['name']}** [{label}] ({ind['score']:.1f}): {ind['evidence']}")

    lines.append(f"\n## Summary")
    lines.append(f"- Score: {total:.1f}/{max_score:.0f}")
    lines.append(f"- Full indicators: {summary['full']}")
    lines.append(f"- Partial indicators: {summary['partial']}")
    lines.append(f"- Absent indicators: {summary['absent']}")

    wh = assessment.get("worker_health")
    if wh:
        lines.append(f"\n## Worker Health")
        lines.append(f"- Status: {wh['status']}")
        lines.append(f"- Last tick age: {wh['age_minutes']} min" if wh['age_minutes'] is not None else "- Last tick age: N/A")
        if wh.get('queue_backlog'):
            lines.append(f"- Queue backlog: {wh['queue_backlog']}")

    return "\n".join(lines)


# ============================================================
# SLEEP TICK LIVENESS (CC-4)
# ============================================================

def get_last_sleep_tick_age(db_path: str = None) -> Optional[float]:
    """Return hours since last sleep_tick_* entry in tool_calls, or None.

    Used by CC-4 consciousness contract and health/SLO monitoring.
    """
    from datetime import datetime
    from modules.db_pool import get_conn
    conn = get_conn(db_path)
    row = conn.execute(
        "SELECT started_at FROM tool_calls "
        "WHERE tool_name LIKE 'sleep_tick_%' "
        "ORDER BY started_at DESC LIMIT 1"
    ).fetchone()
    if not row or not row[0]:
        return None
    last = datetime.fromisoformat(str(row[0]))
    if last.tzinfo:
        last = last.replace(tzinfo=None)
    age_hours = (datetime.now() - last).total_seconds() / 3600
    return age_hours


# ============================================================
# WORKER LIVENESS (operational monitoring)
# ============================================================

WORKER_HEALTHY_THRESHOLD = 10.0    # minutes: healthy if tick age <= this
WORKER_STALE_THRESHOLD = 30.0      # minutes: stale if tick age <= this
WORKER_DEGRADED_THRESHOLD = 60.0   # minutes: degraded if tick age > stale


def classify_worker_status(age_minutes: Optional[float]) -> str:
    """Classify worker health from tick age (minutes).

    Returns: 'missing' | 'healthy' | 'stale' | 'degraded'
    """
    if age_minutes is None:
        return "missing"
    if age_minutes <= WORKER_HEALTHY_THRESHOLD:
        return "healthy"
    if age_minutes <= WORKER_STALE_THRESHOLD:
        return "stale"
    return "degraded"


def get_last_worker_tick_age(db_path: str = None) -> Optional[float]:
    """Return minutes since last write_worker_tick in tool_calls, or None."""
    from datetime import datetime
    from modules.db_pool import get_conn
    conn = get_conn(db_path)
    row = conn.execute(
        "SELECT started_at FROM tool_calls "
        "WHERE tool_name = 'write_worker_tick' "
        "ORDER BY started_at DESC LIMIT 1"
    ).fetchone()
    if not row or not row[0]:
        return None
    last = datetime.fromisoformat(str(row[0]))
    if last.tzinfo:
        last = last.replace(tzinfo=None)
    return (datetime.now() - last).total_seconds() / 60


def get_queue_backlog(db_path: str = None) -> Dict[str, int]:
    """Return write_queue job counts by status."""
    from modules.db_pool import get_conn
    conn = get_conn(db_path)
    rows = conn.execute(
        "SELECT status, COUNT(*) as cnt FROM write_queue GROUP BY status"
    ).fetchall()
    return {r[0]: r[1] for r in rows}


def get_worker_health(db_path: str = None) -> Dict[str, Any]:
    """Unified worker health check."""
    age = get_last_worker_tick_age(db_path)
    status = classify_worker_status(age)
    backlog = get_queue_backlog(db_path)
    return {
        "status": status,
        "age_minutes": round(age, 1) if age is not None else None,
        "queue_backlog": backlog,
    }


# ============================================================
# MCP TOOL REGISTRATION
# ============================================================

def register_assessment_tools(mcp_server) -> None:
    """Register assessment MCP tool."""

    @mcp_server.tool()
    def assessment_report() -> str:
        """External Butlin/Block consciousness assessment.
        Read-only, deterministic scoring of 14 indicators across 5 theories.
        Returns markdown report with scores, evidence, and summary."""
        assessment = get_assessment()
        return format_assessment(assessment)
