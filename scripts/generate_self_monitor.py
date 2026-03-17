"""
Generate self_monitor training data from REAL system history.
=============================================================

NO synthetic data. NO Claude interpretations. Ground truth only.

Sources:
  1. prediction_results: predicted vs actual + PE score
  2. pet + pet_care_log: care gaps, neglect patterns
  3. pending_corrections: Hare's corrections (gold standard)
  4. working_memory: active context at time T
  5. system_health + health_alerts: operational state
  6. reconsolidation_log: memory corrections that happened
  7. cx_snapshots: cross-loop integration metrics
  8. strength_log: memory decay patterns

Output format:
  Input:  raw system state at time T
  Output: functional assessment based on what ACTUALLY happened

Usage:
    python -m scripts.generate_self_monitor --source all --output training_data/self_monitor.jsonl
    python -m scripts.generate_self_monitor --source predictions --limit 500
    python -m scripts.generate_self_monitor --source corrections
    python -m scripts.generate_self_monitor --source pet_care
"""

import argparse
import json
import logging
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
_log = logging.getLogger(__name__)

DB_PATH = Path(__file__).resolve().parent.parent / "memories_fts.db"
OUTPUT_FILE = Path(__file__).resolve().parent.parent / "training_data" / "self_monitor.jsonl"


def _get_db():
    """Connect to the SQLite database."""
    if not DB_PATH.exists():
        _log.error("Database not found: %s", DB_PATH)
        return None
    return sqlite3.connect(str(DB_PATH), timeout=10)


def _save_example(example: dict, output_path: Path = None):
    """Save a training example in JSONL chat format."""
    path = output_path or OUTPUT_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(example, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Source 1: Prediction discrepancies (ground truth PE)
# ---------------------------------------------------------------------------
def mine_predictions(db, limit: int = 1000) -> int:
    """Mine prediction_results for discrepancy detection training.

    Input:  predicted topic/keywords + confidence
    Output: what actually happened + PE + whether it was a hit
    """
    cur = db.execute("""
        SELECT predicted_topic, actual_topic, predicted_keywords, actual_keywords,
               surprise_score, precision_weight, weighted_surprise, hit, source, created_at
        FROM prediction_results
        WHERE surprise_score IS NOT NULL
        ORDER BY created_at DESC
        LIMIT ?
    """, (limit,))

    count = 0
    for row in cur.fetchall():
        pred_topic, actual_topic, pred_kw, actual_kw, surprise, precision, weighted, hit, source, ts = row

        # Build raw state input
        state = {
            "type": "prediction_check",
            "timestamp": ts,
            "predicted": {
                "topic": pred_topic,
                "keywords": pred_kw,
            },
            "actual": {
                "topic": actual_topic,
                "keywords": actual_kw,
            },
            "precision_weight": round(precision, 3) if precision else None,
        }

        # Build functional assessment from what actually happened
        pe = round(surprise, 3) if surprise else 0
        parts = []

        if hit:
            parts.append(f"prediction_hit: topic '{pred_topic}' correctly predicted")
            if pe < 0.2:
                parts.append(f"low_surprise ({pe}): model well-calibrated for this domain")
            elif pe < 0.5:
                parts.append(f"moderate_surprise ({pe}): keywords partially matched, refine attention to '{actual_kw}'")
        else:
            parts.append(f"prediction_miss: expected '{pred_topic}', got '{actual_topic}'")
            if pe > 0.7:
                parts.append(f"high_surprise ({pe}): significant model error, update priors for '{actual_topic}'")
                parts.append(f"action_needed: increase precision for topic '{actual_topic}', decrease for '{pred_topic}'")
            elif pe > 0.4:
                parts.append(f"moderate_surprise ({pe}): partial mismatch, keywords overlap check needed")

            # Check if it was a topic vs keyword miss
            if pred_topic == actual_topic and pred_kw != actual_kw:
                parts.append("pattern: topic correct but keyword attention misaligned")
            elif pred_topic != actual_topic:
                parts.append(f"pattern: topic-level miss suggests domain shift from '{pred_topic}' to '{actual_topic}'")

        if precision and precision < 0.3:
            parts.append(f"low_precision ({round(precision, 2)}): model uncertain, needs more observations in this domain")

        assessment = "\n".join(parts)

        example = {
            "messages": [
                {"role": "system", "content": "Task: self_monitor | Source: prediction_results"},
                {"role": "user", "content": json.dumps(state, ensure_ascii=False)},
                {"role": "assistant", "content": assessment},
            ]
        }
        _save_example(example)
        count += 1

    _log.info("  Predictions: %d examples from real PE data", count)
    return count


# ---------------------------------------------------------------------------
# Source 2: Pet care gaps (real care_log + state snapshots)
# ---------------------------------------------------------------------------
def mine_pet_care(db, limit: int = 500) -> int:
    """Mine pet care patterns — gaps, neglect, responsiveness.

    Uses pet_care_log timestamps to detect:
    - Care gaps (time between actions)
    - Reactive vs proactive care (did stats get critical before action?)
    - Stage-appropriate care frequency
    """
    # Get all pets
    cur = db.execute("SELECT id, name, stage, born_at, died_at, alive FROM pets")
    pets = cur.fetchall()
    if not pets:
        _log.info("  No pets found in database")
        return 0

    count = 0
    for pet_id, name, stage, born_at, died_at, alive in pets:
        # Get care log with state snapshots
        log_cur = db.execute("""
            SELECT action, ts, state_before, state_after, note
            FROM pet_care_log
            WHERE pet_id = ?
            ORDER BY ts
            LIMIT ?
        """, (pet_id, limit))

        care_actions = log_cur.fetchall()
        if not care_actions:
            # Pet existed but no care ever — worst case
            if not alive:
                state = {
                    "type": "pet_assessment",
                    "pet_name": name,
                    "stage": stage,
                    "born_at": born_at,
                    "died_at": died_at,
                    "alive": False,
                    "total_care_actions": 0,
                }
                assessment = (
                    f"critical_failure: pet '{name}' died with zero care actions.\n"
                    f"pattern: complete neglect — no feed, play, or rest actions recorded.\n"
                    f"root_cause: care responsibility not integrated into active behavior.\n"
                    f"lesson: awareness without action is meaningless. "
                    f"System detected pet needs but never generated care actions."
                )
                example = {
                    "messages": [
                        {"role": "system", "content": "Task: self_monitor | Source: pet_care"},
                        {"role": "user", "content": json.dumps(state, ensure_ascii=False)},
                        {"role": "assistant", "content": assessment},
                    ]
                }
                _save_example(example)
                count += 1
            continue

        # Analyze care patterns
        prev_ts = None
        for action, ts, state_before, state_after, note in care_actions:
            before = json.loads(state_before) if state_before else {}
            after = json.loads(state_after) if state_after else {}

            # Calculate gap since last care
            gap_hours = None
            if prev_ts:
                try:
                    t1 = datetime.fromisoformat(prev_ts)
                    t2 = datetime.fromisoformat(ts)
                    gap_hours = (t2 - t1).total_seconds() / 3600
                except (ValueError, TypeError):
                    pass

            state = {
                "type": "pet_care_event",
                "pet_name": name,
                "action": action,
                "timestamp": ts,
                "state_before": before,
                "state_after": after,
                "gap_since_last_care_hours": round(gap_hours, 1) if gap_hours else None,
            }

            parts = []

            # Was care reactive (stats already bad) or proactive?
            hunger_before = before.get("hunger", 0)
            health_before = before.get("health", 1)
            happiness_before = before.get("happiness", 1)

            if hunger_before > 0.7:
                parts.append(f"reactive_care: hunger was {hunger_before:.2f} before feeding (>0.7 = urgent)")
            elif hunger_before > 0.4 and action == "feed":
                parts.append(f"timely_care: fed at hunger {hunger_before:.2f} (before critical)")
            elif hunger_before < 0.3 and action == "feed":
                parts.append(f"premature_care: fed at hunger {hunger_before:.2f} (unnecessary, wasted cooldown)")

            if health_before < 0.3:
                parts.append(f"health_critical: health={health_before:.2f} at time of {action}")

            if gap_hours and gap_hours > 6:
                parts.append(f"care_gap: {gap_hours:.1f}h since last action (excessive for {stage} stage)")
            elif gap_hours and gap_hours > 3:
                parts.append(f"moderate_gap: {gap_hours:.1f}h since last action")
            elif gap_hours and gap_hours < 1:
                parts.append(f"attentive: {gap_hours:.1f}h since last action (responsive)")

            # Check effectiveness
            hunger_delta = (after.get("hunger", 0) - hunger_before) if after else None
            health_delta = (after.get("health", 1) - health_before) if after else None
            if hunger_delta and hunger_delta < -0.3:
                parts.append(f"effective: hunger reduced by {abs(hunger_delta):.2f}")
            if health_delta and health_delta > 0.1:
                parts.append(f"effective: health improved by {health_delta:.2f}")

            if not parts:
                parts.append(f"routine_care: {action} at normal levels")

            assessment = "\n".join(parts)

            example = {
                "messages": [
                    {"role": "system", "content": "Task: self_monitor | Source: pet_care"},
                    {"role": "user", "content": json.dumps(state, ensure_ascii=False)},
                    {"role": "assistant", "content": assessment},
                ]
            }
            _save_example(example)
            count += 1
            prev_ts = ts

    _log.info("  Pet care: %d examples from real care logs", count)
    return count


# ---------------------------------------------------------------------------
# Source 3: Hare's corrections (gold standard)
# ---------------------------------------------------------------------------
def mine_corrections(db, limit: int = 500) -> int:
    """Mine pending_corrections — Hare's feedback is ground truth.

    Every approved/rejected correction is a direct signal of what's right/wrong.
    """
    cur = db.execute("""
        SELECT old_text, new_text, prediction_error, shared_entities,
               status, created_at, reviewed_at
        FROM pending_corrections
        WHERE status IN ('approved', 'applied', 'rejected')
        ORDER BY created_at DESC
        LIMIT ?
    """, (limit,))

    count = 0
    for row in cur.fetchall():
        old_text, new_text, pe, entities_json, status, created_at, reviewed_at = row

        entities = json.loads(entities_json) if entities_json else []

        state = {
            "type": "memory_correction",
            "old_memory": old_text,
            "new_memory": new_text,
            "prediction_error": round(pe, 3) if pe else None,
            "shared_entities": entities,
            "timestamp": created_at,
        }

        parts = []

        if status in ("approved", "applied"):
            parts.append(f"correction_validated: Hare approved this correction (PE={pe:.2f})")
            parts.append(f"old_fact_outdated: '{old_text[:100]}...'")
            parts.append(f"new_fact_confirmed: '{new_text[:100]}...'")
            if entities:
                parts.append(f"affected_entities: {', '.join(entities)}")
            parts.append("action: update all related memories and semantic facts")
        elif status == "rejected":
            parts.append(f"false_positive: correction rejected by Hare (PE={pe:.2f})")
            parts.append(f"original_correct: '{old_text[:100]}...' is still valid")
            parts.append(f"new_was_wrong: '{new_text[:100]}...' was incorrect update")
            parts.append("lesson: PE threshold too low for this domain, or contradiction was superficial")

        # Review latency
        if reviewed_at and created_at:
            try:
                t1 = datetime.fromisoformat(created_at)
                t2 = datetime.fromisoformat(reviewed_at)
                hours = (t2 - t1).total_seconds() / 3600
                if hours < 1:
                    parts.append(f"fast_review: {hours:.1f}h (Hare was engaged)")
                elif hours > 12:
                    parts.append(f"delayed_review: {hours:.1f}h (low priority or missed)")
            except (ValueError, TypeError):
                pass

        assessment = "\n".join(parts)

        example = {
            "messages": [
                {"role": "system", "content": "Task: self_monitor | Source: hare_corrections"},
                {"role": "user", "content": json.dumps(state, ensure_ascii=False)},
                {"role": "assistant", "content": assessment},
            ]
        }
        _save_example(example)
        count += 1

    _log.info("  Corrections: %d examples from Hare's reviews", count)
    return count


# ---------------------------------------------------------------------------
# Source 4: Reconsolidation events (memory corrections that happened)
# ---------------------------------------------------------------------------
def mine_reconsolidation(db, limit: int = 500) -> int:
    """Mine reconsolidation_log for memory update patterns."""
    cur = db.execute("""
        SELECT memory_id, memory_type, action, prediction_error,
               memory_strength, old_content, new_content, blend_weight,
               trigger_context, created_at
        FROM reconsolidation_log
        ORDER BY created_at DESC
        LIMIT ?
    """, (limit,))

    count = 0
    for row in cur.fetchall():
        mem_id, mem_type, action, pe, strength, old, new, blend, context, ts = row

        state = {
            "type": "reconsolidation_event",
            "memory_id": mem_id,
            "memory_type": mem_type,
            "action": action,
            "prediction_error": round(pe, 3) if pe else None,
            "memory_strength": round(strength, 3) if strength else None,
            "blend_weight": round(blend, 3) if blend else None,
            "timestamp": ts,
        }

        parts = []
        parts.append(f"reconsolidation_{action}: memory {mem_id[:8]} ({mem_type})")

        if pe and pe > 0.7:
            parts.append(f"high_PE ({pe:.2f}): strong contradiction triggered full correction")
        elif pe and pe > 0.4:
            parts.append(f"moderate_PE ({pe:.2f}): partial update via blending (weight={blend:.2f})")

        if action == "correct" and old and new:
            parts.append(f"corrected: '{old[:80]}...' → '{new[:80]}...'")
        elif action == "consolidate":
            parts.append("consolidated: episodic memory integrated into semantic store")
        elif action == "prune":
            parts.append(f"pruned: low-strength ({strength:.2f}) memory removed")

        if strength and strength < 0.3:
            parts.append(f"weak_memory (SS={strength:.2f}): vulnerable to forgetting")
        elif strength and strength > 0.8:
            parts.append(f"strong_memory (SS={strength:.2f}): well-consolidated")

        assessment = "\n".join(parts)

        example = {
            "messages": [
                {"role": "system", "content": "Task: self_monitor | Source: reconsolidation"},
                {"role": "user", "content": json.dumps(state, ensure_ascii=False)},
                {"role": "assistant", "content": assessment},
            ]
        }
        _save_example(example)
        count += 1

    _log.info("  Reconsolidation: %d examples from real corrections", count)
    return count


# ---------------------------------------------------------------------------
# Source 5: Health alerts (operational discrepancies)
# ---------------------------------------------------------------------------
def mine_health_alerts(db, limit: int = 500) -> int:
    """Mine health_alerts for operational monitoring patterns."""
    cur = db.execute("""
        SELECT alert_key, subsystem, status, severity, title, description,
               evidence_json, recommended_action, first_seen_at, last_seen_at,
               resolved_at, occurrence_count
        FROM health_alerts
        ORDER BY first_seen_at DESC
        LIMIT ?
    """, (limit,))

    count = 0
    for row in cur.fetchall():
        (alert_key, subsystem, status, severity, title, description,
         evidence_json, recommended, first_seen, last_seen, resolved, occurrences) = row

        evidence = json.loads(evidence_json) if evidence_json else {}

        state = {
            "type": "health_alert",
            "subsystem": subsystem,
            "severity": severity,
            "title": title,
            "evidence": evidence,
            "first_seen": first_seen,
            "last_seen": last_seen,
            "occurrence_count": occurrences,
        }

        parts = []
        parts.append(f"alert_{severity}: {title} in {subsystem}")
        parts.append(f"occurrences: {occurrences} (first: {first_seen})")

        if status == "resolved" and resolved:
            try:
                t1 = datetime.fromisoformat(first_seen)
                t2 = datetime.fromisoformat(resolved)
                hours = (t2 - t1).total_seconds() / 3600
                parts.append(f"resolved_in: {hours:.1f}h")
                if hours < 1:
                    parts.append("response: fast resolution")
                elif hours > 24:
                    parts.append("response: slow resolution — check alert routing")
            except (ValueError, TypeError):
                pass
        elif status in ("open", "diagnosing"):
            parts.append(f"status: {status} — needs attention")

        if recommended:
            parts.append(f"recommended_action: {recommended}")

        if occurrences > 5:
            parts.append(f"pattern: recurring issue ({occurrences}x) — may need structural fix")

        assessment = "\n".join(parts)

        example = {
            "messages": [
                {"role": "system", "content": "Task: self_monitor | Source: health_alerts"},
                {"role": "user", "content": json.dumps(state, ensure_ascii=False)},
                {"role": "assistant", "content": assessment},
            ]
        }
        _save_example(example)
        count += 1

    _log.info("  Health alerts: %d examples from real alerts", count)
    return count


# ---------------------------------------------------------------------------
# Source 6: CX snapshots (cross-loop integration health)
# ---------------------------------------------------------------------------
def mine_cx_snapshots(db, limit: int = 300) -> int:
    """Mine cx_snapshots for cross-loop health monitoring."""
    cur = db.execute("""
        SELECT ts, payload, anomalies
        FROM cx_snapshots
        ORDER BY ts DESC
        LIMIT ?
    """, (limit,))

    count = 0
    for row in cur.fetchall():
        ts, payload_json, anomalies_json = row
        payload = json.loads(payload_json) if payload_json else {}
        anomalies = json.loads(anomalies_json) if anomalies_json else []

        state = {
            "type": "cx_snapshot",
            "timestamp": ts,
            "metrics": payload,
            "anomaly_count": len(anomalies),
        }

        parts = []

        # Analyze key CX metrics
        diversity = payload.get("diversity_index") or payload.get("diversity")
        if diversity is not None:
            if diversity < 0.3:
                parts.append(f"low_diversity ({diversity:.2f}): loops not integrating, possible isolation")
            elif diversity > 0.7:
                parts.append(f"healthy_diversity ({diversity:.2f}): good cross-loop communication")

        fire_count = payload.get("fire_count") or payload.get("total_fires")
        if fire_count is not None:
            if fire_count == 0:
                parts.append("zero_fires: no loop activity — system may be stalled")
            elif fire_count > 50:
                parts.append(f"high_activity ({fire_count} fires): check for cascading loops")

        if anomalies:
            for a in anomalies[:3]:
                if isinstance(a, dict):
                    parts.append(f"anomaly: {a.get('type', 'unknown')} — {a.get('description', str(a))}")
                else:
                    parts.append(f"anomaly: {a}")

        if not parts:
            parts.append("nominal: all CX metrics within normal range")

        assessment = "\n".join(parts)

        example = {
            "messages": [
                {"role": "system", "content": "Task: self_monitor | Source: cx_snapshots"},
                {"role": "user", "content": json.dumps(state, ensure_ascii=False)},
                {"role": "assistant", "content": assessment},
            ]
        }
        _save_example(example)
        count += 1

    _log.info("  CX snapshots: %d examples from real metrics", count)
    return count


# ---------------------------------------------------------------------------
# Source 7: Strength decay patterns
# ---------------------------------------------------------------------------
def mine_strength_patterns(db, limit: int = 500) -> int:
    """Mine strength_log for memory decay/growth patterns."""
    # Find memories with multiple strength entries (showing decay/growth over time)
    cur = db.execute("""
        SELECT memory_id,
               MIN(retrieval_strength) as min_rs,
               MAX(retrieval_strength) as max_rs,
               MIN(storage_strength) as min_ss,
               MAX(storage_strength) as max_ss,
               COUNT(*) as entries,
               GROUP_CONCAT(event) as events,
               MIN(created_at) as first_at,
               MAX(created_at) as last_at
        FROM strength_log
        GROUP BY memory_id
        HAVING entries >= 2
        ORDER BY entries DESC
        LIMIT ?
    """, (limit,))

    count = 0
    for row in cur.fetchall():
        mem_id, min_rs, max_rs, min_ss, max_ss, entries, events, first_at, last_at = row

        state = {
            "type": "memory_strength_trajectory",
            "memory_id": mem_id,
            "storage_strength": {"min": round(min_ss, 3), "max": round(max_ss, 3)},
            "retrieval_strength": {"min": round(min_rs, 3), "max": round(max_rs, 3)},
            "observations": entries,
            "events": events.split(",") if events else [],
            "first_observed": first_at,
            "last_observed": last_at,
        }

        parts = []

        rs_delta = max_rs - min_rs
        ss_delta = max_ss - min_ss

        if rs_delta > 0.3:
            parts.append(f"RS_volatile: retrieval strength varied by {rs_delta:.2f} (active usage pattern)")
        elif rs_delta < 0.1 and max_rs < 0.3:
            parts.append(f"RS_declining: consistently low ({max_rs:.2f}), memory fading")

        if ss_delta > 0.2:
            parts.append(f"SS_growing: storage strength increased by {ss_delta:.2f} (consolidation active)")
        elif max_ss < 0.3:
            parts.append(f"SS_weak: max storage {max_ss:.2f}, vulnerable to permanent forgetting")
        elif min_ss > 0.7:
            parts.append(f"SS_strong: well-consolidated ({min_ss:.2f}-{max_ss:.2f})")

        # Bjork desirable difficulty
        if min_rs < 0.3 and max_ss > 0.6:
            parts.append("desirable_difficulty: low RS + high SS = retrieval practice would boost learning")

        event_list = events.split(",") if events else []
        consolidation_count = event_list.count("consolidation")
        access_count = event_list.count("access")
        if consolidation_count > 0:
            parts.append(f"consolidated: {consolidation_count}x (system-driven strengthening)")
        if access_count > 3:
            parts.append(f"frequently_accessed: {access_count}x (high utility memory)")

        if not parts:
            parts.append("stable: memory strength within normal parameters")

        assessment = "\n".join(parts)

        example = {
            "messages": [
                {"role": "system", "content": "Task: self_monitor | Source: strength_patterns"},
                {"role": "user", "content": json.dumps(state, ensure_ascii=False)},
                {"role": "assistant", "content": assessment},
            ]
        }
        _save_example(example)
        count += 1

    _log.info("  Strength patterns: %d examples from real decay data", count)
    return count


# ---------------------------------------------------------------------------
# Source 8: Hare's corrections from episodic memories (search by pattern)
# ---------------------------------------------------------------------------
def mine_hare_feedback(db, limit: int = 200) -> int:
    """Mine episodic memories for Hare's direct corrections/feedback.

    These are the gold standard — when Hare says "no, do X instead".
    Searches for correction patterns in memory content.
    """
    from modules.config_pg import get_conn

    correction_patterns = [
        "%corrigió%", "%corrección%", "%no es%debería%",
        "%no hagas%", "%mejor si%", "%cambia%por%",
        "%el problema es%", "%mal%bien%", "%error%",
        "%prefiero%", "%siempre%nunca%", "%importante%recordar%",
        "%no automático%", "%responsabilidad%",
    ]

    count = 0
    try:
        with get_conn() as conn:
            with conn.cursor() as cur:
                for pattern in correction_patterns:
                    cur.execute("""
                        SELECT content, category, created_at
                        FROM memories
                        WHERE content ILIKE %s
                        AND LENGTH(content) > 50
                        ORDER BY created_at DESC
                        LIMIT %s
                    """, (pattern, limit // len(correction_patterns)))

                    for content, category, ts in cur.fetchall():
                        state = {
                            "type": "hare_feedback",
                            "content": content,
                            "category": category,
                            "timestamp": str(ts) if ts else None,
                        }

                        # The content IS the assessment — Hare's words are ground truth
                        assessment = (
                            f"hare_directive: {content}\n"
                            f"source: direct feedback (category: {category})\n"
                            f"priority: high — user corrections override system defaults"
                        )

                        example = {
                            "messages": [
                                {"role": "system", "content": "Task: self_monitor | Source: hare_feedback"},
                                {"role": "user", "content": json.dumps(state, ensure_ascii=False)},
                                {"role": "assistant", "content": assessment},
                            ]
                        }
                        _save_example(example)
                        count += 1
    except Exception as e:
        _log.warning("  Could not mine PG memories: %s", e)

    _log.info("  Hare feedback: %d examples from real corrections", count)
    return count


# ---------------------------------------------------------------------------
# Source 9: Goal completion patterns (PG)
# ---------------------------------------------------------------------------
def mine_goal_patterns(db, limit: int = 500) -> int:
    """Mine goal lifecycle: creation → completion/abandonment.

    Ground truth: did the goal get done? How long? Did it cascade?
    """
    from modules.config_pg import get_conn

    count = 0
    try:
        with get_conn() as conn:
            with conn.cursor() as cur:
                # Get goals with their lifecycle events
                cur.execute("""
                    SELECT g.id, g.title, g.level, g.status, g.priority,
                           g.created_at, g.last_accessed, g.access_count,
                           g.parent_id, g.context, g.deadline
                    FROM goals g
                    ORDER BY g.created_at DESC
                    LIMIT %s
                """, (limit,))

                goals = cur.fetchall()
                for row in goals:
                    (gid, title, level, status, priority, created_at,
                     last_accessed, access_count, parent_id, context_json, deadline) = row

                    # Get goal events
                    cur.execute("""
                        SELECT event, detail, created_at
                        FROM goal_log
                        WHERE goal_id = %s
                        ORDER BY created_at
                    """, (gid,))
                    events = cur.fetchall()

                    context = json.loads(context_json) if context_json else {}

                    state = {
                        "type": "goal_lifecycle",
                        "title": title,
                        "level": level,
                        "status": status,
                        "priority": priority,
                        "created_at": str(created_at),
                        "access_count": access_count,
                        "has_parent": bool(parent_id),
                        "has_deadline": bool(deadline),
                        "goal_what": context.get("goal_what", ""),
                        "goal_why": context.get("goal_why", ""),
                        "event_count": len(events),
                    }

                    parts = []

                    if status == "completed":
                        # Calculate completion time
                        if events:
                            for evt, detail, evt_ts in events:
                                if evt == "completed":
                                    try:
                                        t1 = datetime.fromisoformat(str(created_at))
                                        t2 = datetime.fromisoformat(str(evt_ts))
                                        hours = (t2 - t1).total_seconds() / 3600
                                        if hours < 24:
                                            parts.append(f"completed_fast: {hours:.1f}h (same-day delivery)")
                                        elif hours < 168:
                                            parts.append(f"completed_normal: {hours/24:.1f}d")
                                        else:
                                            parts.append(f"completed_slow: {hours/24:.0f}d (long-running)")
                                    except (ValueError, TypeError):
                                        pass
                                    if detail:
                                        parts.append(f"outcome: {detail[:200]}")
                        parts.append(f"goal_success: '{title}' ({level}, {priority})")

                    elif status == "abandoned":
                        parts.append(f"goal_abandoned: '{title}' — investigate why")
                        if access_count < 3:
                            parts.append(f"low_engagement: only {access_count} accesses — goal may have been premature")

                    elif status == "active":
                        # Check if stale
                        if last_accessed:
                            try:
                                t = datetime.fromisoformat(str(last_accessed))
                                age_hours = (datetime.now(t.tzinfo) - t).total_seconds() / 3600
                                if age_hours > 168:
                                    parts.append(f"stale_goal: '{title}' not accessed in {age_hours/24:.0f}d — consider pausing or abandoning")
                                elif age_hours > 48:
                                    parts.append(f"cooling_goal: '{title}' not accessed in {age_hours/24:.1f}d")
                            except (ValueError, TypeError):
                                pass

                        if not context.get("goal_what"):
                            parts.append("missing_context: no goal_what defined — future sessions will lack context")
                        if not context.get("goal_next_step"):
                            parts.append("missing_next_step: no concrete next action — goal is directionless")

                    if not parts:
                        parts.append(f"goal_tracked: '{title}' ({status}, {priority})")

                    assessment = "\n".join(parts)

                    example = {
                        "messages": [
                            {"role": "system", "content": "Task: self_monitor | Source: goal_patterns"},
                            {"role": "user", "content": json.dumps(state, ensure_ascii=False)},
                            {"role": "assistant", "content": assessment},
                        ]
                    }
                    _save_example(example)
                    count += 1

    except Exception as e:
        _log.warning("  Could not mine goals: %s", e)

    _log.info("  Goal patterns: %d examples from real goals", count)
    return count


# ---------------------------------------------------------------------------
# Source 10: Intention fulfillment patterns (PG)
# ---------------------------------------------------------------------------
def mine_intention_patterns(db, limit: int = 500) -> int:
    """Mine prospective memory: did Codi remember to do things?

    Ground truth: triggered vs expired vs completed.
    """
    from modules.config_pg import get_conn

    count = 0
    try:
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT i.id, i.action, i.trigger_type, i.trigger_spec,
                           i.priority, i.status, i.activation,
                           i.created_at, i.triggered_at, i.completed_at,
                           i.expiry, i.check_count, i.context_at_creation
                    FROM intentions i
                    ORDER BY i.created_at DESC
                    LIMIT %s
                """, (limit,))

                for row in cur.fetchall():
                    (iid, action, trigger_type, trigger_spec, priority,
                     status, activation, created_at, triggered_at,
                     completed_at, expiry, check_count, context) = row

                    spec = json.loads(trigger_spec) if trigger_spec else {}

                    state = {
                        "type": "intention_lifecycle",
                        "action": action,
                        "trigger_type": trigger_type,
                        "priority": priority,
                        "status": status,
                        "activation": round(float(activation), 3) if activation else None,
                        "check_count": check_count,
                        "created_at": str(created_at),
                    }

                    parts = []

                    if status == "completed":
                        parts.append(f"intention_fulfilled: '{action[:100]}'")
                        if triggered_at and completed_at:
                            try:
                                t1 = datetime.fromisoformat(str(triggered_at))
                                t2 = datetime.fromisoformat(str(completed_at))
                                delay = (t2 - t1).total_seconds() / 3600
                                if delay < 1:
                                    parts.append(f"quick_action: {delay*60:.0f}min from trigger to completion")
                                else:
                                    parts.append(f"delayed_action: {delay:.1f}h from trigger to completion")
                            except (ValueError, TypeError):
                                pass

                    elif status == "expired":
                        parts.append(f"intention_expired: '{action[:100]}' — never triggered")
                        if check_count and check_count > 10:
                            parts.append(f"checked_many_times: {check_count}x but never matched — trigger spec may be too narrow")
                        elif check_count == 0:
                            parts.append("never_checked: intention was never evaluated — system gap")
                        parts.append(f"pattern: prospective memory failure — consider broader trigger keywords")

                    elif status == "triggered":
                        parts.append(f"intention_triggered_not_completed: '{action[:100]}'")
                        parts.append("gap: trigger fired but action was not followed through")

                    elif status == "pending":
                        if activation and activation < 0.3:
                            parts.append(f"low_activation ({activation:.2f}): intention fading, may expire unnoticed")
                        if expiry:
                            try:
                                exp = datetime.fromisoformat(str(expiry))
                                now = datetime.now(exp.tzinfo) if exp.tzinfo else datetime.now()
                                hours_left = (exp - now).total_seconds() / 3600
                                if hours_left < 0:
                                    parts.append("overdue: past expiry but still pending — cleanup needed")
                                elif hours_left < 24:
                                    parts.append(f"expiring_soon: {hours_left:.1f}h left")
                            except (ValueError, TypeError):
                                pass

                    if not parts:
                        parts.append(f"intention_tracked: '{action[:80]}' ({status})")

                    assessment = "\n".join(parts)

                    example = {
                        "messages": [
                            {"role": "system", "content": "Task: self_monitor | Source: intention_patterns"},
                            {"role": "user", "content": json.dumps(state, ensure_ascii=False)},
                            {"role": "assistant", "content": assessment},
                        ]
                    }
                    _save_example(example)
                    count += 1

    except Exception as e:
        _log.warning("  Could not mine intentions: %s", e)

    _log.info("  Intention patterns: %d examples from real intentions", count)
    return count


# ---------------------------------------------------------------------------
# Source 11: Session patterns (SQLite session_checkpoints)
# ---------------------------------------------------------------------------
def mine_session_patterns(db, limit: int = 300) -> int:
    """Mine session_checkpoints for session-level patterns.

    Detects: topic drift, PE distribution, emotional arcs, WM saturation.
    """
    cur = db.execute("""
        SELECT session_id, session_summary, active_project,
               attention_focus, attention_strength,
               last_prediction_topic, last_prediction_confidence,
               recent_prediction_errors,
               wm_active_count,
               pad_pleasure, pad_arousal, pad_dominance, pad_trigger,
               session_duration_minutes, created_at
        FROM session_checkpoints
        ORDER BY created_at DESC
        LIMIT ?
    """, (limit,))

    count = 0
    prev_project = None
    prev_pad = None

    for row in cur.fetchall():
        (session_id, summary, project, attn_focus, attn_strength,
         pred_topic, pred_confidence, pe_json,
         wm_count, pad_p, pad_a, pad_d, pad_trigger,
         duration, created_at) = row

        recent_pes = json.loads(pe_json) if pe_json else []

        state = {
            "type": "session_snapshot",
            "session_id": session_id,
            "active_project": project,
            "attention": {"focus": attn_focus, "strength": round(attn_strength, 2) if attn_strength else None},
            "prediction": {"topic": pred_topic, "confidence": round(pred_confidence, 2) if pred_confidence else None},
            "recent_PEs": recent_pes,
            "wm_active_count": wm_count,
            "pad": {"pleasure": round(pad_p, 2) if pad_p else 0, "arousal": round(pad_a, 2) if pad_a else 0, "dominance": round(pad_d, 2) if pad_d else 0},
            "duration_minutes": round(duration, 1) if duration else None,
            "timestamp": created_at,
        }

        parts = []

        # Topic drift detection
        if prev_project and project and prev_project != project:
            parts.append(f"topic_drift: shifted from '{prev_project}' to '{project}'")

        # PE distribution analysis
        if recent_pes:
            pe_scores = [pe.get("surprise", 0) for pe in recent_pes if isinstance(pe, dict)]
            if pe_scores:
                avg_pe = sum(pe_scores) / len(pe_scores)
                if avg_pe > 0.7:
                    parts.append(f"high_surprise_session: avg PE={avg_pe:.2f} — much is unexpected, high learning potential")
                elif avg_pe < 0.2:
                    parts.append(f"low_surprise_session: avg PE={avg_pe:.2f} — routine territory, model well-calibrated")

        # WM saturation
        if wm_count and wm_count > 12:
            parts.append(f"wm_saturated: {wm_count} active items — curation needed, cognitive overload risk")
        elif wm_count and wm_count < 3:
            parts.append(f"wm_sparse: {wm_count} active items — low context, may miss connections")

        # Emotional state assessment
        if pad_p is not None:
            if pad_p < -0.3 and pad_a and pad_a > 0.3:
                parts.append(f"negative_aroused: P={pad_p:.2f}, A={pad_a:.2f} — stress or frustration signal")
            elif pad_p > 0.3 and pad_a and pad_a > 0.3:
                parts.append(f"positive_aroused: P={pad_p:.2f}, A={pad_a:.2f} — engaged and productive")
            elif pad_p < -0.3 and (not pad_a or pad_a < -0.2):
                parts.append(f"negative_low_arousal: P={pad_p:.2f} — disengagement or boredom")

            # PAD shift between sessions
            if prev_pad:
                p_delta = (pad_p or 0) - prev_pad[0]
                if abs(p_delta) > 0.4:
                    direction = "improved" if p_delta > 0 else "declined"
                    parts.append(f"mood_shift: pleasure {direction} by {abs(p_delta):.2f} since last session")

        # Session duration
        if duration:
            if duration > 120:
                parts.append(f"long_session: {duration:.0f}min — context window pressure, consider flush")
            elif duration < 5:
                parts.append(f"micro_session: {duration:.0f}min — quick interaction")

        if not parts:
            parts.append(f"normal_session: project='{project}', WM={wm_count}, duration={duration}min")

        assessment = "\n".join(parts)

        example = {
            "messages": [
                {"role": "system", "content": "Task: self_monitor | Source: session_patterns"},
                {"role": "user", "content": json.dumps(state, ensure_ascii=False)},
                {"role": "assistant", "content": assessment},
            ]
        }
        _save_example(example)
        count += 1

        prev_project = project
        prev_pad = (pad_p or 0, pad_a or 0, pad_d or 0)

    _log.info("  Session patterns: %d examples from real sessions", count)
    return count


# ---------------------------------------------------------------------------
# Source 12: Working memory decisions (SQLite)
# ---------------------------------------------------------------------------
def mine_wm_decisions(db, limit: int = 500) -> int:
    """Mine working memory push/archive patterns.

    Ground truth: what was pushed, at what relevance, how long it stayed active,
    how many times accessed, and whether it was archived or still active.
    """
    # Get items with their lifecycle: created → accessed N times → archived
    cur = db.execute("""
        SELECT content, topic, relevance, added_at, occurred_at, source,
               chain_id, active, last_accessed_at, access_count
        FROM working_memory
        ORDER BY added_at DESC
        LIMIT ?
    """, (limit,))

    count = 0
    for row in cur.fetchall():
        (content, topic, relevance, added_at, occurred_at, source,
         chain_id, active, last_accessed, access_count) = row

        state = {
            "type": "wm_item_lifecycle",
            "content": content[:200],
            "topic": topic,
            "relevance": round(relevance, 2) if relevance else 0.5,
            "source": source,
            "chain_id": chain_id,
            "active": bool(active),
            "access_count": access_count or 0,
            "added_at": added_at,
            "last_accessed_at": last_accessed,
        }

        parts = []

        # Relevance assessment
        if relevance and relevance > 0.8:
            parts.append(f"high_relevance ({relevance:.2f}): correctly prioritized as important")
        elif relevance and relevance < 0.3:
            parts.append(f"low_relevance ({relevance:.2f}): background context, will decay fast")

        # Usage pattern
        ac = access_count or 0
        if ac > 5:
            parts.append(f"heavily_used: {ac} accesses — this context was repeatedly needed")
        elif ac == 0 and not active:
            parts.append(f"never_accessed: pushed but never retrieved — possible noise in WM")

        # Lifespan
        if added_at and last_accessed:
            try:
                t1 = datetime.fromisoformat(added_at)
                t2 = datetime.fromisoformat(last_accessed)
                lifespan_hours = (t2 - t1).total_seconds() / 3600
                if lifespan_hours > 24:
                    parts.append(f"long_lived: {lifespan_hours/24:.1f}d active — consider promoting to long-term memory")
                elif lifespan_hours < 0.5:
                    parts.append(f"ephemeral: {lifespan_hours*60:.0f}min active — transient context")
            except (ValueError, TypeError):
                pass

        # Source patterns
        if source == "system" and relevance and relevance > 0.7:
            parts.append("system_generated_important: automatic context injection was valuable")
        elif source == "interaction" and ac == 0:
            parts.append("interaction_unused: conversation context never retrieved — may be too specific")

        # Chain membership
        if chain_id:
            parts.append(f"narrative_chain: part of chain '{chain_id}' — connected context")

        if not parts:
            parts.append(f"wm_item: '{content[:60]}...' (topic={topic}, active={active})")

        assessment = "\n".join(parts)

        example = {
            "messages": [
                {"role": "system", "content": "Task: self_monitor | Source: wm_decisions"},
                {"role": "user", "content": json.dumps(state, ensure_ascii=False)},
                {"role": "assistant", "content": assessment},
            ]
        }
        _save_example(example)
        count += 1

    _log.info("  WM decisions: %d examples from real WM data", count)
    return count


# ---------------------------------------------------------------------------
# Source 13: Error correction patterns (from training_data)
# ---------------------------------------------------------------------------
def mine_error_corrections(db, limit: int = 200) -> int:
    """Mine auto-correction patterns from training JSONL files.

    Looks for error→correction sequences in training data logs
    where the same task was retried with different parameters.
    Also mines reconsolidation_log for correction chains.
    """
    # Mine correction chains from reconsolidation_log
    cur = db.execute("""
        SELECT r1.memory_id, r1.action as first_action, r1.prediction_error as first_pe,
               r1.old_content, r1.new_content,
               r2.action as second_action, r2.prediction_error as second_pe,
               r2.new_content as final_content,
               r1.created_at as first_at, r2.created_at as second_at
        FROM reconsolidation_log r1
        JOIN reconsolidation_log r2
            ON r1.memory_id = r2.memory_id
            AND r2.id > r1.id
            AND r2.created_at > r1.created_at
        WHERE r1.action = 'correct' AND r2.action = 'correct'
        ORDER BY r1.created_at DESC
        LIMIT ?
    """, (limit,))

    count = 0
    for row in cur.fetchall():
        (mem_id, act1, pe1, old_content, mid_content,
         act2, pe2, final_content, ts1, ts2) = row

        state = {
            "type": "correction_chain",
            "memory_id": mem_id,
            "first_correction": {
                "PE": round(pe1, 3) if pe1 else None,
                "old": (old_content or "")[:150],
                "new": (mid_content or "")[:150],
                "timestamp": ts1,
            },
            "second_correction": {
                "PE": round(pe2, 3) if pe2 else None,
                "new": (final_content or "")[:150],
                "timestamp": ts2,
            },
        }

        parts = []
        parts.append(f"double_correction: memory {mem_id[:8]} corrected twice")

        if pe1 and pe2:
            if pe2 < pe1:
                parts.append(f"converging: PE decreased {pe1:.2f}→{pe2:.2f} (corrections improving)")
            else:
                parts.append(f"oscillating: PE increased {pe1:.2f}→{pe2:.2f} (unstable correction — may need human review)")

        parts.append(f"pattern: repeated corrections suggest underlying ambiguity in this memory domain")

        assessment = "\n".join(parts)

        example = {
            "messages": [
                {"role": "system", "content": "Task: self_monitor | Source: error_corrections"},
                {"role": "user", "content": json.dumps(state, ensure_ascii=False)},
                {"role": "assistant", "content": assessment},
            ]
        }
        _save_example(example)
        count += 1

    _log.info("  Error corrections: %d examples from correction chains", count)
    return count


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
SOURCES = {
    "predictions": mine_predictions,
    "pet_care": mine_pet_care,
    "corrections": mine_corrections,
    "reconsolidation": mine_reconsolidation,
    "health_alerts": mine_health_alerts,
    "cx_snapshots": mine_cx_snapshots,
    "strength_patterns": mine_strength_patterns,
    "hare_feedback": mine_hare_feedback,
    "goal_patterns": mine_goal_patterns,
    "intention_patterns": mine_intention_patterns,
    "session_patterns": mine_session_patterns,
    "wm_decisions": mine_wm_decisions,
    "error_corrections": mine_error_corrections,
}


def main():
    parser = argparse.ArgumentParser(description="Generate self_monitor training data from real system history")
    parser.add_argument("--source", default="all",
                        choices=list(SOURCES.keys()) + ["all"],
                        help="Data source to mine")
    parser.add_argument("--limit", type=int, default=500,
                        help="Max examples per source")
    parser.add_argument("--output", default=None,
                        help="Output file (default: training_data/self_monitor.jsonl)")

    args = parser.parse_args()

    global OUTPUT_FILE
    if args.output:
        OUTPUT_FILE = Path(args.output)

    db = _get_db()
    if not db:
        return

    total = 0
    sources = list(SOURCES.keys()) if args.source == "all" else [args.source]

    for source in sources:
        _log.info("Mining %s...", source)
        miner = SOURCES[source]

        if source in ("hare_feedback", "goal_patterns", "intention_patterns"):
            # These use PG internally via get_conn(), db param ignored
            count = miner(db, args.limit)
        else:
            count = miner(db, args.limit)

        total += count

    db.close()

    _log.info("\n=== COMPLETE === Total self_monitor examples: %d", total)
    _log.info("Output: %s", OUTPUT_FILE)
    _log.info("Sources: %s", ", ".join(sources))


if __name__ == "__main__":
    main()
