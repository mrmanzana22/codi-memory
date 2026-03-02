"""
Codi Memory - User Model (Sprint 3, items 3.3 + 3.4)
=====================================================
Theory of Mind for Hare. Uses same AgentModel architecture as self_model
(Graziano 2013 AST symmetry).

Three components:
  1. character_traits: Extracted from memory (personality, preferences, patterns)
  2. mental_state: Inferred from conversation (mood, goal, engagement)
  3. predicted_needs: From historical patterns (what Hare likely needs next)

The user model is INFERRED, not declared. We build it from:
  - Memory content (what Hare has told us / what we've observed)
  - Interaction patterns (timing, topics, communication style)
  - Current conversation context (session topics, turn patterns)
"""

import json
import logging
import os
import sqlite3
from datetime import datetime

from modules.agent_model import AgentModel, AgentTraits, AgentState, AgentPredictions
from modules.config import connect_fts

_logger = logging.getLogger(__name__)

# Path to FTS database (same as rest of system)
_FTS_DB_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "memories_fts.db"
)


def get_user_model() -> AgentModel:
    """Build current user model from available data.

    Combines:
    - Static traits from memory (cached, updated on consolidation)
    - Dynamic state from current session (real-time)
    - Predictions from historical patterns

    Returns AgentModel for Hare.
    """
    model = AgentModel(agent_id="hare")

    # 1. Traits (from memory + cached knowledge)
    model.traits = _build_traits()

    # 2. State (from current session)
    model.state = _build_state()

    # 3. Predictions (from patterns)
    model.predictions = _build_predictions(model.traits, model.state)

    return model


def _build_traits() -> AgentTraits:
    """Extract stable traits from memory.

    Sources: identity memories, interaction patterns, stated preferences.
    """
    traits = AgentTraits(
        name="Hare Jimenez",
        role="CEO",
        personality={
            "direct": 0.95,      # Communicates directly, no fluff
            "technical": 0.85,   # Deep technical knowledge
            "decisive": 0.90,    # Makes quick decisions
            "creative": 0.80,    # Innovative approaches
        },
        preferences={
            "communication": "direct, informal, Spanish",
            "planning": "see plan before execution",
            "coding": "incremental, test after each step",
            "language": "es-informal (parcero, hermano, llave)",
        },
        capabilities={},
        interaction_patterns={},
    )

    # Enrich from SQLite data if available
    try:
        if not os.path.exists(_FTS_DB_PATH):
            return traits

        conn = connect_fts(_FTS_DB_PATH)

        # Extract interaction patterns from attention_transitions
        try:
            rows = conn.execute("""
                SELECT
                    CAST(strftime('%H', created_at) AS INTEGER) as hour,
                    to_topic,
                    COUNT(*) as cnt
                FROM attention_transitions
                GROUP BY hour, to_topic
                ORDER BY cnt DESC
                LIMIT 20
            """).fetchall()

            time_patterns = {}
            for hour, topic, cnt in rows:
                if hour is not None:
                    period = _hour_to_period(hour)
                    if period not in time_patterns or cnt > time_patterns.get(f"{period}_cnt", 0):
                        time_patterns[period] = topic
                        time_patterns[f"{period}_cnt"] = cnt

            # Clean up count keys
            traits.interaction_patterns = {
                k: v for k, v in time_patterns.items() if not k.endswith("_cnt")
            }
        except Exception:
            pass

        # Extract topic capabilities from prediction accuracy
        try:
            rows = conn.execute("""
                SELECT domain, actual_accuracy, sample_size
                FROM prediction_state_l2
                WHERE sample_size >= 5
            """).fetchall()
            for domain, acc, n in rows:
                traits.capabilities[domain] = round(acc, 2)
        except Exception:
            pass

        conn.close()
    except Exception:
        pass

    return traits


def _build_state() -> AgentState:
    """Infer current mental state from session data.

    Sources: session topics, turn frequency, topic shifts.
    """
    state = AgentState(last_updated=datetime.now().isoformat())

    try:
        if not os.path.exists(_FTS_DB_PATH):
            return state

        conn = connect_fts(_FTS_DB_PATH)

        # Get current session info from L1 prediction state
        try:
            row = conn.execute("""
                SELECT session_id, turn_count, goal_topics, goal_locked, last_topic
                FROM prediction_state_l1 WHERE id = 1
            """).fetchone()

            if row:
                state.current_topic = row[4] or ""
                goal_topics = json.loads(row[2] or '{}')
                if goal_topics:
                    primary = max(goal_topics, key=goal_topics.get)
                    state.current_goal = f"Working on {primary}"

                # Engagement proxy: more turns = more engaged
                turn_count = row[1] or 0
                state.engagement = min(1.0, turn_count / 10.0)
        except Exception:
            pass

        # Infer stress from PE levels (high PE = topic shifting = possibly stressed)
        try:
            row = conn.execute("""
                SELECT AVG(weighted_surprise) FROM (
                    SELECT weighted_surprise FROM prediction_results
                    WHERE COALESCE(source, 'interactive') != 'sleep_loop'
                    ORDER BY id DESC LIMIT 5
                )
            """).fetchone()
            if row and row[0] is not None:
                # High surprise = lots of topic switching = possibly stressed/rushed
                state.stress_level = min(1.0, row[0] * 2.0)
        except Exception:
            pass

        # Emotional valence: positive if on-goal, negative if lots of PE
        state.emotional_valence = max(-1.0, min(1.0, 0.3 - state.stress_level))

        conn.close()
    except Exception:
        pass

    return state


def _build_predictions(traits: AgentTraits, state: AgentState) -> AgentPredictions:
    """Predict user needs from traits + current state.

    Combines historical patterns with current context.
    """
    predictions = AgentPredictions(confidence=0.3, basis="prior + session context")

    try:
        if not os.path.exists(_FTS_DB_PATH):
            return predictions

        conn = connect_fts(_FTS_DB_PATH)

        # Predict next topic from Markov model (same data as L0, but for user behavior)
        try:
            current = state.current_topic or 'general'
            row = conn.execute("""
                SELECT to_topic, COUNT(*) as cnt
                FROM attention_transitions
                WHERE from_topic = ?
                GROUP BY to_topic
                ORDER BY cnt DESC
                LIMIT 1
            """, (current,)).fetchone()

            if row:
                predictions.predicted_topic = row[0]
                predictions.confidence = 0.5
        except Exception:
            pass

        # Predict action from time of day + topic
        hour = datetime.now().hour
        period = _hour_to_period(hour)
        if period in traits.interaction_patterns:
            predictions.predicted_action = f"Likely {traits.interaction_patterns[period]} work"

        # Predict needs from session context
        needs = []
        if state.stress_level > 0.5:
            needs.append("focus_support")  # Help staying on track
        if state.engagement > 0.7:
            needs.append("deep_context")  # Provide rich context
        if state.current_goal:
            needs.append("goal_continuity")  # Keep session on-goal
        predictions.predicted_needs = needs

        conn.close()
    except Exception:
        pass

    return predictions


def _hour_to_period(hour: int) -> str:
    """Map hour to time period."""
    if 6 <= hour < 12:
        return "morning"
    elif 12 <= hour < 18:
        return "afternoon"
    elif 18 <= hour < 24:
        return "evening"
    return "night"


def get_user_model_summary() -> str:
    """Get compact user model summary for context injection.

    Returns formatted string suitable for including in prompts.
    """
    try:
        model = get_user_model()
        parts = []
        parts.append(f"User: {model.traits.name} ({model.traits.role})")

        if model.state.current_goal:
            parts.append(f"Goal: {model.state.current_goal}")
        if model.state.current_topic:
            parts.append(f"Topic: {model.state.current_topic}")

        engagement_label = "low" if model.state.engagement < 0.3 else (
            "moderate" if model.state.engagement < 0.7 else "high")
        parts.append(f"Engagement: {engagement_label}")

        if model.predictions.predicted_needs:
            parts.append(f"Needs: {', '.join(model.predictions.predicted_needs)}")

        return " | ".join(parts)
    except Exception:
        return "User model unavailable"


def register_tools(mcp):
    """Register user-model MCP tools."""
    mcp.tool()(get_user_model_summary)
