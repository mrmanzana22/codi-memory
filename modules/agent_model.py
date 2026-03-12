"""
Codi Memory - Agent Model Base (Sprint 3, items 3.3 + 3.4)
==========================================================
Shared architecture for self-model and user-model.

Graziano 2013 (AST): The brain uses the same system to model itself
and model others. Self-awareness = same architecture as social cognition.

Three components per agent:
  1. Traits: Stable characteristics (personality, preferences, capabilities)
  2. State: Current mental/emotional state (dynamic, changes per interaction)
  3. Predictions: Predicted behavior/needs (forward-looking)

Both self_model and user_model implement this interface.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class AgentTraits:
    """Stable characteristics of an agent (self or other).

    Graziano 2013: Attention schema includes a stable model of the
    agent's dispositional properties.
    """
    name: str = ""
    role: str = ""  # e.g., "CTO" for Codi, "CEO" for Hare
    personality: Dict[str, float] = field(default_factory=dict)
    # e.g., {"technical": 0.9, "creative": 0.7, "direct": 0.95}
    preferences: Dict[str, str] = field(default_factory=dict)
    # e.g., {"communication": "direct", "language": "es-informal"}
    capabilities: Dict[str, float] = field(default_factory=dict)
    # e.g., {"python": 0.9, "trading": 0.6}
    interaction_patterns: Dict[str, str] = field(default_factory=dict)
    # e.g., {"morning": "planning", "night": "coding"}


@dataclass
class AgentState:
    """Current dynamic state of an agent.

    Changes per interaction. Represents the agent's "right now".
    """
    current_goal: str = ""
    current_topic: str = ""
    emotional_valence: float = 0.0  # -1 to 1 (negative to positive)
    arousal: float = 0.0  # 0 to 1 (calm to excited)
    engagement: float = 0.5  # 0 to 1 (disengaged to fully engaged)
    stress_level: float = 0.0  # 0 to 1
    last_updated: str = ""


@dataclass
class AgentPredictions:
    """Forward-looking predictions about an agent's behavior.

    What we expect this agent to do/need next.
    """
    predicted_topic: str = ""
    predicted_action: str = ""
    predicted_needs: List[str] = field(default_factory=list)
    confidence: float = 0.5
    basis: str = ""  # What these predictions are based on


@dataclass
class AgentModel:
    """Complete model of an agent (Graziano 2013 AST).

    Used identically for self (Codi) and other (Hare).
    The symmetry is the key insight: self-awareness and
    theory-of-mind are the SAME computational mechanism.
    """
    agent_id: str = ""
    traits: AgentTraits = field(default_factory=AgentTraits)
    state: AgentState = field(default_factory=AgentState)
    predictions: AgentPredictions = field(default_factory=AgentPredictions)

    def summary(self) -> str:
        """Compact summary of the agent model."""
        parts = [f"Agent: {self.agent_id}"]
        if self.traits.role:
            parts.append(f"Role: {self.traits.role}")
        if self.state.current_goal:
            parts.append(f"Goal: {self.state.current_goal}")
        if self.state.current_topic:
            parts.append(f"Topic: {self.state.current_topic}")
        if self.predictions.predicted_needs:
            parts.append(f"Needs: {', '.join(self.predictions.predicted_needs[:3])}")
        return " | ".join(parts)

    def to_dict(self) -> dict:
        """Serialize to dict for storage/transmission."""
        return {
            "agent_id": self.agent_id,
            "traits": {
                "name": self.traits.name,
                "role": self.traits.role,
                "personality": dict(self.traits.personality),
                "preferences": dict(self.traits.preferences),
                "capabilities": dict(self.traits.capabilities),
                "interaction_patterns": dict(self.traits.interaction_patterns),
            },
            "state": {
                "current_goal": self.state.current_goal,
                "current_topic": self.state.current_topic,
                "emotional_valence": self.state.emotional_valence,
                "arousal": self.state.arousal,
                "engagement": self.state.engagement,
                "stress_level": self.state.stress_level,
                "last_updated": self.state.last_updated,
            },
            "predictions": {
                "predicted_topic": self.predictions.predicted_topic,
                "predicted_action": self.predictions.predicted_action,
                "predicted_needs": list(self.predictions.predicted_needs),
                "confidence": self.predictions.confidence,
                "basis": self.predictions.basis,
            },
        }
