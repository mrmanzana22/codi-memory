"""
Codi Memory - Global Workspace Competition Engine (WIRING-6)
============================================================
Central competition gate for Global Workspace Theory (Baars 1988, Dehaene 2014).

Multiple specialized modules generate candidates for conscious access.
This engine scores all candidates uniformly and selects winners for
the limited-capacity workspace broadcast channel.

Design: Pure logic, no Qdrant/SQLite dependencies. Fully unit-testable.

Created: 2026-02-13 (WIRING-6 - Phase 6.1)
"""

import logging
import math
import random
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from modules.config import now_iso

_logger = logging.getLogger(__name__)


# ============================================================
# CONSTANTS
# ============================================================

DEFAULT_WORKSPACE_SLOTS = 5       # Max winners per competition (GWT capacity)
ATTENTION_FOCUS_BONUS = 0.12      # Top-down attentional gain (Desimone & Duncan 1995)
IGNITION_THRESHOLD = 0.25         # Min activation for workspace access (Dehaene & Changeux 2011)
COALITION_TOPIC_BONUS = 0.10      # Bonus when multiple domains converge on same topic

# GWT-4 / S3-01: Iterative nonlinear ignition (Dehaene & Changeux 2003)
# N passes of logistic self-excitation + lateral inhibition.
# Creates bistable phase transition: above threshold → ignites to ~1.0,
# losers suppressed to ~0.0. NMDA-mediated recurrence.
N_AMPLIFICATION_PASSES = 3       # Iterations of recurrent dynamics
W_RECURRENT = 0.8                # Self-excitation weight per pass (logistic growth)
W_LATERAL_INHIBITION = 0.15      # Lateral inhibition weight per pass

# S2-03: Softmax policy selection (Luce 1959, Sutton & Barto 2018)
# Temperature controls exploration: high T = more random, low T = more greedy
SOFTMAX_TEMPERATURE = 8.0         # Gamma: inverse temperature for softmax (higher = more greedy)

VALID_DOMAINS = frozenset([
    "episodic", "semantic", "working_memory",
    "prospective", "prediction", "trigger",
    "session_index",
])


# ============================================================
# DATA TYPES
# ============================================================

@dataclass
class CompetitionCandidate:
    """A candidate competing for workspace access."""
    content: str
    source_domain: str         # One of VALID_DOMAINS
    activation: float          # Score 0-1 (from unified scorer or lightweight proxy)
    memory_id: str = ""
    metadata: dict = field(default_factory=dict)


@dataclass
class CompetitionResult:
    """Result of a workspace competition round."""
    winners: list              # list[CompetitionCandidate]
    losers: list               # list[CompetitionCandidate]
    timestamp: str = ""
    competition_id: str = ""


# ============================================================
# COMPETITION ENGINE
# ============================================================

def run_workspace_competition(
    candidates: list,
    slots: int = DEFAULT_WORKSPACE_SLOTS,
    current_focus: str = None,
) -> CompetitionResult:
    """Run GWT competition: score all candidates, select winners.

    Args:
        candidates: List of CompetitionCandidate from all domains.
        slots: Max winners (workspace capacity).
        current_focus: Current attention schema focus topic (optional).
            Candidates matching this get a small bonus.

    Returns:
        CompetitionResult with winners and losers.
    """
    if not candidates:
        return CompetitionResult(
            winners=[],
            losers=[],
            timestamp=now_iso(),
            competition_id=str(uuid.uuid4())[:8],
        )

    # Phase 1: Apply attention focus bonus (top-down bias)
    if current_focus:
        focus_lower = current_focus.lower()
        for c in candidates:
            topic = c.metadata.get("topic", "")
            if topic and focus_lower in topic.lower():
                c.activation = min(1.0, c.activation + ATTENTION_FOCUS_BONUS)

    # Phase 2: Coalition formation (LIDA-inspired)
    # Candidates from a single domain covering different topics get a bonus (diversity reward).
    _apply_coalition_bonus(candidates)

    # Phase 3: Ignition threshold (Dehaene & Changeux 2011)
    # Below threshold = unconscious processing, no broadcast access.
    above_threshold = [c for c in candidates if c.activation >= IGNITION_THRESHOLD]
    below_threshold = [c for c in candidates if c.activation < IGNITION_THRESHOLD]

    # S2-03: Softmax policy selection (Luce 1959, Sutton & Barto 2018)
    # Replaces deterministic argmax with probabilistic sampling.
    # P(i) = exp(gamma * a_i) / sum(exp(gamma * a_j))
    # This preserves exploration while favoring high-activation candidates.
    winners = _softmax_select(above_threshold, slots)
    winner_set = set(id(w) for w in winners)
    losers = [c for c in above_threshold if id(c) not in winner_set] + below_threshold

    # GWT-4: Recurrent amplification (phase transition / ignition)
    apply_recurrent_amplification(winners, losers)

    result = CompetitionResult(
        winners=winners,
        losers=losers,
        timestamp=now_iso(),
        competition_id=str(uuid.uuid4())[:8],
    )

    # Emit event if event bus is available (non-critical)
    _emit_competition_event(result)

    return result


def _softmax_select(candidates: list, k: int) -> list:
    """Softmax-weighted sampling without replacement (S2-03).

    P(i) = exp(gamma * a_i) / Z, where gamma = SOFTMAX_TEMPERATURE.
    Higher gamma -> more deterministic (approaches argmax).
    Lower gamma -> more exploration (approaches uniform).

    Luce 1959: choice axiom. Sutton & Barto 2018: softmax action selection.

    Falls back to argmax if sampling fails.
    """
    if not candidates or k <= 0:
        return []
    if len(candidates) <= k:
        return sorted(candidates, key=lambda c: c.activation, reverse=True)

    try:
        selected = []
        remaining = list(candidates)
        for _ in range(k):
            if not remaining:
                break
            # Compute softmax weights
            max_a = max(c.activation for c in remaining)
            weights = []
            for c in remaining:
                # Shift by max for numerical stability
                w = math.exp(SOFTMAX_TEMPERATURE * (c.activation - max_a))
                weights.append(w)
            total_w = sum(weights)
            if total_w <= 0:
                # Fallback: pick highest
                remaining.sort(key=lambda c: c.activation, reverse=True)
                selected.append(remaining.pop(0))
                continue
            # Weighted random selection
            probs = [w / total_w for w in weights]
            r = random.random()
            cumulative = 0.0
            chosen_idx = len(remaining) - 1
            for i, p in enumerate(probs):
                cumulative += p
                if r <= cumulative:
                    chosen_idx = i
                    break
            selected.append(remaining.pop(chosen_idx))
        # Sort winners by activation descending so winners[0] is the strongest
        selected.sort(key=lambda c: c.activation, reverse=True)
        return selected
    except Exception:
        _logger.exception("softmax_select failed for %d candidates, falling back to argmax", len(candidates))
        ranked = sorted(candidates, key=lambda c: c.activation, reverse=True)
        return ranked[:k]


def _apply_coalition_bonus(candidates: list) -> None:
    """PID-synergy coalition formation (S0-03, CL-06, G-INV-13).

    Reward DIVERSITY: candidates from a single domain that cover
    different topics get a bonus. This models cross-domain synergy
    (Partial Information Decomposition) — a coalition is more valuable
    when its members bring non-redundant information.

    Previous (broken): same-topic convergence rewarded redundancy.
    """
    from collections import defaultdict

    # Group by source_domain
    domain_map = defaultdict(list)
    for c in candidates:
        domain_map[c.source_domain].append(c)

    # Apply bonus when a domain has candidates covering 2+ different topics
    for domain, members in domain_map.items():
        topics = set()
        for m in members:
            topic = m.metadata.get("topic", "")
            if topic:
                topics.add(topic.lower())
        if len(topics) >= 2:
            # Diversity bonus: each member in a diverse domain gets rewarded
            for m in members:
                m.activation = min(1.0, m.activation + COALITION_TOPIC_BONUS)


def apply_recurrent_amplification(winners: list, losers: list) -> None:
    """GWT-4 / S3-01: Iterative nonlinear ignition (Dehaene & Changeux 2003).

    Sprint 5 FIX-13: ALL winners get amplified (proportionally by rank).
    Lateral inhibition only affects LOSERS, not other winners.

    Lamme 2006: recurrent processing is bidirectional across multiple areas.
    Cowan 2001: workspace holds 4+/-1 items with graded activation.
    Winner[0] gets full boost, winner[i] gets boost * 0.5^i (rank decay).
    """
    if not winners:
        return

    _RANK_DECAY = 0.5  # Each subsequent winner gets 50% less amplification

    for _ in range(N_AMPLIFICATION_PASSES):
        # Amplify ALL winners with rank-based decay
        for rank, w in enumerate(winners):
            a = w.activation
            rank_factor = _RANK_DECAY ** rank  # 1.0, 0.5, 0.25, ...
            boost = W_RECURRENT * rank_factor * a * (1.0 - a)
            w.activation = min(1.0, a + boost)

        # Lateral inhibition only on LOSERS (not other winners)
        top_a = winners[0].activation
        for loser in losers:
            loser.activation = max(0.0, loser.activation - W_LATERAL_INHIBITION * top_a)


def _emit_competition_event(result: CompetitionResult):
    """Emit WORKSPACE_COMPETITION_COMPLETE event. Best-effort."""
    try:
        from modules.events import event_bus, Events
        if hasattr(Events, 'WORKSPACE_COMPETITION_COMPLETE'):
            event_bus.emit(Events.WORKSPACE_COMPETITION_COMPLETE, {
                'competition_id': result.competition_id,
                'winner_count': len(result.winners),
                'loser_count': len(result.losers),
                'winner_domains': [w.source_domain for w in result.winners],
                'loser_ids': [l.memory_id for l in result.losers if l.memory_id],
                'loser_domains': [l.source_domain for l in result.losers],
                'top_activation': result.winners[0].activation if result.winners else 0.0,
                'timestamp': result.timestamp,
            })
    except Exception:
        _logger.exception("Failed to emit WORKSPACE_COMPETITION_COMPLETE event")
