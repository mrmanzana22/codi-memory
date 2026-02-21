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

import uuid
from dataclasses import dataclass, field
from datetime import datetime


# ============================================================
# CONSTANTS
# ============================================================

DEFAULT_WORKSPACE_SLOTS = 5       # Max winners per competition (GWT capacity)
ATTENTION_FOCUS_BONUS = 0.12      # Top-down attentional gain (Desimone & Duncan 1995)
IGNITION_THRESHOLD = 0.25         # Min activation for workspace access (Dehaene & Changeux 2011)
COALITION_TOPIC_BONUS = 0.10      # Bonus when multiple domains converge on same topic

# GWT-4: Recurrent amplification (Dehaene & Changeux 2003)
# Models NMDA-mediated bistable dynamics: winner self-excites, losers suppressed
AMPLIFICATION_GAIN = 0.15         # Winner recurrent self-excitation
LATERAL_INHIBITION = 0.12         # Loser suppression (strong lateral inhibition)

VALID_DOMAINS = frozenset([
    "episodic", "semantic", "working_memory",
    "prospective", "prediction", "trigger",
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
            timestamp=datetime.now().isoformat(),
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
    # Candidates from different domains converging on same topic get a bonus.
    _apply_coalition_bonus(candidates)

    # Phase 3: Ignition threshold (Dehaene & Changeux 2011)
    # Below threshold = unconscious processing, no broadcast access.
    above_threshold = [c for c in candidates if c.activation >= IGNITION_THRESHOLD]
    below_threshold = [c for c in candidates if c.activation < IGNITION_THRESHOLD]

    # Sort survivors by activation descending (highest wins)
    ranked = sorted(above_threshold, key=lambda c: c.activation, reverse=True)

    winners = ranked[:slots]
    losers = ranked[slots:] + below_threshold

    # GWT-4: Recurrent amplification (phase transition / ignition)
    apply_recurrent_amplification(winners, losers)

    result = CompetitionResult(
        winners=winners,
        losers=losers,
        timestamp=datetime.now().isoformat(),
        competition_id=str(uuid.uuid4())[:8],
    )

    # Emit event if event bus is available (non-critical)
    _emit_competition_event(result)

    return result


def _apply_coalition_bonus(candidates: list) -> None:
    """LIDA-inspired coalition formation.

    When candidates from 2+ different domains share a topic,
    they form a coalition and each gets a bonus. This models
    the binding of multi-source signals converging on one theme.
    """
    from collections import defaultdict

    # Group by topic
    topic_map = defaultdict(list)
    for c in candidates:
        topic = c.metadata.get("topic", "")
        if topic:
            topic_map[topic.lower()].append(c)

    # Apply bonus where 2+ domains converge on same topic
    for topic, members in topic_map.items():
        domains = set(m.source_domain for m in members)
        if len(domains) >= 2:
            for m in members:
                m.activation = min(1.0, m.activation + COALITION_TOPIC_BONUS)


def apply_recurrent_amplification(winners: list, losers: list) -> None:
    """GWT-4: Nonlinear ignition dynamics (Dehaene & Changeux 2003).

    After initial ranking, the primary winner gets recurrent self-excitation
    while ALL losers suffer lateral inhibition. This creates a phase transition
    (bistable dynamics): above threshold -> explosive amplification + suppression.

    NMDA-mediated recurrence makes workspace access all-or-none:
    the winner becomes much stronger, losers become much weaker.

    Only the top winner gets amplified (single content in spotlight).
    All losers get suppressed equally (uniform lateral inhibition).
    """
    if not winners:
        return

    # Recurrent self-excitation: top winner gets boosted
    winners[0].activation = min(1.0, winners[0].activation + AMPLIFICATION_GAIN)

    # Lateral inhibition: all losers suppressed
    for loser in losers:
        loser.activation = max(0.0, loser.activation - LATERAL_INHIBITION)


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
        pass
