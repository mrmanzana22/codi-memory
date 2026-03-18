"""
core/pad.py — PAD (Pleasure-Arousal-Dominance) emotion math.

Pure functions operating on PAD values. No mutable state.
The mutable _emotional_state dict stays in config.py (Phase 4 scope).

Functions promoted from utils.py (underscore prefix removed for public API).
"""

import math

# Emotion labels in Spanish (single source of truth)
CODI_EMOTION_MAP = {
    "exuberant": "emocionado y energizado",
    "dependent": "entusiasmado pero necesitando apoyo",
    "relaxed": "satisfecho y tranquilo",
    "docile": "calmado y receptivo",
    "hostile": "frustrado e irritado",
    "anxious": "ansioso e inquieto",
    "disdainful": "desinteresado",
    "bored": "apagado y sin energia",
}

# PAD octant → emotion label
_OCTANT_MAP = {
    "+P+A+D": "exuberant",
    "+P+A-D": "dependent",
    "+P-A+D": "relaxed",
    "+P-A-D": "docile",
    "-P+A+D": "hostile",
    "-P+A-D": "anxious",
    "-P-A+D": "disdainful",
    "-P-A-D": "bored",
}


def clamp_pad_value(value: float) -> float:
    """Force a value to [-1, 1] PAD range."""
    return max(-1.0, min(1.0, value))


def classify_emotion(p: float, a: float, d: float) -> str:
    """Classify PAD state into an emotion label using octants.

    The 8 octants of PAD space:
    +P +A +D = exuberant, +P +A -D = dependent,
    +P -A +D = relaxed,   +P -A -D = docile,
    -P +A +D = hostile,   -P +A -D = anxious,
    -P -A +D = disdainful, -P -A -D = bored
    """
    p_sign = "+" if p >= 0 else "-"
    a_sign = "+" if a >= 0 else "-"
    d_sign = "+" if d >= 0 else "-"
    octant = f"{p_sign}P{a_sign}A{d_sign}D"
    return _OCTANT_MAP.get(octant, "neutral")


def get_emotion_text(label: str) -> str:
    """Return Spanish text for an emotion label."""
    return CODI_EMOTION_MAP.get(label, "en estado neutral")


def calculate_emotional_intensity(p: float, a: float, d: float) -> float:
    """Calculate emotional intensity as distance from origin.

    Value between 0 (neutral) and ~1.73 (maximum).
    """
    return math.sqrt(p ** 2 + a ** 2 + d ** 2)


def compute_pad_retrieval_bias(
    memory_valence: str, current_pleasure: float
) -> float:
    """Compute mood-congruent retrieval bias (Bower 1981).

    Memories whose emotional valence matches current mood get a boost.

    Args:
        memory_valence: 'positive', 'negative', or 'neutral'
        current_pleasure: Current pleasure dimension (-1 to 1)

    Returns:
        Bias value (-0.05 to +0.1)
    """
    valence_map = {"positive": 1.0, "neutral": 0.0, "negative": -1.0}
    mem_val = valence_map.get(memory_valence, 0.0)
    congruence = mem_val * current_pleasure
    if congruence > 0:
        return min(0.1, congruence * 0.1)
    return max(-0.05, congruence * 0.05)


def compute_pad_encoding_boost(arousal: float, pleasure: float) -> float:
    """Compute emotional encoding strength boost (LaBar & Cabeza 2006).

    High arousal strengthens encoding.

    Args:
        arousal: Current arousal level (-1 to 1)
        pleasure: Current pleasure level (-1 to 1)

    Returns:
        Salience boost (0.0 to 0.3)
    """
    arousal_boost = max(0.0, arousal) * 0.2
    valence_boost = abs(pleasure) * 0.1
    return min(0.3, arousal_boost + valence_boost)
