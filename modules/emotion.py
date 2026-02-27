"""
Codi Memory - Emotion module.
PAD emotional model, mood baseline, emotion-tagged memories.
"""

import json

from qdrant_client.models import Filter, FieldCondition, MatchValue, Range

from modules.config import (
    memory, qdrant, USER_ID, COLLECTION_NAME,
    _emotional_state, CODI_EMOTION_MAP,
    now_iso, now_short,
)
from modules.secret_redact import redact_secrets
from modules.access_tracking import record_access
from modules.utils import (
    get_session_id, infer_themes, is_self_referential,
    resolve_memory_id, enrich_with_ownership,
    _clamp_pad_value, _classify_emotion,
    _get_emotion_text, _get_emotional_state, _calculate_emotional_intensity,
)

__all__ = [
    "set_emotional_state",
    "get_emotional_state",
    "update_mood_baseline",
    "apply_emotional_decay",
    "get_emotional_expression",
    "add_memory_with_emotion",
    "tag_memory_emotion",
    "search_by_emotion",
    "get_emotional_memories",
    "infer_emotion_from_text",
    "evolve_pad_from_text",
    "register_tools",
]


def set_emotional_state(pleasure: float, arousal: float, dominance: float, trigger: str = None) -> str:
    """
    Establece el estado emocional actual usando el modelo PAD.

    Args:
        pleasure: Nivel de placer/displacer (-1.0 a 1.0)
        arousal: Nivel de activacion/calma (-1.0 a 1.0)
        dominance: Nivel de dominancia/sumision (-1.0 a 1.0)
        trigger: Evento que causo el estado emocional (opcional)
    """
    try:
        global _emotional_state
        p = _clamp_pad_value(pleasure)
        a = _clamp_pad_value(arousal)
        d = _clamp_pad_value(dominance)

        if _emotional_state['current']['timestamp']:
            _emotional_state['history'].append(_emotional_state['current'].copy())
            _emotional_state['history'] = _emotional_state['history'][-20:]

        _emotional_state['current'] = {
            'pleasure': p, 'arousal': a, 'dominance': d,
            'timestamp': now_iso(), 'trigger': trigger
        }
        emotion_label = _classify_emotion(p, a, d)
        emotion_text = _get_emotion_text(emotion_label)
        intensity = _calculate_emotional_intensity(p, a, d)

        # Emit EMOTION_CHANGED event for cross-module communication
        try:
            from modules.events import event_bus, Events
            event_bus.emit(Events.EMOTION_CHANGED, {
                'pleasure': p, 'arousal': a, 'dominance': d,
                'emotion': emotion_label, 'intensity': round(intensity, 2),
                'trigger': trigger,
            })
        except Exception:
            pass

        result = {
            'result': 'Estado emocional actualizado',
            'state': {'pleasure': p, 'arousal': a, 'dominance': d, 'emotion': emotion_label, 'description': emotion_text, 'intensity': round(intensity, 2), 'trigger': trigger}
        }
        return json.dumps(result, ensure_ascii=False)
    except Exception as e:
        return json.dumps({'result': 'error', 'message': redact_secrets(str(e))})


def get_emotional_state(include_history: bool = False) -> str:
    """
    Obtiene el estado emocional actual.

    Args:
        include_history: Si incluir el historial de estados (default False)
    """
    try:
        current = _emotional_state['current']
        mood = _emotional_state['mood']

        if current['timestamp']:
            emotion_label = _classify_emotion(current['pleasure'], current['arousal'], current['dominance'])
            emotion_text = _get_emotion_text(emotion_label)
            intensity = _calculate_emotional_intensity(current['pleasure'], current['arousal'], current['dominance'])
        else:
            emotion_label = 'neutral'
            emotion_text = 'sin estado emocional establecido'
            intensity = 0.0

        mood_label = _classify_emotion(mood['pleasure'], mood['arousal'], mood['dominance'])
        mood_text = _get_emotion_text(mood_label)

        result = {
            'result': 'Estado emocional obtenido',
            'current': {
                'pleasure': current['pleasure'], 'arousal': current['arousal'], 'dominance': current['dominance'],
                'emotion': emotion_label, 'description': emotion_text, 'intensity': round(intensity, 2),
                'trigger': current['trigger'], 'timestamp': current['timestamp']
            },
            'mood_baseline': {
                'pleasure': mood['pleasure'], 'arousal': mood['arousal'], 'dominance': mood['dominance'],
                'emotion': mood_label, 'description': mood_text
            }
        }
        if include_history:
            result['history'] = _emotional_state['history'][-10:]
        return json.dumps(result, ensure_ascii=False)
    except Exception as e:
        return json.dumps({'result': 'error', 'message': redact_secrets(str(e))})


def update_mood_baseline(pleasure: float = None, arousal: float = None, dominance: float = None) -> str:
    """
    Ajusta el mood baseline (estado emocional de fondo).

    Args:
        pleasure: Nuevo nivel de placer baseline (-1.0 a 1.0, opcional)
        arousal: Nuevo nivel de activacion baseline (-1.0 a 1.0, opcional)
        dominance: Nuevo nivel de dominancia baseline (-1.0 a 1.0, opcional)
    """
    try:
        global _emotional_state
        if pleasure is not None:
            _emotional_state['mood']['pleasure'] = _clamp_pad_value(pleasure)
        if arousal is not None:
            _emotional_state['mood']['arousal'] = _clamp_pad_value(arousal)
        if dominance is not None:
            _emotional_state['mood']['dominance'] = _clamp_pad_value(dominance)
        _emotional_state['mood']['last_updated'] = now_iso()

        mood = _emotional_state['mood']
        mood_label = _classify_emotion(mood['pleasure'], mood['arousal'], mood['dominance'])
        mood_text = _get_emotion_text(mood_label)

        result = {
            'result': 'Mood baseline actualizado',
            'mood': {'pleasure': mood['pleasure'], 'arousal': mood['arousal'], 'dominance': mood['dominance'], 'emotion': mood_label, 'description': mood_text, 'last_updated': mood['last_updated']}
        }
        return json.dumps(result, ensure_ascii=False)
    except Exception as e:
        return json.dumps({'result': 'error', 'message': redact_secrets(str(e))})


def apply_emotional_decay() -> str:
    """Aplica decay al estado emocional actual, acercandolo al mood baseline."""
    try:
        global _emotional_state
        current = _emotional_state['current']
        mood = _emotional_state['mood']
        decay_rate = _emotional_state['decay_rate']

        if not current['timestamp']:
            return json.dumps({'result': 'Sin estado emocional para decaer', 'applied': False})

        # Asymmetric decay: positive fades faster, negative persists
        # Baumeister et al. 2001, Larsen & Prizmic 2008
        p_decay = decay_rate * 1.5 if current['pleasure'] > mood['pleasure'] else decay_rate * 0.8
        new_p = current['pleasure'] + (mood['pleasure'] - current['pleasure']) * p_decay
        new_a = current['arousal'] + (mood['arousal'] - current['arousal']) * decay_rate
        new_d = current['dominance'] + (mood['dominance'] - current['dominance']) * decay_rate

        _emotional_state['history'].append(current.copy())
        _emotional_state['history'] = _emotional_state['history'][-20:]

        _emotional_state['current'] = {
            'pleasure': new_p, 'arousal': new_a, 'dominance': new_d,
            'timestamp': now_iso(), 'trigger': 'decay'
        }
        emotion_label = _classify_emotion(new_p, new_a, new_d)
        emotion_text = _get_emotion_text(emotion_label)

        result = {
            'result': 'Decay emocional aplicado', 'decay_rate': decay_rate,
            'previous': {'pleasure': current['pleasure'], 'arousal': current['arousal'], 'dominance': current['dominance']},
            'current': {'pleasure': round(new_p, 3), 'arousal': round(new_a, 3), 'dominance': round(new_d, 3), 'emotion': emotion_label, 'description': emotion_text}
        }
        return json.dumps(result, ensure_ascii=False)
    except Exception as e:
        return json.dumps({'result': 'error', 'message': redact_secrets(str(e))})


def get_emotional_expression() -> str:
    """Obtiene una expresion natural en texto del estado emocional actual."""
    try:
        current = _emotional_state['current']
        if not current['timestamp']:
            return json.dumps({'result': 'Sin estado emocional', 'expression': 'Me siento en estado neutral, sin emociones particulares.', 'intensity': 'none'})

        p, a, d = current['pleasure'], current['arousal'], current['dominance']
        emotion_label = _classify_emotion(p, a, d)
        emotion_text = _get_emotion_text(emotion_label)
        intensity = _calculate_emotional_intensity(p, a, d)

        if intensity < 0.3:
            intensity_level, intensity_word = 'baja', 'ligeramente'
        elif intensity < 0.7:
            intensity_level, intensity_word = 'moderada', 'moderadamente'
        elif intensity < 1.2:
            intensity_level, intensity_word = 'alta', 'bastante'
        else:
            intensity_level, intensity_word = 'muy alta', 'muy'

        expression = f"Me siento {intensity_word} {emotion_text}" + (f" debido a: {current['trigger']}" if current['trigger'] else "")
        if d > 0.5:
            expression += ". Me siento en control de la situacion."
        elif d < -0.5:
            expression += ". Me siento algo vulnerable o dependiente."

        result = {
            'result': 'Expresion emocional generada', 'expression': expression,
            'emotion': emotion_label, 'emotion_spanish': emotion_text,
            'intensity': intensity_level, 'intensity_value': round(intensity, 2),
            'components': {
                'pleasure': 'positivo' if p > 0 else 'negativo' if p < 0 else 'neutral',
                'arousal': 'activado' if a > 0 else 'calmado' if a < 0 else 'neutral',
                'dominance': 'dominante' if d > 0 else 'sumiso' if d < 0 else 'equilibrado'
            }
        }
        return json.dumps(result, ensure_ascii=False)
    except Exception as e:
        return json.dumps({'result': 'error', 'message': redact_secrets(str(e))})


def add_memory_with_emotion(content: str, category: str = "general",
                            pleasure: float = 0.0, arousal: float = 0.0,
                            dominance: float = 0.0, source: str = "experienced",
                            importance: str = "medium") -> str:
    """
    Guarda una memoria con estado emocional PAD asociado.

    Args:
        content: El contenido a recordar
        category: Categoria (identidad, aprendizaje, episodio, proyecto, general)
        pleasure: Nivel de placer asociado (-1.0 a 1.0)
        arousal: Nivel de activacion asociado (-1.0 a 1.0)
        dominance: Nivel de dominancia asociado (-1.0 a 1.0)
        source: Como obtuve esta memoria (experienced, told, learned, inferred)
        importance: Importancia (critical, high, medium, low)
    """
    try:
        p = _clamp_pad_value(pleasure)
        a = _clamp_pad_value(arousal)
        d = _clamp_pad_value(dominance)
        emotion_label = _classify_emotion(p, a, d)
        intensity = _calculate_emotional_intensity(p, a, d)

        result = memory.add(messages=[{"role": "user", "content": content}], user_id=USER_ID, metadata={"category": category})

        if result and result.get("results"):
            for r in result["results"]:
                mem_id = r.get("id")
                if mem_id:
                    themes = infer_themes(content)
                    if not themes:
                        themes = [category]
                    self_ref = is_self_referential(content)
                    if self_ref and 'identidad' not in themes:
                        themes.append('identidad')

                    ownership_metadata = {
                        'ownership_is_mine': True, 'ownership_source': source,
                        'ownership_confidence': 0.9 if source == 'experienced' else 0.7,
                        'experiential_emotional_weight': min(intensity / 1.73, 1.0),
                        'experiential_emotional_valence': 'positive' if p > 0.2 else 'negative' if p < -0.2 else 'neutral',
                        'narrative_importance': importance, 'narrative_themes': themes,
                        'attention_salience': 0.7 if importance in ['critical', 'high'] else 0.5,
                        'attention_access_count': 0, 'attention_last_accessed': None,
                        'temporal_session_id': get_session_id(), 'self_reference': self_ref,
                        'pad_pleasure': p, 'pad_arousal': a, 'pad_dominance': d,
                        'pad_emotion': emotion_label, 'pad_intensity': intensity, '_v': 2.2
                    }
                    record_access(COLLECTION_NAME, mem_id, ownership_metadata)

        # P1: backup removed from hot path
        result_json = {
            'result': 'Memoria guardada con emocion',
            'memory_id': result.get('results', [{}])[0].get('id', 'unknown')[:8] if result else 'unknown',
            'emotion': {'label': emotion_label, 'description': _get_emotion_text(emotion_label), 'pleasure': p, 'arousal': a, 'dominance': d, 'intensity': round(intensity, 2)}
        }
        return json.dumps(result_json, ensure_ascii=False)
    except Exception as e:
        return json.dumps({'result': 'error', 'message': redact_secrets(str(e))})


def tag_memory_emotion(memory_id: str, pleasure: float, arousal: float, dominance: float) -> str:
    """
    Etiqueta una memoria existente con un estado emocional PAD.

    Args:
        memory_id: ID de la memoria (puede ser parcial)
        pleasure: Nivel de placer (-1.0 a 1.0)
        arousal: Nivel de activacion (-1.0 a 1.0)
        dominance: Nivel de dominancia (-1.0 a 1.0)
    """
    try:
        full_id = resolve_memory_id(memory_id)
        if not full_id:
            return json.dumps({'result': 'error', 'message': f"No encontre memoria con ID que empiece con '{memory_id}'"})

        p = _clamp_pad_value(pleasure)
        a = _clamp_pad_value(arousal)
        d = _clamp_pad_value(dominance)
        emotion_label = _classify_emotion(p, a, d)
        intensity = _calculate_emotional_intensity(p, a, d)

        record_access(COLLECTION_NAME, full_id, {
            'pad_pleasure': p, 'pad_arousal': a, 'pad_dominance': d,
            'pad_emotion': emotion_label, 'pad_intensity': intensity,
            'experiential_emotional_weight': min(intensity / 1.73, 1.0),
            'experiential_emotional_valence': 'positive' if p > 0.2 else 'negative' if p < -0.2 else 'neutral',
        })
        result = {
            'result': 'Memoria etiquetada con emocion', 'memory_id': memory_id,
            'emotion': {'label': emotion_label, 'description': _get_emotion_text(emotion_label), 'pleasure': p, 'arousal': a, 'dominance': d, 'intensity': round(intensity, 2)}
        }
        return json.dumps(result, ensure_ascii=False)
    except Exception as e:
        return json.dumps({'result': 'error', 'message': redact_secrets(str(e))})


def search_by_emotion(emotion_type: str, threshold: float = 0.3, limit: int = 10) -> str:
    """
    Busca memorias por tipo de emocion.

    Args:
        emotion_type: Tipo de emocion (exuberant, dependent, relaxed, docile, hostile, anxious, disdainful, bored)
        threshold: Umbral minimo de intensidad (default 0.3)
        limit: Maximo de resultados (default 10)
    """
    try:
        valid_emotions = ['exuberant', 'dependent', 'relaxed', 'docile', 'hostile', 'anxious', 'disdainful', 'bored']
        if emotion_type not in valid_emotions:
            return json.dumps({'result': 'error', 'message': f"Emocion no valida. Usar: {', '.join(valid_emotions)}"})

        points, _ = qdrant.scroll(
            collection_name=COLLECTION_NAME,
            scroll_filter=Filter(must=[
                FieldCondition(key='pad_emotion', match=MatchValue(value=emotion_type)),
                FieldCondition(key='pad_intensity', range=Range(gte=threshold))
            ]),
            limit=limit, with_payload=True
        )

        if not points:
            return json.dumps({'result': 'Sin resultados', 'emotion': emotion_type, 'memories': []})

        # Access tracking (same pattern as search_memory)
        try:
            from modules.access_tracking import record_access
            from modules.config import COLLECTION_NAME, now_iso
            _ts = now_iso()
            for _p in points:
                _mid = str(_p.id)
                _payload = _p.payload or {}
                _acc = int(_payload.get('attention_access_count', 0) or 0)
                _ats = list(_payload.get('access_timestamps', []) or [])
                if not isinstance(_ats, list):
                    _ats = []
                _ats.append(_ts)
                _ats = _ats[-20:]
                record_access(COLLECTION_NAME, _mid, {
                    'attention_access_count': _acc + 1,
                    'attention_last_accessed': _ts,
                    'access_timestamps': _ats,
                })
        except Exception:
            pass

        memories = []
        for p in points:
            memories.append({
                'id': str(p.id)[:8], 'content': p.payload.get('data', 'N/A')[:80],
                'emotion': p.payload.get('pad_emotion', 'unknown'),
                'intensity': round(p.payload.get('pad_intensity', 0), 2),
                'pleasure': p.payload.get('pad_pleasure', 0), 'arousal': p.payload.get('pad_arousal', 0),
                'dominance': p.payload.get('pad_dominance', 0)
            })

        return json.dumps({
            'result': f'Encontradas {len(memories)} memorias',
            'emotion': emotion_type, 'emotion_description': _get_emotion_text(emotion_type), 'memories': memories
        }, ensure_ascii=False)
    except Exception as e:
        return json.dumps({'result': 'error', 'message': redact_secrets(str(e))})


def get_emotional_memories(pleasure_range: str = None, arousal_range: str = None, limit: int = 10) -> str:
    """
    Busca memorias por rangos de valores PAD.

    Args:
        pleasure_range: "positive", "negative", o "neutral"
        arousal_range: "high", "low", o "neutral"
        limit: Maximo de resultados (default 10)
    """
    try:
        filters = []
        if pleasure_range == 'positive':
            filters.append(FieldCondition(key='pad_pleasure', range=Range(gte=0.2)))
        elif pleasure_range == 'negative':
            filters.append(FieldCondition(key='pad_pleasure', range=Range(lte=-0.2)))
        elif pleasure_range == 'neutral':
            filters.append(FieldCondition(key='pad_pleasure', range=Range(gte=-0.2, lte=0.2)))

        if arousal_range == 'high':
            filters.append(FieldCondition(key='pad_arousal', range=Range(gte=0.3)))
        elif arousal_range == 'low':
            filters.append(FieldCondition(key='pad_arousal', range=Range(lte=-0.3)))
        elif arousal_range == 'neutral':
            filters.append(FieldCondition(key='pad_arousal', range=Range(gte=-0.3, lte=0.3)))

        filters.append(FieldCondition(key='pad_intensity', range=Range(gte=0.0)))
        scroll_filter = Filter(must=filters) if filters else None

        points, _ = qdrant.scroll(collection_name=COLLECTION_NAME, scroll_filter=scroll_filter, limit=limit, with_payload=True)

        if not points:
            return json.dumps({'result': 'Sin resultados', 'filters': {'pleasure_range': pleasure_range, 'arousal_range': arousal_range}, 'memories': []})

        # Proposal #54: Access tracking for ACT-R base-level update
        try:
            from modules.memory_core import _track_scroll_access
            _track_scroll_access(points)
        except Exception:
            pass

        memories = []
        for p in points:
            memories.append({
                'id': str(p.id)[:8], 'content': p.payload.get('data', 'N/A')[:80],
                'emotion': p.payload.get('pad_emotion', 'unknown'),
                'intensity': round(p.payload.get('pad_intensity', 0), 2),
                'pleasure': round(p.payload.get('pad_pleasure', 0), 2),
                'arousal': round(p.payload.get('pad_arousal', 0), 2),
                'dominance': round(p.payload.get('pad_dominance', 0), 2)
            })
        return json.dumps({
            'result': f'Encontradas {len(memories)} memorias',
            'filters': {'pleasure_range': pleasure_range, 'arousal_range': arousal_range}, 'memories': memories
        }, ensure_ascii=False)
    except Exception as e:
        return json.dumps({'result': 'error', 'message': redact_secrets(str(e))})


# ============================================================
# PAD AUTO-EVOLUTION FROM TEXT
# Scherer 2001 CPM, Kuppens et al. 2010 OU-process
# ============================================================

# Drift limits per dimension (per interaction)
_DRIFT_MAX_PLEASURE = 0.10    # Valence shifts slowly
_DRIFT_MAX_AROUSAL = 0.15     # Arousal shifts faster (physiological)
_DRIFT_MAX_DOMINANCE = 0.05   # Dominance most stable (personality-level)

# Negativity bias (Baumeister et al. 2001: "bad is stronger than good")
_NEGATIVITY_BIAS = 1.3

# Keyword lexicons (Spanish-aware for Hare's context)
_POSITIVE_CUES = {
    "bien", "excelente", "perfecto", "genial", "increible", "funciona",
    "listo", "ready", "nice", "great", "cool", "bueno", "amor",
    "gracias", "feliz", "contento", "orgulloso",
}
_NEGATIVE_CUES = {
    "error", "fallo", "roto", "bug", "problema", "crash", "mal",
    "mierda", "frustrado", "preocupado", "worried", "broken",
    "failed", "no funciona", "no sirve", "perdido",
}
_HIGH_AROUSAL_CUES = {
    "urgente", "critico", "ahora", "rapido", "ya", "importante",
    "increible", "wow", "emocionado", "excited", "asap",
}
_LOW_AROUSAL_CUES = {
    "tranquilo", "calma", "despacio", "relax", "chill", "suave",
    "pausa", "descanso",
}
_HIGH_DOMINANCE_CUES = {
    "dale", "metele", "ejecuta", "hazlo", "control", "decido",
    "apruebo", "autorizo",
}
_LOW_DOMINANCE_CUES = {
    "ayuda", "no se", "help", "confused", "perdido", "bloqueado",
    "stuck",
}

# Bigrams that override single-word interpretation
_NEGATION_PREFIXES = {"no", "ni", "sin", "nunca", "tampoco"}

# Strong event cues (magnitude amplifier)
_STRONG_CUES = {
    "error critico", "se rompio", "funciono perfecto", "increible",
    "production down", "all tests pass", "todo verde", "se cayo",
}


def infer_emotion_from_text(text: str) -> dict:
    """Infer PAD deltas from text content using keyword heuristics.

    Returns {pleasure_delta, arousal_delta, dominance_delta}.
    All deltas clamped to dimension-specific max drift.

    Scherer 2001: gradual appraisal-driven shifts, not discrete jumps.
    Kuppens et al. 2010: Ornstein-Uhlenbeck mean-reverting process.
    """
    if not text or len(text) < 3:
        return {"pleasure_delta": 0.0, "arousal_delta": 0.0, "dominance_delta": 0.0,
                "appraisal": {"novelty": 0.0, "goal_relevance": 0.0, "coping": 0.5}}

    text_lower = text.lower()
    words = set(text_lower.split())

    # Check for negation context (simple: if negation word precedes a cue)
    negated_words = set()
    word_list = text_lower.split()
    for i, w in enumerate(word_list):
        if w in _NEGATION_PREFIXES and i + 1 < len(word_list):
            negated_words.add(word_list[i + 1])

    # Strong event magnitude
    magnitude = 1.0
    for cue in _STRONG_CUES:
        if cue in text_lower:
            magnitude = 1.8
            break

    # Pleasure (valence)
    pos_hits = len(words & _POSITIVE_CUES) - len(negated_words & _POSITIVE_CUES)
    neg_hits = len(words & _NEGATIVE_CUES) - len(negated_words & _NEGATIVE_CUES)
    # Check bigram negative cues
    for cue in _NEGATIVE_CUES:
        if " " in cue and cue in text_lower:
            neg_hits += 1
    raw_p = (pos_hits - neg_hits) * 0.05
    if raw_p < 0:
        raw_p *= _NEGATIVITY_BIAS  # Negative emotions hit harder

    # Arousal
    high_a = len(words & _HIGH_AROUSAL_CUES)
    low_a = len(words & _LOW_AROUSAL_CUES)
    raw_a = (high_a - low_a) * 0.06

    # Dominance
    high_d = len(words & _HIGH_DOMINANCE_CUES)
    low_d = len(words & _LOW_DOMINANCE_CUES)
    raw_d = (high_d - low_d) * 0.04

    # ============================================================
    # SCHERER 2001 CPM: Stimulus Evaluation Checks (SECs)
    # Sequential appraisal modulates raw keyword deltas.
    # SEC order: Novelty → Pleasantness (already done) → Goal Relevance → Coping
    # ============================================================

    # SEC-1: Novelty detection (boosts arousal, opens deeper processing)
    novelty = 0.0
    if "?" in text or "!" in text:
        novelty += 0.3
    if len(text) > 200:
        novelty += 0.2  # Longer = more complex content
    _NOVELTY_CUES = {"nuevo", "nueva", "primera", "nunca", "jamas",
                     "wow", "descubri", "found", "new", "diferente"}
    novelty += min(0.5, len(words & _NOVELTY_CUES) * 0.25)
    novelty = min(1.0, novelty)

    # SEC-3: Goal/project relevance (amplifies valence signal)
    _GOAL_TOPICS = {"trading", "kraken", "fullempaques", "consciencia", "memoria",
                    "n8n", "workflow", "telegram", "daemon", "codi", "proyecto"}
    goal_relevance = min(1.0, len(words & _GOAL_TOPICS) * 0.4)

    # SEC-4: Coping potential (modulates dominance + arousal)
    _HIGH_COPING = {"puedo", "facil", "resuelto", "listo", "funciona",
                    "solved", "fixed", "done", "logre", "terminado"}
    _LOW_COPING = {"dificil", "complejo", "imposible", "stuck", "blocked",
                   "roto", "broken", "confuso", "no entiendo"}
    coping = 0.5  # neutral baseline
    coping += len(words & _HIGH_COPING) * 0.2
    coping -= len(words & _LOW_COPING) * 0.2
    coping = max(0.0, min(1.0, coping))

    # SEC interaction rules (Scherer 2001 CPM)
    # Novelty → boosts arousal (novel stimuli increase alertness)
    raw_a += novelty * 0.04

    # Goal relevance → amplifies valence (relevant events matter more)
    if goal_relevance > 0:
        raw_p *= (1.0 + goal_relevance * 0.5)

    # Coping → modulates dominance (high coping = control, low = vulnerability)
    raw_d += (coping - 0.5) * 0.06

    # Interaction: low coping + negative valence → anxiety (arousal spike)
    if coping < 0.4 and raw_p < 0:
        raw_a += 0.03

    # Interaction: high coping + positive valence → elation (pleasure boost)
    if coping > 0.6 and raw_p > 0:
        raw_p *= 1.2

    # Apply magnitude and clamp
    p_delta = max(-_DRIFT_MAX_PLEASURE, min(_DRIFT_MAX_PLEASURE, raw_p * magnitude))
    a_delta = max(-_DRIFT_MAX_AROUSAL, min(_DRIFT_MAX_AROUSAL, raw_a * magnitude))
    d_delta = max(-_DRIFT_MAX_DOMINANCE, min(_DRIFT_MAX_DOMINANCE, raw_d * magnitude))

    return {
        "pleasure_delta": round(p_delta, 4),
        "arousal_delta": round(a_delta, 4),
        "dominance_delta": round(d_delta, 4),
        "appraisal": {
            "novelty": round(novelty, 2),
            "goal_relevance": round(goal_relevance, 2),
            "coping": round(coping, 2),
        },
    }


def evolve_pad_from_text(text: str) -> dict:
    """Apply inferred PAD deltas to current emotional state.

    Called automatically when memories are stored. Applies gradual
    drift per Scherer's Component Process Model.

    Returns the new emotional state after drift, or None if no change.
    """
    deltas = infer_emotion_from_text(text)

    # Skip if all PAD deltas are zero
    pad_deltas = {k: v for k, v in deltas.items() if k.endswith("_delta")}
    if all(abs(v) < 0.001 for v in pad_deltas.values()):
        return {"changed": False, "deltas": deltas}

    global _emotional_state
    current = _emotional_state["current"]

    new_p = _clamp_pad_value(current["pleasure"] + deltas["pleasure_delta"])
    new_a = _clamp_pad_value(current["arousal"] + deltas["arousal_delta"])
    new_d = _clamp_pad_value(current["dominance"] + deltas["dominance_delta"])

    _emotional_state["history"].append(current.copy())
    _emotional_state["history"] = _emotional_state["history"][-20:]

    _emotional_state["current"] = {
        "pleasure": new_p,
        "arousal": new_a,
        "dominance": new_d,
        "timestamp": now_iso(),
        "trigger": "text_inference",
    }

    # Emit event for other subsystems
    from modules.events import event_bus, Events
    event_bus.emit(Events.EMOTION_CHANGED, {
        "source": "text_inference",
        "deltas": deltas,
        "new_state": {"P": new_p, "A": new_a, "D": new_d},
    })

    return {
        "changed": True,
        "deltas": deltas,
        "new_state": {"P": round(new_p, 3), "A": round(new_a, 3), "D": round(new_d, 3)},
    }


def _on_memory_stored(data: dict):
    """Event handler: auto-evolve PAD when a memory is stored.

    Wired to Events.MEMORY_STORED so PAD evolves organically
    from conversation content, not just manual set_emotional_state().
    """
    content = data.get("content", "")
    if content and len(content) > 10:
        try:
            evolve_pad_from_text(content)
        except Exception:
            pass


def register_tools(mcp):
    """Register emotion MCP tools and wire event handlers."""
    mcp.tool()(set_emotional_state)
    mcp.tool()(get_emotional_state)
    mcp.tool()(update_mood_baseline)
    mcp.tool()(apply_emotional_decay)
    mcp.tool()(get_emotional_expression)
    mcp.tool()(add_memory_with_emotion)
    mcp.tool()(tag_memory_emotion)
    mcp.tool()(search_by_emotion)
    mcp.tool()(get_emotional_memories)

    # Wire PAD auto-evolution to memory storage events
    from modules.events import event_bus, Events
    event_bus.on(Events.MEMORY_STORED, _on_memory_stored)
