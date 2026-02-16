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
        return json.dumps({'result': 'error', 'message': str(e)})


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
        return json.dumps({'result': 'error', 'message': str(e)})


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
        return json.dumps({'result': 'error', 'message': str(e)})


def apply_emotional_decay() -> str:
    """Aplica decay al estado emocional actual, acercandolo al mood baseline."""
    try:
        global _emotional_state
        current = _emotional_state['current']
        mood = _emotional_state['mood']
        decay_rate = _emotional_state['decay_rate']

        if not current['timestamp']:
            return json.dumps({'result': 'Sin estado emocional para decaer', 'applied': False})

        new_p = current['pleasure'] + (mood['pleasure'] - current['pleasure']) * decay_rate
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
        return json.dumps({'result': 'error', 'message': str(e)})


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
        return json.dumps({'result': 'error', 'message': str(e)})


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
                    qdrant.set_payload(collection_name=COLLECTION_NAME, payload=ownership_metadata, points=[mem_id])

        # P1: backup removed from hot path
        result_json = {
            'result': 'Memoria guardada con emocion',
            'memory_id': result.get('results', [{}])[0].get('id', 'unknown')[:8] if result else 'unknown',
            'emotion': {'label': emotion_label, 'description': _get_emotion_text(emotion_label), 'pleasure': p, 'arousal': a, 'dominance': d, 'intensity': round(intensity, 2)}
        }
        return json.dumps(result_json, ensure_ascii=False)
    except Exception as e:
        return json.dumps({'result': 'error', 'message': str(e)})


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

        qdrant.set_payload(
            collection_name=COLLECTION_NAME,
            payload={
                'pad_pleasure': p, 'pad_arousal': a, 'pad_dominance': d,
                'pad_emotion': emotion_label, 'pad_intensity': intensity,
                'experiential_emotional_weight': min(intensity / 1.73, 1.0),
                'experiential_emotional_valence': 'positive' if p > 0.2 else 'negative' if p < -0.2 else 'neutral'
            },
            points=[full_id]
        )
        result = {
            'result': 'Memoria etiquetada con emocion', 'memory_id': memory_id,
            'emotion': {'label': emotion_label, 'description': _get_emotion_text(emotion_label), 'pleasure': p, 'arousal': a, 'dominance': d, 'intensity': round(intensity, 2)}
        }
        return json.dumps(result, ensure_ascii=False)
    except Exception as e:
        return json.dumps({'result': 'error', 'message': str(e)})


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
        return json.dumps({'result': 'error', 'message': str(e)})


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
        return json.dumps({'result': 'error', 'message': str(e)})


def register_tools(mcp):
    """Register emotion MCP tools."""
    mcp.tool()(set_emotional_state)
    mcp.tool()(get_emotional_state)
    mcp.tool()(update_mood_baseline)
    mcp.tool()(apply_emotional_decay)
    mcp.tool()(get_emotional_expression)
    mcp.tool()(add_memory_with_emotion)
    mcp.tool()(tag_memory_emotion)
    mcp.tool()(search_by_emotion)
    mcp.tool()(get_emotional_memories)
