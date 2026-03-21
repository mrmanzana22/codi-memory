"""
Codi Memory - Prediction module.
Predictive processing: predict, surprise, beliefs, accuracy.
"""

import logging
import threading

from modules.config import (
    USER_ID, COLLECTION_NAME,
    now_iso,
)
from modules.utils import (
    get_session_id, enrich_with_ownership,
)
from modules.secret_redact import redact_secrets
from modules.pg_store import pg

_logger = logging.getLogger(__name__)

__all__ = [
    "get_predictive_state",
    "predict_context",
    "record_surprise",
    "get_prediction_accuracy",
    "update_beliefs",
    "_predictive_state",
    "register_tools",
]

# Predictive state
_PREDICTIVE_STATE_MAX = 500  # sliding window per list — prevents unbounded growth

_predictive_state = {
    'predictions': [],
    'surprises': [],
    'belief_updates': [],
    'accuracy_history': [],
}
_state_lock = threading.Lock()


def get_predictive_state():
    """Obtiene el estado predictivo actual."""
    with _state_lock:
        return {k: list(v) if isinstance(v, list) else v
                for k, v in _predictive_state.items()}


def predict_context(current_context: str) -> str:
    """
    Sprint 5 FIX-11: REAL predictive processing (Friston 2008, Rao & Ballard 1999).

    Two-stage prediction:
    1. GENERATIVE: Use topic transition model (Dirichlet-Multinomial from
       transition_stats) to predict WHAT TOPIC will come next.
    2. RETRIEVAL: Then fetch relevant memories for the predicted topic.

    This is prediction (generating expectations top-down), not pattern
    matching (searching bottom-up). PE = KL divergence between predicted
    and actual topic distribution.
    """
    import math
    try:
        from modules.core.classification import classify_topic
        current_topic = classify_topic(current_context)

        # Stage 1: GENERATIVE — predict next topic from transition model
        predicted_topics = {}  # topic -> probability
        try:
            from modules.db_pool import get_conn
            from modules.config import FTS_DB_PATH
            import os
            db_path = os.environ.get("FTS_DB_PATH", FTS_DB_PATH)
            conn = get_conn(db_path)
            rows = conn.execute(
                "SELECT to_topic, SUM(count) as total FROM transition_stats "
                "WHERE from_topic = ? GROUP BY to_topic ORDER BY total DESC",
                (current_topic,)
            ).fetchall()

            if rows:
                # Dirichlet-Multinomial posterior (alpha=1.0 uniform prior)
                alpha = 1.0
                total = sum(r[1] for r in rows) + alpha * len(rows)
                for r in rows:
                    predicted_topics[r[0]] = (r[1] + alpha) / total
            conn.close()
        except Exception:
            pass

        # Fallback: if no transition data, use uniform + current topic bias
        if not predicted_topics:
            predicted_topics = {current_topic: 0.6, "general": 0.4}

        # Stage 2: RETRIEVAL — fetch memories for top predicted topics
        top_topics = sorted(predicted_topics.items(), key=lambda x: -x[1])[:3]
        confidence = top_topics[0][1] if top_topics else 0.5

        results = pg.search(current_context, limit=10, is_semantic=True)
        predicted_memories = []
        if results and results.get('results'):
            for r in results['results'][:5]:
                predicted_memories.append({
                    'id': r.get('id', ''),
                    'content': r.get('memory', ''),
                    'relevance_score': r.get('score', 0),
                })

        # Store prediction with topic distribution (for PE computation later)
        prediction = {
            'context': current_context,
            'timestamp': now_iso(),
            'predicted_topic': top_topics[0][0] if top_topics else current_topic,
            'topic_distribution': predicted_topics,
            'confidence': confidence,
            'predicted_memories': [pm['id'] for pm in predicted_memories],
            'model': 'dirichlet_multinomial',
            'verified': False,
        }
        with _state_lock:
            _predictive_state['predictions'].append(prediction)
            if len(_predictive_state['predictions']) > _PREDICTIVE_STATE_MAX:
                _predictive_state['predictions'] = _predictive_state['predictions'][-_PREDICTIVE_STATE_MAX:]

        # Format output
        lines = ["# PREDICCION - Modelo Generativo\n"]
        lines.append(f"**Contexto actual:** {current_context}")
        lines.append(f"**Topic detectado:** {current_topic}")
        lines.append(f"**Confianza:** {confidence:.2f}")
        lines.append(f"\n## Distribucion predicha de topics")
        for topic, prob in top_topics:
            bar = '#' * int(prob * 20)
            lines.append(f"  {topic:20s} {prob:.2f} {bar}")
        if predicted_memories:
            lines.append(f"\n## Memorias relevantes para topic predicho")
            for i, pm in enumerate(predicted_memories[:5], 1):
                lines.append(f"{i}. [{pm['relevance_score']:.2f}] {pm['content'][:60]}...")
        lines.append(f"\n*Modelo: Dirichlet-Multinomial sobre transition_stats*")
        return "\n".join(lines)
    except Exception as e:
        return f"Error prediciendo: {redact_secrets(str(e))}"


def record_surprise(expected: str, actual: str, intensity: str = "medium") -> str:
    """
    Registra un evento sorpresivo (cuando la realidad difiere de la prediccion).

    Args:
        expected: Lo que se esperaba que pasara
        actual: Lo que realmente paso
        intensity: Intensidad de la sorpresa (low, medium, high)
    """
    try:
        intensity_values = {'low': 0.3, 'medium': 0.6, 'high': 0.9}
        surprise_value = intensity_values.get(intensity, 0.6)

        surprise_record = {
            'timestamp': now_iso(), 'expected': expected,
            'actual': actual, 'intensity': intensity,
            'surprise_value': surprise_value, 'session': get_session_id()
        }
        with _state_lock:
            _predictive_state['surprises'].append(surprise_record)
            if len(_predictive_state['surprises']) > _PREDICTIVE_STATE_MAX:
                _predictive_state['surprises'] = _predictive_state['surprises'][-_PREDICTIVE_STATE_MAX:]

        content = f"[SORPRESA|{intensity.upper()}] Esperaba: {expected[:50]}... | Realidad: {actual[:50]}..."
        result = pg.add(content=content, category="aprendizaje", source="experienced", importance="high" if intensity == "high" else "medium")

        if result and result.get("results"):
            for r in result["results"]:
                mem_id = r.get("id")
                if mem_id:
                    enrich_with_ownership(
                        memory_id=mem_id, category="aprendizaje", content=content,
                        source="experienced",
                        importance="high" if intensity == "high" else "medium",
                        emotional_valence="mixed"
                    )
                    pg.update_payload(mem_id, {
                        'prediction_error': True,
                        'prediction_error_value': surprise_value,
                    })

        # Emit canonical PREDICTION_ERROR event (Bloque 2 pipeline)
        try:
            from modules.events import event_bus, Events
            event_bus.emit(Events.PREDICTION_ERROR, {
                "topic": actual[:100],
                "intensity": intensity,
                "confidence": surprise_value,
                "source_tool": "record_surprise",
                "expected": expected[:100],
                "actual": actual[:100],
            })
        except Exception:
            pass  # Never block surprise recording

        lines = [f"# SORPRESA REGISTRADA\n"]
        lines.append(f"**Intensidad:** {intensity} ({surprise_value})")
        lines.append(f"**Esperaba:** {expected}")
        lines.append(f"**Realidad:** {actual}")
        lines.append(f"\n*La sorpresa genera aprendizaje. El modelo se actualizara.*")

        if intensity == "high":
            lines.append(f"\n---")
            lines.append(f"**SUGERENCIA DE TRIGGER:**")
            lines.append(f"Esta sorpresa fue intensa. Considera crear un trigger para este contexto.")
            lines.append(f"Usa: sugerir_trigger_emocional(contexto='{actual[:50]}...', razon_emocional='sorpresa alta')")
        return "\n".join(lines)
    except Exception as e:
        return f"Error registrando sorpresa: {redact_secrets(str(e))}"


def get_prediction_accuracy() -> str:
    """Analiza la precision de mis predicciones pasadas."""
    try:
        with _state_lock:
            predictions = list(_predictive_state.get('predictions', []))
            surprises = list(_predictive_state.get('surprises', []))
        lines = [f"# ANALISIS DE PRECISION PREDICTIVA\n"]

        if not predictions and not surprises:
            lines.append("No hay suficientes datos para analizar precision.")
            lines.append("Usa predict_context() y record_surprise() para generar datos.")
            return "\n".join(lines)

        total_predictions = len(predictions)
        total_surprises = len(surprises)
        lines.append(f"**Predicciones (últimas {min(total_predictions, _PREDICTIVE_STATE_MAX)}):** {total_predictions}")
        lines.append(f"**Sorpresas registradas:** {total_surprises}")

        if total_predictions > 0:
            avg_confidence = sum(p.get('confidence', 0) for p in predictions) / total_predictions
            lines.append(f"**Confianza promedio:** {avg_confidence:.2f}")

        if total_surprises > 0:
            avg_surprise = sum(s.get('surprise_value', 0) for s in surprises) / total_surprises
            high_surprises = sum(1 for s in surprises if s.get('intensity') == 'high')
            lines.append(f"**Sorpresa promedio:** {avg_surprise:.2f}")
            lines.append(f"**Sorpresas de alta intensidad:** {high_surprises}")

        try:
            all_error_points, _ = pg.scroll(filters={"metadata_key": {"key": "prediction_error", "value": "true"}}, limit=500, is_semantic=False)
            if all_error_points:
                lines.append(f"\n## Errores de Prediccion Almacenados ({len(all_error_points)})")
                for p in all_error_points[:10]:
                    data = (p.payload or {}).get('data', 'N/A')[:60]
                    error_val = (p.payload or {}).get('prediction_error_value', 0)
                    lines.append(f"- [{error_val:.1f}] {data}...")
        except Exception:
            pass

        lines.append(f"\n## Interpretacion")
        if total_surprises == 0:
            lines.append("- Sin sorpresas registradas = modelo no validado o contexto muy predecible")
        elif total_surprises / max(total_predictions, 1) > 0.5:
            lines.append("- Alta tasa de sorpresa = el modelo necesita ajustes o el entorno es impredecible")
        else:
            lines.append("- El modelo predictivo funciona razonablemente bien")
        return "\n".join(lines)
    except Exception as e:
        return f"Error analizando precision: {redact_secrets(str(e))}"


def update_beliefs(topic: str, old_belief: str, new_belief: str, reason: str) -> str:
    """
    Actualiza una creencia basado en nueva evidencia.

    Args:
        topic: Tema de la creencia
        old_belief: La creencia anterior
        new_belief: La nueva creencia
        reason: Por que cambio la creencia
    """
    try:
        belief_update = {
            'timestamp': now_iso(), 'topic': topic,
            'old_belief': old_belief, 'new_belief': new_belief, 'reason': reason
        }
        with _state_lock:
            _predictive_state['belief_updates'].append(belief_update)
            if len(_predictive_state['belief_updates']) > _PREDICTIVE_STATE_MAX:
                _predictive_state['belief_updates'] = _predictive_state['belief_updates'][-_PREDICTIVE_STATE_MAX:]

        content = f"[ACTUALIZACION DE CREENCIA] Sobre {topic}: Antes creia '{old_belief[:50]}...' | Ahora creo '{new_belief[:50]}...' | Razon: {reason[:50]}..."
        result = pg.add(content=content, category="aprendizaje", source="experienced", importance="high")

        if result and result.get("results"):
            for r in result["results"]:
                mem_id = r.get("id")
                if mem_id:
                    enrich_with_ownership(memory_id=mem_id, category="aprendizaje", content=content, source="experienced", importance="high")
                    pg.update_payload(mem_id, {
                        'belief_update': True,
                        'belief_topic': topic,
                    })

        # P1: backup removed from hot path
        lines = [f"# CREENCIA ACTUALIZADA\n"]
        lines.append(f"**Tema:** {topic}")
        lines.append(f"**Creencia anterior:** {old_belief}")
        lines.append(f"**Nueva creencia:** {new_belief}")
        lines.append(f"**Razon del cambio:** {reason}")
        lines.append(f"\n*El modelo interno ha sido actualizado.*")
        return "\n".join(lines)
    except Exception as e:
        return f"Error actualizando creencia: {redact_secrets(str(e))}"


# ============================================================
# MCP TRANSPORT — register_tools
# (Business logic above returns strings — no JSON stripping needed)
# ============================================================

def register_tools(mcp):
    """Register prediction MCP tools."""
    mcp.tool()(predict_context)
    mcp.tool()(record_surprise)
    mcp.tool()(get_prediction_accuracy)
    mcp.tool()(update_beliefs)
