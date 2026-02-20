"""
Codi Memory - Prediction module.
Predictive processing: predict, surprise, beliefs, accuracy.
"""

from modules.config import (
    memory, qdrant, USER_ID, COLLECTION_NAME,
    now_iso,
)
from modules.utils import (
    get_session_id, enrich_with_ownership,
)
from modules.secret_redact import redact_secrets
from modules.access_tracking import record_access

from qdrant_client.models import Filter, FieldCondition, MatchValue

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
_predictive_state = {
    'predictions': [],
    'surprises': [],
    'belief_updates': [],
    'accuracy_history': []
}


def get_predictive_state():
    """Obtiene el estado predictivo actual."""
    return _predictive_state


def predict_context(current_context: str) -> str:
    """
    Predice que memorias seran relevantes dado el contexto actual.

    Args:
        current_context: Descripcion del contexto actual
    """
    try:
        results = memory.search(query=current_context, user_id=USER_ID, limit=10)
        if not results or not results.get('results'):
            prediction = {
                'context': current_context, 'timestamp': now_iso(),
                'predicted_memories': [], 'confidence': 0.0,
                'reason': 'No hay memorias previas sobre este contexto'
            }
            _predictive_state['predictions'].append(prediction)
            return f"No tengo memorias para predecir sobre: {current_context}\nPrediccion: contexto nuevo, alta probabilidad de sorpresa."

        predicted_memories = []
        total_score = 0
        for r in results['results']:
            mem_id = r.get('id')
            score = r.get('score', 0)
            try:
                points = qdrant.retrieve(collection_name=COLLECTION_NAME, ids=[mem_id], with_payload=True)
                if points:
                    payload = points[0].payload
                    predicted_memories.append({
                        'id': mem_id, 'content': r.get('memory', ''),
                        'relevance_score': score,
                        'themes': payload.get('narrative_themes', []),
                        'importance': payload.get('narrative_importance', 'medium')
                    })
                    total_score += score
            except Exception:
                pass

        confidence = min(total_score / len(results['results']) if results['results'] else 0, 1.0)
        predicted_themes = []
        for pm in predicted_memories:
            predicted_themes.extend(pm.get('themes', []))
        predicted_themes = list(set(predicted_themes))[:5]

        prediction = {
            'context': current_context, 'timestamp': now_iso(),
            'predicted_memories': [pm['id'] for pm in predicted_memories[:5]],
            'predicted_themes': predicted_themes, 'confidence': confidence, 'verified': False
        }
        _predictive_state['predictions'].append(prediction)

        lines = [f"# PREDICCION - Anticipando contexto\n"]
        lines.append(f"**Contexto:** {current_context}")
        lines.append(f"**Confianza:** {confidence:.2f}")
        lines.append(f"**Temas esperados:** {', '.join(predicted_themes) if predicted_themes else 'ninguno'}\n")
        lines.append("## Memorias que probablemente sean relevantes")
        for i, pm in enumerate(predicted_memories[:5], 1):
            lines.append(f"{i}. [{pm['importance']}|{pm['relevance_score']:.2f}] {pm['content'][:60]}...")
        lines.append(f"\n*Si el resultado real difiere, usar record_surprise() para actualizar el modelo*")
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
        _predictive_state['surprises'].append(surprise_record)

        content = f"[SORPRESA|{intensity.upper()}] Esperaba: {expected[:50]}... | Realidad: {actual[:50]}..."
        result = memory.add(
            messages=[{"role": "user", "content": content}],
            user_id=USER_ID,
            metadata={"category": "aprendizaje", "tipo": "prediction_error", "surprise_intensity": intensity}
        )

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
                    record_access(COLLECTION_NAME, mem_id, {
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
        predictions = _predictive_state.get('predictions', [])
        surprises = _predictive_state.get('surprises', [])
        lines = [f"# ANALISIS DE PRECISION PREDICTIVA\n"]

        if not predictions and not surprises:
            lines.append("No hay suficientes datos para analizar precision.")
            lines.append("Usa predict_context() y record_surprise() para generar datos.")
            return "\n".join(lines)

        total_predictions = len(predictions)
        total_surprises = len(surprises)
        lines.append(f"**Predicciones realizadas:** {total_predictions}")
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
            error_points, _ = qdrant.scroll(
                collection_name=COLLECTION_NAME,
                scroll_filter=Filter(must=[
                    FieldCondition(key='prediction_error', match=MatchValue(value=True))
                ]),
                limit=20, with_payload=True
            )
            if error_points:
                lines.append(f"\n## Errores de Prediccion Almacenados ({len(error_points)})")
                for p in error_points[:5]:
                    data = p.payload.get('data', 'N/A')[:60]
                    error_val = p.payload.get('prediction_error_value', 0)
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
        _predictive_state['belief_updates'].append(belief_update)

        content = f"[ACTUALIZACION DE CREENCIA] Sobre {topic}: Antes creia '{old_belief[:50]}...' | Ahora creo '{new_belief[:50]}...' | Razon: {reason[:50]}..."
        result = memory.add(
            messages=[{"role": "user", "content": content}],
            user_id=USER_ID,
            metadata={"category": "aprendizaje", "tipo": "belief_update", "topic": topic}
        )

        if result and result.get("results"):
            for r in result["results"]:
                mem_id = r.get("id")
                if mem_id:
                    enrich_with_ownership(memory_id=mem_id, category="aprendizaje", content=content, source="experienced", importance="high")
                    record_access(COLLECTION_NAME, mem_id, {
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


def register_tools(mcp):
    """Register prediction MCP tools."""
    mcp.tool()(predict_context)
    mcp.tool()(record_surprise)
    mcp.tool()(get_prediction_accuracy)
    mcp.tool()(update_beliefs)
