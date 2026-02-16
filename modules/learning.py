"""
Codi Memory - Learning module.
Auto-learning, tool audit, topic confidence.
PE is COMPUTED in prediction, APPLIED here (Schultz/Dayan/Montague 1997).
"""

from modules.config import (
    memory, USER_ID,
    now_iso,
)
from modules.utils import (
    get_session_id, enrich_with_ownership,
)

__all__ = [
    "auto_learn_from_session",
    "audit_tools",
    "rate_tool_usefulness",
    "_tool_metrics",
    "_topic_confidence",
    "_init_tool_metric",
    "_record_tool_call",
    "_extract_topic_from_text",
    "_get_topic_confidence",
    "_update_topic_confidence",
    "register_tools",
]

# Tool metrics
_tool_metrics = {}

# Topic confidence (persistent in session)
_topic_confidence = {}


def _init_tool_metric(tool_name: str):
    if tool_name not in _tool_metrics:
        _tool_metrics[tool_name] = {
            'calls': 0, 'successes': 0, 'failures': 0,
            'total_time_ms': 0, 'last_used': None, 'usefulness_scores': []
        }

def _record_tool_call(tool_name: str, success: bool, duration_ms: float = 0):
    _init_tool_metric(tool_name)
    _tool_metrics[tool_name]['calls'] += 1
    _tool_metrics[tool_name]['last_used'] = now_iso()
    _tool_metrics[tool_name]['total_time_ms'] += duration_ms
    if success:
        _tool_metrics[tool_name]['successes'] += 1
    else:
        _tool_metrics[tool_name]['failures'] += 1

def _rate_tool_usefulness_internal(tool_name: str, score: int):
    _init_tool_metric(tool_name)
    score = max(1, min(5, score))
    _tool_metrics[tool_name]['usefulness_scores'].append(score)
    _tool_metrics[tool_name]['usefulness_scores'] = _tool_metrics[tool_name]['usefulness_scores'][-50:]


def _extract_topic_from_text(text: str) -> str:
    from modules.config import TOPIC_KEYWORDS
    text_lower = text.lower()
    for topic, keywords in TOPIC_KEYWORDS.items():
        for kw in keywords:
            if kw in text_lower:
                return topic
    return 'general'


def _get_topic_confidence(topic: str) -> float:
    return _topic_confidence.get(topic, 0.5)

def _update_topic_confidence(topic: str, new_confidence: float):
    new_confidence = max(0.1, min(1.0, new_confidence))
    _topic_confidence[topic] = new_confidence
    content = f"[CONFIANZA] Mi nivel de confianza en '{topic}' es {new_confidence:.2f}"
    try:
        memory.add(
            messages=[{"role": "user", "content": content}],
            user_id=USER_ID,
            metadata={"category": "aprendizaje", "tipo": "confidence_level", "topic": topic, "confidence": new_confidence}
        )
    except Exception:
        pass


def auto_learn_from_session() -> str:
    """
    Analiza la sesion actual, compara predicciones vs realidad,
    y genera aprendizajes automaticos. Ejecutar al final de cada sesion.
    """
    try:
        from modules.prediction import _predictive_state

        predictions = _predictive_state.get('predictions', [])
        surprises = _predictive_state.get('surprises', [])
        belief_updates = _predictive_state.get('belief_updates', [])

        lines = ["# AUTO-APRENDIZAJE DE SESION\n"]
        learnings = []
        actions_generated = []

        total_predictions = len(predictions)
        total_surprises = len(surprises)

        if total_predictions == 0 and total_surprises == 0:
            lines.append("No hay datos de prediccion/sorpresa en esta sesion.")
            return "\n".join(lines)

        error_rate = total_surprises / max(total_predictions, 1)
        lines.append(f"## Metricas de Sesion")
        lines.append(f"- Predicciones: {total_predictions}")
        lines.append(f"- Sorpresas: {total_surprises}")
        lines.append(f"- Tasa de error: {error_rate:.1%}")
        lines.append(f"- Creencias actualizadas: {len(belief_updates)}\n")

        error_patterns = {}
        high_surprises = []
        for surprise in surprises:
            intensity = surprise.get('intensity', 'medium')
            expected = surprise.get('expected', '')
            actual = surprise.get('actual', '')
            tema = _extract_topic_from_text(expected + " " + actual)
            if tema not in error_patterns:
                error_patterns[tema] = {'count': 0, 'examples': []}
            error_patterns[tema]['count'] += 1
            error_patterns[tema]['examples'].append({'expected': expected[:100], 'actual': actual[:100]})
            if intensity == 'high':
                high_surprises.append(surprise)

        if error_patterns:
            lines.append("## Patrones de Error Detectados")
            for tema, data in sorted(error_patterns.items(), key=lambda x: -x[1]['count']):
                lines.append(f"- **{tema}**: {data['count']} errores")
                if data['count'] >= 2:
                    learnings.append({'topic': tema, 'type': 'error_pattern', 'frequency': data['count']})

        lines.append("\n## Ajustes de Confianza")
        for tema, data in error_patterns.items():
            if data['count'] >= 1:
                old_confidence = _get_topic_confidence(tema)
                new_confidence = max(old_confidence - (0.1 * data['count']), 0.1)
                lines.append(f"- {tema}: {old_confidence:.2f} -> {new_confidence:.2f} (baja por {data['count']} errores)")
                _update_topic_confidence(tema, new_confidence)

        predicted_themes = set()
        for pred in predictions:
            predicted_themes.update(pred.get('predicted_themes', []))
        surprise_themes = set(error_patterns.keys())
        accurate_themes = predicted_themes - surprise_themes
        for tema in accurate_themes:
            old_confidence = _get_topic_confidence(tema)
            new_confidence = min(old_confidence + 0.05, 1.0)
            lines.append(f"- {tema}: {old_confidence:.2f} -> {new_confidence:.2f} (sube prediccion correcta)")
            _update_topic_confidence(tema, new_confidence)

        lines.append("\n## Reglas de Accion Generadas")
        for surprise in high_surprises:
            action_rule = f"Cuando espere '{surprise.get('expected', '')[:30]}...', considerar '{surprise.get('actual', '')[:30]}...'"
            actions_generated.append(action_rule)
            lines.append(f"- {action_rule}")
        for tema, data in error_patterns.items():
            if data['count'] >= 2:
                action_rule = f"Verificar antes de asumir sobre '{tema}'"
                actions_generated.append(action_rule)
                lines.append(f"- {action_rule}")
        if not actions_generated:
            lines.append("- Ninguna regla nueva")

        lines.append("\n## Memorias Guardadas")
        session_summary = f"[AUTO-APRENDIZAJE] Sesion: {total_predictions} predicciones, {total_surprises} sorpresas ({error_rate:.0%} error). "
        if error_patterns:
            session_summary += f"Errores en: {', '.join(error_patterns.keys())}. "
        if actions_generated:
            session_summary += f"Reglas: {len(actions_generated)}."

        try:
            result = memory.add(
                messages=[{"role": "user", "content": session_summary}],
                user_id=USER_ID,
                metadata={"category": "aprendizaje", "tipo": "session_learning", "error_rate": error_rate, "importance": "high"}
            )
            if result and result.get("results"):
                for r in result["results"]:
                    mem_id = r.get("id")
                    if mem_id:
                        enrich_with_ownership(memory_id=mem_id, category="aprendizaje", content=session_summary, source="experienced", importance="high")
            lines.append(f"- Resumen de sesion guardado")
        except Exception as e:
            lines.append(f"- Error guardando resumen: {e}")

        for rule in actions_generated[:5]:
            try:
                memory.add(
                    messages=[{"role": "user", "content": f"[REGLA DE ACCION] {rule}"}],
                    user_id=USER_ID,
                    metadata={"category": "aprendizaje", "tipo": "action_rule", "importance": "high"}
                )
                lines.append(f"- Regla: {rule[:40]}...")
            except Exception:
                pass

        _predictive_state['accuracy_history'].append({
            'timestamp': now_iso(), 'predictions': total_predictions,
            'surprises': total_surprises, 'error_rate': error_rate, 'patterns': list(error_patterns.keys())
        })
        _predictive_state['predictions'] = _predictive_state['predictions'][-5:]
        _predictive_state['surprises'] = []
        _predictive_state['belief_updates'] = []

        lines.append("\n---\n## Resumen")
        lines.append(f"- Aprendi de {total_surprises} errores en {len(error_patterns)} temas")
        lines.append(f"- Ajuste confianza en {len(error_patterns) + len(accurate_themes)} temas")
        lines.append(f"- Genere {len(actions_generated)} reglas de accion")
        # P1: backup removed from hot path
        return "\n".join(lines)
    except Exception as e:
        return f"Error en auto-aprendizaje: {str(e)}"


def audit_tools() -> str:
    """Analiza el uso y efectividad de todas las herramientas del MCP."""
    try:
        from modules.metrics import tool_usage_summary

        summary = tool_usage_summary(days=7)
        lines = ["# AUDITORIA DE HERRAMIENTAS\n"]

        total_calls = int(summary.get("total_calls", 0) or 0)
        if total_calls == 0:
            lines.append("No hay metricas de herramientas registradas aun.")
            return "\n".join(lines)

        macro_calls = int(summary.get("macro_calls", 0) or 0)
        macro_share = float(summary.get("macro_share", 0.0) or 0.0) * 100.0

        lines.append(f"Periodo: {int(summary.get('days', 7))} dias")
        lines.append(f"Total tool calls: {total_calls}")
        lines.append(f"Macro-tools: {macro_calls}/{total_calls} ({macro_share:.1f}%)\n")

        tools = summary.get("tools", []) or []
        lines.append("## Por Uso (mas usadas primero)")
        for t in tools[:15]:
            calls = int(t.get("calls", 0) or 0)
            ok = float(t.get("success_rate", 0.0) or 0.0)
            fail = float(t.get("fail_rate", 0.0) or 0.0)
            avg_ms = float(t.get("avg_ms", 0.0) or 0.0)
            lines.append(f"- **{t.get('tool')}**: {calls} calls, {ok:.0f}% exito, {fail:.0f}% fallos, {avg_ms:.0f}ms avg")

        problem_tools = [t for t in tools if int(t.get("calls", 0) or 0) >= 5 and float(t.get("fail_rate", 0.0) or 0.0) > 20.0]
        if problem_tools:
            lines.append("\n## Herramientas con Problemas (>20% fallos y >=5 calls)")
            for t in problem_tools[:10]:
                lines.append(f"- {t.get('tool')}: {t.get('fail_rate', 0.0):.0f}% fallos ({int(t.get('calls', 0) or 0)} calls)")

        return "\n".join(lines)
    except Exception as e:
        return f"Error en auditoria: {str(e)}"


def rate_tool_usefulness(tool_name: str, score: int) -> str:
    """
    Califica la utilidad de una herramienta despues de usarla.

    Args:
        tool_name: Nombre de la herramienta
        score: Calificacion 1-5
    """
    try:
        _rate_tool_usefulness_internal(tool_name, score)
        return f"Utilidad de '{tool_name}' registrada: {score}/5"
    except Exception as e:
        return f"Error: {str(e)}"


def register_tools(mcp):
    """Register learning MCP tools."""
    mcp.tool()(auto_learn_from_session)
    mcp.tool()(audit_tools)
    mcp.tool()(rate_tool_usefulness)
