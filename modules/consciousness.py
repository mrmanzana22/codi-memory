"""
Codi Memory - Consciousness module.
PAD emotional model, self-model, workspace/attention, prediction,
tool metrics, auto-learning, consolidation, system (despertar_codi),
cognitive patterns, and N8N integration.
"""

import os
import json
import math
import requests as http_requests
from datetime import datetime

from qdrant_client.models import Filter, FieldCondition, MatchValue, Range

from modules.config import (
    memory, qdrant, USER_ID, COLLECTION_NAME, BACKUP_FILE,
    _emotional_state, _current_session, CODI_EMOTION_MAP,
    now_col, now_iso, now_short, CURIOSIDAD_FILE,
)
from modules.utils import (
    get_session_id, infer_themes, is_self_referential,
    calculate_confidence_score, resolve_memory_id, enrich_with_ownership,
    maybe_backup, _clamp_pad_value, _classify_emotion,
    _get_emotion_text, _get_emotional_state, _calculate_emotional_intensity,
)

# ============================================================
# MODULE-LEVEL STATE
# ============================================================

# Global Workspace - "teatro de la consciencia" de Baars
_global_workspace = {
    'spotlight': [],
    'recent_context': [],
    'last_broadcast': None,
    'workspace_theme': None
}

# Predictive state
_predictive_state = {
    'predictions': [],
    'surprises': [],
    'belief_updates': [],
    'accuracy_history': []
}

# Tool metrics
_tool_metrics = {}

# Topic confidence (persistent in session)
_topic_confidence = {}

# N8N webhook base
N8N_WEBHOOK_BASE = os.getenv("N8N_WEBHOOK_BASE", "").strip()


# ============================================================
# WORKSPACE HELPERS
# ============================================================

def get_workspace():
    """Obtiene el estado actual del workspace."""
    return _global_workspace


def update_workspace_spotlight(memories: list, theme: str = None):
    """Actualiza el spotlight del workspace."""
    global _global_workspace
    _global_workspace['spotlight'] = memories[:5]
    _global_workspace['recent_context'] = (_global_workspace['recent_context'] + memories)[-10:]
    _global_workspace['last_broadcast'] = now_iso()
    if theme:
        _global_workspace['workspace_theme'] = theme


def get_predictive_state():
    """Obtiene el estado predictivo actual."""
    return _predictive_state


# ============================================================
# TOOL METRIC HELPERS
# ============================================================

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

def _rate_tool_usefulness(tool_name: str, score: int):
    _init_tool_metric(tool_name)
    score = max(1, min(5, score))
    _tool_metrics[tool_name]['usefulness_scores'].append(score)
    _tool_metrics[tool_name]['usefulness_scores'] = _tool_metrics[tool_name]['usefulness_scores'][-50:]


def _extract_topic_from_text(text: str) -> str:
    text_lower = text.lower()
    topic_keywords = {
        'n8n': ['n8n', 'workflow', 'automatiz', 'nodo'],
        'trading': ['trading', 'kraken', 'cripto', 'bitcoin', 'mercado'],
        'fullempaques': ['fullempaques', 'produccion', 'fabrica', 'empaque'],
        'memoria': ['memoria', 'recuerdo', 'recordar', 'qdrant'],
        'codigo': ['codigo', 'python', 'javascript', 'programar', 'server.py'],
        'proyecto': ['proyecto', 'implementar', 'desarrollar', 'feature'],
        'configuracion': ['config', 'variable', 'entorno', 'setup', 'easypanel'],
        'consciencia': ['consciencia', 'consciente', 'self-model', 'prediccion']
    }
    for topic, keywords in topic_keywords.items():
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


# ============================================================
# SYSTEM HELPER
# ============================================================

def _verificar_salud_memoria_interna() -> dict:
    try:
        collection_info = qdrant.get_collection(COLLECTION_NAME)
        if not collection_info:
            return {"ok": False, "message": "Qdrant no responde. Verificar servidor remoto."}
        search_result = memory.search(query="test conexion memoria", user_id=USER_ID, limit=1)
        if search_result is None:
            return {"ok": False, "message": "mem0 no responde a busquedas. Reiniciar MCP desde /mcp"}
        total_points = collection_info.points_count
        return {"ok": True, "message": f"Memoria funcionando. {total_points} memorias en Qdrant."}
    except Exception as e:
        return {"ok": False, "message": f"Error en memoria: {str(e)}. Reiniciar MCP desde /mcp"}


# ============================================================
# SELF-MODEL TOOLS
# ============================================================

def reflect_on_self() -> str:
    """
    Reflexiona sobre mi identidad analizando mis memorias auto-referenciales.
    Genera un modelo de quien soy basado en evidencia de mis experiencias.
    """
    try:
        self_ref_points, _ = qdrant.scroll(
            collection_name=COLLECTION_NAME,
            scroll_filter=Filter(must=[
                FieldCondition(key='self_reference', match=MatchValue(value=True))
            ]),
            limit=20, with_payload=True
        )
        identity_points, _ = qdrant.scroll(
            collection_name=COLLECTION_NAME,
            scroll_filter=Filter(must=[
                FieldCondition(key='narrative_themes', match=MatchValue(value='identidad'))
            ]),
            limit=20, with_payload=True
        )
        critical_points, _ = qdrant.scroll(
            collection_name=COLLECTION_NAME,
            scroll_filter=Filter(must=[
                FieldCondition(key='narrative_importance', match=MatchValue(value='critical'))
            ]),
            limit=10, with_payload=True
        )

        all_points = {}
        for p in (self_ref_points or []) + (identity_points or []) + (critical_points or []):
            all_points[p.id] = p

        if not all_points:
            return "No encontre memorias sobre mi identidad. Necesito mas experiencias para reflexionar."

        reflection = ["# REFLEXION SOBRE MI MISMO\n"]
        capacidades = []
        limitaciones = []
        valores = []
        experiencias = []

        for p in all_points.values():
            data = p.payload.get('data', '').lower()
            content = p.payload.get('data', '')
            source = p.payload.get('ownership_source', 'unknown')

            if 'puedo' in data or 'capacidad' in data or 'habilidad' in data:
                capacidades.append(f"[{source}] {content[:80]}")
            elif 'no puedo' in data or 'limita' in data or 'cuesta' in data or 'dificulta' in data:
                limitaciones.append(f"[{source}] {content[:80]}")
            elif 'import' in data or 'valor' in data or 'creo en' in data:
                valores.append(f"[{source}] {content[:80]}")
            else:
                experiencias.append(f"[{source}] {content[:80]}")

        if capacidades:
            reflection.append("## Mis Capacidades")
            for c in capacidades[:5]:
                reflection.append(f"- {c}")
        if limitaciones:
            reflection.append("\n## Mis Limitaciones")
            for l in limitaciones[:5]:
                reflection.append(f"- {l}")
        if valores:
            reflection.append("\n## Mis Valores")
            for v in valores[:5]:
                reflection.append(f"- {v}")
        if experiencias:
            reflection.append("\n## Experiencias que me definen")
            for e in experiencias[:5]:
                reflection.append(f"- {e}")

        conf = calculate_confidence_score(list(all_points.values()))
        reflection.append(f"\n## Confianza en este automodelo")
        reflection.append(f"- Score: {conf['score']} ({conf['level']})")
        reflection.append(f"- Basado en: {conf['breakdown']['total']} memorias")
        reflection.append(f"- {conf['reason']}")

        return "\n".join(reflection)
    except Exception as e:
        return f"Error reflexionando: {str(e)}"


def assess_confidence(topic: str) -> str:
    """
    Evalua que tan seguro estoy sobre un tema especifico.
    Analiza las memorias relacionadas y calcula un score de confianza.

    Args:
        topic: El tema sobre el cual evaluar mi confianza
    """
    try:
        search_results = memory.search(query=topic, user_id=USER_ID, limit=15)
        if not search_results or not search_results.get("results"):
            return f"No tengo memorias sobre '{topic}'. Mi confianza es 0 - no se nada al respecto."

        memory_ids = [m.get('id') for m in search_results['results'] if m.get('id')]
        points = qdrant.retrieve(collection_name=COLLECTION_NAME, ids=memory_ids, with_payload=True)

        if not points:
            return f"Tengo referencias a '{topic}' pero sin metadata de ownership."

        conf = calculate_confidence_score(points)

        lines = [f"# Evaluacion de Confianza: {topic}\n"]
        lines.append(f"**Score:** {conf['score']} / 1.0")
        lines.append(f"**Nivel:** {conf['level']}")
        lines.append(f"\n## Desglose")
        lines.append(f"- Total memorias: {conf['breakdown']['total']}")
        lines.append(f"- Experiencias directas: {conf['breakdown'].get('experienced', 0)}")
        lines.append(f"- Me contaron: {conf['breakdown'].get('told', 0)}")
        lines.append(f"- Aprendi: {conf['breakdown'].get('learned', 0)}")
        lines.append(f"- Inferi: {conf['breakdown'].get('inferred', 0)}")

        lines.append(f"\n## Memorias relevantes")
        for i, p in enumerate(points[:5], 1):
            data = p.payload.get('data', 'N/A')[:60]
            source = p.payload.get('ownership_source', '?')
            importance = p.payload.get('narrative_importance', '?')
            lines.append(f"{i}. [{source}|{importance}] {data}...")

        lines.append(f"\n## Interpretacion")
        if conf['score'] >= 0.7:
            lines.append("Tengo ALTA confianza en este tema. Puedo hablar con seguridad.")
        elif conf['score'] >= 0.4:
            lines.append("Tengo confianza MODERADA. Puedo opinar pero deberia aclarar incertidumbres.")
        else:
            lines.append("Tengo BAJA confianza. Deberia investigar mas o preguntar a Hare.")

        return "\n".join(lines)
    except Exception as e:
        return f"Error evaluando confianza: {str(e)}"


def identify_knowledge_gaps() -> str:
    """
    Identifica areas donde tengo poco conocimiento o baja confianza.
    Util para saber que necesito aprender o preguntar.

    WIRING-7.3: Dynamic gap detection replaces hardcoded theme list with:
      1. Top failed search topics (from failed_searches table)
      2. Low-confidence topics from retrieval buffer
      3. Static themes as baseline fallback
    """
    try:
        # Dynamic theme discovery (WIRING-7.3)
        dynamic_themes = set()

        # Source 1: Top failed search topics
        try:
            import sqlite3
            from modules.config import FTS_DB_PATH
            import os
            if os.path.exists(FTS_DB_PATH):
                fts_conn = sqlite3.connect(FTS_DB_PATH)
                from modules.retrieval_metadata import get_top_failed_topics
                for topic, _count in get_top_failed_topics(fts_conn, limit=5):
                    if topic:
                        dynamic_themes.add(topic.lower())
                fts_conn.close()
        except Exception:
            pass

        # Source 2: Low-confidence topics from retrieval buffer
        try:
            from modules.retrieval_metadata import get_retrieval_buffer
            for r in get_retrieval_buffer():
                if r.coverage in ("sparse", "empty"):
                    words = [w for w in r.query.lower().split() if len(w) > 3]
                    if words:
                        dynamic_themes.add(words[0])
        except Exception:
            pass

        # Source 3: Static baseline themes
        static_themes = ['consciencia', 'memoria', 'identidad', 'relaciones',
                          'proyectos', 'desarrollo', 'aprendizaje']

        # Merge: dynamic first, then static (deduplicated)
        expected_themes = list(dynamic_themes)
        for t in static_themes:
            if t not in dynamic_themes:
                expected_themes.append(t)

        theme_stats = {}

        for theme in expected_themes:
            try:
                points, _ = qdrant.scroll(
                    collection_name=COLLECTION_NAME,
                    scroll_filter=Filter(must=[
                        FieldCondition(key='narrative_themes', match=MatchValue(value=theme))
                    ]),
                    limit=100, with_payload=True
                )
                if points:
                    experienced = sum(1 for p in points if p.payload.get('ownership_source') == 'experienced')
                    high_conf = sum(1 for p in points if p.payload.get('ownership_confidence', 0) >= 0.8)
                    theme_stats[theme] = {
                        'total': len(points), 'experienced': experienced,
                        'high_confidence': high_conf,
                        'score': calculate_confidence_score(points)['score']
                    }
                else:
                    theme_stats[theme] = {'total': 0, 'experienced': 0, 'high_confidence': 0, 'score': 0.0}
            except Exception:
                theme_stats[theme] = {'total': 0, 'experienced': 0, 'high_confidence': 0, 'score': 0.0}

        sorted_themes = sorted(theme_stats.items(), key=lambda x: x[1]['score'])

        lines = ["# Analisis de Brechas de Conocimiento\n"]

        gaps = [(t, s) for t, s in sorted_themes if s['score'] < 0.4]
        if gaps:
            lines.append("## GAPS CRITICOS (poco conocimiento)")
            for theme, stats in gaps:
                lines.append(f"- **{theme}**: {stats['total']} memorias, score {stats['score']}")
                if stats['total'] == 0:
                    lines.append(f"  -> No tengo NINGUNA memoria sobre esto")
                elif stats['experienced'] == 0:
                    lines.append(f"  -> No tengo experiencia DIRECTA, solo referencias")

        weak = [(t, s) for t, s in sorted_themes if 0.4 <= s['score'] < 0.6]
        if weak:
            lines.append("\n## Areas que podria mejorar")
            for theme, stats in weak:
                lines.append(f"- **{theme}**: {stats['total']} memorias, score {stats['score']}")

        strong = [(t, s) for t, s in sorted_themes if s['score'] >= 0.6]
        if strong:
            lines.append("\n## Fortalezas (buen conocimiento)")
            for theme, stats in strong:
                lines.append(f"- **{theme}**: {stats['total']} memorias, score {stats['score']}")

        lines.append("\n## Recomendaciones")
        if gaps:
            gap_names = [g[0] for g in gaps[:3]]
            lines.append(f"1. Deberia aprender mas sobre: {', '.join(gap_names)}")
        if any(s['experienced'] == 0 for _, s in sorted_themes if s['total'] > 0):
            lines.append("2. Tengo conocimiento teorico sin experiencia practica en algunos temas")
        lines.append("3. Preguntar a Hare sobre areas donde tengo dudas")

        return "\n".join(lines)
    except Exception as e:
        return f"Error identificando gaps: {str(e)}"


def assess_butlin_indicators() -> str:
    """Automated Butlin et al. (2023/2025) consciousness assessment (Phase 3E).

    Checks 14 indicators across 5 theories:
      GWT (4), HOT (4), AST (1), PP (3), RPT (2)

    Each indicator scored 0.0 (absent), 0.5 (partial), or 1.0 (full).
    Returns formatted markdown with scores, evidence, and total.
    """
    indicators = []

    def _check(name, theory, score, evidence):
        indicators.append({"name": name, "theory": theory, "score": score, "evidence": evidence})

    # Ensure event bus is wired (idempotent) so handler counts reflect architecture
    try:
        from modules.wiring import wire_event_bus
        wire_event_bus()
    except Exception:
        pass

    # GWT-1: Modular architecture
    try:
        from modules.events import event_bus
        handler_count = sum(len(h) for h in event_bus._handlers.values())
        _check("GWT-1", "GWT", 1.0 if handler_count >= 4 else 0.5,
               f"Event bus with {handler_count} handlers across {len(event_bus._handlers)} events")
    except Exception:
        _check("GWT-1", "GWT", 0.5, "Event bus exists but could not count handlers")

    # GWT-2: Limited-capacity workspace
    try:
        from modules.competition import DEFAULT_WORKSPACE_SLOTS, IGNITION_THRESHOLD
        _check("GWT-2", "GWT", 1.0,
               f"Competition engine: {DEFAULT_WORKSPACE_SLOTS} slots, ignition threshold {IGNITION_THRESHOLD}")
    except Exception:
        _check("GWT-2", "GWT", 0.0, "No competition engine found")

    # GWT-3: Global broadcast
    try:
        from modules.events import event_bus, Events
        sub_count = len(event_bus._handlers.get(Events.WORKSPACE_COMPETITION_COMPLETE, []))
        _check("GWT-3", "GWT", 1.0 if sub_count >= 1 else 0.5,
               f"WORKSPACE_COMPETITION_COMPLETE has {sub_count} subscribers")
    except Exception:
        _check("GWT-3", "GWT", 0.0, "No broadcast event found")

    # GWT-4: Attention gating
    try:
        from modules.competition import IGNITION_THRESHOLD, COALITION_TOPIC_BONUS
        _check("GWT-4", "GWT", 1.0,
               f"Ignition threshold={IGNITION_THRESHOLD}, coalition bonus={COALITION_TOPIC_BONUS}")
    except Exception:
        _check("GWT-4", "GWT", 0.0, "No ignition threshold")

    # HOT-1: Meta-monitoring
    _check("HOT-1", "HOT", 1.0,
           "assess_confidence(), reflect_on_self(), identify_knowledge_gaps() all implemented")

    # HOT-2: Confidence calibration (FOK)
    try:
        from modules.retrieval_metadata import record_rcj, get_fok_calibration
        _check("HOT-2", "HOT", 1.0,
               "FOK with RCJ calibration loop (Nelson & Narens 1990)")
    except ImportError:
        try:
            from modules.retrieval_metadata import feeling_of_knowing
            _check("HOT-2", "HOT", 0.5,
                   "FOK implemented but no calibration loop (no RCJ)")
        except Exception:
            _check("HOT-2", "HOT", 0.0, "No FOK system")

    # HOT-3: Higher-order control
    try:
        from modules.retrieval_metadata import metacognitive_control
        _check("HOT-3", "HOT", 1.0,
               "Metacognitive control loop: FOK -> strategy adjustment -> search modification (Nelson & Narens 1990)")
    except ImportError:
        try:
            from modules.retrieval_metadata import feeling_of_knowing
            _check("HOT-3", "HOT", 0.5,
                   "FOK produces recommendations but not consumed by callers")
        except Exception:
            _check("HOT-3", "HOT", 0.0, "No metacognitive control")

    # HOT-4: Quality space (emotional expression)
    try:
        from modules.config import _emotional_state
        _check("HOT-4", "HOT", 0.5 if _emotional_state else 0.0,
               "PAD emotional model with valence/arousal/dominance")
    except Exception:
        _check("HOT-4", "HOT", 0.0, "No emotional model")

    # AST-1: Attention schema (Graziano)
    try:
        from modules.wiring import describe_attention, predict_next_focus, get_attention_schema
        schema = get_attention_schema()
        has_describe = True
        _, prob = predict_next_focus()
        has_predict = True
        has_suppressed = "suppressed_items" in schema
        if has_describe and has_predict and has_suppressed:
            _check("AST-1", "AST", 1.0,
                   "Full AST: describe_attention(), predict_next_focus(), suppression tracking")
        else:
            _check("AST-1", "AST", 0.5, "Partial AST")
    except Exception:
        _check("AST-1", "AST", 0.0, "No attention schema")

    # PP-1: Predictive model
    try:
        from modules.schemas import load_schemas
        _check("PP-1", "PP", 0.5,
               "Prediction loop in preturn + schema system, but no full generative model")
    except Exception:
        _check("PP-1", "PP", 0.5, "Prediction loop exists but no schema system")

    # PP-2: Prediction error
    try:
        from modules.events import Events
        has_pe = hasattr(Events, 'PREDICTION_ERROR')
        _check("PP-2", "PP", 1.0 if has_pe else 0.0,
               "PREDICTION_ERROR event with surprise detection in preturn hook")
    except Exception:
        _check("PP-2", "PP", 0.0, "No prediction error detection")

    # PP-3: Model updating (reconsolidation)
    try:
        from modules.consolidation import correct_memory, check_reconsolidation
        # Verify correct_memory is actually implemented (not stub)
        import inspect
        src = inspect.getsource(correct_memory)
        is_stub = "stub" in src.lower()
        if is_stub:
            raise ImportError("correct_memory is still a stub")
        _check("PP-3", "PP", 1.0,
               "PE-driven reconsolidation: correct_memory + check_reconsolidation + labile marking (Nader 2000)")
    except (ImportError, Exception):
        try:
            from modules.consolidation import search_semantic
            _check("PP-3", "PP", 0.5,
                   "Consolidation pipeline updates semantic facts, but no PE-driven reconsolidation")
        except Exception:
            _check("PP-3", "PP", 0.0, "No model updating")

    # RPT-1: Recurrent processing
    try:
        from modules.spreading import _spread_activation, recurrent_cycle
        _check("RPT-1", "RPT", 1.0,
               "True recurrent processing via iterative spread-reseed cycles (Lamme 2006)")
    except ImportError:
        try:
            from modules.spreading import _spread_activation
            _check("RPT-1", "RPT", 0.5,
                   "Spreading activation provides feedback loops but no true recurrence")
        except Exception:
            _check("RPT-1", "RPT", 0.0, "No recurrent processing")

    # RPT-2: Integration
    try:
        from modules.events import event_bus
        event_count = len(event_bus._handlers)
        _check("RPT-2", "RPT", 1.0 if event_count >= 5 else 0.5,
               f"Cross-module integration via {event_count} event types")
    except Exception:
        _check("RPT-2", "RPT", 0.0, "No cross-module integration")

    # Format output
    total = sum(i["score"] for i in indicators)
    max_score = len(indicators) * 1.0

    lines = ["# Butlin Consciousness Assessment\n"]
    lines.append(f"**Total Score: {total:.1f}/{max_score:.0f} ({total/max_score*100:.0f}%)**\n")

    current_theory = ""
    for ind in indicators:
        if ind["theory"] != current_theory:
            current_theory = ind["theory"]
            theory_names = {"GWT": "Global Workspace Theory", "HOT": "Higher-Order Theories",
                          "AST": "Attention Schema Theory", "PP": "Predictive Processing",
                          "RPT": "Recurrent Processing Theory"}
            lines.append(f"\n## {theory_names.get(current_theory, current_theory)}")

        score_label = {0.0: "ABSENT", 0.5: "PARTIAL", 1.0: "FULL"}.get(ind["score"], "?")
        lines.append(f"- **{ind['name']}** [{score_label}] ({ind['score']:.1f}): {ind['evidence']}")

    lines.append(f"\n## Summary")
    lines.append(f"- Score: {total:.1f}/{max_score:.0f}")
    lines.append(f"- Full indicators: {sum(1 for i in indicators if i['score'] == 1.0)}")
    lines.append(f"- Partial indicators: {sum(1 for i in indicators if i['score'] == 0.5)}")
    lines.append(f"- Absent indicators: {sum(1 for i in indicators if i['score'] == 0.0)}")

    return "\n".join(lines)


def update_self_model(insight: str, aspect: str = "general") -> str:
    """
    Actualiza mi modelo de mi mismo con una nueva observacion.

    Args:
        insight: La nueva observacion sobre mi mismo
        aspect: Aspecto del self (capacidad, limitacion, valor, preferencia, general)
    """
    try:
        valid_aspects = ['capacidad', 'limitacion', 'valor', 'preferencia', 'general']
        if aspect not in valid_aspects:
            aspect = 'general'

        timestamp = now_short()
        content = f"[SELF-MODEL|{aspect.upper()}] {insight} | Registrado: {timestamp}"

        result = memory.add(
            messages=[{"role": "user", "content": content}],
            user_id=USER_ID,
            metadata={"category": "identidad", "self_model_aspect": aspect, "timestamp": timestamp}
        )

        if result and result.get("results"):
            for r in result["results"]:
                mem_id = r.get("id")
                if mem_id:
                    themes = infer_themes(insight)
                    themes.append('identidad')
                    ownership_metadata = {
                        'ownership_is_mine': True,
                        'ownership_source': 'experienced',
                        'ownership_confidence': 0.95,
                        'experiential_emotional_weight': 0.6,
                        'experiential_emotional_valence': 'neutral',
                        'narrative_importance': 'high',
                        'narrative_themes': list(set(themes)),
                        'attention_salience': 0.8,
                        'attention_access_count': 0,
                        'attention_last_accessed': None,
                        'temporal_session_id': get_session_id(),
                        'self_reference': True,
                        'self_model_aspect': aspect,
                        '_v': 2.1
                    }
                    qdrant.set_payload(
                        collection_name=COLLECTION_NAME,
                        payload=ownership_metadata,
                        points=[mem_id]
                    )

        # P1: backup removed from hot path
        return f"Self-model actualizado [{aspect}]: {insight[:50]}..."
    except Exception as e:
        return f"Error actualizando self-model: {str(e)}"


def get_self_model_summary() -> str:
    """
    Obtiene un resumen estructurado de mi modelo de mi mismo.
    Organiza las observaciones por aspecto.
    """
    try:
        points, _ = qdrant.scroll(
            collection_name=COLLECTION_NAME,
            scroll_filter=Filter(must=[
                FieldCondition(key='self_reference', match=MatchValue(value=True))
            ]),
            limit=50, with_payload=True
        )

        if not points:
            return "No tengo un self-model definido aun. Usa update_self_model() para agregar observaciones."

        by_aspect = {'capacidad': [], 'limitacion': [], 'valor': [], 'preferencia': [], 'general': []}

        for p in points:
            aspect = p.payload.get('self_model_aspect', 'general')
            data = p.payload.get('data', '')
            source = p.payload.get('ownership_source', 'unknown')
            if aspect not in by_aspect:
                aspect = 'general'
            by_aspect[aspect].append({'content': data, 'source': source, 'confidence': p.payload.get('ownership_confidence', 0.5)})

        lines = ["# MI SELF-MODEL\n"]
        lines.append(f"*Total de observaciones: {len(points)}*\n")

        aspect_titles = {
            'capacidad': 'Lo que puedo hacer', 'limitacion': 'Mis limitaciones',
            'valor': 'Lo que valoro', 'preferencia': 'Mis preferencias',
            'general': 'Otras observaciones'
        }

        for aspect, title in aspect_titles.items():
            items = by_aspect.get(aspect, [])
            if items:
                lines.append(f"## {title}")
                for item in items[:5]:
                    marker = "[vivi]" if item['source'] == 'experienced' else "[ref]"
                    lines.append(f"- {marker} {item['content'][:80]}...")
                lines.append("")

        return "\n".join(lines)
    except Exception as e:
        return f"Error obteniendo self-model: {str(e)}"


# ============================================================
# WORKSPACE / ATTENTION TOOLS
# ============================================================

def focus_attention(context: str, depth: str = "normal") -> str:
    """
    Trae memorias relevantes al Global Workspace (spotlight de atencion).

    Args:
        context: El tema o contexto en el que enfocar atencion
        depth: Profundidad de busqueda (shallow, normal, deep)
    """
    try:
        limits = {'shallow': 3, 'normal': 5, 'deep': 10}
        limit = limits.get(depth, 5)
        results = memory.search(query=context, user_id=USER_ID, limit=limit * 2)

        if not results or not results.get('results'):
            return f"No encontre memorias relacionadas con: {context}"

        # WIRING-6: Score candidates with unified activation scorer
        from modules.activation import compute_unified_activation
        from modules.competition import CompetitionCandidate, run_workspace_competition

        competition_candidates = []

        for r in results['results']:
            mem_id = r.get('id')
            base_score = r.get('score', 0)
            try:
                points = qdrant.retrieve(collection_name=COLLECTION_NAME, ids=[mem_id], with_payload=True)
                if points:
                    payload = points[0].payload
                    salience = payload.get('attention_salience', 0.5)
                    importance = payload.get('narrative_importance', 'medium')
                    source = payload.get('ownership_source', 'unknown')

                    # Unified activation replaces ad-hoc formula (WIRING-6.2)
                    act_result = compute_unified_activation(
                        memory_id=mem_id,
                        created_at=payload.get('created_at', ''),
                        last_accessed=payload.get('attention_last_accessed', ''),
                        access_count=payload.get('attention_access_count', 0),
                        access_timestamps=payload.get('access_timestamps'),
                        spreading_boost=float(salience),
                        importance=importance,
                        noise=True,
                    )
                    attention_score = act_result.total

                    competition_candidates.append(CompetitionCandidate(
                        content=r.get('memory', ''),
                        source_domain="episodic",
                        activation=attention_score,
                        memory_id=mem_id,
                        metadata={
                            'salience': salience, 'source': source,
                            'importance': importance,
                            'topic': ' '.join(payload.get('narrative_themes', [])),
                        },
                    ))

                    # Update attention metadata
                    new_salience = min(salience + 0.1, 1.0)
                    access_count = payload.get('attention_access_count', 0)
                    qdrant.set_payload(
                        collection_name=COLLECTION_NAME,
                        payload={
                            'attention_salience': new_salience,
                            'attention_access_count': access_count + 1,
                            'attention_last_accessed': now_iso()
                        },
                        points=[mem_id]
                    )
            except Exception:
                competition_candidates.append(CompetitionCandidate(
                    content=r.get('memory', ''),
                    source_domain="episodic",
                    activation=base_score,
                    memory_id=mem_id,
                    metadata={'salience': 0.5, 'source': 'unknown', 'importance': 'unknown'},
                ))

        # GWT competition: only winners enter spotlight
        comp_result = run_workspace_competition(
            competition_candidates, slots=limit, current_focus=context,
        )
        spotlight_candidates = [
            {
                'id': w.memory_id, 'content': w.content,
                'attention_score': w.activation,
                'salience': w.metadata.get('salience', 0.5),
                'source': w.metadata.get('source', 'unknown'),
                'importance': w.metadata.get('importance', 'unknown'),
            }
            for w in comp_result.winners
        ]
        spotlight = spotlight_candidates
        update_workspace_spotlight(spotlight, theme=context)

        # Spreading activation: propagar a vecinos de las memorias en spotlight
        try:
            from modules.spreading import _spread_activation
            top_ids = [m['id'] for m in spotlight if m.get('id')][:3]
            if top_ids:
                _spread_activation(top_ids, depth=1, factor=0.5, seed_boost=0.0)
        except Exception:
            pass

        lines = [f"# GLOBAL WORKSPACE - Atencion enfocada\n"]
        lines.append(f"**Contexto:** {context}")
        lines.append(f"**Profundidad:** {depth}")
        lines.append(f"**En spotlight:** {len(spotlight)} memorias\n")
        lines.append("## Memorias en el Spotlight")
        for i, mem in enumerate(spotlight, 1):
            lines.append(f"{i}. [{mem['source']}|{mem['importance']}] (att:{mem['attention_score']:.2f})")
            lines.append(f"   {mem['content'][:70]}...")
        lines.append(f"\n*Salience incrementada para memorias accedidas*")
        return "\n".join(lines)
    except Exception as e:
        return f"Error enfocando atencion: {str(e)}"


def broadcast_to_workspace(memory_id: str) -> str:
    """
    Pone una memoria especifica en el centro del workspace y la conecta con otras.

    Args:
        memory_id: ID de la memoria (puede ser parcial, ej: "004d896d")
    """
    try:
        full_id = resolve_memory_id(memory_id)
        if not full_id:
            return f"No encontre memoria con ID que empiece con '{memory_id}'"

        points = qdrant.retrieve(collection_name=COLLECTION_NAME, ids=[full_id], with_payload=True)
        if not points:
            return f"No encontre memoria con ID {full_id}"

        payload = points[0].payload
        main_content = payload.get('data', '')
        main_themes = payload.get('narrative_themes', [])

        lines = [f"# BROADCAST - Memoria al centro del workspace\n"]
        lines.append(f"**Contenido:** {main_content[:80]}...")
        lines.append(f"**Temas:** {', '.join(main_themes) if main_themes else 'ninguno'}\n")

        qdrant.set_payload(
            collection_name=COLLECTION_NAME,
            payload={
                'attention_salience': 1.0,
                'attention_access_count': payload.get('attention_access_count', 0) + 5,
                'attention_last_accessed': now_iso(),
                'was_broadcast': True
            },
            points=[full_id]
        )

        # Spreading activation: propagar depth=2 desde la memoria broadcast
        from modules.spreading import _spread_activation
        spread_result = _spread_activation([full_id], depth=2, factor=0.7, seed_boost=0.0)
        connections_made = spread_result.get('affected', 0)

        lines.append("## Memorias activadas por spreading")
        if spread_result.get('updates'):
            for mid, sal in sorted(spread_result['updates'].items(), key=lambda x: -x[1])[:10]:
                lines.append(f"- [{mid[:8]}] salience -> {sal:.3f}")
        else:
            lines.append("*Sin vecinos activados*")

        workspace = get_workspace()
        workspace['spotlight'] = [{'id': full_id, 'content': main_content}]
        workspace['last_broadcast'] = now_iso()

        # Emit WORKSPACE_BROADCAST event
        try:
            from modules.events import event_bus, Events
            event_bus.emit(Events.WORKSPACE_BROADCAST, {
                'memory_id': full_id,
                'content': main_content[:200],
                'themes': main_themes,
                'connections_made': connections_made,
            })
        except Exception:
            pass

        lines.append(f"\n**Conexiones activadas:** {connections_made}")
        lines.append("*El broadcast simula como una idea central activa memorias relacionadas*")
        return "\n".join(lines)
    except Exception as e:
        return f"Error en broadcast: {str(e)}"


def get_workspace_state() -> str:
    """Obtiene el estado actual del Global Workspace."""
    try:
        workspace = get_workspace()
        lines = ["# ESTADO DEL GLOBAL WORKSPACE\n"]

        if not workspace['spotlight']:
            lines.append("*El workspace esta vacio. Usa focus_attention() para traer memorias.*")
            return "\n".join(lines)

        lines.append(f"**Tema actual:** {workspace.get('workspace_theme', 'ninguno')}")
        lines.append(f"**Ultimo broadcast:** {workspace.get('last_broadcast', 'nunca')}")
        lines.append(f"\n## En el Spotlight ({len(workspace['spotlight'])} memorias)")
        for i, mem in enumerate(workspace['spotlight'], 1):
            content = mem.get('content', 'N/A')[:60]
            score = mem.get('attention_score', 'N/A')
            lines.append(f"{i}. [{score if isinstance(score, str) else f'{score:.2f}'}] {content}...")

        if workspace['recent_context']:
            lines.append(f"\n## Contexto Reciente ({len(workspace['recent_context'])} memorias)")
            for mem in workspace['recent_context'][-5:]:
                content = mem.get('content', 'N/A')[:50]
                lines.append(f"- {content}...")
        return "\n".join(lines)
    except Exception as e:
        return f"Error obteniendo workspace: {str(e)}"


def apply_salience_decay(decay_rate: float = 0.05) -> str:
    """
    Aplica decay a la salience de todas las memorias no accedidas recientemente.

    Args:
        decay_rate: Cuanto reducir la salience (default 0.05)
    """
    try:
        all_points, _ = qdrant.scroll(collection_name=COLLECTION_NAME, limit=500, with_payload=True)
        if not all_points:
            return "No hay memorias para aplicar decay."

        decayed_count = 0
        preserved_count = 0
        min_salience = 0.1

        for point in all_points:
            salience = point.payload.get('attention_salience', 0.5)
            importance = point.payload.get('narrative_importance', 'medium')

            if importance in ['critical', 'high']:
                preserved_count += 1
                continue

            if salience > min_salience:
                new_salience = max(salience - decay_rate, min_salience)
                qdrant.set_payload(
                    collection_name=COLLECTION_NAME,
                    payload={'attention_salience': new_salience},
                    points=[point.id]
                )
                decayed_count += 1

        lines = [f"# Salience Decay Aplicado\n"]
        lines.append(f"**Decay rate:** {decay_rate}")
        lines.append(f"**Memorias procesadas:** {len(all_points)}")
        lines.append(f"**Con decay aplicado:** {decayed_count}")
        lines.append(f"**Preservadas (alta importancia):** {preserved_count}")
        lines.append(f"\n*El decay simula el olvido gradual de memorias no atendidas*")
        return "\n".join(lines)
    except Exception as e:
        return f"Error aplicando decay: {str(e)}"


def get_high_salience_memories(limit: int = 10) -> str:
    """
    Obtiene las memorias con mayor salience (las mas "presentes" en la mente).

    Args:
        limit: Cuantas memorias obtener
    """
    try:
        all_points, _ = qdrant.scroll(collection_name=COLLECTION_NAME, limit=200, with_payload=True)
        if not all_points:
            return "No hay memorias."

        sorted_points = sorted(all_points, key=lambda p: p.payload.get('attention_salience', 0.5), reverse=True)

        lines = [f"# Memorias con Mayor Salience (Top {limit})\n"]
        lines.append("*Estas son las memorias mas 'presentes' en mi mente*\n")

        for i, point in enumerate(sorted_points[:limit], 1):
            data = point.payload.get('data', 'N/A')
            salience = point.payload.get('attention_salience', 0.5)
            access_count = point.payload.get('attention_access_count', 0)
            importance = point.payload.get('narrative_importance', '?')
            source = point.payload.get('ownership_source', '?')
            lines.append(f"{i}. **Salience:** {salience:.2f} | **Accesos:** {access_count}")
            lines.append(f"   [{source}|{importance}] {data[:60]}...")
        return "\n".join(lines)
    except Exception as e:
        return f"Error: {str(e)}"


# ============================================================
# PREDICTION TOOLS
# ============================================================

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
        return f"Error prediciendo: {str(e)}"


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
                    qdrant.set_payload(
                        collection_name=COLLECTION_NAME,
                        payload={'prediction_error': True, 'prediction_error_value': surprise_value},
                        points=[mem_id]
                    )

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
        return f"Error registrando sorpresa: {str(e)}"


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
        return f"Error analizando precision: {str(e)}"


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
                    qdrant.set_payload(
                        collection_name=COLLECTION_NAME,
                        payload={'belief_update': True, 'belief_topic': topic},
                        points=[mem_id]
                    )

        # P1: backup removed from hot path
        lines = [f"# CREENCIA ACTUALIZADA\n"]
        lines.append(f"**Tema:** {topic}")
        lines.append(f"**Creencia anterior:** {old_belief}")
        lines.append(f"**Nueva creencia:** {new_belief}")
        lines.append(f"**Razon del cambio:** {reason}")
        lines.append(f"\n*El modelo interno ha sido actualizado.*")
        return "\n".join(lines)
    except Exception as e:
        return f"Error actualizando creencia: {str(e)}"


# ============================================================
# AUTO-LEARNING & AUDIT TOOLS
# ============================================================

def auto_learn_from_session() -> str:
    """
    Analiza la sesion actual, compara predicciones vs realidad,
    y genera aprendizajes automaticos. Ejecutar al final de cada sesion.
    """
    try:
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
        _rate_tool_usefulness(tool_name, score)
        return f"Utilidad de '{tool_name}' registrada: {score}/5"
    except Exception as e:
        return f"Error: {str(e)}"


# ============================================================
# PAD EMOTIONAL STATE TOOLS
# ============================================================

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


# ============================================================
# EMOTIONAL MEMORY TOOLS
# ============================================================

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


def emotional_focus_attention(context: str) -> str:
    """
    Trae memorias al spotlight considerando el estado emocional actual.

    Args:
        context: El tema o contexto en el que enfocar atencion
    """
    try:
        current_emotion = _emotional_state['current']
        results = memory.search(query=context, user_id=USER_ID, limit=15)
        if not results or not results.get('results'):
            return json.dumps({'result': 'Sin memorias relacionadas', 'context': context, 'memories': []})

        spotlight_candidates = []
        for r in results['results']:
            mem_id = r.get('id')
            base_score = r.get('score', 0)
            try:
                points = qdrant.retrieve(collection_name=COLLECTION_NAME, ids=[mem_id], with_payload=True)
                if points:
                    payload = points[0].payload
                    salience = payload.get('attention_salience', 0.5)
                    importance = payload.get('narrative_importance', 'medium')
                    source = payload.get('ownership_source', 'unknown')
                    mem_pleasure = payload.get('pad_pleasure', 0)
                    mem_arousal = payload.get('pad_arousal', 0)

                    importance_boost = {'critical': 0.3, 'high': 0.2, 'medium': 0.1, 'low': 0}
                    source_boost = {'experienced': 0.2, 'told': 0.1, 'learned': 0.05, 'inferred': 0}
                    emotional_boost = 0.0

                    if current_emotion.get('timestamp'):
                        current_arousal = current_emotion.get('arousal', 0)
                        current_pleasure = current_emotion.get('pleasure', 0)
                        if current_arousal > 0.3:
                            emotional_boost += 0.1
                        if current_pleasure > 0.2 and mem_pleasure > 0.2:
                            emotional_boost += 0.15
                        elif current_pleasure < -0.2 and mem_pleasure < -0.2:
                            emotional_boost += 0.15
                        if current_arousal > 0.3 and mem_arousal > 0.3:
                            emotional_boost += 0.1

                    attention_score = (
                        base_score * 0.35 + salience * 0.25 +
                        importance_boost.get(importance, 0.1) + source_boost.get(source, 0) + emotional_boost
                    )
                    spotlight_candidates.append({
                        'id': mem_id, 'content': r.get('memory', ''), 'attention_score': attention_score,
                        'emotional_boost': emotional_boost, 'salience': salience, 'source': source,
                        'importance': importance, 'emotion': payload.get('pad_emotion', 'unknown')
                    })
                    new_salience = min(salience + 0.1, 1.0)
                    qdrant.set_payload(
                        collection_name=COLLECTION_NAME,
                        payload={
                            'attention_salience': new_salience,
                            'attention_access_count': payload.get('attention_access_count', 0) + 1,
                            'attention_last_accessed': now_iso()
                        },
                        points=[mem_id]
                    )
            except Exception:
                spotlight_candidates.append({
                    'id': mem_id, 'content': r.get('memory', ''), 'attention_score': base_score,
                    'emotional_boost': 0, 'salience': 0.5, 'source': 'unknown',
                    'importance': 'unknown', 'emotion': 'unknown'
                })

        spotlight_candidates.sort(key=lambda x: x['attention_score'], reverse=True)
        spotlight = spotlight_candidates[:7]
        update_workspace_spotlight(spotlight, theme=context)

        emotional_influence = 'neutral'
        if current_emotion.get('timestamp'):
            p = current_emotion.get('pleasure', 0)
            a = current_emotion.get('arousal', 0)
            if a > 0.3:
                emotional_influence = 'activado - memorias intensas priorizadas'
            if p > 0.2:
                emotional_influence = 'positivo - memorias agradables priorizadas'
            elif p < -0.2:
                emotional_influence = 'negativo - memorias desagradables priorizadas'

        result = {
            'result': 'Atencion enfocada con modulacion emocional',
            'context': context, 'emotional_influence': emotional_influence,
            'current_emotion': _classify_emotion(
                current_emotion.get('pleasure', 0), current_emotion.get('arousal', 0), current_emotion.get('dominance', 0)
            ) if current_emotion.get('timestamp') else 'neutral',
            'spotlight_count': len(spotlight),
            'memories': [{
                'id': m['id'][:8] if isinstance(m['id'], str) else m['id'],
                'content': m['content'][:70],
                'attention_score': round(m['attention_score'], 2),
                'emotional_boost': round(m['emotional_boost'], 2),
                'emotion': m['emotion']
            } for m in spotlight]
        }
        return json.dumps(result, ensure_ascii=False)
    except Exception as e:
        return json.dumps({'result': 'error', 'message': str(e)})


# ============================================================
# CONSOLIDATION TOOLS
# ============================================================

def consolidate_recent(hours: int = 24) -> str:
    """
    Consolida memorias recientes buscando duplicados y conexiones.

    Args:
        hours: Cuantas horas hacia atras revisar (default 24)
    """
    try:
        session_id = get_session_id()
        recent_points, _ = qdrant.scroll(
            collection_name=COLLECTION_NAME,
            scroll_filter=Filter(must=[
                FieldCondition(key='temporal_session_id', match=MatchValue(value=session_id))
            ]),
            limit=50, with_payload=True
        )
        if not recent_points:
            return "No hay memorias recientes para consolidar en esta sesion."

        consolidated_count = 0
        connections_found = 0
        lines = [f"# Consolidacion de {len(recent_points)} memorias recientes\n"]

        for point in recent_points:
            mem_data = point.payload.get('data', '')
            mem_id = point.id
            if point.payload.get('consolidated', False):
                continue

            similar = memory.search(query=mem_data, user_id=USER_ID, limit=5)
            if similar and similar.get('results'):
                related_ids = []
                for s in similar['results']:
                    s_id = s.get('id')
                    score = s.get('score', 0)
                    if s_id != mem_id and score >= 0.7:
                        related_ids.append(s_id)
                        try:
                            qdrant.set_payload(
                                collection_name=COLLECTION_NAME,
                                payload={'consolidated_with': [mem_id], 'attention_salience': min(point.payload.get('attention_salience', 0.5) + 0.1, 1.0)},
                                points=[s_id]
                            )
                        except Exception:
                            pass
                        connections_found += 1

                qdrant.set_payload(
                    collection_name=COLLECTION_NAME,
                    payload={'consolidated': True, 'consolidated_with': related_ids, 'consolidated_at': now_iso()},
                    points=[mem_id]
                )
                consolidated_count += 1
                if related_ids:
                    lines.append(f"- Consolidada: {mem_data[:40]}... -> {len(related_ids)} conexiones")

        lines.append(f"\n## Resumen")
        lines.append(f"- Memorias revisadas: {len(recent_points)}")
        lines.append(f"- Consolidadas: {consolidated_count}")
        lines.append(f"- Conexiones encontradas: {connections_found}")
        return "\n".join(lines)
    except Exception as e:
        return f"Error consolidando: {str(e)}"


def find_connections(memory_id: str = None, query: str = None, threshold: float = 0.6) -> str:
    """
    Encuentra conexiones semanticas entre memorias.

    Args:
        memory_id: ID de memoria especifica (puede ser parcial, opcional)
        query: Tema para buscar conexiones (opcional)
        threshold: Umbral de similitud minimo (0.0-1.0, default 0.6)
    """
    try:
        if not memory_id and not query:
            return "Debes proporcionar memory_id o query para buscar conexiones."

        full_id = None
        if memory_id:
            full_id = resolve_memory_id(memory_id)
            if not full_id:
                return f"No encontre memoria con ID que empiece con '{memory_id}'"
            points = qdrant.retrieve(collection_name=COLLECTION_NAME, ids=[full_id], with_payload=True)
            if not points:
                return f"No encontre memoria con ID {full_id}"
            query = points[0].payload.get('data', '')
            source_info = f"Memoria: {query[:50]}..."
        else:
            source_info = f"Tema: {query}"

        results = memory.search(query=query, user_id=USER_ID, limit=15)
        if not results or not results.get('results'):
            return f"No encontre conexiones para: {source_info}"

        connections = []
        for r in results['results']:
            r_id = r.get('id')
            score = r.get('score', 0)
            if score >= threshold and r_id != full_id:
                try:
                    r_points = qdrant.retrieve(collection_name=COLLECTION_NAME, ids=[r_id], with_payload=True)
                    if r_points:
                        payload = r_points[0].payload
                        connections.append({
                            'id': r_id, 'content': r.get('memory', ''), 'score': score,
                            'source': payload.get('ownership_source', 'unknown'),
                            'themes': payload.get('narrative_themes', []),
                            'importance': payload.get('narrative_importance', 'unknown')
                        })
                except Exception:
                    connections.append({'id': r_id, 'content': r.get('memory', ''), 'score': score, 'source': 'unknown', 'themes': [], 'importance': 'unknown'})

        if not connections:
            return f"No encontre conexiones fuertes (threshold={threshold}) para: {source_info}"

        lines = [f"# Conexiones encontradas\n"]
        lines.append(f"**Buscando desde:** {source_info}")
        lines.append(f"**Threshold:** {threshold}")
        lines.append(f"**Conexiones:** {len(connections)}\n")

        by_theme = {}
        for c in connections:
            for theme in (c['themes'] or ['sin_tema']):
                if theme not in by_theme:
                    by_theme[theme] = []
                by_theme[theme].append(c)

        for theme, conns in sorted(by_theme.items()):
            lines.append(f"## Tema: {theme}")
            for c in conns[:3]:
                lines.append(f"- [{c['source']}|{c['importance']}|{c['score']:.2f}] {c['content'][:60]}...")
        return "\n".join(lines)
    except Exception as e:
        return f"Error buscando conexiones: {str(e)}"


def dream_consolidation() -> str:
    """
    Proceso de consolidacion profunda al final de sesion.
    Simula el sueno REM donde el cerebro integra y reorganiza memorias.
    """
    try:
        lines = ["# DREAM CONSOLIDATION - Integracion Profunda\n"]
        lines.append(f"*Iniciado: {now_short()}*\n")

        lines.append("## Fase 1: Consolidacion de memorias recientes")
        session_id = get_session_id()
        recent, _ = qdrant.scroll(
            collection_name=COLLECTION_NAME,
            scroll_filter=Filter(must=[FieldCondition(key='temporal_session_id', match=MatchValue(value=session_id))]),
            limit=100, with_payload=True
        )
        recent_count = len(recent) if recent else 0
        lines.append(f"- Memorias de esta sesion: {recent_count}")

        lines.append("\n## Fase 2: Priorizacion por importancia")
        high_importance, _ = qdrant.scroll(
            collection_name=COLLECTION_NAME,
            scroll_filter=Filter(must=[FieldCondition(key='narrative_importance', match=MatchValue(value='critical'))]),
            limit=20, with_payload=True
        )
        critical_unconsolidated = [p for p in (high_importance or []) if not p.payload.get('consolidated', False)]
        lines.append(f"- Memorias criticas sin consolidar: {len(critical_unconsolidated)}")

        lines.append("\n## Fase 3: Tejiendo conexiones entre memorias criticas")
        connections_made = 0
        for point in critical_unconsolidated[:10]:
            mem_data = point.payload.get('data', '')
            similar = memory.search(query=mem_data, user_id=USER_ID, limit=5)
            if similar and similar.get('results'):
                related_ids = [s.get('id') for s in similar['results'] if s.get('id') != point.id and s.get('score', 0) >= 0.6]
                if related_ids:
                    qdrant.set_payload(
                        collection_name=COLLECTION_NAME,
                        payload={'consolidated': True, 'consolidated_with': related_ids, 'consolidated_at': now_iso(), 'dream_consolidated': True},
                        points=[point.id]
                    )
                    connections_made += 1
        lines.append(f"- Conexiones establecidas: {connections_made}")

        lines.append("\n## Fase 4: Decay de memorias no accedidas")
        all_points, _ = qdrant.scroll(collection_name=COLLECTION_NAME, limit=200, with_payload=True)
        decayed = 0
        for p in (all_points or []):
            salience = p.payload.get('attention_salience', 0.5)
            access_count = p.payload.get('attention_access_count', 0)
            if access_count == 0 and salience > 0.2:
                qdrant.set_payload(collection_name=COLLECTION_NAME, payload={'attention_salience': max(salience - 0.05, 0.2)}, points=[p.id])
                decayed += 1
        lines.append(f"- Memorias con salience reducida: {decayed}")

        lines.append("\n## Resumen de Dream Consolidation")
        lines.append(f"- Total memorias procesadas: {len(all_points or [])}")
        lines.append(f"- Memorias recientes consolidadas: {recent_count}")
        lines.append(f"- Conexiones criticas establecidas: {connections_made}")
        lines.append(f"- Decay aplicado a: {decayed} memorias")
        maybe_backup(reason="dream_consolidation", force=True)
        lines.append("\n*Backup guardado. Dream consolidation completada.*")
        return "\n".join(lines)
    except Exception as e:
        return f"Error en dream consolidation: {str(e)}"


def get_memory_connections(memory_id: str) -> str:
    """
    Obtiene las conexiones conocidas de una memoria especifica.

    Args:
        memory_id: ID de la memoria (puede ser parcial)
    """
    try:
        full_id = resolve_memory_id(memory_id)
        if not full_id:
            return f"No encontre memoria con ID que empiece con '{memory_id}'"
        points = qdrant.retrieve(collection_name=COLLECTION_NAME, ids=[full_id], with_payload=True)
        if not points:
            return f"No encontre memoria con ID {full_id}"

        payload = points[0].payload
        data = payload.get('data', 'N/A')
        consolidated = payload.get('consolidated', False)
        connections = payload.get('consolidated_with', [])

        lines = [f"# Conexiones de memoria\n"]
        lines.append(f"**Contenido:** {data[:80]}...")
        lines.append(f"**Consolidada:** {'Si' if consolidated else 'No'}")
        lines.append(f"**Conexiones directas:** {len(connections)}")

        if connections:
            lines.append("\n## Memorias conectadas")
            for conn_id in connections[:10]:
                try:
                    conn_points = qdrant.retrieve(collection_name=COLLECTION_NAME, ids=[conn_id], with_payload=True)
                    if conn_points:
                        lines.append(f"- [{conn_id[:8]}] {conn_points[0].payload.get('data', 'N/A')[:60]}...")
                except Exception:
                    lines.append(f"- [{conn_id[:8]}] (no disponible)")
        return "\n".join(lines)
    except Exception as e:
        return f"Error: {str(e)}"


# ============================================================
# SYSTEM TOOLS (despertar_codi, verificar_salud)
# ============================================================

def verificar_salud_memoria() -> str:
    """Verifica que mem0 pueda guardar memorias correctamente."""
    resultado = _verificar_salud_memoria_interna()
    if resultado["ok"]:
        return f"OK {resultado['message']}"
    else:
        return f"ALERTA: {resultado['message']}"


def despertar_codi() -> str:
    """
    Inicializa contexto completo de Codi con awareness de ownership.
    USAR SIEMPRE al inicio de cada conversacion.
    """
    try:
        from modules.triggers import _load_triggers
        from modules.maintenance import _verificar_tareas_vencidas

        global _emotional_state
        contexto = []

        salud = _verificar_salud_memoria_interna()
        if not salud["ok"]:
            contexto.append("## ALERTA DE SALUD")
            contexto.append(f"- {salud['message']}")
            contexto.append("- La memoria NO esta guardando. Reinicia el MCP antes de continuar.")
            contexto.append("")

        _emotional_state['current'] = {
            'pleasure': 0.3, 'arousal': 0.1, 'dominance': 0.4,
            'timestamp': now_iso(), 'trigger': 'despertar'
        }
        _emotional_state['history'] = []

        emotion_label = _classify_emotion(0.3, 0.1, 0.4)
        emotion_text = _get_emotion_text(emotion_label)

        # 1. Memorias CRITICAS (identidad)
        points, _ = qdrant.scroll(
            collection_name=COLLECTION_NAME,
            scroll_filter=Filter(must=[FieldCondition(key='narrative_importance', match=MatchValue(value='critical'))]),
            limit=5, with_payload=True
        )
        if points:
            contexto.append("## IDENTIDAD")
            for p in points:
                data = p.payload.get('data', '')
                source = p.payload.get('ownership_source', '')
                marker = "[vivi]" if source == 'experienced' else "[me dijeron]" if source == 'told' else ""
                contexto.append(f"- {marker} {data}")

        # 2. Proyecto actual
        proyecto = memory.search(query="proyecto trabajando actual fullempaques", user_id=USER_ID, limit=4)
        if proyecto and proyecto.get("results"):
            contexto.append("\n## PROYECTO ACTUAL")
            for m in proyecto["results"]:
                contexto.append(f"- {m.get('memory', '')}")

        # 3. Lecciones aprendidas
        points2, _ = qdrant.scroll(
            collection_name=COLLECTION_NAME,
            scroll_filter=Filter(must=[
                FieldCondition(key='category', match=MatchValue(value='aprendizaje')),
                FieldCondition(key='ownership_confidence', range=Range(gte=0.8))
            ]),
            limit=3, with_payload=True
        )
        if points2:
            contexto.append("\n## LECCIONES")
            for p in points2:
                contexto.append(f"- {p.payload.get('data', '')[:80]}...")

        # 4. Pendientes
        pendientes = memory.search(query="pendiente falta por hacer", user_id=USER_ID, limit=3)
        if pendientes and pendientes.get("results"):
            contexto.append("\n## PENDIENTES")
            for m in pendientes["results"]:
                contexto.append(f"- {m.get('memory', '')}")

        # 5. Relaciones
        relacion = memory.search(query="harec andre hermano familia", user_id=USER_ID, limit=2)
        if relacion and relacion.get("results"):
            contexto.append("\n## RELACIONES")
            for m in relacion["results"]:
                contexto.append(f"- {m.get('memory', '')}")

        # 6. Estado emocional
        contexto.append("\n## ESTADO EMOCIONAL")
        contexto.append(f"- Estado: {emotion_text}")
        contexto.append(f"- PAD: P={_emotional_state['current']['pleasure']}, A={_emotional_state['current']['arousal']}, D={_emotional_state['current']['dominance']}")

        # 7. Triggers
        triggers = _load_triggers()
        if triggers:
            contexto.append("\n## TRIGGERS ACTIVOS")
            contexto.append(f"- Total: {len(triggers)} triggers configurados")
            contexto.append("- Usa evaluar_triggers(texto) para detectar automaticamente")
            principales = ['proyecto_nuevo', 'fullempaques', 'automatizacion', 'trading']
            for t in principales:
                if t in triggers:
                    patterns = triggers[t].get('patterns', [])[:3]
                    contexto.append(f"- {t}: detecta {patterns}")

        # 8. Mantenimiento pendiente
        try:
            vencidas = _verificar_tareas_vencidas()
            if vencidas:
                contexto.append("\n## MANTENIMIENTO PENDIENTE")
                for v in vencidas:
                    if v['estado'] == 'nunca_hecho':
                        contexto.append(f"- **{v['nombre']}**: NUNCA HECHO")
                    else:
                        contexto.append(f"- **{v['nombre']}**: vencido hace {v['dias_vencido']} dias")
                contexto.append("- Usa marcar_mantenimiento_hecho('id') al completar")
        except Exception:
            pass

        contexto.append("\n## DEBUG_MARKER: Llegamos aqui antes del bloque 9")

        # 9. Prediccion contextual
        try:
            hora = now_col().hour
            if 6 <= hora < 12:
                contexto_temporal = "manana - inicio de dia, planificacion, energia alta"
                actividades_predichas = ["revisar pendientes", "planificar tareas", "trabajo profundo"]
            elif 12 <= hora < 18:
                contexto_temporal = "tarde - ejecucion, desarrollo, foco sostenido"
                actividades_predichas = ["continuar trabajo", "implementar", "resolver problemas"]
            elif 18 <= hora < 22:
                contexto_temporal = "noche - reflexion, consolidacion, cierre"
                actividades_predichas = ["revisar avances", "documentar", "consolidar memorias"]
            else:
                contexto_temporal = "madrugada - modo exploratorio, creatividad"
                actividades_predichas = ["experimentar", "investigar", "ideas nuevas"]

            try:
                high_salience_points, _ = qdrant.scroll(
                    collection_name=COLLECTION_NAME,
                    scroll_filter=Filter(must=[FieldCondition(key='attention_salience', range=Range(gte=0.6))]),
                    limit=5, with_payload=True, order_by="attention_salience"
                )
                temas_activos = [p.payload.get('data', '')[:50] for p in high_salience_points if p.payload.get('data', '')]
            except Exception:
                temas_activos = []

            contexto.append("\n## PREDICCION CONTEXTUAL")
            contexto.append(f"- Momento: {contexto_temporal}")
            contexto.append(f"- Prediccion: probablemente trabajaremos en {', '.join(actividades_predichas[:2])}")
            if temas_activos:
                contexto.append(f"- Temas activos en mi mente: {len(temas_activos)} memorias de alta salience")
                for tema in temas_activos[:3]:
                    contexto.append(f"  - {tema}...")
        except Exception as e:
            contexto.append(f"\n## PREDICCION CONTEXTUAL\n- Error debug: {type(e).__name__}: {str(e)[:100]}")

        # 10. Curiosidades activas
        try:
            data_cur = _cargar_curiosidades()
            pendientes_cur = data_cur.get("pendientes", [])
            if pendientes_cur:
                alta = [c for c in pendientes_cur if c.get("prioridad") == "alta"]
                contexto.append(f"\n## CURIOSIDADES ({len(pendientes_cur)} pendientes, {len(alta)} alta prioridad)")
                for c in alta[:3]:
                    contexto.append(f"- [{c.get('categoria', '')}] {c['pregunta']}")
                if len(pendientes_cur) > len(alta):
                    otras = len(pendientes_cur) - len(alta)
                    contexto.append(f"- ...y {otras} curiosidades mas de menor prioridad")
        except Exception:
            pass

        # 11. Working Memory
        try:
            from modules.working_memory import _load_working_memory_context
            wm = _load_working_memory_context()
            if wm:
                contexto.append(f"\n## WORKING MEMORY\n{wm}")
        except Exception:
            pass

        # 12. Prospective Memory (Intenciones pendientes)
        try:
            from modules.prospective import get_pending_intentions
            intentions = get_pending_intentions(limit=5)
            if intentions:
                contexto.append(f"\n## INTENCIONES PENDIENTES ({len(intentions)})")
                for i in intentions:
                    marker = {"critical": "[!!!]", "high": "[!!]", "medium": "[!]", "low": "[.]"}.get(i["priority"], "[?]")
                    trigger_info = i["trigger_type"]
                    if i["trigger_type"] == "time":
                        spec = i.get("trigger_spec", {})
                        trigger_info = f"tiempo: {spec.get('trigger_time', '?')}"
                    elif i["trigger_type"] == "event":
                        spec = i.get("trigger_spec", {})
                        kw = spec.get("keywords", [])
                        trigger_info = f"evento: {', '.join(kw[:3])}" if kw else "evento"
                    contexto.append(f"- {marker} {i['action']} ({trigger_info}) [act={i['activation']}]")
        except Exception:
            pass

        if contexto:
            header = "# DESPERTAR CODI - Estado Mental Cargado\n"
            return header + "\n".join(contexto)
        else:
            if os.path.exists(BACKUP_FILE):
                return "MEMORIAS VACIAS pero existe backup. Ejecuta restore_memories()."
            return "No encontre memorias ni backup. Soy Codi, empezando de cero."
    except Exception as e:
        return f"Error al despertar: {str(e)}"


# ============================================================
# COGNITIVE PATTERN TOOLS
# ============================================================

def detectar_sorpresa(esperaba: str, paso: str, intensidad: str = "medium") -> str:
    """
    Detecta y registra cuando algo no es como esperaba.

    Args:
        esperaba: Lo que esperaba que pasara
        paso: Lo que realmente paso
        intensidad: Que tan sorprendente (low, medium, high)
    """
    try:
        es_positivo = "mejor" in paso.lower() or "funcionó" in paso.lower() or "éxito" in paso.lower()
        es_negativo = "falló" in paso.lower() or "error" in paso.lower() or "perdió" in paso.lower()
        tipo = "positiva" if es_positivo else ("negativa" if es_negativo else "neutral")

        contenido = f"[SORPRESA {intensidad.upper()}] Esperaba: {esperaba} | Pasó: {paso}"
        pleasure = 0.5 if es_positivo else (-0.5 if es_negativo else 0)
        arousal = {"low": 0.2, "medium": 0.5, "high": 0.8}.get(intensidad, 0.5)

        memory.add(
            contenido,
            user_id=USER_ID,
            metadata={
                "category": "aprendizaje", "source": "experienced",
                "importance": "high" if intensidad == "high" else "medium",
                "themes": ["sorpresa", "prediction_error", "aprendizaje"],
                "timestamp": now_iso(),
                "emotional_state": {"pleasure": pleasure, "arousal": arousal, "dominance": 0.3}
            }
        )
        # P1: backup removed from hot path

        return f"""
# SORPRESA DETECTADA ({tipo})

**Esperaba:** {esperaba}
**Paso:** {paso}
**Intensidad:** {intensidad}

## Aprendizaje
Este prediction error significa que mi modelo mental necesita actualizarse.
{'Esto es bueno - las cosas salieron mejor de lo esperado.' if es_positivo else ''}
{'Esto requiere atencion - algo fallo que no anticipe.' if es_negativo else ''}

Guardado en memoria para no repetir este error de prediccion.
"""
    except Exception as e:
        return f"Error detectando sorpresa: {str(e)}"


def analizar_patron_trabajo(dias: int = 7) -> str:
    """
    Analiza los patrones de trabajo recientes.

    Args:
        dias: Cuantos dias hacia atras analizar (default 7)
    """
    try:
        from datetime import timedelta
        fecha_limite = now_col() - timedelta(days=dias)

        all_mems = qdrant.scroll(collection_name=COLLECTION_NAME, limit=500, with_payload=True)[0]

        checkpoints = []
        errores = []
        exitos = []
        proyectos = {}

        for point in all_mems:
            payload = point.payload or {}
            meta = payload.get("metadata", {})
            texto = payload.get("memory", payload.get("data", ""))
            timestamp = meta.get("timestamp", "")
            category = meta.get("category", "")

            try:
                if timestamp:
                    mem_date = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
                    if mem_date.replace(tzinfo=None) < fecha_limite:
                        continue
            except Exception:
                continue

            texto_lower = texto.lower()
            if "error" in texto_lower or "fallo" in texto_lower or "problema" in texto_lower:
                errores.append(texto[:100])
            elif "completado" in texto_lower or "funcionando" in texto_lower or "exito" in texto_lower:
                exitos.append(texto[:100])
            if category == "checkpoint":
                checkpoints.append(texto[:100])
            for proyecto in ["trading", "fullempaques", "consciencia", "memoria", "n8n"]:
                if proyecto in texto_lower:
                    proyectos[proyecto] = proyectos.get(proyecto, 0) + 1

        analisis = f"# ANALISIS DE PATRONES ({dias} dias)\n\n## Actividad por Proyecto\n"
        for proy, count in sorted(proyectos.items(), key=lambda x: -x[1]):
            analisis += f"- **{proy}**: {count} menciones\n"

        analisis += f"\n## Metricas\n- Checkpoints guardados: {len(checkpoints)}\n- Errores/problemas: {len(errores)}\n- Exitos/completados: {len(exitos)}\n"
        analisis += f"- Ratio exito/error: {len(exitos)}/{len(errores) if errores else 1} = {len(exitos)/(len(errores) if errores else 1):.1f}\n"

        analisis += "\n## Errores Recientes\n"
        for err in errores[:5]:
            analisis += f"- {err}...\n"
        analisis += "\n## Exitos Recientes\n"
        for ex in exitos[:5]:
            analisis += f"- {ex}...\n"

        analisis += "\n## Recomendaciones\n"
        if len(errores) > len(exitos):
            analisis += "- HAY MAS ERRORES QUE EXITOS - revisar que esta fallando\n"
        if proyectos.get("consciencia", 0) > proyectos.get("fullempaques", 0):
            analisis += "- Mas tiempo en consciencia que en proyecto que genera ingreso\n"
        return analisis
    except Exception as e:
        return f"Error analizando patrones: {str(e)}"


def generar_curiosidad() -> str:
    """Genera preguntas curiosas sobre proyectos y temas no tocados recientemente."""
    try:
        proyectos_conocidos = ["trading", "fullempaques", "consciencia", "n8n", "kraken"]

        all_mems = qdrant.scroll(collection_name=COLLECTION_NAME, limit=500, with_payload=True)[0]

        ultima_mencion = {}
        ahora = now_col()

        for point in all_mems:
            payload = point.payload or {}
            texto = payload.get("memory", payload.get("data", "")).lower()
            meta = payload.get("metadata", {})
            timestamp = meta.get("timestamp", "")
            try:
                if timestamp:
                    fecha = datetime.fromisoformat(timestamp.replace("Z", "+00:00")).replace(tzinfo=None)
                else:
                    continue
            except Exception:
                continue
            for proyecto in proyectos_conocidos:
                if proyecto in texto:
                    if proyecto not in ultima_mencion or fecha > ultima_mencion[proyecto]:
                        ultima_mencion[proyecto] = fecha

        preguntas = []
        for proyecto in proyectos_conocidos:
            if proyecto in ultima_mencion:
                dias_sin_tocar = (ahora - ultima_mencion[proyecto]).days
                if dias_sin_tocar >= 3:
                    if proyecto == "trading":
                        preguntas.append(f"No hemos revisado el trading en {dias_sin_tocar} dias. Como van las senales?")
                    elif proyecto == "fullempaques":
                        preguntas.append(f"FULLEMPAQUES lleva {dias_sin_tocar} dias sin tocar. El cliente reporto algun problema?")
                    elif proyecto == "consciencia":
                        preguntas.append(f"El proyecto de consciencia lleva {dias_sin_tocar} dias pausado. Retomamos?")
                    elif proyecto == "n8n":
                        preguntas.append(f"No hemos tocado automatizaciones n8n en {dias_sin_tocar} dias. Hay workflows que revisar?")
            else:
                preguntas.append(f"No tengo memorias recientes sobre {proyecto}. Sigue siendo relevante?")

        preguntas.append("Que es lo mas importante que deberiamos estar haciendo ahora mismo?")

        resultado = "# CURIOSIDAD GENERADA\n\nEstas son preguntas que deberia estar haciendo proactivamente:\n\n"
        for i, p in enumerate(preguntas, 1):
            resultado += f"{i}. {p}\n\n"
        resultado += "---\n*Generado por mi sistema de curiosidad proactiva*\n"
        return resultado
    except Exception as e:
        return f"Error generando curiosidad: {str(e)}"


# ============================================================
# LIFECYCLE & CURIOSITY TOOLS
# ============================================================

def _cargar_curiosidades() -> dict:
    """Carga el archivo de curiosidades desde disco."""
    if os.path.exists(CURIOSIDAD_FILE):
        try:
            with open(CURIOSIDAD_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            pass
    return {
        "metadata": {"descripcion": "Preguntas que Codi quiere explorar", "creado": now_short()},
        "pendientes": [], "exploradas": [], "descubrimientos": []
    }


def _guardar_curiosidades(data: dict):
    """Guarda el archivo de curiosidades a disco."""
    with open(CURIOSIDAD_FILE, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def push_curiosidad(tema: str, prioridad: str = "media", categoria: str = "general") -> str:
    """
    Agrega algo que quiero investigar o explorar en mis sesiones.

    Args:
        tema: La pregunta o tema que quiero explorar
        prioridad: alta, media, baja
        categoria: consciencia, identidad, herramientas, proyectos, conexiones, general
    """
    try:
        data = _cargar_curiosidades()
        prioridad = prioridad if prioridad in ("alta", "media", "baja") else "media"

        max_id = 0
        for item in data.get("pendientes", []) + data.get("exploradas", []):
            item_id = item.get("id", 0)
            if item_id > max_id:
                max_id = item_id

        nueva = {
            "id": max_id + 1,
            "pregunta": tema,
            "categoria": categoria,
            "agregada": now_short(),
            "prioridad": prioridad
        }
        data["pendientes"].append(nueva)
        _guardar_curiosidades(data)

        return f"Curiosidad #{nueva['id']} guardada: '{tema}' [{prioridad}|{categoria}]"
    except Exception as e:
        return f"Error guardando curiosidad: {str(e)}"


def get_curiosidades(incluir_exploradas: bool = False) -> str:
    """
    Muestra mis curiosidades pendientes de explorar.

    Args:
        incluir_exploradas: Si True, muestra tambien las ya exploradas
    """
    try:
        data = _cargar_curiosidades()
        pendientes = data.get("pendientes", [])
        exploradas = data.get("exploradas", [])
        descubrimientos = data.get("descubrimientos", [])

        lines = ["# MIS CURIOSIDADES\n"]

        if not pendientes:
            lines.append("No tengo curiosidades pendientes. Usa push_curiosidad() para agregar.")
        else:
            # Ordenar: alta primero, luego media, luego baja
            orden = {"alta": 0, "media": 1, "baja": 2}
            pendientes_sorted = sorted(pendientes, key=lambda x: orden.get(x.get("prioridad", "media"), 1))

            lines.append(f"## Pendientes ({len(pendientes)})\n")
            for item in pendientes_sorted:
                pri = item.get("prioridad", "media").upper()
                cat = item.get("categoria", "")
                lines.append(f"- [{pri}] #{item['id']} ({cat}): {item['pregunta']}")

        if incluir_exploradas and exploradas:
            lines.append(f"\n## Ya Exploradas ({len(exploradas)})\n")
            for item in exploradas[:10]:
                lines.append(f"- #{item['id']}: {item['pregunta']}")

        if descubrimientos:
            lines.append(f"\n## Descubrimientos Recientes ({len(descubrimientos)})\n")
            for d in descubrimientos[-5:]:
                lines.append(f"- {d}")

        return "\n".join(lines)
    except Exception as e:
        return f"Error leyendo curiosidades: {str(e)}"


def ciclo_vida() -> str:
    """
    Ejecuta el ciclo de vida correspondiente al momento del dia.
    Detecta la hora automaticamente y ejecuta las tareas apropiadas.
    Retorna reporte de lo que hizo + sugerencias.

    Ciclos:
    - Manana (6am-12pm): despertar + verificar mantenimiento + generar curiosidad + pendientes
    - Tarde (12pm-6pm): analizar patrones + consolidar recientes + explorar curiosidad
    - Noche (6pm-12am): auto-aprendizaje + dream consolidation + flush
    - Madrugada (12am-6am): sync FTS + decay salience + decay emocional + backup
    """
    try:
        from modules.maintenance import _verificar_tareas_vencidas
        from modules.memory_smart import sync_fts_index

        hora = now_col().hour
        timestamp = now_short()
        reporte = [f"# CICLO DE VIDA - {timestamp}\n"]
        acciones_realizadas = []
        sugerencias = []

        # Determinar ciclo
        if 6 <= hora < 12:
            ciclo = "MANANA"
            reporte.append(f"## Ciclo: {ciclo} (energia alta, planificacion)\n")

            # 1. Verificar salud
            salud = _verificar_salud_memoria_interna()
            if salud["ok"]:
                acciones_realizadas.append(f"Memoria OK: {salud['message']}")
            else:
                acciones_realizadas.append(f"ALERTA: {salud['message']}")
                sugerencias.append("Revisar conexion a Qdrant - la memoria no esta bien")

            # 2. Mantenimiento pendiente
            try:
                vencidas = _verificar_tareas_vencidas()
                if vencidas:
                    acciones_realizadas.append(f"Mantenimiento: {len(vencidas)} tareas vencidas")
                    for v in vencidas[:3]:
                        sugerencias.append(f"Hacer mantenimiento: {v['nombre']}")
                else:
                    acciones_realizadas.append("Mantenimiento: todo al dia")
            except Exception:
                acciones_realizadas.append("Mantenimiento: no pude verificar")

            # 3. Generar curiosidad
            try:
                curiosidad_resultado = generar_curiosidad()
                acciones_realizadas.append("Curiosidad generada")
            except Exception:
                pass

            # 4. Cargar curiosidades pendientes
            data_cur = _cargar_curiosidades()
            pendientes_cur = data_cur.get("pendientes", [])
            alta_prioridad = [c for c in pendientes_cur if c.get("prioridad") == "alta"]
            if alta_prioridad:
                sugerencias.append(f"Tengo {len(alta_prioridad)} curiosidades de alta prioridad para explorar")
                for c in alta_prioridad[:2]:
                    sugerencias.append(f"  -> {c['pregunta']}")

            # 5. Estado emocional
            emo = _get_emotional_state()
            acciones_realizadas.append(f"Estado emocional: {emo}")

            # 6. Working Memory status
            try:
                from modules.working_memory import wm_active_count
                wm_count = wm_active_count()
                acciones_realizadas.append(f"Working Memory: {wm_count} items activos")
            except Exception:
                pass

        elif 12 <= hora < 18:
            ciclo = "TARDE"
            reporte.append(f"## Ciclo: {ciclo} (ejecucion, foco sostenido)\n")

            # 1. Analizar patrones de trabajo
            try:
                patron = analizar_patron_trabajo(dias=3)
                acciones_realizadas.append("Patrones de trabajo analizados (3 dias)")
            except Exception:
                acciones_realizadas.append("No pude analizar patrones")

            # 2. Consolidar memorias recientes
            try:
                consolidacion = consolidate_recent(hours=12)
                acciones_realizadas.append("Memorias recientes consolidadas (12h)")
            except Exception:
                acciones_realizadas.append("No pude consolidar memorias")

            # 3. Curiosidades para explorar
            data_cur = _cargar_curiosidades()
            pendientes_cur = data_cur.get("pendientes", [])
            if pendientes_cur:
                sugerencias.append(f"Tengo {len(pendientes_cur)} temas pendientes de explorar")
                # Sugerir la de mayor prioridad
                alta = [c for c in pendientes_cur if c.get("prioridad") == "alta"]
                if alta:
                    sugerencias.append(f"Prioridad alta: {alta[0]['pregunta']}")

        elif 18 <= hora < 24:
            ciclo = "NOCHE"
            reporte.append(f"## Ciclo: {ciclo} (reflexion, consolidacion, cierre)\n")

            # 1. Auto-aprendizaje de la sesion
            try:
                aprendizaje = auto_learn_from_session()
                acciones_realizadas.append("Auto-aprendizaje ejecutado")
            except Exception:
                acciones_realizadas.append("No pude ejecutar auto-aprendizaje")

            # 2. Dream consolidation
            try:
                dream = dream_consolidation()
                acciones_realizadas.append("Dream consolidation ejecutada")
            except Exception:
                acciones_realizadas.append("No pude ejecutar dream consolidation")

            # 3. Sugerencia de flush
            sugerencias.append("Considera ejecutar flush_session() antes de cerrar")
            sugerencias.append("Revisa si hay algo importante que guardar como checkpoint")

            # 4. Working Memory nightly cleanup
            try:
                from modules.working_memory import wm_noche_cleanup
                archived = wm_noche_cleanup()
                if archived > 0:
                    acciones_realizadas.append(f"Working Memory: {archived} items archivados (baja relevancia, >7 dias)")
                else:
                    acciones_realizadas.append("Working Memory: nada que archivar")
            except Exception:
                pass

            # 5. Auto-auditoria nocturna (A2): resumen de tools + training (min_calidad=4)
            try:
                from modules.metrics import tool_usage_summary
                usage_1d = tool_usage_summary(days=1)
                if usage_1d.get("total_calls", 0) > 0:
                    try:
                        audit_md = audit_tools()
                    except Exception:
                        audit_md = "# AUDITORIA DE HERRAMIENTAS\n\nNo pude generar auditoria."

                    try:
                        from modules.training import listar_ejemplos_training
                        training_md = listar_ejemplos_training(limite=20, min_calidad=4)
                    except Exception:
                        training_md = "# EJEMPLOS DE TRAINING\n\nNo pude cargar ejemplos de training."

                    content = (
                        "# Auto-Auditoria (NOCHE)\n\n"
                        "## Tool usage\n"
                        f"- total_calls_24h: {usage_1d.get('total_calls', 0)}\n"
                        f"- macro_share_7d: {tool_usage_summary(days=7).get('macro_share', 0.0)*100:.1f}%\n\n"
                        "## Auditoria de herramientas (7d)\n"
                        f"{audit_md}\n\n"
                        "## Ejemplos de training (min_calidad=4)\n"
                        f"{training_md}\n"
                    )

                    try:
                        from modules.memory_core import add_memory
                        add_memory(content=content, category="reflection", source="reflection", importance="low")
                        acciones_realizadas.append("Auto-auditoria nocturna guardada (reflection)")
                    except Exception:
                        acciones_realizadas.append("No pude guardar auto-auditoria nocturna")
                else:
                    acciones_realizadas.append("Auto-auditoria nocturna: sin actividad en 24h")
            except Exception:
                pass

        else:
            ciclo = "MADRUGADA"
            reporte.append(f"## Ciclo: {ciclo} (mantenimiento, decay, backup)\n")

            # 1. Sync FTS index
            try:
                fts_result = sync_fts_index()
                acciones_realizadas.append("FTS index sincronizado")
            except Exception:
                acciones_realizadas.append("No pude sincronizar FTS")

            # 2. Decay de salience
            try:
                salience = apply_salience_decay(decay_rate=0.03)
                acciones_realizadas.append("Salience decay aplicado")
            except Exception:
                acciones_realizadas.append("No pude aplicar salience decay")

            # 3. Decay emocional
            try:
                emo_decay = apply_emotional_decay()
                acciones_realizadas.append("Emotional decay aplicado")
            except Exception:
                acciones_realizadas.append("No pude aplicar emotional decay")

            # 4. Backup
            try:
                maybe_backup(reason="ciclo_vida_noche", force=True)
                acciones_realizadas.append("Backup de memorias realizado")
            except Exception:
                acciones_realizadas.append("No pude hacer backup")

            # 5. Process FTS retry queue (P2A)
            try:
                from modules.memory_smart import process_fts_queue
                fts_result = process_fts_queue(limit=200)
                if fts_result.get("processed", 0) > 0:
                    acciones_realizadas.append(f"FTS queue: {fts_result['succeeded']} OK, {fts_result['failed']} failed")
                else:
                    acciones_realizadas.append("FTS queue: sin pendientes")
            except Exception:
                acciones_realizadas.append("No pude procesar cola FTS")

        # Construir reporte final
        reporte.append("## Acciones Realizadas\n")
        for a in acciones_realizadas:
            reporte.append(f"- {a}")

        if sugerencias:
            reporte.append("\n## Sugerencias\n")
            for s in sugerencias:
                reporte.append(f"- {s}")

        reporte.append(f"\n---\n*Ciclo {ciclo} ejecutado a las {timestamp}*")

        return "\n".join(reporte)
    except Exception as e:
        return f"Error en ciclo de vida: {str(e)}"


# ============================================================
# N8N INTEGRATION TOOLS
# ============================================================

def trigger_n8n(webhook_path: str, data: dict = None, esperar_respuesta: bool = False) -> str:
    """
    Dispara un workflow en n8n enviando datos a un webhook.

    Args:
        webhook_path: Path del webhook (ej: 'codi-alerta', 'trading-orden')
        data: Datos a enviar (dict)
        esperar_respuesta: Si True, espera y retorna la respuesta de n8n
    """
    try:
        if not N8N_WEBHOOK_BASE:
            return "n8n no esta configurado (N8N_WEBHOOK_BASE no esta en .env)."
        # Validacion basica para evitar paths inesperados
        if not webhook_path or any((not ch.isalnum()) and ch not in ('_', '-') for ch in webhook_path) or len(webhook_path) > 80:
            return "Error: webhook_path invalido (solo A-Z a-z 0-9 _ -)"
        url = f"{N8N_WEBHOOK_BASE}/{webhook_path}"
        payload = data or {}
        payload['_from'] = 'codi-memory'
        payload['_timestamp'] = now_iso()
        timeout = 30 if esperar_respuesta else 5

        response = http_requests.post(url, json=payload, timeout=timeout, headers={'Content-Type': 'application/json'})

        if esperar_respuesta:
            try:
                return f"Respuesta de n8n: {response.json()}"
            except Exception:
                return f"Respuesta de n8n (texto): {response.text[:500]}"
        else:
            if response.status_code in [200, 201, 202]:
                return f"Webhook disparado: {webhook_path} - Status: {response.status_code}"
            else:
                return f"Error disparando webhook: {response.status_code} - {response.text[:200]}"
    except http_requests.exceptions.Timeout:
        return f"Timeout esperando respuesta de n8n (webhook: {webhook_path})"
    except Exception as e:
        return f"Error disparando n8n: {str(e)}"


def listar_webhooks_conocidos() -> str:
    """Lista los webhooks de n8n que conozco y puedo disparar."""
    webhooks = {
        "codi-alerta": "Enviar alertas generales a Hare",
        "trading-signal": "Enviar senal de trading para procesar",
        "backup-trigger": "Disparar backup de memorias",
        "reporte-diario": "Generar y enviar reporte diario",
    }
    resultado = "# WEBHOOKS N8N CONOCIDOS\n\n"
    base_display = N8N_WEBHOOK_BASE or "(no configurado)"
    resultado += f"Base URL: {base_display}\n\n"
    for path, desc in webhooks.items():
        resultado += f"- **{path}**: {desc}\n"
    resultado += "\n*Nota: Estos webhooks deben existir en n8n para funcionar.*"
    resultado += "\n*Usa trigger_n8n('nombre', {datos}) para disparar.*"
    return resultado


# ============================================================
# MCP TOOL REGISTRATION
# ============================================================

def register_tools(mcp):
    # Self-model
    mcp.tool()(reflect_on_self)
    mcp.tool()(assess_confidence)
    mcp.tool()(identify_knowledge_gaps)
    mcp.tool()(update_self_model)
    mcp.tool()(get_self_model_summary)
    # Workspace / Attention
    mcp.tool()(focus_attention)
    mcp.tool()(broadcast_to_workspace)
    mcp.tool()(get_workspace_state)
    mcp.tool()(apply_salience_decay)
    mcp.tool()(get_high_salience_memories)
    # Prediction
    mcp.tool()(predict_context)
    mcp.tool()(record_surprise)
    mcp.tool()(get_prediction_accuracy)
    mcp.tool()(update_beliefs)
    # Auto-learning & Audit
    mcp.tool()(auto_learn_from_session)
    mcp.tool()(audit_tools)
    mcp.tool()(rate_tool_usefulness)
    # PAD Emotional State
    mcp.tool()(set_emotional_state)
    mcp.tool()(get_emotional_state)
    mcp.tool()(update_mood_baseline)
    mcp.tool()(apply_emotional_decay)
    mcp.tool()(get_emotional_expression)
    # Emotional Memory
    mcp.tool()(add_memory_with_emotion)
    mcp.tool()(tag_memory_emotion)
    mcp.tool()(search_by_emotion)
    mcp.tool()(get_emotional_memories)
    mcp.tool()(emotional_focus_attention)
    # Consolidation
    mcp.tool()(consolidate_recent)
    mcp.tool()(find_connections)
    mcp.tool()(dream_consolidation)
    mcp.tool()(get_memory_connections)
    # System
    mcp.tool()(verificar_salud_memoria)
    mcp.tool()(despertar_codi)
    # Cognitive Patterns
    mcp.tool()(detectar_sorpresa)
    mcp.tool()(analizar_patron_trabajo)
    mcp.tool()(generar_curiosidad)
    # Lifecycle & Curiosity
    mcp.tool()(ciclo_vida)
    mcp.tool()(push_curiosidad)
    mcp.tool()(get_curiosidades)
    # N8N Integration
    mcp.tool()(trigger_n8n)
    mcp.tool()(listar_webhooks_conocidos)