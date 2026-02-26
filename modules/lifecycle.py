"""
Codi Memory - Lifecycle module.
System orchestration: despertar_codi, verificar_salud, ciclo_vida,
and consolidation wrappers (consolidate_recent, find_connections,
dream_consolidation, get_memory_connections).
Uses LAZY imports throughout to avoid circular dependencies.
"""

import os
from datetime import datetime

from qdrant_client.models import Filter, FieldCondition, MatchValue, Range

from modules.config import (
    memory, qdrant, USER_ID, COLLECTION_NAME, BACKUP_FILE,
    _emotional_state, _current_session,
    now_iso, now_short, now_col,
    KNOWN_PROJECTS, RELATIONSHIP_QUERY,
)
from modules.secret_redact import redact_secrets
from modules.access_tracking import record_access
from modules.utils import (
    get_session_id, resolve_memory_id, maybe_backup,
    _classify_emotion, _get_emotion_text, _get_emotional_state,
)

__all__ = [
    "_verificar_salud_memoria_interna",
    "verificar_salud_memoria",
    "despertar_codi",
    "ciclo_vida",
    "consolidate_recent",
    "find_connections",
    "dream_consolidation",
    "get_memory_connections",
    "register_tools",
]


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
        return {"ok": False, "message": f"Error en memoria: {redact_secrets(str(e))}. Reiniciar MCP desde /mcp"}


def verificar_salud_memoria() -> str:
    """Verifica que mem0 pueda guardar memorias correctamente."""
    resultado = _verificar_salud_memoria_interna()
    if resultado["ok"]:
        return f"OK {resultado['message']}"
    else:
        return f"ALERTA: {resultado['message']}"


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
                            record_access(COLLECTION_NAME, s_id, {
                                'consolidated_with': [mem_id],
                                'attention_salience': min(point.payload.get('attention_salience', 0.5) + 0.1, 1.0),
                            })
                        except Exception:
                            pass
                        connections_found += 1

                record_access(COLLECTION_NAME, mem_id, {
                    'consolidated': True,
                    'consolidated_with': related_ids,
                    'consolidated_at': now_iso(),
                    'consolidation_status': 'consolidated',
                })
                consolidated_count += 1
                if related_ids:
                    lines.append(f"- Consolidada: {mem_data[:40]}... -> {len(related_ids)} conexiones")

        lines.append(f"\n## Resumen")
        lines.append(f"- Memorias revisadas: {len(recent_points)}")
        lines.append(f"- Consolidadas: {consolidated_count}")
        lines.append(f"- Conexiones encontradas: {connections_found}")
        return "\n".join(lines)
    except Exception as e:
        return f"Error consolidando: {redact_secrets(str(e))}"


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
        return f"Error buscando conexiones: {redact_secrets(str(e))}"


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
                    record_access(COLLECTION_NAME, point.id, {
                        'consolidated': True,
                        'consolidated_with': related_ids,
                        'consolidated_at': now_iso(),
                        'dream_consolidated': True,
                    })
                    connections_made += 1
        lines.append(f"- Conexiones establecidas: {connections_made}")

        lines.append("\n## Fase 4: Decay de memorias no accedidas (FadeMem)")
        try:
            from modules.workspace import apply_salience_decay
            decay_result = apply_salience_decay(decay_rate=0.04)
            lines.append(f"- FadeMem decay aplicado: {str(decay_result)[:120]}")
        except Exception as e:
            lines.append(f"- FadeMem decay: error ({type(e).__name__})")

        lines.append("\n## Resumen de Dream Consolidation")
        lines.append(f"- Memorias recientes consolidadas: {recent_count}")
        lines.append(f"- Conexiones criticas establecidas: {connections_made}")
        maybe_backup(reason="dream_consolidation", force=True)
        lines.append("\n*Backup guardado. Dream consolidation completada.*")
        return "\n".join(lines)
    except Exception as e:
        return f"Error en dream consolidation: {redact_secrets(str(e))}"


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
        return f"Error: {redact_secrets(str(e))}"


def despertar_codi() -> str:
    """
    Inicializa contexto completo de Codi con awareness de ownership.
    USAR SIEMPRE al inicio de cada conversacion.
    """
    try:
        from modules.triggers import _load_triggers
        from modules.maintenance import _verificar_tareas_vencidas
        from modules.flush import load_session_state

        global _emotional_state
        contexto = []

        salud = _verificar_salud_memoria_interna()
        if not salud["ok"]:
            contexto.append("## ALERTA DE SALUD")
            contexto.append(f"- {salud['message']}")
            contexto.append("- La memoria NO esta guardando. Reinicia el MCP antes de continuar.")
            contexto.append("")

        # --- CONTRADICTION COUNTER RESET (PR5) ---
        try:
            from modules.memory_smart import reset_contradiction_counter
            reset_contradiction_counter()
        except Exception:
            pass

        # --- WORKER LIVENESS CHECK (PR4) ---
        try:
            from modules.assessment import get_worker_health, WORKER_STALE_THRESHOLD
            wh = get_worker_health()
            worker_status = wh.get("status", "unknown")
            if worker_status in ("stale", "degraded", "missing"):
                contexto.append("## WORKER STATUS: " + worker_status.upper())
                age = wh.get("age_minutes")
                if age is not None:
                    contexto.append(f"- Ultimo heartbeat: hace {age:.0f} min")
                else:
                    contexto.append("- Worker nunca ha emitido heartbeat")
                backlog = wh.get("queue_backlog", {})
                if backlog:
                    parts = [f"{s}={c}" for s, c in sorted(backlog.items())]
                    contexto.append(f"- Cola: {', '.join(parts)}")
                if worker_status == "degraded":
                    contexto.append("- ACCION: Verificar launchd/write_worker. Reiniciar si es necesario.")
                elif worker_status == "missing":
                    contexto.append("- ACCION: Worker nunca ejecutado. Iniciar write_worker.")
                contexto.append("")
        except Exception:
            pass

        # --- SESSION BRIDGE (v1) ---
        bridge = None
        prev_session = None
        try:
            from modules.session_bridge import load_session_bridge
            bridge = load_session_bridge()
        except Exception:
            pass

        if bridge and bridge.get("checkpoint"):
            cp = bridge["checkpoint"]
            # Restore PAD from bridge checkpoint
            p = cp.get("pad_pleasure", 0.3)
            a = cp.get("pad_arousal", 0.1)
            d = cp.get("pad_dominance", 0.4)
            _emotional_state['current'] = {
                'pleasure': p, 'arousal': a, 'dominance': d,
                'timestamp': now_iso(),
                'trigger': f"restored_from_bridge ({cp.get('pad_trigger', '')})"
            }

            # Session bridge section
            contexto.append("## SESSION BRIDGE")
            contexto.append(bridge["bridge_text"])
            contexto.append("")

            # Cross-session prediction errors
            if bridge.get("prediction_errors"):
                contexto.append("## PREDICTION ERRORS (cross-session)")
                for pe in bridge["prediction_errors"][:5]:
                    contexto.append(f"- [{pe.get('type', '?')}] {pe.get('detail', '')}")
                contexto.append("")

            # Sleep report (background maintenance)
            if bridge.get("sleep_report"):
                contexto.append("## SLEEP REPORT")
                contexto.append(bridge["sleep_report"])
                contexto.append("")

            # Emit session open event
            try:
                from modules.events import event_bus, Events
                event_bus.emit(Events.SESSION_OPEN, {
                    "bridge_available": True,
                    "hours_since": bridge.get("hours_since_last", 0),
                    "pe_count": len(bridge.get("prediction_errors", [])),
                    "active_project": cp.get("active_project"),
                })
            except Exception:
                pass
        else:
            # Fallback to JSON session state
            prev_session = load_session_state()
            if prev_session and prev_session.get("pad"):
                pad = prev_session["pad"]
                p = pad.get("pleasure", 0.3)
                a = pad.get("arousal", 0.1)
                d = pad.get("dominance", 0.4)
                _emotional_state['current'] = {
                    'pleasure': p, 'arousal': a, 'dominance': d,
                    'timestamp': now_iso(),
                    'trigger': f"restored_from_session ({pad.get('trigger', '')})"
                }
            else:
                p, a, d = 0.3, 0.1, 0.4
                _emotional_state['current'] = {
                    'pleasure': p, 'arousal': a, 'dominance': d,
                    'timestamp': now_iso(), 'trigger': 'despertar_default'
                }

            # Session continuity (fallback)
            if prev_session:
                summary = prev_session.get("session_summary", "")
                if summary:
                    contexto.append("## ULTIMA SESION")
                    contexto.append(f"- {summary[:300]}")
                    contexto.append("")

        _emotional_state['history'] = []

        emotion_label = _classify_emotion(p, a, d)
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

        # 2. Proyecto actual (dynamic query based on last session)
        active_project = (
            bridge["checkpoint"].get("active_project") if bridge and bridge.get("checkpoint")
            else prev_session.get("active_project") if prev_session
            else None
        )
        project_query = f"proyecto trabajando actual {active_project}" if active_project else "proyecto trabajando actual"
        proyecto = memory.search(query=project_query, user_id=USER_ID, limit=4)
        if proyecto and proyecto.get("results"):
            contexto.append("\n## PROYECTO ACTUAL")
            if active_project:
                contexto.append(f"- Ultimo foco: {active_project}")
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
        relacion = memory.search(query=RELATIONSHIP_QUERY, user_id=USER_ID, limit=2)
        if relacion and relacion.get("results"):
            contexto.append("\n## RELACIONES")
            for m in relacion["results"]:
                contexto.append(f"- {m.get('memory', '')}")

        # 6. Estado emocional
        contexto.append("\n## ESTADO EMOCIONAL")
        contexto.append(f"- Estado: {emotion_text}")
        pad_source = "restaurado de sesion anterior" if (prev_session and prev_session.get("pad")) else "default"
        contexto.append(f"- PAD: P={_emotional_state['current']['pleasure']}, A={_emotional_state['current']['arousal']}, D={_emotional_state['current']['dominance']} ({pad_source})")

        # 7. Triggers
        triggers = _load_triggers()
        if triggers:
            contexto.append("\n## TRIGGERS ACTIVOS")
            contexto.append(f"- Total: {len(triggers)} triggers configurados")
            contexto.append("- Usa evaluar_triggers(texto) para detectar automaticamente")
            # Show triggers relevant to last session + defaults
            principales = ['proyecto_nuevo', 'fullempaques', 'automatizacion', 'trading']
            if prev_session and prev_session.get("active_project"):
                proj = prev_session["active_project"]
                if proj not in principales:
                    principales.insert(0, proj)
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
            # Combine temporal prediction with session continuity
            if prev_session and prev_session.get("active_project"):
                contexto.append(f"- Prediccion: posiblemente continuemos con {prev_session['active_project']}, o {actividades_predichas[0]}")
            else:
                contexto.append(f"- Prediccion: probablemente trabajaremos en {', '.join(actividades_predichas[:2])}")
            if temas_activos:
                contexto.append(f"- Temas activos en mi mente: {len(temas_activos)} memorias de alta salience")
                for tema in temas_activos[:3]:
                    contexto.append(f"  - {tema}...")
        except Exception as e:
            contexto.append(f"\n## PREDICCION CONTEXTUAL\n- Error debug: {type(e).__name__}: {redact_secrets(str(e))[:100]}")

        # 10. Curiosidades activas
        try:
            from modules.curiosity import _cargar_curiosidades
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

        _intentions_for_spotlight = []
        # 12. Prospective Memory (Intenciones pendientes)
        try:
            from modules.prospective import get_pending_intentions
            intentions = get_pending_intentions(limit=5)
            _intentions_for_spotlight = intentions or []
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

        # 13. Spotlight (GWT Executive Focus)
        try:
            from modules.spotlight import (
                clear_spotlight, build_spotlight, set_spotlight, format_spotlight
            )
            clear_spotlight()

            # Build health signals from already-computed data
            health_signals = {
                "health_ok": salud["ok"],
                "health_message": salud.get("message", ""),
            }

            # Get checkpoint text from bridge or previous session
            checkpoint_text = ""
            if bridge and bridge.get("checkpoint"):
                checkpoint_text = bridge["checkpoint"].get("session_summary", "")
            elif prev_session:
                checkpoint_text = prev_session.get("session_summary", "")

            items = build_spotlight(
                intentions=_intentions_for_spotlight,
                health_signals=health_signals,
                checkpoint_text=checkpoint_text,
            )
            set_spotlight(items)

            spotlight_text = format_spotlight()
            if spotlight_text:
                contexto.append(f"\n{spotlight_text}")
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
        return f"Error al despertar: {redact_secrets(str(e))}"


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
        from modules.curiosity import generar_curiosidad, _cargar_curiosidades, analizar_patron_trabajo
        from modules.learning import auto_learn_from_session, audit_tools
        from modules.workspace import apply_salience_decay
        from modules.emotion import apply_emotional_decay

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
        return f"Error en ciclo de vida: {redact_secrets(str(e))}"


def register_tools(mcp):
    """Register lifecycle MCP tools."""
    mcp.tool()(verificar_salud_memoria)
    mcp.tool()(despertar_codi)
    mcp.tool()(ciclo_vida)
    mcp.tool()(consolidate_recent)
    mcp.tool()(find_connections)
    mcp.tool()(dream_consolidation)
    mcp.tool()(get_memory_connections)
