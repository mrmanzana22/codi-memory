"""
Codi Memory - Lifecycle module.
System orchestration: despertar_codi, verificar_salud, ciclo_vida,
and consolidation wrappers (consolidate_recent, find_connections,
dream_consolidation, get_memory_connections).
Uses LAZY imports throughout to avoid circular dependencies.
"""

import os
from datetime import datetime

from modules.config import (
    USER_ID, COLLECTION_NAME, BACKUP_FILE,
    _emotional_state, _current_session,
    now_iso, now_short, now_col,
    KNOWN_PROJECTS, RELATIONSHIP_QUERY,
)
from modules.secret_redact import redact_secrets
from modules.pg_store import pg
from modules.memory_smart import search_with_fts_content
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
        collection_info = pg.count(is_semantic=False)
        if not collection_info:
            return {"ok": False, "message": "pg_store no responde. Verificar servidor PostgreSQL."}
        search_result = pg.search("test conexion memoria", limit=1, is_semantic=False)
        if search_result is None:
            return {"ok": False, "message": "pg_store no responde a busquedas. Reiniciar MCP desde /mcp"}
        total_points = collection_info.points_count
        return {"ok": True, "message": f"Memoria funcionando. {total_points} memorias en PostgreSQL."}
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
        cursor = None
        recent_points = []
        while True:
            page_points, cursor = pg.scroll(
                filters={"metadata_key": {"key": "temporal_session_id", "value": session_id}},
                limit=50, cursor=cursor, is_semantic=False
            )
            if not page_points:
                break
            recent_points.extend(page_points)
            if cursor is None:
                break
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

            similar = pg.search(mem_data, limit=5, is_semantic=False)
            if similar and similar.get('results'):
                related_ids = []
                for s in similar['results']:
                    s_id = s.get('id')
                    score = s.get('score', 0)
                    if s_id != mem_id and score >= 0.7:
                        related_ids.append(s_id)
                        try:
                            existing_links = s.get('payload', {}).get('consolidated_with', [])
                            if not isinstance(existing_links, list):
                                existing_links = []
                            merged_links = existing_links if mem_id in existing_links else existing_links + [mem_id]
                            pg.update_payload(s_id, {
                                'consolidated_with': merged_links,
                                'attention_salience': min(point.payload.get('attention_salience', 0.5) + 0.1, 1.0),
                            })
                        except Exception:
                            pass
                        connections_found += 1

                pg.update_payload(mem_id, {
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
            points = pg.get_by_ids([full_id])
            if not points:
                return f"No encontre memoria con ID {full_id}"
            query = points[0].payload.get('data', '')
            source_info = f"Memoria: {query[:50]}..."
        else:
            source_info = f"Tema: {query}"

        results = pg.search(query, limit=15, is_semantic=False)
        if not results or not results.get('results'):
            return f"No encontre conexiones para: {source_info}"

        connections = []
        for r in results['results']:
            r_id = r.get('id')
            score = r.get('score', 0)
            if score >= threshold and r_id != full_id:
                try:
                    r_points = pg.get_by_ids([r_id])
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
        recent, _ = pg.scroll(
            filters={"metadata_key": {"key": "temporal_session_id", "value": session_id}},
            limit=100, is_semantic=False
        )
        recent_count = len(recent) if recent else 0
        lines.append(f"- Memorias de esta sesion: {recent_count}")

        lines.append("\n## Fase 2: Priorizacion por importancia")
        high_importance, _ = pg.scroll(
            filters={"importance": "critical"},
            limit=20, is_semantic=False
        )
        critical_unconsolidated = [p for p in (high_importance or []) if not p.payload.get('consolidated', False)]
        lines.append(f"- Memorias criticas sin consolidar: {len(critical_unconsolidated)}")

        lines.append("\n## Fase 3: Tejiendo conexiones entre memorias criticas")
        connections_made = 0
        for point in critical_unconsolidated[:10]:
            mem_data = point.payload.get('data', '')
            similar = pg.search(mem_data, limit=5, is_semantic=False)
            if similar and similar.get('results'):
                related_ids = [s.get('id') for s in similar['results'] if s.get('id') != point.id and s.get('score', 0) >= 0.6]
                if related_ids:
                    pg.update_payload(point.id, {
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
        points = pg.get_by_ids([full_id])
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
                    conn_points = pg.get_by_ids([conn_id])
                    if conn_points:
                        lines.append(f"- [{conn_id[:8]}] {conn_points[0].payload.get('data', 'N/A')[:60]}...")
                except Exception:
                    lines.append(f"- [{conn_id[:8]}] (no disponible)")
        return "\n".join(lines)
    except Exception as e:
        return f"Error: {redact_secrets(str(e))}"


# Module-level storage for wake JSON (accessible by other modules)
_last_wake_json = {}


def despertar_codi(verbose: bool = False) -> str:
    """
    Executive briefing for session start.
    Gathers data, executes side effects, returns compact operational brief.
    Use verbose=True for full debug dump.
    """
    global _emotional_state, _last_wake_json

    try:
        from modules.maintenance import _verificar_tareas_vencidas
        from modules.flush import load_session_state
        from modules.goals import get_context_goals
        from modules.prospective import get_pending_intentions

        # ── SIDE EFFECTS (always execute) ──
        try:
            from modules.memory_smart import reset_contradiction_counter
            reset_contradiction_counter()
        except Exception:
            pass

        event_handler_error = None
        try:
            from modules.active_inference_integration import register_event_handlers
            register_event_handlers()
        except Exception as e:
            event_handler_error = redact_secrets(str(e))

        # ── DATA GATHERING ──

        # Health
        salud = _verificar_salud_memoria_interna()

        # Worker liveness
        worker_status = "unknown"
        worker_detail = ""
        try:
            from modules.assessment import get_worker_health
            wh = get_worker_health()
            worker_status = wh.get("status", "unknown")
            age = wh.get("age_minutes")
            if age is not None:
                worker_detail = f"heartbeat {age:.0f}min ago"
            else:
                worker_detail = "no heartbeat"
            backlog = wh.get("queue_backlog", {})
            if backlog:
                worker_detail += f", queue: {sum(backlog.values())}"
        except Exception:
            pass

        # Session bridge + PAD restore
        bridge = None
        prev_session = None
        try:
            from modules.session_bridge import load_session_bridge
            bridge = load_session_bridge()
        except Exception:
            pass

        if bridge and bridge.get("checkpoint"):
            cp = bridge["checkpoint"]
            pad_p = cp.get("pad_pleasure", 0.3)
            pad_a = cp.get("pad_arousal", 0.1)
            pad_d = cp.get("pad_dominance", 0.4)
            _emotional_state['current'] = {
                'pleasure': pad_p, 'arousal': pad_a, 'dominance': pad_d,
                'timestamp': now_iso(),
                'trigger': f"restored_from_bridge ({cp.get('pad_trigger', '')})"
            }
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
            prev_session = load_session_state()
            if prev_session and prev_session.get("pad"):
                pad = prev_session["pad"]
                pad_p = pad.get("pleasure", 0.3)
                pad_a = pad.get("arousal", 0.1)
                pad_d = pad.get("dominance", 0.4)
                _emotional_state['current'] = {
                    'pleasure': pad_p, 'arousal': pad_a, 'dominance': pad_d,
                    'timestamp': now_iso(),
                    'trigger': f"restored_from_session ({pad.get('trigger', '')})"
                }
            else:
                pad_p, pad_a, pad_d = 0.3, 0.1, 0.4
                _emotional_state['current'] = {
                    'pleasure': pad_p, 'arousal': pad_a, 'dominance': pad_d,
                    'timestamp': now_iso(), 'trigger': 'despertar_default'
                }

        _emotional_state['history'] = []

        # Active project detection (LAW: explicit project wins)
        active_project = None
        if bridge and bridge.get("checkpoint"):
            active_project = bridge["checkpoint"].get("active_project")
        elif prev_session:
            active_project = prev_session.get("active_project")

        # Goals
        gctx = get_context_goals(limit=5)
        goals = gctx.get("goals", []) if gctx else []

        # Intentions
        intentions = get_pending_intentions(limit=5) or []

        # Spotlight (GNW side effect — build but don't render separately)
        try:
            from modules.spotlight import (
                clear_spotlight, build_spotlight, set_spotlight
            )
            clear_spotlight()
            health_signals = {
                "health_ok": salud["ok"],
                "health_message": salud.get("message", ""),
            }
            checkpoint_text = ""
            if bridge and bridge.get("checkpoint"):
                checkpoint_text = bridge["checkpoint"].get("session_summary", "")
            elif prev_session:
                checkpoint_text = prev_session.get("session_summary", "")
            items = build_spotlight(
                intentions=intentions,
                health_signals=health_signals,
                checkpoint_text=checkpoint_text,
            )
            set_spotlight(items)
        except Exception:
            pass

        # Maintenance (only critically overdue → incidents)
        maint_overdue = []
        try:
            vencidas = _verificar_tareas_vencidas()
            maint_overdue = [
                v for v in vencidas
                if v.get('dias_vencido', 0) > 14 or v.get('estado') == 'nunca_hecho'
            ]
        except Exception:
            pass

        # Identity (for deep_context)
        identity_mems = []
        try:
            id_points, _ = pg.scroll(
                filters={"importance": "critical"}, limit=5, is_semantic=False
            )
            for pt in (id_points or []):
                data = pt.payload.get('data', '')
                source = pt.payload.get('ownership_source', '')
                identity_mems.append({"text": data, "source": source})
        except Exception:
            pass

        # ── BUILD INCIDENTS ──

        incidents = []
        if not salud["ok"]:
            incidents.append({
                "type": "health", "severity": "critical",
                "detail": salud["message"],
                "action": "Reiniciar MCP",
            })
        if worker_status in ("stale", "degraded", "missing"):
            incidents.append({
                "type": "worker",
                "severity": "high" if worker_status == "missing" else "medium",
                "detail": f"Worker {worker_status} ({worker_detail})",
                "action": "Verificar launchd/write_worker",
            })
        if event_handler_error:
            incidents.append({
                "type": "event_handlers",
                "severity": "high",
                "detail": f"Event handler registration failed: {event_handler_error}",
                "action": "Verificar modules.active_inference_integration.register_event_handlers",
            })
        if bridge and bridge.get("hours_since_last", 0) > 6:
            incidents.append({
                "type": "bridge_stale", "severity": "low",
                "detail": f"Ultimo checkpoint hace {bridge['hours_since_last']:.1f}h",
                "action": "Sesion larga sin flush",
            })
        for m in maint_overdue[:2]:
            nombre = m['nombre']
            if m['estado'] == 'nunca_hecho':
                detail = f"{nombre}: nunca hecho"
            else:
                detail = f"{nombre}: vencido {m['dias_vencido']}d"
            incidents.append({
                "type": "maintenance", "severity": "low",
                "detail": detail, "action": "Ejecutar mantenimiento",
            })
        # PAD (0,0,0) is normal when daemon has no active emotional loop.
        # Defaults are (0.3, 0.1, 0.4). Only flag if restored from bridge
        # and then zeroed — unlikely in practice. Removed as noise.

        # ── DETERMINE STATE ──

        has_critical = any(i["severity"] == "critical" for i in incidents)
        has_high = any(i["severity"] == "high" for i in incidents)
        state = "critical" if has_critical else "degraded" if has_high else "estable"

        # ── BRIEF PRIORITIES (from goals) ──

        priorities = []
        for g in goals[:3]:
            ns = g.get("goal_next_step")
            if ns:
                priorities.append({
                    "goal": g["title"], "level": g.get("level", "task"),
                    "action": ns,
                })

        # ── ACTIVE PROJECT CONTEXT ──

        project_goal = None
        if active_project and goals:
            for g in goals:
                if active_project.lower() in g.get("title", "").lower():
                    project_goal = g
                    break
            if not project_goal:
                project_goal = goals[0]
        elif goals:
            project_goal = goals[0]
            if not active_project:
                active_project = project_goal.get("title", "sin proyecto")

        # ── REAL PENDING (intentions only — goals already in priorities) ──

        real_pending = []
        for i in intentions:
            real_pending.append({"source": "intention", "text": i["action"]})

        # ── DEEP CONTEXT (stored, not rendered in brief) ──

        cross_pe = []
        if bridge and bridge.get("prediction_errors"):
            cross_pe = bridge["prediction_errors"][:5]

        emotion_label = _classify_emotion(pad_p, pad_a, pad_d)

        deep_context = {
            "identity": identity_mems,
            "emotional_state": {
                "pad": {"p": pad_p, "a": pad_a, "d": pad_d},
                "label": emotion_label,
            },
            "cross_session_pe": cross_pe,
            "sleep_report": bridge.get("sleep_report") if bridge else None,
            "session_bridge": bridge.get("bridge_text") if bridge else None,
        }

        # ── ASSEMBLE WAKE JSON ──

        wake_json = {
            "brief": {
                "project": active_project or "sin proyecto",
                "state": state,
                "priorities": priorities,
                "risk": incidents[0]["detail"] if incidents else None,
            },
            "incidents": incidents[:5],
            "active_project": {
                "name": active_project,
                "goal": project_goal.get("title") if project_goal else None,
                "what": project_goal.get("goal_what") if project_goal else None,
                "why": project_goal.get("goal_why") if project_goal else None,
                "last_state": project_goal.get("goal_last_state") if project_goal else None,
                "next_step": project_goal.get("goal_next_step") if project_goal else None,
            },
            "real_pending": real_pending[:5],
            "deep_context": deep_context,
        }

        _last_wake_json = wake_json

        # ── RENDER ──

        if verbose:
            return _render_wake_verbose(wake_json)
        return _render_wake_text(wake_json)

    except Exception as e:
        return f"Error al despertar: {redact_secrets(str(e))}"


def _render_wake_text(w: dict) -> str:
    """Compact operational brief: 8-15 dense lines."""
    lines = ["# CODI WAKE BRIEF"]

    b = w["brief"]
    lines.append(f"Proyecto: {b['project']} | Estado: {b['state']}")

    # Priorities
    if b["priorities"]:
        lines.append("Prioridades:")
        for idx, p in enumerate(b["priorities"][:3], 1):
            lines.append(f"  {idx}. [{p['level']}] {p['goal']}: {p['action'][:120]}")

    # Risk
    if b.get("risk"):
        lines.append(f"Riesgo: {b['risk']}")

    # Incidents
    incs = w.get("incidents", [])
    if incs:
        lines.append(f"Incidentes ({len(incs)}):")
        for inc in incs[:3]:
            lines.append(f"  [{inc['severity']}] {inc['detail']} -> {inc['action']}")

    # Active project context
    ap = w.get("active_project", {})
    if ap.get("what"):
        lines.append(f"[{ap['name']}] {ap['what']}")
        if ap.get("next_step"):
            lines.append(f"  Next: {ap['next_step'][:150]}")

    # Pending (intentions only — goals already shown as priorities)
    pending = w.get("real_pending", [])
    if pending:
        lines.append(f"Pendientes ({len(pending)}):")
        for p in pending[:3]:
            lines.append(f"  - {p['text']}")

    return "\n".join(lines)


def _render_wake_verbose(w: dict) -> str:
    """Full debug dump for diagnostics."""
    import json as _json
    lines = [_render_wake_text(w)]
    lines.append("\n--- DEBUG DETAIL ---")

    dc = w.get("deep_context", {})

    # Identity
    identity = dc.get("identity", [])
    if identity:
        lines.append("\n## IDENTITY")
        for m in identity:
            marker = "[vivi]" if m.get("source") == "experienced" else "[told]"
            lines.append(f"  {marker} {m['text']}")

    # Emotional state
    es = dc.get("emotional_state", {})
    pad = es.get("pad", {})
    lines.append(
        f"\n## EMOTIONAL STATE: {es.get('label', '?')}"
        f" (P={pad.get('p')}, A={pad.get('a')}, D={pad.get('d')})"
    )

    # Cross-session PEs
    pes = dc.get("cross_session_pe", [])
    if pes:
        lines.append("\n## CROSS-SESSION PE")
        for pe in pes:
            lines.append(f"  [{pe.get('type', '?')}] {pe.get('detail', '')}")

    # Sleep report
    sr = dc.get("sleep_report")
    if sr:
        lines.append(f"\n## SLEEP REPORT\n{sr}")

    # Bridge
    bt = dc.get("session_bridge")
    if bt:
        lines.append(f"\n## SESSION BRIDGE\n{bt}")

    # Raw JSON (truncated)
    lines.append(
        f"\n## RAW JSON\n```json\n"
        f"{_json.dumps(w, indent=2, default=str)[:3000]}\n```"
    )

    return "\n".join(lines)


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
                sugerencias.append("Revisar conexion a PostgreSQL - la memoria no esta bien")

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
