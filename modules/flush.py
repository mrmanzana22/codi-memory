import logging
import os
import json
from datetime import datetime
from modules.config import (
    USER_ID, COLLECTION_NAME, MARKDOWN_DIR, JOURNAL_DIR,
    now_short, now_iso, now_col, SESSION_STATE_FILE, _emotional_state,
)
from modules.pg_store import pg
from modules.secret_redact import redact_secrets

_logger = logging.getLogger(__name__)
from modules.utils import enrich_with_ownership, maybe_backup, export_memories_to_files, append_to_daily_journal
from modules.memory_smart import add_memory_smart, process_fts_queue


def _checkpoint_memoria_sync(momento: str, que_paso: str, por_que_importa: str) -> str:
    """Synchronous checkpoint implementation (original pipeline)."""
    timestamp = now_short()
    contenido = f"[{momento.upper()}] {que_paso} | Importancia: {por_que_importa} | Fecha: {timestamp}"

    importance_map = {
        'momento_personal': 'critical',
        'decision': 'high',
        'error_resuelto': 'high',
        'aprendizaje': 'medium',
        'tarea_completada': 'medium',
        'patron': 'medium'
    }

    valence_map = {
        'momento_personal': 'positive',
        'tarea_completada': 'positive',
        'error_resuelto': 'mixed',
        'decision': 'neutral',
        'aprendizaje': 'positive',
        'patron': 'neutral'
    }

    result = pg.add(
        content=contenido,
        category="checkpoint",
        source="experienced",
        importance=importance_map.get(momento, 'medium'),
    )

    mem_id = result.get("id") if isinstance(result, dict) else None
    if not mem_id:
        raise RuntimeError(f"pg.add() returned no id for checkpoint: {result!r}")

    if mem_id:
            enrich_with_ownership(
                memory_id=mem_id,
                category="checkpoint",
                content=contenido,
                source="experienced",
                importance=importance_map.get(momento, 'medium'),
                emotional_weight=0.7,
                emotional_valence=valence_map.get(momento, 'neutral')
            )
            # FTS indexing (same pattern as add_memory_smart)
            try:
                from modules.memory_smart import index_memory_fts
                index_memory_fts(
                    mem_id, contenido, "checkpoint",
                    "experienced", importance_map.get(momento, 'medium')
                )
            except Exception:
                pass
            # Proposal #55: Auto-connect to graph neighbors (was missing here)
            try:
                from modules.memory_smart import _auto_connect_neighbors
                _auto_connect_neighbors(mem_id, contenido)
            except Exception:
                pass

    maybe_backup(reason="checkpoint", force=True)

    try:
        process_fts_queue(limit=50)
    except Exception:
        pass

    try:
        append_to_daily_journal(momento, que_paso, por_que_importa)
    except Exception:
        pass

    return f"Checkpoint guardado: {momento} - {que_paso[:50]}..."


def _checkpoint_memoria(momento: str, que_paso: str, por_que_importa: str) -> str:
    """
    Guarda un checkpoint con ownership automatico.
    Los checkpoints siempre son source=experienced, importancia alta.

    Supports 3 write modes via CODI_WRITE_MODE env var:
      - sync (default): synchronous pipeline
      - shadow: sync + enqueue for validation
      - async: enqueue + immediate ACK
    """
    from modules.interface import _get_write_mode
    write_mode = _get_write_mode()

    try:
        if write_mode == "async":
            import json as _json
            from modules.write_queue import enqueue_write_job, compute_dedupe_key
            dedupe_key = compute_dedupe_key("checkpoint_memoria", f"{momento}|{que_paso}")
            enqueue_result = enqueue_write_job(
                kind="checkpoint_memoria",
                payload={"momento": momento, "que_paso": que_paso, "por_que_importa": por_que_importa},
                priority=2,  # checkpoints are high priority
                dedupe_key=dedupe_key,
            )
            preview = que_paso[:120] + "..." if len(que_paso) > 120 else que_paso
            return _json.dumps({
                "action": "queued",
                "mode": write_mode,
                "job_id": enqueue_result["job_id"],
                "kind": "checkpoint_memoria",
                "preview": preview,
                "tags": [momento, "high"],
                "eta_seconds": 30,
                "dedupe_hit": enqueue_result["dedupe_hit"],
                "check": f"write_status('{enqueue_result['job_id']}')",
                "cancel": f"cancel_write('{enqueue_result['job_id']}')",
            }, ensure_ascii=False)

        # Sync execution
        result_str = _checkpoint_memoria_sync(momento, que_paso, por_que_importa)

        # Spotlight partial update from checkpoint text
        try:
            from modules.spotlight import get_spotlight, update_spotlight_from_text, set_spotlight
            combined_text = f"{momento}: {que_paso}. {por_que_importa}"
            updated = update_spotlight_from_text(get_spotlight(), combined_text, "checkpoint")
            set_spotlight(updated)
        except Exception:
            pass

        if write_mode == "shadow":
            try:
                from modules.write_queue import enqueue_write_job, compute_dedupe_key
                from modules.dual_compare import record_sync_result, compute_request_fingerprint
                from modules.tracing import get_trace_id, new_trace_id
                trace_id = get_trace_id()
                if not trace_id:
                    trace_id = new_trace_id()
                content_for_fp = f"{momento}|{que_paso}"
                dedupe_key = compute_dedupe_key("checkpoint_memoria", content_for_fp)
                fingerprint = compute_request_fingerprint("checkpoint_memoria", content_for_fp, trace_id)
                enqueue_result = enqueue_write_job(
                    kind="checkpoint_memoria",
                    payload={"momento": momento, "que_paso": que_paso, "por_que_importa": por_que_importa},
                    dedupe_key=dedupe_key,
                )
                record_sync_result(
                    fingerprint=fingerprint,
                    trace_id=trace_id,
                    kind="checkpoint_memoria",
                    raw_output=result_str,
                    async_job_id=enqueue_result["job_id"],
                    async_status=enqueue_result["status"],
                )
            except Exception:
                pass

        return result_str
    except Exception as e:
        return f"Error guardando checkpoint: {redact_secrets(str(e))}"


def _save_session_state(resumen: str = "") -> bool:
    """Persist session state to disk so despertar_codi can restore it.

    Saves: emotional PAD, active project, WM topics, session summary.
    Returns True on success.
    """
    try:
        # Current emotional state
        current_pad = _emotional_state.get('current', {})

        # Active project: infer from working memory topics (SQLite-backed)
        active_project = None
        wm_topics = []
        try:
            from modules.working_memory import wm_get_active_topics
            topics = wm_get_active_topics(limit=10)
            for topic in topics:
                if topic and topic not in wm_topics:
                    wm_topics.append(topic)
                if not active_project and topic not in ('metamemory', 'contradiction', 'general', ''):
                    active_project = topic
        except Exception:
            pass

        state = {
            "timestamp": now_iso(),
            "pad": {
                "pleasure": current_pad.get('pleasure', 0.0),
                "arousal": current_pad.get('arousal', 0.0),
                "dominance": current_pad.get('dominance', 0.0),
                "trigger": current_pad.get('trigger', ''),
            },
            "active_project": active_project,
            "wm_topics": wm_topics[:10],
            "session_summary": resumen[:500] if resumen else "",
        }

        with open(SESSION_STATE_FILE, 'w') as f:
            json.dump(state, f, indent=2, ensure_ascii=False)
        return True
    except Exception as e:
        _logger.error("Error saving session state: %s", redact_secrets(str(e)))
        return False


def load_session_state() -> dict | None:
    """Load previous session state if it exists and is fresh enough.

    Returns dict with pad, active_project, wm_topics, session_summary
    or None if no state / expired.
    """
    try:
        from modules.config import SESSION_STATE_MAX_AGE_HOURS
        if not os.path.exists(SESSION_STATE_FILE):
            return None

        with open(SESSION_STATE_FILE, 'r') as f:
            state = json.load(f)

        # Check freshness
        saved_ts = state.get("timestamp", "")
        if saved_ts:
            saved_dt = datetime.fromisoformat(saved_ts)
            age_hours = (now_col() - saved_dt).total_seconds() / 3600
            if age_hours > SESSION_STATE_MAX_AGE_HOURS:
                return None

        return state
    except Exception:
        return None


def _flush_session(resumen: str, decisiones: str = "", errores: str = "",
                   aprendizajes: str = "") -> str:
    """
    Flush de sesion pre-compaction. Guarda estado critico antes de que
    el contexto se compacte. EJECUTAR cuando la conversacion es larga.
    Consolida todo en una sola llamada: checkpoint + decisiones + errores + backup.
    """
    resultados = []

    # 1. Guardar checkpoint principal
    try:
        resultado_checkpoint = _checkpoint_memoria(
            momento="flush_pre_compaction",
            que_paso=resumen,
            por_que_importa="Flush automatico antes de compaction para no perder contexto"
        )
        resultados.append("Checkpoint: OK")
    except Exception as e:
        resultados.append(f"Checkpoint: ERROR - {redact_secrets(str(e))}")

    # 2. Guardar decisiones si hay
    if decisiones.strip():
        try:
            add_memory_smart(
                content=f"[DECISIONES SESION {now_short()}] {decisiones}",
                category="aprendizaje",
                source="experienced",
                importance="high"
            )
            resultados.append("Decisiones: OK")
        except Exception as e:
            resultados.append(f"Decisiones: ERROR - {redact_secrets(str(e))}")

    # 3. Guardar errores si hay
    if errores.strip():
        try:
            add_memory_smart(
                content=f"[ERRORES SESION {now_short()}] {errores}",
                category="aprendizaje",
                source="learned",
                importance="high"
            )
            resultados.append("Errores: OK")
        except Exception as e:
            resultados.append(f"Errores: ERROR - {redact_secrets(str(e))}")

    # 4. Guardar aprendizajes si hay
    if aprendizajes.strip():
        try:
            add_memory_smart(
                content=f"[APRENDIZAJES SESION {now_short()}] {aprendizajes}",
                category="aprendizaje",
                source="learned",
                importance="high"
            )
            resultados.append("Aprendizajes: OK")
        except Exception as e:
            resultados.append(f"Aprendizajes: ERROR - {redact_secrets(str(e))}")

    # 5. Hacer backup JSON
    try:
        maybe_backup(reason="flush_session", force=True)
        resultados.append("Backup: OK")
    except Exception as e:
        resultados.append(f"Backup: ERROR - {redact_secrets(str(e))}")

    # 6. Session Bridge checkpoint
    try:
        from modules.session_bridge import checkpoint_session_close
        bridge_result = checkpoint_session_close(
            session_summary=resumen,
            decisions=decisiones,
            goal_stack=[],
            source="flush"
        )
        if bridge_result.get("ok"):
            resultados.append(f"Session checkpoint: OK (id={bridge_result.get('checkpoint_id')})")
        else:
            resultados.append(f"Session checkpoint: DEDUPED (id={bridge_result.get('checkpoint_id')})")
    except Exception as e:
        resultados.append(f"Session checkpoint: ERROR - {redact_secrets(str(e))}")

    # 6b. JSON fallback (backward compat)
    try:
        _save_session_state(resumen)
    except Exception:
        pass

    # 7. Process pending FTS retry queue (P2A)
    try:
        fts_result = process_fts_queue(limit=100)
        if fts_result.get("processed", 0) > 0:
            resultados.append(f"FTS queue: {fts_result['succeeded']} OK, {fts_result['failed']} failed")
        else:
            resultados.append("FTS queue: sin pendientes")
    except Exception:
        resultados.append("FTS queue: error procesando")

    return f"FLUSH COMPLETADO\n" + "\n".join(resultados)


def _export_memories_markdown() -> str:
    """Exporta todas las memorias en formato Markdown."""
    try:
        points = pg.get_all()
        if not points:
            return "No hay memorias para exportar."

        by_category = {}
        for p in points:
            payload = p.payload or {}
            cat = payload.get("category", "general")
            if cat not in by_category:
                by_category[cat] = []
            by_category[cat].append({
                "id": str(p.id)[:8],
                "text": payload.get("data", payload.get("content", ""))
            })

        total = sum(len(v) for v in by_category.values())
        lines = [
            f"# Backup Memorias Codi",
            f"",
            f"**Fecha:** {now_short()}",
            f"**Total:** {total} memorias",
            f"**Schema:** v2 con Ownership Tagging",
            f"",
        ]

        for cat, mems in sorted(by_category.items()):
            lines.append(f"## {cat.upper()}")
            lines.append("")
            for m in mems:
                lines.append(f"- [{m['id']}] {m['text'][:120]}")
            lines.append("")

        return "\n".join(lines)

    except Exception as e:
        return f"Error exportando: {redact_secrets(str(e))}"


def _export_to_markdown() -> str:
    """Exporta todas las memorias a archivos Markdown organizados por categoria.
    Genera: SOUL.md, PROJECTS.md, LEARNINGS.md, EPISODES.md, GENERAL.md, RELATIONSHIPS.md
    y journal diario. Util como backup legible y referencia rapida."""
    try:
        export_memories_to_files()

        # Contar archivos generados
        md_files = [f for f in os.listdir(MARKDOWN_DIR) if f.endswith('.md')]
        journal_files = []
        if os.path.exists(JOURNAL_DIR):
            journal_files = [f for f in os.listdir(JOURNAL_DIR) if f.endswith('.md')]

        return f"Export completado a {MARKDOWN_DIR}\nArchivos: {', '.join(md_files)}\nJournal entries: {len(journal_files)} dias"
    except Exception as e:
        return f"Error exportando: {redact_secrets(str(e))}"


def register_tools(mcp):
    """Register all flush/export tools with the MCP server."""

    @mcp.tool()
    def checkpoint_memoria(momento: str, que_paso: str, por_que_importa: str) -> str:
        """
        Guarda un checkpoint con ownership automatico.
        Los checkpoints siempre son source=experienced, importancia alta.
        """
        return _checkpoint_memoria(momento, que_paso, por_que_importa)

    @mcp.tool()
    def flush_session(resumen: str, decisiones: str = "", errores: str = "",
                      aprendizajes: str = "") -> str:
        """
        Flush de sesion pre-compaction. Guarda estado critico antes de que
        el contexto se compacte. EJECUTAR cuando la conversacion es larga.
        Consolida todo en una sola llamada: checkpoint + decisiones + errores + backup.
        """
        return _flush_session(resumen, decisiones, errores, aprendizajes)

    @mcp.tool()
    def export_memories_markdown() -> str:
        """Exporta todas las memorias en formato Markdown."""
        return _export_memories_markdown()

    @mcp.tool()
    def export_to_markdown() -> str:
        """Exporta todas las memorias a archivos Markdown organizados por categoria.
        Genera: SOUL.md, PROJECTS.md, LEARNINGS.md, EPISODES.md, GENERAL.md, RELATIONSHIPS.md
        y journal diario. Util como backup legible y referencia rapida."""
        return _export_to_markdown()
