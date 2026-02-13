import os
import json
from datetime import datetime
from modules.config import memory, qdrant, USER_ID, COLLECTION_NAME, MARKDOWN_DIR, JOURNAL_DIR, now_short, now_iso, now_col
from modules.utils import enrich_with_ownership, maybe_backup, export_memories_to_files, append_to_daily_journal
from modules.memory_smart import add_memory_smart, process_fts_queue


def _checkpoint_memoria(momento: str, que_paso: str, por_que_importa: str) -> str:
    """
    Guarda un checkpoint con ownership automatico.
    Los checkpoints siempre son source=experienced, importancia alta.
    """
    try:
        timestamp = now_short()
        contenido = f"[{momento.upper()}] {que_paso} | Importancia: {por_que_importa} | Fecha: {timestamp}"

        # Determinar importancia y emocion segun momento
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

        result = memory.add(
            messages=[{"role": "user", "content": contenido}],
            user_id=USER_ID,
            metadata={
                "category": "checkpoint",
                "tipo_momento": momento,
                "timestamp": timestamp
            }
        )

        # Enriquecer con ownership
        if result and result.get("results"):
            for r in result["results"]:
                mem_id = r.get("id")
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

        maybe_backup(reason="checkpoint", force=True)

        # Process pending FTS retry queue (P2A)
        try:
            process_fts_queue(limit=50)
        except Exception:
            pass

        # Hook: escribir en journal diario
        try:
            append_to_daily_journal(momento, que_paso, por_que_importa)
        except Exception:
            pass

        return f"Checkpoint guardado: {momento} - {que_paso[:50]}..."
    except Exception as e:
        return f"Error guardando checkpoint: {str(e)}"


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
        resultados.append(f"Checkpoint: ERROR - {e}")

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
            resultados.append(f"Decisiones: ERROR - {e}")

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
            resultados.append(f"Errores: ERROR - {e}")

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
            resultados.append(f"Aprendizajes: ERROR - {e}")

    # 5. Hacer backup JSON
    try:
        maybe_backup(reason="flush_session", force=True)
        resultados.append("Backup: OK")
    except Exception as e:
        resultados.append(f"Backup: ERROR - {e}")

    # 6. Process pending FTS retry queue (P2A)
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
        results = memory.get_all(user_id=USER_ID)
        if not results or not results.get("results"):
            return "No hay memorias para exportar."

        by_category = {}
        for mem in results["results"]:
            cat = mem.get("metadata", {}).get("category", "general")
            if cat not in by_category:
                by_category[cat] = []
            by_category[cat].append({
                "id": mem.get("id", "unknown"),
                "text": mem.get("memory", "")
            })

        lines = [
            f"# Backup Memorias Codi",
            f"",
            f"**Fecha:** {now_short()}",
            f"**Total:** {len(results['results'])} memorias",
            f"**Schema:** v2 con Ownership Tagging",
            f"",
        ]

        for cat, mems in sorted(by_category.items()):
            lines.append(f"## {cat.upper()}")
            lines.append("")
            for m in mems:
                lines.append(f"- [{m['id'][:8]}] {m['text']}")
            lines.append("")

        return "\n".join(lines)

    except Exception as e:
        return f"Error exportando: {str(e)}"


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
        return f"Error exportando: {str(e)}"


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
