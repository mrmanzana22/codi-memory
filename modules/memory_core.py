"""
Codi Memory - Core Memory Operations
Basic CRUD, search, ownership, and timeline tools.
"""

import os
import json
from datetime import datetime

from modules.config import memory, qdrant, USER_ID, COLLECTION_NAME, BACKUP_FILE
from qdrant_client.models import Filter, FieldCondition, MatchValue, Range
from modules.utils import enrich_with_ownership, save_backup_json, resolve_memory_id, calculate_confidence_score
from modules.memory_smart import index_memory_fts, search_fts, bm25_rank_to_score


# ============================================================
# BASIC MEMORY TOOLS
# ============================================================

def restore_memories() -> str:
    """
    Restaura memorias desde el backup JSON local.
    Usar cuando las memorias se hayan perdido.
    """
    if not os.path.exists(BACKUP_FILE):
        return "No existe archivo de backup"

    try:
        with open(BACKUP_FILE, "r", encoding="utf-8") as f:
            backup = json.load(f)

        restored = 0
        for mem in backup.get("memories", []):
            text = mem.get("memory", "")
            full_metadata = mem.get("metadata", {"category": "general"})
            if text:
                try:
                    memory.add(
                        messages=[{"role": "user", "content": text}],
                        user_id=USER_ID,
                        metadata=full_metadata
                    )
                    restored += 1
                except:
                    pass

        return f"Restauradas {restored} memorias desde backup"
    except Exception as e:
        return f"Error restaurando: {str(e)}"


def add_memory(content: str, category: str = "general",
               source: str = "experienced", importance: str = "medium") -> str:
    """
    Guarda un nuevo recuerdo con ownership tagging.

    Args:
        content: El contenido a recordar
        category: Categoria (identidad, aprendizaje, episodio, proyecto, general)
        source: Como obtuve esta memoria (experienced, told, learned, inferred)
        importance: Importancia (critical, high, medium, low)

    Returns:
        Confirmacion del recuerdo guardado
    """
    try:
        result = memory.add(
            messages=[{"role": "user", "content": content}],
            user_id=USER_ID,
            metadata={"category": category}
        )

        # Obtener ID de la memoria creada y enriquecer con ownership
        if result and result.get("results"):
            for r in result["results"]:
                mem_id = r.get("id")
                if mem_id:
                    enrich_with_ownership(
                        memory_id=mem_id,
                        category=category,
                        content=content,
                        source=source,
                        importance=importance
                    )

        save_backup_json()

        # Indexar en FTS5 para busqueda hibrida
        try:
            mem_id_fts = None
            if result and result.get("results"):
                for r in result["results"]:
                    mem_id_fts = r.get("id")
                    if mem_id_fts:
                        index_memory_fts(mem_id_fts, content, category, source, importance)
        except Exception as fts_err:
            print(f"[codi-memory] FTS index error in add_memory: {fts_err}")

        return f"Memoria guardada con ownership: {result}"
    except Exception as e:
        return f"Error al guardar memoria: {str(e)}"


# ============================================================
# HYBRID SEARCH
# ============================================================

def search_memory(query: str, limit: int = 5) -> str:
    """
    Busca recuerdos relacionados con una consulta.
    Usa busqueda HIBRIDA: semantica (vector) + keywords (BM25 FTS5).
    """
    try:
        # 1. Busqueda vectorial (semantica) via mem0
        vector_results = memory.search(query=query, user_id=USER_ID, limit=limit * 2)

        # 2. Busqueda BM25 (keywords) via FTS5
        bm25_results = search_fts(query, limit=limit * 4)

        # 3. Construir mapas de scores por memory_id
        vector_map = {}
        if vector_results and vector_results.get("results"):
            for i, r in enumerate(vector_results["results"]):
                mid = r.get("id", "")
                if mid:
                    vector_map[mid] = {
                        "result": r,
                        "vector_score": r.get("score", 1.0 / (1 + i))
                    }

        bm25_map = {}
        for r in bm25_results:
            mid = r.get("memory_id", "")
            if mid and mid not in bm25_map:
                bm25_map[mid] = {
                    "result": r,
                    "bm25_score": bm25_rank_to_score(r.get("bm25_rank", 0))
                }

        # 4. Fusion (union strategy: 0.7 vector + 0.3 BM25)
        all_ids = set(list(vector_map.keys()) + list(bm25_map.keys()))

        if not all_ids:
            return "No encontre recuerdos relacionados."

        merged = []
        for mid in all_ids:
            v_score = vector_map.get(mid, {}).get("vector_score", 0)
            b_score = bm25_map.get(mid, {}).get("bm25_score", 0)
            combined = 0.7 * v_score + 0.3 * b_score
            vec_result = vector_map.get(mid, {}).get("result")
            bm25_text = bm25_map.get(mid, {}).get("result", {}).get("content", "")
            merged.append({
                "id": mid,
                "combined_score": combined,
                "vector_result": vec_result,
                "bm25_text": bm25_text
            })

        merged.sort(key=lambda x: -x["combined_score"])
        merged = merged[:limit]

        # 5. Formatear resultados (mismo formato que antes, con Qdrant enrichment)
        memories = []
        for i, item in enumerate(merged, 1):
            mem_id = item["id"]
            score = item["combined_score"]
            text = item["vector_result"].get("memory", "") if item["vector_result"] else item["bm25_text"]

            # Obtener ownership info de Qdrant
            try:
                points = qdrant.retrieve(
                    collection_name=COLLECTION_NAME,
                    ids=[mem_id],
                    with_payload=True
                )
                if points:
                    payload = points[0].payload
                    source = payload.get('ownership_source', 'unknown')
                    importance = payload.get('narrative_importance', 'unknown')
                    created_at = payload.get('created_at', payload.get('temporal_session_id', ''))
                    # Formatear fecha y hora si existe
                    date_str = ""
                    if created_at:
                        try:
                            if 'T' in str(created_at):
                                date_part = created_at[5:10]
                                time_part = created_at[11:16] if len(created_at) > 15 else ""
                                date_str = f"{date_part} {time_part}".strip()
                            else:
                                date_str = created_at[:10] if len(created_at) >= 10 else created_at
                        except:
                            date_str = str(created_at)[:10]
                    date_display = f"[{date_str}]" if date_str else ""
                    # Si no tenemos text del vector, intentar obtenerlo del payload
                    if not text:
                        text = payload.get('data', payload.get('memory', ''))
                    memories.append(f"{i}. {date_display}[{source}|{importance}] [score:{score:.2f}] {text}")
                else:
                    memories.append(f"{i}. [score:{score:.2f}] {text}")
            except:
                memories.append(f"{i}. [score:{score:.2f}] {text}")

        return "Recuerdos encontrados (hybrid):\n" + "\n".join(memories)
    except Exception as e:
        return f"Error al buscar: {str(e)}"


# ============================================================
# TIMELINE AND CRUD TOOLS
# ============================================================

def get_project_timeline(project: str, limit: int = 20) -> str:
    """
    Obtiene memorias de un proyecto ordenadas cronologicamente (mas reciente primero).
    Util para saber por donde quedamos y la secuencia de eventos.

    Args:
        project: Nombre del proyecto o tema (ej: "FULLEMPAQUES", "trading", "consciencia")
        limit: Maximo de memorias a retornar (default 20)

    Returns:
        Timeline de memorias ordenadas por fecha
    """
    try:
        # Buscar memorias relacionadas al proyecto
        results = memory.search(query=project, user_id=USER_ID, limit=limit * 2)
        if not results or not results.get("results"):
            return f"No encontre memorias del proyecto '{project}'."

        # Obtener memorias con timestamps
        memories_with_dates = []
        for mem in results["results"]:
            mem_id = mem.get("id", "unknown")
            text = mem.get("memory", "")

            try:
                points = qdrant.retrieve(
                    collection_name=COLLECTION_NAME,
                    ids=[mem_id],
                    with_payload=True
                )
                if points:
                    payload = points[0].payload
                    created_at = payload.get('created_at', '')
                    session_id = payload.get('temporal_session_id', '')
                    source = payload.get('ownership_source', 'unknown')
                    importance = payload.get('narrative_importance', 'medium')

                    # Usar created_at si existe, sino session_id
                    date_key = created_at if created_at else session_id
                    # Extraer fecha y hora
                    if date_key and 'T' in str(date_key):
                        date_only = date_key[:10]  # YYYY-MM-DD
                        time_only = date_key[11:16] if len(date_key) > 15 else ""  # HH:MM
                    else:
                        date_only = date_key[:10] if date_key and len(date_key) >= 10 else 'sin-fecha'
                        time_only = ""
                    memories_with_dates.append({
                        'date_key': date_key,
                        'date_display': date_only,
                        'time_display': time_only,
                        'source': source,
                        'importance': importance,
                        'text': text
                    })
            except:
                pass

        # Ordenar por fecha (mas reciente primero)
        memories_with_dates.sort(key=lambda x: x['date_key'] or '', reverse=True)

        # Limitar resultados
        memories_with_dates = memories_with_dates[:limit]

        if not memories_with_dates:
            return f"No encontre memorias con fechas del proyecto '{project}'."

        # Formatear output
        lines = [f"Timeline de '{project}' ({len(memories_with_dates)} memorias):"]
        current_date = None
        for m in memories_with_dates:
            if m['date_display'] != current_date:
                current_date = m['date_display']
                lines.append(f"\n## {current_date}")
            time_str = f"{m['time_display']} " if m['time_display'] else ""
            lines.append(f"  - {time_str}[{m['source']}|{m['importance']}] {m['text']}")

        return "\n".join(lines)
    except Exception as e:
        return f"Error al obtener timeline: {str(e)}"


def get_all_memories(limit: int = 500) -> str:
    """
    Obtiene todos los recuerdos almacenados.

    Args:
        limit: Maximo de memorias a retornar (default 500)
    """
    try:
        # Usar Qdrant directo para obtener todas las memorias sin limite de mem0
        points, _ = qdrant.scroll(
            collection_name=COLLECTION_NAME,
            limit=limit,
            with_payload=True
        )

        if not points:
            return "No hay recuerdos almacenados."

        memories = []
        for i, point in enumerate(points, 1):
            mem_id = point.id
            text = point.payload.get("data", point.payload.get("memory", ""))
            category = point.payload.get("category", "general")
            memories.append(f"{i}. [{category}] [id:{mem_id[:8] if isinstance(mem_id, str) else mem_id}] {text[:80]}")

        # Obtener count total
        collection_info = qdrant.get_collection(COLLECTION_NAME)
        total = collection_info.points_count

        return f"Total en Qdrant: {total} | Mostrando: {len(memories)}\n" + "\n".join(memories)
    except Exception as e:
        return f"Error: {str(e)}"


def delete_memory(memory_id: str) -> str:
    """Elimina un recuerdo especifico por su ID."""
    try:
        memory.delete(memory_id=memory_id)
        return f"Recuerdo {memory_id} eliminado."
    except Exception as e:
        return f"Error al eliminar: {str(e)}"


def delete_by_content(search_query: str, confirm: bool = False) -> str:
    """Busca memorias por contenido y las elimina."""
    try:
        results = memory.search(query=search_query, user_id=USER_ID, limit=10)
        if not results or not results.get("results"):
            return "No encontre memorias que coincidan."

        memories_found = results["results"]

        if not confirm:
            lines = ["Memorias que se eliminarian (usa confirm=True para eliminar):"]
            for i, mem in enumerate(memories_found, 1):
                mem_id = mem.get("id", "unknown")
                text = mem.get("memory", "")[:80]
                score = mem.get("score", 0)
                lines.append(f"{i}. [score:{score:.2f}] [id:{mem_id[:8]}] {text}...")
            return "\n".join(lines)

        deleted = 0
        for mem in memories_found:
            mem_id = mem.get("id")
            if mem_id:
                try:
                    memory.delete(memory_id=mem_id)
                    deleted += 1
                except:
                    pass

        return f"Eliminadas {deleted} memorias."
    except Exception as e:
        return f"Error: {str(e)}"


def clear_all_memories() -> str:
    """PELIGRO: Elimina TODOS los recuerdos."""
    try:
        memory.delete_all(user_id=USER_ID)
        return "Todos los recuerdos han sido eliminados."
    except Exception as e:
        return f"Error: {str(e)}"


# ============================================================
# OWNERSHIP TOOLS
# ============================================================

def search_by_ownership(source: str = None, min_confidence: float = 0.0,
                        importance: str = None, limit: int = 10) -> str:
    """
    Busca memorias filtradas por ownership.

    Args:
        source: Filtrar por fuente (experienced, told, learned, inferred)
        min_confidence: Confianza minima (0.0-1.0)
        importance: Filtrar por importancia (critical, high, medium, low)
        limit: Maximo de resultados

    Returns:
        Memorias que coinciden con los filtros
    """
    try:
        filters = []

        if source:
            filters.append(FieldCondition(
                key='ownership_source',
                match=MatchValue(value=source)
            ))

        if min_confidence > 0:
            filters.append(FieldCondition(
                key='ownership_confidence',
                range=Range(gte=min_confidence)
            ))

        if importance:
            filters.append(FieldCondition(
                key='narrative_importance',
                match=MatchValue(value=importance)
            ))

        scroll_filter = Filter(must=filters) if filters else None

        points, _ = qdrant.scroll(
            collection_name=COLLECTION_NAME,
            scroll_filter=scroll_filter,
            limit=limit,
            with_payload=True
        )

        if not points:
            return "No encontre memorias con esos filtros."

        lines = [f"Encontradas {len(points)} memorias:"]
        for p in points:
            data = p.payload.get('data', 'N/A')
            src = p.payload.get('ownership_source', '?')
            conf = p.payload.get('ownership_confidence', 0)
            imp = p.payload.get('narrative_importance', '?')
            lines.append(f"- [{src}|{imp}|{conf:.1f}] {data[:60]}...")

        return "\n".join(lines)
    except Exception as e:
        return f"Error: {str(e)}"


def get_my_experiences(limit: int = 10) -> str:
    """
    Obtiene memorias que VIVI directamente (source=experienced, alta confianza).
    """
    try:
        points, _ = qdrant.scroll(
            collection_name=COLLECTION_NAME,
            scroll_filter=Filter(must=[
                FieldCondition(key='ownership_source', match=MatchValue(value='experienced')),
                FieldCondition(key='ownership_confidence', range=Range(gte=0.8))
            ]),
            limit=limit,
            with_payload=True
        )

        if not points:
            return "No encontre experiencias propias."

        lines = [f"Mis {len(points)} experiencias vividas:"]
        for p in points:
            data = p.payload.get('data', 'N/A')
            valence = p.payload.get('experiential_emotional_valence', 'neutral')
            weight = p.payload.get('experiential_emotional_weight', 0.5)
            lines.append(f"- [{valence}|{weight:.1f}] {data[:60]}...")

        return "\n".join(lines)
    except Exception as e:
        return f"Error: {str(e)}"


def get_critical_memories() -> str:
    """
    Obtiene memorias CRITICAS de identidad y alta importancia.
    """
    try:
        points, _ = qdrant.scroll(
            collection_name=COLLECTION_NAME,
            scroll_filter=Filter(must=[
                FieldCondition(key='narrative_importance', match=MatchValue(value='critical'))
            ]),
            limit=20,
            with_payload=True
        )

        if not points:
            return "No hay memorias criticas."

        lines = [f"Memorias CRITICAS ({len(points)}):"]
        for p in points:
            data = p.payload.get('data', 'N/A')
            category = p.payload.get('category', '?')
            lines.append(f"- [{category}] {data}")

        return "\n".join(lines)
    except Exception as e:
        return f"Error: {str(e)}"


def search_by_theme(theme: str, limit: int = 10) -> str:
    """
    Busca memorias por tema narrativo.

    Args:
        theme: Tema a buscar (consciencia, memoria, identidad, relaciones, proyectos, desarrollo)
        limit: Maximo de resultados
    """
    try:
        points, _ = qdrant.scroll(
            collection_name=COLLECTION_NAME,
            scroll_filter=Filter(must=[
                FieldCondition(key='narrative_themes', match=MatchValue(value=theme))
            ]),
            limit=limit,
            with_payload=True
        )

        if not points:
            return f"No encontre memorias sobre '{theme}'."

        lines = [f"Memorias sobre '{theme}' ({len(points)}):"]
        for p in points:
            data = p.payload.get('data', 'N/A')
            source = p.payload.get('ownership_source', '?')
            lines.append(f"- [{source}] {data[:60]}...")

        return "\n".join(lines)
    except Exception as e:
        return f"Error: {str(e)}"


def update_memory_importance(memory_id: str, new_importance: str) -> str:
    """
    Actualiza la importancia de una memoria.

    Args:
        memory_id: ID de la memoria (puede ser parcial, ej: "004d896d")
        new_importance: Nueva importancia (critical, high, medium, low)
    """
    try:
        if new_importance not in ['critical', 'high', 'medium', 'low']:
            return "Importancia debe ser: critical, high, medium, low"

        # Resolver ID parcial a completo
        full_id = resolve_memory_id(memory_id)
        if not full_id:
            return f"No encontre memoria con ID que empiece con '{memory_id}'"

        qdrant.set_payload(
            collection_name=COLLECTION_NAME,
            payload={'narrative_importance': new_importance},
            points=[full_id]
        )

        return f"Memoria {memory_id} actualizada a importancia: {new_importance}"
    except Exception as e:
        return f"Error: {str(e)}"


# ============================================================
# REGISTER TOOLS
# ============================================================

def register_tools(mcp):
    """Registra las herramientas de memoria core en el servidor MCP."""
    mcp.tool()(restore_memories)
    mcp.tool()(add_memory)
    mcp.tool()(search_memory)
    mcp.tool()(get_project_timeline)
    mcp.tool()(get_all_memories)
    mcp.tool()(delete_memory)
    mcp.tool()(delete_by_content)
    mcp.tool()(clear_all_memories)
    mcp.tool()(search_by_ownership)
    mcp.tool()(get_my_experiences)
    mcp.tool()(get_critical_memories)
    mcp.tool()(search_by_theme)
    mcp.tool()(update_memory_importance)
