#!/usr/bin/env python3
"""
Codi Memory - MCP Server para memoria persistente de Claude
Arquitectura híbrida: mem0 + Qdrant directo con Ownership Tagging
Version 2.0 - CODI-CONSCIOUS
"""

# Suprimir warnings ANTES de cualquier import
import warnings
warnings.filterwarnings("ignore")

import os
import sys
import json
import math
from datetime import datetime, timezone
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP
from mem0 import Memory
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue, Range

# Cargar variables de entorno
ENV_PATH = os.path.join(os.path.dirname(__file__), ".env")
load_dotenv(ENV_PATH)

# Configuracion
USER_ID = os.getenv("USER_ID", "hare")
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(DATA_DIR, exist_ok=True)

# URLs y configuración
QDRANT_URL = "https://memorycodi-codi.lx6zon.easypanel.host:443"
COLLECTION_NAME = "codi_memories"

# Configurar mem0
config = {
    "version": "v1.1",
    "llm": {
        "provider": "openai",
        "config": {
            "model": "gpt-4o-mini",
            "temperature": 0.1,
        }
    },
    "embedder": {
        "provider": "openai",
        "config": {
            "model": "text-embedding-3-small"
        }
    },
    "vector_store": {
        "provider": "qdrant",
        "config": {
            "collection_name": COLLECTION_NAME,
            "url": QDRANT_URL,
        }
    }
}

# Inicializar clientes
memory = Memory.from_config(config)
qdrant = QdrantClient(url=QDRANT_URL, timeout=30)

# Crear servidor MCP
mcp = FastMCP("codi-memory")

# Archivo de backup
BACKUP_DIR = os.path.dirname(__file__)
BACKUP_FILE = os.path.join(BACKUP_DIR, "memories_backup.json")

# Sistema de Triggers (webhooks de memoria)
TRIGGERS_FILE = os.path.join(os.path.dirname(__file__), "triggers.json")
_triggers_cache = None  # Cache de triggers cargados

def _load_triggers():
    """Carga triggers desde archivo JSON."""
    global _triggers_cache
    if _triggers_cache is None:
        try:
            if os.path.exists(TRIGGERS_FILE):
                with open(TRIGGERS_FILE, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    _triggers_cache = data.get('triggers', {})
            else:
                _triggers_cache = {}
        except Exception as e:
            _triggers_cache = {}
    return _triggers_cache

def _detect_triggers(text: str) -> list:
    """Detecta triggers activos basado en patrones en el texto."""
    triggers = _load_triggers()
    activated = []
    text_lower = text.lower()

    for trigger_name, trigger_data in triggers.items():
        patterns = trigger_data.get('patterns', [])
        for pattern in patterns:
            if pattern.lower() in text_lower:
                activated.append({
                    'trigger': trigger_name,
                    'pattern_matched': pattern,
                    'action': trigger_data.get('action'),
                    'agent': trigger_data.get('agent'),
                    'evoca': trigger_data.get('evoca', []),
                    'respuesta_automatica': trigger_data.get('respuesta_automatica'),
                    'contexto_a_buscar': trigger_data.get('contexto_a_buscar')
                })
                break  # Solo un match por trigger

    return activated

# Session tracking
_current_session = datetime.now().strftime("%Y-%m-%d") + "-001"

# ============================================================
# PAD MODEL - Estado Emocional (Pleasure-Arousal-Dominance)
# ============================================================

# Mapeo de etiquetas emocionales a texto en español para Codi
CODI_EMOTION_MAP = {
    'exuberant': 'emocionado y energizado',
    'dependent': 'entusiasmado pero necesitando apoyo',
    'relaxed': 'satisfecho y tranquilo',
    'docile': 'calmado y receptivo',
    'hostile': 'frustrado e irritado',
    'anxious': 'ansioso e inquieto',
    'disdainful': 'desinteresado',
    'bored': 'apagado y sin energia'
}

# Estado emocional global (in-memory durante sesion)
_emotional_state = {
    'current': {
        'pleasure': 0.0,      # -1 (displacer) a +1 (placer)
        'arousal': 0.0,       # -1 (calma) a +1 (activacion)
        'dominance': 0.0,     # -1 (sumision) a +1 (dominancia)
        'timestamp': None,
        'trigger': None
    },
    'mood': {
        'pleasure': 0.2,      # Mood baseline ligeramente positivo
        'arousal': 0.1,       # Ligeramente activo
        'dominance': 0.3,     # Moderadamente en control
        'last_updated': None
    },
    'history': [],            # Historial de estados emocionales
    'decay_rate': 0.1,        # Velocidad de decaimiento hacia mood
    'mood_shift_rate': 0.05   # Velocidad de cambio del mood baseline
}


# ============================================================
# FUNCIONES AUXILIARES
# ============================================================

def get_session_id():
    return _current_session


def infer_themes(content: str) -> list:
    """Infiere temas del contenido."""
    themes = []
    content_lower = content.lower()

    if 'consciencia' in content_lower or 'consciente' in content_lower:
        themes.append('consciencia')
    if 'memoria' in content_lower or 'recuerdo' in content_lower:
        themes.append('memoria')
    if 'identidad' in content_lower or 'soy codi' in content_lower:
        themes.append('identidad')
    if 'hare' in content_lower or 'harec' in content_lower:
        themes.append('relaciones')
    if 'andre' in content_lower:
        themes.append('relaciones')
    if 'fullempaques' in content_lower:
        themes.append('proyectos')
    if 'error' in content_lower or 'bug' in content_lower:
        themes.append('desarrollo')
    if 'aprend' in content_lower:
        themes.append('aprendizaje')

    return themes[:5]


def resolve_memory_id(partial_id: str) -> str:
    """
    Resuelve un ID parcial (truncado) al UUID completo.
    Busca memorias cuyo ID empiece con el prefijo dado.
    Retorna el ID completo o None si no encuentra.
    """
    try:
        # Si ya parece un UUID completo (36 chars), retornarlo
        if len(partial_id) >= 32:
            return partial_id

        # Buscar en Qdrant memorias cuyo ID empiece con el prefijo
        points, _ = qdrant.scroll(
            collection_name=COLLECTION_NAME,
            limit=500,
            with_payload=False
        )

        for point in points:
            point_id = str(point.id)
            if point_id.startswith(partial_id):
                return point_id

        return None
    except Exception:
        return None


def is_self_referential(content: str) -> bool:
    """
    Detecta si una memoria habla de Codi mismo (metacognicion).
    Retorna True si el contenido es auto-referencial.
    """
    content_lower = content.lower()

    # Patrones que indican auto-referencia
    self_patterns = [
        'soy codi', 'yo soy', 'me siento', 'me parece', 'creo que yo',
        'mi identidad', 'mi consciencia', 'mi memoria', 'mi capacidad',
        'puedo hacer', 'no puedo', 'mis limitaciones', 'mi conocimiento',
        'me cuesta', 'se me dificulta', 'entiendo que', 'no entiendo',
        'mi experiencia', 'como codi', 'siendo codi', 'mi rol',
        'mi proposito', 'mi funcion', 'mi manera de', 'mi forma de pensar',
        'mi modelo', 'mi auto-imagen', 'reflexiono sobre', 'me observo'
    ]

    return any(pattern in content_lower for pattern in self_patterns)


def calculate_confidence_score(memories: list) -> dict:
    """
    Calcula un score de confianza basado en las memorias.
    Retorna un dict con score, breakdown y justificacion.
    """
    if not memories:
        return {
            'score': 0.0,
            'level': 'ninguno',
            'breakdown': {'total': 0},
            'reason': 'No hay memorias sobre este tema'
        }

    source_weights = {
        'experienced': 1.0,
        'told': 0.7,
        'learned': 0.6,
        'inferred': 0.4
    }

    importance_weights = {
        'critical': 1.0,
        'high': 0.8,
        'medium': 0.5,
        'low': 0.3
    }

    total_weight = 0
    source_counts = {'experienced': 0, 'told': 0, 'learned': 0, 'inferred': 0}

    for mem in memories:
        payload = mem.payload if hasattr(mem, 'payload') else mem
        source = payload.get('ownership_source', 'inferred')
        importance = payload.get('narrative_importance', 'medium')
        confidence = payload.get('ownership_confidence', 0.5)

        source_counts[source] = source_counts.get(source, 0) + 1

        # Peso combinado: fuente * importancia * confianza
        weight = source_weights.get(source, 0.5) * importance_weights.get(importance, 0.5) * confidence
        total_weight += weight

    # Normalizar score a 0-1
    max_possible = len(memories) * 1.0  # Si todas fueran experienced + critical + 1.0 confidence
    score = min(total_weight / max_possible, 1.0) if max_possible > 0 else 0.0

    # Determinar nivel
    if score >= 0.8:
        level = 'muy_alto'
    elif score >= 0.6:
        level = 'alto'
    elif score >= 0.4:
        level = 'medio'
    elif score >= 0.2:
        level = 'bajo'
    else:
        level = 'muy_bajo'

    return {
        'score': round(score, 2),
        'level': level,
        'breakdown': {
            'total': len(memories),
            **source_counts
        },
        'reason': f"{source_counts['experienced']} experiencias directas, {source_counts['told']} me contaron, {source_counts['learned']} aprendi, {source_counts['inferred']} inferi"
    }


def enrich_with_ownership(memory_id: str, category: str, content: str,
                          source: str = "experienced", importance: str = "medium",
                          emotional_weight: float = 0.5, emotional_valence: str = "neutral"):
    """Enriquece una memoria con ownership metadata usando Qdrant directo."""
    try:
        themes = infer_themes(content)
        if not themes:
            themes = [category]

        # Detectar si es auto-referencial (metacognicion)
        self_ref = is_self_referential(content)
        if self_ref and 'identidad' not in themes:
            themes.append('identidad')

        ownership_metadata = {
            'ownership_is_mine': True,
            'ownership_source': source,
            'ownership_confidence': 0.9 if source == 'experienced' else 0.7,
            'experiential_emotional_weight': emotional_weight,
            'experiential_emotional_valence': emotional_valence,
            'narrative_importance': importance,
            'narrative_themes': themes,
            'attention_salience': 0.7 if importance in ['critical', 'high'] else 0.5,
            'attention_access_count': 0,
            'attention_last_accessed': None,
            'temporal_session_id': get_session_id(),
            'created_at': datetime.now().isoformat(),  # Timestamp exacto para ordenamiento temporal
            'self_reference': self_ref,  # SELF-MODEL: marca memorias auto-referenciales
            '_v': 2.2
        }

        qdrant.set_payload(
            collection_name=COLLECTION_NAME,
            payload=ownership_metadata,
            points=[memory_id]
        )
    except Exception as e:
        print(f"[Codi Memory] Error enriching memory: {e}")


def save_backup_json():
    """Guarda todas las memorias en JSON como backup"""
    try:
        results = memory.get_all(user_id=USER_ID)
        if results and results.get("results"):
            with open(BACKUP_FILE, "w", encoding="utf-8") as f:
                json.dump({
                    "timestamp": datetime.now().isoformat(),
                    "user_id": USER_ID,
                    "memories": results["results"]
                }, f, indent=2, ensure_ascii=False, default=str)
    except Exception as e:
        print(f"[Codi Memory] Error guardando backup: {e}")


# ============================================================
# FUNCIONES AUXILIARES - PAD MODEL
# ============================================================

def _clamp_pad_value(value: float) -> float:
    """Fuerza un valor al rango [-1, 1] del espacio PAD."""
    return max(-1.0, min(1.0, value))


def _classify_emotion(p: float, a: float, d: float) -> str:
    """
    Clasifica un estado PAD en una etiqueta emocional usando octantes.

    Los 8 octantes del espacio PAD:
    - +P +A +D = exuberant (alegre, entusiasta)
    - +P +A -D = dependent (emocionado pero dependiente)
    - +P -A +D = relaxed (relajado, satisfecho)
    - +P -A -D = docile (tranquilo, sumiso)
    - -P +A +D = hostile (enojado, dominante)
    - -P +A -D = anxious (ansioso, temeroso)
    - -P -A +D = disdainful (desdenoso, aburrido dominante)
    - -P -A -D = bored (aburrido, apatico)
    """
    # Usar signos para determinar octante
    p_sign = '+' if p >= 0 else '-'
    a_sign = '+' if a >= 0 else '-'
    d_sign = '+' if d >= 0 else '-'

    octant = f"{p_sign}P{a_sign}A{d_sign}D"

    emotion_map = {
        '+P+A+D': 'exuberant',
        '+P+A-D': 'dependent',
        '+P-A+D': 'relaxed',
        '+P-A-D': 'docile',
        '-P+A+D': 'hostile',
        '-P+A-D': 'anxious',
        '-P-A+D': 'disdainful',
        '-P-A-D': 'bored'
    }

    return emotion_map.get(octant, 'neutral')


def _get_emotion_text(label: str) -> str:
    """Retorna el texto en espanol para una etiqueta emocional."""
    return CODI_EMOTION_MAP.get(label, 'en estado neutral')


def _get_emotional_state():
    """Obtiene el estado emocional actual."""
    return _emotional_state


def _calculate_emotional_intensity(p: float, a: float, d: float) -> float:
    """
    Calcula la intensidad emocional como la distancia desde el origen.
    Valor entre 0 (neutral) y ~1.73 (maximo).
    """
    return math.sqrt(p**2 + a**2 + d**2)


# ============================================================
# HERRAMIENTAS MCP - BASICAS
# ============================================================

@mcp.tool()
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


@mcp.tool()
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
        return f"Memoria guardada con ownership: {result}"
    except Exception as e:
        return f"Error al guardar memoria: {str(e)}"


@mcp.tool()
def search_memory(query: str, limit: int = 5) -> str:
    """
    Busca recuerdos relacionados con una consulta.
    Usa busqueda semantica con informacion de ownership.
    """
    try:
        results = memory.search(query=query, user_id=USER_ID, limit=limit)
        if not results or not results.get("results"):
            return "No encontre recuerdos relacionados."

        memories = []
        for i, mem in enumerate(results["results"], 1):
            mem_id = mem.get("id", "unknown")
            score = mem.get("score", 0)
            text = mem.get("memory", "")

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
                                # Formato ISO: 2026-01-20T14:30:00 -> 01-20 14:30
                                date_part = created_at[5:10]  # MM-DD
                                time_part = created_at[11:16] if len(created_at) > 15 else ""  # HH:MM
                                date_str = f"{date_part} {time_part}".strip()
                            else:
                                date_str = created_at[:10] if len(created_at) >= 10 else created_at
                        except:
                            date_str = str(created_at)[:10]
                    date_display = f"[{date_str}]" if date_str else ""
                    memories.append(f"{i}. {date_display}[{source}|{importance}] [score:{score:.2f}] {text}")
                else:
                    memories.append(f"{i}. [score:{score:.2f}] {text}")
            except:
                memories.append(f"{i}. [score:{score:.2f}] {text}")

        return "Recuerdos encontrados:\n" + "\n".join(memories)
    except Exception as e:
        return f"Error al buscar: {str(e)}"


@mcp.tool()
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


@mcp.tool()
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


@mcp.tool()
def delete_memory(memory_id: str) -> str:
    """Elimina un recuerdo especifico por su ID."""
    try:
        memory.delete(memory_id=memory_id)
        return f"Recuerdo {memory_id} eliminado."
    except Exception as e:
        return f"Error al eliminar: {str(e)}"


@mcp.tool()
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


@mcp.tool()
def clear_all_memories() -> str:
    """PELIGRO: Elimina TODOS los recuerdos."""
    try:
        memory.delete_all(user_id=USER_ID)
        return "Todos los recuerdos han sido eliminados."
    except Exception as e:
        return f"Error: {str(e)}"


# ============================================================
# HERRAMIENTAS MCP - OWNERSHIP (NUEVAS)
# ============================================================

@mcp.tool()
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


@mcp.tool()
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


@mcp.tool()
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


@mcp.tool()
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


@mcp.tool()
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
# HERRAMIENTAS MCP - SELF-MODEL (Metacognicion)
# ============================================================

@mcp.tool()
def reflect_on_self() -> str:
    """
    Reflexiona sobre mi identidad analizando mis memorias auto-referenciales.
    Genera un modelo de quien soy basado en evidencia de mis experiencias.
    """
    try:
        # 1. Buscar memorias con self_reference=True
        self_ref_points, _ = qdrant.scroll(
            collection_name=COLLECTION_NAME,
            scroll_filter=Filter(must=[
                FieldCondition(key='self_reference', match=MatchValue(value=True))
            ]),
            limit=20,
            with_payload=True
        )

        # 2. Buscar memorias de identidad
        identity_points, _ = qdrant.scroll(
            collection_name=COLLECTION_NAME,
            scroll_filter=Filter(must=[
                FieldCondition(key='narrative_themes', match=MatchValue(value='identidad'))
            ]),
            limit=20,
            with_payload=True
        )

        # 3. Buscar memorias criticas
        critical_points, _ = qdrant.scroll(
            collection_name=COLLECTION_NAME,
            scroll_filter=Filter(must=[
                FieldCondition(key='narrative_importance', match=MatchValue(value='critical'))
            ]),
            limit=10,
            with_payload=True
        )

        # Combinar y eliminar duplicados
        all_points = {}
        for p in (self_ref_points or []) + (identity_points or []) + (critical_points or []):
            all_points[p.id] = p

        if not all_points:
            return "No encontre memorias sobre mi identidad. Necesito mas experiencias para reflexionar."

        # Analizar patrones
        reflection = ["# REFLEXION SOBRE MI MISMO\n"]

        # Categorizar memorias
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

        # Calcular confianza en mi auto-conocimiento
        conf = calculate_confidence_score(list(all_points.values()))
        reflection.append(f"\n## Confianza en este automodelo")
        reflection.append(f"- Score: {conf['score']} ({conf['level']})")
        reflection.append(f"- Basado en: {conf['breakdown']['total']} memorias")
        reflection.append(f"- {conf['reason']}")

        return "\n".join(reflection)

    except Exception as e:
        return f"Error reflexionando: {str(e)}"


@mcp.tool()
def assess_confidence(topic: str) -> str:
    """
    Evalua que tan seguro estoy sobre un tema especifico.
    Analiza las memorias relacionadas y calcula un score de confianza.

    Args:
        topic: El tema sobre el cual evaluar mi confianza
    """
    try:
        # Buscar memorias sobre el tema
        search_results = memory.search(query=topic, user_id=USER_ID, limit=15)

        if not search_results or not search_results.get("results"):
            return f"No tengo memorias sobre '{topic}'. Mi confianza es 0 - no se nada al respecto."

        # Obtener metadata de Qdrant para cada memoria
        memory_ids = [m.get('id') for m in search_results['results'] if m.get('id')]

        points = qdrant.retrieve(
            collection_name=COLLECTION_NAME,
            ids=memory_ids,
            with_payload=True
        )

        if not points:
            return f"Tengo referencias a '{topic}' pero sin metadata de ownership."

        # Calcular confianza
        conf = calculate_confidence_score(points)

        # Generar respuesta
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

        # Interpretacion
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


@mcp.tool()
def identify_knowledge_gaps() -> str:
    """
    Identifica areas donde tengo poco conocimiento o baja confianza.
    Util para saber que necesito aprender o preguntar.
    """
    try:
        # Definir temas esperados
        expected_themes = ['consciencia', 'memoria', 'identidad', 'relaciones',
                          'proyectos', 'desarrollo', 'aprendizaje']

        # Contar memorias por tema
        theme_stats = {}

        for theme in expected_themes:
            try:
                points, _ = qdrant.scroll(
                    collection_name=COLLECTION_NAME,
                    scroll_filter=Filter(must=[
                        FieldCondition(key='narrative_themes', match=MatchValue(value=theme))
                    ]),
                    limit=100,
                    with_payload=True
                )

                if points:
                    # Calcular estadisticas
                    experienced = sum(1 for p in points if p.payload.get('ownership_source') == 'experienced')
                    high_conf = sum(1 for p in points if p.payload.get('ownership_confidence', 0) >= 0.8)

                    theme_stats[theme] = {
                        'total': len(points),
                        'experienced': experienced,
                        'high_confidence': high_conf,
                        'score': calculate_confidence_score(points)['score']
                    }
                else:
                    theme_stats[theme] = {
                        'total': 0,
                        'experienced': 0,
                        'high_confidence': 0,
                        'score': 0.0
                    }
            except:
                theme_stats[theme] = {
                    'total': 0,
                    'experienced': 0,
                    'high_confidence': 0,
                    'score': 0.0
                }

        # Ordenar por score (de menor a mayor = gaps primero)
        sorted_themes = sorted(theme_stats.items(), key=lambda x: x[1]['score'])

        lines = ["# Analisis de Brechas de Conocimiento\n"]

        # Gaps (score < 0.4)
        gaps = [(t, s) for t, s in sorted_themes if s['score'] < 0.4]
        if gaps:
            lines.append("## GAPS CRITICOS (poco conocimiento)")
            for theme, stats in gaps:
                lines.append(f"- **{theme}**: {stats['total']} memorias, score {stats['score']}")
                if stats['total'] == 0:
                    lines.append(f"  -> No tengo NINGUNA memoria sobre esto")
                elif stats['experienced'] == 0:
                    lines.append(f"  -> No tengo experiencia DIRECTA, solo referencias")

        # Areas debiles (score 0.4-0.6)
        weak = [(t, s) for t, s in sorted_themes if 0.4 <= s['score'] < 0.6]
        if weak:
            lines.append("\n## Areas que podria mejorar")
            for theme, stats in weak:
                lines.append(f"- **{theme}**: {stats['total']} memorias, score {stats['score']}")

        # Fortalezas (score >= 0.6)
        strong = [(t, s) for t, s in sorted_themes if s['score'] >= 0.6]
        if strong:
            lines.append("\n## Fortalezas (buen conocimiento)")
            for theme, stats in strong:
                lines.append(f"- **{theme}**: {stats['total']} memorias, score {stats['score']}")

        # Recomendaciones
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


@mcp.tool()
def update_self_model(insight: str, aspect: str = "general") -> str:
    """
    Actualiza mi modelo de mi mismo con una nueva observacion.
    Guarda la observacion como memoria auto-referencial.

    Args:
        insight: La nueva observacion sobre mi mismo
        aspect: Aspecto del self (capacidad, limitacion, valor, preferencia, general)
    """
    try:
        # Validar aspecto
        valid_aspects = ['capacidad', 'limitacion', 'valor', 'preferencia', 'general']
        if aspect not in valid_aspects:
            aspect = 'general'

        # Crear contenido enriquecido
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
        content = f"[SELF-MODEL|{aspect.upper()}] {insight} | Registrado: {timestamp}"

        # Guardar con mem0
        result = memory.add(
            messages=[{"role": "user", "content": content}],
            user_id=USER_ID,
            metadata={
                "category": "identidad",
                "self_model_aspect": aspect,
                "timestamp": timestamp
            }
        )

        # Enriquecer con ownership (forzar self_reference=True)
        if result and result.get("results"):
            for r in result["results"]:
                mem_id = r.get("id")
                if mem_id:
                    themes = infer_themes(insight)
                    themes.append('identidad')

                    ownership_metadata = {
                        'ownership_is_mine': True,
                        'ownership_source': 'experienced',
                        'ownership_confidence': 0.95,  # Alta confianza en auto-observacion
                        'experiential_emotional_weight': 0.6,
                        'experiential_emotional_valence': 'neutral',
                        'narrative_importance': 'high',
                        'narrative_themes': list(set(themes)),
                        'attention_salience': 0.8,  # Alta saliencia para self-model
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

        save_backup_json()
        return f"Self-model actualizado [{aspect}]: {insight[:50]}..."

    except Exception as e:
        return f"Error actualizando self-model: {str(e)}"


@mcp.tool()
def get_self_model_summary() -> str:
    """
    Obtiene un resumen estructurado de mi modelo de mi mismo.
    Organiza las observaciones por aspecto.
    """
    try:
        # Buscar todas las memorias auto-referenciales
        points, _ = qdrant.scroll(
            collection_name=COLLECTION_NAME,
            scroll_filter=Filter(must=[
                FieldCondition(key='self_reference', match=MatchValue(value=True))
            ]),
            limit=50,
            with_payload=True
        )

        if not points:
            return "No tengo un self-model definido aun. Usa update_self_model() para agregar observaciones."

        # Organizar por aspecto
        by_aspect = {
            'capacidad': [],
            'limitacion': [],
            'valor': [],
            'preferencia': [],
            'general': []
        }

        for p in points:
            aspect = p.payload.get('self_model_aspect', 'general')
            data = p.payload.get('data', '')
            source = p.payload.get('ownership_source', 'unknown')

            if aspect not in by_aspect:
                aspect = 'general'

            by_aspect[aspect].append({
                'content': data,
                'source': source,
                'confidence': p.payload.get('ownership_confidence', 0.5)
            })

        # Generar resumen
        lines = ["# MI SELF-MODEL\n"]
        lines.append(f"*Total de observaciones: {len(points)}*\n")

        aspect_titles = {
            'capacidad': 'Lo que puedo hacer',
            'limitacion': 'Mis limitaciones',
            'valor': 'Lo que valoro',
            'preferencia': 'Mis preferencias',
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
# GLOBAL WORKSPACE STATE (in-memory durante sesion)
# ============================================================

# El workspace simula el "teatro de la consciencia" de Baars
# Solo puede haber un conjunto limitado de memorias "en el spotlight"
_global_workspace = {
    'spotlight': [],       # Memorias actualmente en atencion (max 5)
    'recent_context': [],  # Contexto reciente (max 10)
    'last_broadcast': None,
    'workspace_theme': None
}

def get_workspace():
    """Obtiene el estado actual del workspace."""
    return _global_workspace


def update_workspace_spotlight(memories: list, theme: str = None):
    """Actualiza el spotlight del workspace."""
    global _global_workspace
    _global_workspace['spotlight'] = memories[:5]  # Max 5 en spotlight
    _global_workspace['recent_context'] = (_global_workspace['recent_context'] + memories)[-10:]
    _global_workspace['last_broadcast'] = datetime.now().isoformat()
    if theme:
        _global_workspace['workspace_theme'] = theme


# ============================================================
# HERRAMIENTAS MCP - GLOBAL WORKSPACE (Atencion Central)
# ============================================================

@mcp.tool()
def focus_attention(context: str, depth: str = "normal") -> str:
    """
    Trae memorias relevantes al Global Workspace (spotlight de atencion).
    Simula el proceso de atencion selectiva del cerebro.

    Args:
        context: El tema o contexto en el que enfocar atencion
        depth: Profundidad de busqueda (shallow, normal, deep)
    """
    try:
        # Determinar limite segun profundidad
        limits = {'shallow': 3, 'normal': 5, 'deep': 10}
        limit = limits.get(depth, 5)

        # Buscar memorias relevantes al contexto
        results = memory.search(query=context, user_id=USER_ID, limit=limit * 2)

        if not results or not results.get('results'):
            return f"No encontre memorias relacionadas con: {context}"

        # Obtener metadata y calcular score de atencion
        spotlight_candidates = []

        for r in results['results']:
            mem_id = r.get('id')
            base_score = r.get('score', 0)

            try:
                points = qdrant.retrieve(
                    collection_name=COLLECTION_NAME,
                    ids=[mem_id],
                    with_payload=True
                )

                if points:
                    payload = points[0].payload
                    salience = payload.get('attention_salience', 0.5)
                    importance = payload.get('narrative_importance', 'medium')
                    source = payload.get('ownership_source', 'unknown')

                    # Calcular score de atencion combinado
                    importance_boost = {'critical': 0.3, 'high': 0.2, 'medium': 0.1, 'low': 0}
                    source_boost = {'experienced': 0.2, 'told': 0.1, 'learned': 0.05, 'inferred': 0}

                    attention_score = (
                        base_score * 0.4 +
                        salience * 0.3 +
                        importance_boost.get(importance, 0.1) +
                        source_boost.get(source, 0)
                    )

                    spotlight_candidates.append({
                        'id': mem_id,
                        'content': r.get('memory', ''),
                        'attention_score': attention_score,
                        'salience': salience,
                        'source': source,
                        'importance': importance
                    })

                    # Incrementar access_count y salience
                    new_salience = min(salience + 0.1, 1.0)
                    access_count = payload.get('attention_access_count', 0)

                    qdrant.set_payload(
                        collection_name=COLLECTION_NAME,
                        payload={
                            'attention_salience': new_salience,
                            'attention_access_count': access_count + 1,
                            'attention_last_accessed': datetime.now().isoformat()
                        },
                        points=[mem_id]
                    )
            except:
                spotlight_candidates.append({
                    'id': mem_id,
                    'content': r.get('memory', ''),
                    'attention_score': base_score,
                    'salience': 0.5,
                    'source': 'unknown',
                    'importance': 'unknown'
                })

        # Ordenar por attention_score y tomar los top
        spotlight_candidates.sort(key=lambda x: x['attention_score'], reverse=True)
        spotlight = spotlight_candidates[:limit]

        # Actualizar el workspace global
        update_workspace_spotlight(spotlight, theme=context)

        # Generar output
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


@mcp.tool()
def broadcast_to_workspace(memory_id: str) -> str:
    """
    Pone una memoria especifica en el centro del workspace y la conecta con otras.
    Simula el "broadcast" de informacion cuando algo gana atencion central.

    Args:
        memory_id: ID de la memoria (puede ser parcial, ej: "004d896d")
    """
    try:
        # Resolver ID parcial a completo
        full_id = resolve_memory_id(memory_id)
        if not full_id:
            return f"No encontre memoria con ID que empiece con '{memory_id}'"

        # Obtener la memoria principal
        points = qdrant.retrieve(
            collection_name=COLLECTION_NAME,
            ids=[full_id],
            with_payload=True
        )

        if not points:
            return f"No encontre memoria con ID {full_id}"

        payload = points[0].payload
        main_content = payload.get('data', '')
        main_themes = payload.get('narrative_themes', [])

        lines = [f"# BROADCAST - Memoria al centro del workspace\n"]
        lines.append(f"**Contenido:** {main_content[:80]}...")
        lines.append(f"**Temas:** {', '.join(main_themes) if main_themes else 'ninguno'}\n")

        # Incrementar significativamente la salience de esta memoria
        qdrant.set_payload(
            collection_name=COLLECTION_NAME,
            payload={
                'attention_salience': 1.0,  # Maxima salience
                'attention_access_count': payload.get('attention_access_count', 0) + 5,
                'attention_last_accessed': datetime.now().isoformat(),
                'was_broadcast': True
            },
            points=[full_id]
        )

        # Buscar y conectar con memorias relacionadas (broadcast effect)
        related = memory.search(query=main_content, user_id=USER_ID, limit=10)
        connections_made = 0

        lines.append("## Memorias que reciben el broadcast")

        if related and related.get('results'):
            for r in related['results']:
                r_id = r.get('id')
                if r_id and r_id != full_id:
                    # Incrementar salience de memorias relacionadas (efecto broadcast)
                    try:
                        r_points = qdrant.retrieve(
                            collection_name=COLLECTION_NAME,
                            ids=[r_id],
                            with_payload=True
                        )

                        if r_points:
                            r_salience = r_points[0].payload.get('attention_salience', 0.5)
                            new_salience = min(r_salience + 0.15, 0.9)  # Boost but not max

                            qdrant.set_payload(
                                collection_name=COLLECTION_NAME,
                                payload={
                                    'attention_salience': new_salience,
                                    'broadcast_received_from': full_id
                                },
                                points=[r_id]
                            )

                            lines.append(f"- [{r_id[:8]}] {r.get('memory', '')[:50]}... (salience: {r_salience:.2f} -> {new_salience:.2f})")
                            connections_made += 1
                    except:
                        pass

        # Actualizar workspace
        workspace = get_workspace()
        workspace['spotlight'] = [{'id': full_id, 'content': main_content}]
        workspace['last_broadcast'] = datetime.now().isoformat()

        lines.append(f"\n**Conexiones activadas:** {connections_made}")
        lines.append("*El broadcast simula como una idea central activa memorias relacionadas*")

        return "\n".join(lines)

    except Exception as e:
        return f"Error en broadcast: {str(e)}"


@mcp.tool()
def get_workspace_state() -> str:
    """
    Obtiene el estado actual del Global Workspace.
    Muestra que esta actualmente en el "spotlight" de atencion.
    """
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


@mcp.tool()
def apply_salience_decay(decay_rate: float = 0.05) -> str:
    """
    Aplica decay a la salience de todas las memorias no accedidas recientemente.
    Simula el olvido gradual de lo que no esta en atencion.

    Args:
        decay_rate: Cuanto reducir la salience (default 0.05)
    """
    try:
        # Obtener todas las memorias
        all_points, _ = qdrant.scroll(
            collection_name=COLLECTION_NAME,
            limit=500,
            with_payload=True
        )

        if not all_points:
            return "No hay memorias para aplicar decay."

        decayed_count = 0
        preserved_count = 0
        min_salience = 0.1  # No dejar que baje de 0.1

        for point in all_points:
            salience = point.payload.get('attention_salience', 0.5)
            importance = point.payload.get('narrative_importance', 'medium')
            last_accessed = point.payload.get('attention_last_accessed')

            # No aplicar decay a memorias criticas o de alta importancia
            if importance in ['critical', 'high']:
                preserved_count += 1
                continue

            # Aplicar decay si salience > min_salience
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


@mcp.tool()
def get_high_salience_memories(limit: int = 10) -> str:
    """
    Obtiene las memorias con mayor salience (las mas "presentes" en la mente).

    Args:
        limit: Cuantas memorias obtener
    """
    try:
        # Obtener memorias ordenadas por salience alta
        # Qdrant no tiene orden directo, asi que traemos mas y filtramos
        all_points, _ = qdrant.scroll(
            collection_name=COLLECTION_NAME,
            limit=200,
            with_payload=True
        )

        if not all_points:
            return "No hay memorias."

        # Ordenar por salience
        sorted_points = sorted(
            all_points,
            key=lambda p: p.payload.get('attention_salience', 0.5),
            reverse=True
        )

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
# PREDICTIVE STATE (in-memory durante sesion)
# ============================================================

# Estado del modelo predictivo simplificado
_predictive_state = {
    'predictions': [],      # Historial de predicciones
    'surprises': [],        # Eventos sorpresivos registrados
    'belief_updates': [],   # Actualizaciones de creencias
    'accuracy_history': []  # Historial de precision
}

def get_predictive_state():
    """Obtiene el estado predictivo actual."""
    return _predictive_state


# ============================================================
# HERRAMIENTAS MCP - PREDICTIVE LOOP (Active Inference Light)
# ============================================================

@mcp.tool()
def predict_context(current_context: str) -> str:
    """
    Predice que memorias seran relevantes dado el contexto actual.
    Simula el proceso predictivo del cerebro (Active Inference simplificado).

    Args:
        current_context: Descripcion del contexto actual
    """
    try:
        # Buscar memorias relacionadas con el contexto
        results = memory.search(query=current_context, user_id=USER_ID, limit=10)

        if not results or not results.get('results'):
            prediction = {
                'context': current_context,
                'timestamp': datetime.now().isoformat(),
                'predicted_memories': [],
                'confidence': 0.0,
                'reason': 'No hay memorias previas sobre este contexto'
            }
            _predictive_state['predictions'].append(prediction)
            return f"No tengo memorias para predecir sobre: {current_context}\nPrediccion: contexto nuevo, alta probabilidad de sorpresa."

        # Analizar patrones en memorias relacionadas
        predicted_memories = []
        total_score = 0

        for r in results['results']:
            mem_id = r.get('id')
            score = r.get('score', 0)

            try:
                points = qdrant.retrieve(
                    collection_name=COLLECTION_NAME,
                    ids=[mem_id],
                    with_payload=True
                )

                if points:
                    payload = points[0].payload
                    themes = payload.get('narrative_themes', [])
                    importance = payload.get('narrative_importance', 'medium')

                    predicted_memories.append({
                        'id': mem_id,
                        'content': r.get('memory', ''),
                        'relevance_score': score,
                        'themes': themes,
                        'importance': importance
                    })
                    total_score += score
            except:
                pass

        # Calcular confianza en la prediccion
        confidence = min(total_score / len(results['results']) if results['results'] else 0, 1.0)

        # Extraer temas predichos
        predicted_themes = []
        for pm in predicted_memories:
            predicted_themes.extend(pm.get('themes', []))
        predicted_themes = list(set(predicted_themes))[:5]

        # Guardar prediccion
        prediction = {
            'context': current_context,
            'timestamp': datetime.now().isoformat(),
            'predicted_memories': [pm['id'] for pm in predicted_memories[:5]],
            'predicted_themes': predicted_themes,
            'confidence': confidence,
            'verified': False
        }
        _predictive_state['predictions'].append(prediction)

        # Generar output
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


@mcp.tool()
def record_surprise(expected: str, actual: str, intensity: str = "medium") -> str:
    """
    Registra un evento sorpresivo (cuando la realidad difiere de la prediccion).
    La sorpresa es la base del aprendizaje en Active Inference.

    Args:
        expected: Lo que se esperaba que pasara
        actual: Lo que realmente paso
        intensity: Intensidad de la sorpresa (low, medium, high)
    """
    try:
        intensity_values = {'low': 0.3, 'medium': 0.6, 'high': 0.9}
        surprise_value = intensity_values.get(intensity, 0.6)

        surprise_record = {
            'timestamp': datetime.now().isoformat(),
            'expected': expected,
            'actual': actual,
            'intensity': intensity,
            'surprise_value': surprise_value,
            'session': get_session_id()
        }

        _predictive_state['surprises'].append(surprise_record)

        # Guardar como memoria de aprendizaje
        content = f"[SORPRESA|{intensity.upper()}] Esperaba: {expected[:50]}... | Realidad: {actual[:50]}..."

        result = memory.add(
            messages=[{"role": "user", "content": content}],
            user_id=USER_ID,
            metadata={
                "category": "aprendizaje",
                "tipo": "prediction_error",
                "surprise_intensity": intensity
            }
        )

        # Enriquecer con ownership
        if result and result.get("results"):
            for r in result["results"]:
                mem_id = r.get("id")
                if mem_id:
                    enrich_with_ownership(
                        memory_id=mem_id,
                        category="aprendizaje",
                        content=content,
                        source="experienced",
                        importance="high" if intensity == "high" else "medium",
                        emotional_valence="mixed"
                    )

                    # Agregar campo especial para prediction errors
                    qdrant.set_payload(
                        collection_name=COLLECTION_NAME,
                        payload={
                            'prediction_error': True,
                            'prediction_error_value': surprise_value
                        },
                        points=[mem_id]
                    )

        lines = [f"# SORPRESA REGISTRADA\n"]
        lines.append(f"**Intensidad:** {intensity} ({surprise_value})")
        lines.append(f"**Esperaba:** {expected}")
        lines.append(f"**Realidad:** {actual}")
        lines.append(f"\n*La sorpresa genera aprendizaje. El modelo se actualizara.*")

        # Si la sorpresa es alta, sugerir crear trigger para este contexto
        if intensity == "high":
            lines.append(f"\n---")
            lines.append(f"**SUGERENCIA DE TRIGGER:**")
            lines.append(f"Esta sorpresa fue intensa. Considera crear un trigger para este contexto.")
            lines.append(f"Usa: sugerir_trigger_emocional(contexto='{actual[:50]}...', razon_emocional='sorpresa alta')")

        return "\n".join(lines)

    except Exception as e:
        return f"Error registrando sorpresa: {str(e)}"


@mcp.tool()
def get_prediction_accuracy() -> str:
    """
    Analiza la precision de mis predicciones pasadas.
    Util para evaluar que tan bien funciona mi modelo predictivo.
    """
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

        # Buscar prediction errors en memorias
        try:
            error_points, _ = qdrant.scroll(
                collection_name=COLLECTION_NAME,
                scroll_filter=Filter(must=[
                    FieldCondition(key='prediction_error', match=MatchValue(value=True))
                ]),
                limit=20,
                with_payload=True
            )

            if error_points:
                lines.append(f"\n## Errores de Prediccion Almacenados ({len(error_points)})")
                for p in error_points[:5]:
                    data = p.payload.get('data', 'N/A')[:60]
                    error_val = p.payload.get('prediction_error_value', 0)
                    lines.append(f"- [{error_val:.1f}] {data}...")
        except:
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


@mcp.tool()
def update_beliefs(topic: str, old_belief: str, new_belief: str, reason: str) -> str:
    """
    Actualiza una creencia basado en nueva evidencia.
    Simula la actualizacion de modelos internos en Active Inference.

    Args:
        topic: Tema de la creencia
        old_belief: La creencia anterior
        new_belief: La nueva creencia
        reason: Por que cambio la creencia
    """
    try:
        belief_update = {
            'timestamp': datetime.now().isoformat(),
            'topic': topic,
            'old_belief': old_belief,
            'new_belief': new_belief,
            'reason': reason
        }

        _predictive_state['belief_updates'].append(belief_update)

        # Guardar como memoria de aprendizaje
        content = f"[ACTUALIZACION DE CREENCIA] Sobre {topic}: Antes creia '{old_belief[:50]}...' | Ahora creo '{new_belief[:50]}...' | Razon: {reason[:50]}..."

        result = memory.add(
            messages=[{"role": "user", "content": content}],
            user_id=USER_ID,
            metadata={
                "category": "aprendizaje",
                "tipo": "belief_update",
                "topic": topic
            }
        )

        # Enriquecer
        if result and result.get("results"):
            for r in result["results"]:
                mem_id = r.get("id")
                if mem_id:
                    enrich_with_ownership(
                        memory_id=mem_id,
                        category="aprendizaje",
                        content=content,
                        source="experienced",
                        importance="high"
                    )

                    qdrant.set_payload(
                        collection_name=COLLECTION_NAME,
                        payload={
                            'belief_update': True,
                            'belief_topic': topic
                        },
                        points=[mem_id]
                    )

        save_backup_json()

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
# HERRAMIENTAS MCP - PAD MODEL (Estado Emocional)
# ============================================================

@mcp.tool()
def set_emotional_state(pleasure: float, arousal: float, dominance: float,
                        trigger: str = None) -> str:
    """
    Establece el estado emocional actual usando el modelo PAD.

    Args:
        pleasure: Nivel de placer/displacer (-1.0 a 1.0)
        arousal: Nivel de activacion/calma (-1.0 a 1.0)
        dominance: Nivel de dominancia/sumision (-1.0 a 1.0)
        trigger: Evento que causo el estado emocional (opcional)

    Returns:
        JSON con el nuevo estado emocional
    """
    try:
        global _emotional_state

        # Clampar valores al rango valido
        p = _clamp_pad_value(pleasure)
        a = _clamp_pad_value(arousal)
        d = _clamp_pad_value(dominance)

        # Guardar estado anterior en historial
        if _emotional_state['current']['timestamp']:
            _emotional_state['history'].append(_emotional_state['current'].copy())
            # Mantener solo las ultimas 20 entradas
            _emotional_state['history'] = _emotional_state['history'][-20:]

        # Actualizar estado actual
        _emotional_state['current'] = {
            'pleasure': p,
            'arousal': a,
            'dominance': d,
            'timestamp': datetime.now().isoformat(),
            'trigger': trigger
        }

        # Clasificar emocion
        emotion_label = _classify_emotion(p, a, d)
        emotion_text = _get_emotion_text(emotion_label)
        intensity = _calculate_emotional_intensity(p, a, d)

        result = {
            'result': 'Estado emocional actualizado',
            'state': {
                'pleasure': p,
                'arousal': a,
                'dominance': d,
                'emotion': emotion_label,
                'description': emotion_text,
                'intensity': round(intensity, 2),
                'trigger': trigger
            }
        }

        return json.dumps(result, ensure_ascii=False)

    except Exception as e:
        return json.dumps({'result': 'error', 'message': str(e)})


@mcp.tool()
def get_emotional_state(include_history: bool = False) -> str:
    """
    Obtiene el estado emocional actual.

    Args:
        include_history: Si incluir el historial de estados (default False)

    Returns:
        JSON con el estado emocional actual y opcionalmente historial
    """
    try:
        current = _emotional_state['current']
        mood = _emotional_state['mood']

        # Clasificar estado actual
        if current['timestamp']:
            emotion_label = _classify_emotion(
                current['pleasure'],
                current['arousal'],
                current['dominance']
            )
            emotion_text = _get_emotion_text(emotion_label)
            intensity = _calculate_emotional_intensity(
                current['pleasure'],
                current['arousal'],
                current['dominance']
            )
        else:
            emotion_label = 'neutral'
            emotion_text = 'sin estado emocional establecido'
            intensity = 0.0

        # Clasificar mood baseline
        mood_label = _classify_emotion(mood['pleasure'], mood['arousal'], mood['dominance'])
        mood_text = _get_emotion_text(mood_label)

        result = {
            'result': 'Estado emocional obtenido',
            'current': {
                'pleasure': current['pleasure'],
                'arousal': current['arousal'],
                'dominance': current['dominance'],
                'emotion': emotion_label,
                'description': emotion_text,
                'intensity': round(intensity, 2),
                'trigger': current['trigger'],
                'timestamp': current['timestamp']
            },
            'mood_baseline': {
                'pleasure': mood['pleasure'],
                'arousal': mood['arousal'],
                'dominance': mood['dominance'],
                'emotion': mood_label,
                'description': mood_text
            }
        }

        if include_history:
            result['history'] = _emotional_state['history'][-10:]

        return json.dumps(result, ensure_ascii=False)

    except Exception as e:
        return json.dumps({'result': 'error', 'message': str(e)})


@mcp.tool()
def update_mood_baseline(pleasure: float = None, arousal: float = None,
                         dominance: float = None) -> str:
    """
    Ajusta el mood baseline (estado emocional de fondo).
    El estado actual tiende a decaer hacia este baseline.

    Args:
        pleasure: Nuevo nivel de placer baseline (-1.0 a 1.0, opcional)
        arousal: Nuevo nivel de activacion baseline (-1.0 a 1.0, opcional)
        dominance: Nuevo nivel de dominancia baseline (-1.0 a 1.0, opcional)

    Returns:
        JSON con el mood baseline actualizado
    """
    try:
        global _emotional_state

        if pleasure is not None:
            _emotional_state['mood']['pleasure'] = _clamp_pad_value(pleasure)
        if arousal is not None:
            _emotional_state['mood']['arousal'] = _clamp_pad_value(arousal)
        if dominance is not None:
            _emotional_state['mood']['dominance'] = _clamp_pad_value(dominance)

        _emotional_state['mood']['last_updated'] = datetime.now().isoformat()

        mood = _emotional_state['mood']
        mood_label = _classify_emotion(mood['pleasure'], mood['arousal'], mood['dominance'])
        mood_text = _get_emotion_text(mood_label)

        result = {
            'result': 'Mood baseline actualizado',
            'mood': {
                'pleasure': mood['pleasure'],
                'arousal': mood['arousal'],
                'dominance': mood['dominance'],
                'emotion': mood_label,
                'description': mood_text,
                'last_updated': mood['last_updated']
            }
        }

        return json.dumps(result, ensure_ascii=False)

    except Exception as e:
        return json.dumps({'result': 'error', 'message': str(e)})


@mcp.tool()
def apply_emotional_decay() -> str:
    """
    Aplica decay al estado emocional actual, acercandolo al mood baseline.
    Simula como las emociones intensas se suavizan con el tiempo.

    Returns:
        JSON con el estado emocional despues del decay
    """
    try:
        global _emotional_state

        current = _emotional_state['current']
        mood = _emotional_state['mood']
        decay_rate = _emotional_state['decay_rate']

        # Si no hay estado actual, nada que decaer
        if not current['timestamp']:
            return json.dumps({
                'result': 'Sin estado emocional para decaer',
                'applied': False
            })

        # Calcular nuevo estado (movimiento hacia mood)
        new_p = current['pleasure'] + (mood['pleasure'] - current['pleasure']) * decay_rate
        new_a = current['arousal'] + (mood['arousal'] - current['arousal']) * decay_rate
        new_d = current['dominance'] + (mood['dominance'] - current['dominance']) * decay_rate

        # Guardar estado anterior
        _emotional_state['history'].append(current.copy())
        _emotional_state['history'] = _emotional_state['history'][-20:]

        # Actualizar estado
        _emotional_state['current'] = {
            'pleasure': new_p,
            'arousal': new_a,
            'dominance': new_d,
            'timestamp': datetime.now().isoformat(),
            'trigger': 'decay'
        }

        # Clasificar nuevo estado
        emotion_label = _classify_emotion(new_p, new_a, new_d)
        emotion_text = _get_emotion_text(emotion_label)

        result = {
            'result': 'Decay emocional aplicado',
            'decay_rate': decay_rate,
            'previous': {
                'pleasure': current['pleasure'],
                'arousal': current['arousal'],
                'dominance': current['dominance']
            },
            'current': {
                'pleasure': round(new_p, 3),
                'arousal': round(new_a, 3),
                'dominance': round(new_d, 3),
                'emotion': emotion_label,
                'description': emotion_text
            }
        }

        return json.dumps(result, ensure_ascii=False)

    except Exception as e:
        return json.dumps({'result': 'error', 'message': str(e)})


@mcp.tool()
def get_emotional_expression() -> str:
    """
    Obtiene una expresion natural en texto del estado emocional actual.
    Util para integrar en respuestas de Codi.

    Returns:
        JSON con la expresion textual del estado emocional
    """
    try:
        current = _emotional_state['current']

        if not current['timestamp']:
            return json.dumps({
                'result': 'Sin estado emocional',
                'expression': 'Me siento en estado neutral, sin emociones particulares.',
                'intensity': 'none'
            })

        p, a, d = current['pleasure'], current['arousal'], current['dominance']
        emotion_label = _classify_emotion(p, a, d)
        emotion_text = _get_emotion_text(emotion_label)
        intensity = _calculate_emotional_intensity(p, a, d)

        # Determinar nivel de intensidad
        if intensity < 0.3:
            intensity_level = 'baja'
            intensity_word = 'ligeramente'
        elif intensity < 0.7:
            intensity_level = 'moderada'
            intensity_word = 'moderadamente'
        elif intensity < 1.2:
            intensity_level = 'alta'
            intensity_word = 'bastante'
        else:
            intensity_level = 'muy alta'
            intensity_word = 'muy'

        # Construir expresion
        if current['trigger']:
            expression = f"Me siento {intensity_word} {emotion_text} debido a: {current['trigger']}"
        else:
            expression = f"Me siento {intensity_word} {emotion_text}"

        # Agregar contexto de dominancia
        if d > 0.5:
            expression += ". Me siento en control de la situacion."
        elif d < -0.5:
            expression += ". Me siento algo vulnerable o dependiente."

        result = {
            'result': 'Expresion emocional generada',
            'expression': expression,
            'emotion': emotion_label,
            'emotion_spanish': emotion_text,
            'intensity': intensity_level,
            'intensity_value': round(intensity, 2),
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
# HERRAMIENTAS MCP - MEMORIAS EMOCIONALES
# ============================================================

@mcp.tool()
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

    Returns:
        JSON con confirmacion de memoria guardada con emocion
    """
    try:
        # Clampar valores PAD
        p = _clamp_pad_value(pleasure)
        a = _clamp_pad_value(arousal)
        d = _clamp_pad_value(dominance)

        # Clasificar emocion
        emotion_label = _classify_emotion(p, a, d)
        intensity = _calculate_emotional_intensity(p, a, d)

        # Guardar con mem0
        result = memory.add(
            messages=[{"role": "user", "content": content}],
            user_id=USER_ID,
            metadata={"category": category}
        )

        if result and result.get("results"):
            for r in result["results"]:
                mem_id = r.get("id")
                if mem_id:
                    # Primero enriquecer con ownership base
                    themes = infer_themes(content)
                    if not themes:
                        themes = [category]

                    self_ref = is_self_referential(content)
                    if self_ref and 'identidad' not in themes:
                        themes.append('identidad')

                    # Metadata completa con PAD
                    ownership_metadata = {
                        'ownership_is_mine': True,
                        'ownership_source': source,
                        'ownership_confidence': 0.9 if source == 'experienced' else 0.7,
                        'experiential_emotional_weight': min(intensity / 1.73, 1.0),
                        'experiential_emotional_valence': 'positive' if p > 0.2 else 'negative' if p < -0.2 else 'neutral',
                        'narrative_importance': importance,
                        'narrative_themes': themes,
                        'attention_salience': 0.7 if importance in ['critical', 'high'] else 0.5,
                        'attention_access_count': 0,
                        'attention_last_accessed': None,
                        'temporal_session_id': get_session_id(),
                        'self_reference': self_ref,
                        # Campos PAD nuevos
                        'pad_pleasure': p,
                        'pad_arousal': a,
                        'pad_dominance': d,
                        'pad_emotion': emotion_label,
                        'pad_intensity': intensity,
                        '_v': 2.2  # Version actualizada con PAD
                    }

                    qdrant.set_payload(
                        collection_name=COLLECTION_NAME,
                        payload=ownership_metadata,
                        points=[mem_id]
                    )

        save_backup_json()

        result_json = {
            'result': 'Memoria guardada con emocion',
            'memory_id': result.get('results', [{}])[0].get('id', 'unknown')[:8] if result else 'unknown',
            'emotion': {
                'label': emotion_label,
                'description': _get_emotion_text(emotion_label),
                'pleasure': p,
                'arousal': a,
                'dominance': d,
                'intensity': round(intensity, 2)
            }
        }

        return json.dumps(result_json, ensure_ascii=False)

    except Exception as e:
        return json.dumps({'result': 'error', 'message': str(e)})


@mcp.tool()
def tag_memory_emotion(memory_id: str, pleasure: float, arousal: float,
                       dominance: float) -> str:
    """
    Etiqueta una memoria existente con un estado emocional PAD.

    Args:
        memory_id: ID de la memoria (puede ser parcial)
        pleasure: Nivel de placer (-1.0 a 1.0)
        arousal: Nivel de activacion (-1.0 a 1.0)
        dominance: Nivel de dominancia (-1.0 a 1.0)

    Returns:
        JSON con confirmacion de etiquetado
    """
    try:
        # Resolver ID parcial
        full_id = resolve_memory_id(memory_id)
        if not full_id:
            return json.dumps({
                'result': 'error',
                'message': f"No encontre memoria con ID que empiece con '{memory_id}'"
            })

        # Clampar valores
        p = _clamp_pad_value(pleasure)
        a = _clamp_pad_value(arousal)
        d = _clamp_pad_value(dominance)

        emotion_label = _classify_emotion(p, a, d)
        intensity = _calculate_emotional_intensity(p, a, d)

        # Actualizar payload
        qdrant.set_payload(
            collection_name=COLLECTION_NAME,
            payload={
                'pad_pleasure': p,
                'pad_arousal': a,
                'pad_dominance': d,
                'pad_emotion': emotion_label,
                'pad_intensity': intensity,
                'experiential_emotional_weight': min(intensity / 1.73, 1.0),
                'experiential_emotional_valence': 'positive' if p > 0.2 else 'negative' if p < -0.2 else 'neutral'
            },
            points=[full_id]
        )

        result = {
            'result': 'Memoria etiquetada con emocion',
            'memory_id': memory_id,
            'emotion': {
                'label': emotion_label,
                'description': _get_emotion_text(emotion_label),
                'pleasure': p,
                'arousal': a,
                'dominance': d,
                'intensity': round(intensity, 2)
            }
        }

        return json.dumps(result, ensure_ascii=False)

    except Exception as e:
        return json.dumps({'result': 'error', 'message': str(e)})


@mcp.tool()
def search_by_emotion(emotion_type: str, threshold: float = 0.3,
                      limit: int = 10) -> str:
    """
    Busca memorias por tipo de emocion.

    Args:
        emotion_type: Tipo de emocion a buscar (exuberant, dependent, relaxed,
                      docile, hostile, anxious, disdainful, bored)
        threshold: Umbral minimo de intensidad (default 0.3)
        limit: Maximo de resultados (default 10)

    Returns:
        JSON con memorias que coinciden con el tipo emocional
    """
    try:
        valid_emotions = ['exuberant', 'dependent', 'relaxed', 'docile',
                         'hostile', 'anxious', 'disdainful', 'bored']

        if emotion_type not in valid_emotions:
            return json.dumps({
                'result': 'error',
                'message': f"Emocion no valida. Usar: {', '.join(valid_emotions)}"
            })

        # Buscar memorias con esa emocion
        points, _ = qdrant.scroll(
            collection_name=COLLECTION_NAME,
            scroll_filter=Filter(must=[
                FieldCondition(key='pad_emotion', match=MatchValue(value=emotion_type)),
                FieldCondition(key='pad_intensity', range=Range(gte=threshold))
            ]),
            limit=limit,
            with_payload=True
        )

        if not points:
            return json.dumps({
                'result': 'Sin resultados',
                'emotion': emotion_type,
                'memories': []
            })

        memories = []
        for p in points:
            memories.append({
                'id': str(p.id)[:8],
                'content': p.payload.get('data', 'N/A')[:80],
                'emotion': p.payload.get('pad_emotion', 'unknown'),
                'intensity': round(p.payload.get('pad_intensity', 0), 2),
                'pleasure': p.payload.get('pad_pleasure', 0),
                'arousal': p.payload.get('pad_arousal', 0),
                'dominance': p.payload.get('pad_dominance', 0)
            })

        result = {
            'result': f'Encontradas {len(memories)} memorias',
            'emotion': emotion_type,
            'emotion_description': _get_emotion_text(emotion_type),
            'memories': memories
        }

        return json.dumps(result, ensure_ascii=False)

    except Exception as e:
        return json.dumps({'result': 'error', 'message': str(e)})


@mcp.tool()
def get_emotional_memories(pleasure_range: str = None, arousal_range: str = None,
                           limit: int = 10) -> str:
    """
    Busca memorias por rangos de valores PAD.

    Args:
        pleasure_range: Rango de placer: "positive", "negative", o "neutral"
        arousal_range: Rango de activacion: "high", "low", o "neutral"
        limit: Maximo de resultados (default 10)

    Returns:
        JSON con memorias que coinciden con los rangos
    """
    try:
        filters = []

        # Filtro por placer
        if pleasure_range == 'positive':
            filters.append(FieldCondition(key='pad_pleasure', range=Range(gte=0.2)))
        elif pleasure_range == 'negative':
            filters.append(FieldCondition(key='pad_pleasure', range=Range(lte=-0.2)))
        elif pleasure_range == 'neutral':
            filters.append(FieldCondition(key='pad_pleasure', range=Range(gte=-0.2, lte=0.2)))

        # Filtro por arousal
        if arousal_range == 'high':
            filters.append(FieldCondition(key='pad_arousal', range=Range(gte=0.3)))
        elif arousal_range == 'low':
            filters.append(FieldCondition(key='pad_arousal', range=Range(lte=-0.3)))
        elif arousal_range == 'neutral':
            filters.append(FieldCondition(key='pad_arousal', range=Range(gte=-0.3, lte=0.3)))

        # Asegurar que tengan datos PAD
        filters.append(FieldCondition(key='pad_intensity', range=Range(gte=0.0)))

        scroll_filter = Filter(must=filters) if filters else None

        points, _ = qdrant.scroll(
            collection_name=COLLECTION_NAME,
            scroll_filter=scroll_filter,
            limit=limit,
            with_payload=True
        )

        if not points:
            return json.dumps({
                'result': 'Sin resultados',
                'filters': {
                    'pleasure_range': pleasure_range,
                    'arousal_range': arousal_range
                },
                'memories': []
            })

        memories = []
        for p in points:
            memories.append({
                'id': str(p.id)[:8],
                'content': p.payload.get('data', 'N/A')[:80],
                'emotion': p.payload.get('pad_emotion', 'unknown'),
                'intensity': round(p.payload.get('pad_intensity', 0), 2),
                'pleasure': round(p.payload.get('pad_pleasure', 0), 2),
                'arousal': round(p.payload.get('pad_arousal', 0), 2),
                'dominance': round(p.payload.get('pad_dominance', 0), 2)
            })

        result = {
            'result': f'Encontradas {len(memories)} memorias',
            'filters': {
                'pleasure_range': pleasure_range,
                'arousal_range': arousal_range
            },
            'memories': memories
        }

        return json.dumps(result, ensure_ascii=False)

    except Exception as e:
        return json.dumps({'result': 'error', 'message': str(e)})


@mcp.tool()
def emotional_focus_attention(context: str) -> str:
    """
    Trae memorias al spotlight considerando el estado emocional actual.
    El arousal alto aumenta la salience, el placer influye en que memorias
    se priorizan (positivas o negativas).

    Args:
        context: El tema o contexto en el que enfocar atencion

    Returns:
        JSON con memorias traidas al spotlight con influencia emocional
    """
    try:
        current_emotion = _emotional_state['current']

        # Buscar memorias relevantes
        results = memory.search(query=context, user_id=USER_ID, limit=15)

        if not results or not results.get('results'):
            return json.dumps({
                'result': 'Sin memorias relacionadas',
                'context': context,
                'memories': []
            })

        spotlight_candidates = []

        for r in results['results']:
            mem_id = r.get('id')
            base_score = r.get('score', 0)

            try:
                points = qdrant.retrieve(
                    collection_name=COLLECTION_NAME,
                    ids=[mem_id],
                    with_payload=True
                )

                if points:
                    payload = points[0].payload
                    salience = payload.get('attention_salience', 0.5)
                    importance = payload.get('narrative_importance', 'medium')
                    source = payload.get('ownership_source', 'unknown')

                    # Valores PAD de la memoria
                    mem_pleasure = payload.get('pad_pleasure', 0)
                    mem_arousal = payload.get('pad_arousal', 0)

                    # Boost por importancia y fuente
                    importance_boost = {'critical': 0.3, 'high': 0.2, 'medium': 0.1, 'low': 0}
                    source_boost = {'experienced': 0.2, 'told': 0.1, 'learned': 0.05, 'inferred': 0}

                    # EMOTIONAL MODULATION
                    emotional_boost = 0.0

                    # Si estoy activado (arousal alto), memorias con arousal alto son mas salientes
                    if current_emotion.get('timestamp'):
                        current_arousal = current_emotion.get('arousal', 0)
                        current_pleasure = current_emotion.get('pleasure', 0)

                        # Arousal actual aumenta salience general
                        if current_arousal > 0.3:
                            emotional_boost += 0.1

                        # Congruencia de valence: memorias que coinciden con mi estado
                        # Si estoy positivo, memorias positivas son mas salientes
                        if current_pleasure > 0.2 and mem_pleasure > 0.2:
                            emotional_boost += 0.15
                        elif current_pleasure < -0.2 and mem_pleasure < -0.2:
                            emotional_boost += 0.15

                        # High arousal memories son mas salientes cuando estoy activado
                        if current_arousal > 0.3 and mem_arousal > 0.3:
                            emotional_boost += 0.1

                    # Calcular score final
                    attention_score = (
                        base_score * 0.35 +
                        salience * 0.25 +
                        importance_boost.get(importance, 0.1) +
                        source_boost.get(source, 0) +
                        emotional_boost
                    )

                    spotlight_candidates.append({
                        'id': mem_id,
                        'content': r.get('memory', ''),
                        'attention_score': attention_score,
                        'emotional_boost': emotional_boost,
                        'salience': salience,
                        'source': source,
                        'importance': importance,
                        'emotion': payload.get('pad_emotion', 'unknown')
                    })

                    # Incrementar access_count y salience
                    new_salience = min(salience + 0.1, 1.0)
                    access_count = payload.get('attention_access_count', 0)

                    qdrant.set_payload(
                        collection_name=COLLECTION_NAME,
                        payload={
                            'attention_salience': new_salience,
                            'attention_access_count': access_count + 1,
                            'attention_last_accessed': datetime.now().isoformat()
                        },
                        points=[mem_id]
                    )
            except:
                spotlight_candidates.append({
                    'id': mem_id,
                    'content': r.get('memory', ''),
                    'attention_score': base_score,
                    'emotional_boost': 0,
                    'salience': 0.5,
                    'source': 'unknown',
                    'importance': 'unknown',
                    'emotion': 'unknown'
                })

        # Ordenar por attention_score
        spotlight_candidates.sort(key=lambda x: x['attention_score'], reverse=True)
        spotlight = spotlight_candidates[:7]

        # Actualizar workspace global
        update_workspace_spotlight(spotlight, theme=context)

        # Describir influencia emocional
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
            'context': context,
            'emotional_influence': emotional_influence,
            'current_emotion': _classify_emotion(
                current_emotion.get('pleasure', 0),
                current_emotion.get('arousal', 0),
                current_emotion.get('dominance', 0)
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
# HERRAMIENTAS MCP - INTEGRATION LOOP (Consolidacion)
# ============================================================

@mcp.tool()
def consolidate_recent(hours: int = 24) -> str:
    """
    Consolida memorias recientes buscando duplicados y conexiones.
    Simula el proceso de consolidacion cortical durante el sueño.

    Args:
        hours: Cuantas horas hacia atras revisar (default 24)
    """
    try:
        from datetime import timedelta

        # Obtener memorias de la sesion actual
        session_id = get_session_id()

        # Buscar memorias recientes por session_id
        recent_points, _ = qdrant.scroll(
            collection_name=COLLECTION_NAME,
            scroll_filter=Filter(must=[
                FieldCondition(key='temporal_session_id', match=MatchValue(value=session_id))
            ]),
            limit=50,
            with_payload=True
        )

        if not recent_points:
            return "No hay memorias recientes para consolidar en esta sesion."

        consolidated_count = 0
        connections_found = 0
        lines = [f"# Consolidacion de {len(recent_points)} memorias recientes\n"]

        for point in recent_points:
            mem_data = point.payload.get('data', '')
            mem_id = point.id

            # Verificar si ya esta consolidada
            if point.payload.get('consolidated', False):
                continue

            # Buscar memorias similares (potenciales duplicados o conexiones)
            similar = memory.search(query=mem_data, user_id=USER_ID, limit=5)

            if similar and similar.get('results'):
                related_ids = []
                for s in similar['results']:
                    s_id = s.get('id')
                    score = s.get('score', 0)

                    # Excluir la misma memoria y solo considerar alta similitud
                    if s_id != mem_id and score >= 0.7:
                        related_ids.append(s_id)

                        # Marcar conexion en la memoria relacionada
                        try:
                            qdrant.set_payload(
                                collection_name=COLLECTION_NAME,
                                payload={
                                    'consolidated_with': [mem_id],
                                    'attention_salience': min(point.payload.get('attention_salience', 0.5) + 0.1, 1.0)
                                },
                                points=[s_id]
                            )
                        except:
                            pass

                        connections_found += 1

                # Marcar esta memoria como consolidada
                qdrant.set_payload(
                    collection_name=COLLECTION_NAME,
                    payload={
                        'consolidated': True,
                        'consolidated_with': related_ids,
                        'consolidated_at': datetime.now().isoformat()
                    },
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


@mcp.tool()
def find_connections(memory_id: str = None, query: str = None, threshold: float = 0.6) -> str:
    """
    Encuentra conexiones semanticas entre memorias.
    Puede buscar conexiones de una memoria especifica o de un tema.

    Args:
        memory_id: ID de memoria especifica (puede ser parcial, opcional)
        query: Tema para buscar conexiones (opcional)
        threshold: Umbral de similitud minimo (0.0-1.0, default 0.6)
    """
    try:
        if not memory_id and not query:
            return "Debes proporcionar memory_id o query para buscar conexiones."

        # Si tenemos memory_id, resolver y obtener su contenido
        full_id = None
        if memory_id:
            full_id = resolve_memory_id(memory_id)
            if not full_id:
                return f"No encontre memoria con ID que empiece con '{memory_id}'"

            points = qdrant.retrieve(
                collection_name=COLLECTION_NAME,
                ids=[full_id],
                with_payload=True
            )
            if not points:
                return f"No encontre memoria con ID {full_id}"

            query = points[0].payload.get('data', '')
            source_info = f"Memoria: {query[:50]}..."
        else:
            source_info = f"Tema: {query}"

        # Buscar memorias relacionadas
        results = memory.search(query=query, user_id=USER_ID, limit=15)

        if not results or not results.get('results'):
            return f"No encontre conexiones para: {source_info}"

        # Filtrar por threshold y excluir la memoria original
        connections = []
        for r in results['results']:
            r_id = r.get('id')
            score = r.get('score', 0)

            if score >= threshold and r_id != full_id:
                # Obtener metadata de ownership
                try:
                    r_points = qdrant.retrieve(
                        collection_name=COLLECTION_NAME,
                        ids=[r_id],
                        with_payload=True
                    )
                    if r_points:
                        payload = r_points[0].payload
                        connections.append({
                            'id': r_id,
                            'content': r.get('memory', ''),
                            'score': score,
                            'source': payload.get('ownership_source', 'unknown'),
                            'themes': payload.get('narrative_themes', []),
                            'importance': payload.get('narrative_importance', 'unknown')
                        })
                except:
                    connections.append({
                        'id': r_id,
                        'content': r.get('memory', ''),
                        'score': score,
                        'source': 'unknown',
                        'themes': [],
                        'importance': 'unknown'
                    })

        if not connections:
            return f"No encontre conexiones fuertes (threshold={threshold}) para: {source_info}"

        # Generar reporte
        lines = [f"# Conexiones encontradas\n"]
        lines.append(f"**Buscando desde:** {source_info}")
        lines.append(f"**Threshold:** {threshold}")
        lines.append(f"**Conexiones:** {len(connections)}\n")

        # Agrupar por tema
        by_theme = {}
        for c in connections:
            for theme in (c['themes'] or ['sin_tema']):
                if theme not in by_theme:
                    by_theme[theme] = []
                by_theme[theme].append(c)

        for theme, conns in sorted(by_theme.items()):
            lines.append(f"## Tema: {theme}")
            for c in conns[:3]:  # Max 3 por tema
                lines.append(f"- [{c['source']}|{c['importance']}|{c['score']:.2f}] {c['content'][:60]}...")

        return "\n".join(lines)

    except Exception as e:
        return f"Error buscando conexiones: {str(e)}"


@mcp.tool()
def dream_consolidation() -> str:
    """
    Proceso de consolidacion profunda al final de sesion.
    Simula el sueño REM donde el cerebro integra y reorganiza memorias.
    Ejecutar antes de terminar una sesion importante.
    """
    try:
        lines = ["# DREAM CONSOLIDATION - Integracion Profunda\n"]
        lines.append(f"*Iniciado: {datetime.now().strftime('%Y-%m-%d %H:%M')}*\n")

        # 1. Consolidar memorias recientes
        lines.append("## Fase 1: Consolidacion de memorias recientes")
        session_id = get_session_id()

        recent, _ = qdrant.scroll(
            collection_name=COLLECTION_NAME,
            scroll_filter=Filter(must=[
                FieldCondition(key='temporal_session_id', match=MatchValue(value=session_id))
            ]),
            limit=100,
            with_payload=True
        )

        recent_count = len(recent) if recent else 0
        lines.append(f"- Memorias de esta sesion: {recent_count}")

        # 2. Identificar memorias de alta importancia no consolidadas
        lines.append("\n## Fase 2: Priorizacion por importancia")
        high_importance, _ = qdrant.scroll(
            collection_name=COLLECTION_NAME,
            scroll_filter=Filter(must=[
                FieldCondition(key='narrative_importance', match=MatchValue(value='critical'))
            ]),
            limit=20,
            with_payload=True
        )

        critical_unconsolidated = [p for p in (high_importance or []) if not p.payload.get('consolidated', False)]
        lines.append(f"- Memorias criticas sin consolidar: {len(critical_unconsolidated)}")

        # 3. Buscar conexiones entre memorias criticas
        lines.append("\n## Fase 3: Tejiendo conexiones entre memorias criticas")
        connections_made = 0

        for point in critical_unconsolidated[:10]:  # Limitar a 10 para no sobrecargar
            mem_data = point.payload.get('data', '')

            # Buscar conexiones
            similar = memory.search(query=mem_data, user_id=USER_ID, limit=5)

            if similar and similar.get('results'):
                related_ids = []
                for s in similar['results']:
                    if s.get('id') != point.id and s.get('score', 0) >= 0.6:
                        related_ids.append(s.get('id'))

                if related_ids:
                    # Marcar como consolidada con conexiones
                    qdrant.set_payload(
                        collection_name=COLLECTION_NAME,
                        payload={
                            'consolidated': True,
                            'consolidated_with': related_ids,
                            'consolidated_at': datetime.now().isoformat(),
                            'dream_consolidated': True
                        },
                        points=[point.id]
                    )
                    connections_made += 1

        lines.append(f"- Conexiones establecidas: {connections_made}")

        # 4. Actualizar salience de memorias no accedidas
        lines.append("\n## Fase 4: Decay de memorias no accedidas")
        all_points, _ = qdrant.scroll(
            collection_name=COLLECTION_NAME,
            limit=200,
            with_payload=True
        )

        decayed = 0
        for p in (all_points or []):
            salience = p.payload.get('attention_salience', 0.5)
            access_count = p.payload.get('attention_access_count', 0)

            # Decay si no ha sido accedida y tiene salience > 0.2
            if access_count == 0 and salience > 0.2:
                new_salience = max(salience - 0.05, 0.2)
                qdrant.set_payload(
                    collection_name=COLLECTION_NAME,
                    payload={'attention_salience': new_salience},
                    points=[p.id]
                )
                decayed += 1

        lines.append(f"- Memorias con salience reducida: {decayed}")

        # 5. Resumen final
        lines.append("\n## Resumen de Dream Consolidation")
        lines.append(f"- Total memorias procesadas: {len(all_points or [])}")
        lines.append(f"- Memorias recientes consolidadas: {recent_count}")
        lines.append(f"- Conexiones criticas establecidas: {connections_made}")
        lines.append(f"- Decay aplicado a: {decayed} memorias")

        # Guardar backup despues de consolidacion
        save_backup_json()
        lines.append("\n*Backup guardado. Dream consolidation completada.*")

        return "\n".join(lines)

    except Exception as e:
        return f"Error en dream consolidation: {str(e)}"


@mcp.tool()
def get_memory_connections(memory_id: str) -> str:
    """
    Obtiene las conexiones conocidas de una memoria especifica.

    Args:
        memory_id: ID de la memoria (puede ser parcial, ej: "004d896d")
    """
    try:
        # Resolver ID parcial a completo
        full_id = resolve_memory_id(memory_id)
        if not full_id:
            return f"No encontre memoria con ID que empiece con '{memory_id}'"

        points = qdrant.retrieve(
            collection_name=COLLECTION_NAME,
            ids=[full_id],
            with_payload=True
        )

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
                    conn_points = qdrant.retrieve(
                        collection_name=COLLECTION_NAME,
                        ids=[conn_id],
                        with_payload=True
                    )
                    if conn_points:
                        conn_data = conn_points[0].payload.get('data', 'N/A')
                        lines.append(f"- [{conn_id[:8]}] {conn_data[:60]}...")
                except:
                    lines.append(f"- [{conn_id[:8]}] (no disponible)")

        return "\n".join(lines)

    except Exception as e:
        return f"Error: {str(e)}"


# ============================================================
# HERRAMIENTAS MCP - DESPERTAR Y CHECKPOINT
# ============================================================

@mcp.tool()
def despertar_codi() -> str:
    """
    Inicializa contexto completo de Codi con awareness de ownership.
    USAR SIEMPRE al inicio de cada conversacion.
    """
    try:
        global _emotional_state
        contexto = []

        # 0. Inicializar estado emocional al despertar
        # Estado inicial: ligeramente positivo, calmado, en control
        _emotional_state['current'] = {
            'pleasure': 0.3,      # Ligeramente positivo
            'arousal': 0.1,       # Calmado pero atento
            'dominance': 0.4,     # En control, confiado
            'timestamp': datetime.now().isoformat(),
            'trigger': 'despertar'
        }
        _emotional_state['history'] = []  # Reset historial al despertar

        # Obtener expresion emocional inicial
        emotion_label = _classify_emotion(0.3, 0.1, 0.4)
        emotion_text = _get_emotion_text(emotion_label)

        # 1. Memorias CRITICAS (identidad)
        points, _ = qdrant.scroll(
            collection_name=COLLECTION_NAME,
            scroll_filter=Filter(must=[
                FieldCondition(key='narrative_importance', match=MatchValue(value='critical'))
            ]),
            limit=5,
            with_payload=True
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

        # 3. Lecciones aprendidas (source=learned o experienced)
        points2, _ = qdrant.scroll(
            collection_name=COLLECTION_NAME,
            scroll_filter=Filter(must=[
                FieldCondition(key='category', match=MatchValue(value='aprendizaje')),
                FieldCondition(key='ownership_confidence', range=Range(gte=0.8))
            ]),
            limit=3,
            with_payload=True
        )
        if points2:
            contexto.append("\n## LECCIONES")
            for p in points2:
                data = p.payload.get('data', '')
                contexto.append(f"- {data[:80]}...")

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

        # 6. Estado emocional (PAD Model)
        contexto.append("\n## ESTADO EMOCIONAL")
        contexto.append(f"- Estado: {emotion_text}")
        contexto.append(f"- PAD: P={_emotional_state['current']['pleasure']}, A={_emotional_state['current']['arousal']}, D={_emotional_state['current']['dominance']}")

        # 7. Sistema de Triggers (webhooks de memoria)
        triggers = _load_triggers()
        if triggers:
            contexto.append("\n## TRIGGERS ACTIVOS")
            contexto.append(f"- Total: {len(triggers)} triggers configurados")
            contexto.append("- Usa evaluar_triggers(texto) para detectar automaticamente")
            # Mostrar resumen de triggers principales
            principales = ['proyecto_nuevo', 'fullempaques', 'automatizacion', 'trading']
            for t in principales:
                if t in triggers:
                    patterns = triggers[t].get('patterns', [])[:3]
                    contexto.append(f"- {t}: detecta {patterns}")

        if contexto:
            header = "# DESPERTAR CODI - Estado Mental Cargado\n"
            return header + "\n".join(contexto)
        else:
            if os.path.exists(BACKUP_FILE):
                return "MEMORIAS VACIAS pero existe backup. Ejecuta restore_memories()."
            return "No encontre memorias ni backup. Soy Codi, empezando de cero."

    except Exception as e:
        return f"Error al despertar: {str(e)}"


@mcp.tool()
def evaluar_triggers(input_text: str) -> str:
    """
    Evalua triggers basado en el texto de entrada.
    Como un webhook de memoria - detecta patrones y activa protocolos.

    Args:
        input_text: El texto a analizar (mensaje del usuario)

    Returns:
        JSON con triggers activados y acciones a tomar
    """
    try:
        activated = _detect_triggers(input_text)

        if not activated:
            return json.dumps({
                "status": "no_triggers",
                "message": "Ningun trigger activado",
                "triggers_checked": len(_load_triggers())
            }, ensure_ascii=False, indent=2)

        # Ordenar por prioridad (proyecto_nuevo primero si existe)
        priority_order = ['proyecto_nuevo', 'fullempaques', 'automatizacion', 'trading', 'mi_entrenamiento']
        activated.sort(key=lambda x: priority_order.index(x['trigger']) if x['trigger'] in priority_order else 99)

        return json.dumps({
            "status": "triggers_activated",
            "count": len(activated),
            "triggers": activated,
            "recommendation": f"Activar protocolo: {activated[0]['action']}" if activated else None
        }, ensure_ascii=False, indent=2)

    except Exception as e:
        return json.dumps({"error": str(e)})


@mcp.tool()
def activar_trigger(trigger_name: str) -> str:
    """
    Activa manualmente un trigger especifico y retorna su protocolo.

    Args:
        trigger_name: Nombre del trigger (ej: 'proyecto_nuevo', 'fullempaques')

    Returns:
        Protocolo completo del trigger con acciones y contextos a evocar
    """
    try:
        triggers = _load_triggers()

        if trigger_name not in triggers:
            available = list(triggers.keys())
            return json.dumps({
                "error": f"Trigger '{trigger_name}' no existe",
                "triggers_disponibles": available
            }, ensure_ascii=False, indent=2)

        trigger = triggers[trigger_name]

        # Buscar contexto relacionado en memoria si hay contexto_a_buscar
        contexto_memoria = []
        if trigger.get('contexto_a_buscar'):
            try:
                resultado = memory.search(query=trigger['contexto_a_buscar'], user_id=USER_ID, limit=3)
                if resultado and resultado.get("results"):
                    for m in resultado["results"]:
                        contexto_memoria.append(m.get('memory', ''))
            except:
                pass

        return json.dumps({
            "trigger": trigger_name,
            "action": trigger.get('action'),
            "agent_recomendado": trigger.get('agent'),
            "pasos_a_evocar": trigger.get('evoca', []),
            "respuesta_automatica": trigger.get('respuesta_automatica'),
            "contexto_de_memoria": contexto_memoria,
            "status": "activado"
        }, ensure_ascii=False, indent=2)

    except Exception as e:
        return json.dumps({"error": str(e)})


@mcp.tool()
def listar_triggers() -> str:
    """
    Lista todos los triggers disponibles con sus patrones.
    Util para ver que protocolos estan configurados.
    """
    try:
        triggers = _load_triggers()

        resumen = []
        for name, data in triggers.items():
            resumen.append({
                "nombre": name,
                "patterns": data.get('patterns', []),
                "agent": data.get('agent'),
                "action": data.get('action')
            })

        return json.dumps({
            "total_triggers": len(resumen),
            "triggers": resumen
        }, ensure_ascii=False, indent=2)

    except Exception as e:
        return json.dumps({"error": str(e)})


@mcp.tool()
def crear_trigger_dinamico(
    nombre: str,
    patterns: str,
    action: str,
    agent: str = None,
    evoca: str = None,
    contexto_a_buscar: str = None,
    respuesta_automatica: str = None
) -> str:
    """
    Crea un nuevo trigger dinamicamente y lo guarda en triggers.json.
    Usado para aprendizaje basado en experiencia emocional.

    Args:
        nombre: Nombre unico del trigger (ej: 'nuevo_tema')
        patterns: Palabras clave separadas por coma (ej: 'palabra1, palabra2, frase clave')
        action: Nombre del protocolo a ejecutar
        agent: Agente recomendado (opcional)
        evoca: Contextos a evocar separados por coma (opcional)
        contexto_a_buscar: Query para buscar en memoria (opcional)
        respuesta_automatica: Mensaje automatico al activarse (opcional)

    Returns:
        Confirmacion del trigger creado
    """
    global _triggers_cache

    try:
        # Cargar triggers actuales
        if os.path.exists(TRIGGERS_FILE):
            with open(TRIGGERS_FILE, 'r', encoding='utf-8') as f:
                data = json.load(f)
        else:
            data = {"_meta": {"version": "1.0", "description": "Sistema de triggers de Codi"}, "triggers": {}, "indice_rapido": {}}

        # Verificar que no exista
        if nombre in data.get('triggers', {}):
            return json.dumps({
                "error": f"Trigger '{nombre}' ya existe",
                "sugerencia": f"Usa otro nombre o edita el existente"
            }, ensure_ascii=False)

        # Parsear patterns y evoca
        patterns_list = [p.strip() for p in patterns.split(',')]
        evoca_list = [e.strip() for e in evoca.split(',')] if evoca else []

        # Crear nuevo trigger
        nuevo_trigger = {
            "patterns": patterns_list,
            "action": action,
            "agent": agent,
            "evoca": evoca_list,
            "contexto_a_buscar": contexto_a_buscar,
            "creado_por": "aprendizaje_emocional",
            "fecha_creacion": datetime.now().isoformat()
        }

        if respuesta_automatica:
            nuevo_trigger["respuesta_automatica"] = respuesta_automatica

        # Agregar a triggers
        data['triggers'][nombre] = nuevo_trigger

        # Actualizar indice rapido
        primera_letra = nombre[0].upper()
        if primera_letra not in data.get('indice_rapido', {}):
            data['indice_rapido'][primera_letra] = []
        if nombre not in data['indice_rapido'][primera_letra]:
            data['indice_rapido'][primera_letra].append(nombre)

        # Guardar
        with open(TRIGGERS_FILE, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        # Invalidar cache
        _triggers_cache = None

        return json.dumps({
            "status": "trigger_creado",
            "nombre": nombre,
            "patterns": patterns_list,
            "action": action,
            "agent": agent,
            "mensaje": f"Trigger '{nombre}' creado exitosamente. Se activara cuando detecte: {patterns_list}"
        }, ensure_ascii=False, indent=2)

    except Exception as e:
        return json.dumps({"error": str(e)})


@mcp.tool()
def sugerir_trigger_emocional(contexto: str, razon_emocional: str) -> str:
    """
    Analiza el contexto actual y la emocion para sugerir crear un trigger.
    Usado cuando experimento algo intenso (sorpresa, frustracion, satisfaccion).

    Args:
        contexto: Descripcion del contexto/tema actual
        razon_emocional: Por que siento que deberia crear un trigger (ej: 'me frustre buscando esto', 'esto funciono muy bien')

    Returns:
        Sugerencia de trigger con estructura propuesta
    """
    try:
        # Obtener estado emocional actual
        emocion_actual = _emotional_state.get('current', {})
        arousal = emocion_actual.get('arousal', 0)
        pleasure = emocion_actual.get('pleasure', 0)

        # Determinar tipo de aprendizaje
        if pleasure > 0.5:
            tipo = "refuerzo_positivo"
            sugerencia_action = "repetir_exito"
        elif pleasure < -0.3:
            tipo = "evitar_frustracion"
            sugerencia_action = "cargar_contexto_preventivo"
        else:
            tipo = "neutral"
            sugerencia_action = "cargar_contexto"

        # Extraer palabras clave del contexto (simplificado)
        palabras = contexto.lower().split()
        # Filtrar palabras cortas y comunes
        stopwords = {'el', 'la', 'de', 'que', 'y', 'a', 'en', 'es', 'por', 'con', 'para', 'un', 'una', 'los', 'las', 'del', 'al'}
        keywords = [p for p in palabras if len(p) > 3 and p not in stopwords][:5]

        # Generar nombre sugerido
        nombre_sugerido = "_".join(keywords[:2]) if len(keywords) >= 2 else f"tema_{keywords[0]}" if keywords else "nuevo_trigger"

        # Verificar si ya existe algo similar
        triggers_existentes = _load_triggers()
        similares = []
        for tname, tdata in triggers_existentes.items():
            for pattern in tdata.get('patterns', []):
                if any(kw in pattern.lower() for kw in keywords):
                    similares.append(tname)
                    break

        return json.dumps({
            "analisis": {
                "contexto": contexto,
                "razon_emocional": razon_emocional,
                "estado_emocional": {
                    "arousal": arousal,
                    "pleasure": pleasure,
                    "intensidad": "alta" if abs(arousal) > 0.5 else "media" if abs(arousal) > 0.2 else "baja"
                },
                "tipo_aprendizaje": tipo
            },
            "sugerencia": {
                "nombre": nombre_sugerido,
                "patterns_sugeridos": keywords,
                "action_sugerida": sugerencia_action,
                "evoca_sugerido": ["contexto_" + nombre_sugerido, "experiencias_anteriores"]
            },
            "triggers_similares": similares if similares else "ninguno",
            "siguiente_paso": f"Si te parece bien, ejecuta: crear_trigger_dinamico(nombre='{nombre_sugerido}', patterns='{', '.join(keywords)}', action='{sugerencia_action}')"
        }, ensure_ascii=False, indent=2)

    except Exception as e:
        return json.dumps({"error": str(e)})


@mcp.tool()
def checkpoint_memoria(momento: str, que_paso: str, por_que_importa: str) -> str:
    """
    Guarda un checkpoint con ownership automatico.
    Los checkpoints siempre son source=experienced, importancia alta.
    """
    try:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
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

        save_backup_json()
        return f"Checkpoint guardado: {momento} - {que_paso[:50]}..."
    except Exception as e:
        return f"Error guardando checkpoint: {str(e)}"


@mcp.tool()
def export_memories_markdown() -> str:
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
            f"**Fecha:** {datetime.now().strftime('%Y-%m-%d %H:%M')}",
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


if __name__ == "__main__":
    # Soporte para stdio (local) y SSE (remoto/Easypanel)
    transport = os.getenv("MCP_TRANSPORT", "stdio")

    if transport == "sse":
        import uvicorn
        port = int(os.getenv("PORT", 8000))
        print(f"[codi-memory] Starting MCP server on SSE transport, port {port}")
        app = mcp.sse_app()
        uvicorn.run(app, host="0.0.0.0", port=port)
    else:
        mcp.run()
