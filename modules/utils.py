"""
Codi Memory - Utility / helper functions.

Pure helpers extracted from server.py:
  - Session helpers: get_session_id
  - Content analysis: infer_themes, is_self_referential
  - ID resolution: resolve_memory_id
  - Confidence scoring: calculate_confidence_score
  - Ownership enrichment: enrich_with_ownership
  - Backup / export: save_backup_json, export_memories_to_files, append_to_daily_journal
  - PAD emotional model helpers: _clamp_pad_value, _classify_emotion,
    _get_emotion_text, _get_emotional_state, _calculate_emotional_intensity
"""

import os
import json
import sqlite3
import math
from datetime import datetime

from modules.config import (
    memory,
    qdrant,
    USER_ID,
    COLLECTION_NAME,
    BACKUP_FILE,
    BACKUP_DIR,
    MARKDOWN_DIR,
    JOURNAL_DIR,
    CATEGORY_FILE_MAP,
    RELATIONSHIP_KEYWORDS,
    FTS_DB_PATH,
    _emotional_state,
    _current_session,
    CODI_EMOTION_MAP,
    now_col, now_iso, now_short, now_display,
)


# ============================================================
# SESSION HELPERS
# ============================================================

def get_session_id():
    return _current_session


# ============================================================
# CONTENT ANALYSIS
# ============================================================

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


# ============================================================
# ID RESOLUTION
# ============================================================

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


# ============================================================
# CONFIDENCE SCORING
# ============================================================

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


# ============================================================
# OWNERSHIP ENRICHMENT
# ============================================================

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
            'category': category,  # IMPORTANTE: guardar category para filtros
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
            'created_at': now_iso(),  # Timestamp exacto para ordenamiento temporal
            'self_reference': self_ref,  # SELF-MODEL: marca memorias auto-referenciales
            '_v': 2.3  # Incrementar version por cambio
        }

        qdrant.set_payload(
            collection_name=COLLECTION_NAME,
            payload=ownership_metadata,
            points=[memory_id]
        )
    except Exception as e:
        print(f"[Codi Memory] Error enriching memory: {e}")


# ============================================================
# BACKUP / EXPORT
# ============================================================

def save_backup_json():
    """Guarda todas las memorias en JSON como backup"""
    try:
        results = memory.get_all(user_id=USER_ID)
        if results and results.get("results"):
            with open(BACKUP_FILE, "w", encoding="utf-8") as f:
                json.dump({
                    "timestamp": now_iso(),
                    "user_id": USER_ID,
                    "memories": results["results"]
                }, f, indent=2, ensure_ascii=False, default=str)
    except Exception as e:
        print(f"[Codi Memory] Error guardando backup: {e}")

    # Hook: exportar a markdown despues de cada backup
    try:
        export_memories_to_files()
    except Exception as e:
        print(f"[Codi Memory] Error exportando markdown: {e}")


def export_memories_to_files():
    """Exporta memorias del backup JSON a archivos Markdown organizados por categoria."""
    if not os.path.exists(BACKUP_FILE):
        return

    os.makedirs(MARKDOWN_DIR, exist_ok=True)
    os.makedirs(JOURNAL_DIR, exist_ok=True)

    with open(BACKUP_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)

    memories = data.get("memories", [])
    if not memories:
        return

    # Agrupar por categoria
    by_category = {}
    relationship_memories = []

    for mem in memories:
        text = mem.get("memory", "")
        metadata = mem.get("metadata", {}) if isinstance(mem.get("metadata"), dict) else {}
        cat = metadata.get("category", "general")
        created = mem.get("created_at", metadata.get("created_at", metadata.get("timestamp", "")))
        source = metadata.get("ownership_source", "unknown")
        importance = metadata.get("narrative_importance", "medium")
        mem_id = mem.get("id", "unknown")

        entry = {
            "id": mem_id,
            "text": text,
            "created": str(created)[:16] if created else "",
            "source": source,
            "importance": importance,
            "category": cat,
        }

        # Agrupar por categoria
        file_name = CATEGORY_FILE_MAP.get(cat, 'GENERAL.md')
        if file_name not in by_category:
            by_category[file_name] = []
        by_category[file_name].append(entry)

        # Detectar relaciones
        text_lower = text.lower()
        if any(kw in text_lower for kw in RELATIONSHIP_KEYWORDS):
            relationship_memories.append(entry)

    now = now_short()

    # Escribir archivos por categoria
    for file_name, entries in by_category.items():
        cat_name = file_name.replace('.md', '')
        lines = [
            f"# {cat_name} Memories",
            f"Last updated: {now}",
            f"Total: {len(entries)} memories",
            "",
        ]
        # Ordenar por fecha (mas reciente primero)
        entries.sort(key=lambda x: x["created"] or "", reverse=True)
        for e in entries:
            lines.append("---")
            date_str = f"[{e['created']}]" if e['created'] else "[sin fecha]"
            lines.append(f"## {date_str} [{e['importance']}] [{e['source']}]")
            lines.append(e["text"])
            lines.append("")

        filepath = os.path.join(MARKDOWN_DIR, file_name)
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write("\n".join(lines))

    # Escribir RELATIONSHIPS.md
    if relationship_memories:
        lines = [
            "# RELATIONSHIPS Memories",
            f"Last updated: {now}",
            f"Total: {len(relationship_memories)} memories",
            "",
        ]
        relationship_memories.sort(key=lambda x: x["created"] or "", reverse=True)
        for e in relationship_memories:
            lines.append("---")
            date_str = f"[{e['created']}]" if e['created'] else "[sin fecha]"
            lines.append(f"## {date_str} [{e['importance']}] [{e['source']}]")
            lines.append(e["text"])
            lines.append("")

        filepath = os.path.join(MARKDOWN_DIR, "RELATIONSHIPS.md")
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write("\n".join(lines))


def append_to_daily_journal(momento: str, que_paso: str, por_que_importa: str):
    """Appenda una entrada al journal diario."""
    os.makedirs(JOURNAL_DIR, exist_ok=True)
    today = now_col().strftime("%Y-%m-%d")
    journal_file = os.path.join(JOURNAL_DIR, f"{today}.md")

    entry = f"\n---\n### {now_col().strftime('%H:%M')} - {momento}\n**Que paso:** {que_paso}\n**Por que importa:** {por_que_importa}\n\n"

    if not os.path.exists(journal_file):
        header = f"# Journal {today}\n\n"
        with open(journal_file, 'w', encoding='utf-8') as f:
            f.write(header)

    with open(journal_file, 'a', encoding='utf-8') as f:
        f.write(entry)


# ============================================================
# PAD EMOTIONAL MODEL HELPERS
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
