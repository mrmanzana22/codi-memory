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
  - ACT-R activation scoring: compute_base_level_activation, compute_activation
"""

import os
import json
import sqlite3
import math
import random
import threading
import time
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
    BACKUP_POLICY,
    BACKUP_MIN_INTERVAL_SEC,
    BACKUP_MAX_FILES,
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

    from modules.config import IMPORTANCE_WEIGHTS as importance_weights

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
    """Enriquece una memoria con ownership metadata usando Qdrant directo.

    Phase 0 ACT-R: Sets initial access_timestamps and applies PAD encoding boost.
    """
    try:
        themes = infer_themes(content)
        if not themes:
            themes = [category]

        # Detectar si es auto-referencial (metacognicion)
        self_ref = is_self_referential(content)
        if self_ref and 'identidad' not in themes:
            themes.append('identidad')

        # Base salience from importance
        base_salience = 0.7 if importance in ['critical', 'high'] else 0.5

        # PAD encoding boost: high arousal/emotion strengthens encoding
        try:
            emotional_state = _get_emotional_state()
            current = emotional_state.get('current', {})
            arousal = current.get('arousal', 0.0)
            pleasure = current.get('pleasure', 0.0)
            pad_boost = compute_pad_encoding_boost(arousal, pleasure)
            base_salience = min(1.0, base_salience + pad_boost)

            # Capture emotional context at encoding time
            emotional_valence_auto = 'neutral'
            if pleasure > 0.3:
                emotional_valence_auto = 'positive'
            elif pleasure < -0.3:
                emotional_valence_auto = 'negative'
            if emotional_valence == 'neutral':
                emotional_valence = emotional_valence_auto
        except Exception:
            arousal = 0.0
            pleasure = 0.0

        created_now = now_iso()

        # 3C: PAD state at encoding (Godden & Baddeley 1975)
        try:
            pad_at_encoding = {
                "P": round(pleasure, 3),
                "A": round(arousal, 3),
                "D": round(current.get('dominance', 0.0), 3),
            }
        except Exception:
            pad_at_encoding = {"P": 0.0, "A": 0.0, "D": 0.0}

        # 3D: Temporal context (Friedman 1993)
        try:
            from datetime import datetime as _dt
            dt = _dt.fromisoformat(created_now.replace('Z', '+00:00')) if 'Z' in created_now else _dt.fromisoformat(created_now)
            hour = dt.hour
            if 5 <= hour < 12:
                time_of_day = "morning"
            elif 12 <= hour < 17:
                time_of_day = "afternoon"
            elif 17 <= hour < 21:
                time_of_day = "evening"
            else:
                time_of_day = "night"
            day_type = "weekend" if dt.weekday() >= 5 else "weekday"
            temporal_context = {"time_of_day": time_of_day, "day_type": day_type}
        except Exception:
            temporal_context = {"time_of_day": "unknown", "day_type": "unknown"}

        ownership_metadata = {
            'category': category,
            'ownership_is_mine': True,
            'ownership_source': source,
            'ownership_confidence': 0.9 if source == 'experienced' else 0.7,
            'experiential_emotional_weight': emotional_weight,
            'experiential_emotional_valence': emotional_valence,
            'narrative_importance': importance,
            'narrative_themes': themes,
            'attention_salience': base_salience,
            'attention_access_count': 0,
            'attention_last_accessed': None,
            'access_timestamps': [created_now],  # ACT-R: track access history
            'temporal_session_id': get_session_id(),
            'created_at': created_now,
            'self_reference': self_ref,
            'pad_at_encoding': pad_at_encoding,       # 3C: State-dependent retrieval
            'temporal_context': temporal_context,      # 3D: Temporal metadata
            '_v': 4.0  # Phase 3: SDR + temporal context
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

_backup_lock = threading.Lock()
_last_backup_time = 0.0


def save_backup_json(reason: str = "manual") -> bool:
    """Guarda todas las memorias en JSON como backup con rotacion de archivos.
    Retorna True si el backup fue exitoso."""
    try:
        results = memory.get_all(user_id=USER_ID)
        if not (results and results.get("results")):
            return False

        payload = {
            "timestamp": now_iso(),
            "reason": reason,
            "user_id": USER_ID,
            "memories": results["results"]
        }

        # Write to canonical path (for compatibility with export_memories_to_files)
        with open(BACKUP_FILE, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False, default=str)

        # Write timestamped copy for rotation
        ts = now_col().strftime("%Y%m%d_%H%M%S")
        rotated_path = os.path.join(BACKUP_DIR, f"memories_backup_{ts}.json")
        try:
            with open(rotated_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, ensure_ascii=False, default=str)
        except Exception as e:
            print(f"[Codi Memory] Error escribiendo backup rotado: {e}")

        # Rotate: keep only BACKUP_MAX_FILES most recent
        _rotate_backups()

        # Hook: exportar a markdown despues de cada backup
        try:
            if os.getenv('EXPORT_MD_ON_BACKUP', '0') in ('1', 'true', 'yes'):
                export_memories_to_files()
        except Exception as e:
            print(f"[Codi Memory] Error exportando markdown: {e}")

        return True

    except Exception as e:
        print(f"[Codi Memory] Error guardando backup: {e}")
        return False


def _rotate_backups():
    """Mantiene solo BACKUP_MAX_FILES backups rotados mas recientes (por mtime)."""
    try:
        files = []
        for name in os.listdir(BACKUP_DIR):
            if name.startswith("memories_backup_") and name.endswith(".json"):
                path = os.path.join(BACKUP_DIR, name)
                try:
                    files.append((os.path.getmtime(path), path))
                except OSError:
                    continue
        files.sort(reverse=True)  # newest first
        for _, path in files[BACKUP_MAX_FILES:]:
            try:
                os.remove(path)
            except OSError:
                pass
    except Exception:
        pass


def maybe_backup(reason: str = "", force: bool = False):
    """
    Backup con debounce. Solo ejecuta si:
    - force=True (flush, checkpoint, noche), o
    - BACKUP_POLICY=='always', o
    - Paso suficiente tiempo desde el ultimo backup (BACKUP_MIN_INTERVAL_SEC)
    """
    global _last_backup_time

    if not force and BACKUP_POLICY != "always":
        elapsed = time.time() - _last_backup_time
        if elapsed < BACKUP_MIN_INTERVAL_SEC:
            return

    acquired = _backup_lock.acquire(blocking=False)
    if not acquired:
        return  # Another backup in progress, skip

    try:
        save_backup_json(reason=reason or "debounced")
        _last_backup_time = time.time()
        if reason:
            print(f"[Codi Memory] Backup completado ({reason})")
    finally:
        _backup_lock.release()


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


# ============================================================
# ACT-R ACTIVATION SCORING (Phase 0 - Neuroscience Foundation)
# ============================================================
# Based on Anderson's ACT-R theory (2007):
#   A_i = B_i + sum(W_j * S_ji) + epsilon
# Where:
#   B_i = base-level activation = ln(sum(t_k^{-d}))
#   W_j * S_ji = spreading activation from context
#   epsilon = stochastic noise for non-deterministic retrieval
# ============================================================

# ACT-R parameters (Phase 0 calibrated - Feb 2026)
# Decay reduced from 0.5 to 0.4 for day/week timescales (Anderson & Schooler 1991)
ACTR_DECAY = 0.4
ACTR_TIME_UNIT = 3600         # Normalize to hours (was seconds) to recentre B_i range
ACTR_NOISE_SIGMA = 0.15       # Was 0.05 (~1% range, deterministic). Now ~3% range for stochastic retrieval
ACTR_SPREAD_WEIGHT = 0.3      # Single-source spreading, reasonable
ACTR_MIN_ACTIVATION = -5.0    # Floor to prevent -inf from log(0)
ACTR_SIGMOID_CENTER = -0.07   # Calibrated for d=0.4, unit=hours
ACTR_SIGMOID_SCALE = 0.91     # Maps useful A_i range to 0.05-0.95


def compute_base_level_activation(
    created_at_str: str,
    last_accessed_str: str = None,
    access_count: int = 0,
    access_timestamps: list = None,
    decay: float = ACTR_DECAY
) -> float:
    """
    Compute ACT-R base-level activation B_i = ln(sum(t_k^{-d})).

    If full access_timestamps are available, uses them directly.
    Otherwise, approximates using created_at, last_accessed, and access_count
    by interpolating access times uniformly between creation and last access.

    Args:
        created_at_str: ISO timestamp of memory creation
        last_accessed_str: ISO timestamp of last access (optional)
        access_count: Number of times accessed
        access_timestamps: List of ISO timestamps of each access (optional, preferred)
        decay: Power-law decay parameter (default 0.5, Anderson 2007)

    Returns:
        Base-level activation value (float). Higher = more accessible.
    """
    now = datetime.now()

    # Parse creation time
    # Default: assume 30 days old if no timestamp (not "now" which inflates activation)
    default_created = now - __import__('datetime').timedelta(days=30)
    try:
        if created_at_str and 'T' in str(created_at_str):
            created_at = datetime.fromisoformat(str(created_at_str).replace('Z', '+00:00'))
            if created_at.tzinfo:
                created_at = created_at.replace(tzinfo=None)
        else:
            created_at = default_created
    except Exception:
        created_at = default_created

    # If we have full access timestamps (best case), use them
    if access_timestamps and len(access_timestamps) > 0:
        total = 0.0
        for ts_str in access_timestamps:
            try:
                ts = datetime.fromisoformat(str(ts_str).replace('Z', '+00:00'))
                if ts.tzinfo:
                    ts = ts.replace(tzinfo=None)
                t_hours = max(1.0, (now - ts).total_seconds() / ACTR_TIME_UNIT)
                total += t_hours ** (-decay)
            except Exception:
                continue
        if total > 0:
            return math.log(total)
        return ACTR_MIN_ACTIVATION

    # Approximation: interpolate access times between creation and last_access
    # The creation itself counts as the first "access"
    t_created = max(1.0, (now - created_at).total_seconds() / ACTR_TIME_UNIT)

    # Parse last_accessed
    t_last = t_created
    if last_accessed_str:
        try:
            last_acc = datetime.fromisoformat(str(last_accessed_str).replace('Z', '+00:00'))
            if last_acc.tzinfo:
                last_acc = last_acc.replace(tzinfo=None)
            t_last = max(1.0, (now - last_acc).total_seconds() / ACTR_TIME_UNIT)
        except Exception:
            pass

    # Total presentations = access_count + 1 (the original encoding)
    n_presentations = max(1, (access_count or 0) + 1)

    if n_presentations == 1:
        # Only the creation event
        total = t_created ** (-decay)
    else:
        # Interpolate: assume accesses are uniformly distributed
        # between t_created (oldest) and t_last (most recent)
        total = 0.0
        for k in range(n_presentations):
            if n_presentations > 1:
                # Fraction: 0 = creation time, 1 = last access time
                frac = k / (n_presentations - 1)
                t_k = t_created + frac * (t_last - t_created)
            else:
                t_k = t_created
            t_k = max(1.0, t_k)
            total += t_k ** (-decay)

    if total > 0:
        return math.log(total)
    return ACTR_MIN_ACTIVATION


def compute_activation(
    payload: dict,
    context_salience: float = 0.0,
    noise: bool = True
) -> float:
    """
    Compute full activation for a memory via unified scorer (WIRING-5).

    Backward-compatible wrapper: delegates to modules.activation.compute_unified_activation.
    All existing callers continue to work unchanged.

    Args:
        payload: Qdrant point payload with memory metadata
        context_salience: Spreading activation from current context (0-1)
        noise: Whether to add stochastic noise

    Returns:
        Activation value normalized to 0-1.
    """
    from modules.activation import compute_unified_activation

    created_at = payload.get('created_at', '')
    last_accessed = payload.get('attention_last_accessed', '')
    access_count = payload.get('attention_access_count', 0)
    access_timestamps = payload.get('access_timestamps', None)
    current_salience = float(payload.get('attention_salience', 0.5))
    importance = payload.get('narrative_importance', 'medium')

    result = compute_unified_activation(
        memory_id=str(payload.get('id', '')),
        created_at=created_at,
        last_accessed=last_accessed,
        access_count=access_count,
        access_timestamps=access_timestamps,
        spreading_boost=max(context_salience, current_salience),
        importance=importance,
        noise=noise,
    )
    return result.total


def compute_pad_retrieval_bias(
    memory_valence: str,
    current_pleasure: float
) -> float:
    """
    Compute mood-congruent retrieval bias based on PAD model.

    Memories whose emotional valence matches current mood get a boost.
    Based on Bower (1981) mood-congruent memory effect.

    Args:
        memory_valence: Emotional valence of the memory ('positive', 'negative', 'neutral')
        current_pleasure: Current pleasure dimension of PAD state (-1 to 1)

    Returns:
        Bias value to add to retrieval score (range: -0.05 to +0.1)
    """
    valence_map = {'positive': 1.0, 'neutral': 0.0, 'negative': -1.0}
    mem_val = valence_map.get(memory_valence, 0.0)

    # Congruence: positive when mood and memory valence align
    congruence = mem_val * current_pleasure

    # Scale to small bias (max 0.1 boost, max -0.05 penalty)
    if congruence > 0:
        return min(0.1, congruence * 0.1)
    else:
        return max(-0.05, congruence * 0.05)


def compute_pad_encoding_boost(arousal: float, pleasure: float) -> float:
    """
    Compute emotional encoding strength boost based on PAD state.

    High arousal strengthens encoding (LaBar & Cabeza 2006).
    Emotional memories (high arousal, strong valence) are encoded more strongly.

    Args:
        arousal: Current arousal level (-1 to 1)
        pleasure: Current pleasure level (-1 to 1)

    Returns:
        Salience boost (0.0 to 0.3) to add to base attention_salience
    """
    # Arousal is the primary driver of encoding strength
    arousal_boost = max(0.0, arousal) * 0.2  # 0 to 0.2

    # Strong emotions (high |pleasure|) also boost encoding
    valence_boost = abs(pleasure) * 0.1  # 0 to 0.1

    return min(0.3, arousal_boost + valence_boost)
