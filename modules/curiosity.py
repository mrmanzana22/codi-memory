"""
Codi Memory - Curiosity module.
Curiosidad proactiva, patrones de trabajo, deteccion de novedad.

Neuroscience basis:
  - Loewenstein 1994: information gap theory (curiosity = gap between
    what you know and what you want to know)
  - Kidd & Hayden 2015: curiosity peaks at intermediate uncertainty
  - Schmidhuber 2010: learning progress as intrinsic reward
  - Connected to prediction loop (Clark 2013): high PE domains
    automatically generate curiosity targets

detectar_sorpresa is a thin wrapper that calls prediction.record_surprise
(conceptually "curiosity" = novelty detection; PE mechanics in prediction).
"""

import math
import os
import json
import sqlite3
from datetime import datetime

from modules.config import (
    memory, qdrant, USER_ID, COLLECTION_NAME,
    now_iso, now_short, now_col,
    CURIOSIDAD_FILE, KNOWN_PROJECTS, CURIOSITY_STALE_DAYS, CURIOSITY_TEMPLATES,
)
from modules.secret_redact import redact_secrets
from modules.qdrant_utils import scroll_all
from modules.access_tracking import record_access

__all__ = [
    "detectar_sorpresa",
    "analizar_patron_trabajo",
    "generar_curiosidad",
    "_cargar_curiosidades",
    "_guardar_curiosidades",
    "push_curiosidad",
    "get_curiosidades",
    "auto_curiosity_tick",
    "register_tools",
]

# FTS DB for prediction data access
_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_FTS_DB = os.path.join(_BASE_DIR, "memories_fts.db")


def detectar_sorpresa(esperaba: str, paso: str, intensidad: str = "medium") -> str:
    """
    Detecta y registra cuando algo no es como esperaba.

    Args:
        esperaba: Lo que esperaba que pasara
        paso: Lo que realmente paso
        intensidad: Que tan sorprendente (low, medium, high)
    """
    try:
        es_positivo = "mejor" in paso.lower() or "funcion\u00f3" in paso.lower() or "\u00e9xito" in paso.lower()
        es_negativo = "fall\u00f3" in paso.lower() or "error" in paso.lower() or "perdi\u00f3" in paso.lower()
        tipo = "positiva" if es_positivo else ("negativa" if es_negativo else "neutral")

        contenido = f"[SORPRESA {intensidad.upper()}] Esperaba: {esperaba} | Pas\u00f3: {paso}"
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
        return f"Error detectando sorpresa: {redact_secrets(str(e))}"


def analizar_patron_trabajo(dias: int = 7) -> str:
    """
    Analiza los patrones de trabajo recientes.

    Args:
        dias: Cuantos dias hacia atras analizar (default 7)
    """
    try:
        from datetime import timedelta
        fecha_limite = now_col() - timedelta(days=dias)

        all_mems = scroll_all(max_results=5000)

        ts = now_iso()
        analyzed_count = 0

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

            if analyzed_count < 500:
                acc = int((payload.get('attention_access_count', 0)) or 0)
                record_access(COLLECTION_NAME, point.id, {
                    'attention_access_count': acc + 1,
                    'attention_last_accessed': ts,
                })
                analyzed_count += 1

            texto_lower = texto.lower()
            if "error" in texto_lower or "fallo" in texto_lower or "problema" in texto_lower:
                errores.append(texto[:100])
            elif "completado" in texto_lower or "funcionando" in texto_lower or "exito" in texto_lower:
                exitos.append(texto[:100])
            if category == "checkpoint":
                checkpoints.append(texto[:100])
            for proyecto in KNOWN_PROJECTS:
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
        return f"Error analizando patrones: {redact_secrets(str(e))}"


def generar_curiosidad() -> str:
    """Genera preguntas curiosas sobre proyectos y temas no tocados recientemente."""
    try:
        proyectos_conocidos = KNOWN_PROJECTS

        all_mems = scroll_all(max_results=5000)

        ts = now_iso()
        tracked_count = 0

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
                        if tracked_count < 500:
                            acc = int((payload.get('attention_access_count', 0)) or 0)
                            record_access(COLLECTION_NAME, point.id, {
                                'attention_access_count': acc + 1,
                                'attention_last_accessed': ts,
                            })
                            tracked_count += 1

        preguntas = []
        for proyecto in proyectos_conocidos:
            if proyecto in ultima_mencion:
                dias_sin_tocar = (ahora - ultima_mencion[proyecto]).days
                if dias_sin_tocar >= CURIOSITY_STALE_DAYS:
                    template = CURIOSITY_TEMPLATES.get(proyecto, f"No hemos tocado {proyecto} en {{dias}} dias. Como va?")
                    preguntas.append(template.format(dias=dias_sin_tocar))
            else:
                preguntas.append(f"No tengo memorias recientes sobre {proyecto}. Sigue siendo relevante?")

        preguntas.append("Que es lo mas importante que deberiamos estar haciendo ahora mismo?")

        # PE-driven curiosity (Loewenstein 1994 + Kidd & Hayden 2015)
        pe_preguntas = []
        pe_domains = _get_high_surprise_domains()
        for domain in pe_domains[:3]:
            pe_preguntas.append(
                f"[PE] Mi modelo falla en '{domain['topic']}' "
                f"(accuracy {domain['accuracy']:.0%}, curiosity={domain['curiosity_score']:.2f})"
            )

        # Knowledge gap curiosity
        gap_preguntas = []
        for gap in _get_knowledge_gaps()[:3]:
            gap_preguntas.append(f"[GAP] Baja confianza en '{gap}' — deberia investigar")

        resultado = "# CURIOSIDAD GENERADA\n\n"

        if pe_preguntas:
            resultado += "## Curiosidad por Prediction Error (dominios poco predecibles)\n\n"
            for i, p in enumerate(pe_preguntas, 1):
                resultado += f"{i}. {p}\n"
            resultado += "\n"

        if gap_preguntas:
            resultado += "## Curiosidad por Knowledge Gaps (baja confianza)\n\n"
            for i, p in enumerate(gap_preguntas, 1):
                resultado += f"{i}. {p}\n"
            resultado += "\n"

        resultado += "## Proyectos Sin Tocar (stale detection)\n\n"
        for i, p in enumerate(preguntas, 1):
            resultado += f"{i}. {p}\n\n"
        resultado += "---\n*Generado por curiosidad proactiva (PE + gaps + stale)*\n"
        return resultado
    except Exception as e:
        return f"Error generando curiosidad: {redact_secrets(str(e))}"


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
        return f"Error guardando curiosidad: {redact_secrets(str(e))}"


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
        return f"Error leyendo curiosidades: {redact_secrets(str(e))}"


def _get_high_surprise_domains(min_observations: int = 3, window: int = 50) -> list:
    """Find domains with persistently high prediction error.

    Queries prediction_results for topics where surprise is consistently
    above average. These are domains where the system's model is weak
    and curiosity should drive information-seeking (Loewenstein 1994).

    Returns list of dicts: [{topic, avg_surprise, count, curiosity_score}]
    """
    if not os.path.exists(_FTS_DB):
        return []
    try:
        conn = sqlite3.connect(_FTS_DB, timeout=3)
        # Get per-topic surprise stats from recent predictions
        rows = conn.execute("""
            SELECT actual_topic,
                   AVG(surprise_score) AS avg_surprise,
                   COUNT(*) AS cnt,
                   AVG(hit) AS accuracy
            FROM (
                SELECT actual_topic, surprise_score, hit
                FROM prediction_results
                WHERE COALESCE(source, 'interactive') != 'sleep_loop'
                ORDER BY id DESC LIMIT ?
            )
            GROUP BY actual_topic
            HAVING cnt >= ?
            ORDER BY avg_surprise DESC
        """, (window, min_observations)).fetchall()

        # Global average surprise for comparison
        global_avg = conn.execute("""
            SELECT AVG(surprise_score) FROM (
                SELECT surprise_score FROM prediction_results
                WHERE COALESCE(source, 'interactive') != 'sleep_loop'
                ORDER BY id DESC LIMIT ?
            )
        """, (window,)).fetchone()
        baseline = global_avg[0] if global_avg and global_avg[0] else 0.5

        conn.close()

        results = []
        for topic, avg_s, cnt, acc in rows:
            if avg_s <= baseline:
                continue  # Below average surprise, not curious
            # Kidd & Hayden 2015: curiosity peaks at INTERMEDIATE uncertainty
            # Very high uncertainty (acc ~0) = too hard, very low (acc ~1) = too easy
            # Peak curiosity at acc ~0.3-0.5 (some knowledge, wants more)
            if acc is not None:
                curiosity_curve = math.exp(-((acc - 0.4) ** 2) / 0.18)
            else:
                curiosity_curve = 0.5
            curiosity_score = (avg_s - baseline) * curiosity_curve
            if curiosity_score > 0.05:
                results.append({
                    "topic": topic,
                    "avg_surprise": round(avg_s, 3),
                    "accuracy": round(acc, 3) if acc else 0,
                    "count": cnt,
                    "curiosity_score": round(curiosity_score, 3),
                })
        return sorted(results, key=lambda x: -x["curiosity_score"])
    except Exception:
        return []


def _get_knowledge_gaps() -> list:
    """Find domains where the system has low confidence.

    Uses self_model.identify_knowledge_gaps() to find areas where
    retrieval has been weak or searches have failed.

    Returns list of topic strings.
    """
    try:
        from modules.self_model import identify_knowledge_gaps
        gaps = identify_knowledge_gaps()
        if isinstance(gaps, str):
            # Parse the text output for topic names
            topics = []
            for line in gaps.split('\n'):
                line = line.strip()
                if line.startswith('- ') and ':' in line:
                    topic = line.split(':')[0].replace('- ', '').strip()
                    if topic and len(topic) > 2:
                        topics.append(topic)
            return topics[:5]
        return []
    except Exception:
        return []


def auto_curiosity_tick() -> dict:
    """Background tick: auto-generate curiosity from prediction errors + knowledge gaps.

    Called by sleep_loop to maintain a fresh curiosity queue.
    Implements Loewenstein 1994: curiosity driven by information gaps.
    Implements Kidd & Hayden 2015: peak curiosity at intermediate uncertainty.

    Returns dict with counts of generated items.
    """
    result = {"generated": 0, "pe_driven": 0, "gap_driven": 0}

    try:
        data = _cargar_curiosidades()
        existing_questions = {item.get("pregunta", "").lower() for item in data.get("pendientes", [])}

        max_id = 0
        for item in data.get("pendientes", []) + data.get("exploradas", []):
            item_id = item.get("id", 0)
            if item_id > max_id:
                max_id = item_id

        # 1. PE-driven curiosity: high-surprise domains
        pe_domains = _get_high_surprise_domains()
        for domain in pe_domains[:3]:
            topic = domain["topic"]
            question = (f"Mi prediccion falla en '{topic}' "
                        f"(accuracy {domain['accuracy']:.0%}, surprise {domain['avg_surprise']:.2f}). "
                        f"Que patrones me estoy perdiendo?")
            if question.lower() not in existing_questions:
                max_id += 1
                data["pendientes"].append({
                    "id": max_id,
                    "pregunta": question,
                    "categoria": topic,
                    "agregada": now_short(),
                    "prioridad": "alta" if domain["curiosity_score"] > 0.15 else "media",
                    "source": "prediction_error",
                    "curiosity_score": domain["curiosity_score"],
                })
                result["pe_driven"] += 1
                result["generated"] += 1

        # 2. Knowledge gap curiosity
        gaps = _get_knowledge_gaps()
        for gap_topic in gaps[:2]:
            question = f"Tengo poca confianza en '{gap_topic}'. Que deberia aprender?"
            if question.lower() not in existing_questions:
                max_id += 1
                data["pendientes"].append({
                    "id": max_id,
                    "pregunta": question,
                    "categoria": gap_topic,
                    "agregada": now_short(),
                    "prioridad": "media",
                    "source": "knowledge_gap",
                })
                result["gap_driven"] += 1
                result["generated"] += 1

        # 3. Prune resolved curiosities: if a PE-driven item's domain now has
        # low surprise, move it to exploradas
        if pe_domains:
            low_surprise_topics = set()
            try:
                conn = sqlite3.connect(_FTS_DB, timeout=3)
                rows = conn.execute("""
                    SELECT actual_topic, AVG(surprise_score) AS avg_s
                    FROM (SELECT actual_topic, surprise_score FROM prediction_results
                          WHERE COALESCE(source, 'interactive') != 'sleep_loop'
                          ORDER BY id DESC LIMIT 20)
                    GROUP BY actual_topic
                    HAVING avg_s < 0.3
                """).fetchall()
                low_surprise_topics = {r[0] for r in rows}
                conn.close()
            except Exception:
                pass

            if low_surprise_topics:
                still_pending = []
                for item in data.get("pendientes", []):
                    if (item.get("source") == "prediction_error"
                            and item.get("categoria") in low_surprise_topics):
                        item["resuelto"] = now_short()
                        item["razon"] = "surprise decreased below 0.3"
                        data.setdefault("exploradas", []).append(item)
                    else:
                        still_pending.append(item)
                data["pendientes"] = still_pending

        if result["generated"] > 0:
            _guardar_curiosidades(data)

    except Exception:
        pass

    return result


def register_tools(mcp):
    """Register curiosity MCP tools."""
    mcp.tool()(detectar_sorpresa)
    mcp.tool()(analizar_patron_trabajo)
    mcp.tool()(generar_curiosidad)
    mcp.tool()(push_curiosidad)
    mcp.tool()(get_curiosidades)
