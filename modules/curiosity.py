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
from datetime import datetime, timezone

from modules.config import (
    USER_ID, COLLECTION_NAME,
    now_iso, now_short, now_col,
    CURIOSIDAD_FILE, KNOWN_PROJECTS, CURIOSITY_STALE_DAYS, CURIOSITY_TEMPLATES,
    connect_fts,
)
from modules.secret_redact import redact_secrets
from modules.qdrant_utils import scroll_all
from modules.pg_store import pg

# Sprint 10: IG constants (AXIOM, Friston 2017)
IG_MIN_OBSERVATIONS = 5     # Min observations before computing IG for a domain
IG_EXPLOITATION_THRESHOLD = 0.1  # Below this IG, domain is "learned" → exploit

__all__ = [
    "detectar_sorpresa",
    "analizar_patron_trabajo",
    "generar_curiosidad",
    "_cargar_curiosidades",
    "_guardar_curiosidades",
    "push_curiosidad",
    "get_curiosidades",
    "resolve_curiosidad",
    "get_curiosity_quality",
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

        pg.add(
            content=contenido,
            category="aprendizaje",
            source="experienced",
            importance="high" if intensidad == "high" else "medium",
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

            if timestamp:
                try:
                    mem_date = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
                except ValueError:
                    continue
                if mem_date.tzinfo is None:
                    mem_date = mem_date.replace(tzinfo=fecha_limite.tzinfo or timezone.utc)
                if mem_date < fecha_limite:
                    continue

            if analyzed_count < 500:
                acc = int((payload.get('attention_access_count', 0)) or 0)
                pg.update_payload(point.id, {
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
                    fecha = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
                    if fecha.tzinfo is None:
                        fecha = fecha.replace(tzinfo=ahora.tzinfo)
                    else:
                        fecha = fecha.astimezone(ahora.tzinfo)
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
                            pg.update_payload(point.id, {
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

        # IG-driven curiosity (Sprint 10: AXIOM + Kidd & Hayden 2015)
        pe_preguntas = []
        pe_domains = _get_high_surprise_domains()
        for domain in pe_domains[:3]:
            phase_tag = "EXPLORE" if domain.get("phase") == "explore" else "EXPLOIT"
            pe_preguntas.append(
                f"[{phase_tag}] Mi prediccion falla en '{domain['topic']}' "
                f"(accuracy {domain['accuracy']:.0%}, "
                f"IG={domain.get('ig', 0):.3f}, "
                f"curiosity={domain['curiosity_score']:.2f})"
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
        with open(CURIOSIDAD_FILE, 'r', encoding='utf-8') as f:
            try:
                return json.load(f)
            except json.JSONDecodeError as e:
                backup_path = f"{CURIOSIDAD_FILE}.corrupt-{now_col().strftime('%Y%m%d%H%M%S')}.bak"
                os.replace(CURIOSIDAD_FILE, backup_path)
                raise ValueError(
                    f"Curiosity file is corrupt and was moved to backup: {backup_path}"
                ) from e
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


def _compute_domain_ig(topic_transitions: dict, alpha: float = 1.0) -> float:
    """Compute Information Gain for a domain's transition model.

    Sprint 10: IG = KL(posterior || prior) for the Dirichlet-Multinomial.
    High IG = model is still learning (posterior far from prior) → explore.
    Low IG = model has stabilized (posterior ≈ observations) → exploit.

    For Dirichlet-Multinomial:
      IG ≈ Σ_k [ (alpha_k - alpha_0) * (digamma(alpha_k) - digamma(sum_alpha)) ]
    where alpha_k = count_k + alpha_0 (posterior), alpha_0 = prior.

    Simplified: use entropy of posterior as proxy for remaining uncertainty.
    IG = H(prior) - H(posterior). Since prior is uniform, H(prior) = log(K).
    So IG = log(K) - H(posterior).

    Args:
        topic_transitions: {next_topic: count} for transitions from this topic
        alpha: Dirichlet prior (1.0 = uniform/Laplace)

    Returns:
        Information Gain in nats. Higher = more to learn.

    References:
        AXIOM Eq.10 (Friston 2024), Friston 2017 EFE epistemic term.
    """
    if not topic_transitions:
        return 0.0

    counts = topic_transitions
    total = sum(counts.values())
    k = len(counts)
    if k <= 1:
        return 0.0

    # Posterior entropy: H(posterior) = -Σ p_k * log(p_k)
    # where p_k = (count_k + alpha) / (total + alpha*K)
    h_posterior = 0.0
    denom = total + alpha * k
    for count in counts.values():
        p = (count + alpha) / denom
        if p > 0:
            h_posterior -= p * math.log(p)

    # Prior entropy: uniform over K categories → log(K)
    h_prior = math.log(k)

    # IG = H(prior) - H(posterior)
    # Clamped to >= 0 (posterior can't be more uncertain than prior in theory,
    # but numerical issues can cause tiny negatives)
    ig = max(0.0, h_prior - h_posterior)
    return ig


def _get_high_surprise_domains(min_observations: int = 3, window: int = 50) -> list:
    """Find domains ranked by Information Gain (Sprint 10).

    Primary ranking: IG from Dirichlet-Multinomial transition model.
    Secondary filter: Kidd & Hayden inverted-U (avoid too-easy/too-hard).

    IG replaces raw PE-based ranking. IG captures how much the model
    would LEARN from observing this domain, not just how surprised it is.

    Returns list of dicts: [{topic, ig, avg_surprise, count, accuracy, curiosity_score, phase}]
    """
    if not os.path.exists(_FTS_DB):
        return []
    try:
        conn = connect_fts(_FTS_DB)

        # Get per-topic transition counts for IG computation
        transition_rows = conn.execute("""
            SELECT from_topic, to_topic, count
            FROM transition_stats
            ORDER BY count DESC
        """).fetchall()

        # Build transition model per source topic
        transitions_by_topic = {}
        for from_t, to_t, cnt in transition_rows:
            if from_t not in transitions_by_topic:
                transitions_by_topic[from_t] = {}
            transitions_by_topic[from_t][to_t] = cnt

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

        conn.close()

        results = []
        for topic, avg_s, cnt, acc in rows:
            # Sprint 10: Compute IG for this domain's transition model
            topic_trans = transitions_by_topic.get(topic, {})
            ig = _compute_domain_ig(topic_trans)

            # Kidd & Hayden 2015: curiosity peaks at INTERMEDIATE uncertainty
            if acc is not None:
                curiosity_curve = math.exp(-((acc - 0.4) ** 2) / 0.18)
            else:
                curiosity_curve = 0.5

            # Sprint 10: Combined score = IG * curiosity_curve
            # IG captures model learning potential, curve filters extremes
            curiosity_score = ig * curiosity_curve

            # Sprint 10.3: Exploration→exploitation transition
            # When IG drops below threshold → domain is "learned"
            if cnt >= IG_MIN_OBSERVATIONS and ig < IG_EXPLOITATION_THRESHOLD:
                phase = "exploit"
            else:
                phase = "explore"

            if curiosity_score > 0.01 or phase == "explore":
                results.append({
                    "topic": topic,
                    "ig": round(ig, 4),
                    "avg_surprise": round(avg_s, 3),
                    "accuracy": round(acc, 3) if acc else 0,
                    "count": cnt,
                    "curiosity_score": round(curiosity_score, 4),
                    "phase": phase,
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
                    topic = line.split(':')[0].replace('- ', '').replace('**', '').strip()
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
        # Proposal #58 Fix 3: Semantic dedup — one PE curiosity per topic
        # Bug fix: check BOTH pendientes AND exploradas to prevent re-creation
        existing_pe_topics = {
            item.get("categoria")
            for item in data.get("pendientes", []) + data.get("exploradas", [])
            if item.get("source") == "prediction_error"
        }
        pe_domains = _get_high_surprise_domains()
        for domain in pe_domains[:3]:
            topic = domain["topic"]
            phase = domain.get("phase", "explore")
            ig = domain.get("ig", 0)
            question = (f"Mi prediccion falla en '{topic}' "
                        f"(accuracy {domain['accuracy']:.0%}, IG={ig:.3f}, phase={phase}). "
                        f"Que patrones me estoy perdiendo?")
            if question.lower() not in existing_questions and topic not in existing_pe_topics:
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
                existing_pe_topics.add(topic)
                result["pe_driven"] += 1
                result["generated"] += 1
            elif topic in existing_pe_topics:
                # Update existing if new score is higher
                for item in data.get("pendientes", []):
                    if (item.get("source") == "prediction_error"
                            and item.get("categoria") == topic
                            and domain["curiosity_score"] > item.get("curiosity_score", 0)):
                        item["pregunta"] = question
                        item["curiosity_score"] = domain["curiosity_score"]
                        item["agregada"] = now_short()
                        break

        # 2. Knowledge gap curiosity
        # Bug fix: strip markdown from gap topics to prevent dedup mismatches
        existing_gap_topics = {
            item.get("categoria", "").replace("**", "").replace("*", "").strip().lower()
            for item in data.get("pendientes", []) + data.get("exploradas", [])
            if item.get("source") == "knowledge_gap"
        }
        gaps = _get_knowledge_gaps()
        for gap_topic in gaps[:2]:
            gap_topic = gap_topic.replace("**", "").replace("*", "").strip()
            if gap_topic.lower() in existing_gap_topics:
                continue
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
                existing_gap_topics.add(gap_topic.lower())
                result["gap_driven"] += 1
                result["generated"] += 1

        # 3. Prune resolved curiosities: if a PE-driven item's domain now has
        # low surprise, move it to exploradas
        if pe_domains:
            low_surprise_topics = set()
            try:
                conn = connect_fts(_FTS_DB)
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

        # Proposal #58 Fix 4: Prune stale manual curiosities (>60 days)
        from datetime import timedelta
        cutoff_date = (datetime.now() - timedelta(days=60)).strftime("%Y-%m-%d")
        still_pending_2 = []
        for item in data.get("pendientes", []):
            if (not item.get("source")  # manual items have no source
                    and item.get("agregada", "") < cutoff_date
                    and not item.get("explored_at")):
                item["resuelto"] = now_short()
                item["razon"] = "stale (>60 days without exploration)"
                data.setdefault("exploradas", []).append(item)
            else:
                still_pending_2.append(item)
        data["pendientes"] = still_pending_2

        _guardar_curiosidades(data)

    except Exception:
        pass

    return result


def resolve_curiosidad(curiosidad_id: int, outcome: str = "discovery",
                       description: str = "") -> str:
    """Mark a curiosity as explored with a structured outcome.

    Closes the curiosity loop (Schmidhuber 2010: learning progress as reward).
    Links exploration outcomes back to the curiosity that generated them.

    Args:
        curiosidad_id: ID of the curiosity to resolve
        outcome: "discovery" (new knowledge gained), "partial" (some insight),
                 "dead_end" (explored but nothing useful found)
        description: What was discovered or why it was a dead end
    """
    valid_outcomes = {"discovery", "partial", "dead_end"}
    if outcome not in valid_outcomes:
        return f"Invalid outcome '{outcome}'. Use: {', '.join(valid_outcomes)}"

    try:
        data = _cargar_curiosidades()
        pendientes = data.get("pendientes", [])

        target = None
        remaining = []
        for item in pendientes:
            if item.get("id") == curiosidad_id:
                target = item
            else:
                remaining.append(item)

        if target is None:
            return f"Curiosidad #{curiosidad_id} not found in pendientes."

        # Enrich with outcome metadata
        target["explored_at"] = now_short()
        target["outcome"] = outcome
        target["outcome_description"] = description
        target["resuelto"] = now_short()
        target["razon"] = f"explored: {outcome}"

        data["pendientes"] = remaining
        data.setdefault("exploradas", []).append(target)

        # If discovery, also record in descubrimientos with link
        if outcome == "discovery" and description:
            discovery_entry = (
                f"[#{curiosidad_id}] {description} "
                f"(from: {target.get('source', 'manual')}, "
                f"topic: {target.get('categoria', 'general')})"
            )
            data.setdefault("descubrimientos", []).append(discovery_entry)

        _guardar_curiosidades(data)

        label = {"discovery": "DESCUBRIMIENTO", "partial": "PARCIAL", "dead_end": "SIN RESULTADO"}
        return (f"Curiosidad #{curiosidad_id} resuelta: {label[outcome]}\n"
                f"Pregunta: {target.get('pregunta', '')}\n"
                f"Outcome: {description or '(no description)'}")
    except Exception as e:
        return f"Error resolving curiosidad: {redact_secrets(str(e))}"


def get_curiosity_quality() -> str:
    """Quality metrics for the curiosity system.

    Schmidhuber 2010: Learning progress as intrinsic reward.
    Tracks which curiosity sources generate the most valuable outcomes.

    Returns:
        Formatted report with discovery rates, source comparison, and stats.
    """
    try:
        data = _cargar_curiosidades()
        pendientes = data.get("pendientes", [])
        exploradas = data.get("exploradas", [])

        if not exploradas:
            return ("# Curiosity Quality Report\n\n"
                    f"Pendientes: {len(pendientes)}, Exploradas: 0\n"
                    "No hay outcomes registrados aun. Usa resolve_curiosidad() para cerrar el loop.")

        # Overall stats
        total_explored = len(exploradas)
        discoveries = [e for e in exploradas if e.get("outcome") == "discovery"]
        partials = [e for e in exploradas if e.get("outcome") == "partial"]
        dead_ends = [e for e in exploradas if e.get("outcome") == "dead_end"]
        auto_resolved = [e for e in exploradas if e.get("outcome") not in
                         {"discovery", "partial", "dead_end"}]

        discovery_rate = len(discoveries) / total_explored if total_explored else 0

        # By source
        sources = {}
        for item in exploradas:
            src = item.get("source", "manual")
            if src not in sources:
                sources[src] = {"total": 0, "discovery": 0, "partial": 0, "dead_end": 0}
            sources[src]["total"] += 1
            outcome = item.get("outcome", "auto_resolved")
            if outcome in sources[src]:
                sources[src][outcome] += 1

        report = ["# Curiosity Quality Report\n"]
        report.append(f"**Pendientes:** {len(pendientes)} | **Exploradas:** {total_explored}\n")
        report.append(f"## Overall Outcomes")
        report.append(f"- Discoveries: {len(discoveries)} ({discovery_rate:.0%})")
        report.append(f"- Partial: {len(partials)}")
        report.append(f"- Dead ends: {len(dead_ends)}")
        report.append(f"- Auto-resolved: {len(auto_resolved)}")

        if sources:
            report.append(f"\n## Quality by Source")
            report.append(f"{'Source':<20} {'Total':<8} {'Disc.':<8} {'Rate':<8}")
            report.append("-" * 44)
            for src, stats in sorted(sources.items(), key=lambda x: -x[1]["total"]):
                rate = stats["discovery"] / stats["total"] if stats["total"] else 0
                report.append(f"{src:<20} {stats['total']:<8} {stats['discovery']:<8} {rate:.0%}")

        # Recent discoveries
        if discoveries:
            report.append(f"\n## Recent Discoveries")
            for d in discoveries[-5:]:
                desc = d.get("outcome_description", "")[:80]
                report.append(f"- #{d.get('id', '?')}: {desc or d.get('pregunta', '')[:80]}")

        return "\n".join(report)
    except Exception as e:
        return f"Error generating quality report: {redact_secrets(str(e))}"


def register_tools(mcp):
    """Register curiosity MCP tools."""
    mcp.tool()(detectar_sorpresa)
    mcp.tool()(analizar_patron_trabajo)
    mcp.tool()(generar_curiosidad)
    mcp.tool()(push_curiosidad)
    mcp.tool()(get_curiosidades)
    mcp.tool()(resolve_curiosidad)
    mcp.tool()(get_curiosity_quality)
