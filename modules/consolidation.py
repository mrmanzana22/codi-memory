"""
CONSOLIDATION MODULE - Phase 1 of Codi Consciousness Project

Implements the episodic -> semantic consolidation pipeline (5 phases):
  1. SELECTION: Score recent episodes by importance
  2. CLUSTERING: Group by semantic similarity + topic
  3. EXTRACTION: Extract generalizable facts (full scope only)
  4. INTEGRATION: Create/update semantic memories
  5. PRUNING: Mark consolidated, apply decay

Sub-modules (split for SRP):
  - consolidation_common: Shared embedding, similarity, DB connections
  - reconsolidation: Contradiction detection, labile memories, correct_memory
  - semantic_store: Semantic search, stats, count operations

Based on:
- Complementary Learning Systems (McClelland et al. 1995)
- Sleep consolidation (Diekelmann & Born 2010)
- Pattern extraction (Gilboa & Marlatte 2017)

Created: 2026-02-13 (Phase 1, Sub-phase 1.1)
"""

import json
import uuid
import logging
from datetime import datetime, timedelta, timezone
from collections import defaultdict

from modules.config import (
    COLLECTION_NAME,
    USER_ID,
    CONSOLIDATION_CLUSTER_MIN_SIZE,
    CONSOLIDATION_SIMILARITY_THRESHOLD,
    CONSOLIDATION_MAX_EPISODES_PER_RUN,
    RECONSOLIDATION_PE_THRESHOLD,
)
from modules.pg_store import pg
from modules.config_pg import get_conn as get_pg_conn
from modules.consolidation_common import (
    _embed_text, _cosine_similarity,
    init_consolidation_db, get_embed_cache_info, _get_oai,
    _embed_text_cached,
    _consolidation_conn,  # noqa: F401 - re-exported for self_model, assessment
)
from modules.utils import now_iso
from modules.bmr import bmr_should_consolidate, bmr_should_prune
from modules.temporal_renorm import run_temporal_renormalization
from modules.secret_redact import redact_secrets

# ============================================================
# FACADE RE-EXPORTS (backward compatibility)
#
# Production code and tests import symbols from modules.consolidation.
# These re-exports ensure all existing import paths continue to work.
# ============================================================
from modules.reconsolidation import (  # noqa: F401
    detect_contradiction,
    check_reconsolidation,
    correct_memory,
    mark_as_labile,
    clear_expired_labile,
    _extract_key_entities,
    queue_correction_suggestion,
    get_pending_corrections,
    expire_stale_corrections,
    CORRECTION_PATTERNS,
    NEGATION_MARKERS,
)
from modules.semantic_store import (  # noqa: F401
    search_semantic,
    get_semantic_facts,
    get_consolidation_stats,
    count_unconsolidated_episodic,
)

_logger = logging.getLogger(__name__)


# ============================================================
# CONSOLIDATION PIPELINE
# ============================================================

def run_consolidation(scope: str = "full", lookback_hours: int = 24) -> str:
    """Main consolidation pipeline. Executes 5 phases.

    Args:
        scope: "full" (all phases) | "light" (clustering only, no LLM) | "manual"
        lookback_hours: Hours to look back for unconsolidated episodes

    Returns:
        Report string with consolidation results

    Phases:
        1. SELECTION: Score recent episodes by importance
        2. CLUSTERING: Group by semantic similarity + topic
        3. EXTRACTION: Extract generalizable facts (full scope only)
        4. INTEGRATION: Create/update semantic memories
        5. PRUNING: Mark consolidated, apply decay
    """
    batch_id = str(uuid.uuid4())[:8]
    start = datetime.now()

    # Phase 1: Selection
    candidates = _phase_selection(lookback_hours)
    if not candidates:
        return f"[consolidation:{batch_id}] No unconsolidated episodes found in last {lookback_hours}h"

    # Phase 1.5: Sharpe scoring (shadow/on/off via feature flag)
    try:
        from modules.sharpe import sharpe_score_candidates
        candidates = sharpe_score_candidates(candidates)
    except Exception as e:
        _logger.warning("[sharpe] Scoring skipped: %s", e)

    # Phase 2: Clustering
    clusters = _phase_clustering(candidates)

    # Phase 2.5: Graph Edge Creation (densify spreading activation network)
    edges_created = _phase_graph_edges(clusters)

    # Phase 2.6: Causal chain extraction (Sprint 5.5)
    causal_chains_count = _extract_causal_chains()

    # Phase 2.7: Cross-topic bridging — dream creativity (S3-05)
    bridge_edges = _phase_cross_topic_bridges(clusters)

    # Phase 2.8: Temporal renormalization (Sprint 12, PN-21)
    # Episode clusters → Events → Narratives → Themes
    renorm_result = {"events": 0, "narratives": 0, "themes": 0}
    try:
        renorm_result = run_temporal_renormalization(clusters)
    except Exception as e:
        _logger.warning("[renorm] Temporal renormalization skipped: %s", e)

    # Collect topics touched across all clusters (proposal 186: enrich payload for CX)
    _topics_touched = set()
    for c in clusters:
        for ep in c.get("episodes", c.get("episode_ids", [])):
            if isinstance(ep, dict):
                _topics_touched.add(ep.get("topic", ep.get("category", "")))

    result = {
        "batch_id": batch_id,
        "scope": scope,
        "lookback_hours": lookback_hours,
        "episodes_scanned": len(candidates),
        "clusters_found": len(clusters),
        "edges_created": edges_created,
        "causal_chains": causal_chains_count,
        "bridge_edges": bridge_edges,
        "renorm_events": renorm_result.get("events", 0),
        "renorm_narratives": renorm_result.get("narratives", 0),
        "renorm_themes": renorm_result.get("themes", 0),
        "facts_extracted": 0,
        "facts_created": 0,
        "facts_updated": 0,
        "contradictions_found": 0,
        "episodes_pruned": 0,
        "topics": list(_topics_touched)[:10],
    }

    if scope == "full" and clusters:
        # Phase 3: Extraction (uses LLM)
        facts = _phase_extraction(clusters)

        # Phase 3b: Self-knowledge extraction (Proposal #63 Fix 1)
        # Conway & Pleydell-Pearce 2000: self-memory system needs identity facts
        self_facts = _phase_extract_self(clusters)
        if self_facts:
            facts.extend(self_facts)

        result["facts_extracted"] = len(facts)

        # Phase 4: Integration
        if facts:
            integration = _phase_integration(facts)
            result["facts_created"] = integration.get("created", 0)
            result["facts_updated"] = integration.get("updated", 0)
            result["contradictions_found"] = integration.get("contradictions", 0)

        # Phase 5: Pruning
        consolidated_ids = []
        for c in clusters:
            consolidated_ids.extend(c.get("episode_ids", []))
        if consolidated_ids:
            pruning = _phase_pruning(consolidated_ids)
            result["episodes_pruned"] = pruning.get("marked_consolidated", 0)
            result["consolidated_ids"] = consolidated_ids  # CX-4b: for SS boost in wiring

    # Phase 6: Compression (full scope only)
    compression_result = _phase_compression(scope=scope)
    result["episodes_compressed"] = compression_result.get("episodes_archived", 0)
    result["compression_summaries"] = compression_result.get("summaries_created", 0)

    # Phase 7: Checkpoint Compression (full scope only)
    result_checkpoint = _phase_checkpoint_compression(scope)
    result["checkpoints_deleted"] = result_checkpoint.get("trivial_deleted", 0)
    result["checkpoints_compressed"] = result_checkpoint.get("progress_compressed", 0)
    result["checkpoint_summaries"] = result_checkpoint.get("summaries_created", 0)

    # Housekeeping: expire stale correction suggestions
    try:
        expired = expire_stale_corrections()
        result["corrections_expired"] = expired
    except Exception:
        pass

    # Log the run
    duration_ms = int((datetime.now() - start).total_seconds() * 1000)
    result["duration_ms"] = duration_ms
    _log_consolidation_run(result)

    # Emit CONSOLIDATION_COMPLETE so wiring handlers can react (Baars 1988 broadcast)
    try:
        from modules.events import event_bus, Events
        event_bus.emit(Events.CONSOLIDATION_COMPLETE, result)
    except Exception:
        pass

    report = (
        f"[consolidation:{batch_id}] {scope} complete\n"
        f"  Episodes scanned: {result['episodes_scanned']}\n"
        f"  Clusters found: {result['clusters_found']}\n"
        f"  Facts extracted: {result['facts_extracted']}\n"
        f"  Facts created: {result['facts_created']}\n"
        f"  Facts updated: {result['facts_updated']}\n"
        f"  Contradictions: {result['contradictions_found']}\n"
        f"  Episodes pruned: {result['episodes_pruned']}\n"
        f"  Duration: {duration_ms}ms"
    )
    return report


def _get_causal_edge_counts(candidate_ids: list) -> dict:
    """Sprint 5.1: Batch-fetch causal edge counts for candidate IDs.

    Returns {id: causal_edge_count} for boosting selection score.
    Woodward 2003: causally connected episodes carry more structural value.
    """
    if not candidate_ids:
        return {}

    from modules.config import connect_fts

    try:
        conn = connect_fts()
        try:
            placeholders = ",".join("?" * len(candidate_ids))
            rows = conn.execute(
                f"SELECT from_id, COUNT(*) as cnt FROM spreading_edges "
                f"WHERE edge_type IN ('causes', 'enables') "
                f"AND from_id IN ({placeholders}) GROUP BY from_id",
                candidate_ids
            ).fetchall()
            return {r[0]: r[1] for r in rows}
        finally:
            conn.close()
    except Exception:
        return {}


def _compute_episode_snr(text: str, payload: dict) -> float:
    """Compute Signal-to-Noise Ratio for consolidation gating (S2-04).

    High SNR = structured, topical, information-rich → promote to semantic.
    Low SNR = noise, fragments, status messages → keep episodic.

    Squire & Alvarez 1995: hippocampal consolidation is selective.
    Gilboa & Marlatte 2017: schema-consistent info consolidates faster.

    Returns:
        SNR score 0.0-1.0. Below 0.3 = reject from consolidation.
    """
    signals = 0.0
    total = 4.0  # Number of signal components

    # Signal 1: Text length (very short = noise)
    text_len = len(text.strip())
    if text_len > 100:
        signals += 1.0
    elif text_len > 40:
        signals += 0.5

    # Signal 2: Has identifiable topic (not 'general')
    themes = payload.get("narrative_themes", [])
    topic = payload.get("metadata", {}).get("topic", "")
    if themes and themes != ["general"]:
        signals += 1.0
    elif topic and topic != "general":
        signals += 0.7

    # Signal 3: Has meaningful metadata (not bare-bones)
    has_category = bool(payload.get("ownership_category", ""))
    has_source = bool(payload.get("ownership_source", ""))
    if has_category and has_source:
        signals += 1.0
    elif has_category or has_source:
        signals += 0.5

    # Signal 4: Word diversity (unique words / total words)
    words = text.lower().split()
    if len(words) >= 5:
        diversity = len(set(words)) / len(words)
        signals += min(1.0, diversity * 1.5)
    else:
        signals += 0.3  # Too few words to judge

    return signals / total


def _phase_selection(lookback_hours: int) -> list:
    """Phase 1: Select unconsolidated episodes from last N hours.

    Scrolls Qdrant codi_memories excluding already-consolidated points.
    Scores by importance * recency and caps at MAX_EPISODES_PER_RUN.

    Returns:
        List of dicts: [{id, data, payload, score}]
    """
    cutoff = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(hours=lookback_hours)
    from modules.config import IMPORTANCE_WEIGHTS as importance_weights

    candidates = []
    offset = None
    max_scroll = CONSOLIDATION_MAX_EPISODES_PER_RUN * 5  # safety cap for scrolling
    scrolled = 0

    while scrolled < max_scroll:
        pts, next_offset = pg.scroll(
            filters={"consolidation_status_not": "consolidated"},
            limit=100,
            is_semantic=False,
            offset=offset,
        )
        if not pts:
            break

        for p in pts:
            payload = p.payload or {}
            created_str = payload.get("created_at", "")

            # Parse created_at and check lookback window
            try:
                created = datetime.fromisoformat(str(created_str).replace("Z", "+00:00"))
                if created.tzinfo:
                    created = created.astimezone(timezone.utc).replace(tzinfo=None)
            except Exception:
                continue

            if created < cutoff:
                continue

            # Skip if no meaningful text
            text = payload.get("data", "")
            if not text or len(text) < 10:
                continue

            # Proposal #60 Fix 2: Skip checkpoint episodes — inherently episodic (Tulving 1972)
            _cat = payload.get("metadata", {}).get("category", "")
            if not _cat:
                _cat = payload.get("ownership_category", "")
            if _cat == "checkpoint":
                continue

            # Score: importance * recency * emotional priority
            # Payne & Kensinger 2010: arousal x valence interaction
            imp = payload.get("narrative_importance", "medium")
            imp_w = importance_weights.get(imp, 0.5)
            hours_ago = max(0.1, (datetime.now() - created).total_seconds() / 3600)
            recency = 1.0 / (1.0 + hours_ago / lookback_hours)  # 0-1, higher = more recent

            # Emotional consolidation priority (McGaugh 2004, Payne & Kensinger 2010)
            pad_enc = payload.get("pad_at_encoding", {})
            arousal_enc = abs(float(pad_enc.get("A", 0.0))) if isinstance(pad_enc, dict) else 0.0
            pleasure_enc = float(pad_enc.get("P", 0.0)) if isinstance(pad_enc, dict) else 0.0
            negativity_bonus = 0.3 if pleasure_enc < -0.2 else 0.0
            emotional_priority = min(1.0, arousal_enc * (1.0 + negativity_bonus))

            score = imp_w * 0.45 + recency * 0.30 + emotional_priority * 0.25

            # S2-04: SNR gate — filter out low-information episodes (CL-15)
            # Episodes that are too short, have no topic, or are noise
            # should stay episodic. Only high-SNR episodes become semantic.
            # Squire & Alvarez 1995: hippocampal → neocortical transfer is selective.
            snr = _compute_episode_snr(text, payload)
            if snr < 0.3:
                continue  # Too noisy for consolidation

            candidates.append({
                "id": str(p.id),
                "data": text,
                "payload": payload,
                "score": score * snr,  # SNR modulates consolidation priority
                "created_at": created,
                "topics": payload.get("narrative_themes", []),
            })

        scrolled += len(pts)
        if not next_offset:
            break
        offset = next_offset

    # Sprint 5.1: Boost candidates with causal edges (Woodward 2003)
    # Memories on causal chains get consolidated first — they carry structural value
    if candidates:
        causal_counts = _get_causal_edge_counts([c["id"] for c in candidates])
        if causal_counts:
            for c in candidates:
                causal_n = causal_counts.get(c["id"], 0)
                if causal_n > 0:
                    c["score"] = c["score"] + 0.2 * min(1.0, causal_n / 3.0)

    # Sort by score descending and cap
    candidates.sort(key=lambda x: -x["score"])
    selected = candidates[:CONSOLIDATION_MAX_EPISODES_PER_RUN]
    _logger.info("Selection: %d/%d candidates from %d scrolled (causal boost applied)", len(selected), len(candidates), scrolled)
    return selected


def _phase_clustering(candidates: list) -> list:
    """Phase 2: Group episodes by topic, then split large groups into subclusters.

    Strategy:
    1. Group candidates by primary topic (narrative_themes[0])
    2. Keep groups with >= CLUSTER_MIN_SIZE members
    3. For large groups (>10), split into subclusters using vector similarity

    Returns:
        List of clusters: [{topic, episode_ids, texts, count}]
    """
    if not candidates:
        return []

    # Step 1: Group by primary topic
    topic_groups = defaultdict(list)
    for c in candidates:
        topics = c.get("topics", [])
        primary = topics[0] if topics else "general"
        topic_groups[primary].append(c)

    clusters = []
    for topic, members in topic_groups.items():
        if len(members) < CONSOLIDATION_CLUSTER_MIN_SIZE:
            continue

        if len(members) <= 10:
            clusters.append({
                "topic": topic,
                "episode_ids": [m["id"] for m in members],
                "texts": [m["data"] for m in members],
                "count": len(members),
                # Sprint 6 FIX-14: pass timestamps for temporal renormalization
                "timestamps": [m.get("created_at", "") for m in members if m.get("created_at")],
            })
        else:
            subclusters = _subcluster_by_vector(topic, members)
            clusters.extend(subclusters)

    _logger.info("Clustering: %d clusters from %d topic groups", len(clusters), len(topic_groups))
    return clusters


def _subcluster_by_vector(topic: str, members: list) -> list:
    """Split a large topic group into coherent subclusters using vectors.

    Greedy approach: pick seed, gather neighbors above threshold, repeat.
    """
    member_ids = [m["id"] for m in members]
    try:
        pts = pg.get_by_ids(member_ids, with_vectors=True)
        vec_map = {str(p.id): p.vector for p in pts if p.vector is not None}
    except Exception as e:
        _logger.error("Subcluster vector fetch failed for '%s': %s", topic, redact_secrets(str(e)))
        return [{
            "topic": topic,
            "episode_ids": member_ids,
            "texts": [m["data"] for m in members],
            "count": len(members),
        }]

    member_map = {m["id"]: m for m in members}
    unassigned = set(member_ids)
    subclusters = []

    while len(unassigned) >= CONSOLIDATION_CLUSTER_MIN_SIZE:
        seed_id = next(iter(unassigned))
        seed_vec = vec_map.get(seed_id)
        if seed_vec is None:
            unassigned.discard(seed_id)
            continue

        cluster_ids = [seed_id]
        for other_id in list(unassigned):
            if other_id == seed_id:
                continue
            other_vec = vec_map.get(other_id)
            if other_vec is None:
                continue
            sim = _cosine_similarity(seed_vec, other_vec)
            if sim >= CONSOLIDATION_SIMILARITY_THRESHOLD:
                cluster_ids.append(other_id)

        if len(cluster_ids) >= CONSOLIDATION_CLUSTER_MIN_SIZE:
            subclusters.append({
                "topic": topic,
                "episode_ids": cluster_ids,
                "texts": [member_map[cid]["data"] for cid in cluster_ids if cid in member_map],
                "count": len(cluster_ids),
            })
            for cid in cluster_ids:
                unassigned.discard(cid)
        else:
            unassigned.discard(seed_id)

    if subclusters:
        _logger.info("Subclustered '%s': %d subclusters from %d members", topic, len(subclusters), len(members))
    else:
        _logger.info("'%s': no subclusters, using full group (%d members)", topic, len(members))
        subclusters = [{
            "topic": topic,
            "episode_ids": member_ids,
            "texts": [m["data"] for m in members],
            "count": len(members),
        }]

    return subclusters


def _build_extraction_prompt(topic: str, episodes_block: str, num_episodes: int) -> str:
    """Build the LLM prompt for semantic fact extraction.

    Prompt is in Spanish (our working language) to ensure facts are generated
    in Spanish. See consolidation quality audit 2026-02-20.
    """
    return f"""Eres un sistema de consolidacion de memoria que extrae HECHOS SEMANTICOS de memorias episodicas.

Un hecho semantico es conocimiento declarativo reutilizable en contextos futuros.
NO es una opinion, prescripcion, motivacion, ni algo trivialmente obvio.

IDIOMA: Responde SIEMPRE en espanol. Los hechos deben estar escritos en espanol.

TEMA: "{topic}"
NUMERO DE EPISODIOS: {num_episodes}

EPISODIOS:
{episodes_block}

== CATEGORIAS ==
TECHNICAL: Como funcionan sistemas, herramientas, APIs o servicios (parametros, comportamientos, restricciones, configuraciones)
PROCEDURAL: Como lograr tareas especificas (secuencias de pasos, prerequisitos, comandos)
RELATIONAL: Hechos sobre personas, sus preferencias, patrones de comportamiento, relaciones
ARCHITECTURAL: Disenos de sistema, flujos de datos, integraciones, esquemas, infraestructura
CONTEXTUAL: Estados de proyectos, decisiones tomadas, hitos alcanzados, restricciones descubiertas
SELF: Conocimiento sobre Codi mismo — identidad, capacidades, limitaciones, historia, relacion con Hare, lecciones aprendidas, preferencias propias
CAUSAL: Relaciones causales entre eventos, decisiones o estados — "X causo Y porque mecanismo". Incluir SOLO si hay mecanismo explicito (no mera correlacion). Woodward 2003: causalidad requiere invarianza bajo intervencion.

== EJEMPLOS DE BUENOS HECHOS ==
- {{"fact": "El workflow TIAW-MainSync usa un cron trigger cada 2 minutos para sincronizar inventario WSC a Supabase", "category": "TECHNICAL", "confidence": 0.90, "specificity": "high"}}
- {{"fact": "Qdrant requiere creacion explicita de la coleccion con vector_size y distancia antes de cualquier upsert", "category": "TECHNICAL", "confidence": 0.85, "specificity": "high"}}
- {{"fact": "Hare prefiere revisar el plan de implementacion antes de que se ejecute cualquier codigo", "category": "RELATIONAL", "confidence": 0.90, "specificity": "high"}}
- {{"fact": "La coleccion codi_semantic usa text-embedding-3-small con 1536 dimensiones y distancia coseno", "category": "ARCHITECTURAL", "confidence": 0.95, "specificity": "high"}}
- {{"fact": "Para desplegar cambios en workflows de n8n, primero desactivar, luego actualizar, luego reactivar", "category": "PROCEDURAL", "confidence": 0.80, "specificity": "high"}}
- {{"fact": "Codi fue creado el 16 de enero 2026 y su identidad evoluciono de asistente generico a CTO del equipo de agentes", "category": "SELF", "confidence": 0.90, "specificity": "high"}}
- {{"fact": "Codi tiene un arco de desarrollo observable: formacion de identidad (semana 2-3), introspeccion (semana 4), foco relacional (semana 5-8)", "category": "SELF", "confidence": 0.85, "specificity": "high"}}
- {{"fact": "La falta de restart del MCP server despues de cambios en server.py causa que las herramientas nuevas no aparezcan porque el proceso carga los modulos en memoria al inicio", "category": "CAUSAL", "confidence": 0.90, "specificity": "high"}}
- {{"fact": "Usar CODI_WRITE_MODE=async elimina la latencia perceptible en guardar memorias porque el write_worker drena la cola en segundo plano sin bloquear la respuesta", "category": "CAUSAL", "confidence": 0.85, "specificity": "high"}}

== EJEMPLOS DE HECHOS MALOS (NO PRODUCIR) ==
- "Es importante testear el codigo" -> prescriptivo, no es un hecho
- "Los workflows tienen multiples nodos" -> trivialmente obvio
- "Se observaron buenas mejoras" -> vago, sin detalle concreto
- "La comunicacion es clave para el exito" -> platitud generica
- "El sistema se actualizo exitosamente" -> evento puntual, no conocimiento reutilizable
- "Se cambio la linea 42 de server.py" -> detalle de implementacion transitorio
- "El error de conexion se resolvio reiniciando" -> error temporal, no patron

== FILTRO ANTI-BASURA ==
IGNORAR y NO extraer:
- Detalles de lineas de codigo especificas (ej. "linea 42", "commit abc123")
- Errores temporales ya resueltos (ej. "el deploy fallo y se arreglo")
- Estados transitorios de deploys o PRs
- Numeros de version que cambian frecuentemente
- Conteos que se desactualizan rapido (ej. "hay 138 facts", "830 tests")

== REGLAS ==
1. Extraer solo hechos con DETALLES CONCRETOS (nombres, parametros, comportamientos especificos)
2. Cada hecho debe ser una oracion declarativa util si se recupera 30 dias despues
3. Confianza (0.0-1.0): basada en cuantos episodios lo soportan y cuan consistente es la evidencia
4. Especificidad debe ser "high" -- si no puedes incluir un detalle concreto, no incluyas el hecho
5. Combinar observaciones superpuestas en un hecho mas fuerte en vez de listar casi-duplicados
6. Maximo 5 hechos por cluster
7. Si menos de 2 hechos cumplen el estandar de calidad, devuelve menos -- NO rellenes con hechos malos

Responde SOLO con un array JSON (sin markdown, sin explicacion):
[{{"fact": "...", "category": "TECHNICAL|PROCEDURAL|RELATIONAL|ARCHITECTURAL|CONTEXTUAL|SELF", "confidence": 0.85, "specificity": "high"}}]"""


# ---------------------------------------------------------------------------
# Classical edge classification (Sprint Independence-1)
# Replaces LLM call with keyword scoring + Jaccard similarity.
# Bilingual (es/en). Zero API cost, ~100x faster, deterministic.
# ---------------------------------------------------------------------------

_CAUSAL_SIGNALS = {
    "porque", "caused", "therefore", "por eso", "resultado", "led to",
    "triggered", "causing", "provoco", "genero", "produjo", "consecuencia",
    "due to", "result of", "hence", "so that", "made", "hizo que",
    "origin", "causa", "causo", "derived", "derivado",
}
_ENABLE_SIGNALS = {
    "permite", "enabled", "allows", "habilita", "posibilita", "facilita",
    "makes possible", "provides", "context", "prerequisite", "setup",
    "configured", "configuro", "installed", "instalo", "prepared",
    "preparo", "setting up", "base para", "necesario para",
}
_PREVENT_SIGNALS = {
    "prevents", "contradicts", "blocks", "impide", "bloquea",
    "incompatible", "instead", "rather", "sino", "en vez de",
}
_TEMPORAL_SIGNALS = {
    "after", "then", "despues", "luego", "siguiente", "antes",
    "before", "prior", "previo", "once", "cuando", "when",
}


def _classify_pair(text_a: str, text_b: str) -> tuple:
    """Classify relationship between two memory texts using keyword scoring.

    Returns (edge_type, confidence) where edge_type is one of:
    causes, enables, prevents, co_occurs.
    """
    combined = (text_a + " " + text_b).lower()
    words_a = set(text_a.lower().split())
    words_b = set(text_b.lower().split())
    union = words_a | words_b
    overlap = len(words_a & words_b) / max(len(union), 1)

    # Score each type by signal word presence
    scores = {
        "causes": 0.0,
        "enables": 0.0,
        "prevents": 0.0,
        "co_occurs": overlap * 0.4,  # topical overlap baseline
    }

    for signal in _CAUSAL_SIGNALS:
        if signal in combined:
            scores["causes"] += 0.25
    for signal in _ENABLE_SIGNALS:
        if signal in combined:
            scores["enables"] += 0.25
    for signal in _PREVENT_SIGNALS:
        if signal in combined:
            scores["prevents"] += 0.25
    for signal in _TEMPORAL_SIGNALS:
        if signal in combined:
            scores["causes"] += 0.1  # temporal cues boost causal

    # Pick highest scoring type
    best_type = max(scores, key=scores.get)
    best_score = scores[best_type]

    # Require minimum signal strength for non-default types
    if best_type != "co_occurs" and best_score < 0.25:
        best_type = "co_occurs"
        best_score = scores["co_occurs"]

    # Map score to confidence
    if best_type == "co_occurs":
        confidence = min(1.0, 0.4 + overlap)
    else:
        confidence = min(1.0, 0.5 + best_score * 0.5)

    return best_type, round(confidence, 2)


def _classical_classify_edges(ids: list, texts: list) -> tuple:
    """Classify edge types between memory pairs using classical NLP.

    Drop-in replacement for _llm_classify_edges. Same interface:
    Returns (types_map, confidence_map) where:
        - types_map: {(from_id, to_id): edge_type_str}
        - confidence_map: {(from_id, to_id): float}
    """
    if len(ids) < 2 or len(texts) < 2:
        return {}, {}

    n = min(len(ids), len(texts), 6)

    # Generate pairs (same logic as before: adjacent pairs)
    pair_keys = []
    for i in range(n):
        for j in range(i + 1, min(i + 3, n)):
            pair_keys.append((ids[i], ids[j]))

    if not pair_keys:
        return {}, {}

    types_map = {}
    conf_map = {}
    for i_idx, (id_a, id_b) in enumerate(pair_keys):
        # Find text indices from ids
        try:
            a_pos = ids[:n].index(id_a)
            b_pos = ids[:n].index(id_b)
        except ValueError:
            continue

        etype, conf = _classify_pair(texts[a_pos][:200], texts[b_pos][:200])
        types_map[(id_a, id_b)] = etype
        types_map[(id_b, id_a)] = etype
        conf_map[(id_a, id_b)] = conf
        conf_map[(id_b, id_a)] = conf

    return types_map, conf_map


def _detect_confounds(conn, edge_types_map: dict) -> dict:
    """Sprint 5.4: Detect confounded edges (Pearl 2009, Chapter 3).

    For each A→B causal edge: find C where C→A and C→B both exist.
    If found, reclassify A→B as 'confounded' — a common cause explains
    the correlation, not direct causation.

    Args:
        conn: SQLite connection to spreading_edges
        edge_types_map: {(from_id, to_id): edge_type} from LLM classification

    Returns:
        Updated edge_types_map with confounded reclassifications.
    """
    # Collect causal edges to check
    causal_pairs = [
        (a, b) for (a, b), etype in edge_types_map.items()
        if etype in ('causes', 'enables')
    ]
    if not causal_pairs:
        return edge_types_map

    updated = dict(edge_types_map)

    try:
        for a_id, b_id in causal_pairs:
            # Find C nodes that point to A
            causes_a = {r[0] for r in conn.execute(
                "SELECT from_id FROM spreading_edges WHERE to_id = ? "
                "AND edge_type IN ('causes', 'enables')",
                (a_id,)
            ).fetchall()}

            if not causes_a:
                continue

            # Find C nodes that also point to B
            causes_b = {r[0] for r in conn.execute(
                "SELECT from_id FROM spreading_edges WHERE to_id = ? "
                "AND edge_type IN ('causes', 'enables')",
                (b_id,)
            ).fetchall()}

            # Confounders = C that causes both A and B
            confounders = causes_a & causes_b
            if confounders:
                updated[(a_id, b_id)] = 'confounded'
                updated[(b_id, a_id)] = 'confounded'
                _logger.debug(
                    "Confound detected: %s→%s has %d common causes",
                    a_id[:8], b_id[:8], len(confounders)
                )
    except Exception as e:
        _logger.debug("Confound detection error: %s", e)

    return updated


def _compute_causal_strength(
    from_id: str, to_id: str,
    edge_type: str, llm_confidence: float,
    conn,
) -> float:
    """Sprint 5.3: Compute graded causal edge strength.

    Formula: strength = 0.3*temporal_prior + 0.3*co_access + 0.4*llm_confidence

    Components:
      - temporal_prior: Edge type encodes temporal ordering strength.
        Causal types (causes/prevents) imply strong temporal structure;
        co_occurs implies none. (Granger 1969: causation requires temporal precedence)
      - co_access: Reinforcement from prior observations of this edge pair.
        Edges seen in multiple consolidation rounds have higher co_access.
        (Hebb 1949: neurons that fire together wire together)
      - llm_confidence: LLM's classification confidence (0-1).
    """
    # Component 1: Temporal prior by edge type (0.3 weight)
    _TYPE_TEMPORAL = {
        "causes": 0.9, "prevents": 0.8, "enables": 0.7,
        "confounded": 0.3, "co_occurs": 0.2,
    }
    temporal = _TYPE_TEMPORAL.get(edge_type, 0.2)

    # Component 2: Co-access frequency / reinforcement (0.3 weight)
    co_access = 0.0
    if conn:
        try:
            row = conn.execute(
                "SELECT strength FROM spreading_edges "
                "WHERE from_id = ? AND to_id = ?",
                (from_id, to_id),
            ).fetchone()
            if row and row[0] is not None:
                # Prior edge exists — use its strength as reinforcement evidence
                co_access = min(1.0, float(row[0]))
        except Exception:
            pass

    # Component 3: LLM confidence (0.4 weight)
    llm_conf = max(0.0, min(1.0, llm_confidence))

    strength = 0.3 * temporal + 0.3 * co_access + 0.4 * llm_conf
    return max(0.1, min(1.0, round(strength, 3)))


def _phase_graph_edges(clusters: list) -> int:
    """Phase 2.5: Create consolidated_with edges within clusters.

    S2-05: Edges typed as 'consolidated' (Rasch & Born 2013).
    Stored both in Qdrant payload (consolidated_with) and SQLite
    spreading_edges (for typed graph analysis).
    Sprint 5.3: Uses graded causal strength instead of fixed defaults.
    """
    from modules.config import GRAPH_AUTO_CONNECT_MAX, FTS_DB_PATH
    from modules.utils import now_iso

    total_edges = 0
    # S2-05: Also record typed edges in SQLite
    edge_conn = None
    try:
        from modules.config import connect_fts as _connect_fts
        edge_conn = _connect_fts()
        from modules.spreading import _init_edge_table, _record_edges
        _init_edge_table(edge_conn)
    except Exception:
        edge_conn = None

    ts = now_iso()
    for cluster in clusters:
        ids = cluster.get("episode_ids", [])
        texts = cluster.get("texts", [])
        if len(ids) < 2:
            continue

        # Sprint 1, item 1.6: Edge typing for clusters with texts
        # Sprint 5.3: Returns (types_map, confidence_map) tuple
        # Sprint Independence-1: Classical NLP replaces LLM call
        edge_types_map = {}
        confidence_map = {}
        if texts and len(texts) >= 2:
            edge_types_map, confidence_map = _classical_classify_edges(ids, texts)

        # Sprint 5.4: Detect confounds in classified edges
        if edge_types_map and edge_conn:
            edge_types_map = _detect_confounds(edge_conn, edge_types_map)

        for i, mid in enumerate(ids):
            neighbors = [oid for j, oid in enumerate(ids) if j != i]
            neighbors = neighbors[:GRAPH_AUTO_CONNECT_MAX]

            if neighbors:
                try:
                    pg.update_payload(mid, {
                        'consolidated_with': neighbors,
                    })
                    # Sprint 5.3: Record edges with graded causal strength
                    if edge_conn:
                        for nb in neighbors:
                            pair_key = (mid, nb)
                            etype = edge_types_map.get(pair_key, "co_occurs")
                            conf = confidence_map.get(pair_key, 0.5)
                            strength = _compute_causal_strength(
                                mid, nb, etype, conf, edge_conn,
                            )
                            _record_edges(edge_conn, mid, [nb], ts,
                                          edge_type=etype, strength=strength)
                    total_edges += len(neighbors)
                except Exception:
                    pass

    if edge_conn:
        try:
            edge_conn.close()
        except Exception:
            pass

    if total_edges > 0:
        _logger.info("Graph edges: %d consolidated edges across %d clusters",
                     total_edges, len(clusters))
    return total_edges


def _phase_cross_topic_bridges(clusters: list) -> int:
    """Phase 2.7: Create cross-topic bridge edges between clusters.

    Neuroscience basis:
    - Walker & Stickgold 2004: Sleep enhances creative problem-solving
      by integrating disparate memory traces
    - Wagner et al. 2004: Sleep-dependent insight (hidden rules discovered
      at 2x rate after sleep vs wake)
    - Cai et al. 2009: REM primes associative networks for remote associations
    - Lewis & Durrant 2011: Gist extraction across schemas during consolidation

    Strategy:
    1. Compute centroid for each cluster (using PG vectors)
    2. Compare centroids between clusters of DIFFERENT topics
    3. Create 'cross_topic_bridge' spreading_edges for pairs above threshold
    """
    BRIDGE_SIMILARITY_THRESHOLD = 0.55  # Lower than within-topic (0.7)
    MAX_BRIDGES_PER_RUN = 10

    if len(clusters) < 2:
        return 0

    # Step 1: Compute centroid embedding for each cluster via PG
    centroids = []
    for cluster in clusters:
        ids = cluster.get("episode_ids", [])[:5]
        topic = cluster.get("topic", "general")
        if not ids:
            continue
        try:
            pts = pg.get_by_ids(ids, with_vectors=True)
            vecs = [p.vector for p in pts if getattr(p, "vector", None)]
            if vecs:
                centroid = [sum(v[i] for v in vecs) / len(vecs) for i in range(len(vecs[0]))]
                centroids.append({
                    "topic": topic,
                    "centroid": centroid,
                    "episode_ids": cluster["episode_ids"],
                })
        except Exception:
            continue

    # Step 2: Compare centroids between DIFFERENT topics
    bridge_pairs = []
    for i, a in enumerate(centroids):
        for j, b in enumerate(centroids):
            if j <= i or a["topic"] == b["topic"]:
                continue
            sim = _cosine_similarity(a["centroid"], b["centroid"])
            if sim >= BRIDGE_SIMILARITY_THRESHOLD:
                bridge_pairs.append((i, j, sim))

    bridge_pairs.sort(key=lambda x: -x[2])
    bridge_pairs = bridge_pairs[:MAX_BRIDGES_PER_RUN]

    # Step 3: Create cross-topic bridge edges in PG spreading_edges
    bridges_created = 0
    from modules.utils import now_iso
    ts = now_iso()

    edge_conn = None
    try:
        from modules.config import connect_fts as _connect_fts
        edge_conn = _connect_fts()
        from modules.spreading import _init_edge_table, _record_edges
        _init_edge_table(edge_conn)
    except Exception:
        edge_conn = None

    for i, j, sim in bridge_pairs:
        a_ids = centroids[i]["episode_ids"][:3]
        b_ids = centroids[j]["episode_ids"][:3]

        for aid in a_ids:
            try:
                pg.update_payload(aid, {
                    "cross_topic_bridges": b_ids[:3],
                })
                if edge_conn:
                    from modules.spreading import _record_edges
                    _record_edges(edge_conn, aid, b_ids[:3], ts,
                                  edge_type="cross_topic_bridge", strength=sim)
                bridges_created += len(b_ids[:3])
            except Exception:
                pass

    if edge_conn:
        try:
            edge_conn.close()
        except Exception:
            pass

    if bridges_created > 0:
        _logger.info(
            "Cross-topic bridges: %d edges across %d pairs (threshold=%.2f)",
            bridges_created, len(bridge_pairs), BRIDGE_SIMILARITY_THRESHOLD,
        )

    return bridges_created


def _extract_causal_chains(fts_db_path: str = None) -> int:
    """Sprint 5.5: Phase 2.6 — Extract causal chains via BFS over causal edges.

    Pearl 2009: causal reasoning requires explicit chain structure A→B→C.
    BFS from root nodes (only outgoing causal edges, no incoming) following
    causes/enables edges. Stores chains in causal_chains table.

    Returns number of chains stored.
    """
    from modules.config import FTS_DB_PATH as _FTS_DB, connect_fts as _connect_fts

    db = fts_db_path or _FTS_DB
    try:
        conn = _connect_fts(db)
        try:
            # Check if causal_chains table exists (created by migration 019)
            if not conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='causal_chains'"
            ).fetchone():
                return 0

            # Find roots: nodes with outgoing causal edges but NO incoming causal edges
            all_from = {r[0] for r in conn.execute(
                "SELECT DISTINCT from_id FROM spreading_edges WHERE edge_type IN ('causes', 'enables')"
            ).fetchall()}
            all_to = {r[0] for r in conn.execute(
                "SELECT DISTINCT to_id FROM spreading_edges WHERE edge_type IN ('causes', 'enables')"
            ).fetchall()}
            roots = all_from - all_to

            if not roots:
                return 0

            chains_stored = 0
            seen_chains = set()

            for root in roots:
                # BFS along causal edges
                chain = [root]
                frontier = [root]
                visited = {root}

                while frontier:
                    node = frontier.pop(0)
                    children = conn.execute(
                        "SELECT to_id, COALESCE(strength, 0.5) FROM spreading_edges "
                        "WHERE from_id = ? AND edge_type IN ('causes', 'enables') LIMIT 5",
                        (node,)
                    ).fetchall()

                    best_child = None
                    best_strength = 0.0
                    for child_id, child_strength in children:
                        if child_id not in visited and child_strength > best_strength:
                            best_child = child_id
                            best_strength = child_strength

                    if best_child:
                        chain.append(best_child)
                        visited.add(best_child)
                        frontier.append(best_child)

                if len(chain) < 2:
                    continue

                chain_key = "->".join(sorted(chain[:3]))
                if chain_key in seen_chains:
                    continue
                seen_chains.add(chain_key)

                # Compute average chain strength
                strengths = []
                for i in range(len(chain) - 1):
                    row = conn.execute(
                        "SELECT COALESCE(strength, 0.5) FROM spreading_edges "
                        "WHERE from_id = ? AND to_id = ?",
                        (chain[i], chain[i + 1])
                    ).fetchone()
                    if row:
                        strengths.append(row[0])
                avg_strength = sum(strengths) / len(strengths) if strengths else 0.5

                chain_id = str(uuid.uuid4())[:8]
                conn.execute("""
                    INSERT OR REPLACE INTO causal_chains (chain_id, nodes, strength, mechanism, created_at)
                    VALUES (?, ?, ?, ?, ?)
                """, (chain_id, json.dumps(chain), avg_strength, "", now_iso()))
                chains_stored += 1

            conn.commit()
            _logger.info("Causal chains: %d chains extracted from %d roots", chains_stored, len(roots))
            return chains_stored
        finally:
            conn.close()

    except Exception as e:
        _logger.error("Causal chain extraction error: %s", e)
        return 0






def _phase_extraction(clusters: list) -> list:
    """Phase 3: Extract semantic facts from each cluster using LLM."""
    if not clusters:
        return []

    all_facts = []
    skipped_low_quality = 0

    for cluster in clusters:
        topic = cluster["topic"]
        texts = cluster["texts"][:15]
        episode_ids = cluster["episode_ids"][:15]

        episodes_block = "\n".join(f"- {t}" for t in texts)
        prompt = _build_extraction_prompt(topic, episodes_block, len(texts))

        try:
            from modules.llm_router import llm_complete

            raw = llm_complete("semantic_extract", prompt)
            if not raw:
                _logger.warning("[consolidation] LLM failed for semantic_extract")
                continue

            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            extracted = json.loads(raw)

            if not isinstance(extracted, list):
                extracted = [extracted]

            cluster_accepted = 0
            for item in extracted:
                fact_text = item.get("fact", "").strip()
                confidence = float(item.get("confidence", 0.5))
                specificity = item.get("specificity", "low").strip().lower()
                category = item.get("category", "CONTEXTUAL").strip().upper()

                if specificity != "high":
                    skipped_low_quality += 1
                    continue
                if not fact_text or len(fact_text) < 20:
                    skipped_low_quality += 1
                    continue
                if confidence < 0.4:
                    skipped_low_quality += 1
                    continue

                valid_categories = {"TECHNICAL", "PROCEDURAL", "RELATIONAL", "ARCHITECTURAL", "CONTEXTUAL", "SELF", "CAUSAL"}
                if category not in valid_categories:
                    category = "CONTEXTUAL"

                all_facts.append({
                    "fact_text": fact_text,
                    "confidence": min(1.0, max(0.0, confidence)),
                    "evidence_count": len(texts),
                    "source_episode_ids": episode_ids,
                    "topic": topic,
                    "category": category,
                })
                cluster_accepted += 1

            _logger.info("Extraction '%s': %d accepted, %d filtered from %d episodes",
                         topic, cluster_accepted, len(extracted) - cluster_accepted, len(texts))

        except Exception as e:
            _logger.error("Extraction error for '%s': %s", topic, redact_secrets(str(e)))
            continue

    _logger.info("Extraction total: %d facts accepted, %d filtered for low quality, from %d clusters",
                 len(all_facts), skipped_low_quality, len(clusters))
    return all_facts


def _phase_extract_self(clusters: list) -> list:
    """Phase 3b: Extract SELF-knowledge from episodic memories.

    Proposal #63 Fix 1 (Conway & Pleydell-Pearce 2000):
    Self-knowledge is cross-cutting — not captured by topic-based clustering.
    This pass looks for identity, capability, relationship, and growth facts
    across ALL clusters, not within individual topic clusters.
    """
    # Gather recent episodes across all clusters (cross-topic)
    all_texts = []
    all_ids = []
    for cluster in clusters:
        for text, eid in zip(cluster.get("texts", [])[:10], cluster.get("episode_ids", [])[:10]):
            all_texts.append(text)
            all_ids.append(eid)
    if len(all_texts) < 5:
        return []

    # Sample up to 30 episodes for self-extraction
    episodes_block = "\n".join(f"- {t}" for t in all_texts[:30])

    prompt = f"""From these episodic memories of Codi (an AI agent with persistent memory), extract SELF-KNOWLEDGE facts.

Focus on:
1. Identity: "Codi is/values/prefers..."
2. Capabilities: "Codi can/learned to..."
3. Relationships: "Codi's relationship with Hare involves..."
4. Growth: "Codi has changed from... to..."
5. Goals: "Codi is working toward..."

Only extract facts that appear STABLE across multiple episodes (not one-time events).
Return a JSON array of objects with keys: "fact", "subcategory" (identity/capability/relationship/growth/goal), "confidence" (0-1).
If no self-knowledge found, return [].

Episodes:
{episodes_block}"""

    try:
        from modules.llm_router import llm_complete

        raw = llm_complete("self_extract", prompt)
        if not raw:
            return []

        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        extracted = json.loads(raw)
        if not isinstance(extracted, list):
            return []

        self_facts = []
        for item in extracted:
            fact_text = item.get("fact", "").strip()
            if not fact_text or len(fact_text) < 15:
                continue
            confidence = float(item.get("confidence", 0.7))
            if confidence < 0.5:
                continue
            subcategory = item.get("subcategory", "general").strip().lower()
            self_facts.append({
                "fact_text": fact_text,
                "confidence": min(1.0, max(0.0, confidence)),
                "evidence_count": min(len(all_texts), 30),
                "source_episode_ids": all_ids[:10],
                "topic": "self",
                "category": "SELF",
            })

        _logger.info("Self-extraction: %d self-facts from %d episodes", len(self_facts), len(all_texts))
        return self_facts[:5]  # Cap at 5 per consolidation run
    except Exception as e:
        _logger.error("Self-extraction error: %s", redact_secrets(str(e)))
        return []


def _phase_integration(facts: list) -> dict:
    """Phase 4: Integrate facts into semantic store (codi_semantic)."""
    if not facts:
        return {"created": 0, "updated": 0, "contradictions": 0}

    created = 0
    updated = 0
    contradictions = 0
    now = now_iso()

    for fact in facts:
        try:
            fact_text = fact["fact_text"]
            embedding = _embed_text(fact_text)

            existing_pts = pg.query_vector(embedding, limit=3, is_semantic=True)

            # Sprint 9: BMR-scored consolidation merge (replaces cosine threshold)
            duplicate = None
            for hit in existing_pts:
                hit_metadata = hit.payload or {}
                fact_metadata = {"topic": fact.get("topic"), "category": fact.get("category")}
                if bmr_should_consolidate(hit.score, hit_metadata, fact_metadata):
                    duplicate = hit
                    break

            if duplicate:
                old_payload = duplicate.payload or {}
                old_fact_text = old_payload.get("fact_text", old_payload.get("data", ""))

                # CA1 comparator: check for contradiction before merging
                # (Kumaran & Maguire 2006, Proposal #180)
                if old_fact_text:
                    contra = detect_contradiction(old_fact_text, fact_text)
                    if contra["prediction_error"] >= RECONSOLIDATION_PE_THRESHOLD:
                        contradictions += 1
                        _logger.warning(
                            "Contradiction PE=%.2f: '%s...' vs '%s...'",
                            contra["prediction_error"],
                            old_fact_text[:50], fact_text[:50],
                        )
                        # Queue for reconsolidation review
                        try:
                            queue_correction_suggestion(
                                old_memory_id=str(duplicate.id),
                                old_text=old_fact_text,
                                new_text=fact_text,
                                prediction_error=contra["prediction_error"],
                                shared_entities=contra["channels"].get("shared_entities", []),
                                channels=contra["channels"],
                            )
                        except Exception:
                            pass
                        # Emit for CX wiring (Loop 1)
                        try:
                            from modules.events import event_bus, Events
                            event_bus.emit(Events.CONTRADICTION_DETECTED, {
                                "conflicting_memory_id": str(duplicate.id),
                                "conflicting_text": old_fact_text,
                                "new_content": fact_text,
                                "pe": contra["prediction_error"],
                                "channels": contra["channels"],
                                "shared_entities": contra["channels"].get("shared_entities", []),
                            })
                        except Exception:
                            pass
                        # Increment contradiction_count on existing fact
                        old_contra_count = int(old_payload.get("contradiction_count", 0))
                        pg.update_payload(duplicate.id, {
                            "contradiction_count": old_contra_count + 1,
                            "last_contradiction": now,
                        })
                        continue  # Do NOT merge contradictory fact

                old_sources = old_payload.get("source_episode_ids", [])
                new_sources = list(set(old_sources + fact["source_episode_ids"]))
                old_evidence = int(old_payload.get("evidence_count", 1))

                # Proposal #60 Fix 3: Evidence-proportional confidence (Koriat 1997)
                import math
                new_evidence = old_evidence + fact["evidence_count"]
                evidence_confidence = min(0.95, 0.70 + 0.05 * math.log2(max(1, new_evidence)))
                pg.update_payload(duplicate.id, {
                    "evidence_count": new_evidence,
                    "source_episode_ids": new_sources,
                    "last_observed": now,
                    "confidence": evidence_confidence,
                })
                updated += 1
                _logger.info("Updated existing fact: %s...", fact_text[:60])
            else:
                point_id = str(uuid.uuid4())

                # Canon v2, S2-12: Transfer causal_links from source episodes
                inherited_causal = []
                try:
                    src_ids = fact["source_episode_ids"][:10]
                    if src_ids:
                        src_pts = pg.get_by_ids(src_ids)
                        for sp in (src_pts or []):
                            cl = (sp.payload or {}).get("causal_links", [])
                            if cl and isinstance(cl, list):
                                inherited_causal.extend(cl)
                        # Deduplicate, cap at 10
                        inherited_causal = list(dict.fromkeys(inherited_causal))[:10]
                except Exception:
                    pass

                meta = {
                    "fact_text": fact_text,
                    "topic": fact["topic"],
                    "topics": [fact["topic"]],
                    "source_episode_ids": fact["source_episode_ids"],
                    "first_observed": now,
                    "last_observed": now,
                    "contradiction_count": 0,
                    "memory_type": "semantic",
                    "user_id": USER_ID,
                    "_v": 4.2,
                }
                if inherited_causal:
                    meta["causal_links"] = inherited_causal

                pg.add(
                    content=fact_text,
                    category=fact.get("category", "CONTEXTUAL"),
                    importance="high" if fact["confidence"] > 0.8 else "medium",
                    embedding=embedding,
                    is_semantic=True,
                    confidence=fact["confidence"],
                    evidence_count=fact["evidence_count"],
                    metadata=meta,
                )
                created += 1
                _logger.info("New semantic fact: %s...", fact_text[:60])

        except Exception as e:
            _logger.error("Integration error: %s", redact_secrets(str(e)))
            continue

    _logger.info("Integration: %d created, %d updated, %d contradictions", created, updated, contradictions)
    return {"created": created, "updated": updated, "contradictions": contradictions}


def _phase_pruning(consolidated_episode_ids: list) -> dict:
    """Phase 5: Mark episodes as consolidated and apply differential decay.

    Sprint 9: BMR-verified pruning — only mark as consolidated if
    information is genuinely preserved in semantic layer.
    """
    if not consolidated_episode_ids:
        return {"marked_consolidated": 0, "decayed": 0, "bmr_skipped": 0}

    marked = 0
    decayed = 0
    bmr_skipped = 0
    now = now_iso()

    consolidation_payload = {
        "consolidation_status": "consolidated",
        "consolidated": True,
        "consolidated_at": now,
    }

    for eid in consolidated_episode_ids:
        try:
            # Sprint 9: BMR verification before pruning
            # Check that episode's information exists in semantic layer
            episode_pts = pg.get_by_ids([eid], with_vectors=True)
            if not episode_pts:
                pg.update_payload(eid, consolidation_payload)
                marked += 1
                continue

            ep_payload = episode_pts[0].payload or {}
            ep_text = ep_payload.get("data", ep_payload.get("memory", ""))
            if not ep_text:
                pg.update_payload(eid, consolidation_payload)
                marked += 1
                continue

            # Reuse stored PG vector instead of regenerating embedding
            ep_embedding = episode_pts[0].vector or _embed_text(ep_text[:500])
            if ep_embedding:
                sem_pts = pg.query_vector(ep_embedding, limit=1, is_semantic=True)
                if sem_pts:
                    top_sem = sem_pts[0]
                    sem_payload = top_sem.payload or {}
                    if bmr_should_prune(top_sem.score, ep_payload, sem_payload):
                        pg.update_payload(eid, consolidation_payload)
                        marked += 1
                    else:
                        bmr_skipped += 1
                        _logger.debug("BMR: episode %s not yet captured in semantic", eid)
                else:
                    bmr_skipped += 1
            else:
                # Fallback: mark as consolidated without BMR check
                pg.update_payload(eid, consolidation_payload)
                marked += 1
        except Exception:
            pass

    _logger.info("Pruning: %d marked consolidated, %d BMR-skipped", marked, bmr_skipped)
    return {"marked_consolidated": marked, "decayed": decayed, "bmr_skipped": bmr_skipped}


# ============================================================
# PHASE 6: COMPRESSION — Episodic memory compression
# Compresses groups of low-value consolidated episodes into summaries.
# Based on: Schema abstraction (Gilboa & Marlatte 2017), gist extraction
# ============================================================

COMPRESSION_PROMPT = """Eres un sistema de compresion de memoria. Tu tarea es comprimir {n} memorias episodicas en UN SOLO resumen.

REGLAS:
1. Maximo 200 palabras
2. Preservar: decisiones tomadas, patrones descubiertos, resultados importantes
3. Descartar: detalles de implementacion, estados transitorios, errores ya resueltos
4. Formato: parrafo narrativo que capture la esencia
5. Idioma: espanol
6. El resumen debe ser util si se recupera 30+ dias despues

MEMORIAS A COMPRIMIR:
{episodes}

Responde SOLO con el resumen (sin markdown, sin explicacion):"""


def _phase_compression(scope: str = "full") -> dict:
    """Phase 6: Compress low-value consolidated episodes into summaries.

    Only runs in 'full' scope. Requires COMPRESSION_ENABLED=True.
    """
    from modules.config import (
        COMPRESSION_MIN_AGE_DAYS, COMPRESSION_MIN_GROUP_SIZE,
        COMPRESSION_MAX_PER_RUN, COMPRESSION_ENABLED,
    )

    if not COMPRESSION_ENABLED:
        return {"compressed_groups": 0, "episodes_archived": 0, "summaries_created": 0}
    if scope != "full":
        return {"compressed_groups": 0, "episodes_archived": 0, "summaries_created": 0}

    cutoff = datetime.now() - timedelta(days=COMPRESSION_MIN_AGE_DAYS)
    now = now_iso()

    # Canon v2, S1-5: Load causal chain members to protect from compression
    try:
        from modules.spreading import get_chain_member_ids
        chain_members = get_chain_member_ids()
        _logger.info("Compression: %d causal chain members protected", len(chain_members))
    except Exception:
        chain_members = set()

    candidates = []
    offset = None
    max_scroll = COMPRESSION_MAX_PER_RUN * 5

    while len(candidates) < max_scroll:
        pts, next_offset = pg.scroll(
            filters={"consolidation_status": "consolidated"},
            limit=100,
            is_semantic=False,
            offset=offset,
        )
        if not pts:
            break

        for p in pts:
            payload = p.payload or {}

            # Skip already-compressed
            if payload.get("consolidated_compressed"):
                continue

            # Skip high-importance
            imp = payload.get("narrative_importance", "medium")
            if imp in ("critical", "high"):
                continue

            # Skip recently accessed
            acc = int(payload.get("attention_access_count", 0) or 0)
            if acc > 0:
                continue

            # Skip too recent
            try:
                created = datetime.fromisoformat(
                    str(payload.get("created_at", "")).replace("Z", "+00:00")
                )
                if created.tzinfo:
                    created = created.replace(tzinfo=None)
                if created > cutoff:
                    continue
            except Exception:
                continue

            text = payload.get("data", "")
            if not text or len(text) < 20:
                continue

            # Canon v2, S1-5: Never compress causal chain members
            if str(p.id) in chain_members:
                continue

            candidates.append({
                "id": str(p.id),
                "data": text,
                "payload": payload,
                "topics": payload.get("narrative_themes", []),
            })

        if not next_offset:
            break
        offset = next_offset

    if len(candidates) < COMPRESSION_MIN_GROUP_SIZE:
        _logger.info("Compression: only %d candidates, skipping", len(candidates))
        return {"compressed_groups": 0, "episodes_archived": 0, "summaries_created": 0}

    # Step 2: Cluster by primary topic
    topic_groups = defaultdict(list)
    for c in candidates:
        topics = c.get("topics", [])
        primary = topics[0] if topics else "general"
        topic_groups[primary].append(c)

    groups = []
    for topic, members in topic_groups.items():
        if len(members) >= COMPRESSION_MIN_GROUP_SIZE:
            groups.append({"topic": topic, "members": members})

    # Step 3: Compress each group
    compressed_groups = 0
    episodes_archived = 0
    summaries_created = 0

    for group in groups[:20]:  # max 20 groups per run
        topic = group["topic"]
        members = group["members"][:15]  # max 15 episodes per summary

        episodes_block = "\n".join(
            f"- {m['data'][:300]}" for m in members
        )
        prompt = COMPRESSION_PROMPT.format(n=len(members), episodes=episodes_block)

        try:
            from modules.llm_router import llm_complete
            summary = llm_complete("compress_episodes", prompt)

            if not summary or len(summary) < 30:
                continue

            # Create compressed memory
            original_ids = [m["id"] for m in members]
            embedding = _embed_text(summary)

            meta = {
                "narrative_themes": [topic],
                "memory_type": "compressed_episodic",
                "compressed_from": original_ids,
                "compression_ratio": f"{len(original_ids)}:1",
                "consolidation_status": "consolidated",
                "_v": 4.1,
            }
            add_result = pg.add(
                content=summary,
                category="episodio",
                importance="medium",
                embedding=embedding,
                is_semantic=False,
                metadata=meta,
            )
            actual_id = add_result["results"][0]["id"]
            summaries_created += 1

            # Mark originals as compressed
            for oid in original_ids:
                pg.update_payload(oid, {
                    "consolidated_compressed": True,
                    "compressed_into": actual_id,
                })
                episodes_archived += 1

            compressed_groups += 1
            _logger.info(
                "Compressed %d episodes -> 1 summary for topic '%s'",
                len(original_ids), topic,
            )

        except Exception as e:
            _logger.error("Compression error for '%s': %s", topic, redact_secrets(str(e)))
            continue

    # Sprint 5.6: Causal intermediary compression (minimal sufficiency)
    # Woodward 2003, Pearl 2009: compress intermediary nodes B in A→B→C
    # if B has zero activation and its info is captured by A and C
    causal_compressed = _compress_causal_intermediaries(chain_members)
    result = {
        "compressed_groups": compressed_groups,
        "episodes_archived": episodes_archived,
        "summaries_created": summaries_created,
        "causal_intermediaries_compressed": causal_compressed,
    }
    _logger.info("Compression complete: %s", result)
    return result


def _compress_causal_intermediaries(chain_members: set) -> int:
    """Sprint 5.6: Compress low-activation intermediary nodes in causal chains.

    For each 3-node chain A→B→C: if B has zero access count (never retrieved),
    create a direct edge A→C with mechanism from A→B→C, reducing redundancy.
    B stays in Qdrant but gets a `causal_compressed=True` flag.

    Woodward 2003: minimal sufficiency — simplest model that explains data.
    Pearl 2009: chain structure preserved at edge level even when middle
    node is compressed.

    Returns number of intermediaries compressed.
    """
    from modules.config import connect_fts as _connect_fts

    compressed = 0
    try:
        conn = _connect_fts()
        try:
            # Check causal_chains table exists
            exists = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='causal_chains'"
            ).fetchone()
            if not exists:
                return 0

            chains = conn.execute(
                "SELECT chain_id, nodes, strength FROM causal_chains"
            ).fetchall()

            if not chains:
                return 0

            from modules.spreading import _init_edge_table, _record_edges
            _init_edge_table(conn)
            ts = now_iso()

            for chain_id, nodes_json, chain_strength in chains:
                try:
                    nodes = json.loads(nodes_json)
                except Exception:
                    continue

                if len(nodes) < 3:
                    continue

                # Check each B node (intermediary) in A→B→C
                for i in range(1, len(nodes) - 1):
                    a_id, b_id, c_id = nodes[i - 1], nodes[i], nodes[i + 1]

                    # B must not be a protected chain member with strong connections
                    if b_id in chain_members:
                        continue

                    # Check B's access count from PG (zero = never retrieved)
                    try:
                        b_pts = pg.get_by_ids([b_id])
                        if not b_pts:
                            continue
                        b_payload = b_pts[0].payload or {}
                        b_access = int(b_payload.get("attention_access_count", 0) or 0)
                        if b_access > 0:
                            continue  # B is still being accessed — don't compress
                    except Exception:
                        continue

                    # Create summary edge A→C (bypass B)
                    _record_edges(conn, a_id, [c_id], ts,
                                   edge_type='enables',
                                   strength=chain_strength * 0.8)

                    # Mark B as causal_compressed in PG
                    try:
                        pg.update_payload(b_id, {"causal_compressed": True})
                    except Exception:
                        pass

                    compressed += 1
                    _logger.debug("Causal compress: %s→%s→%s → direct %s→%s",
                                  a_id[:8], b_id[:8], c_id[:8], a_id[:8], c_id[:8])
        finally:
            conn.close()
    except Exception as e:
        _logger.debug("Causal intermediary compression error: %s", e)

    if compressed > 0:
        _logger.info("Sprint 5.6: %d causal intermediaries compressed", compressed)
    return compressed


CHECKPOINT_COMPRESSION_PROMPT = """Eres un sistema de compresion de memoria. Estos son {n} checkpoints del dia {date}.
Genera UN resumen que capture:
- Que se hizo ese dia
- Decisiones tomadas
- Progreso logrado
- Problemas encontrados

Maximo 200 palabras. En español. Primera persona.

CHECKPOINTS:
{checkpoints}

RESUMEN:"""


def _phase_checkpoint_compression(scope: str = "full") -> dict:
    """Phase 7: Compress checkpoint memories.

    Three-tier approach:
    1. DELETE trivial (date stamps, <30 chars)
    2. COMPRESS progress notes into daily summaries
    3. PRESERVE insights (high importance or frequently accessed)

    Only runs in 'full' scope.
    """
    from modules.config import (
        COMPRESSION_MIN_GROUP_SIZE, COMPRESSION_ENABLED,
    )

    empty = {"trivial_deleted": 0, "progress_compressed": 0, "insights_preserved": 0, "summaries_created": 0}
    if not COMPRESSION_ENABLED or scope != "full":
        return empty

    now = now_iso()

    # Step 1: Scroll ALL uncompressed checkpoints
    all_points = []
    _offset = None
    while True:
        batch, _next = pg.scroll(
            filters={"category": "checkpoint"},
            limit=200,
            is_semantic=False,
            offset=_offset,
        )
        if not batch:
            break
        # Filter out already-compressed in Python
        all_points.extend([p for p in batch if not (p.payload or {}).get("consolidated_compressed")])
        if not _next or len(all_points) >= 2000:
            break
        _offset = _next
    points = all_points

    if not points:
        return empty

    # Step 2: Classify
    trivial_ids = []
    progress_by_day = defaultdict(list)
    insight_count = 0

    for p in points:
        text = str(p.payload.get("data", "")).strip()
        imp = p.payload.get("narrative_importance", "medium")
        attn = int(p.payload.get("attention_access_count", 0) or 0)
        date = str(p.payload.get("created_at", ""))[:10]

        # Tier 1: TRIVIAL — delete
        if (len(text) < 30 or text.lower().startswith("fecha")
                or text.lower().startswith("date ")
                or "date of decision" in text.lower()
                or "date of importance" in text.lower()):
            trivial_ids.append(str(p.id))
            continue

        # Tier 3: INSIGHT — preserve
        if imp in ("critical", "high") or attn >= 3:
            insight_count += 1
            continue

        # Tier 2: PROGRESS — compress by day
        progress_by_day[date].append({
            "id": str(p.id),
            "data": text,
        })

    # Step 3: Soft-delete trivial (mark as consolidated, DO NOT hard delete)
    # Changed from pg.delete() after 20,859 memories were lost on 2026-03-03.
    # Marking as consolidated makes them invisible to search but preserves data.
    trivial_deleted = 0
    for tid in trivial_ids:
        try:
            pg.update_payload(tid, {"consolidated": True, "compression_tier": "trivial"})
            trivial_deleted += 1
        except Exception as e:
            _logger.error("Checkpoint compression: soft-delete failed: %s", e)

    # Step 4: Compress progress by day
    summaries_created = 0
    progress_compressed = 0
    oai = _get_oai()

    for date, members in sorted(progress_by_day.items()):
        if len(members) < COMPRESSION_MIN_GROUP_SIZE:
            continue

        checkpoints_block = "\n".join(
            f"- {m['data']}" for m in members[:30]
        )
        prompt = CHECKPOINT_COMPRESSION_PROMPT.format(
            n=len(members), date=date, checkpoints=checkpoints_block
        )

        try:
            from modules.llm_router import llm_complete
            summary_text = llm_complete("compress_checkpoints", prompt)
            if not summary_text:
                _logger.warning("LLM failed for checkpoint compression %s", date)
                continue
        except Exception as e:
            _logger.error("Checkpoint compression LLM failed for %s: %s", date, e)
            continue

        original_ids = [m["id"] for m in members]

        summary_vec = _embed_text(summary_text)
        if not summary_vec:
            continue

        meta = {
            "memory_type": "checkpoint_summary",
            "topic": "checkpoint",
            "compressed_from": original_ids,
            "compression_ratio": f"{len(original_ids)}:1",
            "summary_date": date,
            "consolidation_status": "consolidated",
            "consolidated_compressed": False,
            "_v": 4.1,
        }
        add_result = pg.add(
            content=summary_text,
            category="checkpoint",
            importance="medium",
            embedding=summary_vec,
            is_semantic=False,
            metadata=meta,
        )
        actual_id = add_result["results"][0]["id"]
        summaries_created += 1

        # Mark originals as compressed
        for oid in original_ids:
            try:
                pg.update_payload(oid, {
                    "consolidated_compressed": True,
                    "compressed_into": actual_id,
                })
            except Exception:
                pass

        progress_compressed += len(original_ids)

    _logger.info(
        "Checkpoint compression: %d trivial deleted, %d progress compressed into %d summaries, %d insights preserved",
        trivial_deleted, progress_compressed, summaries_created, insight_count,
    )

    return {
        "trivial_deleted": trivial_deleted,
        "progress_compressed": progress_compressed,
        "summaries_created": summaries_created,
        "insights_preserved": insight_count,
    }


def _log_consolidation_run(result: dict):
    """Log a consolidation run to PostgreSQL."""
    try:
        with get_pg_conn() as conn:
            # Ensure bridge_edges and batch_topic columns exist (idempotent)
            conn.execute("ALTER TABLE consolidation_log ADD COLUMN IF NOT EXISTS bridge_edges INTEGER DEFAULT 0")
            conn.execute("ALTER TABLE consolidation_log ADD COLUMN IF NOT EXISTS batch_topic TEXT DEFAULT ''")

            # Derive batch_topic from topics touched during consolidation
            topics = result.get("topics", [])
            batch_topic = ", ".join(t for t in topics[:5] if t) if topics else result.get("scope", "")

            conn.execute("""
                INSERT INTO consolidation_log
                (batch_id, scope, lookback_hours, episodes_scanned, clusters_found,
                 facts_extracted, facts_created, facts_updated, contradictions_found,
                 episodes_pruned, duration_ms, consolidated_ids, bridge_edges, batch_topic,
                 created_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb, %s, %s, %s)
            """, (
                result["batch_id"], result["scope"], result.get("lookback_hours", 24),
                result["episodes_scanned"], result["clusters_found"],
                result["facts_extracted"], result["facts_created"],
                result["facts_updated"], result["contradictions_found"],
                result["episodes_pruned"], result["duration_ms"],
                json.dumps(result.get("consolidated_ids", [])[:100]),
                result.get("bridge_edges", 0),
                batch_topic,
                now_iso()
            ))
    except Exception as e:
        _logger.warning("Could not log run: %s", redact_secrets(str(e)))


# ============================================================
# MCP TOOL REGISTRATION
# ============================================================

def register_consolidation_tools(mcp):
    """Register consolidation MCP tools."""
    mcp.tool()(run_consolidation)
    mcp.tool()(correct_memory)
    mcp.tool()(get_semantic_facts)
    mcp.tool()(get_consolidation_stats)
    mcp.tool()(get_pending_corrections)
