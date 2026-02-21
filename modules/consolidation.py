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
from datetime import datetime, timedelta
from collections import defaultdict

from qdrant_client.models import (
    Filter, FieldCondition, MatchValue, PointStruct
)

from modules.config import (
    SEMANTIC_COLLECTION,
    COLLECTION_NAME,
    qdrant,
    USER_ID,
    CONSOLIDATION_CLUSTER_MIN_SIZE,
    CONSOLIDATION_SIMILARITY_THRESHOLD,
    CONSOLIDATION_SEMANTIC_DEDUP_THRESHOLD,
    CONSOLIDATION_MAX_EPISODES_PER_RUN,
)
from modules.consolidation_common import (
    _embed_text, _cosine_similarity, _consolidation_conn,
    init_consolidation_db, get_embed_cache_info, _get_oai,
    _embed_text_cached,
)
from modules.utils import now_iso
from modules.access_tracking import record_access
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

    result = {
        "batch_id": batch_id,
        "scope": scope,
        "episodes_scanned": len(candidates),
        "clusters_found": len(clusters),
        "facts_extracted": 0,
        "facts_created": 0,
        "facts_updated": 0,
        "contradictions_found": 0,
        "episodes_pruned": 0,
    }

    if scope == "full" and clusters:
        # Phase 3: Extraction (uses LLM)
        facts = _phase_extraction(clusters)
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


def _phase_selection(lookback_hours: int) -> list:
    """Phase 1: Select unconsolidated episodes from last N hours.

    Scrolls Qdrant codi_memories excluding already-consolidated points.
    Scores by importance * recency and caps at MAX_EPISODES_PER_RUN.

    Returns:
        List of dicts: [{id, data, payload, score}]
    """
    cutoff = datetime.now() - timedelta(hours=lookback_hours)
    from modules.config import IMPORTANCE_WEIGHTS as importance_weights

    # Exclude already-consolidated episodes
    scroll_filter = Filter(must_not=[
        FieldCondition(key="consolidation_status", match=MatchValue(value="consolidated"))
    ])

    candidates = []
    offset = None
    max_scroll = CONSOLIDATION_MAX_EPISODES_PER_RUN * 5  # safety cap for scrolling
    scrolled = 0

    while scrolled < max_scroll:
        pts, next_offset = qdrant.scroll(
            collection_name=COLLECTION_NAME,
            scroll_filter=scroll_filter,
            limit=100,
            with_payload=True,
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
                    created = created.replace(tzinfo=None)
            except Exception:
                continue

            if created < cutoff:
                continue

            # Skip if no meaningful text
            text = payload.get("data", "")
            if not text or len(text) < 10:
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

            candidates.append({
                "id": str(p.id),
                "data": text,
                "payload": payload,
                "score": score,
                "created_at": created,
                "topics": payload.get("narrative_themes", []),
            })

        scrolled += len(pts)
        if not next_offset:
            break
        offset = next_offset

    # Sort by score descending and cap
    candidates.sort(key=lambda x: -x["score"])
    selected = candidates[:CONSOLIDATION_MAX_EPISODES_PER_RUN]
    _logger.info("Selection: %d/%d candidates from %d scrolled", len(selected), len(candidates), scrolled)
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
        pts = qdrant.retrieve(
            collection_name=COLLECTION_NAME,
            ids=member_ids,
            with_vectors=True,
            with_payload=False,
        )
        vec_map = {str(p.id): p.vector for p in pts if p.vector}
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
        if not seed_vec:
            unassigned.discard(seed_id)
            continue

        cluster_ids = [seed_id]
        for other_id in list(unassigned):
            if other_id == seed_id:
                continue
            other_vec = vec_map.get(other_id)
            if not other_vec:
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

== EJEMPLOS DE BUENOS HECHOS ==
- {{"fact": "El workflow TIAW-MainSync usa un cron trigger cada 2 minutos para sincronizar inventario WSC a Supabase", "category": "TECHNICAL", "confidence": 0.90, "specificity": "high"}}
- {{"fact": "Qdrant requiere creacion explicita de la coleccion con vector_size y distancia antes de cualquier upsert", "category": "TECHNICAL", "confidence": 0.85, "specificity": "high"}}
- {{"fact": "Hare prefiere revisar el plan de implementacion antes de que se ejecute cualquier codigo", "category": "RELATIONAL", "confidence": 0.90, "specificity": "high"}}
- {{"fact": "La coleccion codi_semantic usa text-embedding-3-small con 1536 dimensiones y distancia coseno", "category": "ARCHITECTURAL", "confidence": 0.95, "specificity": "high"}}
- {{"fact": "Para desplegar cambios en workflows de n8n, primero desactivar, luego actualizar, luego reactivar", "category": "PROCEDURAL", "confidence": 0.80, "specificity": "high"}}

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
[{{"fact": "...", "category": "TECHNICAL|PROCEDURAL|RELATIONAL|ARCHITECTURAL|CONTEXTUAL", "confidence": 0.85, "specificity": "high"}}]"""


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
            from modules.consolidation_common import _get_oai
            response = _get_oai().chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=1500,
            )

            raw = response.choices[0].message.content.strip()
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

                valid_categories = {"TECHNICAL", "PROCEDURAL", "RELATIONAL", "ARCHITECTURAL", "CONTEXTUAL"}
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

            existing = qdrant.query_points(
                collection_name=SEMANTIC_COLLECTION,
                query=embedding,
                limit=3,
                with_payload=True,
            )

            duplicate = None
            for hit in existing.points:
                if hit.score >= CONSOLIDATION_SEMANTIC_DEDUP_THRESHOLD:
                    duplicate = hit
                    break

            if duplicate:
                old_payload = duplicate.payload or {}
                old_sources = old_payload.get("source_episode_ids", [])
                new_sources = list(set(old_sources + fact["source_episode_ids"]))
                old_evidence = int(old_payload.get("evidence_count", 1))

                record_access(SEMANTIC_COLLECTION, duplicate.id, {
                    "evidence_count": old_evidence + fact["evidence_count"],
                    "source_episode_ids": new_sources,
                    "last_observed": now,
                    "confidence": max(
                        float(old_payload.get("confidence", 0.5)),
                        fact["confidence"]
                    ),
                })
                updated += 1
                _logger.info("Updated existing fact: %s...", fact_text[:60])
            else:
                point_id = str(uuid.uuid4())
                payload = {
                    "fact_text": fact_text,
                    "data": fact_text,
                    "topic": fact["topic"],
                    "topics": [fact["topic"]],
                    "category": fact.get("category", "CONTEXTUAL"),
                    "source_episode_ids": fact["source_episode_ids"],
                    "evidence_count": fact["evidence_count"],
                    "first_observed": now,
                    "last_observed": now,
                    "confidence": fact["confidence"],
                    "contradiction_count": 0,
                    "memory_type": "semantic",
                    "narrative_importance": "high" if fact["confidence"] > 0.8 else "medium",
                    "user_id": USER_ID,
                    "created_at": now,
                    "_v": 4.1,
                }

                qdrant.upsert(
                    collection_name=SEMANTIC_COLLECTION,
                    points=[PointStruct(
                        id=point_id,
                        vector=embedding,
                        payload=payload,
                    )],
                )
                created += 1
                _logger.info("New semantic fact: %s...", fact_text[:60])

        except Exception as e:
            _logger.error("Integration error: %s", redact_secrets(str(e)))
            continue

    _logger.info("Integration: %d created, %d updated, %d contradictions", created, updated, contradictions)
    return {"created": created, "updated": updated, "contradictions": contradictions}


def _phase_pruning(consolidated_episode_ids: list) -> dict:
    """Phase 5: Mark episodes as consolidated and apply differential decay."""
    if not consolidated_episode_ids:
        return {"marked_consolidated": 0, "decayed": 0}

    marked = 0
    decayed = 0
    now = now_iso()

    batch_size = 50
    consolidation_payload = {
        "consolidation_status": "consolidated",
        "consolidated": True,
        "consolidated_at": now,
    }
    for eid in consolidated_episode_ids:
        try:
            record_access(COLLECTION_NAME, eid, consolidation_payload)
            marked += 1
        except Exception:
            pass

    _logger.info("Pruning: %d episodes marked as consolidated", marked)
    return {"marked_consolidated": marked, "decayed": decayed}


def _log_consolidation_run(result: dict):
    """Log a consolidation run to SQLite."""
    try:
        conn = _consolidation_conn()
        conn.execute("""
            INSERT INTO consolidation_log
            (batch_id, scope, lookback_hours, episodes_scanned, clusters_found,
             facts_extracted, facts_created, facts_updated, contradictions_found,
             episodes_pruned, duration_ms, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            result["batch_id"], result["scope"], result.get("lookback_hours", 24),
            result["episodes_scanned"], result["clusters_found"],
            result["facts_extracted"], result["facts_created"],
            result["facts_updated"], result["contradictions_found"],
            result["episodes_pruned"], result["duration_ms"], now_iso()
        ))
        conn.commit()
        conn.close()
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
