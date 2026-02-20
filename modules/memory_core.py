"""
Codi Memory - Core Memory Operations
Basic CRUD, search, ownership, and timeline tools.
"""

import logging
import os
import json
from datetime import datetime

_logger = logging.getLogger(__name__)

from modules.config import memory, qdrant, USER_ID, COLLECTION_NAME, SEMANTIC_COLLECTION, BACKUP_FILE, now_iso
from modules.secret_redact import redact_secrets
from qdrant_client.models import Filter, FieldCondition, MatchValue, Range
from modules.utils import (
    enrich_with_ownership, resolve_memory_id, calculate_confidence_score,
    compute_pad_retrieval_bias, _get_emotional_state,
)
from modules.activation import compute_unified_activation
from modules.memory_smart import index_memory_fts, delete_memory_fts, search_fts, bm25_rank_to_score
from modules.consolidation import search_semantic
from modules.events import event_bus, Events
from modules.destructive_guard import (
    is_guard_enabled, is_legacy_enabled, compute_fingerprint,
    request_confirmation, validate_and_consume, log_security_event,
)
from modules.retrieval_metadata import (
    wrap_retrieval_result, init_failed_searches_table, log_failed_search,
    metacognitive_control,
)


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
                except Exception:
                    pass

        return f"Restauradas {restored} memorias desde backup"
    except Exception as e:
        return f"Error restaurando: {redact_secrets(str(e))}"


def _add_memory_sync(content: str, category: str, source: str, importance: str) -> str:
    """Synchronous add_memory implementation (original pipeline)."""
    result = memory.add(
        messages=[{"role": "user", "content": content}],
        user_id=USER_ID,
        metadata={"category": category}
    )

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

    mem_id_fts = None
    try:
        if result and result.get("results"):
            for r in result["results"]:
                mem_id_fts = r.get("id")
                if mem_id_fts:
                    index_memory_fts(mem_id_fts, content, category, source, importance)
    except Exception as fts_err:
        _logger.warning("FTS index error in add_memory: %s", fts_err)

    try:
        event_bus.emit(Events.MEMORY_STORED, {
            'memory_id': mem_id_fts,
            'content': content[:200],
            'category': category,
            'source': source,
            'importance': importance,
        })
    except Exception:
        pass

    return f"Memoria guardada con ownership: {result}"


def add_memory(content: str, category: str = "general",
               source: str = "experienced", importance: str = "medium") -> str:
    """
    Guarda un nuevo recuerdo con ownership tagging.

    Supports 3 write modes via CODI_WRITE_MODE env var:
      - sync (default): synchronous pipeline
      - shadow: sync + enqueue for validation
      - async: enqueue + immediate ACK

    Args:
        content: El contenido a recordar
        category: Categoria (identidad, aprendizaje, episodio, proyecto, general)
        source: Como obtuve esta memoria (experienced, told, learned, inferred)
        importance: Importancia (critical, high, medium, low)

    Returns:
        Confirmacion del recuerdo guardado
    """
    from modules.interface import _get_write_mode
    write_mode = _get_write_mode()

    try:
        if write_mode == "async":
            from modules.write_queue import enqueue_write_job, compute_dedupe_key
            dedupe_key = compute_dedupe_key("add_memory", content)
            enqueue_result = enqueue_write_job(
                kind="add_memory",
                payload={"content": content, "category": category, "source": source, "importance": importance},
                priority=3 if importance in ("critical", "high") else 5,
                dedupe_key=dedupe_key,
            )
            preview = content[:120] + "..." if len(content) > 120 else content
            return json.dumps({
                "action": "queued",
                "mode": write_mode,
                "job_id": enqueue_result["job_id"],
                "kind": "add_memory",
                "preview": preview,
                "tags": [category, importance],
                "eta_seconds": 30,
                "dedupe_hit": enqueue_result["dedupe_hit"],
                "check": f"write_status('{enqueue_result['job_id']}')",
                "cancel": f"cancel_write('{enqueue_result['job_id']}')",
            }, ensure_ascii=False)

        # Sync execution
        result_str = _add_memory_sync(content, category, source, importance)

        if write_mode == "shadow":
            try:
                from modules.write_queue import enqueue_write_job, compute_dedupe_key
                from modules.dual_compare import record_sync_result, compute_request_fingerprint
                from modules.tracing import get_trace_id, new_trace_id
                trace_id = get_trace_id()
                if not trace_id:
                    trace_id = new_trace_id()
                dedupe_key = compute_dedupe_key("add_memory", content)
                fingerprint = compute_request_fingerprint("add_memory", content, trace_id)
                enqueue_result = enqueue_write_job(
                    kind="add_memory",
                    payload={"content": content, "category": category, "source": source, "importance": importance},
                    dedupe_key=dedupe_key,
                )
                record_sync_result(
                    fingerprint=fingerprint,
                    trace_id=trace_id,
                    kind="add_memory",
                    raw_output=result_str,
                    async_job_id=enqueue_result["job_id"],
                    async_status=enqueue_result["status"],
                )
            except Exception:
                pass

        return result_str
    except Exception as e:
        return f"Error al guardar memoria: {redact_secrets(str(e))}"


# ============================================================
# HYBRID SEARCH
# ============================================================

def _score_semantic_fact(fact: dict, current_pleasure: float = 0.0) -> float:
    """Compute fusion score for a semantic fact from codi_semantic.

    Semantic facts do NOT have ACT-R access_timestamps or BM25 index entries.
    Their score is based on:
    - Vector similarity (primary relevance signal, from Qdrant query_points)
    - Confidence (quality of the extracted fact, from LLM at consolidation time)
    - Evidence strength (how many episodes support this fact, log-normalized)
    - Recency of last observation (when was this fact last confirmed by new episodes)
    - PAD mood-congruent bias (subtle, same as episodic)

    Formula rationale:
    - vector_score dominates (0.45) because semantic relevance is the primary signal
    - confidence (0.20) acts as a quality gate -- low-confidence facts get suppressed
    - evidence (0.15) rewards facts supported by many episodes (log-normalized
      so that going from 1->5 evidence matters more than 5->25)
    - recency (0.10) keeps recently-confirmed facts slightly preferred
    - pad_bias (0.10) maintains mood-congruent retrieval consistency with episodic path

    Args:
        fact: Dict from search_semantic() with keys: id, fact, topic, confidence,
              evidence_count, score (cosine from query_points)
        current_pleasure: Current PAD pleasure dimension for mood-congruent bias

    Returns:
        Float score in ~0-1 range, comparable to episodic combined scores.
    """
    import math

    # 1. Vector similarity (already 0-1 from Qdrant cosine)
    v_score = float(fact.get("score", 0.0))

    # 2. Confidence (0-1, from LLM extraction)
    confidence = float(fact.get("confidence", 0.5))

    # 3. Evidence strength: log-normalized
    # ln(1)=0, ln(3)=1.1, ln(5)=1.6, ln(10)=2.3, ln(50)=3.9
    # Normalize to 0-1 using ln(evidence+1)/ln(51) so max at 50 episodes = 1.0
    evidence_count = max(1, int(fact.get("evidence_count", 1)))
    evidence_norm = math.log(evidence_count + 1) / math.log(51)
    evidence_norm = min(1.0, evidence_norm)

    # 4. Recency of last observation
    # Semantic facts have last_observed (ISO timestamp from consolidation)
    # More recent = higher score. Decay over 30 days to 0.
    recency = 0.5  # default: moderate recency
    last_observed = fact.get("last_observed", "")
    if last_observed:
        try:
            last_dt = datetime.fromisoformat(str(last_observed).replace("Z", "+00:00"))
            if last_dt.tzinfo:
                last_dt = last_dt.replace(tzinfo=None)
            hours_ago = max(0.1, (datetime.now() - last_dt).total_seconds() / 3600)
            # Sigmoid-like decay: 1.0 at 0h, ~0.5 at 72h, ~0.1 at 720h (30d)
            recency = 1.0 / (1.0 + (hours_ago / 72.0))
        except Exception:
            pass

    # 5. PAD mood-congruent bias (reuse same function, neutral default for semantic)
    pad_bias = compute_pad_retrieval_bias("neutral", current_pleasure)

    # Fusion formula
    combined = (
        0.45 * v_score +
        0.20 * confidence +
        0.15 * evidence_norm +
        0.10 * recency +
        0.10 * max(0, pad_bias + 0.05)
    )

    return combined


def search_memory(query: str, limit: int = 5) -> str:
    """
    Busca recuerdos relacionados con una consulta.
    Usa busqueda HIBRIDA de 3 canales:
      - Canal 1 (Episodico): vector (mem0) + BM25 (FTS5) + ACT-R activation
      - Canal 2 (Semantico): vector (codi_semantic) + confidence + evidence

    Scoring formulas:
      Episodic:  0.40*vector + 0.15*bm25 + 0.45*unified_activation (WIRING-5)
        (unified scorer absorbs importance, emotion, spreading, prediction error)
      Semantic:  0.45*vector + 0.20*confidence + 0.15*evidence + 0.10*recency + 0.10*pad

    Results from both channels compete in a unified ranking.
    Semantic facts are labeled [FACT] in output for transparency.
    """
    try:
        # ============================================================
        # METACOGNITIVE CONTROL (HOT-3): FOK -> strategy adjustment
        # ============================================================
        fok_control = None
        confidence_flag = ""
        try:
            from modules.config import FTS_DB_PATH
            fok_control = metacognitive_control(query, fts_db_path=FTS_DB_PATH)
            confidence_flag = fok_control.get("confidence_flag", "")
            limit_multiplier = fok_control.get("adjusted_limit", 1)
            limit = min(limit * limit_multiplier, 50)  # cap to prevent cascade blowup
            # Emit runtime evidence for HOT-3 scoring (Block 1995)
            try:
                from modules.events import event_bus, Events
                event_bus.emit(Events.METACOGNITIVE_CONTROL_APPLIED, {
                    "strategy": fok_control.get("strategy", ""),
                    "adjusted_limit": limit_multiplier,
                    "fok_score": fok_control.get("fok", {}).get("fok_score", None),
                })
            except Exception:
                pass
        except Exception:
            pass

        # ============================================================
        # CHANNEL 1: EPISODIC (vector + BM25 + ACT-R)
        # ============================================================

        # 1a. Busqueda vectorial (semantica) via mem0
        vector_results = memory.search(query=query, user_id=USER_ID, limit=limit * 2)

        # 1b. Busqueda BM25 (keywords) via FTS5
        bm25_results = search_fts(query, limit=limit * 4)

        # 1c. Construir mapas de scores por memory_id
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

        # ============================================================
        # CHANNEL 2: SEMANTIC (codi_semantic vector search)
        # ============================================================

        # 2a. Search semantic store (safe: returns [] if empty or error)
        semantic_results = search_semantic(query, limit=limit)

        # ============================================================
        # FUSION: Score and merge both channels
        # ============================================================

        # 3. Episodic fusion
        episodic_ids = set(list(vector_map.keys()) + list(bm25_map.keys()))

        # Prefetch episodic payloads en batch
        payload_map = {}
        try:
            id_list = [mid for mid in episodic_ids if mid]
            if id_list:
                pts = qdrant.retrieve(collection_name=COLLECTION_NAME, ids=id_list, with_payload=True)
                if pts:
                    payload_map = {str(p.id): (p.payload or {}) for p in pts}
        except Exception:
            payload_map = {}

        # Get current emotional state for mood-congruent retrieval
        try:
            emotional_state = _get_emotional_state()
            current_pleasure = emotional_state.get('current', {}).get('pleasure', 0.0)
            current_arousal = abs(emotional_state.get('current', {}).get('arousal', 0.0))
        except Exception:
            current_pleasure = 0.0
            current_arousal = 0.0

        merged = []

        # Score episodic memories
        for mid in episodic_ids:
            v_score = vector_map.get(mid, {}).get("vector_score", 0)
            b_score = bm25_map.get(mid, {}).get("bm25_score", 0)
            payload = payload_map.get(str(mid), {})

            # Mood congruence for this memory (Bower 1981)
            mem_valence = payload.get('experiential_emotional_valence', 'neutral')
            valence_num = {'positive': 1.0, 'neutral': 0.0, 'negative': -1.0}.get(mem_valence, 0.0)
            congruence = valence_num * current_pleasure

            # Unified activation (ACT-R base + importance + emotion + noise)
            act_result = compute_unified_activation(
                created_at=payload.get('created_at', ''),
                last_accessed=payload.get('attention_last_accessed', ''),
                access_count=payload.get('attention_access_count', 0),
                access_timestamps=payload.get('access_timestamps'),
                spreading_boost=float(payload.get('attention_salience', 0.5)),
                emotional_arousal=current_arousal,
                emotional_congruence=congruence,
                importance=payload.get('narrative_importance', 'medium'),
                noise=True,
            )
            activation = act_result.total

            # Simplified fusion: unified scorer absorbs importance + emotion
            combined = (
                0.40 * v_score +
                0.15 * b_score +
                0.45 * activation
            )

            # 3C: State-dependent retrieval bonus (Godden & Baddeley 1975)
            pad_enc = payload.get('pad_at_encoding')
            if pad_enc and isinstance(pad_enc, dict):
                try:
                    import math
                    p_diff = current_pleasure - float(pad_enc.get('P', 0))
                    a_diff = current_arousal - float(pad_enc.get('A', 0))
                    d_diff = 0.0  # dominance not tracked at retrieval currently
                    pad_dist = math.sqrt(p_diff**2 + a_diff**2 + d_diff**2)
                    sdr_bonus = 0.08 * math.exp(-2.0 * pad_dist)
                    combined += sdr_bonus
                except Exception:
                    pass

            vec_result = vector_map.get(mid, {}).get("result")
            bm25_text = bm25_map.get(mid, {}).get("result", {}).get("content", "")
            merged.append({
                "id": mid,
                "combined_score": combined,
                "activation": activation,
                "memory_type": "episodic",
                "vector_result": vec_result,
                "bm25_text": bm25_text,
            })

        # Batch-fetch semantic payloads for last_observed and full metadata
        semantic_payload_map = {}
        sem_ids_to_fetch = [f.get("id", "") for f in semantic_results if f.get("id")]
        if sem_ids_to_fetch:
            try:
                sem_pts = qdrant.retrieve(
                    collection_name=SEMANTIC_COLLECTION,
                    ids=sem_ids_to_fetch,
                    with_payload=True,
                )
                if sem_pts:
                    semantic_payload_map = {str(p.id): (p.payload or {}) for p in sem_pts}
            except Exception:
                pass

        # Score semantic facts
        for fact in semantic_results:
            fact_id = fact.get("id", "")
            if not fact_id:
                continue

            # Enrich fact with last_observed from batch-fetched payload
            sem_payload = semantic_payload_map.get(fact_id, {})
            fact_last_observed = sem_payload.get("last_observed", "")
            # Store full payload for access tracking later
            payload_map[f"sem_{fact_id}"] = sem_payload

            fact_with_recency = {**fact, "last_observed": fact_last_observed}
            combined = _score_semantic_fact(fact_with_recency, current_pleasure)

            merged.append({
                "id": fact_id,
                "combined_score": combined,
                "activation": float(fact.get("confidence", 0.5)),  # Use confidence as proxy for display
                "memory_type": "semantic",
                "fact_text": fact.get("fact", ""),
                "topic": fact.get("topic", ""),
                "confidence": fact.get("confidence", 0),
                "evidence_count": fact.get("evidence_count", 0),
                "vector_result": None,
                "bm25_text": "",
            })

        if not merged:
            return "No encontre recuerdos relacionados."

        # Sort unified ranking
        merged.sort(key=lambda x: -x["combined_score"])
        merged = merged[:limit]

        # ============================================================
        # GWT COMPETITION: Filter through workspace (Baars 1988, Dehaene 2011)
        # ============================================================
        try:
            from modules.competition import run_workspace_competition, CompetitionCandidate
            from modules.wiring import get_attention_schema
            focus = get_attention_schema().get("focus", "")
            candidates = [
                CompetitionCandidate(
                    content=(m.get("bm25_text") or m.get("fact_text") or "")[:200],
                    source_domain="episodic" if m["memory_type"] == "episodic" else "semantic",
                    activation=m["combined_score"],
                    memory_id=m["id"],
                    metadata={"topic": query.split()[0] if query.split() else ""},
                )
                for m in merged
            ]
            comp_result = run_workspace_competition(candidates, current_focus=focus)
            if comp_result.winners:
                winner_ids = {w.memory_id for w in comp_result.winners}
                merged = [m for m in merged if m["id"] in winner_ids]
        except Exception:
            pass  # Fallback: no competition, keep all results

        # ============================================================
        # METAMEMORY: Wrap retrieval result (WIRING-7.1)
        # ============================================================
        retrieval_meta = wrap_retrieval_result(query, merged)

        # ============================================================
        # ACCESS TRACKING: Update metadata for retrieved memories
        # ============================================================

        retrieved_ids = [item["id"] for item in merged]
        from modules.access_tracking import record_access
        ts = now_iso()
        for item in merged:
            mid = item["id"]
            try:
                if item["memory_type"] == "episodic":
                    # Episodic: update ACT-R access timestamps in codi_memories
                    payload = payload_map.get(str(mid), {})
                    access_count = int(payload.get('attention_access_count', 0) or 0)

                    access_ts = list(payload.get('access_timestamps', []) or [])
                    if not isinstance(access_ts, list):
                        access_ts = []
                    access_ts.append(ts)
                    access_ts = access_ts[-20:]  # Keep last 20

                    record_access(COLLECTION_NAME, mid, {
                        'attention_access_count': access_count + 1,
                        'attention_last_accessed': ts,
                        'access_timestamps': access_ts,
                    })
                else:
                    # Semantic: update last_observed in codi_semantic
                    record_access(SEMANTIC_COLLECTION, mid, {
                        'last_observed': ts,
                    })
            except Exception:
                pass

        # Emit MEMORY_RETRIEVED event
        try:
            episodic_count = sum(1 for m in merged if m["memory_type"] == "episodic")
            semantic_count = sum(1 for m in merged if m["memory_type"] == "semantic")
            event_bus.emit(Events.MEMORY_RETRIEVED, {
                'query': query,
                'result_count': len(merged),
                'episodic_count': episodic_count,
                'semantic_count': semantic_count,
                'top_activation': merged[0]['activation'] if merged else 0,
                'retrieved_ids': retrieved_ids,
            })
        except Exception:
            pass

        # Emit RETRIEVAL_QUALITY event (WIRING-7.4)
        try:
            event_bus.emit(Events.RETRIEVAL_QUALITY, {
                'query': query,
                'coverage': retrieval_meta.coverage,
                'confidence': retrieval_meta.confidence_estimate,
                'count': retrieval_meta.result_count,
            })
        except Exception:
            pass

        # Log failed/weak searches for FOK estimation (WIRING-7.2)
        if retrieval_meta.result_count < 2 or retrieval_meta.top_activation < 0.3:
            try:
                import sqlite3
                from modules.config import FTS_DB_PATH
                fts_conn = sqlite3.connect(FTS_DB_PATH)
                init_failed_searches_table(fts_conn)
                topic = query.split()[0] if query.split() else ""
                log_failed_search(fts_conn, query, retrieval_meta.result_count, retrieval_meta.top_activation, topic)
                fts_conn.close()
            except Exception:
                pass

        # ============================================================
        # FORMAT OUTPUT
        # ============================================================

        memories = []
        for i, item in enumerate(merged, 1):
            mem_id = item["id"]
            score = item["combined_score"]
            act = item["activation"]

            if item["memory_type"] == "semantic":
                # Format semantic facts distinctly
                text = item.get("fact_text", "")
                topic = item.get("topic", "")
                conf = item.get("confidence", 0)
                evidence = item.get("evidence_count", 0)
                memories.append(
                    f"{i}. [FACT][{topic}] [score:{score:.2f}|conf:{conf:.2f}|ev:{evidence}] {text}"
                )
            else:
                # Format episodic memories (original format)
                text = item["vector_result"].get("memory", "") if item["vector_result"] else item["bm25_text"]

                try:
                    payload = payload_map.get(str(mem_id), {}) or {}
                    source = payload.get("ownership_source", "unknown")
                    importance = payload.get("narrative_importance", "unknown")
                    created_at = payload.get("created_at", payload.get("temporal_session_id", ""))

                    date_str = ""
                    if created_at:
                        try:
                            s = str(created_at)
                            if "T" in s:
                                date_part = s[5:10]
                                time_part = s[11:16] if len(s) > 15 else ""
                                date_str = f"{date_part} {time_part}".strip()
                            else:
                                date_str = s[:10] if len(s) >= 10 else s
                        except Exception:
                            date_str = str(created_at)[:10]

                    date_display = f"[{date_str}]" if date_str else ""

                    # 3D: Temporal distance label (Friedman 1993)
                    temporal_label = ""
                    if created_at:
                        try:
                            from datetime import datetime as _dt, timezone
                            mem_dt = _dt.fromisoformat(str(created_at).replace('Z', '+00:00'))
                            now_dt = _dt.now(timezone.utc) if mem_dt.tzinfo else _dt.now()
                            delta_hours = (now_dt - mem_dt).total_seconds() / 3600
                            if delta_hours < 1:
                                temporal_label = "just_now"
                            elif delta_hours < 24:
                                temporal_label = "today"
                            elif delta_hours < 168:
                                temporal_label = "recently"
                            elif delta_hours < 720:
                                temporal_label = "weeks_ago"
                            else:
                                temporal_label = "long_ago"
                        except Exception:
                            pass
                    temporal_tag = f"[{temporal_label}]" if temporal_label else ""

                    if not text:
                        text = payload.get("data", payload.get("memory", ""))

                    memories.append(f"{i}. {date_display}{temporal_tag}[{source}|{importance}] [score:{score:.2f}|act:{act:.2f}] {text}")
                except Exception:
                    memories.append(f"{i}. [score:{score:.2f}] {text}")

        # ============================================================
        # RCJ CALIBRATION (HOT-2): Record FOK prediction vs actual outcome
        # ============================================================
        if fok_control:
            try:
                from modules.retrieval_metadata import record_rcj
                from modules.config import FTS_DB_PATH
                record_rcj(
                    query=query,
                    fok_predicted=fok_control["fok"]["fok_score"],
                    actual_coverage=retrieval_meta.coverage,
                    actual_count=retrieval_meta.result_count,
                    actual_top_activation=retrieval_meta.top_activation,
                    fts_db_path=FTS_DB_PATH,
                )
            except Exception:
                pass

        header = "Recuerdos encontrados (hybrid+ACT-R+semantic):"
        if confidence_flag:
            header = f"{confidence_flag} {header}"
        return header + "\n" + "\n".join(memories)
    except Exception as e:
        return f"Error al buscar: {redact_secrets(str(e))}"


# ============================================================
# PAD MICRO-UPDATE HELPER
# ============================================================

def _pad_guard_blocked(err: str) -> str:
    """Return GUARD BLOCKED message and trigger PAD micro-update (D-=0.1)."""
    try:
        from modules.spotlight import pad_micro_update
        pad_micro_update("repeated_block")
    except Exception:
        pass
    return f"GUARD BLOCKED: {err}"


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
            except Exception:
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
        return f"Error al obtener timeline: {redact_secrets(str(e))}"


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
        return f"Error: {redact_secrets(str(e))}"


def delete_memory(memory_id: str, dry_run: bool = False, confirm_token: str = "") -> str:
    """Elimina un recuerdo especifico por su ID. Requiere confirm_token (two-step)."""
    tool = "delete_memory"
    fp = compute_fingerprint(tool, memory_id=memory_id)

    if is_guard_enabled():
        if dry_run:
            return f"[DRY RUN] delete_memory eliminaria memoria id={memory_id}."

        if not confirm_token:
            token = request_confirmation(tool, fp)
            return (
                f"GUARD: Esto eliminara memoria id={memory_id}.\n"
                f"Para confirmar, llama de nuevo con confirm_token='{token}'"
            )

        err = validate_and_consume(confirm_token, tool, fp)
        if err:
            log_security_event("destructive_blocked", tool, outcome="blocked",
                               payload_fingerprint=fp,
                               details={"memory_id": memory_id, "reason": err})
            return _pad_guard_blocked(err)

    try:
        memory.delete(memory_id=memory_id)
        delete_memory_fts(memory_id)
        log_security_event("destructive_exec", tool, outcome="executed",
                           payload_fingerprint=fp,
                           details={"memory_id": memory_id})
        return f"Recuerdo {memory_id} eliminado."
    except Exception as e:
        log_security_event("destructive_exec", tool, outcome="error",
                           payload_fingerprint=fp,
                           details={"memory_id": memory_id, "error": str(e)[:200]})
        return f"Error al eliminar: {redact_secrets(str(e))}"


def delete_by_content(search_query: str, dry_run: bool = False,
                      confirm_token: str = "", confirm: bool = False) -> str:
    """Busca memorias por contenido y las elimina. Requiere confirm_token (two-step)."""
    tool = "delete_by_content"
    fp = compute_fingerprint(tool, search_query=search_query)

    try:
        results = memory.search(query=search_query, user_id=USER_ID, limit=10)
        if not results or not results.get("results"):
            return "No encontre memorias que coincidan."

        memories_found = results["results"]

        # Preview (dry_run or first call without token)
        def _preview() -> str:
            lines = [f"[PREVIEW] {len(memories_found)} memorias coinciden con '{search_query}':"]
            for i, mem in enumerate(memories_found, 1):
                mem_id = mem.get("id", "unknown")
                text = mem.get("memory", "")[:80]
                score = mem.get("score", 0)
                lines.append(f"{i}. [score:{score:.2f}] [id:{mem_id[:8]}] {text}...")
            return "\n".join(lines)

        if is_guard_enabled():
            if dry_run:
                return _preview()

            # Legacy bridge: confirm=True only if CODI_DESTRUCTIVE_LEGACY=on
            if confirm and not confirm_token:
                if is_legacy_enabled():
                    pass  # fall through to execute
                else:
                    return (
                        "GUARD: confirm=True ya no es suficiente. "
                        "Llama sin confirm para obtener un confirm_token."
                    )
            elif not confirm_token:
                preview = _preview()
                token = request_confirmation(tool, fp)
                return f"{preview}\n\nPara confirmar, llama con confirm_token='{token}'"
            else:
                err = validate_and_consume(confirm_token, tool, fp)
                if err:
                    return _pad_guard_blocked(err)
        else:
            # Guard off: original behavior
            if not confirm and not confirm_token:
                return _preview()

        deleted = 0
        for mem in memories_found:
            mem_id = mem.get("id")
            if mem_id:
                try:
                    memory.delete(memory_id=mem_id)
                    delete_memory_fts(mem_id)
                    deleted += 1
                except Exception:
                    pass

        log_security_event("destructive_exec", tool, outcome="executed",
                           payload_fingerprint=fp,
                           details={"search_query": search_query, "deleted": deleted,
                                    "found": len(memories_found)})
        return f"Eliminadas {deleted} memorias."
    except Exception as e:
        log_security_event("destructive_exec", tool, outcome="error",
                           payload_fingerprint=fp,
                           details={"search_query": search_query, "error": str(e)[:200]})
        return f"Error: {redact_secrets(str(e))}"


def clear_all_memories(dry_run: bool = False, confirm_token: str = "",
                       confirm_code: str = "") -> str:
    """PELIGRO: Elimina TODOS los recuerdos. Requiere confirm_token (two-step)."""
    tool = "clear_all_memories"
    fp = compute_fingerprint(tool)

    if is_guard_enabled():
        # Legacy bridge: confirm_code only if CODI_DESTRUCTIVE_LEGACY=on
        if confirm_code == "DELETE_ALL_MEMORIES" and not confirm_token:
            if is_legacy_enabled():
                pass  # fall through to execute
            else:
                return (
                    "GUARD: confirm_code ya no es suficiente. "
                    "Llama sin parametros para obtener un confirm_token."
                )

        if dry_run:
            return "[DRY RUN] clear_all_memories eliminaria TODAS las memorias del usuario."

        if not confirm_token and confirm_code != "DELETE_ALL_MEMORIES":
            token = request_confirmation(tool, fp)
            return (
                "GUARD: Esto eliminara TODAS las memorias.\n"
                f"Para confirmar, llama con confirm_token='{token}'"
            )

        if confirm_token:
            err = validate_and_consume(confirm_token, tool, fp)
            if err:
                return _pad_guard_blocked(err)

    try:
        memory.delete_all(user_id=USER_ID)
        log_security_event("destructive_exec", tool, outcome="executed",
                           payload_fingerprint=fp,
                           details={"action": "clear_all"})
        return "Todos los recuerdos han sido eliminados."
    except Exception as e:
        log_security_event("destructive_exec", tool, outcome="error",
                           payload_fingerprint=fp,
                           details={"action": "clear_all", "error": str(e)[:200]})
        return f"Error: {redact_secrets(str(e))}"


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
        return f"Error: {redact_secrets(str(e))}"


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
        return f"Error: {redact_secrets(str(e))}"


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
        return f"Error: {redact_secrets(str(e))}"


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
        return f"Error: {redact_secrets(str(e))}"


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

        from modules.access_tracking import record_access as _record_access
        _record_access(COLLECTION_NAME, full_id, {
            'narrative_importance': new_importance,
        })

        return f"Memoria {memory_id} actualizada a importancia: {new_importance}"
    except Exception as e:
        return f"Error: {redact_secrets(str(e))}"


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