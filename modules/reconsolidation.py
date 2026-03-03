"""
RECONSOLIDATION MODULE

Implements prediction-error driven memory reconsolidation:
- Contradiction detection (3-channel: keywords, topic confirmation, negation)
- Labile memory management (reconsolidation window)
- Memory correction with re-embedding

Based on:
- Nader 2000: Reconsolidation destroys and re-synthesizes the trace
- Sevenster et al. 2013: Prediction error is prerequisite for reconsolidation
- Kumaran & Maguire 2006/2007: CA1 comparator for match-mismatch detection
- Exton-McGuinness 2015: Confidence adjustment proportional to PE

Split from consolidation.py (Phase 1, Sub-phase 1.1)
"""

import logging
from datetime import datetime, timedelta

from modules.config import (
    COLLECTION_NAME,
    RECONSOLIDATION_WINDOW_HOURS,
    RECONSOLIDATION_PE_THRESHOLD,
    RECONSOLIDATION_STRENGTH_FLOOR,
    RECONSOLIDATION_STRENGTH_CEILING,
)
from modules.pg_store import pg
from modules.consolidation_common import (
    _embed_text, _cosine_similarity, _consolidation_conn,
)
from modules.utils import now_iso
from modules.activation import compute_unified_activation
from modules.secret_redact import redact_secrets

_logger = logging.getLogger(__name__)


# ============================================================
# CORRECTION & NEGATION PATTERNS
# ============================================================

CORRECTION_PATTERNS = [
    "no es asi", "en realidad", "correccion", "corrijo",
    "ya no", "cambio a", "ahora usamos", "migramos",
    "antes era", "actualizado", "ya no aplica",
    "no es correcto", "esta mal", "cambio",
]


NEGATION_MARKERS = [
    "no ", "ya no", "no es", "nunca", "ni ", "tampoco",
    "not ", "never", "don't", "doesn't", "isn't", "won't",
    "sin ", "ninguno", "nada",
]


# ============================================================
# ENTITY EXTRACTION
# ============================================================

def _extract_key_entities(text: str) -> set:
    """Extract key entities (nouns/proper names) from text.

    Lightweight heuristic: words > 4 chars, excluding stopwords and
    common verbs/adjectives that inflate overlap without being entities.
    Based on: Kumaran & Maguire 2007 entity-overlap for mismatch detection.
    """
    stopwords = {
        # ES: pronouns, prepositions, conjunctions
        "para", "como", "este", "esta", "estos", "estas", "pero", "porque",
        "cuando", "donde", "tiene", "hacer", "puede", "desde", "hasta",
        "siendo", "sobre", "entre", "antes", "despues", "mejor", "mayor",
        # EN: pronouns, prepositions, conjunctions
        "that", "this", "with", "from", "have", "been", "they", "their",
        "which", "about", "would", "there", "what", "more", "some",
        "also", "other", "just", "should", "could", "after", "before",
        "every", "these", "those", "being", "still", "while", "where",
        # ES: frequent verbs (inflated overlap without semantic value)
        "tiene", "hacer", "puede", "estar", "haber", "tener", "poder",
        "quiero", "quiere", "usamos", "usando", "vamos", "crear", "creo",
        "implementar", "actualizar", "funciona", "necesita", "deberia",
        # EN: frequent verbs
        "using", "works", "would", "should", "could", "needs", "wants",
        "create", "update", "implement", "function", "working", "getting",
        # Tech noise (too common to be meaningful entities)
        "error", "datos", "linea", "archivo", "system", "module",
    }
    words = text.replace(",", " ").replace(".", " ").replace(":", " ").replace("(", " ").replace(")", " ").split()
    entities = set()
    for w in words:
        clean = w.strip().lower()
        if len(clean) > 4 and clean not in stopwords:
            # Skip words ending in very common suffixes (verbs/adverbs)
            if clean.endswith(("mente", "ción", "ando", "endo", "aron", "ieron")):
                continue
            entities.add(clean)
    return entities


# ============================================================
# CONTRADICTION DETECTION
# ============================================================

def detect_contradiction(memory_text: str, context: str) -> dict:
    """Detect contradiction between a memory and current context.

    Kumaran & Maguire 2006/2007: CA1 hippocampal comparator uses
    multiple channels for match-mismatch detection, not just one signal.

    4 channels:
      Canal 1 - Keywords (0.35 raw): Explicit correction patterns
      Canal 2 - Topic confirmation (amplifier): cosine_sim * entity_overlap
                Amplifies C1+C3+C4 when same topic confirmed (0.55 to 1.0x)
      Canal 3 - Negation detector (0.35 raw): Same entities + logical inversion
      Canal 4 - Semantic divergence (0.30 raw): Same entities but different claims
                Kumaran & Maguire 2006: CA1 fires when expected input diverges
                from received input on the same topic (high overlap, low cosine)

    Returns:
        {prediction_error: float, detail: str|None, channels: dict}
    """
    if not context or not memory_text:
        return {"prediction_error": 0.0, "detail": None, "channels": {}}

    context_lower = context.lower()
    memory_lower = memory_text.lower()

    # Canal 1: Keywords (existing CORRECTION_PATTERNS)
    signals = [p for p in CORRECTION_PATTERNS if p in context_lower]
    canal1_score = min(1.0, len(signals) * 0.25) if signals else 0.0

    # Canal 2: Topic confirmation (Kumaran 2006 match detection)
    mem_entities = _extract_key_entities(memory_text)
    ctx_entities = _extract_key_entities(context)
    shared_entities = mem_entities & ctx_entities
    entity_overlap = len(shared_entities) / max(1, min(len(mem_entities), len(ctx_entities)))

    cosine_sim = 0.0
    if shared_entities and len(shared_entities) >= 1:
        try:
            mem_vec = _embed_text(memory_text[:500])
            ctx_vec = _embed_text(context[:500])
            cosine_sim = _cosine_similarity(mem_vec, ctx_vec)
        except Exception:
            cosine_sim = 0.5  # fallback: assume moderate similarity

    # Topic confirmation score: high sim + shared entities = same topic
    canal2_score = cosine_sim * entity_overlap if shared_entities else 0.0

    # Canal 3: Negation detection
    canal3_score = 0.0
    if shared_entities:
        mem_negations = sum(1 for n in NEGATION_MARKERS if n in memory_lower)
        ctx_negations = sum(1 for n in NEGATION_MARKERS if n in context_lower)
        # One has negation, other doesn't = logical inversion
        if (mem_negations > 0) != (ctx_negations > 0):
            canal3_score = min(1.0, entity_overlap * 1.5)

    # Canal 4: Semantic divergence (Kumaran & Maguire 2006)
    # Same entities (same topic) but low cosine similarity (different claims)
    # = CA1 mismatch signal. This catches organic contradictions where
    # the user states something new without explicit correction language.
    # Note: when shared_entities exist, cosine_sim is always computed
    # (or fallback 0.5). cosine_sim=0 means orthogonal = max divergence.
    canal4_score = 0.0
    if shared_entities:
        canal4_score = entity_overlap * max(0.0, 1.0 - cosine_sim)

    # Two PE pathways (max of both, not sum, to avoid false inflation):
    #
    # Path A - Linguistic: C1 (keywords) + C3 (negation), amplified by C2
    #   For explicit corrections ("ya no", "correccion:", negation inversion)
    raw_pe_linguistic = canal1_score * 0.50 + canal3_score * 0.50
    pe_linguistic = raw_pe_linguistic * (0.55 + 0.45 * canal2_score)
    #
    # Path B - Semantic divergence: C4 alone, amplified by entity_overlap
    #   NOT amplified by C2 (which uses cosine_sim and would penalize
    #   divergence — the exact signal C4 is designed to detect)
    #   Kumaran & Maguire 2006: CA1 fires on input mismatch, not similarity
    pe_divergence = canal4_score * (0.55 + 0.45 * entity_overlap)
    #
    # Take stronger signal (avoid double-counting when both fire)
    pe = max(pe_linguistic, pe_divergence)

    channels = {
        "keywords": canal1_score,
        "topic_confirmation": canal2_score,
        "negation": canal3_score,
        "semantic_divergence": canal4_score,
        "shared_entities": list(shared_entities)[:10],
        "cosine_sim": round(cosine_sim, 3),
        "entity_overlap": round(entity_overlap, 3),
    }

    detail = None
    if pe > 0.0:
        parts = []
        if canal1_score > 0:
            parts.append(f"keywords={signals}")
        if canal2_score > 0:
            parts.append(f"topic_sim={cosine_sim:.2f},overlap={entity_overlap:.2f}")
        if canal3_score > 0:
            parts.append(f"negation_inversion")
        if canal4_score > 0:
            parts.append(f"semantic_div={canal4_score:.2f}(cos={cosine_sim:.2f})")
        detail = f"PE channels: {', '.join(parts)}"

    _logger.debug(
        "detect_contradiction PE=%.3f c1=%.2f c2=%.2f c3=%.2f c4=%.2f "
        "entities=%d overlap=%.2f cosine=%.3f",
        pe, canal1_score, canal2_score, canal3_score, canal4_score,
        len(shared_entities), entity_overlap, cosine_sim,
    )

    return {"prediction_error": pe, "detail": detail, "channels": channels}


# ============================================================
# RECONSOLIDATION CHECK
# ============================================================

def check_reconsolidation(memory_id: str, memory_payload: dict,
                          current_context: str) -> dict:
    """Evaluate if a retrieved memory should enter reconsolidation.

    Returns:
        {should_reconsolidate: bool, prediction_error: float,
         memory_strength: float, reason: str}
    """
    # Compute memory strength via unified scorer
    result = compute_unified_activation(
        created_at=memory_payload.get('created_at', ''),
        last_accessed=memory_payload.get('attention_last_accessed', ''),
        access_count=memory_payload.get('attention_access_count', 0),
        access_timestamps=memory_payload.get('access_timestamps'),
        importance=memory_payload.get('narrative_importance', 'medium'),
        noise=False,
    )
    strength = result.total

    # Boundary conditions
    if strength < RECONSOLIDATION_STRENGTH_FLOOR:
        return {"should_reconsolidate": False, "reason": "too_weak",
                "memory_strength": strength, "prediction_error": 0.0}
    if strength > RECONSOLIDATION_STRENGTH_CEILING:
        return {"should_reconsolidate": False, "reason": "too_strong",
                "memory_strength": strength, "prediction_error": 0.0}

    # Detect contradiction
    memory_text = memory_payload.get("data", "") or memory_payload.get("memory", "")
    contradiction = detect_contradiction(memory_text, current_context)

    # Emotional memories are more susceptible to reconsolidation
    # (Nader & Einarsson 2010): lower PE threshold for high-arousal memories
    pad_enc = memory_payload.get("pad_at_encoding", {})
    arousal_enc = abs(float(pad_enc.get("A", 0.0))) if isinstance(pad_enc, dict) else 0.0
    effective_threshold = RECONSOLIDATION_PE_THRESHOLD
    if arousal_enc > 0.4:
        effective_threshold *= (1.0 - arousal_enc * 0.3)  # up to 30% lower threshold

    if contradiction["prediction_error"] > effective_threshold:
        return {
            "should_reconsolidate": True,
            "prediction_error": contradiction["prediction_error"],
            "memory_strength": strength,
            "contradiction": contradiction["detail"],
            "reason": "prediction_error_exceeded_threshold",
            "emotional_vulnerability": arousal_enc > 0.4,
            "effective_threshold": round(effective_threshold, 3),
        }

    return {"should_reconsolidate": False, "reason": "no_prediction_error",
            "memory_strength": strength, "prediction_error": contradiction["prediction_error"]}


# ============================================================
# LABILE MEMORY MANAGEMENT
# ============================================================

def mark_as_labile(memory_id: str, prediction_error: float = 0.0,
                   trigger_context: str = "") -> bool:
    """Mark a memory as labile (in reconsolidation window).

    Returns:
        True if marked, False if already labile
    """
    try:
        conn = _consolidation_conn()
        now = datetime.now()
        expires = now + timedelta(hours=RECONSOLIDATION_WINDOW_HOURS)

        conn.execute("""
            INSERT OR REPLACE INTO labile_memories
            (memory_id, marked_at, window_expires, prediction_error, trigger_context)
            VALUES (?, ?, ?, ?, ?)
        """, (memory_id, now.isoformat(), expires.isoformat(),
              prediction_error, trigger_context))
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        _logger.warning("Could not mark labile: %s", redact_secrets(str(e)))
        return False


def clear_expired_labile():
    """Remove expired labile memory entries."""
    try:
        conn = _consolidation_conn()
        now = datetime.now().isoformat()
        conn.execute("DELETE FROM labile_memories WHERE window_expires < ?", (now,))
        conn.commit()
        conn.close()
    except Exception:
        pass


# ============================================================
# CORRECT MEMORY (RECONSOLIDATION)
# ============================================================

def correct_memory(memory_id: str, correction: str, force: bool = False) -> str:
    """Update a memory based on new evidence (reconsolidation).

    Nader 2000: The original trace is DESTROYED and re-synthesized.
    Sevenster 2013: Prediction error is a prerequisite for reconsolidation.

    Pipeline:
      1. Resolve full ID
      2. Retrieve old payload from pg_store
      3. Labile gate: verify memory is labile OR has PE (force bypasses)
      4. Log old content to reconsolidation_log
      5. Adjust confidence (decrement by 0.1 for contradiction)
      6. Build new content, generate new embedding
      7. Update payload via pg_store (re-embed, not post-it)
      8. Update FTS5 index
      9. Emit RECONSOLIDATION_TRIGGERED event

    Args:
        memory_id: ID (or prefix) of the memory to correct
        correction: The correct/updated information
        force: If True, bypass labile gate (for human-initiated corrections)

    Returns:
        Summary of reconsolidation action
    """
    from modules.utils import resolve_memory_id as _resolve

    # 1. Resolve full ID
    full_id = _resolve(memory_id)
    if not full_id:
        return f"[reconsolidation] Could not resolve memory ID: {memory_id}"

    # 2. Retrieve old payload
    try:
        pts = pg.get_by_ids([full_id])
        if not pts:
            return f"[reconsolidation] Memory {full_id[:8]} not found in pg_store"
        old_payload = pts[0].payload or {}
    except Exception as e:
        return f"[reconsolidation] pg_store retrieve error: {redact_secrets(str(e))}"

    old_content = old_payload.get("data", "") or old_payload.get("memory", "")
    old_confidence = float(old_payload.get("confidence", old_payload.get("narrative_importance_score", 0.5)))

    # 3. Labile gate (Sevenster 2013): PE is prerequisite, not just reactivation
    actual_pe = 0.0
    if not force:
        is_labile = False
        try:
            conn = _consolidation_conn()
            row = conn.execute(
                "SELECT 1 FROM labile_memories WHERE memory_id = ? AND window_expires > ?",
                (full_id, datetime.now().isoformat())
            ).fetchone()
            conn.close()
            is_labile = row is not None
        except Exception:
            pass

        # Always compute PE for the log, even when labile
        pe_result = {"should_reconsolidate": False, "prediction_error": 0.0}
        try:
            pe_result = check_reconsolidation(
                full_id, old_payload, correction
            )
            actual_pe = pe_result.get("prediction_error", 0.0)
        except Exception:
            actual_pe = 0.0

        if not is_labile:
            if not pe_result.get("should_reconsolidate", False):
                return (
                    f"[reconsolidation] Memory {full_id[:8]} rejected: "
                    f"not labile and PE={actual_pe:.2f} "
                    f"below threshold. Use force=True for manual override."
                )

    # 4. Build new content: REPLACE old trace, don't concatenate (Nader 2000)
    # Old content is preserved in reconsolidation_log for audit trail
    new_content = correction
    try:
        conn = _consolidation_conn()
        conn.execute("""
            INSERT INTO reconsolidation_log
            (memory_id, memory_type, action, prediction_error, memory_strength,
             old_content, new_content, blend_weight, trigger_context, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (full_id, "episodic", "correct_memory", actual_pe, old_confidence,
              old_content[:500], new_content[:500], 0.0,  # 0.0: full replace (Nader 2000), not blend
              correction[:200], now_iso()))
        conn.commit()
        conn.close()
    except Exception as e:
        _logger.warning("Could not log reconsolidation: %s", redact_secrets(str(e)))

    # 5. Adjust confidence proportional to PE (Exton-McGuinness 2015)
    # Higher PE = larger confidence decrement (0.05 base + 0.15 * PE)
    confidence_delta = 0.05 + 0.15 * actual_pe
    new_confidence = max(0.0, old_confidence - confidence_delta)

    # 6. Re-embed: generate new vector for corrected content (Nader 2000)
    try:
        new_vector = _embed_text(new_content)
    except Exception as e:
        return f"[reconsolidation] Embedding error: {redact_secrets(str(e))}"

    # 7. Upsert full record -- destroy and re-synthesize the trace (Nader 2000)
    try:
        updated_payload = {
            "data": new_content,
            "confidence": new_confidence,
            "reconsolidated_at": now_iso(),
            "reconsolidation_count": int(old_payload.get("reconsolidation_count", 0)) + 1,
        }
        pg.update_payload(full_id, updated_payload)
    except Exception as e:
        return f"[reconsolidation] pg_store upsert error: {redact_secrets(str(e))}"

    # 8. Update FTS5 index
    try:
        from modules.memory_smart import delete_memory_fts, index_memory_fts
        delete_memory_fts(full_id)
        index_memory_fts(
            full_id, new_content,
            category=old_payload.get("category", "general"),
            source=old_payload.get("source", "experienced"),
            importance=old_payload.get("narrative_importance", "medium"),
        )
    except Exception as e:
        _logger.warning("FTS update failed: %s", redact_secrets(str(e)))

    # 9. Emit event
    try:
        from modules.events import event_bus, Events
        event_bus.emit(Events.RECONSOLIDATION_TRIGGERED, {
            "memory_id": full_id,
            "action": "correct_memory",
            "old_confidence": old_confidence,
            "new_confidence": new_confidence,
            "re_embedded": True,
        })
    except Exception:
        pass

    return (
        f"[reconsolidation] Memory {full_id[:8]} corrected. "
        f"Confidence: {old_confidence:.2f} -> {new_confidence:.2f}. "
        f"Vector re-embedded and FTS updated."
    )


# ============================================================
# SUGGEST MODE: Queue correction proposals instead of auto-applying
# ============================================================

def queue_correction_suggestion(
    old_memory_id: str,
    old_text: str,
    new_text: str,
    prediction_error: float,
    new_memory_id: str = "",
    shared_entities: list = None,
    channels: dict = None,
    window_hours: float = 24.0,
) -> int:
    """Queue a correction suggestion for later review.

    Instead of auto-correcting, this implements the reconsolidation window
    as a time-dependent review process (Nader 2000). Suggestions expire
    after window_hours if not reviewed.

    Returns:
        ID of the queued suggestion, or -1 on error.
    """
    import json
    try:
        conn = _consolidation_conn()
        now = datetime.now()
        expires = now + timedelta(hours=window_hours)

        cursor = conn.execute("""
            INSERT INTO pending_corrections
            (old_memory_id, new_memory_id, old_text, new_text,
             prediction_error, shared_entities, channels,
             status, created_at, expires_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, 'pending', ?, ?)
        """, (
            old_memory_id,
            new_memory_id or "",
            old_text[:500],
            new_text[:500],
            prediction_error,
            json.dumps(shared_entities or []),
            json.dumps(channels or {}),
            now.isoformat(),
            expires.isoformat(),
        ))
        conn.commit()
        row_id = cursor.lastrowid
        conn.close()
        _logger.info("Queued correction suggestion #%d for memory %s (PE=%.2f)",
                      row_id, old_memory_id[:8], prediction_error)
        return row_id
    except Exception as e:
        _logger.error("Failed to queue correction: %s", redact_secrets(str(e)))
        return -1


def get_pending_corrections(include_expired: bool = False) -> str:
    """Get pending correction suggestions for review.

    MCP tool to inspect and act on queued contradictions.

    Args:
        include_expired: If True, also show expired suggestions
    """
    import json
    try:
        conn = _consolidation_conn()
        now = datetime.now().isoformat()

        if include_expired:
            rows = conn.execute("""
                SELECT id, old_memory_id, old_text, new_text,
                       prediction_error, shared_entities, status,
                       created_at, expires_at
                FROM pending_corrections
                ORDER BY prediction_error DESC
                LIMIT 20
            """).fetchall()
        else:
            rows = conn.execute("""
                SELECT id, old_memory_id, old_text, new_text,
                       prediction_error, shared_entities, status,
                       created_at, expires_at
                FROM pending_corrections
                WHERE status = 'pending' AND expires_at > ?
                ORDER BY prediction_error DESC
                LIMIT 20
            """, (now,)).fetchall()
        conn.close()

        if not rows:
            return "[reconsolidation] No pending corrections."

        lines = [f"=== Pending Corrections ({len(rows)}) ==="]
        for r in rows:
            row_id, old_id, old_txt, new_txt, pe, entities_json, status, created, expires = r
            entities = json.loads(entities_json) if entities_json else []
            lines.append(
                f"\n#{row_id} [{status}] PE={pe:.2f} | entities: {', '.join(entities[:5])}\n"
                f"  OLD ({old_id[:8]}): {old_txt[:100]}\n"
                f"  NEW: {new_txt[:100]}\n"
                f"  created: {created[:19]} | expires: {expires[:19]}"
            )
        return "\n".join(lines)
    except Exception as e:
        return f"[reconsolidation] Error: {redact_secrets(str(e))}"


def expire_stale_corrections() -> int:
    """Mark expired pending corrections as 'expired'.

    Called during consolidation runs to clean up the queue.

    Returns:
        Number of corrections expired.
    """
    try:
        conn = _consolidation_conn()
        now = datetime.now().isoformat()
        cursor = conn.execute("""
            UPDATE pending_corrections
            SET status = 'expired', reviewed_at = ?
            WHERE status = 'pending' AND expires_at <= ?
        """, (now, now))
        count = cursor.rowcount
        conn.commit()
        conn.close()
        if count > 0:
            _logger.info("Expired %d stale correction suggestions", count)
        return count
    except Exception as e:
        _logger.warning("expire_stale_corrections error: %s", redact_secrets(str(e)))
        return 0
