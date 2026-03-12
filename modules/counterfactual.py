"""
Codi Memory - Counterfactual Retrieval (Sprint 8)
==================================================
Implements Pearl's Ladder of Causation Level 3: "What if?"

Sprint 8 items:
  8.1: Counterfactual query parser — detect patterns, extract intervention
  8.2: Abduction step — find actual causal chain that led to outcome
  8.3: Intervention step — modify chain by removing/replacing target
  8.4: Prediction step — propagate modified chain to counterfactual outcome

Theory (Pearl 2009, Causality — Ladder of Causation):
  Level 1 (Association): P(y|x) — "What is?"
  Level 2 (Intervention): P(y|do(x)) — "What if I do this?"
  Level 3 (Counterfactual): P(y_x'|x) — "What if it had been different?"

3-step procedure:
  1. Abduction: use evidence to determine U (what actually happened)
  2. Action: modify model by intervening (do-operator, remove/replace target)
  3. Prediction: propagate through modified model

Created: 2026-02-28 (Sprint 8)
"""

import json
import logging
import os
import re
import sqlite3
from typing import Dict, List, Optional, Tuple

_logger = logging.getLogger(__name__)

# Counterfactual trigger patterns
_CF_PATTERNS_ES = [
    r"\bqu[eé]\s+(hubiera|hubiese|habría|habría[n]?\s+pasado)\b",
    r"\bsi\s+no\s+(hubiera|hubiese|hubieran|hubiesen)\b",
    r"\bsi\s+(no\s+)?hub[iié](ra|se|éramos|éramos)\b",
    r"\bqu[eé]\s+pasar[íi]a\s+si\b",
    r"\bqu[eé]\s+pas[oó]\s+si\b",
    r"\bsin\s+\w+,?\s+qu[eé]\b",
    r"\bde\s+no\s+haber\b",
    r"\ben\s+otro\s+escenario\b",
]

_CF_PATTERNS_EN = [
    r"\bwhat\s+if\b",
    r"\bwhat\s+would\s+have\s+happened\s+if\b",
    r"\bif\s+(we\s+)?(hadn't|had\s+not|didn't|did\s+not|never|not)\b",
    r"\bhad\s+\w+\s+not\b",
    r"\bcounterfactual\b",
    r"\balternative\s+scenario\b",
    r"\bwithout\s+\w+,?\s+what\b",
]

# Negation keywords that signal the intervention target follows
_NEGATION_MARKERS = [
    "no", "not", "sin", "never", "hadn't", "hadn't", "hadn't", "no\s+hubiera",
    "si\s+no", "without", "instead", "en\s+lugar",
]


# ============================================================
# 8.1: COUNTERFACTUAL QUERY PARSER
# ============================================================

def parse_counterfactual_query(query: str) -> Optional[Dict]:
    """Sprint 8.1: Parse counterfactual query into structured form.

    Detects "what if X hadn't happened?" patterns and extracts:
        intervention_target: the concept being negated/modified
        negated: True if the intervention removes something
        language: "es" or "en"
        raw_query: original query

    Returns None if query is not counterfactual.

    Examples:
        "what if we hadn't used Docker?" ->
            {intervention_target: "Docker", negated: True, language: "en"}
        "que hubiera pasado si no usabamos Redis?" ->
            {intervention_target: "Redis", negated: True, language: "es"}
    """
    q = query.strip()
    q_low = q.lower()

    # Detect language
    lang = "en"
    for pat in _CF_PATTERNS_ES:
        if re.search(pat, q_low, re.IGNORECASE):
            lang = "es"
            break

    # Check if it's a counterfactual query at all
    is_cf = False
    for pat in (_CF_PATTERNS_ES if lang == "es" else _CF_PATTERNS_EN):
        if re.search(pat, q_low, re.IGNORECASE):
            is_cf = True
            break

    # Also try the other language patterns
    if not is_cf:
        for pat in _CF_PATTERNS_EN:
            if re.search(pat, q_low, re.IGNORECASE):
                is_cf = True
                lang = "en"
                break

    if not is_cf:
        return None

    # Extract intervention target — noun phrase after negation marker
    intervention_target = _extract_intervention_target(q, q_low)
    negated = True  # Most counterfactuals are about negated interventions

    # Detect "what if we HAD used X?" — positive intervention
    positive_markers = ["if we had used", "si hubiéramos usado", "si usáramos",
                        "what if we used", "con", "with ", "usando"]
    has_negation_token = bool(re.search(r"\b(?:not|no)\b", q_low, re.IGNORECASE))
    for marker in positive_markers:
        if marker in q_low and not has_negation_token:
            negated = False
            break

    return {
        "intervention_target": intervention_target,
        "negated": negated,
        "language": lang,
        "raw_query": q,
    }


def _extract_intervention_target(query: str, query_low: str) -> str:
    """Extract the main concept being intervened on from a counterfactual query.

    Strategy:
    1. Look for capitalized words (proper nouns, tech names)
    2. Look for noun after negation marker
    3. Fall back to longest content word
    """
    # Priority 1: Find capitalized tech words (Docker, Redis, Qdrant, etc.)
    caps_words = re.findall(r'\b[A-Z][a-zA-Z]{2,}\b', query)
    # Filter out sentence starters and common words
    stop_caps = {"What", "If", "Would", "Have", "Had", "Why", "When", "How",
                 "The", "That", "This", "Then", "There", "We", "You", "Que",
                 "Si", "Con", "Sin", "En", "De", "La", "El", "Lo"}
    caps_words = [w for w in caps_words if w not in stop_caps]
    if caps_words:
        return caps_words[0]

    # Priority 2: Noun after negation/conditional marker
    negation_re = re.compile(
        r'\b(?:no\s+hubiera|not\s+used?|hadn\'t\s+used?|sin\s+usar?|without\s+using?|'
        r'no\s+usar?|didn\'t\s+use?|never\s+used?)\s+(\w+)',
        re.IGNORECASE
    )
    m = negation_re.search(query)
    if m:
        return m.group(1)

    # Priority 3: Noun after "if" / "si"
    if_re = re.compile(
        r'\b(?:what\s+if|que\s+(?:hubiera|pasaría)\s+si|si\s+no?)\s+\w+\s+(\w{4,})',
        re.IGNORECASE
    )
    m = if_re.search(query)
    if m:
        return m.group(1)

    # Priority 4: Longest non-stop-word
    stop_words = {
        "what", "if", "we", "had", "not", "have", "would", "been", "used",
        "que", "si", "no", "hubiera", "hubiese", "pasado", "con", "sin",
        "para", "por", "una", "uns", "los", "las", "del", "que",
    }
    words = [w.lower() for w in re.findall(r'\b\w{4,}\b', query)]
    content = [w for w in words if w not in stop_words]
    if content:
        return max(content, key=len)

    return query[:50]  # Fallback: first 50 chars of query


# ============================================================
# 8.2: ABDUCTION STEP — find factual causal chain
# ============================================================

def abduction_step(parsed: Dict, fts_db_path: str) -> List[Dict]:
    """Sprint 8.2: Find actual causal chain that led to the outcome.

    Uses causal_chains table (built by Sprint 5.5 during consolidation)
    and memories_text to find the sequence of events related to the
    intervention_target.

    Returns ordered list of memory dicts:
        [{"memory_id": ..., "content": ..., "topic": ..., "created_at": ...}]
    """
    target = (parsed.get("intervention_target") or "").lower()
    if not target:
        return []

    if not os.path.exists(fts_db_path):
        return []

    factual_chain = []

    try:
        from modules.config import connect_fts
        conn = connect_fts(fts_db_path)

        # 1. Find causal chains that include the target concept
        try:
            chains = conn.execute(
                "SELECT chain_id, nodes, strength FROM causal_chains ORDER BY strength DESC LIMIT 20"
            ).fetchall()

            relevant_nodes = set()
            for chain_id, nodes_json, strength in chains:
                try:
                    nodes = json.loads(nodes_json or "[]")
                    # Check if any node's memory content mentions the target
                    for node_id in nodes:
                        row = conn.execute(
                            "SELECT content, topic FROM memories_text WHERE memory_id = ? LIMIT 1",
                            (node_id,)
                        ).fetchone()
                        if row and target in (row[0] or "").lower():
                            # Found a relevant chain — collect all its nodes
                            relevant_nodes.update(nodes)
                except (json.JSONDecodeError, sqlite3.Error) as exc:
                    _logger.warning("Causal chain %s parse/query error: %s", chain_id, exc)
                except (KeyError, IndexError, TypeError) as exc:
                    _logger.warning("Causal chain %s row shape error: %s", chain_id, exc)
        except sqlite3.OperationalError as exc:
            _logger.warning("Causal chain table query failed: %s", exc)
            relevant_nodes = set()

        # 2. Direct text search for target concept in memories_text
        search_rows = conn.execute(
            "SELECT memory_id, content, topic, created_at FROM memories_text "
            "WHERE LOWER(content) LIKE ? ORDER BY created_at ASC LIMIT 15",
            (f"%{target}%",)
        ).fetchall()

        seen_ids = set()
        for mem_id, content, topic, created_at in search_rows:
            if mem_id not in seen_ids:
                factual_chain.append({
                    "memory_id": mem_id,
                    "content": content,
                    "topic": topic or "general",
                    "created_at": created_at or "",
                    "source": "text_match",
                })
                seen_ids.add(mem_id)

        # 3. Fetch relevant chain nodes not already included
        for node_id in list(relevant_nodes):
            if node_id in seen_ids:
                continue
            row = conn.execute(
                "SELECT memory_id, content, topic, created_at FROM memories_text "
                "WHERE memory_id = ? LIMIT 1",
                (node_id,)
            ).fetchone()
            if row:
                factual_chain.append({
                    "memory_id": row[0],
                    "content": row[1],
                    "topic": row[2] or "general",
                    "created_at": row[3] or "",
                    "source": "causal_chain",
                })
                seen_ids.add(node_id)

        conn.close()

        # Sort by creation time
        from modules.config import parse_timestamp

        def _ts_sort_key(x):
            raw = x.get("created_at", "")
            try:
                return (0, parse_timestamp(raw))
            except Exception:
                return (1, raw)

        factual_chain.sort(key=_ts_sort_key)

    except Exception as e:
        _logger.error("Abduction step error: %s", e)

    return factual_chain[:10]  # Cap at 10 memories


# ============================================================
# 8.3: INTERVENTION STEP — modify chain
# ============================================================

def intervention_step(factual_chain: List[Dict], parsed: Dict,
                      fts_db_path: str) -> List[Dict]:
    """Sprint 8.3: Modify factual chain by removing/replacing intervention target.

    For "what if X hadn't happened?":
    1. Remove memories about X from the chain
    2. Find alternative memories for that slot (same topic, different tech)
    3. Return modified chain

    Pearl 2009: do-operator surgically removes X from causal model,
    then propagates from remaining structure.
    """
    target = (parsed.get("intervention_target") or "").lower()
    negated = parsed.get("negated", True)

    if not target or not factual_chain:
        return factual_chain

    # Positive do(X): search for memories mentioning target and build intervention chain
    if not negated:
        target_memories = [m for m in factual_chain if target.lower() in str(m.get("memory", "")).lower()]
        if target_memories:
            # Amplify target evidence in chain
            intervention_chain = list(factual_chain)
            for m in target_memories:
                amplified = dict(m)
                amplified["source"] = "alternative"
                amplified["intervention"] = f"do({target})"
                intervention_chain.append(amplified)
            return intervention_chain
        return factual_chain  # fallback if no target mentions found

    # Partition: keep memories not about target, remove target memories
    kept = []
    removed = []
    for mem in factual_chain:
        content_low = (mem.get("content") or "").lower()
        if target in content_low:
            removed.append(mem)
        else:
            kept.append(mem)

    # If nothing was removed, the target wasn't in the chain
    if not removed:
        return kept  # Return as-is (no intervention possible)

    # Find alternatives for the removed slot
    if negated and fts_db_path and os.path.exists(fts_db_path):
        alternatives = _find_alternatives(removed, target, fts_db_path)
        # Insert alternatives where removed memories were
        # Find insertion point (first removed memory's position in original chain)
        if alternatives:
            removed_idx = next(
                (i for i, m in enumerate(factual_chain)
                 if m.get("memory_id") == removed[0].get("memory_id")),
                len(kept)
            )
            # Adjust for what was already kept before this point
            kept_before = [m for m in kept
                          if m.get("created_at", "") <= removed[0].get("created_at", "")]
            insert_idx = len(kept_before)
            for alt in alternatives:
                alt["source"] = "alternative"
                alt["replaced"] = target
            kept = kept[:insert_idx] + alternatives + kept[insert_idx:]

    return kept


def _find_alternatives(removed_mems: List[Dict], target: str,
                       fts_db_path: str) -> List[Dict]:
    """Find alternative memories that could replace the removed ones.

    Looks for memories with the same topic but WITHOUT the target concept.
    This simulates "what else would have been used/done instead?"
    """
    if not removed_mems:
        return []

    # Get the topic of the removed memories
    target_topic = removed_mems[0].get("topic", "general")
    removed_ids = {m["memory_id"] for m in removed_mems}

    try:
        from modules.config import connect_fts
        conn = connect_fts(fts_db_path)
        # Find similar-topic memories that don't mention the target
        rows = conn.execute(
            "SELECT memory_id, content, topic, created_at FROM memories_text "
            "WHERE topic = ? AND LOWER(content) NOT LIKE ? "
            "AND memory_id NOT IN ({}) "
            "ORDER BY created_at DESC LIMIT 3".format(
                ",".join("?" * len(removed_ids))
            ),
            (target_topic, f"%{target}%") + tuple(removed_ids)
        ).fetchall()
        conn.close()

        return [
            {"memory_id": r[0], "content": r[1], "topic": r[2] or "general",
             "created_at": r[3] or ""}
            for r in rows
        ]
    except Exception as e:
        _logger.debug("Alternative search error: %s", e)
        return []


# ============================================================
# 8.4: PREDICTION STEP — propagate modified chain forward
# ============================================================

def prediction_step(alternative_chain: List[Dict], factual_chain: List[Dict],
                    parsed: Dict, fts_db_path: str) -> Dict:
    """Sprint 8.4: Propagate modified chain forward to counterfactual outcome.

    Pearl 2009, Prediction step:
    Follow causal edges from alternative memories to find what WOULD have
    been caused. Compose natural language counterfactual narrative.

    Returns:
        {
            "factual_outcome": str — what actually happened
            "counterfactual_outcome": str — what would have happened
            "narrative": str — human-readable explanation
            "confidence": float — how confident we are (0-1)
            "factual_memories": int — count of supporting memories
            "alt_memories": int — count of alternative memories
        }
    """
    target = parsed.get("intervention_target", "?")
    language = parsed.get("language", "en")
    query = parsed.get("raw_query", "")

    # Determine factual outcome (what actually happened)
    factual_outcome = _summarize_chain(factual_chain, "factual")

    # Determine counterfactual outcome (follow edges from alternatives)
    alt_outcome = _find_downstream_effects(alternative_chain, fts_db_path)
    if not alt_outcome:
        alt_outcome = _summarize_chain(alternative_chain, "alternative")

    # Compute confidence based on chain quality
    confidence = _compute_confidence(factual_chain, alternative_chain)

    # Compose narrative
    narrative = _compose_narrative(
        target=target,
        factual_chain=factual_chain,
        alt_chain=alternative_chain,
        factual_outcome=factual_outcome,
        alt_outcome=alt_outcome,
        language=language,
        query=query,
        confidence=confidence,
    )

    return {
        "factual_outcome": factual_outcome,
        "counterfactual_outcome": alt_outcome,
        "narrative": narrative,
        "confidence": round(confidence, 2),
        "factual_memories": len(factual_chain),
        "alt_memories": len(alternative_chain),
    }


def _summarize_chain(chain: List[Dict], label: str) -> str:
    """Summarize a causal chain into a short outcome description."""
    if not chain:
        return f"No {label} memories found."

    # Use the last memory in the chain as the "outcome"
    last = chain[-1]
    content = last.get("content", "")
    # Truncate to meaningful summary
    if len(content) > 120:
        content = content[:120].rsplit(" ", 1)[0] + "..."

    topics = list({m.get("topic", "general") for m in chain})
    return f"[{', '.join(topics[:3])}] {content}"


def _find_downstream_effects(alt_chain: List[Dict], fts_db_path: str) -> str:
    """Follow causal edges from alternative memories to find downstream effects."""
    if not alt_chain or not fts_db_path or not os.path.exists(fts_db_path):
        return ""

    try:
        from modules.config import connect_fts
        conn = connect_fts(fts_db_path)
        # For each alt memory, find what it causes (outgoing causal edges)
        downstream_ids = []
        for mem in alt_chain[:3]:  # Look at top 3
            mem_id = mem.get("memory_id")
            if not mem_id:
                continue
            rows = conn.execute(
                "SELECT to_id, edge_type, strength FROM spreading_edges "
                "WHERE from_id = ? AND edge_type IN ('causes', 'enables') "
                "ORDER BY strength DESC LIMIT 3",
                (mem_id,)
            ).fetchall()
            for to_id, etype, strength in rows:
                if strength and strength > 0.3:
                    downstream_ids.append(to_id)

        if not downstream_ids:
            conn.close()
            return ""

        # Fetch content of downstream memories
        downstream = []
        for d_id in downstream_ids[:3]:
            row = conn.execute(
                "SELECT content, topic FROM memories_text WHERE memory_id = ? LIMIT 1",
                (d_id,)
            ).fetchone()
            if row:
                downstream.append(f"[{row[1]}] {row[0][:80]}")

        conn.close()
        return "; ".join(downstream) if downstream else ""

    except Exception as e:
        _logger.debug("Downstream effects error: %s", e)
        return ""


def _compute_confidence(factual_chain: List[Dict], alt_chain: List[Dict]) -> float:
    """Compute confidence score based on chain quality."""
    if not factual_chain:
        return 0.1
    if not alt_chain:
        return 0.2

    # Confidence is higher when:
    # - factual chain is longer (better evidence base)
    # - alternative chain has memories (not just empty)
    factual_score = min(1.0, len(factual_chain) / 5.0)
    alt_score = min(1.0, len([m for m in alt_chain if m.get("source") == "alternative"]) / 2.0)

    # Weight factual evidence more heavily
    return 0.6 * factual_score + 0.4 * alt_score


def _compose_narrative(target: str, factual_chain: List[Dict],
                       alt_chain: List[Dict], factual_outcome: str,
                       alt_outcome: str, language: str, query: str,
                       confidence: float) -> str:
    """Compose a human-readable counterfactual narrative."""
    if language == "es":
        if not factual_chain:
            return f"No encontré evidencia suficiente sobre '{target}' en mi memoria para construir un escenario alternativo."

        alt_mem_count = len([m for m in alt_chain if m.get("source") == "alternative"])

        lines = [
            f"**Análisis Contrafactual: ¿Qué habría pasado sin '{target}'?**\n",
            f"**Escenario Real:**\n{factual_outcome}\n",
        ]

        if alt_chain and alt_outcome:
            lines.append(f"**Escenario Alternativo:**\n{alt_outcome}\n")
            if alt_mem_count > 0:
                lines.append(
                    f"**Interpretación:** Sin '{target}', el proyecto habría seguido "
                    f"un camino diferente basado en {alt_mem_count} recuerdo(s) alternativos. "
                    f"En lugar de los eventos actuales, probablemente se habrían usado "
                    f"las alternativas encontradas en la misma temática."
                )
            else:
                lines.append(
                    f"**Interpretación:** Sin '{target}', no hay evidencia clara de "
                    f"qué habría ocurrido — el camino alternativo no está registrado en mi memoria."
                )
        else:
            lines.append(
                f"**Interpretación:** No encontré un camino alternativo claro para el escenario sin '{target}'."
            )

        lines.append(f"\n*Confianza: {confidence:.0%} | Basado en {len(factual_chain)} recuerdos*")
        return "\n".join(lines)

    else:  # English
        if not factual_chain:
            return f"I don't have enough evidence about '{target}' in memory to construct an alternative scenario."

        alt_mem_count = len([m for m in alt_chain if m.get("source") == "alternative"])

        lines = [
            f"**Counterfactual Analysis: What if '{target}' hadn't been used?**\n",
            f"**What actually happened:**\n{factual_outcome}\n",
        ]

        if alt_chain and alt_outcome:
            lines.append(f"**Alternative outcome:**\n{alt_outcome}\n")
            if alt_mem_count > 0:
                lines.append(
                    f"**Interpretation:** Without '{target}', the project would have taken "
                    f"a different path based on {alt_mem_count} alternative memory/memories. "
                    f"Instead of the current events, the alternatives found in the same "
                    f"topic area would likely have been used."
                )
            else:
                lines.append(
                    f"**Interpretation:** Without '{target}', there's no clear evidence "
                    f"of what would have happened — the alternative path isn't in my memory."
                )
        else:
            lines.append(
                f"**Interpretation:** No clear alternative path found for the scenario without '{target}'."
            )

        lines.append(f"\n*Confidence: {confidence:.0%} | Based on {len(factual_chain)} memories*")
        return "\n".join(lines)


# ============================================================
# MAIN ENTRY POINT
# ============================================================

def run_counterfactual(query: str, fts_db_path: str = None) -> Dict:
    """Sprint 8: Full counterfactual retrieval pipeline.

    Pearl 2009, Ladder of Causation Level 3:
    1. Parse query into structured intervention
    2. Abduction: find actual causal chain
    3. Intervention: modify chain (remove/replace target)
    4. Prediction: propagate to counterfactual outcome

    Args:
        query: Natural language counterfactual question
        fts_db_path: Path to FTS SQLite database

    Returns:
        {parsed, factual_chain, alternative_chain, prediction}
        or {error: "not_counterfactual"} if query isn't counterfactual
    """
    from modules.config import FTS_DB_PATH as _default_fts
    db = fts_db_path or _default_fts

    # 8.1: Parse
    parsed = parse_counterfactual_query(query)
    if parsed is None:
        return {
            "error": "not_counterfactual",
            "message": "Query does not appear to be counterfactual. "
                       "Try: 'what if we hadn't used X?' or 'que hubiera pasado si no...'",
        }

    _logger.info("Counterfactual query: target=%s negated=%s lang=%s",
                 parsed["intervention_target"], parsed["negated"], parsed["language"])

    # 8.2: Abduction — find factual chain
    factual_chain = abduction_step(parsed, db)

    # 8.3: Intervention — modify chain
    alternative_chain = intervention_step(factual_chain, parsed, db)

    # 8.4: Prediction — propagate forward
    prediction = prediction_step(alternative_chain, factual_chain, parsed, db)

    return {
        "parsed": parsed,
        "factual_chain": factual_chain,
        "alternative_chain": alternative_chain,
        "prediction": prediction,
    }
