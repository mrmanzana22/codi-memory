"""
CONSOLIDATION MODULE - Phase 1 of Codi Consciousness Project

Implements:
- Episodic -> Semantic consolidation pipeline (5 phases)
- Reconsolidation with prediction error detection
- Semantic store operations (codi_semantic collection)
- Differential decay support

Based on:
- Complementary Learning Systems (McClelland et al. 1995)
- Sleep consolidation (Diekelmann & Born 2010)
- Reconsolidation (Nader 2000, Sevenster et al. 2013)
- Pattern extraction (Gilboa & Marlatte 2017)

Created: 2026-02-13 (Phase 1, Sub-phase 1.1)
"""

import os
import json
import sqlite3
import uuid
import math
from datetime import datetime, timedelta
from collections import defaultdict

import openai
from qdrant_client.models import (
    Filter, FieldCondition, MatchValue, PointStruct
)

from modules.config import (
    FTS_DB_PATH,
    SEMANTIC_COLLECTION,
    COLLECTION_NAME,
    qdrant,
    USER_ID,
    CONSOLIDATION_CLUSTER_MIN_SIZE,
    CONSOLIDATION_SIMILARITY_THRESHOLD,
    CONSOLIDATION_SEMANTIC_DEDUP_THRESHOLD,
    CONSOLIDATION_MAX_EPISODES_PER_RUN,
    RECONSOLIDATION_WINDOW_HOURS,
    RECONSOLIDATION_PE_THRESHOLD,
    RECONSOLIDATION_STRENGTH_FLOOR,
    RECONSOLIDATION_STRENGTH_CEILING,
)
# Note: RECONSOLIDATION_MAX_BLEND removed -- full replace per Nader 2000 (blend_weight=0.0 always)
from modules.utils import now_iso
from modules.activation import compute_unified_activation

# OpenAI client (lazy, uses OPENAI_API_KEY from env)
_oai_client = None

def _get_oai():
    global _oai_client
    if _oai_client is None:
        _oai_client = openai.OpenAI()
    return _oai_client


def _embed_text(text: str) -> list:
    """Generate embedding using text-embedding-3-small (1536 dims)."""
    resp = _get_oai().embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return resp.data[0].embedding


def _cosine_similarity(a: list, b: list) -> float:
    """Compute cosine similarity between two vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


# ============================================================
# SQLITE INITIALIZATION
# ============================================================

def _consolidation_conn():
    """Get SQLite connection for consolidation tables."""
    return sqlite3.connect(FTS_DB_PATH)


def init_consolidation_db():
    """Initialize consolidation-related tables in memories_fts.db."""
    conn = _consolidation_conn()

    conn.execute("""
        CREATE TABLE IF NOT EXISTS consolidation_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            batch_id TEXT NOT NULL UNIQUE,
            scope TEXT NOT NULL,
            lookback_hours INTEGER,
            episodes_scanned INTEGER DEFAULT 0,
            clusters_found INTEGER DEFAULT 0,
            facts_extracted INTEGER DEFAULT 0,
            facts_created INTEGER DEFAULT 0,
            facts_updated INTEGER DEFAULT 0,
            contradictions_found INTEGER DEFAULT 0,
            episodes_pruned INTEGER DEFAULT 0,
            duration_ms INTEGER DEFAULT 0,
            created_at TEXT NOT NULL
        )
    """)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS reconsolidation_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            memory_id TEXT NOT NULL,
            memory_type TEXT DEFAULT 'episodic',
            action TEXT NOT NULL,
            prediction_error REAL,
            memory_strength REAL,
            old_content TEXT,
            new_content TEXT,
            blend_weight REAL,
            trigger_context TEXT,
            created_at TEXT NOT NULL
        )
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_recon_log_memory
        ON reconsolidation_log(memory_id)
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_recon_log_time
        ON reconsolidation_log(created_at)
    """)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS labile_memories (
            memory_id TEXT PRIMARY KEY,
            marked_at TEXT NOT NULL,
            window_expires TEXT NOT NULL,
            prediction_error REAL,
            trigger_context TEXT
        )
    """)

    conn.commit()
    conn.close()
    print("[consolidation] Tables initialized OK")


# Initialize on import
try:
    init_consolidation_db()
except Exception as e:
    print(f"[consolidation] WARNING: Could not init tables: {e}")


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

            # Score: importance weight * recency (hours since cutoff, inverted)
            imp = payload.get("narrative_importance", "medium")
            imp_w = importance_weights.get(imp, 0.5)
            hours_ago = max(0.1, (datetime.now() - created).total_seconds() / 3600)
            recency = 1.0 / (1.0 + hours_ago / lookback_hours)  # 0-1, higher = more recent

            score = imp_w * 0.6 + recency * 0.4

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
    print(f"[consolidation] Selection: {len(selected)}/{len(candidates)} candidates from {scrolled} scrolled")
    return selected


def _phase_clustering(candidates: list) -> list:
    """Phase 2: Group episodes by topic, then split large groups into subclusters.

    Strategy:
    1. Group candidates by primary topic (narrative_themes[0])
    2. Keep groups with >= CLUSTER_MIN_SIZE members
    3. For large groups (>10), split into subclusters using vector similarity
       to keep each cluster semantically coherent for LLM extraction

    The CONSOLIDATION_SIMILARITY_THRESHOLD (0.65) applies to pairwise
    similarity for subcluster formation, NOT to full-group average.

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
            # Small group: accept as single cluster (topic-based)
            clusters.append({
                "topic": topic,
                "episode_ids": [m["id"] for m in members],
                "texts": [m["data"] for m in members],
                "count": len(members),
            })
        else:
            # Large group: split into subclusters based on vector similarity
            subclusters = _subcluster_by_vector(topic, members)
            clusters.extend(subclusters)

    print(f"[consolidation] Clustering: {len(clusters)} clusters from {len(topic_groups)} topic groups")
    return clusters


def _subcluster_by_vector(topic: str, members: list) -> list:
    """Split a large topic group into coherent subclusters using vectors.

    Greedy approach: pick seed, gather neighbors above threshold, repeat.
    """
    # Retrieve vectors for all members
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
        print(f"[consolidation] Subcluster vector fetch failed for '{topic}': {e}")
        # Fallback: return as single cluster
        return [{
            "topic": topic,
            "episode_ids": member_ids,
            "texts": [m["data"] for m in members],
            "count": len(members),
        }]

    # Build member lookup
    member_map = {m["id"]: m for m in members}
    unassigned = set(member_ids)
    subclusters = []

    while len(unassigned) >= CONSOLIDATION_CLUSTER_MIN_SIZE:
        # Pick seed: first unassigned member
        seed_id = next(iter(unassigned))
        seed_vec = vec_map.get(seed_id)
        if not seed_vec:
            unassigned.discard(seed_id)
            continue

        # Find neighbors above threshold
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

        # Only keep if meets min size
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
            # Seed didn't form a valid cluster, remove it
            unassigned.discard(seed_id)

    if subclusters:
        print(f"[consolidation] Subclustered '{topic}': {len(subclusters)} subclusters from {len(members)} members")
    else:
        # No subclusters formed, fall back to full group
        print(f"[consolidation] '{topic}': no subclusters, using full group ({len(members)} members)")
        subclusters = [{
            "topic": topic,
            "episode_ids": member_ids,
            "texts": [m["data"] for m in members],
            "count": len(members),
        }]

    return subclusters


def _build_extraction_prompt(topic: str, episodes_block: str, num_episodes: int) -> str:
    """Build the LLM prompt for semantic fact extraction.

    Structured with:
    - Explicit categories (TECHNICAL, PROCEDURAL, RELATIONAL, ARCHITECTURAL, CONTEXTUAL)
    - Few-shot examples of good and bad facts
    - Specificity filter (only high-specificity facts pass)
    - Category field in output for downstream filtering

    Based on: Gilboa & Marlatte 2017 (schema extraction from episodes),
    Bartlett 1932 (avoiding constructive distortion in generalization).
    """
    return f"""You are a memory consolidation system extracting SEMANTIC FACTS from episodic memories.

A semantic fact is declarative knowledge that can be reused in future contexts.
It is NOT an opinion, prescription, motivation, or trivially obvious statement.

TOPIC: "{topic}"
NUMBER OF EPISODES: {num_episodes}

EPISODES:
{episodes_block}

== FACT CATEGORIES ==
TECHNICAL: How systems, tools, APIs, or services work (parameters, behaviors, constraints, configs)
PROCEDURAL: How to accomplish specific tasks (step sequences, prerequisites, commands)
RELATIONAL: Facts about people, their preferences, behavioral patterns, relationships
ARCHITECTURAL: System designs, data flows, integrations, schemas, infrastructure
CONTEXTUAL: Project states, decisions made, milestones reached, constraints discovered

== EXAMPLES OF GOOD FACTS ==
- {{"fact": "The TIAW-MainSync workflow uses a cron trigger set to 2-minute intervals to sync WSC inventory to Supabase", "category": "TECHNICAL", "confidence": 0.90, "specificity": "high"}}
- {{"fact": "Qdrant requires explicit collection creation with vector size and distance metric before any upsert operation", "category": "TECHNICAL", "confidence": 0.85, "specificity": "high"}}
- {{"fact": "Hare prefers to review the implementation plan before any code execution begins", "category": "RELATIONAL", "confidence": 0.90, "specificity": "high"}}
- {{"fact": "The codi_semantic collection uses text-embedding-3-small with 1536 dimensions and cosine distance", "category": "ARCHITECTURAL", "confidence": 0.95, "specificity": "high"}}
- {{"fact": "To deploy n8n workflow changes, the workflow must be deactivated first, then updated, then reactivated", "category": "PROCEDURAL", "confidence": 0.80, "specificity": "high"}}

== EXAMPLES OF BAD FACTS (DO NOT PRODUCE THESE) ==
- "It is important to test code thoroughly" -> prescriptive, not a fact
- "Workflows consist of multiple nodes" -> trivially obvious to anyone
- "Good improvements were observed" -> vague, no concrete detail
- "Communication is key for project success" -> generic platitude
- "The system was updated successfully" -> one-time event, not reusable knowledge

== RULES ==
1. Extract only facts with CONCRETE details (names, numbers, parameters, specific behaviors)
2. Each fact must be a single declarative sentence that would be useful if retrieved 30 days from now
3. Confidence (0.0-1.0): base on how many episodes support it and how consistent the evidence is
4. Specificity must be "high" -- if you cannot include a concrete detail, do not include the fact
5. Combine overlapping observations into one stronger fact rather than listing near-duplicates
6. Maximum 5 facts per cluster
7. If fewer than 2 facts meet the quality bar, return fewer -- do NOT pad with low-quality facts

Respond ONLY with a JSON array (no markdown, no explanation):
[{{"fact": "...", "category": "TECHNICAL|PROCEDURAL|RELATIONAL|ARCHITECTURAL|CONTEXTUAL", "confidence": 0.85, "specificity": "high"}}]"""


def _phase_extraction(clusters: list) -> list:
    """Phase 3: Extract semantic facts from each cluster using LLM.

    For each cluster, sends episode texts to GPT-4o-mini with a structured
    prompt that enforces category tagging, few-shot examples, and specificity
    filtering to produce high-quality semantic facts.

    Post-extraction filter: discards any fact where specificity != "high"
    or where the fact text is shorter than 20 chars.

    Returns:
        List of facts: [{fact_text, confidence, evidence_count, source_episode_ids, topic, category}]
    """
    if not clusters:
        return []

    all_facts = []
    skipped_low_quality = 0

    for cluster in clusters:
        topic = cluster["topic"]
        texts = cluster["texts"][:15]  # Cap per cluster to control token usage
        episode_ids = cluster["episode_ids"][:15]

        # Build LLM prompt
        episodes_block = "\n".join(f"- {t}" for t in texts)
        prompt = _build_extraction_prompt(topic, episodes_block, len(texts))

        try:
            response = _get_oai().chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=1500,
            )

            raw = response.choices[0].message.content.strip()
            # Parse JSON (handle markdown code blocks)
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

                # Quality gate: reject low-specificity, too-short, or low-confidence
                if specificity != "high":
                    skipped_low_quality += 1
                    continue
                if not fact_text or len(fact_text) < 20:
                    skipped_low_quality += 1
                    continue
                if confidence < 0.4:
                    skipped_low_quality += 1
                    continue

                # Validate category
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

            print(f"[consolidation] Extraction '{topic}': {cluster_accepted} accepted, "
                  f"{len(extracted) - cluster_accepted} filtered from {len(texts)} episodes")

        except Exception as e:
            print(f"[consolidation] Extraction error for '{topic}': {e}")
            continue

    print(f"[consolidation] Extraction total: {len(all_facts)} facts accepted, "
          f"{skipped_low_quality} filtered for low quality, from {len(clusters)} clusters")
    return all_facts


def _phase_integration(facts: list) -> dict:
    """Phase 4: Integrate facts into semantic store (codi_semantic).

    For each fact:
    1. Generate embedding
    2. Search codi_semantic for duplicates (>0.85 similarity)
    3. If duplicate found: update evidence_count and merge source_episode_ids
    4. If new: create new semantic point

    Returns:
        {created: N, updated: N, contradictions: N}
    """
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

            # Check for existing similar facts (dedup)
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
                # Update existing fact: increment evidence, merge sources
                old_payload = duplicate.payload or {}
                old_sources = old_payload.get("source_episode_ids", [])
                new_sources = list(set(old_sources + fact["source_episode_ids"]))
                old_evidence = int(old_payload.get("evidence_count", 1))

                qdrant.set_payload(
                    collection_name=SEMANTIC_COLLECTION,
                    payload={
                        "evidence_count": old_evidence + fact["evidence_count"],
                        "source_episode_ids": new_sources,
                        "last_observed": now,
                        "confidence": max(
                            float(old_payload.get("confidence", 0.5)),
                            fact["confidence"]
                        ),
                    },
                    points=[duplicate.id],
                )
                updated += 1
                print(f"[consolidation] Updated existing fact: {fact_text[:60]}...")
            else:
                # Create new semantic point
                point_id = str(uuid.uuid4())
                payload = {
                    "fact_text": fact_text,
                    "data": fact_text,  # Compat with existing search
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
                print(f"[consolidation] New semantic fact: {fact_text[:60]}...")

        except Exception as e:
            print(f"[consolidation] Integration error: {e}")
            continue

    print(f"[consolidation] Integration: {created} created, {updated} updated, {contradictions} contradictions")
    return {"created": created, "updated": updated, "contradictions": contradictions}


def _phase_pruning(consolidated_episode_ids: list) -> dict:
    """Phase 5: Mark episodes as consolidated and apply differential decay.

    Sets consolidation_status='consolidated' on processed episodes so they
    won't be re-processed in future runs.

    Returns:
        {marked_consolidated: N, decayed: N}
    """
    if not consolidated_episode_ids:
        return {"marked_consolidated": 0, "decayed": 0}

    marked = 0
    decayed = 0
    now = now_iso()

    # Process in batches of 50
    batch_size = 50
    for i in range(0, len(consolidated_episode_ids), batch_size):
        batch = consolidated_episode_ids[i:i + batch_size]
        try:
            qdrant.set_payload(
                collection_name=COLLECTION_NAME,
                payload={
                    "consolidation_status": "consolidated",
                    "consolidated_at": now,
                },
                points=batch,
            )
            marked += len(batch)
        except Exception as e:
            print(f"[consolidation] Pruning batch error: {e}")
            # Try one by one
            for eid in batch:
                try:
                    qdrant.set_payload(
                        collection_name=COLLECTION_NAME,
                        payload={
                            "consolidation_status": "consolidated",
                            "consolidated_at": now,
                        },
                        points=[eid],
                    )
                    marked += 1
                except Exception:
                    pass

    print(f"[consolidation] Pruning: {marked} episodes marked as consolidated")
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
        print(f"[consolidation] WARNING: Could not log run: {e}")


# ============================================================
# RECONSOLIDATION
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


def detect_contradiction(memory_text: str, context: str) -> dict:
    """Detect contradiction between a memory and current context.

    Kumaran & Maguire 2006/2007: CA1 hippocampal comparator uses
    multiple channels for match-mismatch detection, not just one signal.

    3 channels:
      Canal 1 - Keywords (0.5 raw): Explicit correction patterns
      Canal 2 - Topic confirmation (amplifier): cosine_sim * entity_overlap
                Amplifies C1+C3 when same topic confirmed (0.4 to 1.0x)
      Canal 3 - Negation detector (0.5 raw): Same entities + logical inversion

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
    # High cosine similarity + entity overlap = same topic confirmed.
    # This AMPLIFIES Canal 1 and Canal 3: contradictions only meaningful
    # if texts discuss the same subject. (NOT distance -- contradictions
    # like "Docker is good" vs "Docker is bad" have HIGH similarity.)
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

    # Weighted sum: C2 (topic confirmation) amplifies C1+C3
    # Without C1 or C3, same-topic alone is NOT contradiction
    # With topic confirmed (C2~1), C1+C3 fire at full weight
    # Without topic confirmed (C2~0), C1+C3 still fire at 40% (keywords alone still valid)
    raw_pe = canal1_score * 0.5 + canal3_score * 0.5
    pe = raw_pe * (0.4 + 0.6 * canal2_score)

    channels = {
        "keywords": canal1_score,
        "topic_confirmation": canal2_score,
        "negation": canal3_score,
        "shared_entities": list(shared_entities)[:10],
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
        detail = f"PE channels: {', '.join(parts)}"

    return {"prediction_error": pe, "detail": detail, "channels": channels}


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

    if contradiction["prediction_error"] > RECONSOLIDATION_PE_THRESHOLD:
        return {
            "should_reconsolidate": True,
            "prediction_error": contradiction["prediction_error"],
            "memory_strength": strength,
            "contradiction": contradiction["detail"],
            "reason": "prediction_error_exceeded_threshold"
        }

    return {"should_reconsolidate": False, "reason": "no_prediction_error",
            "memory_strength": strength, "prediction_error": contradiction["prediction_error"]}


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
        print(f"[consolidation] WARNING: Could not mark labile: {e}")
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


def correct_memory(memory_id: str, correction: str, force: bool = False) -> str:
    """Update a memory based on new evidence (reconsolidation).

    Nader 2000: The original trace is DESTROYED and re-synthesized.
    Sevenster 2013: Prediction error is a prerequisite for reconsolidation.

    Pipeline:
      1. Resolve full ID
      2. Retrieve old payload from Qdrant
      3. Labile gate: verify memory is labile OR has PE (force bypasses)
      4. Log old content to reconsolidation_log
      5. Adjust confidence (decrement by 0.1 for contradiction)
      6. Build new content, generate new embedding
      7. Upsert full PointStruct (vector + payload) -- re-embed, not post-it
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
        pts = qdrant.retrieve(collection_name=COLLECTION_NAME, ids=[full_id], with_payload=True)
        if not pts:
            return f"[reconsolidation] Memory {full_id[:8]} not found in Qdrant"
        old_payload = pts[0].payload or {}
    except Exception as e:
        return f"[reconsolidation] Qdrant retrieve error: {e}"

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
        print(f"[consolidation] WARNING: Could not log reconsolidation: {e}")

    # 5. Adjust confidence proportional to PE (Exton-McGuinness 2015)
    # Higher PE = larger confidence decrement (0.05 base + 0.15 * PE)
    confidence_delta = 0.05 + 0.15 * actual_pe
    new_confidence = max(0.0, old_confidence - confidence_delta)

    # 6. Re-embed: generate new vector for corrected content (Nader 2000)
    try:
        new_vector = _embed_text(new_content)
    except Exception as e:
        return f"[reconsolidation] Embedding error: {e}"

    # 7. Upsert full PointStruct -- destroy and re-synthesize the trace
    try:
        updated_payload = dict(old_payload)
        updated_payload.update({
            "data": new_content,
            "confidence": new_confidence,
            "reconsolidated_at": now_iso(),
            "reconsolidation_count": int(old_payload.get("reconsolidation_count", 0)) + 1,
        })
        qdrant.upsert(
            collection_name=COLLECTION_NAME,
            points=[PointStruct(
                id=full_id,
                vector=new_vector,
                payload=updated_payload,
            )],
        )
    except Exception as e:
        return f"[reconsolidation] Qdrant upsert error: {e}"

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
        print(f"[consolidation] WARNING: FTS update failed: {e}")

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
# SEMANTIC STORE OPERATIONS
# ============================================================

def search_semantic(query: str, limit: int = 5) -> list:
    """Search the semantic store (codi_semantic) via vector similarity.

    Returns:
        List of semantic facts with scores
    """
    try:
        info = qdrant.get_collection(SEMANTIC_COLLECTION)
        if info.points_count == 0:
            return []

        query_vector = _embed_text(query)
        results = qdrant.query_points(
            collection_name=SEMANTIC_COLLECTION,
            query=query_vector,
            limit=limit,
            with_payload=True,
        )
        facts = []
        for hit in results.points:
            payload = hit.payload or {}
            facts.append({
                "id": str(hit.id),
                "fact": payload.get("fact_text", payload.get("data", "")),
                "topic": payload.get("topic", ""),
                "confidence": payload.get("confidence", 0),
                "evidence_count": payload.get("evidence_count", 0),
                "score": hit.score,
            })
        return facts
    except Exception as e:
        print(f"[consolidation] Semantic search error: {e}")
        return []


def get_semantic_facts(topic: str = "", limit: int = 10) -> str:
    """Get all semantic facts, optionally filtered by topic.

    MCP tool to inspect consolidated knowledge.

    Args:
        topic: Optional topic to filter by (e.g. 'trading', 'fullempaques')
        limit: Max facts to return (default 10)
    """
    try:
        info = qdrant.get_collection(SEMANTIC_COLLECTION)
        count = info.points_count

        if count == 0:
            return "[semantic] Store is empty (0 facts). Run consolidation first."

        scroll_filter = None
        if topic:
            scroll_filter = Filter(must=[
                FieldCondition(key="topic", match=MatchValue(value=topic))
            ])

        pts, _ = qdrant.scroll(
            collection_name=SEMANTIC_COLLECTION,
            scroll_filter=scroll_filter,
            limit=limit,
            with_payload=True,
        )

        if not pts:
            return f"[semantic] {count} total facts, 0 matching topic='{topic}'"

        lines = [f"=== Semantic Facts ({len(pts)}/{count} total) ==="]
        for p in pts:
            pl = p.payload or {}
            fact = pl.get("fact_text", pl.get("data", "?"))
            topic_val = pl.get("topic", "?")
            conf = pl.get("confidence", 0)
            evidence = pl.get("evidence_count", 0)
            lines.append(f"- [{topic_val}] (conf={conf:.2f}, evidence={evidence}) {fact}")

        return "\n".join(lines)
    except Exception as e:
        return f"[semantic] Error: {e}"


def get_consolidation_stats() -> str:
    """Get statistics about consolidation runs.

    MCP tool for monitoring.
    """
    try:
        conn = _consolidation_conn()
        total_runs = conn.execute(
            "SELECT COUNT(*) FROM consolidation_log"
        ).fetchone()[0]
        total_facts_created = conn.execute(
            "SELECT COALESCE(SUM(facts_created), 0) FROM consolidation_log"
        ).fetchone()[0]
        total_recon = conn.execute(
            "SELECT COUNT(*) FROM reconsolidation_log"
        ).fetchone()[0]
        labile_count = conn.execute(
            "SELECT COUNT(*) FROM labile_memories"
        ).fetchone()[0]
        conn.close()

        semantic_count = 0
        try:
            info = qdrant.get_collection(SEMANTIC_COLLECTION)
            semantic_count = info.points_count
        except Exception:
            pass

        return (
            f"=== Consolidation Stats ===\n"
            f"Total runs: {total_runs}\n"
            f"Total semantic facts created: {total_facts_created}\n"
            f"Semantic store size: {semantic_count}\n"
            f"Total reconsolidation events: {total_recon}\n"
            f"Currently labile memories: {labile_count}"
        )
    except Exception as e:
        return f"Error getting stats: {e}"


def count_unconsolidated_episodic(lookback_hours: int = 24) -> int:
    """Count unconsolidated episodic memories in the last N hours."""
    cutoff = datetime.now() - timedelta(hours=lookback_hours)
    scroll_filter = Filter(must_not=[
        FieldCondition(key="consolidation_status", match=MatchValue(value="consolidated"))
    ])

    count = 0
    offset = None
    while True:
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
            created_str = (p.payload or {}).get("created_at", "")
            try:
                created = datetime.fromisoformat(str(created_str).replace("Z", "+00:00"))
                if created.tzinfo:
                    created = created.replace(tzinfo=None)
                if created >= cutoff:
                    count += 1
            except Exception:
                pass
        if not next_offset:
            break
        offset = next_offset

    return count


# ============================================================
# MCP TOOL REGISTRATION
# ============================================================

def register_consolidation_tools(mcp):
    """Register consolidation MCP tools."""
    mcp.tool()(run_consolidation)
    mcp.tool()(correct_memory)
    mcp.tool()(get_semantic_facts)
    mcp.tool()(get_consolidation_stats)
