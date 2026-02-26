#!/usr/bin/env python3
"""
Codi Memory - Pre-Turn Context Injection Hook
==============================================
Claude Code UserPromptSubmit hook.

Fires BEFORE Codi processes each user message.
Reads user prompt, searches local memory (FTS5 + working memory),
and injects relevant context so Codi doesn't need to call recall().

Design: Local-only (SQLite), no API calls, target < 500ms.
"""

import sys
import json
import sqlite3
import os
import re
import math

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FTS_DB_PATH = os.path.join(BASE_DIR, "memories_fts.db")
TRIGGERS_FILE = os.path.join(BASE_DIR, "triggers.json")
PROSPECTIVE_DB_PATH = os.path.join(BASE_DIR, "prospective.db")

# Config
MAX_FTS_RESULTS = 5
MAX_WM_RESULTS = 4
MIN_PROMPT_LENGTH = 5
MAX_CONTEXT_CHARS = 2000  # Keep injected context compact

# Prediction config (Clark 2013, Friston 2010)
PREDICTION_HISTORY_MAX = 200
PREDICTION_RESULTS_MAX = 500
SURPRISE_ADAPTIVE_WINDOW = 50  # Last N results for threshold
SURPRISE_THRESHOLD_K = 0.75    # mean + K*std
SURPRISE_MIN_THRESHOLD = 0.25  # Minimum surprise to count as prediction error

# Patterns that indicate sensitive content - NEVER inject these
SENSITIVE_PATTERNS = [
    r'api[_\s-]?key', r'api_key', r'apikey',
    r'password', r'passwd', r'secret',
    r'token', r'bearer', r'authorization',
    r'credential', r'credencial',
    r'sk-proj-', r'sk-', r'sm_',
    r'eyj[a-za-z0-9]',  # JWT tokens
    r'sbp_', r'github_pat_',
    r'\b[0-9a-f]{32,}\b',  # Long hex strings (API keys)
    r'supabase_key', r'openai',
    r'evolution.*api', r'818cabbf',
]

SENSITIVE_RE = re.compile('|'.join(SENSITIVE_PATTERNS), re.IGNORECASE)


def contains_sensitive_data(text):
    """Check if text contains credentials or sensitive data."""
    return bool(SENSITIVE_RE.search(text))


def get_db_connection(read_only=True):
    """Get SQLite connection with WAL mode for concurrent reads."""
    conn = sqlite3.connect(FTS_DB_PATH, timeout=3)
    conn.execute("PRAGMA journal_mode=WAL")
    if read_only:
        conn.execute("PRAGMA query_only=ON")
    return conn


def clean_fts_query(prompt):
    """Convert user prompt into FTS5-safe query terms."""
    # Remove special chars that break FTS5
    cleaned = re.sub(r'[^\w\sáéíóúñ]', ' ', prompt.lower())
    # Split into words, filter short ones and stopwords
    stopwords = {
        'que', 'de', 'en', 'el', 'la', 'los', 'las', 'un', 'una', 'es',
        'por', 'con', 'para', 'del', 'al', 'se', 'no', 'si', 'lo', 'como',
        'pero', 'mas', 'su', 'le', 'ya', 'me', 'mi', 'te', 'tu', 'nos',
        'the', 'is', 'at', 'on', 'in', 'to', 'and', 'or', 'of', 'a', 'an',
        'it', 'do', 'has', 'had', 'was', 'are', 'be', 'this', 'that',
        'dame', 'dime', 'quiero', 'necesito', 'puedes', 'vamos', 'haz',
        'mira', 'bueno', 'listo', 'dale', 'hazlo', 'parcero', 'hermano',
    }
    words = [w for w in cleaned.split() if len(w) > 2 and w not in stopwords]
    if not words:
        return None
    # Use OR for broader matching
    return ' OR '.join(words)


def search_fts(conn, query, limit=MAX_FTS_RESULTS):
    """Search FTS5 index for relevant memories."""
    fts_query = clean_fts_query(query)
    if not fts_query:
        return []
    try:
        cursor = conn.execute("""
            SELECT content, memory_id, category, importance, rank
            FROM memories_fts
            JOIN memories_text ON memories_fts.memory_id = memories_text.memory_id
            WHERE memories_fts MATCH ?
            ORDER BY rank
            LIMIT ?
        """, (fts_query, limit))
        return cursor.fetchall()
    except Exception:
        # FTS5 match can fail on malformed queries
        return []


def search_fts_simple(conn, query, limit=MAX_FTS_RESULTS):
    """Simpler FTS5 search using content table directly."""
    fts_query = clean_fts_query(query)
    if not fts_query:
        return []
    try:
        cursor = conn.execute("""
            SELECT mt.content, mt.memory_id, mt.category, mt.importance
            FROM memories_text mt
            WHERE mt.rowid IN (
                SELECT rowid FROM memories_fts WHERE memories_fts MATCH ?
            )
            LIMIT ?
        """, (fts_query, limit))
        return [(row[0], row[1], row[2], row[3], 0) for row in cursor.fetchall()]
    except Exception:
        return []


def get_working_memory(conn, limit=MAX_WM_RESULTS):
    """Get top active working memory items by relevance."""
    try:
        cursor = conn.execute("""
            SELECT content, topic, relevance
            FROM working_memory
            WHERE active = 1
            ORDER BY relevance DESC, occurred_at DESC
            LIMIT ?
        """, (limit,))
        return cursor.fetchall()
    except Exception:
        return []


def detect_triggers(prompt):
    """Check if prompt matches any triggers for context enrichment."""
    if not os.path.exists(TRIGGERS_FILE):
        return []
    try:
        with open(TRIGGERS_FILE, 'r') as f:
            data = json.load(f)
        triggers = data.get('triggers', {})
        activated = []
        prompt_lower = prompt.lower()
        for name, config in triggers.items():
            patterns = config.get('patterns', [])
            for pattern in patterns:
                if pattern.lower() in prompt_lower:
                    activated.append({
                        'name': name,
                        'evoca': config.get('evoca', []),
                        'contexto': config.get('contexto_a_buscar', '')
                    })
                    break
        return activated
    except Exception:
        return []


def check_prospective_memory(prompt):
    """Check pending intentions against current prompt.

    Delegates to the canonical prospective.check_intentions() engine
    which handles: Tier 1 focal, Tier 2 nonfocal, time-based, budget
    enforcement, partial match boosts, stale expiry, and GWT cap of 3.
    """
    try:
        sys.path.insert(0, BASE_DIR)
        from modules.prospective import check_intentions
        return check_intentions(prompt)
    except Exception:
        return []


def _run_lightweight_competition(fts_results, wm_results, intentions, triggers, surprise_info, slots=8):
    """Lightweight GWT competition across domains (WIRING-6.3).

    Normalizes scores per domain, merges into flat list, selects top N.
    No Qdrant calls -- SQLite-only for <500ms budget.

    Returns:
        Dict with domain keys mapping to filtered result lists.
    """
    all_candidates = []  # (score, domain, index, data)

    # FTS memories: rank-based scoring
    if fts_results:
        for i, row in enumerate(fts_results):
            score = 1.0 / (1.0 + i)  # rank 0 -> 1.0, rank 1 -> 0.5, ...
            importance = row[3] if len(row) > 3 else "medium"
            if importance == "critical":
                score = min(1.0, score + 0.2)
            all_candidates.append((score, "fts", i, row))

    # WM items: relevance is already 0-1
    if wm_results:
        for i, row in enumerate(wm_results):
            score = float(row[2]) if len(row) > 2 else 0.5
            all_candidates.append((score, "wm", i, row))

    # PM intentions: priority-weighted
    if intentions:
        priority_w = {"critical": 1.0, "high": 0.85, "medium": 0.65, "low": 0.4}
        for i, intent in enumerate(intentions):
            score = priority_w.get(intent.get("priority", "medium"), 0.5)
            act = intent.get("activation", 0.5)
            score = min(1.0, score * act)
            all_candidates.append((score, "pm", i, intent))

    # Triggers: fixed score (by definition relevant to prompt)
    if triggers:
        for i, t in enumerate(triggers):
            all_candidates.append((0.6, "trigger", i, t))

    # Surprise: if significant, always included
    if surprise_info:
        all_candidates.append((0.8, "surprise", 0, surprise_info))

    # Sort by score descending, take top N
    all_candidates.sort(key=lambda x: x[0], reverse=True)
    winners = all_candidates[:slots]

    # Group winners back by domain
    result = {"fts": [], "wm": [], "pm": [], "trigger": [], "surprise": None}
    for _score, domain, idx, data in winners:
        if domain == "fts":
            result["fts"].append(data)
        elif domain == "wm":
            result["wm"].append(data)
        elif domain == "pm":
            result["pm"].append(data)
        elif domain == "trigger":
            result["trigger"].append(data)
        elif domain == "surprise":
            result["surprise"] = data

    return result


def format_context(fts_results, wm_results, triggers, intentions=None, surprise_info=None):
    """Format search results into compact context string."""
    parts = []

    # Prediction surprise first (if significant)
    if surprise_info:
        parts.append("## Prediction Error Detectado")
        parts.append(f"- Esperaba tema: {surprise_info['predicted_topic']}, "
                     f"actual: {surprise_info['actual_topic']}")
        parts.append(f"- Surprise: {surprise_info['surprise']:.2f} "
                     f"(threshold: {surprise_info['threshold']:.2f})")
        if surprise_info.get('topic_changed'):
            parts.append("- [TOPIC SHIFT] Cambio de contexto detectado")

    # Prospective memory first (actionable, future-oriented)
    if intentions:
        parts.append("## Intenciones Pendientes (Prospective Memory)")
        for intent in intentions:
            marker = {"critical": "[!!!]", "high": "[!!]", "medium": "[!]", "low": "[.]"}.get(intent["priority"], "[?]")
            parts.append(f"- {marker} {intent['action']} ({intent['action_type']})")

    if triggers:
        trigger_names = [t['name'] for t in triggers]
        parts.append(f"[Triggers detectados: {', '.join(trigger_names)}]")

    if fts_results:
        parts.append("## Memorias Relevantes")
        seen = set()
        for content, mem_id, category, importance, _rank in fts_results:
            # SECURITY: Skip memories containing credentials
            if contains_sensitive_data(content):
                continue
            # Truncate and dedup
            short = content[:250].strip()
            if short in seen:
                continue
            seen.add(short)
            prefix = ""
            if importance == 'critical':
                prefix = "[CRITICO] "
            parts.append(f"- {prefix}{short}")

    if wm_results:
        parts.append("\n## Working Memory (contexto reciente)")
        for content, topic, relevance in wm_results:
            # SECURITY: Skip sensitive working memory items
            if contains_sensitive_data(content):
                continue
            short = content[:200].strip()
            parts.append(f"- [{topic}|{relevance:.1f}] {short}")

    context = '\n'.join(parts)

    # Enforce max size
    if len(context) > MAX_CONTEXT_CHARS:
        context = context[:MAX_CONTEXT_CHARS] + "\n[...truncado]"

    return context


# ============================================================
# PREDICTION-COMPARISON LOOP (Clark 2013, Friston 2010)
# WIRING-4: The brain is fundamentally a prediction machine
# ============================================================

# Topic keywords for classification (shared with consciousness.py)
TOPIC_KEYWORDS = {
    'n8n': ['n8n', 'workflow', 'automatiz', 'nodo', 'webhook'],
    'trading': ['trading', 'kraken', 'cripto', 'bitcoin', 'mercado', 'orden'],
    'fullempaques': ['fullempaques', 'produccion', 'fabrica', 'empaque', 'cotiz'],
    'memoria': ['memoria', 'recuerdo', 'recordar', 'qdrant', 'consolidar'],
    'codigo': ['codigo', 'python', 'javascript', 'server', 'bug', 'error', 'fix'],
    'consciencia': ['consciencia', 'consciente', 'prediccion', 'wiring', 'audit'],
    'tiaw': ['tiaw', 'wsc', 'stocklist', 'offer', 'aliados'],
    'pilas': ['pilas', 'whatsapp', 'pendiente', 'bot'],
}

# Action type keywords
ACTION_KEYWORDS = {
    'debugging': ['bug', 'error', 'fix', 'broke', 'wrong', 'fail', 'crash'],
    'building': ['create', 'implement', 'build', 'add', 'new', 'hacer', 'crear'],
    'exploring': ['what', 'how', 'why', 'explain', 'que', 'como', 'por'],
    'reviewing': ['check', 'verify', 'audit', 'review', 'status', 'revisar'],
    'planning': ['plan', 'next', 'should', 'strategy', 'roadmap', 'vamos'],
}


def _init_prediction_tables(conn):
    """Create prediction tables if they don't exist."""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS prediction_state (
            id INTEGER PRIMARY KEY,
            predicted_topic TEXT,
            predicted_keywords TEXT,
            predicted_action TEXT,
            topic_distribution TEXT,
            confidence REAL DEFAULT 0.5,
            created_at TEXT NOT NULL
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS prediction_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            predicted_topic TEXT,
            actual_topic TEXT,
            predicted_keywords TEXT,
            actual_keywords TEXT,
            surprise_score REAL,
            precision_weight REAL,
            weighted_surprise REAL,
            hit INTEGER DEFAULT 0,
            created_at TEXT NOT NULL
        )
    """)
    conn.commit()


def _classify_topic(text):
    """Classify text into a topic using keyword matching. O(n) scan, ~1ms."""
    text_lower = text.lower()
    scores = {}
    for topic, keywords in TOPIC_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw in text_lower)
        if score > 0:
            scores[topic] = score
    if not scores:
        return 'general'
    return max(scores, key=scores.get)


def _classify_action(text):
    """Classify the action mode of the prompt."""
    text_lower = text.lower()
    scores = {}
    for action, keywords in ACTION_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw in text_lower)
        if score > 0:
            scores[action] = score
    if not scores:
        return 'general'
    return max(scores, key=scores.get)


def _extract_keywords(text):
    """Extract significant keywords from text for Jaccard comparison."""
    stopwords = {
        'que', 'de', 'en', 'el', 'la', 'los', 'las', 'un', 'una', 'es',
        'por', 'con', 'para', 'del', 'al', 'se', 'no', 'si', 'lo', 'como',
        'pero', 'mas', 'su', 'le', 'ya', 'me', 'mi', 'te', 'tu', 'nos',
        'the', 'is', 'at', 'on', 'in', 'to', 'and', 'or', 'of', 'a', 'an',
        'it', 'do', 'has', 'had', 'was', 'are', 'be', 'this', 'that',
        'dame', 'dime', 'quiero', 'necesito', 'puedes', 'vamos', 'haz',
        'mira', 'bueno', 'listo', 'dale', 'hazlo', 'parcero', 'hermano',
        'bro', 'hey', 'oye', 'mira', 'ahi',
    }
    words = re.findall(r'\w{3,}', text.lower())
    return set(w for w in words if w not in stopwords)


def _compute_surprise(predicted_topic, predicted_keywords_str, predicted_action,
                       actual_topic, actual_keywords, actual_action):
    """Compute 3-level surprise score (0-1).

    Level 1: Topic surprise (binary match, weighted 0.4)
    Level 2: Keyword surprise (Jaccard distance, weighted 0.4)
    Level 3: Action surprise (binary match, weighted 0.2)
    """
    # Level 1: Topic match
    topic_surprise = 0.0 if predicted_topic == actual_topic else 1.0

    # Level 2: Keyword Jaccard distance
    try:
        predicted_kw = set(json.loads(predicted_keywords_str)) if predicted_keywords_str else set()
    except (json.JSONDecodeError, TypeError):
        predicted_kw = set()

    if predicted_kw or actual_keywords:
        intersection = predicted_kw & actual_keywords
        union = predicted_kw | actual_keywords
        jaccard_sim = len(intersection) / len(union) if union else 0
        keyword_surprise = 1.0 - jaccard_sim
    else:
        keyword_surprise = 0.5  # No data = moderate surprise

    # Level 3: Action match (binary: weights handle relative importance)
    action_surprise = 0.0 if predicted_action == actual_action else 1.0

    # Weighted combination (hierarchical: topic > keyword > action)
    # Higher-level errors (topic) carry more weight (Rao & Ballard 1999)
    surprise = (topic_surprise * 0.50) + (keyword_surprise * 0.30) + (action_surprise * 0.20)
    return min(1.0, max(0.0, surprise))


def _compute_precision(conn, actual_topic):
    """Compute precision weight for current prediction.

    Higher precision = this prediction error matters more.
    Based on: topic familiarity, historical accuracy, stability.
    """
    try:
        # How many times we've predicted this topic correctly (last 20 results only)
        # Subquery needed: LIMIT on aggregate is a no-op (Behrens 2007: recency-weighted)
        row = conn.execute("""
            SELECT COUNT(*) as total,
                   SUM(CASE WHEN hit = 1 THEN 1 ELSE 0 END) as hits
            FROM (
                SELECT hit FROM prediction_results
                WHERE predicted_topic = ?
                ORDER BY id DESC LIMIT 20
            )
        """, (actual_topic,)).fetchone()

        total = row[0] if row else 0
        hits = row[1] if row else 0

        if total < 3:
            return 0.3  # Low precision for unfamiliar topics

        accuracy = hits / total
        # Higher accuracy on this topic = higher precision (we're confident)
        precision = 0.3 + (accuracy * 0.5)
        return min(0.85, max(0.15, precision))
    except Exception:
        return 0.3


def _get_adaptive_threshold(conn):
    """Compute adaptive surprise threshold from recent history.

    Threshold = mean(recent_surprise) + K * std(recent_surprise)
    This prevents "surprised by everything" (EM-LLM inspired).
    """
    try:
        rows = conn.execute("""
            SELECT weighted_surprise FROM prediction_results
            ORDER BY id DESC LIMIT ?
        """, (SURPRISE_ADAPTIVE_WINDOW,)).fetchall()

        if len(rows) < 5:
            return SURPRISE_MIN_THRESHOLD  # Not enough data

        values = [r[0] for r in rows]
        mean = sum(values) / len(values)
        variance = sum((v - mean) ** 2 for v in values) / len(values)
        std = math.sqrt(variance)

        threshold = mean + SURPRISE_THRESHOLD_K * std
        return max(SURPRISE_MIN_THRESHOLD, min(0.8, threshold))
    except Exception:
        return SURPRISE_MIN_THRESHOLD


def _compare_prediction(conn, prompt):
    """PHASE 1 (pre-turn): Compare current prompt against last prediction.

    Returns surprise_info dict if significant, None otherwise.
    """
    try:
        # Get last prediction
        row = conn.execute("""
            SELECT predicted_topic, predicted_keywords, predicted_action,
                   topic_distribution, confidence, created_at
            FROM prediction_state
            WHERE id = 1
        """).fetchone()

        if not row:
            return None  # No prior prediction (first turn)

        predicted_topic = row[0] or 'general'
        predicted_keywords = row[1] or '[]'
        predicted_action = row[2] or 'general'
        confidence = row[4] or 0.5

        # Classify actual prompt
        actual_topic = _classify_topic(prompt)
        actual_keywords = _extract_keywords(prompt)
        actual_action = _classify_action(prompt)

        # Compute raw surprise
        surprise = _compute_surprise(
            predicted_topic, predicted_keywords, predicted_action,
            actual_topic, actual_keywords, actual_action
        )

        # Precision weighting (Friston: two distinct precision signals)
        # 1. Prediction precision: how confident were we? (historical accuracy)
        prediction_precision = _compute_precision(conn, actual_topic)
        # 2. Input precision: how clear/unambiguous is this prompt? (sensory reliability)
        input_precision = min(1.0, len(actual_keywords) / 5.0) if actual_keywords else 0.3
        # Combined precision modulates surprise
        precision = prediction_precision * input_precision
        weighted_surprise = surprise * precision

        # Is it a hit?
        hit = 1 if predicted_topic == actual_topic else 0

        # Record result
        from datetime import datetime
        now = datetime.now().isoformat()
        conn.execute("""
            INSERT INTO prediction_results
            (predicted_topic, actual_topic, predicted_keywords, actual_keywords,
             surprise_score, precision_weight, weighted_surprise, hit, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            predicted_topic, actual_topic,
            predicted_keywords, json.dumps(list(actual_keywords)[:20]),
            surprise, precision, weighted_surprise, hit, now
        ))

        # Cleanup old results
        conn.execute("""
            DELETE FROM prediction_results
            WHERE id NOT IN (
                SELECT id FROM prediction_results ORDER BY id DESC LIMIT ?
            )
        """, (PREDICTION_RESULTS_MAX,))
        conn.commit()

        # Check against adaptive threshold
        threshold = _get_adaptive_threshold(conn)

        if weighted_surprise > threshold:
            # Find memories that influenced the wrong prediction (Lee 2009)
            affected_ids = []
            try:
                predicted_kw_query = clean_fts_query(predicted_topic)
                if predicted_kw_query:
                    cursor = conn.execute("""
                        SELECT mt.memory_id
                        FROM memories_text mt
                        WHERE mt.rowid IN (
                            SELECT rowid FROM memories_fts WHERE memories_fts MATCH ?
                        )
                        LIMIT 3
                    """, (predicted_kw_query,))
                    affected_ids = [row[0] for row in cursor.fetchall()]
            except Exception:
                pass

            return {
                'surprise': weighted_surprise,
                'raw_surprise': surprise,
                'precision': precision,
                'threshold': threshold,
                'predicted_topic': predicted_topic,
                'actual_topic': actual_topic,
                'topic_changed': predicted_topic != actual_topic,
                'actual_keywords': list(actual_keywords)[:10],
                'affected_memory_ids': affected_ids,
            }

        return None  # Below threshold, not surprising
    except Exception:
        return None


def _generate_prediction(conn, prompt, wm_topics=None):
    """PHASE 2 (post-comparison): Generate prediction for next turn.

    Uses current context to predict what the next prompt will be about.
    No API calls, purely keyword-based (<10ms).
    """
    try:
        current_topic = _classify_topic(prompt)
        current_keywords = _extract_keywords(prompt)
        current_action = _classify_action(prompt)

        # Build topic distribution from recent 20 results
        # Subquery needed: LIMIT on GROUP BY limits groups, not input rows
        rows = conn.execute("""
            SELECT actual_topic, COUNT(*) as cnt
            FROM (
                SELECT actual_topic FROM prediction_results
                ORDER BY id DESC LIMIT 20
            )
            GROUP BY actual_topic
        """).fetchall()

        topic_counts = {}
        for topic, cnt in rows:
            topic_counts[topic] = topic_counts.get(topic, 0) + cnt

        # Current topic gets a strong recency boost (people tend to stay on topic)
        topic_counts[current_topic] = topic_counts.get(current_topic, 0) + 5

        # WM topics get a small boost
        if wm_topics:
            for t in wm_topics[:3]:
                topic_counts[t] = topic_counts.get(t, 0) + 1

        # Most likely next topic
        predicted_topic = max(topic_counts, key=topic_counts.get) if topic_counts else current_topic

        # Predicted keywords: current keywords (topic persistence) + WM keywords
        predicted_kw = list(current_keywords)[:15]

        # Predicted action: same as current (people tend to stay in the same mode)
        predicted_action = current_action

        # Confidence: higher if topic is concentrated (low entropy)
        total = sum(topic_counts.values()) or 1
        top_fraction = topic_counts.get(predicted_topic, 0) / total
        confidence = min(0.9, max(0.2, top_fraction))

        from datetime import datetime
        now = datetime.now().isoformat()

        # Upsert prediction (only 1 row, id=1)
        conn.execute("DELETE FROM prediction_state")
        conn.execute("""
            INSERT INTO prediction_state
            (id, predicted_topic, predicted_keywords, predicted_action,
             topic_distribution, confidence, created_at)
            VALUES (1, ?, ?, ?, ?, ?, ?)
        """, (
            predicted_topic,
            json.dumps(predicted_kw),
            predicted_action,
            json.dumps(topic_counts),
            confidence,
            now,
        ))
        conn.commit()
    except Exception:
        pass


def _emit_prediction_error(surprise_info):
    """Emit PREDICTION_ERROR event to the event bus.

    This triggers the wiring handler that boosts encoding and
    captures attention (Schultz 1997 dopaminergic PE signals).
    Also marks affected memories as labile when PE > 0.4 (Nader 2000).
    """
    # 1. Persist to SQLite directly (no dependency on dotenv/db_pool/config)
    try:
        from datetime import datetime
        conn = sqlite3.connect(FTS_DB_PATH, timeout=3)
        conn.execute("""
            INSERT INTO event_counts (event, count, last_seen)
            VALUES ('prediction_error', 1, ?)
            ON CONFLICT(event) DO UPDATE SET
                count = count + 1,
                last_seen = excluded.last_seen
        """, (datetime.now().isoformat(),))
        conn.commit()
        conn.close()
    except Exception:
        pass
    # 2. Mark affected memories as labile when PE > 0.4 (Loop 2: Nader 2000)
    try:
        affected_ids = surprise_info.get('affected_memory_ids', [])
        pe_magnitude = surprise_info.get('surprise', 0)
        if affected_ids and pe_magnitude > 0.4:
            from datetime import datetime, timedelta
            now = datetime.now()
            expires = (now + timedelta(hours=1)).isoformat()
            now_iso = now.isoformat()
            trigger_ctx = (f"Topic PE: predicted={surprise_info.get('predicted_topic', '?')}, "
                          f"actual={surprise_info.get('actual_topic', '?')}")
            conn = sqlite3.connect(FTS_DB_PATH, timeout=3)
            for mem_id in affected_ids[:2]:  # Max 2 memories per PE event
                conn.execute("""
                    INSERT OR REPLACE INTO labile_memories
                    (memory_id, marked_at, window_expires, prediction_error, trigger_context)
                    VALUES (?, ?, ?, ?, ?)
                """, (mem_id, now_iso, expires, pe_magnitude, trigger_ctx))
            conn.commit()
            conn.close()
    except Exception:
        pass
    # 3. Emit to event bus for in-process handlers (best-effort, may fail in system python3)
    try:
        sys.path.insert(0, BASE_DIR)
        from modules.events import event_bus, Events
        event_bus.emit(Events.PREDICTION_ERROR, {
            'error_magnitude': surprise_info['surprise'],
            'topic': surprise_info['actual_topic'],
            'predicted_topic': surprise_info['predicted_topic'],
            'precision': surprise_info['precision'],
            'threshold': surprise_info['threshold'],
            'actual_keywords': surprise_info.get('actual_keywords', []),
            'affected_memory_ids': surprise_info.get('affected_memory_ids', []),
        })
    except Exception:
        pass


def _load_attention_state():
    """Load persisted attention transitions and last focus from SQLite.

    Enables cross-process bigram prediction (Graziano 2013 AST).
    Without this, _attention_schema resets each hook invocation and
    the predictor has no history to achieve pred_prob >= 0.5.
    """
    try:
        conn = sqlite3.connect(FTS_DB_PATH, timeout=3)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS attention_transitions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                from_topic TEXT NOT NULL,
                to_topic TEXT NOT NULL,
                driver TEXT DEFAULT 'unknown',
                created_at TEXT NOT NULL
            )
        """)
        rows = conn.execute(
            "SELECT from_topic, to_topic, driver, created_at "
            "FROM attention_transitions ORDER BY id DESC LIMIT 30"
        ).fetchall()
        conn.close()

        transitions = [
            {"from": r[0], "to": r[1], "driver": r[2], "at": r[3]}
            for r in reversed(rows)
        ]
        last_focus = rows[0][1] if rows else None  # Most recent to_topic
        return transitions, last_focus
    except Exception:
        return [], None


def _save_attention_transition(from_topic, to_topic, driver, created_at):
    """Persist a new topic transition to SQLite for cross-process continuity."""
    try:
        conn = sqlite3.connect(FTS_DB_PATH, timeout=3)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS attention_transitions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                from_topic TEXT NOT NULL,
                to_topic TEXT NOT NULL,
                driver TEXT DEFAULT 'unknown',
                created_at TEXT NOT NULL
            )
        """)
        conn.execute(
            "INSERT INTO attention_transitions (from_topic, to_topic, driver, created_at) "
            "VALUES (?, ?, ?, ?)",
            (from_topic, to_topic, driver, created_at)
        )
        # Prune old entries (keep last 100)
        conn.execute(
            "DELETE FROM attention_transitions WHERE id NOT IN "
            "(SELECT id FROM attention_transitions ORDER BY id DESC LIMIT 100)"
        )
        conn.commit()
        conn.close()
    except Exception:
        pass


def _process_temporal_dynamics():
    """Apply time-based processes: decay, maintenance, emotional drift.

    A brain does not freeze between inputs. This computes elapsed time
    since last interaction and triggers proportional temporal processes.
    """
    try:
        sys.path.insert(0, BASE_DIR)
        from modules.wiring import (
            process_elapsed_time, get_last_interaction_time,
            set_last_interaction_time, _update_attention_schema,
        )
        from datetime import datetime

        now = datetime.now()
        last = get_last_interaction_time()

        if last:
            try:
                last_dt = datetime.fromisoformat(last)
                elapsed = (now - last_dt).total_seconds()
                if elapsed > 60:  # Only process if > 1 min gap
                    process_elapsed_time(elapsed)
            except Exception:
                pass

        set_last_interaction_time(now.isoformat())
    except Exception:
        pass


def _update_attention_from_prompt(prompt: str):
    """Extract topic from prompt and update attention schema.

    Lightweight topic extraction (no API calls) based on keyword density.
    Loads persisted transitions so bigram predictor works cross-process.
    """
    try:
        sys.path.insert(0, BASE_DIR)
        import modules.wiring as _wiring

        # Simple topic extraction: most frequent non-stopword
        stopwords = {
            'el', 'la', 'los', 'las', 'de', 'del', 'en', 'que', 'es', 'un',
            'una', 'por', 'con', 'para', 'se', 'no', 'si', 'me', 'al', 'lo',
            'the', 'is', 'in', 'to', 'and', 'of', 'for', 'on', 'it', 'a',
            'this', 'that', 'with', 'from', 'as', 'but', 'not', 'are', 'was',
            'como', 'este', 'esta', 'ya', 'hay', 'su', 'les', 'nos', 'mas',
            'pero', 'todo', 'eso', 'ser', 'mi', 'te', 'le', 'muy', 'bien',
        }
        words = re.findall(r'\w{3,}', prompt.lower())
        filtered = [w for w in words if w not in stopwords]

        if not filtered:
            return

        from collections import Counter
        topic = Counter(filtered).most_common(1)[0][0]

        # --- AST-1 FIX: Load persisted state into _attention_schema ---
        transitions, last_focus = _load_attention_state()
        if transitions:
            _wiring._attention_schema["topic_transitions"] = transitions
        if last_focus:
            _wiring._attention_schema["current_focus"] = last_focus

        # Count transitions before update (to detect new ones)
        n_before = len(_wiring._attention_schema["topic_transitions"])

        # Call update (bigram predictor now has history!)
        _wiring._update_attention_schema(
            focus=topic,
            driver="user_prompt",
            strength=0.6,
        )

        # --- Persist new transition if one was added ---
        n_after = len(_wiring._attention_schema["topic_transitions"])
        if n_after > n_before:
            latest = _wiring._attention_schema["topic_transitions"][-1]
            _save_attention_transition(
                latest["from"], latest["to"],
                latest.get("driver", "unknown"), latest.get("at", "")
            )

        # --- AST-1: Persist attention PE to event_counts (direct SQLite) ---
        if _wiring._attention_schema.get("last_predicted_focus") is not None:
            try:
                from datetime import datetime
                conn = sqlite3.connect(FTS_DB_PATH, timeout=3)
                conn.execute("""
                    INSERT INTO event_counts (event, count, last_seen)
                    VALUES ('attention_prediction_error', 1, ?)
                    ON CONFLICT(event) DO UPDATE SET
                        count = count + 1,
                        last_seen = excluded.last_seen
                """, (datetime.now().isoformat(),))
                conn.commit()
                conn.close()
            except Exception:
                pass

    except Exception:
        pass


def main():
    try:
        # Read stdin from Claude Code
        input_data = json.loads(sys.stdin.read())
        prompt = input_data.get('prompt', '')

        # Skip very short prompts or greetings
        if not prompt or len(prompt.strip()) < MIN_PROMPT_LENGTH:
            return

        # Check if DB exists
        if not os.path.exists(FTS_DB_PATH):
            return

        # --- Temporal dynamics (time-based decay and maintenance) ---
        _process_temporal_dynamics()

        # --- Update attention schema from prompt ---
        _update_attention_from_prompt(prompt)

        # --- PREDICTION LOOP: Phase 1 - Compare (before processing) ---
        surprise_info = None
        try:
            pred_conn = get_db_connection(read_only=False)
            _init_prediction_tables(pred_conn)
            surprise_info = _compare_prediction(pred_conn, prompt)
            if surprise_info:
                _emit_prediction_error(surprise_info)
        except Exception:
            pred_conn = None

        conn = get_db_connection()
        try:
            # Parallel-ish queries (sequential but fast on SQLite)
            fts_results = search_fts(conn, prompt)
            if not fts_results:
                fts_results = search_fts_simple(conn, prompt)

            wm_results = get_working_memory(conn)
            triggers = detect_triggers(prompt)
            intentions = check_prospective_memory(prompt)

            # --- PREDICTION LOOP: Phase 2 - Generate (after context) ---
            try:
                if pred_conn:
                    wm_topics = [row[1] for row in wm_results] if wm_results else []
                    _generate_prediction(pred_conn, prompt, wm_topics)
                    pred_conn.close()
            except Exception:
                pass

            # Nothing found? Don't inject empty context
            if not fts_results and not wm_results and not triggers and not intentions and not surprise_info:
                return

            # --- WIRING-6.3: Lightweight GWT competition across domains ---
            comp = _run_lightweight_competition(
                fts_results, wm_results, intentions, triggers, surprise_info
            )
            context = format_context(
                comp["fts"], comp["wm"], comp["trigger"], comp["pm"], comp["surprise"]
            )
            if not context.strip():
                return

            # Output for Claude Code hook system
            output = {
                "hookSpecificOutput": {
                    "hookEventName": "UserPromptSubmit",
                    "additionalContext": f"<codi-memory-context>\n{context}\n</codi-memory-context>"
                }
            }
            print(json.dumps(output))

        finally:
            conn.close()

    except Exception:
        # Never block Claude Code if hook fails
        pass


if __name__ == '__main__':
    main()
