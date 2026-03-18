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
import time
from datetime import datetime, timedelta

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
MARKOV_WINDOW = 80             # S2-02: Recency window for transition model (Behrens 2007)
RECENCY_DECAY = 0.03           # #70: Exponential recency decay (Behrens 2007). Half-life ≈ 23 transitions.
LAPLACE_ALPHA = 0.5            # S2-02: Laplace smoothing constant (Jeffreys prior)

# L1: Session-level prediction (Sprint 2, item 2.1)
# Friston 2008: hierarchical predictive coding with temporal abstraction
# L1 operates at session timescale (minutes-hours) vs L0 turn timescale (seconds)
SESSION_GAP_HOURS = 2           # Time gap to detect new session boundary
GOAL_ACCUMULATION_TURNS = 3     # Turns to accumulate before locking session goal
L1_PE_THRESHOLD = 0.35          # Session PE threshold for significance

# L2: Meta-level prediction (Sprint 2, item 2.2)
# Koriat 2000: metamemory monitoring = predicting own performance
# L2 operates at domain timescale (days-weeks) vs L1 session timescale
L2_EMA_ALPHA = 0.1              # Slow EMA for self-prediction update
L2_WINDOW = 15                  # Rolling window for actual accuracy per domain
L2_MIN_SAMPLES = 5              # Minimum samples before computing meta-PE
L2_PE_THRESHOLD = 0.20          # Meta-PE threshold for significance

# PE cascades (Sprint 2, item 2.4)
# Mumford 1992: hierarchical PE propagation — large PE at lower level
# signals model failure to higher level, amplified to ensure attention
CASCADE_THRESHOLD_MULT = 2.0    # PE must exceed N * threshold to trigger cascade
CASCADE_AMPLIFICATION = 1.5     # Amplification factor for upward propagation

# S4-04: PE-gated dual process (Kahneman 2011, Friston 2017)
# Low PE → System 1 (habit): skip FTS heavy search, use WM + triggers only
# High PE → System 2 (deliberate): full search pipeline
# Threshold: surprise_info is None means PE below adaptive threshold → habit
HABIT_MODE_ENABLED = True      # Feature flag for dual-process gating

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

# K.0.1: DDL flag guard — avoid running CREATE TABLE IF NOT EXISTS 3x per turn
_DDL_DONE = False
_PREDICTION_DDL_DONE = False  # 14.1: Guard for _init_prediction_tables (11 DDL statements)

# K.0.2: Module-level import for active_inference_integration
# Avoids cold import inside functions (subprocess model = every call is "first")
try:
    sys.path.insert(0, BASE_DIR)
    from modules.active_inference_integration import get_last_ac as _get_last_ac
    from modules.active_inference_integration import recommend_action as _recommend_action
except Exception:
    _get_last_ac = None
    _recommend_action = None

# 14.1: Module-level imports for hot-path modules
try:
    from modules.prospective import check_intentions as _check_intentions
except Exception:
    _check_intentions = None

# #115: Import PG working memory to replace frozen SQLite WM
try:
    from modules.working_memory import get_working_memory as _pg_get_working_memory
    _use_pg_wm = True
except Exception:
    _pg_get_working_memory = None
    _use_pg_wm = False

try:
    from modules.competition import CompetitionCandidate as _CompetitionCandidate
    from modules.competition import run_workspace_competition as _run_workspace_competition
except Exception:
    _CompetitionCandidate = None
    _run_workspace_competition = None

try:
    from modules.events import event_bus as _event_bus, Events as _Events
except Exception:
    _event_bus = None
    _Events = None


def _ensure_attention_tables(conn):
    """K.0.1: Create attention tables once per process, skip on subsequent calls.

    Replaces 3 inline CREATE TABLE IF NOT EXISTS blocks that each acquire
    a schema lock. Saves ~10-15ms per turn.
    """
    global _DDL_DONE
    if _DDL_DONE:
        return
    conn.execute("""
        CREATE TABLE IF NOT EXISTS attention_state (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS attention_transitions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            from_topic TEXT NOT NULL,
            to_topic TEXT NOT NULL,
            driver TEXT DEFAULT 'unknown',
            created_at TEXT NOT NULL
        )
    """)
    _DDL_DONE = True


def contains_sensitive_data(text):
    """Check if text contains credentials or sensitive data."""
    return bool(SENSITIVE_RE.search(text))


def get_db_connection(read_only=True):
    """Get SQLite connection with WAL mode for concurrent reads."""
    conn = sqlite3.connect(FTS_DB_PATH, timeout=10)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=10000")
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
    """Get top active working memory items by relevance.

    Post-pgvector migration (#115): reads from PostgreSQL via module import.
    Falls back to SQLite if PG unavailable.
    """
    # Sources that are internal system signals — they drive behavior via
    # wiring.py loops but should NOT enter the LLM's verbal context.
    # Neuroscience basis: prediction errors (cerebellum/basal ganglia) adjust
    # motor/cognitive behavior without entering phenomenal consciousness.
    # The LLM should see consequences ("curious about X"), not raw signals.
    _INTERNAL_SOURCES = {"prediction_error", "system", "event_bus", "active_inference"}

    if _use_pg_wm and _pg_get_working_memory:
        try:
            import json as _json
            result = _pg_get_working_memory()
            items = _json.loads(result).get("items", [])[:limit]
            return [
                (it["content"], it["topic"], it["relevance"])
                for it in items
                if it.get("source", "interaction") not in _INTERNAL_SOURCES
            ]
        except Exception:
            pass  # Fall through to SQLite
    # Fallback: SQLite (pre-migration data, frozen)
    try:
        cursor = conn.execute("""
            SELECT content, topic, relevance
            FROM working_memory
            WHERE active = 1 AND COALESCE(source, 'interaction') NOT IN ('prediction_error', 'system', 'event_bus', 'active_inference')
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
        if _check_intentions is None:
            return []
        return _check_intentions(prompt)
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


def _run_gwt_competition(fts_results, wm_results, intentions, triggers, surprise_info, slots=8):
    """K.1.1: Full GNW competition via competition.py (Dehaene & Changeux 2011).

    Converts preturn candidates to CompetitionCandidate objects, runs the
    5-phase GNW competition (attention bias, coalition, ignition, softmax,
    recurrent amplification), then converts back to preturn dict format.

    Falls back to _run_lightweight_competition if competition.py unavailable.
    """
    if _CompetitionCandidate is None or _run_workspace_competition is None:
        return _run_lightweight_competition(fts_results, wm_results, intentions, triggers, surprise_info, slots)

    CompetitionCandidate = _CompetitionCandidate
    run_workspace_competition = _run_workspace_competition

    candidates = []

    # FTS → episodic domain
    if fts_results:
        for i, row in enumerate(fts_results):
            activation = 1.0 / (1.0 + i)
            importance = row[3] if len(row) > 3 else "medium"
            if importance == "critical":
                activation = min(1.0, activation + 0.2)
            content = row[1] if len(row) > 1 else str(row)
            candidates.append(CompetitionCandidate(
                content=str(content)[:200],
                source_domain="episodic",
                activation=activation,
                memory_id=str(row[0]) if row else "",
                metadata={"topic": row[4] if len(row) > 4 else "", "domain": "fts", "index": i, "_data": row},
            ))

    # WM → working_memory domain
    if wm_results:
        for i, row in enumerate(wm_results):
            activation = float(row[2]) if len(row) > 2 else 0.5
            candidates.append(CompetitionCandidate(
                content=str(row[0])[:200] if row else "",
                source_domain="working_memory",
                activation=activation,
                memory_id=str(row[0]) if row else "",
                metadata={"topic": row[1] if len(row) > 1 else "", "domain": "wm", "index": i, "_data": row},
            ))

    # PM → prospective domain
    if intentions:
        priority_w = {"critical": 1.0, "high": 0.85, "medium": 0.65, "low": 0.4}
        for i, intent in enumerate(intentions):
            score = priority_w.get(intent.get("priority", "medium"), 0.5)
            act = intent.get("activation", 0.5)
            activation = min(1.0, score * act)
            candidates.append(CompetitionCandidate(
                content=intent.get("action", "")[:200],
                source_domain="prospective",
                activation=activation,
                metadata={"domain": "pm", "index": i, "_data": intent},
            ))

    # Triggers → trigger domain
    if triggers:
        for i, t in enumerate(triggers):
            candidates.append(CompetitionCandidate(
                content=str(t)[:200],
                source_domain="trigger",
                activation=0.6,
                metadata={"domain": "trigger", "index": i, "_data": t},
            ))

    # Surprise → prediction domain
    if surprise_info:
        candidates.append(CompetitionCandidate(
            content=str(surprise_info.get("predicted_topic", ""))[:200],
            source_domain="prediction",
            activation=0.8,
            metadata={"domain": "surprise", "_data": surprise_info},
        ))

    if not candidates:
        return {"fts": [], "wm": [], "pm": [], "trigger": [], "surprise": None}

    # Get current attention focus for top-down bias
    current_focus = None
    try:
        conn = sqlite3.connect(FTS_DB_PATH, timeout=10)
        conn.execute("PRAGMA busy_timeout=10000")
        row = conn.execute(
            "SELECT value FROM attention_state WHERE key = 'current_focus'"
        ).fetchone()
        current_focus = row[0] if row else None
        conn.close()
    except Exception:
        pass

    # Run full GNW competition
    result = run_workspace_competition(candidates, slots=slots, current_focus=current_focus)

    # Convert winners back to preturn dict format
    output = {"fts": [], "wm": [], "pm": [], "trigger": [], "surprise": None}
    for winner in result.winners:
        domain = winner.metadata.get("domain", "")
        data = winner.metadata.get("_data")
        if domain == "fts":
            output["fts"].append(data)
        elif domain == "wm":
            output["wm"].append(data)
        elif domain == "pm":
            output["pm"].append(data)
        elif domain == "trigger":
            output["trigger"].append(data)
        elif domain == "surprise":
            output["surprise"] = data

    return output


def format_context(fts_results, wm_results, triggers, intentions=None, surprise_info=None, l1_info=None, l2_info=None):
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

    # L1: Session-level PE (Sprint 2, item 2.1)
    if l1_info:
        parts.append("## Session Goal Drift (L1)")
        parts.append(f"- Session goal: {l1_info['session_goal']}, "
                     f"actual: {l1_info['actual_topic']}")
        parts.append(f"- Session PE: {l1_info['session_pe']:.2f} "
                     f"(mean: {l1_info['mean_session_pe']:.2f})")
        parts.append(f"- Turn {l1_info['turn_number']} of session {l1_info['session_id']}")

    # L2: Meta-level PE (Sprint 2, item 2.2)
    if l2_info:
        parts.append("## Meta-PE: Self-Model Discrepancy (L2)")
        parts.append(f"- Domain: {l2_info['domain']} — {l2_info['direction']}")
        parts.append(f"- Self-predicted accuracy: {l2_info['predicted_accuracy']:.0%}, "
                     f"actual: {l2_info['actual_accuracy']:.0%}")
        parts.append(f"- Meta-PE: {l2_info['meta_pe']:.2f} (n={l2_info['sample_size']})")

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
    global _PREDICTION_DDL_DONE
    if _PREDICTION_DDL_DONE:
        return
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
            source TEXT,
            created_at TEXT NOT NULL
        )
    """)
    # Backward-compatible migration: older DBs created before `source` existed.
    prediction_results_columns = {
        row[1] for row in conn.execute("PRAGMA table_info(prediction_results)").fetchall()
    }
    if "source" not in prediction_results_columns:
        conn.execute("ALTER TABLE prediction_results ADD COLUMN source TEXT")
    conn.execute("""
        UPDATE prediction_results
        SET source = 'interactive'
        WHERE source IS NULL
    """)
    # Transition frequency stats for entropy-based precision (Friston 2010, Bogacz 2017)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS transition_stats (
            from_topic TEXT NOT NULL,
            to_topic TEXT NOT NULL,
            count INTEGER DEFAULT 0,
            last_seen TEXT,
            PRIMARY KEY (from_topic, to_topic)
        )
    """)
    # L1: Session-level prediction state (Sprint 2, item 2.1)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS prediction_state_l1 (
            id INTEGER PRIMARY KEY DEFAULT 1,
            session_id TEXT,
            session_start TEXT,
            turn_count INTEGER DEFAULT 0,
            goal_topics TEXT DEFAULT '{}',
            goal_locked INTEGER DEFAULT 0,
            predicted_trajectory TEXT DEFAULT '{}',
            last_topic TEXT,
            last_turn_at TEXT,
            session_pe_sum REAL DEFAULT 0.0,
            session_pe_count INTEGER DEFAULT 0
        )
    """)
    # L1: Session PE history (needed for L1 precision learning in 2.3)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS prediction_results_l1 (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT,
            turn_number INTEGER,
            predicted_goal TEXT,
            actual_topic TEXT,
            session_pe REAL,
            precision_l1 REAL,
            weighted_pe_l1 REAL,
            created_at TEXT
        )
    """)
    # L2: Per-domain self-capability state (Sprint 2, item 2.2)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS prediction_state_l2 (
            domain TEXT PRIMARY KEY,
            predicted_accuracy REAL DEFAULT 0.5,
            actual_accuracy REAL DEFAULT 0.5,
            sample_size INTEGER DEFAULT 0,
            last_updated TEXT
        )
    """)
    # L2: Meta-PE history (needed for L2 precision learning in 2.3)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS prediction_results_l2 (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            domain TEXT,
            predicted_accuracy REAL,
            actual_accuracy REAL,
            meta_pe REAL,
            precision_l2 REAL,
            weighted_pe_l2 REAL,
            created_at TEXT
        )
    """)
    # L3: Project-level prediction state (K.2.2, Kiebel 2008)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS prediction_state_l3 (
            id INTEGER PRIMARY KEY DEFAULT 1,
            primary_project TEXT,
            topic_affinities TEXT DEFAULT '{}',
            active_since TEXT,
            sample_size INTEGER DEFAULT 0,
            updated_at TEXT
        )
    """)
    # L3: Project PE history
    conn.execute("""
        CREATE TABLE IF NOT EXISTS prediction_results_l3 (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            predicted_project TEXT,
            actual_project TEXT,
            project_pe REAL,
            created_at TEXT
        )
    """)
    # Metacognitive cycle traces (K.2.4, Nelson & Narens 1990)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS metacognitive_cycle_traces (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            turn_number INTEGER,
            l0_pe_trend REAL,
            l0_pe_status TEXT,
            session_on_track INTEGER,
            drifted_domains TEXT,
            max_drift REAL DEFAULT 0.0,
            precision_adjustment REAL DEFAULT 0.0,
            wm_alert_pushed INTEGER DEFAULT 0,
            cycle_duration_ms REAL DEFAULT 0.0,
            created_at TEXT
        )
    """)
    conn.commit()
    _PREDICTION_DDL_DONE = True


def _classify_topic(text):
    """Classify text into a topic using keyword matching. O(n) scan, ~1ms."""
    if not text:
        return 'general'
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


def _digamma(x):
    """Pure-Python digamma (psi) function via Stirling series.

    K.0.3: Replaces scipy.special.digamma to avoid 200-500ms cold import.
    Abramowitz & Stegun 1972, section 6.3.18.

    Uses recurrence psi(x) = psi(x+1) - 1/x to shift x >= 6,
    then Stirling series (8 terms, error < 1e-8 for x >= 6).
    """
    result = 0.0
    # Recurrence: shift x into safe zone (x >= 6)
    while x < 6.0:
        result -= 1.0 / x
        x += 1.0
    # Stirling series: psi(x) ~ ln(x) - 1/(2x) - sum B_{2k}/(2k * x^{2k})
    # Bernoulli coefficients: 1/12, 1/120, 1/252, 1/240, 5/660, 691/32760, 1/12
    x2 = 1.0 / (x * x)
    result += (math.log(x) - 0.5 / x
               - x2 * (1.0/12.0
               - x2 * (1.0/120.0
               - x2 * (1.0/252.0
               - x2 * (1.0/240.0
               - x2 * (5.0/660.0
               - x2 * (691.0/32760.0
               - x2 / 12.0)))))))
    return result


def _bayesian_surprise_dirichlet(topic_distribution, actual_topic, alpha=None):
    """True Bayesian Surprise (Itti & Baldi 2009, Eq. 5).

    J.1.3: Upgrades Level 1 PE from Shannon surprisal (-log P) to the full
    Dirichlet-Categorical KL divergence KL(Dir_posterior || Dir_prior).

    For Dirichlet prior Dir(alpha_1,...,alpha_k), after observing topic j:
      Posterior: Dir(alpha_1,...,alpha_j+1,...,alpha_k)
      KL = sum ln(G(a_i)/G(a'_i)) - ln(G(sum a_i)/G(sum a'_i))
           + sum (a'_i - a_i)(psi(a'_i) - psi(sum a'))

    K.0.3: Pure-Python implementation — no scipy/numpy dependency.
    Uses math.lgamma (stdlib) + _digamma() (Stirling series).
    Saves 200-500ms per subprocess invocation.

    Itti & Baldi 2009: S(D,M) = KL(P(M|D) || P(M))
    """
    if alpha is None:
        alpha = LAPLACE_ALPHA
    try:
        all_topics = list(TOPIC_KEYWORDS.keys()) + ["other"]
        prior_alpha = [topic_distribution.get(t, 0) + alpha for t in all_topics]
        post_alpha = list(prior_alpha)
        if actual_topic in all_topics:
            post_alpha[all_topics.index(actual_topic)] += 1.0
        else:
            post_alpha[-1] += 1.0

        sum_prior = sum(prior_alpha)
        sum_post = sum(post_alpha)

        # KL(Dir_post || Dir_prior) in nats, then convert to bits
        lgamma_diff = sum(math.lgamma(p) - math.lgamma(q) for p, q in zip(prior_alpha, post_alpha))
        digamma_sum_post = _digamma(sum_post)
        elementwise = sum(
            (q - p) * (_digamma(q) - digamma_sum_post)
            for p, q in zip(prior_alpha, post_alpha)
        )
        kl_nats = lgamma_diff - (math.lgamma(sum_prior) - math.lgamma(sum_post)) + elementwise
        kl_bits = max(0.0, kl_nats / math.log(2))
        return min(1.0, kl_bits / 4.0)
    except Exception:
        # Fallback to S2-01 Shannon surprisal
        if topic_distribution and sum(topic_distribution.values()) > 0:
            total = sum(topic_distribution.values())
            n_topics = max(len(TOPIC_KEYWORDS) + 1, len(topic_distribution))
            count = topic_distribution.get(actual_topic, 0)
            prob = (count + alpha) / (total + alpha * n_topics)
            return min(1.0, -math.log2(max(prob, 1e-10)) / 4.0)
        return 0.0 if actual_topic else 1.0


def _compute_surprise(predicted_topic, predicted_keywords_str, predicted_action,
                       actual_topic, actual_keywords, actual_action,
                       topic_distribution=None):
    """Compute 3-level surprise score (0-1).

    J.1.3: Level 1 now uses true Dirichlet-Categorical KL divergence
    (Itti & Baldi 2009 Bayesian Surprise) instead of Shannon surprisal.
    S2-01 Shannon surprisal is kept as fallback when scipy unavailable.

    Level 1: KL(Dir_posterior || Dir_prior), normalized, weighted 0.50
    Level 2: Keyword surprise (Jaccard distance, weighted 0.30)
    Level 3: Action surprise (binary match, weighted 0.20)

    Shannon 1948, Friston 2005: surprise = -log p(o|m)
    Itti & Baldi 2009: S(D,M) = KL(P(M|D) || P(M))
    """
    # Level 1: Bayesian topic surprise (J.1.3: Dirichlet KL, was S2-01 Shannon)
    if topic_distribution and sum(topic_distribution.values()) > 0:
        topic_surprise = _bayesian_surprise_dirichlet(
            topic_distribution, actual_topic, alpha=LAPLACE_ALPHA
        )
    else:
        # Fallback: binary match when no distribution available
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


# Sprint 11: Conserved Functional Form (PN-21)
# All prediction levels share the SAME algorithm, only parameters differ.
# Conserved form: observe → predict → compare → HGF precision update → cascade PE
#
# | Level | State Space | Observation    | Omega | Initial Sigma | Prior | Window      |
# |-------|-------------|----------------|-------|---------------|-------|-------------|
# | L0    | topics      | topic_of_turn  | 0.10  | 0.5           | 0.3   | MARKOV (80) |
# | L1    | goals       | goal_alignment | 0.08  | 0.3           | 0.5   | session     |
# | L2    | domains     | accuracy       | 0.05  | 0.2           | 0.7   | all_history |
# | L3    | projects    | project_topic  | 0.03  | 0.15          | 0.8   | 7 days      |
#
# Precision update (shared across ALL levels):
#   sigma(t) = sigma(t-1) + omega * (pe^2 - sigma(t-1))  [Mathys 2011]
#   precision = 1 / (1 + sigma)
#
# What differs per level: state_space, observation_model, omega, window.
# What is CONSERVED: precision update equation, PE computation, cascade logic.

PREDICTION_LEVELS = {
    'L0': {'omega': 0.10, 'initial_sigma': 0.5, 'prior': 0.3, 'window': 'markov'},
    'L1': {'omega': 0.08, 'initial_sigma': 0.3, 'prior': 0.5, 'window': 'session'},
    'L2': {'omega': 0.05, 'initial_sigma': 0.2, 'prior': 0.7, 'window': 'all'},
    'L3': {'omega': 0.03, 'initial_sigma': 0.15, 'prior': 0.8, 'window': '7d'},
}


def _hgf_precision(pe_history, initial_sigma=0.5, omega=0.1):
    """Compute adaptive precision using HGF-style volatility update (K.2.1).

    Mathys et al. 2011: precision = 1/(1 + sigma), where sigma evolves
    via: sigma(t) = sigma(t-1) + omega * (pe^2 - sigma(t-1))

    CONSERVED FORM: This exact function is used by ALL prediction levels
    (L0, L1, L2, L3) with level-specific omega and initial_sigma.
    See PREDICTION_LEVELS dict above.
    """
    if len(pe_history) < 2:
        return 1.0 / (1.0 + initial_sigma)

    sigma = initial_sigma
    for pe in pe_history:
        prediction_error_sq = pe ** 2
        sigma = sigma + omega * (prediction_error_sq - sigma)
        sigma = max(0.05, min(2.0, sigma))  # bound sigma

    precision = 1.0 / (1.0 + sigma)
    return max(0.15, min(0.95, precision))


def _compute_precision(conn, actual_topic):
    """Compute precision weight from PE variance (Friston 2008 M-step).

    pi ~ 1/Var(PE): low PE variance = stable predictions = high precision.
    High PE variance = unstable predictions = low precision.
    Replaces heuristic (0.3 + accuracy * 0.5) with learned precision.
    Canon v2 invariant S2-2, claim G-top.
    """
    try:
        # Fetch recent weighted_surprise values for this topic
        # Source filter: only interactive (PCI fix #42, skip sleep_loop)
        rows = conn.execute("""
            SELECT weighted_surprise FROM (
                SELECT weighted_surprise FROM prediction_results
                WHERE actual_topic = ? AND COALESCE(source, 'interactive') = 'interactive'
                ORDER BY id DESC LIMIT 20
            )
        """, (actual_topic,)).fetchall()

        if len(rows) < 3:
            return 0.3  # Prior for unfamiliar topics (low confidence)

        values = [r[0] for r in rows]
        mean_pe = sum(values) / len(values)
        variance = sum((v - mean_pe) ** 2 for v in values) / len(values)

        # K.2.1: HGF adaptive precision (Mathys 2011)
        # L0: fast timescale — initial_sigma=0.5, omega=0.10
        precision = _hgf_precision(values, initial_sigma=0.5, omega=0.10)

        # Sprint 3, item 3.1: L2→L0 top-down control signal
        # When L2 detects overconfidence in this domain, dampen precision.
        # Nelson & Narens 1990: metacognitive control modulates lower levels.
        try:
            l2_row = conn.execute(
                "SELECT predicted_accuracy, actual_accuracy FROM prediction_state_l2 WHERE domain = ?",
                (actual_topic,)
            ).fetchone()
            if l2_row:
                l2_pred, l2_actual = l2_row
                if l2_pred > l2_actual + 0.15:
                    # L2 says we're overconfident in this domain: dampen precision
                    overconf_penalty = min(0.3, (l2_pred - l2_actual))
                    precision = max(0.15, precision - overconf_penalty)
        except Exception:
            pass

        # Sprint 4, item 4.3: Emotion → precision modulation
        # Positive AC → increase precision (exploit, narrow predictions)
        # Negative AC → decrease precision (explore, broaden predictions)
        # Dayan & Yu 2006: neuromodulators (ACh/NE) modulate precision
        # K.0.2: Use module-level import (avoids cold import per subprocess)
        try:
            if _get_last_ac:
                ac = _get_last_ac()
                if abs(ac) > 0.01:
                    # Scale: AC of ±0.3 → precision shift of ±0.1
                    emotion_modulation = ac * 0.33
                    precision = max(0.15, min(0.95, precision + emotion_modulation))
        except Exception:
            pass

        return precision
    except Exception:
        return 0.3


def _compute_precision_l1(conn):
    """Compute L1 (session-level) precision from PE variance.

    K.2.1: HGF adaptive precision (Mathys 2011).
    L1: medium timescale — initial_sigma=0.3, omega=0.08.
    Uses session_pe history from prediction_results_l1.
    Sprint 2, item 2.3.
    """
    try:
        rows = conn.execute("""
            SELECT session_pe FROM (
                SELECT session_pe FROM prediction_results_l1
                ORDER BY id DESC LIMIT 20
            )
        """).fetchall()

        if len(rows) < 3:
            return 0.5  # Higher prior than L0 (meta-levels start more confident)

        values = [r[0] for r in rows]
        # K.2.1: HGF adaptive precision (Mathys 2011)
        # L1: medium timescale — initial_sigma=0.3, omega=0.08
        precision = _hgf_precision(values, initial_sigma=0.3, omega=0.08)
        return precision
    except Exception:
        return 0.5


def _compute_precision_l2(conn, domain=None):
    """Compute L2 (meta-level) precision from PE variance.

    K.2.1: HGF adaptive precision (Mathys 2011).
    L2: slow timescale — initial_sigma=0.2, omega=0.05.
    Uses meta_pe history from prediction_results_l2.
    Sprint 2, item 2.3.
    """
    try:
        if domain:
            rows = conn.execute("""
                SELECT meta_pe FROM (
                    SELECT meta_pe FROM prediction_results_l2
                    WHERE domain = ?
                    ORDER BY id DESC LIMIT 20
                )
            """, (domain,)).fetchall()
        else:
            rows = conn.execute("""
                SELECT meta_pe FROM (
                    SELECT meta_pe FROM prediction_results_l2
                    ORDER BY id DESC LIMIT 20
                )
            """).fetchall()

        if len(rows) < 3:
            return 0.7  # Highest prior: meta-level should be most stable

        values = [r[0] for r in rows]
        # K.2.1: HGF adaptive precision (Mathys 2011)
        # L2: slow timescale — initial_sigma=0.2, omega=0.05
        precision = _hgf_precision(values, initial_sigma=0.2, omega=0.05)

        # K.2.2: L3→L2 backward prior (Kiebel 2008)
        # Project context modulates L2 precision
        l3_modifier = _l3_backward_prior(conn)
        precision = max(0.15, min(0.95, precision * l3_modifier))

        return precision
    except Exception:
        return 0.7


def _compute_precision_l3(conn):
    """Compute L3 (project-level) precision from PE variance.

    Sprint 11: Conserved functional form (PN-21).
    L3: slowest timescale — initial_sigma=0.15, omega=0.03.
    Uses project_pe history from prediction_results_l3.
    """
    try:
        rows = conn.execute("""
            SELECT project_pe FROM (
                SELECT project_pe FROM prediction_results_l3
                ORDER BY id DESC LIMIT 20
            )
        """).fetchall()

        if len(rows) < 3:
            return PREDICTION_LEVELS['L3']['prior']

        values = [r[0] for r in rows]
        precision = _hgf_precision(
            values,
            initial_sigma=PREDICTION_LEVELS['L3']['initial_sigma'],
            omega=PREDICTION_LEVELS['L3']['omega'],
        )
        return precision
    except Exception:
        return PREDICTION_LEVELS['L3']['prior']


def _compute_transparency(conn):
    """Compute response transparency level from L2 precision (Sprint 4, item 4.4).

    Clark 2016: High precision signals → broadcast content directly (transparent).
    Low precision signals → accompany with metacognitive hedging (opaque).

    Returns a string instruction for response style, or None if no data.
    """
    try:
        # Get L2 precision (meta-level confidence)
        l2_precision = _compute_precision_l2(conn)

        # Get L0 precision for the current topic
        row = conn.execute(
            "SELECT predicted_topic FROM prediction_state WHERE id = 1"
        ).fetchone()
        topic = row[0] if row else "general"
        l0_precision = _compute_precision(conn, topic)

        # Average across levels for overall confidence
        avg_precision = (l0_precision + l2_precision) / 2.0

        if avg_precision >= 0.7:
            return "High confidence — respond directly and concisely"
        elif avg_precision >= 0.5:
            return "Moderate confidence — respond normally, note uncertainties if relevant"
        elif avg_precision >= 0.3:
            return "Low confidence — explain reasoning, flag uncertainty, suggest verification"
        else:
            return "Very low confidence — hedge strongly, ask clarifying questions before acting"
    except Exception:
        return None


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
        topic_distribution_str = row[3] or '{}'
        confidence = row[4] or 0.5

        # Parse stored topic distribution for information-theoretic surprise (S2-01)
        try:
            topic_distribution = json.loads(topic_distribution_str) if topic_distribution_str else {}
            # Ensure values are numeric
            topic_distribution = {k: float(v) for k, v in topic_distribution.items()}
        except (json.JSONDecodeError, TypeError, ValueError):
            topic_distribution = {}

        # Classify actual prompt
        actual_topic = _classify_topic(prompt)
        actual_keywords = _extract_keywords(prompt)
        actual_action = _classify_action(prompt)

        # Compute raw surprise (S2-01: information-theoretic)
        surprise = _compute_surprise(
            predicted_topic, predicted_keywords, predicted_action,
            actual_topic, actual_keywords, actual_action,
            topic_distribution=topic_distribution,
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

        # Record result with source tagging (Proposal #42: PCI fix)

        now = datetime.now().isoformat()

        # Detect source: explicit env var OR timing heuristic (P0 Bug #5 fix)
        # Removed topic=='codigo' restriction — sleep ticks can predict any topic
        source = os.environ.get('CODI_SOURCE', 'interactive')
        if source == 'interactive':
            last_row = conn.execute(
                "SELECT created_at FROM prediction_results ORDER BY id DESC LIMIT 1"
            ).fetchone()
            if last_row:
                last_time = datetime.fromisoformat(last_row[0])
                gap_min = (datetime.now() - last_time).total_seconds() / 60
                if 12 <= gap_min <= 35:
                    source = 'sleep_loop'

        conn.execute("""
            INSERT INTO prediction_results
            (predicted_topic, actual_topic, predicted_keywords, actual_keywords,
             surprise_score, precision_weight, weighted_surprise, hit, source, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            predicted_topic, actual_topic,
            predicted_keywords, json.dumps(list(actual_keywords)[:20]),
            surprise, precision, weighted_surprise, hit, source, now
        ))

        # Cleanup old results
        conn.execute("""
            DELETE FROM prediction_results
            WHERE id NOT IN (
                SELECT id FROM prediction_results ORDER BY id DESC LIMIT ?
            )
        """, (PREDICTION_RESULTS_MAX,))
        conn.commit()

        # Track topic transitions for entropy-based precision (Friston 2010, Bogacz 2017)
        # Bug #5 PCI fix: only record interactive transitions (sleep_loop inflates codigo→codigo)
        prev_actual_topic = None
        if source == 'interactive':
            try:
                _prev = conn.execute("""
                    SELECT actual_topic FROM prediction_results
                    WHERE COALESCE(source, 'interactive') != 'sleep_loop'
                      AND id < (SELECT MAX(id) FROM prediction_results)
                    ORDER BY id DESC LIMIT 1
                """).fetchone()
                if _prev:
                    prev_actual_topic = _prev[0]
                    conn.execute("""
                        INSERT INTO transition_stats (from_topic, to_topic, count, last_seen)
                        VALUES (?, ?, 1, ?)
                        ON CONFLICT(from_topic, to_topic) DO UPDATE SET
                            count = count + 1,
                            last_seen = excluded.last_seen
                    """, (prev_actual_topic, actual_topic, now))
                    conn.commit()
            except Exception:
                pass

        # Check against adaptive threshold
        threshold = _get_adaptive_threshold(conn)

        if weighted_surprise > threshold:
            # Proposal #59: Topic PE doesn't target specific memories.
            # The FTS keyword search was punishing innocent bystanders.
            affected_ids = []

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
                'prev_actual_topic': prev_actual_topic,
            }

        return None  # Below threshold, not surprising
    except Exception:
        return None


def _generate_prediction(conn, prompt, wm_topics=None, backward_msg=None):
    """PHASE 2 (post-comparison): Generate prediction for next turn.

    Sprint 2, item 2.6: Three messages at each node (complete inference).
    P(next_topic) ∝ Forward × Backward × Likelihood

    Forward (bottom-up): Markov transition model P(topic | current_topic)
    Backward (top-down): Session goal prior P(topic | session_goal) from L1/L2
    Likelihood (sensory): Current query evidence P(query | topic)

    Each message is a probability distribution over topics.
    Combined via element-wise multiplication (log-space addition) then normalized.
    Bishop 2006 Ch.8, Rao & Ballard 1999.
    """
    try:
        current_topic = _classify_topic(prompt)
        current_keywords = _extract_keywords(prompt)
        current_action = _classify_action(prompt)

        # All known topics (canonical set)
        all_topics = list(TOPIC_KEYWORDS.keys()) + ['general']

        # === MESSAGE 1: FORWARD (Markov transition model) ===
        # P(next_topic | current_topic) from historical transitions
        # #70: Exponential recency weighting (Behrens 2007) — recent transitions
        # count more than older ones. weight = exp(-λ * position).
        rows = conn.execute("""
            SELECT
                p1.actual_topic AS from_topic,
                p2.actual_topic AS to_topic
            FROM (
                SELECT id, actual_topic, source FROM prediction_results
                WHERE COALESCE(source, 'interactive') != 'sleep_loop'
                ORDER BY id DESC LIMIT ?
            ) p1
            JOIN (
                SELECT id, actual_topic, source FROM prediction_results
                WHERE COALESCE(source, 'interactive') != 'sleep_loop'
                ORDER BY id DESC LIMIT ?
            ) p2 ON p2.id = p1.id + 1
            ORDER BY p1.id DESC
        """, (MARKOV_WINDOW, MARKOV_WINDOW)).fetchall()

        forward_counts = {}
        for pos, (from_t, to_t) in enumerate(rows):
            if from_t == current_topic:
                weight = math.exp(-RECENCY_DECAY * pos)
                forward_counts[to_t] = forward_counts.get(to_t, 0) + weight

        # Fallback: overall frequency if no transitions from current_topic
        if not forward_counts:
            freq_rows = conn.execute("""
                SELECT actual_topic
                FROM prediction_results
                WHERE COALESCE(source, 'interactive') != 'sleep_loop'
                ORDER BY id DESC LIMIT ?
            """, (MARKOV_WINDOW,)).fetchall()
            for pos, (topic,) in enumerate(freq_rows):
                weight = math.exp(-RECENCY_DECAY * pos)
                forward_counts[topic] = forward_counts.get(topic, 0) + weight

        # Laplace smoothing (Behrens 2007)
        for t in all_topics:
            forward_counts[t] = forward_counts.get(t, 0) + LAPLACE_ALPHA

        # WM topics boost forward message
        if wm_topics:
            for t in wm_topics[:3]:
                forward_counts[t] = forward_counts.get(t, 0) + 1

        # CX-7: Causal DAG → weak informative priors (Pearl 2009, Bramley 2017)
        # WARNING: NOTEARS discovers correlation, not causation (Kaiser & Sipos 2022)
        # Uses WEAK priors only (kappa=0.1) — Markov signal must dominate
        DAG_PRIOR_KAPPA = 0.1  # WEAK coupling — do NOT increase without A/B testing
        try:
            dag_row = conn.execute(
                "SELECT w_matrix, topics FROM causal_discovery_state ORDER BY id DESC LIMIT 1"
            ).fetchone()
            if dag_row:
                dag_W = json.loads(dag_row[0])
                dag_topics = json.loads(dag_row[1])
                if current_topic in dag_topics:
                    from_idx = dag_topics.index(current_topic)
                    for j, to_topic in enumerate(dag_topics):
                        if to_topic == current_topic or to_topic not in forward_counts:
                            continue
                        w_ij = dag_W[from_idx][j] if from_idx < len(dag_W) and j < len(dag_W[from_idx]) else 0
                        if abs(w_ij) > 0.1:  # Edge threshold
                            # Multiplicative weak prior (Waldmann & Holyoak 1992)
                            factor = 1.0 + DAG_PRIOR_KAPPA * w_ij
                            forward_counts[to_topic] = max(0.01, forward_counts[to_topic] * factor)
        except Exception:
            pass  # DAG priors are advisory — never block prediction

        # Normalize to probability distribution
        fwd_total = sum(forward_counts.values()) or 1
        forward_dist = {t: forward_counts.get(t, LAPLACE_ALPHA) / fwd_total for t in all_topics}

        # Entropy regularization: mix with uniform prior to prevent absorbing-state dominance.
        # Without this, 'general' (62.5% base rate) collapses forward_dist → 80%,
        # making predictions uninformative. Bayesian mixing (Rao & Ballard 1999).
        _uniform = 1.0 / len(all_topics)
        forward_dist = {t: 0.70 * forward_dist[t] + 0.30 * _uniform for t in all_topics}

        # === MESSAGE 2: BACKWARD (L1 session goal → L0 topic prior) ===
        # P(topic | session_goal) from L1, modulated by L2 meta-confidence
        backward_dist = {}
        if backward_msg and backward_msg.get('session_goal_topics'):
            goal_topics = backward_msg['session_goal_topics']
            goal_weight = backward_msg.get('goal_weight', 2.0)
            bk_counts = {}
            for t in all_topics:
                bk_counts[t] = LAPLACE_ALPHA  # Flat prior
            for t, cnt in goal_topics.items():
                bk_counts[t] = bk_counts.get(t, 0) + cnt * goal_weight
            bk_total = sum(bk_counts.values()) or 1
            backward_dist = {t: bk_counts[t] / bk_total for t in all_topics}
        else:
            # No backward message: uniform (non-informative)
            uniform = 1.0 / len(all_topics)
            backward_dist = {t: uniform for t in all_topics}

        # === MESSAGE 3: LIKELIHOOD (current query evidence) ===
        # P(query | topic): how well does the current prompt match each topic?
        likelihood_dist = {}
        text_lower = prompt.lower()
        for t in all_topics:
            if t == 'general':
                likelihood_dist[t] = 0.1  # Low baseline for general
            else:
                kws = TOPIC_KEYWORDS.get(t, [])
                matches = sum(1 for kw in kws if kw in text_lower)
                # Sigmoid-like: 0 matches → 0.1, 1 → 0.4, 2 → 0.7, 3+ → 0.9
                likelihood_dist[t] = 0.1 + 0.8 * (1.0 - 1.0 / (1.0 + matches))
        # Normalize likelihood
        lk_total = sum(likelihood_dist.values()) or 1
        likelihood_dist = {t: v / lk_total for t, v in likelihood_dist.items()}

        # === COMBINE: Posterior ∝ Forward × Backward × Likelihood ===
        # In log-space to avoid underflow, then normalize
        posterior = {}
        for t in all_topics:
            # Multiply probabilities (equivalent to adding log-probs)
            posterior[t] = forward_dist[t] * backward_dist[t] * likelihood_dist[t]

        # Normalize posterior
        post_total = sum(posterior.values()) or 1e-10
        posterior_dist = {t: v / post_total for t, v in posterior.items()}

        # Convert to pseudo-counts for storage (compatible with _compute_surprise Laplace)
        # Scale factor = forward total, so Laplace smoothing behaves as before
        scale = max(fwd_total, 10.0)
        topic_counts = {t: p * scale for t, p in posterior_dist.items()}

        # Most likely next topic (MAP estimate)
        predicted_topic = max(topic_counts, key=topic_counts.get) if topic_counts else current_topic

        # Predicted keywords: current keywords (topic persistence) + WM keywords
        predicted_kw = list(current_keywords)[:15]

        # Predicted action: same as current (people tend to stay in the same mode)
        predicted_action = current_action

        # Confidence: posterior concentration (1 - normalized entropy)
        total = sum(topic_counts.values()) or 1
        entropy = -sum(p * math.log2(max(p, 1e-10)) for p in topic_counts.values() if p > 0)
        max_entropy = math.log2(len(all_topics))
        confidence = max(0.2, min(0.9, 1.0 - entropy / max_entropy if max_entropy > 0 else 0.5))


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

        # Proposal #62 Fix 4: Sync prediction transitions to attention_transitions
        # Unifies the two transition tracking systems (Tononi 2004 integration)
        prev_row = conn.execute("""
            SELECT actual_topic FROM prediction_results
            WHERE COALESCE(source, 'interactive') != 'sleep_loop'
            ORDER BY id DESC LIMIT 1
        """).fetchone()
        if prev_row and prev_row[0] and current_topic and prev_row[0] != current_topic:
            try:
                conn.execute(
                    "INSERT INTO attention_transitions (from_topic, to_topic, driver, created_at) "
                    "VALUES (?, ?, 'prediction_sync', ?)",
                    (prev_row[0], current_topic, now)
                )
            except Exception:
                pass

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

        conn = sqlite3.connect(FTS_DB_PATH, timeout=10)
        conn.execute("PRAGMA busy_timeout=10000")
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
    # 2. Topic PE labile path REMOVED by Proposal #65 Fix 1.
    # Topic PE means the Markov MODEL is wrong, not that episodic MEMORIES are wrong.
    # Correct reconsolidation trigger is contradiction detection (wiring.py),
    # not topic prediction errors. See Sevenster et al. 2013.
    # Learning from topic PE happens via transition_stats updates (line 604-611).
    # 3. Emit to event bus for in-process handlers (best-effort, may fail in system python3)
    try:
        if _event_bus is None or _Events is None:
            return
        _event_bus.emit(_Events.PREDICTION_ERROR, {
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


# --- L1: Session-Level Prediction (Sprint 2, item 2.1) ---
# Higher temporal level: predicts session trajectory from accumulated goals.
# Session = sequence of turns within SESSION_GAP_HOURS.
# Goal accumulates from first GOAL_ACCUMULATION_TURNS turns, then locks.
# Session PE fires when topic drifts from established goal.
# Friston 2008: hierarchical predictive coding with temporal abstraction.


def _get_or_create_session(conn):
    """Get current L1 session state, detecting new sessions by time gap.

    A new session starts when:
    - No prior state exists, OR
    - Gap since last turn > SESSION_GAP_HOURS

    Returns dict with session state.
    """

    now = datetime.now()

    row = conn.execute("""
        SELECT session_id, session_start, turn_count, goal_topics,
               goal_locked, predicted_trajectory, last_topic, last_turn_at,
               session_pe_sum, session_pe_count
        FROM prediction_state_l1 WHERE id = 1
    """).fetchone()

    if row and row[7]:  # has last_turn_at
        try:
            last_turn = datetime.fromisoformat(row[7])
            gap_hours = (now - last_turn).total_seconds() / 3600
            if gap_hours < SESSION_GAP_HOURS:
                # Same session — return existing state
                return {
                    'session_id': row[0],
                    'session_start': row[1],
                    'turn_count': row[2],
                    'goal_topics': json.loads(row[3] or '{}'),
                    'goal_locked': bool(row[4]),
                    'predicted_trajectory': json.loads(row[5] or '{}'),
                    'last_topic': row[6],
                    'last_turn_at': row[7],
                    'session_pe_sum': row[8] or 0.0,
                    'session_pe_count': row[9] or 0,
                    'is_new': False,
                }
        except (ValueError, TypeError):
            pass

    # New session: time gap exceeded or first ever
    session_id = f"s_{now.strftime('%Y%m%d%H%M')}"
    conn.execute("DELETE FROM prediction_state_l1")
    conn.execute("""
        INSERT INTO prediction_state_l1
        (id, session_id, session_start, turn_count, goal_topics,
         goal_locked, predicted_trajectory, last_topic, last_turn_at)
        VALUES (1, ?, ?, 0, '{}', 0, '{}', NULL, NULL)
    """, (session_id, now.isoformat()))
    conn.commit()

    return {
        'session_id': session_id,
        'session_start': now.isoformat(),
        'turn_count': 0,
        'goal_topics': {},
        'goal_locked': False,
        'predicted_trajectory': {},
        'last_topic': None,
        'last_turn_at': None,
        'session_pe_sum': 0.0,
        'session_pe_count': 0,
        'is_new': True,
    }


def _session_prediction_cycle(conn, prompt):
    """L1 session-level prediction: accumulate goal or compute session PE.

    Phase 1 (turns 1-N): Accumulate topic counts into session goal.
    Phase 2 (turns N+1...): Compare current topic against goal distribution,
    compute session PE using information-theoretic surprise.

    Returns dict with session PE info if goal is locked and PE is significant,
    None otherwise.
    """

    try:
        state = _get_or_create_session(conn)
        current_topic = _classify_topic(prompt)
        now = datetime.now().isoformat()

        state['turn_count'] += 1
        state['last_topic'] = current_topic
        state['last_turn_at'] = now

        session_pe_info = None

        if not state['goal_locked']:
            # Phase 1: Accumulate topic into session goal
            topics = state['goal_topics']
            topics[current_topic] = topics.get(current_topic, 0) + 1
            state['goal_topics'] = topics

            if state['turn_count'] >= GOAL_ACCUMULATION_TURNS:
                state['goal_locked'] = True
                # Predicted trajectory = the accumulated topic distribution
                # Kept as counts for proper Laplace smoothing during PE computation
                state['predicted_trajectory'] = dict(topics)
        else:
            # Phase 2: Compute session PE
            trajectory = state['predicted_trajectory']
            total = sum(trajectory.values()) or 1

            # Laplace-smoothed probability of current topic under session goal
            n_topics = max(len(trajectory), len(TOPIC_KEYWORDS) + 1)
            count = trajectory.get(current_topic, 0)
            prob = (count + LAPLACE_ALPHA) / (total + LAPLACE_ALPHA * n_topics)

            # Information-theoretic surprise: -log2(P), capped at 4 bits
            raw_pe = -math.log2(max(prob, 1e-10))
            session_pe = min(1.0, raw_pe / 4.0)

            # Per-level precision learning (Sprint 2, item 2.3)
            precision_l1 = _compute_precision_l1(conn)
            weighted_pe_l1 = session_pe * precision_l1

            # Track cumulative session PE
            state['session_pe_sum'] += weighted_pe_l1
            state['session_pe_count'] += 1

            # Record L1 result with precision
            primary_goal = max(trajectory, key=trajectory.get) if trajectory else 'general'
            conn.execute("""
                INSERT INTO prediction_results_l1
                (session_id, turn_number, predicted_goal, actual_topic,
                 session_pe, precision_l1, weighted_pe_l1, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (state['session_id'], state['turn_count'], primary_goal,
                  current_topic, session_pe, precision_l1, weighted_pe_l1, now))

            # Return info if weighted PE exceeds threshold
            if weighted_pe_l1 > L1_PE_THRESHOLD:
                session_pe_info = {
                    'session_pe': weighted_pe_l1,
                    'raw_session_pe': session_pe,
                    'precision_l1': precision_l1,
                    'session_id': state['session_id'],
                    'session_goal': primary_goal,
                    'goal_distribution': trajectory,
                    'actual_topic': current_topic,
                    'turn_number': state['turn_count'],
                    'mean_session_pe': (state['session_pe_sum'] / state['session_pe_count']
                                        if state['session_pe_count'] > 0 else 0.0),
                }

        # Persist updated state
        conn.execute("""
            UPDATE prediction_state_l1 SET
                turn_count = ?,
                goal_topics = ?,
                goal_locked = ?,
                predicted_trajectory = ?,
                last_topic = ?,
                last_turn_at = ?,
                session_pe_sum = ?,
                session_pe_count = ?
            WHERE id = 1
        """, (
            state['turn_count'],
            json.dumps(state['goal_topics']),
            1 if state['goal_locked'] else 0,
            json.dumps(state['predicted_trajectory']),
            current_topic,
            now,
            state['session_pe_sum'],
            state['session_pe_count'],
        ))
        conn.commit()

        return session_pe_info
    except Exception:
        return None


# --- L2: Meta-Level Prediction (Sprint 2, item 2.2) ---
# Highest temporal level: predicts own performance per domain.
# Koriat 2000: metamemory monitoring = predicting own capability.
# Self-prediction updates slowly (EMA alpha=0.1), reality updates fast (rolling window).
# Meta-PE fires when self-model diverges from actual performance.


def _meta_prediction_cycle(conn, actual_topic, l0_hit):
    """L2 meta-level prediction: track self-capability per domain.

    Maintains predicted_accuracy per domain (slow EMA self-model) and compares
    against actual_accuracy (fast rolling window). Meta-PE = divergence.

    Args:
        conn: SQLite connection (read-write)
        actual_topic: The topic classified for this turn
        l0_hit: Whether L0 prediction was correct (1) or wrong (0)

    Returns dict with meta-PE info if significant, None otherwise.
    """

    try:
        now = datetime.now().isoformat()
        domain = actual_topic or 'general'

        # Get current L2 state for this domain
        row = conn.execute("""
            SELECT predicted_accuracy, actual_accuracy, sample_size, last_updated
            FROM prediction_state_l2 WHERE domain = ?
        """, (domain,)).fetchone()

        if row:
            predicted_acc = row[0]
            old_actual_acc = row[1]
            sample_size = row[2]
        else:
            predicted_acc = 0.05  # Empirical prior: non-general domains avg ~5% L0 accuracy
            old_actual_acc = 0.05
            sample_size = 0

        # Compute actual accuracy from rolling window of recent results for this domain
        domain_rows = conn.execute("""
            SELECT hit FROM (
                SELECT hit FROM prediction_results
                WHERE actual_topic = ? AND COALESCE(source, 'interactive') != 'sleep_loop'
                ORDER BY id DESC LIMIT ?
            )
        """, (domain, L2_WINDOW)).fetchall()

        if len(domain_rows) >= L2_MIN_SAMPLES:
            actual_acc = sum(r[0] for r in domain_rows) / len(domain_rows)
        else:
            # Not enough data: update state but don't compute meta-PE
            conn.execute("""
                INSERT INTO prediction_state_l2 (domain, predicted_accuracy, actual_accuracy, sample_size, last_updated)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(domain) DO UPDATE SET
                    sample_size = excluded.sample_size,
                    last_updated = excluded.last_updated
            """, (domain, predicted_acc, old_actual_acc, len(domain_rows), now))
            conn.commit()
            return None

        # Meta-PE: divergence between self-prediction and reality
        # Normalized to 0-1: |predicted - actual| / 0.5 (max possible divergence from center)
        raw_meta_pe = abs(predicted_acc - actual_acc)
        meta_pe = min(1.0, raw_meta_pe / 0.5)

        # Update predicted_accuracy via slow EMA (self-model adapts slowly)
        new_predicted = predicted_acc + L2_EMA_ALPHA * (actual_acc - predicted_acc)

        # Persist updated state
        conn.execute("""
            INSERT INTO prediction_state_l2 (domain, predicted_accuracy, actual_accuracy, sample_size, last_updated)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(domain) DO UPDATE SET
                predicted_accuracy = excluded.predicted_accuracy,
                actual_accuracy = excluded.actual_accuracy,
                sample_size = excluded.sample_size,
                last_updated = excluded.last_updated
        """, (domain, new_predicted, actual_acc, len(domain_rows), now))

        # Per-level precision learning (Sprint 2, item 2.3)
        precision_l2 = _compute_precision_l2(conn, domain)
        weighted_pe_l2 = meta_pe * precision_l2

        # Record L2 result with precision
        conn.execute("""
            INSERT INTO prediction_results_l2
            (domain, predicted_accuracy, actual_accuracy, meta_pe,
             precision_l2, weighted_pe_l2, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (domain, predicted_acc, actual_acc, meta_pe, precision_l2, weighted_pe_l2, now))

        # Prune old L2 results (keep last 200)
        conn.execute("""
            DELETE FROM prediction_results_l2
            WHERE id NOT IN (
                SELECT id FROM prediction_results_l2 ORDER BY id DESC LIMIT 200
            )
        """)
        conn.commit()

        # Return info if weighted meta-PE exceeds threshold
        if weighted_pe_l2 > L2_PE_THRESHOLD:
            direction = "overconfident" if predicted_acc > actual_acc else "underconfident"
            return {
                'meta_pe': weighted_pe_l2,
                'raw_meta_pe': meta_pe,
                'precision_l2': precision_l2,
                'domain': domain,
                'predicted_accuracy': predicted_acc,
                'actual_accuracy': actual_acc,
                'direction': direction,
                'sample_size': len(domain_rows),
            }

        return None
    except Exception:
        return None


# --- L3: Project-Level Prediction (K.2.2, Kiebel 2008) ---
# Slowest temporal level: identifies active project from working memory topics.
# Projects operate at timescale of days→weeks.
# L3 backward prior modulates L2 precision based on project volatility.

L3_WM_WINDOW = 50       # Working memory items to consider
L3_MIN_SAMPLES = 5      # Minimum turns before L3 backward prior activates


def _l3_prediction_cycle(conn, actual_topic):
    """L3 project-level prediction: track which project is active.

    Identifies primary project from recent working memory topics (7-day window).
    Computes project PE when the active project changes.

    K.2.2 (Kiebel 2008 timescale hierarchy).
    """

    try:
        now = datetime.now().isoformat()

        # Step 1: Determine current project from working memory topics
        try:
            wm_rows = conn.execute("""
                SELECT topic, COUNT(*) as cnt FROM working_memory
                WHERE active = 1
                  AND created_at > datetime('now', '-7 days')
                GROUP BY topic
                ORDER BY cnt DESC
                LIMIT 10
            """).fetchall()
        except Exception:
            wm_rows = []

        if not wm_rows:
            # Fallback: use actual_topic from this turn
            current_project = actual_topic or 'general'
            topic_affinities = {current_project: 1.0}
        else:
            total = sum(r[1] for r in wm_rows)
            topic_affinities = {r[0]: round(r[1] / total, 3) for r in wm_rows}
            current_project = wm_rows[0][0]  # Highest count

        # Step 2: Get previous L3 state
        row = conn.execute(
            "SELECT primary_project, sample_size FROM prediction_state_l3 WHERE id = 1"
        ).fetchone()

        if row:
            predicted_project = row[0]
            sample_size = row[1]
        else:
            predicted_project = None
            sample_size = 0

        # Step 3: Compute project PE with HGF precision (Sprint 11: conserved form)
        if predicted_project is not None:
            raw_project_pe = 0.0 if current_project == predicted_project else 1.0
            # Sprint 11: Apply conserved HGF precision to L3
            precision_l3 = _compute_precision_l3(conn)
            project_pe = raw_project_pe * precision_l3

            # Record L3 result
            conn.execute("""
                INSERT INTO prediction_results_l3
                (predicted_project, actual_project, project_pe, created_at)
                VALUES (?, ?, ?, ?)
            """, (predicted_project, current_project, project_pe, now))

            # Prune old L3 results (keep last 100)
            conn.execute("""
                DELETE FROM prediction_results_l3
                WHERE id NOT IN (
                    SELECT id FROM prediction_results_l3 ORDER BY id DESC LIMIT 100
                )
            """)

        # Step 4: Update L3 state

        active_since = now if (predicted_project != current_project or not row) else (
            conn.execute("SELECT active_since FROM prediction_state_l3 WHERE id=1").fetchone() or (now,)
        )[0]

        conn.execute("""
            INSERT INTO prediction_state_l3 (id, primary_project, topic_affinities, active_since, sample_size, updated_at)
            VALUES (1, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                primary_project = excluded.primary_project,
                topic_affinities = excluded.topic_affinities,
                active_since = excluded.active_since,
                sample_size = excluded.sample_size,
                updated_at = excluded.updated_at
        """, (current_project, json.dumps(topic_affinities), active_since, sample_size + 1, now))
        conn.commit()

        return {
            'primary_project': current_project,
            'topic_affinities': topic_affinities,
            'project_pe': project_pe if predicted_project else None,
            'sample_size': sample_size + 1,
        }
    except Exception:
        return None


def _l3_backward_prior(conn):
    """L3→L2 backward prior: project context modulates L2 precision.

    Kiebel 2008: higher timescale levels provide priors to lower levels.
    Different projects have different PE volatility profiles.

    Returns precision modifier (multiplier, 1.0 = no effect).
    """
    try:
        row = conn.execute(
            "SELECT primary_project, sample_size FROM prediction_state_l3 WHERE id = 1"
        ).fetchone()
        if not row or row[1] < L3_MIN_SAMPLES:
            return 1.0  # Insufficient data: no modifier

        # Compute project stability from L3 PE history
        pe_rows = conn.execute("""
            SELECT project_pe FROM (
                SELECT project_pe FROM prediction_results_l3
                ORDER BY id DESC LIMIT 20
            )
        """).fetchall()

        if len(pe_rows) < 3:
            return 1.0

        # Project switch rate: how often does the project change?
        switch_rate = sum(r[0] for r in pe_rows) / len(pe_rows)

        # Stable project (low switch rate) → increase L2 precision (narrow predictions)
        # Volatile project (high switch rate) → decrease L2 precision (broaden)
        # Modifier range: [0.85, 1.15]
        modifier = 1.0 + 0.15 * (1.0 - 2.0 * switch_rate)
        return max(0.85, min(1.15, modifier))
    except Exception:
        return 1.0


# --- K.2.4: 12-Step Metacognitive Cycle (Nelson & Narens 1990) ---
# Proactive sweep every N turns. Monitoring → Evaluation → Control.
# Catches gradual drift that reactive checks miss.

METACOGNITIVE_CYCLE_PERIOD = 10  # Run every 10 turns


def _metacognitive_cycle(conn, turn_number):
    """Proactive metacognitive sweep: monitoring → evaluation → control.

    Nelson & Narens 1990: effective metacognition requires periodic monitoring
    (Object→Meta) and control (Meta→Object), not just reactive responses.

    Runs every METACOGNITIVE_CYCLE_PERIOD turns. Each step is fault-tolerant.
    K.2.4.
    """


    _start = time.monotonic()
    now = datetime.now().isoformat()

    # ---- Phase 1: MONITORING (Object → Meta, steps 1-4) ----

    # Step 1-3: L0 PE trend (last 10 turns)
    l0_pe_trend = 0.0
    l0_pe_status = 'stable'
    try:
        pe_rows = conn.execute("""
            SELECT weighted_surprise FROM (
                SELECT id, weighted_surprise FROM prediction_results
                WHERE COALESCE(source, 'interactive') != 'sleep_loop'
                ORDER BY id DESC LIMIT 10
            ) ORDER BY id
        """).fetchall()
        if len(pe_rows) >= 4:
            values = [r[0] for r in pe_rows]
            n = len(values)
            # Simple linear regression slope
            x_mean = (n - 1) / 2.0
            y_mean = sum(values) / n
            num = sum((i - x_mean) * (v - y_mean) for i, v in enumerate(values))
            den = sum((i - x_mean) ** 2 for i in range(n))
            l0_pe_trend = num / den if den > 0 else 0.0
            if l0_pe_trend > 0.02:
                l0_pe_status = 'degrading'
            elif l0_pe_trend < -0.02:
                l0_pe_status = 'improving'
    except Exception:
        pass

    # Step 4: Domain accuracies from L2
    domain_accuracies = {}
    try:
        rows = conn.execute(
            "SELECT domain, actual_accuracy, sample_size FROM prediction_state_l2"
        ).fetchall()
        domain_accuracies = {r[0]: r[1] for r in rows if r[2] >= 3}
    except Exception:
        pass

    # ---- Phase 2: EVALUATION (Meta-level comparison, steps 5-8) ----

    # Step 6: Session goal coherence
    session_on_track = True
    try:
        l1_row = conn.execute(
            "SELECT session_pe_sum, session_pe_count FROM prediction_state_l1 WHERE id=1"
        ).fetchone()
        if l1_row and l1_row[1] > 3:
            avg_session_pe = l1_row[0] / l1_row[1]
            if avg_session_pe > 0.6:
                session_on_track = False
    except Exception:
        pass

    # Step 7: Domain drift detection (compare against previous cycle)
    drifted_domains = []
    max_drift = 0.0
    try:
        prev_trace = conn.execute("""
            SELECT drifted_domains FROM metacognitive_cycle_traces
            ORDER BY id DESC LIMIT 1
        """).fetchone()
        # Compare current domain accuracies against L2 predicted
        for domain, actual_acc in domain_accuracies.items():
            l2_pred = conn.execute(
                "SELECT predicted_accuracy FROM prediction_state_l2 WHERE domain=?",
                (domain,)
            ).fetchone()
            if l2_pred and abs(l2_pred[0] - actual_acc) > 0.15:
                drift = abs(l2_pred[0] - actual_acc)
                drifted_domains.append(domain)
                max_drift = max(max_drift, drift)
    except Exception:
        pass

    # ---- Phase 3: CONTROL (Meta → Object, steps 9-12) ----

    # Step 9: Precision adjustment signal
    precision_adjustment = 0.0
    if l0_pe_status == 'degrading':
        precision_adjustment = -0.05  # Lower precision → more exploration
    elif l0_pe_status == 'improving':
        precision_adjustment = 0.03   # Raise precision → more exploitation

    # Step 10: WM alert for drifted domains
    wm_alert_pushed = False
    if drifted_domains:
        try:
            alert_content = (
                f"[Metacognitive alert] Domain drift detected: "
                f"{', '.join(drifted_domains)} (max drift: {max_drift:.2f}). "
                f"Prediction accuracy degrading in these areas."
            )
            conn.execute("""
                INSERT INTO working_memory (content, topic, relevance, active, added_at)
                VALUES (?, 'metacognition', 0.9, 1, ?)
            """, (alert_content, now))
            wm_alert_pushed = True
        except Exception:
            pass

    # Step 12: Trace logging
    duration_ms = (time.monotonic() - _start) * 1000
    try:
        conn.execute("""
            INSERT INTO metacognitive_cycle_traces
            (turn_number, l0_pe_trend, l0_pe_status, session_on_track,
             drifted_domains, max_drift, precision_adjustment,
             wm_alert_pushed, cycle_duration_ms, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (turn_number, l0_pe_trend, l0_pe_status,
              1 if session_on_track else 0,
              json.dumps(drifted_domains), max_drift,
              precision_adjustment, 1 if wm_alert_pushed else 0,
              duration_ms, now))
        # Keep last 200 traces
        conn.execute("""
            DELETE FROM metacognitive_cycle_traces
            WHERE id NOT IN (
                SELECT id FROM metacognitive_cycle_traces ORDER BY id DESC LIMIT 200
            )
        """)
        conn.commit()
    except Exception:
        pass

    return {
        'l0_pe_trend': l0_pe_trend,
        'l0_pe_status': l0_pe_status,
        'session_on_track': session_on_track,
        'drifted_domains': drifted_domains,
        'precision_adjustment': precision_adjustment,
        'wm_alert_pushed': wm_alert_pushed,
        'duration_ms': duration_ms,
    }


def _apply_pe_cascades(surprise_info, l1_info, l2_info):
    """Apply upward PE cascades across hierarchy levels.

    Mumford 1992, Friston 2008: When PE at a lower level exceeds 2x threshold,
    it amplifies 1.5x and propagates upward, signaling model failure.

    L0 → L1: Large topic PE implies session goal may be wrong.
    L1 → L2: Large session PE implies meta-level capability estimate may be wrong.

    Sprint 2, item 2.4.
    Returns (surprise_info, l1_info, l2_info) — possibly modified with cascade markers.
    """
    # L0 → L1 cascade
    if surprise_info:
        l0_pe = surprise_info.get('surprise', 0)
        l0_threshold = surprise_info.get('threshold', SURPRISE_MIN_THRESHOLD)
        if l0_pe > CASCADE_THRESHOLD_MULT * l0_threshold:
            cascade_pe = l0_pe * CASCADE_AMPLIFICATION
            if l1_info:
                # Boost existing L1 PE with cascade contribution
                l1_info['session_pe'] = l1_info.get('session_pe', 0) + cascade_pe * 0.3
                l1_info['cascade_from_l0'] = cascade_pe
            else:
                # Create L1 info from cascade alone (session goal implicitly challenged)
                l1_info = {
                    'session_pe': cascade_pe * 0.3,
                    'session_id': 'cascade',
                    'session_goal': surprise_info.get('predicted_topic', 'unknown'),
                    'goal_distribution': {},
                    'actual_topic': surprise_info.get('actual_topic', 'unknown'),
                    'turn_number': 0,
                    'mean_session_pe': 0.0,
                    'cascade_from_l0': cascade_pe,
                }

    # L1 → L2 cascade
    if l1_info:
        l1_pe = l1_info.get('session_pe', 0)
        if l1_pe > CASCADE_THRESHOLD_MULT * L1_PE_THRESHOLD:
            cascade_pe = l1_pe * CASCADE_AMPLIFICATION
            if l2_info:
                # Boost existing L2 meta-PE
                l2_info['meta_pe'] = l2_info.get('meta_pe', 0) + cascade_pe * 0.3
                l2_info['cascade_from_l1'] = cascade_pe
            else:
                # Create L2 info from cascade alone (meta-model implicitly challenged)
                l2_info = {
                    'meta_pe': cascade_pe * 0.3,
                    'domain': l1_info.get('actual_topic', 'general'),
                    'predicted_accuracy': 0.5,
                    'actual_accuracy': 0.0,
                    'direction': 'cascade',
                    'sample_size': 0,
                    'cascade_from_l1': cascade_pe,
                }

    return surprise_info, l1_info, l2_info


def _build_backward_message(conn):
    """Build top-down backward message from L1/L2 to constrain L0 prediction.

    Sprint 2, item 2.5. Rao & Ballard 1999: top-down predictions as priors.

    L1 → L0: Session goal topics bias next-topic prediction.
    L2 → L1: Domain accuracy modulates the strength of L1's influence.
      High L2 precision for session's domain → stronger goal prior.
      Low L2 precision → weaker goal prior (uncertain about this domain).

    Returns backward_msg dict or None.
    """
    try:
        # Get L1 session state
        row = conn.execute("""
            SELECT goal_topics, goal_locked, predicted_trajectory
            FROM prediction_state_l1 WHERE id = 1
        """).fetchone()

        if not row or not row[1]:  # No session or goal not locked
            return None

        goal_topics = json.loads(row[0] or '{}')
        if not goal_topics:
            return None

        # Determine primary session domain
        primary_goal = max(goal_topics, key=goal_topics.get)

        # L2 → L1: Get meta-level confidence for this domain
        l2_row = conn.execute("""
            SELECT predicted_accuracy, actual_accuracy
            FROM prediction_state_l2 WHERE domain = ?
        """, (primary_goal,)).fetchone()

        if l2_row:
            # L2 accuracy modulates goal weight: high accuracy → strong prior
            domain_accuracy = l2_row[1] or 0.5
            # Scale: 50% accuracy → weight 1.0, 80% → weight 2.5, 30% → weight 0.3
            goal_weight = max(0.3, domain_accuracy * 3.0)
        else:
            goal_weight = 2.0  # Default: moderate prior when no L2 data

        return {
            'session_goal_topics': goal_topics,
            'goal_weight': goal_weight,
            'primary_goal': primary_goal,
        }
    except Exception:
        return None


def _load_attention_state():
    """Load persisted attention transitions and last focus from SQLite.

    Enables cross-process bigram prediction (Graziano 2013 AST).
    Without this, _attention_schema resets each hook invocation and
    the predictor has no history to achieve pred_prob >= 0.5.
    """
    last_focus = None
    try:
        conn = sqlite3.connect(FTS_DB_PATH, timeout=10)
        conn.execute("PRAGMA busy_timeout=10000")
        _ensure_attention_tables(conn)
        rows = conn.execute(
            "SELECT from_topic, to_topic, driver, created_at "
            "FROM attention_transitions ORDER BY id DESC LIMIT 30"
        ).fetchall()

        transitions = [
            {"from": r[0], "to": r[1], "driver": r[2], "at": r[3]}
            for r in reversed(rows)
        ]
        last_focus = rows[0][1] if rows else None

        # Bootstrap: load saved focus when no transitions exist yet
        if not last_focus:
            try:
                row = conn.execute(
                    "SELECT value FROM attention_state WHERE key = 'current_focus'"
                ).fetchone()
                last_focus = row[0] if row else None
            except Exception:
                pass

        conn.close()
        return transitions, last_focus
    except Exception:
        return [], None


def _save_attention_transition(from_topic, to_topic, driver, created_at):
    """Persist a new topic transition to SQLite for cross-process continuity."""
    try:
        conn = sqlite3.connect(FTS_DB_PATH, timeout=10)
        conn.execute("PRAGMA busy_timeout=10000")
        _ensure_attention_tables(conn)
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



def _update_attention_from_prompt(prompt: str):
    """Update attention state using pure SQLite (no module imports).

    The heavy AST-1 prediction and self-correction happens in-process
    in the MCP server via wiring.py. The hook only records topic shifts.

    Proposal #62: Eliminated wiring.py import (776ms -> <8ms).
    Fixed Sisyphean transition persistence bug (count-based -> topic match).
    """
    topic = _classify_topic(prompt)
    if not topic:
        return
    try:

        now = datetime.now().isoformat()
        conn = sqlite3.connect(FTS_DB_PATH, timeout=10)
        conn.execute("PRAGMA busy_timeout=10000")
        _ensure_attention_tables(conn)

        # Get previous focus
        row = conn.execute(
            "SELECT value FROM attention_state WHERE key = 'current_focus'"
        ).fetchone()
        prev_focus = row[0] if row else None

        # Record transition if topic changed
        if prev_focus and prev_focus != topic:
            conn.execute(
                "INSERT INTO attention_transitions (from_topic, to_topic, driver, created_at) "
                "VALUES (?, ?, 'user_prompt', ?)",
                (prev_focus, topic, now)
            )
            # Prune old entries (keep last 100)
            conn.execute(
                "DELETE FROM attention_transitions WHERE id NOT IN "
                "(SELECT id FROM attention_transitions ORDER BY id DESC LIMIT 100)"
            )

        # Update current focus
        conn.execute("""
            INSERT INTO attention_state (key, value, updated_at)
            VALUES ('current_focus', ?, ?)
            ON CONFLICT(key) DO UPDATE SET
                value = excluded.value,
                updated_at = excluded.updated_at
        """, (topic, now))

        conn.commit()
        conn.close()
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

        # --- Update attention schema from prompt (pure SQLite, no module imports) ---
        _update_attention_from_prompt(prompt)

        # --- PREDICTION LOOP: Phase 1 - Compare (before processing) ---
        surprise_info = None
        l1_info = None
        l2_info = None
        try:
            pred_conn = get_db_connection(read_only=False)
            _init_prediction_tables(pred_conn)
            surprise_info = _compare_prediction(pred_conn, prompt)
            if surprise_info:
                _emit_prediction_error(surprise_info)
            # L1: Session-level prediction (Sprint 2, item 2.1)
            l1_info = _session_prediction_cycle(pred_conn, prompt)
            # Classify topic for L2/L3
            actual_topic = _classify_topic(prompt)
            # L3: Project-level prediction (K.2.2, Kiebel 2008)
            # Runs before L2 so backward prior is available
            l3_info = _l3_prediction_cycle(pred_conn, actual_topic)
            # L2: Meta-level prediction (Sprint 2, item 2.2)
            # Needs actual_topic and l0_hit from the compare phase
            l0_hit = 1 if (surprise_info and not surprise_info.get('topic_changed')) else (
                0 if surprise_info else None)
            # When no surprise_info, check if there was a prior prediction and it matched
            if l0_hit is None:
                _prior = pred_conn.execute(
                    "SELECT hit FROM prediction_results ORDER BY id DESC LIMIT 1"
                ).fetchone()
                l0_hit = _prior[0] if _prior else None
            if l0_hit is not None:
                l2_info = _meta_prediction_cycle(pred_conn, actual_topic, l0_hit)
            # PE cascades: upward propagation (Sprint 2, item 2.4)
            surprise_info, l1_info, l2_info = _apply_pe_cascades(surprise_info, l1_info, l2_info)
            # K.2.4: Proactive metacognitive sweep every N turns
            try:
                l1_state = pred_conn.execute(
                    "SELECT turn_count FROM prediction_state_l1 WHERE id=1"
                ).fetchone()
                if l1_state and l1_state[0] > 0 and l1_state[0] % METACOGNITIVE_CYCLE_PERIOD == 0:
                    _metacognitive_cycle(pred_conn, l1_state[0])
            except Exception:
                pass
        except Exception:
            pred_conn = None

        # --- S4-04: PE-gated Dual Process (Kahneman 2011) ---
        # System 1 (habit): low PE → skip FTS, use WM only (fast, <50ms)
        # System 2 (deliberate): high PE or first turn → full pipeline
        habit_mode = False
        if HABIT_MODE_ENABLED and surprise_info is None:
            # No significant PE — candidate for habit mode
            # But first turn (no prediction history) should be deliberate
            try:
                if pred_conn:
                    has_prior = pred_conn.execute(
                        "SELECT 1 FROM prediction_state WHERE id = 1"
                    ).fetchone()
                    habit_mode = has_prior is not None
            except Exception:
                pass

        conn = get_db_connection()
        try:
            # S4-04: Branch by processing mode
            if habit_mode:
                # System 1 (habit): skip FTS search, WM is sufficient
                fts_results = []
            else:
                # System 2 (deliberate): full FTS search pipeline
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
                    # Sprint 2, item 2.5: Backward message from L1/L2 -> L0
                    backward_msg = _build_backward_message(pred_conn)
                    _generate_prediction(pred_conn, prompt, wm_topics, backward_msg=backward_msg)
                    pred_conn.close()
            except Exception:
                pass

            # Nothing found? Don't inject empty context
            if not fts_results and not wm_results and not triggers and not intentions and not surprise_info and not l1_info and not l2_info:
                return

            # --- WIRING-6.3: GWT competition across domains ---
            # K.1.1: Full 5-phase GNW via competition.py (fallback to lightweight)
            comp = _run_gwt_competition(
                fts_results, wm_results, intentions, triggers, surprise_info
            )
            context = format_context(
                comp["fts"], comp["wm"], comp["trigger"], comp["pm"], comp["surprise"],
                l1_info=l1_info, l2_info=l2_info,
            )
            if not context.strip():
                return

            # --- S5-05: Active Inference Action Hint ---
            # K.0.2: Use module-level import (avoids cold import per subprocess)
            try:
                if _recommend_action:
                    rec = _recommend_action()
                else:
                    rec = None
                if rec and rec.get("recommended"):
                    context += (
                        f"\n## Active Inference Hint\n"
                        f"- Recommended: {rec['recommended']} ({rec['description']})\n"
                        f"- Rationale: {rec['rationale']}\n"
                        f"- Model: {rec['model_observations']} observations"
                    )
            except Exception:
                pass  # Never block hook

            # --- Sprint 4, item 4.4: Transparency from precision ---
            # High L2 precision → transparent (direct answer)
            # Low L2 precision → opaque (explain reasoning, hedge)
            # Clark 2016: precision determines whether predictions are
            # broadcast transparently or accompanied by meta-commentary.
            try:
                transparency = _compute_transparency(conn)
                if transparency:
                    context += f"\n## Response Style (Precision-Adaptive)\n- {transparency}"
            except Exception:
                pass  # Never block hook

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
