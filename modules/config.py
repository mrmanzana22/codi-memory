"""
Codi Memory - Shared configuration, constants, state, and initialization.
All modules import shared state from here.
"""

import logging
import os
import json
import math
import signal
import sqlite3
from datetime import datetime, timezone, timedelta
from zoneinfo import ZoneInfo
try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = lambda: None  # No-op if dotenv not available

_logger = logging.getLogger(__name__)

# ============================================================
# TIMEZONE: Colombia (America/Bogota, UTC-5)
# ============================================================
TZ_COL = ZoneInfo("America/Bogota")

def now_col() -> datetime:
    """Retorna datetime actual en timezone Colombia."""
    return datetime.now(TZ_COL)

def now_iso() -> str:
    """Retorna timestamp ISO 8601 con timezone Colombia."""
    return now_col().isoformat()

def now_display() -> str:
    """Retorna timestamp legible con zona: '2026-02-07 07:30 COT'"""
    return now_col().strftime("%Y-%m-%d %H:%M COT")

def now_short() -> str:
    """Retorna timestamp corto: '2026-02-07 07:30'"""
    return now_col().strftime("%Y-%m-%d %H:%M")
try:
    from mcp.server.fastmcp import FastMCP
    from mcp.server.transport_security import TransportSecuritySettings
    _HAS_MCP = True
except ImportError:
    _HAS_MCP = False
from mem0 import Memory
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue, Range
from supabase import create_client, Client

# ============================================================
# PATHS AND CONSTANTS
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(__file__))  # /Users/harecjimenez/codi-memory
PID_FILE = os.path.join(BASE_DIR, ".codi-memory.pid")
ENV_PATH = os.path.join(BASE_DIR, ".env")
DATA_DIR = os.path.join(BASE_DIR, "data")
BACKUP_DIR = BASE_DIR
BACKUP_FILE = os.path.join(BACKUP_DIR, "memories_backup.json")
TRIGGERS_FILE = os.path.join(BASE_DIR, "triggers.json")
FTS_DB_PATH = os.path.join(BASE_DIR, "memories_fts.db")
PROSPECTIVE_DB_PATH = os.path.join(os.path.dirname(FTS_DB_PATH), "prospective.db")

# ---------------------------------------------------------------------------
# Centralized SQLite connection factory
# ---------------------------------------------------------------------------
_SQLITE_TIMEOUT = 30          # seconds to wait for lock
_SQLITE_BUSY_TIMEOUT = 30000  # ms — PRAGMA busy_timeout


class _PooledConn:
    """Wraps a pooled SQLite connection so .close() never kills the pool conn.

    With autocommit mode (isolation_level=None) in the pool, there are no
    implicit transactions to commit. close() is a true no-op.
    """
    __slots__ = ('_real',)

    def __init__(self, real: sqlite3.Connection):
        object.__setattr__(self, '_real', real)

    def close(self):
        pass  # Pool manages lifecycle; autocommit handles writes

    def __getattr__(self, name):
        return getattr(self._real, name)

    def __setattr__(self, name, value):
        if name == '_real':
            object.__setattr__(self, name, value)
        else:
            setattr(self._real, name, value)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass  # No-op with autocommit


def connect_fts(db_path: str = None) -> sqlite3.Connection:
    """Get a thread-local pooled SQLite connection with WAL and autocommit.

    Returns a pooled connection wrapped so that .close() is a no-op.
    All callers share the same connection per thread — no more leaks.
    Falls back to a fresh connection if the pool import fails.
    """
    try:
        from modules.db_pool import get_conn
        return _PooledConn(get_conn(db_path or FTS_DB_PATH))
    except Exception:
        # Fallback: raw connection (during early init or circular import)
        conn = sqlite3.connect(
            db_path or FTS_DB_PATH,
            timeout=_SQLITE_TIMEOUT,
            isolation_level=None,  # AUTOCOMMIT — match pool behavior
        )
        conn.execute(f"PRAGMA busy_timeout={_SQLITE_BUSY_TIMEOUT}")
        conn.execute("PRAGMA journal_mode=WAL")
        return conn
MARKDOWN_DIR = os.path.join(BASE_DIR, "markdown")
JOURNAL_DIR = os.path.join(MARKDOWN_DIR, "journal")
CURIOSIDAD_FILE = os.path.join(BASE_DIR, "preguntas_curiosidad.json")
SESSION_STATE_FILE = os.path.join(DATA_DIR, "session_state.json")
SESSION_STATE_MAX_AGE_HOURS = 24  # Ignore session state older than this

# Working Memory limits
WORKING_MEMORY_MAX_ACTIVE = 30

# Spreading Activation (Fase 3)
SPREAD_DEFAULT_FACTOR = 0.7
SPREAD_DEFAULT_DEPTH = 2
SPREAD_MIN_ACTIVATION = 0.05
SPREAD_MAX_NEIGHBORS = 15
SPREAD_SALIENCE_CAP = 1.0
SPREAD_SALIENCE_FLOOR = 0.1

# --- Graph Densification (Phase 5.5) ---
GRAPH_AUTO_CONNECT_K = 5          # Top-K candidates to evaluate
GRAPH_AUTO_CONNECT_MAX = 3        # Max connections to create per memory
GRAPH_AUTO_CONNECT_MIN_SCORE = 0.5  # Minimum similarity to connect

os.makedirs(DATA_DIR, exist_ok=True)

# Cargar variables de entorno
load_dotenv(ENV_PATH)

USER_ID = os.getenv("USER_ID", "hare")

# ---------------------------------------------------------------------------
# Storage backend: "legacy" (Qdrant+mem0+SQLite) or "pg" (PostgreSQL+pgvector)
# ---------------------------------------------------------------------------
STORAGE_BACKEND = os.getenv("CODI_STORAGE_BACKEND", "legacy")

# Backup policy (P1) - must be after load_dotenv
BACKUP_POLICY = os.getenv("BACKUP_POLICY", "on_demand").strip()  # "on_demand" | "always"
BACKUP_MIN_INTERVAL_SEC = int(os.getenv("BACKUP_MIN_INTERVAL_SEC", "600"))  # 10 min debounce
BACKUP_MAX_FILES = int(os.getenv("BACKUP_MAX_FILES", "20"))  # rotation cap
QDRANT_URL = os.getenv("QDRANT_URL", "").strip()
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "").strip()
COLLECTION_NAME = "codi_memories"          # Episodic store (existing)
SEMANTIC_COLLECTION = "codi_semantic"       # Semantic store (Phase 1)

# ============================================================
# PHASE 1: CONSOLIDATION PARAMETERS
# ============================================================
CONSOLIDATION_CLUSTER_MIN_SIZE = 3          # Min episodes to form a pattern
CONSOLIDATION_SIMILARITY_THRESHOLD = 0.65   # Cosine sim threshold for clustering
CONSOLIDATION_SEMANTIC_DEDUP_THRESHOLD = 0.85  # Dedup threshold for semantic facts
CONSOLIDATION_MAX_EPISODES_PER_RUN = 200    # Cap per consolidation run

# ============================================================
# PHASE 6: COMPRESSION PARAMETERS
# ============================================================
COMPRESSION_MIN_AGE_DAYS = 7
COMPRESSION_MIN_GROUP_SIZE = 3
COMPRESSION_MAX_PER_RUN = 100
COMPRESSION_SIMILARITY_THRESHOLD = 0.60
COMPRESSION_ENABLED = True

# ============================================================
# PHASE 1: RECONSOLIDATION PARAMETERS
# ============================================================
RECONSOLIDATION_WINDOW_HOURS = 1.0          # Lability window duration
RECONSOLIDATION_PE_THRESHOLD = 0.3          # Min prediction error to trigger
RECONSOLIDATION_STRENGTH_FLOOR = 0.15       # Too weak to reconsolidate
RECONSOLIDATION_STRENGTH_CEILING = 0.90     # Too strong to reconsolidate
RECONSOLIDATION_MAX_BLEND = 0.3             # Max 30% new content blends in

# ============================================================
# PHASE 5: INLINE CONTRADICTION DETECTION (Kumaran & Maguire 2007)
# ============================================================
CONTRADICTION_SCORE_FLOOR = 0.35            # Min similarity to compare (lowered from 0.50: contradictions often have low cosine sim)
CONTRADICTION_PE_SILENT = 0.20              # PE >= this: silent WM note (lowered from 0.30: still needs 2-channel convergence)
CONTRADICTION_PE_ALERT = 0.50               # PE >= this: visible alert + mark labile
CONTRADICTION_PE_CRITICAL = 0.70            # PE >= this: critical alert, LLM should ask user
CONTRADICTION_MIN_ENTITIES = 1              # Min shared entities to compare (lowered from 2: short memories rarely share 2+)
CONTRADICTION_COOLDOWN_MINUTES = 30         # Same-topic cooldown
CONTRADICTION_MAX_PER_SESSION = 3           # Max alerts before silent-only mode

# ============================================================
# PHASE 1: DIFFERENTIAL DECAY PARAMETERS
# DEPRECATED (WIRING-5): Authoritative decay constants now live in
# modules/activation.py (DECAY_EPISODIC, DECAY_SEMANTIC, etc.).
# These are kept ONLY for backward compat; no code imports them.
# ============================================================
EPISODIC_DECAY_BASE = 0.5                   # DEPRECATED -> activation.py DECAY_EPISODIC=0.40
EPISODIC_DECAY_CRITICAL = 0.2               # DEPRECATED -> activation.py DECAY_EPISODIC_CRITICAL=0.20
EPISODIC_DECAY_HIGH = 0.35                  # DEPRECATED -> activation.py DECAY_EPISODIC_HIGH=0.30
EPISODIC_DECAY_EMOTIONAL = 0.25             # DEPRECATED -> activation.py DECAY_EPISODIC_EMOTIONAL=0.25
SEMANTIC_DECAY_BASE = 0.15                  # DEPRECATED -> activation.py DECAY_SEMANTIC=0.15
SEMANTIC_DECAY_CRITICAL = 0.05              # DEPRECATED -> activation.py DECAY_SEMANTIC_MIN=0.05
SEMANTIC_EVIDENCE_BOOST = 0.02              # DEPRECATED -> activation.py EVIDENCE_DECAY_REDUCTION=0.02
EPISODIC_PRUNE_THRESHOLD = 0.05             # Below this activation = prune candidate
EPISODIC_PRUNE_MIN_AGE_DAYS = 30            # Don't prune anything younger

# Mapeo de categoría a archivo markdown
CATEGORY_FILE_MAP = {
    'identidad': 'SOUL.md',
    'proyecto': 'PROJECTS.md',
    'aprendizaje': 'LEARNINGS.md',
    'episodio': 'EPISODES.md',
    'general': 'GENERAL.md',
    'checkpoint': 'GENERAL.md',
}

RELATIONSHIP_KEYWORDS = ['andre', 'harec', 'hijo', 'esposa', 'familia', 'papa', 'mamá', 'mama']

# ============================================================
# KNOWN PROJECTS & TOPIC KEYWORDS (single source of truth)
# ============================================================
KNOWN_PROJECTS = ["trading", "fullempaques", "consciencia", "n8n", "kraken", "memoria", "pilas", "portal-aliados-mrmanzana"]

TOPIC_KEYWORDS = {
    'n8n': ['n8n', 'workflow', 'automatiz', 'nodo'],
    'trading': ['trading', 'kraken', 'cripto', 'bitcoin', 'mercado'],
    'fullempaques': ['fullempaques', 'produccion', 'fabrica', 'empaque'],
    'memoria': ['memoria', 'recuerdo', 'recordar', 'qdrant'],
    'codigo': ['codigo', 'python', 'javascript', 'programar', 'server.py'],
    'proyecto': ['proyecto', 'implementar', 'desarrollar', 'feature'],
    'configuracion': ['config', 'variable', 'entorno', 'setup', 'easypanel'],
    'consciencia': [
        'consciencia', 'consciente', 'self-model', 'prediccion',
        'consciousness', 'awareness', 'metacognicion', 'metacognition',
        'self_model', 'sleep_loop', 'reconsolidacion', 'reconsolidation',
        'preturn', 'butlin', 'gwt', 'gnw', 'fok_calibration',
    ],
}

TRIGGER_PRIORITY_ORDER = ['proyecto_nuevo', 'fullempaques', 'automatizacion', 'trading', 'mi_entrenamiento']


def classify_topic(text: str) -> str:
    """Classify text into a known topic using keyword matching.

    Loewenstein 1994: Curiosity is driven by information gaps in specific
    DOMAINS, not random words. Proper classification ensures gaps are
    meaningful and actionable.

    Returns topic string or 'general' if no match.
    """
    text_lower = text.lower()
    scores = {}
    for topic, keywords in TOPIC_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw in text_lower)
        if score > 0:
            scores[topic] = score
    if not scores:
        return "general"
    return max(scores, key=scores.get)

# ============================================================
# PERFORMANCE CONTRACTS - p95/p99 latency budgets (ms)
# ============================================================
PERF_CONTRACTS = {
    "macro": {"p95": 2000, "p99": 5000, "tools": ["recall", "remember", "context_snapshot"]},
    "search": {"p95": 1500, "p99": 3000, "tools": ["search_memory", "search_by_theme", "search_by_ownership", "search_by_emotion"]},
    "write": {"p95": 1500, "p99": 3000, "tools": ["add_memory", "add_memory_smart"]},
    "fast": {"p95": 200, "p99": 500, "tools": ["get_emotional_state", "get_working_memory", "get_workspace_state", "listar_triggers", "audit_tools"]},
    "consolidation": {"p95": 5000, "p99": 10000, "tools": ["run_consolidation", "dream_consolidation", "consolidate_recent"]},
    "default": {"p95": 1000, "p99": 3000, "tools": []},
}

# Build reverse lookup: tool_name -> contract category
PERF_TOOL_CONTRACT = {}
for _cat, _spec in PERF_CONTRACTS.items():
    for _tool in _spec["tools"]:
        PERF_TOOL_CONTRACT[_tool] = _cat

CURIOSITY_TEMPLATES = {
    'trading': "No hemos revisado el trading en {dias} dias. Como van las senales?",
    'fullempaques': "FULLEMPAQUES lleva {dias} dias sin tocar. El cliente reporto algun problema?",
    'consciencia': "El proyecto de consciencia lleva {dias} dias pausado. Retomamos?",
    'n8n': "No hemos tocado automatizaciones n8n en {dias} dias. Hay workflows que revisar?",
}

# ============================================================
# IMPORTANCE WEIGHTS (single source of truth)
# ============================================================
IMPORTANCE_WEIGHTS = {'critical': 1.0, 'high': 0.8, 'medium': 0.5, 'low': 0.2}

# ============================================================
# OPERATIONAL THRESHOLDS
# ============================================================
CURIOSITY_STALE_DAYS = 3                     # Days before a project is "stale"
WM_IMPORTANCE_THRESHOLD = 0.7                # Min importance to push to WM
MEMORY_SEARCH_DEFAULT_LIMIT = 10             # Default limit for memory searches
RELATIONSHIP_QUERY = ' '.join(RELATIONSHIP_KEYWORDS[:5])  # Dynamic query from config

# ============================================================
# AUTO-CLEANUP: Matar instancias anteriores del MCP
# ============================================================

def register_pid():
    """Registra el PID actual para debugging. No mata instancias anteriores."""
    try:
        with open(PID_FILE, 'w') as f:
            f.write(str(os.getpid()))
        _logger.info("Instancia iniciada (PID %d)", os.getpid())
    except Exception:
        pass

register_pid()

# ============================================================
# MEM0 CONFIGURATION
# ============================================================

mem0_config = {
    "version": "v1.1",
    "llm": {
        "provider": "openai",
        "config": {
            "model": "gpt-4o-mini",
            "temperature": 0.1,
        }
    },
    "embedder": {
        "provider": "openai",
        "config": {
            "model": "text-embedding-3-small"
        }
    },
    "vector_store": {
        "provider": "qdrant",
        "config": {
            "collection_name": COLLECTION_NAME,
            "url": QDRANT_URL,
            **({"api_key": QDRANT_API_KEY} if QDRANT_API_KEY else {}),
        }
    }
}

# ============================================================
# LAZY INITIALIZATION
# ============================================================

_memory = None
_qdrant = None
_init_error = None


def get_memory():
    """Lazy init de mem0."""
    global _memory, _init_error
    if _memory is None:
        try:
            _memory = Memory.from_config(mem0_config)
            _init_error = None
            _logger.info("mem0 conectado OK")
        except Exception as e:
            _init_error = str(e)
            from modules.secret_redact import redact_secrets
            _logger.error("ERROR conectando mem0: %s", redact_secrets(str(e)))
            raise
    return _memory


def _sanitize_url(url: str) -> str:
    """Return scheme://host:port only (strip path, query, creds)."""
    try:
        from urllib.parse import urlparse
        p = urlparse(url)
        host = p.hostname or "unknown"
        port = f":{p.port}" if p.port else ""
        return f"{p.scheme}://{host}{port}"
    except Exception:
        return "<redacted>"


def _is_remote_qdrant(url: str) -> bool:
    """True if URL points to a non-localhost host."""
    try:
        from urllib.parse import urlparse
        host = (urlparse(url).hostname or "").lower()
        return host not in ("localhost", "127.0.0.1", "::1", "")
    except Exception:
        return True


def get_qdrant():
    """Lazy init de Qdrant with API key auth and remote guardrail."""
    global _qdrant
    if _qdrant is None:
        if not QDRANT_URL:
            raise RuntimeError(
                "QDRANT_URL no esta configurada. "
                "Configurala en .env (ej: QDRANT_URL=https://<host>:443)."
            )

        # Guardrail: remote Qdrant without API key is dangerous
        if _is_remote_qdrant(QDRANT_URL) and not QDRANT_API_KEY:
            allow_insecure = os.getenv("CODI_ALLOW_INSECURE_QDRANT", "").strip()
            if allow_insecure != "1":
                raise RuntimeError(
                    f"Remote Qdrant ({_sanitize_url(QDRANT_URL)}) requires QDRANT_API_KEY. "
                    "Set CODI_ALLOW_INSECURE_QDRANT=1 to bypass (dev only)."
                )

        try:
            _qdrant = QdrantClient(
                url=QDRANT_URL,
                api_key=QDRANT_API_KEY or None,
                timeout=30,
            )
            safe_url = _sanitize_url(QDRANT_URL)
            auth_status = "with API key" if QDRANT_API_KEY else "WITHOUT auth"
            _logger.info("Qdrant conectado OK (%s, %s)", safe_url, auth_status)
        except Exception as e:
            _logger.error("ERROR conectando Qdrant: %s", type(e).__name__)
            raise
    return _qdrant


class _LazyMemory:
    """Proxy que inicializa mem0 en el primer uso."""
    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        return getattr(get_memory(), name)


class _LazyQdrant:
    """Proxy que inicializa Qdrant en el primer uso."""
    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        return getattr(get_qdrant(), name)


memory = _LazyMemory()
qdrant = _LazyQdrant()

# ============================================================
# SUPABASE CLIENT (guarded — PR3 C-01)
# ============================================================
# Supabase is only used by training.py for training_examples.
# Guard with CODI_SUPABASE_ENABLED to prevent accidental init
# in environments where the key shouldn't be active.

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY") or ""
SUPABASE_KEY_1 = os.getenv("SUPABASE_KEY_1") or ""
SUPABASE_KEY_2 = os.getenv("SUPABASE_KEY_2") or ""
if SUPABASE_KEY_1 and SUPABASE_KEY_2:
    SUPABASE_KEY = SUPABASE_KEY_1 + SUPABASE_KEY_2

CODI_SUPABASE_ENABLED = os.getenv("CODI_SUPABASE_ENABLED", "1").strip()

supabase: Client = None
if SUPABASE_URL and SUPABASE_KEY:
    if CODI_SUPABASE_ENABLED == "1":
        try:
            supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
            _logger.info("Supabase conectado para training examples")
        except Exception as e:
            from modules.secret_redact import redact_secrets as _redact
            _logger.warning("Supabase no disponible: %s", _redact(str(e)))
    else:
        _logger.info("Supabase desactivado (CODI_SUPABASE_ENABLED != 1)")

# ============================================================
# MCP SERVER
# ============================================================

if _HAS_MCP:
    mcp = FastMCP(
        "codi-memory",
        transport_security=TransportSecuritySettings(
            enable_dns_rebinding_protection=False,
        )
    )
else:
    mcp = None  # Tests/scripts without mcp installed

# ============================================================
# SHARED IN-MEMORY STATE
# ============================================================

_current_session = now_col().strftime("%Y-%m-%d") + "-001"

# PAD MODEL - Estado Emocional (Pleasure-Arousal-Dominance)
CODI_EMOTION_MAP = {
    'exuberant': 'emocionado y energizado',
    'dependent': 'entusiasmado pero necesitando apoyo',
    'relaxed': 'satisfecho y tranquilo',
    'docile': 'calmado y receptivo',
    'hostile': 'frustrado e irritado',
    'anxious': 'ansioso e inquieto',
    'disdainful': 'desinteresado',
    'bored': 'apagado y sin energia'
}

_emotional_state = {
    'current': {
        'pleasure': 0.0,
        'arousal': 0.0,
        'dominance': 0.0,
        'timestamp': None,
        'trigger': None
    },
    'mood': {
        'pleasure': 0.2,
        'arousal': 0.1,
        'dominance': 0.3,
        'last_updated': None
    },
    'history': [],
    'decay_rate': 0.1,
    'mood_shift_rate': 0.05
}
