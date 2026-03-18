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
    def load_dotenv(*args, **kwargs):
        """No-op fallback when python-dotenv is not installed."""
        return False

_logger = logging.getLogger(__name__)

# ============================================================
# TIMEZONE & TIMESTAMPS (Phase 3 — extracted to core/time.py)
# Re-exported here for backward compatibility.
# ============================================================
from modules.core.time import (
    TZ_COL,
    now_col,
    now_iso,
    now_display,
    now_short,
    parse_timestamp,
    parse_sqlite_utc,
    now_sqlite,
    to_sqlite_utc,
    to_sqlite_local,
)


try:
    from mcp.server.fastmcp import FastMCP
    from mcp.server.transport_security import TransportSecuritySettings
    _HAS_MCP = True
except ImportError:
    _HAS_MCP = False
# mem0 and qdrant_client are imported lazily inside get_memory() / get_qdrant()
# to avoid import failures when STORAGE_BACKEND != "legacy".
try:
    from supabase import create_client, Client
except ImportError:
    create_client = None
    Client = None

# ============================================================
# EARLY DOTENV LOAD (must be before INSTANCE CONFIG)
# Instance config reads CODI_PG_URL from env — dotenv must load first.
# ============================================================
_EARLY_ENV_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env")
load_dotenv(_EARLY_ENV_PATH)

# ============================================================
# INSTANCE CONFIG (must be before PATHS)
# ============================================================
from modules.instance_config import get_instance as _get_instance
_inst = _get_instance()

# ============================================================
# PATHS AND CONSTANTS
# ============================================================

# ============================================================
# PATHS (Phase 3 — extracted to core/paths.py)
# Re-exported here for backward compatibility.
# ============================================================
from modules.core.paths import (
    BASE_DIR, DATA_DIR, BACKUP_DIR, BACKUP_FILE,
    FTS_DB_PATH, PROSPECTIVE_DB_PATH,
    PID_FILE, ENV_PATH, TRIGGERS_FILE,
    MARKDOWN_DIR, JOURNAL_DIR, CURIOSIDAD_FILE, SESSION_STATE_FILE,
)
_INSTANCE_DATA_DIR = _inst.data_dir  # kept for local use in connect_fts()

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
# MARKDOWN_DIR, JOURNAL_DIR, CURIOSIDAD_FILE, SESSION_STATE_FILE
# imported from core/paths.py above
# SESSION_STATE_MAX_AGE_HOURS, WORKING_MEMORY_MAX_ACTIVE imported from core/constants.py above

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

USER_ID = _inst.user_id
TENANT_ID = _inst.tenant_id

# ---------------------------------------------------------------------------
# Storage backend: "legacy" (Qdrant+mem0+SQLite) or "pg" (PostgreSQL+pgvector)
# ---------------------------------------------------------------------------
STORAGE_BACKEND = os.getenv("CODI_STORAGE_BACKEND", "legacy")

# Backup policy (P1) - override core defaults with env vars
BACKUP_POLICY = os.getenv("BACKUP_POLICY", "on_demand").strip()
BACKUP_MIN_INTERVAL_SEC = int(os.getenv("BACKUP_MIN_INTERVAL_SEC", "600"))
BACKUP_MAX_FILES = int(os.getenv("BACKUP_MAX_FILES", "20"))
QDRANT_URL = os.getenv("QDRANT_URL", "").strip()
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "").strip()
COLLECTION_NAME = _inst.qdrant_collection   # Episodic store (per-tenant)
SEMANTIC_COLLECTION = _inst.qdrant_semantic  # Semantic store (per-tenant)

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

# Proposal #086: Deprecated decay constants removed (WIRING-5).
# Authoritative values live in modules/activation.py.

# Mapeo de categoría a archivo markdown
# ============================================================
# CONSTANTS, CLASSIFICATION, PERF CONTRACTS (Phase 3 — extracted to core/)
# Re-exported here for backward compatibility.
# ============================================================
from modules.core.constants import (
    IMPORTANCE_WEIGHTS, CATEGORY_FILE_MAP, RELATIONSHIP_KEYWORDS,
    RELATIONSHIP_QUERY, PERF_CONTRACTS, PERF_TOOL_CONTRACT,
    CURIOSITY_TEMPLATES, SESSION_STATE_MAX_AGE_HOURS,
    WORKING_MEMORY_MAX_ACTIVE, WM_IMPORTANCE_THRESHOLD,
    MEMORY_SEARCH_DEFAULT_LIMIT, CURIOSITY_STALE_DAYS,
    # BACKUP_POLICY, BACKUP_MIN_INTERVAL_SEC, BACKUP_MAX_FILES — kept as env-var-based above
)
from modules.core.classification import (
    KNOWN_PROJECTS, TOPIC_KEYWORDS, TRIGGER_PRIORITY_ORDER, classify_topic,
)

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
            from mem0 import Memory
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
            from qdrant_client import QdrantClient
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
# PAD emotion map (Phase 3 — extracted to core/pad.py)
from modules.core.pad import CODI_EMOTION_MAP

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
