"""
Codi Memory - Shared configuration, constants, state, and initialization.
All modules import shared state from here.
"""

import os
import json
import math
import signal
import sqlite3
from datetime import datetime, timezone, timedelta
from zoneinfo import ZoneInfo
from dotenv import load_dotenv

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
from mcp.server.fastmcp import FastMCP
from mcp.server.transport_security import TransportSecuritySettings
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
MARKDOWN_DIR = os.path.join(BASE_DIR, "markdown")
JOURNAL_DIR = os.path.join(MARKDOWN_DIR, "journal")
CURIOSIDAD_FILE = os.path.join(BASE_DIR, "preguntas_curiosidad.json")

# Working Memory limits
WORKING_MEMORY_MAX_ACTIVE = 30

# Spreading Activation (Fase 3)
SPREAD_DEFAULT_FACTOR = 0.7
SPREAD_DEFAULT_DEPTH = 2
SPREAD_MIN_ACTIVATION = 0.05
SPREAD_MAX_NEIGHBORS = 15
SPREAD_SALIENCE_CAP = 1.0
SPREAD_SALIENCE_FLOOR = 0.1

os.makedirs(DATA_DIR, exist_ok=True)

# Cargar variables de entorno
load_dotenv(ENV_PATH)

USER_ID = os.getenv("USER_ID", "hare")

# Backup policy (P1) - must be after load_dotenv
BACKUP_POLICY = os.getenv("BACKUP_POLICY", "on_demand").strip()  # "on_demand" | "always"
BACKUP_MIN_INTERVAL_SEC = int(os.getenv("BACKUP_MIN_INTERVAL_SEC", "600"))  # 10 min debounce
BACKUP_MAX_FILES = int(os.getenv("BACKUP_MAX_FILES", "20"))  # rotation cap
QDRANT_URL = os.getenv("QDRANT_URL", "").strip()
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
# PHASE 1: RECONSOLIDATION PARAMETERS
# ============================================================
RECONSOLIDATION_WINDOW_HOURS = 1.0          # Lability window duration
RECONSOLIDATION_PE_THRESHOLD = 0.3          # Min prediction error to trigger
RECONSOLIDATION_STRENGTH_FLOOR = 0.15       # Too weak to reconsolidate
RECONSOLIDATION_STRENGTH_CEILING = 0.90     # Too strong to reconsolidate
RECONSOLIDATION_MAX_BLEND = 0.3             # Max 30% new content blends in

# ============================================================
# PHASE 1: DIFFERENTIAL DECAY PARAMETERS
# ============================================================
EPISODIC_DECAY_BASE = 0.5                   # Standard Anderson decay
EPISODIC_DECAY_CRITICAL = 0.2               # 2.5x slower for critical
EPISODIC_DECAY_HIGH = 0.35                  # 1.4x slower for high importance
EPISODIC_DECAY_EMOTIONAL = 0.25             # 2x slower for high arousal
SEMANTIC_DECAY_BASE = 0.15                  # 3.3x slower than episodic
SEMANTIC_DECAY_CRITICAL = 0.05              # Near permanent
SEMANTIC_EVIDENCE_BOOST = 0.02              # Each evidence reduces d by 0.02
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
# AUTO-CLEANUP: Matar instancias anteriores del MCP
# ============================================================

def register_pid():
    """Registra el PID actual para debugging. No mata instancias anteriores."""
    try:
        with open(PID_FILE, 'w') as f:
            f.write(str(os.getpid()))
        print(f"[codi-memory] Instancia iniciada (PID {os.getpid()})")
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
            print("[codi-memory] mem0 conectado OK")
        except Exception as e:
            _init_error = str(e)
            print(f"[codi-memory] ERROR conectando mem0: {e}")
            raise
    return _memory


def get_qdrant():
    """Lazy init de Qdrant."""
    global _qdrant
    if _qdrant is None:
        if not QDRANT_URL:
            raise RuntimeError(
                "QDRANT_URL no esta configurada. "
                "Configurala en .env (ej: QDRANT_URL=https://<host>:443)."
            )
        try:
            _qdrant = QdrantClient(url=QDRANT_URL, timeout=30)
            print("[codi-memory] Qdrant conectado OK")
        except Exception as e:
            print(f"[codi-memory] ERROR conectando Qdrant: {e}")
            raise
    return _qdrant


class _LazyMemory:
    """Proxy que inicializa mem0 en el primer uso."""
    def __getattr__(self, name):
        return getattr(get_memory(), name)


class _LazyQdrant:
    """Proxy que inicializa Qdrant en el primer uso."""
    def __getattr__(self, name):
        return getattr(get_qdrant(), name)


memory = _LazyMemory()
qdrant = _LazyQdrant()

# ============================================================
# SUPABASE CLIENT
# ============================================================

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY") or ""
SUPABASE_KEY_1 = os.getenv("SUPABASE_KEY_1") or ""
SUPABASE_KEY_2 = os.getenv("SUPABASE_KEY_2") or ""
if SUPABASE_KEY_1 and SUPABASE_KEY_2:
    SUPABASE_KEY = SUPABASE_KEY_1 + SUPABASE_KEY_2

supabase: Client = None
if SUPABASE_URL and SUPABASE_KEY:
    try:
        supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
        print("[codi-memory] Supabase conectado para training examples")
    except Exception as e:
        print(f"[codi-memory] Warning: Supabase no disponible: {e}")

# ============================================================
# MCP SERVER
# ============================================================

mcp = FastMCP(
    "codi-memory",
    transport_security=TransportSecuritySettings(
        enable_dns_rebinding_protection=False,
    )
)

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
