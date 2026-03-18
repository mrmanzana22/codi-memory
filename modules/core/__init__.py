"""
modules.core — Pure foundation layer for codi-memory.

All exports are stateless, side-effect-free, and have zero dependencies
on other modules/ code (only stdlib + instance_config).

Import either from the specific submodule or directly from core:
    from modules.core.time import now_iso
    from modules.core import now_iso  # convenience re-export
"""

# Time (highest-traffic exports — 42+ importers)
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

# Paths (15+ importers)
from modules.core.paths import (
    BASE_DIR,
    DATA_DIR,
    BACKUP_DIR,
    BACKUP_FILE,
    FTS_DB_PATH,
    PROSPECTIVE_DB_PATH,
    PID_FILE,
    ENV_PATH,
    TRIGGERS_FILE,
    MARKDOWN_DIR,
    JOURNAL_DIR,
    CURIOSIDAD_FILE,
    SESSION_STATE_FILE,
)

# Constants
from modules.core.constants import (
    IMPORTANCE_WEIGHTS,
    CATEGORY_FILE_MAP,
    RELATIONSHIP_KEYWORDS,
    RELATIONSHIP_QUERY,
    PERF_CONTRACTS,
    PERF_TOOL_CONTRACT,
    CURIOSITY_TEMPLATES,
    SESSION_STATE_MAX_AGE_HOURS,
    WORKING_MEMORY_MAX_ACTIVE,
    WM_IMPORTANCE_THRESHOLD,
    MEMORY_SEARCH_DEFAULT_LIMIT,
    CURIOSITY_STALE_DAYS,
    BACKUP_POLICY,
    BACKUP_MIN_INTERVAL_SEC,
    BACKUP_MAX_FILES,
)

# Classification
from modules.core.classification import (
    KNOWN_PROJECTS,
    TOPIC_KEYWORDS,
    TRIGGER_PRIORITY_ORDER,
    classify_topic,
)

# PAD emotion math
from modules.core.pad import (
    CODI_EMOTION_MAP,
    clamp_pad_value,
    classify_emotion,
    get_emotion_text,
    calculate_emotional_intensity,
    compute_pad_retrieval_bias,
    compute_pad_encoding_boost,
)

# Content analysis
from modules.core.analysis import (
    infer_themes,
    is_self_referential,
    calculate_confidence_score,
)
