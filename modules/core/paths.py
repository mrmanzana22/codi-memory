"""
core/paths.py — All path constants for codi-memory.

Depends only on instance_config (which has zero modules/ dependencies).
instance_config loads dotenv independently (idempotent).

No side effects — does NOT create directories (config.py handles that).
"""

import os

from modules.instance_config import get_instance as _get_instance

_inst = _get_instance()

# Project root
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))  # up from modules/core/

# Instance-aware data directory
_INSTANCE_DATA_DIR = _inst.data_dir
DATA_DIR = os.path.join(_INSTANCE_DATA_DIR, "data")

# Backup
BACKUP_DIR = _INSTANCE_DATA_DIR
BACKUP_FILE = os.path.join(BACKUP_DIR, "memories_backup.json")

# Database paths
FTS_DB_PATH = os.path.join(_INSTANCE_DATA_DIR, "memories_fts.db")
PROSPECTIVE_DB_PATH = os.path.join(_INSTANCE_DATA_DIR, "prospective.db")

# Shared config files
PID_FILE = os.path.join(BASE_DIR, ".codi-memory.pid")
ENV_PATH = os.path.join(BASE_DIR, ".env")
TRIGGERS_FILE = os.path.join(BASE_DIR, "triggers.json")

# Content directories
MARKDOWN_DIR = os.path.join(BASE_DIR, "markdown")
JOURNAL_DIR = os.path.join(MARKDOWN_DIR, "journal")

# Data files
CURIOSIDAD_FILE = os.path.join(BASE_DIR, "preguntas_curiosidad.json")
SESSION_STATE_FILE = os.path.join(DATA_DIR, "session_state.json")
