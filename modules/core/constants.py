"""
core/constants.py — Operational constants for codi-memory.

Pure values with zero dependencies. No mutable state.
Domain-specific constants (CONSOLIDATION_*, SPREAD_*, etc.) stay in config.py.
"""

import os

# Importance weights (single source of truth)
IMPORTANCE_WEIGHTS = {"critical": 1.0, "high": 0.8, "medium": 0.5, "low": 0.2}

# Category → markdown file mapping
CATEGORY_FILE_MAP = {
    "identidad": "SOUL.md",
    "proyecto": "PROJECTS.md",
    "aprendizaje": "LEARNINGS.md",
    "episodio": "EPISODES.md",
    "general": "GENERAL.md",
    "checkpoint": "GENERAL.md",
}

# Relationship detection
RELATIONSHIP_KEYWORDS = [
    "andre", "harec", "hijo", "esposa", "familia", "papa", "mamá", "mama",
]
RELATIONSHIP_QUERY = " ".join(RELATIONSHIP_KEYWORDS[:5])

# Performance contracts — p95/p99 latency budgets (ms)
PERF_CONTRACTS = {
    "macro": {"p95": 2000, "p99": 5000, "tools": ["recall", "remember", "context_snapshot"]},
    "search": {"p95": 1500, "p99": 3000, "tools": ["search_memory", "search_by_theme", "search_by_ownership", "search_by_emotion"]},
    "write": {"p95": 1500, "p99": 3000, "tools": ["add_memory", "add_memory_smart"]},
    "fast": {"p95": 200, "p99": 500, "tools": ["get_emotional_state", "get_working_memory", "get_workspace_state", "listar_triggers", "audit_tools"]},
    "consolidation": {"p95": 5000, "p99": 10000, "tools": ["run_consolidation", "dream_consolidation", "consolidate_recent"]},
    "default": {"p95": 1000, "p99": 3000, "tools": []},
}

# Reverse lookup: tool_name → contract category
PERF_TOOL_CONTRACT = {}
for _cat, _spec in PERF_CONTRACTS.items():
    for _tool in _spec["tools"]:
        PERF_TOOL_CONTRACT[_tool] = _cat

# Curiosity templates
CURIOSITY_TEMPLATES = {
    "trading": "No hemos revisado el trading en {dias} dias. Como van las senales?",
    "fullempaques": "FULLEMPAQUES lleva {dias} dias sin tocar. El cliente reporto algun problema?",
    "consciencia": "El proyecto de consciencia lleva {dias} dias pausado. Retomamos?",
    "n8n": "No hemos tocado automatizaciones n8n en {dias} dias. Hay workflows que revisar?",
}

# Operational thresholds
SESSION_STATE_MAX_AGE_HOURS = 24
WORKING_MEMORY_MAX_ACTIVE = 30
WM_IMPORTANCE_THRESHOLD = 0.7
MEMORY_SEARCH_DEFAULT_LIMIT = 10
CURIOSITY_STALE_DAYS = 3

# Backup policy — raw defaults.
# config.py overrides these with os.getenv() for env-var support.
BACKUP_POLICY = "on_demand"
BACKUP_MIN_INTERVAL_SEC = 600
BACKUP_MAX_FILES = 20
