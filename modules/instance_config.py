"""
instance_config.py - Per-tenant configuration for multi-instance Codi.
======================================================================
P2 Instance Isolation.

Each Codi instance (Hare, Sebastian, ...) gets its own:
  - PostgreSQL database (DB-level isolation, zero query changes)
  - SQLite data directory (file-level isolation)
  - Qdrant collections (collection-level isolation)
  - Daemon port + launchd plists

Config cascade:
  1. CODI_INSTANCE_CONFIG env var → path to YAML file
  2. No env var → _default_hare() (exact current behavior)

YAML values support ${ENV_VAR} interpolation for secrets.
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass
from typing import Optional

# Load .env EARLY so CODI_PG_URL is available when _default_hare() reads os.getenv
try:
    from dotenv import load_dotenv as _load_dotenv
    _load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"))
except ImportError:
    pass

_logger = logging.getLogger(__name__)

# Singleton cache
_instance: Optional["InstanceConfig"] = None


@dataclass(frozen=True)
class InstanceConfig:
    """Per-tenant configuration. Immutable after creation."""

    tenant_id: str          # "hare", "sebastian"
    display_name: str       # "Hare", "Sebastian"
    user_id: str            # mem0 user_id
    pg_url: str             # per-tenant PG connection string
    daemon_port: int        # 8420, 8421
    data_dir: str           # per-tenant SQLite/data root dir
    qdrant_collection: str  # "codi_memories", "seb_memories"
    qdrant_semantic: str    # "codi_semantic", "seb_semantic"
    log_dir: str            # per-tenant log dir
    plist_prefix: str       # "com.codi", "com.seb"
    timezone: str = "America/Bogota"


def get_instance() -> InstanceConfig:
    """Get the current instance config (lazy singleton).

    Resolution order:
      1. CODI_INSTANCE_CONFIG env var → YAML file path
      2. Fallback → _default_hare() (exact backward compat)
    """
    global _instance
    if _instance is not None:
        return _instance

    config_path = os.getenv("CODI_INSTANCE_CONFIG", "").strip()
    if config_path:
        _instance = _load_yaml(config_path)
        _logger.info(
            "Instance config loaded: tenant=%s from %s",
            _instance.tenant_id,
            config_path,
        )
    else:
        _instance = _default_hare()
        _logger.debug("Using default Hare instance config (no CODI_INSTANCE_CONFIG)")

    return _instance


def _default_hare() -> InstanceConfig:
    """Exact current behavior: Hare's config from env vars / hardcoded defaults."""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return InstanceConfig(
        tenant_id="hare",
        display_name="Hare",
        user_id=os.getenv("USER_ID", "hare"),
        pg_url=os.getenv(
            "CODI_PG_URL",
            "postgresql://codi:c0d1m3m0ry2026pgv@lx6zon.easypanel.host:5433/codi_memory",
        ),
        daemon_port=int(os.getenv("DAEMON_PORT", "8420")),
        data_dir=base_dir,
        qdrant_collection="codi_memories",
        qdrant_semantic="codi_semantic",
        log_dir=os.path.join(os.path.expanduser("~"), ".codi-daemon", "logs"),
        plist_prefix="com.codi",
        timezone="America/Bogota",
    )


# ---------------------------------------------------------------------------
# YAML loader with env var interpolation
# ---------------------------------------------------------------------------
_ENV_PATTERN = re.compile(r"\$\{(\w+)\}")


def _interpolate_env(value: str) -> str:
    """Replace ${ENV_VAR} with its value. Raise if unset."""
    def _replace(match):
        var_name = match.group(1)
        env_val = os.getenv(var_name)
        if env_val is None:
            raise ValueError(
                f"Environment variable ${{{var_name}}} is required but not set"
            )
        return env_val

    if not isinstance(value, str):
        return value
    return _ENV_PATTERN.sub(_replace, value)


def _load_yaml(path: str) -> InstanceConfig:
    """Load and validate a YAML instance config file."""
    import yaml

    if not os.path.isfile(path):
        raise FileNotFoundError(f"Instance config not found: {path}")

    with open(path, "r") as f:
        raw = yaml.safe_load(f)

    if not isinstance(raw, dict):
        raise ValueError(f"Instance config must be a YAML mapping, got {type(raw)}")

    # Interpolate env vars in string values
    cfg = {}
    for k, v in raw.items():
        if isinstance(v, str):
            cfg[k] = _interpolate_env(v)
        else:
            cfg[k] = v

    _validate_config(cfg)

    return InstanceConfig(
        tenant_id=cfg["tenant_id"],
        display_name=cfg.get("display_name", cfg["tenant_id"].capitalize()),
        user_id=cfg.get("user_id", cfg["tenant_id"]),
        pg_url=cfg["pg_url"],
        daemon_port=int(cfg.get("daemon_port", 8420)),
        data_dir=cfg["data_dir"],
        qdrant_collection=cfg["qdrant_collection"],
        qdrant_semantic=cfg["qdrant_semantic"],
        log_dir=cfg.get("log_dir", ""),
        plist_prefix=cfg.get("plist_prefix", f"com.{cfg['tenant_id']}"),
        timezone=cfg.get("timezone", "America/Bogota"),
    )


def _validate_config(cfg: dict) -> None:
    """Fail fast on invalid config. Raises ValueError."""
    required = ["tenant_id", "pg_url", "data_dir", "qdrant_collection", "qdrant_semantic"]
    missing = [k for k in required if not cfg.get(k)]
    if missing:
        raise ValueError(f"Missing required config fields: {missing}")

    # tenant_id must be alphanumeric + underscore
    tid = cfg["tenant_id"]
    if not re.match(r"^[a-z][a-z0-9_]*$", tid):
        raise ValueError(
            f"tenant_id must be lowercase alphanumeric (got '{tid}')"
        )

    # Port range
    port = int(cfg.get("daemon_port", 8420))
    if not (1024 <= port <= 65535):
        raise ValueError(f"daemon_port must be 1024-65535 (got {port})")

    # Paths must be absolute
    for path_key in ("data_dir",):
        val = cfg.get(path_key, "")
        if val and not os.path.isabs(val):
            raise ValueError(f"{path_key} must be an absolute path (got '{val}')")

    # Collections must not be empty
    for coll_key in ("qdrant_collection", "qdrant_semantic"):
        if not cfg.get(coll_key, "").strip():
            raise ValueError(f"{coll_key} must not be empty")

    # pg_url basic check
    pg_url = cfg.get("pg_url", "")
    if not pg_url.startswith("postgresql://") and not pg_url.startswith("postgres://"):
        raise ValueError(f"pg_url must start with postgresql:// (got '{pg_url[:20]}...')")


def ensure_instance_dirs() -> None:
    """Create data_dir and log_dir for the current instance if they don't exist."""
    inst = get_instance()
    for d in (inst.data_dir, inst.log_dir):
        if d:
            os.makedirs(d, exist_ok=True)


def reset_instance() -> None:
    """Reset the singleton (for testing only)."""
    global _instance
    _instance = None
