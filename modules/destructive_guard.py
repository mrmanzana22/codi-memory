"""
Destructive Guard: server-side two-step confirmation for dangerous MCP tools.

Prevents accidental or injected execution of destructive operations
(clear_all_memories, delete_memory, delete_by_content, sync_fts_index).

Flow:
  1. First call (no token): returns preview + issues a confirm_token (UUID, TTL 60s)
  2. Second call (with token): validates tool_name + payload fingerprint, executes, invalidates token

Config:
  - CODI_DESTRUCTIVE_GUARD=off  -> bypass (dev/testing only)
  - .destructive_guard file      -> fallback if env not set
  - Default: guard ON
"""

import hashlib
import json
import os
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional


TOKEN_TTL_SECONDS = 60

_GUARD_FILE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    ".destructive_guard",
)


@dataclass
class TokenInfo:
    token: str
    tool_name: str
    fingerprint: str
    issued_at: datetime = field(default_factory=datetime.now)
    expires_at: datetime = field(default_factory=lambda: datetime.now() + timedelta(seconds=TOKEN_TTL_SECONDS))


_lock = threading.Lock()
_pending_tokens: dict[str, TokenInfo] = {}


def _purge_expired() -> None:
    """Remove expired tokens. Must be called under _lock."""
    now = datetime.now()
    expired = [k for k, v in _pending_tokens.items() if v.expires_at <= now]
    for k in expired:
        del _pending_tokens[k]


def is_guard_enabled() -> bool:
    """Check if destructive guard is active."""
    val = os.environ.get("CODI_DESTRUCTIVE_GUARD", "").strip().lower()
    if val:
        return val != "off"
    try:
        with open(_GUARD_FILE) as f:
            return f.read().strip().lower() != "off"
    except FileNotFoundError:
        return True  # default: guard ON


def is_legacy_enabled() -> bool:
    """Check if legacy confirm_code/confirm=True paths are allowed."""
    return os.environ.get("CODI_DESTRUCTIVE_LEGACY", "").strip().lower() == "on"


def compute_fingerprint(tool_name: str, **kwargs) -> str:
    """Deterministic fingerprint tying a token to a specific tool + args."""
    normalized = json.dumps(kwargs, sort_keys=True, default=str)
    raw = f"{tool_name}|{normalized}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def request_confirmation(tool_name: str, fingerprint: str) -> str:
    """Issue a single-use confirmation token for a destructive operation."""
    token = uuid.uuid4().hex[:16]
    info = TokenInfo(
        token=token,
        tool_name=tool_name,
        fingerprint=fingerprint,
    )
    with _lock:
        _purge_expired()
        _pending_tokens[token] = info
    return token


def validate_and_consume(token: str, tool_name: str, fingerprint: str) -> Optional[str]:
    """
    Validate a confirm_token. Returns None on success, error message on failure.
    Token is consumed (single-use) on success.
    """
    if not token:
        return "No confirm_token provided."

    with _lock:
        _purge_expired()
        info = _pending_tokens.get(token)
        if info is None:
            return "Token invalid or expired."
        if info.tool_name != tool_name:
            return f"Token was issued for '{info.tool_name}', not '{tool_name}'."
        if info.fingerprint != fingerprint:
            return "Token payload mismatch (arguments changed since preview)."
        del _pending_tokens[token]

    return None  # success


def log_security_event(
    event_type: str,
    tool_name: str,
    outcome: str = "executed",
    payload_fingerprint: str = None,
    details: dict = None,
) -> None:
    """Log a destructive operation to security_audit_log (best-effort).

    INSERT-only, immutable audit trail for forensics/compliance.
    """
    import logging
    _logger = logging.getLogger(__name__)
    try:
        from modules.db_pool import get_conn
        from modules.config import now_iso
        conn = get_conn()
        details_json = json.dumps(details, default=str) if details else None
        conn.execute(
            "INSERT INTO security_audit_log "
            "(event_type, tool_name, payload_fingerprint, details_json, outcome, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (event_type, tool_name, payload_fingerprint, details_json, outcome, now_iso()),
        )
        conn.commit()
    except Exception as e:
        _logger.debug("security audit log write failed: %s", e)
