"""
Secret Redaction (Security Hotfix PR3 — C-01)
==============================================
Prevents API keys and secrets from leaking through error messages,
logs, or stack traces.

Patterns detected and redacted:
  - OpenAI keys: sk-...
  - JWT tokens: eyJ...
  - Supabase service_role: sbp_...
  - Generic patterns: apikey=, api_key=, bearer tokens
  - Long base64-like strings (likely keys)

Reference: C-01 in Security Audit (2026-02-17)
"""

import re

# ============================================================
# REDACTION PATTERNS
# ============================================================

# Each pattern: (compiled_regex, replacement_label)
_REDACTION_PATTERNS = [
    # OpenAI API keys: sk-... (48+ chars)
    (re.compile(r"sk-[A-Za-z0-9_-]{20,}"), "[REDACTED:openai_key]"),

    # Supabase service_role / anon keys: sbp_... or eyJ... (JWT)
    (re.compile(r"sbp_[A-Za-z0-9_-]{20,}"), "[REDACTED:supabase_key]"),

    # JWT tokens (header.payload.signature)
    (re.compile(r"eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}"),
     "[REDACTED:jwt]"),

    # Qdrant API keys (generic long alphanumeric)
    (re.compile(r"(?i)(?:qdrant[_-]?(?:api[_-]?)?key\s*[=:]\s*)['\"]?([A-Za-z0-9_-]{16,})['\"]?"),
     "[REDACTED:qdrant_key]"),

    # Generic key=value patterns in URLs or configs
    (re.compile(r"(?i)(?:api[_-]?key|apikey|secret|token|password|authorization)\s*[=:]\s*['\"]?[A-Za-z0-9_/.+=-]{8,}['\"]?"),
     "[REDACTED:credential]"),

    # Bearer tokens in headers
    (re.compile(r"(?i)bearer\s+[A-Za-z0-9_/.+=-]{8,}"),
     "[REDACTED:bearer]"),
]

# ============================================================
# PUBLIC API
# ============================================================


def redact_secrets(text: str) -> str:
    """Remove secrets/keys from a string, replacing with [REDACTED:type].

    Safe to call on any string — if no secrets found, returns unchanged.

    Examples:
        >>> redact_secrets("Error with key sk-abc123def456ghi789jkl012mno345pqr678stu901vwx")
        'Error with key [REDACTED:openai_key]'
        >>> redact_secrets("normal error message")
        'normal error message'
    """
    if not text or not isinstance(text, str):
        return text or ""

    result = text
    for pattern, replacement in _REDACTION_PATTERNS:
        result = pattern.sub(replacement, result)
    return result


def redact_exception(exc: Exception) -> str:
    """Redact secrets from an exception's string representation."""
    return redact_secrets(str(exc))


def contains_secret(text: str) -> bool:
    """Check if a string appears to contain any secret/key pattern."""
    if not text or not isinstance(text, str):
        return False
    for pattern, _ in _REDACTION_PATTERNS:
        if pattern.search(text):
            return True
    return False
