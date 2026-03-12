"""
FTS5 Query Sanitization (Security Hotfix PR1)
==============================================
Prevents FTS5 MATCH injection by converting untrusted input
into a safe, deterministic token-based query.

Strategy: tokenize + quote. No FTS5 syntax ever reaches MATCH
from user input. All tokens are individually quoted, making
operators like OR, NOT, NEAR, *, etc. into harmless literals.

Reference: C-02 in Security Audit (2026-02-17)
"""

import re

# ============================================================
# CONSTANTS
# ============================================================

FTS_MAX_RESULTS = 50          # Hard cap on any FTS query result count
FTS_MAX_QUERY_LENGTH = 512    # Max chars before tokenization
FTS_MAX_TOKENS = 12           # Max tokens per query
FTS_MIN_TOKEN_LENGTH = 2      # Tokens shorter than this are dropped
FTS_MAX_TOKEN_LENGTH = 64     # Tokens longer than this are truncated

# Regex: only alphanumeric + underscore, 2-64 chars
_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")


# ============================================================
# PUBLIC API
# ============================================================

def sanitize_fts_query(query: str) -> str:
    """Convert untrusted user input into a safe FTS5 MATCH query.

    Strategy:
      1. Strip and length-cap the raw input
      2. Extract alphanumeric tokens via regex (no FTS5 syntax survives)
      3. Filter by min/max token length
      4. Cap token count
      5. Quote each token individually -> AND-implicit search

    Returns:
        Safe FTS5 query string (e.g. '"docker" "qdrant"'), or "" if
        no valid tokens remain. Caller must check for "" and skip MATCH.

    Examples:
        >>> sanitize_fts_query("docker qdrant")
        '"docker" "qdrant"'
        >>> sanitize_fts_query("*")
        ''
        >>> sanitize_fts_query("foo OR bar")
        '"foo" "bar"'
        >>> sanitize_fts_query("NEAR(foo bar)")
        '"near" "foo" "bar"'
    """
    if not query or not isinstance(query, str):
        return ""

    # Step 1: strip + length cap
    raw = query.strip()[:FTS_MAX_QUERY_LENGTH]

    if not raw:
        return ""

    # Step 2: explicit block of standalone wildcard
    if raw == "*":
        return ""

    # Step 3: tokenize -- only alphanumeric+underscore survive
    raw_tokens = _TOKEN_RE.findall(raw.lower())

    # Step 4: filter by token length
    tokens = []
    for t in raw_tokens:
        if len(t) < FTS_MIN_TOKEN_LENGTH:
            continue
        if len(t) > FTS_MAX_TOKEN_LENGTH:
            t = t[:FTS_MAX_TOKEN_LENGTH]
        tokens.append(t)

    # Step 5: cap token count
    tokens = tokens[:FTS_MAX_TOKENS]

    if not tokens:
        return ""

    # Step 6: quote each token individually (AND-implicit)
    return " ".join(f'"{t}"' for t in tokens)


def is_fts_query_safe(query: str) -> bool:
    """Check if a raw query would produce a non-empty safe FTS query.

    Useful for logging/metrics before executing search.
    """
    return sanitize_fts_query(query) != ""


def clamp_fts_limit(limit: int) -> int:
    """Ensure FTS result limit does not exceed hard cap."""
    if not isinstance(limit, int) or limit < 1:
        return 1
    return min(limit, FTS_MAX_RESULTS)
