# FTS5 Query Hardening (C-02)

**PR:** Security Hotfix PR1
**Date:** 2026-02-17
**Module:** `modules/fts_safety.py`
**Tests:** `tests/test_fts_safety.py` (50 tests)

## Threat

SQLite FTS5 has its own query syntax. If user input reaches `MATCH` unsanitized:

| Attack | Example | Impact |
|--------|---------|--------|
| Wildcard dump | `*` | Returns every document in the index |
| Boolean injection | `foo OR bar NOT secret` | Attacker controls search logic |
| Column filter | `content:password` | Targets specific columns |
| NEAR proximity | `NEAR(secret password, 5)` | Proximity-based extraction |

The single choke point was `memory_smart.py:search_fts()` executing `WHERE content MATCH ?` with raw user input.

## Fix: Tokenize + Quote

Instead of stripping FTS5 operators (error-prone, whack-a-mole), we use a deterministic tokenization approach:

```
User input: "foo OR bar NOT secret*"
     |
     v
Step 1: Strip + length cap (512 chars max)
Step 2: Regex extract [A-Za-z0-9_]+ tokens, lowercased
     -> ["foo", "or", "bar", "not", "secret"]
Step 3: Filter by length (min 2, max 64 chars)
Step 4: Cap token count (max 12)
Step 5: Quote each token individually
     -> "foo" "or" "bar" "not" "secret"
```

**Result:** FTS5 treats each quoted token as a literal. `"or"` is the word "or", not the `OR` operator. No FTS5 syntax survives.

## What Gets Blocked

| Input | Output | Why safe |
|-------|--------|----------|
| `*` | `""` (empty) | No MATCH executed |
| `foo OR bar` | `"foo" "or" "bar"` | OR becomes literal |
| `NEAR(foo bar)` | `"near" "foo" "bar"` | Parens stripped, NEAR literal |
| `content:password` | `"content" "password"` | Colon stripped |
| `a` | `""` (empty) | Single-char dropped |
| `foo) UNION SELECT *` | `"foo" "union" "select"` | SQL injection neutralized |

## Hardcaps

| Constant | Value | Purpose |
|----------|-------|---------|
| `FTS_MAX_QUERY_LENGTH` | 512 | Truncate input before tokenizing |
| `FTS_MAX_TOKENS` | 12 | Limit tokens per query |
| `FTS_MIN_TOKEN_LENGTH` | 2 | Drop single-char noise |
| `FTS_MAX_TOKEN_LENGTH` | 64 | Truncate oversized tokens |
| `FTS_MAX_RESULTS` | 50 | Hard cap on result count |

## Integration

`search_fts()` in `memory_smart.py` calls `sanitize_fts_query()` before any MATCH. If the sanitized result is empty, it returns `[]` without executing MATCH.

```python
safe_query = sanitize_fts_query(query)
if not safe_query:
    return []
```

All other callers (`memory_core.py`, `recall()`, etc.) go through `search_fts()` -- single choke point, single fix.
