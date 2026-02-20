#!/usr/bin/env python3
"""
FTS5 SAFETY TESTS (Security Hotfix PR1 — C-02)
================================================
Verify that sanitize_fts_query() prevents FTS5 MATCH injection.

Strategy: tokenize + quote. No FTS5 syntax reaches MATCH from user input.

18 tests:
  1. Normal query passes through as quoted tokens
  2. Wildcard * blocked
  3. Empty / whitespace -> empty
  4. OR operator neutralized (becomes literal token)
  5. NOT operator neutralized
  6. NEAR operator neutralized
  7. AND operator neutralized
  8. Quoted phrases become individual tokens
  9. Parentheses stripped (not alphanumeric)
  10. Colons stripped
  11. Single-char tokens dropped (min length 2)
  12. Token count capped at FTS_MAX_TOKENS
  13. Token length capped at FTS_MAX_TOKEN_LENGTH
  14. Query length capped at FTS_MAX_QUERY_LENGTH
  15. All output tokens are quoted
  16. search_fts returns [] for empty sanitized query
  17. clamp_fts_limit enforces FTS_MAX_RESULTS
  18. Fuzz: dangerous chars never produce unsafe output

Run: ./venv/bin/pytest tests/test_fts_safety.py -v
"""

import sys
import os
import re
import sqlite3

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.fts_safety import (
    sanitize_fts_query,
    is_fts_query_safe,
    clamp_fts_limit,
    FTS_MAX_RESULTS,
    FTS_MAX_TOKENS,
    FTS_MAX_TOKEN_LENGTH,
    FTS_MAX_QUERY_LENGTH,
    FTS_MIN_TOKEN_LENGTH,
)


# ============================================================
# HELPERS
# ============================================================

# Dangerous FTS5 operators/chars that must never appear unquoted
_FTS5_OPERATORS = {"OR", "AND", "NOT", "NEAR"}
_FTS5_DANGEROUS_CHARS = set('*()[]{}:^~')


def _assert_output_safe(result: str):
    """Assert that sanitize output contains no unquoted FTS5 operators or dangerous chars."""
    if not result:
        return  # empty is safe

    # Every non-space portion should be a quoted token: "xxx"
    parts = result.split()
    for part in parts:
        assert part.startswith('"') and part.endswith('"'), \
            f"Token not quoted: {part!r} in {result!r}"
        inner = part[1:-1]
        # Inner content should be lowercase alphanumeric + underscore only
        assert re.match(r'^[a-z0-9_]+$', inner), \
            f"Token has unexpected chars: {inner!r} in {result!r}"


# ============================================================
# TEST 1-3: Basic input handling
# ============================================================

class TestBasicInputs:
    """Normal queries, wildcards, empty inputs."""

    def test_normal_query_produces_quoted_tokens(self):
        """Normal multi-word query -> quoted tokens."""
        result = sanitize_fts_query("docker qdrant memory")
        assert result == '"docker" "qdrant" "memory"'

    def test_single_word_query(self):
        """Single word -> single quoted token."""
        result = sanitize_fts_query("consolidation")
        assert result == '"consolidation"'

    def test_wildcard_star_blocked(self):
        """Standalone * must return empty (prevents dump-all)."""
        assert sanitize_fts_query("*") == ""

    def test_star_in_query_stripped(self):
        """Star mixed with words: star stripped, words survive."""
        # * is not alphanumeric, so regex won't capture it
        result = sanitize_fts_query("foo* bar")
        assert '"foo"' in result
        assert '"bar"' in result
        assert "*" not in result

    def test_empty_string(self):
        assert sanitize_fts_query("") == ""

    def test_whitespace_only(self):
        assert sanitize_fts_query("   ") == ""

    def test_none_input(self):
        assert sanitize_fts_query(None) == ""


# ============================================================
# TEST 4-7: FTS5 operators neutralized
# ============================================================

class TestOperatorNeutralization:
    """FTS5 operators become harmless quoted literal tokens."""

    def test_or_operator(self):
        """'foo OR bar' -> OR becomes quoted token (neutralized)."""
        result = sanitize_fts_query("foo OR bar")
        _assert_output_safe(result)
        assert '"foo"' in result
        assert '"bar"' in result
        # OR is lowercased to "or" and quoted -- harmless
        assert "OR" not in result  # uppercase operator gone

    def test_not_operator(self):
        result = sanitize_fts_query("NOT secret")
        _assert_output_safe(result)
        assert '"not"' in result
        assert '"secret"' in result

    def test_near_operator(self):
        result = sanitize_fts_query("NEAR(foo bar)")
        _assert_output_safe(result)
        assert '"near"' in result
        assert '"foo"' in result
        assert '"bar"' in result
        assert "(" not in result
        assert ")" not in result

    def test_and_operator(self):
        result = sanitize_fts_query("foo AND bar")
        _assert_output_safe(result)
        assert '"foo"' in result
        assert '"bar"' in result


# ============================================================
# TEST 8-10: Special characters stripped
# ============================================================

class TestSpecialCharsStripped:
    """Non-alphanumeric characters don't survive tokenization."""

    def test_quoted_phrases_become_tokens(self):
        """User quotes are stripped; words become separate tokens."""
        result = sanitize_fts_query('"exact phrase"')
        _assert_output_safe(result)
        assert '"exact"' in result
        assert '"phrase"' in result

    def test_parentheses_stripped(self):
        result = sanitize_fts_query("(foo) [bar] {baz}")
        _assert_output_safe(result)
        assert '"foo"' in result
        assert '"bar"' in result
        assert '"baz"' in result

    def test_colons_stripped(self):
        result = sanitize_fts_query("title:secret content:password")
        _assert_output_safe(result)
        assert ":" not in result
        assert '"title"' in result
        assert '"secret"' in result


# ============================================================
# TEST 11-14: Caps and limits enforced
# ============================================================

class TestCapsEnforced:
    """Token count, token length, query length caps."""

    def test_single_char_tokens_dropped(self):
        """Tokens shorter than FTS_MIN_TOKEN_LENGTH are dropped."""
        result = sanitize_fts_query("a b c docker")
        assert result == '"docker"'

    def test_token_count_capped(self):
        """More than FTS_MAX_TOKENS tokens -> only first N kept."""
        words = " ".join(f"word{i}" for i in range(50))
        result = sanitize_fts_query(words)
        token_count = len(result.split())
        assert token_count == FTS_MAX_TOKENS

    def test_token_length_capped(self):
        """Token longer than FTS_MAX_TOKEN_LENGTH is truncated."""
        long_word = "a" * 200
        result = sanitize_fts_query(long_word)
        inner = result.strip('"')
        assert len(inner) == FTS_MAX_TOKEN_LENGTH

    def test_query_length_capped(self):
        """Input longer than FTS_MAX_QUERY_LENGTH is truncated before tokenizing."""
        # Create a query that's way over the limit
        long_query = "word " * 200  # ~1000 chars
        result = sanitize_fts_query(long_query)
        # Should still produce tokens, just from the truncated portion
        assert result != ""
        _assert_output_safe(result)


# ============================================================
# TEST 15: All output tokens are properly quoted
# ============================================================

class TestOutputFormat:
    """Every token in output must be individually quoted."""

    def test_all_tokens_are_quoted(self):
        """Output format: each token is \"word\" separated by spaces."""
        queries = [
            "docker qdrant",
            "foo OR bar NOT baz",
            "NEAR(hello world)",
            "simple",
            "multi word query with many tokens",
        ]
        for q in queries:
            result = sanitize_fts_query(q)
            if result:
                _assert_output_safe(result)


# ============================================================
# TEST 16: search_fts integration (empty query = no MATCH)
# ============================================================

class TestSearchFtsIntegration:
    """search_fts returns [] when sanitized query is empty."""

    def test_search_fts_star_returns_empty(self):
        """search_fts('*') must return [] -- no dump-all."""
        from modules.memory_smart import search_fts
        result = search_fts("*")
        assert result == []

    def test_search_fts_empty_returns_empty(self):
        from modules.memory_smart import search_fts
        result = search_fts("")
        assert result == []

    def test_search_fts_single_char_returns_empty(self):
        from modules.memory_smart import search_fts
        result = search_fts("a")
        assert result == []


# ============================================================
# TEST 17: clamp_fts_limit
# ============================================================

class TestClampLimit:
    """FTS result limit is hard-capped."""

    def test_clamp_over_max(self):
        assert clamp_fts_limit(1000) == FTS_MAX_RESULTS

    def test_clamp_normal(self):
        assert clamp_fts_limit(20) == 20

    def test_clamp_zero(self):
        assert clamp_fts_limit(0) == FTS_MAX_RESULTS

    def test_clamp_negative(self):
        assert clamp_fts_limit(-5) == FTS_MAX_RESULTS

    def test_clamp_non_int(self):
        assert clamp_fts_limit("abc") == FTS_MAX_RESULTS


# ============================================================
# TEST 18: Fuzz with dangerous characters
# ============================================================

class TestFuzzDangerousChars:
    """Directed fuzz: dangerous FTS5 chars never produce unsafe output."""

    DANGEROUS_INPUTS = [
        '*',
        '""',
        "' OR '1'='1",
        "* OR *",
        "NEAR(secret password, 5)",
        'content:* OR 1=1',
        "foo) UNION SELECT * FROM memories_fts--",
        "^prefix*",
        "~fuzzy",
        "+required -excluded",
        "{col1 col2}:term",
        'a"b"c',
        "OR OR OR",
        "NOT NOT NOT",
        "***",
        "\x00\x01\x02",
        "a" * 1000,
        " ".join(["token"] * 100),
        "NEAR/5(foo bar)",
        "col1:col2:col3",
    ]

    @pytest.mark.parametrize("dangerous_input", DANGEROUS_INPUTS)
    def test_dangerous_input_is_safe(self, dangerous_input):
        """No dangerous input can produce unsafe FTS5 output."""
        result = sanitize_fts_query(dangerous_input)
        _assert_output_safe(result)
        # Must never contain raw * outside quotes
        assert "*" not in result


# ============================================================
# is_fts_query_safe convenience
# ============================================================

class TestIsFtsQuerySafe:
    """Convenience function mirrors sanitize behavior."""

    def test_safe_query(self):
        assert is_fts_query_safe("docker qdrant") is True

    def test_unsafe_star(self):
        assert is_fts_query_safe("*") is False

    def test_unsafe_empty(self):
        assert is_fts_query_safe("") is False
