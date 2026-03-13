"""
llm_router.py — Unified LLM router for all Codi subsystems.
=============================================================

Single entry point for ALL LLM calls in the system.
Replaces the fragmented Ollama→NVIDIA→OpenAI fallback chains.

Priority chain:
  1. Ollama (local, free — only if CODI_USE_OLLAMA=true and available)
  2. Anthropic Claude Haiku (our API, reliable, cheap)
  3. OpenAI gpt-4o-mini (backup)

Usage:
    from modules.llm_router import llm_complete

    answer = llm_complete("resolve_curiosity", prompt)
    # answer is str or None

Cost estimates (Claude Haiku 4.5):
    $0.80/MTok input, $4.00/MTok output
    ~50 calls/day × ~1K tokens = ~$0.04/day
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Optional

_logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Task configuration
# ---------------------------------------------------------------------------
TASK_CONFIG = {
    "resolve_curiosity":    {"temperature": 0.3, "max_tokens": 2048},
    "edge_classify":        {"temperature": 0.0, "max_tokens": 1024},
    "semantic_extract":     {"temperature": 0.1, "max_tokens": 2048},
    "self_extract":         {"temperature": 0.1, "max_tokens": 2048},
    "compress_episodes":    {"temperature": 0.2, "max_tokens": 2048},
    "compress_checkpoints": {"temperature": 0.3, "max_tokens": 1024},
    "mine_tool_patterns":   {"temperature": 0.2, "max_tokens": 2048},
    "study_simple":         {"temperature": 0.2, "max_tokens": 4096},
    "study_complex":        {"temperature": 0.3, "max_tokens": 4096},
    "schema_extract":       {"temperature": 0.0, "max_tokens": 2048},
}

# Claude model for sleep loop tasks (cheapest, fastest)
CLAUDE_MODEL = os.getenv("CODI_LLM_MODEL", "claude-haiku-4-5-20251001")

# OpenAI fallback model
OPENAI_MODEL = "gpt-4o-mini"


# ---------------------------------------------------------------------------
# Provider: Ollama (optional, free)
# ---------------------------------------------------------------------------
def _try_ollama(task_type: str, prompt: str) -> Optional[str]:
    """Try Ollama if enabled and available."""
    if os.getenv("CODI_USE_OLLAMA", "").lower() not in ("true", "1", "yes"):
        return None
    try:
        from modules.ollama_router import ollama_chat_completion
        return ollama_chat_completion(task_type, prompt)
    except Exception as e:
        _logger.debug("[llm_router] Ollama failed: %s", e)
        return None


# ---------------------------------------------------------------------------
# Provider: Anthropic Claude
# ---------------------------------------------------------------------------
_anthropic_client = None


def _get_anthropic():
    """Lazy-init Anthropic client."""
    global _anthropic_client
    if _anthropic_client is None:
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            return None
        try:
            import anthropic
            _anthropic_client = anthropic.Anthropic(api_key=api_key)
        except ImportError:
            _logger.warning("[llm_router] anthropic SDK not installed")
            return None
    return _anthropic_client


def _try_claude(task_type: str, prompt: str) -> Optional[str]:
    """Try Claude Haiku via Anthropic API."""
    client = _get_anthropic()
    if client is None:
        return None

    config = TASK_CONFIG.get(task_type, {"temperature": 0.2, "max_tokens": 2048})

    try:
        response = client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=config["max_tokens"],
            temperature=config["temperature"],
            messages=[{"role": "user", "content": prompt}],
        )
        text = response.content[0].text.strip() if response.content else ""
        if text:
            _logger.info("[llm_router] Claude succeeded for %s (%d chars)", task_type, len(text))
            return text
    except Exception as e:
        _logger.warning("[llm_router] Claude failed for %s: %s", task_type, e)

    return None


# ---------------------------------------------------------------------------
# Provider: OpenAI (backup)
# ---------------------------------------------------------------------------
_openai_client = None


def _get_openai():
    """Lazy-init OpenAI client."""
    global _openai_client
    if _openai_client is None:
        try:
            import openai
            _openai_client = openai.OpenAI()  # uses OPENAI_API_KEY from env
        except Exception:
            return None
    return _openai_client


def _try_openai(task_type: str, prompt: str) -> Optional[str]:
    """Try OpenAI gpt-4o-mini as backup."""
    client = _get_openai()
    if client is None:
        return None

    config = TASK_CONFIG.get(task_type, {"temperature": 0.2, "max_tokens": 2048})

    try:
        response = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=config["temperature"],
            max_tokens=config["max_tokens"],
        )
        text = response.choices[0].message.content.strip()
        if text:
            _logger.info("[llm_router] OpenAI succeeded for %s (%d chars)", task_type, len(text))
            return text
    except Exception as e:
        _logger.warning("[llm_router] OpenAI failed for %s: %s", task_type, e)

    return None


# ---------------------------------------------------------------------------
# PG Logging
# ---------------------------------------------------------------------------
_log_table_ensured = False


def _log_to_pg(task_type: str, provider: str, model: str, duration_ms: int,
               prompt_chars: int, output_chars: int, success: bool, error: str = ""):
    """Log LLM call to PG for observability."""
    global _log_table_ensured
    try:
        from modules.config_pg import get_conn
        if not _log_table_ensured:
            with get_conn() as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        CREATE TABLE IF NOT EXISTS llm_calls (
                            id              BIGSERIAL PRIMARY KEY,
                            task_type       TEXT NOT NULL,
                            provider        TEXT NOT NULL,
                            model           TEXT NOT NULL,
                            success         BOOLEAN NOT NULL,
                            duration_ms     INTEGER,
                            prompt_chars    INTEGER,
                            output_chars    INTEGER,
                            error           TEXT,
                            created_at      TIMESTAMPTZ DEFAULT NOW()
                        )
                    """)
            _log_table_ensured = True

        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """INSERT INTO llm_calls
                       (task_type, provider, model, success, duration_ms,
                        prompt_chars, output_chars, error)
                       VALUES (%s, %s, %s, %s, %s, %s, %s, %s)""",
                    (task_type, provider, model, success, duration_ms,
                     prompt_chars, output_chars, error[:500]),
                )
    except Exception as e:
        _logger.debug("[llm_router] PG log failed: %s", e)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def llm_complete(task_type: str, prompt: str, system: str = "") -> Optional[str]:
    """Route LLM call through priority chain: Ollama → Claude → OpenAI.

    Args:
        task_type: Key for task config (temperature, max_tokens).
        prompt: The user/task prompt.
        system: Optional system prompt (only used by Claude).

    Returns:
        Response text, or None if all providers fail.
    """
    full_prompt = prompt
    if system:
        full_prompt = f"{system}\n\n{prompt}"

    t0 = time.time()
    prompt_chars = len(full_prompt)

    # 1. Ollama (free, local)
    answer = _try_ollama(task_type, full_prompt)
    if answer:
        duration_ms = int((time.time() - t0) * 1000)
        _log_to_pg(task_type, "ollama", "local", duration_ms,
                   prompt_chars, len(answer), True)
        return answer

    # 2. Claude Haiku (our API)
    answer = _try_claude(task_type, full_prompt)
    if answer:
        duration_ms = int((time.time() - t0) * 1000)
        _log_to_pg(task_type, "anthropic", CLAUDE_MODEL, duration_ms,
                   prompt_chars, len(answer), True)
        return answer

    # 3. OpenAI gpt-4o-mini (backup)
    answer = _try_openai(task_type, full_prompt)
    if answer:
        duration_ms = int((time.time() - t0) * 1000)
        _log_to_pg(task_type, "openai", OPENAI_MODEL, duration_ms,
                   prompt_chars, len(answer), True)
        return answer

    # All failed
    duration_ms = int((time.time() - t0) * 1000)
    _log_to_pg(task_type, "none", "all_failed", duration_ms,
               prompt_chars, 0, False, "all providers failed")
    _logger.error("[llm_router] All providers failed for %s", task_type)
    return None
