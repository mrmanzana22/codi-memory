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
import pathlib
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
# Provider: MLX-LM (local fine-tuned Qwen3-4B, free, fastest)
# ---------------------------------------------------------------------------
_mlx_model = None
_mlx_tokenizer = None
_MLX_ADAPTER_PATH = pathlib.Path(__file__).resolve().parent.parent / "adapters" / "codi-v2"
_MLX_BASE_MODEL = "mlx-community/Qwen3-4B-Instruct-2507-4bit"

# Tasks suitable for the fine-tuned model (trained on these task types)
_MLX_SUPPORTED_TASKS = {
    "semantic_extract", "self_extract", "compress_episodes",
    "compress_checkpoints",
}


def _get_mlx():
    """Lazy-load MLX model + adapter. Returns (model, tokenizer) or (None, None)."""
    global _mlx_model, _mlx_tokenizer
    if _mlx_model is not None:
        return _mlx_model, _mlx_tokenizer
    if not _MLX_ADAPTER_PATH.exists():
        return None, None
    try:
        from mlx_lm import load
        _mlx_model, _mlx_tokenizer = load(
            str(_MLX_BASE_MODEL),
            adapter_path=str(_MLX_ADAPTER_PATH),
        )
        _logger.info("[llm_router] MLX model loaded: %s + %s", _MLX_BASE_MODEL, _MLX_ADAPTER_PATH.name)
        return _mlx_model, _mlx_tokenizer
    except Exception as e:
        _logger.warning("[llm_router] MLX load failed: %s", e)
        return None, None


def _try_mlx(task_type: str, prompt: str, system: str = "") -> Optional[str]:
    """Try local fine-tuned model via MLX-LM. Only for supported task types."""
    if task_type not in _MLX_SUPPORTED_TASKS:
        return None
    model, tokenizer = _get_mlx()
    if model is None:
        return None

    config = TASK_CONFIG.get(task_type, {"temperature": 0.2, "max_tokens": 2048})

    try:
        from mlx_lm import generate
        full_prompt = f"{system}\n\n{prompt}" if system else prompt
        messages = [{"role": "user", "content": full_prompt}]
        formatted = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        response = generate(
            model, tokenizer, prompt=formatted,
            max_tokens=config["max_tokens"],
        )
        # Strip thinking tags if present
        text = response.strip()
        if text.startswith("<think>"):
            think_end = text.find("</think>")
            if think_end != -1:
                text = text[think_end + len("</think>"):].strip()
        if text:
            _logger.info("[llm_router] MLX succeeded for %s (%d chars)", task_type, len(text))
            return text
    except Exception as e:
        _logger.warning("[llm_router] MLX failed for %s: %s", task_type, e)

    return None


# ---------------------------------------------------------------------------
# Provider: Ollama (optional, free)
# ---------------------------------------------------------------------------
def _try_ollama(task_type: str, prompt: str, system: str = "") -> Optional[str]:
    """Try Ollama if enabled and available."""
    if os.getenv("CODI_USE_OLLAMA", "").lower() not in ("true", "1", "yes"):
        return None
    try:
        from modules.ollama_router import ollama_chat_completion
        # Ollama has no dedicated system param — concatenate into prompt
        full_prompt = f"{system}\n\n{prompt}" if system else prompt
        return ollama_chat_completion(task_type, full_prompt)
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


def _try_claude(task_type: str, prompt: str, system: str = "") -> Optional[str]:
    """Try Claude Haiku via Anthropic API."""
    client = _get_anthropic()
    if client is None:
        return None

    config = TASK_CONFIG.get(task_type, {"temperature": 0.2, "max_tokens": 2048})

    try:
        kwargs: dict = dict(
            model=CLAUDE_MODEL,
            max_tokens=config["max_tokens"],
            temperature=config["temperature"],
            messages=[{"role": "user", "content": prompt}],
            timeout=15.0,
        )
        if system:
            kwargs["system"] = system  # pass as dedicated top-level parameter
        response = client.messages.create(**kwargs)
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


def _try_openai(task_type: str, prompt: str, system: str = "") -> Optional[str]:
    """Try OpenAI gpt-4o-mini as backup."""
    client = _get_openai()
    if client is None:
        return None

    config = TASK_CONFIG.get(task_type, {"temperature": 0.2, "max_tokens": 2048})

    try:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})  # dedicated system message
        messages.append({"role": "user", "content": prompt})
        response = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=messages,
            temperature=config["temperature"],
            max_tokens=config["max_tokens"],
            timeout=15.0,
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
    """Log LLM call to PG for observability.

    Table created by migration 026_llm_calls.sql.
    """
    global _log_table_ensured
    try:
        from modules.config_pg import get_conn
        if not _log_table_ensured:
            # Verify table exists (created by migration 026)
            with get_conn() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        "SELECT 1 FROM information_schema.tables "
                        "WHERE table_name = 'llm_calls'"
                    )
                    if not cur.fetchone():
                        return  # table not yet migrated, skip logging
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
# Training Data Logging (Fase 1: destilacion pasiva)
# Logs successful LLM input/output pairs in MLX-LM chat JSONL format.
# Used to train our own model later (Qwen3-4B via QLoRA).
# ---------------------------------------------------------------------------
_TRAINING_DATA_DIR = pathlib.Path(__file__).resolve().parent.parent / "training_data"

# Don't log these — already replaced with classical or not useful for training
_SKIP_TRAINING_LOG = {"edge_classify"}


def _log_training_data(task_type: str, prompt: str, response: str, system: str = ""):
    """Save successful LLM call as training example in JSONL chat format."""
    if task_type in _SKIP_TRAINING_LOG:
        return
    if not response or len(response) < 10:
        return
    try:
        _TRAINING_DATA_DIR.mkdir(parents=True, exist_ok=True)
        filepath = _TRAINING_DATA_DIR / f"{task_type}.jsonl"
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        messages.append({"role": "assistant", "content": response})
        example = {"messages": messages}
        with open(filepath, "a", encoding="utf-8") as f:
            f.write(json.dumps(example, ensure_ascii=False) + "\n")
    except Exception as e:
        _logger.debug("[llm_router] Training data log failed: %s", e)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def llm_complete(task_type: str, prompt: str, system: str = "",
                 _no_log: bool = False) -> Optional[str]:
    """Route LLM call through priority chain: Ollama → Claude → OpenAI.

    Args:
        task_type: Key for task config (temperature, max_tokens).
        prompt: The user/task prompt.
        system: Optional system prompt (only used by Claude).
        _no_log: If True, skip training data logging (use for DPO/synthetic
                 calls whose responses must not contaminate SFT files).

    Returns:
        Response text, or None if all providers fail.
    """
    t0 = time.time()
    prompt_chars = len(prompt) + len(system)

    # 0. MLX fine-tuned model (free, local, fastest — only for trained tasks)
    answer = _try_mlx(task_type, prompt, system)
    if answer:
        duration_ms = int((time.time() - t0) * 1000)
        _log_to_pg(task_type, "mlx_local", "codi-v2", duration_ms,
                   prompt_chars, len(answer), True)
        if not _no_log:
            _log_training_data(task_type, prompt, answer, system)
        return answer

    # 1. Ollama (free, local) — concatenates system into prompt internally
    answer = _try_ollama(task_type, prompt, system)
    if answer:
        duration_ms = int((time.time() - t0) * 1000)
        _log_to_pg(task_type, "ollama", "local", duration_ms,
                   prompt_chars, len(answer), True)
        if not _no_log:
            _log_training_data(task_type, prompt, answer, system)
        return answer

    # 2. Claude Haiku (our API) — system passed as top-level parameter
    answer = _try_claude(task_type, prompt, system)
    if answer:
        duration_ms = int((time.time() - t0) * 1000)
        _log_to_pg(task_type, "anthropic", CLAUDE_MODEL, duration_ms,
                   prompt_chars, len(answer), True)
        if not _no_log:
            _log_training_data(task_type, prompt, answer, system)
        return answer

    # 3. OpenAI gpt-4o-mini (backup) — system prepended as system message
    answer = _try_openai(task_type, prompt, system)
    if answer:
        duration_ms = int((time.time() - t0) * 1000)
        _log_to_pg(task_type, "openai", OPENAI_MODEL, duration_ms,
                   prompt_chars, len(answer), True)
        if not _no_log:
            _log_training_data(task_type, prompt, answer, system)
        return answer

    # All failed
    duration_ms = int((time.time() - t0) * 1000)
    _log_to_pg(task_type, "none", "all_failed", duration_ms,
               prompt_chars, 0, False, "all providers failed")
    _logger.error("[llm_router] All providers failed for %s", task_type)
    return None
