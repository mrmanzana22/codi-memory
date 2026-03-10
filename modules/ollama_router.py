"""
ollama_router.py - Local LLM router for token savings.
=======================================================

Routes LLM tasks to local Ollama models instead of OpenAI.
Quality gates validate output per task type.
Logs everything to PG for observability.

Usage:
    from modules.ollama_router import get_router

    router = get_router()
    result = router.route("edge_classify", prompt)
    # result = {"ok": True, "output": "...", "model": "qwen3.5:9b", ...}

Control:
    CODI_USE_OLLAMA=true   → route to Ollama (default: false)
    OLLAMA_URL             → Ollama endpoint (default: localhost:11434)
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
import urllib.error
import urllib.request
from typing import Optional

_logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_CHAT_URL = f"{OLLAMA_URL}/api/chat"
OLLAMA_TIMEOUT_S = int(os.getenv("OLLAMA_TIMEOUT_S", "120"))
OLLAMA_MAX_RETRIES = 2

# Feature flag: opt-in to Ollama routing
def _use_ollama() -> bool:
    return os.getenv("CODI_USE_OLLAMA", "false").lower() in ("true", "1", "yes")


# ---------------------------------------------------------------------------
# Task → Model mapping
# ---------------------------------------------------------------------------
TASK_MODEL_MAP = {
    # Consolidation tasks (all use qwen3.5:4b — 9b was removed 2026-03-09)
    "edge_classify":        "qwen3.5:4b",
    "semantic_extract":     "qwen3.5:4b",
    "self_extract":         "qwen3.5:4b",
    "compress_episodes":    "qwen3.5:4b",
    "compress_checkpoints": "qwen3.5:4b",
    # Learning tasks
    "resolve_curiosity":    "qwen3.5:4b",
    "mine_tool_patterns":   "qwen3.5:4b",
    "study_simple":         "qwen3.5:4b",
    "study_complex":        "qwen3.5:4b",
    "schema_extract":       "qwen3.5:4b",
}

# Model-specific options
MODEL_OPTIONS = {
    "qwen3.5:4b": {
        "num_ctx": 8192,
        "num_predict": 2048,
        "top_k": 40,
        "top_p": 0.9,
        "repeat_penalty": 1.0,
    },
}

# Task-specific temperature overrides
TASK_TEMPERATURE = {
    "edge_classify":        0.0,
    "semantic_extract":     0.1,
    "self_extract":         0.1,
    "compress_episodes":    0.2,
    "compress_checkpoints": 0.3,
    "resolve_curiosity":    0.3,
    "mine_tool_patterns":   0.2,
    "study_simple":         0.2,
    "study_complex":        0.3,
    "schema_extract":       0.0,
}


# ---------------------------------------------------------------------------
# Ollama Client
# ---------------------------------------------------------------------------
class OllamaClient:
    """Wraps Ollama HTTP API with retry logic."""

    def __init__(self, base_url: str = OLLAMA_CHAT_URL, timeout: int = OLLAMA_TIMEOUT_S):
        self.base_url = base_url
        self.timeout = timeout

    def chat(
        self,
        model: str,
        messages: list[dict],
        temperature: float = 0.2,
        options: Optional[dict] = None,
        think: bool = False,
    ) -> str:
        """Send chat completion to Ollama. Returns response text.

        Args:
            think: Enable thinking mode (for models like qwen3.5).
                   When False, sends think=false to get direct output.
                   DeepSeek R1 always thinks (it's built-in).

        Raises RuntimeError on failure after retries.
        """
        opts = dict(options or {})
        opts["temperature"] = temperature

        body = {
            "model": model,
            "messages": messages,
            "stream": False,
            "options": opts,
        }
        # Qwen 3.5 models support think=true/false at the API level.
        # DeepSeek R1 always thinks regardless of this flag.
        if not think and "qwen" in model.lower():
            body["think"] = False

        payload = json.dumps(body).encode()

        last_error = None
        for attempt in range(1, OLLAMA_MAX_RETRIES + 1):
            try:
                req = urllib.request.Request(
                    self.base_url,
                    data=payload,
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    data = json.loads(resp.read())
                    content = data.get("message", {}).get("content", "").strip()
                    if content:
                        return content
                    last_error = "Empty response from Ollama"
            except (urllib.error.URLError, TimeoutError, OSError) as e:
                last_error = str(e)
                _logger.warning(
                    "Ollama attempt %d/%d failed for %s: %s",
                    attempt, OLLAMA_MAX_RETRIES, model, last_error,
                )
                if attempt < OLLAMA_MAX_RETRIES:
                    time.sleep(2 * attempt)  # Brief backoff

        raise RuntimeError(f"Ollama failed after {OLLAMA_MAX_RETRIES} attempts: {last_error}")

    def is_available(self) -> bool:
        """Check if Ollama is running."""
        try:
            req = urllib.request.Request(
                OLLAMA_URL + "/api/tags",
                method="GET",
            )
            with urllib.request.urlopen(req, timeout=5) as resp:
                return resp.status == 200
        except Exception:
            return False

    def list_models(self) -> list[str]:
        """List available models."""
        try:
            req = urllib.request.Request(
                OLLAMA_URL + "/api/tags",
                method="GET",
            )
            with urllib.request.urlopen(req, timeout=5) as resp:
                data = json.loads(resp.read())
                return [m["name"] for m in data.get("models", [])]
        except Exception:
            return []


# ---------------------------------------------------------------------------
# Quality Gates
# ---------------------------------------------------------------------------
class OllamaQualityGate:
    """Validates Ollama output per task type."""

    GATES = {
        "edge_classify": {
            "format": "json_array",
            "required_keys": ["type"],
            "valid_values": {
                "type": ["causes", "enables", "prevents", "co_occurs"],
            },
            "min_items": 1,
        },
        "semantic_extract": {
            "format": "json_array",
            "required_keys": ["fact", "confidence"],
            "min_items": 1,
        },
        "self_extract": {
            "format": "json_array",
            "required_keys": ["fact"],
            "min_items": 0,  # Empty array is valid (no self-knowledge found)
        },
        "compress_episodes": {
            "format": "plain_text",
            "min_length": 30,
            "max_length": 2000,
        },
        "compress_checkpoints": {
            "format": "plain_text",
            "min_length": 20,
            "max_length": 600,
        },
        "resolve_curiosity": {
            "format": "plain_text",
            "min_length": 100,
            "max_length": 5000,
        },
        "mine_tool_patterns": {
            "format": "plain_text",
            "min_length": 30,
            "max_length": 2000,
        },
        "study_simple": {
            "format": "plain_text",
            "min_length": 50,
            "max_length": 5000,
        },
        "study_complex": {
            "format": "plain_text",
            "min_length": 100,
            "max_length": 8000,
        },
        "schema_extract": {
            "format": "json_array",
            "required_keys": [],
            "min_items": 1,
        },
    }

    @classmethod
    def validate(cls, task_type: str, output: str) -> tuple[bool, str]:
        """Validate output against gate rules.

        Returns (is_valid, reason).
        """
        gate = cls.GATES.get(task_type)
        if gate is None:
            return True, "no gate defined"

        fmt = gate["format"]

        if fmt == "json_array":
            return cls._validate_json_array(output, gate)
        elif fmt == "plain_text":
            return cls._validate_plain_text(output, gate)
        else:
            return True, "unknown format"

    @classmethod
    def _validate_json_array(cls, output: str, gate: dict) -> tuple[bool, str]:
        """Validate JSON array output."""
        # Strip markdown code blocks if present
        clean = output.strip()
        if clean.startswith("```"):
            # Extract content between first and last ```
            parts = clean.split("```")
            if len(parts) >= 3:
                clean = parts[1]
                if clean.startswith("json"):
                    clean = clean[4:]
            clean = clean.strip()

        # DeepSeek R1 wraps output in <think>...</think> tags
        clean = _strip_think_tags(clean)

        try:
            parsed = json.loads(clean)
        except json.JSONDecodeError:
            return False, f"invalid JSON: {clean[:100]}"

        if not isinstance(parsed, list):
            return False, f"expected array, got {type(parsed).__name__}"

        min_items = gate.get("min_items", 0)
        if len(parsed) < min_items:
            return False, f"too few items: {len(parsed)} < {min_items}"

        required_keys = gate.get("required_keys", [])
        valid_values = gate.get("valid_values", {})

        for i, item in enumerate(parsed):
            if not isinstance(item, dict):
                # Allow string items for edge_classify backward compat
                if isinstance(item, str) and "type" in required_keys:
                    if valid_values.get("type") and item not in valid_values["type"]:
                        return False, f"item {i}: invalid value '{item}'"
                    continue
                return False, f"item {i}: expected dict, got {type(item).__name__}"

            for key in required_keys:
                if key not in item:
                    return False, f"item {i}: missing key '{key}'"

            for key, allowed in valid_values.items():
                val = item.get(key)
                if val is not None and val not in allowed:
                    return False, f"item {i}: '{key}' value '{val}' not in {allowed}"

        return True, "ok"

    @classmethod
    def _validate_plain_text(cls, output: str, gate: dict) -> tuple[bool, str]:
        """Validate plain text output."""
        clean = _strip_think_tags(output).strip()
        length = len(clean)

        min_len = gate.get("min_length", 0)
        max_len = gate.get("max_length", 100_000)

        if length < min_len:
            return False, f"too short: {length} < {min_len}"
        if length > max_len:
            return False, f"too long: {length} > {max_len}"

        return True, "ok"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _strip_think_tags(text: str) -> str:
    """Remove <think>...</think> tags from DeepSeek R1 output."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def _extract_json_from_response(text: str) -> str:
    """Extract JSON content from a response that may have markdown or think tags."""
    clean = _strip_think_tags(text).strip()
    if clean.startswith("```"):
        parts = clean.split("```")
        if len(parts) >= 3:
            clean = parts[1]
            if clean.startswith("json"):
                clean = clean[4:]
        clean = clean.strip()
    return clean


# ---------------------------------------------------------------------------
# PG Logging
# ---------------------------------------------------------------------------
_tables_ensured = False


def _ensure_tables():
    """Create ollama_tasks table in PG if it doesn't exist."""
    global _tables_ensured
    if _tables_ensured:
        return

    try:
        from modules.config_pg import get_conn
        with get_conn() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS ollama_tasks (
                    id              BIGSERIAL PRIMARY KEY,
                    task_type       TEXT NOT NULL,
                    model           TEXT NOT NULL,
                    status          TEXT NOT NULL DEFAULT 'pending',
                    quality_ok      BOOLEAN,
                    quality_reason  TEXT,
                    duration_ms     INTEGER,
                    attempts        INTEGER DEFAULT 1,
                    prompt_chars    INTEGER,
                    output_chars    INTEGER,
                    error           TEXT,
                    created_at      TIMESTAMPTZ DEFAULT NOW()
                )
            """)
            # Index for stats queries
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_ollama_tasks_type_created
                ON ollama_tasks (task_type, created_at)
            """)
        _tables_ensured = True
        _logger.info("ollama_tasks table ensured")
    except Exception as e:
        _logger.warning("Could not ensure ollama_tasks table: %s", e)


def _log_task(
    task_type: str,
    model: str,
    status: str,
    quality_ok: Optional[bool] = None,
    quality_reason: str = "",
    duration_ms: int = 0,
    attempts: int = 1,
    prompt_chars: int = 0,
    output_chars: int = 0,
    error: str = "",
):
    """Log a task execution to PG."""
    try:
        from modules.config_pg import get_conn
        _ensure_tables()
        with get_conn() as conn:
            conn.execute(
                """INSERT INTO ollama_tasks
                   (task_type, model, status, quality_ok, quality_reason,
                    duration_ms, attempts, prompt_chars, output_chars, error)
                   VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)""",
                (task_type, model, status, quality_ok, quality_reason,
                 duration_ms, attempts, prompt_chars, output_chars, error),
            )
    except Exception as e:
        _logger.debug("Could not log ollama task: %s", e)


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------
def get_ollama_stats(task_type: Optional[str] = None, days: int = 7) -> dict:
    """Query PG for Ollama task statistics.

    Returns success rates, avg duration, task counts by type.
    """
    try:
        from modules.config_pg import get_conn
        _ensure_tables()
        with get_conn() as conn:
            where = "WHERE created_at > NOW() - make_interval(days => %s)"
            params = [int(days)]
            if task_type:
                where += " AND task_type = %s"
                params.append(task_type)

            # Overall stats
            row = conn.execute(
                f"""SELECT
                    COUNT(*) as total,
                    COUNT(*) FILTER (WHERE status = 'success') as success,
                    COUNT(*) FILTER (WHERE status = 'failed') as failed,
                    COUNT(*) FILTER (WHERE quality_ok = true) as quality_passed,
                    AVG(duration_ms) FILTER (WHERE status = 'success') as avg_duration_ms,
                    SUM(prompt_chars) as total_prompt_chars,
                    SUM(output_chars) as total_output_chars
                FROM ollama_tasks {where}""",
                params,
            ).fetchone()

            total = row[0] or 0
            success = row[1] or 0

            # Per-task-type breakdown
            rows = conn.execute(
                f"""SELECT task_type, model,
                    COUNT(*) as total,
                    COUNT(*) FILTER (WHERE status = 'success') as success,
                    AVG(duration_ms) FILTER (WHERE status = 'success') as avg_ms
                FROM ollama_tasks {where}
                GROUP BY task_type, model
                ORDER BY total DESC""",
                params,
            ).fetchall()

            breakdown = []
            for r in rows:
                breakdown.append({
                    "task_type": r[0],
                    "model": r[1],
                    "total": r[2],
                    "success": r[3],
                    "avg_ms": round(r[4]) if r[4] else 0,
                    "success_rate": f"{r[3] / r[2] * 100:.0f}%" if r[2] > 0 else "n/a",
                })

            return {
                "period_days": days,
                "total_tasks": total,
                "success": success,
                "failed": row[2] or 0,
                "quality_passed": row[3] or 0,
                "success_rate": f"{success / total * 100:.0f}%" if total > 0 else "n/a",
                "avg_duration_ms": round(row[4]) if row[4] else 0,
                "total_prompt_chars": row[5] or 0,
                "total_output_chars": row[6] or 0,
                "breakdown": breakdown,
            }
    except Exception as e:
        return {"error": str(e)}


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------
class OllamaRouter:
    """Main entry: route(task_type, prompt) -> validated output.

    Selects model by task type, calls Ollama, validates via quality gate,
    logs to PG, returns structured result.
    """

    def __init__(self):
        self.client = OllamaClient()

    def route(
        self,
        task_type: str,
        prompt: str,
        system: str = "",
        context: str = "",
    ) -> dict:
        """Route a task to the appropriate Ollama model.

        Args:
            task_type: Key from TASK_MODEL_MAP
            prompt: The user prompt
            system: Optional system prompt
            context: Optional additional context prepended to prompt

        Returns:
            {
                "ok": bool,
                "output": str,         # Raw output (cleaned of think tags)
                "model": str,
                "duration_ms": int,
                "attempts": int,
                "quality_reason": str,
            }
        """
        model = TASK_MODEL_MAP.get(task_type)
        if model is None:
            return {
                "ok": False,
                "output": "",
                "model": "unknown",
                "duration_ms": 0,
                "attempts": 0,
                "quality_reason": f"unknown task_type: {task_type}",
            }

        # Build messages
        messages = []
        if system:
            messages.append({"role": "system", "content": system})

        full_prompt = f"{context}\n\n{prompt}" if context else prompt
        messages.append({"role": "user", "content": full_prompt})

        # Get model options
        options = dict(MODEL_OPTIONS.get(model, {}))
        temperature = TASK_TEMPERATURE.get(task_type, 0.2)

        # Execute with quality gate retry
        t0 = time.time()
        attempts = 0
        last_output = ""
        last_reason = ""

        for attempt in range(1, 3):  # Max 2 attempts (retry once on quality fail)
            attempts = attempt
            try:
                raw_output = self.client.chat(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    options=options,
                )
                clean_output = _strip_think_tags(raw_output)
                last_output = clean_output

                # Quality gate
                is_valid, reason = OllamaQualityGate.validate(task_type, raw_output)
                last_reason = reason

                if is_valid:
                    duration_ms = int((time.time() - t0) * 1000)
                    _log_task(
                        task_type=task_type,
                        model=model,
                        status="success",
                        quality_ok=True,
                        quality_reason=reason,
                        duration_ms=duration_ms,
                        attempts=attempts,
                        prompt_chars=len(full_prompt),
                        output_chars=len(clean_output),
                    )
                    return {
                        "ok": True,
                        "output": clean_output,
                        "model": model,
                        "duration_ms": duration_ms,
                        "attempts": attempts,
                        "quality_reason": reason,
                    }

                _logger.warning(
                    "Quality gate failed for %s (attempt %d): %s",
                    task_type, attempt, reason,
                )

            except RuntimeError as e:
                last_reason = str(e)
                _logger.error("Ollama call failed for %s: %s", task_type, e)
                break

        # All attempts failed
        duration_ms = int((time.time() - t0) * 1000)
        _log_task(
            task_type=task_type,
            model=model,
            status="failed",
            quality_ok=False,
            quality_reason=last_reason,
            duration_ms=duration_ms,
            attempts=attempts,
            prompt_chars=len(full_prompt),
            output_chars=len(last_output),
            error=last_reason,
        )
        return {
            "ok": False,
            "output": last_output,
            "model": model,
            "duration_ms": duration_ms,
            "attempts": attempts,
            "quality_reason": last_reason,
        }


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------
_router: Optional[OllamaRouter] = None


def get_router() -> OllamaRouter:
    """Get or create the singleton router."""
    global _router
    if _router is None:
        _router = OllamaRouter()
    return _router


# ---------------------------------------------------------------------------
# Convenience: drop-in replacement for consolidation LLM calls
# ---------------------------------------------------------------------------
def ollama_chat_completion(
    task_type: str,
    prompt: str,
    system: str = "",
) -> Optional[str]:
    """Drop-in helper for consolidation.py call sites.

    Fallback chain: Ollama → NVIDIA → None (caller falls back to OpenAI).
    Returns the response text if any provider succeeds.
    Returns None if all fail.
    """
    if not _use_ollama():
        return None

    router = get_router()

    # Quick availability check (cached per call, but cheap)
    if not router.client.is_available():
        _logger.debug("Ollama not available, trying NVIDIA fallback")
        return _nvidia_fallback(task_type, prompt, system)

    result = router.route(task_type, prompt, system=system)
    if result["ok"]:
        return result["output"]

    _logger.info(
        "Ollama failed quality gate for %s (%s), trying NVIDIA fallback",
        task_type, result["quality_reason"],
    )
    return _nvidia_fallback(task_type, prompt, system)


# ── NVIDIA fallback for sleep loop tasks ──

# Map task types to best NVIDIA model
_NVIDIA_TASK_MODELS = {
    "resolve_curiosity":    "deepseek-ai/deepseek-v3.2",
    "study_complex":        "deepseek-ai/deepseek-r1-distill-qwen-32b",
    "study_simple":         "meta/llama-3.3-70b-instruct",
    "semantic_extract":     "meta/llama-3.3-70b-instruct",
    "self_extract":         "meta/llama-3.3-70b-instruct",
    "compress_episodes":    "meta/llama-3.3-70b-instruct",
    "compress_checkpoints": "meta/llama-3.3-70b-instruct",
    "edge_classify":        "meta/llama-3.3-70b-instruct",
    "mine_tool_patterns":   "meta/llama-3.3-70b-instruct",
    "schema_extract":       "qwen/qwen2.5-coder-32b-instruct",
}


def _nvidia_fallback(
    task_type: str,
    prompt: str,
    system: str = "",
) -> Optional[str]:
    """Try NVIDIA as fallback when Ollama fails."""
    nvidia_key = os.getenv("NVIDIA_API_KEY", "")
    if not nvidia_key:
        return None

    nvidia_model = _NVIDIA_TASK_MODELS.get(task_type, "meta/llama-3.3-70b-instruct")
    nvidia_url = os.getenv("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1")
    temperature = TASK_TEMPERATURE.get(task_type, 0.2)

    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    body = json.dumps({
        "model": nvidia_model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": 2048,
        "stream": False,
    }).encode()

    try:
        req = urllib.request.Request(
            f"{nvidia_url}/chat/completions",
            data=body,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {nvidia_key}",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read())
            choices = data.get("choices", [])
            if choices:
                content = choices[0].get("message", {}).get("content", "").strip()
                if content:
                    # Validate through quality gate
                    is_valid, reason = OllamaQualityGate.validate(task_type, content)
                    if is_valid:
                        _logger.info(
                            "NVIDIA fallback succeeded for %s (model=%s)",
                            task_type, nvidia_model,
                        )
                        _log_task(
                            task_type=task_type,
                            model=f"nvidia:{nvidia_model}",
                            status="success",
                            quality_ok=True,
                            quality_reason=reason,
                            prompt_chars=len(prompt),
                            output_chars=len(content),
                        )
                        return content
                    else:
                        _logger.warning(
                            "NVIDIA output failed quality gate for %s: %s",
                            task_type, reason,
                        )
    except Exception as e:
        _logger.warning("NVIDIA fallback failed for %s: %s", task_type, e)

    return None


# ---------------------------------------------------------------------------
# MCP Tool Registration
# ---------------------------------------------------------------------------
def register_tools(mcp_instance):
    """Register Ollama router tools with the MCP server."""

    @mcp_instance.tool()
    async def get_ollama_delegation_stats(task_type: str = "", days: int = 7) -> str:
        """Get Ollama local LLM usage statistics.

        Shows success rates, avg duration, task counts by type.
        Use to monitor token savings from Ollama delegation.

        Args:
            task_type: Filter by task type (empty = all)
            days: Number of days to look back (default 7)

        Returns:
            JSON string with stats breakdown
        """
        stats = get_ollama_stats(task_type or None, days)
        return json.dumps(stats, indent=2)

    @mcp_instance.tool()
    async def ollama_health() -> str:
        """Check Ollama availability and list installed models.

        Returns:
            JSON with availability status and model list
        """
        client = OllamaClient()
        available = client.is_available()
        models = client.list_models() if available else []
        use_ollama = _use_ollama()

        return json.dumps({
            "ollama_available": available,
            "ollama_enabled": use_ollama,
            "env_var": "CODI_USE_OLLAMA",
            "models_installed": models,
            "task_model_map": TASK_MODEL_MAP,
        }, indent=2)
