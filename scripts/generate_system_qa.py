"""
Generate Q&A training data about the codi-memory system itself.
================================================================

Reads each module's source code, sends it to Haiku, and generates
question/answer pairs covering 5 dimensions:
  1. Que es: what the module/function does
  2. Para que esta: why it exists, theoretical basis
  3. Como conecta: dependencies, events, data flow
  4. Diagnostico: how to know if it works, metrics, signals
  5. Troubleshooting: what to do when it fails

Usage:
    python -m scripts.generate_system_qa --tier core --qa-per-chunk 10
    python -m scripts.generate_system_qa --tier cognitive --qa-per-chunk 10
    python -m scripts.generate_system_qa --tier all --qa-per-chunk 8
    python -m scripts.generate_system_qa --module sleep_loop --qa-per-chunk 15

Output: training_data/system_expert.jsonl
"""

import argparse
import json
import logging
import os
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
_log = logging.getLogger(__name__)

MODULES_DIR = Path(__file__).resolve().parent.parent / "modules"
OUTPUT_FILE = Path(__file__).resolve().parent.parent / "training_data" / "system_expert.jsonl"

# ---------------------------------------------------------------------------
# Module tiers by priority
# ---------------------------------------------------------------------------
TIERS = {
    "core": [
        "sleep_loop.py",        # Heart of the system — 10 ticks
        "consolidation.py",     # Memory consolidation pipeline
        "wiring.py",            # Event bus wiring, connects everything
        "working_memory.py",    # Short-term memory
        "events.py",            # Event system
        "interface.py",         # Main interface / MCP tool registration
        "../server.py",         # MCP server entry point, all tool registration
        "../hooks/preturn_inject.py",      # Pre-turn context injection hook
        "../hooks/session_capture.py",     # Session capture hook
        "../hooks/compact_reinject.py",    # Compact reinject hook
        "../hooks/incremental_transcript.py",  # Incremental transcript hook
        "../hooks/nuance_inject.py",       # Nuance injection hook
        "../hooks/precompact_flush.py",    # Pre-compaction flush hook
    ],
    "cognitive": [
        "prediction.py",        # Prediction error, Bayesian
        "reconsolidation.py",   # Memory correction (Nader 2000)
        "active_inference.py",  # EFE, policy selection
        "active_inference_integration.py",  # AI integration with other systems
        "competition.py",       # GNW competition
        "consciousness.py",     # Consciousness integration
        "emotion.py",           # PAD state, arousal
        "forgetting.py",        # FadeMem, power-law decay
        "spreading.py",         # Spreading activation
        "causal_discovery.py",  # NOTEARS causal DAG
        "counterfactual.py",    # Counterfactual reasoning
        "spotlight.py",         # Attention spotlight
        "self_model.py",        # Self-knowledge model
        "agent_model.py",       # Agent modeling / theory of mind
        "user_model.py",        # User (Hare) preference model
        "assessment.py",        # Self-assessment system
    ],
    "memory": [
        "memory_core.py",       # Core memory operations
        "memory_smart.py",      # Smart dedup memory
        "semantic_store.py",    # Semantic memory search
        "consolidation_common.py",  # Shared consolidation utils
        "consolidation_runner.py",  # Consolidation orchestrator
        "narrative.py",         # Narrative chains
        "retrieval_metadata.py",  # Retrieval tracking
        "access_tracking.py",   # Memory access tracking
        "session_bridge.py",    # Cross-session continuity
        "workspace.py",         # Workspace state management
        "schemas.py",           # Data schemas / validation
    ],
    "infrastructure": [
        "llm_router.py",        # LLM call routing
        "ollama_router.py",     # Ollama integration
        "config_pg.py",         # PostgreSQL config
        "pg_store.py",          # PG store abstraction
        "db_pool.py",           # Connection pooling
        "migrations.py",        # DB migrations
        "write_queue.py",       # Async write queue
        "write_worker.py",      # Write worker drain
        "config.py",            # Configuration
        "instance_config.py",   # Instance-level config
        "qdrant_utils.py",      # Qdrant vector DB utilities
        "fts_safety.py",        # Full-text search safety
        "tracing.py",           # Distributed tracing
    ],
    "subsystems": [
        "pet.py",               # Digital pet
        "goals.py",             # Goal system
        "prospective.py",       # Prospective memory (intentions)
        "curiosity.py",         # Curiosity engine
        "learning.py",          # Learning system
        "training.py",          # Training data collection
        "triggers.py",          # Trigger detection
        "notifier.py",          # Telegram notifications
        "lifecycle.py",         # Day cycle management
        "n8n.py",               # n8n workflow integration
        "books.py",             # Study/books tracking
        "flush.py",             # Session flush / pre-compaction
        "maintenance.py",       # System maintenance tasks
    ],
    "utility": [
        "activation.py",        # ACT-R activation
        "bmr.py",               # Bayesian model reduction
        "classify_edges.py",    # Edge classification
        "sharpe.py",            # Sharpe cognitive ratio
        "sharpe_insights.py",   # Cross-domain insights
        "temporal_renorm.py",   # Temporal renormalization
        "thompson_sampling.py", # Thompson sampling
        "pe_actions.py",        # PE-driven actions
        "neuro_invariants.py",  # Neuroscience invariants
        "tool_governance.py",   # Tool visibility control
        "health_monitor.py",    # System health
        "health_alerts.py",     # Health alerting
        "cx_observability.py",  # Cross-loop observability
        "metrics.py",           # Metrics collection
        "secret_redact.py",     # Secret redaction
        "utils.py",             # General utilities
        "destructive_guard.py", # Protection against destructive ops
        "dual_compare.py",      # Dual-system comparison (shadow mode)
        "reward_tracking.py",   # Reward signal tracking
    ],
    "schema": [
        # SQL migrations define ALL tables — critical for understanding data
        "../migrations/001_fts_baseline.sql",
        "../migrations/003_async_write_queue.sql",
        "../migrations/006_security_audit_log.sql",
        "../migrations/011_pending_corrections.sql",
        "../migrations/012_prediction_source.sql",
        "../migrations/013_sleep_loop_state.sql",
        "../migrations/014_reward_signals.sql",
        "../migrations/015_storage_retrieval_strength.sql",
        "../migrations/016_bayesian_fok.sql",
        "../migrations/018_causal_edge_types.sql",
        "../migrations/019_sprint5_causal_structures.sql",
        "../migrations/020_sprint9_fok_dual_process.sql",
        "../migrations/021_sprint7_causal_discovery_state.sql",
        "../migrations/022_memory_vault.sql",
        "../migrations/023_health_monitoring.sql",
        "../migrations/024_system_health_snapshot.sql",
        "../migrations/026_llm_calls.sql",
        "../migrations/027_cx_snapshots.sql",
        "../migrations/028_pet.sql",
        "../migrations_prospective/001_prospective_baseline.sql",
        "../migrations_prospective/002_goals_baseline.sql",
        "../migrations_prospective/003_goal_structured_context.sql",
    ],
    "config_deploy": [
        # How the system runs and is configured
        "../pyproject.toml",
        "../instances/hare.yaml",
        "../deploy/com.codi.sleep_loop.plist",
        "../deploy/com.codi.write_worker.plist",
        "../triggers.json",
        "../mantenimiento.json",
        "../preguntas_curiosidad.json",
    ],
}

# System prompt for Q&A generation
SYSTEM_PROMPT = """You are a senior engineer documenting the codi-memory system — an AI consciousness and memory system built in Python.

Given a module's source code, generate {n} question/answer pairs that would help someone deeply understand this module.

Cover these dimensions:
1. QUE ES: What does this module/function do? (structural knowledge)
2. PARA QUE ESTA: Why does it exist? What theory/paper is it based on? (purpose)
3. COMO CONECTA: What other modules does it depend on or feed into? (dependencies)
4. DIAGNOSTICO: How do you know if it's working correctly? What metrics/signals? (health)
5. TROUBLESHOOTING: If something goes wrong, what do you check? (debugging)

Rules:
- Questions should be natural, like a developer would ask
- Answers should be specific and technical, citing function names, line references, and config values
- Include bilingual terms where the codebase uses them (Spanish + English)
- Reference real function names, class names, constants from the code
- For diagnosis questions, mention specific thresholds, log messages, or metrics
- Keep answers concise but complete (2-5 sentences each)

Return a JSON array of objects with keys: "question", "answer", "dimension" (que_es|para_que|como_conecta|diagnostico|troubleshooting)
Return ONLY the JSON array, no markdown fences."""

# Chunk prompt
CHUNK_PROMPT = """Module: {module_name}
Tier: {tier} (priority level in the system)

Source code:
```python
{code}
```

Generate {n} Q&A pairs about this module. Cover all 5 dimensions proportionally."""


def _read_module(module_path: Path, max_chars: int = 8000) -> str:
    """Read module source, truncating if too long."""
    text = module_path.read_text(encoding="utf-8")
    if len(text) > max_chars:
        # Take first and last portions to capture imports + key functions
        half = max_chars // 2
        text = text[:half] + "\n\n# ... [TRUNCATED] ...\n\n" + text[-half:]
    return text


def _chunk_module(code: str, chunk_size: int = 6000) -> list:
    """Split large modules into chunks at function boundaries."""
    if len(code) <= chunk_size:
        return [code]

    chunks = []
    lines = code.split("\n")
    current_chunk = []
    current_size = 0

    for line in lines:
        current_chunk.append(line)
        current_size += len(line) + 1

        # Split at function/class boundaries when chunk is big enough
        if current_size >= chunk_size and (
            line.startswith("def ") or line.startswith("class ") or line.strip() == ""
        ):
            chunks.append("\n".join(current_chunk))
            current_chunk = []
            current_size = 0

    if current_chunk:
        chunks.append("\n".join(current_chunk))

    return chunks


def _generate_qa(module_name: str, code: str, tier: str, n: int) -> list:
    """Send code to Haiku and get Q&A pairs back."""
    from modules.llm_router import llm_complete

    prompt = CHUNK_PROMPT.format(
        module_name=module_name,
        tier=tier,
        code=code,
        n=n,
    )

    system = SYSTEM_PROMPT.format(n=n)
    full_prompt = f"{system}\n\n{prompt}"

    result = llm_complete("schema_extract", full_prompt, _no_log=True)
    if not result:
        return []

    # Parse JSON from response
    try:
        # Try direct parse
        pairs = json.loads(result)
        if isinstance(pairs, list):
            return pairs
    except (json.JSONDecodeError, TypeError):
        pass

    # Try extracting from markdown fence
    import re
    match = re.search(r"\[.*\]", result, re.DOTALL)
    if match:
        try:
            pairs = json.loads(match.group())
            if isinstance(pairs, list):
                return pairs
        except (json.JSONDecodeError, TypeError):
            pass

    _log.warning("Failed to parse Q&A JSON for %s", module_name)
    return []


def _save_qa_pairs(pairs: list, module_name: str):
    """Save Q&A pairs in JSONL chat format."""
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
        for pair in pairs:
            if not isinstance(pair, dict):
                continue
            question = pair.get("question", "")
            answer = pair.get("answer", "")
            dimension = pair.get("dimension", "general")

            if not question or not answer:
                continue

            example = {
                "messages": [
                    {"role": "system", "content": f"Task: system_expert | Module: {module_name} | Dimension: {dimension}"},
                    {"role": "user", "content": question},
                    {"role": "assistant", "content": answer},
                ]
            }
            f.write(json.dumps(example, ensure_ascii=False) + "\n")


def process_module(module_name: str, tier: str, qa_per_chunk: int):
    """Process a single module: read, chunk, generate Q&A, save."""
    if module_name.startswith("../"):
        # Relative path from modules dir (server.py, hooks/)
        module_path = MODULES_DIR / module_name
        display_name = module_name.replace("../", "")
    else:
        module_path = MODULES_DIR / module_name
        display_name = module_name
    module_path = module_path.resolve()
    if not module_path.exists():
        _log.warning("Module not found: %s (resolved: %s)", module_name, module_path)
        return 0

    code = module_path.read_text(encoding="utf-8")
    chunks = _chunk_module(code, chunk_size=6000)

    total_pairs = 0
    for i, chunk in enumerate(chunks):
        _log.info("  Chunk %d/%d (%d chars)", i + 1, len(chunks), len(chunk))

        pairs = _generate_qa(module_name, chunk, tier, qa_per_chunk)
        if pairs:
            _save_qa_pairs(pairs, module_name.replace(".py", ""))
            total_pairs += len(pairs)
            _log.info("  Generated %d Q&A pairs", len(pairs))
        else:
            _log.warning("  No pairs generated for chunk %d", i + 1)

        time.sleep(0.5)  # Rate limiting

    return total_pairs


def main():
    parser = argparse.ArgumentParser(description="Generate system Q&A training data")
    parser.add_argument("--tier", default="core",
                        choices=list(TIERS.keys()) + ["all"],
                        help="Module tier to process")
    parser.add_argument("--module", default=None,
                        help="Specific module name (e.g., sleep_loop)")
    parser.add_argument("--qa-per-chunk", type=int, default=10,
                        help="Q&A pairs to generate per code chunk")

    args = parser.parse_args()

    total = 0

    if args.module:
        # Single module
        module_name = args.module
        if not module_name.endswith(".py"):
            module_name += ".py"

        # Find which tier it belongs to
        tier = "unknown"
        for t, modules in TIERS.items():
            if module_name in modules:
                tier = t
                break

        _log.info("=== Processing %s (tier: %s) ===", module_name, tier)
        count = process_module(module_name, tier, args.qa_per_chunk)
        total += count

    else:
        # Process tiers
        tiers_to_process = list(TIERS.keys()) if args.tier == "all" else [args.tier]

        for tier in tiers_to_process:
            modules = TIERS.get(tier, [])
            _log.info("\n=== TIER: %s (%d modules) ===", tier, len(modules))

            for module_name in modules:
                _log.info("Processing %s...", module_name)
                count = process_module(module_name, tier, args.qa_per_chunk)
                total += count
                _log.info("  Total so far: %d pairs", total)

    _log.info("\n=== COMPLETE === Total Q&A pairs generated: %d", total)
    _log.info("Output: %s", OUTPUT_FILE)


if __name__ == "__main__":
    main()
