"""
Tool Governance - Bundle-based tool visibility control.

Reduces MCP surface area from ~114 tools to ~28 (core) by default.
Tools are never deleted — only hidden from the MCP registry.

Feature flag precedence (same pattern as .sharpe_mode):
  1. Env var CODI_TOOLSET
  2. Dot-file .toolset in repo root
  3. Default: "core"

Valid toolset values: core | ops | research | full

Presets expand to bundle unions:
  core     -> {core}
  ops      -> {core, ops}
  research -> {core, research}
  full     -> {core, ops, research, full}
"""

import os
import logging

_logger = logging.getLogger("codi-memory.governance")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOLSET_FILE = os.path.join(BASE_DIR, ".toolset")

VALID_TOOLSETS = frozenset({"core", "ops", "research", "full"})

# ============================================================
# BUNDLE DEFINITIONS
# ============================================================

BUNDLE_CORE = frozenset({
    # Interface (macro tools)
    "recall", "remember", "context_snapshot", "get_runtime_flags",
    # Lifecycle
    "despertar_codi", "ciclo_vida", "verificar_salud_memoria",
    # Session
    "checkpoint_memoria", "flush_session",
    # Memory CRUD (essential)
    "add_memory", "add_memory_smart", "search_memory", "get_critical_memories",
    # Working Memory
    "get_working_memory", "push_to_working_memory", "update_working_memory",
    "get_narrative_chain", "link_narrative_trace",
    # Triggers
    "evaluar_triggers", "activar_trigger", "listar_triggers",
    # Sharpe
    "get_sharpe_report", "get_sharpe_insights_report",
    # Emotional (read-only essentials)
    "get_emotional_state", "get_emotional_expression",
    # Prospective (intentions - daily use)
    "ver_intenciones", "crear_intencion", "completar_intencion",
    # Goals (Sprint 15 - daily use)
    "crear_goal", "ver_goals", "actualizar_goal", "completar_goal",
    "contexto_goals", "arbol_goals",
    # CPO v2: CX observability
    "get_cx_health",
    # Pet (tamagochi digital)
    "pet_status", "adopt_pet_tool", "care_for_pet_tool",
    # Source Monitoring (Johnson et al. 1993)
    "get_memory_source",
    # Governance meta (always visible)
    "get_toolset_status",
})

BUNDLE_OPS = frozenset({
    # Maintenance
    "registrar_mantenimiento", "verificar_mantenimiento",
    "marcar_mantenimiento_hecho", "mantenimiento_memorias",
    "ver_recordatorios_externos", "limpiar_recordatorios",
    # FTS
    "sync_fts_index", "process_fts_queue_now", "get_fts_queue_stats",
    # Write Queue
    "write_status", "cancel_write",
    # Consolidation + lifecycle ops
    "run_consolidation", "get_consolidation_stats",
    "get_semantic_facts", "correct_memory",
    "consolidate_recent", "find_connections",
    "dream_consolidation", "get_memory_connections",
    # Diagnostics
    "perf_report_tool", "tool_usage_summary_tool", "embed_cache_stats",
    # Status
    "session_bridge_status", "sleep_report_status", "assessment_report",
    # Backup / Export
    "restore_memories", "export_memories_markdown", "export_to_markdown",
    # Prospective ops
    "cancelar_intencion", "posponer_intencion", "mantenimiento_intenciones",
    # Learning
    "auto_learn_from_session", "audit_tools", "rate_tool_usefulness",
    # Training (self-correction protocol)
    "capturar_ejemplo_training", "guardar_ejemplo_training",
    "listar_ejemplos_training", "contar_ejemplos_training",
})

BUNDLE_RESEARCH = frozenset({
    # Self-model
    "reflect_on_self", "assess_confidence", "identify_knowledge_gaps",
    "update_self_model", "get_self_model_summary",
    # Curiosity
    "detectar_sorpresa", "analizar_patron_trabajo", "generar_curiosidad",
    "push_curiosidad", "get_curiosidades",
    # Spreading activation
    "spread_activation", "get_activation_map",
    # Prediction
    "predict_context", "record_surprise",
    "get_prediction_accuracy", "update_beliefs",
    # Workspace (GNW experimental)
    "focus_attention", "broadcast_to_workspace", "get_workspace_state",
    "apply_salience_decay", "get_high_salience_memories",
    "emotional_focus_attention",
    # Books (knowledge introspection)
    "listar_libros", "ver_libro", "agregar_capitulo", "crear_libro",
})

# Tools only visible in "full" mode (destructive, rarely-used, edge-case)
# Not defined explicitly — anything registered but not in core/ops/research
# automatically lands here.

BUNDLES = {
    "core": BUNDLE_CORE,
    "ops": BUNDLE_OPS,
    "research": BUNDLE_RESEARCH,
    "full": frozenset(),
}

# Preset -> which bundles are active
TOOLSET_PRESETS = {
    "core": {"core"},
    "ops": {"core", "ops"},
    "research": {"core", "research"},
    "full": {"core", "ops", "research", "full"},
}


# ============================================================
# RESOLVER
# ============================================================

def resolve_toolset() -> str:
    """Resolve active toolset from env var > dot-file > default.

    Returns one of: core, ops, research, full.
    Invalid values fall back to core with a warning.
    """
    # 1. Env var (highest precedence)
    env_val = os.environ.get("CODI_TOOLSET", "").strip().lower()
    if env_val:
        if env_val in VALID_TOOLSETS:
            return env_val
        _logger.warning(
            "CODI_TOOLSET=%r is invalid (valid: %s). Falling back to 'core'.",
            env_val, ", ".join(sorted(VALID_TOOLSETS)),
        )
        return "core"

    # 2. Dot-file
    if os.path.isfile(TOOLSET_FILE):
        try:
            with open(TOOLSET_FILE, "r") as f:
                file_val = f.read().strip().lower()
            if file_val in VALID_TOOLSETS:
                return file_val
            _logger.warning(
                ".toolset contains %r (invalid). Falling back to 'core'.", file_val,
            )
            return "core"
        except Exception as e:
            _logger.warning("Error reading .toolset: %s. Falling back to 'core'.", e)
            return "core"

    # 3. Default
    return "core"


def get_allowed_tools(toolset: str = None) -> frozenset:
    """Return the set of tool names allowed for the given toolset.

    For 'full', returns None (meaning: allow everything).
    """
    if toolset is None:
        toolset = resolve_toolset()

    if toolset == "full":
        return None  # signal: no filtering

    active_bundles = TOOLSET_PRESETS.get(toolset, {"core"})
    allowed = set()
    for bundle_name in active_bundles:
        bundle_tools = BUNDLES.get(bundle_name, frozenset())
        allowed.update(bundle_tools)
    return frozenset(allowed)


# ============================================================
# APPLY FILTER
# ============================================================

def apply_allowlist(mcp, allowed_tools):
    """Remove tools from MCP registry that are not in allowed_tools.

    If allowed_tools is None (full mode), no tools are removed.
    Returns (removed_count, removed_names) for logging.
    """
    if allowed_tools is None:
        return 0, []

    # Get all currently registered tool names
    all_tools = set(mcp._tool_manager._tools.keys())
    to_remove = all_tools - set(allowed_tools)

    removed_names = []
    failed_names = []
    for name in sorted(to_remove):
        try:
            mcp.remove_tool(name)
        except Exception as e:
            failed_names.append(name)
            _logger.warning("Failed to remove tool %r: %s", name, e)
        else:
            removed_names.append(name)

    if failed_names:
        _logger.warning("Tool allowlist left %d tool(s) visible: %s", len(failed_names), failed_names)

    return len(removed_names), removed_names


# ============================================================
# STATUS TOOL (registered separately in server.py)
# ============================================================

def _get_toolset_status_impl(mcp) -> str:
    """Implementation of get_toolset_status tool."""
    toolset = resolve_toolset()
    active_bundles = TOOLSET_PRESETS.get(toolset, {"core"})

    visible_count = len(mcp._tool_manager._tools)

    # Calculate what's in each active bundle
    bundle_counts = {}
    for b in sorted(active_bundles):
        if b == "full":
            bundle_counts[b] = "all"
        elif b in BUNDLES:
            bundle_counts[b] = len(BUNDLES[b])

    # Calculate total defined across all bundles
    all_defined = set()
    for b_tools in BUNDLES.values():
        all_defined.update(b_tools)

    lines = [
        f"# Tool Governance Status",
        f"",
        f"Toolset: **{toolset}**",
        f"Active bundles: {', '.join(sorted(active_bundles))}",
        f"Visible tools: {visible_count}",
        f"",
        f"## Bundle sizes",
    ]
    for b_name in ["core", "ops", "research"]:
        marker = " (active)" if b_name in active_bundles else ""
        lines.append(f"  {b_name}: {len(BUNDLES[b_name])}{marker}")
    lines.append(f"  full-only: (everything else)")

    lines.append(f"")
    lines.append(f"## Source")
    env_val = os.environ.get("CODI_TOOLSET", "").strip()
    file_exists = os.path.isfile(TOOLSET_FILE)
    if env_val:
        lines.append(f"  Resolved from: env CODI_TOOLSET={env_val}")
    elif file_exists:
        try:
            with open(TOOLSET_FILE) as f:
                fv = f.read().strip()
            lines.append(f"  Resolved from: .toolset file (value: {fv})")
        except Exception:
            lines.append(f"  Resolved from: .toolset file (read error)")
    else:
        lines.append(f"  Resolved from: default (no env, no .toolset)")

    lines.append(f"")
    lines.append(f"## Quick switch")
    lines.append(f"  echo 'ops' > .toolset    # enable core+ops")
    lines.append(f"  echo 'full' > .toolset   # enable everything")
    lines.append(f"  rm .toolset              # back to core default")

    return "\n".join(lines)


def register_governance_tool(mcp):
    """Register the get_toolset_status tool with MCP."""
    @mcp.tool()
    def get_toolset_status() -> str:
        """Show current tool governance status: active toolset, bundles, visible tool count, and how to switch."""
        return _get_toolset_status_impl(mcp)
