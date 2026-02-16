"""
Codi Memory - Consciousness module (FACADE).
Re-exports from: emotion, workspace, prediction, self_model,
learning, curiosity, lifecycle, n8n.

D5: De-God-Module — consciousness.py split into 8 cohesive modules.
This facade maintains backward compatibility: all external imports
(`from modules.consciousness import X`) continue to work.

AJUSTE: lifecycle se importa LAZY via __getattr__ para evitar import hell
(lifecycle importa de 8+ modulos; cargarlo al inicio puede disparar ciclos).
"""

# --- Back-compat re-exports (tests monkeypatch these names) ---
from modules.config import qdrant, memory  # noqa: F401
from modules.utils import _classify_emotion, _get_emotion_text  # noqa: F401

# --- Eager imports (leaf modules, no risk) ---
from modules.n8n import trigger_n8n, listar_webhooks_conocidos, N8N_WEBHOOK_BASE  # noqa: F401

from modules.emotion import (  # noqa: F401
    set_emotional_state, get_emotional_state, update_mood_baseline,
    apply_emotional_decay, get_emotional_expression,
    add_memory_with_emotion, tag_memory_emotion, search_by_emotion,
    get_emotional_memories,
)

from modules.prediction import (  # noqa: F401
    get_predictive_state, predict_context, record_surprise,
    get_prediction_accuracy, update_beliefs, _predictive_state,
)

from modules.workspace import (  # noqa: F401
    get_workspace, update_workspace_spotlight,
    focus_attention, broadcast_to_workspace, get_workspace_state,
    apply_salience_decay, get_high_salience_memories,
    emotional_focus_attention, _global_workspace,
)

from modules.learning import (  # noqa: F401
    auto_learn_from_session, audit_tools, rate_tool_usefulness,
    _tool_metrics, _topic_confidence,
    _init_tool_metric, _record_tool_call,
    _extract_topic_from_text, _get_topic_confidence, _update_topic_confidence,
)

from modules.self_model import (  # noqa: F401
    reflect_on_self, assess_confidence, identify_knowledge_gaps,
    assess_butlin_indicators, _legacy_assess_butlin,
    update_self_model, get_self_model_summary,
)

from modules.curiosity import (  # noqa: F401
    detectar_sorpresa, analizar_patron_trabajo, generar_curiosidad,
    _cargar_curiosidades, _guardar_curiosidades,
    push_curiosidad, get_curiosidades,
)


# --- Lazy imports (lifecycle = orchestrator, heavy deps) ---
_LIFECYCLE_NAMES = {
    "despertar_codi", "verificar_salud_memoria", "_verificar_salud_memoria_interna",
    "ciclo_vida", "consolidate_recent", "find_connections",
    "dream_consolidation", "get_memory_connections",
}


def __getattr__(name):
    if name in _LIFECYCLE_NAMES:
        import modules.lifecycle as _lc
        return getattr(_lc, name)
    raise AttributeError(f"module 'modules.consciousness' has no attribute {name!r}")


def register_tools(mcp):
    """Register all consciousness tools with MCP server.
    Delegates to per-module register_tools() functions."""
    from modules.n8n import register_tools as _rt_n8n
    from modules.emotion import register_tools as _rt_emotion
    from modules.prediction import register_tools as _rt_prediction
    from modules.workspace import register_tools as _rt_workspace
    from modules.learning import register_tools as _rt_learning
    from modules.self_model import register_tools as _rt_self_model
    from modules.curiosity import register_tools as _rt_curiosity
    from modules.lifecycle import register_tools as _rt_lifecycle

    _rt_n8n(mcp)
    _rt_emotion(mcp)
    _rt_prediction(mcp)
    _rt_workspace(mcp)
    _rt_learning(mcp)
    _rt_self_model(mcp)
    _rt_curiosity(mcp)
    _rt_lifecycle(mcp)
