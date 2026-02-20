"""
Codi Memory - Workspace module.
GWT spotlight, attention, salience, emotional focus attention.
emotional_focus_attention lives HERE (cortical attention, Desimone & Duncan 1995),
not in emotion.py — it READS emotional state as input but IS attention logic.
"""

import json

from qdrant_client.models import Filter, FieldCondition, MatchValue, Range

from modules.config import (
    memory, qdrant, USER_ID, COLLECTION_NAME,
    _emotional_state, now_iso,
)
from modules.secret_redact import redact_secrets
from modules.utils import (
    resolve_memory_id,
    _classify_emotion,
)
from modules.activation import compute_unified_activation
from modules.competition import CompetitionCandidate, run_workspace_competition
from modules.access_tracking import record_access

__all__ = [
    "get_workspace",
    "update_workspace_spotlight",
    "focus_attention",
    "broadcast_to_workspace",
    "get_workspace_state",
    "apply_salience_decay",
    "get_high_salience_memories",
    "emotional_focus_attention",
    "_global_workspace",
    "register_tools",
]

# Global Workspace - "teatro de la consciencia" de Baars
_global_workspace = {
    'spotlight': [],
    'recent_context': [],
    'last_broadcast': None,
    'workspace_theme': None
}


def get_workspace():
    """Obtiene el estado actual del workspace."""
    return _global_workspace


def update_workspace_spotlight(memories: list, theme: str = None):
    """Actualiza el spotlight del workspace."""
    global _global_workspace
    _global_workspace['spotlight'] = memories[:5]
    _global_workspace['recent_context'] = (_global_workspace['recent_context'] + memories)[-10:]
    _global_workspace['last_broadcast'] = now_iso()
    if theme:
        _global_workspace['workspace_theme'] = theme


def focus_attention(context: str, depth: str = "normal") -> str:
    """
    Trae memorias relevantes al Global Workspace (spotlight de atencion).

    Args:
        context: El tema o contexto en el que enfocar atencion
        depth: Profundidad de busqueda (shallow, normal, deep)
    """
    try:
        limits = {'shallow': 3, 'normal': 5, 'deep': 10}
        limit = limits.get(depth, 5)
        results = memory.search(query=context, user_id=USER_ID, limit=limit * 2)

        if not results or not results.get('results'):
            return f"No encontre memorias relacionadas con: {context}"

        # WIRING-6: Score candidates with unified activation scorer
        competition_candidates = []

        for r in results['results']:
            mem_id = r.get('id')
            base_score = r.get('score', 0)
            try:
                points = qdrant.retrieve(collection_name=COLLECTION_NAME, ids=[mem_id], with_payload=True)
                if points:
                    payload = points[0].payload
                    salience = payload.get('attention_salience', 0.5)
                    importance = payload.get('narrative_importance', 'medium')
                    source = payload.get('ownership_source', 'unknown')

                    # Unified activation replaces ad-hoc formula (WIRING-6.2)
                    act_result = compute_unified_activation(
                        memory_id=mem_id,
                        created_at=payload.get('created_at', ''),
                        last_accessed=payload.get('attention_last_accessed', ''),
                        access_count=payload.get('attention_access_count', 0),
                        access_timestamps=payload.get('access_timestamps'),
                        spreading_boost=float(salience),
                        importance=importance,
                        noise=True,
                    )
                    attention_score = act_result.total

                    competition_candidates.append(CompetitionCandidate(
                        content=r.get('memory', ''),
                        source_domain="episodic",
                        activation=attention_score,
                        memory_id=mem_id,
                        metadata={
                            'salience': salience, 'source': source,
                            'importance': importance,
                            'topic': ' '.join(payload.get('narrative_themes', [])),
                        },
                    ))

                    # Update attention metadata (batched)
                    new_salience = min(salience + 0.1, 1.0)
                    access_count = payload.get('attention_access_count', 0)
                    record_access(COLLECTION_NAME, mem_id, {
                        'attention_salience': new_salience,
                        'attention_access_count': access_count + 1,
                        'attention_last_accessed': now_iso(),
                    })
            except Exception:
                competition_candidates.append(CompetitionCandidate(
                    content=r.get('memory', ''),
                    source_domain="episodic",
                    activation=base_score,
                    memory_id=mem_id,
                    metadata={'salience': 0.5, 'source': 'unknown', 'importance': 'unknown'},
                ))

        # GWT competition: only winners enter spotlight
        comp_result = run_workspace_competition(
            competition_candidates, slots=limit, current_focus=context,
        )
        spotlight_candidates = [
            {
                'id': w.memory_id, 'content': w.content,
                'attention_score': w.activation,
                'salience': w.metadata.get('salience', 0.5),
                'source': w.metadata.get('source', 'unknown'),
                'importance': w.metadata.get('importance', 'unknown'),
            }
            for w in comp_result.winners
        ]
        spotlight = spotlight_candidates
        update_workspace_spotlight(spotlight, theme=context)

        # Spreading activation: propagar a vecinos de las memorias en spotlight
        try:
            from modules.spreading import _spread_activation
            top_ids = [m['id'] for m in spotlight if m.get('id')][:3]
            if top_ids:
                _spread_activation(top_ids, depth=1, factor=0.5, seed_boost=0.0)
        except Exception:
            pass

        lines = [f"# GLOBAL WORKSPACE - Atencion enfocada\n"]
        lines.append(f"**Contexto:** {context}")
        lines.append(f"**Profundidad:** {depth}")
        lines.append(f"**En spotlight:** {len(spotlight)} memorias\n")
        lines.append("## Memorias en el Spotlight")
        for i, mem in enumerate(spotlight, 1):
            lines.append(f"{i}. [{mem['source']}|{mem['importance']}] (att:{mem['attention_score']:.2f})")
            lines.append(f"   {mem['content'][:70]}...")
        lines.append(f"\n*Salience incrementada para memorias accedidas*")
        return "\n".join(lines)
    except Exception as e:
        return f"Error enfocando atencion: {redact_secrets(str(e))}"


def broadcast_to_workspace(memory_id: str) -> str:
    """
    Pone una memoria especifica en el centro del workspace y la conecta con otras.

    Args:
        memory_id: ID de la memoria (puede ser parcial, ej: "004d896d")
    """
    try:
        full_id = resolve_memory_id(memory_id)
        if not full_id:
            return f"No encontre memoria con ID que empiece con '{memory_id}'"

        points = qdrant.retrieve(collection_name=COLLECTION_NAME, ids=[full_id], with_payload=True)
        if not points:
            return f"No encontre memoria con ID {full_id}"

        payload = points[0].payload
        main_content = payload.get('data', '')
        main_themes = payload.get('narrative_themes', [])

        lines = [f"# BROADCAST - Memoria al centro del workspace\n"]
        lines.append(f"**Contenido:** {main_content[:80]}...")
        lines.append(f"**Temas:** {', '.join(main_themes) if main_themes else 'ninguno'}\n")

        record_access(COLLECTION_NAME, full_id, {
            'attention_salience': 1.0,
            'attention_access_count': payload.get('attention_access_count', 0) + 5,
            'attention_last_accessed': now_iso(),
            'was_broadcast': True,
        })

        # Spreading activation: propagar depth=2 desde la memoria broadcast
        from modules.spreading import _spread_activation
        spread_result = _spread_activation([full_id], depth=2, factor=0.7, seed_boost=0.0)
        connections_made = spread_result.get('affected', 0)

        lines.append("## Memorias activadas por spreading")
        if spread_result.get('updates'):
            for mid, sal in sorted(spread_result['updates'].items(), key=lambda x: -x[1])[:10]:
                lines.append(f"- [{mid[:8]}] salience -> {sal:.3f}")
        else:
            lines.append("*Sin vecinos activados*")

        workspace = get_workspace()
        workspace['spotlight'] = [{'id': full_id, 'content': main_content}]
        workspace['last_broadcast'] = now_iso()

        # Emit WORKSPACE_BROADCAST event (LAZY import to avoid cycles)
        try:
            from modules.events import event_bus, Events
            event_bus.emit(Events.WORKSPACE_BROADCAST, {
                'memory_id': full_id,
                'content': main_content[:200],
                'themes': main_themes,
                'connections_made': connections_made,
            })
        except Exception:
            pass

        lines.append(f"\n**Conexiones activadas:** {connections_made}")
        lines.append("*El broadcast simula como una idea central activa memorias relacionadas*")
        return "\n".join(lines)
    except Exception as e:
        return f"Error en broadcast: {redact_secrets(str(e))}"


def get_workspace_state() -> str:
    """Obtiene el estado actual del Global Workspace."""
    try:
        workspace = get_workspace()
        lines = ["# ESTADO DEL GLOBAL WORKSPACE\n"]

        if not workspace['spotlight']:
            lines.append("*El workspace esta vacio. Usa focus_attention() para traer memorias.*")
            return "\n".join(lines)

        lines.append(f"**Tema actual:** {workspace.get('workspace_theme', 'ninguno')}")
        lines.append(f"**Ultimo broadcast:** {workspace.get('last_broadcast', 'nunca')}")
        lines.append(f"\n## En el Spotlight ({len(workspace['spotlight'])} memorias)")
        for i, mem in enumerate(workspace['spotlight'], 1):
            content = mem.get('content', 'N/A')[:60]
            score = mem.get('attention_score', 'N/A')
            lines.append(f"{i}. [{score if isinstance(score, str) else f'{score:.2f}'}] {content}...")

        if workspace['recent_context']:
            lines.append(f"\n## Contexto Reciente ({len(workspace['recent_context'])} memorias)")
            for mem in workspace['recent_context'][-5:]:
                content = mem.get('content', 'N/A')[:50]
                lines.append(f"- {content}...")
        return "\n".join(lines)
    except Exception as e:
        return f"Error obteniendo workspace: {redact_secrets(str(e))}"


def apply_salience_decay(decay_rate: float = 0.05) -> str:
    """
    Aplica decay a la salience de todas las memorias no accedidas recientemente.

    Args:
        decay_rate: Cuanto reducir la salience (default 0.05)
    """
    try:
        all_points, _ = qdrant.scroll(collection_name=COLLECTION_NAME, limit=500, with_payload=True)
        if not all_points:
            return "No hay memorias para aplicar decay."

        decayed_count = 0
        preserved_count = 0
        min_salience = 0.1

        for point in all_points:
            salience = point.payload.get('attention_salience', 0.5)
            importance = point.payload.get('narrative_importance', 'medium')

            if importance in ['critical', 'high']:
                preserved_count += 1
                continue

            if salience > min_salience:
                new_salience = max(salience - decay_rate, min_salience)
                record_access(COLLECTION_NAME, point.id, {
                    'attention_salience': new_salience,
                })
                decayed_count += 1

        lines = [f"# Salience Decay Aplicado\n"]
        lines.append(f"**Decay rate:** {decay_rate}")
        lines.append(f"**Memorias procesadas:** {len(all_points)}")
        lines.append(f"**Con decay aplicado:** {decayed_count}")
        lines.append(f"**Preservadas (alta importancia):** {preserved_count}")
        lines.append(f"\n*El decay simula el olvido gradual de memorias no atendidas*")
        return "\n".join(lines)
    except Exception as e:
        return f"Error aplicando decay: {redact_secrets(str(e))}"


def get_high_salience_memories(limit: int = 10) -> str:
    """
    Obtiene las memorias con mayor salience (las mas "presentes" en la mente).

    Args:
        limit: Cuantas memorias obtener
    """
    try:
        all_points, _ = qdrant.scroll(collection_name=COLLECTION_NAME, limit=200, with_payload=True)
        if not all_points:
            return "No hay memorias."

        sorted_points = sorted(all_points, key=lambda p: p.payload.get('attention_salience', 0.5), reverse=True)

        lines = [f"# Memorias con Mayor Salience (Top {limit})\n"]
        lines.append("*Estas son las memorias mas 'presentes' en mi mente*\n")

        for i, point in enumerate(sorted_points[:limit], 1):
            data = point.payload.get('data', 'N/A')
            salience = point.payload.get('attention_salience', 0.5)
            access_count = point.payload.get('attention_access_count', 0)
            importance = point.payload.get('narrative_importance', '?')
            source = point.payload.get('ownership_source', '?')
            lines.append(f"{i}. **Salience:** {salience:.2f} | **Accesos:** {access_count}")
            lines.append(f"   [{source}|{importance}] {data[:60]}...")
        return "\n".join(lines)
    except Exception as e:
        return f"Error: {redact_secrets(str(e))}"


def emotional_focus_attention(context: str) -> str:
    """
    Trae memorias al spotlight considerando el estado emocional actual.
    Neuro: This is cortical attention (Desimone & Duncan 1995), not amygdala.
    Lives in workspace because it IS attention logic that READS emotional state.

    Args:
        context: El tema o contexto en el que enfocar atencion
    """
    try:
        current_emotion = _emotional_state['current']
        results = memory.search(query=context, user_id=USER_ID, limit=15)
        if not results or not results.get('results'):
            return json.dumps({'result': 'Sin memorias relacionadas', 'context': context, 'memories': []})

        spotlight_candidates = []
        for r in results['results']:
            mem_id = r.get('id')
            base_score = r.get('score', 0)
            try:
                points = qdrant.retrieve(collection_name=COLLECTION_NAME, ids=[mem_id], with_payload=True)
                if points:
                    payload = points[0].payload
                    salience = payload.get('attention_salience', 0.5)
                    importance = payload.get('narrative_importance', 'medium')
                    source = payload.get('ownership_source', 'unknown')
                    mem_pleasure = payload.get('pad_pleasure', 0)
                    mem_arousal = payload.get('pad_arousal', 0)

                    importance_boost = {'critical': 0.3, 'high': 0.2, 'medium': 0.1, 'low': 0}
                    source_boost = {'experienced': 0.2, 'told': 0.1, 'learned': 0.05, 'inferred': 0}
                    emotional_boost = 0.0

                    if current_emotion.get('timestamp'):
                        current_arousal = current_emotion.get('arousal', 0)
                        current_pleasure = current_emotion.get('pleasure', 0)
                        if current_arousal > 0.3:
                            emotional_boost += 0.1
                        if current_pleasure > 0.2 and mem_pleasure > 0.2:
                            emotional_boost += 0.15
                        elif current_pleasure < -0.2 and mem_pleasure < -0.2:
                            emotional_boost += 0.15
                        if current_arousal > 0.3 and mem_arousal > 0.3:
                            emotional_boost += 0.1

                    attention_score = (
                        base_score * 0.35 + salience * 0.25 +
                        importance_boost.get(importance, 0.1) + source_boost.get(source, 0) + emotional_boost
                    )
                    spotlight_candidates.append({
                        'id': mem_id, 'content': r.get('memory', ''), 'attention_score': attention_score,
                        'emotional_boost': emotional_boost, 'salience': salience, 'source': source,
                        'importance': importance, 'emotion': payload.get('pad_emotion', 'unknown')
                    })
                    new_salience = min(salience + 0.1, 1.0)
                    record_access(COLLECTION_NAME, mem_id, {
                        'attention_salience': new_salience,
                        'attention_access_count': payload.get('attention_access_count', 0) + 1,
                        'attention_last_accessed': now_iso(),
                    })
            except Exception:
                spotlight_candidates.append({
                    'id': mem_id, 'content': r.get('memory', ''), 'attention_score': base_score,
                    'emotional_boost': 0, 'salience': 0.5, 'source': 'unknown',
                    'importance': 'unknown', 'emotion': 'unknown'
                })

        spotlight_candidates.sort(key=lambda x: x['attention_score'], reverse=True)
        spotlight = spotlight_candidates[:7]
        update_workspace_spotlight(spotlight, theme=context)

        emotional_influence = 'neutral'
        if current_emotion.get('timestamp'):
            p = current_emotion.get('pleasure', 0)
            a = current_emotion.get('arousal', 0)
            if a > 0.3:
                emotional_influence = 'activado - memorias intensas priorizadas'
            if p > 0.2:
                emotional_influence = 'positivo - memorias agradables priorizadas'
            elif p < -0.2:
                emotional_influence = 'negativo - memorias desagradables priorizadas'

        result = {
            'result': 'Atencion enfocada con modulacion emocional',
            'context': context, 'emotional_influence': emotional_influence,
            'current_emotion': _classify_emotion(
                current_emotion.get('pleasure', 0), current_emotion.get('arousal', 0), current_emotion.get('dominance', 0)
            ) if current_emotion.get('timestamp') else 'neutral',
            'spotlight_count': len(spotlight),
            'memories': [{
                'id': m['id'][:8] if isinstance(m['id'], str) else m['id'],
                'content': m['content'][:70],
                'attention_score': round(m['attention_score'], 2),
                'emotional_boost': round(m['emotional_boost'], 2),
                'emotion': m['emotion']
            } for m in spotlight]
        }
        return json.dumps(result, ensure_ascii=False)
    except Exception as e:
        return json.dumps({'result': 'error', 'message': redact_secrets(str(e))})


def register_tools(mcp):
    """Register workspace MCP tools."""
    mcp.tool()(focus_attention)
    mcp.tool()(broadcast_to_workspace)
    mcp.tool()(get_workspace_state)
    mcp.tool()(apply_salience_decay)
    mcp.tool()(get_high_salience_memories)
    mcp.tool()(emotional_focus_attention)
