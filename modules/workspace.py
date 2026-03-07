"""
Codi Memory - Workspace module.
GWT spotlight, attention, salience, emotional focus attention.
emotional_focus_attention lives HERE (cortical attention, Desimone & Duncan 1995),
not in emotion.py — it READS emotional state as input but IS attention logic.
"""

import json

from modules.config import (
    USER_ID,
    _emotional_state, now_iso,
)
from modules.pg_store import pg
from modules.secret_redact import redact_secrets
from modules.memory_smart import search_with_fts_content
from modules.utils import (
    resolve_memory_id,
    _classify_emotion,
)
from modules.activation import compute_unified_activation
from modules.competition import CompetitionCandidate, run_workspace_competition

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
    'workspace_theme': None,
    # GWT-6: Cognitive cycle tracking (Baars 1988, LIDA Franklin 2006)
    'cycle_count': 0,
    'cycle_history': [],          # Last 10 winners [{cycle, theme, content, timestamp}]
    'last_cycle_timestamp': None,
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


def record_workspace_cycle(theme: str, winner_content: str, winning_domain: str = "unknown") -> None:
    """Record a cognitive cycle for seriality tracking (GWT-6).

    Baars 1988: Consciousness is serial -- one content at a time in the spotlight.
    LIDA (Franklin 2006): Cognitive cycles are the fundamental unit of processing.
    S3-02: Records winning_domain for Phi_E integration measurement.

    Args:
        theme: Topic of the winning content
        winner_content: Text of the winner (truncated to 100 chars)
        winning_domain: Source domain of the winner (for Phi_E computation)
    """
    global _global_workspace
    now = now_iso()

    _global_workspace['cycle_count'] += 1
    _global_workspace['last_cycle_timestamp'] = now

    cycle_entry = {
        'cycle': _global_workspace['cycle_count'],
        'theme': theme,
        'content': winner_content[:100],
        'timestamp': now,
        'winning_domain': winning_domain,  # S3-02: For Phi_E
    }
    _global_workspace['cycle_history'].append(cycle_entry)
    # Keep last 10 cycles
    _global_workspace['cycle_history'] = _global_workspace['cycle_history'][-10:]


def get_coalition_timeseries() -> list:
    """Return workspace cycle history for Phi_E computation (S3-02).

    Barrett & Seth 2011: approximate Phi on coalition time-series.
    Returns list of dicts with theme, winning_domain, timestamp.
    """
    return list(_global_workspace.get('cycle_history', []))


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
        results = search_with_fts_content(query=context, user_id=USER_ID, limit=limit * 2)

        if not results or not results.get('results'):
            return f"No encontre memorias relacionadas con: {context}"

        # WIRING-6: Score candidates with unified activation scorer
        competition_candidates = []

        for r in results['results']:
            mem_id = r.get('id')
            base_score = r.get('score', 0)
            try:
                points = pg.get_by_ids([mem_id])
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
                    pg.update_payload(mem_id, {
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

        # GWT-6: Record cognitive cycle (S3-02: include winning domain for Phi_E)
        if spotlight:
            winning_domain = comp_result.winners[0].source_domain if comp_result.winners else "unknown"
            record_workspace_cycle(context, spotlight[0].get('content', ''), winning_domain=winning_domain)

        # Proposal #67 Fix 4: Emit retrieval event for focus_attention spotlight winners
        try:
            from modules.events import event_bus, Events
            winner_ids = [m['id'] for m in spotlight if m.get('id')]
            if winner_ids:
                event_bus.emit(Events.MEMORY_RETRIEVED, {
                    'query': context,
                    'result_count': len(spotlight),
                    'episodic_count': len(spotlight),
                    'semantic_count': 0,
                    'top_activation': spotlight[0].get('attention_score', 0.5) if spotlight else 0,
                    'retrieved_ids': winner_ids,
                    'source': 'focus_attention',
                })
        except Exception:
            pass

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

        points = pg.get_by_ids([full_id])
        if not points:
            return f"No encontre memoria con ID {full_id}"

        payload = points[0].payload
        main_content = payload.get('data', '')
        main_themes = payload.get('narrative_themes', [])

        lines = [f"# BROADCAST - Memoria al centro del workspace\n"]
        lines.append(f"**Contenido:** {main_content[:80]}...")
        lines.append(f"**Temas:** {', '.join(main_themes) if main_themes else 'ninguno'}\n")

        pg.update_payload(full_id, {
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

        # GWT-6: Record cognitive cycle
        record_workspace_cycle(
            main_themes[0] if main_themes else "broadcast",
            main_content,
        )

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

        # GWT-5: Recruitment cycle -- modules react to broadcast (Baars 1988)
        try:
            from modules.events import event_bus, Events
            event_bus.emit(Events.WORKSPACE_RECRUITMENT, {
                'broadcast_content': main_content[:200],
                'broadcast_theme': main_themes[0] if main_themes else 'unknown',
                'memory_id': full_id,
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
        lines.append(f"**Ciclos cognitivos:** {workspace.get('cycle_count', 0)}")
        if workspace.get('cycle_history'):
            last = workspace['cycle_history'][-1]
            lines.append(f"**Ultimo ciclo:** #{last['cycle']} - {last['theme']}")
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
    Aplica decay a la salience usando FadeMem importance-modulated curves.

    Uses proper forgetting curves (Ebbinghaus 1885, Wixted & Ebbesen 1991)
    instead of flat subtraction. Decay rate varies by importance, consolidation
    status, and emotional arousal at encoding.

    Falls back to flat decay for memories without timing data.

    Args:
        decay_rate: Scaling factor for decay intensity (default 0.05).
                    Maps to decay_multiplier = decay_rate / 0.05 for FadeMem.
    """
    from modules.forgetting import (
        compute_fadem_strength, compute_fadem_strength_ss_rs,
        _get_hours_since_access,
        _get_emotional_arousal, _is_consolidated, _get_memory_type,
        FADEM_FLOOR,
    )

    try:
        decayed_count = 0
        preserved_count = 0
        fadem_count = 0
        total_scanned = 0

        # Map decay_rate to FadeMem multiplier (0.05 = 1.0x, 0.03 = 0.6x)
        decay_multiplier = decay_rate / 0.05

        # Canon v2, S1-5: Load causal chain members for reduced decay
        try:
            from modules.spreading import get_chain_member_ids
            _chain_members = get_chain_member_ids()
        except Exception:
            _chain_members = set()

        # Pre-filter: only scroll memories that can actually be decayed
        # (salience above floor, not critical importance)
        # Proposal #66 Fix 1: Only critical immune. High decays via FadeMem lambda.
        # Bjork & Bjork 1992: storage strength != retrieval strength immunity.
        # Range filter (attention_salience > FADEM_FLOOR) applied in Python post-filter.

        # Paginated scroll (same pattern as consolidation.py:208-267)
        offset = None
        MAX_SCROLL = 500  # reduced for remote DB latency
        pending_updates = []  # batch updates to avoid N+1 round-trips

        while total_scanned < MAX_SCROLL:
            batch, next_offset = pg.scroll(
                filters={"narrative_importance__not": "critical"},
                limit=200,
                is_semantic=False,
                offset=offset,
            )
            if not batch:
                break

            for point in batch:
                salience = point.payload.get('attention_salience', 0.5)
                # Post-filter: skip memories at or below FADEM_FLOOR (Range replacement)
                if salience <= FADEM_FLOOR:
                    continue
                importance = point.payload.get('narrative_importance', 'medium')

                # K.1.4: Read SS/RS from payload (defaults for legacy memories)
                ss = float(point.payload.get('storage_strength', 0.3) or 0.3)
                rs = float(point.payload.get('retrieval_strength', salience) or salience)

                # Try SS/RS dual strength model (Bjork & Bjork 1992)
                hours = _get_hours_since_access(point.payload)
                if hours is not None:
                    new_ss, new_rs, effective = compute_fadem_strength_ss_rs(
                        ss=ss, rs=rs,
                        hours_since_access=hours,
                        importance=importance,
                        consolidated=_is_consolidated(point.payload),
                        memory_type=_get_memory_type(point.payload),
                        emotional_arousal=_get_emotional_arousal(point.payload),
                    )
                    new_salience = effective
                    fadem_count += 1
                else:
                    # Flat fallback for memories without timing data
                    new_salience = max(FADEM_FLOOR, salience - decay_rate)
                    new_ss = ss
                    new_rs = max(FADEM_FLOOR, rs - decay_rate)

                # Canon v2, S1-5: Chain members get halved decay
                if str(point.id) in _chain_members:
                    new_salience = salience + (new_salience - salience) * 0.5
                    new_rs = rs + (new_rs - rs) * 0.5

                if new_salience < salience:
                    pending_updates.append((str(point.id), new_salience, new_ss, new_rs))
                    decayed_count += 1

            total_scanned += len(batch)
            if not next_offset:
                break
            offset = next_offset

        # Batch update all decayed memories in one transaction
        if pending_updates:
            from modules.config_pg import get_conn as _pg_conn
            from datetime import datetime as _dt
            _now = _dt.now()
            with _pg_conn() as conn:
                for mid, sal, ss_val, rs_val in pending_updates:
                    conn.execute(
                        "UPDATE memories SET activation_score = %s, "
                        "storage_strength = %s, retrieval_strength = %s, "
                        "updated_at = %s WHERE id = %s",
                        (sal, ss_val, rs_val, _now, mid),
                    )

        if total_scanned == 0:
            return "No hay memorias para aplicar decay."

        lines = [f"# Salience Decay Aplicado (FadeMem)\n"]
        lines.append(f"**Decay multiplier:** {decay_multiplier:.2f}x")
        lines.append(f"**Memorias procesadas:** {total_scanned}")
        lines.append(f"**Con decay aplicado:** {decayed_count}")
        lines.append(f"**FadeMem curves:** {fadem_count}")
        lines.append(f"**Flat fallback:** {decayed_count - fadem_count}")
        lines.append(f"**Preservadas (filtradas):** critical/high excluded via pre-filter")
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
        all_points, _ = pg.scroll(filters={}, limit=200, is_semantic=False)
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
        results = search_with_fts_content(query=context, user_id=USER_ID, limit=15)
        if not results or not results.get('results'):
            return json.dumps({'result': 'Sin memorias relacionadas', 'context': context, 'memories': []})

        spotlight_candidates = []
        for r in results['results']:
            mem_id = r.get('id')
            base_score = r.get('score', 0)
            try:
                points = pg.get_by_ids([mem_id])
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
                    pg.update_payload(mem_id, {
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


def recalibrate_importance(
    max_scan: int = 200,
    critical_decay_days: int = 45,
    high_decay_days: int = 90,
) -> str:
    """Recalibrate importance based on access patterns.

    Proposal #61 Fix 2 (Craik & Lockhart 1972, Bjork & Bjork 1992):
    Importance should reflect actual processing depth, not just encoding label.
    Memories tagged critical/high but never re-accessed have shallow effective
    processing and should lose their privileged status over time.

    Downgrades:
      critical -> high: if no access in critical_decay_days AND
                        not a checkpoint with tipo_momento=momento_personal
      high -> medium:   if no access in high_decay_days AND
                        not source=experienced with emotional_weight >= 0.8

    Never downgrades below medium (safety floor).
    """
    from datetime import datetime, timedelta
    import logging

    logger = logging.getLogger(__name__)
    downgraded_crit = 0
    downgraded_high = 0
    scanned = 0
    now = datetime.now()
    critical_cutoff = (now - timedelta(days=critical_decay_days)).isoformat()
    high_cutoff = (now - timedelta(days=high_decay_days)).isoformat()

    # Collect batch updates to avoid N+1 round-trips to remote PG
    downgrade_to_high = []
    downgrade_to_medium = []

    # --- Scan critical memories ---
    offset = None
    while scanned < max_scan:
        batch, next_offset = pg.scroll(
            filters={"importance": "critical"},
            limit=100,
            is_semantic=False,
            offset=offset,
        )
        if not batch:
            break

        for point in batch:
            scanned += 1
            payload = point.payload

            # Protected: legitimate critical content (personal moments)
            meta = payload.get('metadata', {})
            if meta.get('tipo_momento') == 'momento_personal':
                continue

            # S1-02: Bayesian importance uses SS + access pattern
            # Low SS = low real importance regardless of LLM label
            ss = float(payload.get('storage_strength', 1.0) or 1.0)
            timestamps = payload.get('access_timestamps', [])
            last_access = timestamps[-1] if timestamps else payload.get('created_at', '')

            # Effective decay threshold: SS < 2.0 means barely accessed, decay sooner
            effective_cutoff = critical_cutoff
            if ss < 2.0:
                # Low SS: halve the grace period (degrade sooner)
                half_days = critical_decay_days // 2
                effective_cutoff = (now - timedelta(days=half_days)).isoformat()

            if last_access and last_access < effective_cutoff:
                downgrade_to_high.append(str(point.id))
                downgraded_crit += 1

        if next_offset is None:
            break
        offset = next_offset

    # --- Scan high memories ---
    offset = None
    while scanned < max_scan:
        batch, next_offset = pg.scroll(
            filters={"importance": "high"},
            limit=100,
            is_semantic=False,
            offset=offset,
        )
        if not batch:
            break

        for point in batch:
            scanned += 1
            payload = point.payload

            # Protected: emotionally weighted memories
            ew = payload.get('experiential_emotional_weight', 0)
            try:
                if ew and float(ew) >= 0.8:
                    continue
            except (ValueError, TypeError):
                pass

            # S1-02: Same SS-aware decay for high->medium
            ss = float(payload.get('storage_strength', 1.0) or 1.0)
            timestamps = payload.get('access_timestamps', [])
            last_access = timestamps[-1] if timestamps else payload.get('created_at', '')

            effective_cutoff = high_cutoff
            if ss < 2.0:
                half_days = high_decay_days // 2
                effective_cutoff = (now - timedelta(days=half_days)).isoformat()

            if last_access and last_access < effective_cutoff:
                downgrade_to_medium.append(str(point.id))
                downgraded_high += 1

        if next_offset is None:
            break
        offset = next_offset

    # Batch update downgrades in one transaction
    if downgrade_to_high or downgrade_to_medium:
        from modules.config_pg import get_conn as _pg_conn
        with _pg_conn() as conn:
            if downgrade_to_high:
                conn.execute(
                    "UPDATE memories SET importance = 'high', updated_at = NOW() "
                    "WHERE id = ANY(%s::uuid[])",
                    (downgrade_to_high,),
                )
            if downgrade_to_medium:
                conn.execute(
                    "UPDATE memories SET importance = 'medium', updated_at = NOW() "
                    "WHERE id = ANY(%s::uuid[])",
                    (downgrade_to_medium,),
                )

    total_downgraded = downgraded_crit + downgraded_high
    if total_downgraded > 0:
        logger.info(f"[importance_recal] {downgraded_crit} critical→high, {downgraded_high} high→medium")

    return (
        f"# Importance Recalibration\n\n"
        f"**Scanned:** {scanned}\n"
        f"**Downgraded:** {total_downgraded} "
        f"({downgraded_crit} critical→high, {downgraded_high} high→medium)\n"
        f"**Criteria:** critical→high after {critical_decay_days}d, "
        f"high→medium after {high_decay_days}d without access"
    )


def register_tools(mcp):
    """Register workspace MCP tools."""
    mcp.tool()(focus_attention)
    mcp.tool()(broadcast_to_workspace)
    mcp.tool()(get_workspace_state)
    mcp.tool()(apply_salience_decay)
    mcp.tool()(get_high_salience_memories)
    mcp.tool()(emotional_focus_attention)
