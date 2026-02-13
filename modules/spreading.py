"""
Codi Memory - Spreading Activation (Fase 3)
BFS propagation over outgoing edges with fan-effect, hop decay, and multi-path accumulation.

Limitation: Only outgoing edges (from payload). Incoming edges (memories pointing TO you) deferred to Fase 4.
Field: attention_salience (float, default 0.5, clamped to [FLOOR..CAP]).
"""

import json
from modules.config import (
    qdrant, memory, COLLECTION_NAME, USER_ID,
    now_iso,
    SPREAD_DEFAULT_FACTOR, SPREAD_DEFAULT_DEPTH,
    SPREAD_MIN_ACTIVATION, SPREAD_MAX_NEIGHBORS,
    SPREAD_SALIENCE_CAP, SPREAD_SALIENCE_FLOOR,
)
from modules.utils import resolve_memory_id


# ============================================================
# HELPERS
# ============================================================

def _clamp_salience(value: float) -> float:
    """Clamp salience to [FLOOR..CAP]."""
    return max(SPREAD_SALIENCE_FLOOR, min(SPREAD_SALIENCE_CAP, float(value)))


def _get_neighbors(point_id: str, payload: dict) -> list:
    """
    Extract connected IDs from payload outgoing edges.
    Order: related_to, consolidated_with, broadcast_received_from.
    Deduplicates, excludes self, caps at SPREAD_MAX_NEIGHBORS.
    """
    neighbors = []
    seen = {point_id}

    # related_to (str)
    rt = payload.get('related_to')
    if rt and isinstance(rt, str) and rt not in seen:
        neighbors.append(rt)
        seen.add(rt)

    # consolidated_with (list[str])
    cw = payload.get('consolidated_with')
    if cw and isinstance(cw, list):
        for cid in cw:
            if isinstance(cid, str) and cid not in seen:
                neighbors.append(cid)
                seen.add(cid)

    # broadcast_received_from (str)
    brf = payload.get('broadcast_received_from')
    if brf and isinstance(brf, str) and brf not in seen:
        neighbors.append(brf)
        seen.add(brf)

    return neighbors[:SPREAD_MAX_NEIGHBORS]


# ============================================================
# CORE ENGINE
# ============================================================

def _spread_activation(seed_ids: list, depth: int = SPREAD_DEFAULT_DEPTH,
                       factor: float = SPREAD_DEFAULT_FACTOR,
                       seed_boost: float = 0.0) -> dict:
    """
    BFS spreading activation over outgoing edges.

    Args:
        seed_ids: Starting point IDs (full UUIDs)
        depth: Max hops to propagate
        factor: Decay factor per hop (energy * factor / fan)
        seed_boost: Extra salience boost applied to seeds (0 = no boost)

    Returns:
        {affected, max_depth_reached, total_nodes_visited, updates: {id: new_salience}}
    """
    delta_map = {}        # id -> total accumulated delta
    payload_cache = {}    # id -> payload dict
    expanded = set()      # IDs already expanded (neighbors extracted)
    hop_delta = {}        # hop_number -> {id -> delta received in this hop}

    # 1. Batch retrieve seed payloads
    valid_seeds = []
    try:
        pts = qdrant.retrieve(collection_name=COLLECTION_NAME, ids=seed_ids, with_payload=True)
        if pts:
            for p in pts:
                pid = str(p.id)
                payload_cache[pid] = p.payload or {}
                valid_seeds.append(pid)
    except Exception:
        return {'affected': 0, 'max_depth_reached': 0, 'total_nodes_visited': 0, 'updates': {}}

    if not valid_seeds:
        return {'affected': 0, 'max_depth_reached': 0, 'total_nodes_visited': 0, 'updates': {}}

    # 2. Apply seed_boost
    if seed_boost > 0:
        for sid in valid_seeds:
            delta_map[sid] = delta_map.get(sid, 0) + seed_boost

    # 3. BFS propagation
    frontier = set(valid_seeds)
    max_depth_reached = 0

    for hop in range(1, depth + 1):
        hop_delta[hop] = {}
        frontier_next = set()

        for node in frontier:
            expanded.add(node)

            # Determine energy this node propagates
            if hop == 1:
                # Seeds propagate their current salience (+ seed_boost if applied)
                base_sal = payload_cache.get(node, {}).get('attention_salience', 0.5)
                node_energy = base_sal + delta_map.get(node, 0)
            else:
                # Non-seeds propagate what they received in the previous hop
                node_energy = hop_delta.get(hop - 1, {}).get(node, 0)
                if node_energy <= 0:
                    continue

            neighbors = _get_neighbors(node, payload_cache.get(node, {}))
            fan = len(neighbors)
            if fan == 0:
                continue

            spread_delta = (node_energy * factor) / fan
            if spread_delta < SPREAD_MIN_ACTIVATION:
                continue

            for nb in neighbors:
                # Accumulate in hop_delta
                hop_delta[hop][nb] = hop_delta[hop].get(nb, 0) + spread_delta
                # Accumulate in total delta_map
                delta_map[nb] = delta_map.get(nb, 0) + spread_delta

                if nb not in expanded:
                    frontier_next.add(nb)

        # Batch fetch payloads for new frontier nodes
        unfetched = [nid for nid in frontier_next if nid not in payload_cache]
        if unfetched:
            try:
                pts = qdrant.retrieve(collection_name=COLLECTION_NAME, ids=unfetched, with_payload=True)
                if pts:
                    for p in pts:
                        payload_cache[str(p.id)] = p.payload or {}
            except Exception:
                pass

        frontier = frontier_next
        if frontier_next:
            max_depth_reached = hop

    # 4. Build updates (exclude seeds if seed_boost == 0 to avoid double-boosting)
    updates = {}
    seed_set = set(valid_seeds)

    for mid, total_delta in delta_map.items():
        if seed_boost == 0 and mid in seed_set:
            continue

        old_sal = payload_cache.get(mid, {}).get('attention_salience', 0.5)
        new_sal = _clamp_salience(old_sal + total_delta)

        if abs(new_sal - old_sal) >= 0.01:
            updates[mid] = new_sal

    # 5. Batch update Qdrant
    for mid, new_sal in updates.items():
        try:
            qdrant.set_payload(
                collection_name=COLLECTION_NAME,
                payload={
                    'attention_salience': new_sal,
                    'attention_last_accessed': now_iso()
                },
                points=[mid]
            )
        except Exception:
            pass

    return {
        'affected': len(updates),
        'max_depth_reached': max_depth_reached,
        'total_nodes_visited': len(expanded),
        'updates': updates
    }


# ============================================================
# MCP TOOLS
# ============================================================

def register_tools(mcp):

    @mcp.tool()
    def spread_activation(memory_id_or_query: str, depth: int = SPREAD_DEFAULT_DEPTH,
                          factor: float = SPREAD_DEFAULT_FACTOR) -> str:
        """
        Propaga activacion desde una memoria hacia sus vecinas (spreading activation).
        Usa BFS con fan-effect y decay por hop. Solo edges salientes (outgoing).

        Args:
            memory_id_or_query: ID de memoria (parcial o completo) o query de texto
            depth: Profundidad maxima de propagacion (1-3, default 2)
            factor: Factor de decay por hop (0.1-1.0, default 0.7)
        """
        try:
            depth = max(1, min(3, int(depth)))
            factor = max(0.1, min(1.0, float(factor)))

            # Heuristic: if input has spaces or > 40 chars, treat as query
            seed_ids = []
            if ' ' in memory_id_or_query or len(memory_id_or_query) > 40:
                # Query mode: search for seeds
                results = memory.search(query=memory_id_or_query, user_id=USER_ID, limit=3)
                if results and results.get('results'):
                    for r in results['results']:
                        rid = r.get('id')
                        if rid:
                            seed_ids.append(rid)
            else:
                # ID mode: resolve
                full_id = resolve_memory_id(memory_id_or_query)
                if full_id:
                    seed_ids = [full_id]

            if not seed_ids:
                return f"No encontre memorias para '{memory_id_or_query}'"

            result = _spread_activation(seed_ids, depth=depth, factor=factor, seed_boost=0.1)

            lines = ["# SPREADING ACTIVATION\n"]
            lines.append(f"**Seeds:** {len(seed_ids)}")
            lines.append(f"**Depth:** {depth} | **Factor:** {factor}")
            lines.append(f"**Nodos visitados:** {result['total_nodes_visited']}")
            lines.append(f"**Memorias afectadas:** {result['affected']}")
            lines.append(f"**Profundidad alcanzada:** {result['max_depth_reached']}\n")

            if result.get('updates'):
                lines.append("## Top cambios de salience")
                sorted_updates = sorted(result['updates'].items(), key=lambda x: -x[1])[:10]
                for mid, sal in sorted_updates:
                    lines.append(f"- [{mid[:8]}] salience -> {sal:.3f}")
            else:
                lines.append("*Sin cambios de salience (memorias sin vecinos o delta muy pequeno)*")

            return "\n".join(lines)
        except Exception as e:
            return f"Error en spreading activation: {str(e)}"

    @mcp.tool()
    def get_activation_map(topic_or_id: str) -> str:
        """
        Muestra el mapa de activacion de una memoria: su salience y la de sus vecinos directos.
        Solo edges salientes (outgoing). Direction = OUT.

        Args:
            topic_or_id: ID de memoria (parcial o completo) o query de texto
        """
        try:
            # Resolve seeds
            seed_ids = []
            if ' ' in topic_or_id or len(topic_or_id) > 40:
                results = memory.search(query=topic_or_id, user_id=USER_ID, limit=3)
                if results and results.get('results'):
                    for r in results['results']:
                        rid = r.get('id')
                        if rid:
                            seed_ids.append(rid)
            else:
                full_id = resolve_memory_id(topic_or_id)
                if full_id:
                    seed_ids = [full_id]

            if not seed_ids:
                return f"No encontre memorias para '{topic_or_id}'"

            # Fetch seed payloads
            lines = ["# ACTIVATION MAP\n"]
            lines.append(f"**Direction:** OUT (solo edges salientes)")
            lines.append(f"**Limitacion:** Incoming edges no incluidos (Fase 4)\n")

            try:
                pts = qdrant.retrieve(collection_name=COLLECTION_NAME, ids=seed_ids, with_payload=True)
            except Exception:
                return "Error conectando a Qdrant"

            if not pts:
                return "No se encontraron las memorias seed"

            for p in pts:
                pid = str(p.id)
                payload = p.payload or {}
                sal = _clamp_salience(payload.get('attention_salience', 0.5))
                content = (payload.get('data', '') or '')[:60]

                lines.append(f"## Seed: [{pid[:8]}]")
                lines.append(f"**Content:** {content}...")
                lines.append(f"**Salience:** {sal:.3f}")

                neighbors = _get_neighbors(pid, payload)
                if not neighbors:
                    lines.append("*Sin vecinos (outgoing)*\n")
                    continue

                # Fetch neighbor payloads
                lines.append(f"**Vecinos (OUT):** {len(neighbors)}")
                try:
                    nb_pts = qdrant.retrieve(collection_name=COLLECTION_NAME, ids=neighbors, with_payload=True)
                    nb_map = {str(np.id): np.payload or {} for np in (nb_pts or [])}
                except Exception:
                    nb_map = {}

                for nb_id in neighbors:
                    nb_payload = nb_map.get(nb_id, {})
                    nb_sal = _clamp_salience(nb_payload.get('attention_salience', 0.5))
                    nb_content = (nb_payload.get('data', '') or '')[:50]
                    # Determine edge type
                    edge = "related_to"
                    cw = payload.get('consolidated_with', [])
                    if isinstance(cw, list) and nb_id in cw:
                        edge = "consolidated_with"
                    elif payload.get('broadcast_received_from') == nb_id:
                        edge = "broadcast_from"
                    lines.append(f"  - [{nb_id[:8]}] sal:{nb_sal:.3f} ({edge}) {nb_content}...")

                lines.append("")

            return "\n".join(lines)
        except Exception as e:
            return f"Error en activation map: {str(e)}"
