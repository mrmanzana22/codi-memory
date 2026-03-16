"""
Codi Memory - Spreading Activation (Fase 4)
BFS propagation over bidirectional edges with fan-effect, hop decay,
multi-path accumulation, and lateral inhibition.

Fase 4 upgrades (Feb 26, 2026):
  - Bidirectional edges: outgoing (payload) + incoming (SQLite reverse index)
  - Lateral inhibition: k-WTA competitive selection (Desimone & Duncan 1995)

Field: attention_salience (float, default 0.5, clamped to [FLOOR..CAP]).
"""

import json
import logging
import math
import os
import sqlite3
from modules.config import (
    COLLECTION_NAME, USER_ID,
    now_iso, connect_fts,
    SPREAD_DEFAULT_FACTOR, SPREAD_DEFAULT_DEPTH,
    SPREAD_MIN_ACTIVATION, SPREAD_MAX_NEIGHBORS,
    SPREAD_SALIENCE_CAP, SPREAD_SALIENCE_FLOOR,
)
from modules.pg_store import pg
from modules.utils import resolve_memory_id
from modules.secret_redact import redact_secrets

_logger = logging.getLogger(__name__)

# SQLite edge index for bidirectional lookup — uses config FTS_DB_PATH
# (instance-aware: each tenant has its own SQLite directory)
from modules.config import FTS_DB_PATH as _EDGE_DB

# Lateral inhibition parameters (Desimone & Duncan 1995)
_INHIBITION_K = 5            # Top-k winners retain full activation
_INHIBITION_FACTOR = 0.3     # Losers get delta * this factor

# Edge-type weight multipliers (Canon v2 Sprint 1, CC-3/S2-5/S2-6)
# Pearl 2009: causal edges propagate; co-occurrence does NOT.
# 5 canonical types: causes, enables, prevents, co_occurs, confounded
_EDGE_TYPE_WEIGHT = {
    'causes': 1.0,        # A caused B -> strong propagation
    'enables': 0.8,       # A enabled B -> moderate propagation
    'prevents': -0.5,     # A prevents B -> INHIBITORY (S2-6)
    'co_occurs': 0.0,     # Mere co-occurrence = ZERO spreading (S2-5, item 1.4)
    'confounded': 0.0,    # Confounded = do not propagate
    'similarity': 0.5,    # Semantic similarity (payload: related_to, related_memories)
}


# ============================================================
# EDGE INDEX (SQLite reverse index for incoming edges)
# ============================================================

def _init_edge_table(conn):
    """Ensure spreading_edges table exists with strength column (Sprint 5.3)."""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS spreading_edges (
            from_id TEXT NOT NULL,
            to_id TEXT NOT NULL,
            edge_type TEXT DEFAULT 'co_occurs',
            strength REAL DEFAULT 0.5,
            last_seen TEXT,
            PRIMARY KEY (from_id, to_id)
        )
    """)
    try:
        conn.execute("CREATE INDEX IF NOT EXISTS idx_edges_to ON spreading_edges(to_id)")
    except sqlite3.OperationalError:
        pass
    # Idempotent column migrations (ALTER TABLE ADD COLUMN is no-op if exists)
    for col_sql in [
        "ALTER TABLE spreading_edges ADD COLUMN strength REAL DEFAULT 0.5",          # Sprint 5.3
        "ALTER TABLE spreading_edges ADD COLUMN discovery_source TEXT DEFAULT NULL",  # Sprint 7.4
        "ALTER TABLE spreading_edges ADD COLUMN directed INTEGER DEFAULT 0",         # Sprint 7.4
    ]:
        try:
            conn.execute(col_sql)
        except sqlite3.OperationalError:
            pass  # Column already exists
    conn.commit()


def _record_edges(conn, from_id: str, neighbor_ids: list, ts: str,
                   edge_type: str = "co_occurs", strength: float = None):
    """Record edges in SQLite with typed relationships (S2-05, Sprint 1).

    Edge types: causes, enables, prevents, co_occurs, confounded.
    Pearl 2009: causal structure requires directed typed edges.

    Sprint 5.3: strength encodes causal reliability (0-1).
    If not provided, defaults by edge type: causal=0.7, enables=0.5, else=0.3.
    """
    if strength is None:
        if edge_type == 'causes':
            strength = 0.7
        elif edge_type == 'enables':
            strength = 0.5
        else:
            strength = 0.3
    for to_id in neighbor_ids:
        conn.execute("""
            INSERT INTO spreading_edges (from_id, to_id, edge_type, strength, last_seen)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(from_id, to_id) DO UPDATE SET
                edge_type = excluded.edge_type,
                strength = excluded.strength,
                last_seen = excluded.last_seen
        """, (from_id, to_id, edge_type, strength, ts))
    conn.commit()


def is_causal_chain_member(point_id: str, fts_db_path: str = None) -> bool:
    """Check if a memory is part of a causal chain.

    A chain member has CAUSAL edges (causes/enables) both incoming and outgoing.
    Chain members must never be pruned/compressed (Canon v2, S1-5).

    Pearl 2009: causal chain members carry irreplaceable structural information.
    """
    if not fts_db_path:
        fts_db_path = _EDGE_DB

    if not os.path.exists(fts_db_path):
        return False

    try:
        conn = connect_fts(fts_db_path)
        _init_edge_table(conn)

        # Check outgoing causal edges (this node -> other)
        has_outgoing = conn.execute(
            "SELECT 1 FROM spreading_edges WHERE from_id = ? "
            "AND edge_type IN ('causes', 'enables') LIMIT 1",
            (str(point_id),)
        ).fetchone()

        if not has_outgoing:
            conn.close()
            return False

        # Check incoming causal edges (other -> this node)
        has_incoming = conn.execute(
            "SELECT 1 FROM spreading_edges WHERE to_id = ? "
            "AND edge_type IN ('causes', 'enables') LIMIT 1",
            (str(point_id),)
        ).fetchone()

        conn.close()
        return bool(has_incoming)
    except sqlite3.Error as exc:
        _logger.exception(
            "Failed causal chain membership lookup for point_id=%s db=%s",
            point_id,
            fts_db_path,
        )
        raise


def get_chain_member_ids(fts_db_path: str = None) -> set:
    """Batch query: get all memory IDs that are causal chain members.

    More efficient than calling is_causal_chain_member() per ID.
    Returns set of point_id strings.
    """
    if not fts_db_path:
        fts_db_path = _EDGE_DB

    if not os.path.exists(fts_db_path):
        return set()

    try:
        conn = connect_fts(fts_db_path)
        _init_edge_table(conn)

        # Nodes with outgoing causal edges
        outgoing = {r[0] for r in conn.execute(
            "SELECT DISTINCT from_id FROM spreading_edges "
            "WHERE edge_type IN ('causes', 'enables')"
        ).fetchall()}

        # Nodes with incoming causal edges
        incoming = {r[0] for r in conn.execute(
            "SELECT DISTINCT to_id FROM spreading_edges "
            "WHERE edge_type IN ('causes', 'enables')"
        ).fetchall()}

        conn.close()
        # Chain members = intersection (have both in and out)
        return outgoing & incoming
    except sqlite3.Error as exc:
        _logger.exception(
            "Failed batch causal chain member lookup for db=%s",
            fts_db_path,
        )
        raise


def _get_incoming_neighbors(conn, point_id: str, limit: int = None) -> list:
    """Find memories that point TO this node (reverse direction).

    Collins & Loftus 1975: activation spreads bidirectionally along
    associative links. Incoming edges represent memories that reference
    this node but aren't captured by outgoing payload fields.

    Returns list of (from_id, edge_type, strength) tuples for type-differentiated
    spreading (Canon v2, S2-5/S2-6). Sprint 5.3: includes strength.
    """
    if limit is None:
        limit = SPREAD_MAX_NEIGHBORS
    rows = conn.execute(
        "SELECT from_id, edge_type, COALESCE(strength, 0.5) FROM spreading_edges WHERE to_id = ? LIMIT ?",
        (point_id, limit)
    ).fetchall()
    return [(r[0], r[1] or 'co_occurs', float(r[2])) for r in rows]


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

    # related_memories (list[str]) -- auto-connected neighbors (Phase 5.5)
    rm = payload.get('related_memories')
    if rm and isinstance(rm, list):
        for rid in rm:
            if isinstance(rid, str) and rid not in seen:
                neighbors.append(rid)
                seen.add(rid)

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
        pts = pg.get_by_ids(seed_ids)
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

    # 3. BFS propagation (bidirectional: outgoing + incoming edges)
    frontier = set(valid_seeds)
    max_depth_reached = 0

    # Open SQLite connection for edge index (bidirectional lookup)
    edge_conn = None
    try:
        if os.path.exists(_EDGE_DB):
            edge_conn = connect_fts(_EDGE_DB)
            _init_edge_table(edge_conn)
    except Exception:
        pass

    ts = now_iso()

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

            # Bidirectional neighbors: outgoing (payload) + incoming (SQLite)
            outgoing = _get_neighbors(node, payload_cache.get(node, {}))

            # Build neighbor->edge_type+strength map (Canon v2, S2-5/S2-6)
            # Outgoing from payload default to 'similarity' strength=0.3
            nb_types = {nb_id: 'similarity' for nb_id in outgoing}
            nb_strength = {nb_id: 0.3 for nb_id in outgoing}  # Sprint 5.3

            incoming_typed = []
            if edge_conn:
                try:
                    incoming_typed = _get_incoming_neighbors(edge_conn, node)
                    # Record outgoing edges for future reverse lookups
                    if outgoing:
                        _record_edges(edge_conn, node, outgoing, ts, edge_type='similarity')
                except Exception:
                    pass

            # Merge and deduplicate (outgoing have priority for type)
            for inc_id, inc_type, inc_strength in incoming_typed:
                if inc_id not in nb_types and inc_id != node:
                    outgoing.append(inc_id)
                    nb_types[inc_id] = inc_type
                    nb_strength[inc_id] = inc_strength  # Sprint 5.3
            neighbors = outgoing[:SPREAD_MAX_NEIGHBORS]

            fan = len(neighbors)
            if fan == 0:
                continue

            # S0-02: Fan effect S-ln(fan) (G-INV-10). Was linear /fan (over-aggressive, kills fan>7).
            # Sub-linear: hubs still propagate. Collins & Loftus 1975.
            base_delta = (node_energy * factor) / (1.0 + math.log(fan))
            if base_delta < SPREAD_MIN_ACTIVATION:
                continue

            for nb in neighbors:
                # Edge-type weight (Canon v2 Sprint 1, CC-3/S2-5/S2-6)
                etype = nb_types.get(nb, 'co_occurs')
                weight = _EDGE_TYPE_WEIGHT.get(etype, 0.0)
                if weight == 0.0:
                    continue  # co_occurs/confounded = zero spreading (item 1.4)
                # Sprint 5.3: modulate by causal strength (Woodward 2003)
                strength = nb_strength.get(nb, 0.5)
                spread_delta = base_delta * weight * strength
                # For positive: skip if below min activation
                # For negative (PREVENTS): skip if suppression is negligible
                if abs(spread_delta) < SPREAD_MIN_ACTIVATION:
                    continue
                # Accumulate in hop_delta (negative = inhibitory, item 1.3)
                hop_delta[hop][nb] = hop_delta[hop].get(nb, 0) + spread_delta
                # Accumulate in total delta_map
                delta_map[nb] = delta_map.get(nb, 0) + spread_delta

                if nb not in expanded:
                    frontier_next.add(nb)

        # Batch fetch payloads for new frontier nodes
        unfetched = [nid for nid in frontier_next if nid not in payload_cache]
        if unfetched:
            try:
                pts = pg.get_by_ids(unfetched)
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

    # 4b. Lateral inhibition (Desimone & Duncan 1995)
    # k-WTA: top-k nodes retain full activation, rest get suppressed
    if len(updates) > _INHIBITION_K:
        sorted_nodes = sorted(updates.items(), key=lambda x: -x[1])
        inhibited = {}
        for i, (nid, sal) in enumerate(sorted_nodes):
            if i < _INHIBITION_K:
                inhibited[nid] = sal  # Winner: full
            else:
                # Suppress by pulling delta toward old salience
                old_sal = payload_cache.get(nid, {}).get('attention_salience', 0.5)
                inhibited[nid] = old_sal + (sal - old_sal) * _INHIBITION_FACTOR
        updates = {k: v for k, v in inhibited.items()
                   if abs(v - payload_cache.get(k, {}).get('attention_salience', 0.5)) >= 0.01}

    # 5. Close edge index connection
    if edge_conn:
        try:
            edge_conn.close()
        except Exception:
            pass

    # 6. Batch update Qdrant (via access_tracking aggregator)
    from modules.access_tracking import record_spreading
    record_spreading(COLLECTION_NAME, updates, last_accessed=now_iso())

    return {
        'affected': len(updates),
        'max_depth_reached': max_depth_reached,
        'total_nodes_visited': len(expanded),
        'updates': updates
    }


# ============================================================
# RECURRENT PROCESSING (Lamme 2006)
# ============================================================

def recurrent_cycle(seed_ids: list, cycles: int = 2, depth: int = 1,
                    factor: float = 0.6) -> dict:
    """True recurrent processing: output of one spreading cycle feeds next.

    Lamme 2006: Re-entrant processing creates stable representations
    through iterative feedback between processing stages. Unlike single-pass
    BFS (feedforward sweep), this feeds activated nodes back as seeds.

    Args:
        seed_ids: Starting point IDs (full UUIDs)
        cycles: Number of recurrent iterations (2-5, default 2)
        depth: BFS depth per cycle (default 1)
        factor: Decay factor per hop (default 0.6)

    Returns:
        {cycles_run, total_affected, stable (bool), updates_per_cycle}
    """
    cycles = max(1, min(5, int(cycles)))
    updates_per_cycle = []
    current_seeds = list(seed_ids)
    prev_top_nodes = set()
    stable = False

    for i in range(cycles):
        if not current_seeds:
            break

        result = _spread_activation(current_seeds, depth=depth, factor=factor)
        cycle_updates = result.get("updates", {})
        updates_per_cycle.append(len(cycle_updates))

        # Get top activated nodes from this cycle to use as next seeds
        if cycle_updates:
            sorted_nodes = sorted(cycle_updates.items(), key=lambda x: -x[1])
            top_nodes = set(nid for nid, _ in sorted_nodes[:3])

            # Check stability: same top nodes as previous cycle
            if top_nodes and top_nodes == prev_top_nodes:
                stable = True

            prev_top_nodes = top_nodes
            current_seeds = list(top_nodes)
        else:
            break

    total_affected = sum(updates_per_cycle)
    return {
        "cycles_run": len(updates_per_cycle),
        "total_affected": total_affected,
        "stable": stable,
        "updates_per_cycle": updates_per_cycle,
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
                results = pg.search(memory_id_or_query, limit=3, is_semantic=False)
                if results:
                    for r in results:
                        rid = r.get('id') if isinstance(r, dict) else getattr(r, 'id', None)
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
            return f"Error en spreading activation: {redact_secrets(str(e))}"

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
                results = pg.search(topic_or_id, limit=3, is_semantic=False)
                if results:
                    for r in results:
                        rid = r.get('id') if isinstance(r, dict) else getattr(r, 'id', None)
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
                pts = pg.get_by_ids(seed_ids)
            except Exception:
                return "Error conectando a pg_store"

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
                    nb_pts = pg.get_by_ids(neighbors)
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
            return f"Error en activation map: {redact_secrets(str(e))}"
