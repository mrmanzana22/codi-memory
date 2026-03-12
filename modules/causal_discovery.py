"""
Codi Memory - NOTEARS Causal Discovery (Sprint 7)
==================================================
Learns causal DAG structure from memory co-occurrence patterns.

Sprint 7 items:
  7.1: Co-occurrence matrix from attention_transitions + prediction_results
  7.2: NOTEARS core (augmented Lagrangian, L-BFGS-B)
  7.3: Edge threshold and extraction (typed by weight magnitude)
  7.4: Integration with existing spreading_edges (merge with LLM-classified)
  7.5: Periodic re-learning via sleep_loop tick

Theory (Zheng et al. 2018 NOTEARS):
  Converts NP-hard DAG learning to continuous optimization:
    min ||X - XW||_F^2 + lambda * ||W||_1  s.t. h(W) = 0
  where h(W) = trace(e^(W∘W)) - d = 0 enforces acyclicity.
  Gradient: dh/dW = 2 * (e^(W∘W))^T ∘ W

References:
  - Zheng et al. 2018: "DAGs with NO TEARS: Continuous Optimization for Structure Learning"
  - Pearl 2009: Causality — causal DAG as the ground truth of interventional queries
  - Woodward 2003: Making Things Happen — invariance under intervention

Created: 2026-02-28 (Sprint 7)
"""

import json
import logging
import os
import sqlite3
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
from modules.config import now_iso

_logger = logging.getLogger(__name__)

# NOTEARS hyperparameters
_LAMBDA = 0.1          # L1 regularization (sparsity penalty)
_MAX_ITER = 100        # Max L-BFGS-B iterations
_H_TOL = 1e-8          # Acyclicity tolerance
_EDGE_THRESHOLD = 0.1  # Minimum |w_ij| to extract as edge
_MIN_TRANSITIONS = 50  # Minimum new transitions before re-learning


# ============================================================
# 7.1: CO-OCCURRENCE MATRIX
# ============================================================

def _get_all_topics(conn: sqlite3.Connection) -> List[str]:
    """Get unique topics from attention_transitions and prediction_results."""
    topics = set()

    try:
        rows = conn.execute(
            "SELECT DISTINCT from_topic FROM attention_transitions WHERE from_topic IS NOT NULL"
        ).fetchall()
        topics.update(r[0] for r in rows if r[0])
    except Exception:
        pass

    try:
        rows = conn.execute(
            "SELECT DISTINCT topic FROM prediction_results WHERE topic IS NOT NULL"
        ).fetchall()
        topics.update(r[0] for r in rows if r[0])
    except Exception:
        pass

    # Filter out noise
    topics = {t for t in topics if t and t != "general" and len(t) > 2}
    return sorted(topics)


def build_cooccurrence_matrix(conn: sqlite3.Connection) -> Tuple[np.ndarray, List[str]]:
    """Sprint 7.1: Build N×N co-occurrence matrix from transition data.

    Cell (i,j) = count of times topic_i → topic_j appear in same session
    or within close temporal proximity (from attention_transitions).

    Returns:
        (X: N×N matrix, topics: list of topic names in index order)
    """
    topics = _get_all_topics(conn)
    n = len(topics)

    if n < 2:
        return np.zeros((0, 0)), []

    topic_idx = {t: i for i, t in enumerate(topics)}
    X = np.zeros((n, n), dtype=float)

    # Fill from attention_transitions (from_topic → to_topic)
    try:
        rows = conn.execute(
            "SELECT from_topic, to_topic, count FROM attention_transitions "
            "WHERE from_topic IS NOT NULL AND to_topic IS NOT NULL"
        ).fetchall()
        for from_t, to_t, count in rows:
            if from_t in topic_idx and to_t in topic_idx and from_t != to_t:
                i, j = topic_idx[from_t], topic_idx[to_t]
                X[i][j] += float(count or 1)
    except Exception:
        pass

    # Add co-occurrence within same session window (prediction_results proximity)
    try:
        rows = conn.execute(
            "SELECT topic, created_at FROM prediction_results "
            "WHERE topic IS NOT NULL ORDER BY created_at ASC LIMIT 500"
        ).fetchall()

        # Find pairs within 5-minute windows
        for k in range(len(rows)):
            t1_topic, t1_time = rows[k]
            if t1_topic not in topic_idx:
                continue
            for m in range(k + 1, min(k + 5, len(rows))):
                t2_topic, t2_time = rows[m]
                if t2_topic not in topic_idx or t2_topic == t1_topic:
                    continue
                try:
                    dt = abs((datetime.fromisoformat(t2_time) - datetime.fromisoformat(t1_time)).total_seconds())
                    if dt <= 300:  # 5-minute window
                        i, j = topic_idx[t1_topic], topic_idx[t2_topic]
                        X[i][j] += 0.5  # Weaker signal than direct transitions
                except Exception:
                    pass
    except Exception:
        pass

    # Normalize each row to [0, 1] for numerical stability
    row_maxes = X.max(axis=1, keepdims=True)
    row_maxes[row_maxes == 0] = 1.0
    X_norm = X / row_maxes

    _logger.info("Co-occurrence matrix: %d topics, %d non-zero entries",
                 n, int((X > 0).sum()))
    return X_norm, topics


# ============================================================
# 7.2: NOTEARS CORE ALGORITHM
# ============================================================

def _acyclicity_constraint(W: np.ndarray) -> float:
    """h(W) = trace(e^(W∘W)) - d.

    h(W) = 0 iff W is a DAG. Always >= 0.
    Zheng et al. 2018, Eq. (9).
    """
    d = W.shape[0]
    M = W * W  # element-wise square
    E = np.linalg.matrix_power(np.eye(d) + M / d, d)  # (I + M/d)^d approximates e^M
    return float(np.trace(E)) - d


def _acyclicity_gradient(W: np.ndarray) -> np.ndarray:
    """dh/dW = 2 * (e^(W∘W))^T ∘ W.

    Zheng et al. 2018: acyclicity constraint gradient for augmented Lagrangian.
    """
    d = W.shape[0]
    M = W * W
    E = np.linalg.matrix_power(np.eye(d) + M / d, d)
    return 2.0 * E.T * W


def run_notears(X: np.ndarray, lambda_l1: float = _LAMBDA,
                max_iter: int = _MAX_ITER) -> np.ndarray:
    """Sprint 7.2: NOTEARS algorithm — continuous DAG learning.

    Zheng et al. 2018: solve the continuous-constrained optimization:
        min ||X - XW||_F^2 / n + lambda * ||W||_1  s.t. h(W) = 0

    Uses augmented Lagrangian + L-BFGS-B for inner optimization.

    Args:
        X: N×N co-occurrence matrix (N topics)
        lambda_l1: Sparsity regularization
        max_iter: Max iterations per L-BFGS-B call

    Returns:
        W: N×N weighted adjacency matrix (DAG)
    """
    from scipy.optimize import minimize

    n, d = X.shape
    if n < 2:
        return np.zeros((d, d))

    # Objective: ||X - XW||_F^2 / n + lambda * ||W||_1
    def objective(w_flat):
        W = w_flat.reshape(d, d)
        # Set diagonal to 0 (no self-loops)
        np.fill_diagonal(W, 0)
        residual = X - X @ W
        loss = 0.5 * np.sum(residual ** 2) / n
        l1_pen = lambda_l1 * np.sum(np.abs(W))
        return loss + l1_pen

    def gradient(w_flat):
        W = w_flat.reshape(d, d)
        np.fill_diagonal(W, 0)
        residual = X - X @ W
        grad_loss = -X.T @ residual / n
        grad_l1 = lambda_l1 * np.sign(W)
        g = grad_loss + grad_l1
        np.fill_diagonal(g, 0)
        return g.flatten()

    # Augmented Lagrangian: add h(W) penalty
    rho = 1.0
    alpha = 0.0  # Lagrange multiplier
    W = np.zeros((d, d))

    for outer_iter in range(20):  # Outer Lagrangian iterations
        def aug_objective(w_flat):
            W_inner = w_flat.reshape(d, d)
            np.fill_diagonal(W_inner, 0)
            h = _acyclicity_constraint(W_inner)
            penalty = 0.5 * rho * h ** 2 + alpha * h
            return objective(w_flat) + penalty

        def aug_gradient(w_flat):
            W_inner = w_flat.reshape(d, d)
            np.fill_diagonal(W_inner, 0)
            h = _acyclicity_constraint(W_inner)
            dh = _acyclicity_gradient(W_inner)
            penalty_grad = (rho * h + alpha) * dh.flatten()
            return gradient(w_flat) + penalty_grad

        result = minimize(
            fun=aug_objective,
            x0=W.flatten(),
            jac=aug_gradient,
            method='L-BFGS-B',
            options={'maxiter': max_iter, 'ftol': 1e-8},
        )
        W = result.x.reshape(d, d)
        np.fill_diagonal(W, 0)

        h = _acyclicity_constraint(W)
        if h < _H_TOL:
            break

        alpha += rho * h
        rho = min(rho * 1.5, 100.0)

    _logger.info("NOTEARS: converged at h(W)=%.2e after %d outer iters",
                 _acyclicity_constraint(W), outer_iter + 1)
    return W


# ============================================================
# 7.3: EDGE THRESHOLD AND EXTRACTION
# ============================================================

def extract_edges(W: np.ndarray, topics: List[str],
                  threshold: float = _EDGE_THRESHOLD) -> List[Dict]:
    """Sprint 7.3: Extract edges from W matrix above threshold.

    Maps weight magnitude to edge types:
        |w| > 0.3 → 'causes' (strong causal signal)
        |w| > 0.1 → 'enables' (moderate causal signal)
        w < -0.1  → 'prevents' (inhibitory)

    Returns list of {from_topic, to_topic, weight, edge_type} dicts.
    """
    edges = []
    n = len(topics)

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            w = float(W[i][j])
            if abs(w) < threshold:
                continue

            if w < 0:
                etype = "prevents"
            elif w > 0.3:
                etype = "causes"
            else:
                etype = "enables"

            edges.append({
                "from_topic": topics[i],
                "to_topic": topics[j],
                "weight": round(w, 4),
                "edge_type": etype,
            })

    return edges


# ============================================================
# 7.4: INTEGRATION WITH SPREADING_EDGES
# ============================================================

def integrate_discovered_edges(edges: List[Dict], fts_db_path: str) -> int:
    """Sprint 7.4: Merge discovered topic-level edges into spreading_edges.

    For each topic pair with a discovered edge: find memory pairs between
    those topics and create/update edges. Only if no existing LLM-classified
    edge exists (to preserve higher-confidence manual classifications).

    Returns number of new edges created.
    """
    from modules.config import FTS_DB_PATH as _default_fts, connect_fts
    from modules.utils import now_iso
    from modules.spreading import _init_edge_table, _record_edges

    db = fts_db_path or _default_fts

    if not os.path.exists(db):
        return 0

    try:
        conn = connect_fts(db)
        _init_edge_table(conn)
        ts = now_iso()
        created = 0

        for edge in edges:
            from_topic = edge["from_topic"]
            to_topic = edge["to_topic"]
            etype = edge["edge_type"]
            weight = abs(edge["weight"])

            # Find memory IDs for these topics from memories_text
            from_ids = [r[0] for r in conn.execute(
                "SELECT DISTINCT memory_id FROM memories_text WHERE topic = ? LIMIT 10",
                (from_topic,)
            ).fetchall()]
            to_ids = [r[0] for r in conn.execute(
                "SELECT DISTINCT memory_id FROM memories_text WHERE topic = ? LIMIT 10",
                (to_topic,)
            ).fetchall()]

            if not from_ids or not to_ids:
                continue

            # For each pair: only create if no existing causal edge
            for from_id in from_ids[:3]:
                for to_id in to_ids[:3]:
                    existing = conn.execute(
                        "SELECT edge_type FROM spreading_edges WHERE from_id = ? AND to_id = ?",
                        (from_id, to_id)
                    ).fetchone()

                    if existing and existing[0] in ('causes', 'enables', 'prevents'):
                        continue  # Don't overwrite LLM-classified edges

                    _record_edges(conn, from_id, [to_id], ts,
                                  edge_type=etype, strength=weight)
                    created += 1

        # Record discovery source (mark with notears tag via new column if exists)
        try:
            for edge in edges:
                conn.execute(
                    "UPDATE spreading_edges SET discovery_source = 'notears' "
                    "WHERE edge_type = ? AND last_seen = ?",
                    (edge["edge_type"], ts)
                )
        except Exception:
            pass

        conn.close()
        _logger.info("NOTEARS integration: %d edges created", created)
        return created

    except Exception as e:
        _logger.error("NOTEARS integration error: %s", e)
        return 0


# ============================================================
# 7.5: STATE PERSISTENCE
# ============================================================

def _save_discovery_state(W: np.ndarray, topics: List[str], h_value: float,
                          n_edges: int, fts_db_path: str) -> None:
    """Persist NOTEARS W matrix to causal_discovery_state table.

    Table created by migration 021_sprint7_causal_discovery_state.sql.
    """
    try:
        from modules.config import connect_fts
        conn = connect_fts(fts_db_path)
        conn.execute("""
            INSERT INTO causal_discovery_state (w_matrix, topics, h_value, lambda_val, n_edges, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            json.dumps(W.tolist()),
            json.dumps(topics),
            float(h_value),
            _LAMBDA,
            n_edges,
            now_iso(),
        ))
        # Keep only last 5 states
        conn.execute("""
            DELETE FROM causal_discovery_state WHERE id NOT IN (
                SELECT id FROM causal_discovery_state ORDER BY created_at DESC LIMIT 5
            )
        """)
        conn.commit()
        conn.close()
    except Exception as e:
        _logger.debug("Failed to save discovery state: %s", e)


def _count_new_transitions(fts_db_path: str, since_ts: str) -> int:
    """Count new attention_transitions since last discovery run."""
    try:
        from modules.config import connect_fts
        conn = connect_fts(fts_db_path)
        count = conn.execute(
            "SELECT COUNT(*) FROM attention_transitions WHERE created_at > ?",
            (since_ts,)
        ).fetchone()[0]
        conn.close()
        return count
    except Exception:
        return 0


# ============================================================
# MAIN DISCOVERY FUNCTION
# ============================================================

def run_causal_discovery(fts_db_path: str = None) -> Dict:
    """Sprint 7: Full causal discovery pipeline.

    1. Build co-occurrence matrix
    2. Run NOTEARS
    3. Extract typed edges
    4. Integrate into spreading_edges
    5. Save state

    Returns summary dict.
    """
    from modules.config import FTS_DB_PATH as _default_fts
    db = fts_db_path or _default_fts

    result = {
        "topics": 0,
        "edges_discovered": 0,
        "edges_integrated": 0,
        "h_value": 1.0,
        "ran": False,
    }

    if not os.path.exists(db):
        return result

    try:
        from modules.config import connect_fts
        conn = connect_fts(db)

        # 7.1: Co-occurrence matrix
        X, topics = build_cooccurrence_matrix(conn)
        conn.close()

        if X.shape[0] < 2:
            _logger.info("NOTEARS: not enough topics (%d), skipping", X.shape[0])
            return result

        result["topics"] = len(topics)

        # 7.2: NOTEARS
        W = run_notears(X)
        h_val = _acyclicity_constraint(W)
        result["h_value"] = round(float(h_val), 6)

        # 7.3: Extract edges
        edges = extract_edges(W, topics)
        result["edges_discovered"] = len(edges)

        # 7.4: Integrate
        integrated = integrate_discovered_edges(edges, db)
        result["edges_integrated"] = integrated

        # 7.5: Save state
        _save_discovery_state(W, topics, h_val, len(edges), db)
        result["ran"] = True

        _logger.info("Causal discovery: %d topics, %d discovered, %d integrated, h=%.2e",
                     len(topics), len(edges), integrated, h_val)

    except Exception as e:
        _logger.error("Causal discovery error: %s", e)

    return result
