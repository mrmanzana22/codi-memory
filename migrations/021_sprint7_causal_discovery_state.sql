-- Migration 021: Sprint 7 — NOTEARS Causal Discovery state table
-- Zheng et al. 2018: DAG learning requires persistent W matrix storage
-- Stores serialized adjacency matrices and discovery metadata for each run
-- When: 2026-02-28

CREATE TABLE IF NOT EXISTS causal_discovery_state (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    w_matrix TEXT NOT NULL,
    topics TEXT NOT NULL,
    h_value REAL,
    lambda_val REAL,
    n_edges INTEGER,
    created_at TEXT NOT NULL
);
