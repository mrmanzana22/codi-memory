-- Migration 019: Sprint 5 — Causal Consolidation structures
-- Sprint 5.3: Add strength column to spreading_edges
-- Sprint 5.5: Create causal_chains table
-- Sprint 10.1: Create retrieval_weights table (also needed for Sprint 10)
-- Woodward 2003, Pearl 2009: causal graph requires graded edge strengths and chain storage

-- Sprint 5.3: Strength column for spreading_edges (graded causal reliability)
ALTER TABLE spreading_edges ADD COLUMN strength REAL DEFAULT 0.5;

-- Sprint 5.5: Causal chain storage (A→B→C explicit structure)
CREATE TABLE IF NOT EXISTS causal_chains (
    chain_id TEXT PRIMARY KEY,
    nodes TEXT NOT NULL,
    strength REAL DEFAULT 0.5,
    mechanism TEXT DEFAULT '',
    created_at TEXT NOT NULL
);

-- Sprint 10.1: Per-topic retrieval weight adaptation (Gershman & Daw 2017)
CREATE TABLE IF NOT EXISTS retrieval_weights (
    topic TEXT PRIMARY KEY,
    w_vector REAL DEFAULT 0.40,
    w_bm25 REAL DEFAULT 0.15,
    w_activation REAL DEFAULT 0.45,
    n_updates INTEGER DEFAULT 0,
    updated_at TEXT
);

