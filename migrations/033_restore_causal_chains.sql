-- 033_restore_causal_chains.sql
-- Restore causal_chains table (was in migration 019, lost in DB reset).
-- Required by Sprint 6 FIX-14 in consolidation.py (_store_causal_chains).
-- Pearl 2009, Woodward 2003: causal graph requires explicit chain storage.

CREATE TABLE IF NOT EXISTS causal_chains (
    chain_id TEXT PRIMARY KEY,
    nodes TEXT NOT NULL,
    strength REAL DEFAULT 0.5,
    mechanism TEXT DEFAULT '',
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_causal_chains_strength
  ON causal_chains (strength DESC);
