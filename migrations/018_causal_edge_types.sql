-- Migration 018: Transform associative edges to causal schema (Canon v2, CC-3)
-- Pearl 2009: Graph must support causal, not just associative, relationships
-- Renames old types to co_occurs, adds schema for 5 causal types:
--   causes, enables, prevents, co_occurs, confounded
-- When: 2026-02-28

-- Ensure table exists (normally created at runtime by spreading.py)
CREATE TABLE IF NOT EXISTS spreading_edges (
    from_id TEXT NOT NULL,
    to_id TEXT NOT NULL,
    edge_type TEXT DEFAULT 'co_occurs',
    last_seen TEXT,
    PRIMARY KEY (from_id, to_id)
);
CREATE INDEX IF NOT EXISTS idx_edges_to ON spreading_edges(to_id);

-- Rename all existing associative types to co_occurs
UPDATE spreading_edges SET edge_type = 'co_occurs' WHERE edge_type IN ('similarity', 'consolidated', 'bridge', 'outgoing', 'temporal', 'broadcast');
