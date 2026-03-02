-- S1-03: Storage Strength vs Retrieval Strength (Bjork & Bjork 1992)
-- SS = monotonically non-decreasing, RS = decays with time.
-- Tracked as Qdrant payload fields: storage_strength, retrieval_strength.
-- This migration adds a tracking table for SS/RS history (debugging).
CREATE TABLE IF NOT EXISTS strength_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    memory_id TEXT NOT NULL,
    storage_strength REAL NOT NULL,
    retrieval_strength REAL NOT NULL,
    access_count INTEGER DEFAULT 0,
    event TEXT DEFAULT 'access',  -- 'access', 'decay', 'consolidation'
    created_at TEXT DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_strength_memory ON strength_log(memory_id, created_at);
