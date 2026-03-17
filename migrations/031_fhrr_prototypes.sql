-- Migration 031: FHRR Schema Prototypes
-- Tracks averaged role vectors per topic for novelty detection.
-- Paper: van Kesteren 2012 (mPFC schema congruency)

CREATE TABLE IF NOT EXISTS fhrr_schema_prototypes (
    topic TEXT PRIMARY KEY,
    num_sessions INTEGER NOT NULL DEFAULT 0,
    updated_at TEXT NOT NULL,
    prototype_path TEXT NOT NULL  -- path to .npz file with averaged role heads
);
