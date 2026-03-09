-- postgres-only
-- Memory Vault schema for PostgreSQL memories table.
-- SQLite migration runner skips this file via modules/migrations.py.

ALTER TABLE memories ADD COLUMN IF NOT EXISTS is_dormant BOOLEAN DEFAULT FALSE;
ALTER TABLE memories ADD COLUMN IF NOT EXISTS dormant_at TIMESTAMPTZ;
ALTER TABLE memories ADD COLUMN IF NOT EXISTS reactivation_count INTEGER DEFAULT 0;

CREATE INDEX IF NOT EXISTS idx_memories_active
  ON memories (activation_score DESC)
  WHERE NOT is_dormant;

CREATE INDEX IF NOT EXISTS idx_memories_vault
  ON memories (dormant_at DESC)
  WHERE is_dormant;

CREATE INDEX IF NOT EXISTS idx_memories_vault_vector
  ON memories
  USING hnsw (embedding vector_cosine_ops)
  WHERE is_dormant;
