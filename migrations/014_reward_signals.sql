-- S1-05: Reward signal infrastructure for Thompson Sampling
-- Tracks whether retrieved memories are re-accessed within 5 min (binary reward).
-- Channel attribution enables per-channel Beta posterior updates.
CREATE TABLE IF NOT EXISTS reward_signals (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    query TEXT NOT NULL,
    memory_id TEXT NOT NULL,
    channel TEXT NOT NULL,          -- 'vector', 'bm25', 'semantic'
    retrieved_at TEXT NOT NULL,
    rewarded INTEGER DEFAULT NULL,  -- NULL=pending, 1=rewarded, 0=expired
    reward_at TEXT,
    created_at TEXT DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_reward_pending ON reward_signals(rewarded, retrieved_at);
CREATE INDEX IF NOT EXISTS idx_reward_memory ON reward_signals(memory_id, retrieved_at);
