-- 011: Pending corrections queue (reconsolidation suggest mode)
--
-- When contradiction is detected (PE >= ALERT), instead of auto-correcting,
-- we queue a correction suggestion for review. This implements the
-- reconsolidation window as a TIME-DEPENDENT process (Nader 2000).
--
-- Status flow: pending -> approved -> applied | rejected | expired

CREATE TABLE IF NOT EXISTS pending_corrections (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    old_memory_id TEXT NOT NULL,
    new_memory_id TEXT,
    old_text TEXT NOT NULL,
    new_text TEXT NOT NULL,
    prediction_error REAL NOT NULL,
    shared_entities TEXT,           -- JSON array of shared entity strings
    channels TEXT,                  -- JSON dict of PE channel scores
    status TEXT NOT NULL DEFAULT 'pending',  -- pending|approved|applied|rejected|expired
    created_at TEXT NOT NULL,
    reviewed_at TEXT,
    applied_at TEXT,
    expires_at TEXT NOT NULL        -- Reconsolidation window: 24h default
);

CREATE INDEX IF NOT EXISTS idx_pending_corrections_status
    ON pending_corrections(status);
CREATE INDEX IF NOT EXISTS idx_pending_corrections_expires
    ON pending_corrections(expires_at);
