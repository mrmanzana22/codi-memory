-- 001_prospective_baseline.sql
-- Baseline schema for prospective.db (2 tables, 4 indexes)
-- Captures the full schema as of D4 implementation (2026-02-15).

-- ============================================================
-- Prospective Memory - Intentions (prospective.py)
-- ============================================================

CREATE TABLE IF NOT EXISTS intentions (
    id TEXT PRIMARY KEY,
    action TEXT NOT NULL,
    action_type TEXT DEFAULT 'remind',
    trigger_type TEXT NOT NULL,
    trigger_spec TEXT NOT NULL,
    cue_focality TEXT DEFAULT 'focal',
    priority TEXT DEFAULT 'medium',
    status TEXT DEFAULT 'pending',
    activation REAL DEFAULT 0.7,
    created_at TEXT NOT NULL,
    triggered_at TEXT,
    completed_at TEXT,
    expiry TEXT,
    snooze_until TEXT,
    context_at_creation TEXT,
    creator TEXT DEFAULT 'codi',
    recurrence TEXT,
    recurrence_spec TEXT,
    check_count INTEGER DEFAULT 0,
    partial_match_count INTEGER DEFAULT 0,
    last_checked_at TEXT,
    last_maintained_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_intentions_status ON intentions(status);
CREATE INDEX IF NOT EXISTS idx_intentions_activation ON intentions(activation DESC);
CREATE INDEX IF NOT EXISTS idx_intentions_expiry ON intentions(expiry);

-- ============================================================
-- Intention Log (prospective.py)
-- ============================================================

CREATE TABLE IF NOT EXISTS intention_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    intention_id TEXT NOT NULL,
    event TEXT NOT NULL,
    detail TEXT,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_intention_log_intention ON intention_log(intention_id);
