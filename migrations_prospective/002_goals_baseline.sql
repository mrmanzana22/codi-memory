-- 002_goals_baseline.sql
-- Sprint 15: Goal System (ICARUS Activation-Based)
-- Papers: Altmann & Trafton 2002, Cox et al 2017, Pink et al 2025
-- Date: 2026-03-02

-- ============================================================
-- Goals Table (goals.py)
-- ============================================================
-- Activation-based priority agenda (NOT a stack).
-- Uses ACT-R base-level activation with decay.
-- Hierarchy: project > phase > sprint > task.

CREATE TABLE IF NOT EXISTS goals (
    id            TEXT PRIMARY KEY,
    title         TEXT NOT NULL,
    parent_id     TEXT REFERENCES goals(id),
    level         TEXT CHECK(level IN ('project','phase','sprint','task')) NOT NULL,
    status        TEXT CHECK(status IN ('active','paused','completed','abandoned')) DEFAULT 'active',
    priority      TEXT CHECK(priority IN ('critical','high','medium','low')) DEFAULT 'medium',
    access_count  INTEGER DEFAULT 0,
    created_at    TEXT NOT NULL,
    last_accessed TEXT NOT NULL,
    deadline      TEXT,
    assigned_to   TEXT,
    context       TEXT,
    metadata      TEXT
);

CREATE INDEX IF NOT EXISTS idx_goals_status ON goals(status);
CREATE INDEX IF NOT EXISTS idx_goals_parent ON goals(parent_id);
CREATE INDEX IF NOT EXISTS idx_goals_level ON goals(level);
CREATE INDEX IF NOT EXISTS idx_goals_last_accessed ON goals(last_accessed);

-- ============================================================
-- Goal Log (audit trail)
-- ============================================================

CREATE TABLE IF NOT EXISTS goal_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    goal_id TEXT NOT NULL,
    event TEXT NOT NULL,
    detail TEXT,
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_goal_log_goal ON goal_log(goal_id);

-- ============================================================
-- Link intentions to goals (Cox 2017: achieve operation)
-- ============================================================

ALTER TABLE intentions ADD COLUMN goal_id TEXT REFERENCES goals(id);
