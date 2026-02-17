-- 002_session_checkpoints.sql
-- Session Bridge: structured checkpoints for cross-session continuity.
-- Neuroscience basis: hippocampal replay + autobiographical self + pattern completion.

CREATE TABLE IF NOT EXISTS session_checkpoints (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    -- Identifiers
    trace_id TEXT,
    session_id TEXT,
    source TEXT DEFAULT 'flush',      -- 'flush' | 'hook'
    -- Core state
    goal_stack TEXT,                   -- JSON: [{goal, status, next_step}]
    active_project TEXT,
    session_summary TEXT,             -- max 500 chars
    -- Attention schema
    attention_focus TEXT,
    attention_strength REAL,
    last_prediction_topic TEXT,
    last_prediction_confidence REAL,
    recent_prediction_errors TEXT,    -- JSON: last 3 PEs
    -- Working memory
    wm_items TEXT,                    -- JSON: top 15 WM items
    wm_active_count INTEGER,
    -- Emotional state
    pad_pleasure REAL DEFAULT 0.0,
    pad_arousal REAL DEFAULT 0.0,
    pad_dominance REAL DEFAULT 0.0,
    pad_trigger TEXT,
    -- Decisions & plans
    last_decisions TEXT,              -- JSON: [{decision, why}]
    -- Prospective memory
    pending_intentions TEXT,          -- JSON: [{id, action, priority, activation}]
    -- Narrative traces
    active_traces TEXT,               -- JSON: [{trace_name, theme}]
    -- Metadata
    session_duration_minutes REAL,
    created_at TEXT NOT NULL,
    sleep_report TEXT                 -- NULL initially, daemon fills later
);

CREATE INDEX IF NOT EXISTS idx_sc_created ON session_checkpoints(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_sc_session ON session_checkpoints(session_id);
