-- 001_fts_baseline.sql
-- Baseline schema for memories_fts.db (16 tables, 14 indexes, 3 triggers)
-- Captures the full schema as of D4 implementation (2026-02-15).
-- All statements use IF NOT EXISTS for idempotency on existing installs.

-- ============================================================
-- FTS5 Full-Text Search (memory_smart.py)
-- ============================================================

CREATE TABLE IF NOT EXISTS memories_text (
    memory_id TEXT PRIMARY KEY,
    content TEXT NOT NULL,
    category TEXT,
    source TEXT,
    importance TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
    content,
    memory_id UNINDEXED,
    category UNINDEXED,
    source UNINDEXED,
    content='memories_text',
    content_rowid='rowid'
);

-- FTS sync triggers
CREATE TRIGGER IF NOT EXISTS memories_text_ai AFTER INSERT ON memories_text BEGIN
    INSERT INTO memories_fts(rowid, content, memory_id, category, source)
    VALUES (new.rowid, new.content, new.memory_id, new.category, new.source);
END;

CREATE TRIGGER IF NOT EXISTS memories_text_ad AFTER DELETE ON memories_text BEGIN
    INSERT INTO memories_fts(memories_fts, rowid, content, memory_id, category, source)
    VALUES ('delete', old.rowid, old.content, old.memory_id, old.category, old.source);
END;

CREATE TRIGGER IF NOT EXISTS memories_text_au AFTER UPDATE ON memories_text BEGIN
    INSERT INTO memories_fts(memories_fts, rowid, content, memory_id, category, source)
    VALUES ('delete', old.rowid, old.content, old.memory_id, old.category, old.source);
    INSERT INTO memories_fts(rowid, content, memory_id, category, source)
    VALUES (new.rowid, new.content, new.memory_id, new.category, new.source);
END;

-- FTS retry queue
CREATE TABLE IF NOT EXISTS fts_retry_queue (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    memory_id TEXT NOT NULL,
    op TEXT NOT NULL,
    payload_json TEXT,
    status TEXT DEFAULT 'pending',
    attempts INTEGER DEFAULT 0,
    last_error TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE UNIQUE INDEX IF NOT EXISTS uq_fts_retry_mem_op ON fts_retry_queue(memory_id, op);
CREATE INDEX IF NOT EXISTS idx_fts_retry_status ON fts_retry_queue(status, updated_at);

-- ============================================================
-- Metrics & Auditing (metrics.py)
-- ============================================================

CREATE TABLE IF NOT EXISTS tool_calls (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    tool_name TEXT NOT NULL,
    started_at TEXT NOT NULL,
    duration_ms INTEGER NOT NULL,
    success INTEGER NOT NULL,
    error_type TEXT,
    args_size INTEGER DEFAULT 0,
    result_size INTEGER DEFAULT 0,
    session_id TEXT,
    tag TEXT
);
CREATE INDEX IF NOT EXISTS idx_tool_calls_time ON tool_calls(started_at);
CREATE INDEX IF NOT EXISTS idx_tool_calls_tool ON tool_calls(tool_name, started_at);

-- ============================================================
-- Consolidation Pipeline (consolidation.py)
-- ============================================================

CREATE TABLE IF NOT EXISTS consolidation_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    batch_id TEXT NOT NULL UNIQUE,
    scope TEXT NOT NULL,
    lookback_hours INTEGER,
    episodes_scanned INTEGER DEFAULT 0,
    clusters_found INTEGER DEFAULT 0,
    facts_extracted INTEGER DEFAULT 0,
    facts_created INTEGER DEFAULT 0,
    facts_updated INTEGER DEFAULT 0,
    contradictions_found INTEGER DEFAULT 0,
    episodes_pruned INTEGER DEFAULT 0,
    duration_ms INTEGER DEFAULT 0,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS reconsolidation_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    memory_id TEXT NOT NULL,
    memory_type TEXT DEFAULT 'episodic',
    action TEXT NOT NULL,
    prediction_error REAL,
    memory_strength REAL,
    old_content TEXT,
    new_content TEXT,
    blend_weight REAL,
    trigger_context TEXT,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_recon_log_memory ON reconsolidation_log(memory_id);
CREATE INDEX IF NOT EXISTS idx_recon_log_time ON reconsolidation_log(created_at);

CREATE TABLE IF NOT EXISTS labile_memories (
    memory_id TEXT PRIMARY KEY,
    marked_at TEXT NOT NULL,
    window_expires TEXT NOT NULL,
    prediction_error REAL,
    trigger_context TEXT
);

-- ============================================================
-- Knowledge Schemas (schemas.py)
-- ============================================================

CREATE TABLE IF NOT EXISTS schemas (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    description TEXT,
    topic TEXT NOT NULL,
    slots_json TEXT NOT NULL,
    instance_count INTEGER DEFAULT 1,
    created_at TEXT NOT NULL,
    last_matched TEXT
);

-- ============================================================
-- Working Memory (working_memory.py)
-- ============================================================

CREATE TABLE IF NOT EXISTS working_memory (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    content TEXT NOT NULL,
    topic TEXT DEFAULT 'general',
    relevance REAL DEFAULT 0.5,
    added_at TEXT NOT NULL,
    occurred_at TEXT,
    source TEXT DEFAULT 'interaction',
    related_memory_id TEXT,
    chain_id TEXT,
    active INTEGER DEFAULT 1,
    last_accessed_at TEXT,
    access_count INTEGER DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_wm_active_relevance ON working_memory(active, relevance DESC);
CREATE INDEX IF NOT EXISTS idx_wm_chain_active_time ON working_memory(chain_id, active, occurred_at);
CREATE INDEX IF NOT EXISTS idx_wm_topic_active_time ON working_memory(topic, active, occurred_at);
CREATE INDEX IF NOT EXISTS idx_wm_added_at ON working_memory(added_at);

CREATE TABLE IF NOT EXISTS narrative_traces (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    trace_name TEXT NOT NULL UNIQUE,
    chain_ids TEXT NOT NULL,
    theme TEXT,
    created_at TEXT NOT NULL,
    last_updated TEXT NOT NULL,
    active INTEGER DEFAULT 1
);
CREATE INDEX IF NOT EXISTS idx_traces_active_updated ON narrative_traces(active, last_updated);

CREATE TABLE IF NOT EXISTS trace_chains (
    trace_id INTEGER NOT NULL,
    chain_id TEXT NOT NULL,
    FOREIGN KEY (trace_id) REFERENCES narrative_traces(id)
);
CREATE UNIQUE INDEX IF NOT EXISTS uq_trace_chain ON trace_chains(trace_id, chain_id);
CREATE INDEX IF NOT EXISTS idx_trace_chains_chain ON trace_chains(chain_id);
CREATE INDEX IF NOT EXISTS idx_trace_chains_trace ON trace_chains(trace_id);

-- ============================================================
-- Retrieval Metadata (retrieval_metadata.py)
-- ============================================================

CREATE TABLE IF NOT EXISTS failed_searches (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    query TEXT NOT NULL,
    result_count INTEGER DEFAULT 0,
    top_activation REAL DEFAULT 0.0,
    topic TEXT,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS fok_calibration_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    query TEXT,
    fok_predicted REAL,
    actual_coverage TEXT,
    actual_count INTEGER,
    actual_top_activation REAL,
    created_at TEXT
);

-- ============================================================
-- Event Counts (events.py)
-- ============================================================

CREATE TABLE IF NOT EXISTS event_counts (
    event TEXT PRIMARY KEY,
    count INTEGER DEFAULT 0,
    last_seen TEXT
);

-- ============================================================
-- Prediction Loop (hooks/preturn_inject.py)
-- ============================================================

CREATE TABLE IF NOT EXISTS prediction_state (
    id INTEGER PRIMARY KEY,
    predicted_topic TEXT,
    predicted_keywords TEXT,
    predicted_action TEXT,
    topic_distribution TEXT,
    confidence REAL DEFAULT 0.5,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS prediction_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    predicted_topic TEXT,
    actual_topic TEXT,
    predicted_keywords TEXT,
    actual_keywords TEXT,
    surprise_score REAL,
    precision_weight REAL,
    weighted_surprise REAL,
    hit INTEGER DEFAULT 0,
    created_at TEXT NOT NULL
);
