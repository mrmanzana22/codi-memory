-- 030: FHRR Hippocampal Index tracking table
-- Tracks which sessions have been encoded as FHRR session records.
-- Paper: Teyler & DiScenna 1986 (Hippocampal Indexing Theory)

CREATE TABLE IF NOT EXISTS fhrr_session_index (
    session_id TEXT PRIMARY KEY,
    encoded_at TEXT NOT NULL,
    num_turns INTEGER,
    num_chunks INTEGER,
    file_size_bytes INTEGER,
    encoding_time_ms REAL,
    topics_json TEXT DEFAULT '[]'
);
