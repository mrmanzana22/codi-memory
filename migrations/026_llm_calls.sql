-- 026_llm_calls.sql
-- LLM router call log table for observability.

CREATE TABLE IF NOT EXISTS llm_calls (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    task_type       TEXT NOT NULL,
    provider        TEXT NOT NULL,
    model           TEXT NOT NULL,
    success         INTEGER NOT NULL DEFAULT 1,
    duration_ms     INTEGER,
    prompt_chars    INTEGER,
    output_chars    INTEGER,
    error           TEXT,
    created_at      TEXT DEFAULT (datetime('now'))
);
