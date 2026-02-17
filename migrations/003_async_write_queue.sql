-- 003_async_write_queue.sql
-- Async write-path queue for non-blocking memory operations (E2.3).
-- Context: remember/add_memory/checkpoint_memoria block 30-80s due to
-- synchronous mem0 -> OpenAI -> Qdrant pipeline. This table enables
-- immediate ACK + background processing with retries and observability.

CREATE TABLE IF NOT EXISTS write_queue (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id TEXT NOT NULL UNIQUE,
    kind TEXT NOT NULL,                    -- 'remember' | 'add_memory' | 'checkpoint_memoria'
    payload_json TEXT NOT NULL,            -- full args needed to execute without original process
    status TEXT NOT NULL DEFAULT 'queued', -- 'queued' | 'running' | 'done' | 'failed' | 'dead'
    priority INTEGER NOT NULL DEFAULT 5,  -- 1=highest, 10=lowest
    attempts INTEGER NOT NULL DEFAULT 0,
    max_attempts INTEGER NOT NULL DEFAULT 8,
    lease_until TEXT,                      -- ISO timestamp: worker claim expiry
    last_error TEXT,
    dedupe_key TEXT,                       -- SHA256(kind + normalized_text + day_bucket)
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    completed_at TEXT                      -- set when status -> 'done'
);

-- Primary query: find next queued job by priority + age
CREATE INDEX IF NOT EXISTS idx_wq_status_priority_created
    ON write_queue(status, priority, created_at);

-- Dedupe lookups
CREATE INDEX IF NOT EXISTS idx_wq_dedupe ON write_queue(dedupe_key)
    WHERE dedupe_key IS NOT NULL;

-- Lease expiry checks (stale worker detection)
CREATE INDEX IF NOT EXISTS idx_wq_lease ON write_queue(lease_until)
    WHERE status = 'running';

-- Observability: job completion log (lightweight, separate from tool_calls)
CREATE TABLE IF NOT EXISTS write_queue_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id TEXT NOT NULL,
    kind TEXT NOT NULL,
    status TEXT NOT NULL,                  -- final status: 'done' | 'failed' | 'dead'
    attempts INTEGER NOT NULL DEFAULT 0,
    duration_ms INTEGER,                   -- total wall time from created_at to completion
    error_class TEXT,
    error_msg TEXT,
    session_id TEXT,
    created_at TEXT NOT NULL               -- when this log entry was written
);

CREATE INDEX IF NOT EXISTS idx_wql_job ON write_queue_log(job_id);
CREATE INDEX IF NOT EXISTS idx_wql_time ON write_queue_log(created_at);
