-- 007_dual_compare_log.sql
-- Dual-mode compare layer: tracks sync vs async results for shadow comparison.
--
-- When dual-mode is active, the sync path records its result here immediately.
-- After the async worker completes the same job, it UPDATEs the async columns.
-- A comparison function then computes match/divergence.
--
-- request_fingerprint = SHA256(kind | normalized_content | trace_id)[:32]
-- This is distinct from write_queue.dedupe_key (which uses day_bucket).
--
-- Created: 2026-02-16

CREATE TABLE IF NOT EXISTS dual_compare_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,

    -- Correlation
    request_fingerprint TEXT NOT NULL UNIQUE,   -- SHA256(kind|content|trace_id)[:32]
    trace_id            TEXT NOT NULL,          -- from modules.tracing
    kind                TEXT NOT NULL,          -- 'remember' | 'add_memory' | 'checkpoint_memoria'

    -- Sync side (filled at record time)
    sync_completed_at   TEXT NOT NULL,          -- ISO timestamp
    sync_result_json    TEXT,                   -- normalized result, cap 8192 chars
    sync_preview        TEXT,                   -- human-readable preview, cap 200 chars

    -- Async side (filled by worker UPDATE after job completes)
    async_job_id        TEXT,                   -- write_queue.job_id
    async_status_at_record TEXT,                -- status when dual row was created
    async_completed_at  TEXT,                   -- ISO timestamp when worker finished
    async_result_json   TEXT,                   -- normalized result, cap 8192 chars
    async_last_error    TEXT,                   -- last error if failed, cap 500 chars
    failure_reason      TEXT,                   -- classified reason (timeout, rate_limited, etc.)

    -- Comparison (filled by compute_comparison_for_job)
    compare_status      TEXT NOT NULL DEFAULT 'pending',  -- 'pending' | 'computed'
    compare_computed_at TEXT,                   -- ISO timestamp when comparison ran
    match               INTEGER,               -- 1 = match, 0 = divergence, NULL = not yet computed
    divergence_code     TEXT,                   -- e.g. 'action_mismatch', 'memory_id_diff', 'async_failed'
    diff_summary        TEXT,                   -- human-readable diff, cap 300 chars
    diff_json           TEXT                    -- structured diff, cap 4096 chars
);

-- Fast lookup by async_job_id (worker UPDATE path)
CREATE INDEX IF NOT EXISTS idx_dcl_async_job_id
    ON dual_compare_log(async_job_id)
    WHERE async_job_id IS NOT NULL;

-- Pending comparisons scan
CREATE INDEX IF NOT EXISTS idx_dcl_compare_status_time
    ON dual_compare_log(compare_status, sync_completed_at);

-- Divergence analysis queries
CREATE INDEX IF NOT EXISTS idx_dcl_divergence_code
    ON dual_compare_log(divergence_code)
    WHERE divergence_code IS NOT NULL;
