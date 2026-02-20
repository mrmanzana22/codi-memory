-- 009_cancel_tombstone.sql
-- Cancel tombstone: allow marking running jobs for cancellation.
--
-- Today cancel_write_job() only works on 'queued' jobs.
-- For 'running' jobs, it returns 'too_late' with no recourse.
--
-- With tombstone semantics:
--   - queued  -> canceled immediately (same as before)
--   - running -> cancel_requested=1 (tombstone), worker checks before/during exec
--   - done/dead/failed/canceled -> no-op
--
-- Worker behavior:
--   1. After claim: if cancel_requested=1, skip and mark canceled
--   2. Before exec: re-check cancel_requested (race window protection)
--   3. If exec already started: log as 'late_cancel_race' (tolerable for remember/add_memory)
--
-- Created: 2026-02-17

ALTER TABLE write_queue ADD COLUMN cancel_requested INTEGER NOT NULL DEFAULT 0;
ALTER TABLE write_queue ADD COLUMN canceled_at TEXT;
ALTER TABLE write_queue ADD COLUMN cancel_reason TEXT;

-- Index for worker to quickly skip canceled jobs during claim
CREATE INDEX IF NOT EXISTS idx_wq_cancel_requested
    ON write_queue(cancel_requested)
    WHERE cancel_requested = 1;
