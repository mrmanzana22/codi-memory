-- 010_dual_sync_status.sql
-- Add sync-compare thread observability + created_at columns to dual_compare_log.
--
-- Problems solved:
--   1. sync_completed_at is a placeholder for dual_ack rows (set by record_async_intent).
--      created_at is the true row creation timestamp for time-window filtering.
--   2. sync_compare_status tracks the outcome of the background sync-compare thread
--      (ok/timeout/skipped_capacity/error), distinct from compare_status which
--      tracks the lifecycle of the comparison computation.
--   3. sync_last_error stores error messages from the sync-compare thread.
--
-- Created: 2026-02-17

-- Step a: Add created_at column
ALTER TABLE dual_compare_log
  ADD COLUMN created_at TEXT NOT NULL DEFAULT '';

-- Backfill: rows without created_at use sync_completed_at as fallback
UPDATE dual_compare_log SET created_at = sync_completed_at WHERE created_at = '';

-- Step b: Add sync_compare_status column
-- Values: 'pending' | 'ok' | 'timeout' | 'skipped_capacity' | 'error'
ALTER TABLE dual_compare_log
  ADD COLUMN sync_compare_status TEXT NOT NULL DEFAULT 'pending';

-- Step c: Add sync_last_error column (truncated to 500 chars)
ALTER TABLE dual_compare_log
  ADD COLUMN sync_last_error TEXT;
