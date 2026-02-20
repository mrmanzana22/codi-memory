-- 008_dual_integrity.sql
-- Fix dual_compare_log integrity: dedup + UNIQUE async_job_id + write_mode per row.
--
-- Problems solved:
--   1. Duplicate rows with same async_job_id but different fingerprint
--      (record_sync_result called twice per request in shadow mode).
--   2. Worker process doesn't know write_mode from env, so compare_results
--      misclassifies expected_shadow_dedup as action_mismatch.
--
-- Changes:
--   a) Dedup: delete duplicate rows, keeping MIN(id) per async_job_id
--   b) UNIQUE index on async_job_id (partial: WHERE async_job_id IS NOT NULL)
--   c) Add write_mode column (TEXT DEFAULT 'unknown')
--   d) Backfill existing rows to 'shadow'
--
-- Created: 2026-02-17

-- Step a: Dedup — keep the oldest row per async_job_id, delete the rest
DELETE FROM dual_compare_log
WHERE id NOT IN (
    SELECT MIN(id)
    FROM dual_compare_log
    WHERE async_job_id IS NOT NULL
    GROUP BY async_job_id
)
AND async_job_id IS NOT NULL
AND id NOT IN (
    SELECT MIN(id)
    FROM dual_compare_log
    WHERE async_job_id IS NOT NULL
    GROUP BY async_job_id
);

-- Step b: Drop old non-unique index, create unique one
DROP INDEX IF EXISTS idx_dcl_async_job_id;

CREATE UNIQUE INDEX IF NOT EXISTS idx_dcl_async_job_id_unique
    ON dual_compare_log(async_job_id)
    WHERE async_job_id IS NOT NULL;

-- Step c: Add write_mode column
ALTER TABLE dual_compare_log ADD COLUMN write_mode TEXT NOT NULL DEFAULT 'unknown';

-- Step d: Backfill all existing rows to 'shadow' (we've been in shadow mode since creation)
UPDATE dual_compare_log SET write_mode = 'shadow' WHERE write_mode = 'unknown';
