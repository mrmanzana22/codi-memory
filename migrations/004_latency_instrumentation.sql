-- 004_latency_instrumentation.sql
-- Add timing columns to write_queue for latency breakdown.
--
-- Enables separating:
--   queue_wait_ms = claimed_at - created_at   (time sitting in queue)
--   exec_ms       = completed_at - started_at (actual executor runtime)
--   total_ms      = completed_at - created_at (end-to-end)
--
-- Created: 2026-02-16

ALTER TABLE write_queue ADD COLUMN claimed_at TEXT;
ALTER TABLE write_queue ADD COLUMN started_at TEXT;
