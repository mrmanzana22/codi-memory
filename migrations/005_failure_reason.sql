-- 005_failure_reason.sql
-- Add failure_reason column for classified error categories.
--
-- Categories: timeout, rate_limited, auth, upstream_5xx,
--             conn_error, payload_error, unknown
--
-- Enables shadow_report breakdown by failure reason instead of
-- raw exception class names.
--
-- Created: 2026-02-16

ALTER TABLE write_queue ADD COLUMN failure_reason TEXT;
ALTER TABLE write_queue_log ADD COLUMN failure_reason TEXT;
