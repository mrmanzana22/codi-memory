-- Migration 025: Add session_id to write_queue for trace propagation
-- Bug #025: enqueue_write_job accepted session_id but never stored it

ALTER TABLE write_queue ADD COLUMN session_id TEXT;
