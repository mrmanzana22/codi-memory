-- Migration 032: Add content dedup index to working_memory
-- postgres-only
-- Prevents duplicate active WM items with same content+topic
-- Addresses Bug: remember() creates 2 items instead of 1 (dual_ack + restart scenarios)

CREATE UNIQUE INDEX IF NOT EXISTS idx_wm_active_content_hash
ON working_memory (
    md5(left(content, 300)),
    topic
)
WHERE active = TRUE;

COMMENT ON INDEX idx_wm_active_content_hash IS
    'Prevents duplicate active WM entries with same content prefix and topic.
     Only active=TRUE rows are constrained (archived items can repeat).
     Complements in-memory _push_dedup (120s window).';
