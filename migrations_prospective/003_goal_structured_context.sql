-- 003_goal_structured_context.sql
-- Sprint 15.5: Structured Goal Context (research-backed)
--
-- Replaces monolithic `context` blob with 4 structured fields
-- based on cognitive architecture research:
--   - ACT-R: goal buffer (control) vs imaginal buffer (data)
--   - SOAR: O-support (committed/permanent) vs I-support (derived/recomputable)
--   - Duncan 2013: chunking structure prevents goal neglect
--   - Altmann & Trafton 2007: cascading priming for incremental recovery
--
-- Papers: Goal Drift arXiv:2505.02709, HiAgent ACL 2025
-- Date: 2026-03-02

-- ============================================================
-- Committed fields (permanent, set at creation)
-- ============================================================
ALTER TABLE goals ADD COLUMN goal_what TEXT;
ALTER TABLE goals ADD COLUMN goal_why TEXT;

-- ============================================================
-- Derivable fields (updated on touch/flush, recomputable)
-- ============================================================
ALTER TABLE goals ADD COLUMN goal_last_state TEXT;
ALTER TABLE goals ADD COLUMN goal_next_step TEXT;

-- ============================================================
-- Context freshness tracking
-- ============================================================
ALTER TABLE goals ADD COLUMN context_updated_at TEXT;

-- ============================================================
-- Migrate existing context blob to goal_what
-- ============================================================
UPDATE goals SET goal_what = context WHERE context IS NOT NULL AND context != '';
UPDATE goals SET context_updated_at = last_accessed WHERE context IS NOT NULL AND context != '';
