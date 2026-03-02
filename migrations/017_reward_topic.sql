-- Migration 017: Add topic column to reward_signals for per-topic Thompson Sampling
-- Canon v2, CC-8: Domain-specific retrieval allocation
-- When: 2026-02-28

ALTER TABLE reward_signals ADD COLUMN topic TEXT DEFAULT 'general';
CREATE INDEX IF NOT EXISTS idx_reward_topic ON reward_signals(topic, channel, rewarded);
