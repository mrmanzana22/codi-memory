-- 023_health_monitoring.sql
-- Persisted health alerts for sleep-loop operational monitoring.

CREATE TABLE IF NOT EXISTS health_alerts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    alert_key TEXT NOT NULL,
    subsystem TEXT NOT NULL,
    status TEXT NOT NULL,              -- open | diagnosing | actioned | resolved | suppressed
    severity TEXT NOT NULL,            -- warning | critical
    title TEXT NOT NULL,
    description TEXT NOT NULL,
    evidence_json TEXT NOT NULL,
    recommended_action TEXT,
    first_seen_at TEXT NOT NULL,
    last_seen_at TEXT NOT NULL,
    resolved_at TEXT,
    occurrence_count INTEGER NOT NULL DEFAULT 1,
    source_run_id TEXT NOT NULL,
    owner TEXT,
    dedupe_hash TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_health_alerts_subsystem
    ON health_alerts(subsystem);

CREATE INDEX IF NOT EXISTS idx_health_alerts_status
    ON health_alerts(status);

CREATE INDEX IF NOT EXISTS idx_health_alerts_last_seen_at
    ON health_alerts(last_seen_at);

CREATE INDEX IF NOT EXISTS idx_health_alerts_dedupe_status
    ON health_alerts(dedupe_hash, status);
