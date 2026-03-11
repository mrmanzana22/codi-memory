-- 024_system_health_snapshot.sql
-- Hourly operational snapshots for system self-awareness (P0 roadmap).

CREATE TABLE IF NOT EXISTS system_health (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    snapshot_at TEXT NOT NULL,          -- ISO8601 UTC
    tick_stats_json TEXT NOT NULL,      -- {tick: {last_at, age_min}} per tick
    wm_json TEXT NOT NULL,              -- {active, writes_24h, latest_write_at}
    predictions_json TEXT NOT NULL,     -- {count_24h, accuracy_pct, avg_pe}
    ai_json TEXT NOT NULL,              -- {total_observations, table_exists}
    alerts_json TEXT NOT NULL,          -- {open_count, critical_count, by_subsystem}
    global_json TEXT NOT NULL,          -- {events_24h, writes_24h}
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_system_health_snapshot_at
    ON system_health(snapshot_at);
