-- Migration 027: CX Production Observability snapshots
-- One row per snapshot (tier-2 tick ~1.5h cadence)
-- JSON payload contains all CX metrics for dashboard

CREATE TABLE IF NOT EXISTS cx_snapshots (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    ts          TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%S', 'now', 'localtime')),
    payload     TEXT    NOT NULL,   -- JSON: {fires, latencies, diversity, ei_ratio, ...}
    anomalies   TEXT    DEFAULT '[]' -- JSON array of detected anomalies
);

CREATE INDEX IF NOT EXISTS idx_cx_snapshots_ts ON cx_snapshots(ts);
