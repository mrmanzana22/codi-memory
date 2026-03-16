-- Migration 028: Digital Pet (Tamagochi)
-- A living being that depends on Codi's conscious care.
-- Stats decay with real time (lazy evaluation), no tick needed.

CREATE TABLE IF NOT EXISTS pets (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    name        TEXT    NOT NULL,
    stage       TEXT    NOT NULL DEFAULT 'egg',   -- egg|baby|child|teen|adult
    born_at     TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%S', 'now', 'localtime')),
    hatched_at  TEXT,                              -- NULL until egg hatches
    died_at     TEXT,                              -- NULL unless dead
    alive       INTEGER NOT NULL DEFAULT 1,        -- 0 = dead (irreversible)
    -- Stats (last snapshot values, real state = lazy eval with elapsed time)
    hunger      REAL    NOT NULL DEFAULT 0.0,      -- 0=full, 1=starving
    happiness   REAL    NOT NULL DEFAULT 0.8,      -- 0=sad, 1=happy
    energy      REAL    NOT NULL DEFAULT 1.0,      -- 0=exhausted, 1=full
    health      REAL    NOT NULL DEFAULT 1.0,      -- 0=dead, 1=healthy
    -- Timestamps for lazy evaluation
    last_interaction_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%S', 'now', 'localtime')),
    last_fed_at     TEXT,
    last_played_at  TEXT,
    last_rested_at  TEXT,
    last_cleaned_at TEXT,
    last_medicine_at TEXT,
    -- Counters
    times_fed       INTEGER NOT NULL DEFAULT 0,
    times_played    INTEGER NOT NULL DEFAULT 0,
    times_rested    INTEGER NOT NULL DEFAULT 0,
    total_care_actions INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS pet_care_log (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    pet_id      INTEGER NOT NULL REFERENCES pets(id),
    action      TEXT    NOT NULL,  -- feed|play|rest|clean|medicine
    ts          TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%S', 'now', 'localtime')),
    state_before TEXT,             -- JSON snapshot before action
    state_after  TEXT,             -- JSON snapshot after action
    note        TEXT
);

CREATE INDEX IF NOT EXISTS idx_pet_care_log_pet ON pet_care_log(pet_id);
CREATE INDEX IF NOT EXISTS idx_pet_care_log_ts ON pet_care_log(ts);
