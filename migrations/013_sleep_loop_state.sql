-- S0-05: VOC tiering tick counter persistence
CREATE TABLE IF NOT EXISTS sleep_loop_state (
    key TEXT PRIMARY KEY,
    value TEXT
);
