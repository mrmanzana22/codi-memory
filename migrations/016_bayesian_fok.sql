-- S1-01: Beta-Binomial FOK model with conjugate updates
-- Replaces heuristic FOK with proper Bayesian estimation.
-- Alpha/beta track per-topic retrieval success/failure history.
CREATE TABLE IF NOT EXISTS fok_priors (
    topic TEXT PRIMARY KEY,
    alpha REAL DEFAULT 1.0,         -- Beta prior: successes + 1
    beta REAL DEFAULT 1.0,          -- Beta prior: failures + 1
    n_observations INTEGER DEFAULT 0,
    last_updated TEXT DEFAULT (datetime('now'))
);
