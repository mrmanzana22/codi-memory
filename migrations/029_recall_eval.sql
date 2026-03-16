-- Migration 029: Recall Quality Evaluation Tables
-- Sprint C: systematic recall quality measurement

CREATE TABLE IF NOT EXISTS recall_eval_cases (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    query TEXT NOT NULL,
    expected_keywords TEXT NOT NULL DEFAULT '[]',  -- JSON array of keywords that MUST appear in results
    category TEXT DEFAULT 'episodic',               -- episodic|semantic|bilingual|paraphrase
    active INTEGER DEFAULT 1,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS recall_eval_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_source TEXT NOT NULL DEFAULT 'manual',      -- sleep_loop|manual|ci
    n_cases INTEGER NOT NULL,
    precision_at_5 REAL,
    recall_at_5 REAL,
    mrr REAL,
    n_failures INTEGER DEFAULT 0,
    failure_details TEXT DEFAULT '[]',               -- JSON array of failed queries
    elapsed_ms INTEGER,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);
