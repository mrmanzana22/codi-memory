"""
Recall Quality Evaluation Harness
==================================
Ground-truth test suite: known queries mapped to expected keywords.
Runs precision@k, recall@k, MRR, stores results for trending.

Integration:
  - Sleep loop: tick_recall_eval() (Tier 3)
  - On-demand: MCP tool eval_recall_quality()
  - Results: recall_eval_results table (memories_fts.db)

Neuroscience basis:
  - Tulving 1985: cued recall paradigm
  - Nelson & Narens 1990: metamemory monitoring
  - Koriat 1993: FOK calibration
"""

import json
import time
import logging
import os
from dataclasses import dataclass

_logger = logging.getLogger("codi.recall_eval")


@dataclass
class EvalCase:
    id: int
    query: str
    expected_keywords: list
    category: str = "episodic"


def _get_conn():
    """Get FTS DB connection."""
    from modules.config import connect_fts, FTS_DB_PATH
    if not os.path.exists(FTS_DB_PATH):
        return None
    return connect_fts(FTS_DB_PATH)


def ensure_tables(conn):
    """Create eval tables if they don't exist."""
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS recall_eval_cases (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            query TEXT NOT NULL,
            expected_keywords TEXT NOT NULL DEFAULT '[]',
            category TEXT DEFAULT 'episodic',
            active INTEGER DEFAULT 1,
            created_at TEXT NOT NULL DEFAULT (datetime('now')),
            updated_at TEXT NOT NULL DEFAULT (datetime('now'))
        );
        CREATE TABLE IF NOT EXISTS recall_eval_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_source TEXT NOT NULL DEFAULT 'manual',
            n_cases INTEGER NOT NULL,
            precision_at_5 REAL,
            recall_at_5 REAL,
            mrr REAL,
            n_failures INTEGER DEFAULT 0,
            failure_details TEXT DEFAULT '[]',
            elapsed_ms INTEGER,
            created_at TEXT NOT NULL DEFAULT (datetime('now'))
        );
    """)


def load_active_cases(conn=None):
    """Load all active eval cases from DB."""
    _conn = conn or _get_conn()
    if not _conn:
        return []
    try:
        ensure_tables(_conn)
        rows = _conn.execute(
            "SELECT id, query, expected_keywords, category FROM recall_eval_cases "
            "WHERE active = 1 ORDER BY id"
        ).fetchall()
        cases = []
        for r in rows:
            try:
                kws = json.loads(r[2]) if r[2] else []
            except json.JSONDecodeError:
                kws = []
            cases.append(EvalCase(id=r[0], query=r[1], expected_keywords=kws, category=r[3] or "episodic"))
        return cases
    finally:
        if conn is None:
            _conn.close()


def add_eval_case(query, expected_keywords, category="episodic"):
    """Add a new eval case. Returns case ID."""
    conn = _get_conn()
    if not conn:
        return -1
    try:
        ensure_tables(conn)
        cursor = conn.execute(
            "INSERT INTO recall_eval_cases (query, expected_keywords, category) VALUES (?, ?, ?)",
            (query, json.dumps(expected_keywords, ensure_ascii=False), category),
        )
        conn.commit()
        return cursor.lastrowid
    finally:
        conn.close()


def _search_for_eval(query, limit=10):
    """Run search and extract result texts.

    Uses raw pg_store search directly (search_memory returns formatted text).
    """
    try:
        from modules.pg_store import pg
        from modules.consolidation_common import _embed_text

        items = []

        # Vector search (episodic)
        embedding = _embed_text(query)
        if embedding:
            points = pg.query_vector(embedding, limit=limit, is_semantic=False)
            for p in points:
                text = (p.payload or {}).get("data", "")
                items.append({"id": str(p.id), "text": str(text).lower()})

        # BM25 search
        bm25_results = pg.search_fts(query, limit=limit)
        seen_ids = {i["id"] for i in items}
        for r in bm25_results:
            mid = str(r.get("memory_id", ""))
            if mid not in seen_ids:
                items.append({"id": mid, "text": str(r.get("content", "")).lower()})
                seen_ids.add(mid)

        # Semantic search
        try:
            from modules.consolidation import search_semantic
            sem_results = search_semantic(query, limit=min(5, limit))
            for f in sem_results:
                fid = str(f.get("id", ""))
                if fid not in seen_ids:
                    items.append({"id": fid, "text": str(f.get("fact", "")).lower()})
                    seen_ids.add(fid)
        except Exception:
            pass

        return items
    except Exception as e:
        _logger.debug("search_for_eval failed: %s", e)
        return []


def compute_precision_at_k(results, keywords, k=5):
    """Fraction of top-k results that contain at least one expected keyword."""
    top_k = results[:k]
    if not top_k or not keywords:
        return 0.0
    relevant = 0
    for r in top_k:
        text = r["text"]
        if any(kw.lower() in text for kw in keywords):
            relevant += 1
    return relevant / len(top_k)


def compute_recall_at_k(results, keywords, k=5):
    """Fraction of expected keywords found in top-k results."""
    if not keywords:
        return 1.0
    top_k_text = " ".join(r["text"] for r in results[:k])
    found = sum(1 for kw in keywords if kw.lower() in top_k_text)
    return found / len(keywords)


def compute_mrr(results, keywords):
    """Mean Reciprocal Rank: 1/rank of first result containing any expected keyword."""
    for i, r in enumerate(results):
        text = r["text"]
        if any(kw.lower() in text for kw in keywords):
            return 1.0 / (i + 1)
    return 0.0


def run_eval_suite(cases, limit=10):
    """Run full evaluation battery."""
    start = time.monotonic()
    precisions = []
    recalls = []
    mrrs = []
    failures = []

    for case in cases:
        results = _search_for_eval(case.query, limit=limit)

        p = compute_precision_at_k(results, case.expected_keywords, k=5)
        r = compute_recall_at_k(results, case.expected_keywords, k=5)
        m = compute_mrr(results, case.expected_keywords)

        precisions.append(p)
        recalls.append(r)
        mrrs.append(m)

        if m == 0.0:
            failures.append({
                "query": case.query,
                "expected_keywords": case.expected_keywords,
                "category": case.category,
            })

    elapsed_ms = int((time.monotonic() - start) * 1000)
    n = len(cases)

    aggregate = {
        "precision_at_5": round(sum(precisions) / n, 3) if n else 0,
        "recall_at_5": round(sum(recalls) / n, 3) if n else 0,
        "mrr": round(sum(mrrs) / n, 3) if n else 0,
        "n_cases": n,
        "n_failures": len(failures),
    }

    from datetime import datetime
    return {
        "aggregate": aggregate,
        "failures": failures,
        "timestamp": datetime.now().isoformat(),
        "elapsed_ms": elapsed_ms,
    }


def store_eval_results(results, run_source="manual"):
    """Store eval results to DB for trending."""
    conn = _get_conn()
    if not conn:
        return
    try:
        ensure_tables(conn)
        agg = results.get("aggregate", {})
        conn.execute(
            "INSERT INTO recall_eval_results "
            "(run_source, n_cases, precision_at_5, recall_at_5, mrr, n_failures, "
            " failure_details, elapsed_ms) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                run_source,
                agg.get("n_cases", 0),
                agg.get("precision_at_5"),
                agg.get("recall_at_5"),
                agg.get("mrr"),
                agg.get("n_failures", 0),
                json.dumps(results.get("failures", []), ensure_ascii=False),
                results.get("elapsed_ms", 0),
            ),
        )
        conn.commit()
        # Retention: keep last 100
        conn.execute(
            "DELETE FROM recall_eval_results WHERE id NOT IN "
            "(SELECT id FROM recall_eval_results ORDER BY created_at DESC LIMIT 100)"
        )
        conn.commit()
    except Exception as e:
        _logger.debug("store_eval_results failed: %s", e)
    finally:
        conn.close()


def get_latest_eval_summary():
    """Get latest eval results for dashboard integration."""
    conn = _get_conn()
    if not conn:
        return {"status": "unavailable"}
    try:
        ensure_tables(conn)
        row = conn.execute(
            "SELECT precision_at_5, recall_at_5, mrr, n_cases, n_failures, created_at "
            "FROM recall_eval_results ORDER BY created_at DESC LIMIT 1"
        ).fetchone()
        if not row:
            return {"status": "no_data"}
        return {
            "status": "ok",
            "precision_at_5": row[0],
            "recall_at_5": row[1],
            "mrr": row[2],
            "n_cases": row[3],
            "n_failures": row[4],
            "last_run": row[5],
        }
    finally:
        conn.close()


def register_tools(mcp):
    """Register recall eval MCP tools."""

    @mcp.tool()
    def eval_recall_quality(seed_if_empty: bool = True) -> str:
        """Run recall quality evaluation and return metrics report.

        Runs 10 standardized queries against memory, measures precision@5,
        recall@5, and MRR. Results stored for trending.

        Args:
            seed_if_empty: Auto-seed with 10 critical test cases if no cases exist.
        """
        if seed_if_empty:
            seed_eval_cases()

        cases = load_active_cases()
        if not cases:
            return json.dumps({"status": "no_cases", "message": "No eval cases found."})

        results = run_eval_suite(cases, limit=10)
        store_eval_results(results, run_source="manual")

        agg = results["aggregate"]
        report = "# Recall Quality Report\n\n"
        report += f"**Cases:** {agg['n_cases']}\n"
        report += f"**Precision@5:** {agg['precision_at_5']}\n"
        report += f"**Recall@5:** {agg['recall_at_5']}\n"
        report += f"**MRR:** {agg['mrr']}\n"
        report += f"**Failures:** {agg['n_failures']}/{agg['n_cases']}\n"
        report += f"**Elapsed:** {results['elapsed_ms']}ms\n"

        if results["failures"]:
            report += "\n## Failed Queries\n"
            for f in results["failures"]:
                report += f"- [{f['category']}] {f['query']} (expected: {', '.join(f['expected_keywords'])})\n"

        if agg["mrr"] >= 0.7:
            report += "\n**Status:** HEALTHY"
        elif agg["mrr"] >= 0.3:
            report += "\n**Status:** DEGRADED"
        else:
            report += "\n**Status:** CRITICAL"

        return report


# Seed eval cases
SEED_CASES = [
    {"query": "cerebro operativo", "keywords": ["cerebro", "operativo", "consciencia", "5 capas"], "category": "bilingual"},
    {"query": "source tracking implementation", "keywords": ["source", "tracking", "Johnson", "memory_sources"], "category": "episodic"},
    {"query": "edit tracking hooks", "keywords": ["edit_tracker", "PostToolUse", "session_edits"], "category": "episodic"},
    {"query": "query expansion bilingual", "keywords": ["query_expansion", "aliases", "bilingual", "BM25"], "category": "episodic"},
    {"query": "cross loop silent dead", "keywords": ["CX", "dead", "silent", "wiring"], "category": "episodic"},
    {"query": "sleep loop consolidation ticks", "keywords": ["sleep", "tick", "consolidation"], "category": "semantic"},
    {"query": "NOTEARS causal discovery", "keywords": ["NOTEARS", "causal", "DAG"], "category": "bilingual"},
    {"query": "temporal contiguity boost recall", "keywords": ["temporal", "contiguity", "Howard", "Kahana"], "category": "episodic"},
    {"query": "inferencia activa opciones", "keywords": ["active inference", "EFE", "options"], "category": "bilingual"},
    {"query": "CX-9 homeostatic plasticity", "keywords": ["CX-9", "homeostatic", "Turrigiano", "refractory"], "category": "episodic"},
]


def seed_eval_cases():
    """Seed the eval table with critical test cases if empty."""
    conn = _get_conn()
    if not conn:
        return 0
    try:
        ensure_tables(conn)
        existing = conn.execute("SELECT COUNT(*) FROM recall_eval_cases").fetchone()[0]
        if existing > 0:
            return 0
        count = 0
        for case in SEED_CASES:
            conn.execute(
                "INSERT INTO recall_eval_cases (query, expected_keywords, category) VALUES (?, ?, ?)",
                (case["query"], json.dumps(case["keywords"], ensure_ascii=False), case["category"]),
            )
            count += 1
        conn.commit()
        return count
    finally:
        conn.close()
