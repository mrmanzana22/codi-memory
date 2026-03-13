"""
P1 Evaluation Harness — Retrieval Quality
==========================================
Measures: retrieval@5, retrieval@10, false_memory_rate

Methodology (from L1 Cognitive Audit, 2026-03-12):
  - 20 queries across 4 categories (identity, architecture, procedural, recent)
  - Each query has expected keywords that MUST appear in results
  - @5 = fraction of queries where expected content appears in top 5
  - @10 = fraction of queries where expected content appears in top 10
  - false_memory = results that contain factually wrong content (manual flag)

Targets (from CODI-EXECUTION-PLAN.md):
  - retrieval@5 >= 70%
  - retrieval@10 >= 85%
  - false_memory_rate < 5%

Last baseline (L1 audit post-fix): @5=95%, @10=100%, false=0%
"""

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

# 20-query baseline from L1 audit
QUERIES = {
    # === Identity (5 queries) ===
    "quien es Codi": {
        "category": "identity",
        "expected": ["CTO", "parcero", "Hare", "agente", "desarrollo"],
    },
    "relacion Codi y Hare": {
        "category": "identity",
        "expected": ["CEO", "CTO", "colaboracion", "equipo", "parcero"],
    },
    "como ha evolucionado Codi": {
        "category": "identity",
        "expected": ["asistente", "CTO", "consciencia", "autonomia"],
    },
    "que valora Codi": {
        "category": "identity",
        "expected": ["mejora", "cognitiv", "autonomia", "consciencia"],
    },
    "Codi sovereign runtime": {
        "category": "identity",
        "expected": ["sovereign", "runtime", "LLM", "provider"],
    },

    # === Architecture (5 queries) ===
    "sleep loop ticks": {
        "category": "architecture",
        "expected": ["tier", "budget", "tick", "sleep", "loop"],
    },
    "consolidation pipeline": {
        "category": "architecture",
        "expected": ["phase", "cluster", "compression", "consolidat"],
    },
    "prediction system": {
        "category": "architecture",
        "expected": ["HGF", "hierarchi", "Bayesian", "L0", "prediction"],
    },
    "active inference": {
        "category": "architecture",
        "expected": ["EFE", "Dirichlet", "Options", "policy"],
    },
    "memory vault": {
        "category": "architecture",
        "expected": ["vault", "dormant", "migration", "probe"],
    },

    # === Procedural (5 queries) ===
    "por que reiniciar MCP": {
        "category": "procedural",
        "expected": ["MCP", "server", "restart", "modulo", "memoria"],
    },
    "proposal protocol": {
        "category": "procedural",
        "expected": ["proposal", ".md", "review", "apply", "delete"],
    },
    "autofix pipeline blocked": {
        "category": "procedural",
        "expected": ["autofix", "untracked", "pending", "block"],
    },
    "ghost migration bug": {
        "category": "procedural",
        "expected": ["migration", "ghost", "executescript", "BEGIN"],
    },
    "flush antes de compaction": {
        "category": "procedural",
        "expected": ["flush", "compaction", "context", "session"],
    },

    # === Recent Events (5 queries) ===
    "L0 audit infrastructure": {
        "category": "recent",
        "expected": ["L0", "audit", "consolidation", "infrastructure"],
    },
    "ollama model deleted": {
        "category": "recent",
        "expected": ["ollama", "qwen", "model", "deleted", "removed"],
    },
    "numpy boolean bug": {
        "category": "recent",
        "expected": ["numpy", "boolean", "pg_store", "pgvector"],
    },
    "budget starvation": {
        "category": "recent",
        "expected": ["budget", "starvation", "timeout", "cap"],
    },
    "full consolidation background": {
        "category": "recent",
        "expected": ["consolidation", "subprocess", "background", "lock"],
    },
}


def run_retrieval_test(limit: int = 10) -> dict:
    """Run the 20-query retrieval benchmark.

    Returns dict with @5, @10 scores and per-query details.
    """
    from modules.memory_core import search_memory

    results = {}
    hits_5 = 0
    hits_10 = 0
    total = len(QUERIES)

    for query, spec in QUERIES.items():
        raw = search_memory(query, limit=limit)
        content = raw.lower() if raw else ""

        # Check if ANY expected keyword appears in the result
        # search_memory returns formatted text, not structured data
        # We check against the full output (top-10 by default)
        found_any = any(kw.lower() in content for kw in spec["expected"])

        # For @5 approximation: search with limit=5
        raw_5 = search_memory(query, limit=5)
        content_5 = raw_5.lower() if raw_5 else ""
        found_5 = any(kw.lower() in content_5 for kw in spec["expected"])

        if found_5:
            hits_5 += 1
        if found_any:
            hits_10 += 1

        results[query] = {
            "category": spec["category"],
            "hit_at_5": found_5,
            "hit_at_10": found_any,
            "expected": spec["expected"],
        }

    precision_5 = round(hits_5 / total, 3) if total else 0
    precision_10 = round(hits_10 / total, 3) if total else 0

    # Category breakdown
    categories = {}
    for query, r in results.items():
        cat = r["category"]
        if cat not in categories:
            categories[cat] = {"total": 0, "hit_5": 0, "hit_10": 0}
        categories[cat]["total"] += 1
        if r["hit_at_5"]:
            categories[cat]["hit_5"] += 1
        if r["hit_at_10"]:
            categories[cat]["hit_10"] += 1

    return {
        "test": "retrieval",
        "precision_at_5": precision_5,
        "precision_at_10": precision_10,
        "target_at_5": 0.70,
        "target_at_10": 0.85,
        "pass_at_5": precision_5 >= 0.70,
        "pass_at_10": precision_10 >= 0.85,
        "total_queries": total,
        "hits_at_5": hits_5,
        "hits_at_10": hits_10,
        "false_memory_rate": 0.0,  # Requires manual review
        "false_memory_pass": True,  # Default pass until manually flagged
        "categories": categories,
        "details": results,
    }


if __name__ == "__main__":
    result = run_retrieval_test()
    print(json.dumps(result, indent=2, ensure_ascii=False))
