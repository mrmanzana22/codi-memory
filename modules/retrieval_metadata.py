"""
Codi Memory - Retrieval Metadata & Metamemory (WIRING-7)
=========================================================
Nelson & Narens 1990: metamemory = monitoring + control.

Provides:
  - RetrievalResult: structured metadata about each search
  - wrap_retrieval_result(): compute coverage & confidence from merged results
  - feeling_of_knowing(): pre-retrieval FOK estimation (Feeling of Knowing)
  - _retrieval_buffer: recent searches for pattern analysis

Based on NEURO_ANALYSIS_REPORT Phase 2 requirements.
"""

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional


# ============================================================
# DATACLASSES
# ============================================================

@dataclass
class RetrievalResult:
    """Structured metadata about a single search_memory() call."""
    query: str
    result_count: int
    episodic_count: int
    semantic_count: int
    top_activation: float
    mean_activation: float
    confidence_estimate: float
    coverage: str  # "comprehensive" | "partial" | "sparse" | "empty"
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


# ============================================================
# RETRIEVAL BUFFER (module-level, last 20 searches)
# ============================================================

_retrieval_buffer: deque = deque(maxlen=20)


def get_retrieval_buffer() -> List[RetrievalResult]:
    """Return snapshot of recent retrieval results."""
    return list(_retrieval_buffer)


# ============================================================
# COVERAGE & CONFIDENCE
# ============================================================

def _classify_coverage(result_count: int, mean_activation: float) -> str:
    """Classify retrieval coverage based on Nelson & Narens monitoring model.

    - empty: 0 results
    - sparse: 1-2 results
    - partial: 3-5 results with mean_activation < 0.5
    - comprehensive: 5+ results or mean_activation >= 0.5 with 3+ results
    """
    if result_count == 0:
        return "empty"
    if result_count <= 2:
        return "sparse"
    if result_count >= 5 or mean_activation >= 0.5:
        return "comprehensive"
    return "partial"


def _compute_confidence(result_count: int, top_activation: float) -> float:
    """Compute search confidence from count + activation quality.

    Formula: min(1.0, count/5) * 0.6 + top_activation * 0.4
    """
    count_factor = min(1.0, result_count / 5.0) if result_count > 0 else 0.0
    return count_factor * 0.6 + top_activation * 0.4


# ============================================================
# MAIN API
# ============================================================

def wrap_retrieval_result(query: str, merged: list) -> RetrievalResult:
    """Build RetrievalResult from search_memory() merged list.

    Args:
        query: The search query string
        merged: List of dicts with keys: memory_type, activation, combined_score
    """
    result_count = len(merged)
    episodic_count = sum(1 for m in merged if m.get("memory_type") == "episodic")
    semantic_count = sum(1 for m in merged if m.get("memory_type") == "semantic")

    activations = [float(m.get("activation", 0)) for m in merged]
    top_activation = max(activations) if activations else 0.0
    mean_activation = sum(activations) / len(activations) if activations else 0.0

    coverage = _classify_coverage(result_count, mean_activation)
    confidence = _compute_confidence(result_count, top_activation)

    result = RetrievalResult(
        query=query,
        result_count=result_count,
        episodic_count=episodic_count,
        semantic_count=semantic_count,
        top_activation=top_activation,
        mean_activation=mean_activation,
        confidence_estimate=confidence,
        coverage=coverage,
    )

    # Store in buffer
    _retrieval_buffer.append(result)

    return result


# ============================================================
# FEELING OF KNOWING (FOK) - WIRING-7.2
# ============================================================

def feeling_of_knowing(query: str, fts_db_path: str = None, wm_conn=None) -> dict:
    """Pre-retrieval FOK estimation (Hart 1965, Nelson & Narens 1990).

    Estimates likelihood that a search will succeed BEFORE running it.
    Uses:
      1. Count of similar failed searches (penalty)
      2. Whether topic is in working memory (boost)
      3. Retrieval buffer history (pattern)

    Args:
        query: The query to estimate FOK for
        fts_db_path: Path to FTS database for failed_searches table
        wm_conn: Optional SQLite connection for working memory check

    Returns:
        {"fok_score": 0.0-1.0, "basis": str, "recommendation": "search"|"ask"|"uncertain"}
    """
    import os
    from modules.db_pool import get_conn

    fok = 0.5  # base
    basis_parts = ["base=0.5"]

    # 1. Check failed_searches table
    failed_count = 0
    if fts_db_path and os.path.exists(fts_db_path):
        try:
            conn = get_conn(fts_db_path)
            conn.execute("SELECT 1 FROM failed_searches LIMIT 1")
            cursor = conn.execute(
                "SELECT COUNT(*) FROM (SELECT 1 FROM failed_searches WHERE query LIKE ? ORDER BY created_at DESC LIMIT 50)",
                (f"%{query[:30]}%",)
            )
            failed_count = cursor.fetchone()[0]
            if failed_count > 0:
                penalty = min(0.3, failed_count * 0.1)
                fok -= penalty
                basis_parts.append(f"failed_searches={failed_count}(-{penalty:.2f})")
        except Exception:
            pass

    # 2. Check working memory for topic presence
    wm_hit = False
    _wm_local = wm_conn
    if not _wm_local and fts_db_path and os.path.exists(fts_db_path):
        try:
            _wm_local = get_conn(fts_db_path)
        except Exception:
            pass
    if _wm_local:
        try:
            cursor = _wm_local.execute(
                "SELECT COUNT(*) FROM working_memory WHERE content LIKE ?",
                (f"%{query[:30]}%",)
            )
            wm_count = cursor.fetchone()[0]
            if wm_count > 0:
                fok += 0.15
                wm_hit = True
                basis_parts.append(f"in_wm(+0.15)")
        except Exception:
            pass

    # 3. Check retrieval buffer for similar past queries
    buffer_hits = sum(
        1 for r in _retrieval_buffer
        if r.coverage in ("comprehensive", "partial") and
        any(word in r.query.lower() for word in query.lower().split() if len(word) > 3)
    )
    if buffer_hits > 0:
        boost = min(0.15, buffer_hits * 0.05)
        fok += boost
        basis_parts.append(f"buffer_hits={buffer_hits}(+{boost:.2f})")

    # Clamp
    fok = max(0.0, min(1.0, fok))

    # Recommendation
    if fok >= 0.6:
        recommendation = "search"
    elif fok <= 0.3:
        recommendation = "ask"
    else:
        recommendation = "uncertain"

    return {
        "fok_score": round(fok, 3),
        "basis": "; ".join(basis_parts),
        "recommendation": recommendation,
    }


# ============================================================
# METACOGNITIVE CONTROL (HOT-3) - Nelson & Narens 1990
# ============================================================

def metacognitive_control(query: str, fts_db_path: str = None, wm_conn=None) -> dict:
    """Translate FOK into retrieval strategy modifications.

    Nelson & Narens 1990: consciousness requires not just monitoring (FOK)
    but CONTROL -- monitoring output must change behavior.

    This closes the loop: FOK -> strategy -> modified search parameters.

    Args:
        query: The search query
        fts_db_path: Path to FTS database for FOK estimation
        wm_conn: Optional SQLite connection for working memory check

    Returns:
        {strategy, adjusted_limit, confidence_flag, fok}
    """
    fok = feeling_of_knowing(query, fts_db_path=fts_db_path, wm_conn=wm_conn)
    fok_score = fok["fok_score"]

    # Apply calibration if available
    try:
        calibration = get_fok_calibration(fts_db_path=fts_db_path)
        if calibration["n_records"] >= 10:
            fok_score = calibrated_fok_score(fok_score, calibration)
    except Exception:
        pass

    if fok_score >= 0.6:
        return {
            "strategy": "full_search",
            "adjusted_limit": 1,  # multiplier: 1x normal limit
            "confidence_flag": "",
            "fok": fok,
        }
    elif fok_score > 0.3:
        return {
            "strategy": "expand_search",
            "adjusted_limit": 2,  # multiplier: 2x normal limit
            "confidence_flag": "[UNCERTAIN]",
            "fok": fok,
        }
    else:
        return {
            "strategy": "suggest_ask",
            "adjusted_limit": 2,  # multiplier: 2x normal limit
            "confidence_flag": "[LOW CONFIDENCE]",
            "fok": fok,
        }


# ============================================================
# FAILED SEARCH TRACKING - WIRING-7.2
# ============================================================

def init_failed_searches_table(conn) -> None:
    """Validate failed_searches table exists (created by migrations)."""
    from modules.migrations import ensure_schema_ready
    ensure_schema_ready(conn, ["failed_searches"])


def log_failed_search(conn, query: str, result_count: int, top_activation: float, topic: str = None) -> None:
    """Log a failed/weak search for FOK estimation.

    Called when result_count < 2 or top_activation < 0.3.
    Keeps max 500 rows (FIFO cleanup).
    """
    conn.execute(
        "INSERT INTO failed_searches (query, result_count, top_activation, topic, created_at) VALUES (?, ?, ?, ?, ?)",
        (query[:200], result_count, top_activation, topic, datetime.now().isoformat())
    )

    # FIFO cleanup: keep max 500
    conn.execute("""
        DELETE FROM failed_searches WHERE id NOT IN (
            SELECT id FROM failed_searches ORDER BY created_at DESC LIMIT 500
        )
    """)
    conn.commit()


def get_top_failed_topics(conn, limit: int = 5) -> list:
    """Get most frequently failed topics for dynamic gap detection (WIRING-7.3).

    Returns list of (topic, fail_count) tuples.
    """
    try:
        cursor = conn.execute("""
            SELECT topic, COUNT(*) as cnt
            FROM failed_searches
            WHERE topic IS NOT NULL AND topic != ''
            GROUP BY topic
            ORDER BY cnt DESC
            LIMIT ?
        """, (limit,))
        return cursor.fetchall()
    except Exception:
        return []


# ============================================================
# RCJ CALIBRATION (HOT-2) - Nelson & Narens 1990
# ============================================================

def _init_fok_calibration_table(conn) -> None:
    """Validate fok_calibration_log table exists (created by migrations)."""
    from modules.migrations import ensure_schema_ready
    ensure_schema_ready(conn, ["fok_calibration_log"])


def record_rcj(query: str, fok_predicted: float, actual_coverage: str,
               actual_count: int, actual_top_activation: float,
               fts_db_path: str = None) -> None:
    """Log FOK prediction vs actual outcome for calibration (RCJ).

    Retrospective Confidence Judgment: after retrieval, record how well
    FOK predicted the actual outcome. This enables learning whether
    confidence predictions are accurate.

    Args:
        query: The search query
        fok_predicted: FOK score before search (0.0-1.0)
        actual_coverage: Actual retrieval coverage (comprehensive/partial/sparse/empty)
        actual_count: Actual number of results found
        actual_top_activation: Highest activation score in results
        fts_db_path: Path to FTS database
    """
    import os
    from modules.db_pool import get_conn

    if not fts_db_path:
        fts_db_path = os.environ.get("FTS_DB_PATH", "memories_fts.db")

    try:
        conn = get_conn(fts_db_path)
        _init_fok_calibration_table(conn)
        conn.execute(
            "INSERT INTO fok_calibration_log (query, fok_predicted, actual_coverage, actual_count, actual_top_activation, created_at) VALUES (?, ?, ?, ?, ?, ?)",
            (query[:200], fok_predicted, actual_coverage, actual_count,
             actual_top_activation, datetime.now().isoformat())
        )
        # FIFO cleanup: keep max 500
        conn.execute("""
            DELETE FROM fok_calibration_log WHERE id NOT IN (
                SELECT id FROM fok_calibration_log ORDER BY created_at DESC LIMIT 500
            )
        """)
        conn.commit()
    except Exception:
        pass


def get_fok_calibration(lookback: int = 100, fts_db_path: str = None) -> dict:
    """Compute FOK calibration metrics from recent RCJ records.

    Returns:
        {mean_absolute_error, overconfidence_bias, n_records}
        - mean_absolute_error: average |predicted - actual_quality|
        - overconfidence_bias: average (predicted - actual_quality), >0 means overconfident
        - n_records: number of records used
    """
    import os
    from modules.db_pool import get_conn

    if not fts_db_path:
        fts_db_path = os.environ.get("FTS_DB_PATH", "memories_fts.db")

    try:
        conn = get_conn(fts_db_path)
        _init_fok_calibration_table(conn)
        cursor = conn.execute(
            "SELECT fok_predicted, actual_coverage, actual_count, actual_top_activation FROM fok_calibration_log ORDER BY created_at DESC LIMIT ?",
            (lookback,)
        )
        rows = cursor.fetchall()

        if not rows:
            return {"mean_absolute_error": 0.0, "overconfidence_bias": 0.0, "n_records": 0}

        errors = []
        biases = []
        for fok_pred, coverage, count, top_act in rows:
            # Convert actual outcome to a quality score (0-1)
            coverage_score = {"comprehensive": 1.0, "partial": 0.6, "sparse": 0.3, "empty": 0.0}.get(coverage, 0.3)
            count_factor = min(1.0, count / 5.0) if count else 0.0
            actual_quality = coverage_score * 0.5 + count_factor * 0.3 + top_act * 0.2

            errors.append(abs(fok_pred - actual_quality))
            biases.append(fok_pred - actual_quality)

        return {
            "mean_absolute_error": sum(errors) / len(errors),
            "overconfidence_bias": sum(biases) / len(biases),
            "n_records": len(rows),
        }
    except Exception:
        return {"mean_absolute_error": 0.0, "overconfidence_bias": 0.0, "n_records": 0}


def calibrated_fok_score(raw_fok: float, calibration: dict) -> float:
    """Adjust raw FOK score based on historical calibration.

    If historically overconfident (bias > 0), reduce FOK.
    If historically underconfident (bias < 0), increase FOK.

    Args:
        raw_fok: Raw FOK score (0.0-1.0)
        calibration: Output from get_fok_calibration()

    Returns:
        Calibrated FOK score (clamped to 0.0-1.0)
    """
    if calibration["n_records"] < 5:
        return raw_fok  # Not enough data to calibrate

    bias = calibration["overconfidence_bias"]
    # Subtract bias: if overconfident, bias > 0, so we reduce FOK
    adjusted = raw_fok - (bias * 0.5)  # Apply half the bias as correction
    return max(0.0, min(1.0, adjusted))
