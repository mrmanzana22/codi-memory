"""
Codi Memory - Retrieval Metadata & Metamemory (WIRING-7)
=========================================================
Nelson & Narens 1990: metamemory = monitoring + control.

Provides:
  - RetrievalResult: structured metadata about each search
  - wrap_retrieval_result(): compute coverage & confidence from merged results
  - feeling_of_knowing(): pre-retrieval FOK estimation (Feeling of Knowing)
  - compute_memory_confidence(): per-memory reliability score (Koriat 1997)
  - diagnose_retrieval_failure(): classify WHY retrieval failed (Schacter 1999)
  - quality_class: experiential classification (Tulving 1985)
  - _retrieval_buffer: recent searches for pattern analysis

Based on NEURO_ANALYSIS_REPORT Phase 2 requirements.
"""

import math
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional

from modules.config import now_iso
from modules.pg_store import pg


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
    quality_class: str = "blank"  # Tulving 1985 remembering-vs-knowing
    # Values: "confident_recall" | "partial_recall" | "recognition_only" | "tip_of_tongue" | "blank"
    failure_diagnosis: str = ""  # Schacter 1999: why retrieval failed (if applicable)
    # Values: "" | "never_stored" | "decayed" | "tip_of_tongue" | "retrieval_failure"
    cox_failure_type: str = ""  # Cox taxonomy (Sprint 3, item 3.2)
    # Values: "" | "retrieval" | "application" | "knowledge" | "novel" | "success"
    cox_remedy: str = ""  # Recommended remediation action
    timestamp: str = field(default_factory=now_iso)


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
# PER-MEMORY CONFIDENCE (Koriat 1997, Nelson & Narens 1990)
# ============================================================

# Source reliability weights (Nelson & Narens 1990)
_SOURCE_WEIGHTS = {
    "experienced": 1.0,
    "learned": 0.7,
    "told": 0.5,
    "inferred": 0.4,
    "unknown": 0.3,
}


def compute_memory_confidence(payload: dict, activation: float = 0.0,
                              source_verified: bool = None) -> float:
    """Compute per-memory confidence score from 5 signals + source verification.

    Koriat 1997: confidence = retrieval fluency + source reliability.
    Dunlosky & Metcalfe 2009: corroboration increases JOL.
    Johnson et al. 1993 / Nelson & Narens 1990: source monitoring modulates FOK.

    Args:
        payload: Memory's Qdrant payload dict
        activation: ACT-R activation score (0-1) from search scoring
        source_verified: If False, apply 0.10 penalty (no provenance). None = no penalty.

    Returns:
        Confidence score (0.0-1.0)
    """
    # 1. Source reliability (0-1)
    source = payload.get("ownership_source", "unknown")
    source_score = _SOURCE_WEIGHTS.get(source, 0.3)

    # 2. Corroboration (log-normalized evidence_count)
    evidence = int(payload.get("evidence_count", 1) or 1)
    corroboration = min(1.0, math.log(evidence + 1) / math.log(6))  # 5+ evidence = 1.0

    # 3. Contradiction penalty (already tracked by sharpe.py)
    contradictions = int(payload.get("contradiction_count", 0) or 0)
    contradiction_penalty = min(0.45, contradictions * 0.15)

    # 4. Retrieval fluency (activation normalized 0-1)
    fluency = max(0.0, min(1.0, activation))

    # 5. Staleness (inverse hours since last access)
    staleness_score = 0.5  # default if no timing data
    last_accessed = payload.get("attention_last_accessed", "")
    if last_accessed:
        try:
            last_dt = datetime.fromisoformat(
                str(last_accessed).replace("Z", "+00:00")
            )
            # Use now_col() (Colombia-aware) for consistent tz comparison.
            # attention_last_accessed is written with now_iso() (Colombia-aware).
            from modules.config import now_col as _now_col
            hours_since = max(0, (_now_col() - last_dt).total_seconds() / 3600)
            staleness_score = math.exp(-0.01 * hours_since)  # ~0.79 at 24h, ~0.51 at 7d
        except (ValueError, TypeError):
            pass

    # Weighted combination
    raw = (
        0.25 * source_score
        + 0.20 * corroboration
        + 0.25 * fluency
        + 0.15 * staleness_score
        + 0.15 * 1.0  # base reliability
    )

    # 6. Source verification penalty (Johnson et al. 1993, Nelson & Narens 1990)
    # Memories without provenance tracking are less trustworthy
    source_penalty = 0.10 if source_verified is False else 0.0

    return round(max(0.0, min(1.0, raw - contradiction_penalty - source_penalty)), 3)


# ============================================================
# QUALITY SPACE (Tulving 1985, Yonelinas 2002)
# ============================================================

def _classify_quality(coverage: str, mean_activation: float, confidence: float) -> str:
    """Classify retrieval experience quality.

    Tulving 1985: remembering (vivid, detailed) vs knowing (familiar, vague).
    Yonelinas 2002: dual-process model of recognition.

    Returns: confident_recall | partial_recall | recognition_only | blank
             (tip_of_tongue is set by diagnose_retrieval_failure)
    """
    if coverage == "empty":
        return "blank"
    if coverage == "comprehensive" and confidence >= 0.7:
        return "confident_recall"
    if coverage in ("comprehensive", "partial") and confidence >= 0.4:
        return "partial_recall"
    return "recognition_only"


# ============================================================
# FAILURE DIAGNOSTICS (Schacter 1999)
# ============================================================

def diagnose_retrieval_failure(
    coverage: str,
    result_count: int,
    top_activation: float,
    query: str,
) -> str:
    """Diagnose WHY retrieval failed (Schacter 1999 seven sins of memory).

    Only meaningful when coverage is sparse or empty.

    Categories:
    - never_stored: no memory of this topic ever existed
    - decayed: was stored but has faded (trace decay)
    - tip_of_tongue: almost there, partial retrieval (Brown & McNeill 1966)
    - retrieval_failure: stored but can't access (cue mismatch)

    Returns:
        Diagnosis string or "" if not a failure
    """
    if coverage not in ("empty", "sparse"):
        return ""

    # Check past successes in the retrieval buffer
    query_words = {w.lower() for w in query.split() if len(w) > 3}
    past_success = any(
        r.coverage in ("comprehensive", "partial")
        and any(w in r.query.lower() for w in query_words)
        for r in _retrieval_buffer
    ) if query_words else False

    if coverage == "empty":
        if past_success:
            return "decayed"
        return "never_stored"

    # Sparse: 1-2 results
    if top_activation < 0.3:
        return "tip_of_tongue"

    if past_success:
        return "retrieval_failure"

    return "retrieval_failure"


def classify_failure_cox(
    diagnosis: str,
    query: str,
    result_count: int,
    channel_scores: dict = None,
    fts_db_path: str = None,
) -> dict:
    """Classify retrieval failure using Cox taxonomy (Sprint 3, item 3.2).

    Four failure types with specific remediation strategies:
    - retrieval: Memory exists but wrong cues used (cue-dependent forgetting).
      Remedy: reformulate query, use different keywords.
    - application: Wrong retrieval strategy (channel mismatch).
      Remedy: try different channel allocation.
    - knowledge: Topic known but specific knowledge missing (knowledge gap).
      Remedy: acquire new information, study.
    - novel: Truly novel topic, no prior knowledge exists.
      Remedy: acknowledge ignorance, explore from scratch.

    Cox & Griggs 1982 failure taxonomy, adapted for memory systems.
    Schacter 1999 seven sins provide the diagnostic input.

    Args:
        diagnosis: From diagnose_retrieval_failure() (never_stored, decayed, etc.)
        query: The search query
        result_count: Number of results returned
        channel_scores: Optional {channel: score} from Thompson sampling
        fts_db_path: Path to FTS database

    Returns:
        {type, label, remedy, confidence, diagnosis}
    """
    # Default
    cox_type = "novel"
    label = "Novel topic"
    remedy = "Acknowledge ignorance. Explore from scratch."
    confidence = 0.5

    if not diagnosis or diagnosis == "":
        # Not a failure — retrieval succeeded
        return {
            "type": "success",
            "label": "Retrieval succeeded",
            "remedy": "",
            "confidence": 1.0,
            "diagnosis": "",
        }

    if diagnosis == "never_stored":
        # Check if related topics exist (knowledge vs novel)
        has_related = False
        try:
            from modules.config import FTS_DB_PATH as _fts_path, connect_fts
            _path = fts_db_path or _fts_path
            conn = connect_fts(_path)
            try:
                # Check if any words from query appear in FTS index
                words = [w for w in query.lower().split() if len(w) > 3]
                for w in words[:3]:
                    row = conn.execute(
                        "SELECT COUNT(*) FROM memories_fts WHERE memories_fts MATCH ?",
                        (w,)
                    ).fetchone()
                    if row and row[0] > 0:
                        has_related = True
                        break
            finally:
                conn.close()
        except Exception:
            pass

        if has_related:
            cox_type = "knowledge"
            label = "Knowledge gap"
            remedy = "Related topics exist but specific knowledge missing. Study or acquire."
            confidence = 0.6
        else:
            cox_type = "novel"
            label = "Novel topic"
            remedy = "No prior knowledge. Explore from scratch."
            confidence = 0.7

    elif diagnosis == "decayed":
        cox_type = "retrieval"
        label = "Retrieval failure (decayed trace)"
        remedy = "Memory existed but decayed. Try broader cues or check semantic store."
        confidence = 0.6

    elif diagnosis == "tip_of_tongue":
        cox_type = "retrieval"
        label = "Retrieval failure (partial access)"
        remedy = "Memory partially accessible. Reformulate query with different keywords."
        confidence = 0.7

    elif diagnosis == "retrieval_failure":
        # Distinguish retrieval vs application by checking channel diversity
        if channel_scores:
            dominant_channel = max(channel_scores, key=channel_scores.get)
            score_range = max(channel_scores.values()) - min(channel_scores.values())
            if score_range > 0.3:
                # Strong channel preference — might have picked wrong one
                cox_type = "application"
                label = "Application failure (strategy mismatch)"
                remedy = f"Channel {dominant_channel} dominated. Try rebalancing retrieval strategy."
                confidence = 0.5
            else:
                cox_type = "retrieval"
                label = "Retrieval failure (cue mismatch)"
                remedy = "Stored but inaccessible. Try alternative search terms."
                confidence = 0.6
        else:
            cox_type = "retrieval"
            label = "Retrieval failure (cue mismatch)"
            remedy = "Stored but inaccessible. Try alternative search terms."
            confidence = 0.6

    return {
        "type": cox_type,
        "label": label,
        "remedy": remedy,
        "confidence": confidence,
        "diagnosis": diagnosis,
    }


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

    # Quality space classification (Tulving 1985)
    quality = _classify_quality(coverage, mean_activation, confidence)

    # Failure diagnosis (Schacter 1999) -- only for sparse/empty
    diagnosis = diagnose_retrieval_failure(coverage, result_count, top_activation, query)

    # Cox failure taxonomy (Sprint 3, item 3.2) — type-specific remediation
    cox = classify_failure_cox(diagnosis, query, result_count)

    # Override quality_class for tip_of_tongue (Brown & McNeill 1966)
    if diagnosis == "tip_of_tongue":
        quality = "tip_of_tongue"

    result = RetrievalResult(
        query=query,
        result_count=result_count,
        episodic_count=episodic_count,
        semantic_count=semantic_count,
        top_activation=top_activation,
        mean_activation=mean_activation,
        confidence_estimate=confidence,
        coverage=coverage,
        quality_class=quality,
        failure_diagnosis=diagnosis,
        cox_failure_type=cox["type"],
        cox_remedy=cox["remedy"],
    )

    # Store in buffer
    _retrieval_buffer.append(result)

    return result


# ============================================================
# FEELING OF KNOWING (FOK) - WIRING-7.2
# ============================================================

def _get_topic_from_query(query: str) -> str:
    """Extract primary topic from query for Beta-Binomial lookup."""
    from modules.config import TOPIC_KEYWORDS
    q_lower = query.lower()
    for topic, keywords in TOPIC_KEYWORDS.items():
        if any(kw in q_lower for kw in keywords):
            return topic
    # Fallback: first significant word
    words = [w for w in query.split() if len(w) > 3]
    return words[0].lower() if words else "general"


def _get_beta_prior(topic: str, fts_db_path: str) -> tuple:
    """Get Beta(alpha, beta) prior for a topic from SQLite.

    Returns (alpha, beta) with default (1.0, 1.0) = uniform prior.
    """
    import os
    if not fts_db_path or not os.path.exists(fts_db_path):
        return (1.0, 1.0)
    try:
        from modules.db_pool import get_conn
        conn = get_conn(fts_db_path)
        row = conn.execute(
            "SELECT alpha, beta FROM fok_priors WHERE topic = ?", (topic,)
        ).fetchone()
        if row:
            return (float(row[0]), float(row[1]))
    except Exception:
        pass
    return (1.0, 1.0)


def update_fok_prior(topic: str, success: bool, fts_db_path: str = None):
    """Conjugate update of Beta-Binomial FOK prior (S1-01).

    Beta(a, b) + observation -> Beta(a+1, b) if success, Beta(a, b+1) if failure.
    Discount: multiply both by gamma=0.995 before update (non-stationary).

    Args:
        topic: Topic string
        success: True if retrieval was successful (comprehensive/partial)
        fts_db_path: Path to FTS database
    """
    import os
    if not fts_db_path:
        from modules.config import FTS_DB_PATH
        fts_db_path = FTS_DB_PATH
    if not os.path.exists(fts_db_path):
        return

    DISCOUNT_GAMMA = 0.995  # Non-stationary discount

    try:
        from modules.db_pool import get_conn
        conn = get_conn(fts_db_path)
        row = conn.execute(
            "SELECT alpha, beta, n_observations FROM fok_priors WHERE topic = ?",
            (topic,)
        ).fetchone()

        if row:
            a, b, n = float(row[0]), float(row[1]), int(row[2])
            # Discount (forget old observations slowly)
            a *= DISCOUNT_GAMMA
            b *= DISCOUNT_GAMMA
        else:
            a, b, n = 1.0, 1.0, 0

        # Conjugate update
        if success:
            a += 1.0
        else:
            b += 1.0

        conn.execute("""
            INSERT INTO fok_priors (topic, alpha, beta, n_observations, last_updated)
            VALUES (?, ?, ?, ?, datetime('now'))
            ON CONFLICT(topic) DO UPDATE SET
                alpha = excluded.alpha,
                beta = excluded.beta,
                n_observations = excluded.n_observations,
                last_updated = excluded.last_updated
        """, (topic, round(a, 4), round(b, 4), n + 1))
    except Exception:
        pass


def brier_score(predicted: float, actual: bool) -> float:
    """Brier score: proper scoring rule for probabilistic predictions.

    BS = (predicted - actual)^2. Lower is better. Range [0, 1].
    """
    return (predicted - (1.0 if actual else 0.0)) ** 2


# Sprint 9: Dual-process FOK blend weight (adapts over time)
# Alpha = weight given to familiarity vs accessibility
# Metcalfe et al. 1993: two signals, adaptive combination
_FOK_ALPHA_DEFAULT = 0.6  # Trust familiarity slightly more initially


def estimate_familiarity(query: str, fts_db_path: str = None, wm_conn=None) -> float:
    """Sprint 9.1: Fast familiarity signal — cheap signals only (< 5ms).

    Uses: FTS5 match count, topic keyword match, working memory overlap.
    NO embeddings, NO vector calls.

    Metcalfe et al. 1993: familiarity is automatic and fast.
    Reder 1992: familiarity based on cue strength, not accessibility.

    Returns:
        Familiarity score 0-1.
    """
    import os
    score = 0.0
    n_signals = 0

    # Signal 1: FTS keyword match count
    if fts_db_path and os.path.exists(fts_db_path):
        try:
            from modules.db_pool import get_conn as _get_conn
            conn = _get_conn(fts_db_path)
            words = [w for w in query.split() if len(w) > 2 and w.isalnum()]
            if words:
                fts_q = " OR ".join(words[:5])
                cnt = conn.execute(
                    "SELECT COUNT(*) FROM memories_fts WHERE content MATCH ?",
                    (fts_q,)
                ).fetchone()[0]
                score += min(1.0, math.log(cnt + 1) / math.log(100))
                n_signals += 1
        except Exception:
            pass

    # Signal 2: Topic keyword match
    try:
        topic = _get_topic_from_query(query)
        if topic != "general":
            score += 0.4
            n_signals += 1
        else:
            n_signals += 1
    except Exception:
        pass

    # Signal 3: Working memory presence (recent context) — PostgreSQL backend
    try:
        from modules.working_memory import wm_search_content
        cnt = wm_search_content(query[:20], active_only=True)
        if cnt > 0:
            score += 0.3
        n_signals += 1
    except Exception:
        n_signals += 1

    if n_signals == 0:
        return 0.5
    return max(0.0, min(1.0, score / n_signals))


def estimate_accessibility(query: str, fts_db_path: str = None) -> float:
    """Sprint 9.2: Slow accessibility probe — vector search for top result (< 50ms).

    Checks if query has a strong match in semantic/episodic store.
    score > 0.8 = accessible, 0.5-0.8 = tip-of-tongue, < 0.5 = inaccessible.

    Koriat 1993: accessibility reflects actual memory trace strength.

    Returns:
        Accessibility score 0-1.
    """
    try:
        from modules.consolidation_common import _embed_text_cached

        vec = _embed_text_cached(query[:200])
        if not vec:
            return 0.5

        results = pg.query_vector(vec, limit=1, is_semantic=False)
        if not results:
            return 0.1
        top_score = float(results[0].score)
        return max(0.0, min(1.0, top_score))
    except Exception:
        return 0.5


def _get_fok_dual_alpha(fts_db_path: str = None) -> float:
    """Sprint 9.4: Read adaptive alpha from metamemory_params.

    Alpha starts at 0.6 (trust familiarity more). Adapts from calibration history.
    Returns alpha in [0.2, 0.8] range.
    """
    try:
        import os
        if not fts_db_path:
            from modules.config import FTS_DB_PATH as _fts
            fts_db_path = os.environ.get("FTS_DB_PATH", _fts)
        from modules.db_pool import get_conn as _get_conn
        conn = _get_conn(fts_db_path)
        row = conn.execute(
            "SELECT value FROM metamemory_params WHERE param = 'fok_dual_alpha'"
        ).fetchone()
        if row:
            return max(0.2, min(0.8, float(row[0])))
    except Exception:
        pass
    return _FOK_ALPHA_DEFAULT


def feeling_of_knowing(query: str, fts_db_path: str = None, wm_conn=None) -> dict:
    """Pre-retrieval FOK estimation using Beta-Binomial posterior (S1-01).

    Sprint 9.3: Upgraded to dual-process model (Metcalfe et al. 1993, Koriat 1993).
    FOK = alpha * familiarity + (1-alpha) * accessibility
    Alpha adapts over time based on which signal is more predictive (Sprint 9.4).

    S1-01: Replaces additive heuristic with Beta(a,b) conjugate model.
    Base FOK = E[Beta(a,b)] = a/(a+b) for the query's topic.
    Evidence adjustments from FTS count and working memory still apply
    but are now additive corrections to the Bayesian base.

    Hart 1965, Nelson & Narens 1990.

    Args:
        query: The query to estimate FOK for
        fts_db_path: Path to FTS database
        wm_conn: Optional SQLite connection for working memory check

    Returns:
        {"fok_score": 0.0-1.0, "basis": str, "recommendation": str,
         "brier_ready": bool, "topic": str}
    """
    import os
    from modules.db_pool import get_conn

    topic = _get_topic_from_query(query)
    basis_parts = []

    # S1-01: Beta-Binomial base (conjugate posterior mean)
    alpha, beta_param = _get_beta_prior(topic, fts_db_path)
    bayesian_base = alpha / (alpha + beta_param)
    fok = bayesian_base
    basis_parts.append(f"Beta({alpha:.1f},{beta_param:.1f})={bayesian_base:.3f}")

    # Evidence 1: FTS count (Reder 1992 accessibility heuristic)
    if fts_db_path and os.path.exists(fts_db_path):
        try:
            fts_conn = get_conn(fts_db_path)
            words = [w for w in query.split() if len(w) > 2 and w.isalnum()]
            if words:
                fts_query = " OR ".join(words[:5])
                cursor = fts_conn.execute(
                    "SELECT COUNT(*) FROM memories_fts WHERE content MATCH ?",
                    (fts_query,)
                )
                fts_count = cursor.fetchone()[0]
                if fts_count > 0:
                    boost = min(0.25, 0.08 * math.log(fts_count + 1))
                    fok += boost
                    basis_parts.append(f"fts={fts_count}(+{boost:.2f})")
                else:
                    fok -= 0.1
                    basis_parts.append("fts=0(-0.10)")
        except Exception:
            pass

    # Evidence 2: Working memory presence — PostgreSQL backend
    try:
        from modules.working_memory import wm_search_content
        cnt = wm_search_content(query[:30], active_only=False)
        if cnt > 0:
            fok += 0.12
            basis_parts.append("wm(+0.12)")
    except Exception:
        pass

    # Clamp beta-based FOK (this becomes the familiarity signal)
    familiarity = max(0.0, min(1.0, fok))

    # Sprint 9.1-9.3: Dual-process integration (Metcalfe et al. 1993)
    # Fast familiarity already computed above (FTS + WM signals)
    # Now compute slow accessibility (vector top-1 score)
    fok_process = "familiarity_only"
    accessibility = None

    try:
        import time as _time
        _t_start = _time.monotonic()
        accessibility = estimate_accessibility(query, fts_db_path)
        _acc_ms = (_time.monotonic() - _t_start) * 1000

        if _acc_ms < 3500:  # Proposal #111: pgvector p99 ~3022ms (was 60ms for local Qdrant)
            dual_alpha = _get_fok_dual_alpha(fts_db_path)
            fok = dual_alpha * familiarity + (1.0 - dual_alpha) * accessibility
            fok_process = "dual_process"
            basis_parts.append(
                f"dual(fam={familiarity:.2f},acc={accessibility:.2f},α={dual_alpha:.2f})"
            )
        else:
            fok = familiarity
            basis_parts.append(f"familiarity_only(acc_timeout={_acc_ms:.0f}ms)")
    except Exception:
        fok = familiarity

    # Clamp combined FOK
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
        "topic": topic,
        "brier_ready": (alpha + beta_param) > 3.0,  # Enough data for Brier scoring
        "fok_process": fok_process,              # Sprint 9.3: process used
        "familiarity_score": round(familiarity, 3),  # Sprint 9.4: for adaptive alpha
        "accessibility_score": round(accessibility, 3) if accessibility is not None else None,
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

    # Apply calibration if available (Nelson & Narens 1990: monitoring + control)
    try:
        calibration = get_fok_calibration(fts_db_path=fts_db_path)
        if calibration["n_records"] >= 10:
            # Read adaptive correction factor from self_model (Proposal #64 Fix 2)
            correction_factor = 0.5
            try:
                import sqlite3 as _sql
                _conn = _sql.connect(fts_db_path, timeout=3)
                _row = _conn.execute(
                    "SELECT value FROM metamemory_params "
                    "WHERE param = 'fok_correction_factor'"
                ).fetchone()
                if _row:
                    correction_factor = float(_row[0])
                _conn.close()
            except Exception:
                pass
            fok_score = calibrated_fok_score(fok_score, calibration, correction_factor)
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
        (query[:200], result_count, top_activation, topic, now_iso())
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
               fts_db_path: str = None,
               familiarity_score: float = None,
               accessibility_score: float = None,
               fok_process: str = None) -> None:
    """Log FOK prediction vs actual outcome for calibration (RCJ).

    Retrospective Confidence Judgment: after retrieval, record how well
    FOK predicted the actual outcome. This enables learning whether
    confidence predictions are accurate.

    Sprint 9.4: Also logs familiarity_score and accessibility_score for
    adaptive alpha computation (which signal was more predictive?).

    Args:
        query: The search query
        fok_predicted: FOK score before search (0.0-1.0)
        actual_coverage: Actual retrieval coverage (comprehensive/partial/sparse/empty)
        actual_count: Actual number of results found
        actual_top_activation: Highest activation score in results
        fts_db_path: Path to FTS database
        familiarity_score: Familiarity component used (Sprint 9.4)
        accessibility_score: Accessibility component used (Sprint 9.4)
        fok_process: Process used: 'dual_process' or 'familiarity_only' (Sprint 9.4)
    """
    import os
    from modules.db_pool import get_conn

    if not fts_db_path:
        from modules.config import FTS_DB_PATH as _default_path
        fts_db_path = os.environ.get("FTS_DB_PATH", _default_path)

    try:
        conn = get_conn(fts_db_path)
        _init_fok_calibration_table(conn)
        conn.execute(
            "INSERT INTO fok_calibration_log "
            "(query, fok_predicted, actual_coverage, actual_count, actual_top_activation, "
            " familiarity_score, accessibility_score, fok_process, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (query[:200], fok_predicted, actual_coverage, actual_count,
             actual_top_activation, familiarity_score, accessibility_score,
             fok_process or "single_process", now_iso())
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


def update_fok_dual_alpha(fts_db_path: str = None) -> float:
    """Sprint 9.4: Compute and persist adaptive alpha from calibration history.

    Compare Brier scores for familiarity-only vs accessibility-only predictions.
    Set alpha to minimize combined Brier score.

    Metcalfe et al. 1993: optimal blend adapts to task and memory system.

    Returns:
        New alpha value, or current alpha if insufficient data.
    """
    import os
    from modules.db_pool import get_conn

    if not fts_db_path:
        from modules.config import FTS_DB_PATH as _default_path
        fts_db_path = os.environ.get("FTS_DB_PATH", _default_path)

    try:
        conn = get_conn(fts_db_path)
        rows = conn.execute(
            "SELECT familiarity_score, accessibility_score, actual_top_activation "
            "FROM fok_calibration_log "
            "WHERE familiarity_score IS NOT NULL AND accessibility_score IS NOT NULL "
            "ORDER BY created_at DESC LIMIT 50"
        ).fetchall()

        if len(rows) < 10:
            return _FOK_ALPHA_DEFAULT

        # Compute Brier score for each signal independently
        fam_brier, acc_brier = 0.0, 0.0
        for fam, acc, actual_activation in rows:
            actual = 1.0 if actual_activation >= 0.6 else 0.0
            fam_brier += (float(fam) - actual) ** 2
            acc_brier += (float(acc) - actual) ** 2

        fam_brier /= len(rows)
        acc_brier /= len(rows)

        # Alpha = proportion of weight to give familiarity
        # If familiarity has lower Brier score → higher alpha
        total = fam_brier + acc_brier
        if total > 0:
            # Lower Brier = better → more weight. Inverted: acc_brier/(total) = familiarity alpha
            new_alpha = acc_brier / total  # acc_brier large → acc bad → trust familiarity more
            new_alpha = max(0.2, min(0.8, new_alpha))
        else:
            new_alpha = _FOK_ALPHA_DEFAULT

        # Persist to metamemory_params
        conn.execute("""
            INSERT OR REPLACE INTO metamemory_params (param, value, updated_at)
            VALUES ('fok_dual_alpha', ?, ?)
        """, (str(new_alpha), now_iso()))
        conn.commit()
        return new_alpha

    except Exception:
        return _FOK_ALPHA_DEFAULT


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
        from modules.config import FTS_DB_PATH as _default_path
        fts_db_path = os.environ.get("FTS_DB_PATH", _default_path)

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


def get_fok_resolution(lookback: int = 200, fts_db_path: str = None) -> dict:
    """Compute FOK resolution via Goodman-Kruskal gamma (Nelson 1984).

    Resolution measures whether higher FOK predictions discriminate
    better retrieval outcomes from worse ones. Unlike calibration (MAE/bias),
    resolution captures the RANKING ability of the metacognitive monitor.

    Gamma = (concordant - discordant) / (concordant + discordant)
    - +1.0: perfect discrimination (high FOK always predicts better outcome)
    -  0.0: no discrimination (FOK unrelated to outcome)
    - -1.0: inverse discrimination (high FOK predicts worse outcomes)

    Canon v2 invariant S3-2, claim I-top.

    Returns:
        {gamma, concordant, discordant, tied, n_pairs, n_records}
    """
    import os
    from modules.db_pool import get_conn

    if not fts_db_path:
        from modules.config import FTS_DB_PATH as _default_path
        fts_db_path = os.environ.get("FTS_DB_PATH", _default_path)

    try:
        conn = get_conn(fts_db_path)
        _init_fok_calibration_table(conn)
        cursor = conn.execute(
            "SELECT fok_predicted, actual_coverage, actual_count, actual_top_activation "
            "FROM fok_calibration_log ORDER BY created_at DESC LIMIT ?",
            (lookback,)
        )
        rows = cursor.fetchall()

        if len(rows) < 5:
            return {"gamma": 0.0, "concordant": 0, "discordant": 0,
                    "tied": 0, "n_pairs": 0, "n_records": len(rows)}

        # Compute actual quality scores (same formula as get_fok_calibration)
        pairs = []
        for fok_pred, coverage, count, top_act in rows:
            coverage_score = {"comprehensive": 1.0, "partial": 0.6,
                              "sparse": 0.3, "empty": 0.0}.get(coverage, 0.3)
            count_factor = min(1.0, count / 5.0) if count else 0.0
            actual_quality = coverage_score * 0.5 + count_factor * 0.3 + top_act * 0.2
            pairs.append((fok_pred, actual_quality))

        concordant = 0
        discordant = 0
        tied = 0

        for i in range(len(pairs)):
            for j in range(i + 1, len(pairs)):
                fok_diff = pairs[i][0] - pairs[j][0]
                qual_diff = pairs[i][1] - pairs[j][1]
                product = fok_diff * qual_diff

                if abs(fok_diff) < 1e-9 or abs(qual_diff) < 1e-9:
                    tied += 1
                elif product > 0:
                    concordant += 1
                else:
                    discordant += 1

        denom = concordant + discordant
        gamma = (concordant - discordant) / denom if denom > 0 else 0.0

        return {
            "gamma": round(gamma, 4),
            "concordant": concordant,
            "discordant": discordant,
            "tied": tied,
            "n_pairs": concordant + discordant + tied,
            "n_records": len(rows),
        }
    except Exception:
        return {"gamma": 0.0, "concordant": 0, "discordant": 0,
                "tied": 0, "n_pairs": 0, "n_records": 0}


def calibrated_fok_score(raw_fok: float, calibration: dict,
                         correction_factor: float = 0.5) -> float:
    """Adjust raw FOK score based on historical calibration.

    If historically overconfident (bias > 0), reduce FOK.
    If historically underconfident (bias < 0), increase FOK.

    Nelson & Narens 1990: correction_factor adapts based on persistent bias.
    Managed by detect_self_discrepancies() in self_model.py.

    Args:
        raw_fok: Raw FOK score (0.0-1.0)
        calibration: Output from get_fok_calibration()
        correction_factor: Adaptive factor [0.3-0.9], default 0.5.
            Updated by metamemory control loop.

    Returns:
        Calibrated FOK score (clamped to 0.0-1.0)
    """
    if calibration["n_records"] < 5:
        return raw_fok  # Not enough data to calibrate

    bias = calibration["overconfidence_bias"]
    adjusted = raw_fok - (bias * correction_factor)
    return max(0.0, min(1.0, adjusted))
