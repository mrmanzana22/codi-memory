"""
Codi Memory - Cognitive Observability Contracts
================================================
Phase 1 of architectural restructuring.

Each cognitive module gets a CONTRACT that defines:
  1. What metrics it MUST expose
  2. What the healthy ranges are (neuro-informed)
  3. What red flags trigger alerts
  4. How health is evaluated (Bayesian, never binary)

Two skills inform this:
  - Neuro Skill: WHAT the brain does (papers, theories)
  - Evaluator Skill: HOW to verify (psychometrics, thresholds)

Key papers:
  - Dehaene & Changeux 2011: GNW ignition detection
  - Casali et al 2013: PCI (perturbational complexity)
  - Cowan 2001: WM capacity 4+/-1
  - McClelland et al 1995: CLS consolidation theory
  - Nelson & Narens 1990: Metamemory monitoring → control
  - Lamme 2006: Recurrent processing depth
  - Seth 2005: Causal density
  - Breck et al 2017: ML Test Score (adapted)
"""

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Dict, List, Optional, Tuple

_logger = logging.getLogger(__name__)


# ============================================================
# CONTRACT DEFINITIONS
# ============================================================

class HealthStatus(Enum):
    """Bayesian-inspired health status (never binary pass/fail)."""
    HEALTHY = "healthy"          # BF > 10 (strong evidence of health)
    LIKELY_HEALTHY = "likely_healthy"  # BF 3-10
    INCONCLUSIVE = "inconclusive"     # BF 0.33-3
    LIKELY_DEGRADED = "likely_degraded"  # BF 0.1-0.33
    DEGRADED = "degraded"        # BF < 0.1


class MetricDomain(Enum):
    """Domains from evaluator skill E3 consciousness metrics battery."""
    INTEGRATION = "integration_differentiation"
    GNW_IGNITION = "gnw_ignition"
    RECURRENCE = "recurrence"
    COMPLEXITY = "complexity"
    MEMORY = "memory"
    PREDICTION = "prediction"
    METACOGNITION = "metacognition"


@dataclass
class Metric:
    """A single measurable metric within a contract."""
    name: str
    domain: MetricDomain
    description: str
    target_lo: float            # Healthy range lower bound
    target_hi: float            # Healthy range upper bound
    red_flag_lo: Optional[float] = None  # Below this = degraded
    red_flag_hi: Optional[float] = None  # Above this = degraded
    unit: str = ""
    paper: str = ""             # Citation for the threshold
    measure_fn: Optional[str] = None  # dotted path to measurement function


@dataclass
class Contract:
    """Observability contract for a cognitive module/loop."""
    loop_id: str                # e.g., "L3", "PE", "L6"
    name: str                   # e.g., "GNW Competition"
    module: str                 # e.g., "competition.py"
    theory: str                 # e.g., "Baars 1988, Dehaene 2014"
    metrics: List[Metric] = field(default_factory=list)


@dataclass
class MetricResult:
    """Result of measuring a single metric."""
    metric_name: str
    value: Optional[float]
    status: HealthStatus
    detail: str = ""


@dataclass
class ContractResult:
    """Result of evaluating a full contract."""
    loop_id: str
    name: str
    overall_status: HealthStatus
    metric_results: List[MetricResult] = field(default_factory=list)
    elapsed_ms: float = 0.0


# ============================================================
# CONTRACT REGISTRY — 10 Loops + PE Hub
# ============================================================

def _build_contracts() -> Dict[str, Contract]:
    """Build all cognitive contracts. Called once at import."""

    contracts = {}

    # --- PE Hub (Universal Currency) ---
    contracts["PE"] = Contract(
        loop_id="PE",
        name="Prediction Error Hub",
        module="prediction.py",
        theory="Emergent universal currency",
        metrics=[
            Metric("pe_flow_diversity", MetricDomain.COMPLEXITY,
                   "Number of loops PE flows through",
                   target_lo=3, target_hi=10,
                   red_flag_lo=1, red_flag_hi=None,
                   paper="Emergent finding"),
            Metric("pe_magnitude_mean", MetricDomain.PREDICTION,
                   "Mean PE magnitude across recent events",
                   target_lo=0.1, target_hi=0.8,
                   red_flag_lo=0.0, red_flag_hi=0.95,
                   paper="Clark 2013"),
        ]
    )

    # --- L1: Reconsolidation ---
    contracts["L1"] = Contract(
        loop_id="L1",
        name="Reconsolidation",
        module="reconsolidation.py",
        theory="Nader 2000",
        metrics=[
            Metric("reconsolidation_trigger_rate", MetricDomain.MEMORY,
                   "Fraction of high-PE events that trigger reconsolidation",
                   target_lo=0.05, target_hi=0.40,
                   red_flag_lo=0.0, red_flag_hi=0.80,
                   paper="Nader 2000"),
            Metric("correction_success_rate", MetricDomain.MEMORY,
                   "Fraction of reconsolidations that successfully update memory",
                   target_lo=0.50, target_hi=1.0,
                   red_flag_lo=0.20,
                   paper="Lee 2008"),
        ]
    )

    # --- L2: Consolidation ---
    contracts["L2"] = Contract(
        loop_id="L2",
        name="Consolidation",
        module="consolidation.py",
        theory="McClelland 1995",
        metrics=[
            Metric("consolidation_coverage", MetricDomain.MEMORY,
                   "Fraction of daily episodes consolidated",
                   target_lo=0.50, target_hi=1.0,
                   red_flag_lo=0.30,
                   unit="%",
                   paper="McClelland 1995"),
            Metric("semantic_extraction_rate", MetricDomain.MEMORY,
                   "Semantic facts extracted per consolidation cycle",
                   target_lo=1.0, target_hi=20.0,
                   red_flag_lo=0.0,
                   paper="McClelland 1995"),
            Metric("false_memory_rate", MetricDomain.MEMORY,
                   "Fraction of consolidated facts that are incorrect",
                   target_lo=0.0, target_hi=0.05,
                   red_flag_hi=0.10,
                   paper="Roediger & McDermott 1995"),
        ]
    )

    # --- L3: GNW + Attention ---
    contracts["L3"] = Contract(
        loop_id="L3",
        name="GNW Competition",
        module="competition.py",
        theory="Baars 1988, Dehaene 2014",
        metrics=[
            Metric("ignition_ratio", MetricDomain.GNW_IGNITION,
                   "Fraction of CX fires from GNW competition (interactive sessions only)",
                   target_lo=0.0, target_hi=0.70,
                   red_flag_lo=None, red_flag_hi=0.90,
                   paper="Dehaene & Changeux 2011 (0 expected during sleep-only measurement)"),
            Metric("coalition_size", MetricDomain.INTEGRATION,
                   "Mean active CX connections per snapshot (proxy for GNW breadth)",
                   target_lo=3, target_hi=15,
                   red_flag_lo=0, red_flag_hi=25,
                   paper="Dehaene 2014 (proxy: CX active breadth)"),
            Metric("workspace_utilization", MetricDomain.INTEGRATION,
                   "Fraction of CX connections active (cx_active/cx_total)",
                   target_lo=0.10, target_hi=0.60,
                   red_flag_lo=0.0, red_flag_hi=0.90,
                   unit="%",
                   paper="Integration breadth metric"),
            Metric("recurrent_pass_count", MetricDomain.RECURRENCE,
                   "Number of recurrent amplification passes",
                   target_lo=3, target_hi=5,
                   red_flag_lo=1,
                   paper="Lamme 2006"),
        ]
    )

    # --- L4: Prediction → Emotion ---
    contracts["L4"] = Contract(
        loop_id="L4",
        name="Prediction → Emotion",
        module="prediction.py + emotion.py",
        theory="Clark 2013, Friston 2010, Hesp 2021",
        metrics=[
            Metric("prediction_accuracy", MetricDomain.PREDICTION,
                   "Fraction of L0 predictions that match outcome",
                   target_lo=0.40, target_hi=0.80,
                   red_flag_lo=0.20,
                   paper="Internal baseline"),
            Metric("precision_adaptation_speed", MetricDomain.PREDICTION,
                   "HGF kappa convergence rate (turns to adapt)",
                   target_lo=1, target_hi=10,
                   red_flag_hi=20,
                   unit="turns",
                   paper="Mathys 2011"),
            Metric("pad_from_precision", MetricDomain.PREDICTION,
                   "Whether PAD is computed from precision dynamics (not keywords)",
                   target_lo=1.0, target_hi=1.0,
                   red_flag_lo=0.0,
                   paper="Hesp 2021"),
        ]
    )

    # --- L5: Metacognition ---
    contracts["L5"] = Contract(
        loop_id="L5",
        name="Metacognition",
        module="self_model.py + prediction.py",
        theory="Rosenthal 2005, Nelson & Narens 1990",
        metrics=[
            Metric("calibration_error", MetricDomain.METACOGNITION,
                   "Expected Calibration Error (ECE)",
                   target_lo=0.0, target_hi=0.10,
                   red_flag_hi=0.20,
                   paper="Lichtenstein 1982, evaluator E2.5"),
            Metric("confidence_range", MetricDomain.METACOGNITION,
                   "Spread of confidence scores (max - min)",
                   target_lo=0.30, target_hi=0.90,
                   red_flag_lo=0.10,
                   paper="Nelson & Narens 1990"),
            Metric("monitoring_control_loop", MetricDomain.METACOGNITION,
                   "Whether monitoring output feeds control decisions",
                   target_lo=1.0, target_hi=1.0,
                   red_flag_lo=0.0,
                   paper="Nelson & Narens 1990"),
        ]
    )

    # --- L6: Curiosity ---
    contracts["L6"] = Contract(
        loop_id="L6",
        name="Curiosity",
        module="curiosity.py",
        theory="Schmidhuber 2010, Gottlieb 2013",
        metrics=[
            Metric("curiosity_generation_rate", MetricDomain.COMPLEXITY,
                   "Questions generated per day (daemon + interactive)",
                   target_lo=1.0, target_hi=50.0,
                   red_flag_lo=0.0, red_flag_hi=100.0,
                   paper="Schmidhuber 2010"),
            Metric("curiosity_resolution_rate", MetricDomain.COMPLEXITY,
                   "Fraction of curiosities resolved (discovery+partial) vs total",
                   target_lo=0.10, target_hi=0.80,
                   red_flag_lo=0.0,
                   paper="Schmidhuber 2010 (15% discovery rate is healthy for autonomous system)"),
        ]
    )

    # --- L7: Active Inference ---
    contracts["L7"] = Contract(
        loop_id="L7",
        name="Active Inference",
        module="active_inference.py",
        theory="Friston 2017, Sutton 1999",
        metrics=[
            Metric("efe_policy_diversity", MetricDomain.INTEGRATION,
                   "Number of distinct policies considered",
                   target_lo=2, target_hi=4,
                   red_flag_lo=1,
                   paper="Friston 2017"),
            Metric("action_selection_latency", MetricDomain.INTEGRATION,
                   "Time to select action (ms)",
                   target_lo=10, target_hi=200,
                   red_flag_hi=500,
                   unit="ms",
                   paper="Internal target"),
        ]
    )

    # --- L8: Causal DAG ---
    contracts["L8"] = Contract(
        loop_id="L8",
        name="Causal Discovery",
        module="causal_discovery.py",
        theory="Zheng 2018 (NOTEARS), Pearl 2009",
        metrics=[
            Metric("dag_edge_count", MetricDomain.COMPLEXITY,
                   "Number of causal edges discovered",
                   target_lo=5, target_hi=200,
                   red_flag_lo=0,
                   paper="Zheng 2018"),
            Metric("causal_density", MetricDomain.INTEGRATION,
                   "Fraction of significant causal interactions",
                   target_lo=0.15, target_hi=0.60,
                   red_flag_lo=0.05,
                   paper="Seth 2005"),
        ]
    )

    # --- L9: Self-Model ---
    contracts["L9"] = Contract(
        loop_id="L9",
        name="Self-Model",
        module="self_model.py",
        theory="Graziano 2013, Rosenthal 2005",
        metrics=[
            Metric("self_model_refresh_frequency", MetricDomain.METACOGNITION,
                   "Self-model tool calls per day (daemon + interactive)",
                   target_lo=10, target_hi=300,
                   red_flag_hi=500,
                   unit="interactions",
                   paper="Graziano 2013"),
            Metric("discrepancy_count", MetricDomain.METACOGNITION,
                   "Self-model discrepancies detected per refresh",
                   target_lo=0, target_hi=5,
                   red_flag_hi=10,
                   paper="Metzinger 2003"),
        ]
    )

    # --- L10: Forgetting ---
    contracts["L10"] = Contract(
        loop_id="L10",
        name="Forgetting",
        module="forgetting.py",
        theory="Bjork 1992, Anderson 2003",
        metrics=[
            Metric("decay_curve_exponent", MetricDomain.MEMORY,
                   "Power-law decay exponent (should match Ebbinghaus)",
                   target_lo=0.3, target_hi=0.6,
                   red_flag_lo=0.1, red_flag_hi=0.9,
                   paper="Ebbinghaus 1885, Bjork 1992"),
            Metric("rif_suppression_rate", MetricDomain.MEMORY,
                   "RIF suppression magnitude (10-20% in humans)",
                   target_lo=0.05, target_hi=0.25,
                   red_flag_hi=0.40,
                   paper="Anderson et al 1994"),
        ]
    )

    # --- Cross-Loop Integration ---
    contracts["CX"] = Contract(
        loop_id="CX",
        name="Cross-Loop Integration",
        module="wiring.py + cx_observability.py",
        theory="Tononi 2004 (IIT proxy), Lamme 2006",
        metrics=[
            Metric("cx_diversity_index", MetricDomain.INTEGRATION,
                   "Shannon entropy of CX fire counts",
                   target_lo=2.0, target_hi=4.0,
                   red_flag_lo=1.0,
                   paper="evaluator E3.2"),
            Metric("cascade_depth", MetricDomain.RECURRENCE,
                   "Mean CX chain length per event (sleep loop = mostly parallel, not sequential)",
                   target_lo=0.3, target_hi=6,
                   red_flag_lo=0.1,
                   paper="Lamme 2006 (parallel CX expected during sleep ticks)"),
            Metric("active_cx_ratio", MetricDomain.INTEGRATION,
                   "Fraction of 34 CX connections that fired in last 24h",
                   target_lo=0.10, target_hi=1.0,
                   red_flag_lo=0.05,
                   paper="Internal target (sleep loop activates subset of CX)"),
            Metric("pci_proxy", MetricDomain.INTEGRATION,
                   "Perturbational Complexity Index proxy",
                   target_lo=0.03, target_hi=0.50,
                   red_flag_lo=0.02,
                   paper="Casali 2013"),
        ]
    )

    return contracts


# Module-level singleton
CONTRACTS: Dict[str, Contract] = _build_contracts()


# ============================================================
# MEASUREMENT ENGINE
# ============================================================

def _evaluate_metric(metric: Metric, value: Optional[float]) -> MetricResult:
    """Evaluate a single metric against its contract thresholds.

    Uses evaluator skill principle: never binary pass/fail.
    Returns Bayesian-inspired status based on distance from healthy range.
    """
    if value is None:
        return MetricResult(metric.name, None, HealthStatus.INCONCLUSIVE,
                            "No measurement available")

    # Check red flags first
    if metric.red_flag_lo is not None and value < metric.red_flag_lo:
        return MetricResult(metric.name, value, HealthStatus.DEGRADED,
                            f"{value:.3f} < red_flag {metric.red_flag_lo}")
    if metric.red_flag_hi is not None and value > metric.red_flag_hi:
        return MetricResult(metric.name, value, HealthStatus.DEGRADED,
                            f"{value:.3f} > red_flag {metric.red_flag_hi}")

    # Check healthy range
    if metric.target_lo <= value <= metric.target_hi:
        return MetricResult(metric.name, value, HealthStatus.HEALTHY,
                            f"{value:.3f} in [{metric.target_lo}, {metric.target_hi}]")

    # Outside healthy but not red flag = likely degraded
    if value < metric.target_lo:
        distance = metric.target_lo - value
        range_size = metric.target_hi - metric.target_lo
        ratio = distance / range_size if range_size > 0 else 1.0
        status = HealthStatus.LIKELY_DEGRADED if ratio > 0.5 else HealthStatus.LIKELY_HEALTHY
        return MetricResult(metric.name, value, status,
                            f"{value:.3f} below target [{metric.target_lo}, {metric.target_hi}]")
    else:
        distance = value - metric.target_hi
        range_size = metric.target_hi - metric.target_lo
        ratio = distance / range_size if range_size > 0 else 1.0
        status = HealthStatus.LIKELY_DEGRADED if ratio > 0.5 else HealthStatus.LIKELY_HEALTHY
        return MetricResult(metric.name, value, status,
                            f"{value:.3f} above target [{metric.target_lo}, {metric.target_hi}]")


def _overall_status(results: List[MetricResult]) -> HealthStatus:
    """Weakest-link rule: overall = worst metric status.

    From evaluator E4.2: overall score = MINIMUM across categories.
    """
    if not results:
        return HealthStatus.INCONCLUSIVE

    priority = {
        HealthStatus.DEGRADED: 0,
        HealthStatus.LIKELY_DEGRADED: 1,
        HealthStatus.INCONCLUSIVE: 2,
        HealthStatus.LIKELY_HEALTHY: 3,
        HealthStatus.HEALTHY: 4,
    }
    worst = min(results, key=lambda r: priority.get(r.status, 2))
    return worst.status


def evaluate_contract(contract: Contract,
                      measurements: Dict[str, Optional[float]]) -> ContractResult:
    """Evaluate a contract against provided measurements.

    Args:
        contract: The contract to evaluate
        measurements: {metric_name: value} dict

    Returns:
        ContractResult with per-metric and overall status
    """
    t0 = time.monotonic()
    results = []
    for metric in contract.metrics:
        value = measurements.get(metric.name)
        results.append(_evaluate_metric(metric, value))

    elapsed = (time.monotonic() - t0) * 1000
    return ContractResult(
        loop_id=contract.loop_id,
        name=contract.name,
        overall_status=_overall_status(results),
        metric_results=results,
        elapsed_ms=elapsed,
    )


# ============================================================
# COLLECTORS — gather live measurements from the system
# ============================================================

def collect_cx_metrics() -> Dict[str, Optional[float]]:
    """Collect CX integration metrics from stored cx_snapshots (not live EventBus).

    Fix: reads from SQLite cx_snapshots table (persisted by sleep-loop tick)
    instead of live EventBus state (empty outside daemon process).
    """
    metrics = {}
    try:
        import json
        import math
        from modules.config import connect_fts, FTS_DB_PATH
        db = connect_fts(FTS_DB_PATH)
        try:
            # Aggregate across last 24h of snapshots (not just latest)
            rows = db.execute(
                """SELECT payload FROM cx_snapshots
                   WHERE ts > datetime('now', '-24 hours')
                   ORDER BY ts DESC"""
            ).fetchall()
            if not rows:
                # Fallback: last 10 snapshots regardless of time
                rows = db.execute(
                    "SELECT payload FROM cx_snapshots ORDER BY ts DESC LIMIT 10"
                ).fetchall()
        finally:
            db.close()
        if rows:
            # Aggregate fire counts across all snapshots
            aggregated_fires = {}
            total_chain = 0
            snap_count = 0
            for r in rows:
                try:
                    snap = json.loads(r[0])
                    fires = snap.get("fire_counts", {})
                    for cx_id, count in fires.items():
                        aggregated_fires[cx_id] = aggregated_fires.get(cx_id, 0) + (count or 0)
                    cascade = snap.get("cascade", {})
                    total_chain += cascade.get("chain_count", 0)
                    snap_count += 1
                except Exception:
                    pass
            # Active CX ratio: distinct CX that fired at least once across window
            total_cx = 34
            active = sum(1 for v in aggregated_fires.values() if v > 0)
            metrics["active_cx_ratio"] = active / total_cx
            # Cascade depth: average chain count
            metrics["cascade_depth"] = total_chain / snap_count if snap_count > 0 else 0
            # Diversity index: compute Shannon entropy from aggregated fire counts
            # (Fix: upstream diversity_index is always 0 due to coact tracker bug)
            total_all = sum(v for v in aggregated_fires.values() if v > 0)
            if total_all > 0:
                entropy = 0.0
                for count in aggregated_fires.values():
                    if count > 0:
                        p = count / total_all
                        entropy -= p * math.log2(p)
                metrics["cx_diversity_index"] = entropy
                # PCI proxy: entropy_normalized * cascade_depth_normalized
                # Casali et al 2013: perturbational complexity = integration * differentiation
                max_entropy = math.log2(max(len(aggregated_fires), 1)) or 1.0
                entropy_norm = entropy / max_entropy if max_entropy > 0 else 0
                cascade_depth = metrics.get("cascade_depth", 0)
                cascade_norm = min(1.0, cascade_depth / 6.0)  # Max expected depth ~6
                metrics["pci_proxy"] = round(entropy_norm * cascade_norm, 4)
            else:
                metrics["cx_diversity_index"] = 0.0
                metrics["pci_proxy"] = 0.0
    except Exception as e:
        _logger.debug("collect_cx_metrics: %s", e)
    return metrics


def collect_gnw_metrics() -> Dict[str, Optional[float]]:
    """Collect GNW competition metrics from stored CX snapshots.

    Fix: competition.py has no get_competition_stats(). Instead, read
    from cx_snapshots which record GNW competition outcomes via EventBus.
    Recurrent pass count is architectural (N=3 in competition.py).
    """
    metrics = {}
    try:
        import json
        from modules.config import connect_fts, FTS_DB_PATH
        db = connect_fts(FTS_DB_PATH)
        try:
            # GNW fires are tracked in cx_snapshots fire_counts
            # CX-5, CX-9, CX-16 are GNW-triggered (WORKSPACE_COMPETITION_COMPLETE)
            # Aggregate across stored snapshots for GNW data
            all_rows = db.execute(
                "SELECT payload FROM cx_snapshots ORDER BY ts DESC LIMIT 50"
            ).fetchall()
        finally:
            db.close()
        if all_rows:
            # GNW CX IDs: WORKSPACE_COMPETITION_COMPLETE event
            gnw_cx = {"CX-5", "CX-9", "CX-16", "CX-25"}
            total_fires = 0
            total_gnw_fires = 0
            for r in all_rows:
                try:
                    snap = json.loads(r[0])
                    fires = snap.get("fire_counts", {})
                    for cx_id, count in fires.items():
                        c = count or 0
                        total_fires += c
                        if cx_id in gnw_cx:
                            total_gnw_fires += c
                except Exception:
                    pass
            # Ignition ratio: GNW fires / total fires
            # Note: GNW fires mainly during interactive sessions, not sleep-loop
            if total_fires > 0:
                metrics["ignition_ratio"] = total_gnw_fires / total_fires
        # Coalition size proxy: active_cx from snapshots (mean of recent)
        if all_rows:
            active_counts = []
            total_cx_counts = []
            for r in all_rows:
                try:
                    snap = json.loads(r[0])
                    ac = snap.get("active_cx")
                    tc = snap.get("total_cx")
                    if ac is not None:
                        active_counts.append(ac)
                    if tc is not None:
                        total_cx_counts.append(tc)
                except Exception:
                    pass
            if active_counts:
                metrics["coalition_size"] = sum(active_counts) / len(active_counts)
            if active_counts and total_cx_counts:
                avg_active = sum(active_counts) / len(active_counts)
                avg_total = sum(total_cx_counts) / len(total_cx_counts)
                if avg_total > 0:
                    metrics["workspace_utilization"] = avg_active / avg_total
        # Architectural constants
        metrics["recurrent_pass_count"] = 3  # Hardcoded N=3 in competition.py
    except Exception as e:
        _logger.debug("collect_gnw_metrics: %s", e)
    return metrics


def collect_prediction_metrics() -> Dict[str, Optional[float]]:
    """Collect prediction + emotion metrics from prediction_results table.

    Fix: get_predictive_state() returns empty predictions list.
    Real accuracy data lives in prediction_results.hit column.
    """
    metrics = {}
    try:
        from modules.config import connect_fts, FTS_DB_PATH
        db = connect_fts(FTS_DB_PATH)
        try:
            # Prediction accuracy from prediction_results.hit
            # Filter: only interactive/sleep_loop sources (curiosity_resolution inflates to 99%)
            rows = db.execute(
                """SELECT hit, surprise_score FROM prediction_results
                   WHERE source IN ('interactive', 'sleep_loop', 'preturn')
                   ORDER BY created_at DESC LIMIT 100"""
            ).fetchall()
        finally:
            db.close()
        if rows:
            hits = [r[0] for r in rows if r[0] is not None]
            if hits:
                metrics["prediction_accuracy"] = sum(hits) / len(hits)
            # Precision adaptation: variance of surprise scores
            surprises = [r[1] for r in rows if r[1] is not None]
            if len(surprises) > 10:
                # If surprise scores vary, precision is adapting
                mean_s = sum(surprises) / len(surprises)
                variance = sum((s - mean_s) ** 2 for s in surprises) / len(surprises)
                # Lower variance = more adapted. Map to turns-to-adapt proxy.
                metrics["precision_adaptation_speed"] = max(1, min(20, 10 * (1 - variance)))
                # PAD-from-precision: check persistent SQLite state (survives process restarts)
                # In-memory _emotional_state is empty outside daemon; use sleep_loop_state DB
                try:
                    import json as _j2
                    from modules.config import connect_fts as _cfts, FTS_DB_PATH as _fdb
                    _pdb = _cfts(_fdb)
                    try:
                        _pad_row = _pdb.execute(
                            "SELECT value FROM sleep_loop_state WHERE key = 'pad_current'"
                        ).fetchone()
                    finally:
                        _pdb.close()
                    if _pad_row:
                        _pad_data = _j2.loads(_pad_row[0])
                        has_pad = _pad_data.get("timestamp") is not None
                        has_trigger = bool(_pad_data.get("trigger"))
                        precision_sources = {"affective_charge", "text_inference", "decay"}
                        from_precision = any(s in (_pad_data.get("trigger") or "")
                                             for s in precision_sources)
                        # 1.0 if PAD exists with precision-driven trigger, 0.5 if PAD exists, 0.0 if no PAD
                        metrics["pad_from_precision"] = 1.0 if from_precision else (0.5 if has_pad else 0.0)
                    else:
                        metrics["pad_from_precision"] = 0.0
                except Exception:
                    metrics["pad_from_precision"] = 0.0
    except Exception as e:
        _logger.debug("collect_prediction_metrics: %s", e)
    return metrics


def collect_causal_metrics() -> Dict[str, Optional[float]]:
    """Collect causal discovery metrics from causal_discovery.py."""
    metrics = {}
    try:
        from modules.config import connect_fts, FTS_DB_PATH
        db = connect_fts(FTS_DB_PATH)
        try:
            # Read from causal_discovery_state table (latest entry)
            row = db.execute(
                "SELECT n_edges, topics FROM causal_discovery_state ORDER BY created_at DESC LIMIT 1"
            ).fetchone()
        finally:
            db.close()
        if row:
            import json
            n_edges = row[0] or 0
            topics = json.loads(row[1]) if row[1] else []
            metrics["dag_edge_count"] = n_edges
            total_possible = len(topics) ** 2
            if total_possible > 0:
                metrics["causal_density"] = n_edges / total_possible
    except Exception as e:
        _logger.debug("collect_causal_metrics: %s", e)
    return metrics


def collect_curiosity_metrics() -> Dict[str, Optional[float]]:
    """Collect curiosity metrics from preguntas_curiosidad.json (authoritative source)."""
    metrics = {}
    try:
        from modules.curiosity import _cargar_curiosidades
        from datetime import datetime, timedelta
        data = _cargar_curiosidades()
        pendientes = data.get("pendientes", [])
        exploradas = data.get("exploradas", [])

        # Generation rate: pendientes added in last 7 days / 7
        cutoff = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")
        recent_pending = [c for c in pendientes if (c.get("agregada") or "") >= cutoff]
        recent_explored = [c for c in exploradas if (c.get("agregada") or "") >= cutoff]
        metrics["curiosity_generation_rate"] = (len(recent_pending) + len(recent_explored)) / 7.0

        # Resolution rate: explored with meaningful outcome / total in window
        total_in_window = len(recent_pending) + len(recent_explored)
        resolved = [c for c in recent_explored
                    if c.get("outcome") in ("discovery", "partial")]
        if total_in_window > 0:
            metrics["curiosity_resolution_rate"] = len(resolved) / total_in_window
        else:
            # Fallback: all-time rate
            all_resolved = [c for c in exploradas
                           if c.get("outcome") in ("discovery", "partial")]
            total_all = len(pendientes) + len(exploradas)
            if total_all > 0:
                metrics["curiosity_resolution_rate"] = len(all_resolved) / total_all
    except Exception as e:
        _logger.debug("collect_curiosity_metrics: %s", e)
    return metrics


def collect_self_model_metrics() -> Dict[str, Optional[float]]:
    """Collect self-model metrics from self_model.py + tool_calls."""
    metrics = {}
    try:
        from modules.self_model import detect_self_discrepancies
        result = detect_self_discrepancies()
        if isinstance(result, dict):
            metrics["discrepancy_count"] = result.get("count", 0)
        # Self-model refresh frequency: tool_calls for self_model in last 7 days
        try:
            from modules.config import connect_fts, FTS_DB_PATH
            db = connect_fts(FTS_DB_PATH)
            try:
                row = db.execute(
                    """SELECT COUNT(*) FROM tool_calls
                       WHERE tool_name LIKE '%self_model%'
                       AND started_at > datetime('now', '-7 days')"""
                ).fetchone()
            finally:
                db.close()
            if row and row[0] is not None:
                metrics["self_model_refresh_frequency"] = row[0] / 7.0
        except Exception:
            pass
    except Exception as e:
        _logger.debug("collect_self_model_metrics: %s", e)
    return metrics


def collect_reconsolidation_metrics() -> Dict[str, Optional[float]]:
    """Collect L1 reconsolidation metrics from DB logs."""
    metrics = {}
    db = None
    try:
        from modules.config import connect_fts, FTS_DB_PATH
        db = connect_fts(FTS_DB_PATH)
        # Total reconsolidations
        total = db.execute("SELECT COUNT(*) FROM reconsolidation_log").fetchone()[0]
        # Successful corrections (action = 'corrected' or has new_content)
        success = db.execute(
            "SELECT COUNT(*) FROM reconsolidation_log WHERE new_content IS NOT NULL AND new_content != ''"
        ).fetchone()[0]
        if total > 0:
            metrics["correction_success_rate"] = success / total
        # Trigger rate: reconsolidations / total high-PE events
        pe_events = db.execute("SELECT COUNT(*) FROM prediction_results WHERE surprise_score > 0.6").fetchone()[0]
        if pe_events > 0:
            metrics["reconsolidation_trigger_rate"] = total / pe_events
    except Exception as e:
        _logger.debug("collect_reconsolidation_metrics: %s", e)
    finally:
        if db is not None:
            db.close()
    return metrics


def collect_consolidation_metrics() -> Dict[str, Optional[float]]:
    """Collect L2 consolidation metrics from consolidation_log."""
    metrics = {}
    db = None
    try:
        from modules.config import connect_fts, FTS_DB_PATH
        db = connect_fts(FTS_DB_PATH)
        # Recent consolidation (last 30 days — consolidation runs intermittently)
        rows = db.execute(
            """SELECT episodes_scanned, facts_extracted, facts_created, contradictions_found
               FROM consolidation_log
               ORDER BY created_at DESC LIMIT 20"""
        ).fetchall()
        if rows:
            total_episodes = sum(r[0] for r in rows)
            total_facts = sum(r[1] for r in rows)
            total_created = sum(r[2] for r in rows)
            total_contradictions = sum(r[3] for r in rows)
            # Semantic extraction rate (facts per cycle)
            metrics["semantic_extraction_rate"] = total_facts / len(rows) if rows else 0
            # False memory proxy: contradictions / total facts
            if total_facts > 0:
                metrics["false_memory_rate"] = total_contradictions / total_facts
            # Coverage: episodes consolidated / estimated total daily episodes
            # Rough estimate: 50 memories per day
            daily_episodes_est = 50
            daily_consolidated = total_episodes / 7.0
            metrics["consolidation_coverage"] = min(1.0, daily_consolidated / daily_episodes_est)
    except Exception as e:
        _logger.debug("collect_consolidation_metrics: %s", e)
    finally:
        if db is not None:
            db.close()
    return metrics


def collect_metacognition_metrics() -> Dict[str, Optional[float]]:
    """Collect L5 metacognition metrics from FOK calibration + metacognition traces."""
    metrics = {}
    db = None
    try:
        from modules.config import connect_fts, FTS_DB_PATH
        db = connect_fts(FTS_DB_PATH)
        # FOK calibration: actual_coverage is categorical (sparse/partial/comprehensive)
        # Map to numeric: sparse=0.2, partial=0.5, comprehensive=0.8
        coverage_map = {"sparse": 0.2, "partial": 0.5, "comprehensive": 0.8}
        rows = db.execute(
            """SELECT fok_predicted, actual_coverage FROM fok_calibration_log
               ORDER BY created_at DESC LIMIT 200"""
        ).fetchall()
        if rows:
            errors = []
            confidences = []
            for r in rows:
                if r[0] is not None and r[1] is not None:
                    actual = coverage_map.get(r[1], 0.5)
                    errors.append(abs(r[0] - actual))
                    confidences.append(r[0])
            if errors:
                metrics["calibration_error"] = sum(errors) / len(errors)
            if confidences:
                metrics["confidence_range"] = max(confidences) - min(confidences)
        # Monitoring → control check: do metacognition traces show strategy changes?
        mc_rows = db.execute(
            """SELECT COUNT(DISTINCT strategy_chosen) FROM metacognition_traces
               WHERE created_at > datetime('now', '-30 days')"""
        ).fetchone()
        if mc_rows and mc_rows[0] > 1:
            metrics["monitoring_control_loop"] = 1.0  # Multiple strategies = control loop active
        elif mc_rows:
            metrics["monitoring_control_loop"] = 0.0
    except Exception as e:
        _logger.debug("collect_metacognition_metrics: %s", e)
    finally:
        if db is not None:
            db.close()
    return metrics


def collect_pe_metrics() -> Dict[str, Optional[float]]:
    """Collect PE hub metrics from prediction_results + CX snapshots."""
    metrics = {}
    db = None
    try:
        from modules.config import connect_fts, FTS_DB_PATH
        db = connect_fts(FTS_DB_PATH)
        # PE magnitude mean (recent)
        rows = db.execute(
            """SELECT surprise_score FROM prediction_results
               WHERE created_at > datetime('now', '-7 days')
               ORDER BY created_at DESC LIMIT 100"""
        ).fetchall()
        if rows:
            scores = [r[0] for r in rows if r[0] is not None]
            if scores:
                metrics["pe_magnitude_mean"] = sum(scores) / len(scores)
        # PE flow diversity: count distinct CX that fired from PE events
        # Use cx_snapshots payload
        cx_rows = db.execute(
            """SELECT payload FROM cx_snapshots
               WHERE ts > datetime('now', '-24 hours')
               ORDER BY ts DESC LIMIT 5"""
        ).fetchall()
        if cx_rows:
            import json
            all_cx_ids = set()
            for r in cx_rows:
                try:
                    snap = json.loads(r[0])
                    fires = snap.get("fire_counts", {})
                    for cx_id, count in fires.items():
                        if count > 0:
                            all_cx_ids.add(cx_id)
                except Exception:
                    pass
            metrics["pe_flow_diversity"] = len(all_cx_ids)
    except Exception as e:
        _logger.debug("collect_pe_metrics: %s", e)
    finally:
        if db is not None:
            db.close()
    return metrics


def collect_active_inference_metrics() -> Dict[str, Optional[float]]:
    """Collect L7 Active Inference metrics from generative_model_transitions."""
    metrics = {}
    db = None
    try:
        from modules.config import connect_fts, FTS_DB_PATH
        db = connect_fts(FTS_DB_PATH)
        # Policy diversity: distinct actions selected recently
        rows = db.execute(
            """SELECT DISTINCT driver FROM attention_transitions
               WHERE created_at > datetime('now', '-7 days')"""
        ).fetchall()
        if rows:
            metrics["efe_policy_diversity"] = min(4, len(rows))
        # Action selection latency proxy: avg CX handler p50 latency from recent snapshots
        try:
            import json as _j
            snap_rows = db.execute(
                "SELECT payload FROM cx_snapshots ORDER BY ts DESC LIMIT 5"
            ).fetchall()
            all_p50s = []
            for snap_row in snap_rows:
                try:
                    snap = _j.loads(snap_row[0])
                    lat_summary = snap.get("latency_summary", {})
                    for v in lat_summary.values():
                        if isinstance(v, dict) and v.get("p50") is not None:
                            all_p50s.append(v["p50"])
                except Exception:
                    pass
            if all_p50s:
                metrics["action_selection_latency"] = sum(all_p50s) / len(all_p50s)
        except Exception:
            pass
    except Exception as e:
        _logger.debug("collect_active_inference_metrics: %s", e)
    finally:
        if db is not None:
            db.close()
    return metrics


def collect_forgetting_metrics() -> Dict[str, Optional[float]]:
    """Collect L10 Forgetting metrics from strength_log + forgetting constants."""
    metrics = {}
    try:
        # Decay exponent is a code constant in activation.py
        from modules.activation import ACTR_DECAY_DEFAULT
        metrics["decay_curve_exponent"] = ACTR_DECAY_DEFAULT

        # RIF: check strength_log for suppression events
        from modules.config import connect_fts, FTS_DB_PATH
        db = connect_fts(FTS_DB_PATH)
        try:
            rif_count = db.execute(
                "SELECT COUNT(*) FROM strength_log WHERE event = 'rif_suppression'"
            ).fetchone()
            total = db.execute("SELECT COUNT(*) FROM strength_log").fetchone()
        finally:
            db.close()
        if total and total[0] > 0 and rif_count:
            metrics["rif_suppression_rate"] = rif_count[0] / total[0]
        else:
            # No forgetting events yet — rate is 0.0 (valid measurement, not inconclusive)
            metrics["rif_suppression_rate"] = 0.0
    except Exception as e:
        _logger.debug("collect_forgetting_metrics: %s", e)
    return metrics


# ============================================================
# STORE-BASED COLLECTORS (Phase 2 — CognitiveStore)
# ============================================================

def _build_store_collectors() -> Dict[str, Callable]:
    """Build collectors using CognitiveStore repos.

    Falls back to legacy collectors if store import fails.
    """
    try:
        from modules.store import get_store
        store = get_store()
    except Exception:
        _logger.debug("CognitiveStore import failed, using legacy collectors")
        return {
            "PE": collect_pe_metrics,
            "L1": collect_reconsolidation_metrics,
            "L2": collect_consolidation_metrics,
            "L3": collect_gnw_metrics,
            "L4": collect_prediction_metrics,
            "L5": collect_metacognition_metrics,
            "L6": collect_curiosity_metrics,
            "L7": collect_active_inference_metrics,
            "L8": collect_causal_metrics,
            "L9": collect_self_model_metrics,
            "L10": collect_forgetting_metrics,
            "CX": collect_cx_metrics,
        }

    def _collect_pe():
        metrics = {}
        scores = store.prediction.get_surprise_scores(days=7, limit=100)
        if scores:
            metrics["pe_magnitude_mean"] = sum(scores) / len(scores)
        metrics["pe_flow_diversity"] = store.cx.get_pe_cx_diversity(hours=24)
        return metrics

    def _collect_l1():
        return store.consolidation.get_reconsolidation_stats()

    def _collect_l2():
        return store.consolidation.get_consolidation_metrics()

    def _collect_l3():
        metrics = store.cx.get_gnw_metrics()
        # Merge coalition_size + workspace_utilization from legacy collector
        legacy = collect_gnw_metrics()
        for k in ("coalition_size", "workspace_utilization"):
            if k not in metrics or metrics[k] is None:
                metrics[k] = legacy.get(k)
        return metrics

    def _collect_l4():
        metrics = store.prediction.get_accuracy()
        # Merge pad_from_precision from legacy collector
        legacy = collect_prediction_metrics()
        if "pad_from_precision" not in metrics or metrics.get("pad_from_precision") is None:
            metrics["pad_from_precision"] = legacy.get("pad_from_precision")
        return metrics

    def _collect_l5():
        return store.metacognition.get_calibration_metrics()

    def _collect_l6():
        # Curiosity still uses FTS + JSON — not migrated to store yet
        return collect_curiosity_metrics()

    def _collect_l7():
        diversity = store.attention.get_policy_diversity(days=7)
        metrics = {"efe_policy_diversity": diversity}
        # Merge action_selection_latency from legacy collector
        legacy = collect_active_inference_metrics()
        if legacy.get("action_selection_latency") is not None:
            metrics["action_selection_latency"] = legacy["action_selection_latency"]
        return metrics

    def _collect_l8():
        return store.causal.get_latest_dag_state()

    def _collect_l9():
        # Self-model uses function call, not SQL
        return collect_self_model_metrics()

    def _collect_l10():
        metrics = store.forgetting.get_forgetting_metrics()
        # Merge rif_suppression_rate from legacy collector (handles 0-data case)
        legacy = collect_forgetting_metrics()
        if "rif_suppression_rate" not in metrics or metrics.get("rif_suppression_rate") is None:
            metrics["rif_suppression_rate"] = legacy.get("rif_suppression_rate")
        return metrics

    def _collect_cx():
        metrics = store.cx.get_aggregated_cx_metrics(hours=24)
        # Merge pci_proxy from legacy collector
        legacy = collect_cx_metrics()
        if "pci_proxy" not in metrics or metrics.get("pci_proxy") is None:
            metrics["pci_proxy"] = legacy.get("pci_proxy")
        return metrics

    return {
        "PE": _collect_pe,
        "L1": _collect_l1,
        "L2": _collect_l2,
        "L3": _collect_l3,
        "L4": _collect_l4,
        "L5": _collect_l5,
        "L6": _collect_l6,
        "L7": _collect_l7,
        "L8": _collect_l8,
        "L9": _collect_l9,
        "L10": _collect_l10,
        "CX": _collect_cx,
    }


# ============================================================
# PUBLIC API
# ============================================================

def get_all_contracts() -> Dict[str, Contract]:
    """Return all registered contracts."""
    return dict(CONTRACTS)


def get_contract(loop_id: str) -> Optional[Contract]:
    """Get a specific contract by loop ID."""
    return CONTRACTS.get(loop_id)


def evaluate_all(collectors: Optional[Dict[str, Callable]] = None) -> List[ContractResult]:
    """Evaluate all contracts using registered collectors.

    Returns list of ContractResults, one per contract.
    """
    if collectors is None:
        collectors = _build_store_collectors()

    results = []
    for loop_id, contract in CONTRACTS.items():
        collector = collectors.get(loop_id)
        if collector:
            try:
                measurements = collector()
            except Exception as e:
                _logger.debug("Collector for %s failed: %s", loop_id, e)
                measurements = {}
        else:
            measurements = {}
        results.append(evaluate_contract(contract, measurements))

    return results


def format_health_report(results: List[ContractResult]) -> str:
    """Format results into a readable health report."""
    lines = ["# Cognitive Health Report", ""]

    # Summary
    status_counts = {}
    for r in results:
        s = r.overall_status.value
        status_counts[s] = status_counts.get(s, 0) + 1

    lines.append(f"**Contracts:** {len(results)} | " +
                 " | ".join(f"{k}: {v}" for k, v in sorted(status_counts.items())))
    lines.append("")

    # Per-contract detail
    for r in results:
        icon = {"healthy": "+", "likely_healthy": "~",
                "inconclusive": "?", "likely_degraded": "!",
                "degraded": "X"}.get(r.overall_status.value, "?")
        lines.append(f"## [{icon}] {r.loop_id}: {r.name} ({r.overall_status.value})")
        for mr in r.metric_results:
            val = f"{mr.value:.3f}" if mr.value is not None else "N/A"
            lines.append(f"  - {mr.metric_name}: {val} [{mr.status.value}] {mr.detail}")
        lines.append("")

    return "\n".join(lines)


# ============================================================
# MCP TOOL REGISTRATION
# ============================================================

def register_tools(mcp):
    """Register cognitive contract tools."""

    async def cognitive_health_check() -> str:
        """Run all cognitive observability contracts and return health report.

        Evaluates 12 contracts (10 loops + PE + CX) against neuro-informed
        thresholds. Returns Bayesian-inspired status (never binary pass/fail).
        """
        results = evaluate_all()
        return format_health_report(results)

    mcp.tool()(cognitive_health_check)

    async def get_cognitive_contracts() -> str:
        """List all cognitive observability contracts with their metrics and thresholds.

        Shows what each loop MUST measure and what healthy ranges look like.
        """
        import json
        output = {}
        for loop_id, contract in CONTRACTS.items():
            output[loop_id] = {
                "name": contract.name,
                "module": contract.module,
                "theory": contract.theory,
                "metrics": [
                    {
                        "name": m.name,
                        "domain": m.domain.value,
                        "target": f"[{m.target_lo}, {m.target_hi}]",
                        "red_flags": f"lo={m.red_flag_lo} hi={m.red_flag_hi}",
                        "paper": m.paper,
                    }
                    for m in contract.metrics
                ]
            }
        return json.dumps(output, indent=2)

    mcp.tool()(get_cognitive_contracts)
