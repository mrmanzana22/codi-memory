#!/usr/bin/env python3
"""
P1 Evaluation Harness — Run All Tests
=======================================
Runs all automated evaluation tests and outputs a JSON report.

Usage:
    python tests/harness/run_all.py              # Full run (all tests)
    python tests/harness/run_all.py --fast       # Skip retrieval (slow, needs embeddings)
    python tests/harness/run_all.py --save       # Save report to CODI-BASELINES.md

Exit criteria (from CODI-EXECUTION-PLAN.md):
  - All 10 tests exist as executable scripts
  - run_all.py completes and outputs JSON report
  - Baseline measurements recorded
  - At least 7/10 tests meet targets
"""

import json
import os
import sys
import time
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

REPORT_PATH = os.path.expanduser("~/Desktop/CODI-BASELINES.md")


def run_all(skip_retrieval: bool = False) -> dict:
    """Run all harness tests and return combined report."""
    report = {
        "timestamp": datetime.now().isoformat(),
        "tests": {},
        "summary": {"total": 0, "passed": 0, "failed": 0, "skipped": 0, "errors": 0},
    }

    tests = []

    # 1-3: Retrieval (@5, @10, false_memory_rate)
    if skip_retrieval:
        report["tests"]["retrieval"] = {"test": "retrieval", "skipped": True}
        report["summary"]["skipped"] += 3
    else:
        print("Running: retrieval (@5, @10, false_memory)...", flush=True)
        t0 = time.time()
        try:
            from tests.harness.test_retrieval import run_retrieval_test
            result = run_retrieval_test()
            result["duration_s"] = round(time.time() - t0, 1)
            report["tests"]["retrieval"] = result

            # Count as 3 tests: @5, @10, false_memory
            for key in ["pass_at_5", "pass_at_10", "false_memory_pass"]:
                report["summary"]["total"] += 1
                if result.get(key):
                    report["summary"]["passed"] += 1
                else:
                    report["summary"]["failed"] += 1
        except Exception as e:
            report["tests"]["retrieval"] = {"test": "retrieval", "error": str(e)}
            report["summary"]["errors"] += 3

    # 4: Prediction accuracy
    print("Running: prediction_accuracy...", flush=True)
    t0 = time.time()
    try:
        from tests.harness.test_prediction import run_prediction_test
        result = run_prediction_test()
        result["duration_s"] = round(time.time() - t0, 1)
        report["tests"]["prediction_accuracy"] = result
        report["summary"]["total"] += 1
        if result.get("error"):
            report["summary"]["errors"] += 1
        elif result.get("pass"):
            report["summary"]["passed"] += 1
        else:
            report["summary"]["failed"] += 1
    except Exception as e:
        report["tests"]["prediction_accuracy"] = {"test": "prediction_accuracy", "error": str(e)}
        report["summary"]["errors"] += 1

    # 5: Consolidation coverage
    print("Running: consolidation_coverage...", flush=True)
    t0 = time.time()
    try:
        from tests.harness.test_consolidation import run_consolidation_test
        result = run_consolidation_test()
        result["duration_s"] = round(time.time() - t0, 1)
        report["tests"]["consolidation_coverage"] = result
        if result.get("skipped"):
            report["summary"]["skipped"] += 1
        else:
            report["summary"]["total"] += 1
            if result.get("error"):
                report["summary"]["errors"] += 1
            elif result.get("pass"):
                report["summary"]["passed"] += 1
            else:
                report["summary"]["failed"] += 1
    except Exception as e:
        report["tests"]["consolidation_coverage"] = {"test": "consolidation_coverage", "error": str(e)}
        report["summary"]["errors"] += 1

    # 6: Emotion variance
    print("Running: emotion_variance...", flush=True)
    t0 = time.time()
    try:
        from tests.harness.test_emotion import run_emotion_test
        result = run_emotion_test()
        result["duration_s"] = round(time.time() - t0, 1)
        report["tests"]["emotion_variance"] = result
        if result.get("skipped"):
            report["summary"]["skipped"] += 1
        else:
            report["summary"]["total"] += 1
            if result.get("error"):
                report["summary"]["errors"] += 1
            elif result.get("pass"):
                report["summary"]["passed"] += 1
            else:
                report["summary"]["failed"] += 1
    except Exception as e:
        report["tests"]["emotion_variance"] = {"test": "emotion_variance", "error": str(e)}
        report["summary"]["errors"] += 1

    # 7: Proactive relevance (manual — placeholder)
    report["tests"]["proactive_relevance"] = {
        "test": "proactive_relevance",
        "manual": True,
        "instructions": "Rate 20 proactive messages: useful/not useful. Target >= 60%.",
        "pass": None,
    }
    report["summary"]["skipped"] += 1

    # 8: Override success (manual — placeholder)
    report["tests"]["override_success"] = {
        "test": "override_success",
        "manual": True,
        "instructions": "Run 10 override test cases via Telegram. Target 100%.",
        "pass": None,
    }
    report["summary"]["skipped"] += 1

    # 9: Tick health
    print("Running: tick_health...", flush=True)
    t0 = time.time()
    try:
        from tests.harness.test_tick_health import run_tick_health_test
        result = run_tick_health_test()
        result["duration_s"] = round(time.time() - t0, 1)
        report["tests"]["tick_health"] = result
        report["summary"]["total"] += 1
        if result.get("error"):
            report["summary"]["errors"] += 1
        elif result.get("pass"):
            report["summary"]["passed"] += 1
        else:
            report["summary"]["failed"] += 1
    except Exception as e:
        report["tests"]["tick_health"] = {"test": "tick_health", "error": str(e)}
        report["summary"]["errors"] += 1

    # 10: Body handoff (manual — placeholder)
    report["tests"]["body_handoff"] = {
        "test": "body_handoff",
        "manual": True,
        "instructions": "Send message on Telegram, Claude Code, HTTP. Verify memory sees all 3. Target 100%.",
        "pass": None,
    }
    report["summary"]["skipped"] += 1

    # ---- P2 TESTS: Cross-Loop Integration (Fase 1) ----

    # 11: Event flow & instrumentation
    print("Running: event_flow...", flush=True)
    t0 = time.time()
    try:
        from tests.harness.test_event_flow import run_event_flow_test
        result = run_event_flow_test()
        result["duration_s"] = round(time.time() - t0, 1)
        report["tests"]["event_flow"] = result
        report["summary"]["total"] += 1
        if result.get("pass"):
            report["summary"]["passed"] += 1
        else:
            report["summary"]["failed"] += 1
    except Exception as e:
        report["tests"]["event_flow"] = {"test": "event_flow", "error": str(e)}
        report["summary"]["errors"] += 1

    # 12: Cross-loop chains
    print("Running: cross_loop_chains...", flush=True)
    t0 = time.time()
    try:
        from tests.harness.test_cross_loop_chains import run_cross_loop_chain_test
        result = run_cross_loop_chain_test()
        result["duration_s"] = round(time.time() - t0, 1)
        report["tests"]["cross_loop_chains"] = result
        report["summary"]["total"] += 1
        if result.get("pass"):
            report["summary"]["passed"] += 1
        else:
            report["summary"]["failed"] += 1
    except Exception as e:
        report["tests"]["cross_loop_chains"] = {"test": "cross_loop_chains", "error": str(e)}
        report["summary"]["errors"] += 1

    # 13: E/I balance
    print("Running: ei_balance...", flush=True)
    t0 = time.time()
    try:
        from tests.harness.test_ei_balance import run_ei_balance_test
        result = run_ei_balance_test()
        result["duration_s"] = round(time.time() - t0, 1)
        report["tests"]["ei_balance"] = result
        report["summary"]["total"] += 1
        if result.get("pass"):
            report["summary"]["passed"] += 1
        else:
            report["summary"]["failed"] += 1
    except Exception as e:
        report["tests"]["ei_balance"] = {"test": "ei_balance", "error": str(e)}
        report["summary"]["errors"] += 1

    # 14: Feedback stability
    print("Running: feedback_stability...", flush=True)
    t0 = time.time()
    try:
        from tests.harness.test_feedback_stability import run_feedback_stability_test
        result = run_feedback_stability_test()
        result["duration_s"] = round(time.time() - t0, 1)
        report["tests"]["feedback_stability"] = result
        report["summary"]["total"] += 1
        if result.get("pass"):
            report["summary"]["passed"] += 1
        else:
            report["summary"]["failed"] += 1
    except Exception as e:
        report["tests"]["feedback_stability"] = {"test": "feedback_stability", "error": str(e)}
        report["summary"]["errors"] += 1

    # 15: Ablation framework
    print("Running: ablation...", flush=True)
    t0 = time.time()
    try:
        from tests.harness.test_ablation import run_ablation_test
        result = run_ablation_test()
        result["duration_s"] = round(time.time() - t0, 1)
        report["tests"]["ablation"] = result
        report["summary"]["total"] += 1
        if result.get("pass"):
            report["summary"]["passed"] += 1
        else:
            report["summary"]["failed"] += 1
    except Exception as e:
        report["tests"]["ablation"] = {"test": "ablation", "error": str(e)}
        report["summary"]["errors"] += 1

    # 16: PCI (Perturbational Complexity Index)
    print("Running: pci...", flush=True)
    t0 = time.time()
    try:
        from tests.harness.test_pci import run_pci_test
        result = run_pci_test()
        result["duration_s"] = round(time.time() - t0, 1)
        report["tests"]["pci"] = result
        report["summary"]["total"] += 1
        if result.get("pass"):
            report["summary"]["passed"] += 1
        else:
            report["summary"]["failed"] += 1
    except Exception as e:
        report["tests"]["pci"] = {"test": "pci", "error": str(e)}
        report["summary"]["errors"] += 1

    # 17: Failure modes
    print("Running: failure_modes...", flush=True)
    t0 = time.time()
    try:
        from tests.harness.test_failure_modes import run_failure_modes_test
        result = run_failure_modes_test()
        result["duration_s"] = round(time.time() - t0, 1)
        report["tests"]["failure_modes"] = result
        report["summary"]["total"] += 1
        if result.get("pass"):
            report["summary"]["passed"] += 1
        else:
            report["summary"]["failed"] += 1
    except Exception as e:
        report["tests"]["failure_modes"] = {"test": "failure_modes", "error": str(e)}
        report["summary"]["errors"] += 1

    # Final summary
    total_scored = report["summary"]["total"]
    report["summary"]["pass_rate"] = (
        round(report["summary"]["passed"] / total_scored, 3) if total_scored else 0
    )
    report["summary"]["meets_p1_target"] = report["summary"]["passed"] >= 7
    report["summary"]["meets_p2_target"] = report["summary"]["passed"] >= 10

    return report


def save_baselines(report: dict, path: str = REPORT_PATH) -> str:
    """Save report as CODI-BASELINES.md."""
    s = report["summary"]
    lines = [
        "# CODI Baselines",
        f"**Generated:** {report['timestamp']}",
        f"**Summary:** {s['passed']}/{s['total']} passed, "
        f"{s['failed']} failed, {s['skipped']} skipped, {s['errors']} errors",
        f"**P1 Target (7/10 pass):** {'MET' if s['meets_p1_target'] else 'NOT MET'}",
        f"**P2 Target (10/15 pass):** {'MET' if s.get('meets_p2_target') else 'NOT MET'}",
        "",
        "## Results",
        "",
        "| # | Test | Result | Value | Target |",
        "|---|------|--------|-------|--------|",
    ]

    test_rows = [
        ("1", "retrieval@5", report["tests"].get("retrieval", {}), "precision_at_5", "pass_at_5", ">= 0.70"),
        ("2", "retrieval@10", report["tests"].get("retrieval", {}), "precision_at_10", "pass_at_10", ">= 0.85"),
        ("3", "false_memory_rate", report["tests"].get("retrieval", {}), "false_memory_rate", "false_memory_pass", "< 0.05"),
        ("4", "prediction_accuracy", report["tests"].get("prediction_accuracy", {}), "accuracy", "pass", ">= 0.40"),
        ("5", "consolidation_coverage", report["tests"].get("consolidation_coverage", {}), "cumulative_coverage", "pass", ">= 0.80"),
        ("6", "emotion_variance", report["tests"].get("emotion_variance", {}), "max_range", "pass", "range > 0.1"),
        ("7", "proactive_relevance", report["tests"].get("proactive_relevance", {}), None, "pass", ">= 60%"),
        ("8", "override_success", report["tests"].get("override_success", {}), None, "pass", "100%"),
        ("9", "tick_health", report["tests"].get("tick_health", {}), "health_pct", "pass", ">= 0.95"),
        ("10", "body_handoff", report["tests"].get("body_handoff", {}), None, "pass", "100%"),
        # P2 — Cross-Loop Integration
        ("11", "event_flow", report["tests"].get("event_flow", {}), "total_checks", "pass", "all checks"),
        ("12", "cross_loop_chains", report["tests"].get("cross_loop_chains", {}), "total_chains", "pass", "all chains"),
        ("13", "ei_balance", report["tests"].get("ei_balance", {}), None, "pass", "65-85% E"),
        ("14", "feedback_stability", report["tests"].get("feedback_stability", {}), None, "pass", "no runaway"),
        ("15", "ablation", report["tests"].get("ablation", {}), "total_checks", "pass", "all checks"),
        ("16", "pci", report["tests"].get("pci", {}), "avg_pci", "pass", "0.05-0.80"),
        ("17", "failure_modes", report["tests"].get("failure_modes", {}), "total_checks", "pass", "all checks"),
    ]

    for num, name, data, val_key, pass_key, target in test_rows:
        if data.get("skipped"):
            lines.append(f"| {num} | {name} | SKIP | - | {target} |")
        elif data.get("manual"):
            lines.append(f"| {num} | {name} | MANUAL | pending | {target} |")
        elif data.get("error"):
            lines.append(f"| {num} | {name} | ERROR | {data['error'][:40]} | {target} |")
        else:
            val = data.get(val_key, "?") if val_key else "?"
            passed = data.get(pass_key)
            status = "PASS" if passed else "FAIL"
            lines.append(f"| {num} | {name} | **{status}** | {val} | {target} |")

    lines.append("")
    lines.append("## Raw JSON")
    lines.append("```json")
    # Compact version without per-query details
    compact = {k: {kk: vv for kk, vv in v.items() if kk != "details"}
                for k, v in report["tests"].items()}
    lines.append(json.dumps(compact, indent=2, ensure_ascii=False))
    lines.append("```")

    content = "\n".join(lines)
    with open(path, "w") as f:
        f.write(content)
    return path


if __name__ == "__main__":
    skip_retrieval = "--fast" in sys.argv
    save = "--save" in sys.argv

    report = run_all(skip_retrieval=skip_retrieval)

    # Always print JSON
    print(json.dumps(report, indent=2, ensure_ascii=False))

    if save:
        path = save_baselines(report)
        print(f"\nBaselines saved to: {path}", file=sys.stderr)

    # Exit code: 0 if P1 target met, 1 otherwise
    sys.exit(0 if report["summary"]["meets_p1_target"] else 1)
