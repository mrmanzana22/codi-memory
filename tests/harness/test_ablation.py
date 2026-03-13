"""
P2 Evaluation Harness — Ablation Framework
============================================
Tests the CX ablation API and provides framework for ablation studies.

Methodology (from neuroscience consultant + systems architect):
  - Disable individual CX via event_bus.off()
  - Verify downstream handlers stop firing
  - Re-enable and verify recovery
  - Sham control: "disable" a non-existent CX, measure no change
  - Double-dissociation setup: disable CX-A shows effect on metric-X but not metric-Y

Phase 3 will use this framework for full ablation studies.
This test validates the mechanism works correctly.

References:
  - Casali et al. (2013): Perturbational Complexity Index methodology
  - Beggs & Plenz (2003): Neuronal avalanches — criticality testing
"""

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))


def run_ablation_test() -> dict:
    """Test ablation API correctness."""
    os.environ.setdefault('QDRANT_DISABLED', '1')

    from modules.events import event_bus, Events
    from modules.wiring import (
        wire_event_bus, get_cx_registry, disable_cx, enable_cx,
        get_disabled_cx, get_cx_summary,
    )

    wire_event_bus()

    results = {
        'test': 'ablation',
        'checks': {},
    }
    passed = 0
    failed = 0

    # --- Check 1: Disable CX-1, verify handler stops ---
    trace_before = 'ablation-before-cx1'
    event_bus.emit(Events.PREDICTION_ERROR, {
        'topic': 'ablation_test',
        'surprise_score': 0.7,
        'predicted_topic': 'general',
        'actual_topic': 'ablation_test',
        'learning_progress': 0.2,
        'trace_id': trace_before,
    })

    before_cascade = event_bus.get_cascade_log(trace_id=trace_before)
    cx1_before = [e for e in before_cascade if '_on_pe_drives_curiosity' in e.get('handler', '')]

    # Disable CX-1
    disabled_ok = disable_cx('CX-1')
    assert disabled_ok, "disable_cx('CX-1') should return True"

    trace_during = 'ablation-during-cx1'
    event_bus.emit(Events.PREDICTION_ERROR, {
        'topic': 'ablation_test',
        'surprise_score': 0.7,
        'predicted_topic': 'general',
        'actual_topic': 'ablation_test',
        'learning_progress': 0.2,
        'trace_id': trace_during,
    })

    during_cascade = event_bus.get_cascade_log(trace_id=trace_during)
    cx1_during = [e for e in during_cascade if '_on_pe_drives_curiosity' in e.get('handler', '')]

    # Re-enable CX-1
    enabled_ok = enable_cx('CX-1')
    assert enabled_ok, "enable_cx('CX-1') should return True"

    trace_after = 'ablation-after-cx1'
    event_bus.emit(Events.PREDICTION_ERROR, {
        'topic': 'ablation_test',
        'surprise_score': 0.7,
        'predicted_topic': 'general',
        'actual_topic': 'ablation_test',
        'learning_progress': 0.2,
        'trace_id': trace_after,
    })

    after_cascade = event_bus.get_cascade_log(trace_id=trace_after)
    cx1_after = [e for e in after_cascade if '_on_pe_drives_curiosity' in e.get('handler', '')]

    check_1 = {
        'name': 'disable_enable_cx1',
        'before_fires': len(cx1_before),
        'during_disabled_fires': len(cx1_during),
        'after_reenable_fires': len(cx1_after),
        'disabled_correctly': len(cx1_during) == 0,
        'recovered_correctly': len(cx1_after) > 0,
        'pass': len(cx1_during) == 0 and len(cx1_after) > 0 and len(cx1_before) > 0,
    }
    results['checks']['disable_enable'] = check_1
    if check_1['pass']:
        passed += 1
    else:
        failed += 1

    # --- Check 2: Pull-model CX cannot be disabled ---
    pull_result = disable_cx('CX-6')
    check_2 = {
        'name': 'pull_model_guard',
        'cx': 'CX-6',
        'disable_returned': pull_result,
        'pass': pull_result is False,
    }
    results['checks']['pull_guard'] = check_2
    if check_2['pass']:
        passed += 1
    else:
        failed += 1

    # --- Check 3: Sham control — disable non-existent CX ---
    sham_result = disable_cx('CX-99')
    check_3 = {
        'name': 'sham_control',
        'cx': 'CX-99',
        'disable_returned': sham_result,
        'pass': sham_result is False,
    }
    results['checks']['sham_control'] = check_3
    if check_3['pass']:
        passed += 1
    else:
        failed += 1

    # --- Check 4: Idempotent enable (already enabled) ---
    idem_result = enable_cx('CX-1')
    check_4 = {
        'name': 'idempotent_enable',
        'cx': 'CX-1',
        'result': idem_result,
        'pass': idem_result is True,
    }
    results['checks']['idempotent'] = check_4
    if check_4['pass']:
        passed += 1
    else:
        failed += 1

    # --- Check 5: Multi-CX disable (Tier 5 ablation) ---
    tier5_cx = ['CX-22', 'CX-23', 'CX-28']
    all_disabled = all(disable_cx(cx) for cx in tier5_cx)
    disabled_set = get_disabled_cx()
    tier5_in_disabled = all(cx in disabled_set for cx in tier5_cx)

    # Re-enable all
    all_enabled = all(enable_cx(cx) for cx in tier5_cx)
    final_disabled = get_disabled_cx()

    check_5 = {
        'name': 'multi_cx_ablation',
        'cx_ids': tier5_cx,
        'all_disabled': all_disabled,
        'in_disabled_set': tier5_in_disabled,
        'all_reenabled': all_enabled,
        'final_disabled_count': len(final_disabled),
        'pass': all_disabled and tier5_in_disabled and all_enabled and len(final_disabled) == 0,
    }
    results['checks']['multi_ablation'] = check_5
    if check_5['pass']:
        passed += 1
    else:
        failed += 1

    # --- Check 6: Summary still correct after ablation cycle ---
    summary = get_cx_summary()
    check_6 = {
        'name': 'post_ablation_integrity',
        'total_cx': summary['total'],
        'disabled': summary['disabled'],
        'pass': summary['disabled'] == 0 and summary['total'] >= 28,
    }
    results['checks']['integrity'] = check_6
    if check_6['pass']:
        passed += 1
    else:
        failed += 1

    # --- Summary ---
    results['passed'] = passed
    results['failed'] = failed
    results['total_checks'] = passed + failed
    results['pass'] = failed == 0

    return results


if __name__ == "__main__":
    result = run_ablation_test()
    print(json.dumps(result, indent=2, ensure_ascii=False))
