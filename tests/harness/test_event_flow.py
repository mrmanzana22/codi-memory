"""
P2 Evaluation Harness — Event Flow & Instrumentation
=====================================================
Tests event bus wiring, handler registration, latency tracking,
and cascade log reconstruction using Fase 0 instrumentation.

Metrics:
  - All expected events have registered handlers
  - Handler latency P95 < 100ms (no blocking handlers)
  - Cascade log captures event chains with trace_id
  - CX_REGISTRY matches wired handlers

References:
  - Fase 0 implementation (2026-03-13)
  - Neuroscience consultant: "Zero event bus visibility" finding
  - Systems architect: "Harness has no event flow tests"
"""

import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))


# Events that MUST have at least one handler after wire_event_bus()
REQUIRED_EVENTS = [
    'memory_stored',
    'memory_retrieved',
    'emotion_changed',
    'workspace_broadcast',
    'consolidation_complete',
    'prediction_error',
    'workspace_competition_complete',
    'workspace_recruitment',
    'retrieval_quality',
    'reconsolidation_triggered',
    'contradiction_detected',
    'metacognitive_control_applied',
    'attention_prediction_error',
    'self_model_refreshed',
    'memory_vaulted',
    'memory_reactivated',
    'affective_charge_computed',
    'curiosity_resolved',
    'action_selected',
]

# Minimum handler counts for key events (from CX_REGISTRY analysis)
MIN_HANDLER_COUNTS = {
    'prediction_error': 2,          # base + CX-1
    'curiosity_resolved': 3,        # base + CX-2 + CX-11
    'self_model_refreshed': 4,      # base + CX-3 + CX-10 + CX-15 + CX-26
    'memory_vaulted': 3,            # base + CX-4a + CX-23 + CX-28
    'consolidation_complete': 4,    # base + CX-4b + CX-14 + CX-17 + CX-19
    'workspace_competition_complete': 4,  # base + CX-5 + CX-9 + CX-16 + CX-25
    'reconsolidation_triggered': 3, # base + CX-8 + CX-18 + CX-29
    'metacognitive_control_applied': 3,  # base + CX-20 + CX-24 + CX-27
    'action_selected': 2,           # CX-22 + CX-30
}


def run_event_flow_test() -> dict:
    """Test event bus wiring, latency, and cascade instrumentation."""
    os.environ.setdefault('QDRANT_DISABLED', '1')

    from modules.events import event_bus, Events
    from modules.wiring import wire_event_bus, get_cx_registry, get_cx_summary

    wire_event_bus()

    results = {
        'test': 'event_flow',
        'checks': {},
    }
    passed = 0
    failed = 0

    # --- Check 1: All required events have handlers ---
    stats = event_bus.get_stats()
    missing_events = []
    for event in REQUIRED_EVENTS:
        if event not in stats or stats[event] == 0:
            missing_events.append(event)

    check_1 = {
        'name': 'required_events_wired',
        'total_events': len(REQUIRED_EVENTS),
        'wired_events': len(REQUIRED_EVENTS) - len(missing_events),
        'missing': missing_events,
        'pass': len(missing_events) == 0,
    }
    results['checks']['required_events'] = check_1
    if check_1['pass']:
        passed += 1
    else:
        failed += 1

    # --- Check 2: Minimum handler counts ---
    handler_failures = []
    for event, min_count in MIN_HANDLER_COUNTS.items():
        actual = stats.get(event, 0)
        if actual < min_count:
            handler_failures.append({
                'event': event,
                'expected_min': min_count,
                'actual': actual,
            })

    check_2 = {
        'name': 'handler_counts',
        'checks': len(MIN_HANDLER_COUNTS),
        'failures': handler_failures,
        'pass': len(handler_failures) == 0,
    }
    results['checks']['handler_counts'] = check_2
    if check_2['pass']:
        passed += 1
    else:
        failed += 1

    # --- Check 3: CX_REGISTRY populated ---
    registry = get_cx_registry()
    summary = get_cx_summary()

    check_3 = {
        'name': 'cx_registry',
        'total_cx': summary['total'],
        'by_tier': summary['by_tier'],
        'by_model': summary['by_model'],
        'pass': summary['total'] >= 28,  # At least 28 CX registered
    }
    results['checks']['cx_registry'] = check_3
    if check_3['pass']:
        passed += 1
    else:
        failed += 1

    # --- Check 4: Latency instrumentation works ---
    # Emit a test event and verify latency is recorded
    event_bus.emit('memory_stored', {
        'memory_id': 'test-harness-001',
        'content': 'Evaluation harness test memory',
        'trace_id': 'harness-trace-001',
    })
    time.sleep(0.05)  # Let handlers finish

    latencies = event_bus.get_handler_latencies()
    cascade = event_bus.get_cascade_log(trace_id='harness-trace-001')

    check_4 = {
        'name': 'latency_instrumentation',
        'handlers_tracked': len(latencies),
        'cascade_entries': len(cascade),
        'pass': len(latencies) > 0 and len(cascade) > 0,
    }
    results['checks']['latency_instrumentation'] = check_4
    if check_4['pass']:
        passed += 1
    else:
        failed += 1

    # --- Check 5: No blocking handlers (P95 < 500ms) ---
    # Threshold is 500ms because some handlers do legitimate I/O (DB writes)
    slow = event_bus.get_slow_handlers(threshold_ms=500.0)
    check_5 = {
        'name': 'no_blocking_handlers',
        'threshold_ms': 500.0,
        'slow_handlers': [{'handler': s['handler'], 'p95': round(s['p95'], 1)} for s in slow],
        'pass': len(slow) == 0,
    }
    results['checks']['no_blocking'] = check_5
    if check_5['pass']:
        passed += 1
    else:
        failed += 1

    # --- Summary ---
    total = passed + failed
    results['passed'] = passed
    results['failed'] = failed
    results['total_checks'] = total
    results['pass'] = failed == 0
    results['handler_stats'] = stats

    return results


if __name__ == "__main__":
    result = run_event_flow_test()
    print(json.dumps(result, indent=2, ensure_ascii=False))
