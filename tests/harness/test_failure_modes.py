"""
P2 Evaluation Harness — Failure Mode Battery
==============================================
Tests system behavior under degraded conditions.

Methodology (from neuroscience consultant):
  "The most revealing test of a conscious system is what happens
   when components fail. Graceful degradation = integration.
   Catastrophic failure = mere aggregation."

Failure modes tested:
  1. Silent handler: Handler raises exception — should be caught, logged, not propagate
  2. Missing handler: CX disabled — downstream still functions
  3. Cascade isolation: Error in one chain doesn't affect parallel chains
  4. Recovery after ablation: Disable + re-enable CX, verify full recovery
  5. Event storm: 100 rapid events — system doesn't OOM or deadlock
  6. Orphan event: Emit event with no handlers — no crash

References:
  - Tononi (2004): Integration and information — phi as integration measure
  - Dehaene (2014): Consciousness and the Brain — graceful degradation in GNW
"""

import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))


def run_failure_modes_test() -> dict:
    """Test system resilience under various failure conditions."""
    os.environ.setdefault('QDRANT_DISABLED', '1')

    from modules.events import event_bus, Events
    from modules.wiring import wire_event_bus, disable_cx, enable_cx

    wire_event_bus()
    event_bus.reset_instrumentation()

    results = {
        'test': 'failure_modes',
        'modes': {},
    }
    passed = 0
    failed = 0

    # --- Mode 1: Silent handler exception ---
    # Register a handler that raises, verify it doesn't propagate
    error_count = [0]

    def _bad_handler(data):
        error_count[0] += 1
        raise ValueError("Intentional test failure")

    event_bus.on('memory_stored', _bad_handler)
    try:
        # This should NOT raise despite the bad handler
        event_bus.emit('memory_stored', {
            'memory_id': 'fail-test-001',
            'content': 'test',
            'trace_id': 'fail-mode-1',
        })
        no_propagation = True
    except Exception:
        no_propagation = False
    finally:
        event_bus.off('memory_stored', _bad_handler)

    mode_1 = {
        'name': 'silent_handler_exception',
        'handler_called': error_count[0] > 0,
        'exception_caught': no_propagation,
        'pass': no_propagation and error_count[0] > 0,
    }
    results['modes']['silent_exception'] = mode_1
    if mode_1['pass']:
        passed += 1
    else:
        failed += 1

    # --- Mode 2: Missing handler (CX disabled) ---
    # Disable CX-1, emit PE, verify base handler still fires
    event_bus.reset_instrumentation()
    disable_cx('CX-1')

    event_bus.emit(Events.PREDICTION_ERROR, {
        'topic': 'fail_test',
        'surprise_score': 0.7,
        'predicted_topic': 'general',
        'actual_topic': 'fail_test',
        'learning_progress': 0.2,
        'trace_id': 'fail-mode-2',
    })

    cascade = event_bus.get_cascade_log(trace_id='fail-mode-2')
    base_handler_fired = any('_on_prediction_error' in e.get('handler', '')
                            and 'curiosity' not in e.get('handler', '')
                            for e in cascade)
    cx1_fired = any('_on_pe_drives_curiosity' in e.get('handler', '') for e in cascade)

    enable_cx('CX-1')

    mode_2 = {
        'name': 'missing_handler_graceful',
        'base_handler_fired': base_handler_fired,
        'cx1_correctly_missing': not cx1_fired,
        'pass': base_handler_fired and not cx1_fired,
    }
    results['modes']['missing_handler'] = mode_2
    if mode_2['pass']:
        passed += 1
    else:
        failed += 1

    # --- Mode 3: Cascade isolation ---
    # Error in one handler shouldn't prevent other handlers from firing
    event_bus.reset_instrumentation()

    call_log = []

    def _failing_handler(data):
        call_log.append('failing')
        raise RuntimeError("Cascade isolation test")

    def _surviving_handler(data):
        call_log.append('surviving')

    test_event = 'test_cascade_isolation'
    event_bus.on(test_event, _failing_handler)
    event_bus.on(test_event, _surviving_handler)

    event_bus.emit(test_event, {'trace_id': 'fail-mode-3'})

    event_bus.off(test_event, _failing_handler)
    event_bus.off(test_event, _surviving_handler)

    mode_3 = {
        'name': 'cascade_isolation',
        'failing_called': 'failing' in call_log,
        'surviving_called': 'surviving' in call_log,
        'pass': 'failing' in call_log and 'surviving' in call_log,
    }
    results['modes']['cascade_isolation'] = mode_3
    if mode_3['pass']:
        passed += 1
    else:
        failed += 1

    # --- Mode 4: Recovery after ablation ---
    event_bus.reset_instrumentation()
    disable_cx('CX-22')
    disable_cx('CX-30')

    # Fire with both disabled
    event_bus.emit(Events.ACTION_SELECTED, {
        'action': 'test',
        'trace_id': 'fail-mode-4a',
    })
    disabled_cascade = event_bus.get_cascade_log(trace_id='fail-mode-4a')
    cx22_during = any('_on_action_selected_to_metacognition' in e.get('handler', '')
                      for e in disabled_cascade)

    # Re-enable both
    enable_cx('CX-22')
    enable_cx('CX-30')

    event_bus.emit(Events.ACTION_SELECTED, {
        'action': 'test',
        'trace_id': 'fail-mode-4b',
    })
    enabled_cascade = event_bus.get_cascade_log(trace_id='fail-mode-4b')
    cx22_after = any('_on_action_selected_to_metacognition' in e.get('handler', '')
                     for e in enabled_cascade)
    cx30_after = any('_on_action_outcome_updates_causal_dag' in e.get('handler', '')
                     for e in enabled_cascade)

    mode_4 = {
        'name': 'recovery_after_ablation',
        'disabled_correctly': not cx22_during,
        'cx22_recovered': cx22_after,
        'cx30_recovered': cx30_after,
        'pass': not cx22_during and cx22_after and cx30_after,
    }
    results['modes']['recovery'] = mode_4
    if mode_4['pass']:
        passed += 1
    else:
        failed += 1

    # --- Mode 5: Event storm (100 rapid events) ---
    event_bus.reset_instrumentation()
    t0 = time.monotonic()

    for i in range(100):
        event_bus.emit(Events.MEMORY_RETRIEVED, {
            'query': f'storm query {i}',
            'result_count': 5,
            'trace_id': f'fail-mode-5-{i}',
        })

    storm_duration_ms = (time.monotonic() - t0) * 1000
    storm_cascade = event_bus.get_cascade_log(limit=200)

    mode_5 = {
        'name': 'event_storm_100',
        'events_sent': 100,
        'duration_ms': round(storm_duration_ms, 1),
        'cascade_entries': len(storm_cascade),
        'avg_ms_per_event': round(storm_duration_ms / 100, 2),
        'no_deadlock': True,  # If we got here, no deadlock
        'pass': storm_duration_ms < 30000,  # 100 events in under 30s
    }
    results['modes']['event_storm'] = mode_5
    if mode_5['pass']:
        passed += 1
    else:
        failed += 1

    # --- Mode 6: Orphan event (no handlers) ---
    event_bus.reset_instrumentation()
    try:
        event_bus.emit('completely_unknown_event_type', {
            'data': 'should not crash',
            'trace_id': 'fail-mode-6',
        })
        orphan_safe = True
    except Exception:
        orphan_safe = False

    mode_6 = {
        'name': 'orphan_event',
        'event': 'completely_unknown_event_type',
        'no_crash': orphan_safe,
        'pass': orphan_safe,
    }
    results['modes']['orphan_event'] = mode_6
    if mode_6['pass']:
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
    result = run_failure_modes_test()
    print(json.dumps(result, indent=2, ensure_ascii=False))
