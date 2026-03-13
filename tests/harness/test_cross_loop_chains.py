"""
P2 Evaluation Harness — Cross-Loop Chain Integration
=====================================================
Tests that cross-loop chains fire correctly end-to-end.

Methodology:
  1. Wire the event bus with CX_REGISTRY
  2. Emit a trigger event
  3. Verify cascade flows through expected CX handlers via cascade_log
  4. Test all 5 tiers independently

Key chains tested:
  - PE → CX-1 → curiosity (Tier 1)
  - Curiosity → CX-2 → prediction + CX-11 → causal (Tier 1+3)
  - Self-model → CX-3 → workspace + CX-10 → metacognition (Tier 1+3)
  - Consolidation → CX-4b + CX-14 + CX-17 + CX-19 (Tier 1+3+4)
  - Workspace → CX-5 + CX-9 + CX-16 + CX-25 (Tier 2+3+4+5)
  - Reconsolidation → CX-8 + CX-18 + CX-29 (Tier 2+4+5)
  - Metacognition → CX-20 + CX-24 + CX-27 (Tier 4+5)
  - Memory vaulted → CX-4a + CX-23 + CX-28 (Tier 1+5)
  - Action selected → CX-22 + CX-30 (Tier 5)

References:
  - Cross-loop implementation: CROSS_LOOP_FINDINGS.md (Tiers 1-5)
  - CX_REGISTRY: wiring.py Fase 0
"""

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))


# Define expected chains: trigger_event -> expected handler substrings in cascade
CHAIN_TESTS = {
    'pe_chain': {
        'desc': 'PREDICTION_ERROR → CX-1 curiosity',
        'event': 'prediction_error',
        'payload': {
            'topic': 'test_topic',
            'surprise_score': 0.7,
            'predicted_topic': 'general',
            'actual_topic': 'test_topic',
            'learning_progress': 0.3,
        },
        'expected_handlers': ['_on_prediction_error', '_on_pe_drives_curiosity'],
        'tiers': [1],
    },
    'curiosity_chain': {
        'desc': 'CURIOSITY_RESOLVED → CX-2 prediction + CX-11 causal',
        'event': 'curiosity_resolved',
        'payload': {
            'question': 'test curiosity question',
            'topic': 'test_topic',
            'discovery': 'test discovery result',
            'saved_memory_id': 'test-mem-001',
        },
        'expected_handlers': [
            '_on_curiosity_resolved',
            '_on_curiosity_resolved_prediction',
            '_on_curiosity_feeds_causal',
        ],
        'tiers': [1, 3],
    },
    'self_model_chain': {
        'desc': 'SELF_MODEL_REFRESHED → CX-3 workspace + CX-10 meta + CX-15 forget + CX-26 action',
        'event': 'self_model_refreshed',
        'payload': {
            'summary': 'I am a test cognitive system',
            'discrepancy_score': 0.2,
            'confidence': 0.8,
            'topics': ['test'],
        },
        'expected_handlers': [
            '_on_self_model_refreshed',
            '_on_self_model_to_competition',
            '_on_self_model_to_metacognition',
            '_on_self_model_modulates_forgetting',
            '_on_self_model_constrains_action',
        ],
        'tiers': [1, 3, 4, 5],
    },
    'consolidation_chain': {
        'desc': 'CONSOLIDATION_COMPLETE → CX-4b + CX-14 + CX-17 + CX-19',
        'event': 'consolidation_complete',
        'payload': {
            'batch_id': 'test-batch',
            'episodes_scanned': 50,
            'clusters_found': 5,
            'facts_created': 3,
            'pruned_ids': ['m1', 'm2'],
            'schemas': [{'topic': 'test', 'confidence': 0.9}],
            'contradictions': [],
            'bridge_edges': [],
            'low_density_clusters': [],
        },
        'expected_handlers': [
            '_on_consolidation_complete',
            '_on_consolidation_protects_decay',
            '_on_consolidation_gaps_drive_curiosity',
            '_on_consolidation_updates_prediction_priors',
            '_on_consolidation_feeds_self_model',
        ],
        'tiers': [1, 3, 4],
    },
    'workspace_chain': {
        'desc': 'WORKSPACE_COMPETITION_COMPLETE → CX-5 + CX-9 + CX-16 + CX-25',
        'event': 'workspace_competition_complete',
        'payload': {
            'winner_content': 'test workspace winner',
            'winner_score': 0.85,
            'coalition_size': 3,
            'novelty': 0.6,
            'loser_ids': [],
            'competition_id': 'test-comp-001',
        },
        'expected_handlers': [
            '_on_competition_complete',
            '_on_gnw_broadcast_to_action',
            '_on_workspace_to_self_model',
            '_on_workspace_broadcast_to_metacognition',
            '_on_workspace_retrieval_modulates_forgetting',
        ],
        'tiers': [2, 3, 4, 5],
    },
    'reconsolidation_chain': {
        'desc': 'RECONSOLIDATION_TRIGGERED → CX-8 + CX-18 + CX-29',
        'event': 'reconsolidation_triggered',
        'payload': {
            'memory_id': 'test-mem-recon',
            'old_content': 'old version',
            'new_content': 'corrected version',
            'new_confidence': 0.7,
            'correction_type': 'update',
        },
        'expected_handlers': [
            '_on_reconsolidation_triggered',
            '_on_reconsolidation_protects_decay',
            '_on_reconsolidation_to_metacognition',
            '_on_reconsolidation_invalidates_causal_edges',
        ],
        'tiers': [2, 4, 5],
    },
    'metacognition_chain': {
        'desc': 'METACOGNITIVE_CONTROL_APPLIED → CX-20 + CX-24 + CX-27',
        'event': 'metacognitive_control_applied',
        'payload': {
            'action': 'test_precision_damping',
            'target': 'L0',
            'l2_overconfidence': 0.15,
            'fok_score': 0.3,
            'topic': 'test_topic',
        },
        'expected_handlers': [
            '_on_metacognitive_control',
            '_on_metacognitive_uncertainty_to_curiosity',
            '_on_metacognition_gates_reconsolidation',
            '_on_metacognition_suppresses_causal_edges',
        ],
        'tiers': [4, 5],
    },
    'vault_chain': {
        'desc': 'MEMORY_VAULTED → CX-4a + CX-23 + CX-28',
        'event': 'memory_vaulted',
        'payload': {
            'memory_id': 'test-mem-vault',
            'importance': 0.6,
            'topic': 'test_topic',
            'content_snippet': 'test vaulted memory',
        },
        'expected_handlers': [
            '_on_memory_vaulted',
            '_on_vault_tracks_urgency',
            '_on_forgetting_suppresses_curiosity',
            '_on_forgetting_degrades_metacognition',
        ],
        'tiers': [1, 5],
    },
    'action_chain': {
        'desc': 'ACTION_SELECTED → CX-22 + CX-30',
        'event': 'action_selected',
        'payload': {
            'action': 'topic_exploration',
            'efe_score': -0.5,
            'option_count': 4,
            'selected_index': 0,
            'outcome': 'success',
        },
        'expected_handlers': [
            '_on_action_selected_to_metacognition',
            '_on_action_outcome_updates_causal_dag',
        ],
        'tiers': [5],
    },
}


def run_cross_loop_chain_test() -> dict:
    """Test all CX chain paths fire correctly."""
    os.environ.setdefault('QDRANT_DISABLED', '1')

    from modules.events import event_bus
    from modules.wiring import wire_event_bus, get_cx_registry

    wire_event_bus()

    results = {
        'test': 'cross_loop_chains',
        'chains': {},
    }
    passed = 0
    failed = 0

    for chain_id, chain_spec in CHAIN_TESTS.items():
        trace_id = f'chain-test-{chain_id}'

        # Clear cascade log by emitting enough events to flush old ones
        # (ring buffer is 200 entries, we just need fresh trace_id)

        # Emit trigger event
        payload = dict(chain_spec['payload'])
        payload['trace_id'] = trace_id

        event_bus.emit(chain_spec['event'], payload)

        # Check cascade log for expected handlers
        cascade = event_bus.get_cascade_log(trace_id=trace_id)
        fired_handlers = [entry['handler'] for entry in cascade]

        # Check which expected handlers actually fired
        found = []
        missing = []
        for expected in chain_spec['expected_handlers']:
            matched = any(expected in h for h in fired_handlers)
            if matched:
                found.append(expected)
            else:
                missing.append(expected)

        chain_pass = len(missing) == 0

        results['chains'][chain_id] = {
            'desc': chain_spec['desc'],
            'event': chain_spec['event'],
            'tiers': chain_spec['tiers'],
            'expected': len(chain_spec['expected_handlers']),
            'fired': len(found),
            'missing': missing,
            'found': found,
            'cascade_entries': len(cascade),
            'latencies_ms': [round(e['latency_ms'], 2) for e in cascade],
            'pass': chain_pass,
        }

        if chain_pass:
            passed += 1
        else:
            failed += 1

    # --- Tier coverage summary ---
    tiers_tested = set()
    for chain_spec in CHAIN_TESTS.values():
        tiers_tested.update(chain_spec['tiers'])

    results['passed'] = passed
    results['failed'] = failed
    results['total_chains'] = len(CHAIN_TESTS)
    results['tiers_covered'] = sorted(tiers_tested)
    results['pass'] = failed == 0

    return results


if __name__ == "__main__":
    result = run_cross_loop_chain_test()
    print(json.dumps(result, indent=2, ensure_ascii=False))
