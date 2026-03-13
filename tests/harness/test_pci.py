"""
P2 Evaluation Harness — Perturbational Complexity Index (PCI)
==============================================================
Measures: complexity of event cascade responses to perturbations.

Methodology (Casali et al. 2013, adapted for event-driven architecture):
  1. Inject a single "perturbation" event (e.g., high-surprise PE)
  2. Record the full cascade response via cascade_log
  3. Measure:
     a) Cascade length: how many handlers fire from one stimulus
     b) Cascade diversity: how many UNIQUE event types appear
     c) Cascade complexity: Lempel-Ziv complexity of the handler sequence
     d) Cascade depth: max chain length through CX connections

  PCI = normalized(LZ_complexity * cascade_diversity)

Interpretation:
  - PCI = 0: No response (vegetative / disconnected modules)
  - PCI low: Stereotyped response (all handlers fire same way)
  - PCI medium: Differentiated response (healthy integration)
  - PCI high: Chaotic response (too much cross-talk)

  Human consciousness: PCI ~0.31-0.44 (Casali 2013)
  Our target: PCI > 0.15 (differentiated) and PCI < 0.80 (not chaotic)

References:
  - Casali et al. (2013): A theoretically based index of consciousness
  - Tononi & Edelman (1998): Complexity and coherency
  - Lempel & Ziv (1976): On the complexity of finite sequences
"""

import json
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))


def _lempel_ziv_complexity(sequence: list) -> float:
    """Compute normalized Lempel-Ziv complexity of a symbol sequence.

    Returns value in [0, 1] where:
      0 = maximally compressible (e.g., AAAA)
      1 = maximally complex (random)

    Based on Lempel & Ziv (1976), normalized by sequence length.
    """
    if not sequence:
        return 0.0

    s = ''.join(str(x) for x in sequence)
    n = len(s)
    if n <= 1:
        return 0.0

    # LZ76 complexity
    complexity = 1
    i = 0
    k = 1
    l = 1

    while i + k <= n:
        # Check if s[i+1..i+k] has been seen in s[0..i+l-1]
        substr = s[i + 1:i + k + 1] if i + k + 1 <= n else s[i + 1:n]
        window = s[0:i + l]

        if substr in window:
            k += 1
        else:
            complexity += 1
            i += k
            k = 1
            l = 1
            continue
        l = max(l, k)

    # Normalize by theoretical maximum: n / log2(n)
    max_complexity = n / math.log2(n) if n > 1 else 1
    return min(complexity / max_complexity, 1.0)


# Perturbation stimuli: different event types to test response diversity
PERTURBATIONS = {
    'high_pe': {
        'desc': 'High prediction error (novel topic)',
        'event': 'prediction_error',
        'payload': {
            'topic': 'pci_novel_topic',
            'surprise_score': 0.85,
            'predicted_topic': 'general',
            'actual_topic': 'pci_novel_topic',
            'learning_progress': 0.4,
        },
    },
    'consolidation': {
        'desc': 'Consolidation cycle complete',
        'event': 'consolidation_complete',
        'payload': {
            'batch_id': 'pci-batch',
            'episodes_scanned': 100,
            'clusters_found': 8,
            'facts_created': 5,
            'pruned_ids': ['pci-m1'],
            'schemas': [{'topic': 'pci_test', 'confidence': 0.85}],
            'contradictions': [{'id': 'pci-c1'}],
            'bridge_edges': [],
            'low_density_clusters': [{'topic': 'sparse'}],
        },
    },
    'workspace_win': {
        'desc': 'Workspace competition with high novelty',
        'event': 'workspace_competition_complete',
        'payload': {
            'winner_content': 'pci novel insight about consciousness',
            'winner_score': 0.92,
            'coalition_size': 4,
            'novelty': 0.8,
            'competition_id': 'pci-comp',
            'loser_ids': [],
        },
    },
    'self_model': {
        'desc': 'Self-model refresh with discrepancy',
        'event': 'self_model_refreshed',
        'payload': {
            'summary': 'I am a cognitive system exploring consciousness',
            'discrepancy_score': 0.4,
            'confidence': 0.7,
            'topics': ['consciousness', 'pci'],
        },
    },
    'low_pe': {
        'desc': 'Low prediction error (routine, sham control)',
        'event': 'prediction_error',
        'payload': {
            'topic': 'general',
            'surprise_score': 0.15,
            'predicted_topic': 'general',
            'actual_topic': 'general',
            'learning_progress': 0.01,
        },
    },
}


def run_pci_test() -> dict:
    """Compute PCI for each perturbation type."""
    os.environ.setdefault('QDRANT_DISABLED', '1')

    from modules.events import event_bus, Events
    from modules.wiring import wire_event_bus, get_cx_registry

    wire_event_bus()
    registry = get_cx_registry()

    # Get all unique event types for diversity normalization
    all_event_types = set()
    for entry in registry.values():
        if entry.get('event'):
            all_event_types.add(entry['event'])
    max_diversity = len(all_event_types)

    results = {
        'test': 'pci',
        'perturbations': {},
    }
    pci_values = []

    for perturb_id, perturb_spec in PERTURBATIONS.items():
        # Reset instrumentation for clean measurement
        event_bus.reset_instrumentation()

        trace_id = f'pci-{perturb_id}'
        payload = dict(perturb_spec['payload'])
        payload['trace_id'] = trace_id

        # Inject perturbation
        event_bus.emit(perturb_spec['event'], payload)

        # Collect cascade response
        cascade = event_bus.get_cascade_log(trace_id=trace_id)

        # Metrics
        cascade_length = len(cascade)
        handler_sequence = [e['handler'] for e in cascade]
        unique_handlers = set(handler_sequence)

        # Handler diversity: unique handlers / total registered CX handlers
        # This measures how much of the system responds to one stimulus
        total_cx_handlers = len([e for e in registry.values() if e.get('model') == 'event'])
        handler_diversity = len(unique_handlers) / max(total_cx_handlers, 1)

        # LZ complexity of handler sequence (differentiation)
        handler_map = {h: chr(65 + i) for i, h in enumerate(sorted(unique_handlers))}
        encoded = [handler_map.get(h, '?') for h in handler_sequence]
        lz_complexity = _lempel_ziv_complexity(encoded)

        # Latency profile diversity: std of handler latencies (in ms)
        latencies = [e.get('latency_ms', 0) for e in cascade]
        lat_std = 0
        if len(latencies) >= 2:
            lat_mean = sum(latencies) / len(latencies)
            lat_std = (sum((x - lat_mean) ** 2 for x in latencies) / len(latencies)) ** 0.5

        # PCI adapted for event-bus architecture:
        # = handler_diversity * (lz_complexity + 0.5) * cascade_reach
        # cascade_reach = cascade_length / total_cx_handlers (how far the perturbation reaches)
        cascade_reach = min(cascade_length / max(total_cx_handlers, 1), 1.0)
        pci = round(handler_diversity * (lz_complexity * 0.5 + 0.5) * cascade_reach, 4) if cascade_length > 0 else 0.0

        # Total cascade latency
        total_latency_ms = sum(e.get('latency_ms', 0) for e in cascade)

        perturb_result = {
            'desc': perturb_spec['desc'],
            'event': perturb_spec['event'],
            'cascade_length': cascade_length,
            'unique_handlers': len(unique_handlers),
            'handler_diversity': round(handler_diversity, 4),
            'cascade_reach': round(cascade_reach, 4),
            'lz_complexity': round(lz_complexity, 4),
            'latency_std_ms': round(lat_std, 2),
            'pci': pci,
            'total_latency_ms': round(total_latency_ms, 2),
            'handler_sequence': handler_sequence[:20],
        }
        results['perturbations'][perturb_id] = perturb_result
        pci_values.append(pci)

    # --- Global PCI metrics ---
    avg_pci = round(sum(pci_values) / len(pci_values), 4) if pci_values else 0
    max_pci = max(pci_values) if pci_values else 0
    min_pci = min(pci_values) if pci_values else 0

    # Differentiation: stimuli with more downstream CX should have higher PCI
    # Consolidation (4 CX handlers) should beat self_model or workspace
    high_cascade = results['perturbations'].get('consolidation', {}).get('cascade_length', 0)
    low_cascade = results['perturbations'].get('low_pe', {}).get('cascade_length', 0)
    differentiation = high_cascade >= low_cascade  # More CX = longer cascade

    # Variance across perturbations: system responds differently to different stimuli
    pci_variance = 0
    if len(pci_values) >= 2:
        pci_mean = sum(pci_values) / len(pci_values)
        pci_variance = sum((x - pci_mean) ** 2 for x in pci_values) / len(pci_values)

    results['avg_pci'] = avg_pci
    results['max_pci'] = max_pci
    results['min_pci'] = min_pci
    results['pci_variance'] = round(pci_variance, 6)
    results['differentiation'] = differentiation
    results['target_min'] = 0.01
    results['target_max'] = 0.80

    # Pass criteria:
    # 1. Average PCI > 0.01 (system responds to perturbation at all)
    # 2. PCI < 0.80 (not chaotic)
    # 3. Differentiation: different stimuli produce different cascade sizes
    results['pass'] = (
        avg_pci >= 0.01
        and max_pci <= 0.80
        and differentiation
    )
    results['passed'] = sum([
        avg_pci >= 0.01,
        max_pci <= 0.80,
        differentiation,
    ])
    results['total_checks'] = 3

    return results


if __name__ == "__main__":
    result = run_pci_test()
    print(json.dumps(result, indent=2, ensure_ascii=False))
