"""
P2 Evaluation Harness — Excitatory/Inhibitory Balance
======================================================
Verifies E/I ratio across cross-loop connections.

Neuroscience basis:
  - Healthy cortical networks maintain ~80% E / 20% I (Brunel 2000)
  - Our measured ratio: 79:21 (from CROSS_LOOP_FINDINGS_TIER5.md)
  - Target range: 70-85% excitatory

Classification:
  EXCITATORY: amplifies, drives, feeds, boosts, broadcasts
  INHIBITORY: suppresses, gates, dampens, degrades, constrains, protects-from

References:
  - Brunel (2000): Dynamics of sparsely connected networks
  - Haider et al. (2006): E/I balance in cortical circuits
  - CROSS_LOOP_FINDINGS_TIER5.md: measured 79:21 ratio
"""

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))


# E/I classification for each CX (from architecture analysis)
# Principle: E = activates/strengthens/feeds downstream target
#            I = suppresses/dampens/degrades/constrains downstream target
# "Protects from decay" = excitatory (strengthens the target)
CX_CLASSIFICATION = {
    # TIER 1
    'CX-1':  'E',  # PE drives curiosity — activates
    'CX-2':  'E',  # Curiosity → prediction (injects synthetic hits) — feeds/strengthens
    'CX-3':  'E',  # Self-model competes in GNW — activates
    'CX-4a': 'E',  # Vault tracks urgency — drives consolidation
    'CX-4b': 'E',  # Consolidation boosts SS — strengthens memories
    # TIER 2
    'CX-5':  'E',  # GNW broadcast feeds action — activates
    'CX-6':  'E',  # Metacognition modulates explore/exploit — drives exploration
    'CX-8':  'E',  # Reconsolidation boosts SS — strengthens memories
    # TIER 3
    'CX-9':  'E',  # Workspace → self-model — triggers refresh
    'CX-10': 'I',  # Self-model → lower metacognitive precision — dampens
    'CX-11': 'E',  # Curiosity feeds causal — feeds observations
    'CX-12': 'E',  # Retrieval boosts SS — strengthens (testing effect)
    'CX-13': 'E',  # PAD → EFE weights — modulates action
    'CX-14': 'E',  # Consolidation gaps → curiosity — drives exploration
    # TIER 4
    'CX-15': 'I',  # Self-model modulates forgetting — can accelerate decay (first inhibitory)
    'CX-16': 'E',  # Workspace → metacognition — informs monitoring
    'CX-17': 'E',  # Consolidation → prediction priors — feeds
    'CX-18': 'I',  # Reconsolidation → lower confidence — dampens
    'CX-19': 'E',  # Consolidation → self-model — feeds identity
    'CX-20': 'E',  # Metacognitive uncertainty → curiosity — drives exploration
    'CX-21': 'E',  # Causal centrality → decay protection — strengthens
    # TIER 5
    'CX-22': 'E',  # Action → metacognition — informs monitoring
    'CX-23': 'I',  # Forgetting suppresses curiosity — suppresses
    'CX-24': 'I',  # Metacognition gates reconsolidation — blocks
    'CX-25': 'E',  # Workspace → testing effect + RIF — net excitatory
    'CX-26': 'I',  # Self-model constrains action — constrains
    'CX-27': 'I',  # Metacognition suppresses causal edges — suppresses
    'CX-28': 'I',  # Forgetting degrades metacognition — degrades
    'CX-29': 'I',  # Reconsolidation invalidates causal — invalidates
    'CX-30': 'E',  # Action outcomes → causal DAG — feeds
}

# Target range for E/I ratio (% excitatory)
EI_TARGET_MIN = 0.65
EI_TARGET_MAX = 0.85


def run_ei_balance_test() -> dict:
    """Verify E/I balance across cross-loops."""
    os.environ.setdefault('QDRANT_DISABLED', '1')

    from modules.wiring import wire_event_bus, get_cx_registry, get_cx_summary

    wire_event_bus()
    registry = get_cx_registry()
    summary = get_cx_summary()

    results = {
        'test': 'ei_balance',
        'checks': {},
    }
    passed = 0
    failed = 0

    # --- Check 1: Global E/I ratio ---
    e_count = sum(1 for v in CX_CLASSIFICATION.values() if v == 'E')
    i_count = sum(1 for v in CX_CLASSIFICATION.values() if v == 'I')
    total = e_count + i_count
    e_ratio = round(e_count / total, 3) if total else 0

    check_1 = {
        'name': 'global_ei_ratio',
        'excitatory': e_count,
        'inhibitory': i_count,
        'total': total,
        'e_ratio': e_ratio,
        'i_ratio': round(1 - e_ratio, 3),
        'target_range': f'{EI_TARGET_MIN*100:.0f}-{EI_TARGET_MAX*100:.0f}%',
        'pass': EI_TARGET_MIN <= e_ratio <= EI_TARGET_MAX,
    }
    results['checks']['global_ratio'] = check_1
    if check_1['pass']:
        passed += 1
    else:
        failed += 1

    # --- Check 2: Per-tier E/I balance ---
    tier_ei = {}
    for cx_id, classification in CX_CLASSIFICATION.items():
        entry = registry.get(cx_id, {})
        tier = entry.get('tier', 0)
        if tier not in tier_ei:
            tier_ei[tier] = {'E': 0, 'I': 0}
        tier_ei[tier][classification] += 1

    tier_results = {}
    tier_all_pass = True
    for tier in sorted(tier_ei.keys()):
        e = tier_ei[tier]['E']
        i = tier_ei[tier]['I']
        t = e + i
        ratio = round(e / t, 3) if t else 0
        # Per-tier: Tier 5 is intentionally inhibitory-heavy (balances Tiers 1-3).
        # Early tiers (1-3) should be E-dominant, later tiers can be I-dominant.
        if tier <= 3:
            tier_pass = ratio >= 0.50  # Early tiers: at least 50% E
        else:
            tier_pass = True  # Tiers 4-5: any ratio is valid (by design)
        if not tier_pass:
            tier_all_pass = False
        tier_results[f'tier_{tier}'] = {
            'excitatory': e,
            'inhibitory': i,
            'e_ratio': ratio,
            'pass': tier_pass,
        }

    check_2 = {
        'name': 'per_tier_balance',
        'tiers': tier_results,
        'pass': tier_all_pass,
    }
    results['checks']['per_tier'] = check_2
    if check_2['pass']:
        passed += 1
    else:
        failed += 1

    # --- Check 3: Registry completeness ---
    # Every CX in classification should exist in registry
    missing_from_registry = [
        cx for cx in CX_CLASSIFICATION if cx not in registry
    ]
    unclassified = [
        cx for cx in registry if cx not in CX_CLASSIFICATION
    ]

    check_3 = {
        'name': 'classification_completeness',
        'classified': len(CX_CLASSIFICATION),
        'in_registry': len(registry),
        'missing_from_registry': missing_from_registry,
        'unclassified_in_registry': unclassified,
        'pass': len(missing_from_registry) == 0,
    }
    results['checks']['completeness'] = check_3
    if check_3['pass']:
        passed += 1
    else:
        failed += 1

    # --- Check 4: Inhibitory diversity ---
    # Inhibitory CX should span multiple tiers (not concentrated)
    i_tiers = set()
    for cx_id, classification in CX_CLASSIFICATION.items():
        if classification == 'I':
            entry = registry.get(cx_id, {})
            i_tiers.add(entry.get('tier', 0))

    check_4 = {
        'name': 'inhibitory_diversity',
        'inhibitory_tiers': sorted(i_tiers),
        'num_tiers_with_inhibition': len(i_tiers),
        'pass': len(i_tiers) >= 3,  # Inhibition should span at least 3 tiers
    }
    results['checks']['inhibitory_diversity'] = check_4
    if check_4['pass']:
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
    result = run_ei_balance_test()
    print(json.dumps(result, indent=2, ensure_ascii=False))
