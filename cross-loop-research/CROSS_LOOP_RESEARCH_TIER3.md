# Cross-Loop Research Brief — TIER 3
> Generated: 2026-03-13 | For: Research Terminal (Deep Research Agent)
> Status: READY FOR RESEARCH

---

## Context

TIER 1 (CX-1 through CX-4b) and TIER 2 (CX-5 through CX-8) are **implemented and tested**:
- 9 cross-loops active, 158 papers with DOI
- 18/45 connections working (40% coverage)
- All 10 loops now have at least one cross-loop connection
- 41 cross-loop tests passing, 1709 total tests, 0 regressions

TIER 3 targets the next 6 most impactful missing connections to reach ~53% coverage.

---

## Research Targets (6 Cross-Loops)

### CX-9: L3→L9 — GNW Workspace → Self-Model Refresh

**What exists:**
- `competition.py` emits `WORKSPACE_COMPETITION_COMPLETE` with `winner_domains`, `top_activation`, `winner_count`
- `self_model.py` has `reflect_on_self()` (line 39) for self-model refresh
- `SELF_MODEL_REFRESHED` event already emitted when discrepancies detected
- CX-3 (L9→L3) already implemented: self-model pushes TO workspace with DMN gateway bonus

**What's missing:**
- Reverse direction: when workspace broadcasts self-referential content, it should trigger self-model UPDATE
- No detection of "self-referential" content in workspace winners
- No connection from competition results to self_model.py

**Research questions:**
1. When should workspace content trigger self-model update? (every broadcast? only self-referential?)
2. How to detect self-referential content in workspace winners?
3. What's the boundary between GNW informing self-model vs self-model informing GNW (CX-3)?
4. Does the Default Mode Network (DMN) literature support bidirectional self↔workspace flow?
5. Risk: feedback loop — self-model → GNW (CX-3) → self-model (CX-9) → GNW... How to break?

**Keywords:** Default Mode Network, self-referential processing, GNW access, autonoetic consciousness, self-awareness competition, metacognitive monitoring

---

### CX-10: L9↔L5 — Self-Model ↔ Metacognition

**What exists:**
- `self_model.py` emits `SELF_MODEL_REFRESHED` with `source`, `discrepancy_count`, `domains`
- `preturn_inject.py` has `_meta_prediction_cycle()` computing L2 meta-PE per domain
- L2 state in `prediction_state_l2` table: `predicted_accuracy`, `actual_accuracy`, `sample_size` per domain
- L2 already controls L0 precision (dampens overconfidence)
- CX-6 (L5→L7) uses meta-confidence for EFE temperature

**What's missing:**
- Self-model discrepancies should UPDATE metacognitive accuracy estimates
- When self-model detects it's wrong about X domains, L2 should lower predicted_accuracy for those domains
- Reverse: when metacognition detects systematic errors, self-model should be notified (triggers reconsolidation?)
- No current connection between self_model.py events and prediction_state_l2 updates

**Research questions:**
1. How does self-knowledge accuracy feed metacognitive confidence? (Dunning-Kruger, unskilled-unaware)
2. Should self-model discrepancies directly modify L2 predicted_accuracy, or go through an intermediate?
3. Is this the same as "metacognitive sensitivity" (Fleming & Lau 2014)?
4. What's the causal direction: does better self-model → better metacognition, or vice versa?
5. Risk: cascading confidence collapse — self-model detects error → lowers meta-confidence → CX-6 raises temperature → explores randomly → more errors → more self-model discrepancies

**Keywords:** metacognitive sensitivity, self-knowledge accuracy, Dunning-Kruger, meta-d prime, self-awareness metacognition, introspective accuracy, feeling of knowing

---

### CX-11: L6→L8 — Curiosity → Causal Discovery

**What exists:**
- `curiosity.py` has `push_curiosidad(tema, prioridad, categoria)` for generating questions
- `CURIOSITY_RESOLVED` event defined in events.py (line 65) but NOT YET consistently emitted
- `causal_discovery.py` has `run_causal_discovery()` → builds co-occurrence matrix → runs NOTEARS → extracts edges
- Co-occurrence matrix built from `attention_transitions` + `prediction_results` tables
- Causal discovery runs in sleep_loop tick independently

**What's missing:**
- Curiosity exploration should FEED new co-occurrence data to causal discovery
- When curiosity resolves a question about topic X, the exploration pattern should update attention_transitions
- No mechanism to prioritize causal discovery for curiosity-driven topics
- CURIOSITY_RESOLVED event needs consistent emission with topic/category data

**Research questions:**
1. How does exploration/curiosity improve causal model learning? (Bramley 2017 active learning)
2. Should curiosity-driven observations get HIGHER weight in co-occurrence matrix?
3. Is there a distinction between observational and interventional data for NOTEARS?
4. Should causal discovery run IMMEDIATELY after curiosity resolution, or wait for batch?
5. Can curiosity be DIRECTED by causal model gaps (missing edges, uncertain edges)?
6. Risk: curiosity-driven bias — exploring what's interesting may create sampling bias in DAG

**Keywords:** active causal learning, curiosity-driven exploration, interventional data, structure learning, directed exploration, information gain, causal structure

---

### CX-12: L7→L10 — Action Outcomes → Forgetting

**What exists:**
- `active_inference.py` has `select_action()` returning `(Action, {name: efe_value})`
- 6 primitive actions: retrieve, store, consolidate, forget, attend, explore
- `forgetting.py` has `compute_fadem_strength()` with `decay_multiplier` parameter (line 113)
- Power-law decay: `R(t) = (1 + α*t)^{-β}` with importance modulation
- `FADEM_LAMBDA_BASE = 0.008`, modulated by importance and decay_multiplier
- No `ACTION_OUTCOME` event exists yet

**What's missing:**
- No tracking of which topics are USED (accessed via actions) vs UNUSED
- No mechanism to increase decay for unused/irrelevant topics
- No action_outcome event to close the loop
- `decay_multiplier` parameter exists in compute_fadem_strength but nobody passes non-default values

**Research questions:**
1. How does action-based retrieval practice protect memories from forgetting? (testing effect, Roediger 2006)
2. Should unused topics decay FASTER or just not get protection? (active forgetting vs passive decay)
3. What's the relationship between action frequency and memory strength? (spacing effect)
4. How to compute "usage score" per topic from action history?
5. Should action-FAILURE (high PE) increase or decrease decay for that topic?
6. Risk: creating a "rich get richer" dynamic where used topics never decay and unused topics vanish

**Keywords:** testing effect, retrieval practice, spacing effect, desirable difficulties, use-dependent plasticity, action-based memory, active forgetting, directed forgetting

---

### CX-13: L4→L7 — Emotion (PAD) → Action Selection (EFE)

**What exists:**
- `emotion.py` provides PAD state: pleasure [-1,1], arousal [0,1], dominance [-1,1]
- AC→PAD blending: `_AC_PAD_BLEND = 0.6` (60% AC-derived, 40% text-inferred)
- `active_inference.py` EFE computation (line 505-507): `G = -pragmatic - epistemic + cost`
- `SystemState` already has `emotional_valence` (pleasure) and `pe_magnitude`
- EFE is currently emotion-agnostic — no PAD modulation

**What's missing:**
- PAD should modulate the pragmatic/epistemic BALANCE in EFE
- High arousal → more pragmatic (exploit known paths, reduce uncertainty exploration)
- Low arousal → more epistemic (explore, reduce uncertainty)
- Valence could modulate risk tolerance (positive → risk-seeking, negative → risk-averse)
- Dominance could modulate autonomy (high → independent action, low → conservative)

**Research questions:**
1. How does emotional arousal modulate explore/exploit tradeoff? (Aston-Jones & Cohen 2005, LC-NE)
2. Does mood affect decision-making quality or just strategy? (Isen 1987, mood-as-information)
3. Should PAD modulate EFE WEIGHTS (pragmatic vs epistemic) or TEMPERATURE (CX-6 already does temperature)?
4. Risk of double-counting with CX-6: meta-confidence already modulates temperature. PAD should modulate DIFFERENT parameter.
5. What's the neuroscience of emotion→action? (somatic marker hypothesis, Damasio 1994)
6. Risk: mood-congruent perseveration — negative mood → conservative actions → no improvement → stays negative

**Keywords:** somatic marker hypothesis, affect heuristic, mood-congruent judgment, arousal and decision-making, emotional regulation of action, LC-NE system, approach-avoidance motivation

---

### CX-14: L2→L6 — Consolidation Gaps → Curiosity

**What exists:**
- `consolidation.py` emits `CONSOLIDATION_COMPLETE` with rich result dict including:
  - `contradictions_found` — semantic conflicts detected
  - `facts_extracted`, `facts_created` — knowledge extracted
  - `clusters_found` — memory clusters
  - `consolidated_ids` — processed memory IDs
- `curiosity.py` has `push_curiosidad(tema, prioridad, categoria)`
- CX-1 already drives curiosity FROM prediction error
- CX-2 closes the loop: resolved curiosity boosts precision

**What's missing:**
- Consolidation does NOT report "gaps" — only counts and contradictions
- No mechanism to detect "this cluster is missing expected knowledge"
- No connection from consolidation results to curiosity generation
- Gap detection would need comparing cluster content against expected semantic structure

**Research questions:**
1. How does the brain detect knowledge gaps during consolidation? (offline replay, Diekelmann 2010)
2. Is gap detection a comparison against schema/expectations (schema theory, Bartlett 1932)?
3. Should contradiction detection (existing) also trigger curiosity?
4. How to define "expected knowledge" for a topic cluster? (semantic network completeness)
5. Can consolidation's graph analysis (bridge_edges, causal_chains) reveal structural gaps?
6. Risk: generating too many curiosity items from every consolidation run (noise → curiosity overload)

**Keywords:** schema-driven consolidation, knowledge gap detection, offline replay, curiosity from contradiction, memory completeness, schema violation, information gap theory (Loewenstein 1994)

---

## Agent Grouping (Efficiency-Optimized)

Based on TIER 2 improvement (26% fewer tokens via grouping):

### Research Agent 1: Self-Model + Metacognition (CX-9 + CX-10)
- Shared domain: self-referential processing, metacognitive monitoring, DMN
- Shared literature: Fleming, Dunning-Kruger, Nelson & Narens, autonoetic consciousness
- ~30 papers expected

### Research Agent 2: Curiosity + Causal + Consolidation (CX-11 + CX-14)
- Shared domain: knowledge gaps, exploration, learning-driven discovery
- Shared literature: Bramley, Loewenstein, active learning, information gain
- ~25 papers expected

### Research Agent 3: Action + Emotion + Forgetting (CX-12 + CX-13)
- Shared domain: action selection, emotional modulation, use-dependent plasticity
- Shared literature: Roediger, Damasio, Aston-Jones, spacing effect
- ~25 papers expected

### Verification Agent 4: Blind Spots + Counter-Evidence (ALL)
- Run SIMULTANEOUSLY with research agents
- Focus: feedback loops, double-counting risks, cascading failures
- Special attention to: CX-9/CX-3 feedback loop, CX-13/CX-6 double-counting, CX-12 rich-get-richer

### Verification Agent 5: Codebase Audit (ALL)
- Run SIMULTANEOUSLY with research agents
- Focus: feasibility scores, exact insertion points, blockers
- Special attention to: CURIOSITY_RESOLVED emission gap (CX-11), ACTION_OUTCOME event design (CX-12), gap detection in consolidation (CX-14)

---

## Deliverable Format

Same as TIER 1+2:

For EACH cross-loop:
1. **Papers** — Table with citation + DOI (minimum 3 per CX)
2. **Mechanism** — Neuroscience-backed description
3. **Evidence** — Key experimental findings
4. **Implementation Minima** — Minimal code (handler + registration + helper)
5. **Risks** — What can go wrong
6. **Blind Spots** — Counter-evidence, boundary conditions, alternatives
7. **Test Plan** — 3-4 test cases

Plus:
- Cross-cutting blind spots
- Combined feasibility audit
- Implementation order (easiest → hardest, safest → riskiest)
- TIER 1+2+3 combined status table

---

## Existing Architecture Reference

### Current Cross-Loop Status
| CX | Loop | Status | LOC |
|----|------|--------|-----|
| CX-1 | L4→L6 | IMPLEMENTED | ~25 |
| CX-2 | L6→L4 | IMPLEMENTED | ~30 |
| CX-3 | L9→L3 | IMPLEMENTED | ~35 |
| CX-4a | L10→L2 | IMPLEMENTED | ~20 |
| CX-4b | L2→L10 | IMPLEMENTED | ~25 |
| CX-5 | L3→L7 | IMPLEMENTED | ~30 |
| CX-6 | L5→L7 | IMPLEMENTED | ~45 |
| CX-7 | L8→L4 | IMPLEMENTED | ~20 |
| CX-8 | L1→L10 | IMPLEMENTED | ~25 |

### Key Design Constraints
1. **Event bus is per-process** — preturn_inject.py hooks share the server event bus
2. **SQLite for cross-process state** — prediction_state_l2, causal_discovery_state, etc.
3. **No new events without justification** — prefer reusing existing 23 events
4. **Weak coupling always** — new connections should be advisory, never blocking
5. **Guards on everything** — every handler needs boundary conditions and cooldowns
6. **Tests required** — minimum 3 tests per handler
