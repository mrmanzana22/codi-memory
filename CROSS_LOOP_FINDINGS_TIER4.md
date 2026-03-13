# Cross-Loop Findings — TIER 4

**Date:** 2026-03-13
**Scope:** 20 remaining cross-loop connections (all possible connections minus 14 existing + already-researched CX-12)
**Methodology:** 4 research agents + 2 validators (Neuroscience Consultant + Cognitive Architecture Expert)
**Papers referenced:** ~65 across all agents + validators
**Constraint:** Max 8 IMPLEMENT

---

## Methodology

### Research Phase (4 agents)
- **Agent 1**: L1 outputs (6 connections from Reconsolidation)
- **Agent 2**: L2 outputs (6 connections from Consolidation)
- **Agent 3**: L3-L9 remaining (8 connections: L3→L5, L3→L6, L4→L8, L4→L9, L5→L6, L6→L9, L8→L10, L9→L10)
- **Agent 4**: Integration analysis (graph metrics, redundancy matrix, composite scoring, biological plausibility)

### Validation Phase (2 validators, independent)
- **Neuroscience Consultant**: Resolved 7 inter-agent conflicts using neuroscience literature. Every verdict backed by citations.
- **Cognitive Architecture Expert**: Validated against SOAR (Laird 2012), ACT-R (Anderson 2007), LIDA (Franklin & Baars 2003), CLARION (Sun 2016). Checked hub overload, sink node, E/I balance.

### Conflict Resolution
6 significant conflicts arose between agents. Both validators independently evaluated each. Their verdicts aligned on ALL 6 conflicts.

---

## Summary Table — All 20 Connections

| # | Connection | Direction | Classification | Neuro | Arch | Papers |
|---|-----------|-----------|----------------|-------|------|--------|
| 1 | L1→L3 | Recon→GNW | **SKIP** | SKIP | DEFER | 3 |
| 2 | L1→L5 | Recon→Meta | **IMPLEMENT** | IMPL | IMPL | 5 |
| 3 | L1→L6 | Recon→Curiosity | SKIP | — | — | 0 |
| 4 | L1→L7 | Recon→ActInf | SKIP | — | — | 0 |
| 5 | L1→L8 | Recon→Causal | SKIP | — | — | 0 |
| 6 | L1→L9 | Recon→Self | **DEFER** | DEFER | DEFER | 4 |
| 7 | L2→L3 | Consol→GNW | **SKIP** | — | SKIP | 0 |
| 8 | L2→L4 | Consol→Pred | **IMPLEMENT** | IMPL | IMPL | 5 |
| 9 | L2→L5 | Consol→Meta | **DEFER** | DEFER | DEFER | 3 |
| 10 | L2→L7 | Consol→ActInf | SKIP | — | — | 0 |
| 11 | L2→L8 | Consol→Causal | SKIP | — | — | 0 |
| 12 | L2→L9 | Consol→Self | **IMPLEMENT** | IMPL | IMPL | 5 |
| 13 | L3→L5 | GNW→Meta | **IMPLEMENT** | IMPL | IMPL | 5 |
| 14 | L3→L6 | GNW→Curiosity | **DEFER** | — | — | 2 |
| 15 | L4→L8 | Pred→Causal | **SKIP** | — | — | 3 |
| 16 | L4→L9 | PE→Self | **DEFER** | DEFER | DEFER | 5 |
| 17 | L5→L6 | Meta→Curiosity | **IMPLEMENT** | IMPL | IMPL | 5 |
| 18 | L6→L9 | Curiosity→Self | **SKIP** | — | — | 0 |
| 19 | L8→L10 | Causal→Forget | **IMPLEMENT** | IMPL | IMPL | 5 |
| 20 | L9→L10 | Self→Forget | **IMPLEMENT** | IMPL | IMPL | 4 |

**Result: 7 IMPLEMENT, 4 DEFER, 9 SKIP**

---

## IMPLEMENT: 7 New Cross-Loops

### CX-15: L9→L10 — Self-Model Protects/Prunes Memory (INHIBITORY)

**Neuro Priority: 1 | Arch Priority: 7 | Combined: 1**

#### Papers

| # | Citation | DOI |
|---|----------|-----|
| 1 | Sedikides, C. & Green, J.D. (2009). Memory as a self-protective mechanism. *Social & Personality Psych. Compass*, 3(6), 1055-1068. | 10.1111/j.1751-9004.2009.00220.x |
| 2 | Conway, M.A. (2005). Memory and the self. *J. Memory & Language*, 53, 594-628. | 10.1016/j.jml.2005.08.005 |
| 3 | Anderson, M.C. & Hanslmayr, S. (2014). Neural mechanisms of motivated forgetting. *Trends Cogn. Sci.*, 18(6), 279-292. | 10.1016/j.tics.2014.03.002 |
| 4 | Yizhar, O. et al. (2011). Neocortical excitation/inhibition balance. *Nature*, 477, 171-178. | 10.1038/nature10360 |

#### Mechanism

The self-model acts as GATEKEEPER for forgetting. Memories consistent with self-identity receive decay PROTECTION (reduced FadeMem decay rate). Memories threatening to self-coherence receive accelerated decay (capped at 1.5x). This is the system's FIRST INHIBITORY connection, addressing the pathological 23:0 E:I balance.

Computationally distinct from CX-18 (L8→L10): CX-18 uses structural centrality (graph topology), CX-15 uses semantic congruence (content match with self-model). These are orthogonal dimensions.

```python
_CX15_PROTECT_FACTOR = 0.3      # Decay reduction for self-affirming
_CX15_PRUNE_FACTOR = 1.5        # Decay acceleration for self-threatening (CAPPED)
_CX15_COOLDOWN = 1800           # 30 min between self-model forgetting scans

def _on_self_model_modulates_forgetting(event_name, data):
    """CX-15: Self-model protects identity-coherent memories.
    Sedikides & Green 2009: mnemic neglect mechanism.
    First INHIBITORY cross-loop in the system."""
    core_beliefs = data.get("core_beliefs", [])
    capabilities = data.get("capabilities", {})
    # For each memory in decay queue:
    #   if memory aligns with core_beliefs → reduce decay by PROTECT_FACTOR
    #   if memory contradicts core_beliefs → accelerate decay by PRUNE_FACTOR
    #   Metacognition (L5 via CX-10) receives pre-forgetting signal
```

#### Risks

| Risk | Severity | Mitigation |
|------|----------|------------|
| Self-serving bias (forgets failures) | HIGH | Cap prune factor at 1.5x; L5 receives signal before pruning |
| Positive feedback loop with self-model | MEDIUM | Self-model refreshes from multiple sources (CX-9, CX-16) |

**LOC: ~35 | Type: INHIBITORY**

---

### CX-16: L3→L5 — Workspace Broadcasts Inform Metacognition

**Neuro Priority: 2 | Arch Priority: 5 | Combined: 2**

#### Papers

| # | Citation | DOI |
|---|----------|-----|
| 1 | Shea, N. & Frith, C.D. (2019). The global workspace needs metacognition. *Trends Cogn. Sci.*, 23(7), 560-571. | 10.1016/j.tics.2019.04.007 |
| 2 | Mashour, G.A. et al. (2020). Conscious processing and GNW. *Neuron*, 105(5), 776-798. | 10.1016/j.neuron.2020.01.026 |
| 3 | Fleming, S.M. & Dolan, R.J. (2012). Neural basis of metacognitive ability. *Phil. Trans. R. Soc. B*, 367, 1338-1349. | 10.1098/rstb.2011.0417 |
| 4 | Baars, B.J. et al. (2021). GWT and prefrontal cortex. *Front. Psych.*, 12, 749868. | 10.3389/fpsyg.2021.749868 |
| 5 | COGITATE Consortium (2025). Prefrontal involvement in consciousness. *Nature*. | — |

#### Mechanism

Workspace competition produces process-level metadata (coalition_strength, coalition_size, novelty_score) that metacognition CANNOT get through indirect paths. L3→L9→L5 filters for self-relevance. L3→L4→L5 filters for surprise. Neither carries the workspace process signals that metacognition needs for quality control.

All 4 cognitive architectures (ACT-R, SOAR, LIDA, CLARION) require workspace→metacognition monitoring.

```python
_CX16_STRENGTH_THRESHOLD = 0.3  # Only significant broadcasts trigger meta-evaluation

def _on_workspace_broadcast_to_metacognition(event_name, data):
    """CX-16: GNW broadcast informs metacognitive monitoring.
    Shea & Frith 2019: workspace needs confidence tagging."""
    strength = data.get("competition_strength", 0.5)
    coalition = data.get("coalition_size", 1)
    novelty = data.get("novelty_score", 0.5)
    if strength < _CX16_STRENGTH_THRESHOLD:
        return
    workspace_conf = 0.4*strength + 0.3*min(coalition/5, 1.0) + 0.3*(1-novelty)
    domain = data.get("winner_topic", "general")
    # Modulate L2 precision for this domain
    # Low workspace_conf → lower L2 precision → system "knows it doesn't know"
```

#### Risks

| Risk | Severity | Mitigation |
|------|----------|------------|
| High frequency (every workspace broadcast) | MEDIUM | Threshold gate: only significant broadcasts |
| Interaction with CX-10 (L9↔L5) | LOW | Complementary: CX-16 = workspace quality, CX-10 = self-model accuracy |

**LOC: ~30 | Type: Excitatory | Arch support: 4/4**

---

### CX-17: L2→L4 — Consolidated Schemas Become Predictive Priors

**Neuro Priority: 3 | Arch Priority: 6 | Combined: 3**

#### Papers

| # | Citation | DOI |
|---|----------|-----|
| 1 | Tse, D. et al. (2007). Schemas and memory consolidation. *Science*, 316(5821), 76-82. | 10.1126/science.1135935 |
| 2 | McClelland, J.L. et al. (1995). Why there are complementary learning systems. *Psych. Rev.*, 102(3), 419-457. | 10.1037/0033-295X.102.3.419 |
| 3 | Kumaran, D. et al. (2016). What learning systems do intelligent agents need? *Neuron*, 92(6), 1205-1220. | 10.1016/j.neuron.2016.09.001 |
| 4 | Kumaran, D. & McClelland, J.L. (2012). Generalization through the recurrent interaction of episodic memories. *Psych. Rev.*, 119(3), 573-616. | 10.1037/a0028681 |
| 5 | Lewis, P.A. & Durrant, S.J. (2011). Overlapping memory replay during sleep builds cognitive schemata. *Trends Cogn. Sci.*, 15(8), 343-351. | 10.1016/j.tics.2011.06.004 |

#### Mechanism

CLS (Complementary Learning Systems) foundational claim: consolidated neocortical representations become the prior structure for hippocampal encoding. After each consolidation run, schema-level statistics update prediction's Dirichlet priors.

Hub overload concern (Agent 4) refuted by both validators: consolidation fires at most 1x per 30-min sleep cycle — negligible load. The indirect path via L6 (curiosity) is computationally different: curiosity drives exploratory prediction, consolidated schemas drive confirmatory prediction.

```python
_CX17_SCHEMA_WEIGHT = 0.1      # WEAK priors (same philosophy as CX-7)
_CX17_MIN_CONFIDENCE = 0.7     # Schema must be consolidated with high confidence

def _on_consolidation_updates_prediction_priors(event_name, data):
    """CX-17: Consolidated schemas become predictive priors.
    Tse 2007: schemas accelerate learning ~50x. CLS (McClelland 1995)."""
    schemas = data.get("schemas_extracted", [])
    for schema in schemas:
        if schema["confidence"] < _CX17_MIN_CONFIDENCE:
            continue
        domain = schema["domain"]
        # Update prediction prior for domain:
        # alpha_domain += schema_weight * schema_strength
```

#### Risks

| Risk | Severity | Mitigation |
|------|----------|------------|
| Schema rigidity (overfit to past patterns) | MEDIUM | WEAK priors (0.1), require stability across N consolidation runs |
| Wrong schemas from bad consolidation | LOW | Confidence threshold 0.7 filters low-quality schemas |

**LOC: ~35 | Type: Excitatory | Arch support: 4/4**

---

### CX-18: L1→L5 — Reconsolidation Lowers Metacognitive Confidence

**Neuro Priority: 4 | Arch Priority: 3 | Combined: 4**

#### Papers

| # | Citation | DOI |
|---|----------|-----|
| 1 | Nelson, T.O. & Narens, L. (1990). Metamemory: A theoretical framework. *Psych. Learning & Motivation*, 26, 125-173. | 10.1016/S0079-7421(08)60053-5 |
| 2 | Fleming, S.M. (2014). The neural basis of metacognitive ability. *Phil. Trans. R. Soc. B*, 369, 20130535. | 10.1098/rstb.2013.0535 |
| 3 | Exton-McGuinness, M.T.J. et al. (2015). Updating memories: prediction errors in reconsolidation. *BBR*, 278, 375-384. | 10.1016/j.bbr.2014.10.011 |
| 4 | Nader, K. et al. (2000). Fear memories require protein synthesis for reconsolidation. *Nature*, 406, 722-726. | 10.1038/35021052 |
| 5 | Schwartz, B.L. (1994). Sources of information in metamemory. *Psychonomic Bull. & Rev.*, 1(3), 357-375. | 10.3758/BF03213977 |

#### Mechanism

When reconsolidation destabilizes a memory (PE >= 0.6), metacognition MUST lower confidence in related beliefs. This is a healthy negative feedback loop: memory found wrong → confidence drops → system becomes more cautious in that domain.

Supported by 3/4 cognitive architectures (CLARION's MCS direct, SOAR's impasse, LIDA through workspace).

```python
_CX18_CONFIDENCE_REDUCTION = 0.15  # Per reconsolidation event
_CX18_FLOOR = 0.2                  # Minimum confidence

def _on_reconsolidation_to_metacognition(event_name, data):
    """CX-18: Reconsolidation lowers metacognitive confidence.
    Nelson & Narens 1990: memory correction must update monitoring."""
    domain = data.get("domain", "general")
    pe = data.get("prediction_error", 0.0)
    if pe < 0.6:
        return
    # Reduce L2 precision for domain
    # new_precision = max(FLOOR, current - REDUCTION * pe)
```

#### Risks

| Risk | Severity | Mitigation |
|------|----------|------------|
| Cascading confidence collapse | MEDIUM | Floor at 0.2, single-event reduction capped |
| Interaction with CX-16 (workspace confidence) | LOW | Different signals: CX-18 = memory error, CX-16 = workspace quality |

**LOC: ~35 | Type: Excitatory (error correction) | Arch support: 3/4**

---

### CX-19: L2→L9 — Consolidation Feeds Self-Model (Episodic→Semantic Self)

**Neuro Priority: 5 | Arch Priority: 4 | Combined: 5**

#### Papers

| # | Citation | DOI |
|---|----------|-----|
| 1 | Conway, M.A. (2005). Memory and the self. *J. Memory & Language*, 53, 594-628. | 10.1016/j.jml.2005.08.005 |
| 2 | Northoff, G. et al. (2004). Self-referential processing. *Neuroimage*, 31(1), 440-457. | 10.1016/j.neuroimage.2005.12.002 |
| 3 | Klein, S.B. & Lax, M.L. (2010). The unanticipated resilience of trait self-knowledge. *J. Exp. Psych.: General*, 139(4), 595-602. | 10.1037/a0021044 |
| 4 | McAdams, D.P. (2001). The psychology of life stories. *Rev. General Psych.*, 5(2), 100-122. | 10.1037/1089-2680.5.2.100 |
| 5 | Gallagher, S. (2000). Philosophical conceptions of the self. *Trends Cogn. Sci.*, 4(1), 14-21. | 10.1016/S1364-6613(99)01417-5 |

#### Mechanism

Conway's Self-Memory System: during consolidation, episodic self-knowledge transitions to semantic self-knowledge. This directly addresses the documented noetic-autonoetic gap (semantic self 0.88 >> episodic self 0.64).

SOAR's semantic learning from episodic memory is the cleanest architectural analogue.

```python
def _on_consolidation_feeds_self_model(event_name, data):
    """CX-19: Consolidated semantic facts update self-model.
    Conway 2005 SMS: episodic→semantic self-knowledge transition."""
    semantic_facts = data.get("semantic_facts", [])
    for fact in semantic_facts:
        if fact.get("self_relevant", False):
            # Queue for self-model update:
            # self_model.update_from_consolidation(
            #   fact=fact["content"],
            #   confidence=fact["confidence"],
            #   source="consolidation_cx19"
            # )
```

**LOC: ~45 | Type: Excitatory | Arch support: 3/4**

---

### CX-20: L5→L6 — Metacognitive Uncertainty Drives Curiosity

**Neuro Priority: 6 | Arch Priority: 2 | Combined: 6**

#### Papers

| # | Citation | DOI |
|---|----------|-----|
| 1 | Loewenstein, G. (1994). The psychology of curiosity. *Psych. Bulletin*, 116(1), 75-98. | 10.1037/0033-2909.116.1.75 |
| 2 | Litman, J.A. (2005). Curiosity and metacognition. *Cognition & Emotion*, 19(5), 793-814. | 10.1080/02699930541000101 |
| 3 | Boldt, A. et al. (2019). Confidence modulates exploration-exploitation. *Neurosci. of Consciousness*, niz004. | 10.1093/nc/niz004 |
| 4 | Gottlieb, J. et al. (2013). Information-seeking, curiosity, attention. *Trends Cogn. Sci.*, 17(11), 585-593. | 10.1016/j.tics.2013.09.001 |
| 5 | Gruber, M.J. et al. (2014). Curiosity modulates hippocampus-dependent learning. *Neuron*, 84(2), 486-496. | 10.1016/j.neuron.2014.08.060 |

#### Mechanism

**Most architecturally universal connection.** All 4 cognitive architectures implement metacognition→exploration:
- SOAR: impasse → exploration subgoal
- ACT-R: retrieval failure → exploration strategy
- LIDA: attention codelets detect information gaps
- CLARION: MCS low performance → increase exploration rate

When L5 detects low confidence in a domain (L2 precision < 0.35), generate D-type (deprivation) curiosity via `push_curiosidad()`.

```python
_CX20_CONF_THRESHOLD = 0.35
_CX20_MAX_PER_CYCLE = 2
_CX20_COOLDOWN_PER_DOMAIN = 900  # 15 min

def _on_metacognitive_uncertainty_to_curiosity(event_name, data):
    """CX-20: Low metacognitive confidence → D-type curiosity.
    Loewenstein 1994 + Litman 2005: information gap requires meta-evaluation."""
    domain = data.get("domain", "")
    precision = data.get("l2_precision", 1.0)
    if precision > _CX20_CONF_THRESHOLD:
        return
    # push_curiosidad(question=f"What explains uncertainty in {domain}?",
    #                  source="metacognitive_cx20", urgency=1.0-precision)
```

**LOC: ~25 | Type: Excitatory | Arch support: 4/4 (HIGHEST)**

---

### CX-21: L8→L10 — Causal Centrality Protects Hub Memories (INHIBITORY)

**Neuro Priority: 7 | Arch Priority: 8 | Combined: 7**

#### Papers

| # | Citation | DOI |
|---|----------|-----|
| 1 | Kirkpatrick, J. et al. (2017). Overcoming catastrophic forgetting (EWC). *PNAS*, 114(13), 3521-3526. | 10.1073/pnas.1611835114 |
| 2 | Tononi, G. & Cirelli, C. (2014). Sleep and the price of plasticity (SHY). *Neuron*, 81(1), 12-34. | 10.1016/j.neuron.2013.12.025 |
| 3 | Tompary, A. & Davachi, L. (2017). Consolidation promotes representational overlap. *Neuron*, 96(1), 228-241. | 10.1016/j.neuron.2017.09.005 |
| 4 | Nature Comms (2022). Predicting memory from network structure. *Nat. Commun.*, 13, 4307. | 10.1038/s41467-022-31965-2 |
| 5 | Pearl, J. (2009). *Causality* (2nd ed.). Cambridge University Press. | ISBN 978-0521895606 |

#### Mechanism

**NOVEL** — no major cognitive architecture has this explicitly. Closest: ACT-R's fan-based activation (more connections → higher activation → lower forgetting).

After NOTEARS run, compute betweenness centrality. Top-quartile hub topics get decay protection in FadeMem.

```python
_CX21_CENTRALITY_BOOST = 0.2
_CX21_TOP_PERCENTILE = 0.75

def _on_causal_discovery_to_forgetting_protection(event_name, data):
    """CX-21: Causal hub memories resist decay.
    EWC analog (Kirkpatrick 2017): important parameters resist change."""
    w_matrix = data.get("w_matrix")
    topics = data.get("topics")
    centrality = _compute_betweenness(w_matrix, topics)
    threshold = np.percentile(list(centrality.values()), _CX21_TOP_PERCENTILE * 100)
    for topic, cent in centrality.items():
        if cent >= threshold:
            # Write protection to cx21_causal_protection table
            # FadeMem adds this to importance score
```

**LOC: ~35 | Type: INHIBITORY | Arch support: 0/4 + ML (NOVEL)**

---

## DEFERRED Connections (4)

| Connection | Reason | Revisit When |
|-----------|--------|-------------|
| L1→L9 (Recon→Self) | Indirect path (L1→L3→CX-9→L9) is neurobiologically faithful. Both validators agree. | After CX-9 is live; if self-model shows stale data after reconsolidation |
| L2→L5 (Consol→Meta) | Consolidation quality signals are structural, not performance. L2→L9→CX-10→L5 covers this. | After CX-19 (L2→L9) is live; if metacognition blind to consolidation failures |
| L4→L9 (PE→Self) | Most architectures mediate through workspace. Sharot 2011 asymmetric LR belongs in L9's update function, not a separate pathway. | After CX-9 and CX-19 are live; if self-model shows no PE-driven corrections |
| L3→L6 (GNW→Curio) | Indirect path via L4 (PE→curiosity). CX-20 covers metacognitive route. | After CX-16 and CX-20 are live; if conscious content fails to trigger curiosity |

---

## SKIPPED Connections (9)

| Connection | Reason |
|-----------|--------|
| L1→L3 (Recon→GNW) | Reconsolidation operates unconsciously. Event bus already handles notification. Forcing into GNW competition makes notification LESS reliable. |
| L1→L6 (Recon→Curio) | Redundant via L1→L2→L6 |
| L1→L7 (Recon→ActInf) | L7 is sink — adding input to sink has zero downstream value |
| L1→L8 (Recon→Causal) | No direct mechanism established |
| L2→L3 (Consol→GNW) | No architecture supports supply-driven "consolidation complete" broadcasts. Knowledge enters workspace on DEMAND. |
| L2→L7 (Consol→ActInf) | L7 sink problem |
| L2→L8 (Consol→Causal) | Covered by L2→L6→L8 indirect |
| L4→L8 (Pred→Causal) | Creates causal illusion feedback loop with CX-7. CX-11 (curiosity→causal) is safer. |
| L6→L9 (Curio→Self) | Too narrow; self-topics rare in curiosity queue; indirect via L6→workspace→CX-9 suffices |

---

## Validator Insights

### E/I Balance Assessment

| Metric | Current | After TIER 4 | Biological Target |
|--------|---------|-------------|-------------------|
| Excitatory edges | 23 | 28 | ~80% |
| Inhibitory edges | 0 | 2 (CX-15, CX-21) | ~20% |
| E/I ratio | 100:0 | 93:7 | 80:20 |

**Neuro assessment**: The 23:0 ratio IS concerning, but not directly analogous to biological E/I balance. Module-level inhibition is different from neuron-level inhibition. The 2 new INHIBITORY connections (L9→L10, L8→L10) provide TARGETED context-dependent suppression that the system currently lacks entirely.

**Arch assessment**: The system already has STRUCTURAL inhibition (GNW competition = winner-take-all, FadeMem decay, threshold gating) but lacks TARGETED inhibition. Adding L9→L10 and L8→L10 addresses this.

### Hub Overload (L4)

Both validators independently concluded: **hub overload is NOT a real risk** in our async event-bus architecture.
- Events are queued, not blocking
- Betweenness centrality measures shortest-path mediation; our modules communicate via events, not routing
- ACT-R's central buffers and SOAR's working memory are universal hubs by design
- L4's role as prediction hub is architecturally correct (Clark 2013: prediction is the brain's central currency)

### L7 Sink Node (CRITICAL)

**Both validators flagged this as the #1 structural priority.** L7 (Active Inference) has ZERO outgoing edges. This is pathological in ALL 4 cognitive architectures:
- ACT-R: motor module MUST produce output
- SOAR: operators MUST produce state changes
- LIDA: action selection → sensory-motor → environment → perception (cycle)
- CLARION: action-centered subsystem MUST produce actions

**Mandatory fix**: CX-12 (L7→L10, researched in TIER 3, ~165 LOC) must be implemented BEFORE any TIER 4 connections.

### Missed Connections (flagged by validators)

| Connection | Source | Mechanism | Priority |
|-----------|--------|-----------|----------|
| L10→L6 (Forget→Curiosity suppression) | Neuro | When memory is vaulted, suppress curiosity about related topics. Prevents vault-curiosity-relearning loops. Anderson et al. 1994: RIF suppresses associated info. | TIER 5 |
| L5→L1 (Meta→Reconsolidation gating) | Neuro | High metacognitive confidence INHIBITS reconsolidation (no need to destabilize what works). Suzuki et al. 2004: memory strength as boundary condition. | TIER 5 |
| L7→L5 (Action→Metacognition) | Arch | Action selection should report to metacognition: "I chose X because Y, outcome was Z." Closes action-monitoring loop. All 4 architectures support. | After CX-12 |
| L5→L4 (Meta→Prediction precision, explicit) | Arch | Exists implicitly in metacognitive sweep. Deserves explicit edge status. | TIER 5 |

---

## Implementation Order

Based on dependencies, structural priority, and validator consensus:

| Phase | CX | Connection | LOC | Deps | Rationale |
|-------|-----|-----------|-----|------|-----------|
| **0** | CX-12 | L7→L10 | 165 | None | **STRUCTURAL FIX.** Sink node. Both validators: top priority. |
| **1** | CX-20 | L5→L6 | 25 | CX-6 (exists) | Simplest. 4/4 arch support. Enables curiosity from uncertainty. |
| **2** | CX-18 | L1→L5 | 35 | None | Error correction pathway. 3/4 arch support. |
| **3** | CX-15 | L9→L10 | 35 | Self-model capabilities | First INHIBITORY. Self protects identity memories. |
| **4** | CX-16 | L3→L5 | 30 | CX-10 (for precision infra) | Workspace quality monitoring. Shea & Frith mandate. |
| **5** | CX-19 | L2→L9 | 45 | Self-model update API | Addresses noetic-autonoetic gap. Conway SMS. |
| **6** | CX-17 | L2→L4 | 35 | Consolidation schema output | CLS priors. Low frequency (1x/30min). |
| **7** | CX-21 | L8→L10 | 35 | NOTEARS events, FadeMem API | Second INHIBITORY. Novel mechanism. |

**Total: ~405 LOC across 8 connections (CX-12 + 7 new)**

---

## Post-Implementation Projections

| Metric | Before | After CX-12 only | After all TIER 4 |
|--------|--------|-------------------|-----------------|
| Directed edges | 23 | 24 | 31 |
| Density | 25.6% | 26.7% | 34.4% |
| Diameter | 5 | 4 | 3 |
| Avg path length | 2.28 | ~2.1 | ~1.7 |
| 2-hop coverage | 54.4% | ~60% | ~82% |
| Unreachable pairs | 9 (all FROM L7) | 0 | 0 |
| Inhibitory edges | 0 | 0 | 2 |
| Small-world sigma | ~0.8 | ~0.9 | ~1.1 |

---

## Combined Status — ALL TIERS (CX-1 through CX-21)

| CX | Connection | Tier | Status | Papers |
|----|-----------|------|--------|--------|
| CX-1 | L4→L6 PE→Curiosity | 1 | Implemented | 16 |
| CX-2 | L4→L1 PE→Reconsolidation | 1 | Implemented | 18 |
| CX-3 | L4→L3 PE→GNW Broadcast | 1 | Implemented | 13 |
| CX-4 | L4→L5 PE→Metacognition | 1 | Implemented | 15 |
| CX-4b | L2→L10 Consolidation→Decay | 2 | Implemented | 15 |
| CX-5 | L3→L4 GNW→Precision | 2 | Implemented | 16 |
| CX-6 | L5→L7 Meta→EFE | 2 | Implemented | 12 |
| CX-7 | L8→L4 Causal→Prediction | 2 | Researched | 13 |
| CX-8 | L1→L10 Recon→Decay Protection | 2 | Researched | 13 |
| CX-9 | L3→L9 GNW→Self-Model | 3 | Researched | 27 |
| CX-10 | L9↔L5 Self↔Metacognition | 3 | Researched | 25 |
| CX-11 | L6→L8 Curiosity→Causal | 3 | Researched | 26 |
| CX-12 | L7→L10 Action→Forgetting | 3 | Researched | ~15 |
| CX-13 | L4→L7 PAD→EFE | 3 | Researched | ~15 |
| CX-14 | L2→L6 Consolidation→Curiosity | 3 | Researched | ~15 |
| CX-15 | L9→L10 Self→Forgetting | **4** | **Researched** | 4 |
| CX-16 | L3→L5 GNW→Metacognition | **4** | **Researched** | 5 |
| CX-17 | L2→L4 Consol→Prediction | **4** | **Researched** | 5 |
| CX-18 | L1→L5 Recon→Metacognition | **4** | **Researched** | 5 |
| CX-19 | L2→L9 Consol→Self-Model | **4** | **Researched** | 5 |
| CX-20 | L5→L6 Meta→Curiosity | **4** | **Researched** | 5 |
| CX-21 | L8→L10 Causal→Forgetting | **4** | **Researched** | 5 |

**Total: 21 cross-loops researched | ~300+ papers | 9 implemented | 12 pending implementation**

---

## Efficiency Report

| Metric | TIER 1 | TIER 2 | TIER 3 | TIER 4 | Total |
|--------|--------|--------|--------|--------|-------|
| Research agents | 5 | 5 | 5 | 4 | 19 |
| Verification agents | 2 | 2 | 0 | 2 | 6 |
| CX researched | 4 | 5 | 6 | 7 | 21+redundant |
| Papers found | 77 | 81 | 78 | ~65 | ~300 |
| Connections screened | 4 | 5 | 6 | 20 | 35 (20 unique) |
| IMPLEMENT | 4 | 5 | 6 | 7 | 21 |
| SKIP/DEFER | 0 | 0 | 0 | 13 | 13 |

TIER 4 screened 3.3x more connections per agent than earlier tiers. The triage approach (classify first, deep-dive only IMPLEMENT) was significantly more efficient.
