# Cross-Loop Findings — TIER 3
> Generated: 2026-03-13 | Agents: 5 (3 research + blind spots + codebase audit)
> Papers: ~78 | Combined TIER 1+2+3: ~236 papers
> Coverage: 18/45 active (40%) → target 24/45 (53%) after TIER 3

---

## Executive Summary

TIER 3 researched 6 cross-loops (CX-9 through CX-14) using 5 parallel agents. Key findings:

1. **All 6 proposals are theoretically sound** — neuroscience supports every connection
2. **All 6 are architecturally under-constrained** — every proposal lacks inhibitory mechanisms
3. **CX-9 is the most dangerous** — creates rumination loop with CX-3, needs mandatory salience gate
4. **CX-14 is the safest** — genuinely distinct from CX-1 (PACE framework), easiest to safeguard
5. **Bug discovered**: `causal_discovery.py:97` queries non-existent `count` column in `attention_transitions`
6. **Bug confirmed**: `resolve_curiosidad()` doesn't emit `CURIOSITY_RESOLVED` event (blocks CX-11)
7. **Cross-cutting risk**: "100% excitatory" — all proposals ADD connections without inhibition

---

## CX-9: L3→L9 — GNW Workspace → Self-Model Refresh

### Papers (15)

| # | Citation | DOI |
|---|----------|-----|
| 1 | Northoff, G. & Bermpohl, F. (2004). Cortical midline structures and the self. *Trends Cogn. Sci.*, 8(3), 102-107. | 10.1016/j.tics.2004.01.004 |
| 2 | Qin, P. & Northoff, G. (2011). Self and default-mode network. *NeuroImage*, 57(3), 1221-1233. | 10.1016/j.neuroimage.2011.05.028 |
| 3 | Andrews-Hanna, J.R. et al. (2014). Default network and self-generated thought. *Ann. NY Acad. Sci.*, 1316(1), 29-52. | 10.1111/nyas.12360 |
| 4 | Luppi, A.I. et al. (2024). Synergistic workspace revealed by IID. *eLife*, 13, e88173. | 10.7554/eLife.88173 |
| 5 | Davey, C.G. et al. (2016). Mapping the self in DMN. *NeuroImage*, 132, 390-397. | 10.1016/j.neuroimage.2016.02.022 |
| 6 | Shea, N. & Frith, C.D. (2019). GNW needs metacognition. *Trends Cogn. Sci.*, 23(7), 560-571. | 10.1016/j.tics.2019.04.003 |
| 7 | Raichle, M.E. (2015). The brain's default mode network. *Ann. Rev. Neurosci.*, 38, 433-447. | 10.1146/annurev-neuro-071013-014030 |
| 8 | Sui, J. & Humphreys, G.W. (2015). The integrative self. *Trends Cogn. Sci.*, 19(12), 719-728. | 10.1016/j.tics.2015.08.015 |
| 9 | Cleeremans, A. (2011). Radical Plasticity Thesis. *Front. Psychol.*, 2, 86. | 10.3389/fpsyg.2011.00086 |
| 10 | Lou, H.C. et al. (2017). Towards cognitive neuroscience of self-awareness. *Neurosci. Biobehav. Rev.*, 83, 765-773. | 10.1016/j.neubiorev.2016.04.004 |
| 11 | Graziano, M.S.A. (2019). *Rethinking Consciousness*. Norton. | ISBN: 978-0393541342 |
| 12 | Whitfield-Gabrieli, S. & Ford, J.M. (2012). DMN activity in psychopathology. *Ann. Rev. Clin. Psychol.*, 8, 49-76. | 10.1146/annurev-clinpsy-032511-143049 |
| 13 | Nolen-Hoeksema, S. (1991). Responses to depression. *J. Abnorm. Psychol.*, 100(4), 569-582. | 10.1037/0021-843X.100.4.569 |
| 14 | Mashour, G.A. et al. (2020). Conscious Processing and GNW. *Neuron*, 105(5), 776-798. | 10.1016/j.neuron.2020.01.026 |
| 15 | Garrison, K.A. et al. (2015). Meditation reduces DMN activity. *Cogn. Affect. Behav. Neurosci.*, 15(3), 712-720. | 10.3758/s13415-015-0358-3 |

### Mechanism

Workspace broadcasts self-referential content → CMS ventral cluster detects self-relevance (Northoff 2004) → graded 0.0-1.0 score via keyword/theme/source matching → triggers `reflect_on_self()` when score > 0.3. DMN bidirectional flow confirmed by DCM (Davey 2016) and synergistic workspace model (Luppi 2024). CX-3 (L9→L3) is proactive (self pushes); CX-9 (L3→L9) is reactive (self receives).

### Evidence

- Qin & Northoff 2011: MPFC self-specific across 87 studies (1433 participants)
- Sui & Humphreys 2015: Self-associated stimuli 30-50ms faster RT (automatic detection)
- Whitfield-Gabrieli & Ford 2012: DMN hyperconnectivity → pathological self-referential loops
- Garrison 2015: Meditation reduces PCC activity ~35% (biological circuit breaker)

### Implementation (~60 LOC)

```python
_CX9_COOLDOWN = 300.0          # 5 min refractory period
_CX9_NOVELTY_THRESHOLD = 0.3   # Minimum self-relevance score
_SELF_REF_KEYWORDS = {"self_model", "identity", "capability", "performance", ...}
_SELF_REF_SOURCES = {"self_model_gwt", "self_model_refreshed"}  # Anti-echo

def _compute_self_relevance(data: dict) -> float:
    """Graded 0.0-1.0 (Northoff 2004 CMS analog)."""
    # Source exclusion (anti-echo) + theme matching + keyword scan + pronoun detection

def _on_workspace_broadcast_to_self_model(event_name, data):
    """3 circuit breakers: anti-echo, cooldown, novelty gate."""
    # 1. Skip if source in _SELF_REF_SOURCES (anti-echo)
    # 2. Skip if < 5 min since last update (refractory)
    # 3. Skip if self_relevance < 0.3 (novelty gate)
    # → reflect_on_self(trigger="workspace_broadcast", context=...)

# Registration: event_bus.on(Events.WORKSPACE_COMPETITION_COMPLETE, ...)
# Requires: enrich competition.py payload with winner_topics
```

### Risks

| Risk | Severity | Mitigation |
|------|----------|------------|
| Rumination loop (CX-3↔CX-9) | CRITICAL | Anti-echo source exclusion + 5 min cooldown + novelty gate |
| Self-model update storm | HIGH | Cooldown ensures max 1 refresh per 5 min |
| False self-reference detection | MEDIUM | Graded scoring with 0.3 threshold, multiple signal types |
| Performance cost of reflect_on_self() | MEDIUM | Background thread, cooldown |

### Blind Spots (Agent 4)

- **Salience gate missing**: Brain has SN (anterior insula) gating DMN-ECN crosstalk (Menon 2011). CX-9 has no equivalent. Need PE-threshold accumulator before triggering.
- **No valence distinction**: Negative self-referential broadcasts → negativity-biased self-model (Gotlib & Joormann 2010).
- **Recommended kill switch**: If self-model updated >3 times in 100 cycles, disable CX-9 until next sleep cycle.

### Codebase (Agent 5)

- **Feasibility**: 7/10 | **Risk**: 4/10 | **LOC**: ~60
- **Blocker**: `_emit_competition_event` lacks `winner_topics` — need to enrich `competition.py:264-274`
- **reflect_on_self()** is heavy (~200-500ms, PG queries) — must run in background thread
- **Files**: wiring.py, competition.py (enrich), test_cross_loops.py

### Tests

| # | Test | Expected |
|---|------|----------|
| T1 | Broadcast with self-referential themes | Handler fires, calls reflect_on_self(), logs self_relevance >= 0.3 |
| T2 | Broadcast with source="self_model_gwt" | Handler returns early (anti-echo) |
| T3 | Two broadcasts within 5 min | First triggers, second blocked by cooldown |
| T4 | Broadcast with non-self themes | self_relevance < 0.3, returns early |
| T5 | Same themes after cooldown | Both trigger but second blocked by novelty gate |

---

## CX-10: L9↔L5 — Self-Model ↔ Metacognition

### Papers (12)

| # | Citation | DOI |
|---|----------|-----|
| 1 | Fleming, S.M. & Lau, H.C. (2014). How to measure metacognition. *Front. Hum. Neurosci.*, 8, 443. | 10.3389/fnhum.2014.00443 |
| 2 | Maniscalco, B. & Lau, H. (2012). Meta-d' from confidence ratings. *Conscious. Cogn.*, 21(1), 422-430. | 10.1016/j.concog.2011.09.021 |
| 3 | Fleming, S.M. & Dolan, R.J. (2012). Neural basis of metacognitive ability. *Phil. Trans. R. Soc. B*, 367, 1338-1349. | 10.1098/rstb.2011.0417 |
| 4 | Fleming, S.M. et al. (2010). Introspective accuracy and brain structure. *Science*, 329, 1541-1543. | 10.1126/science.1191883 |
| 5 | Yeung, N. & Summerfield, C. (2012). Metacognition in decision-making. *Phil. Trans. R. Soc. B*, 367, 1310-1321. | 10.1098/rstb.2011.0416 |
| 6 | Kruger, J. & Dunning, D. (1999). Unskilled and unaware. *J. Pers. Soc. Psychol.*, 77(6), 1121-1134. | 10.1037/0022-3514.77.6.1121 |
| 7 | Jansen, R.A. et al. (2021). Rational model of Dunning-Kruger. *Nat. Hum. Behav.*, 5(6), 756-763. | 10.1038/s41562-021-01057-0 |
| 8 | Nelson, T.O. & Narens, L. (1990). Metamemory framework. *Psychol. Learn. Motiv.*, 26, 125-173. | 10.1016/S0079-7421(08)60053-5 |
| 9 | Koriat, A. (1993). Accessibility model of FOK. *Psychol. Rev.*, 100(4), 609-639. | 10.1037/0033-295X.100.4.609 |
| 10 | Rouault, M. et al. (2018). Psychiatric symptoms and metacognition. *Biol. Psychiatry*, 84(6), 443-451. | 10.1016/j.biopsych.2017.12.017 |
| 11 | Vaccaro, A.G. & Fleming, S.M. (2018). Metacognitive neuroimaging meta-analysis. *Brain Neurosci. Adv.*, 2. | 10.1177/2398212818810591 |
| 12 | Mazancieux, A. et al. (2020). G factor for metacognition. *J. Exp. Psychol. Gen.*, 149(9), 1788-1799. | 10.1037/xge0000746 |

### Mechanism

**Direction A (L9→L5)**: Self-model discrepancies lower L2 metacognitive precision per domain (Nelson & Narens 1990 MONITORING signal). Inaccurate self-model → noisy metacognitive cues (Koriat 1993) → lower effective meta-d'. Modifies `prediction_state_l2` precision via `cx10_precision_modifiers` table.

**Direction B (L5→L9)**: Systematic L2 bias (consistent over/under-prediction) triggers self-model reassessment (Nelson & Narens 1990 CONTROL signal). Kruger & Dunning 1999: incompetence prevents recognition of incompetence.

**Asymmetric timescales**: Meta→Self is fast (per judgment), Self→Meta is slow (accumulated evidence). Fleming & Dolan 2012: metacognition neurally dissociable from task performance (aPFC BA10 vs sensory/motor areas).

### Evidence

- Fleming et al. 2010: BA10 gray matter correlates with metacognitive ability (r=0.36) but NOT task performance (r=0.04). N=32.
- Kruger & Dunning 1999: Bottom-quartile overestimated by 50 percentile points. Training improved both performance AND self-assessment. N>300.
- Rouault et al. 2018: N=995. Anxiety → lower confidence + higher meta-d'/d'. Compulsive → higher confidence + lower meta-d'/d'. Task performance unaffected.
- Mazancieux et al. 2020: Cross-domain metacognitive correlations (r=0.15-0.25) across 4 tasks in N=181.

### Implementation (~40 LOC)

```python
_CX10_PRECISION_FLOOR = 0.15     # Prevents confidence collapse
_CX10_ADJUSTMENT_DECAY = 0.95    # Modifier decays back to 1.0
_CX10_DISC_WEIGHT = 0.08         # Discrepancy → precision reduction
_CX10_BIAS_THRESHOLD = 0.25      # Systematic bias triggers reassessment

# Direction A: L9→L5
def _on_self_model_discrepancy_to_metacognition(event_name, data):
    """MONITORING signal. Discrepancies lower L2 precision per domain.
    Writes to cx10_precision_modifiers table (SQLite)."""
    # Floor at 0.15, cap at 5 discrepancies, domain-isolated

# Direction B: L5→L9
def _check_metacognitive_bias_to_self_model(conn, domain):
    """CONTROL signal. Systematic bias → self-model reassessment.
    Checks avg(predicted - actual) over 20-sample window."""
    # If |mean_bias| > 0.25, emit SELF_MODEL_REFRESHED with source="metacognitive_bias_cx10b"

# Integration: preturn_inject.py reads cx10_precision_modifiers
# Registration: event_bus.on(Events.SELF_MODEL_REFRESHED, ...)
```

### Risks

| Risk | Severity | Mitigation |
|------|----------|------------|
| Confidence death spiral | HIGH | Precision floor 0.15, asymmetric rates (lower=0.08, recover=0.02/cycle) |
| Interaction with CX-6 | HIGH | CX-10 modulates precision, CX-6 modulates temperature — different parameters |
| Temporal scale confusion | MEDIUM | Domain isolation, windowed evidence (20 samples) |
| Metacognitive noise contaminating self-model | MEDIUM | Direction B requires 10+ samples + systematic bias > 0.25 |

### Blind Spots (Agent 4)

- **Local vs global confidence collapsed**: Fleming & Daw 2017 show local (trial) and global (trait) confidence diverge. CX-10 treats them as one.
- **Three-way convergence**: CX-10 + CX-13 + CX-6 all affect explore/exploit. Under stress, all push toward exploration simultaneously.
- **Recommended**: Confidence floor 0.3 (not 0.15), asymmetric learning rate 0.02, 5-event evidence window.

### Codebase (Agent 5)

- **Feasibility**: 8/10 | **Risk**: 3/10 | **LOC**: ~40
- **Clean data flow**: `SELF_MODEL_REFRESHED` emission #3 (discrepancy_detection) already has `domains` list
- **SQLite WAL** handles concurrent access between wiring.py and preturn_inject.py
- **Files**: wiring.py, test_cross_loops.py

### Tests

| # | Test | Expected |
|---|------|----------|
| T1 | Emit SELF_MODEL_REFRESHED with discrepancy_count=3, domains=["trading"] | cx10_precision_modifiers["trading"] reduced by ~0.24 |
| T2 | Emit with discrepancy_count=0 | No changes, handler returns early |
| T3 | Multiple discrepancies → check floor | Precision never drops below 0.15 |
| T4 | Systematic L2 overconfidence (20 samples) | CX-10B triggers self-model reassessment |

---

## CX-11: L6→L8 — Curiosity → Causal Discovery

### Papers (13)

| # | Citation | DOI |
|---|----------|-----|
| 1 | Bramley, N.R. et al. (2015). Conservative forgetful scholars. *J. Exp. Psychol.: LMC*, 41(3), 708-731. | 10.1037/xlm0000061 |
| 2 | Steyvers, M. et al. (2003). Causal networks from observations and interventions. *Cogn. Sci.*, 27(3), 453-489. | 10.1207/s15516709cog2703_6 |
| 3 | Coenen, A. et al. (2015). Strategies to intervene on causal systems. *Cogn. Psychol.*, 79, 102-133. | 10.1016/j.cogpsych.2015.02.004 |
| 4 | Scherrer, N. et al. (2022). Learning Neural Causal Models with Active Interventions. *NeurIPS 2022*. | 10.48550/arXiv.2109.02429 |
| 5 | Tigas, P. et al. (2022). Interventions, Where and How? *NeurIPS 2022*. | 10.48550/arXiv.2203.02016 |
| 6 | Eberhardt, F. & Scheines, R. (2007). Interventions and Causal Inference. *Phil. Sci.*, 74(5), 981-995. | 10.1086/525638 |
| 7 | Hauser, A. & Buhlmann, P. (2012). Interventional Markov equivalence classes. *JMLR*, 13, 2409-2464. | N/A (JMLR) |
| 8 | Gottlieb, J. et al. (2013). Information-seeking, curiosity, attention. *Trends Cogn. Sci.*, 17(11), 585-593. | 10.1016/j.tics.2013.09.001 |
| 9 | Oudeyer, P.-Y. & Kaplan, F. (2007). What is intrinsic motivation? *Front. Neurorobot.*, 1, 6. | 10.3389/neuro.12.006.2007 |
| 10 | Pathak, D. et al. (2017). Curiosity-driven Exploration. *ICML 2017*. | 10.5555/3305890.3305968 |
| 11 | Burda, Y. et al. (2018). Exploration by Random Network Distillation. *ICLR 2019*. | 10.48550/arXiv.1810.12894 |
| 12 | Bramley, N.R. et al. (2017). Formalizing Neurath's ship. *Psychol. Rev.*, 124(3), 301-338. | 10.1037/rev0000061 |
| 13 | Gruber, M.J. et al. (2014). Curiosity modulates hippocampus-dependent learning. *Neuron*, 84(2), 486-496. | 10.1016/j.neuron.2014.08.060 |

### Mechanism

Curiosity resolution = computational intervention (Bramley 2017, Steyvers 2003). Interventional data resolves Markov equivalence ambiguity that observational data cannot (Eberhardt & Scheines 2007). Curiosity-resolved observations get 1.5x weight (Gruber 2014: curiosity-enhanced encoding), bounded to prevent sampling bias (Burda 2018: Noisy TV Problem). Buffer 20 observations before flushing to NOTEARS (Bramley 2015: local batch updating).

**Reverse flow (CX-11b)**: Uncertain NOTEARS edges (w=0.05-0.15) generate targeted curiosity questions (Coenen 2015: downstream connectivity heuristic, Tigas 2022: BOED information gain). Max 2 questions per causal discovery run.

### Evidence

- Steyvers et al. 2003: Interventional data improved causal structure identification ~35%
- Scherrer et al. 2022: AIT reduced required interventions 2-5x vs random
- Gruber et al. 2014: Curiosity-state memories showed enhanced hippocampal encoding + 24h retention
- Eberhardt & Scheines 2007: log(N)+1 multi-variable interventions suffice for N-variable DAG
- Burda et al. 2018: Prediction-error curiosity creates sampling bias toward stochastic domains

### Implementation (~45 LOC)

```python
_CX11_CURIOSITY_WEIGHT = 1.5     # vs 1.0 observational (Gruber 2014)
_CX11_MAX_PENDING_OBS = 20       # Batch threshold (Bramley 2015)
_cx11_pending_observations = []   # Buffer

def _on_curiosity_feeds_causal(event_name, data):
    """CX-11a: Curiosity → causal discovery. Buffer interventional observations."""
    # Extract from_topic (attention focus) and to_topic (category)
    # Append with weight=1.5, source="curiosity_intervention"
    # Flush when buffer >= 20

def _flush_curiosity_observations_to_causal():
    """Write buffered observations to transition_stats table."""

def _on_causal_gaps_direct_curiosity():
    """CX-11b reverse: Uncertain edges (0.05 < |w| < 0.15) → curiosity questions."""
    # Max 2 per run, via push_curiosidad()
    # Called from _tick_causal_discovery after NOTEARS succeeds

# Registration: event_bus.on(Events.CURIOSITY_RESOLVED, _on_curiosity_feeds_causal)
```

### Risks

| Risk | Severity | Mitigation |
|------|----------|------------|
| Sampling bias (Burda 2018) | HIGH | Bounded 1.5x weight, observational base data preserved |
| Feedback loop (gaps → curiosity → edges → new gaps) | MEDIUM | Max 2 questions per run, cooldown per topic pair |
| NOTEARS unsuitability (Kaiser 2021) | MEDIUM | Use as approximate signal, not ground truth |
| Stale interventional data | LOW | NOTEARS re-runs periodically, timestamps on observations |

### Blind Spots (Agent 4)

- **NOTEARS finds correlation, not causation** (Kaiser & Sipos 2022). Curiosity-driven co-occurrence ≠ causal relationship.
- **Faithfulness assumption violated**: Curiosity-driven sampling introduces dependencies (curious topics co-occur because explored together).
- **Recommended**: Separate data matrices (curiosity vs routine). NOTEARS on routine only. Curiosity data = hypothesis generation, not direct feed.

### Codebase (Agent 5)

- **Feasibility**: 7/10 | **Risk**: 2/10 | **LOC**: ~45
- **Bug**: `causal_discovery.py:97` queries `count FROM attention_transitions` — no such column. Should use `transition_stats` instead.
- **Prerequisite**: `resolve_curiosidad()` must emit `CURIOSITY_RESOLVED` (currently only sleep_loop auto-resolve does)
- **Write target**: `transition_stats` (has `count`), not `attention_transitions`
- **Files**: wiring.py, test_cross_loops.py, optionally causal_discovery.py (bug fix)

### Tests

| # | Test | Expected |
|---|------|----------|
| T1 | Emit CURIOSITY_RESOLVED with category="trading" | Observation buffered with weight=1.5 |
| T2 | Push 20 observations | Flush to transition_stats, buffer empty |
| T3 | Insert W matrix with uncertain edge (w=0.08) | 1-2 curiosity questions generated |
| T4 | 50 resolutions same topic pair | Weight bounded, DAG edge ≤ 3x baseline |

---

## CX-12: L7→L10 — Action Outcomes → Forgetting

### Papers (13)

| # | Citation | DOI |
|---|----------|-----|
| 1 | Roediger, H.L. & Karpicke, J.D. (2006a). Power of Testing Memory. *Perspect. Psychol. Sci.*, 1(3), 181-210. | 10.1111/j.1745-6916.2006.00012.x |
| 2 | Roediger, H.L. & Karpicke, J.D. (2006b). Test-Enhanced Learning. *Psychol. Sci.*, 17(3), 249-255. | 10.1111/j.1467-9280.2006.01693.x |
| 3 | Karpicke, J.D. & Roediger, H.L. (2008). Critical Importance of Retrieval. *Science*, 319, 966-968. | 10.1126/science.1152408 |
| 4 | Rowland, C.A. (2014). Testing effect meta-analysis. *Psychol. Bull.*, 140(6), 1432-1463. | 10.1037/a0037559 |
| 5 | Cepeda, N.J. et al. (2006). Distributed Practice meta-analysis. *Psychol. Bull.*, 132(3), 354-380. | 10.1037/0033-2909.132.3.354 |
| 6 | Bjork, R.A. & Bjork, E.L. (1992). New Theory of Disuse. In *From Learning Processes to Cognitive Processes*. Erlbaum. | N/A (book chapter) |
| 7 | Bjork, R.A. (1994). Desirable Difficulties. In *Metacognition*. MIT Press. | N/A (book chapter) |
| 8 | Anderson, M.C. et al. (1994). Remembering can cause forgetting. *J. Exp. Psychol.: LMC*, 20(5), 1063-1087. | 10.1037/0278-7393.20.5.1063 |
| 9 | Anderson, M.C. & Hanslmayr, S. (2014). Motivated Forgetting. *Trends Cogn. Sci.*, 18(6), 279-292. | 10.1016/j.tics.2014.03.002 |
| 10 | Wimber, M. et al. (2015). Adaptive forgetting via cortical pattern suppression. *Nat. Neurosci.*, 18, 582-589. | 10.1038/nn.3973 |
| 11 | Kornell, N. et al. (2009). Unsuccessful retrieval enhances learning. *J. Exp. Psychol.: LMC*, 35(4), 989-998. | 10.1037/a0015729 |
| 12 | Storm, B.C. & Levy, B.J. (2012). RIF inhibitory account progress. *Mem. Cogn.*, 40, 827-843. | 10.3758/s13421-012-0211-7 |
| 13 | Steyvers, M. & Tenenbaum, J.B. (2005). Semantic network structure. *Cogn. Sci.*, 29(1), 41-78. | 10.1207/s15516709cog2901_3 |

### Mechanism

Each RETRIEVE action triggers `compute_fadem_strength_ss_rs(retrieval_event=True)` — SS grows monotonically (Bjork & Bjork 1992), modulated by difficulty bonus `max(0.5, 1.5 - RS)` (Bjork 1994: desirable difficulties). The SS/RS model already exists in forgetting.py but is NEVER called from the event pipeline. Failed retrieval still provides smaller SS boost (Kornell 2009: pretesting effect, rate × 0.4). Rich-get-richer mitigation via `1/(1 + 0.1*sqrt(N))` per topic (Steyvers & Tenenbaum 2005). FORGET action increases `decay_multiplier` (Anderson & Hanslmayr 2014: active forgetting).

### Evidence

| Finding | Source | Effect Size |
|---------|--------|-------------|
| Testing > Restudying at 2d retention | Roediger & Karpicke 2006a | 13% vs 56% forgetting |
| Overall testing effect | Rowland 2014 | g = 0.50 (159 studies) |
| Failed retrieval still helps | Kornell et al. 2009 | Significant pretesting effect |
| RIF is strength-independent | Storm & Levy 2012 | Competitors suppressed regardless |
| Retrieval suppresses competitor patterns | Wimber et al. 2015 | fMRI evidence |

### Implementation (~165 LOC total, multi-file)

```python
# events.py: ACTION_OUTCOME = 'action_outcome'
# Payload: {action, topic, success, pe, retrieved_ids}

# wiring.py handler:
CX12_FAILED_RETRIEVAL_SS_RATE = 0.4   # Kornell 2009
CX12_USAGE_DAMPENING = 0.1            # Rich-get-richer mitigation
CX12_FORGET_DECAY_BOOST = 1.8         # Anderson & Hanslmayr 2014

async def _on_action_outcome(event_name, data):
    """Route action outcomes to forgetting module."""
    if action == "retrieve" and retrieved_ids:
        # Update topic usage counter
        usage_dampening = 1.0 / (1.0 + CX12_USAGE_DAMPENING * math.sqrt(n_retrievals))
        # For each memory: SS boost with difficulty bonus, dampened by usage
        # Success: full SS_LEARNING_RATE × usage_dampening
        # Failure: SS_LEARNING_RATE × 0.4 × usage_dampening (Kornell 2009)
    elif action == "forget":
        # Active decay boost (Anderson & Hanslmayr 2014)

# Emission from active_inference_integration.py after action execution
```

### Risks

| Risk | Severity | Mitigation |
|------|----------|------------|
| Rich-get-richer monopolization | HIGH | Usage dampening 1/(1+0.1√N), logarithmic decay of gains |
| Irreversible knowledge loss | MEDIUM | Natural power-law RS decay preserves high-SS memories longer |
| SS inflation ceiling | LOW | (1-SS)×lr formula provides diminishing returns near 1.0 |
| RIF double-counting | LOW | RIF on salience, CX-12 on SS/RS — independent (Storm & Levy 2012) |

### Blind Spots (Agent 4)

- **No global downscaling**: Tononi & Cirelli SHY demands periodic renormalization. Need 0.95x multiplier per sleep cycle.
- **No diversity floor**: Identity/critical memories need minimum 0.3 SS regardless of usage.
- **Context switch vulnerability**: Switching projects kills all protection for previous project.
- **Recommended**: Add RIF for competitors + global downscaling + importance floor.

### Codebase (Agent 5)

- **Feasibility**: 4/10 | **Risk**: 6/10 | **LOC**: ~165
- **Requires building 3 non-existent things**: outcome tracking, per-topic state, FadeMem integration
- **No ACTION_OUTCOME event** — needs new event in events.py
- **No outcome observation loop** — system recommends actions but never tracks outcomes
- **Per-topic decay_multiplier not supported** — forgetting.py takes global parameter only
- **Files**: events.py, active_inference_integration.py, wiring.py, forgetting.py, sleep_loop.py, test_cross_loops.py

### Tests

| # | Test | Expected |
|---|------|----------|
| T1 | Retrieve with SS=0.3, RS=0.4, success=True | SS increases to ~0.415, RS resets to 1.0 |
| T2 | Retrieve with success=False | SS increases less (×0.4), RS partial reset |
| T3 | 100 retrievals same topic | SS gain at #100 < 50% of #1 (dampening) |
| T4 | SS=0.5 RS=0.9 vs SS=0.5 RS=0.2 | RS=0.2 gets larger SS gain (spacing effect) |

---

## CX-13: L4→L7 — Emotion (PAD) → Action Selection (EFE)

### Papers (13)

| # | Citation | DOI |
|---|----------|-----|
| 1 | Damasio, A.R. (1996). Somatic marker hypothesis. *Phil. Trans. R. Soc. B*, 351, 1413-1420. | 10.1098/rstb.1996.0125 |
| 2 | Slovic, P. et al. (2007). The affect heuristic. *Eur. J. Oper. Res.*, 177(3), 1333-1352. | 10.1016/j.ejor.2005.04.006 |
| 3 | Schwarz, N. & Clore, G.L. (2003). Mood as information: 20 years later. *Psychol. Inquiry*, 14(3-4), 296-303. | 10.1080/1047840X.2003.9682896 |
| 4 | Aston-Jones, G. & Cohen, J.D. (2005). LC-NE adaptive gain theory. *Ann. Rev. Neurosci.*, 28, 403-450. | 10.1146/annurev.neuro.28.061604.135709 |
| 5 | Elliot, A.J. (2006). Approach-Avoidance Motivation. *Motiv. Emot.*, 30, 111-116. | 10.1007/s11031-006-9028-7 |
| 6 | Lerner, J.S. et al. (2015). Emotion and Decision Making. *Ann. Rev. Psychol.*, 66, 799-823. | 10.1146/annurev-psych-010213-115043 |
| 7 | Scherer, K.R. (2009). Dynamic architecture of emotion. *Cogn. Emot.*, 23(7), 1307-1351. | 10.1080/02699930902928969 |
| 8 | Doya, K. (2008). Modulators of Decision Making. *Nat. Neurosci.*, 11(4), 410-416. | 10.1038/nn2077 |
| 9 | Yu, A.J. & Dayan, P. (2005). Uncertainty, Neuromodulation, Attention. *Neuron*, 46(4), 681-692. | 10.1016/j.neuron.2005.04.026 |
| 10 | Vinckier, F. et al. (2018). Mood and decisions. *Nat. Commun.*, 9, 1708. | 10.1038/s41467-018-03774-z |
| 11 | Eldar, E. & Niv, Y. (2015). Mood as moving average of RPEs. *Nat. Commun.*, 6, 6149. | 10.1038/ncomms7149 |
| 12 | Dreisbach, G. & Goschke, T. (2004). Positive affect and cognitive flexibility. *J. Exp. Psychol.: LMC*, 30(2), 343-353. | 10.1037/0278-7393.30.2.343 |
| 13 | Dunn, B.D. et al. (2006). Critical review of somatic marker hypothesis. *Neurosci. Biobehav. Rev.*, 30(2), 239-271. | 10.1016/j.neubiorev.2005.07.001 |

### Mechanism

PAD modulates EFE **weights** (NOT temperature — CX-6 owns temperature). Three-parameter mapping following Doya 2008:
- **Pleasure** → pragmatic weight (Vinckier 2018: positive mood amplifies gain sensitivity)
- **Arousal** → epistemic weight (Aston-Jones & Cohen 2005: high tonic LC = exploration)
- **Dominance** → cost weight (Lerner 2015: high certainty appraisal = risk-seeking)

`G(a) = -(w_prag × pragmatic) - (w_epist × epistemic) + (w_cost × cost)` where weights are PAD-modulated with ±0.4 max delta, floored at 0.3.

Anti-perseveration: same action >3 times → epistemic boost +0.3 (Dreisbach & Goschke 2004). Model confidence dampening: PAD influence stronger when model uncertain (Dunn 2006: SMH weaker for structured decisions).

### Evidence

| Finding | Source | Implication |
|---------|--------|-------------|
| Somatic markers bias decisions | Damasio 1996 | PAD → pragmatic weights |
| Happy = heuristic, sad = systematic | Schwarz & Clore 2003 | Pleasure → exploit/explore bias |
| LC tonic = explore, phasic = exploit | Aston-Jones & Cohen 2005 | Arousal → epistemic weight |
| Mood modulates gain/loss weights | Vinckier 2018 | Pleasure → pragmatic gain/cost ratio |
| Mood-outcome positive feedback | Eldar & Niv 2015 | RISK: perseveration trap |
| SMH weaker for structured decisions | Dunn et al. 2006 | Scale by inverse model confidence |

### Implementation (~60 LOC)

```python
CX13_MAX_WEIGHT_DELTA = 0.4           # Homeostatic bound
CX13_AROUSAL_EPIST_SCALE = 0.35       # Aston-Jones & Cohen 2005
CX13_PLEASURE_PRAG_SCALE = 0.30       # Vinckier 2018
CX13_DOMINANCE_COST_SCALE = 0.25      # Lerner 2015
CX13_ANTI_PERSEVERATION_BOOST = 0.3   # Dreisbach & Goschke 2004

def compute_pad_efe_weights(pleasure, arousal, dominance,
                            model_observations=0, consecutive_same_action=0):
    """CX-13: PAD-modulated EFE weights. Returns {w_pragmatic, w_epistemic, w_cost}."""
    # Each PAD dimension → different EFE parameter (Doya 2008)
    # Confidence dampening: 1/(1 + 0.01*model_observations) (Dunn 2006)
    # Anti-perseveration check (Dreisbach & Goschke 2004)
    # Floors at 0.3 for all weights

# Modified in select_action() — pull model, no event handler needed
# Reads PAD from config._emotional_state, computes weights, passes to compute_efe()
```

### Risks

| Risk | Severity | Mitigation |
|------|----------|------------|
| Double-counting with CX-6 | HIGH | CX-6 = temperature, CX-13 = weights (orthogonal) |
| Mood-congruent perseveration | HIGH | Anti-perseveration boost + model confidence dampening |
| Triple convergence (CX-6 + CX-10 + CX-13) | HIGH | Each modulates different parameter; combined bound needed |
| Emotional hijacking (transient spike) | MEDIUM | Weight deltas capped at ±0.4, floors at 0.3 |

### Blind Spots (Agent 4)

- **Inverted-U violation** (Cools & D'Esposito 2011): Two modulatory signals individually optimal can combine to exceed inverted-U peak.
- **No integration rule for CX-6 + CX-13**: Are they additive? Multiplicative? Need principled integration.
- **Yerkes-Dodson ignored**: Task complexity determines optimal arousal — CX-13 applies same modulation regardless.
- **Alternative**: Integrate PAD INTO precision (same channel as CX-6) via single allostatic integrator. This is biologically more accurate but conflicts with CX-6 ownership.

### Codebase (Agent 5)

- **Feasibility**: 7/10 | **Risk**: 5/10 | **LOC**: ~60
- **SystemState.as_tuple() blocker**: Returns `(topic, uncertainty_level, wm_level)`. Adding emotional dimensions fragments Dirichlet model. Resolution: inject at `select_action()` level, NOT in as_tuple().
- **emotional_valence** already in SystemState but unused by compute_efe()
- **Pull model** — no event handler, reads PAD directly in select_action()
- **Files**: active_inference.py (SystemState, get_current_state, select_action), test_cross_loops.py

### Tests

| # | Test | Expected |
|---|------|----------|
| T1 | PAD=(0.8, 0.0, 0.0) | w_pragmatic > 1.0, w_epistemic ≈ 1.0 |
| T2 | PAD=(0.0, 0.8, 0.0) | w_epistemic > 1.0 (exploration mode) |
| T3 | PAD=(0.0, 0.0, 0.8) | w_cost < 1.0 (bolder actions) |
| T4 | Same action 4 times | Epistemic boost +0.3 (anti-perseveration) |
| T5 | PAD neutral, 1000 model observations | Weight deltas near zero (confidence dampening) |

---

## CX-14: L2→L6 — Consolidation Gaps → Curiosity

### Papers (12)

| # | Citation | DOI |
|---|----------|-----|
| 1 | Loewenstein, G. (1994). Psychology of curiosity. *Psychol. Bull.*, 116(1), 75-98. | 10.1037/0033-2909.116.1.75 |
| 2 | Berlyne, D.E. (1960). *Conflict, Arousal, and Curiosity*. McGraw-Hill. | N/A (book) |
| 3 | Litman, J.A. (2005). Curiosity and the pleasures of learning. *Cogn. Emot.*, 19(6), 793-814. | 10.1080/02699930541000101 |
| 4 | Kang, M.J. et al. (2009). Wick in the candle of learning. *Psychol. Sci.*, 20(8), 963-973. | 10.1111/j.1467-9280.2009.02402.x |
| 5 | Diekelmann, S. & Born, J. (2010). Memory function of sleep. *Nat. Rev. Neurosci.*, 11, 114-126. | 10.1038/nrn2762 |
| 6 | Lewis, P.A. & Durrant, S.J. (2011). Overlapping memory replay builds schemata. *Trends Cogn. Sci.*, 15(8), 343-351. | 10.1016/j.tics.2011.06.004 |
| 7 | Ghosh, V.E. & Gilboa, A. (2014). What is a memory schema? *Neuropsychologia*, 53, 104-114. | 10.1016/j.neuropsychologia.2013.11.010 |
| 8 | van Kesteren, M.T.R. et al. (2012). Schema and novelty augment memory. *Trends Neurosci.*, 35(4), 211-219. | 10.1016/j.tins.2012.02.001 |
| 9 | Tse, D. et al. (2007). Schemas and memory consolidation. *Science*, 316, 76-82. | 10.1126/science.1135935 |
| 10 | Kumaran, D. & Maguire, E.A. (2007). Hippocampal match-mismatch processes. *J. Neurosci.*, 27(32), 8517-8524. | 10.1523/JNEUROSCI.1677-07.2007 |
| 11 | Wagner, U. et al. (2004). Sleep inspires insight. *Nature*, 427, 352-355. | 10.1038/nature02223 |
| 12 | Stachenfeld, K.L. et al. (2017). Hippocampus as predictive map. *Nat. Neurosci.*, 20, 1643-1653. | 10.1038/nn.4650 |

### Mechanism

Three gap detection channels during consolidation:

1. **Contradictions** (Berlyne 1960: conceptual conflict → D-type curiosity, Kumaran & Maguire 2007: hippocampal mismatch). `contradictions_found > 0` → high-priority question.
2. **Low fact density** (Lewis & Durrant 2011 iOtA: weak schema integration). `facts_extracted / clusters_found < 0.3` → medium-priority question about sparse topics.
3. **Bridge edges without shared facts** (Ghosh & Gilboa 2014: schema expects associative completeness). Structural gaps in relational graph → low-priority question.

Gated by selectivity (Diekelmann & Born 2010): max 3 questions per consolidation run, 4h cooldown per topic. PACE framework (Gruber 2019): appraisal gate — only gaps above significance threshold generate curiosity.

### Evidence

- Loewenstein 1994: Curiosity follows inverted-U with knowledge level (maximum at intermediate knowledge)
- Kumaran & Maguire 2007: Hippocampal activation maximal when sequence predictions violated
- Wagner et al. 2004: 59% insight with sleep vs 25% without (consolidation reveals hidden patterns)
- van Kesteren et al. 2012: Schema-incongruent information triggers hippocampal encoding enhancement
- Tse et al. 2007: Schema-consistent consolidates in 48h; inconsistent restructures schema

### Implementation (~45 LOC)

```python
_CX14_MIN_CONTRADICTIONS = 1         # Berlyne 1960
_CX14_FACT_DENSITY_THRESHOLD = 0.3   # Lewis & Durrant 2011
_CX14_MAX_QUESTIONS_PER_RUN = 3      # Selectivity gate
_CX14_COOLDOWN_HOURS = 4             # Per-topic cooldown
_cx14_recent_gap_topics = {}

def _on_consolidation_gaps_drive_curiosity(event_name, data):
    """CX-14: Three gap detection channels from consolidation results."""
    # Only on scope="full" consolidation runs
    # Channel 1: contradictions → high-priority D-type curiosity
    # Channel 2: low fact density → medium-priority I-type curiosity
    # Channel 3: bridge edges without facts → low-priority structural query
    # Cap at 3 questions, 4h cooldown per topic

# Registration: event_bus.on(Events.CONSOLIDATION_COMPLETE, ...)
```

### Risks

| Risk | Severity | Mitigation |
|------|----------|------------|
| Curiosity queue saturation | MEDIUM | Max 3 per run + 4h cooldowns |
| Stale questions | LOW | Check if gap already resolved before pushing |
| Interaction with CX-1 | LOW | CX-1 = prediction-error curiosity (PACE type a), CX-14 = gap curiosity (PACE type b) — distinct |
| CX-11 contamination | MEDIUM | Tag CX-14 questions, firewall from causal discovery data |

### Blind Spots (Agent 4)

- **PACE appraisal gate missing**: Not all gaps produce curiosity. Lateral PFC evaluates: Can I close this? Is it worth the cost? Is it anxiety-provoking or curiosity-provoking?
- **No gap-quality metric**: Gap between two well-established facts > gap at frontier of knowledge.
- **Resolution tracking**: Same gap flagged repeatedly without tracking prior resolution attempts.
- **Recommended**: Appraisal gate + budget cap + resolution tracking + CX-11 firewall.

### Codebase (Agent 5)

- **Feasibility**: 8/10 | **Risk**: 2/10 | **LOC**: ~45
- **Clean event payload**: CONSOLIDATION_COMPLETE has `contradictions_found`, `clusters_found`, `facts_extracted`, `bridge_edges`
- **push_curiosidad()** is safe to call from event handlers (CX-1 already does)
- **Missing from payload**: cluster topics/themes, per-topic fact counts
- **Files**: wiring.py, test_cross_loops.py

### Tests

| # | Test | Expected |
|---|------|----------|
| T1 | Consolidation with 2 contradictions | 1 high-priority curiosity question |
| T2 | 5 clusters, 1 fact (density=0.2) | Medium-priority gap question |
| T3 | Same topic within 4h cooldown | Question NOT pushed |
| T4 | scope="minimal" consolidation | Handler returns early |

---

## Cross-Cutting Analysis

### Risk Ranking

| Rank | CX | Risk | Primary Danger | Reversibility |
|------|-----|------|----------------|---------------|
| 1 | **CX-9** | CRITICAL | Rumination loop with CX-3, no natural termination | Low |
| 2 | **CX-13** | HIGH | Double-counting with CX-6, triple convergence with CX-10 | Medium |
| 3 | **CX-11** | HIGH | Undetectable causal graph bias from curiosity sampling | Low |
| 4 | **CX-10** | HIGH | Confidence death spiral, amplified by CX-6 and CX-13 | Medium |
| 5 | **CX-12** | MEDIUM | Rich-get-richer topic monopolization | Low |
| 6 | **CX-14** | MEDIUM | Curiosity overload, most manageable | High |

### Interaction Risks

1. **CX-10 + CX-13 + CX-6 triple convergence on EFE**: Three mechanisms all modulating explore/exploit. Under stress, all push toward exploration simultaneously → behavioral chaos.
2. **CX-9 + CX-10 self-referential amplification**: Self-referential error in workspace → CX-9 updates self-model → CX-10 lowers meta-confidence → CX-6 raises temperature → more errors → CX-3 pushes back to workspace. **5-node feedback loop**.
3. **CX-11 + CX-14 + CX-12 knowledge tunnel**: Curiosity → biased exploration → biased DAG → use-dependent protection immortalizes biased topics → consolidation gaps only in biased area → more biased curiosity. **Closed-loop knowledge tunnel**.

### Systemic Blind Spot: 100% Excitatory

Biological neural systems are ~20% inhibitory neurons. Every excitatory connection has inhibitory counterparts. All 6 proposals ADD connections without any corresponding inhibition, gating, or damping. This is a recipe for runaway dynamics.

### Recommended Circuit Breakers

| CX | Circuit Breaker | Mechanism |
|----|----------------|-----------|
| CX-9 | Salience gate + refractory | PE threshold > 0.4 + 10-cycle minimum + 3/100 kill switch |
| CX-10 | Confidence floor + asymmetric | Floor 0.3, lr_down=0.02, lr_up=0.15, 5-event window |
| CX-11 | Data stream separation | Separate matrices: curiosity vs routine. NOTEARS on routine only. |
| CX-12 | RIF + global downscaling | Competitors 0.95x penalty + global 0.95x per sleep cycle + identity floor 0.3 |
| CX-13 | Single allostatic integrator | Consider PAD → precision (same channel as CX-6) instead of separate pathway |
| CX-14 | PACE appraisal gate | P(resolvable) > 0.3, budget cap 3/cycle, resolution tracking, CX-11 firewall |

---

## Feasibility Audit

| CX | Feasibility | Risk | LOC | New Event? | Files Changed |
|----|------------|------|-----|------------|---------------|
| CX-9 | 7/10 | 4/10 | ~60 | No | wiring.py, competition.py |
| CX-10 | 8/10 | 3/10 | ~40 | No | wiring.py |
| CX-11 | 7/10 | 2/10 | ~45 | No | wiring.py, (causal_discovery.py bug fix) |
| CX-12 | 4/10 | 6/10 | ~165 | Yes: ACTION_OUTCOME | events.py, active_inference_integration.py, wiring.py, forgetting.py, sleep_loop.py |
| CX-13 | 7/10 | 5/10 | ~60 | No (pull model) | active_inference.py |
| CX-14 | 8/10 | 2/10 | ~45 | No | wiring.py |

### Implementation Order (safest → riskiest)

1. **CX-14** (Feasibility 8/10, Risk 2/10) — Clean event payload, push_curiosidad() safe from handlers
2. **CX-10** (Feasibility 8/10, Risk 3/10) — Clean data flow, existing emission has all needed data
3. **CX-11** (Feasibility 7/10, Risk 2/10) — Needs causal_discovery.py bug fix first, CURIOSITY_RESOLVED prerequisite
4. **CX-13** (Feasibility 7/10, Risk 5/10) — Pull model avoids event complexity, calibration challenge
5. **CX-9** (Feasibility 7/10, Risk 4/10) — Requires competition.py enrichment + full circuit breaker suite
6. **CX-12** (Feasibility 4/10, Risk 6/10) — Largest scope, builds 3 non-existent subsystems

### Known Bugs

1. **`causal_discovery.py:97`**: Queries `count FROM attention_transitions` — no such column. `except Exception: pass` silently swallows error. NOTEARS only uses prediction_results proximity, not attention transitions.
2. **`resolve_curiosidad()`**: Doesn't emit `CURIOSITY_RESOLVED` event. Only `sleep_loop.py` auto-resolve emits it. Manual resolution via MCP tool bypasses event entirely.

---

## TIER 1+2+3 Combined Status

| CX | Loop | Status | LOC | Tier |
|----|------|--------|-----|------|
| CX-1 | L4→L6 PE→Curiosity | IMPLEMENTED | ~25 | 1 |
| CX-2 | L6→L4 Curiosity→Precision | IMPLEMENTED | ~30 | 1 |
| CX-3 | L9→L3 Self→GNW | IMPLEMENTED | ~35 | 1 |
| CX-4a | L10→L2 Vault→Consolidation | IMPLEMENTED | ~20 | 1 |
| CX-4b | L2→L10 Consolidation→Decay | IMPLEMENTED | ~25 | 2 |
| CX-5 | L3→L7 GNW→Action | IMPLEMENTED | ~30 | 2 |
| CX-6 | L5→L7 Meta→Explore/Exploit | IMPLEMENTED | ~45 | 2 |
| CX-7 | L8→L4 Causal→Prediction | IMPLEMENTED | ~20 | 2 |
| CX-8 | L1→L10 Reconsolidation→Decay | IMPLEMENTED | ~25 | 2 |
| CX-9 | L3→L9 GNW→Self-Model | RESEARCHED | ~60 | 3 |
| CX-10 | L9↔L5 Self↔Metacognition | RESEARCHED | ~40 | 3 |
| CX-11 | L6→L8 Curiosity→Causal | RESEARCHED | ~45 | 3 |
| CX-12 | L7→L10 Action→Forgetting | RESEARCHED | ~165 | 3 |
| CX-13 | L4→L7 Emotion→Action | RESEARCHED | ~60 | 3 |
| CX-14 | L2→L6 Consolidation→Curiosity | RESEARCHED | ~45 | 3 |

**Coverage**: 9 implemented + 6 researched = 15/45 (33% active, 53% after implementation)

### Paper Count

| Tier | Papers | DOI-verified |
|------|--------|-------------|
| TIER 1 | 77 | 77 |
| TIER 2 | 81 | 81 |
| TIER 3 | 78 | 73 (5 books/chapters) |
| **Total** | **236** | **231** |

---

## Agent Efficiency

| Metric | TIER 1 | TIER 2 | TIER 3 |
|--------|--------|--------|--------|
| Agents | 6 | 5 | 5 |
| Cross-loops | 4 | 5 | 6 |
| Papers | 77 | 81 | 78 |
| CX per agent | 0.67 | 1.0 | 1.2 |
| Grouping | Individual | Paired + verification | Paired + verification |
