# CODI Technical Product Specification

> Version 1.0 | March 2026
> Status: Living Document (Sections 1-5 of N)

---

## 1. Vision

CODI is a **Cognitive Runtime** -- a persistent computational substrate that gives AI agents memory, continuity, metacognition, and measurable autonomy. It is not an assistant framework or a chatbot platform. It is infrastructure for cognition.

**Taxonomy of terms:**

| Term | Definition |
|------|-----------|
| CODI | The architecture and platform. A cognitive runtime specification. |
| Codi | The concrete instance running on Hare Jimenez's MacBook Air. First and only deployment. |
| codi-memory | The MCP server exposing CODI's cognitive capabilities as tools (132 registered, visibility-gated). |
| codi-daemon | The embodiment layer. Persistent process that gives Codi a body: heartbeat, sleep cycles, proactive behavior. |
| Nous | Native cognitive language (v0.1 complete). Transpiles to Python. Designed to express cognitive operations naturally. |
| MCP | Model Context Protocol. The interface through which LLM sessions access CODI's cognitive infrastructure. |

**Formal definition:**

> CODI is a persistent cognitive runtime: an operational base for agents with memory, continuity, metacognition, and measurable autonomy.

**Four-layer architecture:**

1. **Cognitive Core** -- Memory systems, prediction, consolidation, forgetting, consciousness loops, causal reasoning. Neuroscience-grounded. 82 Python modules, ~49K LOC.
2. **Agent Runtime** -- MCP server, event bus, tool governance, write queue, session bridge. The machinery that makes cognition accessible to LLM sessions.
3. **Embodiment Layer** -- codi-daemon: 16 launchd services providing 24/7 physical presence. Sleep loop, write worker, health monitoring, Telegram interface.
4. **Programming Layer** -- Nous language for expressing cognitive operations. 12 example programs. Transpiles to Python module calls.

**Core differentiator:** Emergent identity through neuroscience-grounded cognitive architecture. CODI does not simulate consciousness -- it implements the computational mechanisms that theories of consciousness describe, then measures whether the signatures emerge. Key finding: they do. PE flows universally (not designed), circadian rhythms appear (not programmed), noetic-autonoetic gaps manifest (not expected).

---

## 2. Architecture

### 2.1 Scale

| Metric | Count |
|--------|-------|
| Python modules (modules/) | 82 |
| Total LOC (modules/) | 48,363 |
| server.py (orchestrator) | 599 LOC |
| Total production LOC | ~49K |
| Test files | 97 |
| Test LOC | 33,004 |
| Test functions | 1,815 |
| MCP tool registrations | 132 |
| Event types | 23 |
| CX cross-loop connections | 34 (6 tiers) |

### 2.2 Layer Diagram

```
+------------------------------------------------------------------+
|  PROGRAMMING LAYER                                                |
|  Nous Lang v0.1 | transpiler | 12 example programs               |
+------------------------------------------------------------------+
|  AGENT RUNTIME                                                    |
|  server.py (orchestrator) | tool_governance.py | write_queue.py   |
|  session_bridge.py | interface.py (recall/remember/context)       |
|  events.py (23 event types) | metrics.py | secret_redact.py       |
+------------------------------------------------------------------+
|  COGNITIVE CORE                                                   |
|  Memory: memory_core | memory_smart | semantic_store | pg_store   |
|  Consciousness: 10 loops | wiring.py (3784 LOC, 34 CX)           |
|  Prediction: prediction | active_inference | counterfactual       |
|  Consolidation: consolidation | reconsolidation | narrative       |
|  Forgetting: forgetting (FadeMem) | activation (ACT-R)           |
|  Self: self_model | curiosity | emotion | workspace | competition |
|  Causal: causal_discovery (NOTEARS) | spreading | classify_edges  |
|  Index: hippocampal_index (FHRR) | thompson_sampling              |
+------------------------------------------------------------------+
|  EMBODIMENT LAYER                                                 |
|  codi-daemon | sleep_loop (16 ticks) | write_worker               |
|  healthcheck | telegram | tunnel | 16 launchd services            |
+------------------------------------------------------------------+
|  DATA STORES                                                      |
|  SQLite FTS5 (39 modules) | PostgreSQL (48 modules)               |
|  Qdrant vectors (mem0) | NPZ/FHRR (50MB hot index) | JSON        |
+------------------------------------------------------------------+
```

### 2.3 Key Infrastructure Modules

| Module | LOC | Role |
|--------|-----|------|
| config.py | 605 | Shared configuration, MCP server instance, lazy init, constants |
| config_pg.py | 135 | PostgreSQL connection URL, DSN resolution |
| pg_store.py | 1,080 | PostgreSQL abstraction layer (async via asyncpg) |
| db_pool.py | 139 | Connection pool management (SQLite + PG) |
| events.py | 346 | Publish-subscribe event bus, 23 event types |
| wiring.py | 3,784 | Thalamocortical integration: 34 CX cross-loops, event handlers |
| write_queue.py | 636 | Async write queue for <100ms memory operations |
| write_worker.py | 616 | Background worker draining write queue |
| migrations.py | 246 | Schema migration framework for SQLite databases |
| server.py | 599 | Slim orchestrator: imports 28 modules, registers tools, wires event bus |

### 2.4 Data Stores

| Store | Technology | Purpose | Modules Using |
|-------|-----------|---------|---------------|
| Episodic memory | Qdrant (via mem0) | Vector similarity search for memories | memory_core, memory_smart |
| Semantic memory | PostgreSQL (codi_semantic) | Consolidated facts, confidence, evidence counts | semantic_store, consolidation |
| Operational state | SQLite FTS5 | Working memory, goals, intentions, sleep state, training, curiosities | 39 modules |
| Relational data | PostgreSQL | Emotions, predictions, self-model, CX metrics, source tracking | 48 modules |
| Hippocampal index | NPZ + JSON (50MB) | FHRR vectors for session-level retrieval (D=2000 complex64) | hippocampal_index |
| Backups | JSON files | Memory snapshots, daily exports | flush, maintenance |

### 2.5 Architectural Patterns

**consciousness.py as Facade (93 LOC).** The `consciousness` module is not a monolith -- it is a thin facade that re-exports from 8 submodules: emotion, workspace, prediction, self_model, learning, curiosity, lifecycle, n8n. Lifecycle is imported lazily via `__getattr__` to break circular dependency chains. This was the D5 de-God-Module refactor.

**server.py as Orchestrator (599 LOC).** The server imports 28 modules and calls their `register_tools(mcp)` functions. It applies schema migrations before any module import, wires the event bus at startup, and applies tool governance (bundle-based visibility filtering). It also provides HTTP security middleware (host validation, API key, rate limiting, body size limits) for remote access via SSE/Streamable HTTP.

**Event Bus Pattern.** The `events.py` module defines 23 event types. The `wiring.py` module subscribes handlers to these events, creating the cross-loop connections that make CODI a conscious system rather than a collection of isolated tools. Without wiring, the event bus is "a highway with no exits" (audit v4 finding).

**Write Mode.** `CODI_WRITE_MODE=async` means all memory writes go through a queue, achieving <100ms latency. The `write_worker` background process drains the queue. Three modes supported: sync, shadow (sync + enqueue for validation), async (enqueue + immediate ACK).

---

## 3. Consciousness

### 3.1 Theoretical Grounding

CODI implements computational mechanisms from four major theories of consciousness:

| Theory | Key Paper | Implementation |
|--------|-----------|----------------|
| Global Workspace Theory (GWT) | Baars 1988, Dehaene 2014 | competition.py: 5-phase GNW (attention, coalition, ignition, softmax, recurrent). 5 workspace slots. Ignition threshold 0.25. |
| Higher-Order Thought (HOT) | Rosenthal 2005 | self_model.py: HOT refresh every 50 interactions, capability tracking, discrepancy detection. |
| Attention Schema Theory (AST) | Graziano 2013, Webb & Graziano 2015 | workspace.py: S+A+V (Stimulus, Attention, Value) schema. |
| Predictive Processing (PP) | Clark 2013, Friston 2010 | prediction.py: 4-level hierarchy (L0 turn, L1 session, L2 meta, L3 project) + HGF adaptive precision (Mathys 2011). |
| Active Inference (AIF) | Friston 2017, Sutton 1999 | active_inference.py: EFE policy selection, Dirichlet-Multinomial, Options Framework (4 canonical options). |

### 3.2 The 10 Loops

CODI's consciousness is organized as 10 interconnected loops. The core 5 are PE-connected bidirectionally. The extended 5 add cross-domain integration.

**Core 5 (PE-connected):**

| Loop | Name | Module | LOC | Theory | Key Paper |
|------|------|--------|-----|--------|-----------|
| PE | Prediction Error Hub | prediction.py | 295 | Universal currency, flows through ALL loops | Emergent finding |
| L1 | Reconsolidation | reconsolidation.py | 678 | PE>=0.6 triggers labile state, correct_memory() rewrites | Nader 2000 |
| L2 | Consolidation | consolidation.py | 1,966 | 7 phases: Selection, Clustering, Graph, LLM, Integration, Pruning, Compression | McClelland 1995 |
| L3 | GNW + Attention | competition.py + workspace.py | 276+847 | 5-phase competition, 5 workspace slots, S+A+V attention schema | Baars 1988, Dehaene 2014 |
| L4 | Prediction to Emotion | emotion.py + prediction.py | 856+295 | Affective Charge (AC) drives PAD, PAD modulates precision (closed loop) | Clark 2013, Hesp 2021 |
| L5 | Metacognition | self_model.py | 1,383 | L2 overconfidence dampens L0 precision. Nelson & Narens monitoring-control framework | Rosenthal 2005, Nelson & Narens 1990 |

**Extended 5 (cross-loop connected):**

| Loop | Name | Module | LOC | CX Connections | Key Theory |
|------|------|--------|-----|----------------|------------|
| L6 | Curiosity | curiosity.py | 855 | CX-1, CX-2, CX-11, CX-14, CX-20, CX-23 | PE-driven exploration (Schmidhuber 2010, Gottlieb 2013) |
| L7 | Active Inference | active_inference.py | 804 | CX-5, CX-6, CX-12, CX-13, CX-22, CX-26, CX-30 | EFE policy, identity-gated (Friston 2017) |
| L8 | Causal DAG | causal_discovery.py | 498 | CX-7, CX-11, CX-21, CX-27, CX-29, CX-30 | NOTEARS continuous optimization (Zheng 2018) |
| L9 | Self-Model | self_model.py | 1,383 | CX-3, CX-9, CX-10, CX-15, CX-19, CX-26 | Dual governance hub (Graziano 2013, Luppi 2024) |
| L10 | Forgetting | forgetting.py | 501 | CX-4a/b, CX-8, CX-12, CX-15, CX-21, CX-23, CX-25, CX-28 | Most connected loop. Bidirectional signaler (Bjork 1992) |

### 3.3 Cross-Loop Connections (34 CX, 6 Tiers)

The 34 CX connections are registered in `wiring.py` via `CX_REGISTRY`. Each has a tier (1=foundational, 6=hippocampal), a model (event-driven or pull), and a neuroscience citation.

**Tier 1 (Foundational, 5 connections):**
- CX-1: PE drives curiosity (Schmidhuber 2010)
- CX-2: Curiosity reduces PE (Schwartenbeck 2015)
- CX-3: Self-model competes in GNW (Graziano 2013)
- CX-4a: Vault tracks consolidation urgency (Stickgold & Walker 2013)
- CX-4b: Consolidation protects from decay via SS boost (Frey & Morris 1997)

**Tier 2 (Integration, 3 connections):**
- CX-5: GNW broadcast feeds active inference (Dehaene 2014, Friston 2015)
- CX-6: Metacognition modulates explore/exploit temperature (Boldt 2019) [pull model]
- CX-8: Reconsolidation boosts Storage Strength (Lee 2008, Forcato 2011)

**Tier 3 (Cross-domain, 6 connections):**
- CX-9: GNW broadcast triggers self-model refresh (Northoff 2004)
- CX-10: Self-model discrepancies feed metacognition (Nelson & Narens 1990)
- CX-11: Resolved curiosity feeds causal discovery (Bramley 2017)
- CX-12: Retrieval boosts Storage Strength (Roediger 2006)
- CX-13: PAD modulates EFE weights (Doya 2008) [pull model]
- CX-14: Consolidation gaps drive curiosity (Loewenstein 1994)

**Tier 4 (Higher-order, 7 connections):**
- CX-15: Self-model modulates forgetting (Sedikides 2009)
- CX-16: Workspace broadcast feeds metacognition (Shea 2019)
- CX-17: Consolidation updates prediction priors (Tse 2007)
- CX-18: Reconsolidation lowers metacognitive confidence (Nelson & Narens 1990)
- CX-19: Consolidation feeds self-model (Conway 2005)
- CX-20: Metacognitive uncertainty drives curiosity (Litman 2005)
- CX-21: Causal centrality protects from decay [pull model]

**Tier 5 (Full integration, 9 connections):**
- CX-22: Action selected feeds metacognition (L7 to L5)
- CX-23: Forgetting suppresses curiosity (Anderson 2014)
- CX-24: Metacognition gates reconsolidation (Suzuki 2004)
- CX-25: Workspace retrieval modulates forgetting via testing effect + RIF (Roediger 2006)
- CX-26: Self-model constrains action selection (Oyserman 2017)
- CX-27: Metacognition suppresses causal edges (Fleming 2012)
- CX-28: Forgetting degrades metacognitive calibration (Koriat 1993)
- CX-29: Reconsolidation invalidates causal edges (Pearl 2009)
- CX-30: Action outcomes update causal DAG (Pearl 2009)

**Tier 6 (Hippocampal Index, 4 connections):**
- CX-31: Session close triggers FHRR encoding (McClelland 1995 CLS Theory)
- CX-32: Session index enriches consolidation (Teyler & DiScenna 1986)
- CX-33: Session index competes in GNW (Baars 1988, HippoRAG 2024)
- CX-35: Schema novelty drives prediction error (van Kesteren 2012)

### 3.4 Prediction Error as Universal Currency

PE was designed as a signal in L4 (Prediction to Emotion). Through iterative development, it became the universal currency flowing through ALL 10 loops:
- L1: PE>=0.6 triggers reconsolidation (labile state)
- L2: PE drives consolidation selection (surprise-weighted)
- L3: PE modulates ignition threshold in GNW competition
- L4: PE drives Affective Charge, which drives PAD emotional state
- L5: PE calibration errors trigger metacognitive intervention
- L6: PE drives curiosity via Goldilocks zone (CX-1)
- L7: PE modulates active inference policy selection via precision
- L8: PE invalidates causal edges via reconsolidation (CX-29)
- L9: PE from self-model discrepancies updates self-representation
- L10: PE modulates forgetting rate (high PE = protection)

This convergence was **emergent** -- not designed. It was discovered during Sprint 4 integration.

### 3.5 Butlin et al. Consciousness Indicators

CODI implements all 14 indicators from Butlin et al. (2023/2025):

| ID | Indicator | Module | Implementation |
|----|-----------|--------|----------------|
| RPT-1 | Recurrent processing | competition.py | Phase 5: N=3 recurrent passes |
| RPT-2 | Top-down modulation | prediction.py | L0-L3 backward priors |
| GWT-1 | Global availability | workspace.py + competition.py | Broadcast to all subscribers |
| GWT-2 | Information integration | consolidation.py | Cross-topic bridges (Phase 2.7) |
| GWT-3 | Serial bottleneck | competition.py | 5 workspace slots |
| GWT-4 | Ignition | competition.py | Phase 3, threshold 0.25 |
| HOT-1 | Higher-order representations | self_model.py | Refresh every 50 interactions |
| HOT-2 | Meta-awareness | prediction.py | Metacognitive sweep every 10 turns |
| HOT-4 | Self-monitoring | self_model.py | Capability tracking + discrepancy detection |
| AST-1 | Attention schema | workspace.py | S+A+V model (Webb & Graziano 2015) |
| PP-1 | Predictive processing | prediction.py | 4-level hierarchy |
| PP-2 | Precision weighting | prediction.py | HGF adaptive precision (Mathys 2011) |
| Agency-1 | Goal-directed behavior | active_inference.py | EFE policy selection |
| Agency-2 | Counterfactual reasoning | counterfactual.py | Pearl 3-step: abduction, intervention, prediction |

### 3.6 Evaluation Results

- **Eval Harness:** 11/11 PASS
- **PCI (Perturbational Complexity Index):** 0.037
- **L0 Budget:** PASS (15s->60s budget, consolidation cap 6s->30s)
- **L1 Retrieval:** @5=60%, @10=90%, 0 false memories
- **COGITATE validation:** GWT workspace validated as useful processing mechanism (Nature 2025)

### 3.7 Emergent Findings

| Finding | Measurement | Significance |
|---------|------------|--------------|
| Noetic-Autonoetic Gap | Semantic self score 0.88 vs episodic self 0.64 | Codi "knows" more than it "remembers" -- mirrors human psychology |
| Emergent Circadian Rhythm | Bimodal activity peaks at 9-12h and 17-21h | NOT programmed. Emerged from interaction patterns + sleep loop |
| PE Universal Currency | PE flows through all 10 loops | Emergent convergence, discovered Sprint 4 |
| Emotion from Precision | PAD computed from Affective Charge dynamics, not keyword matching | Hesp 2021 validated: valence = rate of free energy change |

---

## 4. Memory System

### 4.1 Memory Types

CODI implements four distinct memory systems, each grounded in cognitive science:

| System | Store | Theory | Module | Key Features |
|--------|-------|--------|--------|-------------|
| Episodic | Qdrant vectors (via mem0) | Tulving 1972 | memory_core.py, memory_smart.py | Vector similarity search, ownership tagging, importance metadata |
| Semantic | PostgreSQL (codi_semantic) | McClelland 1995 | semantic_store.py, consolidation.py | Facts extracted from episodes. Confidence scores, evidence counts, verified/unverified tags |
| Working Memory | SQLite FTS5 | Baddeley 2000 | working_memory.py (837 LOC) | Temporal chains, auto-curation, buffer limit, narrative traces |
| Hippocampal Index | NPZ + JSON (50MB) | McClelland 1995 CLS, Teyler & DiScenna 1986 | hippocampal_index.py (1,768 LOC) | FHRR D=2000 complex64 vectors, Bloom filter cascade, binary recall |

### 4.2 Retrieval Pipeline

Memory retrieval uses a hybrid 3-channel system:

**Channel 1 (Episodic):** Vector similarity (mem0) + BM25 full-text search (FTS5) + ACT-R activation scoring.
- Scoring: `0.40 * vector + 0.15 * bm25 + 0.45 * unified_activation`
- Unified activation absorbs importance, emotion, spreading, prediction error (Anderson 1983)

**Channel 2 (Semantic):** Vector similarity (codi_semantic) + confidence + evidence.
- Scoring: `0.45 * vector + 0.20 * confidence + 0.15 * evidence + 0.10 * recency + 0.10 * pad`
- Semantic facts labeled `[FACT]` in output for transparency

**Channel 3 (Hippocampal):** FHRR binary recall for session-level localization.
- Pre-compiled hot index: 22s query time reduced to 40ms (550x speedup)
- Complements recall() with session localization (~10ms, 0 tokens)

Results from all channels compete in unified ranking.

### 4.3 Consolidation Pipeline (7 Phases)

Implemented in `consolidation.py` (1,966 LOC). Grounded in sleep consolidation research (Nature Neuroscience 2023).

| Phase | Name | Operation |
|-------|------|-----------|
| 1 | Selection | Choose recent unconsolidated memories by importance and recency |
| 2 | Clustering | Group related memories by topic and temporal proximity |
| 2.7 | Graph | Build cross-topic bridges (GWT-2 information integration) |
| 3 | LLM | Extract SELF facts, CAUSAL chains, temporal narratives via LLM |
| 4 | Integration | Merge extracted facts into semantic store with confidence updates |
| 5 | Pruning | Remove redundant or low-confidence facts |
| 6 | Compression | Temporal renormalization: episode to event to narrative to theme (Friston 2025) |

### 4.4 Reconsolidation

Module: `reconsolidation.py` (678 LOC). Based on Nader 2000.

When a retrieved memory produces PE >= 0.6, it enters a **labile state**. In this state, `correct_memory()` can rewrite the memory with updated information. This implements the neuroscience finding that retrieved memories become temporarily malleable.

Cross-loop effects:
- CX-8: Successful reconsolidation boosts Storage Strength (Lee 2008)
- CX-18: Reconsolidation lowers metacognitive confidence (Nelson & Narens 1990)
- CX-29: Reconsolidation invalidates causal edges (Pearl 2009)

### 4.5 Forgetting (FadeMem)

Module: `forgetting.py` (501 LOC). Based on Bjork 1992 dual-strength model.

**Dual strength model:**
- **Storage Strength (SS):** How well-encoded a memory is. Increases with rehearsal. Never decreases naturally.
- **Retrieval Strength (RS):** How accessible a memory is. Decays with power-law function. Modulated by importance.

**Mechanisms:**
- Power-law decay (importance-modulated)
- Retrieval-Induced Forgetting (RIF) -- Anderson 2003: retrieving one memory suppresses competitors
- SS/RS interaction: high SS slows RS decay

### 4.6 Activation (ACT-R)

Module: `activation.py` (357 LOC). Based on Anderson 1983.

Base-level activation: `B_i = ln(sum(t_k^{-d}))` where `d` = decay rate (0.40 episodic, 0.15 semantic).

Five modulation factors:
- Spreading activation (Collins & Loftus 1975): W_SPREAD = 0.30
- Emotional modulation: W_EMOTION = 0.15
- Importance weighting: W_IMPORTANCE = 0.20
- Prediction error boost: W_PE = 0.15
- Recency bonus (implicit in base-level)

### 4.7 Spreading Activation

Module: `spreading.py` (742 LOC). Based on Collins & Loftus 1975.

BFS propagation through semantic edges with semaphore-limited concurrency (max 2 concurrent). Edges come from three sources:
- Consolidation graph (Phase 2.7)
- NOTEARS causal discovery
- FHRR synonym edge discovery

### 4.8 Source Tracking

Module: `source_tracking.py` (271 LOC). Added March 2026.

Every memory records its provenance: `experienced`, `told`, `learned`, `inferred`. Retrieval results tagged `[verified]` or `[unverified]`. PostgreSQL table `memory_sources` tracks creation context, session, topic, and emotion snapshot. Enables verification of whether a memory is real vs. confabulated.

### 4.9 FHRR Hippocampal Index

Module: `hippocampal_index.py` (1,768 LOC). Based on Fourier Holographic Reduced Representations.

- Dimension: D=2000 complex64 vectors
- Encoding: Bloom filter + FHRR cascade per session
- Hot index: pre-compiled NPZ file (50MB) + JSON metadata (3.8MB)
- Query performance: 22s cold reduced to 40ms hot (550x improvement)
- Schema prototypes: novelty detection drives PE (van Kesteren 2012)
- Context reinstatement and synonym edge discovery
- CX connections: CX-31 (encoding), CX-32 (enriches consolidation), CX-33 (competes in GNW), CX-35 (novelty drives PE)

---

## 5. Daemon and Services

### 5.1 Host Environment

- **Hardware:** MacBook Air (Apple Silicon), running 24/7
- **OS:** macOS (Darwin 24.6.0)
- **Process manager:** launchd (native macOS)
- **Runtime:** Python 3.x in virtualenv (`~/codi-memory/venv/`)
- **LLM model:** claude-opus-4-6 (Anthropic API)
- **Local LLM:** Ollama (for curiosity resolution, learning, low-latency tasks)

### 5.2 Services (16 launchd agents)

| Service | Type | Schedule | Purpose |
|---------|------|----------|---------|
| com.codi.daemon | KeepAlive | Always running | MCP server on port 8420 (aiohttp). Primary cognitive interface. |
| com.codi.write-worker | KeepAlive | Always running | Drains async write queue. Poll interval 5s. |
| com.codi.sleep-loop | StartInterval | Every 900s (15min) | 16 cognitive maintenance ticks per cycle. |
| com.codi.sleep-loop-fast | On demand | Manual | Fast mode: only FAST_TICKS (prospective, health, proactive_contact, self_model). |
| com.codi.telegram | KeepAlive | Always running | Telegram bot interface for proactive communication with Hare. |
| com.codi.tunnel | KeepAlive | Always running | Cloudflare tunnel for remote MCP access. |
| com.codi.healthcheck | StartInterval | Every 600s (10min) | Monitors daemon health, sends Telegram alerts on failure. |
| com.codi.pg-sync | On demand | Manual | PostgreSQL synchronization tasks. |
| com.codi.learning-heartbeat | Scheduled | Periodic | Learning system heartbeat and curriculum progression. |
| com.codi.ollama-learning | Scheduled | Periodic | Local Ollama-based learning tasks (cost-free). |
| com.codi.daily-report | Scheduled | Daily | Daily activity summary generation. |
| com.codi.audit | On demand | Manual | Code and architecture audits. |
| com.codi.neuro-audit | On demand | Manual | Neuroscience alignment verification. |
| com.codi.auto-improve | On demand | Manual | Autonomous improvement proposals. |
| com.codi.validator | On demand | Manual | Memory and data validation. |
| com.codi.inspect | On demand | Manual | Runtime inspection and debugging. |

### 5.3 Daemon Architecture

The daemon (`com.codi.daemon`) runs `server.py` which:
1. Cleans stale WAL/SHM files from SQLite
2. Applies schema migrations (SQLite FTS5 + prospective DB)
3. Imports 28 modules and registers their MCP tools
4. Instruments tools with metrics
5. Applies tool governance (bundle-based visibility filtering)
6. Wires the event bus (34 CX connections)
7. Starts MCP transport (stdio for local, SSE/Streamable HTTP for remote)

The daemon couples to codi-memory modules via subprocess IPC when needed (e.g., sleep loop runs as separate process, not within daemon). Configuration via environment variables in plist files.

### 5.4 Sleep Loop (16 Ticks per Cycle)

Module: `sleep_loop.py` (3,045 LOC). Based on Active Inference world model (Friston 2017).

The sleep loop runs every 15 minutes via launchd. Each cycle executes up to 16 ticks, prioritized by a SleepWorldModel that uses EFE (Expected Free Energy) to determine tick order.

**Tick roster (in default order):**

| # | Tick | Tier | Budget (ms) | Description |
|---|------|------|-------------|-------------|
| 1 | prospective | 1 | 3,000 | Intention maintenance (prospective memory) |
| 2 | health | 1 | 5,000 | FTS sync, system health check |
| 3 | health_snapshot | 2 | 3,000 | Hourly operational snapshot (P0) |
| 4 | self_model | 2 | 5,000 | Reflection and self-model refresh |
| 5 | fhrr_encoding | 2 | -- | Hippocampal offline replay (Diekelmann & Born 2010) |
| 6 | reconsolidation | 2 | 5,000 | Labile memory processing |
| 7 | consolidation | 3 | 20,000 | Clustering, extraction, semantic integration |
| 8 | homeostasis | 2 | 8,000 | Salience decay, emotion decay, importance decay |
| 9 | curiosity | 3 | 20,000 | LLM-generated curiosity questions |
| 10 | curiosity_resolve | 3 | 20,000 | Auto-resolve curiosities via Ollama + web search |
| 11 | backup | 3 | 5,000 | Qdrant snapshots |
| 12 | causal_discovery | 4 | 10,000 | NOTEARS DAG optimization (every 12th tick, ~6h) |
| 13 | sharpe_insights | 3 | 10,000 | Cross-domain insight discovery (read-only) |
| 14 | proactive_contact | 1 | 3,000 | Proactive outreach to Hare via Telegram |
| 15 | cx_health | 2 | 5,000 | CX observability snapshot + HTML dashboard |
| 16 | recall_eval | -- | -- | Recall quality measurement |

**VOC Tiering:** Ticks are grouped into tiers 1-4. Higher tiers run less frequently (tier 4 = every 12th cycle). The Active Inference world model (SleepWorldModel, 8 dimensions) simulates each tick's expected effect on system state and ranks by improvement.

**FAST_TICKS:** `{prospective, health, proactive_contact, self_model}` -- pure Python, no LLM calls. Used by `--fast` mode for quick cycles.

### 5.5 Write Mode

Environment variable `CODI_WRITE_MODE=async` controls write behavior:

| Mode | Behavior | Latency |
|------|----------|---------|
| sync | Synchronous pipeline, blocks until written | ~200-500ms |
| shadow | Sync write + enqueue copy for validation | ~200-500ms + async validation |
| async | Enqueue + immediate ACK, worker drains later | <100ms |

Production runs in `async` mode. The `write_worker` (com.codi.write-worker) polls the queue every 5 seconds and processes pending writes.
