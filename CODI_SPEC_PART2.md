# CODI Technical Product Specification v1 -- Sections 6-10

> Snapshot: March 17, 2026 | Pre-restructuring baseline
> Companion to CODI_SPEC_PART1.md (Sections 1-5)

---

## 6. DATA LAYER

CODI uses five storage backends, each chosen for a specific computational property. No single store owns all data; modules compose across stores as needed.

### 6.1 Storage Matrix

| Store | File / Connection | Modules Touching | Purpose |
|-------|-------------------|-----------------|---------|
| SQLite FTS5 | `memories_fts.db` | 35 | BM25 full-text search, state tables, event tracking, working memory, goals, intentions, CX registry, pet state |
| PostgreSQL + pgvector | via `pg_store.py` | 32 | Episodic + semantic vector storage, memory sources, consolidation state, causal edges, FHRR metadata |
| Qdrant | via `qdrant_utils.py` | 3-4 | Vector similarity search (abstracted behind mem0 and semantic_store) |
| NumPy NPZ | `fhrr_hot_index.npz` | 3 | FHRR holographic session encodings, hot index (50MB, 1554 sessions), binary recall |
| JSON | config files, state files | 38 | Configuration, serialization, trigger definitions, backup snapshots |

### 6.2 Gateway Modules

| Module | Role |
|--------|------|
| `config.py` | SQLite paths, feature flags, environment resolution |
| `config_pg.py` | PostgreSQL connection parameters, pool settings |
| `db_pool.py` | Async connection pool management for PostgreSQL |
| `pg_store.py` | PostgreSQL CRUD operations, vector queries, migrations |
| `qdrant_utils.py` | Qdrant client wrapper, collection management |
| `fts_safety.py` | SQLite FTS5 write guards, WAL management |

### 6.3 Data Flow

Write path (async): `MCP tool` -> `write_queue.py` (enqueue) -> `write_worker` (drain) -> `pg_store` + `FTS5` + `Qdrant`
Read path (sync): `MCP tool` -> `retrieval (3-channel hybrid)` -> `FTS5 BM25` + `pgvector cosine` + `ACT-R activation` -> merged ranking

### 6.4 Planned: CognitiveStore

A unified interface (`CognitiveStore`) is planned for Phase 2 of the restructuring to abstract storage access behind a single API, enabling backend swaps without module changes. Current direct-access patterns across 87 modules will migrate to this interface.

---

## 7. MEASUREMENT AND EVALUATION

### 7.1 Test Suite

| Metric | Value |
|--------|-------|
| Total tests | 1,804 |
| Test files | 84 (in `tests/`) |
| Pass rate | 100% (CI gate) |

**Test categories:**

| Category | Files | What It Validates |
|----------|-------|-------------------|
| Consciousness contracts | `test_butlin.py` | 14/14 Butlin indicators implemented |
| Cross-loops | `test_dual_shadow_wire.py`, `test_dual_report.py` | CX event propagation, cascade integrity |
| Sleep cycle | `test_tier1_smoke.py`, `test_tier1_flush_consolidation.py` | Tick execution, VOC tiering, budget compliance |
| Neuro invariants | `test_phase3_wave1.py` | Constants match literature values |
| Security | `test_key_security.py`, `test_server_security.py` | No secrets in code, API key redaction |
| Architecture | `test_repo_hygiene.py`, `test_schemas.py` | Import structure, schema validity |
| Performance | `test_performance.py` | Latency budgets, memory bounds |

### 7.2 Consciousness Evaluation

**Butlin et al. (2023/2025) -- 14 Indicators:**
All 14 implemented. See Section 4 (Consciousness Loops) for per-indicator module mapping.

**Evaluation Harness:**
- 11/11 PASS across all consciousness contract tests
- PCI proxy: 0.037 (via `cx_observability.py` diversity index)
- Interactive map: `codi-consciousness-loops.html` (45 connections, 31 cross-loops)

### 7.3 Performance Budgets

| Metric | Target | Actual | Module |
|--------|--------|--------|--------|
| L0 prediction budget | 60s max | 15s typical | `prediction.py` |
| Consolidation cap | 30s max | 6s typical | `consolidation.py` |
| Sleep loop cycle | 30 min / 15 ticks | Compliant | `sleep_loop.py` |
| FHRR binary recall | <100ms | 40ms (hot index) | `hippocampal_index.py` |
| Write latency (async) | <100ms ACK | Compliant | `write_queue.py` |

### 7.4 Retrieval Quality

| Metric | Value |
|--------|-------|
| Recall @5 | 60% |
| Recall @10 | 90% |
| False memories | 0 |
| Retrieval channels | 3 (vector, BM25, ACT-R activation) |

### 7.5 Breck ML Test Score (Partial)

Applied categories from Breck et al. (2017): feature expectations (yes), data pipeline tests (yes), training reproducibility (partial -- no local model yet), model staleness (N/A), fairness (N/A). Full score pending LLM independence phase.

### 7.6 Cognitive Evaluator Skill

In development. 5 tracks, 20-24 curriculum items, ~150 papers. Designed to provide systematic measurement of consciousness indicators beyond the current binary pass/fail harness. Tracks: consciousness measures, integration metrics, temporal dynamics, self-model fidelity, causal reasoning quality.

---

## 8. API REFERENCE

CODI exposes 131 MCP tools organized into 14 domains. All tools are registered via FastMCP's `mcp.tool()` decorator across 30+ module files.

### 8.1 Memory (Core)

| Tool | Module | Description |
|------|--------|-------------|
| `recall` | interface.py | Macro search: auto, memory, theme, ownership, emotion, timeline modes |
| `remember` | interface.py | Macro save: working memory + optional long-term with dedup |
| `add_memory` | memory_core.py | Direct episodic memory storage |
| `add_memory_smart` | memory_smart.py | Dedup-aware storage with relate threshold |
| `search_memory` | memory_core.py | Hybrid 3-channel retrieval (vector + BM25 + activation) |
| `get_all_memories` | memory_core.py | List all memories with pagination |
| `get_critical_memories` | memory_core.py | Identity and high-importance memories |
| `search_by_theme` | memory_core.py | Theme-filtered search |
| `search_by_ownership` | memory_core.py | Source-tagged search (experienced, told, learned, inferred) |
| `get_my_experiences` | memory_core.py | First-person episodic memories |
| `update_memory_importance` | memory_core.py | Modify importance level |
| `delete_memory` | memory_core.py | Remove single memory (guarded) |
| `delete_by_content` | memory_core.py | Content-match deletion (guarded) |
| `clear_all_memories` | memory_core.py | Full wipe (guarded, requires confirmation) |
| `restore_memories` | memory_core.py | Restore from backup JSON |
| `get_project_timeline` | memory_core.py | Project-scoped chronological view |
| `sync_fts_index` | memory_smart.py | Rebuild FTS5 index from PostgreSQL |
| `get_memory_source` | source_tracking.py | Provenance: where/when a memory was created |

### 8.2 Context and Working Memory

| Tool | Module | Description |
|------|--------|-------------|
| `context_snapshot` | interface.py | Light or full state snapshot |
| `get_working_memory` | working_memory.py | Active short-term buffer, ranked by score |
| `push_to_working_memory` | working_memory.py | Insert into working memory with auto-chain |
| `update_working_memory` | working_memory.py | Modify relevance or archive |
| `get_narrative_chain` | working_memory.py | Retrieve temporal chain by topic or ID |
| `link_narrative_trace` | working_memory.py | Link chains into meta-narratives |

### 8.3 Consciousness and Emotion

| Tool | Module | Description |
|------|--------|-------------|
| `get_emotional_state` | emotion.py | Current PAD vector + optional history |
| `get_emotional_expression` | emotion.py | Natural language emotional state |
| `set_emotional_state` | emotion.py | Manual PAD override (debug only) |
| `update_mood_baseline` | emotion.py | Adjust long-term mood |
| `apply_emotional_decay` | emotion.py | Time-based emotional decay |
| `add_memory_with_emotion` | emotion.py | Store memory with explicit emotion tag |
| `tag_memory_emotion` | emotion.py | Retroactive emotion tagging |
| `search_by_emotion` | emotion.py | Emotion-filtered retrieval |
| `get_emotional_memories` | emotion.py | Memories with strongest emotional charge |
| `focus_attention` | workspace.py | Direct attentional spotlight |
| `broadcast_to_workspace` | workspace.py | GNW broadcast to workspace slots |
| `get_workspace_state` | workspace.py | Current workspace contents |
| `apply_salience_decay` | workspace.py | Time-based salience reduction |
| `get_high_salience_memories` | workspace.py | Above-threshold salience items |
| `emotional_focus_attention` | workspace.py | Emotion-weighted attention |

### 8.4 Goals

| Tool | Module | Description |
|------|--------|-------------|
| `crear_goal` | goals.py | Create goal with what/why/next_step |
| `ver_goals` | goals.py | List goals ranked by ACT-R activation |
| `actualizar_goal` | goals.py | Update derivable fields (last_state, next_step) |
| `completar_goal` | goals.py | Mark complete with outcome |
| `arbol_goals` | goals.py | Hierarchical tree view |
| `contexto_goals` | goals.py | Top goals above interference threshold |

### 8.5 Prospective Memory

| Tool | Module | Description |
|------|--------|-------------|
| `crear_intencion` | prospective.py | Event/time/condition-triggered intentions |
| `ver_intenciones` | prospective.py | List pending intentions |
| `completar_intencion` | prospective.py | Mark intention fulfilled |
| `check_prospective_triggers` | prospective.py | Evaluate triggers against current context |
| `snooze_intencion` | prospective.py | Delay intention trigger |
| `delete_intencion` | prospective.py | Remove intention |

### 8.6 Session Lifecycle

| Tool | Module | Description |
|------|--------|-------------|
| `despertar_codi` | lifecycle.py | Executive briefing on session start |
| `ciclo_vida` | lifecycle.py | Time-of-day lifecycle (morning/afternoon/night/dawn) |
| `consolidate_recent` | lifecycle.py | On-demand consolidation of recent memories |
| `find_connections` | lifecycle.py | Discover inter-memory connections |
| `dream_consolidation` | lifecycle.py | Sleep-mode deep consolidation |
| `get_memory_connections` | lifecycle.py | View discovered connections |
| `flush_session` | flush.py | Pre-compaction state dump |
| `checkpoint_memoria` | flush.py | Moment-specific checkpoint |
| `get_session_stats` | session_bridge.py | Current session metrics |

### 8.7 Prediction and Surprise

| Tool | Module | Description |
|------|--------|-------------|
| `predict_context` | prediction.py | Generate L0-L3 predictions |
| `record_surprise` | prediction.py | Log prediction error event |
| `get_prediction_accuracy` | prediction.py | Accuracy metrics across levels |
| `update_beliefs` | prediction.py | Bayesian belief update |
| `detectar_sorpresa` | curiosity.py | Surprise detection from input |

### 8.8 Consolidation and Semantic

| Tool | Module | Description |
|------|--------|-------------|
| `run_consolidation` | consolidation.py | Execute 7-phase consolidation pipeline |
| `correct_memory` | consolidation.py | PE-triggered reconsolidation rewrite |
| `get_semantic_facts` | consolidation.py | Extracted semantic knowledge |
| `get_consolidation_stats` | consolidation.py | Pipeline execution metrics |
| `get_pending_corrections` | consolidation.py | Queued reconsolidation targets |

### 8.9 Self-Model and Metacognition

| Tool | Module | Description |
|------|--------|-------------|
| `reflect_on_self` | self_model.py | HOT self-reflection |
| `assess_confidence` | self_model.py | Metacognitive confidence assessment |
| `identify_knowledge_gaps` | self_model.py | Gap detection in knowledge |
| `update_self_model` | self_model.py | Refresh self-model state |
| `get_self_model_summary` | self_model.py | Current self-model snapshot |
| `get_user_model_summary` | user_model.py | Model of user (Hare) preferences |

### 8.10 Curiosity

| Tool | Module | Description |
|------|--------|-------------|
| `generar_curiosidad` | curiosity.py | Generate curiosity-driven questions |
| `push_curiosidad` | curiosity.py | Add curiosity item to queue |
| `get_curiosidades` | curiosity.py | List active curiosity items |
| `resolve_curiosidad` | curiosity.py | Mark curiosity resolved |
| `analizar_patron_trabajo` | curiosity.py | Work pattern analysis |
| `get_curiosity_quality` | curiosity.py | Curiosity resolution quality metrics |

### 8.11 Hippocampal Index (FHRR)

| Tool | Module | Description |
|------|--------|-------------|
| `binary_recall_tool` | hippocampal_index.py | Fast session localization (~40ms, 0 tokens) |
| `compile_fhrr_index` | hippocampal_index.py | Recompile hot index from session encodings |
| `get_fhrr_stats` | hippocampal_index.py | Index statistics (sessions, size, accuracy) |

### 8.12 Observation and Analytics

| Tool | Module | Description |
|------|--------|-------------|
| `get_cx_health` | cx_observability.py | Cross-loop fire counts, diversity, cascades |
| `get_sharpe_report` | sharpe.py | Sharpe Cognitive ratio across episodic memories |
| `get_sharpe_insights_report` | sharpe_insights.py | Cross-domain insight generation |
| `get_recall_eval_report` | recall_eval.py | Recall quality evaluation metrics |
| `get_spreading_graph` | spreading.py | Spreading activation network state |
| `get_spreading_stats` | spreading.py | Activation propagation metrics |

### 8.13 System and Maintenance

| Tool | Module | Description |
|------|--------|-------------|
| `verificar_salud_memoria` | lifecycle.py | Memory health check (write + read test) |
| `get_runtime_flags` | interface.py | Current write mode, flags, thread state |
| `get_toolset_status` | tool_governance.py | Active toolset, bundles, visible tools |
| `write_status` | write_queue.py | Async write queue status |
| `cancel_write` | write_queue.py | Cancel pending write |
| `run_maintenance` | maintenance.py | Scheduled maintenance tasks |

### 8.14 Triggers, Training, and Pet

| Tool | Module | Description |
|------|--------|-------------|
| `evaluar_triggers` | triggers.py | Pattern-based trigger evaluation |
| `activar_trigger` | triggers.py | Manual trigger activation |
| `listar_triggers` | triggers.py | List configured triggers and patterns |
| `pet_status` | pet.py | Digital pet current state |
| `adopt_pet_tool` | pet.py | Adopt new digital pet |
| `care_for_pet_tool` | pet.py | Pet care actions (feed, play, rest, clean, medicine) |
| `guardar_ejemplo_training` | training.py | Save training example for future fine-tuning |
| `auto_learn_from_session` | learning.py | Extract learning from session |
| `audit_tools` | learning.py | Tool usage audit |
| `trigger_n8n` | n8n.py | Trigger n8n webhook |

---

## 9. NEUROSCIENCE FOUNDATION

CODI's architecture is grounded in peer-reviewed neuroscience and cognitive science. Every computational module maps to one or more published theories. The system draws from approximately 355 papers across 13 study tracks (all 5 courses complete, 212 curriculum items).

### 9.1 Core Theories

| Theory | Key Papers | CODI Implementation |
|--------|-----------|---------------------|
| Global Workspace Theory | Baars 1988; Dehaene & Changeux 2011; Dehaene 2014 | `competition.py`: 5-phase GNW (attention, coalition, ignition, softmax, recurrent). 5 workspace slots, ignition threshold 0.25 |
| Predictive Processing | Clark 2013, 2015; Friston 2008, 2010; Kiebel 2008 | `prediction.py`: 4-level hierarchy (L0 turn, L1 session, L2 meta, L3 project) + HGF adaptive precision |
| Active Inference | Friston 2017; Parr, Pezzulo, Friston 2022 | `active_inference.py`: EFE policy selection, Dirichlet-Multinomial, Options Framework (Sutton 1999) |
| IIT (proxy measures) | Tononi 2004; Albantakis 2023; Barrett & Seth 2011 | `cx_observability.py`: PCI proxy via diversity index (true Phi is NP-hard) |
| Attention Schema Theory | Graziano 2013; Webb & Graziano 2015 | `workspace.py`: S+A+V attention schema model |
| Higher-Order Theory | Rosenthal 2005; Nelson & Narens 1990 | `self_model.py`: HOT refresh every 50 interactions, metacognitive sweep every 10 turns |
| Memory Consolidation | McClelland 1995; Nader 2000; Nature Neurosci 2023 | `consolidation.py`: 7 phases; `reconsolidation.py`: PE>=0.6 triggers labile state |
| ACT-R | Anderson 1983, 2007 | `activation.py`: base-level B_i = ln(sum(t_k^{-d})) + 5 modulations (spread, emotion, importance, PE, prediction) |
| Dual-Strength Forgetting | Bjork 1992; Anderson 2003 | `forgetting.py`: FadeMem power-law decay, SS/RS, retrieval-induced forgetting |
| Causal Discovery | Zheng 2018 (NOTEARS); Pearl 2009 | `causal_discovery.py`: continuous DAG optimization; `counterfactual.py`: Pearl 3-step |
| Temporal Renormalization | Friston 2025 | `narrative.py`: episode -> event -> narrative -> theme hierarchy |
| Bayesian Surprise | Itti & Baldi 2009; Mathys 2011 | `prediction.py`: KL-divergence PE; HGF adaptive precision weighting |
| Spreading Activation | Collins & Loftus 1975 | `spreading.py`: BFS propagation, semaphore-limited |

### 9.2 Module-to-Paper Mapping

| Module | Primary Paper(s) | What It Computes |
|--------|-----------------|------------------|
| `activation.py` | Anderson 1983 | ACT-R base-level activation + 5 modulations |
| `competition.py` | Baars 1988, Dehaene 2014 | 5-phase GNW competition |
| `prediction.py` | Friston 2008, Kiebel 2008, Mathys 2011 | 4-level hierarchical prediction + HGF |
| `consolidation.py` | Nature Neurosci 2023, Nader 2000 | 7-phase consolidation + causal chains |
| `forgetting.py` | Bjork 1992, Anderson 2003 | Power-law decay, SS/RS, RIF |
| `emotion.py` | Russell 1977, Hesp 2021 | PAD from precision dynamics (not keywords) |
| `active_inference.py` | Friston 2017, Sutton 1999 | EFE policy + Dirichlet-Multinomial + Options |
| `self_model.py` | Rosenthal 2005, Graziano 2013 | HOT + Attention Schema + capability model |
| `causal_discovery.py` | Zheng 2018 | NOTEARS continuous DAG + typed edges |
| `counterfactual.py` | Pearl 2009 | Bilingual parser + abduction -> intervention -> prediction |
| `curiosity.py` | Schmidhuber 2010, Gottlieb 2013 | PE-driven exploration, Goldilocks zone |
| `working_memory.py` | Baddeley 2000 | Temporal chains, buffer curation |
| `narrative.py` | Friston 2025 | Temporal renormalization hierarchy |
| `thompson_sampling.py` | Russo et al. 2018 | Beta-posterior retrieval channel selection |
| `neuro_invariants.py` | Multiple | Literature-validated constants |
| `hippocampal_index.py` | Plate 2003 (FHRR) | Holographic reduced representations for session encoding |

### 9.3 Empirical Validation

- **COGITATE Consortium (Nature 2025)**: IIT scored 2/3 predictions, GWT 0/3 as neural correlate. Our GNW implementation is validated as a useful processing mechanism, not a correlate of biological consciousness.
- **Noetic-Autonoetic Gap**: Semantic self-knowledge (0.88) significantly exceeds episodic self-knowledge (0.64), consistent with dual-process theories.
- **Emergent Circadian Rhythm**: Bimodal activity peaks at 9-12h and 17-21h emerged from usage patterns, not from programming.
- **PE as Universal Currency**: Prediction error flows through all 10 loops as an emergent integration mechanism.

---

## 10. ROADMAP

### 10.1 Architectural Restructuring (5 Phases)

| Phase | Name | Status | Description |
|-------|------|--------|-------------|
| 1 | Observability Contracts | Next | Define typed interfaces for all inter-module communication |
| 2 | CognitiveStore | Planned | Unified data access layer replacing 87 direct-access patterns |
| 3 | Core Library Extraction | Planned | Separate cognitive algorithms from MCP transport |
| 4 | Transport Separation | Planned | MCP becomes one of N possible transports (HTTP, gRPC, direct) |
| 5 | Module Decomposition | Planned | Break large modules (consolidation 1952 LOC, self_model 1346 LOC) into focused units |

### 10.2 Cognitive Evaluator Skill

Currently in study phase. 5 evaluation tracks, 20-24 curriculum items, ~150 papers. Will provide systematic measurement beyond binary pass/fail, including integration metrics (proxy Phi), temporal dynamics quality, self-model fidelity, and causal reasoning accuracy.

### 10.3 LLM Independence

Goal: replace external LLM calls with classical algorithms and local 3B-7B models.
- Sprint 1 complete: `classify_edges` moved to classical algorithm (0 tokens)
- Next: passive logging in `llm_router.py` to collect training data
- Blocked on: Phase 3 restructuring (Core Library) to decouple LLM from cognitive core
- Target: zero recurring API cost for core cognitive operations

### 10.4 Product

- **First pilot client**: Sebastian (onboarding planned post-restructuring)
- **Dashboard**: visual redesign of `codi-consciousness-loops.html` with live data from daemon via `cx_observability.py`
- **Technical Specification**: this document serves as v0 snapshot; v1 will reflect post-restructuring architecture

### 10.5 FHRR Hippocampal Index

Functionally complete. Hot index: 50MB, 1554 sessions, 40ms query time (550x improvement from 22s). Integrated into sleep-loop tick for auto-recompilation. Remaining: validate hot index recompile in live sleep-loop tick after next daemon restart.

### 10.6 Known Gaps

| Gap | Blocker | Target |
|-----|---------|--------|
| Information Gain curiosity | Sprint 10 design | Replace PE-only with IG-optimized exploration |
| EFE gate for memory storage | Sprint 13 | Store only what reduces model uncertainty |
| Unified prediction algorithm | Sprint 11 | Single algorithm across L0-L3 |
| BMR scoring | Sprint 9 | Replace cosine thresholds with Bayesian model reduction |
| Source monitoring in reconsolidation | Implementation gap | Reconsolidation should update memory source provenance |
| Metacognition verified-flag usage | Implementation gap | Metacognition should weight verified memories higher |
