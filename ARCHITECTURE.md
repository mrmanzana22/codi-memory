# CODI Architecture Map

> Generated: March 18, 2026 | Post-restructuring (5/5 phases complete)
> 102 modules, 51K LOC, 1812 tests, 133 MCP tools, 34 CX cross-loops

---

## System Diagram

```
LLM Session (Claude Code / Telegram / API)
        │ MCP Protocol (stdio/SSE)
        ▼
server.py ─── Orchestrator (imports all modules, register_tools)
        │
        ▼
modules/ ─── 102 files, 51K LOC
├── core/        ← Pure functions (time, paths, constants, pad)
├── store/       ← Data access (8 domain stores, @store_traced)
├── wiring/      ← Event bus (56 handlers, 34 CX cross-loops)
├── sleep_loop/  ← Background ticks (22 ticks, SleepWorldModel)
└── 81 modules   ← Cognitive + infrastructure
        │
        ▼
Data: PostgreSQL (pgvector) + SQLite (FTS5) + JSON files
```

---

## Layer Architecture

```
MCP Transport ─── JSON wrappers, register_tools(mcp)
Business Logic ── _impl functions (25), return dicts
Store Layer ───── modules/store/ (8 stores, SQL queries)
Core Layer ────── modules/core/ (pure functions, no I/O)
```

---

## Complete Module Inventory (102 files)

### PACKAGES

#### core/ — Pure Functions (0 I/O, 0 state)
| File | What | Used By |
|------|------|---------|
| time.py | TZ_COL, now_col, now_iso, now_short | Every module |
| paths.py | BASE_DIR, FTS_DB_PATH, all paths | config, db_pool |
| constants.py | Thresholds, limits, policy values | config |
| pad.py | PAD: clamp, classify, intensity, emotion text | emotion, utils |
| classification.py | Importance weights, theme inference | memory_core, utils |
| analysis.py | Confidence scoring | utils |

#### store/ — Data Access (SQL only)
| File | Domain | Key Methods |
|------|--------|-------------|
| cx_store.py | CX metrics | get_aggregated_cx_metrics, get_gnw_metrics |
| prediction_store.py | Predictions | get_accuracy, get_surprise_scores |
| consolidation_store.py | Consolidation | get_consolidation_metrics, get_reconsolidation_stats |
| metacognition_store.py | Metacognition | get_calibration_metrics |
| causal_store.py | Causal DAGs | get_latest_dag_state |
| attention_store.py | Attention | get_policy_diversity |
| forgetting_store.py | Forgetting | get_forgetting_metrics |
| sleep_store.py | Sleep loop | get_tick_history |

#### wiring/ — Event Bus (3784 LOC, 56 handlers)
| Handler Group | CX IDs | Connects |
|--------------|--------|----------|
| Core events (memory, workspace, emotion, consolidation) | — | memory_core, working_memory |
| Attention schema (S+A+V model) | — | self_model, active_inference |
| FHRR hippocampal | CX-32,33,35 | hippocampal_index |
| Curiosity-PE coupling | CX-1,2 | curiosity, prediction |
| Consolidation urgency | CX-4a,4b,14 | forgetting, curiosity |
| GNW→Action | CX-5,9,16 | active_inference, self_model |
| Self-model loops | CX-3,10,15 | competition, forgetting |
| Causal plasticity | CX-11,29,30 | causal_discovery |
| Forgetting gates | CX-23,28,24 | curiosity, reconsolidation |
| Identity constraints | CX-26,27 | active_inference, causal |
| Registry + ablation API | — | server, cx_observability |

#### sleep_loop/ — Background Ticks (3091 LOC)
| Tick | Tier | What | Connects |
|------|------|------|----------|
| prospective | 1 | Review intentions | prospective.py |
| health | 1 | FTS sync, Qdrant check | db_pool, pg_store |
| proactive_contact | 1 | Telegram notifications | notifier |
| health_snapshot | 2 | Dashboard snapshot | config |
| self_model | 2 | Reflection + identity | self_model.py |
| reconsolidation | 2 | Labile memory processing | reconsolidation.py |
| homeostasis | 2 | Salience/emotion/importance decay | forgetting, emotion |
| fhrr_encoding | 2 | Hippocampal replay | hippocampal_index |
| cx_health | 2 | CX observability | cx_observability |
| recall_eval | 2 | Recall quality | recall_eval.py |
| consolidation | 3 | Episodic→semantic | consolidation.py |
| curiosity | 3 | Question generation (LLM) | curiosity, llm_router |
| curiosity_resolve | 3 | Auto-resolution (Ollama) | curiosity, ollama_router |
| backup | 3 | Snapshots + SQLite backup | pg_store |
| sharpe_insights | 3 | Cross-domain patterns | sharpe_insights |
| cognitive_health | 3 | 31 cognitive metrics | cognitive_contracts |
| causal_discovery | 4 | NOTEARS DAG optimization | causal_discovery |

---

### TOP-LEVEL MODULES (81 files)

#### Memory System
| Module | LOC | Tools | What | Connects |
|--------|-----|-------|------|----------|
| memory_core.py | 1725 | 13 | CRUD: add, search, delete, scroll | pg_store, events |
| memory_smart.py | 1214 | 4 | FTS5, dedup, graph neighbors | pg_store, db_pool |
| pg_store.py | 1080 | 0 | PostgreSQL/pgvector wrapper | config_pg |
| consolidation.py | 1966 | 5 | Episodic→semantic (5 phases). Facade | reconsolidation, semantic_store |
| consolidation_common.py | 120 | 0 | Embeddings, similarity | llm_router |
| consolidation_runner.py | 177 | 0 | Subprocess launcher | consolidation |
| reconsolidation.py | — | 0 | Contradictions, correct_memory | pg_store, events |
| semantic_store.py | 158 | 0 | Semantic memory search | pg_store |
| forgetting.py | — | 0 | FadeMem: power-law, RIF, SS/RS | activation, pg_store |
| activation.py | — | 0 | ACT-R base-level activation | db_pool |
| spreading.py | — | 2 | Spreading activation BFS | pg_store |

#### Prediction & Inference
| Module | LOC | Tools | What | Connects |
|--------|-----|-------|------|----------|
| prediction.py | 766 | 4 | Predict, surprise, beliefs | pg_store, events |
| active_inference.py | — | 0 | EFE policy, Dirichlet-Multinomial | config, wiring |
| active_inference_integration.py | — | 0 | Affective Charge, AC→PAD | emotion |
| competition.py | — | 0 | GNW 5-phase workspace competition | events |

#### Emotion & Self
| Module | LOC | Tools | What | Connects |
|--------|-----|-------|------|----------|
| emotion.py | 895 | 9 | PAD model, mood, text→PAD (9 _impl) | config, pg_store, events |
| self_model.py | 1387 | 5 | Reflection, discrepancy detection | pg_store, working_memory |
| workspace.py | 851 | 6 | GNW broadcast, salience decay | pg_store, events |

#### Curiosity & Causal
| Module | LOC | Tools | What | Connects |
|--------|-----|-------|------|----------|
| curiosity.py | 860 | 7 | Info gap, IG explore/exploit | config, pg_store |
| causal_discovery.py | — | 0 | NOTEARS DAG, augmented Lagrangian | db_pool |
| counterfactual.py | — | 0 | Abduction→intervention→prediction | pg_store |
| learning.py | — | 3 | Curriculum, courses | config |

#### Hippocampal & Retrieval
| Module | LOC | Tools | What | Connects |
|--------|-----|-------|------|----------|
| hippocampal_index.py | 1768 | 3 | FHRR encoding, binary_recall, schemas | db_pool, pg_store |
| retrieval_metadata.py | 1170 | 0 | FOK calibration, retrieval quality | db_pool |
| query_expansion.py | 188 | 0 | Query broadening | pg_store |

#### Observability
| Module | LOC | Tools | What | Connects |
|--------|-----|-------|------|----------|
| cognitive_contracts.py | 1198 | 2 | 12 contracts, 31 metrics | store/ |
| cx_observability.py | — | 1 | CX fires, diversity, cascades | wiring |
| metrics.py | — | 1 | Tool call instrumentation | db_pool |
| neuro_invariants.py | — | 0 | Neuroscience regression checks | sleep_loop |
| health_monitor.py | 241 | 0 | System health | config |
| health_alerts.py | 155 | 0 | Alert thresholds | db_pool |

#### Working Memory & Goals
| Module | LOC | Tools | What | Connects |
|--------|-----|-------|------|----------|
| working_memory.py | 837 | 5 | Items, chains, traces (5 _impl) | config_pg |
| goals.py | 917 | 6 | ACT-R goal agenda, hierarchy | config_pg |
| prospective.py | 978 | 6 | Intentions (event/time triggers) | config_pg |
| narrative.py | — | 0 | Narrative trace mgmt | working_memory |

#### Interface & Governance
| Module | LOC | Tools | What | Connects |
|--------|-----|-------|------|----------|
| interface.py | 931 | 4 | recall, remember, context_snapshot | memory_core, working_memory |
| tool_governance.py | — | 1 | Visibility bundles (full/core/learning) | config |
| session_bridge.py | — | 1 | Session state persistence | working_memory |
| assessment.py | 913 | 0 | Cognitive assessment orchestrator | multiple |

#### Infrastructure
| Module | LOC | Tools | What | Connects |
|--------|-----|-------|------|----------|
| config.py | — | 0 | Central config, mcp instance | core/ |
| config_pg.py | 135 | 0 | PostgreSQL pool (psycopg3) | — |
| db_pool.py | 139 | 0 | SQLite pool | core/paths |
| events.py | — | 0 | EventBus: 23 event types | — |
| migrations.py | 246 | 0 | Schema migrations (029+) | db_pool |
| schemas.py | 267 | 0 | Data validation | — |
| instance_config.py | 211 | 0 | Multi-instance YAML | — |
| secret_redact.py | 89 | 0 | PII redaction | — |
| tracing.py | 22 | 0 | Trace ID generation | — |
| fts_safety.py | 105 | 0 | FTS query sanitization | — |
| destructive_guard.py | 142 | 0 | Confirmation tokens | — |
| utils.py | — | 0 | Ownership, backup, journal | core/ |
| qdrant_utils.py | 69 | 0 | Qdrant scroll helpers | pg_store |

#### Auxiliary
| Module | LOC | Tools | What | Connects |
|--------|-----|-------|------|----------|
| books.py | — | 6 | Knowledge books (6 _impl) | pg_store |
| triggers.py | — | 5 | Pattern webhooks (5 _impl) | config, memory_smart |
| flush.py | — | 4 | Checkpoints, export | pg_store, memory_smart |
| training.py | — | 4 | LoRA training data | config |
| maintenance.py | — | 6 | Periodic tasks | pg_store |
| pet.py | — | 3 | Digital pet | db_pool |
| n8n.py | 87 | 2 | n8n webhooks | config |
| notifier.py | 154 | 0 | Telegram notifications | — |
| user_model.py | — | 1 | User prefs | pg_store |
| agent_model.py | 122 | 0 | Agent modeling | — |

#### Analytics & Routing
| Module | LOC | Tools | What | Connects |
|--------|-----|-------|------|----------|
| sharpe.py | — | 1 | Sharpe Cognitive report | db_pool |
| sharpe_insights.py | — | 1 | Cross-domain insights | db_pool |
| recall_eval.py | — | 1 | Recall quality | db_pool, pg_store |
| spotlight.py | — | 0 | Attention spotlight | working_memory |
| source_tracking.py | — | 1 | Memory provenance | config_pg |
| reward_tracking.py | — | 0 | Action→outcome | db_pool |
| thompson_sampling.py | 174 | 0 | Thompson sampling | — |
| bmr.py | 199 | 0 | Bayesian Model Reduction | — |
| temporal_renorm.py | — | 0 | Importance renorm | pg_store |
| classify_edges.py | 222 | 0 | Edge classification | — |
| llm_router.py | — | 0 | LLM routing | config |
| ollama_router.py | 819 | 0 | Ollama local models | — |
| dual_compare.py | — | 0 | Sync vs async compare | write_queue |
| write_queue.py | — | 2 | Async write queue | db_pool |
| write_worker.py | — | 0 | Queue drain | write_queue |
| access_tracking.py | — | 0 | Memory access events | events |
| pe_actions.py | 229 | 0 | PE-driven actions | events |
| consciousness.py | 93 | 0 | Facade: re-exports | emotion, workspace |

---

## Data Stores

| Store | Tech | Key Tables |
|-------|------|-----------|
| PG episodic | pgvector | codi_memories (vectors + payload) |
| PG semantic | pgvector | codi_semantic (consolidated facts) |
| PG ops | psycopg3 | working_memory, goals, narrative_traces, intentions |
| SQLite FTS | FTS5 | memories_fts, memories_text, fts_retry_queue |
| SQLite ops | same DB | prediction_results, reconsolidation_log, consolidation_log, strength_log, cx_snapshots, tool_calls, fok_calibration_log, metacognition_traces, attention_transitions, sleep_loop_state, causal_discovery_state |
| JSON | files | triggers.json, preguntas_curiosidad.json, libros.json, mantenimiento.json |

---

## Multi-Instance

| Instance | PG Collection | Data Dir | Status |
|----------|--------------|----------|--------|
| Codi (Hare) | codi_memories / codi_semantic | ~/codi-memory/ | Production |
| Sebastian | sebastian_memories / sebastian_semantic | ~/codi-memory/data-sebastian/ | Active |

---

## codi-daemon (~/codi-daemon/)

| Service | Plist | Purpose |
|---------|-------|---------|
| daemon | com.codi.daemon | aiohttp:8420, agent_loop, auto_improve |
| write-worker | com.codi.write-worker | Async write drain (5s poll) |
| sleep-loop | com.codi.sleep-loop | 22 ticks / 30 min |
| telegram | com.codi.telegram | Telegram bot |
