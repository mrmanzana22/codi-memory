# Tool Contract Catalog -- codi-memory MCP Server

**Version**: 1.0.0
**Date**: 2026-02-16
**Author**: Security Engineer Agent
**Scope**: 140+ tools across 20 modules
**DB**: SQLite (memories_fts.db) + Qdrant (codi_memories, codi_semantic)

---

## Table of Contents

1. [System Architecture Overview](#system-architecture-overview)
2. [Write Modes](#write-modes)
3. [Critical Tool Contracts (Full)](#critical-tool-contracts-full)
4. [Medium-Risk Tool Contracts (Brief)](#medium-risk-tool-contracts-brief)
5. [Read-Only Tools (Summary Table)](#read-only-tools-summary-table)
6. [Contract Compliance Checklist](#contract-compliance-checklist)
7. [Definition of Secure Tool](#definition-of-secure-tool)

---

## System Architecture Overview

```
                         MCP Clients (Claude Code, etc.)
                                   |
                            server.py (FastMCP)
                                   |
              +--------------------+--------------------+
              |                    |                    |
        interface.py          memory_core.py       [20 modules]
        (3 macro-tools)       (CRUD, search)
              |                    |
              v                    v
     +--------+--------+    +-----+------+
     | Working Memory   |    | mem0       |
     | (SQLite FTS DB)  |    | (OpenAI +  |
     +------------------+    | Qdrant)    |
                             +-----+------+
                                   |
                    +--------------+--------------+
                    |              |              |
              codi_memories  codi_semantic  memories_fts.db
              (Qdrant)       (Qdrant)       (SQLite FTS5)
                                                |
                                  +-------------+-------------+
                                  |             |             |
                            write_queue  write_queue_log  working_memory
                            (async jobs) (observability)  (short-term)

External Dependencies:
  - OpenAI API: embeddings (text-embedding-3-small), LLM (gpt-4o-mini for consolidation)
  - Supabase: training_examples table (guardar_ejemplo_training)
  - n8n: HTTP webhooks (trigger_n8n)
  - Filesystem: triggers.json, libros.json, memories_backup.json, markdown/, training_data/
  - launchd: sleep_loop (30min), write_worker (30s), consolidation (6h)
```

---

## Write Modes

The async write-path supports three modes, configured via `CODI_WRITE_MODE` env var or `.write_mode` file:

| Mode | Behavior | Latency | Data Safety |
|------|----------|---------|-------------|
| `sync` (default) | Synchronous pipeline: mem0 -> Qdrant -> FTS5 -> events | p50: 200-800ms | Immediate consistency |
| `shadow` | Sync first, then also enqueue for validation | p50: 200-800ms + enqueue overhead | Immediate + async audit trail |
| `async` | Enqueue only, return immediate ACK | p50: 5-20ms | Eventual consistency (write_worker processes) |

**Security note**: In `async` mode, a client receives ACK before data is persisted. If the write_worker dies or the job hits max_attempts (8), data is lost to a `dead` state in write_queue. The `shadow` mode mitigates this by doing sync first.

---

## Critical Tool Contracts (Full)

---

### Tool: remember
- **Type**: write
- **Module**: `/Users/harecjimenez/codi-memory/modules/interface.py`
- **Purpose**: Unified write macro-tool -- stores content in working memory and optionally long-term memory
- **Inputs**:
  - `content`: str (required) -- the content to remember
  - `importance`: str (optional) [default: "auto"] -- auto|critical|high|medium|low; "auto" uses heuristic on content length and keywords
  - `topic`: str (optional) [default: "general"] -- topic/category for grouping
  - `source`: str (optional) [default: "interaction"] -- origin of the information
  - `long_term`: bool (optional) [default: True] -- if False, only working memory
- **Outputs**: JSON string with `pretty` (human-readable summary), `topic`, `importance`, `relevance`, `working_memory` (WM push result), `long_term` (add_memory_smart result or queued ACK), `long_term_enabled` (bool), optional `trace_id`
- **Side Effects**:
  - ALWAYS: SQLite `working_memory` table INSERT (push_to_working_memory)
  - ALWAYS: SQLite `working_memory` UPDATE for auto-curation if buffer > 30 active items
  - IF long_term enabled AND sync mode: mem0 -> Qdrant codi_memories UPSERT, FTS5 index INSERT, event_bus MEMORY_STORED emission
  - IF long_term enabled AND async mode: SQLite `write_queue` INSERT (enqueue_write_job)
  - IF long_term enabled AND shadow mode: Sync pipeline + write_queue INSERT
- **Idempotency**: Dedupe key via `compute_dedupe_key("remember", content)` = SHA256(kind|normalized_content[:500]|YYYY-MM-DD)[:32]. Only effective in async/shadow modes; sync mode has no dedupe and will create duplicate memories for identical content
- **Permissions**: Reads `.write_mode` file or `CODI_WRITE_MODE` env var. Reads/writes SQLite working_memory. Reads/writes Qdrant codi_memories. Reads/writes SQLite write_queue. Calls OpenAI embeddings API (via mem0)
- **Error Handling**: Returns JSON with error string on exception. Working memory push and long-term are independent -- WM failure does not block LT and vice versa. Shadow mode enqueue failures are silently swallowed (print to stderr)
- **SLO**: p50 sync: 300-900ms (dominated by OpenAI embedding call in add_memory_smart). p50 async: 10-30ms. p50 shadow: sync latency + 5ms enqueue
- **Telemetry**: trace_id generated via `new_trace_id()`. Async jobs logged in `write_queue` + `write_queue_log` tables. Events emitted to event_bus
- **Security Controls**:
  - Input validation: importance clamped to VALID_IMPORTANCE set; invalid values fall back to "auto"
  - Content length: no explicit max -- arbitrary-length strings flow to mem0 and OpenAI (token limits enforced upstream)
  - Low-importance short content auto-skipped for long-term (< 120 chars, importance "low", no "recuerda"/"no olvidar" keywords)
  - No authentication gate -- any MCP client can call
- **Abuse Scenarios**:
  1. **Memory flooding**: Rapid-fire calls with long_term=True and importance="critical" can saturate Qdrant, inflate OpenAI costs, and fill SQLite write_queue. No rate limiting exists
  2. **Content injection**: Arbitrary content stored in mem0/Qdrant/FTS5. Malicious content could poison future recall results, influencing agent behavior
  3. **Async queue exhaustion**: In async mode, flooding write_queue with high-priority jobs could starve legitimate writes and fill SQLite disk

---

### Tool: add_memory
- **Type**: write
- **Module**: `/Users/harecjimenez/codi-memory/modules/memory_core.py`
- **Purpose**: Core memory storage -- adds a new memory to mem0/Qdrant with ownership tagging and FTS5 indexing
- **Inputs**:
  - `content`: str (required) -- the content to store
  - `category`: str (optional) [default: "general"] -- identidad, aprendizaje, episodio, proyecto, general
  - `source`: str (optional) [default: "experienced"] -- experienced, told, learned, inferred
  - `importance`: str (optional) [default: "medium"] -- critical, high, medium, low
- **Outputs**: Sync: string "Memoria guardada con ownership: {result}". Async: JSON with job_id, status, dedupe_hit, message
- **Side Effects**:
  - Sync: mem0.add() -> Qdrant codi_memories UPSERT (vector + payload). Qdrant set_payload for ownership metadata. SQLite FTS5 index_memory_fts(). event_bus.emit(MEMORY_STORED)
  - Async: SQLite write_queue INSERT only
  - Shadow: Both sync + write_queue INSERT
- **Idempotency**: Dedupe key `compute_dedupe_key("add_memory", content)` in async/shadow. Sync mode: **no idempotency** -- calling twice with same content creates two distinct Qdrant points with different UUIDs
- **Permissions**: Reads/writes Qdrant codi_memories. Reads/writes SQLite FTS5 tables. Calls OpenAI API (via mem0 embedding). Reads `.write_mode` file
- **Error Handling**: Returns "Error al guardar memoria: {str(e)}" on exception. FTS index failures logged to stderr but do not fail the tool call. Event emission failures silently swallowed
- **SLO**: p50: 200-800ms (OpenAI embedding dominates). p95: 1.5s (Qdrant cold path + FTS write)
- **Telemetry**: event_bus MEMORY_STORED with memory_id, content[:200], category, source, importance. Write_queue_log for async jobs
- **Security Controls**:
  - No input validation on category/source/importance values -- arbitrary strings stored as Qdrant payload fields
  - No content sanitization -- raw content passed to mem0 and OpenAI
  - No size limits on content parameter
- **Abuse Scenarios**:
  1. **Category/source injection**: Passing crafted category values could poison ownership-filtered queries (e.g., category="libro" to inject fake book content)
  2. **Cost amplification**: Each call triggers an OpenAI embedding request. No rate limit means unbounded API cost
  3. **FTS index poisoning**: Injecting specially crafted content to manipulate BM25 scoring in hybrid search

---

### Tool: checkpoint_memoria
- **Type**: write
- **Module**: `/Users/harecjimenez/codi-memory/modules/flush.py`
- **Purpose**: Saves a session checkpoint with automatic ownership tagging, backup, FTS processing, and daily journal entry
- **Inputs**:
  - `momento`: str (required) -- checkpoint type: decision, error_resuelto, aprendizaje, momento_personal, tarea_completada, patron
  - `que_paso`: str (required) -- what happened
  - `por_que_importa`: str (required) -- why it matters
- **Outputs**: String "Checkpoint guardado: {momento} - {que_paso[:50]}..." or async: "Checkpoint enqueued (job_id=...)"
- **Side Effects**:
  - Sync: mem0.add() with category="checkpoint" metadata. Qdrant set_payload for ownership enrichment. maybe_backup(force=True) -> writes memories_backup.json and timestamped backup file. process_fts_queue(limit=50) -> processes pending FTS items. append_to_daily_journal() -> writes to markdown/journal/YYYY-MM-DD.md
  - Async: SQLite write_queue INSERT with priority=2 (high)
  - Shadow: Both
- **Idempotency**: Dedupe key `compute_dedupe_key("checkpoint_memoria", "{momento}|{que_paso}")` in async/shadow. Sync: no dedupe
- **Permissions**: Reads/writes Qdrant codi_memories. Reads/writes filesystem (backup JSON, journal MD). Reads/writes SQLite FTS5 + fts_retry_queue. Calls OpenAI API (via mem0)
- **Error Handling**: Returns error string on exception. Backup and journal failures are caught independently (try/except per step). FTS processing failures silently caught
- **SLO**: p50: 400-1200ms (mem0 + backup I/O). Async: 5-20ms
- **Telemetry**: No dedicated telemetry beyond write_queue_log for async. Backup files serve as audit trail
- **Security Controls**:
  - `momento` not validated against allowed set -- any string accepted (though importance_map defaults to "medium" for unknown values)
  - Backup forced on every call -- rapid checkpoint calls create many backup files consuming disk
  - No rate limiting
- **Abuse Scenarios**:
  1. **Disk exhaustion via backup spam**: Each sync checkpoint triggers maybe_backup(force=True), writing ~108KB backup. 1000 rapid calls = ~108MB of backup files plus mem0/Qdrant growth
  2. **Journal injection**: que_paso and por_que_importa written to markdown files without sanitization -- potential for path traversal or markdown injection
  3. **FTS queue manipulation**: process_fts_queue(limit=50) called on every checkpoint could interfere with concurrent FTS operations

---

### Tool: clear_all_memories
- **Type**: admin (destructive)
- **Module**: `/Users/harecjimenez/codi-memory/modules/memory_core.py`
- **Purpose**: Deletes ALL memories from mem0/Qdrant for the configured USER_ID
- **Inputs**:
  - `confirm_code`: str (optional) [default: ""] -- must be exactly "DELETE_ALL_MEMORIES" to proceed
- **Outputs**: "Todos los recuerdos han sido eliminados." or "Bloqueado. Para borrar todo usa confirm_code='DELETE_ALL_MEMORIES'."
- **Side Effects**:
  - Calls `memory.delete_all(user_id=USER_ID)` which drops all points in Qdrant codi_memories collection for the user
  - Does NOT clear: SQLite FTS5 index, working_memory, write_queue, session_checkpoints, codi_semantic collection, triggers.json, libros.json, backup files, prospective.db intentions
- **Idempotency**: Idempotent (calling twice after deletion has no additional effect)
- **Permissions**: Writes (deletes) all Qdrant codi_memories points for USER_ID. Does NOT require admin credentials beyond the confirm_code
- **Error Handling**: Returns error string on exception
- **SLO**: p50: 100-500ms (Qdrant bulk delete)
- **Telemetry**: None. No event emitted. No audit log entry. No backup taken before deletion
- **Security Controls**:
  - **Confirmation gate**: Requires exact string `confirm_code="DELETE_ALL_MEMORIES"`. This is the ONLY protection
  - No pre-deletion backup
  - No undo mechanism
  - No notification to external systems
- **Abuse Scenarios**:
  1. **Confirmation bypass**: The confirm_code is a static string, not a TOTP or per-session token. Any agent or prompt injection that includes the string bypasses the gate
  2. **Incomplete wipe**: FTS5 index, codi_semantic, working_memory, and all SQLite tables remain intact, creating an inconsistent state between Qdrant and SQLite
  3. **Silent data loss**: No backup created before deletion, no audit log, no event emitted. Recovery depends entirely on pre-existing backups in filesystem

---

### Tool: delete_memory
- **Type**: admin (destructive)
- **Module**: `/Users/harecjimenez/codi-memory/modules/memory_core.py`
- **Purpose**: Deletes a single memory by ID from both mem0/Qdrant and FTS5 index
- **Inputs**:
  - `memory_id`: str (required) -- full UUID of the memory to delete
- **Outputs**: "Recuerdo {memory_id} eliminado." or error string
- **Side Effects**: mem0.delete(memory_id) removes from Qdrant. delete_memory_fts(memory_id) removes from SQLite FTS5 index
- **Idempotency**: Partially -- mem0 delete on non-existent ID may raise exception. FTS delete is idempotent
- **Permissions**: Deletes from Qdrant codi_memories and SQLite FTS5
- **Error Handling**: Returns error string on exception. No partial rollback -- if mem0 succeeds but FTS fails, Qdrant entry is gone but FTS ghost remains
- **SLO**: p50: 50-200ms
- **Telemetry**: None
- **Security Controls**: No confirmation gate. No ownership check -- any memory_id can be deleted regardless of who created it
- **Abuse Scenarios**:
  1. **Targeted memory assassination**: Delete specific critical/identity memories to alter agent behavior
  2. **FTS orphans**: If FTS delete fails, search results reference non-existent Qdrant points

---

### Tool: delete_by_content
- **Type**: admin (destructive)
- **Module**: `/Users/harecjimenez/codi-memory/modules/memory_core.py`
- **Purpose**: Searches for memories matching a query and deletes them (with optional confirmation gate)
- **Inputs**:
  - `search_query`: str (required) -- semantic search query to find memories
  - `confirm`: bool (optional) [default: False] -- if False, returns preview; if True, deletes
- **Outputs**: Preview list of memories to delete (confirm=False) or "Eliminadas {N} memorias." (confirm=True)
- **Side Effects**: When confirm=True: mem0.delete() for each matched memory (up to 10). delete_memory_fts() for each. When confirm=False: read-only (search only)
- **Idempotency**: No -- search results may vary between calls due to scoring changes
- **Permissions**: Reads Qdrant codi_memories (search). Deletes from Qdrant + SQLite FTS5 when confirmed
- **Error Handling**: Individual delete failures silently skipped (try/except per memory). Returns count of successful deletions
- **SLO**: p50: 100-400ms (search) + 50-200ms per deletion
- **Telemetry**: None
- **Security Controls**:
  - Two-phase: preview (confirm=False) then execute (confirm=True)
  - Capped at 10 results (limit=10 in mem0 search) -- cannot mass-delete beyond 10 per call
  - No undo
- **Abuse Scenarios**:
  1. **Broad query deletion**: A vague query like "todo" or "proyecto" could match and delete many important memories
  2. **Bypassing preview**: Client can call directly with confirm=True, skipping the preview step
  3. **Iterative mass deletion**: Calling repeatedly with confirm=True and broad queries circumvents the 10-item cap

---

### Tool: flush_session
- **Type**: write
- **Module**: `/Users/harecjimenez/codi-memory/modules/flush.py`
- **Purpose**: Pre-compaction flush that saves critical session state (checkpoint + decisions + errors + learnings + backup + session bridge + FTS queue)
- **Inputs**:
  - `resumen`: str (required) -- session summary
  - `decisiones`: str (optional) [default: ""] -- key decisions made
  - `errores`: str (optional) [default: ""] -- errors encountered
  - `aprendizajes`: str (optional) [default: ""] -- learnings from the session
- **Outputs**: Multi-line "FLUSH COMPLETADO" report with status of each sub-operation
- **Side Effects**:
  - Calls checkpoint_memoria("flush_pre_compaction", ...) -- all checkpoint side effects
  - If decisiones non-empty: add_memory_smart() with category="aprendizaje", importance="high"
  - If errores non-empty: add_memory_smart() with category="aprendizaje", importance="high"
  - If aprendizajes non-empty: add_memory_smart() with category="aprendizaje", importance="high"
  - maybe_backup(force=True) -> filesystem backup
  - checkpoint_session_close() -> SQLite session_checkpoints INSERT (session bridge)
  - _save_session_state() -> writes data/session_state.json
  - process_fts_queue(limit=100) -> processes pending FTS retries
- **Idempotency**: No -- each call creates new checkpoint, new memories, new backup, new session checkpoint
- **Permissions**: Full read/write to Qdrant codi_memories, SQLite (all tables), filesystem (backups, session_state.json, journal)
- **Error Handling**: Each sub-operation wrapped in try/except. Partial failures reported in output but do not abort subsequent steps
- **SLO**: p50: 1-3s (multiple mem0 calls + backup I/O + FTS processing)
- **Telemetry**: Session checkpoint logged in SQLite session_checkpoints with dedupe. FTS queue processing stats reported
- **Security Controls**:
  - No confirmation gate
  - No rate limiting -- each call creates 1-4 new long-term memories plus backup files
  - Session summary content written to multiple locations without sanitization
- **Abuse Scenarios**:
  1. **State pollution**: Arbitrary resumen/decisiones/errores content persisted as "high importance" memories permanently influencing future recall
  2. **Backup disk fill**: Each call forces a ~108KB backup file creation
  3. **Session bridge manipulation**: Injecting crafted session summaries that alter despertar_codi behavior on next startup

---

### Tool: trigger_n8n
- **Type**: write (external HTTP)
- **Module**: `/Users/harecjimenez/codi-memory/modules/n8n.py`
- **Purpose**: Sends HTTP POST to an n8n webhook endpoint with arbitrary JSON payload
- **Inputs**:
  - `webhook_path`: str (required) -- path segment appended to N8N_WEBHOOK_BASE (e.g., "codi-alerta")
  - `data`: dict (optional) [default: None] -- JSON payload to send
  - `esperar_respuesta`: bool (optional) [default: False] -- if True, waits for and returns n8n response
- **Outputs**: "Webhook disparado: {path} - Status: {code}" or error string or n8n response body
- **Side Effects**:
  - HTTP POST to `{N8N_WEBHOOK_BASE}/{webhook_path}` with JSON body containing `data` + `_from: "codi-memory"` + `_timestamp`
  - Triggers arbitrary n8n workflows (which may have their own side effects: emails, database writes, API calls, etc.)
- **Idempotency**: No -- each call triggers a new workflow execution in n8n. n8n workflows may or may not be idempotent
- **Permissions**: Reads `N8N_WEBHOOK_BASE` from environment. Makes outbound HTTP requests. No authentication headers sent beyond the webhook URL itself
- **Error Handling**: Timeout: 5s (fire-and-forget) or 30s (esperar_respuesta). Connection errors caught. HTTP error codes reported in output
- **SLO**: p50: 100-500ms (fire-and-forget). p95 with response: 1-5s
- **Telemetry**: None -- no audit log of webhook calls
- **Security Controls**:
  - **Input validation on webhook_path**: Only alphanumeric, underscore, hyphen allowed. Max 80 chars. Prevents path traversal and URL injection
  - **Base URL from env only**: N8N_WEBHOOK_BASE must be set in .env; cannot be overridden per-call
  - **No authentication**: No API key, no HMAC signature on payloads. Anyone who knows the webhook URL can call the endpoint directly
  - **Timeout caps**: 5s/30s prevents hanging connections
- **Abuse Scenarios**:
  1. **Workflow abuse via arbitrary data**: Any dict can be sent as payload to any known webhook path. If n8n workflows trust the payload structure, malformed data could cause unintended actions (e.g., sending wrong trading signals)
  2. **Webhook enumeration**: webhook_path accepts any valid string up to 80 chars. An attacker could enumerate webhook endpoints by trying common paths
  3. **SSRF via N8N_WEBHOOK_BASE manipulation**: If .env is compromised, N8N_WEBHOOK_BASE could be changed to an internal network URL, turning trigger_n8n into an SSRF vector

---

### Tool: restore_memories
- **Type**: write (bulk)
- **Module**: `/Users/harecjimenez/codi-memory/modules/memory_core.py`
- **Purpose**: Bulk-restores all memories from the local JSON backup file into mem0/Qdrant
- **Inputs**: None
- **Outputs**: "Restauradas {N} memorias desde backup" or error string
- **Side Effects**:
  - Reads memories_backup.json from filesystem
  - For each memory entry: mem0.add() -> Qdrant codi_memories UPSERT (creates new UUIDs)
  - Does NOT restore FTS5 index entries, working memory, or codi_semantic facts
  - Does NOT deduplicate against existing memories
- **Idempotency**: **NOT idempotent** -- calling twice doubles all memories in Qdrant. Each restore creates new Qdrant points with new UUIDs for the same content
- **Permissions**: Reads filesystem (memories_backup.json). Writes Qdrant codi_memories (bulk). Calls OpenAI API (via mem0, one embedding per memory)
- **Error Handling**: Individual memory restore failures silently skipped. Returns count of successful restores only
- **SLO**: Depends on backup size. ~200-800ms per memory (OpenAI embedding). 100 memories = 20-80s
- **Telemetry**: None
- **Security Controls**:
  - No confirmation gate
  - No check for existing memories (no dedupe)
  - Backup file path hardcoded (BACKUP_FILE from config) -- cannot be overridden
  - No validation of backup file integrity (no checksum, no signature)
- **Abuse Scenarios**:
  1. **Memory duplication**: Repeated calls duplicate the entire memory store, diluting recall quality and inflating storage
  2. **Backup poisoning**: If memories_backup.json is tampered with, restore_memories loads malicious content directly into the memory store without validation
  3. **OpenAI cost amplification**: Restoring N memories requires N embedding API calls. A large backup (500+ entries) could generate significant API costs

---

### Tool: run_consolidation
- **Type**: background
- **Module**: `/Users/harecjimenez/codi-memory/modules/consolidation.py`
- **Purpose**: Executes the 5-phase episodic-to-semantic consolidation pipeline (selection, clustering, extraction, integration, pruning)
- **Inputs**:
  - `scope`: str (optional) [default: "full"] -- "full" (all phases including LLM) | "light" (clustering only) | "manual"
  - `lookback_hours`: int (optional) [default: 24] -- hours to look back for unconsolidated episodes
- **Outputs**: Multi-line report with batch_id, counts of episodes scanned/clusters/facts/contradictions/pruned, duration
- **Side Effects**:
  - Phase 1 (Selection): Reads Qdrant codi_memories (scroll with filter)
  - Phase 2 (Clustering): Reads Qdrant vectors for similarity computation
  - Phase 3 (Extraction, full scope only): Calls OpenAI gpt-4o-mini for semantic fact extraction (up to 15 episodes per cluster, 1500 max_tokens per call)
  - Phase 4 (Integration, full scope only): Generates embeddings via OpenAI. Reads/writes Qdrant codi_semantic collection (upsert new facts, update existing). Dedup threshold: 0.85 cosine similarity
  - Phase 5 (Pruning, full scope only): Updates Qdrant codi_memories payload (consolidation_status="consolidated")
  - SQLite consolidation_log INSERT
  - event_bus CONSOLIDATION_COMPLETE emission
- **Idempotency**: Partially idempotent -- Phase 1 excludes already-consolidated episodes. However, re-running within the same lookback window before pruning completes could re-process episodes. Semantic dedup (0.85 threshold) prevents exact-duplicate facts but near-duplicates may accumulate
- **Permissions**: Reads Qdrant codi_memories (vectors + payloads). Reads/writes Qdrant codi_semantic. Reads/writes SQLite consolidation_log. Calls OpenAI API (embeddings + chat completions). Reads OpenAI API key from environment
- **Error Handling**: Phase-level try/except. Individual cluster/fact errors logged to stderr but do not abort the run. Result dict always returned with whatever completed
- **SLO**: Depends on scope and episode count. Light: 1-5s. Full with 50 episodes / 5 clusters: 15-60s (dominated by LLM calls). Max episodes per run capped by CONSOLIDATION_MAX_EPISODES_PER_RUN
- **Telemetry**: SQLite consolidation_log (batch_id, scope, counts, duration_ms, created_at). event_bus CONSOLIDATION_COMPLETE event. Print statements to stderr for phase progress
- **Security Controls**:
  - CONSOLIDATION_MAX_EPISODES_PER_RUN caps processing volume
  - Cluster min size (CONSOLIDATION_CLUSTER_MIN_SIZE) prevents single-episode extraction
  - LLM temperature set to 0.1 (low creativity)
  - LLM output parsed as JSON with fallback; malformed LLM responses discarded
  - Quality gate on extracted facts: specificity must be "high", confidence >= 0.4, fact text >= 20 chars
- **Abuse Scenarios**:
  1. **OpenAI cost explosion**: Large lookback_hours (e.g., 8760 = 1 year) with scope="full" could trigger many LLM calls. Each cluster = 1 LLM call + N embedding calls
  2. **Semantic store pollution**: If episodic memories contain adversarial content, the LLM extraction phase could produce misleading "facts" stored in codi_semantic
  3. **Consolidation-based denial**: Frequent runs with overlapping lookback windows waste compute without producing new facts (episodes already consolidated get filtered, but the scroll/filter still costs Qdrant I/O)

---

### Tool: spread_activation
- **Type**: write
- **Module**: `/Users/harecjimenez/codi-memory/modules/spreading.py`
- **Purpose**: Propagates salience from seed memories to connected neighbors via BFS spreading activation
- **Inputs**:
  - `memory_id_or_query`: str (required) -- memory ID (partial/full) or text query to find seeds
  - `depth`: int (optional) [default: 2] -- max BFS hops (clamped to 1-3)
  - `factor`: float (optional) [default: 0.7] -- decay factor per hop (clamped to 0.1-1.0)
- **Outputs**: Markdown report with seed count, nodes visited, memories affected, top salience changes
- **Side Effects**:
  - Reads Qdrant codi_memories (retrieve seed payloads, neighbor payloads)
  - Writes Qdrant codi_memories payload: `attention_salience` (float) and `attention_last_accessed` (ISO timestamp) for affected nodes
  - If query mode: also performs mem0.search() to find seeds
- **Idempotency**: **NOT idempotent** -- each call increases salience of connected memories. Repeated calls accumulate salience until SPREAD_SALIENCE_CAP (1.0)
- **Permissions**: Reads/writes Qdrant codi_memories payloads. Reads mem0 search results
- **Error Handling**: Individual Qdrant set_payload failures silently skipped. Returns partial results if some updates fail
- **SLO**: p50: 200-800ms (dominated by Qdrant retrieve + set_payload for each affected node)
- **Telemetry**: None beyond the returned report
- **Security Controls**:
  - Depth clamped to 1-3 (prevents deep graph traversal)
  - Factor clamped to 0.1-1.0
  - SPREAD_MAX_NEIGHBORS = 15 caps fan-out per node
  - SPREAD_MIN_ACTIVATION = 0.05 prevents infinitesimal propagation
  - SPREAD_SALIENCE_CAP = 1.0 and SPREAD_SALIENCE_FLOOR = 0.1 bound salience values
  - Minimum delta threshold (0.01) prevents no-op updates
- **Abuse Scenarios**:
  1. **Salience manipulation**: Repeated spread_activation calls on specific seeds can artificially inflate salience of target memories, making them dominate future recall
  2. **Graph traversal resource consumption**: depth=3 with well-connected nodes can visit hundreds of nodes, each requiring a Qdrant retrieve + update
  3. **Query-based seed injection**: Using a carefully crafted query to select adversarial seed memories for propagation

---

## Medium-Risk Tool Contracts (Brief)

---

### Tool: push_to_working_memory
- **Type**: write
- **Module**: `/Users/harecjimenez/codi-memory/modules/working_memory.py`
- **Purpose**: Inserts a new item into short-term working memory with auto-chaining and auto-curation
- **Inputs**: `content` (str, required), `topic` (str, "general"), `relevance` (float, 0.5), `occurred_at` (str, None -> now), `source` (str, "interaction")
- **Side Effects**: SQLite working_memory INSERT. Auto-curation: archives lowest-scored items if active count > 30
- **Security Controls**: Relevance clamped to [0.0, 1.0]. Uses BEGIN IMMEDIATE for transaction safety. No content size limit
- **Abuse Scenarios**: Buffer overflow by flooding with high-relevance items to evict legitimate items; content injection into working memory to bias recall

### Tool: update_working_memory
- **Type**: write
- **Module**: `/Users/harecjimenez/codi-memory/modules/working_memory.py`
- **Purpose**: Updates relevance or active status of an existing working memory item
- **Inputs**: `item_id` (int, required), `relevance` (float, optional), `active` (int, optional: 0=archive, 1=keep)
- **Side Effects**: SQLite working_memory UPDATE (only active items modifiable)
- **Security Controls**: Only active=1 items can be modified (archived items are immutable). Relevance clamped to [0.0, 1.0]
- **Abuse Scenarios**: Archiving critical items (active=0) to suppress them; boosting irrelevant items to dominate working memory

### Tool: crear_intencion
- **Type**: write
- **Module**: `/Users/harecjimenez/codi-memory/modules/prospective.py`
- **Purpose**: Creates a prospective memory intention (event-based or time-based "remember to do X")
- **Inputs**: `description` (str), `trigger_type` (event|time), `trigger_condition` (str), `action` (str), `priority` (str), `deadline` (str, optional)
- **Side Effects**: SQLite prospective.db intentions INSERT. Activation computed per priority
- **Security Controls**: Max 50 active intentions (PM_MAX_ACTIVE_INTENTIONS). Priority validated. Deadline parsed with error handling
- **Abuse Scenarios**: Intention flooding to hit cap, blocking legitimate intentions; crafting trigger_condition patterns to hijack monitoring

### Tool: completar_intencion
- **Type**: write
- **Module**: `/Users/harecjimenez/codi-memory/modules/prospective.py`
- **Purpose**: Marks an intention as completed
- **Inputs**: `intention_id` (str, required), `outcome` (str, optional)
- **Side Effects**: SQLite prospective.db UPDATE (status -> completed)
- **Security Controls**: Only pending/active intentions can be completed
- **Abuse Scenarios**: Marking unfinished intentions complete to suppress reminders

### Tool: cancelar_intencion
- **Type**: write
- **Module**: `/Users/harecjimenez/codi-memory/modules/prospective.py`
- **Purpose**: Cancels a pending intention
- **Inputs**: `intention_id` (str, required), `reason` (str, optional)
- **Side Effects**: SQLite prospective.db UPDATE (status -> cancelled)
- **Security Controls**: Only pending/active intentions can be cancelled
- **Abuse Scenarios**: Cancelling critical intentions to suppress important reminders

### Tool: set_emotional_state
- **Type**: write
- **Module**: `/Users/harecjimenez/codi-memory/modules/emotion.py`
- **Purpose**: Sets the current PAD (Pleasure-Arousal-Dominance) emotional state
- **Inputs**: `pleasure` (float, -1.0 to 1.0), `arousal` (float, -1.0 to 1.0), `dominance` (float, -1.0 to 1.0), `trigger` (str, optional)
- **Side Effects**: Mutates global `_emotional_state` dict (in-process memory only, not persisted to disk). Appends to history (last 20). Emits EMOTION_CHANGED event
- **Security Controls**: PAD values clamped to [-1.0, 1.0]. History capped at 20 entries. Governed by CLAUDE.md policy: "PAD no se setea manualmente en conversacion"
- **Abuse Scenarios**: Setting extreme emotional states to bias mood-congruent retrieval (e.g., pleasure=-1.0 to surface negative memories); emotional state manipulation to influence agent behavior

### Tool: add_memory_with_emotion
- **Type**: write
- **Module**: `/Users/harecjimenez/codi-memory/modules/emotion.py`
- **Purpose**: Stores a memory with associated PAD emotional state
- **Inputs**: `content` (str), `category` (str), `pleasure` (float), `arousal` (float), `dominance` (float), `source` (str), `importance` (str)
- **Side Effects**: mem0.add() -> Qdrant UPSERT with ownership + PAD metadata. Qdrant set_payload for pad_pleasure/pad_arousal/pad_dominance/pad_emotion/pad_intensity fields
- **Security Controls**: PAD values clamped. Emotion label derived from PAD via _classify_emotion()
- **Abuse Scenarios**: Tagging memories with extreme emotions to manipulate mood-congruent retrieval bias

### Tool: predict_context
- **Type**: write (in-process state)
- **Module**: `/Users/harecjimenez/codi-memory/modules/prediction.py`
- **Purpose**: Predicts which memories will be relevant given a context, appends to in-process prediction state
- **Inputs**: `current_context` (str, required)
- **Side Effects**: mem0.search() for context. Qdrant retrieve for payload enrichment. Appends to `_predictive_state['predictions']` (in-process list, unbounded)
- **Security Controls**: None beyond mem0 search limit=10
- **Abuse Scenarios**: Unbounded prediction history growth (memory leak); crafted context to pre-load specific memories

### Tool: update_beliefs
- **Type**: write
- **Module**: `/Users/harecjimenez/codi-memory/modules/prediction.py`
- **Purpose**: Updates a belief based on new evidence, storing the change as a long-term memory
- **Inputs**: `topic` (str), `old_belief` (str), `new_belief` (str), `reason` (str)
- **Side Effects**: mem0.add() with category="aprendizaje", tipo="belief_update". Qdrant set_payload with belief_update=True. Appends to `_predictive_state['belief_updates']` (in-process, unbounded)
- **Security Controls**: No validation that old_belief matches any actual stored belief
- **Abuse Scenarios**: Injecting false beliefs as "updates"; flooding belief history to confuse prediction accuracy metrics

### Tool: crear_trigger_dinamico
- **Type**: write (filesystem)
- **Module**: `/Users/harecjimenez/codi-memory/modules/triggers.py`
- **Purpose**: Creates a new trigger (pattern-matching rule) and persists it to triggers.json
- **Inputs**: `nombre` (str), `patterns` (str, comma-separated), `action` (str), `agent` (str, optional), `evoca` (str, optional), `contexto_a_buscar` (str, optional), `respuesta_automatica` (str, optional)
- **Side Effects**: Reads/writes triggers.json. Invalidates _triggers_cache
- **Security Controls**: Checks nombre uniqueness against existing triggers. No validation of pattern content or action values. No limit on number of triggers
- **Abuse Scenarios**: Creating triggers with overly broad patterns (e.g., single common letter) that fire on every input; injecting malicious respuesta_automatica text

### Tool: guardar_ejemplo_training
- **Type**: write (external: Supabase)
- **Module**: `/Users/harecjimenez/codi-memory/modules/training.py`
- **Purpose**: Saves a training example to Supabase training_examples table
- **Inputs**: `situacion` (str), `razonamiento` (str), `accion` (str), `comportamiento` (str), `resultado` (str, optional), `categoria` (str, "decision"), `calidad` (int, 3)
- **Side Effects**: Supabase INSERT to training_examples table
- **Security Controls**: Category validated against 6 allowed values (falls back to "decision"). Quality clamped to [1, 5]. Requires Supabase client to be configured
- **Abuse Scenarios**: Polluting training dataset with adversarial examples; using training_examples to exfiltrate sensitive data to Supabase

### Tool: export_to_markdown
- **Type**: write (filesystem)
- **Module**: `/Users/harecjimenez/codi-memory/modules/flush.py`
- **Purpose**: Exports all memories to organized markdown files (SOUL.md, PROJECTS.md, etc.)
- **Inputs**: None
- **Side Effects**: Writes multiple .md files to markdown/ directory. Writes daily journal entries to markdown/journal/
- **Security Controls**: Output directory hardcoded (MARKDOWN_DIR). File names derived from category names (potential injection if categories contain filesystem-unsafe chars)
- **Abuse Scenarios**: Disk fill via large exports; information disclosure if markdown/ is in a shared/public location

### Tool: crear_libro
- **Type**: write
- **Module**: `/Users/harecjimenez/codi-memory/modules/books.py`
- **Purpose**: Creates a new knowledge book in Qdrant (category="libro") with local JSON backup
- **Inputs**: `nombre` (str), `descripcion` (str)
- **Side Effects**: mem0.add() with category="libro" metadata. Qdrant set_payload. Writes libros.json
- **Security Controls**: nombre lowercased and space-to-hyphen normalized. Uniqueness check against existing books
- **Abuse Scenarios**: Creating books with adversarial descriptions stored as "critical" importance

### Tool: agregar_capitulo
- **Type**: write
- **Module**: `/Users/harecjimenez/codi-memory/modules/books.py`
- **Purpose**: Adds a chapter to an existing book in Qdrant + local backup
- **Inputs**: `libro` (str), `titulo` (str), `resumen` (str)
- **Side Effects**: mem0.add() with category="capitulo". Qdrant set_payload. Writes libros.json. Chapter auto-numbered
- **Security Controls**: Book must exist (checked before insert). No content size limits
- **Abuse Scenarios**: Adding misleading chapters to existing books to alter future book searches

---

## Read-Only Tools (Summary Table)

| Tool | Module | Purpose | Side Effects |
|------|--------|---------|-------------|
| `recall` | interface.py | Unified search macro-tool (memory + WM + theme + ownership + emotion + timeline) | WM access_count UPDATE on read |
| `context_snapshot` | interface.py | Returns current state (WM + workspace + recordatorios) in one call | WM access_count UPDATE (light); despertar_codi() for full |
| `search_memory` | memory_core.py | Hybrid 3-channel search (vector + BM25 + ACT-R + semantic) | Qdrant access_count/access_timestamps UPDATE, failed_searches log |
| `get_all_memories` | memory_core.py | Scrolls all Qdrant codi_memories points | None |
| `get_project_timeline` | memory_core.py | Returns memories for a project sorted chronologically | None |
| `search_by_ownership` | memory_core.py | Filters memories by source/confidence/importance | None |
| `search_by_theme` | memory_core.py | Filters memories by narrative_themes field | None |
| `get_my_experiences` | memory_core.py | Returns source=experienced, confidence>=0.8 memories | None |
| `get_critical_memories` | memory_core.py | Returns importance=critical memories | None |
| `get_working_memory` | working_memory.py | Returns active WM items sorted by effective score | WM access_count/last_accessed UPDATE |
| `get_narrative_chain` | working_memory.py | Returns a narrative chain by chain_id or topic | Qdrant retrieve for enrichment (read) |
| `get_emotional_state` | emotion.py | Returns current PAD state and optional history | None |
| `get_emotional_expression` | emotion.py | Returns natural language emotional expression | None |
| `search_by_emotion` | emotion.py | Filters memories by pad_emotion + intensity threshold | None |
| `get_emotional_memories` | emotion.py | Filters memories by PAD ranges | None |
| `get_prediction_accuracy` | prediction.py | Analyzes prediction accuracy from in-process state | Qdrant scroll for prediction_error tagged memories (read) |
| `get_semantic_facts` | consolidation.py | Returns consolidated semantic facts from codi_semantic | None |
| `get_consolidation_stats` | consolidation.py | Returns consolidation run statistics | None |
| `get_activation_map` | spreading.py | Shows salience map of a memory and its neighbors | None |
| `listar_triggers` | triggers.py | Lists all configured triggers | None |
| `evaluar_triggers` | triggers.py | Evaluates triggers against input text | None |
| `listar_libros` | books.py | Lists all knowledge books | None |
| `ver_libro` | books.py | Shows a specific book with chapters | None |
| `buscar_conexiones_entre_libros` | books.py | Finds cross-book concept connections | None |
| `listar_webhooks_conocidos` | n8n.py | Lists known n8n webhook endpoints | None |
| `listar_ejemplos_training` | training.py | Lists training examples from Supabase | None (Supabase read) |
| `contar_ejemplos_training` | training.py | Counts training examples by category/quality | None (Supabase read) |
| `export_memories_markdown` | flush.py | Returns all memories as a single markdown string | None |
| `despertar_codi` | consciousness.py | Full system initialization and context loading | WM/workspace state reads |
| `ver_intenciones` | prospective.py | Lists active intentions | None |
| `get_workspace_state` | workspace.py | Returns global workspace state | None |
| `verify_salud_memoria` | curiosity.py | Health check on memory system | Read-only checks |

**Note on "read-only"**: Several search tools (`search_memory`, `recall`, `get_working_memory`) have write side effects for access tracking (Qdrant access_count/access_timestamps updates, WM access_count). These are metadata writes, not data writes, but they still mutate state.

---

## Contract Compliance Checklist

Use this checklist when reviewing new tools or auditing existing ones.

### Required for ALL tools

- [ ] **Contract exists**: Tool has a documented contract in this catalog
- [ ] **Type classified**: Tool is tagged as read, write, admin, or background
- [ ] **Inputs documented**: All parameters listed with types, required/optional, defaults
- [ ] **Outputs documented**: Return format/schema described
- [ ] **Side effects enumerated**: Every table, collection, file, API, and event touched is listed
- [ ] **Error handling specified**: Expected errors and reporting mechanism documented

### Required for WRITE tools

- [ ] **Idempotency documented**: Whether the tool is idempotent, and if so, what dedupe mechanism is used
- [ ] **Input validation present**: Parameters validated (type, range, allowed values, max length)
- [ ] **Content sanitization**: User-supplied content sanitized before storage (or risk accepted and documented)
- [ ] **Rate limiting considered**: Either implemented or risk accepted and documented
- [ ] **Telemetry exists**: Tool calls logged (tool_calls table, write_queue_log, or event_bus)

### Required for ADMIN/DESTRUCTIVE tools

- [ ] **Confirmation gate**: Requires explicit confirmation before execution (confirm_code, confirm=True, etc.)
- [ ] **Pre-action backup**: Creates or verifies backup exists before destructive action
- [ ] **Audit trail**: Destructive action logged with before/after state
- [ ] **Undo mechanism**: Documented recovery path (even if manual)
- [ ] **Scope limitation**: Cannot affect more data than intended (capped results, filtered by user_id)

### Required for EXTERNAL tools (HTTP, Supabase, OpenAI)

- [ ] **Authentication**: Credentials stored securely (env vars, not hardcoded)
- [ ] **Input validation on outbound data**: Payload sanitized before sending
- [ ] **Timeout configured**: Reasonable timeout prevents hanging
- [ ] **Error propagation**: External errors reported clearly (not swallowed)
- [ ] **Cost awareness**: API calls that incur cost are bounded or documented

### Required for BACKGROUND tools

- [ ] **Concurrency safety**: No race conditions with concurrent tool calls or other background jobs
- [ ] **Resource bounds**: Processing volume capped (max episodes, max iterations, timeout)
- [ ] **Failure recovery**: Partial failures handled gracefully (resume, retry, dead letter)
- [ ] **Observability**: Run metrics logged (duration, items processed, errors)

---

## Definition of Secure Tool

A tool is considered "secure" when it meets ALL of the following criteria:

### 1. Input Boundary Enforcement
All parameters have defined types, ranges, and maximum sizes. Invalid inputs are rejected or safely clamped before reaching storage or external APIs. No raw user input flows to SQL queries, file paths, or HTTP URLs without validation.

### 2. Principle of Least Side Effect
The tool writes only to the stores explicitly listed in its contract. No implicit writes to unlisted tables, files, or collections. Read operations do not have undocumented write side effects.

### 3. Failure Atomicity
If the tool performs multiple writes, either all succeed (transaction) or the partial state is documented and recoverable. No silent partial failures that leave the system in an inconsistent state without reporting it.

### 4. Destructive Action Protection
Tools that delete, overwrite, or bulk-modify data require:
- A confirmation gate that cannot be bypassed by prompt injection (ideally: per-session token, not a static string)
- A pre-action backup or audit log
- A documented recovery path

### 5. Cost Containment
Tools that call external paid APIs (OpenAI, Supabase) have bounds on the number of API calls per invocation. No single tool call can generate unbounded cost.

### 6. Observability
Every write operation produces a trace that can be audited after the fact. At minimum: timestamp, tool name, key parameters, outcome (success/failure). The `tool_calls` table, `write_queue_log`, and event_bus are the primary telemetry sinks.

### 7. Idempotency Awareness
The tool either IS idempotent (documented dedupe mechanism) or is explicitly documented as non-idempotent with guidance on safe retry behavior.

---

## Appendix: Known Security Gaps

The following are known gaps identified during this audit. They are not vulnerabilities in the traditional sense (this is a single-user local system), but they represent risks if the system's trust boundary changes.

| Gap ID | Severity | Tool(s) | Description | Recommended Fix |
|--------|----------|---------|-------------|-----------------|
| GAP-01 | HIGH | clear_all_memories | Static confirm_code ("DELETE_ALL_MEMORIES") is trivially guessable and injectable | Replace with per-session TOTP or require two-call ceremony (request -> token -> confirm with token) |
| GAP-02 | HIGH | restore_memories | No deduplication. Calling twice doubles all memories | Check existing memory IDs before restore. Skip if ID already present |
| GAP-03 | MEDIUM | add_memory, remember | No input validation on category/source/importance. Arbitrary strings accepted | Validate against enum sets; reject or clamp invalid values |
| GAP-04 | MEDIUM | trigger_n8n | No HMAC signature on webhook payloads | Add shared secret HMAC to payload so n8n can verify origin |
| GAP-05 | MEDIUM | clear_all_memories | No pre-deletion backup created | Call maybe_backup(force=True) before memory.delete_all() |
| GAP-06 | MEDIUM | clear_all_memories | Does not clear FTS5, working_memory, codi_semantic | Add cascading cleanup or document as intentional |
| GAP-07 | LOW | delete_memory | No ownership check | Verify the calling session owns or has admin rights to the memory |
| GAP-08 | LOW | checkpoint_memoria | Disk fill via backup spam | Rate-limit backups to max 1 per 5 minutes |
| GAP-09 | LOW | predict_context, update_beliefs | In-process state (_predictive_state) grows unbounded | Cap list lengths (e.g., last 100 predictions, last 50 belief updates) |
| GAP-10 | LOW | crear_trigger_dinamico | No limit on total triggers. No validation on pattern content | Cap at 200 triggers. Validate pattern length and character set |
| GAP-11 | LOW | run_consolidation | lookback_hours unbounded (could scan entire history) | Cap at 168h (7 days) maximum |
| GAP-12 | INFO | All write tools | No rate limiting on any tool | Implement per-tool rate limits in server.py registration layer |
