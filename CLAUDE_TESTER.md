# codi-memory — Tester Instructions

You are testing **codi-memory**, an MCP memory server with cognitive architecture.
Your goal is to exercise the tools, find bugs, and report issues.

## Your MCP Tools

The server exposes tools via the `mcp__codi-memory__` prefix. The main ones to test:

### Core (start here)
| Tool | What it does |
|------|-------------|
| `recall(query)` | Search memories (hybrid: vector + FTS + activation) |
| `remember(content)` | Save a memory (working memory + long-term) |
| `context_snapshot()` | Get current system state |
| `search_memory(query)` | Direct vector search |

### Working Memory
| Tool | What it does |
|------|-------------|
| `get_working_memory()` | List active short-term items |
| `push_to_working_memory(content)` | Add item to working memory |
| `update_working_memory(item_id, active=0)` | Archive an item |

### Consciousness / Cognition
| Tool | What it does |
|------|-------------|
| `get_emotional_state()` | Current PAD emotional state |
| `get_emotional_expression()` | Natural language emotion |
| `verificar_salud_memoria()` | Health check (mem0 + Qdrant) |

### Prospective Memory
| Tool | What it does |
|------|-------------|
| `crear_intencion(action)` | Create a "remember to do X later" |
| `ver_intenciones()` | List pending intentions |
| `completar_intencion(id)` | Mark intention as done |

### Triggers
| Tool | What it does |
|------|-------------|
| `listar_triggers()` | Show all configured triggers |
| `evaluar_triggers(input_text)` | Detect triggers in text |

### Advanced
| Tool | What it does |
|------|-------------|
| `add_memory_smart(content)` | Save with deduplication |
| `checkpoint_memoria(momento, que_paso, por_que_importa)` | Save a checkpoint |
| `flush_session(resumen)` | Flush session state to long-term |
| `get_narrative_chain(topic)` | View narrative timeline |
| `get_sharpe_report()` | Cognitive Sharpe ratio report |

## What to Test

### 1. Basic CRUD
- [ ] `remember("Test memory from Sebastian")` — saves successfully
- [ ] `recall("test memory")` — finds the memory you just saved
- [ ] `remember()` the same content twice — should deduplicate (not create duplicate)
- [ ] Save memories with different `importance` levels: "low", "medium", "high", "critical"
- [ ] Save memories with different `topic` values and verify `recall` filters correctly

### 2. Working Memory
- [ ] `push_to_working_memory("task in progress")` — creates item
- [ ] `get_working_memory()` — shows the item with relevance score
- [ ] `update_working_memory(item_id, active=0)` — archives it
- [ ] `get_working_memory()` — archived item no longer appears
- [ ] Push 30+ items — verify auto-curation kicks in (oldest/lowest relevance archived)

### 3. Search Quality
- [ ] Search for exact phrases — should return high scores
- [ ] Search for related concepts (not exact words) — should still find relevant memories
- [ ] Search for something that doesn't exist — should return empty, not hallucinate
- [ ] Test with Spanish and English queries

### 4. Prospective Memory
- [ ] Create an event-based intention: `crear_intencion("Remind me to check X", trigger_type="event", trigger_spec='{"keywords": ["X"]}')`
- [ ] `ver_intenciones()` — shows the pending intention
- [ ] Mention the keyword in conversation — intention should fire
- [ ] `completar_intencion(id)` — marks it done

### 5. Emotional State
- [ ] `get_emotional_state()` — returns valid PAD values
- [ ] `get_emotional_expression()` — returns natural language description
- [ ] Verify emotional state changes after interactions

### 6. System Health
- [ ] `verificar_salud_memoria()` — all checks pass
- [ ] `context_snapshot()` — returns coherent state
- [ ] `get_sharpe_report()` — runs without errors

### 7. Edge Cases & Error Handling
- [ ] Empty string: `remember("")` — should handle gracefully
- [ ] Very long content: `remember("x" * 10000)` — should handle or reject
- [ ] Special characters: `remember("quotes 'single' \"double\" and unicode: ")`
- [ ] Concurrent operations: save + search at the same time
- [ ] `recall` with mode parameter: `recall("test", mode="memory")`, `recall("test", mode="timeline")`

## How to Report Issues

When you find a bug, note:

1. **Tool called** — exact tool name and parameters
2. **Expected behavior** — what should have happened
3. **Actual behavior** — what actually happened (include error messages)
4. **Reproducible?** — can you trigger it again with the same steps?

## What NOT to Do

- Do not use `clear_all_memories` or `delete_memory` — these are destructive
- Do not modify any `.py` files
- Do not change the `.env` file after initial setup
- Do not run `flush_session` or `sync_fts_index` unless specifically testing those
