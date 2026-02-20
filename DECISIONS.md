# Architecture Decisions

Registro de decisiones arquitectonicas relevantes del sistema (ADR-lite).
Cada entrada incluye: contexto, decision, alternativas, tradeoffs, estado.

---

## D001 — Continuity Pipeline (2026-02-15)

**Contexto:** Perdida completa de hilo entre sesiones. Cada despertar era "desde cero".

**Decision:** Pipeline en 4 etapas:
1. `checkpoint_session_close()` — captura estado minimo suficiente de 9 fuentes
2. Sleep Loop — mantenimiento entre sesiones (4 ticks con budget)
3. `load_session_bridge()` — reconstruye narrativa determinista (sin LLM)
4. `despertar_codi()` — inyecta bridge + restaura PAD + intenciones

**Alternativas consideradas:**
- Persistir contexto completo (descartado: demasiado grande, no escala)
- Continuidad via LLM summary (descartado: no determinista, costoso)

**Tradeoff:** "Ilusion estable" de continuidad vs exactitud total. El sistema fabrica continuidad desde checkpoints, prediction errors y narrativa, igual que el cerebro.

**Estado:** DONE. Validado con 31 continuity tests + 2 golden E2E.

---

## D002 — Session Checkpoints en SQLite + Retention (2026-02-15)

**Contexto:** Necesidad de persistir estado entre sesiones sin depender de servicios externos.

**Decision:** Tabla `session_checkpoints` en SQLite con:
- Dedupe window de 120 segundos
- Source priority: flush (2) > hook (1)
- Retention max 50 filas (FIFO)
- Sleep report como columna nullable

**Alternativas:**
- JSON file (descartado: no atomico, corrupcion en crash)
- Supabase (descartado: dependencia externa para path critico)

**Tradeoff:** Menos historia (50 checkpoints) a cambio de DB estable y rapida.

**Estado:** DONE.

---

## D003 — Budget Enforcement por Ticks (Sleep Loop) (2026-02-16)

**Contexto:** El sleep loop hace 4 tareas de mantenimiento entre sesiones. Sin control de presupuesto, una tarea lenta bloquea las demas.

**Decision:**
- Orden fijo: `prospective -> health -> consolidation -> homeostasis`
- Ticks rapidos primero, pesados al final
- Budget gating: cada tick verifica tiempo restante antes de ejecutar
- Total budget: 8,000ms configurable via CLI
- Minimum budget por tick (consolidation: 1,500ms, otros: 200ms)

**Alternativas:**
- Timeouts por tick (descartado: mas complejo, kill de threads)
- Orden dinamico por prioridad (descartado: no predecible, dificil de testear)

**Tradeoff:** Si el budget se agota, homeostasis (menos critico) se sacrifica primero. Prospective (mas urgente) siempre corre.

**Estado:** DONE. Tick-level metrics implementados (E2.2).

---

## D004 — Migraciones como Unica Fuente de Verdad (2026-02-15)

**Contexto:** Tablas SQLite se creaban en multiples lugares con `CREATE TABLE IF NOT EXISTS`. Esto hacia imposible saber el schema real y causaba drift silencioso.

**Decision:**
- Cero `CREATE TABLE` fuera de `migrations/` y `migrations_prospective/`
- `ensure_schema_ready()` valida que las migraciones se aplicaron
- `apply_migrations()` es idempotente (tracking en tabla `_migrations`)
- Migraciones son forward-only (no rollback)

**Alternativas:**
- Auto-heal con `CREATE TABLE IF NOT EXISTS` (descartado: oculta bugs)
- Alembic (descartado: overkill para SQLite local)

**Tradeoff:** Fail-fast en vez de auto-heal silencioso. Un schema incorrecto falla rapido en lugar de producir datos corruptos.

**Estado:** DONE. 2 migration dirs: `migrations/` (FTS DB, 2 files) + `migrations_prospective/` (1 file).

---

## D005 — Strangler Fig + Facade (D5 De-God-Module) (2026-02-16)

**Contexto:** `consciousness.py` era un god module de 3,277 lineas con 41 MCP tools. Imposible de navegar, testear o refactorizar de forma segura.

**Decision:** Strangler fig pattern:
1. Extraer funcionalidad a 8 modulos cohesivos: `n8n`, `emotion`, `prediction`, `workspace`, `learning`, `self_model`, `curiosity`, `lifecycle`
2. Mantener `consciousness.py` como facade (94 lineas) que re-exporta todo
3. Lazy import para `lifecycle` via `__getattr__` (evita import cycles)
4. `register_tools(mcp)` delega a 8 sub-module register functions

**Alternativas:**
- Reescribir de cero (descartado: riesgo alto, rompe todo)
- Partir en 2-3 modulos grandes (descartado: seguirian siendo god modules)

**Tradeoff:** Una capa extra de re-exports a cambio de estabilidad de API. Ningun consumidor externo (tests, server.py) necesito cambios.

**Resultado:**
- 8 modulos: 82-845 lineas cada uno (total: 3,433 lineas)
- Facade: 94 lineas
- 41 MCP tools intactos (9+4+6+3+5+5+7+2)
- 386/386 tests green, 0 cambios externos

**Estado:** DONE. Verificado con contract checks + parity harness.

---

## D006 — SLOs en 3 Capas (2026-02-16)

**Contexto:** Sin SLOs, no hay forma de saber si un cambio degrada performance o si el sistema "se siente lento".

**Decision:** 3 capas de SLOs documentadas en `docs/SLO.md`:
1. **Interactive** (UX): despertar <= 4s p50, recall <= 3s p50, fast tools <= 50ms p50
2. **Write-path** (eventual): ACK <= 1s p50 (target), completitud <= 5min p95
3. **Sleep Loop**: 8s budget total, tick-level SLOs

**Budget categories** en `config.py:PERF_CONTRACTS`: macro, search, write, fast, consolidation, default.

**Alternativas:**
- Un solo SLO global (descartado: no distingue UX-blocking de background)
- Sin SLOs (descartado: "vibes-based engineering")

**Tradeoff:** SLOs interim para write-path son honestos pero feos (remember p50 = 45s). La solucion real es E2.3 (async write-path).

**Estado:** DONE. Baseline documentado en `docs/perf/baseline_2026-02-16.md`.

---

## D007 — Parity Harness para Refactors Seguros (2026-02-16)

**Contexto:** El D5 split y futuros refactors pueden cambiar outputs criticos sin que nadie se de cuenta hasta produccion.

**Decision:** Snapshot tests deterministas en `tests/parity/`:
- 10 snapshots cubriendo 5 outputs criticos (bridge, despertar, sleep report, tick status, write-path)
- Reloj congelado + random seed + mocks para 100% determinismo
- `UPDATE_SNAPSHOTS=1` para regenerar despues de cambios intencionales
- Clock patching solo en modulos bajo test (no config global)

**Alternativas:**
- Assertion-only tests sin snapshots (descartado: no captura regresiones sutiles en formato)
- Golden file testing con LLM output (descartado: no determinista)

**Tradeoff:** Snapshots son fragiles ante cambios intencionales (requieren regeneracion). Pero capturan regresiones que assertion tests no detectan.

**Estado:** DONE. 11 tests, 42/42 health suite green x3 runs.

---

## D008 — Async Write-Path (Propuesta) (2026-02-16)

**Contexto:** Write-path sincrono (mem0 -> OpenAI -> Qdrant) bloquea 30-80s por memoria. El p99 de `remember` llega a 302s (5 minutos).

**Decision propuesta:**
- `remember()` retorna ACK inmediato (< 2s) con `memory_id` + `status="queued"`
- Background worker procesa la cola
- Completitud SLO: <= 5 min p95
- Backpressure: si la cola excede N items, degradar a local-only mode

**Alternativas:**
- Optimizar pipeline sincrono (limitado: el cuello es OpenAI API)
- Cambiar a embeddings locales (descartado: calidad inferior, nueva dependencia)

**Tradeoff:** Eventual consistency (la memoria no esta disponible para busqueda inmediata) a cambio de UX no-bloqueante.

**Estado:** OPEN (E2.3, issue #29).
