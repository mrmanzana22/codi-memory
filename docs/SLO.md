# Codi Memory - Service Level Objectives (SLOs) v0

**Status:** Draft (pending CTO approval)
**Based on:** Baseline report 2026-02-16 (413 calls, 8 days of data)
**Related issues:** E0.1 (#3), E0.2 (#4), G1 (#6)

## Principles

1. **SLOs have two layers:** interactive (UX) and completitud (consistency)
2. **Write-path requires contract change:** sync pipeline (mem0/OpenAI) cannot meet interactive SLOs
3. **Measure, don't guess:** all SLOs are based on real baseline data
4. **Degrade gracefully:** exceed budget = skip/queue, never block the session

## Layer 1: Interactive SLOs

These protect user experience. Violations here mean the system feels "frozen".

| Tool | p50 | p95 | p99 | Error Budget |
|------|-----|-----|-----|-------------|
| despertar_codi | <= 4,000ms | <= 8,000ms | <= 15,000ms | 99% success/week |
| recall | <= 3,000ms | <= 7,000ms | <= 12,000ms | 99% success/week |
| search_memory | <= 2,500ms | <= 5,000ms | <= 8,000ms | 99% success/week |
| context_snapshot (light) | <= 500ms | <= 2,000ms | <= 5,000ms | 99% success/week |
| get_working_memory | <= 50ms | <= 200ms | <= 500ms | 99.9% success/week |
| get_workspace_state | <= 50ms | <= 200ms | <= 500ms | 99.9% success/week |
| evaluar_triggers | <= 50ms | <= 200ms | <= 500ms | 99.9% success/week |
| assessment_report | <= 200ms | <= 500ms | <= 1,000ms | 99% success/week |

### Degradation actions

| Level | Condition | Action |
|-------|-----------|--------|
| Warning | p95 exceeded | Log as `degraded=true`, continue |
| Alert | p99 exceeded | Log + emit PERF_BUDGET_VIOLATION event |
| Critical | Error budget exhausted (>1% failures) | Alert + investigate |

## Layer 2: Write-path SLOs (Async Target)

These apply AFTER async write pipeline is implemented.

| Tool | ACK p50 | ACK p95 | ACK p99 | Completitud p95 |
|------|---------|---------|---------|-----------------|
| remember | <= 1,000ms | <= 2,000ms | <= 4,000ms | <= 5 min |
| add_memory | <= 1,000ms | <= 2,000ms | <= 4,000ms | <= 5 min |
| checkpoint_memoria | <= 1,000ms | <= 2,000ms | <= 4,000ms | <= 30s |

### Interim Sync SLOs (until async is implemented)

These are the "honest" SLOs for current synchronous behavior:

| Tool | p50 | p95 | p99 |
|------|-----|-----|-----|
| remember | <= 45,000ms | <= 90,000ms | <= 180,000ms |
| add_memory | <= 45,000ms | <= 60,000ms | <= 120,000ms |
| checkpoint_memoria | <= 50,000ms | <= 70,000ms | <= 120,000ms |

## Layer 3: Sleep Loop SLOs

| Metric | Target | Measurement |
|--------|--------|-------------|
| Total run time | <= 8,000ms | budget_ms in sleep_loop config |
| prospective tick | <= 1,000ms p95 | tick elapsed_ms |
| health tick | <= 2,000ms p95 | tick elapsed_ms |
| consolidation tick | <= 4,000ms p95 | tick elapsed_ms |
| homeostasis tick | <= 3,000ms p95 | tick elapsed_ms |
| Tick skip rate | < 10% of runs | skipped ticks / total ticks |
| Run success rate | >= 95% | ok runs / total runs |
| Lock contention rate | < 1% | lock acquisition failures |

### Tick ordering contract

Ticks MUST execute in this order: `prospective -> health -> consolidation -> homeostasis`

Rationale: fast ticks first, heavy ticks last. If budget exhaustion occurs, it starves homeostasis (least critical), not prospective (most time-sensitive).

## Budgets by Category

These map to `PERF_CONTRACTS` in `config.py`:

| Category | p95 Budget | p99 Budget | Tools |
|----------|-----------|-----------|-------|
| macro | 2,000ms | 5,000ms | recall, remember (ACK), context_snapshot |
| search | 1,500ms | 3,000ms | search_memory, search_by_theme, search_by_ownership, search_by_emotion |
| write | 1,500ms | 3,000ms | add_memory (ACK), add_memory_smart (ACK) |
| fast | 200ms | 500ms | get_emotional_state, get_working_memory, get_workspace_state, listar_triggers, audit_tools |
| consolidation | 5,000ms | 10,000ms | run_consolidation, dream_consolidation, consolidate_recent |
| default | 1,000ms | 3,000ms | everything else |

## How to Measure

### Automated (already instrumented)
- All MCP tools are automatically timed via `metrics.instrument_mcp()` wrapper
- Data stored in `memories_fts.db` table `tool_calls`
- Use `perf_report_tool(days=7)` for quick check
- Use `scripts/analyze_perf_baseline.py --days 7 --json` for detailed analysis

### Manual verification
```bash
# Quick SLO check
python scripts/analyze_perf_baseline.py --days 7

# Full report with JSON export
python scripts/analyze_perf_baseline.py --days 30 --json --output docs/perf/
```

## Review Schedule

- **Weekly:** Check perf_report_tool for violations
- **Monthly:** Run full baseline analysis and compare to previous month
- **On release:** Run baseline before and after significant changes
- **On violation:** Investigate root cause, document in DECISIONS.md

## Changelog

| Date | Change | Rationale |
|------|--------|-----------|
| 2026-02-16 | v0 draft | Based on first baseline (413 calls, 8 days) |
