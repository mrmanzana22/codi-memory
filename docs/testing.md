# Testing Guide

## Philosophy

Preferimos tests que validen **mecanismos** (continuity, prediction error, homeostasis, parity) y no solo "no crashea". Cada test bucket tiene base neurocientifica o contractual documentada.

## Test Suite Overview

**386 tests** across 20 files, organized in 5 suites.

| Suite | Marker | Tests | Purpose | Speed |
|-------|--------|-------|---------|-------|
| Full regression | (none) | 386 | Everything | ~40s |
| Continuity battery | `continuity` | 31 | Cross-session pipeline (10 buckets) | ~22s |
| Parity harness | `parity` | 11 | Snapshot tests for critical outputs | ~10s |
| Behavioral battery | `battery` | 20 | End-to-end cognitive mechanisms + ablations | ~15s |
| Health signal | `continuity or parity` | 42 | Quick gate check | ~25s |

## Running Tests

```bash
# Full regression
./venv/bin/pytest tests/ -v

# Health signal (recommended for pre-commit)
./venv/bin/pytest -m "continuity or parity" -v

# Continuity battery only
./venv/bin/pytest -m continuity -v

# Parity snapshots only
./venv/bin/pytest -m parity -v

# Behavioral battery + ablations
./venv/bin/pytest -m battery -v

# Single test file
./venv/bin/pytest tests/test_sleep_loop.py -v
```

## Continuity Battery (31 tests, 10 buckets)

File: `tests/test_continuity_battery.py`

| Bucket | Tests | Validates |
|--------|-------|-----------|
| 1. Checkpoint Fidelity | 3 | PAD, intentions, WM items captured correctly |
| 2. Bridge Reconstruction | 2 | Narrative text grounded in data, hours_since accuracy |
| 3. Cross-Session PE | 5 | Expired intentions, external events generate PEs |
| 4. Synaptic Homeostasis | 3 | Salience decay, floor respect, emotional decay |
| 5. Prospective Memory | 3 | Activation floor, deadline urgency, roundtrip survival |
| 6. Autobiographical Continuity | 2 | Bridge text cap, temporal markers |
| 7. State-Dependent Retrieval | 2 | PAD restored at wakeup, trigger marks bridge |
| 8. Sleep Loop Mechanics | 2 | Tick order, report format cap |
| 9. Source Priority | 1 | Hook cannot upgrade flush checkpoint |
| 10. Golden E2E | 2 | Full pipeline grounded, graceful without checkpoint |

## Parity Harness (11 tests, 5 snapshot classes)

File: `tests/parity/test_parity_harness.py`

Protects load-bearing contracts via deterministic snapshot comparison. If a refactor silently changes a critical output, parity tests catch it.

| Snapshot Class | Tests | What's Protected |
|---------------|-------|-----------------|
| Bridge Output | 2 | `load_session_bridge()` structure + PE for expired intentions |
| Despertar Sections | 2 | Section headers, grounding, intentions in wake-up text |
| Sleep Report Format | 2 | `format_sleep_report()` string shape (normal + skipped tick) |
| Tick Status | 3 | Tick order + status under normal/zero/tight budgets |
| Write-path Smoke | 2 | Report produced + idempotent write |

### Regenerating Snapshots

After an **intentional** change to a protected output:

```bash
UPDATE_SNAPSHOTS=1 pytest tests/parity/ -v
```

This overwrites `tests/parity/snapshots/*.json` and `*.txt`. Review the diff before committing.

### Determinism Guarantees

The `parity_env` fixture provides:
- Frozen clock (`FIXED_NOW = 2026-02-16T14:00:00-05:00`)
- `random.seed(0)` + `random.gauss` stubbed to 0
- All external deps mocked (Qdrant, mem0, OpenAI)
- Isolated SQLite DBs with migrations applied
- Normalization layer: timestamps replaced, floats rounded, IDs masked

**Important:** Clock patching only targets module-level references in modules under test (`session_bridge`, `sleep_loop`), NOT `modules.config`. Patching config globally causes cross-test contamination.

## Behavioral Battery (20 tests)

Files: `tests/test_task_battery.py` (12) + `tests/test_task_battery_ablations.py` (4+)

Tests 6 cognitive mechanisms end-to-end. Ablation tests disable a single module and verify measurable degradation, proving each module contributes to behavior (not dead code).

## Key Fixtures

| Fixture | Scope | What It Does |
|---------|-------|-------------|
| `_isolate_sqlite` | autouse | Redirects all SQLite to tmp_path per test |
| `clean_event_bus` | manual | Saves/restores event bus history |
| `continuity_db` | manual | Isolated DB with full migrations + WM tables |
| `mock_external_deps` | manual | Mocks 7+ external sources for checkpoint tests |
| `parity_env` | manual | Deterministic world (frozen clock, mocked externals) |
| `clean_pad` | manual | Saves/restores PAD emotional state |

## Gates

| Gate | Marker/Command | When to Run |
|------|---------------|-------------|
| G1 (SLOs defined) | N/A | Manual review of `docs/SLO.md` |
| G2 (Battery + Parity) | `pytest -m "continuity or parity"` | Before any merge to main |
| G3 (Perf contracts) | `perf_report_tool(days=7)` | Weekly + before releases |
| G4 (D5 complete) | `pytest tests/ -q` | After module splits |
| G5 (Schema migrations) | Manual review | Before data model changes |

## Debug Tips

1. **Mock where consumed, not where re-exported.** If `consciousness.py` re-exports `despertar_codi` from `lifecycle.py`, patch `modules.lifecycle.despertar_codi`, not `modules.consciousness.despertar_codi`.

2. **Clock contamination.** If a test uses frozen time, only patch the module-level `now_col`/`now_iso` in the specific module you're testing. Never patch `modules.config.now_col` globally.

3. **SQLite isolation.** The autouse `_isolate_sqlite` fixture ensures each test gets a fresh DB. If you need to pre-populate data, use the `continuity_db` fixture or create your own with `apply_migrations()`.

4. **Parity failures after intentional changes.** Run `UPDATE_SNAPSHOTS=1 pytest tests/parity/ -v`, review the diff, commit the new snapshots.

5. **Flaky tests.** Most flakiness comes from real-time dependencies. Use frozen clocks and deterministic mocks. The tolerance window in time-based assertions should be generous (>= 0.5h).
