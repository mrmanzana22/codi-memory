# Testing Protocol: Sprint 15.5 — Structured Goal Context

## Overview
Sprint 15.5 replaces the monolithic `context` blob in goals with 4 structured fields:
- `goal_what` (committed, permanent)
- `goal_why` (committed, permanent)
- `goal_last_state` (derivable, refreshable)
- `goal_next_step` (derivable, refreshable)
- `context_updated_at` (auto-managed timestamp)

## Phase 1: Unit Tests (Automated)

**Run:** `./venv/bin/pytest tests/test_goals.py -v`

**Expected:** 45/45 passed

| Test Class | Count | What it verifies |
|-----------|-------|-----------------|
| TestCreateGoal | 10 | Formulate + structured fields + warnings |
| TestGetActiveGoals | 4 | Select + activation ranking |
| TestUpdateGoal | 5 | Change + access tracking |
| TestAssignGoal | 1 | Delegate |
| TestCompleteGoal | 4 | Achieve + cascade |
| TestGoalHygiene | 3 | Monitor + staleness |
| TestActivation | 2 | ACT-R activation dynamics |
| TestInterferenceLevel | 2 | Altmann & Trafton threshold |
| TestGoalTree | 2 | Hierarchy view |
| TestGoalLog | 3 | Audit trail |
| TestStructuredContext | 5 | NEW: structured fields persist, touch updates derivable only |
| TestContextStaleness | 4 | NEW: stale/empty warnings |

## Phase 2: Integration Test (Semi-Automated)

### 2.1 Verify Migration Applied
```bash
sqlite3 ~/codi-memory/prospective.db "PRAGMA table_info(goals);"
```
**Expected:** columns 13-17 = goal_what, goal_why, goal_last_state, goal_next_step, context_updated_at

### 2.2 Verify Existing Goals Enriched
```bash
sqlite3 ~/codi-memory/prospective.db "SELECT id, title, goal_what IS NOT NULL, goal_why IS NOT NULL FROM goals;"
```
**Expected:** All 5 goals have goal_what=1 and goal_why=1

### 2.3 Verify MCP Tool Parameters
Start MCP server and check that `crear_goal` accepts: goal_what, goal_why, goal_next_step
Check that `actualizar_goal` accepts: goal_last_state, goal_next_step

## Phase 3: End-to-End Test (Manual — Requires Claude Code Restart)

### 3.1 Despertar Shows Structured Context
1. Restart Claude Code (new session)
2. Verify despertar_codi() output includes:
   ```
   ## ACTIVE GOALS (X/Y above interference)
   - [project][high] Proyecto Consciencia (act=X.XX)
     WHAT: Sistema cognitivo con 5 loops de integracion...
     NEXT: Poblar goals reales del proyecto...
   ```
3. **PASS criteria:** Each goal above interference shows WHAT + NEXT lines

### 3.2 Staleness Warning Works
1. Manually age one goal's context:
   ```bash
   sqlite3 ~/codi-memory/prospective.db "UPDATE goals SET context_updated_at = '2026-01-01' WHERE id = '010eb724';"
   ```
2. Run despertar or contexto_goals()
3. **PASS criteria:** Output shows `STALE CONTEXT (1 goals)` section
4. Clean up:
   ```bash
   sqlite3 ~/codi-memory/prospective.db "UPDATE goals SET context_updated_at = datetime('now') WHERE id = '010eb724';"
   ```

### 3.3 No-Context Warning Works
1. Create a goal without structured fields:
   ```
   crear_goal(title="Test empty", level="task", priority="high")
   ```
2. Run contexto_goals()
3. **PASS criteria:** Output includes warning about "NO context set"
4. Verify response includes `warnings` array with goal_what and goal_why messages
5. Clean up: delete or complete the test goal

### 3.4 Touch Updates Derivable Only
1. Create a goal with full context:
   ```
   crear_goal(title="Touch test", level="task", goal_what="Original what", goal_why="Original why", goal_next_step="Original step")
   ```
2. Update derivable fields:
   ```
   actualizar_goal(goal_id="<id>", goal_last_state="New state", goal_next_step="New step")
   ```
3. Verify with `ver_goals()` that:
   - goal_what = "Original what" (unchanged)
   - goal_why = "Original why" (unchanged)
   - goal_last_state = "New state" (updated)
   - goal_next_step = "New step" (updated)
4. **PASS criteria:** Committed fields unchanged, derivable fields updated

### 3.5 New Codi Context Comprehension Test
1. Restart Claude Code (completely new session)
2. After despertar loads, ask: "Que sabes de nuestros goals actuales?"
3. **PASS criteria:** Codi should accurately describe what each goal is about
   based on the WHAT/NEXT fields, not hallucinate details
4. **FAIL criteria:** Codi invents details not present in goal context

## Phase 4: Metrics Collection

After 1 week of use, measure:

### 4.1 Context Coverage
```sql
SELECT
  count(*) as total,
  sum(CASE WHEN goal_what IS NOT NULL THEN 1 ELSE 0 END) as has_what,
  sum(CASE WHEN goal_why IS NOT NULL THEN 1 ELSE 0 END) as has_why,
  sum(CASE WHEN goal_next_step IS NOT NULL THEN 1 ELSE 0 END) as has_next
FROM goals WHERE status = 'active';
```
**Target:** >90% coverage on what + why

### 4.2 Context Freshness
```sql
SELECT
  id, title,
  CAST(julianday('now') - julianday(context_updated_at) AS INTEGER) as days_old
FROM goals
WHERE status = 'active'
ORDER BY days_old DESC;
```
**Target:** No active goal with context > 14 days old

### 4.3 Goal Drift Indicator
Compare what Codi says about goals (from despertar output) vs what the goals actually contain.
Log discrepancies as "drift events" via checkpoint_memoria().
**Target:** 0 drift events per week

## Files Changed

| File | Change |
|------|--------|
| `migrations_prospective/003_goal_structured_context.sql` | NEW: 5 ALTER TABLE + migrate |
| `modules/goals.py` | MODIFIED: create_goal, get_active_goals, get_context_goals, update_goal, touch_goal, MCP tools |
| `modules/lifecycle.py` | MODIFIED: despertar section 12b shows WHAT+NEXT |
| `modules/interface.py` | MODIFIED: context_snapshot shows structured fields |
| `CLAUDE.md` | MODIFIED: GOAL PROTOCOL section added |
| `tests/test_goals.py` | MODIFIED: 13 new tests (32→45 total) |
| `tests/PROTOCOL_sprint15_5.md` | NEW: This document |
