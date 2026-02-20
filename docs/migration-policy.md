# Migration Policy

**Status:** Active
**Decision:** D004 (DECISIONS.md)
**Issue:** #25 (E5.1)

## Principles

1. **Forward-only.** No rollback migrations. If a migration is wrong, write a new one that fixes it.
2. **Migrations are the single source of truth.** Zero `CREATE TABLE` outside of migration directories.
3. **Fail fast over auto-heal.** Checksum mismatch or missing table = immediate error, not silent `IF NOT EXISTS`.
4. **One transaction per migration.** Failure rolls back the individual migration, not the whole set.
5. **Immutable after application.** Applied migration files must never be modified (checksum enforcement).

## Migration Directories

| Directory | Database | Purpose |
|-----------|----------|---------|
| `migrations/` | `memories_fts.db` | FTS, metrics, consolidation, working memory, events, predictions |
| `migrations_prospective/` | `prospective.db` | Prospective memory (intentions, intention_log) |

Each directory is independent. Migrations in `migrations/` do NOT affect `prospective.db` and vice versa.

## Naming Convention

```
NNN_short_description.sql
```

- `NNN` = zero-padded 3-digit sequence (001, 002, 003...)
- `short_description` = snake_case, describes what the migration does
- Examples: `001_fts_baseline.sql`, `003_async_write_queue.sql`

## File Format

```sql
-- NNN_short_description.sql
-- Brief description of what this migration adds/changes.
-- Context: why this migration exists.

CREATE TABLE IF NOT EXISTS new_table (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ...
);

CREATE INDEX IF NOT EXISTS idx_new_table_col ON new_table(col);
```

Rules:
- Use `IF NOT EXISTS` for idempotency (safe to re-run on existing installs)
- One logical change per migration (don't mix unrelated tables)
- Include comments explaining the "why"
- SQLite-compatible syntax only

## How Migrations Run

1. `apply_migrations(db_path, migrations_dir)` discovers all `.sql` files sorted by version prefix
2. Compares against `schema_migrations` table (tracks what's already applied)
3. For each pending migration:
   - Executes the SQL via `executescript()`
   - Computes schema fingerprint from `sqlite_master`
   - Records version, name, checksum, fingerprint in `schema_migrations`
4. If a previously applied migration's checksum changed: **RuntimeError** (immutability enforcement)

## Adding a New Migration

### Step 1: Create the SQL file

```bash
# In the appropriate directory
touch migrations/003_async_write_queue.sql
```

### Step 2: Write the SQL

Follow the file format above. Test it manually first:

```bash
python3 -c "
import sqlite3
conn = sqlite3.connect(':memory:')
with open('migrations/003_async_write_queue.sql') as f:
    conn.executescript(f.read())
print('OK')
"
```

### Step 3: Apply locally

```bash
python3 -c "
from modules.migrations import apply_migrations
result = apply_migrations('memories_fts.db', 'migrations')
print(result)
"
```

### Step 4: Verify

```bash
# Run full test suite
./venv/bin/pytest tests/ -q

# Check schema_migrations table
python3 -c "
import sqlite3
db = sqlite3.connect('memories_fts.db')
for r in db.execute('SELECT * FROM schema_migrations ORDER BY version'):
    print(r)
"
```

### Step 5: Commit

Include the migration file and any code changes in the same PR.

## Backfill Strategy

When a migration adds columns or tables that need historical data populated:

### For new tables (no existing data)
- Migration creates the table
- Code starts writing to it immediately
- No backfill needed

### For new columns on existing tables
- Migration adds the column with a sensible DEFAULT
- Write a backfill script in `scripts/backfill_NNN_description.py`
- Backfill script must be:
  - **Idempotent** (safe to run multiple times)
  - **Batched** (process N rows at a time, not entire table)
  - **Logged** (print progress, write to consolidation_cron.log)
- Run backfill after migration is applied
- Document in the migration's comments or PR description

### For data transformations
- Never modify existing data in the migration SQL itself
- Use a separate backfill script that reads old format and writes new format
- Keep old columns until backfill is confirmed complete

## Operational Procedures

### Checking current schema version

```bash
python3 -c "
from modules.migrations import get_current_version
print('FTS:', get_current_version('memories_fts.db'))
print('Prospective:', get_current_version('prospective.db'))
"
```

### Detecting schema drift

```bash
python3 -c "
from modules.migrations import apply_migrations
# Dry check: will report what would be applied
import sqlite3
from modules.migrations import discover_migrations, get_applied_versions
conn = sqlite3.connect('memories_fts.db')
applied = get_applied_versions(conn)
available = discover_migrations('migrations')
for v, name, sql, checksum in available:
    status = 'applied' if v in applied else 'PENDING'
    print(f'  [{status}] {name}')
conn.close()
"
```

### Emergency: corrupted migration state

If `schema_migrations` is inconsistent (should be extremely rare):

1. **Never delete rows from schema_migrations** without understanding why
2. Check the actual schema against what migrations should have created
3. If needed, manually insert the correct row:

```sql
INSERT INTO schema_migrations (version, name, checksum, schema_fingerprint)
VALUES ('001', '001_fts_baseline', '<correct_checksum>', '<fingerprint>');
```

## Gate: G5

No schema change may be merged without:
- [ ] A numbered migration file in the correct directory
- [ ] Backfill script (if applicable) with idempotency guarantee
- [ ] Migration tested locally (apply + full test suite)
- [ ] PR description documents what tables/columns change and why
