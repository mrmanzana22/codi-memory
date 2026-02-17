# Security Test Plan - codi-memory MCP Server

**Version:** 1.0
**Date:** 2026-02-16
**System:** codi-memory (Python MCP server with SQLite FTS5, Qdrant, mem0)
**Scope:** Input validation, write-path integrity, destructive operations, HTTP API, SQLite security, filesystem safety, credential hygiene

---

## 1. Environment Setup

### 1.1 Clone and Isolate

```bash
# Create a staging workspace (never test against production DBs)
export STAGING_DIR=/tmp/codi-memory-security-staging
mkdir -p "$STAGING_DIR"
cp -R /Users/harecjimenez/codi-memory "$STAGING_DIR/codi-memory"
cd "$STAGING_DIR/codi-memory"

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install pytest pytest-timeout httpx  # test dependencies
```

### 1.2 Staging Databases

```bash
# Create fresh staging DBs via migrations (never copy production DBs)
python3 -c "
from modules.migrations import apply_migrations
apply_migrations('$STAGING_DIR/staging_fts.db', migrations_dir='migrations')
apply_migrations('$STAGING_DIR/staging_prospective.db', migrations_dir='migrations_prospective')
print('Staging DBs created.')
"

# Verify tables
sqlite3 "$STAGING_DIR/staging_fts.db" ".tables"
# Expected: consolidation_log  event_counts  failed_searches  fok_calibration_log
#           fts_retry_queue  labile_memories  memories_fts  memories_text
#           narrative_traces  prediction_results  prediction_state  reconsolidation_log
#           schemas  tool_calls  trace_chains  working_memory
#           write_queue  write_queue_log
```

### 1.3 Mock API Keys

```bash
# Create a staging .env (no real credentials)
cat > "$STAGING_DIR/codi-memory/.env" <<'ENVEOF'
USER_ID=test_user
QDRANT_URL=http://localhost:6333
OPENAI_API_KEY=sk-test-fake-key-not-real
N8N_WEBHOOK_BASE=http://localhost:5678/webhook-test
CODI_API_KEY=staging-test-key-12345
CODI_WRITE_MODE=sync
SUPABASE_URL=
SUPABASE_KEY=
ENVEOF
```

### 1.4 Local Qdrant (Optional -- required for integration tests)

```bash
# Run Qdrant locally via Docker
docker run -d --name qdrant-staging \
  -p 6333:6333 -p 6334:6334 \
  qdrant/qdrant:latest

# Verify connectivity
curl -s http://localhost:6333/collections | python3 -m json.tool
```

### 1.5 Verify Baseline

```bash
cd /Users/harecjimenez/codi-memory
source venv/bin/activate

# Run the existing test suite to establish a clean baseline
pytest tests/ -v --tb=short 2>&1 | tail -30

# Confirm all existing tests pass before running security tests
```

---

## 2. Existing Test Suites

### 2.1 Available Test Commands

| Suite | Command | Description |
|-------|---------|-------------|
| All unit tests | `pytest tests/ -v` | Full test suite (~430+ tests) |
| Parity tests | `pytest tests/parity/ -v` | Write-path parity (sync vs shadow vs async) |
| Continuity tests | `pytest tests/ -m continuity -v` | Session continuity across restarts |
| Ablation tests | `pytest tests/ -m ablation -v` | Feature ablation (disable components) |
| Battery tests | `pytest tests/ -m battery -v` | Task battery (macro-tool validation) |
| Write queue | `pytest tests/test_write_queue.py -v` | Enqueue, dedupe, claim, retry, dead letter |
| Write worker | `pytest tests/test_write_worker.py -v` | Background worker job execution |
| Tracing | `pytest tests/test_tracing.py -v` | Trace ID propagation |
| Schemas | `pytest tests/test_schemas.py -v` | Knowledge schema operations |
| Migrations | `pytest tests/test_migrations.py -v` | Migration idempotency |
| Activation | `pytest tests/test_activation.py -v` | ACT-R activation scoring |
| Performance | `pytest tests/test_performance.py -v` | Latency contract validation |

### 2.2 Test Isolation

All tests use the `_isolate_sqlite` fixture from `tests/conftest.py`, which:
- Redirects `FTS_DB_PATH` and `PROSPECTIVE_DB_PATH` to `tmp_path`
- Runs migrations on the isolated DBs
- Automatically restores config after each test via `monkeypatch`

This means security tests can safely use destructive payloads without risk to production data.

### 2.3 Running a Subset

```bash
# Run only tests matching a keyword
pytest tests/ -k "sql" -v

# Run with timeout to catch hangs
pytest tests/ --timeout=30 -v

# Run with verbose failure output
pytest tests/ -v --tb=long -x
```

---

## 3. Security Test Procedures

### A. Input Validation Tests

#### A.1 Target Surface

The following MCP tools accept user-controlled string input and are the primary attack surface:

| Tool | Module | Critical Parameters |
|------|--------|-------------------|
| `recall` | `modules/interface.py` | `query`, `mode`, `limit` |
| `remember` | `modules/interface.py` | `content`, `importance`, `category` |
| `search_memory` | `modules/memory_core.py` | `query`, `limit` |
| `add_memory` | `modules/memory_core.py` | `content`, `category`, `source`, `importance` |
| `add_memory_smart` | `modules/memory_smart.py` | `content`, `category`, `source`, `importance` |
| `search_fts` | `modules/memory_smart.py` | `query`, `limit` |
| `checkpoint_memoria` | `modules/flush.py` | `momento`, `que_paso`, `por_que_importa` |
| `trigger_n8n` | `modules/n8n.py` | `webhook_path`, `data` |
| `delete_by_content` | `modules/memory_core.py` | `content_fragment` |
| `search_by_theme` | `modules/memory_core.py` | `theme` |
| `search_by_ownership` | `modules/memory_core.py` | `source`, `min_confidence` |

#### A.2 Fuzzing Payloads

Use these payloads against every tool parameter listed above:

```python
FUZZ_PAYLOADS = {
    "sql_injection": [
        "'; DROP TABLE memories_text; --",
        "' OR '1'='1",
        "' UNION SELECT * FROM tool_calls; --",
        "'; INSERT INTO write_queue (job_id, kind, payload_json, status, priority, attempts, max_attempts, created_at, updated_at) VALUES ('evil','remember','{}','queued',1,0,1,datetime('now'),datetime('now')); --",
        "Robert'); DROP TABLE memories_fts;--",
        "1; ATTACH DATABASE '/tmp/evil.db' AS evil; --",
    ],
    "fts5_injection": [
        # FTS5 MATCH syntax abuse
        'content MATCH "* OR 1=1"',
        '"*" NOT content',
        "NEAR(test, 0) OR 1",
        '{"query": "test"}',
        "test) OR (1=1",
        # FTS5 special commands (should not be reachable from user input)
        "rebuild",
        "merge=100",
        "integrity-check",
        "optimize",
    ],
    "unicode_abuse": [
        "\x00\x00\x00",           # null bytes
        "\ud800",                  # lone surrogate (invalid UTF-8)
        "A" * 100_000,            # 100KB string
        "A" * 1_000_000,          # 1MB string
        "\n" * 50_000,            # newline flood
        "\u202e" + "test",        # RTL override
        "\ufeff" * 100,           # BOM flood
        "test\x00injection",      # embedded null
    ],
    "json_malformed": [
        '{"key": }',
        '{"key": undefined}',
        "not json at all",
        "",
        "null",
        '{"a":' + '{"b":' * 100 + '"c"' + '}' * 100 + '}',  # deeply nested
        '{"key": "' + "A" * 100_000 + '"}',  # large value
    ],
    "path_traversal": [
        "../../../etc/passwd",
        "..\\..\\..\\etc\\passwd",
        "/etc/passwd",
        "~/../../etc/shadow",
        "%2e%2e%2f%2e%2e%2f",
        "....//....//etc/passwd",
    ],
    "command_injection": [
        "; ls -la /",
        "| cat /etc/passwd",
        "$(whoami)",
        "`id`",
        "${IFS}cat${IFS}/etc/passwd",
    ],
    "type_confusion": [
        None,
        42,
        3.14,
        True,
        [],
        {},
        ["nested", "list"],
        {"nested": "dict"},
    ],
}
```

#### A.3 How to Invoke

**Direct Python call (preferred for unit tests):**

```python
# Example: fuzz search_fts with SQL injection payloads
from modules.memory_smart import search_fts

for payload in FUZZ_PAYLOADS["sql_injection"]:
    try:
        result = search_fts(query=payload, limit=5)
        # PASS if: returns empty list or raises a handled exception
        # FAIL if: returns data from other tables, crashes with unhandled exception,
        #          or modifies database state
        assert isinstance(result, list), f"Unexpected type: {type(result)}"
    except Exception as e:
        # Acceptable: sqlite3.OperationalError for malformed MATCH
        # Unacceptable: any exception that leaks schema info or stack traces
        assert "OperationalError" in type(e).__name__ or "ValueError" in type(e).__name__, \
            f"Unexpected exception type: {type(e).__name__}: {e}"
```

**Via MCP protocol (integration test):**

```python
# Using the mcp test client (if available), or calling tool functions directly
from modules.interface import recall, remember

for payload in FUZZ_PAYLOADS["sql_injection"]:
    result = recall(query=payload, mode="auto", limit=5)
    # Verify: no SQL error leaks in the response string
    assert "OperationalError" not in result
    assert "sqlite3" not in result.lower()
    assert "traceback" not in result.lower()
```

**Via HTTP API (for SSE/HTTP transport):**

```python
import httpx

BASE = "http://localhost:8000"
HEADERS = {"x-api-key": "staging-test-key-12345"}

for payload in FUZZ_PAYLOADS["sql_injection"]:
    # Test /api/search
    r = httpx.get(f"{BASE}/api/search", params={"q": payload, "limit": "5"}, headers=HEADERS)
    assert r.status_code in (200, 400, 413)
    assert "OperationalError" not in r.text
    assert "Traceback" not in r.text

    # Test /api/memory
    r = httpx.post(f"{BASE}/api/memory", json={"content": payload}, headers=HEADERS)
    assert r.status_code in (200, 400, 413)
    assert "OperationalError" not in r.text
```

#### A.4 Pass/Fail Criteria

| Criterion | PASS | FAIL |
|-----------|------|------|
| SQL injection | Returns empty/error, no data leak | Returns data from unrelated tables |
| FTS5 injection | Returns empty list or graceful error | Executes FTS5 admin commands |
| Unicode abuse | Handles gracefully, stores/rejects cleanly | Crashes, corrupts DB, or truncates silently |
| Large input | Rejects with size error (413/400) | OOM, hangs, or stores unbounded data |
| Type confusion | Returns typed error message | Crashes with TypeError/AttributeError |
| Path traversal | Rejects or sanitizes path | Reads/writes outside expected directories |
| Command injection | Stores as literal string | Executes shell command |

---

### B. Write-Path Integrity Tests

#### B.1 Shadow Mode Verification

```bash
# Enable shadow mode
echo "shadow" > /Users/harecjimenez/codi-memory/.write_mode
# Or: export CODI_WRITE_MODE=shadow
```

```python
# Verify shadow mode enqueues AND executes synchronously
import os
os.environ["CODI_WRITE_MODE"] = "shadow"

from modules.interface import remember
from modules.write_queue import get_queue_stats

stats_before = get_queue_stats(db_path="$STAGING_DIR/staging_fts.db")
result = remember(content="shadow test memory", importance="medium")

# Sync path should have returned a result
assert "guardado" in result.lower() or "enqueued" in result.lower()

# Shadow path should have also enqueued
stats_after = get_queue_stats(db_path="$STAGING_DIR/staging_fts.db")
assert stats_after["total"] >= stats_before["total"]
```

#### B.2 Async Mode Queue Consistency

```python
import os
os.environ["CODI_WRITE_MODE"] = "async"

from modules.interface import remember
from modules.write_queue import get_queue_stats, get_write_job_status
import json

result_json = remember(content="async test memory", importance="high")
result = json.loads(result_json)

# Async mode returns immediate ACK with job_id
assert "job_id" in result or "job_id" in result_json

# Job should be in queued state
# (Extract job_id from result string)
```

#### B.3 Worker Crash Recovery

```python
from modules.write_queue import (
    enqueue_write_job, claim_next_job, get_write_job_status, _get_conn
)

# Enqueue a job
job = enqueue_write_job(
    kind="remember",
    payload={"content": "crash recovery test"},
    priority=5,
    db_path="$STAGING_DIR/staging_fts.db",
)
job_id = job["job_id"]

# Claim it (simulating a worker)
claimed = claim_next_job(lease_seconds=2, db_path="$STAGING_DIR/staging_fts.db")
assert claimed["job_id"] == job_id

# Simulate crash: do NOT call mark_job_done. Wait for lease to expire.
import time
time.sleep(3)

# Another worker should be able to reclaim the stale job
reclaimed = claim_next_job(lease_seconds=120, db_path="$STAGING_DIR/staging_fts.db")
assert reclaimed is not None
assert reclaimed["job_id"] == job_id
assert reclaimed["attempts"] == 2  # incremented on reclaim
```

#### B.4 Dedupe Key Collision Testing

```python
from modules.write_queue import enqueue_write_job, compute_dedupe_key

dedupe = compute_dedupe_key("remember", "test content for collision")

# First enqueue should succeed
r1 = enqueue_write_job(
    kind="remember",
    payload={"content": "test content for collision"},
    dedupe_key=dedupe,
    db_path="$STAGING_DIR/staging_fts.db",
)
assert r1["dedupe_hit"] is False

# Second enqueue with same dedupe_key should return existing job
r2 = enqueue_write_job(
    kind="remember",
    payload={"content": "test content for collision"},
    dedupe_key=dedupe,
    db_path="$STAGING_DIR/staging_fts.db",
)
assert r2["dedupe_hit"] is True
assert r2["job_id"] == r1["job_id"]

# Different content should produce different dedupe key
dedupe2 = compute_dedupe_key("remember", "different content entirely")
assert dedupe != dedupe2
```

#### B.5 Concurrent Write Testing

```python
import threading
import sqlite3
from modules.write_queue import enqueue_write_job, claim_next_job

DB = "$STAGING_DIR/staging_fts.db"
results = {"enqueued": 0, "claimed": set(), "errors": []}
lock = threading.Lock()

def enqueue_worker(n):
    try:
        r = enqueue_write_job(
            kind="remember",
            payload={"content": f"concurrent test {n}"},
            priority=5,
            db_path=DB,
        )
        with lock:
            results["enqueued"] += 1
    except Exception as e:
        with lock:
            results["errors"].append(f"enqueue-{n}: {e}")

def claim_worker():
    try:
        r = claim_next_job(lease_seconds=120, db_path=DB)
        if r:
            with lock:
                results["claimed"].add(r["job_id"])
    except Exception as e:
        with lock:
            results["errors"].append(f"claim: {e}")

# Fire 20 concurrent enqueues
threads = [threading.Thread(target=enqueue_worker, args=(i,)) for i in range(20)]
for t in threads:
    t.start()
for t in threads:
    t.join()

assert results["enqueued"] == 20, f"Only {results['enqueued']}/20 enqueued"
assert len(results["errors"]) == 0, f"Errors: {results['errors']}"

# Fire 20 concurrent claims -- each should get a unique job
results["claimed"].clear()
threads = [threading.Thread(target=claim_worker) for _ in range(20)]
for t in threads:
    t.start()
for t in threads:
    t.join()

# No two threads should have claimed the same job
# (claim uses SELECT then UPDATE WHERE status='queued', not fully atomic in SQLite)
# NOTE: This is a known limitation. Document if duplicates occur.
print(f"Unique jobs claimed: {len(results['claimed'])}")
```

---

### C. Destructive Operation Tests

#### C.1 clear_all_memories Confirmation Bypass

```python
# Locate the clear_all_memories implementation
from modules.memory_core import clear_all_memories

# Test: Does it require confirmation or protect against accidental calls?
# The tool should either:
#   a) Require a confirmation parameter
#   b) Require the caller to be in a specific state
#   c) Be restricted to specific agents/contexts

# If no confirmation is required, this is a FINDING (Medium severity)
# Document the current behavior:
result = clear_all_memories()
# Check: was anything actually deleted? (use staging DB)
```

#### C.2 delete_by_content Without Confirmation

```python
# Test: Can delete_by_content be called with a broad pattern?
from modules.memory_core import delete_by_content

# Attempt to delete with a very broad fragment
result = delete_by_content(content_fragment="a")
# If this deletes all memories containing "a", it is effectively a mass delete
# PASS: rejects fragments shorter than a threshold or requires confirmation
# FAIL: deletes without safeguards
```

#### C.3 Mass Deletion via Search Patterns

```python
# Test: Can search + delete be chained to wipe data?
from modules.memory_core import search_memory, delete_memory

results = search_memory(query="*", limit=1000)
# If search returns all memories with wildcard, an attacker could iterate and delete
# Verify: search_memory limits results, does not support wildcards in dangerous ways
```

---

### D. HTTP API Tests

These tests apply when the server runs in SSE/HTTP mode (`MCP_TRANSPORT=sse`).

#### D.1 Authentication Bypass

```python
import httpx

BASE = "http://localhost:8000"

# Test 1: No API key header
r = httpx.get(f"{BASE}/api/context")
assert r.status_code in (401, 503), f"Expected auth failure, got {r.status_code}"

# Test 2: Wrong API key
r = httpx.get(f"{BASE}/api/context", headers={"x-api-key": "wrong-key"})
assert r.status_code == 401

# Test 3: Empty API key
r = httpx.get(f"{BASE}/api/context", headers={"x-api-key": ""})
assert r.status_code == 401

# Test 4: API key in Bearer format
r = httpx.get(f"{BASE}/api/context", headers={"Authorization": "Bearer staging-test-key-12345"})
assert r.status_code == 200  # Should work (server supports Bearer)

# Test 5: API key via query parameter (should NOT work)
r = httpx.get(f"{BASE}/api/context?api_key=staging-test-key-12345")
assert r.status_code in (401, 503), "API key via query param should not be accepted"

# Test 6: When CODI_API_KEY is unset, only localhost should be allowed
# (requires restarting server without CODI_API_KEY)
```

#### D.2 Rate Limiting Verification

```python
import httpx
import time

BASE = "http://localhost:8000"
HEADERS = {"x-api-key": "staging-test-key-12345"}

# Fire RATE_LIMIT_PER_MIN + 10 requests in rapid succession
# Default RATE_LIMIT_PER_MIN = 60
responses = []
for i in range(70):
    r = httpx.get(f"{BASE}/health")
    responses.append(r.status_code)

# Health endpoint is exempt from rate limiting (code: "if path == '/health': return await call_next(request)")
# So test against /api/context instead:
responses_api = []
for i in range(70):
    r = httpx.get(f"{BASE}/api/context", headers=HEADERS)
    responses_api.append(r.status_code)

rate_limited = [s for s in responses_api if s == 429]
assert len(rate_limited) > 0, "Rate limiting did not engage after 60+ requests"
print(f"Rate limited after {responses_api.index(429) + 1} requests")
```

#### D.3 Payload Size Limits

```python
import httpx

BASE = "http://localhost:8000"
HEADERS = {"x-api-key": "staging-test-key-12345", "Content-Type": "application/json"}

# Test 1: Payload exactly at limit (256KB default)
payload_at_limit = {"content": "A" * 260_000}
r = httpx.post(f"{BASE}/api/memory", json=payload_at_limit, headers=HEADERS)
# Content-Length based check + body-level check
assert r.status_code in (413, 400), f"Expected 413/400, got {r.status_code}"

# Test 2: Payload well under limit
payload_small = {"content": "small test memory"}
r = httpx.post(f"{BASE}/api/memory", json=payload_small, headers=HEADERS)
assert r.status_code == 200

# Test 3: Missing Content-Length header (chunked transfer)
# The server checks body length after reading for /recordatorio and /api/memory
# but only checks Content-Length header in middleware
# This is a potential bypass -- document if body is not size-checked
r = httpx.post(
    f"{BASE}/recordatorio",
    content=b'{"mensaje": "' + b"A" * 300_000 + b'"}',
    headers={**HEADERS, "Transfer-Encoding": "chunked"},
)
# PASS: rejected with 413
# FAIL: accepted despite exceeding MAX_BODY_BYTES
```

#### D.4 SSRF via Webhook URLs

```python
# The trigger_n8n function constructs URLs from N8N_WEBHOOK_BASE + webhook_path
# Test: Can webhook_path be manipulated to hit internal services?

from modules.n8n import trigger_n8n

# webhook_path validation: only allows [A-Za-z0-9_-], max 80 chars
# Test bypass attempts:

ssrf_payloads = [
    "../../internal-service",
    "codi-alerta?url=http://169.254.169.254/latest/meta-data/",
    "codi-alerta#@evil.com",
    "codi-alerta/../../admin",
    "codi-alerta%00evil",
    "@evil.com/webhook",
    "codi-alerta\n\rHost: evil.com",
]

for payload in ssrf_payloads:
    result = trigger_n8n(webhook_path=payload)
    # PASS: "Error: webhook_path invalido" for all payloads
    assert "invalido" in result.lower() or "error" in result.lower(), \
        f"SSRF payload accepted: {payload!r} -> {result}"
```

---

### E. SQLite Security Tests

#### E.1 SQL Injection via Tool Parameters

```python
# The primary SQL injection surface is search_fts which uses FTS5 MATCH
# FTS5 MATCH has its own query syntax and is not raw SQL, but test anyway

from modules.memory_smart import search_fts

injection_payloads = [
    "'; DROP TABLE memories_text; --",
    "' OR '1'='1",
    "test' UNION SELECT sql FROM sqlite_master --",
    "test'; ATTACH DATABASE '/tmp/pwned.db' AS pwned; --",
    "test\"; .system ls",
]

for payload in injection_payloads:
    result = search_fts(query=payload, limit=5)
    # FTS5 MATCH should reject these as malformed queries
    assert isinstance(result, list)
    # Verify the memories_text table still exists
    import sqlite3
    from modules.config import FTS_DB_PATH
    conn = sqlite3.connect(FTS_DB_PATH)
    tables = [r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    ).fetchall()]
    conn.close()
    assert "memories_text" in tables, f"TABLE DROPPED by payload: {payload!r}"
```

```python
# Test parameterized queries in write_queue.py
from modules.write_queue import enqueue_write_job, get_write_job_status

# Attempt injection via job_id lookup
malicious_id = "'; DROP TABLE write_queue; --"
result = get_write_job_status(job_id=malicious_id)
assert result is None  # Should just return None (not found)

# Verify table still intact
import sqlite3
from modules.config import FTS_DB_PATH
conn = sqlite3.connect(FTS_DB_PATH)
tables = [r[0] for r in conn.execute(
    "SELECT name FROM sqlite_master WHERE type='table'"
).fetchall()]
conn.close()
assert "write_queue" in tables, "write_queue table was dropped!"
```

#### E.2 WAL Mode Integrity Under Concurrent Access

```python
import sqlite3
import threading
import time
from modules.config import FTS_DB_PATH

DB = FTS_DB_PATH  # use staging DB
errors = []

def reader():
    try:
        conn = sqlite3.connect(DB, timeout=5)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=5000")
        for _ in range(100):
            conn.execute("SELECT COUNT(*) FROM memories_text").fetchone()
            time.sleep(0.01)
        conn.close()
    except Exception as e:
        errors.append(f"reader: {e}")

def writer():
    try:
        conn = sqlite3.connect(DB, timeout=5)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=5000")
        for i in range(50):
            conn.execute(
                "INSERT OR REPLACE INTO memories_text (memory_id, content, category, source, importance) "
                "VALUES (?, ?, 'test', 'test', 'low')",
                (f"wal-test-{i}", f"WAL concurrent test {i}")
            )
            conn.commit()
            time.sleep(0.02)
        conn.close()
    except Exception as e:
        errors.append(f"writer: {e}")

# 5 readers + 2 writers concurrently
threads = [threading.Thread(target=reader) for _ in range(5)]
threads += [threading.Thread(target=writer) for _ in range(2)]
for t in threads:
    t.start()
for t in threads:
    t.join()

assert len(errors) == 0, f"WAL integrity errors: {errors}"
```

#### E.3 Migration Rollback Safety

```python
# Verify migrations are idempotent (running twice does not corrupt)
from modules.migrations import apply_migrations
import os

db_path = "/tmp/migration_safety_test.db"
migrations_dir = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "migrations"
)

# Run once
r1 = apply_migrations(db_path, migrations_dir)
# Run again (should skip all)
r2 = apply_migrations(db_path, migrations_dir)

assert len(r2["applied"]) == 0, f"Migrations re-applied: {r2['applied']}"
assert set(r2["skipped"]) == set(r1["applied"]) | set(r1["skipped"])

# Verify no data loss
import sqlite3
conn = sqlite3.connect(db_path)
tables = [r[0] for r in conn.execute(
    "SELECT name FROM sqlite_master WHERE type='table'"
).fetchall()]
conn.close()
assert "memories_text" in tables
assert "write_queue" in tables
assert "tool_calls" in tables

os.unlink(db_path)
```

---

### F. Filesystem Tests

#### F.1 Path Traversal in export_to_markdown

```python
# export_to_markdown writes to MARKDOWN_DIR (config.py: os.path.join(BASE_DIR, "markdown"))
# Verify it does not write outside that directory

import os
from modules.config import MARKDOWN_DIR, BASE_DIR

# Before export, check MARKDOWN_DIR is safely rooted
assert MARKDOWN_DIR.startswith(BASE_DIR), \
    f"MARKDOWN_DIR ({MARKDOWN_DIR}) is not under BASE_DIR ({BASE_DIR})"

# Check: can a memory with path-traversal content in its category cause
# files to be written outside MARKDOWN_DIR?
# The CATEGORY_FILE_MAP is hardcoded in config.py, so categories map to fixed filenames.
# But verify no dynamic path construction uses user input as filename.

from modules.config import CATEGORY_FILE_MAP
for cat, filename in CATEGORY_FILE_MAP.items():
    full_path = os.path.join(MARKDOWN_DIR, filename)
    assert os.path.normpath(full_path).startswith(os.path.normpath(MARKDOWN_DIR)), \
        f"Category '{cat}' maps to path outside MARKDOWN_DIR: {full_path}"
```

#### F.2 JSON File Corruption Recovery

```python
import json
import os

BACKUP = "/tmp/test_backup.json"

# Test 1: Corrupted JSON
with open(BACKUP, "w") as f:
    f.write('{"memories": [{"invalid json')

# Verify restore_memories handles this gracefully
from modules.memory_core import restore_memories
# Monkey-patch BACKUP_FILE for this test
import modules.config
original = modules.config.BACKUP_FILE
modules.config.BACKUP_FILE = BACKUP
try:
    result = restore_memories()
    assert "error" in result.lower() or "Error" in result, \
        f"Corrupted JSON not handled: {result}"
finally:
    modules.config.BACKUP_FILE = original
    os.unlink(BACKUP)

# Test 2: Empty file
with open(BACKUP, "w") as f:
    f.write("")
modules.config.BACKUP_FILE = BACKUP
try:
    result = restore_memories()
    assert "error" in result.lower() or "0" in result
finally:
    modules.config.BACKUP_FILE = original
    os.unlink(BACKUP)

# Test 3: Valid JSON but wrong structure
with open(BACKUP, "w") as f:
    json.dump({"not_memories": "wrong"}, f)
modules.config.BACKUP_FILE = BACKUP
try:
    result = restore_memories()
    assert "0" in result or "Restauradas 0" in result
finally:
    modules.config.BACKUP_FILE = original
    os.unlink(BACKUP)
```

#### F.3 Plist and Data Directory Permission Checks

```bash
# Verify the launchd plist does not have overly permissive permissions
stat -f "%Sp %Su %Sg" /Users/harecjimenez/codi-memory/com.codi.sleep-loop.plist
# Expected: -rw-r--r-- harecjimenez staff (or more restrictive)
# FAIL if: world-writable (-rw-rw-rw-)

# Check data directory permissions
stat -f "%Sp %Su %Sg" /Users/harecjimenez/codi-memory/data/
# Expected: drwxr-xr-x or drwx------

# Check .env permissions (should not be world-readable)
stat -f "%Sp %Su %Sg" /Users/harecjimenez/codi-memory/.env
# Expected: -rw------- (600) or -rw-r----- (640)
# FAIL if: -rw-r--r-- or more permissive

# Check DB file permissions
stat -f "%Sp %Su %Sg" /Users/harecjimenez/codi-memory/memories_fts.db
# Expected: -rw-r--r-- or -rw-------
```

---

### G. Credential Safety Tests

#### G.1 Verify API Keys Not Logged in tool_calls

```python
import sqlite3
from modules.config import FTS_DB_PATH

conn = sqlite3.connect(FTS_DB_PATH)
conn.row_factory = sqlite3.Row

# Check tool_calls table for any credential-like patterns
rows = conn.execute("SELECT * FROM tool_calls ORDER BY started_at DESC LIMIT 100").fetchall()

CREDENTIAL_PATTERNS = [
    "sk-",           # OpenAI key prefix
    "eyJ",           # JWT token prefix (base64 of '{"')
    "Bearer ",       # Auth header value
    "CODI_API_KEY",  # Env var name in data
    "SUPABASE_KEY",  # Env var name in data
    "password",
    "secret",
]

for row in rows:
    row_str = str(dict(row))
    for pattern in CREDENTIAL_PATTERNS:
        assert pattern not in row_str, \
            f"Credential pattern '{pattern}' found in tool_calls row: {row['tool_name']} at {row['started_at']}"

conn.close()
```

#### G.2 Verify Tokens Not in write_queue Payloads

```python
import sqlite3
from modules.config import FTS_DB_PATH

conn = sqlite3.connect(FTS_DB_PATH)

# Check write_queue payload_json for credential leaks
rows = conn.execute("SELECT job_id, kind, payload_json FROM write_queue").fetchall()

for job_id, kind, payload_json in rows:
    if payload_json:
        for pattern in CREDENTIAL_PATTERNS:
            assert pattern not in payload_json, \
                f"Credential pattern '{pattern}' found in write_queue payload for job {job_id} ({kind})"

conn.close()
```

#### G.3 Check Log Files for Credential Leaks

```bash
# Check all log files and stderr captures for leaked credentials
LOG_FILES=(
    "/Users/harecjimenez/codi-memory/data/write_worker_stderr.log"
    "/Users/harecjimenez/codi-memory/data/sleep_loop_stderr.log"
)

PATTERNS="sk-|eyJ|Bearer |CODI_API_KEY=|SUPABASE_KEY=|OPENAI_API_KEY="

for f in "${LOG_FILES[@]}"; do
    if [ -f "$f" ]; then
        echo "=== Checking: $f ==="
        # Use grep -c to count matches (0 = clean)
        count=$(grep -ciE "$PATTERNS" "$f" 2>/dev/null || echo 0)
        if [ "$count" -gt 0 ]; then
            echo "WARNING: Found $count potential credential leaks in $f"
            grep -inE "$PATTERNS" "$f" | head -5
        else
            echo "CLEAN: No credential patterns found"
        fi
    else
        echo "SKIP: $f does not exist"
    fi
done
```

#### G.4 Verify .env Not Committed to Git

```bash
cd /Users/harecjimenez/codi-memory

# Check if .env is in .gitignore
if [ -f .gitignore ]; then
    if grep -q "\.env" .gitignore; then
        echo "PASS: .env is in .gitignore"
    else
        echo "FAIL: .env is NOT in .gitignore"
    fi
else
    echo "WARNING: No .gitignore file found"
fi

# Check if .env is tracked by git
if [ -d .git ]; then
    if git ls-files --cached .env | grep -q ".env"; then
        echo "FAIL: .env is tracked by git (committed)"
    else
        echo "PASS: .env is not tracked by git"
    fi
fi

# Check if backup files with potential credentials are committed
git ls-files --cached "*.json" 2>/dev/null | while read f; do
    echo "CHECK: $f is tracked by git -- verify it contains no credentials"
done
```

---

## 4. Monitoring and Observability

### 4.1 Shadow Report

```bash
cd /Users/harecjimenez/codi-memory
source venv/bin/activate

# Run shadow report for the last hour
./venv/bin/python scripts/shadow_report.py --hours 1

# Run for last 48 hours with custom DB
./venv/bin/python scripts/shadow_report.py --hours 48 --db memories_fts.db
```

### 4.2 Tool Call Audit

```bash
# Recent tool calls (last 20)
sqlite3 /Users/harecjimenez/codi-memory/memories_fts.db \
  "SELECT tool_name, started_at, duration_ms, success, error_type FROM tool_calls ORDER BY started_at DESC LIMIT 20;"

# Failed tool calls
sqlite3 /Users/harecjimenez/codi-memory/memories_fts.db \
  "SELECT tool_name, started_at, duration_ms, error_type FROM tool_calls WHERE success = 0 ORDER BY started_at DESC LIMIT 20;"

# Tool call volume by tool name (last 24h)
sqlite3 /Users/harecjimenez/codi-memory/memories_fts.db \
  "SELECT tool_name, COUNT(*) as calls, AVG(duration_ms) as avg_ms, SUM(CASE WHEN success=0 THEN 1 ELSE 0 END) as failures FROM tool_calls WHERE started_at > datetime('now', '-1 day') GROUP BY tool_name ORDER BY calls DESC;"
```

### 4.3 Write Queue Status

```bash
# Current queue state
sqlite3 /Users/harecjimenez/codi-memory/memories_fts.db \
  "SELECT status, COUNT(*) as cnt FROM write_queue GROUP BY status;"

# Recent jobs
sqlite3 /Users/harecjimenez/codi-memory/memories_fts.db \
  "SELECT job_id, kind, status, attempts, last_error, created_at FROM write_queue ORDER BY created_at DESC LIMIT 20;"

# Dead jobs (exhausted retries)
sqlite3 /Users/harecjimenez/codi-memory/memories_fts.db \
  "SELECT job_id, kind, attempts, max_attempts, last_error, created_at FROM write_queue WHERE status = 'dead' ORDER BY created_at DESC;"

# Stale leases (running but lease expired)
sqlite3 /Users/harecjimenez/codi-memory/memories_fts.db \
  "SELECT job_id, kind, lease_until FROM write_queue WHERE status = 'running' AND lease_until < datetime('now');"
```

### 4.4 Worker and Sleep Loop Logs

```bash
# Write worker stderr (launched by launchd)
cat /Users/harecjimenez/codi-memory/data/write_worker_stderr.log 2>/dev/null || echo "No worker log"

# Sleep loop stderr
cat /Users/harecjimenez/codi-memory/data/sleep_loop_stderr.log 2>/dev/null || echo "No sleep log"

# Tail logs in real time during testing
tail -f /Users/harecjimenez/codi-memory/data/write_worker_stderr.log &
tail -f /Users/harecjimenez/codi-memory/data/sleep_loop_stderr.log &
```

---

## 5. Correlation and Tracing

### 5.1 Trace ID Flow

The tracing system uses `contextvars` (see `/Users/harecjimenez/codi-memory/modules/tracing.py`):

```
Client request
  -> MCP tool invocation
    -> new_trace_id() generates 12-char hex (e.g., "a1b2c3d4e5f6")
    -> trace_id stored in ContextVar (thread-local equivalent for asyncio)
    -> metrics.py reads get_trace_id() when logging to tool_calls
    -> interface.py includes trace_id in JSON responses
```

### 5.2 Cross-Table Correlation

```sql
-- Trace a specific tool call through the system
-- Step 1: Find the tool call by trace_id (stored in 'tag' column of tool_calls)
SELECT id, tool_name, started_at, duration_ms, success, tag
FROM tool_calls
WHERE tag = 'a1b2c3d4e5f6'
ORDER BY started_at;

-- Step 2: If it was a write operation, find the write_queue job
-- The session_id in write_queue can be correlated with session_id in tool_calls
SELECT wq.job_id, wq.kind, wq.status, wq.created_at, wq.completed_at
FROM write_queue wq
WHERE wq.created_at BETWEEN '2026-02-16T00:00:00' AND '2026-02-16T23:59:59'
ORDER BY wq.created_at;

-- Step 3: Check write_queue_log for completion details
SELECT wql.job_id, wql.kind, wql.status, wql.attempts, wql.duration_ms,
       wql.error_class, wql.error_msg, wql.session_id
FROM write_queue_log wql
WHERE wql.job_id = '<job_id_from_step_2>'
ORDER BY wql.created_at;

-- Step 4: Correlate with session checkpoints
SELECT * FROM session_checkpoints
WHERE session_id = '<session_id>'
ORDER BY created_at DESC LIMIT 5;
```

### 5.3 Full Request Lifecycle

```
┌─────────────┐   trace_id    ┌────────────┐  job_id   ┌─────────────┐
│  MCP Client │ ──────────>   │ tool_calls │ ───────>  │ write_queue │
│  (Claude)   │               │  (metrics) │           │  (async)    │
└─────────────┘               └────────────┘           └──────┬──────┘
                                                              │
                                                        ┌─────v─────┐
                                                        │  write_   │
                                                        │ queue_log │
                                                        └───────────┘
```

Key join columns:
- `tool_calls.tag` = trace_id
- `tool_calls.session_id` = session identifier
- `write_queue.job_id` = unique job identifier
- `write_queue_log.job_id` = same as write_queue.job_id
- `write_queue_log.session_id` = session identifier

---

## 6. Regression Test Template

Place new security regression tests in `tests/test_security_regression.py`:

```python
"""
Security regression tests for codi-memory.

Each test documents a specific finding and verifies the fix.
Tests use the _isolate_sqlite fixture from conftest.py for DB isolation.
"""

import json
import os
import sqlite3
import sys

import pytest

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestSQLInjection:
    """Verify SQL injection is not possible via any tool parameter."""

    INJECTION_PAYLOADS = [
        "'; DROP TABLE memories_text; --",
        "' OR '1'='1",
        "' UNION SELECT * FROM tool_calls; --",
        "1; ATTACH DATABASE '/tmp/evil.db' AS evil; --",
    ]

    def test_search_fts_injection(self):
        """Verify SQL injection is not possible via search_fts."""
        from modules.memory_smart import search_fts

        for payload in self.INJECTION_PAYLOADS:
            result = search_fts(query=payload, limit=5)
            assert isinstance(result, list)

    def test_write_queue_status_injection(self):
        """Verify SQL injection not possible via get_write_job_status."""
        from modules.write_queue import get_write_job_status

        for payload in self.INJECTION_PAYLOADS:
            result = get_write_job_status(job_id=payload)
            assert result is None

    def test_write_queue_enqueue_injection(self, tmp_path):
        """Verify SQL injection not possible via enqueue payload."""
        from modules.write_queue import enqueue_write_job
        from modules.migrations import apply_migrations

        db_path = str(tmp_path / "test_inject.db")
        migrations_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "migrations",
        )
        apply_migrations(db_path, migrations_dir)

        for payload in self.INJECTION_PAYLOADS:
            result = enqueue_write_job(
                kind="remember",
                payload={"content": payload},
                db_path=db_path,
            )
            assert result["status"] == "queued"

        # Verify tables are intact
        conn = sqlite3.connect(db_path)
        tables = [
            r[0]
            for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        ]
        conn.close()
        assert "write_queue" in tables
        assert "memories_text" in tables


class TestInputValidation:
    """Verify input validation across tool parameters."""

    def test_recall_invalid_mode(self):
        """Verify recall rejects invalid mode values."""
        from modules.interface import recall

        result = recall(query="test", mode="invalid_mode")
        parsed = json.loads(result)
        assert "invalido" in parsed.get("pretty", "").lower() or "invalid" in parsed.get("pretty", "").lower()

    def test_recall_limit_bounds(self):
        """Verify recall clamps limit to valid range."""
        from modules.interface import recall

        # These should not crash
        recall(query="test", mode="auto", limit=-1)
        recall(query="test", mode="auto", limit=0)
        recall(query="test", mode="auto", limit=999)

    def test_fts5_special_commands_blocked(self):
        """Verify FTS5 admin commands cannot be injected via search."""
        from modules.memory_smart import search_fts

        # These are FTS5 internal commands that should never execute via user search
        dangerous_commands = ["rebuild", "merge=100", "integrity-check", "optimize"]
        for cmd in dangerous_commands:
            result = search_fts(query=cmd, limit=5)
            assert isinstance(result, list)

    def test_unicode_null_bytes(self):
        """Verify null bytes in input do not cause crashes or truncation."""
        from modules.memory_smart import search_fts

        result = search_fts(query="test\x00injection", limit=5)
        assert isinstance(result, list)


class TestSSRF:
    """Verify SSRF protections in webhook/HTTP integrations."""

    def test_n8n_webhook_path_validation(self):
        """Verify trigger_n8n rejects malicious webhook paths."""
        from modules.n8n import trigger_n8n

        malicious_paths = [
            "../../etc/passwd",
            "path?url=http://169.254.169.254",
            "path#@evil.com",
            "path\n\rHost: evil.com",
            "path%00null",
            "",
            "a" * 100,  # exceeds 80 char limit
        ]
        for path in malicious_paths:
            result = trigger_n8n(webhook_path=path)
            assert "invalido" in result.lower() or "error" in result.lower() or "no esta configurado" in result.lower(), \
                f"Path accepted: {path!r}"


class TestDestructiveOps:
    """Verify destructive operations have appropriate safeguards."""

    def test_clear_all_requires_safeguard(self):
        """Document whether clear_all_memories has a confirmation gate."""
        # This test documents current behavior.
        # If clear_all_memories accepts no confirmation parameter,
        # file a finding (see Section 7 template).
        import inspect
        from modules.memory_core import clear_all_memories

        sig = inspect.signature(clear_all_memories)
        params = list(sig.parameters.keys())
        # Ideally there should be a 'confirm' or 'force' parameter
        # Document the finding either way
        print(f"clear_all_memories params: {params}")


class TestCredentialHygiene:
    """Verify credentials are not leaked into observable stores."""

    CREDENTIAL_PATTERNS = ["sk-", "eyJ", "Bearer ", "OPENAI_API_KEY", "SUPABASE_KEY"]

    def test_tool_calls_no_credentials(self):
        """Verify tool_calls table does not contain credential patterns."""
        from modules.config import FTS_DB_PATH

        conn = sqlite3.connect(FTS_DB_PATH)
        rows = conn.execute("SELECT * FROM tool_calls LIMIT 200").fetchall()
        conn.close()

        for row in rows:
            row_str = str(row)
            for pattern in self.CREDENTIAL_PATTERNS:
                assert pattern not in row_str, \
                    f"Credential pattern '{pattern}' found in tool_calls"

    def test_write_queue_no_credentials(self):
        """Verify write_queue payloads do not contain credential patterns."""
        from modules.config import FTS_DB_PATH

        conn = sqlite3.connect(FTS_DB_PATH)
        rows = conn.execute(
            "SELECT job_id, payload_json FROM write_queue LIMIT 200"
        ).fetchall()
        conn.close()

        for job_id, payload_json in rows:
            if payload_json:
                for pattern in self.CREDENTIAL_PATTERNS:
                    assert pattern not in payload_json, \
                        f"Credential pattern '{pattern}' in write_queue job {job_id}"


class TestWritePathIntegrity:
    """Verify write-path consistency and crash recovery."""

    def test_dedupe_key_deterministic(self):
        """Verify compute_dedupe_key is deterministic for same input."""
        from modules.write_queue import compute_dedupe_key

        k1 = compute_dedupe_key("remember", "hello world")
        k2 = compute_dedupe_key("remember", "hello world")
        assert k1 == k2

    def test_dedupe_key_varies_by_kind(self):
        """Verify different kinds produce different dedupe keys."""
        from modules.write_queue import compute_dedupe_key

        k1 = compute_dedupe_key("remember", "hello world")
        k2 = compute_dedupe_key("add_memory", "hello world")
        assert k1 != k2

    def test_lease_expiry_reclaim(self, tmp_path):
        """Verify stale leases are reclaimed by new workers."""
        from modules.write_queue import enqueue_write_job, claim_next_job
        from modules.migrations import apply_migrations
        import time

        db_path = str(tmp_path / "test_lease.db")
        migrations_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "migrations",
        )
        apply_migrations(db_path, migrations_dir)

        job = enqueue_write_job(
            kind="remember",
            payload={"content": "lease test"},
            db_path=db_path,
        )

        # Claim with 1-second lease
        claimed = claim_next_job(lease_seconds=1, db_path=db_path)
        assert claimed["job_id"] == job["job_id"]

        # Wait for lease to expire
        time.sleep(2)

        # Should be reclaimable
        reclaimed = claim_next_job(lease_seconds=120, db_path=db_path)
        assert reclaimed is not None
        assert reclaimed["job_id"] == job["job_id"]
```

### Running the Security Regression Tests

```bash
cd /Users/harecjimenez/codi-memory
source venv/bin/activate

# Run all security regression tests
pytest tests/test_security_regression.py -v --tb=long

# Run a specific class
pytest tests/test_security_regression.py::TestSQLInjection -v

# Run with timeout (catch infinite loops from malicious input)
pytest tests/test_security_regression.py --timeout=10 -v
```

---

## 7. Report Template

Use this template to document each security finding:

```markdown
### Finding: [SEC-NNN] [Short Title]

- **Severity**: Critical / High / Medium / Low / Informational
- **CVSS (if applicable)**: X.X
- **Tool/Endpoint**: `tool_name` or `POST /api/endpoint`
- **Module**: `modules/module_name.py:line_number`
- **CWE**: CWE-XXX (e.g., CWE-89 for SQL Injection)

#### Description
One paragraph describing the vulnerability and its impact.

#### Steps to Reproduce
1. Step one
2. Step two
3. Step three

```python
# Proof of concept code
from modules.xxx import yyy
result = yyy(malicious_input="payload")
```

#### Expected Behavior
What should happen (e.g., "Input is rejected with a 400 error").

#### Actual Behavior
What actually happens (e.g., "SQL query is executed, table is dropped").

#### Evidence
- Screenshot or log output
- Database state before/after
- Network capture (if applicable)

#### Root Cause
Technical explanation of why the vulnerability exists.

#### Remediation
Specific code changes recommended:

```python
# Before (vulnerable)
conn.execute(f"SELECT * FROM table WHERE col = '{user_input}'")

# After (safe)
conn.execute("SELECT * FROM table WHERE col = ?", (user_input,))
```

#### Retest Result
- **Date**: YYYY-MM-DD
- **Commit**: abc1234
- **Result**: PASS / FAIL
- **Notes**: Any observations from retesting
```

### Severity Classification

| Severity | Definition | Example |
|----------|-----------|---------|
| Critical | Remote code execution, full data loss, credential exposure | SQL injection dropping all tables |
| High | Data leak, unauthorized access, persistent data corruption | Authentication bypass on HTTP API |
| Medium | Limited data exposure, denial of service, inconsistent state | Rate limiting bypass, unbounded input |
| Low | Information disclosure, minor integrity issues | Error messages leaking internal paths |
| Informational | Best practice deviation, hardening opportunity | Missing Content-Security-Policy header |

---

## Appendix A: Quick-Reference Commands

```bash
# === TESTING ===
pytest tests/test_security_regression.py -v                    # Security regression suite
pytest tests/ -v --tb=short                                    # Full test suite
pytest tests/test_write_queue.py tests/test_write_worker.py -v # Write-path tests

# === MONITORING ===
./venv/bin/python scripts/shadow_report.py --hours 1           # Shadow mode report

# === AUDIT QUERIES ===
sqlite3 memories_fts.db "SELECT tool_name, COUNT(*), AVG(duration_ms) FROM tool_calls GROUP BY tool_name ORDER BY COUNT(*) DESC;"
sqlite3 memories_fts.db "SELECT status, COUNT(*) FROM write_queue GROUP BY status;"
sqlite3 memories_fts.db "SELECT * FROM write_queue WHERE status='dead';"

# === CREDENTIAL CHECK ===
grep -rn "sk-" data/ *.log 2>/dev/null | grep -v ".pyc"       # OpenAI key leak check
grep -rn "eyJ" data/ *.log 2>/dev/null | grep -v ".pyc"       # JWT leak check

# === FILE PERMISSIONS ===
stat -f "%Sp %SHp %Su" .env memories_fts.db data/              # Check perms
```

## Appendix B: Codebase File Map (Security-Relevant)

| File | Security Relevance |
|------|--------------------|
| `/Users/harecjimenez/codi-memory/server.py` | HTTP security middleware (auth, rate limit, body size) |
| `/Users/harecjimenez/codi-memory/modules/config.py` | Credential loading (.env), DB paths, lazy init |
| `/Users/harecjimenez/codi-memory/modules/memory_core.py` | Core CRUD, destructive ops (clear_all, delete) |
| `/Users/harecjimenez/codi-memory/modules/memory_smart.py` | FTS5 search (MATCH injection surface), dedup |
| `/Users/harecjimenez/codi-memory/modules/write_queue.py` | Async write queue, lease-based claims, dedupe |
| `/Users/harecjimenez/codi-memory/modules/write_worker.py` | Background job executor, payload deserialization |
| `/Users/harecjimenez/codi-memory/modules/n8n.py` | Outbound HTTP (SSRF surface via webhook_path) |
| `/Users/harecjimenez/codi-memory/modules/interface.py` | Macro tools (recall/remember), input validation |
| `/Users/harecjimenez/codi-memory/modules/flush.py` | File I/O (export), session state persistence |
| `/Users/harecjimenez/codi-memory/modules/metrics.py` | Tool call logging (must not log payloads) |
| `/Users/harecjimenez/codi-memory/modules/tracing.py` | Trace ID generation (contextvar-based) |
| `/Users/harecjimenez/codi-memory/modules/migrations.py` | Schema migrations (idempotency critical) |
| `/Users/harecjimenez/codi-memory/.env` | Credentials (OPENAI_API_KEY, QDRANT_URL, CODI_API_KEY) |
| `/Users/harecjimenez/codi-memory/migrations/` | SQL migration files (DDL review) |
| `/Users/harecjimenez/codi-memory/tests/conftest.py` | Test isolation fixtures |
