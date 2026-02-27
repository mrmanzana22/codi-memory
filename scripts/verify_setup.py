#!/usr/bin/env python3
"""
codi-memory setup verification script.
Run after completing SETUP_GUIDE.md to verify everything works.

Usage:
    python scripts/verify_setup.py
"""

import os
import sys

# Resolve paths relative to repo root (one level up from scripts/)
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"
WARN = "\033[93mWARN\033[0m"

results = []


def check(name, fn):
    """Run a check function and record result."""
    try:
        ok, msg = fn()
        status = PASS if ok else FAIL
        results.append((name, ok, msg))
        print(f"  [{status}] {name}: {msg}")
    except Exception as e:
        results.append((name, False, str(e)))
        print(f"  [{FAIL}] {name}: {e}")


# ── Check 1: Python version ──────────────────────────────────

def check_python_version():
    v = sys.version_info
    ok = v >= (3, 11)
    return ok, f"{v.major}.{v.minor}.{v.micro}" + ("" if ok else " (need 3.11+)")

# ── Check 2: .env file exists ────────────────────────────────

def check_env_file():
    env_path = os.path.join(REPO_ROOT, ".env")
    if not os.path.exists(env_path):
        return False, ".env not found — copy .env.example to .env"
    return True, ".env exists"

# ── Check 3: Required env vars ───────────────────────────────

def check_required_vars():
    from dotenv import load_dotenv
    load_dotenv(os.path.join(REPO_ROOT, ".env"))

    missing = []
    for var in ["OPENAI_API_KEY", "QDRANT_URL"]:
        val = os.getenv(var, "").strip()
        if not val or val.startswith("sk-your-"):
            missing.append(var)

    if missing:
        return False, f"missing or placeholder: {', '.join(missing)}"
    return True, "OPENAI_API_KEY and QDRANT_URL set"

# ── Check 4: Dependencies installed ──────────────────────────

def check_dependencies():
    missing = []
    for pkg in ["mcp", "mem0", "openai", "qdrant_client", "dotenv", "pydantic", "supabase"]:
        try:
            if pkg == "dotenv":
                __import__("dotenv")
            else:
                __import__(pkg)
        except ImportError:
            missing.append(pkg)

    if missing:
        return False, f"missing packages: {', '.join(missing)} — run: pip install -r requirements.txt"
    return True, "all required packages installed"

# ── Check 5: Qdrant connection ───────────────────────────────

def check_qdrant():
    from dotenv import load_dotenv
    load_dotenv(os.path.join(REPO_ROOT, ".env"))

    url = os.getenv("QDRANT_URL", "").strip()
    if not url:
        return False, "QDRANT_URL not set"

    from qdrant_client import QdrantClient
    api_key = os.getenv("QDRANT_API_KEY", "").strip() or None
    client = QdrantClient(url=url, api_key=api_key, timeout=5)

    # Try a simple operation
    collections = client.get_collections()
    names = [c.name for c in collections.collections]
    return True, f"connected — {len(names)} collections: {', '.join(names) or '(empty, will be created on first use)'}"

# ── Check 6: OpenAI API key works ────────────────────────────

def check_openai():
    from dotenv import load_dotenv
    load_dotenv(os.path.join(REPO_ROOT, ".env"))

    key = os.getenv("OPENAI_API_KEY", "").strip()
    if not key or key.startswith("sk-your-"):
        return False, "OPENAI_API_KEY not set or still placeholder"

    import openai
    client = openai.OpenAI(api_key=key)
    # Minimal API call: list models (cheap, no tokens consumed)
    resp = client.models.list()
    if resp.data:
        return True, "API key valid"
    return False, "API key returned no models"

# ── Check 7: Migrations apply ────────────────────────────────

def check_migrations():
    migrations_dir = os.path.join(REPO_ROOT, "migrations")
    if not os.path.isdir(migrations_dir):
        return False, "migrations/ directory not found"

    files = sorted(f for f in os.listdir(migrations_dir) if f.endswith(".sql"))
    if not files:
        return False, "no .sql files in migrations/"

    # Test that apply_migrations runs without error on a temp DB
    import tempfile
    import sqlite3
    from modules.migrations import apply_migrations

    with tempfile.NamedTemporaryFile(suffix=".db", delete=True) as tmp:
        tmp_path = tmp.name

    try:
        apply_migrations(tmp_path, migrations_dir=migrations_dir)
        # Verify tables were created
        conn = sqlite3.connect(tmp_path)
        tables = [r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
        conn.close()
        os.unlink(tmp_path)
        return True, f"{len(files)} migrations applied — tables: {', '.join(sorted(tables))}"
    except Exception as e:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        return False, f"migration error: {e}"

# ── Check 8: server.py imports cleanly ───────────────────────

def check_server_imports():
    """Verify that all module imports in server.py resolve without errors."""
    # We test each module import individually to give specific error messages
    modules_to_check = [
        "modules.config",
        "modules.migrations",
        "modules.triggers",
        "modules.memory_core",
        "modules.memory_smart",
        "modules.working_memory",
        "modules.interface",
        "modules.consciousness",
        "modules.flush",
    ]

    failed = []
    for mod in modules_to_check:
        try:
            __import__(mod)
        except Exception as e:
            failed.append(f"{mod}: {type(e).__name__}")

    if failed:
        return False, f"import errors: {'; '.join(failed)}"
    return True, f"all {len(modules_to_check)} core modules import OK"


# ── Run all checks ───────────────────────────────────────────

def main():
    print()
    print("codi-memory setup verification")
    print("=" * 45)
    print()

    check("Python version", check_python_version)
    check(".env file", check_env_file)
    check("Required env vars", check_required_vars)
    check("Dependencies", check_dependencies)
    check("Qdrant connection", check_qdrant)
    check("OpenAI API key", check_openai)
    check("Migrations", check_migrations)
    check("Server imports", check_server_imports)

    print()
    print("-" * 45)

    passed = sum(1 for _, ok, _ in results if ok)
    total = len(results)

    if passed == total:
        print(f"\n  All {total} checks passed. You're ready to go.\n")
    else:
        failed_names = [name for name, ok, _ in results if not ok]
        print(f"\n  {passed}/{total} passed. Fix these: {', '.join(failed_names)}\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
