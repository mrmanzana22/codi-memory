"""
PARITY HARNESS - Snapshot tests for critical outputs
=====================================================
Protects load-bearing contracts: if a refactor silently changes a critical
output, these tests catch it before it reaches prod.

5 snapshots:
  1. load_session_bridge() — core continuity contract
  2. despertar_codi() — section contract (SESSION BRIDGE block)
  3. format_sleep_report() — report string shape
  4. Tick order + status list — budget gating behavior
  5. Write-path smoke — sleep_loop idempotent report

Usage:
  # Run parity tests
  pytest tests/parity/test_parity_harness.py -v

  # Regenerate snapshots after intentional changes
  UPDATE_SNAPSHOTS=1 pytest tests/parity/test_parity_harness.py -v
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import json
import random
import re
import sqlite3
import pytest
from datetime import datetime, timedelta
from unittest.mock import MagicMock
from zoneinfo import ZoneInfo

TZ_COL = ZoneInfo("America/Bogota")

pytestmark = pytest.mark.parity

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SNAPSHOTS_DIR = os.path.join(os.path.dirname(__file__), "snapshots")
FTS_MIGRATIONS_DIR = os.path.join(PROJECT_ROOT, "migrations")
PROSP_MIGRATIONS_DIR = os.path.join(PROJECT_ROOT, "migrations_prospective")


# ============================================================
# SNAPSHOT HELPERS
# ============================================================

def _should_update() -> bool:
    return os.environ.get("UPDATE_SNAPSHOTS", "").strip() in ("1", "true", "yes")


def _snapshot_path(name: str, ext: str = "json") -> str:
    return os.path.join(SNAPSHOTS_DIR, f"{name}.{ext}")


def assert_snapshot(name: str, obj, *, ext: str = "json"):
    """Compare obj against stored snapshot. Update if UPDATE_SNAPSHOTS=1."""
    os.makedirs(SNAPSHOTS_DIR, exist_ok=True)
    path = _snapshot_path(name, ext)

    if ext == "json":
        current = json.dumps(obj, sort_keys=True, indent=2, ensure_ascii=False)
    else:
        current = str(obj)

    if _should_update():
        with open(path, "w", encoding="utf-8") as f:
            f.write(current)
        return  # Updated, no assertion needed

    if not os.path.exists(path):
        # First run: write snapshot and pass
        with open(path, "w", encoding="utf-8") as f:
            f.write(current)
        return

    with open(path, "r", encoding="utf-8") as f:
        expected = f.read()

    assert current == expected, (
        f"Parity violation for snapshot '{name}'.\n"
        f"Run UPDATE_SNAPSHOTS=1 pytest tests/parity/ -v to update.\n"
        f"--- EXPECTED ---\n{expected[:500]}\n"
        f"--- ACTUAL ---\n{current[:500]}"
    )


# ============================================================
# NORMALIZATION LAYER
# ============================================================

def _normalize_bridge(bridge_data: dict) -> dict:
    """Normalize bridge output for stable snapshotting."""
    if bridge_data is None:
        return None

    normalized = {
        "bridge_text": bridge_data.get("bridge_text", ""),
        "hours_since_last": round(bridge_data.get("hours_since_last", 0.0), 1),
        "prediction_errors": [],
        "has_sleep_report": bridge_data.get("sleep_report") is not None,
    }

    for pe in bridge_data.get("prediction_errors", []):
        normalized["prediction_errors"].append({
            "type": pe.get("type", ""),
            "severity": pe.get("severity", ""),
            "detail_prefix": pe.get("detail", "")[:60],
        })

    # Normalize variable parts of bridge_text
    text = normalized["bridge_text"]
    # Replace "Han pasado X.X horas" with stable token
    text = re.sub(r"Han pasado \d+\.\d+ horas", "Han pasado ELAPSED horas", text)
    normalized["bridge_text"] = text

    return normalized


def _normalize_despertar_sections(wake_text: str) -> dict:
    """Extract section headers and key contract fields from despertar_codi output."""
    sections = re.findall(r"^## (.+)$", wake_text, re.MULTILINE)

    # Normalize dynamic counts inside parentheses in section headers.
    # e.g. "CURIOSIDADES (19 pendientes, 7 alta prioridad)" ->
    #      "CURIOSIDADES (N pendientes, N alta prioridad)"
    def _replace_nums_in_parens(m):
        return re.sub(r"\d+", "N", m.group(0))

    normalized_sections = [
        re.sub(r"\([^)]*\d[^)]*\)", _replace_nums_in_parens, s)
        for s in sections
    ]

    result = {
        "sections_present": sorted(normalized_sections),
        "has_session_bridge": "SESSION BRIDGE" in wake_text,
        "has_prediction_errors": "PREDICTION ERRORS" in wake_text,
        "has_sleep_report": "SLEEP REPORT" in wake_text,
        "bridge_grounded": False,
        "intentions_present": [],
    }

    # Check grounding: summary text appears in bridge
    if "baseline E0.1" in wake_text or "SLOs" in wake_text:
        result["bridge_grounded"] = True

    # Extract pending intentions mentioned
    for match in re.finditer(r"Pendiente \[(\w+)\]: (.+?)$", wake_text, re.MULTILINE):
        result["intentions_present"].append({
            "priority": match.group(1),
            "action_prefix": match.group(2)[:50],
        })

    return result


def _normalize_tick_results(tick_results: list) -> list:
    """Extract stable contract fields from tick results (no timing)."""
    return [
        {
            "tick": r.get("tick", ""),
            "status": r.get("status", "unknown"),
            "ok": r.get("ok", False),
        }
        for r in tick_results
    ]


# ============================================================
# FIXTURES
# ============================================================

FIXED_NOW = datetime(2026, 2, 16, 14, 0, 0, tzinfo=TZ_COL)
CHECKPOINT_TIME = FIXED_NOW - timedelta(hours=1)


@pytest.fixture
def parity_env(tmp_path, monkeypatch):
    """Deterministic world for parity testing.

    Freezes clock, seeds random, mocks all external deps.
    Returns db_path for further use.
    """
    # --- Freeze clock ---
    # IMPORTANT: Only patch module-level references in the modules under test.
    # Do NOT patch modules.config.now_col/now_iso — other tests import from
    # config and a leaked frozen clock causes flaky failures.
    _frozen_now = lambda: FIXED_NOW
    _frozen_iso = lambda: FIXED_NOW.isoformat()
    monkeypatch.setattr("modules.session_bridge.now_col", _frozen_now)
    monkeypatch.setattr("modules.session_bridge.now_iso", _frozen_iso)
    monkeypatch.setattr("modules.sleep_loop.now_iso", _frozen_iso)

    # --- Seed random ---
    random.seed(0)
    monkeypatch.setattr("random.gauss", lambda *a, **k: 0.0)

    # --- Isolated DBs ---
    db_path = str(tmp_path / "parity_fts.db")
    prosp_path = str(tmp_path / "parity_prosp.db")

    monkeypatch.setenv("FTS_DB_PATH", db_path)
    monkeypatch.setenv("PROSPECTIVE_DB_PATH", prosp_path)
    monkeypatch.setattr("modules.config.FTS_DB_PATH", db_path)
    monkeypatch.setattr("modules.config.PROSPECTIVE_DB_PATH", prosp_path)
    monkeypatch.setattr("modules.session_bridge.FTS_DB_PATH", db_path)
    monkeypatch.setattr("modules.sleep_loop.FTS_DB_PATH", db_path)
    monkeypatch.setattr("modules.sleep_loop.DATA_DIR", str(tmp_path))
    monkeypatch.setattr("modules.sleep_loop.LOCK_FILE", str(tmp_path / "sleep_loop.lock"))

    from modules.migrations import apply_migrations
    apply_migrations(db_path, migrations_dir=FTS_MIGRATIONS_DIR)
    apply_migrations(prosp_path, migrations_dir=PROSP_MIGRATIONS_DIR)

    # Ensure WM tables exist
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS working_memory (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            content TEXT NOT NULL,
            topic TEXT DEFAULT 'general',
            relevance REAL DEFAULT 0.5,
            added_at TEXT NOT NULL,
            occurred_at TEXT,
            source TEXT DEFAULT 'interaction',
            related_memory_id TEXT,
            chain_id TEXT,
            active INTEGER DEFAULT 1,
            last_accessed_at TEXT,
            access_count INTEGER DEFAULT 0
        );
        CREATE TABLE IF NOT EXISTS narrative_traces (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            trace_name TEXT NOT NULL UNIQUE,
            chain_ids TEXT NOT NULL,
            theme TEXT,
            created_at TEXT NOT NULL,
            last_updated TEXT NOT NULL,
            active INTEGER DEFAULT 1
        );
    """)
    conn.commit()
    conn.close()

    monkeypatch.setattr("modules.db_pool.FTS_DB_PATH", db_path)
    monkeypatch.setattr("modules.working_memory._TABLES_INITIALIZED", False)

    # --- Mock external deps ---
    monkeypatch.setattr("modules.wiring._attention_schema", {
        "current_focus": "parity_test",
        "focus_strength": 0.80,
    })
    monkeypatch.setattr("modules.session_bridge._emotional_state", {
        "current": {
            "pleasure": 0.5, "arousal": -0.2, "dominance": 0.6,
            "trigger": "parity_fixture",
        },
        "mood": {"pleasure": 0.2, "arousal": 0.1, "dominance": 0.3, "last_updated": None},
        "history": [],
        "decay_rate": 0.1,
        "mood_shift_rate": 0.05,
    })
    monkeypatch.setattr("modules.session_bridge.get_trace_id", lambda: "TRACE_ID")
    monkeypatch.setattr("modules.session_bridge._current_session", "SESSION_ID")
    monkeypatch.setattr(
        "modules.prospective.get_pending_intentions",
        lambda limit=10: [
            {"id": "int-001", "action": "Deploy trading bot v2",
             "priority": "high", "activation": 0.9, "expiry": None},
            {"id": "int-002", "action": "Review session bridge health",
             "priority": "medium", "activation": 0.6, "expiry": None},
        ]
    )

    # No external reminders file
    monkeypatch.setattr(
        "modules.session_bridge.REMINDERS_FILE",
        str(tmp_path / "nonexistent_reminders.json")
    )

    return db_path


@pytest.fixture
def parity_checkpoint(parity_env):
    """Insert a known checkpoint + sleep report. Returns (db_path, checkpoint_id)."""
    from modules.session_bridge import checkpoint_session_close

    result = checkpoint_session_close(
        session_summary="Completamos baseline E0.1 y definimos SLOs v0",
        decisions="Usar async ACK para write-path",
        goal_stack=[{"goal": "Production-grade Codi", "status": "in_progress", "next_step": "E1.2 parity harness"}],
        source="flush",
    )
    assert result["ok"] is True
    cp_id = result["checkpoint_id"]

    # Write a known sleep report
    conn = sqlite3.connect(parity_env)
    conn.execute(
        "UPDATE session_checkpoints SET sleep_report = ? WHERE id = ?",
        (
            "SLEEP REPORT (launchd, 3200ms)\n"
            "- prospective: OK (150ms) intention maintenance applied\n"
            "- health: OK (800ms) fts: queue empty; health: OK (127 mems)\n"
            "- consolidation: OK (1800ms) no output\n"
            "- homeostasis: OK (400ms) salience: decayed 5 mems; emotional: PAD shifted",
            cp_id,
        )
    )
    conn.commit()
    conn.close()

    return parity_env, cp_id


@pytest.fixture
def clean_pad():
    """Save/restore _emotional_state to prevent PAD leaks."""
    from modules.config import _emotional_state
    import copy
    saved = copy.deepcopy(_emotional_state)
    yield _emotional_state
    _emotional_state.clear()
    _emotional_state.update(saved)


# ============================================================
# SNAPSHOT 1: load_session_bridge()
# ============================================================

class TestBridgeSnapshot:
    """Snapshot 1: load_session_bridge() output contract."""

    def test_bridge_output_stable(self, parity_checkpoint):
        """Bridge text, PE list, and structure match snapshot."""
        db_path, cp_id = parity_checkpoint

        from modules.session_bridge import load_session_bridge
        bridge = load_session_bridge()

        assert bridge is not None, "Bridge returned None (stale or missing)"
        normalized = _normalize_bridge(bridge)
        assert_snapshot("bridge_output", normalized)

    def test_bridge_with_expired_intention(self, parity_env, monkeypatch):
        """Bridge generates PE for expired intention."""
        from modules.session_bridge import checkpoint_session_close, load_session_bridge

        # Create intention that's already expired
        expired_time = (FIXED_NOW - timedelta(hours=2)).isoformat()
        monkeypatch.setattr(
            "modules.prospective.get_pending_intentions",
            lambda limit=10: [
                {"id": "exp-001", "action": "Submit report to client",
                 "priority": "high", "activation": 0.8, "expiry": expired_time},
            ]
        )

        result = checkpoint_session_close(
            session_summary="Session with expiring intention",
            source="flush",
        )
        assert result["ok"] is True

        # Reset intentions mock (bridge reads from checkpoint)
        monkeypatch.setattr(
            "modules.prospective.get_pending_intentions",
            lambda limit=10: []
        )

        bridge = load_session_bridge()
        assert bridge is not None
        normalized = _normalize_bridge(bridge)
        assert_snapshot("bridge_expired_intention", normalized)


# ============================================================
# SNAPSHOT 2: despertar_codi() section contract
# ============================================================

class TestDespertarSections:
    """Snapshot 2: despertar_codi section headers and contract fields."""

    def _setup_despertar_mocks(self, monkeypatch, db_path):
        """Minimal mocks for despertar_codi."""
        monkeypatch.setattr(
            "modules.lifecycle._verificar_salud_memoria_interna",
            lambda: {"ok": True, "total_memories": 100, "message": "OK"}
        )
        monkeypatch.setattr("modules.triggers._load_triggers", lambda: [])
        monkeypatch.setattr("modules.maintenance._verificar_tareas_vencidas", lambda: [])

        # Mock pg_store (replaces old qdrant/memory mocks after pg migration)
        mock_pg = MagicMock()
        mock_pg.scroll.return_value = ([], None)
        mock_pg.count.return_value = MagicMock(points_count=100)
        mock_pg.search.return_value = []
        monkeypatch.setattr("modules.lifecycle.pg", mock_pg)

        # Mock goals (deterministic, no real DB)
        monkeypatch.setattr("modules.goals.get_context_goals", lambda limit=5: {
            "goals": [
                {"title": "Consciencia", "level": "project",
                 "goal_what": "Sistema cognitivo con 5 loops",
                 "goal_why": "Core identity project",
                 "goal_next_step": "E1.2 parity harness",
                 "goal_last_state": "Baseline E0.1 complete"},
            ]
        })

        # Mock worker health (deterministic)
        monkeypatch.setattr("modules.assessment.get_worker_health", lambda: {
            "status": "ok", "age_minutes": 1.0, "queue_backlog": {}
        })

        monkeypatch.setattr("modules.flush.load_session_state", lambda: None)

        from modules.events import event_bus
        monkeypatch.setattr(event_bus, "emit", lambda *a, **kw: None)

    def test_section_contract(self, parity_checkpoint, monkeypatch, clean_pad):
        """Sections present in wake-up text match snapshot."""
        db_path, cp_id = parity_checkpoint
        self._setup_despertar_mocks(monkeypatch, db_path)

        from modules.consciousness import despertar_codi
        wake_text = despertar_codi()

        normalized = _normalize_despertar_sections(wake_text)
        assert_snapshot("despertar_sections", normalized)

    def test_no_checkpoint_sections(self, parity_env, monkeypatch, clean_pad):
        """Without checkpoint, bridge sections are absent."""
        self._setup_despertar_mocks(monkeypatch, parity_env)

        from modules.consciousness import despertar_codi
        wake_text = despertar_codi()

        normalized = _normalize_despertar_sections(wake_text)
        assert_snapshot("despertar_no_checkpoint", normalized)


# ============================================================
# SNAPSHOT 3: format_sleep_report()
# ============================================================

class TestSleepReportSnapshot:
    """Snapshot 3: format_sleep_report() string shape."""

    def test_report_format(self):
        """Sleep report format matches snapshot."""
        from modules.sleep_loop import format_sleep_report

        tick_results = [
            {"tick": "prospective", "ok": True, "elapsed_ms": 150, "detail": "intention maintenance applied"},
            {"tick": "health", "ok": True, "elapsed_ms": 800, "detail": "fts: queue empty; health: OK (127 mems)"},
            {"tick": "consolidation", "ok": True, "elapsed_ms": 1800, "detail": "no output"},
            {"tick": "homeostasis", "ok": True, "elapsed_ms": 400, "detail": "salience: decayed 5 mems; emotional: PAD shifted"},
        ]
        report = format_sleep_report(tick_results, total_ms=3200, reason="launchd")
        assert_snapshot("sleep_report_format", report, ext="txt")

    def test_report_with_skipped_tick(self):
        """Report with a skipped tick matches snapshot."""
        from modules.sleep_loop import format_sleep_report

        tick_results = [
            {"tick": "prospective", "ok": True, "elapsed_ms": 150, "detail": "intention maintenance applied"},
            {"tick": "health", "ok": True, "elapsed_ms": 800, "detail": "fts: queue empty"},
            {"tick": "consolidation", "ok": True, "elapsed_ms": 6500, "detail": "light consolidation"},
            {"tick": "homeostasis", "ok": False, "elapsed_ms": 0, "detail": "skipped (need 200ms, only 50ms left)"},
        ]
        report = format_sleep_report(tick_results, total_ms=7950, reason="launchd")
        assert_snapshot("sleep_report_skipped", report, ext="txt")


# ============================================================
# SNAPSHOT 4: Tick order + status list
# ============================================================

class TestTickStatusSnapshot:
    """Snapshot 4: tick execution order and status under different budget scenarios."""

    @staticmethod
    def _mock_all_ticks(monkeypatch, elapsed_ms=50):
        """Mock all 13 ticks as instant, deterministic functions."""
        def _make_tick(name):
            return lambda budget_ms: {"tick": name, "ok": True, "elapsed_ms": elapsed_ms, "detail": "ok"}

        for tick_name in ("prospective", "health", "health_snapshot", "self_model",
                          "reconsolidation", "consolidation", "homeostasis",
                          "curiosity", "curiosity_resolve", "backup",
                          "causal_discovery", "sharpe_insights", "proactive_contact"):
            monkeypatch.setattr(f"modules.sleep_loop._tick_{tick_name}", _make_tick(tick_name))
        monkeypatch.setattr("modules.sleep_loop._log_tick_metric", lambda *a, **k: None)
        # S4-05: Mock world model prioritization to preserve original tick order
        # (parity tests validate execution behavior, not ordering strategy)
        monkeypatch.setattr(
            "modules.sleep_loop.SleepWorldModel.prioritize_ticks",
            lambda self, eligible, conn=None: eligible
        )

    def test_normal_budget(self, parity_env, monkeypatch):
        """All ticks run OK with generous budget."""
        # Mock all 8 tick functions to be fast and deterministic
        self._mock_all_ticks(monkeypatch)

        from modules.sleep_loop import run_sleep_loop
        result = run_sleep_loop(reason="parity_test", budget_ms=8000)

        normalized = _normalize_tick_results(result["tick_results"])
        assert_snapshot("tick_status_normal", normalized)

    def test_zero_budget_skips_all(self, parity_env, monkeypatch):
        """Zero budget skips every tick."""
        monkeypatch.setattr("modules.sleep_loop._log_tick_metric", lambda *a, **k: None)
        # S4-05: Mock world model to preserve original tick order
        monkeypatch.setattr(
            "modules.sleep_loop.SleepWorldModel.prioritize_ticks",
            lambda self, eligible, conn=None: eligible
        )

        from modules.sleep_loop import run_sleep_loop
        result = run_sleep_loop(reason="parity_test", budget_ms=0)

        normalized = _normalize_tick_results(result["tick_results"])
        assert_snapshot("tick_status_zero_budget", normalized)

    def test_tight_budget_skips_heavy(self, parity_env, monkeypatch):
        """Tight budget: consolidation (needs 1500ms) is skipped, rest runs.

        All ticks are mocked as instant lambdas. Budget gating uses
        time.monotonic() so real time barely passes — the only tick
        skipped is consolidation (TICK_MIN_MS=1500 > 500ms budget).
        """
        self._mock_all_ticks(monkeypatch)

        from modules.sleep_loop import run_sleep_loop
        result = run_sleep_loop(reason="parity_test", budget_ms=500)

        normalized = _normalize_tick_results(result["tick_results"])
        assert_snapshot("tick_status_tight_budget", normalized)


# ============================================================
# SNAPSHOT 5: Write-path smoke (idempotent report)
# ============================================================

class TestWritePathSmoke:
    """Snapshot 5: sleep_loop produces idempotent report."""

    def test_report_written_once(self, parity_env, monkeypatch):
        """run_sleep_loop produces a report; second call doesn't clobber."""
        # Mock ALL 13 ticks to be fast and deterministic
        def _make_tick(name):
            return lambda budget_ms: {"tick": name, "ok": True, "elapsed_ms": 10, "detail": "ok"}

        for tick_name in ("prospective", "health", "health_snapshot", "self_model",
                          "reconsolidation", "consolidation", "homeostasis",
                          "curiosity", "curiosity_resolve", "backup",
                          "causal_discovery", "sharpe_insights", "proactive_contact"):
            monkeypatch.setattr(f"modules.sleep_loop._tick_{tick_name}", _make_tick(tick_name))
        monkeypatch.setattr("modules.sleep_loop._log_tick_metric", lambda *a, **k: None)
        # S4-05: Mock world model to preserve original tick order
        monkeypatch.setattr(
            "modules.sleep_loop.SleepWorldModel.prioritize_ticks",
            lambda self, eligible, conn=None: eligible
        )

        from modules.sleep_loop import run_sleep_loop, _write_sleep_report

        result = run_sleep_loop(reason="parity_test", budget_ms=8000)
        report_text = result["report"]

        assert report_text is not None
        assert len(report_text) > 0
        assert len(report_text) <= 600, f"Report exceeds 600 char cap: {len(report_text)}"

        # Snapshot the report structure (not timing)
        # Normalize dynamic total_ms: "SLEEP REPORT (parity_test, 484ms)"
        # -> "SLEEP REPORT (parity_test, Nms)"
        # Normalize the full first line before truncating so the length is stable.
        report_prefix = re.sub(r"(\w+),\s*\d+ms", r"\1, Nms", report_text[:50])[:30]
        snapshot_data = {
            "ok": result["ok"],
            "ticks_total": len(result["tick_results"]),
            "ticks_ok": sum(1 for t in result["tick_results"] if t.get("ok")),
            "report_starts_with": report_prefix,
            "report_line_count": len(report_text.strip().split("\n")),
            "report_length_within_cap": len(report_text) <= 600,
        }
        assert_snapshot("write_path_smoke", snapshot_data)

    def test_idempotent_write(self, parity_env):
        """_write_sleep_report doesn't clobber existing report."""
        from modules.sleep_loop import _write_sleep_report

        # Insert checkpoint with no report
        conn = sqlite3.connect(parity_env)
        conn.execute(
            "INSERT INTO session_checkpoints (source, created_at, session_summary) "
            "VALUES (?, ?, ?)",
            ("flush", FIXED_NOW.isoformat(), "parity test")
        )
        conn.commit()
        cp_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
        conn.close()

        # First write succeeds
        assert _write_sleep_report(cp_id, "REPORT v1") is True

        # Second write is idempotent (no clobber)
        assert _write_sleep_report(cp_id, "REPORT v2") is False

        # Verify original report preserved
        conn = sqlite3.connect(parity_env)
        row = conn.execute(
            "SELECT sleep_report FROM session_checkpoints WHERE id = ?", (cp_id,)
        ).fetchone()
        conn.close()
        assert row[0] == "REPORT v1"


# ============================================================
# FAKE QDRANT (reused from battery)
# ============================================================

class FakeQdrantPoint:
    def __init__(self, point_id, payload):
        self.id = point_id
        self.payload = payload


class FakeQdrant:
    def __init__(self, points):
        self.points = points
        self.payloads_set = []

    def scroll(self, collection_name, scroll_filter=None, limit=100,
               with_payload=True, with_vectors=False, offset=None):
        return (self.points, None)

    def set_payload(self, collection_name, payload, points):
        self.payloads_set.append({"payload": payload, "points": points})
