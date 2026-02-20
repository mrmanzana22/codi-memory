"""
Tests for PR-2: Reconsolidation Suggest Mode
=============================================
Covers:
  1. queue_correction_suggestion inserts rows correctly
  2. get_pending_corrections returns formatted output
  3. expire_stale_corrections marks old entries
  4. Wiring handler queues suggestion on high PE

Run: ./venv/bin/pytest tests/test_pending_corrections.py -v
"""

import os
import sys
import json
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ============================================================
# FIXTURES
# ============================================================

@pytest.fixture
def correction_db(tmp_path, monkeypatch):
    """Isolated DB with migrations applied for pending_corrections tests."""
    db_path = str(tmp_path / "test_corrections.db")
    migrations_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "migrations",
    )
    from modules.migrations import apply_migrations
    apply_migrations(db_path, migrations_dir)

    # Patch consolidation_common to use our test DB
    monkeypatch.setattr(
        "modules.consolidation_common.FTS_DB_PATH", db_path, raising=False
    )
    # Close any cached connections so they reconnect to new path
    try:
        from modules.consolidation_common import _consolidation_conn
    except Exception:
        pass

    return db_path


# ============================================================
# 1. QUEUE CORRECTION SUGGESTION
# ============================================================

class TestQueueCorrectionSuggestion:

    def test_queue_inserts_row(self, correction_db):
        """queue_correction_suggestion should insert a pending row."""
        from modules.reconsolidation import queue_correction_suggestion
        import sqlite3

        row_id = queue_correction_suggestion(
            old_memory_id="abc-123",
            old_text="Qdrant uses port 6333",
            new_text="Qdrant uses port 6334 after migration",
            prediction_error=0.55,
            shared_entities=["qdrant", "port"],
            channels={"keywords": 0.5, "negation": 0.3},
        )

        assert row_id > 0

        conn = sqlite3.connect(correction_db)
        row = conn.execute(
            "SELECT * FROM pending_corrections WHERE id = ?", (row_id,)
        ).fetchone()
        conn.close()

        assert row is not None

    def test_queue_sets_pending_status(self, correction_db):
        """Queued suggestions start with status='pending'."""
        from modules.reconsolidation import queue_correction_suggestion
        import sqlite3

        row_id = queue_correction_suggestion(
            old_memory_id="def-456",
            old_text="old",
            new_text="new",
            prediction_error=0.60,
        )

        conn = sqlite3.connect(correction_db)
        status = conn.execute(
            "SELECT status FROM pending_corrections WHERE id = ?", (row_id,)
        ).fetchone()[0]
        conn.close()

        assert status == "pending"

    def test_queue_sets_expiry_24h(self, correction_db):
        """Default expiry should be ~24h from now."""
        from modules.reconsolidation import queue_correction_suggestion
        import sqlite3

        row_id = queue_correction_suggestion(
            old_memory_id="ghi-789",
            old_text="old",
            new_text="new",
            prediction_error=0.45,
        )

        conn = sqlite3.connect(correction_db)
        expires = conn.execute(
            "SELECT expires_at FROM pending_corrections WHERE id = ?", (row_id,)
        ).fetchone()[0]
        conn.close()

        expires_dt = datetime.fromisoformat(expires)
        expected = datetime.now() + timedelta(hours=24)
        # Should be within 1 minute of expected
        delta = abs((expires_dt - expected).total_seconds())
        assert delta < 60

    def test_queue_truncates_long_text(self, correction_db):
        """Text fields should be truncated to 500 chars."""
        from modules.reconsolidation import queue_correction_suggestion
        import sqlite3

        long_text = "x" * 1000
        row_id = queue_correction_suggestion(
            old_memory_id="trunc-test",
            old_text=long_text,
            new_text=long_text,
            prediction_error=0.50,
        )

        conn = sqlite3.connect(correction_db)
        old_txt, new_txt = conn.execute(
            "SELECT old_text, new_text FROM pending_corrections WHERE id = ?",
            (row_id,)
        ).fetchone()
        conn.close()

        assert len(old_txt) == 500
        assert len(new_txt) == 500

    def test_queue_stores_entities_as_json(self, correction_db):
        """shared_entities should be stored as JSON array."""
        from modules.reconsolidation import queue_correction_suggestion
        import sqlite3

        row_id = queue_correction_suggestion(
            old_memory_id="json-test",
            old_text="old",
            new_text="new",
            prediction_error=0.50,
            shared_entities=["qdrant", "docker", "supabase"],
        )

        conn = sqlite3.connect(correction_db)
        entities_json = conn.execute(
            "SELECT shared_entities FROM pending_corrections WHERE id = ?",
            (row_id,)
        ).fetchone()[0]
        conn.close()

        parsed = json.loads(entities_json)
        assert parsed == ["qdrant", "docker", "supabase"]


# ============================================================
# 2. GET PENDING CORRECTIONS
# ============================================================

class TestGetPendingCorrections:

    def test_empty_returns_no_pending(self, correction_db):
        """Empty table should return 'No pending corrections'."""
        from modules.reconsolidation import get_pending_corrections

        result = get_pending_corrections()
        assert "No pending" in result

    def test_returns_queued_items(self, correction_db):
        """Should return formatted list of pending items."""
        from modules.reconsolidation import (
            queue_correction_suggestion,
            get_pending_corrections,
        )

        queue_correction_suggestion(
            old_memory_id="show-test-1",
            old_text="The API uses REST",
            new_text="The API migrated to GraphQL",
            prediction_error=0.65,
            shared_entities=["api"],
        )

        result = get_pending_corrections()
        assert "Pending Corrections" in result
        assert "PE=0.65" in result
        assert "show-tes" in result  # truncated ID
        assert "REST" in result


# ============================================================
# 3. EXPIRE STALE CORRECTIONS
# ============================================================

class TestExpireStaleCorrections:

    def test_expire_marks_old_entries(self, correction_db):
        """Entries past expires_at should be marked 'expired'."""
        from modules.reconsolidation import (
            queue_correction_suggestion,
            expire_stale_corrections,
        )
        import sqlite3

        # Insert with window=0 so it expires immediately
        row_id = queue_correction_suggestion(
            old_memory_id="expire-test",
            old_text="old",
            new_text="new",
            prediction_error=0.50,
            window_hours=0.0,  # expires immediately
        )

        count = expire_stale_corrections()
        assert count >= 1

        conn = sqlite3.connect(correction_db)
        status = conn.execute(
            "SELECT status FROM pending_corrections WHERE id = ?", (row_id,)
        ).fetchone()[0]
        conn.close()

        assert status == "expired"

    def test_expire_does_not_touch_fresh(self, correction_db):
        """Fresh entries (not expired) should remain pending."""
        from modules.reconsolidation import (
            queue_correction_suggestion,
            expire_stale_corrections,
        )
        import sqlite3

        row_id = queue_correction_suggestion(
            old_memory_id="fresh-test",
            old_text="old",
            new_text="new",
            prediction_error=0.50,
            window_hours=48.0,  # far future
        )

        expire_stale_corrections()

        conn = sqlite3.connect(correction_db)
        status = conn.execute(
            "SELECT status FROM pending_corrections WHERE id = ?", (row_id,)
        ).fetchone()[0]
        conn.close()

        assert status == "pending"


# ============================================================
# 4. WIRING HANDLER INTEGRATION
# ============================================================

class TestWiringHandlerSuggestMode:

    def test_handler_queues_on_high_pe(self, correction_db):
        """_on_contradiction_detected should call queue_correction_suggestion when PE >= ALERT."""
        from modules.wiring import _on_contradiction_detected

        with patch("modules.working_memory.push_to_working_memory"), \
             patch("modules.wiring._update_attention_schema"), \
             patch("modules.consolidation.mark_as_labile"), \
             patch("modules.consolidation.queue_correction_suggestion") as mock_queue:

            _on_contradiction_detected("contradiction_detected", {
                "pe": 0.55,
                "conflicting_memory_id": "conflict-001",
                "conflicting_text": "Old fact about Qdrant",
                "new_content": "Updated fact about Qdrant",
                "shared_entities": ["qdrant"],
                "new_memory_id": "new-001",
                "channels": {"keywords": 0.5},
            })

            mock_queue.assert_called_once()
            call_kwargs = mock_queue.call_args[1]
            assert call_kwargs["old_memory_id"] == "conflict-001"
            assert call_kwargs["prediction_error"] == 0.55

    def test_handler_does_not_queue_on_low_pe(self, correction_db):
        """Low PE should NOT trigger queue_correction_suggestion."""
        from modules.wiring import _on_contradiction_detected

        with patch("modules.working_memory.push_to_working_memory"), \
             patch("modules.wiring._update_attention_schema"), \
             patch("modules.consolidation.queue_correction_suggestion") as mock_queue:

            _on_contradiction_detected("contradiction_detected", {
                "pe": 0.25,  # Below ALERT threshold
                "conflicting_memory_id": "conflict-002",
                "conflicting_text": "Some fact",
                "new_content": "Some other fact",
                "shared_entities": ["something"],
            })

            mock_queue.assert_not_called()
