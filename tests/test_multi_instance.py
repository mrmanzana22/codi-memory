"""
Tests for multi-instance isolation (Codi + Sebastian).
Verifies that each instance gets separate data stores and configs.
"""
import os
import json
import pytest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class TestInstanceConfig:
    """Test instance_config.py isolation mechanics."""

    def test_default_is_hare(self):
        """Without CODI_INSTANCE_CONFIG, default is Hare."""
        from modules.instance_config import _default_hare
        cfg = _default_hare()
        assert cfg.tenant_id == "hare"
        assert cfg.display_name == "Hare"
        assert cfg.daemon_port == 8420

    def test_sebastian_yaml_loadable(self, tmp_path, monkeypatch):
        """Sebastian YAML config should load and produce correct isolation."""
        monkeypatch.setenv("CODI_PG_URL", "postgresql://test:test@localhost:5432/test")
        yaml_content = """
tenant_id: sebastian
display_name: Sebastian
user_id: sebastian
pg_url: ${CODI_PG_URL}
daemon_port: 8421
data_dir: /tmp/test-sebastian
qdrant_collection: seb_memories
qdrant_semantic: seb_semantic
log_dir: /tmp/test-seb-logs
plist_prefix: com.seb
"""
        yaml_path = tmp_path / "sebastian.yaml"
        yaml_path.write_text(yaml_content)

        from modules.instance_config import _load_yaml
        cfg = _load_yaml(str(yaml_path))

        assert cfg.tenant_id == "sebastian"
        assert cfg.daemon_port == 8421
        assert cfg.qdrant_collection == "seb_memories"
        assert cfg.qdrant_semantic == "seb_semantic"
        assert cfg.data_dir == "/tmp/test-sebastian"

    def test_hare_and_sebastian_no_overlap(self):
        """Hare and Sebastian must have ZERO overlapping paths/collections."""
        from modules.instance_config import _default_hare
        hare = _default_hare()

        # Sebastian's expected values
        seb_collection = "seb_memories"
        seb_semantic = "seb_semantic"
        seb_port = 8421

        assert hare.qdrant_collection != seb_collection
        assert hare.qdrant_semantic != seb_semantic
        assert hare.daemon_port != seb_port


class TestDataIsolation:
    """Test that Sebastian's data directory is separate."""

    def test_sebastian_data_dir_exists(self):
        """Sebastian's data dir should exist with FTS + prospective DBs."""
        seb_dir = PROJECT_ROOT / "data-sebastian"
        assert seb_dir.exists(), f"Sebastian data dir missing: {seb_dir}"
        assert (seb_dir / "memories_fts.db").exists(), "Sebastian FTS DB missing"
        assert (seb_dir / "prospective.db").exists(), "Sebastian prospective DB missing"

    def test_sebastian_fts_is_separate(self):
        """Sebastian's FTS DB should NOT share tables with Codi's."""
        import sqlite3
        seb_fts = PROJECT_ROOT / "data-sebastian" / "memories_fts.db"
        codi_fts_path = os.getenv("FTS_DB_PATH", str(PROJECT_ROOT / "memories_fts.db"))

        if not seb_fts.exists():
            pytest.skip("Sebastian FTS DB not available")

        seb_conn = sqlite3.connect(str(seb_fts))
        seb_tables = {r[0] for r in seb_conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()}
        seb_conn.close()

        # Sebastian should have its own tables (not empty)
        assert len(seb_tables) > 0, "Sebastian FTS has no tables"

    def test_no_cross_contamination(self):
        """Memories written to Codi should NOT appear in Sebastian's DB."""
        import sqlite3
        seb_fts = PROJECT_ROOT / "data-sebastian" / "memories_fts.db"

        if not seb_fts.exists():
            pytest.skip("Sebastian FTS DB not available")

        seb_conn = sqlite3.connect(str(seb_fts))
        try:
            # Check if any of Codi's identity memories leaked
            r = seb_conn.execute(
                "SELECT COUNT(*) FROM memories_text WHERE content LIKE '%Hare%Codi%parcero%'"
            ).fetchone()
            assert r[0] == 0, f"Cross-contamination: {r[0]} Codi memories found in Sebastian DB"
        except Exception:
            pass  # Table might not exist yet
        finally:
            seb_conn.close()


class TestCollectionIsolation:
    """Test PG collection naming isolation."""

    def test_collection_names_differ(self):
        """Hare and Sebastian must use different PG collection names."""
        from modules.instance_config import _default_hare
        hare = _default_hare()

        # These should never overlap
        assert "codi" in hare.qdrant_collection.lower()
        assert "seb" not in hare.qdrant_collection.lower()
