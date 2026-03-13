"""
test_tenant_isolation.py - P2 Instance Isolation tests
=======================================================
15 test cases covering:
  - Config loading & validation
  - Data isolation (PG, SQLite, Qdrant paths)
  - Backward compatibility (no env var = Hare defaults)
  - Path isolation
"""

import os
import tempfile
import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def reset_instance_singleton():
    """Reset instance config singleton between tests."""
    from modules.instance_config import reset_instance
    reset_instance()
    yield
    reset_instance()


def _write_yaml(tmp_dir: str, content: str) -> str:
    """Write a YAML file and return its path."""
    path = os.path.join(tmp_dir, "test_instance.yaml")
    with open(path, "w") as f:
        f.write(content)
    return path


# ===========================================================================
# CONFIG TESTS (5-8)
# ===========================================================================

class TestConfigLoading:
    """Test 5-8: Config loading and validation."""

    def test_5_no_env_var_defaults_to_hare(self):
        """No CODI_INSTANCE_CONFIG → exact current behavior."""
        os.environ.pop("CODI_INSTANCE_CONFIG", None)
        from modules.instance_config import get_instance, reset_instance
        reset_instance()

        inst = get_instance()
        assert inst.tenant_id == "hare"
        assert inst.user_id == "hare"
        assert inst.qdrant_collection == "codi_memories"
        assert inst.qdrant_semantic == "codi_semantic"
        assert inst.daemon_port == 8420
        assert inst.plist_prefix == "com.codi"

    def test_6_valid_yaml_loads(self):
        """Valid YAML loads correctly with env var interpolation."""
        from modules.instance_config import _load_yaml

        with tempfile.TemporaryDirectory() as tmp:
            os.environ["_TEST_PG_URL"] = "postgresql://test:pass@localhost/test_db"
            try:
                path = _write_yaml(tmp, """
tenant_id: testuser
display_name: Test User
user_id: testuser
pg_url: ${_TEST_PG_URL}
daemon_port: 8422
data_dir: /tmp/test-codi
qdrant_collection: test_memories
qdrant_semantic: test_semantic
log_dir: /tmp/test-codi/logs
plist_prefix: com.test
""")
                inst = _load_yaml(path)
                assert inst.tenant_id == "testuser"
                assert inst.display_name == "Test User"
                assert inst.pg_url == "postgresql://test:pass@localhost/test_db"
                assert inst.daemon_port == 8422
                assert inst.qdrant_collection == "test_memories"
            finally:
                os.environ.pop("_TEST_PG_URL", None)

    def test_7_missing_pg_url_crashes(self):
        """Invalid YAML (missing pg_url) → crash with clear error."""
        from modules.instance_config import _load_yaml

        with tempfile.TemporaryDirectory() as tmp:
            path = _write_yaml(tmp, """
tenant_id: bad
data_dir: /tmp/bad
qdrant_collection: bad_mem
qdrant_semantic: bad_sem
""")
            with pytest.raises(ValueError, match="pg_url"):
                _load_yaml(path)

    def test_8_invalid_port_crashes(self):
        """Invalid YAML (bad port) → crash with clear error."""
        from modules.instance_config import _load_yaml

        with tempfile.TemporaryDirectory() as tmp:
            os.environ["_TEST_PG_URL2"] = "postgresql://x:y@localhost/db"
            try:
                path = _write_yaml(tmp, """
tenant_id: bad
pg_url: ${_TEST_PG_URL2}
daemon_port: 80
data_dir: /tmp/bad
qdrant_collection: bad_mem
qdrant_semantic: bad_sem
""")
                with pytest.raises(ValueError, match="daemon_port"):
                    _load_yaml(path)
            finally:
                os.environ.pop("_TEST_PG_URL2", None)

    def test_invalid_tenant_id_crashes(self):
        """tenant_id with uppercase or special chars → crash."""
        from modules.instance_config import _validate_config

        with pytest.raises(ValueError, match="tenant_id"):
            _validate_config({
                "tenant_id": "Bad-User",
                "pg_url": "postgresql://x:y@localhost/db",
                "data_dir": "/tmp/x",
                "qdrant_collection": "mem",
                "qdrant_semantic": "sem",
            })

    def test_relative_data_dir_crashes(self):
        """Relative data_dir → crash."""
        from modules.instance_config import _validate_config

        with pytest.raises(ValueError, match="absolute path"):
            _validate_config({
                "tenant_id": "test",
                "pg_url": "postgresql://x:y@localhost/db",
                "data_dir": "relative/path",
                "qdrant_collection": "mem",
                "qdrant_semantic": "sem",
            })

    def test_unset_env_var_crashes(self):
        """YAML with ${MISSING_VAR} → crash with clear error."""
        from modules.instance_config import _load_yaml

        os.environ.pop("TOTALLY_MISSING_VAR", None)
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_yaml(tmp, """
tenant_id: test
pg_url: ${TOTALLY_MISSING_VAR}
data_dir: /tmp/x
qdrant_collection: mem
qdrant_semantic: sem
""")
            with pytest.raises(ValueError, match="TOTALLY_MISSING_VAR"):
                _load_yaml(path)

    def test_frozen_dataclass(self):
        """InstanceConfig is immutable after creation."""
        from modules.instance_config import get_instance, reset_instance
        os.environ.pop("CODI_INSTANCE_CONFIG", None)
        reset_instance()

        inst = get_instance()
        with pytest.raises(AttributeError):
            inst.tenant_id = "hacked"


# ===========================================================================
# DATA ISOLATION TESTS (1-4, 12-15)
# ===========================================================================

class TestDataIsolation:
    """Test that different instances get different paths/collections."""

    def test_hare_collection_names(self):
        """Hare gets codi_memories / codi_semantic."""
        os.environ.pop("CODI_INSTANCE_CONFIG", None)
        from modules.instance_config import reset_instance
        reset_instance()

        # Re-import to get fresh values
        from modules.instance_config import get_instance
        inst = get_instance()
        assert inst.qdrant_collection == "codi_memories"
        assert inst.qdrant_semantic == "codi_semantic"

    def test_sebastian_collection_names(self):
        """Sebastian gets seb_memories / seb_semantic."""
        from modules.instance_config import _load_yaml

        with tempfile.TemporaryDirectory() as tmp:
            os.environ["CODI_PG_URL_SEB"] = "postgresql://test:pass@localhost/seb_db"
            try:
                path = _write_yaml(tmp, """
tenant_id: sebastian
display_name: Sebastian
user_id: sebastian
pg_url: ${CODI_PG_URL_SEB}
daemon_port: 8421
data_dir: /tmp/seb-data
qdrant_collection: seb_memories
qdrant_semantic: seb_semantic
log_dir: /tmp/seb-logs
plist_prefix: com.seb
""")
                inst = _load_yaml(path)
                assert inst.qdrant_collection == "seb_memories"
                assert inst.qdrant_semantic == "seb_semantic"
                assert inst.user_id == "sebastian"
                assert inst.daemon_port == 8421
            finally:
                os.environ.pop("CODI_PG_URL_SEB", None)

    def test_separate_data_dirs(self):
        """Different instances have different data directories."""
        from modules.instance_config import _default_hare, _load_yaml

        hare = _default_hare()

        with tempfile.TemporaryDirectory() as tmp:
            os.environ["_TEST_PG"] = "postgresql://x:y@localhost/db"
            try:
                path = _write_yaml(tmp, """
tenant_id: sebastian
pg_url: ${_TEST_PG}
daemon_port: 8421
data_dir: /Users/codi-air/codi-memory/data-sebastian
qdrant_collection: seb_memories
qdrant_semantic: seb_semantic
""")
                seb = _load_yaml(path)
                assert hare.data_dir != seb.data_dir
                assert "data-sebastian" in seb.data_dir
            finally:
                os.environ.pop("_TEST_PG", None)

    def test_separate_pg_urls(self):
        """Different instances have different PG connection strings."""
        from modules.instance_config import _default_hare, _load_yaml

        hare = _default_hare()

        with tempfile.TemporaryDirectory() as tmp:
            os.environ["_TEST_PG_SEB"] = "postgresql://seb:pass@localhost/codi_sebastian"
            try:
                path = _write_yaml(tmp, """
tenant_id: sebastian
pg_url: ${_TEST_PG_SEB}
daemon_port: 8421
data_dir: /tmp/seb
qdrant_collection: seb_memories
qdrant_semantic: seb_semantic
""")
                seb = _load_yaml(path)
                assert hare.pg_url != seb.pg_url
                assert "codi_sebastian" in seb.pg_url
            finally:
                os.environ.pop("_TEST_PG_SEB", None)

    def test_ensure_instance_dirs(self):
        """ensure_instance_dirs() creates data_dir and log_dir."""
        from modules.instance_config import reset_instance, InstanceConfig

        with tempfile.TemporaryDirectory() as tmp:
            data_dir = os.path.join(tmp, "data")
            log_dir = os.path.join(tmp, "logs")

            # Manually set instance for this test
            import modules.instance_config as ic
            ic._instance = InstanceConfig(
                tenant_id="test",
                display_name="Test",
                user_id="test",
                pg_url="postgresql://x:y@localhost/db",
                daemon_port=8422,
                data_dir=data_dir,
                qdrant_collection="test_mem",
                qdrant_semantic="test_sem",
                log_dir=log_dir,
                plist_prefix="com.test",
            )

            assert not os.path.exists(data_dir)
            assert not os.path.exists(log_dir)

            from modules.instance_config import ensure_instance_dirs
            ensure_instance_dirs()

            assert os.path.isdir(data_dir)
            assert os.path.isdir(log_dir)
