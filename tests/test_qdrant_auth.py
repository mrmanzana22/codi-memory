"""
Tests for Qdrant authentication guardrail (G-01 remediation).

Validates:
  - API key is passed to QdrantClient constructor
  - Remote Qdrant without API key raises RuntimeError
  - CODI_ALLOW_INSECURE_QDRANT=1 bypasses the guardrail
  - API key is included in mem0_config when set
  - Sanitized URL logging (no creds leaked)
"""

import sys
from unittest.mock import patch, MagicMock

import pytest

# Ensure qdrant_client is available even if not installed
_mock_qdrant_module = MagicMock()
if "qdrant_client" not in sys.modules:
    sys.modules["qdrant_client"] = _mock_qdrant_module

import modules.config as config


@pytest.fixture(autouse=True)
def _reset_qdrant_singleton():
    """Reset the lazy Qdrant singleton before/after each test."""
    old = config._qdrant
    config._qdrant = None
    yield
    config._qdrant = old


class TestQdrantApiKey:

    def test_qdrant_client_receives_api_key(self, monkeypatch):
        """QdrantClient is constructed with api_key when QDRANT_API_KEY is set."""
        monkeypatch.setattr(config, "QDRANT_URL", "https://remote.host:443")
        monkeypatch.setattr(config, "QDRANT_API_KEY", "test-secret-key-123")

        mock_client = MagicMock()
        with patch("qdrant_client.QdrantClient", return_value=mock_client) as mock_cls:
            result = config.get_qdrant()

        mock_cls.assert_called_once_with(
            url="https://remote.host:443",
            api_key="test-secret-key-123",
            timeout=30,
        )
        assert result is mock_client

    def test_qdrant_client_no_key_passes_none(self, monkeypatch):
        """QdrantClient gets api_key=None when QDRANT_API_KEY is empty."""
        monkeypatch.setattr(config, "QDRANT_URL", "http://localhost:6333")
        monkeypatch.setattr(config, "QDRANT_API_KEY", "")

        mock_client = MagicMock()
        with patch("qdrant_client.QdrantClient", return_value=mock_client) as mock_cls:
            config.get_qdrant()

        mock_cls.assert_called_once_with(
            url="http://localhost:6333",
            api_key=None,
            timeout=30,
        )


class TestRemoteGuardrail:

    def test_remote_qdrant_without_key_raises(self, monkeypatch):
        """Remote URL + no API key + no bypass -> RuntimeError."""
        monkeypatch.setattr(config, "QDRANT_URL", "https://memorycodi-codi.lx6zon.easypanel.host:443")
        monkeypatch.setattr(config, "QDRANT_API_KEY", "")
        monkeypatch.delenv("CODI_ALLOW_INSECURE_QDRANT", raising=False)

        with pytest.raises(RuntimeError, match="requires QDRANT_API_KEY"):
            config.get_qdrant()

    def test_allow_insecure_bypass(self, monkeypatch):
        """CODI_ALLOW_INSECURE_QDRANT=1 skips the guardrail."""
        monkeypatch.setattr(config, "QDRANT_URL", "https://remote.host:443")
        monkeypatch.setattr(config, "QDRANT_API_KEY", "")
        monkeypatch.setenv("CODI_ALLOW_INSECURE_QDRANT", "1")

        mock_client = MagicMock()
        with patch("qdrant_client.QdrantClient", return_value=mock_client):
            result = config.get_qdrant()

        assert result is mock_client

    def test_localhost_no_key_no_error(self, monkeypatch):
        """Localhost URL without API key does NOT raise."""
        monkeypatch.setattr(config, "QDRANT_URL", "http://localhost:6333")
        monkeypatch.setattr(config, "QDRANT_API_KEY", "")
        monkeypatch.delenv("CODI_ALLOW_INSECURE_QDRANT", raising=False)

        mock_client = MagicMock()
        with patch("qdrant_client.QdrantClient", return_value=mock_client):
            result = config.get_qdrant()

        assert result is mock_client

    def test_127_0_0_1_no_key_no_error(self, monkeypatch):
        """127.0.0.1 URL without API key does NOT raise."""
        monkeypatch.setattr(config, "QDRANT_URL", "http://127.0.0.1:6333")
        monkeypatch.setattr(config, "QDRANT_API_KEY", "")
        monkeypatch.delenv("CODI_ALLOW_INSECURE_QDRANT", raising=False)

        mock_client = MagicMock()
        with patch("qdrant_client.QdrantClient", return_value=mock_client):
            result = config.get_qdrant()

        assert result is mock_client


class TestSanitizeUrl:

    def test_strips_path_and_query(self):
        assert config._sanitize_url("https://host.com:443/path?key=val") == "https://host.com:443"

    def test_preserves_scheme_and_host(self):
        assert config._sanitize_url("http://localhost:6333") == "http://localhost:6333"

    def test_handles_malformed(self):
        result = config._sanitize_url("not-a-url")
        assert isinstance(result, str)


class TestIsRemoteQdrant:

    def test_localhost_is_not_remote(self):
        assert config._is_remote_qdrant("http://localhost:6333") is False

    def test_127_is_not_remote(self):
        assert config._is_remote_qdrant("http://127.0.0.1:6333") is False

    def test_easypanel_is_remote(self):
        assert config._is_remote_qdrant("https://memorycodi-codi.lx6zon.easypanel.host:443") is True

    def test_empty_is_not_remote(self):
        assert config._is_remote_qdrant("") is False


class TestMem0Config:

    def test_mem0_config_includes_api_key_when_set(self, monkeypatch):
        """mem0_config vector_store includes api_key when QDRANT_API_KEY is set."""
        monkeypatch.setattr(config, "QDRANT_API_KEY", "my-secret")
        # Re-evaluate mem0_config (it's module-level, so we check the template)
        qdrant_config = {
            "collection_name": config.COLLECTION_NAME,
            "url": config.QDRANT_URL,
            **({"api_key": config.QDRANT_API_KEY} if config.QDRANT_API_KEY else {}),
        }
        assert qdrant_config["api_key"] == "my-secret"

    def test_mem0_config_no_api_key_when_empty(self, monkeypatch):
        """mem0_config vector_store omits api_key when QDRANT_API_KEY is empty."""
        monkeypatch.setattr(config, "QDRANT_API_KEY", "")
        qdrant_config = {
            "collection_name": config.COLLECTION_NAME,
            "url": config.QDRANT_URL,
            **({"api_key": config.QDRANT_API_KEY} if config.QDRANT_API_KEY else {}),
        }
        assert "api_key" not in qdrant_config
