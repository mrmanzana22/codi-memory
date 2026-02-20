"""Tests for modules/n8n.py.

Covers:
  - trigger_n8n (webhook dispatch with mocked HTTP)
  - listar_webhooks_conocidos (pure function)
  - Input validation (invalid paths, missing config)
"""

import pytest
from unittest.mock import patch, MagicMock


class TestTriggerN8n:
    """Tests for trigger_n8n."""

    def test_no_base_url_returns_error(self):
        from modules.n8n import trigger_n8n
        with patch("modules.n8n.N8N_WEBHOOK_BASE", ""):
            result = trigger_n8n("test-webhook", {"key": "value"})
            assert "no esta configurado" in result

    def test_invalid_path_rejected(self):
        from modules.n8n import trigger_n8n
        with patch("modules.n8n.N8N_WEBHOOK_BASE", "https://n8n.example.com"):
            # Path with spaces
            result = trigger_n8n("bad path", {})
            assert "invalido" in result

    def test_invalid_path_special_chars(self):
        from modules.n8n import trigger_n8n
        with patch("modules.n8n.N8N_WEBHOOK_BASE", "https://n8n.example.com"):
            result = trigger_n8n("../../etc/passwd", {})
            assert "invalido" in result

    def test_empty_path_rejected(self):
        from modules.n8n import trigger_n8n
        with patch("modules.n8n.N8N_WEBHOOK_BASE", "https://n8n.example.com"):
            result = trigger_n8n("", {})
            assert "invalido" in result

    @patch("modules.n8n.http_requests")
    def test_successful_fire_and_forget(self, mock_requests):
        from modules.n8n import trigger_n8n
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_requests.post.return_value = mock_response

        with patch("modules.n8n.N8N_WEBHOOK_BASE", "https://n8n.example.com"):
            result = trigger_n8n("test-hook", {"data": "value"})
            assert "disparado" in result
            mock_requests.post.assert_called_once()
            call_args = mock_requests.post.call_args
            assert "test-hook" in call_args[0][0]

    @patch("modules.n8n.http_requests")
    def test_adds_metadata_to_payload(self, mock_requests):
        from modules.n8n import trigger_n8n
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_requests.post.return_value = mock_response

        with patch("modules.n8n.N8N_WEBHOOK_BASE", "https://n8n.example.com"):
            trigger_n8n("hook", {"custom": "data"})
            sent_payload = mock_requests.post.call_args[1]["json"]
            assert sent_payload["_from"] == "codi-memory"
            assert "_timestamp" in sent_payload
            assert sent_payload["custom"] == "data"

    @patch("modules.n8n.http_requests")
    def test_wait_for_response(self, mock_requests):
        from modules.n8n import trigger_n8n
        mock_response = MagicMock()
        mock_response.json.return_value = {"status": "ok"}
        mock_requests.post.return_value = mock_response

        with patch("modules.n8n.N8N_WEBHOOK_BASE", "https://n8n.example.com"):
            result = trigger_n8n("hook", esperar_respuesta=True)
            assert "ok" in result
            # Timeout should be 30s for wait mode
            call_kwargs = mock_requests.post.call_args[1]
            assert call_kwargs["timeout"] == 30

    @patch("modules.n8n.http_requests")
    def test_error_status_code(self, mock_requests):
        from modules.n8n import trigger_n8n
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.text = "Internal Server Error"
        mock_requests.post.return_value = mock_response

        with patch("modules.n8n.N8N_WEBHOOK_BASE", "https://n8n.example.com"):
            result = trigger_n8n("hook")
            assert "Error" in result
            assert "500" in result

    @patch("modules.n8n.http_requests")
    def test_timeout_handled(self, mock_requests):
        import requests
        from modules.n8n import trigger_n8n
        mock_requests.post.side_effect = requests.exceptions.Timeout()
        mock_requests.exceptions = requests.exceptions

        with patch("modules.n8n.N8N_WEBHOOK_BASE", "https://n8n.example.com"):
            result = trigger_n8n("hook")
            assert "Timeout" in result


class TestListarWebhooks:
    """Tests for listar_webhooks_conocidos."""

    def test_returns_markdown(self):
        from modules.n8n import listar_webhooks_conocidos
        result = listar_webhooks_conocidos()
        assert "WEBHOOKS N8N" in result

    def test_includes_known_webhooks(self):
        from modules.n8n import listar_webhooks_conocidos
        result = listar_webhooks_conocidos()
        assert "codi-alerta" in result
        assert "trading-signal" in result

    def test_shows_base_url(self):
        from modules.n8n import listar_webhooks_conocidos
        with patch("modules.n8n.N8N_WEBHOOK_BASE", "https://test.example.com"):
            result = listar_webhooks_conocidos()
            assert "test.example.com" in result
