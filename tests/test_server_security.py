#!/usr/bin/env python3
"""
SERVER SECURITY TESTS (Security Hotfix PR2 -- C-03 + H-05)
============================================================
Test pure helper functions and middleware behavior for:
  - Bind host resolution (safe defaults)
  - DNS rebinding protection (Host header validation)
  - Timing-safe API key comparison

12 tests:
  1-3. get_bind_host: default, 0.0.0.0 with key, 0.0.0.0 without key
  4-6. get_allowed_hosts: default, custom CSV, always includes defaults
  7-10. parse_host_header: plain, with port, IPv6, empty
  11-12. verify_api_key: correct key, wrong key, timing-safe

Run: ./venv/bin/pytest tests/test_server_security.py -v
"""

import sys
import os
import unittest.mock

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from server import (
    get_bind_host,
    get_allowed_hosts,
    parse_host_header,
    verify_api_key,
    _DEFAULT_ALLOWED_HOSTS,
)


# ============================================================
# get_bind_host()
# ============================================================

class TestGetBindHost:
    """Bind host resolution with safe defaults."""

    def test_default_is_localhost(self):
        """No env vars -> 127.0.0.1."""
        with unittest.mock.patch.dict(os.environ, {}, clear=True):
            os.environ.pop("CODI_SERVER_HOST", None)
            os.environ.pop("CODI_API_KEY", None)
            assert get_bind_host() == "127.0.0.1"

    def test_explicit_localhost(self):
        """CODI_SERVER_HOST=127.0.0.1 -> 127.0.0.1."""
        with unittest.mock.patch.dict(os.environ, {
            "CODI_SERVER_HOST": "127.0.0.1"
        }, clear=True):
            assert get_bind_host() == "127.0.0.1"

    def test_all_interfaces_with_api_key(self):
        """0.0.0.0 with CODI_API_KEY set -> allowed."""
        with unittest.mock.patch.dict(os.environ, {
            "CODI_SERVER_HOST": "0.0.0.0",
            "CODI_API_KEY": "test-secret-key",
        }, clear=True):
            assert get_bind_host() == "0.0.0.0"

    def test_all_interfaces_without_api_key_forced_localhost(self):
        """0.0.0.0 without CODI_API_KEY -> forced to 127.0.0.1."""
        with unittest.mock.patch.dict(os.environ, {
            "CODI_SERVER_HOST": "0.0.0.0",
        }, clear=True):
            os.environ.pop("CODI_API_KEY", None)
            assert get_bind_host() == "127.0.0.1"

    def test_ipv6_all_without_key_forced_localhost(self):
        """:: without CODI_API_KEY -> forced to 127.0.0.1."""
        with unittest.mock.patch.dict(os.environ, {
            "CODI_SERVER_HOST": "::",
        }, clear=True):
            os.environ.pop("CODI_API_KEY", None)
            assert get_bind_host() == "127.0.0.1"

    def test_custom_host_passthrough(self):
        """Custom host (not 0.0.0.0) passes through regardless of API key."""
        with unittest.mock.patch.dict(os.environ, {
            "CODI_SERVER_HOST": "10.0.0.5",
        }, clear=True):
            os.environ.pop("CODI_API_KEY", None)
            assert get_bind_host() == "10.0.0.5"


# ============================================================
# get_allowed_hosts()
# ============================================================

class TestGetAllowedHosts:
    """Allowed hosts parsing for DNS rebinding protection."""

    def test_default_hosts(self):
        """No CODI_ALLOWED_HOSTS -> localhost, 127.0.0.1, ::1."""
        with unittest.mock.patch.dict(os.environ, {}, clear=True):
            os.environ.pop("CODI_ALLOWED_HOSTS", None)
            result = get_allowed_hosts()
            assert result == set(_DEFAULT_ALLOWED_HOSTS)

    def test_custom_csv(self):
        """CODI_ALLOWED_HOSTS adds custom entries on top of defaults."""
        with unittest.mock.patch.dict(os.environ, {
            "CODI_ALLOWED_HOSTS": "codi.example.com, myserver.local",
        }, clear=True):
            result = get_allowed_hosts()
            assert "codi.example.com" in result
            assert "myserver.local" in result
            # Defaults always present
            assert "localhost" in result
            assert "127.0.0.1" in result
            assert "::1" in result

    def test_always_includes_defaults(self):
        """Even if user provides only custom hosts, defaults remain."""
        with unittest.mock.patch.dict(os.environ, {
            "CODI_ALLOWED_HOSTS": "only-this.com",
        }, clear=True):
            result = get_allowed_hosts()
            assert "only-this.com" in result
            assert "localhost" in result

    def test_handles_whitespace_and_empty(self):
        """Whitespace in CSV is stripped, empty entries ignored."""
        with unittest.mock.patch.dict(os.environ, {
            "CODI_ALLOWED_HOSTS": " foo.com , , bar.com , ",
        }, clear=True):
            result = get_allowed_hosts()
            assert "foo.com" in result
            assert "bar.com" in result
            assert "" not in result


# ============================================================
# parse_host_header()
# ============================================================

class TestParseHostHeader:
    """Host header parsing for DNS rebinding check."""

    def test_plain_hostname(self):
        assert parse_host_header("localhost") == "localhost"

    def test_hostname_with_port(self):
        assert parse_host_header("localhost:8765") == "localhost"

    def test_ip_with_port(self):
        assert parse_host_header("127.0.0.1:8000") == "127.0.0.1"

    def test_ipv6_with_brackets_and_port(self):
        assert parse_host_header("[::1]:8000") == "::1"

    def test_ipv6_brackets_no_port(self):
        assert parse_host_header("[::1]") == "::1"

    def test_empty_string(self):
        assert parse_host_header("") == ""

    def test_none_like_missing(self):
        """Missing header comes as empty string from request.headers.get."""
        assert parse_host_header("") == ""

    def test_uppercase_normalized(self):
        assert parse_host_header("MyServer.COM:8080") == "myserver.com"

    def test_domain_no_port(self):
        assert parse_host_header("codi.example.com") == "codi.example.com"

    def test_malformed_ipv6_no_close_bracket(self):
        """Malformed IPv6 (no closing bracket) -> empty (safe rejection)."""
        assert parse_host_header("[::1") == ""


# ============================================================
# verify_api_key()
# ============================================================

class TestVerifyApiKey:
    """Timing-safe API key verification."""

    def test_correct_key(self):
        assert verify_api_key("my-secret-key", "my-secret-key") is True

    def test_wrong_key(self):
        assert verify_api_key("wrong-key", "my-secret-key") is False

    def test_empty_provided(self):
        assert verify_api_key("", "my-secret-key") is False

    def test_empty_expected(self):
        assert verify_api_key("any-key", "") is False

    def test_both_empty(self):
        assert verify_api_key("", "") is False

    def test_unicode_key(self):
        assert verify_api_key("clave-secreta-123", "clave-secreta-123") is True

    def test_near_match_fails(self):
        """Keys that differ by one char must fail."""
        assert verify_api_key("my-secret-key-a", "my-secret-key-b") is False
