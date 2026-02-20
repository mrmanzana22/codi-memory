# Server Hardening (C-03 + H-05)

**PR:** Security Hotfix PR2
**Date:** 2026-02-17
**Module:** `server.py` (helpers + middleware)
**Tests:** `tests/test_server_security.py` (27 tests)

## Threats Addressed

| ID | Threat | Impact |
|----|--------|--------|
| C-03 | Server binds to 0.0.0.0 without auth | Any device on the network can access memory API |
| H-05 | No Host header validation | DNS rebinding bypasses localhost-only check |

## Fix: Secure Defaults + Host Validation

### 1. Bind Host (C-03)

Default bind is now `127.0.0.1` (localhost only). To expose remotely:

```bash
# Requires BOTH vars -- without API key, 0.0.0.0 is forced back to 127.0.0.1
export CODI_SERVER_HOST="0.0.0.0"
export CODI_API_KEY="your-secret-key-here"
```

| CODI_SERVER_HOST | CODI_API_KEY | Result |
|-----------------|--------------|--------|
| (not set) | (not set) | `127.0.0.1` (safe default) |
| `0.0.0.0` | set | `0.0.0.0` (allowed) |
| `0.0.0.0` | (not set) | `127.0.0.1` (forced + warning) |
| `::` | (not set) | `127.0.0.1` (forced + warning) |
| `10.0.0.5` | (any) | `10.0.0.5` (custom passthrough) |

### 2. DNS Rebinding Protection (H-05)

Every request (including `/health`) must have a valid `Host` header:

```bash
# Default allowed: localhost, 127.0.0.1, ::1
# Add custom hosts for reverse proxy / Easypanel:
export CODI_ALLOWED_HOSTS="codi.yourdomain.com,internal.server.local"
```

- Host header is parsed, port stripped, lowercased
- Checked against allowed set (defaults always included)
- Missing or invalid Host -> `403 Forbidden`

### 3. Timing-Safe API Key (bonus)

API key comparison now uses `hmac.compare_digest()` to prevent timing side-channel attacks.

## Environment Variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `CODI_SERVER_HOST` | `127.0.0.1` | Bind address |
| `CODI_API_KEY` | (none) | API key for auth (required for remote bind) |
| `CODI_ALLOWED_HOSTS` | `localhost,127.0.0.1,::1` | Valid Host headers (CSV) |
| `MAX_BODY_BYTES` | `262144` | Max request body (256KB) |
| `RATE_LIMIT_PER_MIN` | `60` | Per-IP rate limit |

## Deployment: Exposing Securely

For Easypanel/nginx reverse proxy:

```bash
export MCP_TRANSPORT=sse
export CODI_SERVER_HOST=0.0.0.0
export CODI_API_KEY="generate-a-strong-key-here"
export CODI_ALLOWED_HOSTS="codi.yourdomain.com"
export PORT=8000
```

The middleware validates `Host` header (not `X-Forwarded-Host`) to keep the check simple and hard to bypass.
