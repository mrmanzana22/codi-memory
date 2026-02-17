# Qdrant Hardening Runbook

## Overview

Qdrant is the vector database backing Codi's episodic and semantic memory.
By default, Qdrant ships with **no authentication**. This runbook documents
how to enable API key auth and verify the setup.

## Current Setup

- **Host**: Easypanel (`memorycodi-codi.lx6zon.easypanel.host:443`)
- **TLS**: Yes (via Easypanel reverse proxy)
- **Auth**: Configured via `QDRANT_API_KEY` env var

## Step 1: Generate API Key

```bash
# 32 bytes / 64 hex chars — good entropy
openssl rand -hex 32
```

Store the key in a password manager. You'll need it in two places.

## Step 2: Configure Qdrant Server (Easypanel)

1. Open Easypanel dashboard
2. Navigate to the `memorycodi-codi` service
3. Go to **Environment Variables**
4. Add: `QDRANT__SERVICE__API_KEY=<your-key>`
5. (Optional) Add: `QDRANT__SERVICE__READ_ONLY_API_KEY=<another-key>` for read-only access
6. **Restart** the service

## Step 3: Configure Codi Memory Client

Add `QDRANT_API_KEY` in two places:

### a) `.env` file (for local/daemon processes)

```bash
# In /Users/harecjimenez/codi-memory/.env
QDRANT_API_KEY=<your-key>
```

### b) Claude Code settings (for MCP server)

```json
// In ~/.claude/settings.json -> mcpServers.codi-memory.env
{
  "QDRANT_API_KEY": "<your-key>"
}
```

### c) Restart Claude Code

The MCP server needs a restart to pick up the new env var.

## Step 4: Verify

### Request WITHOUT auth (should fail)

```bash
curl -s -o /dev/null -w "%{http_code}" \
  https://memorycodi-codi.lx6zon.easypanel.host:443/collections
# Expected: 401 or 403
```

### Request WITH auth (should succeed)

```bash
curl -s -o /dev/null -w "%{http_code}" \
  -H "api-key: <your-key>" \
  https://memorycodi-codi.lx6zon.easypanel.host:443/collections
# Expected: 200
```

### Verify client guardrail

```bash
# Without QDRANT_API_KEY set, the server should refuse to start:
# RuntimeError: Remote Qdrant requires QDRANT_API_KEY
```

## Key Rotation Procedure

1. Generate new key: `openssl rand -hex 32`
2. Update Qdrant server env var in Easypanel
3. Restart Qdrant service
4. Update `.env` and `settings.json` with new key
5. Restart Claude Code / MCP server
6. Verify with curl commands above

**Important**: Between steps 2-4, the client will fail to connect. Plan for
a brief maintenance window (~2 minutes).

## Guardrail Behavior

The code in `modules/config.py` enforces:

| URL Type | API Key Set | Result |
|----------|-------------|--------|
| localhost / 127.0.0.1 | No | Connects (no auth needed) |
| localhost / 127.0.0.1 | Yes | Connects with auth |
| Remote (any) | No | **RuntimeError** (blocks startup) |
| Remote (any) | Yes | Connects with auth |

### Dev Bypass

For development with a remote Qdrant without auth (NOT recommended):

```bash
export CODI_ALLOW_INSECURE_QDRANT=1
```

This bypass is logged and should never be used in production.

## Network Hardening (Optional)

If you have access to the Easypanel networking config:

1. **Bind to private interface**: Configure Qdrant to listen only on
   private network interfaces (not `0.0.0.0`)
2. **Firewall rules**: Restrict incoming connections to known IPs
3. **Reverse proxy auth**: Add Basic Auth or mTLS at the proxy level

## Related Files

- `modules/config.py` - Client initialization with API key
- `docs/security/THREAT_MODEL.md` - Threat T-010 (Qdrant without auth)
- `docs/security/TOOL_CONTRACTS.md` - GAP-01
