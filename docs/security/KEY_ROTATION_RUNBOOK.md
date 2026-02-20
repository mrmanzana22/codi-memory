# Key Rotation Runbook (C-01)

**PR:** Security Hotfix PR3
**Date:** 2026-02-17
**Cadence:** Every 30-90 days, or immediately on suspected compromise

## Secret Inventory

| Secret | Env Var | Used By | Blast Radius |
|--------|---------|---------|--------------|
| OpenAI API Key | `OPENAI_API_KEY` | mem0 (embeddings + LLM) | Can generate embeddings, costs $ |
| Qdrant API Key | `QDRANT_API_KEY` | qdrant_client (vector store) | Read/write all vectors |
| Supabase URL | `SUPABASE_URL` | training.py | Identifies project |
| Supabase Key | `SUPABASE_KEY` (or `KEY_1`+`KEY_2`) | training.py | Insert/select training_examples |
| Codi API Key | `CODI_API_KEY` | server.py middleware | HTTP API access |

### Where Secrets Live

1. **Local dev:** `.env.local` (gitignored)
2. **Claude Code MCP:** `~/.claude/settings.json` -> `mcpServers.codi-memory.env`
3. **Easypanel/prod:** Environment variables in container config
4. **n8n:** Credential store (for webhooks calling codi-memory)

## Rotation Procedure

### Step 1: Generate New Keys

| Provider | How to Rotate |
|----------|---------------|
| OpenAI | [platform.openai.com/api-keys](https://platform.openai.com/api-keys) -> Create new key |
| Qdrant | Qdrant Cloud dashboard -> Cluster -> API Keys -> Create |
| Supabase | Supabase dashboard -> Settings -> API -> Generate new key |
| Codi API Key | Generate locally: `python3 -c "import secrets; print(secrets.token_urlsafe(32))"` |

### Step 2: Update All Locations

```bash
# 1. Update .env.local
vim .env.local   # Replace old key with new

# 2. Update Claude Code MCP settings
vim ~/.claude/settings.json   # Update env block

# 3. Update Easypanel (if deployed)
# Easypanel UI -> Service -> Environment -> Update vars

# 4. Update n8n credentials (if applicable)
# n8n UI -> Credentials -> Edit
```

### Step 3: Restart Services

```bash
# Local: restart MCP server (Claude Code reconnects automatically)
# Kill existing server process, Claude Code will restart it

# Easypanel: redeploy container
# n8n: no restart needed (credentials hot-reload)
```

### Step 4: Verify

```bash
# 1. Health check
curl -s http://localhost:8000/health | jq .

# 2. MCP tool call (via Claude Code)
# Run: recall("test query")
# Expected: results (not auth error)

# 3. Qdrant connectivity
# Run: search_memory("test")
# Expected: results (not 401)

# 4. Supabase (if enabled)
# Run: contar_ejemplos_training()
# Expected: stats (not "Supabase no configurado")
```

### Step 5: Revoke Old Keys

**Wait 1 hour** after verifying new keys work, then:

| Provider | How to Revoke |
|----------|---------------|
| OpenAI | platform.openai.com -> API Keys -> Delete old key |
| Qdrant | Qdrant Cloud -> API Keys -> Delete old key |
| Supabase | Dashboard -> Settings -> API (anon key auto-rotates; service_role: regenerate) |
| Codi API Key | Just removing from env is sufficient |

## Rollback

If new keys don't work:

1. Revert `.env.local` to old values
2. Revert `~/.claude/settings.json`
3. Restart services
4. Investigate what went wrong before retrying

**Keep old keys for 24h after rotation** in a secure note (not in code/git).

## Emergency: Suspected Compromise

1. **Immediately** rotate ALL keys (Steps 1-3 above)
2. Check audit logs: `SELECT * FROM security_audit_log ORDER BY created_at DESC LIMIT 50`
3. Check Supabase logs for unauthorized access
4. Check OpenAI usage dashboard for unexpected spend
5. Check Qdrant access logs

## Automation (Future)

- Add pre-commit hook to scan for leaked keys (e.g., `detect-secrets`)
- Add startup preflight that warns if keys haven't been rotated in 90 days
- Consider HashiCorp Vault or similar for production secret management
