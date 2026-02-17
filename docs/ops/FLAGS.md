# Feature Flags (dot-files)

## Available flags

| Flag file | Scope | Values | Default |
|-----------|-------|--------|---------|
| `.write_mode` | Global (all write tools) | sync, async, shadow, dual_ack | sync |
| `.remember_mode` | Per-tool (remember() only) | sync, async, shadow, dual_ack | falls back to .write_mode |

## Precedence

1. Environment variable (`CODI_WRITE_MODE` / `CODI_REMEMBER_MODE`)
2. Dot-file in project root (`.write_mode` / `.remember_mode`)
3. Default (`sync`)

**Note:** Claude Code does NOT propagate custom env vars to MCP servers.
Dot-files are the reliable mechanism today.

## How to change

1. Edit the file: `echo "dual_ack" > /path/to/codi-memory/.remember_mode`
2. Kill zombie MCP processes: `pkill -f "/codi-memory/server.py"`
3. Reopen Claude Code (or start new session)

MCP processes survive Claude Code restarts. Always kill them manually.
