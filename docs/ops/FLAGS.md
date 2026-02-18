# Feature Flags (dot-files)

## Available flags

| Flag file | Scope | Values | Default |
|-----------|-------|--------|---------|
| `.write_mode` | Global (all write tools) | sync, async, shadow, dual_ack | sync |
| `.remember_mode` | Per-tool (remember() only) | sync, async, shadow, dual_ack | falls back to .write_mode |
| `.toolset` | Tool visibility (MCP surface area) | core, ops, research, full | core |

## Precedence

1. Environment variable (`CODI_WRITE_MODE` / `CODI_REMEMBER_MODE` / `CODI_TOOLSET`)
2. Dot-file in project root (`.write_mode` / `.remember_mode` / `.toolset`)
3. Default (`sync` for write modes, `core` for toolset)

**Note:** Claude Code does NOT propagate custom env vars to MCP servers.
Dot-files are the reliable mechanism today.

## How to change

1. Edit the file: `echo "dual_ack" > /path/to/codi-memory/.remember_mode`
2. Kill zombie MCP processes: `pkill -f "/codi-memory/server.py"`
3. Reopen Claude Code (or start new session)

MCP processes survive Claude Code restarts. Always kill them manually.

## Toolset bundles

The `.toolset` flag controls which tools are visible in the MCP registry.

| Toolset | Bundles active | Tools visible | Use case |
|---------|---------------|---------------|----------|
| `core` | core | ~29 | Daily conversation, recall/remember/checkpoint |
| `ops` | core + ops | ~67 | Maintenance, diagnostics, consolidation, training |
| `research` | core + research | ~55 | Introspection, self-model, curiosity, spreading |
| `full` | all | ~115 | Debug, audit, destructive ops |

Quick switch:
```bash
echo "full" > /path/to/codi-memory/.toolset   # enable everything
echo "ops" > /path/to/codi-memory/.toolset    # maintenance mode
rm /path/to/codi-memory/.toolset              # back to core (default)
```

Use `get_toolset_status` tool to check current state.
