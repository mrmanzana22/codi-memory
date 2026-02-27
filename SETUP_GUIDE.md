# codi-memory — Setup Guide

Get a local instance of codi-memory running from scratch.

## Prerequisites

- **Python 3.11+**
- **Docker** (for Qdrant vector store)
- **OpenAI API key** (for embeddings and LLM calls via mem0)
- **Claude Code CLI** (`claude` command available)

## 1. Clone the repo

```bash
git clone <repo-url> codi-memory
cd codi-memory
```

## 2. Create virtual environment and install dependencies

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## 3. Configure environment variables

```bash
cp .env.example .env
```

Edit `.env` and set at minimum:

| Variable | Value | Notes |
|----------|-------|-------|
| `OPENAI_API_KEY` | `sk-...` | Required. Used for embeddings + LLM |
| `QDRANT_URL` | `http://localhost:6333` | Required. Points to your Qdrant instance |
| `USER_ID` | `your_name` | Your identity for memory ownership |

Everything else is optional and has safe defaults. See `.env.example` for the full list.

## 4. Start Qdrant (local, via Docker)

```bash
docker run -d \
  --name qdrant \
  -p 6333:6333 \
  -p 6334:6334 \
  -v qdrant_data:/qdrant/storage \
  qdrant/qdrant
```

Verify it's running:

```bash
curl http://localhost:6333/healthz
# Expected: {"title":"qdrant - vectorass engine","version":"..."}
```

## 5. Verify setup

Run the verification script:

```bash
python scripts/verify_setup.py
```

You should see all checks PASS. If something fails, the script tells you what to fix.

## 6. Test that server.py starts

```bash
python server.py
```

If it starts without errors and prints `[codi-memory] All modules loaded`, the server is working. Press `Ctrl+C` to stop it (you don't need to keep it running manually — Claude Code launches it).

## 7. Register MCP in Claude Code

Edit (or create) `~/.claude.json` and add the `codi-memory` MCP server:

```json
{
  "mcpServers": {
    "codi-memory": {
      "command": "/absolute/path/to/codi-memory/venv/bin/python3",
      "args": ["/absolute/path/to/codi-memory/server.py"],
      "cwd": "/absolute/path/to/codi-memory"
    }
  }
}
```

Replace `/absolute/path/to/codi-memory` with your actual path (e.g., `/Users/sebastian/codi-memory`).

## 8. Quick test

Open Claude Code in any directory:

```bash
claude
```

Then try these MCP tools to verify the connection:

```
> Use recall("test memory") to search for memories
> Use remember("Setup complete - codi-memory is working") to save a memory
> Use recall("setup complete") to verify it was saved
```

If `recall` and `remember` work, you're done.

## Troubleshooting

### "QDRANT_URL no esta configurada"
Your `.env` is missing `QDRANT_URL` or the file isn't being loaded. Check the file exists in the repo root.

### "Connection refused" on Qdrant
Qdrant container isn't running. Check `docker ps` and restart if needed:
```bash
docker start qdrant
```

### OpenAI errors
Verify your API key is valid and has credit:
```bash
python -c "import openai; c=openai.OpenAI(); print(c.models.list().data[0].id)"
```

### MCP not connecting in Claude Code
- Verify paths in `~/.claude.json` are absolute (no `~` or relative paths)
- Verify `cwd` points to the repo root (where `.env` lives)
- Restart Claude Code after editing `~/.claude.json`

### Import errors
Make sure you're using the venv's Python, not system Python:
```bash
which python3  # Should point to codi-memory/venv/bin/python3
```
