#!/usr/bin/env python3
"""
Codi Memory - MCP Server v3.0 (Modular Architecture)
Slim orchestrator: imports modules and registers all tools.

Modules:
  config.py       - Shared configuration, MCP server, lazy init
  utils.py        - Helpers (session, analysis, backup, PAD)
  triggers.py     - Trigger detection, evaluation, creation
  books.py        - Libro/book management
  memory_core.py  - Core memory CRUD, search, ownership
  memory_smart.py - FTS5, smart add, hybrid search
  training.py     - Training examples for Supabase
  maintenance.py  - Maintenance tasks, reminders
  flush.py        - Checkpoint, flush_session, export
  consciousness.py- PAD emotional, self-model, workspace, prediction,
                    consolidation, system, cognitive, N8N
"""

# Suppress warnings before any import
import warnings
warnings.filterwarnings("ignore")

import os
import hmac
import logging

_logger = logging.getLogger("codi-memory")

# ============================================================
# STALE WAL/SHM CLEANUP - remove orphaned lock files on startup
# ============================================================
import subprocess

def _cleanup_stale_wal(db_path: str) -> None:
    """Remove WAL/SHM files if no process has the DB open."""
    shm = db_path + "-shm"
    wal = db_path + "-wal"
    if not os.path.exists(shm):
        return
    try:
        result = subprocess.run(
            ["lsof", db_path], capture_output=True, text=True, timeout=5
        )
        # lsof outputs lines for each process holding the file
        # If only the header or empty, nobody has it open
        lines = [l for l in result.stdout.strip().splitlines() if l and not l.startswith("COMMAND")]
        if not lines:
            for f in (shm, wal):
                if os.path.exists(f):
                    os.remove(f)
                    _logger.info("Removed stale %s", f)
    except Exception:
        pass  # lsof not available or timeout — skip cleanup

# ============================================================
# SCHEMA MIGRATIONS (D4) - must run BEFORE any module import
# ============================================================
from modules.config import FTS_DB_PATH, PROSPECTIVE_DB_PATH
from modules.migrations import apply_migrations

_server_dir = os.path.dirname(os.path.abspath(__file__))
_cleanup_stale_wal(FTS_DB_PATH)
_cleanup_stale_wal(PROSPECTIVE_DB_PATH)
apply_migrations(FTS_DB_PATH, migrations_dir=os.path.join(_server_dir, "migrations"))
apply_migrations(PROSPECTIVE_DB_PATH, migrations_dir=os.path.join(_server_dir, "migrations_prospective"))

# ============================================================
# NOW import modules (tables already exist from migrations)
# ============================================================
from modules.config import mcp, memory, qdrant, USER_ID, COLLECTION_NAME, now_iso
from modules import metrics

# Import all modules with register_tools
from modules import triggers
from modules import books
from modules import memory_core
from modules import memory_smart
from modules import training
from modules import maintenance
from modules import flush
from modules import consciousness
from modules import working_memory
from modules import interface
from modules import spreading
from modules import consolidation
from modules import sharpe
from modules import sharpe_insights
from modules import prospective
from modules import goals
from modules import wiring
from modules import assessment
from modules import session_bridge
from modules import sleep_loop
from modules import write_queue
from modules import narrative
from modules import tool_governance
from modules import ollama_router

# ============================================================
# INSTRUMENTATION (A1) - must run BEFORE register_tools()
# ============================================================
metrics.instrument_mcp(mcp)

# ============================================================
# REGISTER ALL TOOLS
# ============================================================

metrics.register_metrics_tools(mcp)
triggers.register_tools(mcp)
books.register_tools(mcp)
memory_core.register_tools(mcp)
memory_smart.register_tools(mcp)
training.register_tools(mcp)
maintenance.register_tools(mcp)
flush.register_tools(mcp)
consciousness.register_tools(mcp)
working_memory.register_tools(mcp)
interface.register_tools(mcp)
spreading.register_tools(mcp)
consolidation.register_consolidation_tools(mcp)
sharpe.register_sharpe_tools(mcp)
sharpe_insights.register_insights_tools(mcp)
prospective.register_prospective_tools(mcp)
goals.register_goal_tools(mcp)
assessment.register_assessment_tools(mcp)
session_bridge.register_tools(mcp)
sleep_loop.register_tools(mcp)
write_queue.register_tools(mcp)
narrative.register_tools(mcp)
tool_governance.register_governance_tool(mcp)
ollama_router.register_tools(mcp)

# ============================================================
# TOOL GOVERNANCE - Apply bundle-based visibility filter
# ============================================================
_toolset = tool_governance.resolve_toolset()
_allowed = tool_governance.get_allowed_tools(_toolset)
_total_before = len(mcp._tool_manager._tools)
_removed_count, _removed_names = tool_governance.apply_allowlist(mcp, _allowed)
_total_after = len(mcp._tool_manager._tools)

# ============================================================
# WIRE EVENT BUS (Thalamocortical Integration)
# ============================================================
wiring.wire_event_bus()

print(f"[codi-memory] All modules loaded. Event bus wired.")
print(f"[codi-memory] [toolset={_toolset}] visible={_total_after} hidden={_removed_count} (of {_total_before} total)")


# ============================================================
# HTTP SECURITY HELPERS (PR2 — C-03 + H-05)
# ============================================================
# Pure functions, importable and testable without starting the server.

_DEFAULT_ALLOWED_HOSTS = frozenset({"localhost", "127.0.0.1", "::1"})


def get_bind_host() -> str:
    """Resolve bind host with safe defaults.

    - Default: 127.0.0.1 (localhost only)
    - If CODI_SERVER_HOST is 0.0.0.0 or ::, requires CODI_API_KEY
    - Forces 127.0.0.1 + warning if binding remotely without API key
    """
    requested = os.getenv("CODI_SERVER_HOST", "127.0.0.1").strip()
    api_key = os.getenv("CODI_API_KEY", "").strip()

    if requested in ("0.0.0.0", "::"):
        has_allowed_hosts = bool(os.getenv("CODI_ALLOWED_HOSTS", "").strip())
        if not api_key and not has_allowed_hosts:
            _logger.warning(
                "CODI_SERVER_HOST=%s without CODI_API_KEY or CODI_ALLOWED_HOSTS"
                " -- forcing 127.0.0.1",
                requested,
            )
            return "127.0.0.1"
    return requested


def get_allowed_hosts() -> set:
    """Parse CODI_ALLOWED_HOSTS CSV into a set.

    Default: localhost, 127.0.0.1, ::1.
    Custom entries are added on top of the defaults (never replaces them).
    """
    raw = os.getenv("CODI_ALLOWED_HOSTS", "").strip()
    hosts = set(_DEFAULT_ALLOWED_HOSTS)
    if raw:
        for h in raw.split(","):
            h = h.strip().lower()
            if h:
                hosts.add(h)
    return hosts


def parse_host_header(host_header: str) -> str:
    """Extract hostname from Host header, stripping port.

    Handles: 'localhost:8000', '127.0.0.1:8765', '[::1]:8000', 'example.com'.
    Returns lowercased hostname, or '' if missing/invalid.
    """
    if not host_header:
        return ""
    host_header = host_header.strip()
    # IPv6 with brackets: [::1]:8000
    if host_header.startswith("["):
        bracket_end = host_header.find("]")
        if bracket_end != -1:
            return host_header[1:bracket_end].lower()
        return ""
    # host:port
    if ":" in host_header:
        return host_header.rsplit(":", 1)[0].lower()
    return host_header.lower()


def verify_api_key(provided: str, expected: str) -> bool:
    """Timing-safe API key comparison using hmac.compare_digest."""
    if not expected or not provided:
        return False
    return hmac.compare_digest(provided.encode("utf-8"), expected.encode("utf-8"))


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    # Soporte para stdio (local) y SSE/HTTP (remoto/Easypanel)
    transport = os.getenv("MCP_TRANSPORT", "stdio")

    if transport in ("sse", "http", "streamable-http"):
        import uvicorn
        from starlette.applications import Starlette
        from starlette.routing import Mount, Route
        from starlette.responses import JSONResponse
        from qdrant_client.models import Filter, FieldCondition, MatchValue
        from modules.maintenance import _cargar_recordatorios, _guardar_recordatorios
        from starlette.middleware.base import BaseHTTPMiddleware
        from starlette.middleware import Middleware
        import json
        import time

        # ============================================================
        # HTTP SECURITY (Host validation + API key + rate limit + size)
        # ============================================================
        CODI_API_KEY = os.getenv("CODI_API_KEY", "").strip()
        MAX_BODY_BYTES = int(os.getenv("MAX_BODY_BYTES", "262144"))  # 256KB
        RATE_LIMIT_PER_MIN = int(os.getenv("RATE_LIMIT_PER_MIN", "60"))

        _allowed_hosts = get_allowed_hosts()
        _rate_window = {}  # ip -> (window_start_epoch, count)

        def _client_ip(request) -> str:
            try:
                return request.client.host if request.client else "unknown"
            except Exception:
                return "unknown"

        class SecurityMiddleware(BaseHTTPMiddleware):
            async def dispatch(self, request, call_next):
                # 1. DNS rebinding protection (ALL routes, including /health)
                host_header = request.headers.get("host", "")
                hostname = parse_host_header(host_header)
                # Support wildcard suffixes: *.trycloudflare.com
                host_ok = hostname in _allowed_hosts
                if not host_ok:
                    host_ok = any(
                        a.startswith("*.") and hostname.endswith(a[1:])
                        for a in _allowed_hosts
                    )
                if not hostname or not host_ok:
                    return JSONResponse(
                        {"error": "forbidden host"}, status_code=403
                    )

                path = request.url.path

                # 2. Health checks skip auth + rate-limit
                if path == "/health":
                    return await call_next(request)

                # 2b. MCP protocol paths (/sse, /messages) from allowed
                # tunnel hosts skip API key auth. The tunnel URL itself
                # serves as the access control (random, unguessable).
                _MCP_PATHS = ("/sse", "/messages", "/messages/", "/mcp", "/mcp/")
                _is_tunnel = hostname not in _DEFAULT_ALLOWED_HOSTS
                if _is_tunnel and host_ok and (path in _MCP_PATHS or path.startswith("/mcp")):
                    return await call_next(request)

                ip = _client_ip(request)

                # 3. Authentication (timing-safe):
                # - If CODI_API_KEY is set, require it.
                # - If not set, allow only localhost (safer default).
                if CODI_API_KEY:
                    token = request.headers.get("x-api-key", "")
                    auth = request.headers.get("authorization", "")
                    if not token and auth.lower().startswith("bearer "):
                        token = auth.split(" ", 1)[1].strip()
                    if not verify_api_key(token, CODI_API_KEY):
                        return JSONResponse({"error": "unauthorized"}, status_code=401)
                else:
                    if ip not in ("127.0.0.1", "::1"):
                        return JSONResponse({"error": "server not configured (CODI_API_KEY missing)"}, status_code=503)

                # 4. Rate limiting per IP (rolling 60s window)
                now = int(time.time())
                window_start, count = _rate_window.get(ip, (now, 0))
                if now - window_start >= 60:
                    window_start, count = now, 0
                count += 1
                _rate_window[ip] = (window_start, count)
                if count > RATE_LIMIT_PER_MIN:
                    return JSONResponse({"error": "rate limited"}, status_code=429)

                # 5. Body size limit (best-effort using Content-Length)
                if request.method in ("POST", "PUT", "PATCH"):
                    cl = request.headers.get("content-length")
                    if cl:
                        try:
                            if int(cl) > MAX_BODY_BYTES:
                                return JSONResponse({"error": "payload too large"}, status_code=413)
                        except Exception:
                            pass

                return await call_next(request)


        # Endpoint para recibir recordatorios de n8n u otros sistemas
        async def recibir_recordatorio(request):
            try:
                raw = await request.body()
                if raw and len(raw) > MAX_BODY_BYTES:
                    return JSONResponse({"error": "payload too large"}, status_code=413)
                try:
                    body = json.loads(raw.decode("utf-8") if raw else "{}")
                except Exception:
                    return JSONResponse({"error": "json invalido"}, status_code=400)

                mensaje = body.get('mensaje', '')
                prioridad = body.get('prioridad', 'normal')
                origen = body.get('origen', 'externo')

                if not isinstance(mensaje, str) or not mensaje.strip():
                    return JSONResponse({"error": "mensaje requerido"}, status_code=400)
                if len(mensaje) > 2000:
                    return JSONResponse({"error": "mensaje demasiado largo"}, status_code=413)
                if prioridad not in ("baja", "media", "alta", "normal"):
                    return JSONResponse({"error": "prioridad invalida"}, status_code=400)
                if not isinstance(origen, str) or len(origen) > 64:
                    return JSONResponse({"error": "origen invalido"}, status_code=400)

                # Guardar recordatorio
                data = _cargar_recordatorios()
                data['pendientes'].append({
                    "mensaje": mensaje,
                    "prioridad": prioridad,
                    "origen": origen,
                    "timestamp": now_iso()
                })
                _guardar_recordatorios(data)

                # Tambien guardar como memoria si es alta prioridad
                if prioridad == "alta":
                    memory.add(
                        f"[RECORDATORIO EXTERNO] {mensaje} (de {origen})",
                        user_id=USER_ID,
                        metadata={
                            'category': 'recordatorio',
                            'origen': origen,
                            'narrative_importance': 'high'
                        }
                    )

                return JSONResponse({
                    "status": "ok",
                    "mensaje": f"Recordatorio guardado: {mensaje[:50]}..."
                })

            except Exception as e:
                from modules.secret_redact import redact_secrets
                _logger.error("recordatorio error: %s", e)
                return JSONResponse({"error": redact_secrets(str(e))}, status_code=500)

        # Health check
        async def health(request):
            return JSONResponse({"status": "ok", "service": "codi-memory"})

        # ============================================================
        # API HTTP PARA N8N Y CODI-LOOP
        # ============================================================

        async def api_context(request):
            """GET /api/context - Retorna contexto de despertar para sesiones autonomas"""
            try:
                contexto = []

                # 1. Memorias CRITICAS (identidad)
                points, _ = qdrant.scroll(
                    collection_name=COLLECTION_NAME,
                    scroll_filter=Filter(must=[
                        FieldCondition(key='narrative_importance', match=MatchValue(value='critical'))
                    ]),
                    limit=5,
                    with_payload=True
                )
                identidad = []
                if points:
                    for p in points:
                        data = p.payload.get('data', '')
                        source = p.payload.get('ownership_source', '')
                        identidad.append({"memory": data, "source": source})

                # 2. Proyecto actual
                proyecto = memory.search(query="proyecto trabajando actual", user_id=USER_ID, limit=3)
                proyectos = []
                if proyecto and proyecto.get("results"):
                    for m in proyecto["results"]:
                        proyectos.append(m.get('memory', ''))

                # 3. Pendientes
                pendientes_search = memory.search(query="pendiente falta por hacer bloqueador", user_id=USER_ID, limit=3)
                pendientes = []
                if pendientes_search and pendientes_search.get("results"):
                    for m in pendientes_search["results"]:
                        pendientes.append(m.get('memory', ''))

                # 4. Memorias recientes
                recientes_search = memory.search(query="reciente ultimo hoy ayer memoria checkpoint", user_id=USER_ID, limit=10)
                recientes = []
                if recientes_search and recientes_search.get("results"):
                    for m in recientes_search["results"]:
                        recientes.append({
                            "memory": m.get('memory', ''),
                            "score": m.get('score', 0)
                        })

                return JSONResponse({
                    "status": "ok",
                    "context": {
                        "identidad": identidad,
                        "proyectos": proyectos,
                        "pendientes": pendientes,
                        "recientes": recientes
                    },
                    "timestamp": now_iso()
                })

            except Exception as e:
                from modules.secret_redact import redact_secrets
                _logger.error("api_context error: %s", e)
                return JSONResponse({"error": redact_secrets(str(e))}, status_code=500)

        async def api_memory(request):
            """POST /api/memory - Guarda una nueva memoria"""
            try:
                raw = await request.body()
                if raw and len(raw) > MAX_BODY_BYTES:
                    return JSONResponse({"error": "payload too large"}, status_code=413)
                try:
                    body = json.loads(raw.decode("utf-8") if raw else "{}")
                except Exception:
                    return JSONResponse({"error": "json invalido"}, status_code=400)

                content = body.get('content', '')
                category = body.get('category', 'general')
                source = body.get('source', 'experienced')
                importance = body.get('importance', 'medium')

                if not isinstance(content, str):
                    return JSONResponse({"error": "content debe ser string"}, status_code=400)
                if len(content) > 10000:
                    return JSONResponse({"error": "content demasiado largo"}, status_code=413)
                if not isinstance(category, str) or len(category) > 64:
                    return JSONResponse({"error": "category invalida"}, status_code=400)
                if not isinstance(source, str) or len(source) > 32:
                    return JSONResponse({"error": "source invalida"}, status_code=400)
                if not isinstance(importance, str) or len(importance) > 32:
                    return JSONResponse({"error": "importance invalida"}, status_code=400)

                if not content:
                    return JSONResponse({"error": "content requerido"}, status_code=400)

                result = memory.add(
                    content,
                    user_id=USER_ID,
                    metadata={
                        'category': category,
                        'ownership_source': source,
                        'ownership_confidence': 1.0 if source == 'experienced' else 0.8,
                        'narrative_importance': importance,
                        'created_at': now_iso()
                    }
                )

                return JSONResponse({
                    "status": "ok",
                    "message": f"Memoria guardada: {content[:50]}...",
                    "result": result
                })

            except Exception as e:
                from modules.secret_redact import redact_secrets
                _logger.error("api_memory error: %s", e)
                return JSONResponse({"error": redact_secrets(str(e))}, status_code=500)

        async def api_search(request):
            """GET /api/search?q=query&limit=5 - Busca memorias"""
            try:
                query = request.query_params.get('q', '')
                if len(query) > 512:
                    return JSONResponse({'error': 'query demasiado largo'}, status_code=413)
                try:
                    limit = int(request.query_params.get('limit', 5))
                except (ValueError, TypeError):
                    limit = 5
                limit = max(1, min(20, limit))

                if not query:
                    return JSONResponse({"error": "q parameter requerido"}, status_code=400)

                results = memory.search(query=query, user_id=USER_ID, limit=limit)

                memorias = []
                if results and results.get("results"):
                    for m in results["results"]:
                        memorias.append({
                            "memory": m.get('memory', ''),
                            "score": m.get('score', 0)
                        })

                return JSONResponse({
                    "status": "ok",
                    "query": query,
                    "results": memorias,
                    "count": len(memorias)
                })

            except Exception as e:
                from modules.secret_redact import redact_secrets
                _logger.error("api_search error: %s", e)
                return JSONResponse({"error": redact_secrets(str(e))}, status_code=500)

        # Streamable HTTP transport (for Codex CLI and modern MCP clients)
        import contextlib
        from mcp.server.streamable_http_manager import StreamableHTTPSessionManager
        streamable_manager = StreamableHTTPSessionManager(
            app=mcp._mcp_server,
            stateless=False,
        )

        @contextlib.asynccontextmanager
        async def lifespan(app):
            async with streamable_manager.run():
                yield

        port = int(os.getenv("PORT", 8000))
        host = get_bind_host()
        print(f"[codi-memory] Starting MCP server on {transport} transport, {host}:{port}")
        print(f"[codi-memory] Transports: SSE at /sse, Streamable HTTP at /mcp")
        print(f"[codi-memory] Allowed hosts: {sorted(_allowed_hosts)}")

        # Normalize /mcp -> /mcp/ so Mount matches (redirect breaks POST body)
        class TrailingSlashMCP:
            def __init__(self, app):
                self.app = app
            async def __call__(self, scope, receive, send):
                if scope["type"] == "http" and scope["path"] == "/mcp":
                    scope = dict(scope, path="/mcp/")
                await self.app(scope, receive, send)

        # Rutas: MCP SSE en /, Streamable HTTP en /mcp, API HTTP, health
        inner = Starlette(
            routes=[
                Route("/api/context", api_context, methods=["GET"]),
                Route("/api/memory", api_memory, methods=["POST"]),
                Route("/api/search", api_search, methods=["GET"]),
                Route("/recordatorio", recibir_recordatorio, methods=["POST"]),
                Route("/health", health, methods=["GET"]),
                Mount("/mcp", app=streamable_manager.handle_request),
                Mount("/", app=mcp.sse_app())
            ],
            middleware=[Middleware(SecurityMiddleware)],
            lifespan=lifespan,
        )
        app = TrailingSlashMCP(inner)
        uvicorn.run(app, host=host, port=port)
    else:
        mcp.run()