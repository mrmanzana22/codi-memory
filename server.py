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
from datetime import datetime, timezone

# Import MCP server and shared config from modules
from modules.config import mcp, memory, qdrant, USER_ID, COLLECTION_NAME

# Import all modules with register_tools
from modules import triggers
from modules import books
from modules import memory_core
from modules import memory_smart
from modules import training
from modules import maintenance
from modules import flush
from modules import consciousness

# ============================================================
# REGISTER ALL TOOLS
# ============================================================

triggers.register_tools(mcp)
books.register_tools(mcp)
memory_core.register_tools(mcp)
memory_smart.register_tools(mcp)
training.register_tools(mcp)
maintenance.register_tools(mcp)
flush.register_tools(mcp)
consciousness.register_tools(mcp)

print(f"[codi-memory] All modules loaded. Tools registered.")


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

        # Endpoint para recibir recordatorios de n8n u otros sistemas
        async def recibir_recordatorio(request):
            try:
                body = await request.json()
                mensaje = body.get('mensaje', '')
                prioridad = body.get('prioridad', 'normal')
                origen = body.get('origen', 'externo')

                if not mensaje:
                    return JSONResponse({"error": "mensaje requerido"}, status_code=400)

                # Guardar recordatorio
                data = _cargar_recordatorios()
                data['pendientes'].append({
                    "mensaje": mensaje,
                    "prioridad": prioridad,
                    "origen": origen,
                    "timestamp": datetime.now().isoformat()
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
                return JSONResponse({"error": str(e)}, status_code=500)

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
                    "timestamp": datetime.now().isoformat()
                })

            except Exception as e:
                return JSONResponse({"error": str(e)}, status_code=500)

        async def api_memory(request):
            """POST /api/memory - Guarda una nueva memoria"""
            try:
                body = await request.json()
                content = body.get('content', '')
                category = body.get('category', 'general')
                source = body.get('source', 'experienced')
                importance = body.get('importance', 'medium')

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
                        'created_at': datetime.now(timezone.utc).isoformat()
                    }
                )

                return JSONResponse({
                    "status": "ok",
                    "message": f"Memoria guardada: {content[:50]}...",
                    "result": result
                })

            except Exception as e:
                return JSONResponse({"error": str(e)}, status_code=500)

        async def api_search(request):
            """GET /api/search?q=query&limit=5 - Busca memorias"""
            try:
                query = request.query_params.get('q', '')
                limit = int(request.query_params.get('limit', 5))

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
                return JSONResponse({"error": str(e)}, status_code=500)

        port = int(os.getenv("PORT", 8000))
        print(f"[codi-memory] Starting MCP server on {transport} transport, port {port}")

        # Rutas: MCP en /, API HTTP para n8n, recordatorios, health
        app = Starlette(routes=[
            Route("/api/context", api_context, methods=["GET"]),
            Route("/api/memory", api_memory, methods=["POST"]),
            Route("/api/search", api_search, methods=["GET"]),
            Route("/recordatorio", recibir_recordatorio, methods=["POST"]),
            Route("/health", health, methods=["GET"]),
            Mount("/", app=mcp.sse_app())
        ])
        uvicorn.run(app, host="0.0.0.0", port=port)
    else:
        mcp.run()
