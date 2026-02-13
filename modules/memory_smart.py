"""
Codi Memory - Smart Memory (FTS5 + Deduplication)
FTS5 functions and add_memory_smart tool.
"""

import os
import json
import sqlite3


from datetime import datetime

from modules.config import memory, qdrant, USER_ID, COLLECTION_NAME, BACKUP_FILE, FTS_DB_PATH, now_iso

def _fts_conn():
    conn = sqlite3.connect(FTS_DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA busy_timeout=3000")
    return conn

from modules.utils import enrich_with_ownership
from modules.events import event_bus, Events


# ============================================================
# FTS5 FUNCTIONS
# ============================================================

def init_fts_db():
    """Inicializa la base de datos SQLite con FTS5 para busqueda por keywords."""
    conn = _fts_conn()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS memories_text (
            memory_id TEXT PRIMARY KEY,
            content TEXT NOT NULL,
            category TEXT DEFAULT 'general',
            source TEXT DEFAULT 'experienced',
            importance TEXT DEFAULT 'medium',
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.execute("""
        CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts
        USING fts5(
            content,
            memory_id UNINDEXED,
            category UNINDEXED,
            source UNINDEXED,
            content=memories_text,
            content_rowid=rowid
        )
    """)
    # Triggers para mantener FTS sincronizado
    conn.execute("""
        CREATE TRIGGER IF NOT EXISTS memories_text_ai AFTER INSERT ON memories_text BEGIN
            INSERT INTO memories_fts(rowid, content, memory_id, category, source)
            VALUES (new.rowid, new.content, new.memory_id, new.category, new.source);
        END
    """)
    conn.execute("""
        CREATE TRIGGER IF NOT EXISTS memories_text_ad AFTER DELETE ON memories_text BEGIN
            INSERT INTO memories_fts(memories_fts, rowid, content, memory_id, category, source)
            VALUES('delete', old.rowid, old.content, old.memory_id, old.category, old.source);
        END
    """)
    conn.execute("""
        CREATE TRIGGER IF NOT EXISTS memories_text_au AFTER UPDATE ON memories_text BEGIN
            INSERT INTO memories_fts(memories_fts, rowid, content, memory_id, category, source)
            VALUES('delete', old.rowid, old.content, old.memory_id, old.category, old.source);
            INSERT INTO memories_fts(rowid, content, memory_id, category, source)
            VALUES (new.rowid, new.content, new.memory_id, new.category, new.source);
        END
    """)
    # Retry queue for FTS consistency (P2A)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS fts_retry_queue (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            memory_id TEXT NOT NULL,
            op TEXT NOT NULL,
            payload_json TEXT,
            status TEXT DEFAULT 'pending',
            attempts INTEGER DEFAULT 0,
            last_error TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
    """)
    conn.execute("""
        CREATE UNIQUE INDEX IF NOT EXISTS uq_fts_retry_mem_op
        ON fts_retry_queue(memory_id, op)
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_fts_retry_status
        ON fts_retry_queue(status, updated_at)
    """)
    conn.commit()
    conn.close()
    print("[codi-memory] FTS5 index initialized")


def _index_memory_fts_raw(memory_id: str, content: str, category: str = "general",
                          source: str = "experienced", importance: str = "medium"):
    """Raw FTS insert. Triggers sync FTS5 automatically. Raises on failure."""
    conn = _fts_conn()
    try:
        conn.execute("""
            INSERT OR REPLACE INTO memories_text (memory_id, content, category, source, importance, created_at)
            VALUES (?, ?, ?, ?, ?, datetime('now'))
        """, (memory_id, content, category, source, importance))
        conn.commit()
    finally:
        conn.close()


def _delete_memory_fts_raw(memory_id: str):
    """Raw FTS delete. Triggers sync FTS5 automatically. Raises on failure."""
    conn = _fts_conn()
    try:
        conn.execute("DELETE FROM memories_text WHERE memory_id = ?", (memory_id,))
        conn.commit()
    finally:
        conn.close()


def index_memory_fts(memory_id: str, content: str, category: str = "general",
                     source: str = "experienced", importance: str = "medium") -> bool:
    """Safe FTS index: tries raw insert, queues retry on failure."""
    try:
        _index_memory_fts_raw(memory_id, content, category, source, importance)
        return True
    except Exception as e:
        queue_fts_op(
            memory_id=memory_id,
            op="upsert",
            payload={"content": content, "category": category, "source": source, "importance": importance},
            error=f"index_memory_fts failed: {e}",
        )
        return False


def delete_memory_fts(memory_id: str) -> bool:
    """Safe FTS delete: tries raw delete, queues retry on failure."""
    try:
        _delete_memory_fts_raw(memory_id)
        return True
    except Exception as e:
        queue_fts_op(
            memory_id=memory_id,
            op="delete",
            payload=None,
            error=f"delete_memory_fts failed: {e}",
        )
        return False


# ============================================================
# FTS RETRY QUEUE (P2A)
# ============================================================

def queue_fts_op(memory_id: str, op: str, payload: dict = None, error: str = None) -> dict:
    """Enqueue a failed FTS operation for retry. Idempotent by (memory_id, op)."""
    op = (op or "").strip().lower()
    if op not in ("upsert", "delete"):
        return {"ok": False, "error": f"invalid op: {op}"}

    now = now_iso()
    payload_json = json.dumps(payload, ensure_ascii=False) if payload else None
    error_short = (str(error) or "")[:500] if error else None

    conn = _fts_conn()
    try:
        conn.execute("""
            INSERT INTO fts_retry_queue
                (memory_id, op, payload_json, status, attempts, last_error, created_at, updated_at)
            VALUES (?, ?, ?, 'pending', 0, ?, ?, ?)
            ON CONFLICT(memory_id, op) DO UPDATE SET
                payload_json = COALESCE(excluded.payload_json, payload_json),
                status = 'pending',
                last_error = COALESCE(excluded.last_error, last_error),
                updated_at = excluded.updated_at
        """, (memory_id, op, payload_json, error_short, now, now))
        conn.commit()
        return {"ok": True, "memory_id": memory_id, "op": op}
    except Exception as e:
        print(f"[codi-memory] Error enqueuing FTS op: {e}")
        return {"ok": False, "error": str(e)}
    finally:
        conn.close()


def process_fts_queue(limit: int = 50, max_attempts: int = 10) -> dict:
    """Process pending FTS retry queue items. Idempotent and safe."""
    limit = max(1, min(500, int(limit or 50)))
    max_attempts = max(1, min(50, int(max_attempts or 10)))
    now = now_iso()

    conn = _fts_conn()
    try:
        rows = conn.execute("""
            SELECT id, memory_id, op, payload_json, attempts
            FROM fts_retry_queue
            WHERE status = 'pending' AND attempts < ?
            ORDER BY updated_at ASC
            LIMIT ?
        """, (max_attempts, limit)).fetchall()

        processed = 0
        succeeded = 0
        failed = 0

        for row in rows:
            qid, mem_id, op, payload_json, attempts = row
            processed += 1

            try:
                if op == "upsert":
                    payload = json.loads(payload_json) if payload_json else {}
                    content = payload.get("content", "")
                    if not content:
                        raise ValueError("missing content for upsert")
                    _index_memory_fts_raw(
                        mem_id,
                        content=content,
                        category=payload.get("category", "general"),
                        source=payload.get("source", "experienced"),
                        importance=payload.get("importance", "medium"),
                    )
                elif op == "delete":
                    _delete_memory_fts_raw(mem_id)

                conn.execute("""
                    UPDATE fts_retry_queue
                    SET status = 'done', updated_at = ?, last_error = NULL
                    WHERE id = ?
                """, (now, qid))
                conn.commit()
                succeeded += 1

            except Exception as e:
                new_attempts = (attempts or 0) + 1
                new_status = "failed" if new_attempts >= max_attempts else "pending"
                conn.execute("""
                    UPDATE fts_retry_queue
                    SET attempts = ?, last_error = ?, status = ?, updated_at = ?
                    WHERE id = ?
                """, (new_attempts, str(e)[:500], new_status, now, qid))
                conn.commit()
                failed += 1

        pending = conn.execute(
            "SELECT COUNT(*) FROM fts_retry_queue WHERE status = 'pending'"
        ).fetchone()[0]

        return {
            "ok": True,
            "processed": processed,
            "succeeded": succeeded,
            "failed": failed,
            "remaining_pending": pending,
        }
    except Exception as e:
        return {"ok": False, "error": str(e)}
    finally:
        conn.close()


def fts_queue_stats() -> dict:
    """Get FTS retry queue statistics."""
    conn = _fts_conn()
    try:
        pending = conn.execute("SELECT COUNT(*) FROM fts_retry_queue WHERE status='pending'").fetchone()[0]
        failed = conn.execute("SELECT COUNT(*) FROM fts_retry_queue WHERE status='failed'").fetchone()[0]
        done = conn.execute("SELECT COUNT(*) FROM fts_retry_queue WHERE status='done'").fetchone()[0]
        max_att = conn.execute("SELECT COALESCE(MAX(attempts), 0) FROM fts_retry_queue").fetchone()[0]
        oldest = conn.execute("""
            SELECT created_at FROM fts_retry_queue
            WHERE status = 'pending'
            ORDER BY created_at ASC LIMIT 1
        """).fetchone()
        return {
            "ok": True,
            "pending": pending,
            "failed": failed,
            "done": done,
            "max_attempts": int(max_att or 0),
            "oldest_pending": oldest[0] if oldest else None,
        }
    except Exception as e:
        return {"ok": False, "error": str(e)}
    finally:
        conn.close()


def search_fts(query: str, limit: int = 20) -> list:
    """Busca en FTS5 usando BM25 ranking."""
    conn = _fts_conn()
    try:
        results = conn.execute("""
            SELECT memory_id, content, category, source, rank
            FROM memories_fts
            WHERE content MATCH ?
            ORDER BY rank
            LIMIT ?
        """, (query, limit)).fetchall()
        return [{"memory_id": r[0], "content": r[1], "category": r[2],
                 "source": r[3], "bm25_rank": r[4]} for r in results]
    except Exception:
        return []
    finally:
        conn.close()


def bm25_rank_to_score(rank: float) -> float:
    """Convierte BM25 rank a score 0-1. Rank 0 = score 1.0."""
    normalized = max(0, abs(rank)) if rank is not None else 999
    return 1 / (1 + normalized)


def sync_fts_from_backup():
    """Sincroniza todas las memorias existentes al indice FTS5."""
    if not os.path.exists(BACKUP_FILE):
        print("[codi-memory] No backup found for FTS sync")
        return "No backup found"
    try:
        with open(BACKUP_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)
        memories = data if isinstance(data, list) else data.get("memories", [])
        count = 0
        for mem in memories:
            memory_id = mem.get("id", str(count))
            content = mem.get("memory", "")
            category = mem.get("metadata", {}).get("category", "general") if isinstance(mem.get("metadata"), dict) else "general"
            source = "experienced"
            importance = "medium"
            if content:
                index_memory_fts(memory_id, content, category, source, importance)
                count += 1
        # Rebuild FTS index para asegurar consistencia despues de bulk insert
        try:
            conn = _fts_conn()
            conn.execute("INSERT INTO memories_fts(memories_fts) VALUES('rebuild')")
            conn.commit()
            conn.close()
        except Exception:
            pass
        print(f"[codi-memory] Synced {count} memories to FTS index")
        return f"Synced {count} memories to FTS index"
    except Exception as e:
        print(f"[codi-memory] Error syncing FTS: {e}")
        return f"Error syncing FTS: {e}"


# ============================================================
# MCP TOOL FUNCTIONS
# ============================================================

def add_memory_smart(content: str, category: str = "general",
                     source: str = "experienced", importance: str = "medium",
                     dedup_threshold: float = 0.90,
                     relate_threshold: float = 0.75) -> str:
    """
    Guarda memoria con deduplicacion inteligente.
    Basado en neurociencia: el cerebro consolida, no duplica.

    Args:
        content: El contenido a recordar
        category: Categoria (identidad, aprendizaje, episodio, proyecto, general)
        source: Como obtuve esta memoria (experienced, told, learned, inferred)
        importance: Importancia (critical, high, medium, low)
        dedup_threshold: Umbral para considerar duplicado (default 0.90)
        relate_threshold: Umbral para marcar como relacionada (default 0.75)

    Returns:
        Resultado de la operacion con explicacion
    """
    try:
        # 1. Buscar memorias similares
        similar_results = memory.search(query=content, user_id=USER_ID, limit=3)

        if similar_results and similar_results.get("results"):
            top_result = similar_results["results"][0]
            top_score = top_result.get("score", 0)
            top_id = top_result.get("id", "")
            top_text = top_result.get("memory", "")[:80]

            # 2. Decidir accion segun similitud
            if top_score > dedup_threshold:
                # DUPLICADO - no guardar
                return json.dumps({
                    "action": "skipped_duplicate",
                    "score": round(top_score, 3),
                    "existing_memory": top_text,
                    "existing_id": top_id,
                    "message": f"Memoria ya existe (similitud {top_score:.2f})"
                }, ensure_ascii=False)

            elif top_score > relate_threshold:
                # SIMILAR - guardar pero relacionar
                result = memory.add(
                    messages=[{"role": "user", "content": content}],
                    user_id=USER_ID,
                    metadata={"category": category}
                )

                new_id = None
                if result and result.get("results"):
                    for r in result["results"]:
                        new_id = r.get("id")
                        if new_id:
                            # Enriquecer con ownership + relacion
                            enrich_with_ownership(
                                memory_id=new_id,
                                category=category,
                                content=content,
                                source=source,
                                importance=importance
                            )
                            # Agregar metadata de relacion
                            try:
                                qdrant.set_payload(
                                    collection_name=COLLECTION_NAME,
                                    payload={
                                        'related_to': top_id,
                                        'relation_score': top_score,
                                        'relation_type': 'semantic_similar'
                                    },
                                    points=[new_id]
                                )
                            except Exception:
                                pass

                            # Mini spreading activation: activar vecinos de la nueva relacion
                            try:
                                from modules.spreading import _spread_activation
                                spread_seeds = [top_id]
                                if new_id:
                                    spread_seeds.append(new_id)
                                _spread_activation(spread_seeds, depth=1, factor=0.3, seed_boost=0.05)
                            except Exception:
                                pass

                # P1: backup removed from hot path

                # Indexar en FTS5 (safe: auto-queues on failure)
                if new_id:
                    index_memory_fts(new_id, content, category, source, importance)

                # Emit MEMORY_STORED event
                try:
                    event_bus.emit(Events.MEMORY_STORED, {
                        'memory_id': new_id,
                        'content': content[:200],
                        'category': category,
                        'source': source,
                        'importance': importance,
                        'action': 'saved_with_relation',
                        'related_to': top_id,
                    })
                except Exception:
                    pass

                return json.dumps({
                    "action": "saved_with_relation",
                    "new_id": new_id,
                    "related_to": top_id,
                    "score": round(top_score, 3),
                    "message": f"Memoria guardada y relacionada con existente (similitud {top_score:.2f})"
                }, ensure_ascii=False)

        # 3. NUEVA - guardar normal
        result = memory.add(
            messages=[{"role": "user", "content": content}],
            user_id=USER_ID,
            metadata={"category": category}
        )

        new_id = None
        if result and result.get("results"):
            for r in result["results"]:
                new_id = r.get("id")
                if new_id:
                    enrich_with_ownership(
                        memory_id=new_id,
                        category=category,
                        content=content,
                        source=source,
                        importance=importance
                    )

        # P1: backup removed from hot path

        # Indexar en FTS5 (safe: auto-queues on failure)
        if new_id:
            index_memory_fts(new_id, content, category, source, importance)

        # Emit MEMORY_STORED event
        try:
            event_bus.emit(Events.MEMORY_STORED, {
                'memory_id': new_id,
                'content': content[:200],
                'category': category,
                'source': source,
                'importance': importance,
                'action': 'saved_new',
            })
        except Exception:
            pass

        return json.dumps({
            "action": "saved_new",
            "new_id": new_id,
            "message": f"Nueva memoria guardada: {content[:50]}..."
        }, ensure_ascii=False)

    except Exception as e:
        return json.dumps({
            "action": "error",
            "message": f"Error: {str(e)}"
        }, ensure_ascii=False)


def sync_fts_index() -> str:
    """Resincroniza el indice FTS5 desde el backup JSON. Usar si el indice se desincroniza."""
    try:
        init_fts_db()
        result = sync_fts_from_backup()
        return result
    except Exception as e:
        return f"Error sincronizando FTS: {str(e)}"


# ============================================================
# REGISTER TOOLS
# ============================================================

def _process_fts_queue_now(limit: int = 100) -> str:
    """Procesa la cola de reintentos FTS pendientes. Usar si sospechas desincronizacion FTS/Qdrant.

    Args:
        limit: Maximo de items a procesar (default 100)
    """
    result = process_fts_queue(limit=limit)
    if result.get("ok"):
        lines = ["# FTS Retry Queue - Procesado"]
        lines.append(f"- Procesados: {result['processed']}")
        lines.append(f"- Exitosos: {result['succeeded']}")
        lines.append(f"- Fallidos: {result['failed']}")
        lines.append(f"- Pendientes restantes: {result['remaining_pending']}")
        return "\n".join(lines)
    return f"Error procesando cola FTS: {result.get('error')}"


def _get_fts_queue_stats() -> str:
    """Muestra estadisticas de la cola de reintentos FTS. Util para diagnosticar desincronizacion."""
    s = fts_queue_stats()
    if s.get("ok"):
        lines = ["# FTS Queue Stats"]
        lines.append(f"- Pendientes: {s['pending']}")
        lines.append(f"- Fallidos (max reintentos): {s['failed']}")
        lines.append(f"- Completados: {s['done']}")
        lines.append(f"- Max intentos en cola: {s['max_attempts']}")
        lines.append(f"- Pendiente mas antiguo: {s['oldest_pending'] or 'ninguno'}")
        if s['pending'] == 0 and s['failed'] == 0:
            lines.append("\n*FTS y Qdrant estan sincronizados*")
        elif s['pending'] > 0:
            lines.append(f"\n*Hay {s['pending']} operaciones pendientes. Usa process_fts_queue_now() para procesarlas.*")
        return "\n".join(lines)
    return f"Error obteniendo stats: {s.get('error')}"


def register_tools(mcp):
    """Registra las herramientas de memoria smart en el servidor MCP."""
    mcp.tool()(add_memory_smart)
    mcp.tool()(sync_fts_index)

    @mcp.tool()
    def process_fts_queue_now(limit: int = 100) -> str:
        """Procesa la cola de reintentos FTS pendientes. Usar si sospechas desincronizacion FTS/Qdrant."""
        return _process_fts_queue_now(limit=limit)

    @mcp.tool()
    def get_fts_queue_stats() -> str:
        """Muestra estadisticas de la cola de reintentos FTS. Util para diagnosticar desincronizacion."""
        return _get_fts_queue_stats()
