"""
Codi Memory - Smart Memory (FTS5 + Deduplication)
FTS5 functions and add_memory_smart tool.
"""

import os
import json
import sqlite3
from datetime import datetime

from modules.config import memory, qdrant, USER_ID, COLLECTION_NAME, BACKUP_FILE, FTS_DB_PATH
from modules.utils import enrich_with_ownership, save_backup_json


# ============================================================
# FTS5 FUNCTIONS
# ============================================================

def init_fts_db():
    """Inicializa la base de datos SQLite con FTS5 para busqueda por keywords."""
    conn = sqlite3.connect(FTS_DB_PATH)
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
    conn.commit()
    conn.close()
    print("[codi-memory] FTS5 index initialized")


def index_memory_fts(memory_id: str, content: str, category: str = "general",
                     source: str = "experienced", importance: str = "medium"):
    """Indexa una memoria en SQLite FTS5."""
    conn = sqlite3.connect(FTS_DB_PATH)
    conn.execute("""
        INSERT OR REPLACE INTO memories_text (memory_id, content, category, source, importance, created_at)
        VALUES (?, ?, ?, ?, ?, datetime('now'))
    """, (memory_id, content, category, source, importance))
    conn.commit()
    conn.close()


def search_fts(query: str, limit: int = 20) -> list:
    """Busca en FTS5 usando BM25 ranking."""
    conn = sqlite3.connect(FTS_DB_PATH)
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
            conn = sqlite3.connect(FTS_DB_PATH)
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
                            except:
                                pass

                save_backup_json()

                # Indexar en FTS5
                try:
                    if new_id:
                        index_memory_fts(new_id, content, category, source, importance)
                except Exception as fts_err:
                    print(f"[codi-memory] FTS index error in add_memory_smart (relation): {fts_err}")

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

        save_backup_json()

        # Indexar en FTS5
        try:
            if new_id:
                index_memory_fts(new_id, content, category, source, importance)
        except Exception as fts_err:
            print(f"[codi-memory] FTS index error in add_memory_smart (new): {fts_err}")

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

def register_tools(mcp):
    """Registra las herramientas de memoria smart en el servidor MCP."""
    mcp.tool()(add_memory_smart)
    mcp.tool()(sync_fts_index)
