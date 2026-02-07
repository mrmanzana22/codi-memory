#!/usr/bin/env python3
"""
Script para importar conocimiento del curso M5-95 a la memoria de Codi.
Usa mem0 con Qdrant como backend.
"""

import json
import os
from datetime import datetime
from mem0 import Memory

# Configuracion (misma que server.py)
QDRANT_URL = "https://memorycodi-codi.lx6zon.easypanel.host:443"
COLLECTION_NAME = "codi_memories"
USER_ID = "hare"
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")

config = {
    "llm": {
        "provider": "openai",
        "config": {
            "model": "gpt-4o-mini",
            "temperature": 0.1
        }
    },
    "embedder": {
        "provider": "openai",
        "config": {
            "model": "text-embedding-3-small"
        }
    },
    "vector_store": {
        "provider": "qdrant",
        "config": {
            "collection_name": COLLECTION_NAME,
            "url": QDRANT_URL,
        }
    },
    "version": "v1.1"
}

def load_knowledge():
    """Carga el archivo JSON con el conocimiento del M5-95"""
    with open("/Users/harecjimenez/codi-memory/m95_knowledge_import.json", "r", encoding="utf-8") as f:
        return json.load(f)

def import_memories():
    """Importa las memorias a Qdrant via mem0"""
    print("Inicializando mem0...")
    memory = Memory.from_config(config)

    print("Cargando conocimiento M5-95...")
    knowledge = load_knowledge()

    total = len(knowledge["memories"])
    imported = 0
    errors = 0

    print(f"Importando {total} memorias...")

    for i, mem in enumerate(knowledge["memories"], 1):
        content = mem["content"]
        category = mem.get("category", "aprendizaje")
        importance = mem.get("importance", "medium")
        tags = mem.get("tags", [])

        try:
            # Agregar metadata enriquecida
            metadata = {
                "category": category,
                "importance": importance,
                "tags": tags,
                "source": "m5-95-course",
                "imported_at": datetime.now().isoformat(),
                "ownership_is_mine": True,
                "ownership_source": mem.get("source", "learned"),
                "ownership_confidence": 0.95
            }

            result = memory.add(
                messages=[{"role": "user", "content": content}],
                user_id=USER_ID,
                metadata=metadata
            )

            imported += 1
            print(f"[{i}/{total}] OK: {content[:50]}...")

        except Exception as e:
            errors += 1
            print(f"[{i}/{total}] ERROR: {str(e)[:50]}...")

    print(f"\n--- RESUMEN ---")
    print(f"Total: {total}")
    print(f"Importadas: {imported}")
    print(f"Errores: {errors}")

    return imported, errors

if __name__ == "__main__":
    if not OPENAI_API_KEY:
        print("ERROR: Necesitas configurar OPENAI_API_KEY")
        print("Ejecuta: export OPENAI_API_KEY='tu-api-key'")
        exit(1)

    import_memories()
    print("\nImportacion completada!")
