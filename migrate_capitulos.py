#!/usr/bin/env python3
"""
Script para migrar capítulos del JSON local a Qdrant.
Los libros ya existen en Qdrant, solo agregamos los capítulos.
"""

import json
import os
from mem0 import Memory
from qdrant_client import QdrantClient
from dotenv import load_dotenv

load_dotenv()

# Configuracion
USER_ID = os.getenv("USER_ID", "hare")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
QDRANT_URL = "https://memorycodi-codi.lx6zon.easypanel.host:443"
COLLECTION_NAME = "codi_memories"
LIBROS_FILE = os.path.join(os.path.dirname(__file__), "libros.json")

# Clientes
config = {
    "llm": {
        "provider": "openai",
        "config": {
            "model": "gpt-4o-mini",
            "temperature": 0.1,
            "api_key": OPENAI_API_KEY,
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

memory = Memory.from_config(config)
qdrant = QdrantClient(url=QDRANT_URL, timeout=30)

def migrate_capitulos():
    """Migra solo los capítulos del JSON a Qdrant."""

    # Cargar libros del JSON
    if not os.path.exists(LIBROS_FILE):
        print("No existe libros.json")
        return

    with open(LIBROS_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)

    libros = data.get('libros', {})
    total_caps = 0

    for libro_key, libro in libros.items():
        capitulos = libro.get('capitulos', [])
        if not capitulos:
            print(f"[{libro['nombre']}] Sin capítulos, saltando...")
            continue

        print(f"\n--- {libro['nombre']}: {len(capitulos)} capítulos ---")

        for cap in capitulos:
            numero = cap.get('numero', 0)
            titulo = cap.get('titulo', '')
            resumen = cap.get('resumen', '')
            fecha = cap.get('fecha', '')

            print(f"  Cap {numero}: {titulo[:40]}...")

            # Guardar con memory.add
            result = memory.add(
                f"[{libro['nombre']}] Capitulo {numero}: {titulo}. {resumen}",
                user_id=USER_ID,
                metadata={
                    'category': 'capitulo',
                    'libro': libro_key,
                    'numero': numero,
                    'titulo': titulo,
                    'resumen': resumen,
                    'fecha': fecha,
                    'narrative_importance': 'high'
                }
            )

            # Agregar metadata directamente en Qdrant
            if result and result.get("results"):
                for r in result["results"]:
                    mem_id = r.get("id")
                    if mem_id:
                        qdrant.set_payload(
                            collection_name=COLLECTION_NAME,
                            payload={
                                'category': 'capitulo',
                                'libro': libro_key,
                                'numero': numero,
                                'titulo': titulo,
                                'resumen': resumen,
                                'fecha': fecha,
                                'narrative_importance': 'high'
                            },
                            points=[mem_id]
                        )
                        print(f"    -> Guardado con metadata en Qdrant")

            total_caps += 1

    print(f"\n=== Migración completada: {total_caps} capítulos ===")

if __name__ == "__main__":
    migrate_capitulos()
