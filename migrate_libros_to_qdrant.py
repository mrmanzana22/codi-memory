#!/usr/bin/env python3
"""
Script para migrar libros del JSON local a Qdrant.
Ejecutar una sola vez despues de actualizar server.py.
"""

import json
import os
from datetime import datetime
from mem0 import Memory
from dotenv import load_dotenv

load_dotenv()

# Configuracion
USER_ID = os.getenv("USER_ID", "hare")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
QDRANT_URL = "https://memorycodi-codi.lx6zon.easypanel.host:443"
COLLECTION_NAME = "codi_memories"
LIBROS_FILE = os.path.join(os.path.dirname(__file__), "libros.json")

# Configurar mem0
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

def migrate():
    """Migra libros del JSON a Qdrant."""

    # Cargar libros del JSON
    if not os.path.exists(LIBROS_FILE):
        print("No existe libros.json")
        return

    with open(LIBROS_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)

    libros = data.get('libros', {})
    print(f"Encontrados {len(libros)} libros para migrar")

    for libro_key, libro in libros.items():
        print(f"\n--- Migrando libro: {libro['nombre']} ---")

        # Guardar el libro
        result = memory.add(
            f"LIBRO: {libro['nombre']} - {libro.get('descripcion', '')}",
            user_id=USER_ID,
            metadata={
                'category': 'libro',
                'libro_key': libro_key,
                'nombre': libro['nombre'],
                'descripcion': libro.get('descripcion', ''),
                'iniciado': libro.get('iniciado', ''),
                'estado': libro.get('estado', 'activo'),
                'siguiente_paso': libro.get('siguiente_paso'),
                'narrative_importance': 'critical'
            }
        )
        print(f"  Libro guardado: {result}")

        # Guardar cada capitulo
        capitulos = libro.get('capitulos', [])
        for cap in capitulos:
            result = memory.add(
                f"[{libro['nombre']}] Capitulo {cap['numero']}: {cap['titulo']}. {cap.get('resumen', '')}",
                user_id=USER_ID,
                metadata={
                    'category': 'capitulo',
                    'libro': libro_key,
                    'numero': cap['numero'],
                    'titulo': cap['titulo'],
                    'resumen': cap.get('resumen', ''),
                    'fecha': cap.get('fecha', ''),
                    'narrative_importance': 'high'
                }
            )
            print(f"  Cap {cap['numero']}: {cap['titulo'][:30]}...")

    print("\n=== Migracion completada ===")

if __name__ == "__main__":
    migrate()
