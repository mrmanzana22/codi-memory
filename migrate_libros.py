#!/usr/bin/env python3
"""
Script para migrar libros desde libros.json a Qdrant.
"""
import json
import os
from datetime import datetime
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue, PointStruct
from mem0 import Memory
import uuid

# Configuración
QDRANT_URL = "https://memorycodi-codi.lx6zon.easypanel.host:443"
COLLECTION_NAME = "codi_memories"
USER_ID = "hare"
LIBROS_FILE = "/Users/harecjimenez/codi-memory/libros.json"

# Inicializar conexiones
qdrant = QdrantClient(url=QDRANT_URL, timeout=30)

config = {
    "version": "v1.1",
    "llm": {
        "provider": "openai",
        "config": {
            "model": "gpt-4o-mini",
            "temperature": 0.1,
        }
    },
    "embedder": {
        "provider": "openai",
        "config": {"model": "text-embedding-3-small"}
    },
    "vector_store": {
        "provider": "qdrant",
        "config": {
            "collection_name": COLLECTION_NAME,
            "url": QDRANT_URL,
        }
    }
}

memory = Memory.from_config(config)

def limpiar_libros_existentes():
    """Elimina libros y capítulos existentes en Qdrant."""
    print("Limpiando libros existentes en Qdrant...")

    # Buscar y eliminar libros
    libro_points, _ = qdrant.scroll(
        collection_name=COLLECTION_NAME,
        scroll_filter=Filter(
            must=[FieldCondition(key="category", match=MatchValue(value="libro"))]
        ),
        limit=100,
        with_payload=False,
        with_vectors=False
    )

    if libro_points:
        ids_libros = [p.id for p in libro_points]
        print(f"  Eliminando {len(ids_libros)} libros...")
        qdrant.delete(
            collection_name=COLLECTION_NAME,
            points_selector=ids_libros
        )

    # Buscar y eliminar capítulos
    cap_points, _ = qdrant.scroll(
        collection_name=COLLECTION_NAME,
        scroll_filter=Filter(
            must=[FieldCondition(key="category", match=MatchValue(value="capitulo"))]
        ),
        limit=500,
        with_payload=False,
        with_vectors=False
    )

    if cap_points:
        ids_caps = [p.id for p in cap_points]
        print(f"  Eliminando {len(ids_caps)} capítulos...")
        qdrant.delete(
            collection_name=COLLECTION_NAME,
            points_selector=ids_caps
        )

    print("  Limpieza completada.")

def migrar_libros():
    """Migra libros desde archivo local a Qdrant."""

    # Cargar libros del archivo
    with open(LIBROS_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)

    libros = data.get('libros', {})
    print(f"\nMigrando {len(libros)} libros a Qdrant...")

    total_caps = 0

    for libro_key, libro in libros.items():
        print(f"\n[LIBRO] {libro['nombre']}")

        # Crear memoria para el libro
        texto_libro = f"LIBRO: {libro['nombre']} - {libro.get('descripcion', '')}"

        result = memory.add(
            texto_libro,
            user_id=USER_ID,
            metadata={
                'category': 'libro',
                'libro_key': libro_key,
                'nombre': libro['nombre'],
                'descripcion': libro.get('descripcion', ''),
                'iniciado': libro.get('iniciado', datetime.now().strftime('%Y-%m-%d')),
                'estado': libro.get('estado', 'activo'),
                'siguiente_paso': libro.get('siguiente_paso'),
                'narrative_importance': 'critical'
            }
        )

        # Asegurar metadata en Qdrant
        if result and result.get("results"):
            for r in result["results"]:
                mem_id = r.get("id")
                if mem_id:
                    qdrant.set_payload(
                        collection_name=COLLECTION_NAME,
                        payload={
                            'category': 'libro',
                            'libro_key': libro_key,
                            'nombre': libro['nombre'],
                            'descripcion': libro.get('descripcion', ''),
                            'iniciado': libro.get('iniciado', ''),
                            'estado': libro.get('estado', 'activo'),
                            'siguiente_paso': libro.get('siguiente_paso'),
                            'narrative_importance': 'critical'
                        },
                        points=[mem_id]
                    )

        # Migrar capítulos
        capitulos = libro.get('capitulos', [])
        print(f"  -> {len(capitulos)} capítulos")

        for cap in capitulos:
            texto_cap = f"[{libro['nombre']}] Capítulo {cap['numero']}: {cap['titulo']}. {cap['resumen']}"

            cap_result = memory.add(
                texto_cap,
                user_id=USER_ID,
                metadata={
                    'category': 'capitulo',
                    'libro': libro_key,
                    'numero': cap['numero'],
                    'titulo': cap['titulo'],
                    'resumen': cap['resumen'],
                    'fecha': cap.get('fecha', ''),
                    'narrative_importance': 'high'
                }
            )

            # Asegurar metadata en Qdrant
            if cap_result and cap_result.get("results"):
                for r in cap_result["results"]:
                    mem_id = r.get("id")
                    if mem_id:
                        qdrant.set_payload(
                            collection_name=COLLECTION_NAME,
                            payload={
                                'category': 'capitulo',
                                'libro': libro_key,
                                'numero': cap['numero'],
                                'titulo': cap['titulo'],
                                'resumen': cap['resumen'],
                                'fecha': cap.get('fecha', ''),
                                'narrative_importance': 'high'
                            },
                            points=[mem_id]
                        )

            total_caps += 1
            print(f"     Cap {cap['numero']}: {cap['titulo'][:40]}...")

    print(f"\n{'='*50}")
    print(f"MIGRACIÓN COMPLETADA")
    print(f"  Libros: {len(libros)}")
    print(f"  Capítulos: {total_caps}")
    print(f"{'='*50}")

if __name__ == "__main__":
    limpiar_libros_existentes()
    migrar_libros()
