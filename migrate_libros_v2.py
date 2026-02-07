#!/usr/bin/env python3
"""
Migración v2: Usa Qdrant directo con OpenAI embeddings para evitar duplicados.
"""
import json
import os
import uuid
from datetime import datetime
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct
from openai import OpenAI

QDRANT_URL = "https://memorycodi-codi.lx6zon.easypanel.host:443"
COLLECTION_NAME = "codi_memories"
USER_ID = "hare"
LIBROS_FILE = "/Users/harecjimenez/codi-memory/libros.json"

qdrant = QdrantClient(url=QDRANT_URL, timeout=30)
openai_client = OpenAI()

def get_embedding(text: str) -> list:
    """Genera embedding con OpenAI."""
    response = openai_client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return response.data[0].embedding

def migrar():
    with open(LIBROS_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)

    libros = data.get('libros', {})
    print(f"Migrando {len(libros)} libros...")

    points = []

    for libro_key, libro in libros.items():
        print(f"\n[LIBRO] {libro['nombre']}")

        # Crear punto para el libro
        texto_libro = f"LIBRO: {libro['nombre']} - {libro.get('descripcion', '')}"
        libro_id = str(uuid.uuid4())

        points.append(PointStruct(
            id=libro_id,
            vector=get_embedding(texto_libro),
            payload={
                "memory": texto_libro,
                "user_id": USER_ID,
                "category": "libro",
                "libro_key": libro_key,
                "nombre": libro['nombre'],
                "descripcion": libro.get('descripcion', ''),
                "iniciado": libro.get('iniciado', datetime.now().strftime('%Y-%m-%d')),
                "estado": libro.get('estado', 'activo'),
                "siguiente_paso": libro.get('siguiente_paso'),
                "narrative_importance": "critical",
                "created_at": datetime.now().isoformat()
            }
        ))

        # Crear puntos para cada capítulo
        capitulos = libro.get('capitulos', [])
        print(f"  -> {len(capitulos)} capítulos")

        for cap in capitulos:
            texto_cap = f"[{libro['nombre']}] Capítulo {cap['numero']}: {cap['titulo']}. {cap['resumen']}"
            cap_id = str(uuid.uuid4())

            points.append(PointStruct(
                id=cap_id,
                vector=get_embedding(texto_cap),
                payload={
                    "memory": texto_cap,
                    "user_id": USER_ID,
                    "category": "capitulo",
                    "libro": libro_key,
                    "numero": cap['numero'],
                    "titulo": cap['titulo'],
                    "resumen": cap['resumen'],
                    "fecha": cap.get('fecha', ''),
                    "narrative_importance": "high",
                    "created_at": datetime.now().isoformat()
                }
            ))
            print(f"     Cap {cap['numero']}: {cap['titulo'][:40]}...")

    # Insertar todos los puntos de una vez
    print(f"\nInsertando {len(points)} puntos en Qdrant...")
    qdrant.upsert(
        collection_name=COLLECTION_NAME,
        points=points
    )

    libros_count = sum(1 for p in points if p.payload.get('category') == 'libro')
    caps_count = sum(1 for p in points if p.payload.get('category') == 'capitulo')

    print(f"\n{'='*50}")
    print(f"MIGRACIÓN COMPLETADA")
    print(f"  Libros: {libros_count}")
    print(f"  Capítulos: {caps_count}")
    print(f"{'='*50}")

if __name__ == "__main__":
    migrar()
