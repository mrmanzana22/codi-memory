"""
Codi Memory - Books/Library Module.
Manages Codi's knowledge books (libros de conocimiento) stored in Qdrant
with local JSON backup. Each book represents a topic/project with chapters.

Extracted from server.py - contains:
  Helper functions: _cargar_libros_de_qdrant, _cargar_libros_local,
                    _cargar_libros, _guardar_libros
  MCP tools: listar_libros, ver_libro, agregar_capitulo, crear_libro,
             actualizar_siguiente_paso, buscar_conexiones_entre_libros
"""

import os
import json
from datetime import datetime

from modules.config import memory, qdrant, USER_ID, COLLECTION_NAME, BASE_DIR

from qdrant_client.models import Filter, FieldCondition, MatchValue

# Archivo local como backup/fallback
LIBROS_FILE = os.path.join(BASE_DIR, "libros.json")


def _cargar_libros_de_qdrant():
    """Carga libros desde Qdrant buscando memorias con category=libro."""
    try:
        libros = {}

        # Buscar memorias que son libros
        libro_points, _ = qdrant.scroll(
            collection_name=COLLECTION_NAME,
            scroll_filter=Filter(
                must=[
                    FieldCondition(key="category", match=MatchValue(value="libro"))
                ]
            ),
            limit=100,
            with_payload=True,
            with_vectors=False
        )

        for point in libro_points:
            payload = point.payload
            libro_key = payload.get('libro_key', '').lower()
            if libro_key:
                libros[libro_key] = {
                    "nombre": payload.get('nombre', libro_key.upper()),
                    "descripcion": payload.get('descripcion', ''),
                    "iniciado": payload.get('iniciado', ''),
                    "estado": payload.get('estado', 'activo'),
                    "siguiente_paso": payload.get('siguiente_paso'),
                    "capitulos": [],
                    "memory_id": str(point.id)
                }

        # Buscar capitulos para cada libro
        for libro_key in libros:
            cap_points, _ = qdrant.scroll(
                collection_name=COLLECTION_NAME,
                scroll_filter=Filter(
                    must=[
                        FieldCondition(key="category", match=MatchValue(value="capitulo")),
                        FieldCondition(key="libro", match=MatchValue(value=libro_key))
                    ]
                ),
                limit=100,
                with_payload=True,
                with_vectors=False
            )

            capitulos = []
            for cap_point in cap_points:
                cap_payload = cap_point.payload
                capitulos.append({
                    "numero": cap_payload.get('numero', 0),
                    "titulo": cap_payload.get('titulo', ''),
                    "fecha": cap_payload.get('fecha', ''),
                    "resumen": cap_payload.get('resumen', ''),
                    "memory_id": str(cap_point.id)
                })

            # Ordenar por numero
            capitulos.sort(key=lambda x: x.get('numero', 0))
            libros[libro_key]['capitulos'] = capitulos

        return {"metadata": {"source": "qdrant"}, "libros": libros}

    except Exception as e:
        print(f"[libros] Error cargando de Qdrant: {e}")
        # Fallback a archivo local
        return _cargar_libros_local()

def _cargar_libros_local():
    """Fallback: Carga libros desde archivo local."""
    try:
        if os.path.exists(LIBROS_FILE):
            with open(LIBROS_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {"metadata": {}, "libros": {}}
    except Exception:
        return {"metadata": {}, "libros": {}}

def _cargar_libros():
    """Carga libros - primero intenta Qdrant, si no hay usa archivo local."""
    qdrant_data = _cargar_libros_de_qdrant()

    # Si Qdrant tiene libros, usarlos
    if qdrant_data.get('libros'):
        return qdrant_data

    # Fallback a archivo local si Qdrant no tiene libros
    local_data = _cargar_libros_local()
    if local_data.get('libros'):
        return local_data

    return {"metadata": {}, "libros": {}}

def _guardar_libros(data):
    """Guarda libros en archivo local como backup."""
    with open(LIBROS_FILE, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def register_tools(mcp):
    """Register all book/library MCP tools."""

    @mcp.tool()
    def listar_libros() -> str:
        """
        Lista todos los libros de conocimiento de Codi.
        Cada libro es un tema/proyecto importante con sus capitulos.

        Returns:
            Lista de libros con su estado
        """
        try:
            data = _cargar_libros()
            libros = data.get('libros', {})

            if not libros:
                return "No hay libros creados todavia."

            resultado = "# MIS LIBROS DE CONOCIMIENTO\n\n"

            for key, libro in libros.items():
                estado = libro.get('estado', 'activo')
                caps = len(libro.get('capitulos', []))
                emoji_estado = "activo" if estado == "activo" else "pausado" if estado == "pausado" else "completo"

                resultado += f"## {libro['nombre']} [{emoji_estado}]\n"
                resultado += f"- {libro.get('descripcion', '')}\n"
                resultado += f"- Capitulos: {caps}\n"
                if libro.get('siguiente_paso'):
                    resultado += f"- Siguiente: {libro['siguiente_paso']}\n"
                resultado += f"- Evocar: `{key}`\n\n"

            return resultado

        except Exception as e:
            return f"Error listando libros: {str(e)}"


    @mcp.tool()
    def ver_libro(nombre: str) -> str:
        """
        Ve el contenido de un libro especifico con todos sus capitulos.

        Args:
            nombre: Nombre del libro (ej: 'codi-consciencia', 'fullempaques')

        Returns:
            Indice completo del libro con capitulos
        """
        try:
            data = _cargar_libros()
            libro = data.get('libros', {}).get(nombre.lower())

            if not libro:
                disponibles = list(data.get('libros', {}).keys())
                return f"Libro '{nombre}' no encontrado. Disponibles: {disponibles}"

            resultado = f"# {libro['nombre']}\n\n"
            resultado += f"**{libro.get('descripcion', '')}**\n\n"
            resultado += f"- Estado: {libro.get('estado', 'activo')}\n"
            resultado += f"- Iniciado: {libro.get('iniciado', 'desconocido')}\n"
            if libro.get('siguiente_paso'):
                resultado += f"- Siguiente paso: {libro['siguiente_paso']}\n"

            capitulos = libro.get('capitulos', [])
            if capitulos:
                resultado += f"\n## Indice ({len(capitulos)} capitulos)\n\n"
                for cap in capitulos:
                    resultado += f"### Cap {cap['numero']}: {cap['titulo']}\n"
                    resultado += f"*{cap.get('fecha', '')}*\n\n"
                    resultado += f"{cap.get('resumen', '')}\n\n"
            else:
                resultado += "\n*Este libro no tiene capitulos todavia.*\n"

            return resultado

        except Exception as e:
            return f"Error viendo libro: {str(e)}"


    @mcp.tool()
    def agregar_capitulo(libro: str, titulo: str, resumen: str) -> str:
        """
        Agrega un nuevo capitulo a un libro.
        Usar cuando se complete una fase importante de un proyecto.

        Args:
            libro: Nombre del libro (ej: 'codi-consciencia')
            titulo: Titulo del capitulo
            resumen: Resumen de lo que paso/aprendimos

        Returns:
            Confirmacion del capitulo agregado
        """
        try:
            data = _cargar_libros()
            libro_key = libro.lower()

            if libro_key not in data.get('libros', {}):
                return f"Libro '{libro}' no existe. Usa crear_libro() primero."

            libro_data = data['libros'][libro_key]
            capitulos = libro_data.get('capitulos', [])
            nuevo_numero = len(capitulos) + 1
            fecha = datetime.now().strftime('%Y-%m-%d')

            # Guardar en Qdrant como memoria con category=capitulo
            result = memory.add(
                f"[{libro.upper()}] Capitulo {nuevo_numero}: {titulo}. {resumen}",
                user_id=USER_ID,
                metadata={
                    'category': 'capitulo',
                    'libro': libro_key,
                    'numero': nuevo_numero,
                    'titulo': titulo,
                    'resumen': resumen,
                    'fecha': fecha,
                    'narrative_importance': 'high'
                }
            )

            # Agregar metadata directamente en Qdrant para poder filtrar
            if result and result.get("results"):
                for r in result["results"]:
                    mem_id = r.get("id")
                    if mem_id:
                        qdrant.set_payload(
                            collection_name=COLLECTION_NAME,
                            payload={
                                'category': 'capitulo',
                                'libro': libro_key,
                                'numero': nuevo_numero,
                                'titulo': titulo,
                                'resumen': resumen,
                                'fecha': fecha,
                                'narrative_importance': 'high'
                            },
                            points=[mem_id]
                        )

            # También guardar en archivo local como backup
            nuevo_cap = {
                "numero": nuevo_numero,
                "titulo": titulo,
                "fecha": fecha,
                "resumen": resumen,
                "memorias_clave": []
            }
            capitulos.append(nuevo_cap)
            libro_data['capitulos'] = capitulos
            _guardar_libros(data)

            return f"""
Capitulo agregado a **{libro_data['nombre']}**:

**Cap {nuevo_numero}: {titulo}**
{resumen}

Total capitulos: {len(capitulos)}
Guardado en: Qdrant + backup local
"""

        except Exception as e:
            return f"Error agregando capitulo: {str(e)}"


    @mcp.tool()
    def crear_libro(nombre: str, descripcion: str) -> str:
        """
        Crea un nuevo libro de conocimiento para un tema/proyecto.

        Args:
            nombre: Nombre corto del libro (sin espacios, ej: 'nuevo-proyecto')
            descripcion: De que trata este libro

        Returns:
            Confirmacion del libro creado
        """
        try:
            data = _cargar_libros()
            nombre_key = nombre.lower().replace(' ', '-')

            if nombre_key in data.get('libros', {}):
                return f"Ya existe un libro llamado '{nombre_key}'"

            fecha_inicio = datetime.now().strftime('%Y-%m-%d')

            # Guardar en Qdrant como memoria
            result = memory.add(
                f"LIBRO: {nombre.upper()} - {descripcion}",
                user_id=USER_ID,
                metadata={
                    'category': 'libro',
                    'libro_key': nombre_key,
                    'nombre': nombre.upper(),
                    'descripcion': descripcion,
                    'iniciado': fecha_inicio,
                    'estado': 'activo',
                    'siguiente_paso': None,
                    'narrative_importance': 'critical'
                }
            )

            # Agregar metadata directamente en Qdrant para poder filtrar
            if result and result.get("results"):
                for r in result["results"]:
                    mem_id = r.get("id")
                    if mem_id:
                        qdrant.set_payload(
                            collection_name=COLLECTION_NAME,
                            payload={
                                'category': 'libro',
                                'libro_key': nombre_key,
                                'nombre': nombre.upper(),
                                'descripcion': descripcion,
                                'iniciado': fecha_inicio,
                                'estado': 'activo',
                                'siguiente_paso': None,
                                'narrative_importance': 'critical'
                            },
                            points=[mem_id]
                        )

            # También guardar en archivo local como backup
            data['libros'][nombre_key] = {
                "nombre": nombre.upper(),
                "descripcion": descripcion,
                "iniciado": fecha_inicio,
                "capitulos": [],
                "estado": "activo",
                "siguiente_paso": None
            }
            _guardar_libros(data)

            return f"""
Libro creado: **{nombre.upper()}**

- Clave: `{nombre_key}`
- Descripcion: {descripcion}
- Guardado en: Qdrant + backup local

Usa `agregar_capitulo('{nombre_key}', 'titulo', 'resumen')` para agregar contenido.
"""

        except Exception as e:
            return f"Error creando libro: {str(e)}"


    @mcp.tool()
    def actualizar_siguiente_paso(libro: str, siguiente: str) -> str:
        """
        Actualiza el siguiente paso de un libro/proyecto.

        Args:
            libro: Nombre del libro
            siguiente: Descripcion del siguiente paso

        Returns:
            Confirmacion
        """
        try:
            data = _cargar_libros()
            libro_key = libro.lower()

            if libro_key not in data.get('libros', {}):
                return f"Libro '{libro}' no existe."

            libro_data = data['libros'][libro_key]

            # Actualizar en Qdrant si tenemos el memory_id
            if libro_data.get('memory_id'):
                try:
                    qdrant.set_payload(
                        collection_name=COLLECTION_NAME,
                        payload={"payload": {"siguiente_paso": siguiente}},
                        points=[libro_data['memory_id']]
                    )
                except Exception as e:
                    print(f"[libros] Warning: no se pudo actualizar en Qdrant: {e}")

            # Actualizar en archivo local
            data['libros'][libro_key]['siguiente_paso'] = siguiente
            _guardar_libros(data)

            return f"Siguiente paso de {libro.upper()} actualizado: {siguiente}"

        except Exception as e:
            return f"Error: {str(e)}"


    @mcp.tool()
    def buscar_conexiones_entre_libros() -> str:
        """
        Busca conexiones y patrones entre diferentes libros/proyectos.
        Encuentra conocimiento de un contexto que puede aplicar a otro.
        Como el cerebro durante el sueno - conecta cosas que parecen no relacionadas.

        Returns:
            Conexiones encontradas entre libros
        """
        try:
            data = _cargar_libros()
            libros = data.get('libros', {})

            if len(libros) < 2:
                return "Necesito al menos 2 libros para buscar conexiones."

            resultado = "# CONEXIONES ENTRE LIBROS\n\n"
            conexiones_encontradas = []

            # Extraer palabras clave de cada libro
            keywords_por_libro = {}
            for nombre, libro in libros.items():
                keywords = set()
                keywords.update(libro.get('descripcion', '').lower().split())
                for cap in libro.get('capitulos', []):
                    keywords.update(cap.get('titulo', '').lower().split())
                    keywords.update(cap.get('resumen', '').lower().split())
                keywords = {k for k in keywords if len(k) > 4 and k not in
                           ['para', 'como', 'desde', 'hasta', 'entre', 'sobre', 'cuando', 'donde', 'tiene', 'hacer']}
                keywords_por_libro[nombre] = keywords

            # Buscar intersecciones entre libros
            libros_list = list(libros.keys())
            resultado += "## Conceptos compartidos\n\n"
            hay_conceptos = False
            for i, libro1 in enumerate(libros_list):
                for libro2 in libros_list[i+1:]:
                    comunes = keywords_por_libro[libro1] & keywords_por_libro[libro2]
                    if comunes:
                        hay_conceptos = True
                        resultado += f"**{libro1.upper()}** <-> **{libro2.upper()}**\n"
                        resultado += f"- Conceptos: {', '.join(list(comunes)[:5])}\n\n"
                        conexiones_encontradas.append((libro1, libro2, list(comunes)))

            if not hay_conceptos:
                resultado += "No encontre conceptos compartidos todavia.\n\n"

            # Conexiones potenciales basadas en patrones conocidos
            resultado += "## Conexiones potenciales\n\n"
            patrones = [
                ("trading", "automatizaciones", "Senales de trading pueden disparar workflows de n8n"),
                ("trading", "codi-consciencia", "Analisis de patrones del bot informa como analizo mis propios patrones de trabajo"),
                ("codi-consciencia", "fullempaques", "Sistema de checkpoints y seguimiento puede aplicarse a produccion"),
                ("automatizaciones", "fullempaques", "Workflows pueden automatizar reportes y alertas de produccion"),
                ("codi-consciencia", "automatizaciones", "Mi sistema de triggers es como un workflow interno"),
            ]

            for libro1, libro2, insight in patrones:
                if libro1 in libros and libro2 in libros:
                    resultado += f"- **{libro1}** -> **{libro2}**: {insight}\n"

            # Ideas para explorar
            resultado += "\n## Para explorar\n\n"
            resultado += "Preguntas que conectan dominios:\n"
            resultado += "- Que aprendi en un proyecto que no estoy aplicando en otro?\n"
            resultado += "- Que patron se repite en diferentes contextos?\n"
            resultado += "- Que error cometi en un lugar que podria estar cometiendo en otro?\n"

            return resultado

        except Exception as e:
            return f"Error buscando conexiones: {str(e)}"
