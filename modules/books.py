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

import logging
import os
import json
from datetime import datetime

_logger = logging.getLogger(__name__)

from modules.config import USER_ID, COLLECTION_NAME, BASE_DIR, now_col
from modules.pg_store import pg
from modules.secret_redact import redact_secrets

# Archivo local como backup/fallback
LIBROS_FILE = os.path.join(BASE_DIR, "libros.json")


def _normalizar_libro_key(nombre: str) -> str:
    """Canonical book key: lowercase + hyphens."""
    return nombre.lower().replace(' ', '-')


def _cargar_libros_de_qdrant():
    """Carga libros desde pg_store buscando memorias con category=libro."""
    try:
        libros = {}

        # Buscar memorias que son libros (paginated)
        libro_points = []
        _lb_offset = None
        while True:
            _batch, _lb_offset = pg.scroll(filters={"category": "libro"}, limit=100, is_semantic=False, offset=_lb_offset)
            if not _batch:
                break
            libro_points.extend(_batch)
            if _lb_offset is None:
                break

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

        # Buscar capitulos (paginated, once for all libros)
        all_cap_points = []
        _cap_offset = None
        while True:
            _batch, _cap_offset = pg.scroll(filters={"category": "capitulo"}, limit=100, is_semantic=False, offset=_cap_offset)
            if not _batch:
                break
            all_cap_points.extend(_batch)
            if _cap_offset is None:
                break

        for libro_key in libros:
            # Post-filter for this specific libro
            cap_points = [p for p in all_cap_points if p.payload.get('libro') == libro_key]

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

        return {"metadata": {"source": "pg_store"}, "libros": libros}

    except Exception as e:
        _logger.error("Error cargando de pg_store: %s", redact_secrets(str(e)))
        # Fallback a archivo local
        return _cargar_libros_local()

def _cargar_libros_local():
    """Fallback: Carga libros desde archivo local."""
    try:
        if os.path.exists(LIBROS_FILE):
            with open(LIBROS_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {"metadata": {}, "libros": {}}
    except FileNotFoundError:
        return {"metadata": {}, "libros": {}}
    except Exception as exc:
        _logger.exception("Failed to load local libros backup: %s", exc)
        return {"metadata": {"error": "local_backup_read_failed"}, "libros": {}}

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


# ============================================================
# BUSINESS LOGIC — returns strings/dicts, raises exceptions
# ============================================================

def _listar_libros_impl() -> str:
    """List all knowledge books. Returns markdown string."""
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


def _ver_libro_impl(nombre: str) -> str:
    """View a specific book with all chapters. Returns markdown string."""
    data = _cargar_libros()
    libro = data.get('libros', {}).get(_normalizar_libro_key(nombre))

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


def _agregar_capitulo_impl(libro: str, titulo: str, resumen: str) -> str:
    """Add a chapter to a book. Returns confirmation string."""
    data = _cargar_libros()
    libro_key = _normalizar_libro_key(libro)

    if libro_key not in data.get('libros', {}):
        return f"Libro '{libro}' no existe. Usa crear_libro() primero."

    libro_data = data['libros'][libro_key]
    capitulos = libro_data.get('capitulos', [])
    nuevo_numero = len(capitulos) + 1
    fecha = now_col().strftime('%Y-%m-%d')

    # Guardar en pg_store como memoria con category=capitulo
    result = pg.add(
        content=f"[{libro.upper()}] Capitulo {nuevo_numero}: {titulo}. {resumen}",
        category='capitulo',
        source='experienced',
        importance='high'
    )

    # Actualizar payload con metadata adicional
    if result and result.get("results"):
        mem_id = result["results"][0]["id"]
        if mem_id:
            pg.update_payload(mem_id, {
                'libro': libro_key,
                'numero': nuevo_numero,
                'titulo': titulo,
                'resumen': resumen,
                'fecha': fecha,
                'narrative_importance': 'high',
            })

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


def _crear_libro_impl(nombre: str, descripcion: str) -> str:
    """Create a new knowledge book. Returns confirmation string."""
    data = _cargar_libros()
    nombre_key = _normalizar_libro_key(nombre)

    if nombre_key in data.get('libros', {}):
        return f"Ya existe un libro llamado '{nombre_key}'"

    fecha_inicio = now_col().strftime('%Y-%m-%d')

    # Guardar en pg_store como memoria
    result = pg.add(
        content=f"LIBRO: {nombre.upper()} - {descripcion}",
        category='libro',
        source='experienced',
        importance='critical'
    )

    # Actualizar payload con metadata adicional
    if result and result.get("results"):
        mem_id = result["results"][0]["id"]
        if mem_id:
            pg.update_payload(mem_id, {
                'libro_key': nombre_key,
                'nombre': nombre.upper(),
                'descripcion': descripcion,
                'iniciado': fecha_inicio,
                'estado': 'activo',
                'siguiente_paso': None,
                'narrative_importance': 'critical',
            })

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


def _actualizar_siguiente_paso_impl(libro: str, siguiente: str) -> str:
    """Update next step for a book/project. Returns confirmation string."""
    data = _cargar_libros()
    libro_key = _normalizar_libro_key(libro)

    if libro_key not in data.get('libros', {}):
        return f"Libro '{libro}' no existe."

    libro_data = data['libros'][libro_key]

    # Actualizar en pg_store si tenemos el memory_id
    if libro_data.get('memory_id'):
        try:
            pg.update_payload(libro_data['memory_id'], {
                "siguiente_paso": siguiente,
            })
        except Exception as e:
            _logger.warning("No se pudo actualizar en pg_store: %s", redact_secrets(str(e)))

    # Actualizar en archivo local
    data['libros'][libro_key]['siguiente_paso'] = siguiente
    _guardar_libros(data)

    return f"Siguiente paso de {libro.upper()} actualizado: {siguiente}"


def _buscar_conexiones_entre_libros_impl() -> str:
    """Find connections between books/projects. Returns markdown string."""
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


# ============================================================
# MCP TRANSPORT — error wrapping, register_tools
# ============================================================

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
            return _listar_libros_impl()
        except Exception as e:
            return f"Error listando libros: {redact_secrets(str(e))}"

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
            return _ver_libro_impl(nombre)
        except Exception as e:
            return f"Error viendo libro: {redact_secrets(str(e))}"

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
            return _agregar_capitulo_impl(libro, titulo, resumen)
        except Exception as e:
            return f"Error agregando capitulo: {redact_secrets(str(e))}"

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
            return _crear_libro_impl(nombre, descripcion)
        except Exception as e:
            return f"Error creando libro: {redact_secrets(str(e))}"

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
            return _actualizar_siguiente_paso_impl(libro, siguiente)
        except Exception as e:
            return f"Error: {redact_secrets(str(e))}"

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
            return _buscar_conexiones_entre_libros_impl()
        except Exception as e:
            return f"Error buscando conexiones: {redact_secrets(str(e))}"
