"""Tests for modules/books.py.

Covers:
  - _cargar_libros_local / _guardar_libros (file I/O)
  - _cargar_libros_de_qdrant (mocked qdrant)
  - _cargar_libros (fallback logic)
  - MCP tools via register_tools (crear_libro, agregar_capitulo, etc.)
"""

import json
import os
import pytest
from unittest.mock import patch, MagicMock
from tests.conftest import FakeQdrantPoint


@pytest.fixture
def books_file(tmp_path, monkeypatch):
    """Redirect books file to tmp_path."""
    filepath = str(tmp_path / "libros.json")
    monkeypatch.setattr("modules.books.LIBROS_FILE", filepath)
    return filepath


class MockMCP:
    """Minimal mock MCP server that captures registered tools."""

    def __init__(self):
        self._tools = {}

    def tool(self):
        def decorator(fn):
            self._tools[fn.__name__] = fn
            return fn
        return decorator


class TestCargarLibrosLocal:
    """Tests for _cargar_libros_local and _guardar_libros."""

    def test_no_file_returns_empty(self, books_file):
        from modules.books import _cargar_libros_local
        data = _cargar_libros_local()
        assert data["libros"] == {}

    def test_roundtrip(self, books_file):
        from modules.books import _cargar_libros_local, _guardar_libros
        data = {"metadata": {}, "libros": {"test": {"nombre": "TEST", "capitulos": []}}}
        _guardar_libros(data)
        loaded = _cargar_libros_local()
        assert "test" in loaded["libros"]
        assert loaded["libros"]["test"]["nombre"] == "TEST"


class TestCargarLibrosDeQdrant:
    """Tests for _cargar_libros_de_qdrant."""

    def test_empty_qdrant_returns_empty(self, patch_externals, books_file):
        from modules.books import _cargar_libros_de_qdrant
        data = _cargar_libros_de_qdrant()
        assert data["libros"] == {}

    def test_with_libro_points(self, patch_externals, books_file):
        from modules.books import _cargar_libros_de_qdrant
        fake_pg = patch_externals["pg"]

        # Pre-populate FakePG with a libro point (pg migration: books use pg.scroll now)
        fake_pg.add_point("codi_memories", "lib-1", {
            'data': 'LIBRO: TRADING - Libro de trading',
            'category': 'libro',
            'libro_key': 'trading',
            'nombre': 'TRADING',
            'descripcion': 'Libro de trading',
            'iniciado': '2026-01-01',
            'estado': 'activo',
            'siguiente_paso': 'Siguiente paso',
        })

        data = _cargar_libros_de_qdrant()
        assert "trading" in data["libros"]
        assert data["libros"]["trading"]["nombre"] == "TRADING"


class TestCargarLibrosFallback:
    """Tests for _cargar_libros (Qdrant -> local fallback)."""

    def test_uses_local_when_qdrant_empty(self, patch_externals, books_file):
        from modules.books import _cargar_libros, _guardar_libros
        # Pre-populate local file
        _guardar_libros({
            "metadata": {},
            "libros": {"local-book": {"nombre": "LOCAL", "capitulos": []}}
        })
        data = _cargar_libros()
        assert "local-book" in data["libros"]


class TestMCPTools:
    """Tests for MCP tools registered via register_tools."""

    @pytest.fixture
    def mcp_tools(self, patch_externals, books_file):
        from modules.books import register_tools
        mcp = MockMCP()
        register_tools(mcp)
        return mcp._tools

    def test_listar_libros_empty(self, mcp_tools):
        result = mcp_tools["listar_libros"]()
        assert "No hay libros" in result

    def test_crear_libro(self, mcp_tools):
        result = mcp_tools["crear_libro"]("test-book", "A test book")
        assert "TEST-BOOK" in result
        assert "creado" in result.lower()

    def test_crear_libro_duplicate(self, mcp_tools):
        mcp_tools["crear_libro"]("dupe", "First")
        result = mcp_tools["crear_libro"]("dupe", "Second")
        assert "Ya existe" in result

    def test_listar_after_create(self, mcp_tools):
        mcp_tools["crear_libro"]("my-book", "My description")
        result = mcp_tools["listar_libros"]()
        assert "MY-BOOK" in result

    def test_agregar_capitulo(self, mcp_tools):
        mcp_tools["crear_libro"]("cap-test", "Capitulo test book")
        result = mcp_tools["agregar_capitulo"]("cap-test", "Intro", "This is the intro")
        assert "Cap 1" in result
        assert "Intro" in result

    def test_agregar_capitulo_nonexistent_book(self, mcp_tools):
        result = mcp_tools["agregar_capitulo"]("fake", "Title", "Content")
        assert "no existe" in result

    def test_ver_libro(self, mcp_tools):
        mcp_tools["crear_libro"]("ver-test", "View test")
        mcp_tools["agregar_capitulo"]("ver-test", "First Chapter", "Summary here")
        result = mcp_tools["ver_libro"]("ver-test")
        assert "VER-TEST" in result
        assert "First Chapter" in result
        assert "Summary here" in result

    def test_ver_libro_not_found(self, mcp_tools):
        result = mcp_tools["ver_libro"]("nonexistent")
        assert "no encontrado" in result

    def test_actualizar_siguiente_paso(self, mcp_tools):
        mcp_tools["crear_libro"]("paso-test", "Step test")
        result = mcp_tools["actualizar_siguiente_paso"]("paso-test", "Do next thing")
        assert "actualizado" in result
        assert "Do next thing" in result

    def test_buscar_conexiones_needs_two_books(self, mcp_tools):
        mcp_tools["crear_libro"]("solo", "Only one")
        result = mcp_tools["buscar_conexiones_entre_libros"]()
        assert "al menos 2" in result

    def test_buscar_conexiones_with_two_books(self, mcp_tools):
        mcp_tools["crear_libro"]("book-a", "Trading analysis")
        mcp_tools["crear_libro"]("book-b", "Market research")
        result = mcp_tools["buscar_conexiones_entre_libros"]()
        assert "CONEXIONES" in result
