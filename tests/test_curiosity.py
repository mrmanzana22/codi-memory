"""Tests for modules/curiosity.py.

Covers:
  - detectar_sorpresa (surprise detection)
  - _cargar_curiosidades / _guardar_curiosidades (file I/O)
  - push_curiosidad (add curiosity)
  - get_curiosidades (list curiosities)
"""

import json
import os
import pytest
from unittest.mock import patch, MagicMock


@pytest.fixture
def curiosity_file(tmp_path, monkeypatch):
    """Redirect curiosity file to tmp_path."""
    filepath = str(tmp_path / "preguntas_curiosidad.json")
    monkeypatch.setattr("modules.curiosity.CURIOSIDAD_FILE", filepath)
    return filepath


class TestDetectarSorpresa:
    """Tests for detectar_sorpresa."""

    def test_positive_surprise(self, patch_externals):
        from modules.curiosity import detectar_sorpresa
        result = detectar_sorpresa(
            esperaba="Que fallara el deploy",
            paso="Funcionó perfecto al primer intento",
            intensidad="high"
        )
        assert "SORPRESA DETECTADA" in result
        assert "positiva" in result
        assert "mejor de lo esperado" in result or "bueno" in result

    def test_negative_surprise(self, patch_externals):
        from modules.curiosity import detectar_sorpresa
        result = detectar_sorpresa(
            esperaba="Que todo funcione bien",
            paso="Error grave en produccion",
            intensidad="high"
        )
        assert "SORPRESA DETECTADA" in result
        assert "negativa" in result

    def test_neutral_surprise(self, patch_externals):
        from modules.curiosity import detectar_sorpresa
        result = detectar_sorpresa(
            esperaba="Un resultado normal",
            paso="Algo completamente diferente",
            intensidad="medium"
        )
        assert "SORPRESA DETECTADA" in result
        assert "neutral" in result

    def test_intensity_levels(self, patch_externals):
        from modules.curiosity import detectar_sorpresa
        for level in ["low", "medium", "high"]:
            result = detectar_sorpresa("A", "B", intensidad=level)
            assert level in result.lower()


class TestCuriosidadesFileIO:
    """Tests for _cargar_curiosidades / _guardar_curiosidades."""

    def test_load_nonexistent_returns_default(self, curiosity_file):
        from modules.curiosity import _cargar_curiosidades
        data = _cargar_curiosidades()
        assert "pendientes" in data
        assert "exploradas" in data
        assert "descubrimientos" in data

    def test_save_and_load_roundtrip(self, curiosity_file):
        from modules.curiosity import _cargar_curiosidades, _guardar_curiosidades
        data = _cargar_curiosidades()
        data["pendientes"].append({"id": 1, "pregunta": "Test?"})
        _guardar_curiosidades(data)
        loaded = _cargar_curiosidades()
        assert len(loaded["pendientes"]) == 1
        assert loaded["pendientes"][0]["pregunta"] == "Test?"


class TestPushCuriosidad:
    """Tests for push_curiosidad."""

    def test_push_basic(self, curiosity_file):
        from modules.curiosity import push_curiosidad
        result = push_curiosidad("Como funciona la memoria?")
        assert "Curiosidad #" in result
        assert "memoria" in result

    def test_push_with_priority(self, curiosity_file):
        from modules.curiosity import push_curiosidad
        result = push_curiosidad("Urgente!", prioridad="alta", categoria="consciencia")
        assert "alta" in result
        assert "consciencia" in result

    def test_push_invalid_priority_defaults_to_media(self, curiosity_file):
        from modules.curiosity import push_curiosidad
        result = push_curiosidad("Test", prioridad="invalid")
        assert "media" in result

    def test_push_increments_id(self, curiosity_file):
        from modules.curiosity import push_curiosidad
        r1 = push_curiosidad("First")
        r2 = push_curiosidad("Second")
        # Extract IDs from "Curiosidad #N"
        id1 = int(r1.split("#")[1].split(" ")[0])
        id2 = int(r2.split("#")[1].split(" ")[0])
        assert id2 > id1


class TestGetCuriosidades:
    """Tests for get_curiosidades."""

    def test_empty_returns_message(self, curiosity_file):
        from modules.curiosity import get_curiosidades
        result = get_curiosidades()
        assert "MIS CURIOSIDADES" in result
        assert "No tengo curiosidades" in result or "Pendientes" in result

    def test_shows_pushed_items(self, curiosity_file):
        from modules.curiosity import push_curiosidad, get_curiosidades
        push_curiosidad("Como funciona X?", prioridad="alta")
        push_curiosidad("Por que Y?", prioridad="baja")
        result = get_curiosidades()
        assert "funciona X" in result
        assert "ALTA" in result

    def test_sorted_by_priority(self, curiosity_file):
        from modules.curiosity import push_curiosidad, get_curiosidades
        push_curiosidad("Low priority", prioridad="baja")
        push_curiosidad("High priority", prioridad="alta")
        result = get_curiosidades()
        # ALTA should appear before BAJA
        alta_pos = result.find("ALTA")
        baja_pos = result.find("BAJA")
        assert alta_pos < baja_pos

    def test_include_exploradas(self, curiosity_file):
        from modules.curiosity import _cargar_curiosidades, _guardar_curiosidades, get_curiosidades
        data = _cargar_curiosidades()
        data["exploradas"].append({"id": 99, "pregunta": "Old question"})
        _guardar_curiosidades(data)
        result = get_curiosidades(incluir_exploradas=True)
        assert "Ya Exploradas" in result
        assert "Old question" in result
