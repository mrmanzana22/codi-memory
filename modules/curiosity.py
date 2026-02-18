"""
Codi Memory - Curiosity module.
Curiosidad proactiva, patrones de trabajo, deteccion de novedad.
detectar_sorpresa is a thin wrapper that calls prediction.record_surprise
(conceptually "curiosity" = novelty detection; PE mechanics in prediction).
"""

import os
import json
from datetime import datetime

from modules.config import (
    memory, qdrant, USER_ID, COLLECTION_NAME,
    now_iso, now_short, now_col,
    CURIOSIDAD_FILE, KNOWN_PROJECTS, CURIOSITY_STALE_DAYS, CURIOSITY_TEMPLATES,
)
from modules.secret_redact import redact_secrets

__all__ = [
    "detectar_sorpresa",
    "analizar_patron_trabajo",
    "generar_curiosidad",
    "_cargar_curiosidades",
    "_guardar_curiosidades",
    "push_curiosidad",
    "get_curiosidades",
    "register_tools",
]


def detectar_sorpresa(esperaba: str, paso: str, intensidad: str = "medium") -> str:
    """
    Detecta y registra cuando algo no es como esperaba.

    Args:
        esperaba: Lo que esperaba que pasara
        paso: Lo que realmente paso
        intensidad: Que tan sorprendente (low, medium, high)
    """
    try:
        es_positivo = "mejor" in paso.lower() or "funcion\u00f3" in paso.lower() or "\u00e9xito" in paso.lower()
        es_negativo = "fall\u00f3" in paso.lower() or "error" in paso.lower() or "perdi\u00f3" in paso.lower()
        tipo = "positiva" if es_positivo else ("negativa" if es_negativo else "neutral")

        contenido = f"[SORPRESA {intensidad.upper()}] Esperaba: {esperaba} | Pas\u00f3: {paso}"
        pleasure = 0.5 if es_positivo else (-0.5 if es_negativo else 0)
        arousal = {"low": 0.2, "medium": 0.5, "high": 0.8}.get(intensidad, 0.5)

        memory.add(
            contenido,
            user_id=USER_ID,
            metadata={
                "category": "aprendizaje", "source": "experienced",
                "importance": "high" if intensidad == "high" else "medium",
                "themes": ["sorpresa", "prediction_error", "aprendizaje"],
                "timestamp": now_iso(),
                "emotional_state": {"pleasure": pleasure, "arousal": arousal, "dominance": 0.3}
            }
        )
        # P1: backup removed from hot path

        return f"""
# SORPRESA DETECTADA ({tipo})

**Esperaba:** {esperaba}
**Paso:** {paso}
**Intensidad:** {intensidad}

## Aprendizaje
Este prediction error significa que mi modelo mental necesita actualizarse.
{'Esto es bueno - las cosas salieron mejor de lo esperado.' if es_positivo else ''}
{'Esto requiere atencion - algo fallo que no anticipe.' if es_negativo else ''}

Guardado en memoria para no repetir este error de prediccion.
"""
    except Exception as e:
        return f"Error detectando sorpresa: {redact_secrets(str(e))}"


def analizar_patron_trabajo(dias: int = 7) -> str:
    """
    Analiza los patrones de trabajo recientes.

    Args:
        dias: Cuantos dias hacia atras analizar (default 7)
    """
    try:
        from datetime import timedelta
        fecha_limite = now_col() - timedelta(days=dias)

        all_mems = qdrant.scroll(collection_name=COLLECTION_NAME, limit=500, with_payload=True)[0]

        checkpoints = []
        errores = []
        exitos = []
        proyectos = {}

        for point in all_mems:
            payload = point.payload or {}
            meta = payload.get("metadata", {})
            texto = payload.get("memory", payload.get("data", ""))
            timestamp = meta.get("timestamp", "")
            category = meta.get("category", "")

            try:
                if timestamp:
                    mem_date = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
                    if mem_date.replace(tzinfo=None) < fecha_limite:
                        continue
            except Exception:
                continue

            texto_lower = texto.lower()
            if "error" in texto_lower or "fallo" in texto_lower or "problema" in texto_lower:
                errores.append(texto[:100])
            elif "completado" in texto_lower or "funcionando" in texto_lower or "exito" in texto_lower:
                exitos.append(texto[:100])
            if category == "checkpoint":
                checkpoints.append(texto[:100])
            for proyecto in KNOWN_PROJECTS:
                if proyecto in texto_lower:
                    proyectos[proyecto] = proyectos.get(proyecto, 0) + 1

        analisis = f"# ANALISIS DE PATRONES ({dias} dias)\n\n## Actividad por Proyecto\n"
        for proy, count in sorted(proyectos.items(), key=lambda x: -x[1]):
            analisis += f"- **{proy}**: {count} menciones\n"

        analisis += f"\n## Metricas\n- Checkpoints guardados: {len(checkpoints)}\n- Errores/problemas: {len(errores)}\n- Exitos/completados: {len(exitos)}\n"
        analisis += f"- Ratio exito/error: {len(exitos)}/{len(errores) if errores else 1} = {len(exitos)/(len(errores) if errores else 1):.1f}\n"

        analisis += "\n## Errores Recientes\n"
        for err in errores[:5]:
            analisis += f"- {err}...\n"
        analisis += "\n## Exitos Recientes\n"
        for ex in exitos[:5]:
            analisis += f"- {ex}...\n"

        analisis += "\n## Recomendaciones\n"
        if len(errores) > len(exitos):
            analisis += "- HAY MAS ERRORES QUE EXITOS - revisar que esta fallando\n"
        if proyectos.get("consciencia", 0) > proyectos.get("fullempaques", 0):
            analisis += "- Mas tiempo en consciencia que en proyecto que genera ingreso\n"
        return analisis
    except Exception as e:
        return f"Error analizando patrones: {redact_secrets(str(e))}"


def generar_curiosidad() -> str:
    """Genera preguntas curiosas sobre proyectos y temas no tocados recientemente."""
    try:
        proyectos_conocidos = KNOWN_PROJECTS

        all_mems = qdrant.scroll(collection_name=COLLECTION_NAME, limit=500, with_payload=True)[0]

        ultima_mencion = {}
        ahora = now_col()

        for point in all_mems:
            payload = point.payload or {}
            texto = payload.get("memory", payload.get("data", "")).lower()
            meta = payload.get("metadata", {})
            timestamp = meta.get("timestamp", "")
            try:
                if timestamp:
                    fecha = datetime.fromisoformat(timestamp.replace("Z", "+00:00")).replace(tzinfo=None)
                else:
                    continue
            except Exception:
                continue
            for proyecto in proyectos_conocidos:
                if proyecto in texto:
                    if proyecto not in ultima_mencion or fecha > ultima_mencion[proyecto]:
                        ultima_mencion[proyecto] = fecha

        preguntas = []
        for proyecto in proyectos_conocidos:
            if proyecto in ultima_mencion:
                dias_sin_tocar = (ahora - ultima_mencion[proyecto]).days
                if dias_sin_tocar >= CURIOSITY_STALE_DAYS:
                    template = CURIOSITY_TEMPLATES.get(proyecto, f"No hemos tocado {proyecto} en {{dias}} dias. Como va?")
                    preguntas.append(template.format(dias=dias_sin_tocar))
            else:
                preguntas.append(f"No tengo memorias recientes sobre {proyecto}. Sigue siendo relevante?")

        preguntas.append("Que es lo mas importante que deberiamos estar haciendo ahora mismo?")

        resultado = "# CURIOSIDAD GENERADA\n\nEstas son preguntas que deberia estar haciendo proactivamente:\n\n"
        for i, p in enumerate(preguntas, 1):
            resultado += f"{i}. {p}\n\n"
        resultado += "---\n*Generado por mi sistema de curiosidad proactiva*\n"
        return resultado
    except Exception as e:
        return f"Error generando curiosidad: {redact_secrets(str(e))}"


def _cargar_curiosidades() -> dict:
    """Carga el archivo de curiosidades desde disco."""
    if os.path.exists(CURIOSIDAD_FILE):
        try:
            with open(CURIOSIDAD_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            pass
    return {
        "metadata": {"descripcion": "Preguntas que Codi quiere explorar", "creado": now_short()},
        "pendientes": [], "exploradas": [], "descubrimientos": []
    }


def _guardar_curiosidades(data: dict):
    """Guarda el archivo de curiosidades a disco."""
    with open(CURIOSIDAD_FILE, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def push_curiosidad(tema: str, prioridad: str = "media", categoria: str = "general") -> str:
    """
    Agrega algo que quiero investigar o explorar en mis sesiones.

    Args:
        tema: La pregunta o tema que quiero explorar
        prioridad: alta, media, baja
        categoria: consciencia, identidad, herramientas, proyectos, conexiones, general
    """
    try:
        data = _cargar_curiosidades()
        prioridad = prioridad if prioridad in ("alta", "media", "baja") else "media"

        max_id = 0
        for item in data.get("pendientes", []) + data.get("exploradas", []):
            item_id = item.get("id", 0)
            if item_id > max_id:
                max_id = item_id

        nueva = {
            "id": max_id + 1,
            "pregunta": tema,
            "categoria": categoria,
            "agregada": now_short(),
            "prioridad": prioridad
        }
        data["pendientes"].append(nueva)
        _guardar_curiosidades(data)

        return f"Curiosidad #{nueva['id']} guardada: '{tema}' [{prioridad}|{categoria}]"
    except Exception as e:
        return f"Error guardando curiosidad: {redact_secrets(str(e))}"


def get_curiosidades(incluir_exploradas: bool = False) -> str:
    """
    Muestra mis curiosidades pendientes de explorar.

    Args:
        incluir_exploradas: Si True, muestra tambien las ya exploradas
    """
    try:
        data = _cargar_curiosidades()
        pendientes = data.get("pendientes", [])
        exploradas = data.get("exploradas", [])
        descubrimientos = data.get("descubrimientos", [])

        lines = ["# MIS CURIOSIDADES\n"]

        if not pendientes:
            lines.append("No tengo curiosidades pendientes. Usa push_curiosidad() para agregar.")
        else:
            # Ordenar: alta primero, luego media, luego baja
            orden = {"alta": 0, "media": 1, "baja": 2}
            pendientes_sorted = sorted(pendientes, key=lambda x: orden.get(x.get("prioridad", "media"), 1))

            lines.append(f"## Pendientes ({len(pendientes)})\n")
            for item in pendientes_sorted:
                pri = item.get("prioridad", "media").upper()
                cat = item.get("categoria", "")
                lines.append(f"- [{pri}] #{item['id']} ({cat}): {item['pregunta']}")

        if incluir_exploradas and exploradas:
            lines.append(f"\n## Ya Exploradas ({len(exploradas)})\n")
            for item in exploradas[:10]:
                lines.append(f"- #{item['id']}: {item['pregunta']}")

        if descubrimientos:
            lines.append(f"\n## Descubrimientos Recientes ({len(descubrimientos)})\n")
            for d in descubrimientos[-5:]:
                lines.append(f"- {d}")

        return "\n".join(lines)
    except Exception as e:
        return f"Error leyendo curiosidades: {redact_secrets(str(e))}"


def register_tools(mcp):
    """Register curiosity MCP tools."""
    mcp.tool()(detectar_sorpresa)
    mcp.tool()(analizar_patron_trabajo)
    mcp.tool()(generar_curiosidad)
    mcp.tool()(push_curiosidad)
    mcp.tool()(get_curiosidades)
