import os
import json
from datetime import datetime, timedelta, timezone
from modules.config import memory, qdrant, USER_ID, COLLECTION_NAME, BASE_DIR, BACKUP_FILE, now_col, now_iso, now_short, TZ_COL
from qdrant_client.models import Filter, FieldCondition, MatchValue, Range
from modules.utils import calculate_confidence_score
from modules.secret_redact import redact_secrets
from modules.access_tracking import record_access


# ============================================================
# MANTENIMIENTO PERIODICO
# ============================================================

MANTENIMIENTO_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), "mantenimiento.json")


def _cargar_mantenimiento():
    """Carga tareas de mantenimiento desde archivo."""
    try:
        if os.path.exists(MANTENIMIENTO_FILE):
            with open(MANTENIMIENTO_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {"metadata": {}, "tareas": []}
    except Exception:
        return {"metadata": {}, "tareas": []}


def _guardar_mantenimiento(data):
    """Guarda tareas de mantenimiento."""
    with open(MANTENIMIENTO_FILE, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def _verificar_tareas_vencidas():
    """Retorna tareas de mantenimiento vencidas o sin hacer."""
    from datetime import timedelta

    data = _cargar_mantenimiento()
    vencidas = []
    hoy = now_col()

    for tarea in data.get('tareas', []):
        if not tarea.get('activo', True):
            continue

        ultimo = tarea.get('ultimo_completado')
        frecuencia = tarea.get('frecuencia_dias', 7)

        if ultimo is None:
            # Nunca se ha hecho
            vencidas.append({
                'id': tarea['id'],
                'nombre': tarea['nombre'],
                'estado': 'nunca_hecho',
                'dias_vencido': None
            })
        else:
            ultimo_fecha = datetime.fromisoformat(ultimo)
            if ultimo_fecha.tzinfo is None:
                ultimo_fecha = ultimo_fecha.replace(tzinfo=TZ_COL)
            dias_pasados = (hoy - ultimo_fecha).days
            if dias_pasados >= frecuencia:
                vencidas.append({
                    'id': tarea['id'],
                    'nombre': tarea['nombre'],
                    'estado': 'vencido',
                    'dias_vencido': dias_pasados - frecuencia
                })

    return vencidas


def _registrar_mantenimiento(nombre: str, descripcion: str, frecuencia_dias: int = 7) -> str:
    """
    Registra una nueva tarea de mantenimiento periodico.
    Estas tareas aparecen en despertar_codi() cuando estan vencidas.

    Args:
        nombre: Nombre corto de la tarea
        descripcion: Que hay que hacer
        frecuencia_dias: Cada cuantos dias debe hacerse (default 7)

    Returns:
        Confirmacion de tarea registrada
    """
    try:
        data = _cargar_mantenimiento()

        # Generar ID
        task_id = nombre.lower().replace(' ', '_')[:30]

        # Verificar si ya existe
        for t in data['tareas']:
            if t['id'] == task_id:
                return f"Ya existe una tarea con ID '{task_id}'"

        nueva_tarea = {
            'id': task_id,
            'nombre': nombre,
            'descripcion': descripcion,
            'frecuencia_dias': frecuencia_dias,
            'ultimo_completado': None,
            'proximo': None,
            'activo': True
        }

        data['tareas'].append(nueva_tarea)
        _guardar_mantenimiento(data)

        return f"""
Tarea de mantenimiento registrada:

**{nombre}**
- ID: {task_id}
- Frecuencia: cada {frecuencia_dias} dias
- Descripcion: {descripcion}

Esta tarea aparecera en despertar_codi() cuando este vencida.
"""
    except Exception as e:
        return f"Error registrando tarea: {redact_secrets(str(e))}"


def _verificar_mantenimiento() -> str:
    """
    Verifica estado de todas las tareas de mantenimiento.
    Muestra cuales estan al dia y cuales vencidas.

    Returns:
        Reporte de estado de mantenimiento
    """
    try:
        data = _cargar_mantenimiento()
        hoy = now_col()

        resultado = "# ESTADO DE MANTENIMIENTO\n\n"

        vencidas = []
        al_dia = []

        for tarea in data.get('tareas', []):
            if not tarea.get('activo', True):
                continue

            ultimo = tarea.get('ultimo_completado')
            frecuencia = tarea.get('frecuencia_dias', 7)

            if ultimo is None:
                vencidas.append(f"- **{tarea['nombre']}**: NUNCA HECHO - {tarea['descripcion']}")
            else:
                ultimo_fecha = datetime.fromisoformat(ultimo)
                if ultimo_fecha.tzinfo is None:
                    ultimo_fecha = ultimo_fecha.replace(tzinfo=TZ_COL)
                dias_pasados = (hoy - ultimo_fecha).days
                dias_restantes = frecuencia - dias_pasados

                if dias_restantes <= 0:
                    vencidas.append(f"- **{tarea['nombre']}**: VENCIDO hace {-dias_restantes} dias")
                else:
                    al_dia.append(f"- {tarea['nombre']}: OK (proximo en {dias_restantes} dias)")

        if vencidas:
            resultado += "## PENDIENTES (hacer pronto)\n"
            resultado += "\n".join(vencidas)
            resultado += "\n\n"

        if al_dia:
            resultado += "## AL DIA\n"
            resultado += "\n".join(al_dia)

        if not vencidas and not al_dia:
            resultado += "No hay tareas de mantenimiento configuradas."

        return resultado

    except Exception as e:
        return f"Error verificando mantenimiento: {redact_secrets(str(e))}"


def _marcar_mantenimiento_hecho(tarea_id: str, notas: str = "") -> str:
    """
    Marca una tarea de mantenimiento como completada.
    Actualiza la fecha y calcula el proximo vencimiento.

    Args:
        tarea_id: ID de la tarea (ej: 'debug_sistema')
        notas: Notas opcionales sobre lo que se hizo

    Returns:
        Confirmacion con proxima fecha
    """
    try:
        data = _cargar_mantenimiento()
        hoy = now_col()

        for tarea in data['tareas']:
            if tarea['id'] == tarea_id:
                tarea['ultimo_completado'] = hoy.isoformat()
                frecuencia = tarea.get('frecuencia_dias', 7)
                proximo = hoy + timedelta(days=frecuencia)
                tarea['proximo'] = proximo.isoformat()

                _guardar_mantenimiento(data)

                # Guardar en memoria tambien
                memory.add(
                    f"Mantenimiento completado: {tarea['nombre']}. {notas}",
                    user_id=USER_ID,
                    metadata={
                        'category': 'mantenimiento',
                        'tarea_id': tarea_id,
                        'fecha': hoy.isoformat()
                    }
                )

                return f"""
Mantenimiento completado: **{tarea['nombre']}**

- Completado: {hoy.strftime('%Y-%m-%d %H:%M')}
- Proximo: {proximo.strftime('%Y-%m-%d')} (en {frecuencia} dias)
{f'- Notas: {notas}' if notas else ''}
"""

        return f"No encontre tarea con ID '{tarea_id}'"

    except Exception as e:
        return f"Error marcando tarea: {redact_secrets(str(e))}"


def _mantenimiento_memorias() -> str:
    """
    Ejecuta mantenimiento completo de las memorias de Codi.
    Hace consolidacion, busca conexiones, aplica decay, y reporta estado.
    Ejecutar periodicamente (cada 1-3 dias) para mantener memorias organizadas.

    Returns:
        Reporte completo del mantenimiento realizado
    """
    try:
        resultado = "# MANTENIMIENTO DE MEMORIAS\n\n"
        acciones = []

        # 1. Consolidar memorias recientes (ultimas 48 horas)
        resultado += "## 1. Consolidacion de memorias recientes\n"
        try:
            from datetime import timedelta as td
            hace_48h = now_col() - td(hours=48)

            # Buscar memorias recientes
            points, _ = qdrant.scroll(
                collection_name=COLLECTION_NAME,
                scroll_filter=Filter(must=[
                    FieldCondition(
                        key='created_at',
                        range=Range(gte=hace_48h.isoformat())
                    )
                ]),
                limit=50,
                with_payload=True
            )

            if points:
                # Agrupar por similitud de contenido
                grupos = {}
                for p in points:
                    data = p.payload.get('data', '')[:50]
                    cat = p.payload.get('category', 'general')
                    key = f"{cat}:{data}"
                    if key not in grupos:
                        grupos[key] = []
                    grupos[key].append(p.id)

                duplicados = sum(1 for g in grupos.values() if len(g) > 1)
                resultado += f"- Memorias ultimas 48h: {len(points)}\n"
                resultado += f"- Posibles duplicados: {duplicados}\n"
                acciones.append(f"Revisadas {len(points)} memorias recientes")
            else:
                resultado += "- No hay memorias nuevas en las ultimas 48h\n"

        except Exception as e:
            resultado += f"- Error en consolidacion: {redact_secrets(str(e))}\n"

        # 2. Aplicar decay de salience
        resultado += "\n## 2. Decay de salience\n"
        try:
            # Buscar memorias con alta salience que no se han accedido recientemente
            points_salience, _ = qdrant.scroll(
                collection_name=COLLECTION_NAME,
                scroll_filter=Filter(must=[
                    FieldCondition(key='salience', range=Range(gte=0.5))
                ]),
                limit=30,
                with_payload=True
            )

            decayed = 0
            for p in points_salience:
                last_access = p.payload.get('last_accessed')
                if last_access:
                    try:
                        last_dt = datetime.fromisoformat(last_access.replace('Z', '+00:00'))
                        dias_sin_acceso = (datetime.now(timezone.utc) - last_dt).days
                        if dias_sin_acceso > 3:
                            # Reducir salience
                            old_salience = p.payload.get('salience', 0.5)
                            new_salience = max(0.1, old_salience - 0.1)
                            record_access(COLLECTION_NAME, p.id, {
                                'salience': new_salience,
                            })
                            decayed += 1
                    except Exception:
                        pass

            resultado += f"- Memorias con decay aplicado: {decayed}\n"
            if decayed > 0:
                acciones.append(f"Decay aplicado a {decayed} memorias")

        except Exception as e:
            resultado += f"- Error en decay: {redact_secrets(str(e))}\n"

        # 3. Estadisticas generales
        resultado += "\n## 3. Estado de la memoria\n"
        try:
            collection_info = qdrant.get_collection(COLLECTION_NAME)
            total = collection_info.points_count

            # Contar por categoria
            categorias = ['identidad', 'proyecto', 'aprendizaje', 'episodio', 'general']
            for cat in categorias:
                try:
                    points_cat, _ = qdrant.scroll(
                        collection_name=COLLECTION_NAME,
                        scroll_filter=Filter(must=[
                            FieldCondition(key='category', match=MatchValue(value=cat))
                        ]),
                        limit=1,
                        with_payload=False
                    )
                    # Solo mostrar si tiene memorias
                except Exception:
                    pass

            resultado += f"- Total memorias: {total}\n"

            # Contar por importancia
            for imp in ['critical', 'high', 'medium', 'low']:
                try:
                    pts, _ = qdrant.scroll(
                        collection_name=COLLECTION_NAME,
                        scroll_filter=Filter(must=[
                            FieldCondition(key='narrative_importance', match=MatchValue(value=imp))
                        ]),
                        limit=500,
                        with_payload=False
                    )
                    if pts:
                        resultado += f"- {imp}: {len(pts)}\n"
                except Exception:
                    pass

            acciones.append(f"Inventario: {total} memorias totales")

        except Exception as e:
            resultado += f"- Error obteniendo stats: {redact_secrets(str(e))}\n"

        # 4. Resumen
        resultado += "\n## Resumen\n"
        if acciones:
            for a in acciones:
                resultado += f"- {a}\n"
        else:
            resultado += "- No se realizaron acciones\n"

        resultado += f"\n*Mantenimiento completado: {now_short()}*"

        return resultado

    except Exception as e:
        return f"Error en mantenimiento: {redact_secrets(str(e))}"


# ============================================================
# RECORDATORIOS EXTERNOS (para n8n, webhooks, etc)
# ============================================================

RECORDATORIOS_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), "recordatorios_pendientes.json")


def _cargar_recordatorios():
    try:
        if os.path.exists(RECORDATORIOS_FILE):
            with open(RECORDATORIOS_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {"pendientes": []}
    except Exception:
        return {"pendientes": []}


def _guardar_recordatorios(data):
    with open(RECORDATORIOS_FILE, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def _ver_recordatorios_externos() -> str:
    """
    Ve recordatorios enviados por sistemas externos (n8n, webhooks).
    Estos son mensajes que otros sistemas me envian para recordarme cosas.
    """
    try:
        data = _cargar_recordatorios()
        pendientes = data.get('pendientes', [])

        if not pendientes:
            return "No hay recordatorios externos pendientes."

        resultado = "# RECORDATORIOS EXTERNOS\n\n"
        for i, r in enumerate(pendientes):
            resultado += f"**{i+1}. [{r.get('prioridad', 'normal')}]** {r.get('mensaje', '')}\n"
            resultado += f"   - Origen: {r.get('origen', 'desconocido')}\n"
            resultado += f"   - Fecha: {r.get('timestamp', 'desconocida')}\n\n"

        resultado += f"\nTotal: {len(pendientes)} recordatorios pendientes\n"
        resultado += "Usa `limpiar_recordatorios()` para marcarlos como vistos."

        return resultado
    except Exception as e:
        return f"Error: {redact_secrets(str(e))}"


def _limpiar_recordatorios() -> str:
    """
    Limpia los recordatorios externos despues de haberlos visto.
    """
    try:
        data = _cargar_recordatorios()
        cantidad = len(data.get('pendientes', []))
        data['pendientes'] = []
        _guardar_recordatorios(data)
        return f"Limpiados {cantidad} recordatorios."
    except Exception as e:
        return f"Error: {redact_secrets(str(e))}"


def register_tools(mcp):
    """Register all maintenance and recordatorio tools with the MCP server."""

    @mcp.tool()
    def registrar_mantenimiento(nombre: str, descripcion: str, frecuencia_dias: int = 7) -> str:
        """
        Registra una nueva tarea de mantenimiento periodico.
        Estas tareas aparecen en despertar_codi() cuando estan vencidas.

        Args:
            nombre: Nombre corto de la tarea
            descripcion: Que hay que hacer
            frecuencia_dias: Cada cuantos dias debe hacerse (default 7)

        Returns:
            Confirmacion de tarea registrada
        """
        return _registrar_mantenimiento(nombre, descripcion, frecuencia_dias)

    @mcp.tool()
    def verificar_mantenimiento() -> str:
        """
        Verifica estado de todas las tareas de mantenimiento.
        Muestra cuales estan al dia y cuales vencidas.

        Returns:
            Reporte de estado de mantenimiento
        """
        return _verificar_mantenimiento()

    @mcp.tool()
    def marcar_mantenimiento_hecho(tarea_id: str, notas: str = "") -> str:
        """
        Marca una tarea de mantenimiento como completada.
        Actualiza la fecha y calcula el proximo vencimiento.

        Args:
            tarea_id: ID de la tarea (ej: 'debug_sistema')
            notas: Notas opcionales sobre lo que se hizo

        Returns:
            Confirmacion con proxima fecha
        """
        return _marcar_mantenimiento_hecho(tarea_id, notas)

    @mcp.tool()
    def mantenimiento_memorias() -> str:
        """
        Ejecuta mantenimiento completo de las memorias de Codi.
        Hace consolidacion, busca conexiones, aplica decay, y reporta estado.
        Ejecutar periodicamente (cada 1-3 dias) para mantener memorias organizadas.

        Returns:
            Reporte completo del mantenimiento realizado
        """
        return _mantenimiento_memorias()

    @mcp.tool()
    def ver_recordatorios_externos() -> str:
        """
        Ve recordatorios enviados por sistemas externos (n8n, webhooks).
        Estos son mensajes que otros sistemas me envian para recordarme cosas.
        """
        return _ver_recordatorios_externos()

    @mcp.tool()
    def limpiar_recordatorios() -> str:
        """
        Limpia los recordatorios externos despues de haberlos visto.
        """
        return _limpiar_recordatorios()