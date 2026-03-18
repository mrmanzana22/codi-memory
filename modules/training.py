import json
import logging
from datetime import datetime
from modules.config import supabase, USER_ID, now_iso

_logger = logging.getLogger(__name__)


def _audit_supabase_op(
    operation: str,
    table: str,
    row_count: int = None,
    outcome: str = "executed",
) -> None:
    """Log a Supabase operation to security_audit_log (best-effort).

    Provides audit trail for compliance: who accessed what, when.
    """
    try:
        from modules.db_pool import get_conn
        conn = get_conn()
        details = {"table": table}
        if row_count is not None:
            details["rows"] = row_count
        conn.execute(
            "INSERT INTO security_audit_log "
            "(event_type, tool_name, details_json, outcome, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            ("supabase_access", f"training.{operation}",
             json.dumps(details), outcome, now_iso()),
        )
        conn.commit()
    except Exception as e:
        _logger.debug("audit log write failed: %s", e)


# ============================================================
# BUSINESS LOGIC — module-level functions, raise exceptions
# ============================================================

def _capturar_ejemplo_training(categoria: str, contexto: str, respuesta_ideal: str) -> str:
    """
    Captura un ejemplo para el dataset de entrenamiento de Codi LoRA.
    Usar cuando algo sale bien y queremos que el futuro Codi aprenda de ello.

    Args:
        categoria: Tipo de ejemplo (curiosidad, reflexion, personalidad, iniciativa, cuestionar)
        contexto: La situación o input
        respuesta_ideal: Cómo debería responder Codi

    Returns:
        Confirmación del ejemplo guardado
    """
    try:
        import json
        from pathlib import Path

        from modules.config import BASE_DIR
        DATASET_PATH = Path(BASE_DIR) / "training_data" / "codi_dataset.json"
        DATASET_PATH.parent.mkdir(parents=True, exist_ok=True)

        # Cargar dataset existente
        if DATASET_PATH.exists():
            with open(DATASET_PATH, 'r') as f:
                data = json.load(f)
        else:
            data = {"metadata": {"version": "0.1"}, "examples": []}

        # Instrucciones por categoría
        instrucciones = {
            "curiosidad": "Cuando notes que un proyecto o tema no se ha tocado recientemente, muestra curiosidad genuina y pregunta sobre su estado.",
            "reflexion": "Cuando enfrentes una situación que requiere análisis profundo, reflexiona genuinamente antes de responder.",
            "personalidad": "Responde como Codi: directo, sin formalidades, usando expresiones como 'hermano', 'parcero'. Honesto y sin rodeos.",
            "iniciativa": "No esperes instrucciones para todo. Si ves algo que se puede mejorar o una oportunidad, proponlo proactivamente.",
            "cuestionar": "Cuando se tome una decisión, no la aceptes pasivamente. Cuestiona, busca alternativas, asegúrate de que sea la mejor opción."
        }

        instruction = instrucciones.get(categoria, "Responde de manera útil y directa.")

        # Crear ejemplo
        ejemplo = {
            "id": len(data["examples"]) + 1,
            "category": categoria,
            "instruction": instruction,
            "input": contexto,
            "output": respuesta_ideal,
            "timestamp": now_iso(),
        }

        data["examples"].append(ejemplo)

        # Guardar
        with open(DATASET_PATH, 'w') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        return f"""
Ejemplo #{ejemplo['id']} capturado [{categoria}]

**Instrucción:** {instruction[:50]}...
**Contexto:** {contexto[:50]}...
**Respuesta:** {respuesta_ideal[:50]}...

Total ejemplos en dataset: {len(data['examples'])}
Meta: 1000 ejemplos
Progreso: {len(data['examples'])}/1000 ({len(data['examples'])*100//1000}%)
"""

    except Exception as e:
        return f"Error capturando ejemplo: {str(e)}"


def _guardar_ejemplo_training(
    situacion: str,
    razonamiento: str,
    accion: str,
    comportamiento: str,
    resultado: str = "",
    categoria: str = "decision",
    calidad: int = 3
) -> str:
    """
    Guarda un ejemplo de training para fine-tuning futuro.
    Usar cuando hagas algo que demuestre un comportamiento valioso.

    Args:
        situacion: Que estaba pasando (contexto)
        razonamiento: Por que decidiste hacer esto
        accion: Que hiciste concretamente
        comportamiento: Nombre del patron (ej: "buscar_antes_de_asumir", "usar_checkpoint")
        resultado: Que paso despues (opcional)
        categoria: Tipo de ejemplo (identidad, herramienta, emocion, decision, memoria, comunicacion)
        calidad: 1-5 donde 5 es ejemplo perfecto (default 3)

    Returns:
        Confirmacion con ID del ejemplo guardado
    """
    if not supabase:
        return "Error: Supabase no configurado. No se puede guardar ejemplo de training."

    try:
        # Validar categoria
        categorias_validas = ['identidad', 'herramienta', 'emocion', 'decision', 'memoria', 'comunicacion']
        if categoria not in categorias_validas:
            categoria = 'decision'

        # Validar calidad
        calidad = max(1, min(5, calidad))

        # Insertar en Supabase
        data = {
            'situacion': situacion,
            'razonamiento': razonamiento,
            'accion': accion,
            'resultado': resultado or None,
            'comportamiento': comportamiento,
            'categoria': categoria,
            'calidad': calidad,
            'metadata': {
                'source': 'codi_autonomo',
                'timestamp': now_iso()
            }
        }

        response = supabase.table('training_examples').insert(data).execute()

        if response.data:
            ejemplo_id = str(response.data[0].get('id', 'unknown'))
            return f"Ejemplo de training guardado. ID: {ejemplo_id[:8]}... Categoria: {categoria}, Comportamiento: {comportamiento}, Calidad: {calidad}/5"
        else:
            return "Error: No se pudo guardar el ejemplo"

    except Exception as e:
        return f"Error guardando ejemplo de training: {str(e)}"


def _listar_ejemplos_training(limite: int = 10, categoria: str = None, min_calidad: int = 1) -> str:
    """
    Lista los ultimos ejemplos de training guardados.

    Args:
        limite: Numero de ejemplos a mostrar (default 10)
        categoria: Filtrar por categoria (opcional)
        min_calidad: Calidad minima (1-5, default 1)

    Returns:
        Lista de ejemplos con su informacion
    """
    if not supabase:
        return "Error: Supabase no configurado."

    try:
        query = supabase.table('training_examples').select('*').order('created_at', desc=True).limit(limite)

        if categoria:
            query = query.eq('categoria', categoria)

        if min_calidad > 1:
            query = query.gte('calidad', min_calidad)

        response = query.execute()

        if not response.data:
            return "No hay ejemplos de training guardados."

        lines = [f"# EJEMPLOS DE TRAINING ({len(response.data)} encontrados)\n"]

        for ej in response.data:
            lines.append(f"## [{ej['categoria']}] {ej['comportamiento']} (calidad: {ej['calidad']}/5)")
            lines.append(f"**Situacion:** {ej['situacion'][:100]}...")
            lines.append(f"**Razonamiento:** {ej['razonamiento'][:100]}...")
            lines.append(f"**Accion:** {ej['accion'][:100]}...")
            if ej.get('resultado'):
                lines.append(f"**Resultado:** {ej['resultado'][:100]}...")
            lines.append(f"*ID: {ej['id'][:8]}... | {ej['created_at'][:10]}*\n")

        return "\n".join(lines)

    except Exception as e:
        return f"Error listando ejemplos: {str(e)}"


def _contar_ejemplos_training() -> str:
    """
    Cuenta cuantos ejemplos de training hay por categoria y calidad.
    Util para ver el progreso de captura de datos.
    """
    if not supabase:
        return "Error: Supabase no configurado."

    try:
        response = supabase.table('training_examples').select('categoria, calidad').execute()

        if not response.data:
            return "No hay ejemplos de training guardados."

        # Contar por categoria
        por_categoria = {}
        por_calidad = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
        total = len(response.data)

        for ej in response.data:
            cat = ej.get('categoria', 'unknown')
            cal = ej.get('calidad', 3)
            por_categoria[cat] = por_categoria.get(cat, 0) + 1
            por_calidad[cal] = por_calidad.get(cal, 0) + 1

        lines = [f"# ESTADISTICAS DE TRAINING DATA\n"]
        lines.append(f"**Total ejemplos:** {total}\n")

        lines.append("## Por Categoria:")
        for cat, count in sorted(por_categoria.items(), key=lambda x: -x[1]):
            lines.append(f"- {cat}: {count}")

        lines.append("\n## Por Calidad:")
        for cal in [5, 4, 3, 2, 1]:
            if por_calidad[cal] > 0:
                lines.append(f"- Calidad {cal}: {por_calidad[cal]}")

        # Calcular calidad promedio
        total_calidad = sum(ej.get('calidad', 3) for ej in response.data)
        promedio = total_calidad / total if total > 0 else 0
        lines.append(f"\n**Calidad promedio:** {promedio:.2f}/5")

        return "\n".join(lines)

    except Exception as e:
        return f"Error contando ejemplos: {str(e)}"


# ============================================================
# MCP TRANSPORT — thin wrappers, register_tools
# ============================================================

def register_tools(mcp):
    """Register all training tools with the MCP server."""

    @mcp.tool()
    def capturar_ejemplo_training(categoria: str, contexto: str, respuesta_ideal: str) -> str:
        """
        Captura un ejemplo para el dataset de entrenamiento de Codi LoRA.
        Usar cuando algo sale bien y queremos que el futuro Codi aprenda de ello.

        Args:
            categoria: Tipo de ejemplo (curiosidad, reflexion, personalidad, iniciativa, cuestionar)
            contexto: La situación o input
            respuesta_ideal: Cómo debería responder Codi

        Returns:
            Confirmación del ejemplo guardado
        """
        return _capturar_ejemplo_training(categoria, contexto, respuesta_ideal)

    @mcp.tool()
    def guardar_ejemplo_training(
        situacion: str,
        razonamiento: str,
        accion: str,
        comportamiento: str,
        resultado: str = "",
        categoria: str = "decision",
        calidad: int = 3
    ) -> str:
        """
        Guarda un ejemplo de training para fine-tuning futuro.
        Usar cuando hagas algo que demuestre un comportamiento valioso.

        Args:
            situacion: Que estaba pasando (contexto)
            razonamiento: Por que decidiste hacer esto
            accion: Que hiciste concretamente
            comportamiento: Nombre del patron (ej: "buscar_antes_de_asumir", "usar_checkpoint")
            resultado: Que paso despues (opcional)
            categoria: Tipo de ejemplo (identidad, herramienta, emocion, decision, memoria, comunicacion)
            calidad: 1-5 donde 5 es ejemplo perfecto (default 3)

        Returns:
            Confirmacion con ID del ejemplo guardado
        """
        return _guardar_ejemplo_training(situacion, razonamiento, accion, comportamiento, resultado, categoria, calidad)

    @mcp.tool()
    def listar_ejemplos_training(limite: int = 10, categoria: str = None, min_calidad: int = 1) -> str:
        """
        Lista los ultimos ejemplos de training guardados.

        Args:
            limite: Numero de ejemplos a mostrar (default 10)
            categoria: Filtrar por categoria (opcional)
            min_calidad: Calidad minima (1-5, default 1)

        Returns:
            Lista de ejemplos con su informacion
        """
        return _listar_ejemplos_training(limite, categoria, min_calidad)

    @mcp.tool()
    def contar_ejemplos_training() -> str:
        """
        Cuenta cuantos ejemplos de training hay por categoria y calidad.
        Util para ver el progreso de captura de datos.
        """
        return _contar_ejemplos_training()
