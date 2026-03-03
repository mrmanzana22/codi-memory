"""
Codi Memory - Triggers module.
Handles trigger loading, detection, evaluation, and dynamic creation.
Triggers act as 'memory webhooks' that detect patterns and activate protocols.
"""

import os
import json
from datetime import datetime

from modules.config import USER_ID, COLLECTION_NAME, TRIGGERS_FILE, _current_session, _emotional_state, now_iso
from modules.secret_redact import redact_secrets
from modules.memory_smart import search_with_fts_content

# ============================================================
# MODULE-LEVEL STATE AND HELPERS
# ============================================================

_triggers_cache = None  # Cache de triggers cargados


def _load_triggers():
    """Carga triggers desde archivo JSON."""
    global _triggers_cache
    if _triggers_cache is None:
        try:
            if os.path.exists(TRIGGERS_FILE):
                with open(TRIGGERS_FILE, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    _triggers_cache = data.get('triggers', {})
            else:
                _triggers_cache = {}
        except Exception as e:
            _triggers_cache = {}
    return _triggers_cache


def _detect_triggers(text: str) -> list:
    """Detecta triggers activos basado en patrones en el texto."""
    triggers = _load_triggers()
    activated = []
    text_lower = text.lower()

    for trigger_name, trigger_data in triggers.items():
        patterns = trigger_data.get('patterns', [])
        for pattern in patterns:
            if pattern.lower() in text_lower:
                activated.append({
                    'trigger': trigger_name,
                    'pattern_matched': pattern,
                    'action': trigger_data.get('action'),
                    'agent': trigger_data.get('agent'),
                    'evoca': trigger_data.get('evoca', []),
                    'respuesta_automatica': trigger_data.get('respuesta_automatica'),
                    'contexto_a_buscar': trigger_data.get('contexto_a_buscar')
                })
                break  # Solo un match por trigger

    return activated


# ============================================================
# MCP TOOL REGISTRATION
# ============================================================

def register_tools(mcp):

    @mcp.tool()
    def evaluar_triggers(input_text: str) -> str:
        """
        Evalua triggers basado en el texto de entrada.
        Como un webhook de memoria - detecta patrones y activa protocolos.

        Args:
            input_text: El texto a analizar (mensaje del usuario)

        Returns:
            JSON con triggers activados y acciones a tomar
        """
        try:
            activated = _detect_triggers(input_text)

            if not activated:
                return json.dumps({
                    "status": "no_triggers",
                    "message": "Ningun trigger activado",
                    "triggers_checked": len(_load_triggers())
                }, ensure_ascii=False, indent=2)

            # Ordenar por prioridad (proyecto_nuevo primero si existe)
            from modules.config import TRIGGER_PRIORITY_ORDER
            priority_order = TRIGGER_PRIORITY_ORDER
            activated.sort(key=lambda x: priority_order.index(x['trigger']) if x['trigger'] in priority_order else 99)

            return json.dumps({
                "status": "triggers_activated",
                "count": len(activated),
                "triggers": activated,
                "recommendation": f"Activar protocolo: {activated[0]['action']}" if activated else None
            }, ensure_ascii=False, indent=2)

        except Exception as e:
            return json.dumps({"error": redact_secrets(str(e))})

    @mcp.tool()
    def activar_trigger(trigger_name: str) -> str:
        """
        Activa manualmente un trigger especifico y retorna su protocolo.

        Args:
            trigger_name: Nombre del trigger (ej: 'proyecto_nuevo', 'fullempaques')

        Returns:
            Protocolo completo del trigger con acciones y contextos a evocar
        """
        try:
            triggers = _load_triggers()

            if trigger_name not in triggers:
                available = list(triggers.keys())
                return json.dumps({
                    "error": f"Trigger '{trigger_name}' no existe",
                    "triggers_disponibles": available
                }, ensure_ascii=False, indent=2)

            trigger = triggers[trigger_name]

            # Buscar contexto relacionado en memoria si hay contexto_a_buscar
            contexto_memoria = []
            if trigger.get('contexto_a_buscar'):
                try:
                    resultado = search_with_fts_content(query=trigger['contexto_a_buscar'], user_id=USER_ID, limit=3)
                    if resultado and resultado.get("results"):
                        for m in resultado["results"]:
                            contexto_memoria.append(m.get('memory', ''))
                except Exception:
                    pass

            return json.dumps({
                "trigger": trigger_name,
                "action": trigger.get('action'),
                "agent_recomendado": trigger.get('agent'),
                "pasos_a_evocar": trigger.get('evoca', []),
                "respuesta_automatica": trigger.get('respuesta_automatica'),
                "contexto_de_memoria": contexto_memoria,
                "status": "activado"
            }, ensure_ascii=False, indent=2)

        except Exception as e:
            return json.dumps({"error": redact_secrets(str(e))})

    @mcp.tool()
    def listar_triggers() -> str:
        """
        Lista todos los triggers disponibles con sus patrones.
        Util para ver que protocolos estan configurados.
        """
        try:
            triggers = _load_triggers()

            resumen = []
            for name, data in triggers.items():
                resumen.append({
                    "nombre": name,
                    "patterns": data.get('patterns', []),
                    "agent": data.get('agent'),
                    "action": data.get('action')
                })

            return json.dumps({
                "total_triggers": len(resumen),
                "triggers": resumen
            }, ensure_ascii=False, indent=2)

        except Exception as e:
            return json.dumps({"error": redact_secrets(str(e))})

    @mcp.tool()
    def crear_trigger_dinamico(
        nombre: str,
        patterns: str,
        action: str,
        agent: str = None,
        evoca: str = None,
        contexto_a_buscar: str = None,
        respuesta_automatica: str = None
    ) -> str:
        """
        Crea un nuevo trigger dinamicamente y lo guarda en triggers.json.
        Usado para aprendizaje basado en experiencia emocional.

        Args:
            nombre: Nombre unico del trigger (ej: 'nuevo_tema')
            patterns: Palabras clave separadas por coma (ej: 'palabra1, palabra2, frase clave')
            action: Nombre del protocolo a ejecutar
            agent: Agente recomendado (opcional)
            evoca: Contextos a evocar separados por coma (opcional)
            contexto_a_buscar: Query para buscar en memoria (opcional)
            respuesta_automatica: Mensaje automatico al activarse (opcional)

        Returns:
            Confirmacion del trigger creado
        """
        global _triggers_cache

        try:
            # Cargar triggers actuales
            if os.path.exists(TRIGGERS_FILE):
                with open(TRIGGERS_FILE, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            else:
                data = {"_meta": {"version": "1.0", "description": "Sistema de triggers de Codi"}, "triggers": {}, "indice_rapido": {}}

            # Verificar que no exista
            if nombre in data.get('triggers', {}):
                return json.dumps({
                    "error": f"Trigger '{nombre}' ya existe",
                    "sugerencia": f"Usa otro nombre o edita el existente"
                }, ensure_ascii=False)

            # Parsear patterns y evoca
            patterns_list = [p.strip() for p in patterns.split(',')]
            evoca_list = [e.strip() for e in evoca.split(',')] if evoca else []

            # Crear nuevo trigger
            nuevo_trigger = {
                "patterns": patterns_list,
                "action": action,
                "agent": agent,
                "evoca": evoca_list,
                "contexto_a_buscar": contexto_a_buscar,
                "creado_por": "aprendizaje_emocional",
                "fecha_creacion": now_iso()
            }

            if respuesta_automatica:
                nuevo_trigger["respuesta_automatica"] = respuesta_automatica

            # Agregar a triggers
            data['triggers'][nombre] = nuevo_trigger

            # Actualizar indice rapido
            primera_letra = nombre[0].upper()
            if primera_letra not in data.get('indice_rapido', {}):
                data['indice_rapido'][primera_letra] = []
            if nombre not in data['indice_rapido'][primera_letra]:
                data['indice_rapido'][primera_letra].append(nombre)

            # Guardar
            with open(TRIGGERS_FILE, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

            # Invalidar cache
            _triggers_cache = None

            return json.dumps({
                "status": "trigger_creado",
                "nombre": nombre,
                "patterns": patterns_list,
                "action": action,
                "agent": agent,
                "mensaje": f"Trigger '{nombre}' creado exitosamente. Se activara cuando detecte: {patterns_list}"
            }, ensure_ascii=False, indent=2)

        except Exception as e:
            return json.dumps({"error": redact_secrets(str(e))})

    @mcp.tool()
    def sugerir_trigger_emocional(contexto: str, razon_emocional: str) -> str:
        """
        Analiza el contexto actual y la emocion para sugerir crear un trigger.
        Usado cuando experimento algo intenso (sorpresa, frustracion, satisfaccion).

        Args:
            contexto: Descripcion del contexto/tema actual
            razon_emocional: Por que siento que deberia crear un trigger (ej: 'me frustre buscando esto', 'esto funciono muy bien')

        Returns:
            Sugerencia de trigger con estructura propuesta
        """
        try:
            # Obtener estado emocional actual
            emocion_actual = _emotional_state.get('current', {})
            arousal = emocion_actual.get('arousal', 0)
            pleasure = emocion_actual.get('pleasure', 0)

            # Determinar tipo de aprendizaje
            if pleasure > 0.5:
                tipo = "refuerzo_positivo"
                sugerencia_action = "repetir_exito"
            elif pleasure < -0.3:
                tipo = "evitar_frustracion"
                sugerencia_action = "cargar_contexto_preventivo"
            else:
                tipo = "neutral"
                sugerencia_action = "cargar_contexto"

            # Extraer palabras clave del contexto (simplificado)
            palabras = contexto.lower().split()
            # Filtrar palabras cortas y comunes
            stopwords = {'el', 'la', 'de', 'que', 'y', 'a', 'en', 'es', 'por', 'con', 'para', 'un', 'una', 'los', 'las', 'del', 'al'}
            keywords = [p for p in palabras if len(p) > 3 and p not in stopwords][:5]

            # Generar nombre sugerido
            nombre_sugerido = "_".join(keywords[:2]) if len(keywords) >= 2 else f"tema_{keywords[0]}" if keywords else "nuevo_trigger"

            # Verificar si ya existe algo similar
            triggers_existentes = _load_triggers()
            similares = []
            for tname, tdata in triggers_existentes.items():
                for pattern in tdata.get('patterns', []):
                    if any(kw in pattern.lower() for kw in keywords):
                        similares.append(tname)
                        break

            return json.dumps({
                "analisis": {
                    "contexto": contexto,
                    "razon_emocional": razon_emocional,
                    "estado_emocional": {
                        "arousal": arousal,
                        "pleasure": pleasure,
                        "intensidad": "alta" if abs(arousal) > 0.5 else "media" if abs(arousal) > 0.2 else "baja"
                    },
                    "tipo_aprendizaje": tipo
                },
                "sugerencia": {
                    "nombre": nombre_sugerido,
                    "patterns_sugeridos": keywords,
                    "action_sugerida": sugerencia_action,
                    "evoca_sugerido": ["contexto_" + nombre_sugerido, "experiencias_anteriores"]
                },
                "triggers_similares": similares if similares else "ninguno",
                "siguiente_paso": f"Si te parece bien, ejecuta: crear_trigger_dinamico(nombre='{nombre_sugerido}', patterns='{', '.join(keywords)}', action='{sugerencia_action}')"
            }, ensure_ascii=False, indent=2)

        except Exception as e:
            return json.dumps({"error": redact_secrets(str(e))})