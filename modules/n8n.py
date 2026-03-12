"""
Codi Memory - N8N integration module.
Webhook dispatch to n8n workflows.
"""

import logging
import os
import requests as http_requests

from modules.config import now_iso
from modules.secret_redact import redact_secrets

__all__ = [
    "trigger_n8n",
    "listar_webhooks_conocidos",
    "N8N_WEBHOOK_BASE",
    "register_tools",
]

# N8N webhook base
N8N_WEBHOOK_BASE = os.getenv("N8N_WEBHOOK_BASE", "").strip()


def trigger_n8n(webhook_path: str, data: dict = None, esperar_respuesta: bool = False) -> str:
    """
    Dispara un workflow en n8n enviando datos a un webhook.

    Args:
        webhook_path: Path del webhook (ej: 'codi-alerta', 'trading-orden')
        data: Datos a enviar (dict)
        esperar_respuesta: Si True, espera y retorna la respuesta de n8n
    """
    try:
        if not N8N_WEBHOOK_BASE:
            return "n8n no esta configurado (N8N_WEBHOOK_BASE no esta en .env)."
        # Validacion basica para evitar paths inesperados
        if not webhook_path or any((not ch.isalnum()) and ch not in ('_', '-') for ch in webhook_path) or len(webhook_path) > 80:
            return "Error: webhook_path invalido (solo A-Z a-z 0-9 _ -)"
        url = f"{N8N_WEBHOOK_BASE}/{webhook_path}"
        payload = dict(data or {})
        payload['_from'] = 'codi-memory'
        payload['_timestamp'] = now_iso()
        timeout = 30 if esperar_respuesta else 5

        response = http_requests.post(url, json=payload, timeout=timeout, headers={'Content-Type': 'application/json'})

        if esperar_respuesta:
            if response.status_code not in [200, 201, 202]:
                return f"Error disparando webhook: {response.status_code} - {response.text[:200]}"
            try:
                return f"Respuesta de n8n: {response.json()}"
            except Exception:
                return f"Respuesta de n8n (texto): {response.text[:500]}"
        else:
            if response.status_code in [200, 201, 202]:
                return f"Webhook disparado: {webhook_path} - Status: {response.status_code}"
            else:
                return f"Error disparando webhook: {response.status_code} - {response.text[:200]}"
    except http_requests.exceptions.Timeout:
        return f"Timeout esperando respuesta de n8n (webhook: {webhook_path})"
    except Exception as e:
        logging.error("Error disparando n8n (webhook: %s): %s", webhook_path, redact_secrets(str(e)), exc_info=True)
        return f"Error disparando n8n: {redact_secrets(str(e))}"


def listar_webhooks_conocidos() -> str:
    """Lista los webhooks de n8n que conozco y puedo disparar."""
    webhooks = {
        "codi-alerta": "Enviar alertas generales a Hare",
        "trading-signal": "Enviar senal de trading para procesar",
        "backup-trigger": "Disparar backup de memorias",
        "reporte-diario": "Generar y enviar reporte diario",
    }
    resultado = "# WEBHOOKS N8N CONOCIDOS\n\n"
    base_display = N8N_WEBHOOK_BASE or "(no configurado)"
    resultado += f"Base URL: {base_display}\n\n"
    for path, desc in webhooks.items():
        resultado += f"- **{path}**: {desc}\n"
    resultado += "\n*Nota: Estos webhooks deben existir en n8n para funcionar.*"
    resultado += "\n*Usa trigger_n8n('nombre', {datos}) para disparar.*"
    return resultado


def register_tools(mcp):
    """Register N8N MCP tools."""
    mcp.tool()(trigger_n8n)
    mcp.tool()(listar_webhooks_conocidos)
