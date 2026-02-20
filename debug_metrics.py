#!/usr/bin/env python3
"""
debug_metrics.py - Diagnóstico automatizado del Sistema de Auto-Auditoría

Corre todos los checks del protocolo de debug en secuencia.
Uso: python3 debug_metrics.py
"""

import sqlite3
import sys
from pathlib import Path

# Colors para terminal
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
RESET = "\033[0m"
BOLD = "\033[1m"

def check(name: str, test_fn, critical=False):
    """Ejecuta un check y reporta resultado."""
    try:
        result = test_fn()
        if result:
            print(f"{GREEN}✅ {name}{RESET}")
            return True
        else:
            print(f"{YELLOW}⚠️  {name} - NO PASSED{RESET}")
            return False
    except Exception as e:
        print(f"{RED}❌ {name} - ERROR: {e}{RESET}")
        if critical:
            print(f"{RED}CRÍTICO: No se puede continuar sin este check.{RESET}")
            sys.exit(1)
        return False

print(f"\n{BOLD}=== DIAGNÓSTICO SISTEMA AUTO-AUDITORÍA ==={RESET}\n")

# L0: Básico
print(f"{BOLD}L0: VERIFICACIÓN BÁSICA{RESET}")

def l0_1_import():
    import modules.metrics
    return True

def l0_2_schema():
    from modules.config import FTS_DB_PATH
    conn = sqlite3.connect(FTS_DB_PATH, timeout=3.0)
    try:
        result = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='tool_calls'").fetchone()
        return result is not None
    finally:
        conn.close()

def l0_3_columns():
    from modules.config import FTS_DB_PATH
    conn = sqlite3.connect(FTS_DB_PATH, timeout=3.0)
    try:
        cols = conn.execute("PRAGMA table_info(tool_calls)").fetchall()
        col_names = [c[1] for c in cols]
        expected = ["id", "tool_name", "started_at", "duration_ms", "success",
                   "error_type", "args_size", "result_size", "session_id", "tag"]
        return set(expected).issubset(set(col_names))
    finally:
        conn.close()

check("L0.1: Import metrics.py", l0_1_import, critical=True)
check("L0.2: Tabla tool_calls existe", l0_2_schema)
check("L0.3: Columnas correctas (10 esperadas)", l0_3_columns)

# L1: Instrumentación
print(f"\n{BOLD}L1: INSTRUMENTACIÓN ACTIVA{RESET}")

def l1_1_server():
    server_py = Path("server.py")
    if not server_py.exists():
        return False
    content = server_py.read_text()
    return "metrics.instrument_mcp" in content

def l1_2_before_register():
    server_py = Path("server.py")
    content = server_py.read_text()
    lines = content.split('\n')
    instrument_line = None
    register_line = None
    for i, line in enumerate(lines):
        if "metrics.instrument_mcp" in line:
            instrument_line = i
        if "register_tools(mcp)" in line and register_line is None:
            register_line = i
    return instrument_line is not None and register_line is not None and instrument_line < register_line

check("L1.1: metrics.instrument_mcp() en server.py", l1_1_server)
check("L1.2: Instrumentación ANTES de register_tools", l1_2_before_register)

# L2: Captura de datos
print(f"\n{BOLD}L2: CAPTURA DE DATOS{RESET}")

def l2_1_has_data():
    from modules.config import FTS_DB_PATH
    conn = sqlite3.connect(FTS_DB_PATH, timeout=3.0)
    try:
        count = conn.execute("SELECT COUNT(*) FROM tool_calls").fetchone()[0]
        if count > 0:
            print(f"   → {count} tool calls capturadas")
        return count > 0
    finally:
        conn.close()

def l2_2_macro_tags():
    from modules.config import FTS_DB_PATH
    conn = sqlite3.connect(FTS_DB_PATH, timeout=3.0)
    try:
        macro_tools = conn.execute("""
            SELECT tool_name, tag FROM tool_calls
            WHERE tool_name IN ('recall','remember','context_snapshot')
            LIMIT 5
        """).fetchall()
        if not macro_tools:
            return None  # No data yet, not a failure
        has_tags = all(tag is not None and tag.startswith('macro:') for _, tag in macro_tools)
        if has_tags:
            print(f"   → {len(macro_tools)} macro-tools con tags correctos")
        return has_tags
    finally:
        conn.close()

def l2_3_sizes():
    from modules.config import FTS_DB_PATH
    conn = sqlite3.connect(FTS_DB_PATH, timeout=3.0)
    try:
        rows = conn.execute("""
            SELECT args_size, result_size FROM tool_calls
            WHERE args_size > 0 AND result_size > 0
            LIMIT 1
        """).fetchall()
        return len(rows) > 0
    finally:
        conn.close()

has_data = check("L2.1: Hay datos capturados", l2_1_has_data)
if has_data:
    check("L2.2: Tags macro presentes", l2_2_macro_tags)
    check("L2.3: Tamaños (args/result) > 0", l2_3_sizes)
else:
    print(f"{YELLOW}   ⚠️  No hay datos aún. Usa recall/remember/context_snapshot para generar datos.{RESET}")

# L3: Agregaciones
print(f"\n{BOLD}L3: AGREGACIONES{RESET}")

def l3_1_summary():
    from modules.metrics import tool_usage_summary
    summary = tool_usage_summary(days=7)
    total = summary.get("total_calls", 0)
    macro = summary.get("macro_calls", 0)
    share = summary.get("macro_share", 0.0) * 100
    if total > 0:
        print(f"   → Total: {total}, Macro: {macro} ({share:.1f}%)")
    return total > 0

def l3_2_audit():
    from modules.consciousness import audit_tools
    result = audit_tools()
    has_metrics = "No hay metricas" not in result
    if has_metrics:
        lines = result.split('\n')
        for line in lines[:4]:  # Primeras 4 líneas
            if line.strip():
                print(f"   → {line.strip()}")
    return has_metrics

if has_data:
    check("L3.1: tool_usage_summary() funciona", l3_1_summary)
    check("L3.2: audit_tools() lee de DB", l3_2_audit)
else:
    print(f"{YELLOW}   ⚠️  Saltando L3 (no hay datos){RESET}")

# L4: Hooks
print(f"\n{BOLD}L4: HOOKS AUTOMÁTICOS{RESET}")

def l4_1_hook_present():
    consciousness_py = Path("modules/consciousness.py")
    if not consciousness_py.exists():
        return False
    content = consciousness_py.read_text()
    return "Auto-auditoria nocturna" in content

def l4_2_reflection_exists():
    from modules.config import FTS_DB_PATH
    conn = sqlite3.connect(FTS_DB_PATH, timeout=3.0)
    try:
        # Buscar en memories si existe
        result = conn.execute("""
            SELECT COUNT(*) FROM memories
            WHERE source = 'reflection'
            AND content LIKE '%Auto-Auditoria (NOCHE)%'
        """).fetchone()
        count = result[0] if result else 0
        if count > 0:
            print(f"   → {count} memoria(s) de reflection encontrada(s)")
        return count > 0
    except sqlite3.OperationalError:
        # Tabla memories no existe o no tiene esas columnas
        return None
    finally:
        conn.close()

check("L4.1: Hook nocturno presente en consciousness.py", l4_1_hook_present)
reflection = check("L4.2: Memoria reflection existe", l4_2_reflection_exists)
if reflection is None:
    print(f"{YELLOW}   ⚠️  No se pudo verificar (tabla memories no accesible){RESET}")

# Resumen final
print(f"\n{BOLD}=== RESUMEN ==={RESET}\n")

from modules.config import FTS_DB_PATH
conn = sqlite3.connect(FTS_DB_PATH, timeout=3.0)
try:
    total_calls = conn.execute("SELECT COUNT(*) FROM tool_calls").fetchone()[0]
    if total_calls > 0:
        # Últimas 5 llamadas
        print(f"{BOLD}Últimas 5 tool calls:{RESET}")
        recent = conn.execute("""
            SELECT
                substr(started_at, 12, 8) as time,
                tool_name,
                CASE success WHEN 1 THEN '✅' ELSE '❌' END as ok,
                duration_ms || 'ms' as dur,
                COALESCE(tag, '-') as tag
            FROM tool_calls
            ORDER BY started_at DESC
            LIMIT 5
        """).fetchall()
        for row in recent:
            print(f"  {row[0]} | {row[1]:20s} | {row[2]} | {row[3]:6s} | {row[4]}")

        # Top 5 tools
        print(f"\n{BOLD}Top 5 tools más usadas (7d):{RESET}")
        top = conn.execute("""
            SELECT
                tool_name,
                COUNT(*) as calls,
                ROUND(AVG(duration_ms), 0) as avg_ms
            FROM tool_calls
            WHERE started_at >= datetime('now', '-7 days')
            GROUP BY tool_name
            ORDER BY calls DESC
            LIMIT 5
        """).fetchall()
        for row in top:
            print(f"  {row[0]:20s} | {row[1]:4d} calls | {int(row[2]):4d}ms avg")
    else:
        print(f"{YELLOW}No hay datos capturados aún.{RESET}")
        print(f"\n{BOLD}Próximos pasos:{RESET}")
        print("1. Reiniciar MCP server si no lo has hecho")
        print("2. Usar recall(), remember(), context_snapshot() desde el MCP client")
        print("3. Volver a ejecutar este diagnóstico")

except Exception as e:
    print(f"{RED}Error al generar resumen: {e}{RESET}")
finally:
    conn.close()

print(f"\n{BOLD}Diagnóstico completo.{RESET}")
print(f"Ver protocolo completo en: {BOLD}DEBUG_PROTOCOL.md{RESET}\n")
