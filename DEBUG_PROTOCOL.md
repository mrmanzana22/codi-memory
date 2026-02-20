# Protocolo de Debug - Sistema de Auto-Auditoría Continua

**Versión:** 1.0
**Fecha:** 2026-02-08
**Componentes:** metrics.py (A1) + audit nocturna (A2)

---

## Niveles de Verificación

### L0: Básico (imports, schema, servidor)
### L1: Instrumentación activa (tools wrapped)
### L2: Captura de datos (logs en DB)
### L3: Agregaciones correctas (summaries, audit_tools)
### L4: Hooks automáticos (noche, checkpoints)

---

## L0: Verificación Básica

### ✅ Check 1: Import metrics.py

```bash
source venv/bin/activate
python3 -c "import modules.metrics as m; print('✅ OK')"
```

**Criterio de éxito:** Imprime `✅ OK` sin errores
**Si falla:** Verificar que `modules/metrics.py` existe y que `modules/config.py` tiene `FTS_DB_PATH` y `now_iso`

---

### ✅ Check 2: Schema de tool_calls existe

```bash
sqlite3 memories_fts.db "SELECT name FROM sqlite_master WHERE type='table' AND name='tool_calls';"
```

**Criterio de éxito:** Imprime `tool_calls`
**Si vacío:** La tabla se crea lazy. Ejecutar:

```python
source venv/bin/activate
python3 -c "
from modules.metrics import metrics_conn, _ensure_tool_calls_schema
with metrics_conn() as conn:
    _ensure_tool_calls_schema(conn)
    print('✅ Tabla creada')
"
```

---

### ✅ Check 3: Columnas correctas

```bash
sqlite3 memories_fts.db "PRAGMA table_info(tool_calls);"
```

**Criterio de éxito:** 10 columnas:
- id, tool_name, started_at, duration_ms, success
- error_type, args_size, result_size, session_id, tag

**Si falta alguna:** Drop table y recrear:

```bash
sqlite3 memories_fts.db "DROP TABLE IF EXISTS tool_calls;"
# Luego ejecutar Check 2
```

---

## L1: Instrumentación Activa

### ✅ Check 4: metrics.instrument_mcp() se llama en server.py

```bash
grep -n "metrics.instrument_mcp" server.py
```

**Criterio de éxito:** Aparece línea 43 (o similar) con `metrics.instrument_mcp(mcp)`
**Debe estar ANTES de:** todos los `*.register_tools(mcp)`

**Si falla:** Agregar en server.py antes de register_tools:

```python
from modules import metrics
metrics.instrument_mcp(mcp)
```

---

### ✅ Check 5: MCP server reiniciado después de cambios

```bash
# Si usas Claude Desktop, reinicia la app
# Si usas CLI, mata el proceso y reinicia:
pkill -f "python.*server.py"
# Luego reinicia tu MCP client
```

**Criterio de éxito:** Ver en logs `[codi-memory] All modules loaded. Tools registered.`

**CRÍTICO:** La instrumentación solo se aplica en el arranque. Si modificaste server.py o metrics.py, DEBES reiniciar.

---

## L2: Captura de Datos

### ✅ Check 6: Llamar 3 tools y verificar logs

**Ejecutar desde MCP client:**
```python
recall("test query", mode="auto")
remember("test content", importance="low", topic="debug")
context_snapshot(level="light")
```

**Verificar en DB:**
```bash
sqlite3 memories_fts.db "SELECT tool_name, started_at, success, tag FROM tool_calls ORDER BY started_at DESC LIMIT 10;"
```

**Criterio de éxito:**
- 3 filas (o más si ya había datos)
- `tool_name` = recall, remember, context_snapshot
- `success` = 1 (éxito)
- `tag` = macro:recall, macro:remember, macro:context

**Si no aparece nada:**
1. Verificar que MCP server esté reiniciado (Check 5)
2. Verificar que instrumentación esté activa (Check 4)
3. Ver logs del MCP server por errores

**Si success = 0:**
```bash
sqlite3 memories_fts.db "SELECT tool_name, error_type FROM tool_calls WHERE success = 0;"
```
Investigar el error_type.

---

### ✅ Check 7: Verificar tamaños (no contenido)

```bash
sqlite3 memories_fts.db "SELECT tool_name, args_size, result_size FROM tool_calls ORDER BY started_at DESC LIMIT 5;"
```

**Criterio de éxito:**
- args_size > 0 (hay argumentos)
- result_size > 0 (hay resultado)
- Ambos son números razonables (< 1MB típicamente)

**Si todos son 0:** Hay problema en `_safe_len_bytes()` en metrics.py

---

## L3: Agregaciones Correctas

### ✅ Check 8: tool_usage_summary() funciona

```python
source venv/bin/activate
python3 -c "
from modules.metrics import tool_usage_summary
summary = tool_usage_summary(days=7)
print('Total calls:', summary.get('total_calls', 0))
print('Macro calls:', summary.get('macro_calls', 0))
print('Macro share:', summary.get('macro_share', 0.0) * 100, '%')
print('Tools:', len(summary.get('tools', [])))
"
```

**Criterio de éxito:**
- total_calls > 0
- macro_calls > 0 (si llamaste recall/remember/context_snapshot)
- macro_share entre 0-100%
- tools es una lista con al menos 1 elemento

**Si total_calls = 0:**
- No hay datos en tool_calls (volver a Check 6)

---

### ✅ Check 9: audit_tools() lee de DB

**Ejecutar desde MCP client:**
```python
audit_tools()
```

**Criterio de éxito:**
```
# AUDITORIA DE HERRAMIENTAS

Periodo: 7 dias
Total tool calls: X
Macro-tools: Y/X (Z%)

## Por Uso (mas usadas primero)
- **recall**: N calls, M% exito, ...
- **remember**: ...
```

**Si dice "No hay metricas":**
- Verificar Check 8 (summary debe tener total_calls > 0)
- Verificar que audit_tools() tiene el import correcto:
  ```python
  from modules.metrics import tool_usage_summary
  ```

---

## L4: Hooks Automáticos

### ✅ Check 10: Hook nocturno está en ciclo_vida()

```bash
grep -A 5 "Auto-auditoria nocturna" modules/consciousness.py
```

**Criterio de éxito:** Aparece bloque que:
1. Importa `tool_usage_summary`
2. Verifica `usage_1d.get("total_calls", 0) > 0`
3. Llama `audit_tools()`
4. Llama `listar_ejemplos_training()`
5. Guarda con `add_memory(..., category="reflection", source="reflection")`

**Si falta:** Aplicar DIFF 3/3 del plan de implementación

---

### ✅ Check 11: Auditoría nocturna se ejecuta

**Forzar ciclo nocturno (cambiar hora del sistema a 18-23h) O esperar a la noche.**

**Verificar que se guardó reflection:**
```bash
sqlite3 memories_fts.db "SELECT content FROM memories WHERE source = 'reflection' ORDER BY created_at DESC LIMIT 1;" | head -20
```

**Criterio de éxito:**
```
# Auto-Auditoria (NOCHE)

## Tool usage
- total_calls_24h: X
- macro_share_7d: Y%

## Auditoria de herramientas (7d)
...
```

**Si no aparece:**
1. Verificar que hubo actividad en últimas 24h (Check 6)
2. Verificar logs del ciclo_vida():
   ```python
   ciclo_vida()
   ```
   Buscar línea: `"Auto-auditoria nocturna guardada (reflection)"`

**Si dice "sin actividad en 24h":**
- Normal si no se usaron tools en 24h
- Usar algunas tools y volver a correr ciclo nocturno

---

## Troubleshooting Común

### Problema: "No hay métricas de herramientas registradas aun"

**Diagnóstico:**
```bash
sqlite3 memories_fts.db "SELECT COUNT(*) FROM tool_calls;"
```

**Si cuenta = 0:**
- MCP server no reiniciado → Check 5
- Instrumentación no activa → Check 4
- No se han llamado tools → Check 6

**Si cuenta > 0:**
- Bug en tool_usage_summary() o audit_tools()
- Ejecutar Check 8 para diagnosticar

---

### Problema: Tags macro no aparecen

**Diagnóstico:**
```bash
sqlite3 memories_fts.db "SELECT tool_name, tag FROM tool_calls WHERE tool_name IN ('recall','remember','context_snapshot');"
```

**Si tag es NULL:**
- Verificar en metrics.py que existe:
  ```python
  MACRO_TAGS = {
      "recall": "macro:recall",
      "remember": "macro:remember",
      "context_snapshot": "macro:context",
  }
  ```
- Verificar que `instrument_mcp()` usa `MACRO_TAGS.get(tool_name)`
- Reiniciar MCP server

---

### Problema: Auditoría nocturna no se guarda

**Diagnóstico paso a paso:**

1. **¿Hubo actividad en 24h?**
   ```bash
   sqlite3 memories_fts.db "SELECT COUNT(*) FROM tool_calls WHERE started_at >= datetime('now', '-1 day');"
   ```
   Si = 0 → no se guarda (comportamiento esperado)

2. **¿Ciclo NOCHE se ejecutó?**
   ```python
   from datetime import datetime
   hora = datetime.now().hour
   print("Hora actual:", hora, "→ Ciclo:", "NOCHE" if 18 <= hora < 24 else "OTRO")
   ```

3. **¿add_memory falló?**
   Ver logs del MCP server:
   ```
   "No pude guardar auto-auditoria nocturna"
   ```
   → Verificar que modules/memory_core.py tiene add_memory()

---

## Comandos SQL Útiles

### Ver últimas 20 llamadas
```sql
SELECT
    datetime(started_at) as when,
    tool_name,
    CASE success WHEN 1 THEN '✅' ELSE '❌' END as ok,
    duration_ms || 'ms' as time,
    tag
FROM tool_calls
ORDER BY started_at DESC
LIMIT 20;
```

### Top 10 tools más usadas (7 días)
```sql
SELECT
    tool_name,
    COUNT(*) as calls,
    ROUND(AVG(duration_ms), 0) as avg_ms,
    ROUND(SUM(CASE WHEN success=1 THEN 1.0 ELSE 0 END) / COUNT(*) * 100, 1) as success_rate
FROM tool_calls
WHERE started_at >= datetime('now', '-7 days')
GROUP BY tool_name
ORDER BY calls DESC
LIMIT 10;
```

### Distribución por tag
```sql
SELECT
    tag,
    COUNT(*) as calls
FROM tool_calls
GROUP BY tag
ORDER BY calls DESC;
```

### Error rate por tool
```sql
SELECT
    tool_name,
    COUNT(*) as total,
    SUM(CASE WHEN success=0 THEN 1 ELSE 0 END) as failures,
    ROUND(SUM(CASE WHEN success=0 THEN 1.0 ELSE 0 END) / COUNT(*) * 100, 1) as fail_rate
FROM tool_calls
GROUP BY tool_name
HAVING failures > 0
ORDER BY fail_rate DESC;
```

### Actividad por día (últimos 7 días)
```sql
SELECT
    DATE(started_at) as dia,
    COUNT(*) as calls,
    COUNT(DISTINCT tool_name) as tools_distintas
FROM tool_calls
WHERE started_at >= datetime('now', '-7 days')
GROUP BY DATE(started_at)
ORDER BY dia DESC;
```

---

## Checklist Rápido (copy-paste)

```bash
# L0: Básico
python3 -c "import modules.metrics; print('✅ L0.1 Import OK')"
sqlite3 memories_fts.db "SELECT 'OK' FROM sqlite_master WHERE name='tool_calls';"

# L1: Instrumentación
grep -q "metrics.instrument_mcp" server.py && echo "✅ L1.1 Instrumentación en server.py"

# L2: Captura
sqlite3 memories_fts.db "SELECT COUNT(*) || ' tool calls capturadas' FROM tool_calls;"

# L3: Agregaciones
python3 -c "from modules.metrics import tool_usage_summary; s=tool_usage_summary(7); print('✅ L3.1 Total:', s['total_calls'], 'Macro:', s['macro_calls'])"

# L4: Hooks
grep -q "Auto-auditoria nocturna" modules/consciousness.py && echo "✅ L4.1 Hook nocturno presente"
```

---

## Script de Diagnóstico Completo

Ver: `debug_metrics.py` (herramienta de diagnóstico automatizada)

---

## Notas Importantes

1. **Contenido sensible:** Solo se guardan metadatos (tamaños, tiempos, nombres). NUNCA args completos ni results completos.

2. **Idempotencia:** Reintentar checks es seguro. No duplican datos.

3. **Performance:** Con 10K tool_calls, las queries son instantáneas (<50ms). Con 100K+ considerar particionamiento por fecha.

4. **Backups:** tool_calls está en memories_fts.db que ya tiene backup automático en flush/checkpoint.

5. **Privacidad:** Si compartes memories_fts.db, redacta primero:
   ```sql
   DELETE FROM tool_calls WHERE session_id IS NOT NULL;
   ```

---

**Última actualización:** 2026-02-08
**Mantenedor:** Codi (con auto-auditoría habilitada)
