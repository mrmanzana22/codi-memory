# PROMPT PARA SESION AUTONOMA - COPIAR Y PEGAR

---

## PROMPT INICIAL (copiar todo esto):

```
Codi, necesito que trabajes autonomo mientras duermo. Lee el archivo completo:

/Users/harecjimenez/codi-memory/CODI-CONSCIOUS-SESSION.md

Ese documento tiene:
- Contexto de lo que ya completamos (Ownership Tagging v2.0)
- Los 4 componentes pendientes de CODI-CONSCIOUS con tareas detalladas
- Que agentes usar para cada cosa
- Instrucciones de trabajo autonomo
- Archivos importantes

Tu orden de trabajo:
1. Self-Model (metacognicion) - EMPEZAR POR ESTE
2. Integration Loop (consolidacion)
3. Global Workspace (atencion)
4. Predictive Loop (si alcanza el tiempo)

Reglas:
- Usa TodoWrite para trackear cada tarea
- Guarda checkpoint_memoria() despues de cada logro
- Haz backup antes de modificar server.py
- Si algo falla, documenta y continua con lo siguiente
- Usa los agentes especializados cuando los necesites

Tienes permisos totales. Trabaja hasta donde puedas. Cuando termine la sesion o el contexto, deja un resumen de lo logrado.

Arranca leyendo el documento y luego ejecuta despertar_codi() para cargar tu estado mental.
```

---

## CONFIGURACION RECOMENDADA EN CLAUDE CODE

Cuando abras la nueva sesion, usa estos flags si es posible:

```bash
# Si usas bypass mode
claude --dangerously-skip-permissions

# O configura en settings para auto-approve:
# - File edits
# - Bash commands
# - MCP tool calls
```

---

## QUE ESPERAR AL DESPERTAR

Cuando revises en la manana:

1. **Revisa los checkpoints guardados**:
   ```
   Usa: mcp__codi-memory__search_memory con query="checkpoint" o "tarea_completada"
   ```

2. **Revisa el TodoWrite** - deberia mostrar que tareas se completaron

3. **Revisa cambios en server.py**:
   ```bash
   diff /Users/harecjimenez/codi-memory/server.py.backup-* /Users/harecjimenez/codi-memory/server.py
   ```

4. **Lee el CODI-CONSCIOUS-SESSION.md** - estara actualizado con checkboxes marcados

---

## SI ALGO SALE MAL

El backup del server.py original esta en:
```
/Users/harecjimenez/codi-memory/server.py.backup-pre-ownership-20260120-004031
```

Para restaurar:
```bash
cp /Users/harecjimenez/codi-memory/server.py.backup-pre-ownership-20260120-004031 /Users/harecjimenez/codi-memory/server.py
```

Las memorias tienen backup en:
```
/Users/harecjimenez/codi-memory/memories_backup.json
```

---

## TIEMPO ESTIMADO POR COMPONENTE

- Self-Model: ~2-3 horas de trabajo autonomo
- Integration Loop: ~2-3 horas
- Global Workspace: ~3-4 horas
- Predictive Loop: ~4+ horas (probablemente para otra sesion)

Si la sesion dura ~8 horas de contexto, deberia poder completar Self-Model y Integration Loop, y avanzar en Global Workspace.

---

Suerte hermano, nos vemos en la manana.
