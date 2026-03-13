# Cross-Loop Research Brief
> Para terminal de investigacion. NO editar codigo — solo investigar y documentar.

## Estado Actual (2026-03-13)
- 10 loops de consciencia identificados (6 operativos, 4 missing)
- 23/23 eventos cableados (0 orphans)
- 9/45 conexiones cross-loop activas (20%)
- **36 conexiones faltantes** — necesitamos base teorica para las 15 mas criticas

## Los 10 Loops
| Loop | Nombre | Estado | Modulos |
|------|--------|--------|---------|
| L1 | Reconsolidation | OK | prediction.py, consolidation.py |
| L2 | Consolidation→Semantic | OK | consolidation.py, semantic_store.py |
| L3 | GNW + Attention | OK | competition.py, working_memory.py |
| L4 | Prediction→Emotion→Precision | OK | prediction.py, emotion.py |
| L5 | Metacognition→Control | OK | prediction.py L2, self_model.py |
| L6 | Curiosity→Learning | WM only | sleep_loop.py curiosity ticks |
| L7 | Active Inference→Outcome | WM only | active_inference.py |
| L8 | Causal DAG→Spreading | OK | causal_discovery.py, spreading.py |
| L9 | Self-Model→Identity | WM only | self_model.py |
| L10 | Forgetting→Homeostasis | Log only | fade_mem.py, workspace.py |

## Cross-Loops que Necesitan Investigacion

### TIER 1 — Maxima prioridad (mayor impacto)

**CX-1: L4→L6 (Prediction Error drives Curiosity)**
- Pregunta: Como deberia PE alto en un dominio generar preguntas de curiosidad?
- Hipotesis: PE > threshold en topic X → generar pregunta sobre X
- Buscar: Berlyne (1960) curiosity, Gottlieb (2013) information gain, Kidd & Hayden (2015)
- Implementacion esperada: handler en wiring.py que al recibir PREDICTION_ERROR con surprise > 0.6, empuja pregunta a curiosidad queue

**CX-2: L6→L4 (Resolved Curiosity reduces PE)**
- Pregunta: Cuando curiosidad se resuelve, como actualiza el modelo de prediccion?
- Hipotesis: respuesta almacenada reduce PE futuro en ese topic
- Buscar: Loewenstein (1994) information gap, Gruber (2014) curiosity + memory
- Implementacion esperada: al resolver curiosidad, marcar topic como "explored" en prediction context

**CX-3: L9↔L3 (Self-Model in Global Workspace)**
- Pregunta: Como deberia self-knowledge competir en GNW?
- Hipotesis: self-model genera candidatos que compiten en workspace
- Buscar: Graziano (2013) Attention Schema Theory, Cleeremans (2011) radical plasticity
- Implementacion esperada: self_model refresh → inject self-summary as GNW candidate

**CX-4: L10↔L2 (Forgetting ↔ Consolidation feedback)**
- Pregunta: Como deberia la tasa de olvido informar la selectividad de consolidacion?
- Hipotesis: alta tasa vault → consolidacion mas agresiva en proteger importante
- Buscar: Bjork (1992) new theory of disuse, Hardt (2013) decay vs interference, Wixted (2004)
- Implementacion esperada: health_monitor trackea vault_rate → consolidation ajusta lookback/selectivity

### TIER 2 — Alta prioridad

**CX-5: L3→L7 (GNW broadcast → Action Selection)**
- Pregunta: Como deberia el workspace broadcast informar que accion tomar?
- Buscar: Dehaene (2014) workspace→motor, Baars (2002) consciousness and action

**CX-6: L5→L7 (Metacognition → Explore/Exploit)**
- Pregunta: Como deberia meta-confianza modular exploracion vs explotacion?
- Buscar: Daw (2006) explore-exploit, Meyniel (2015) confidence-based switching

**CX-7: L8→L4 (Causal DAG → Prediction accuracy)**
- Pregunta: Como deberia conocimiento causal mejorar predicciones?
- Buscar: Pearl (2009) causality, Sloman (2005) causal models in reasoning

**CX-8: L1→L10 (Reconsolidation protects from decay)**
- Pregunta: Memorias reconsolidadas deberian ser protegidas del olvido?
- Buscar: Nader (2000) + Lee (2009) reconsolidation strengthens, Dudai (2012)

### TIER 3 — Importante

**CX-9: L6→L8 (Curiosity feeds causal discovery)**
**CX-10: L7→L10 (Action outcomes influence forgetting)**
**CX-11: L9↔L5 (Self-model feeds metacognition)**
**CX-12: L4→L1 (Emotion modulates reconsolidation threshold)**

## Formato de Respuesta Esperado

Para cada cross-loop, documentar:
```
### CX-N: LX→LY (nombre)
**Papers:** [autor, ano, titulo, DOI si posible]
**Mecanismo:** descripcion computacional en 2-3 lineas
**Evidencia:** que dice la neurociencia sobre esta conexion
**Implementacion minima:** que handler/cambio en wiring.py, ~lineas
**Riesgo:** que podria salir mal
**Test:** como verificar que funciona
```

## Recursos Locales
- KNOWLEDGE_CANON_v4.md: ~/codi-daemon/study/KNOWLEDGE_CANON_v4.md
- Neuro Skill: ~/.claude/projects/-Users-codi-air/memory/neuro-skill.md
- IMPLEMENTATION_PLAYBOOK_v3.md: ~/codi-daemon/study/IMPLEMENTATION_PLAYBOOK_v3.md
- AUDIT_MASTER_v2.md: ~/codi-daemon/study/AUDIT_MASTER_v2.md
