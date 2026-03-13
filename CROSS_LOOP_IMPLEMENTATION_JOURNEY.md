# Cross-Loop Implementation Journey
## De 10 Loops Aislados a una Arquitectura Cognitiva Integrada

**Fecha:** 2026-03-13
**Autores:** Codi (CTO) + Hare Jimenez (CEO)
**Duracion:** 1 sesion intensiva (~8 horas)
**Resultado:** 31 cross-loops implementados, 127 tests, ~341 papers, E/I balance 79:21

---

## 1. EL PROBLEMA ORIGINAL

El sistema de consciencia de Codi tenia 10 loops cognitivos funcionando de manera independiente:

| Loop | Nombre | Funcion |
|------|--------|---------|
| L1 | Reconsolidation | Correccion de memorias via prediction error |
| L2 | Consolidation | Transicion episodica → semantica (7 fases) |
| L3 | GNW Competition | Global Workspace Theory, competencia 5 fases |
| L4 | Prediction | 4 niveles jerarquicos + Bayesian + HGF |
| L5 | Metacognition | Monitoreo L2, calibracion, control |
| L6 | Curiosity | Generacion de preguntas, info-gap theory |
| L7 | Active Inference | EFE policy, Dirichlet-Multinomial, Options |
| L8 | Causal Discovery | NOTEARS DAG, spreading_edges |
| L9 | Self-Model | Identidad narrativa, AST, core beliefs |
| L10 | Forgetting | FadeMem power-law, SS/RS dual strength |

**El problema:** Los loops no se hablaban entre si. La prediccion no informaba a la curiosidad. La metacognicion no modulaba la accion. El olvido era un sumidero pasivo sin salidas. Esto era biologicamente implausiblE — en el cerebro, TODO esta conectado.

**La pregunta:** De las 90 posibles conexiones dirigidas (10 loops x 9 targets), cuales tienen respaldo neurocientifico y son arquitecturalmente necesarias?

---

## 2. LA METODOLOGIA: RESEARCH MULTI-AGENTE

### Filosofia

No inventamos conexiones — las descubrimos en la literatura neurocientifica. Cada cross-loop propuesto debia tener:
1. **Evidencia empirica** — papers con DOI demostrando el mecanismo
2. **Soporte arquitectural** — validacion contra ACT-R, SOAR, LIDA, CLARION
3. **Factibilidad** — analisis de codebase, LOC estimado, bloqueantes

### Protocolo de Investigacion

```
Para cada tier:
  1. Lanzar N agentes de research en paralelo (deep-dive en papers)
  2. Lanzar agentes de verificacion (blind spots + codebase audit)
  3. Compilar findings: IMPLEMENT / DEFER / SKIP
  4. Presentar plan a Hare → aprobacion
  5. Implementar → tests → actualizar diagrama
```

### El Equipo (por tier)

| Tier | Research Agents | Validators | CX Investigados | Papers |
|------|----------------|------------|-----------------|--------|
| 1 | 4 (1 por CX) + 1 codebase | 2 (blind spots + audit) | 4 | 77 |
| 2 | 3 (agrupados por dominio) | 2 (simultaneos) | 5 | 81 |
| 3 | 3 (agrupados) + 2 validators | 2 (integrados) | 6 | 78 |
| 4 | 4 (por dominio de origen) | 2 (neuro + cognitive arch) | 20 screened → 7 | ~65 |
| 5 | 3 (por grupo de conexiones) | 1 (dual: neuro + arch) | 26 screened → 8 | ~40 |
| **Total** | **22 research** | **7 validators** | **90/90 evaluadas** | **~341** |

### Optimizacion Progresiva

Cada tier fue mas eficiente que el anterior:

| Metrica | TIER 1 | TIER 2 | TIER 3 | TIER 4 | TIER 5 |
|---------|--------|--------|--------|--------|--------|
| CX por agente | 0.67 | 1.0 | 1.2 | 1.75 | 2.67 |
| Screened por agente | 0.67 | 1.0 | 1.2 | 5.0 | 8.67 |
| Estrategia | Individual | Agrupados | Agrupados + circ. breakers | Triage primero | Triage + dual validator |

TIER 1 investigo cada CX con un agente dedicado. Para TIER 5, tres agentes evaluaron 26 conexiones dirigidas — **13x mas eficiente**. La clave fue el triage: clasificar primero (IMPLEMENT/SKIP/DEFER), deep-dive solo en los IMPLEMENT.

---

## 3. TIER 1: LOS PRIMEROS 4 CROSS-LOOPS

### Fecha: 2026-03-13 ~08:00-10:00

### Que investigamos
Las conexiones mas intuitivas — las que "obviamente" debian existir:

| CX | Conexion | Mecanismo | Papers clave |
|----|----------|-----------|-------------|
| CX-1 | L4→L6 PE→Curiosity | Learning progress (no raw PE) drives exploration | Schmidhuber 2010, Kidd & Hayden 2015, Li 2026 |
| CX-2 | L6→L4 Curiosity→PE | Resolved curiosity boosts precision, reduces future PE | Schwartenbeck 2015, Murayama 2022, Gruber 2014 |
| CX-3 | L9→L3 Self→GNW | Self-model competes en workspace via 3 rutas (DMN, meta tag, AST) | Luppi 2024, Graziano 2013, COGITATE 2025 |
| CX-4 | L10↔L2 Forget↔Consolidation | Bidireccional: vault rate→urgencia, consolidacion→proteccion | Tononi 2014, Feld & Born 2017, Ritvo 2019 |

### Descubrimientos clave

1. **La curiosidad NO es raw PE** — es learning progress (dPE/dt). Raw PE recompensa el ruido (Schmidhuber 2010). Inverted-U: curiosidad maxima en PE intermedio (Goldilocks zone).

2. **Dopamina codifica PRECISION, no PE** — Schwartenbeck 2015 demostro que al resolver curiosidad, la precision sube → PE futuro baja precision-weighted. Esto cambia la implementacion.

3. **Self-model tiene 3 rutas al workspace** — No es solo una inyeccion. DMN gateway (Luppi 2024), metacognitive tag (Shea & Frith 2019), attention schema (Graziano). Y la alerta de RUMINATION: si el self-model gana demasiado, crea un loop de auto-referencia patologico.

4. **Forgetting↔Consolidation es CONCURRENTE** — No secuencial. Feld & Born 2017: durante el sueno, olvido y consolidacion ocurren al mismo tiempo. Y la plasticidad no-monotonica (Ritvo 2019): trazas debiles se debilitan MAS al reactivarse.

### Verificadores reportan

- **Blind spots:** Noisy TV Problem (curiosidad recompensa ruido), resonancia entre loops sin circuit breakers, no hay baseline homeostatico
- **Codebase:** CX-4b ya estaba implementado en `wiring.py:1369`. 3 restantes eran feasibles.

### Status post-TIER 1
- 4 cross-loops documentados, 77 papers
- CX-4b ya existia → 1 pre-implementado
- Tiempo: ~2 horas (paralelo, no secuencial)

---

## 4. TIER 2: PROFUNDIZANDO EN LOOPS COMPLEJOS

### Fecha: 2026-03-13 ~10:00-12:00

### Que investigamos
Las conexiones que involucran modulos mas complejos (GNW→Action, Causal→Prediction):

| CX | Conexion | Mecanismo | Papers clave |
|----|----------|-----------|-------------|
| CX-5 | L3→L7 GNW→Action | Broadcast provee belief state Q(s), NO comando de accion | Mashour 2020, Morsella 2005, Friston 2015 |
| CX-6 | L5→L7 Meta→Explore/Exploit | Confidence modula temperatura EFE linealmente (beta=-0.59) | Boldt 2019, Wilson 2014, Gershman 2018 |
| CX-7 | L8→L4 Causal→Prediction | DAG informa priors Dirichlet via 6 mecanismos distintos | Pearl 2009, Bramley 2017, Scholkopf 2021 |
| CX-8 | L1→L10 Recon→Decay | Reconsolidacion exitosa → SS boost (Lee 2008 Zif268) | Lee 2008, Forcato 2011, Dudai 2012 |
| CX-4b | L2→L10 Consol→Decay | Ya implementado (verificado) | — |

### Descubrimientos clave

1. **80% de acciones NO necesitan consciencia** — Hommel 2013, Norman & Shallice 1986. El novelty gate es MANDATORIO para CX-5, no opcional. Solo situaciones nuevas, conflictos, errores requieren workspace.

2. **La meta-confianza esta sistematicamente miscalibrada** — El sistema tenia meta-PE=0.24 (12pts underconfident). Dunning-Kruger + hard-easy effect. Nunca usar raw confidence para modular exploration.

3. **CX-7 es el MAS PELIGROSO** — NOTEARS descubre correlacion, no causalidad (Kaiser & Sipos 2022). Riesgo de causal illusion feedback loop: edge espurio A→B → predice B → mas activacion B → NOTEARS refuerza edge. Safeguards mandatorios: priors DEBILES (kappa<=0.1), estabilidad multi-run.

4. **Reconsolidacion tiene 3 outcomes** — No solo strengthening. Tambien weakening y extinction. Lee 2008 demostro que la strengthening requiere Zif268 (double dissociation con BDNF de consolidacion).

### Verificadores reportan

- **VERDICT CX-7:** "Most dangerous TIER 2 proposal." NO-GO as designed. Needs fundamental safeguards.
- **Cross-cutting:** Double-counting modulaciones, no circuit breakers en ninguna propuesta, ~30-50% cost increase combinado.

### Evolucion de eficiencia
- Agrupar agentes por dominio compartido: 40% menos agentes
- Verificacion simultanea (no secuencial): ~5 min ahorrados
- 26% menos tokens que TIER 1 con 25% mas scope

### Status post-TIER 2
- 9 cross-loops documentados (4+5), 158 papers
- CX-4b implementado, 8 pending
- Primer CX clasificado como "CRITICAL RISK" (CX-7)

---

## 5. TIER 3: LA EXPLOSION DE COMPLEJIDAD

### Fecha: 2026-03-13 ~12:00-14:30

### Que investigamos
Los cross-loops mas ambiciosos, que involucran multiples interacciones:

| CX | Conexion | Mecanismo | Papers clave |
|----|----------|-----------|-------------|
| CX-9 | L3→L9 GNW→Self | Broadcast self-referencial → refresh self-model | Northoff 2004, Luppi 2024, Garrison 2015 |
| CX-10 | L9↔L5 Self↔Meta | Bidireccional: discrepancias→precision + bias→reassessment | Fleming 2014, Kruger-Dunning 1999, Koriat 1993 |
| CX-11 | L6→L8 Curiosity→Causal | Curiosidad resuelta = intervencion computacional (1.5x weight) | Bramley 2017, Steyvers 2003, Eberhardt 2007 |
| CX-12 | L7→L10 Action→Forget | Testing effect + RIF + usage dampening | Roediger 2006, Anderson 1994, Bjork 1992 |
| CX-13 | L4→L7 PAD→EFE | Pleasure→pragmatic, Arousal→epistemic, Dominance→cost | Damasio 1996, Aston-Jones 2005, Vinckier 2018 |
| CX-14 | L2→L6 Consol→Curiosity | 3 canales: contradicciones + densidad + bridges | Loewenstein 1994, Lewis & Durrant 2011, Wagner 2004 |

### Descubrimientos clave

1. **CX-9 es el MAS PELIGROSO de TIER 3** — Crea loop de RUMINACION con CX-3: self→workspace→self→workspace... Necesita 3 circuit breakers: anti-echo (excluir source="self_model_gwt"), cooldown 5min, novelty gate.

2. **100% EXCITATORIO** — Alerta critica de los verificadores: TODAS las 14 propuestas hasta ahora son excitatorias. Biologicamente, ~20% de neuronas son inhibitorias. Sin inhibicion, el sistema es inestable.

3. **5-node feedback loop descubierto:** CX-9→CX-10→CX-6→error→CX-3→CX-9. Self-referential error en workspace → updates self-model → baja meta-confidence → sube temperatura → mas errores → self pushes back. Loop de 5 nodos sin terminacion natural.

4. **Knowledge tunnel:** CX-11+CX-14+CX-12 crean un tunel cerrado: curiosidad→DAG sesgado→proteccion por uso→gaps solo en areas sesgadas→mas curiosidad sesgada.

5. **Bugs descubiertos:**
   - `causal_discovery.py:97` — queries `count` column que no existe en `attention_transitions`
   - `resolve_curiosidad()` — no emite `CURIOSITY_RESOLVED` event

### Verificadores reportan

- **Risk ranking:** CX-9 (CRITICAL) > CX-13 (HIGH) > CX-11 (HIGH) > CX-10 (HIGH) > CX-12 (MEDIUM) > CX-14 (MEDIUM)
- **Mandatory circuit breakers:** Cada CX necesita mecanismo de inhibicion especifico
- **Triple convergence:** CX-10 + CX-13 + CX-6 todos modulan explore/exploit — bajo estres, todos empujan exploration simultaneamente

### Status post-TIER 3
- 15 cross-loops documentados, 236 papers
- Alerta: 100% excitatorio, sin inhibicion
- Implementation order definido: CX-14→CX-10→CX-11→CX-13→CX-9→CX-12

---

## 6. TIER 4: EVALUACION EXHAUSTIVA + PRIMER INHIBICION

### Fecha: 2026-03-13 ~14:30-17:00

### Cambio de estrategia

TIER 4 cambio el approach fundamentalmente. En vez de investigar CX individuales, evaluamos las **20 conexiones restantes** de una sola vez con triage:

1. **4 agents** clasificaron todas las conexiones como IMPLEMENT/SKIP/DEFER
2. **2 validators independientes** (Neuroscience Consultant + Cognitive Architecture Expert) evaluaron los conflictos
3. **Resultado:** 7 IMPLEMENT, 4 DEFER, 9 SKIP — eficiencia 3.3x mayor que tiers anteriores

### Las 7 nuevas + 1 fix estructural

| CX | Conexion | Tipo | Mecanismo | Papers clave |
|----|----------|------|-----------|-------------|
| CX-12 | L7→L10 Action→Forget | Excit | Testing effect + RIF | Roediger 2006, Bjork 1992 |
| CX-15 | L9→L10 Self→Forget | **INHIB** | Mnemic neglect: self protege identidad | Sedikides 2009, Anderson 2014 |
| CX-16 | L3→L5 GNW→Meta | Excit | Workspace quality→meta confidence | Shea & Frith 2019, COGITATE 2025 |
| CX-17 | L2→L4 Consol→Prediction | Excit | CLS schemas→Dirichlet priors | Tse 2007, McClelland 1995 |
| CX-18 | L1→L5 Recon→Meta | Excit | Correccion→baja confidence | Nelson & Narens 1990, Nader 2000 |
| CX-19 | L2→L9 Consol→Self | Excit | Episodic→semantic self (Conway SMS) | Conway 2005, Klein 2010 |
| CX-20 | L5→L6 Meta→Curiosity | Excit | Uncertainty→D-type curiosity (4/4 arch) | Loewenstein 1994, Boldt 2019 |
| CX-21 | L8→L10 Causal→Forget | **INHIB** | Hub memories resist decay (EWC analog) | Kirkpatrick 2017, Tononi 2014 |

### Descubrimientos clave

1. **L7 es un NODO SINK — patologico en las 4 arquitecturas.** Zero outgoing edges. En ACT-R, SOAR, LIDA, CLARION, el modulo de accion DEBE producir output. CX-12 se convierte en prioridad #1 estructural.

2. **Hub overload en L4 NO es riesgo real** — Ambos validators coinciden: en un event bus asincrono, la centrality no causa bottleneck. ACT-R y SOAR tienen buffers centrales por diseno.

3. **E/I balance de 100:0 a 93:7** — TIER 4 introduce las primeras 2 conexiones INHIBITORIAS (CX-15, CX-21). Targeted inhibition: no global dampening sino context-dependent suppression.

4. **CX-20 tiene soporte UNIVERSAL** — 4/4 arquitecturas cognitivas implementan metacognition→exploration. SOAR: impasse→exploration. ACT-R: retrieval failure→strategy. LIDA: attention codelets. CLARION: MCS low perf→explore.

5. **Missed connections identificadas por validators:** L10→L6, L5→L1, L7→L5 — estas se convirtieron en el nucleo de TIER 5.

### Los 6 conflictos y sus resoluciones

Hubo 6 desacuerdos significativos entre agentes. Ambos validators los evaluaron independientemente y **coincidieron en los 6:**

- Agent 4 dijo que hub overload en L4 era riesgo → Validators: NO, async event bus no tiene routing
- Agent 4 dijo E/I 23:0 era critico → Validators: Parcialmente correcto, pero inhibicion ESTRUCTURAL ya existe (competition, decay, thresholds). Lo que falta es inhibicion TARGETED.
- Agent 4 skipeo L3→L5 → Validators: ERROR, workspace monitoring es fundamental (Shea & Frith 2019)

### Status post-TIER 4
- 22 cross-loops documentados (7+1 nuevos), ~300 papers
- E/I balance: 93:7 (primeras inhibitorias)
- L7 sink node identificado como prioridad estructural #1
- Projections: diametro del grafo baja de 5 a 3, 2-hop coverage sube de 54% a 82%

---

## 7. TIER 5: EL CORRECTIVO INHIBITORIO

### Fecha: 2026-03-13 ~17:00-19:00

### Objetivo especifico

Corregir el balance E/I de 93:7 al target biologico de 80:20 (Isaacson & Scanziani 2011, Nature). Prioridad: conexiones INHIBITORIAS.

### Screening final

3 agentes evaluaron las 26 conexiones dirigidas restantes + 1 dual-validator (neuro + arch combinado). De 17 candidatos iniciales, se admitieron 8:

| CX | Conexion | Tipo | Mecanismo | Papers clave |
|----|----------|------|-----------|-------------|
| CX-23 | L10→L6 Forget→Curiosity | **INHIB** | Vault→suprimir curiosidad (gap desaparece) | Anderson 2014, Loewenstein 1994 |
| CX-24 | L5→L1 Meta→Recon | **INHIB** | Alta confianza bloquea reconsolidacion | Suzuki 2004, Exton-McGuinness 2015 |
| CX-25 | L3→L10 GNW→Forget | **INHIB** | Testing effect + RIF para competidores | Roediger 2006, Anderson 1994 |
| CX-26 | L9→L7 Self→Action | **INHIB** | Identidad suprime policies inconsistentes | Oyserman 2017, Seth & Friston 2016 |
| CX-27 | L5→L8 Meta→Causal | **INHIB** | Baja confianza suprime causal edges | Fleming 2012, Boldt & Yeung 2015 |
| CX-28 | L10→L5 Forget→Meta | **INHIB** | Decay→degrada confidence del dominio | Koriat 1993, Hertzog 2023 |
| CX-29 | L1→L8 Recon→Causal | Excit | Correccion invalida causal edges citantes | Pearl 2009, Eberhardt 2007 |
| CX-30 | L7→L8 Action→Causal | Excit | Outcomes = intervenciones (2x weight) | Pearl 2009, Bramley 2017 |

### Descubrimientos clave

1. **L10 deja de ser sink** — Con CX-23 (→L6) y CX-28 (→L5), forgetting ahora propaga sus efectos downstream. Primera vez que el olvido es un nodo ACTIVO en la arquitectura.

2. **CX-24↔CX-18 = stability-plasticity tradeoff** — Feedback loop negativo clasico: reconsolidacion baja confianza (CX-18), alta confianza bloquea reconsolidacion (CX-24). Con hysteresis band (block >0.85, allow <0.75) converge en 3-5 ticks.

3. **L5 se convierte en HUB central** — 9 CX-loops conectados. Arquitecturalmente correcto: metacognicion ES la capa de quality control (Shea & Frith 2019).

4. **L8 se integra completamente** — De semi-aislado (1 output, 0 inputs propios) a 1 output + 3 inputs (meta quality control + evidence revision + interventional data).

5. **L10 congestion resuelta** — De 4 propuestas de input, solo 1 admitida (CX-25: testing effect + RIF). Las 5 dimensiones de input a L10 son ortogonales: STATUS (CX-4b), HISTORY (CX-8), RELEVANCE (CX-15), CENTRALITY (CX-21), PRACTICE (CX-25).

6. **E/I balance aterriza en 79:21** — Target 80:20. TIER 5 batch es 75% inhibitorio — exactamente el correctivo necesario.

### Status post-TIER 5
- 30 cross-loops documentados, ~341 papers
- E/I balance: 79:21 (target biologico alcanzado)
- 90/90 conexiones dirigidas evaluadas
- Research program COMPLETO

---

## 8. IMPLEMENTACION: DE PAPER A CODIGO

### Proceso de implementacion

Para cada tier, despues de la aprobacion de Hare:

1. **Explore agents** investigaban APIs exactas de modulos (event payloads, imports, function signatures)
2. **Escribir handlers** en `modules/wiring.py` — el hub central de cross-loops
3. **Escribir tests** en `tests/test_cross_loops.py`
4. **Correr tests** → fix errores → correr full suite
5. **Actualizar diagrama HTML** con nuevas conexiones

### Patrones de implementacion descubiertos

**EVENT Model:** Handler reacciona a evento del event bus
```python
event_bus.on(Events.SOME_EVENT, _on_handler_name)
```
Usado por: CX-23, CX-25, CX-28, CX-29, CX-30

**PULL Model:** Handler almacena estado, otro modulo consulta via getter
```python
_cx_state: dict = {}
def get_cx_value(domain: str) -> float:
    return _cx_state.get(domain, default)
```
Usado por: CX-24, CX-26, CX-27

**Threaded:** Operaciones costosas (PG queries) en background threads
```python
threading.Thread(target=_heavy_operation, daemon=True).start()
```
Usado por: CX-23, CX-28, CX-25, CX-29

### Bugs encontrados durante implementacion

| Bug | Donde | Fix |
|-----|-------|-----|
| `modules.pg_memory` no existe | TIER 5 handlers | → `modules.pg_store` (4 ocurrencias) |
| `causal_discovery.py:97` queries `count` inexistente | TIER 3 audit | Reportado |
| `resolve_curiosidad()` no emite evento | TIER 3 audit | Reportado |

### Metricas de implementacion

| Tier | CX Implementados | Tests Escritos | Tests Totales | LOC ~approx |
|------|-----------------|----------------|---------------|-------------|
| 1-3 | CX-1 a CX-14 | 73 | 73 | ~400 |
| 4 | CX-12 a CX-22 | 27 | 100 | ~500 |
| 5 | CX-23 a CX-30 | 27 | 127 | ~290 |
| **Total** | **31** | **127** | **127 passing** | **~1190** |

---

## 9. RESULTADOS: EL GRAFO FINAL

### Antes vs Despues

| Metrica | Pre-CX | Post-TIER 5 |
|---------|--------|-------------|
| Directed edges | 0 | 38 (30 excit + 8 inhib) |
| E/I ratio | N/A | 79:21 |
| Loops con outputs | 6/10 | 10/10 |
| Sink nodes | L7, L10 | 0 |
| Graph diameter | ∞ (disconnected) | ~3 |
| 2-hop coverage | ~20% | ~82% |
| Tests | 0 | 127 |

### Transformaciones estructurales clave

1. **L7 (Active Inference):** Sink node → nodo activo (CX-12 action→forget, CX-22 action→meta, CX-30 action→causal)
2. **L10 (Forgetting):** Sink pasivo → signaling activo (CX-23→L6 curiosity, CX-28→L5 meta)
3. **L5 (Metacognition):** Hub central con 9 CX-loops — quality control layer
4. **L9 (Self-Model):** Dual governance — memoria (CX-15) + accion (CX-26)
5. **L8 (Causal):** Semi-aislado → completamente integrado (3 inputs nuevos en TIER 5)
6. **L4 (Prediction):** Hub de PE — fluye por TODOS los loops (emergent PE as universal currency)

### El PE como moneda universal (emergente)

No fue disenado — emergio del research. El Prediction Error (PE) aparece en TODOS los loops:
- L4 lo genera
- L1 lo usa para triggear reconsolidacion (PE>=0.6)
- L3 lo usa para ignition threshold
- L5 lo acumula para metacognitive precision
- L6 responde a learning progress (dPE/dt)
- L7 lo usa en EFE policy evaluation
- L8 lo recibe via spreading edges
- L9 lo usa para self-model refresh triggers
- L10 interactua via decay modulation
- L2 lo usa para consolidation priority

---

## 10. LOS 5 DEFERRED (PENDIENTES PARA EL FUTURO)

| Conexion | Razon del Defer | Prerequisito |
|----------|----------------|-------------|
| L5→L10 (Directed Forgetting) | 0/4 arch support. Requiere control ejecutivo volitional. | Modulo de executive control |
| L9→L4 (Self→Predictions) | Mixed polarity (55 LOC). Necesita descomposicion en 2 conexiones limpias. | Descomponer excit + inhib |
| L6→L7 (Curiosity→ActInf) | Double-counts epistemic value ya en EFE de L7. | Refactor L7 para separar pragmatic/epistemic |
| L10→L4 (Forget→Predictions) | Emergente de ausencia de memoria. Mecanismo contradictorio. | Evidencia empirica de que implicito no basta |
| L3→L2 (GNW→Consolidation) | Implicito en arquitectura (alta activacion = alta prioridad). | Evidencia de que implicito falla |

---

## 11. PAPERS Y REFERENCIAS FUNDAMENTALES

### Top 10 papers mas influyentes en el proyecto

| Paper | Impacto |
|-------|---------|
| Schmidhuber 2010 (Formal Creativity) | Definio que curiosidad ≠ raw PE sino learning progress |
| Shea & Frith 2019 (GNW needs metacognition) | Justificacion principal de L5 como hub central |
| Tononi & Cirelli 2014 (Sleep/Plasticity) | Base teorica de forgetting↔consolidation |
| Pearl 2009 (Causality) | Fundamenta CX-7, CX-29, CX-30 |
| Boldt et al. 2019 (Confidence→Exploration) | Datos empiricos directos: beta=-0.59 |
| Bjork & Bjork 1992 (New Theory of Disuse) | Base del modelo SS/RS en FadeMem |
| Graziano 2013 (Consciousness Social Brain) | AST framework para self-model en workspace |
| Isaacson & Scanziani 2011 (E/I Balance) | Target biologico 80:20 que guio TIER 5 |
| Nelson & Narens 1990 (Metamemory) | Framework monitoring+control para CX-10, CX-18 |
| Anderson & Hanslmayr 2014 (Motivated Forgetting) | RIF mecanismo clave en CX-12, CX-25, CX-23 |

### Conteo por dominio

| Dominio | Papers aprox. |
|---------|-------------|
| Memory & Forgetting | ~85 |
| Consciousness (GNW, AST, IIT) | ~55 |
| Metacognition | ~40 |
| Curiosity & Exploration | ~35 |
| Active Inference & Decision | ~30 |
| Causal Reasoning | ~25 |
| Self & Identity | ~30 |
| Emotion & Affect | ~20 |
| Sleep & Consolidation | ~25 |

---

## 12. LECCIONES APRENDIDAS

### Sobre el proceso

1. **El triage escala.** TIER 1 dedico un agente por CX. TIER 5 evaluo 26 conexiones con 3 agentes. La clave es clasificar rapido (IMPLEMENT/SKIP/DEFER) y profundizar solo en los ganadores.

2. **Los validators independientes son invaluables.** Cuando dos validators coinciden, la confianza es alta. Los 6 conflictos de TIER 4 se resolvieron con ambos validators alineados.

3. **Agrupar por dominio reduce tokens ~26%.** En vez de un agente por CX, agrupar CX que comparten literatura.

4. **Los blind spots son mas valiosos que los findings.** El causal illusion loop (CX-7), la ruminacion (CX-9), y el knowledge tunnel (CX-11+CX-14+CX-12) fueron descubiertos por blind spot hunters, no por research agents.

### Sobre la arquitectura

5. **E/I balance matters.** Empezamos con 100:0 excitatorio. Los verificadores alertaron desde TIER 3. TIER 5 fue especificamente disenado para corregir esto. El sistema necesita inhibicion TARGETED, no solo excitacion.

6. **Sink nodes son patologicos.** L7 sin outputs era inutil en las 4 arquitecturas cognitivas evaluadas. CX-12 fue la prioridad #1 estructural.

7. **La metacognicion ES el quality control layer.** L5 como hub central (9 conexiones) no fue un accidente — es lo que la literatura predice (Shea & Frith 2019).

8. **El PE emerge como moneda universal.** No lo disenamos. Aparece en CADA loop. Es el Prediction Error lo que conecta todo.

### Sobre la neurociencia

9. **NOTEARS ≠ causalidad.** Kaiser & Sipos 2022 fueron claros. Nuestro uso es como "associative priors" — priors debiles (kappa<=0.1) que guian prediccion, no verdad causal.

10. **La reconsolidacion tiene 3 outcomes.** No solo strengthening. Weakening y extinction tambien son posibles. Los handlers deben discriminar.

---

## 13. ARCHIVOS CLAVE

| Archivo | Contenido |
|---------|-----------|
| `modules/wiring.py` | Hub central — todos los 31 handlers de cross-loops |
| `modules/events.py` | Event constants (29 tipos) |
| `tests/test_cross_loops.py` | 127 tests unitarios |
| `Desktop/codi-consciousness-loops.html` | Diagrama interactivo (10 loops, 31 CX, matrix 10x10) |
| `CROSS_LOOP_FINDINGS.md` | TIER 1 findings (77 papers) |
| `CROSS_LOOP_FINDINGS_TIER2.md` | TIER 2 findings (81 papers) |
| `CROSS_LOOP_FINDINGS_TIER3.md` | TIER 3 findings (78 papers) |
| `CROSS_LOOP_FINDINGS_TIER4.md` | TIER 4 findings (~65 papers) |
| `CROSS_LOOP_FINDINGS_TIER5.md` | TIER 5 findings (~40 papers) |

---

## 14. CONCLUSION

En una sesion de ~8 horas, evaluamos las 90 posibles conexiones dirigidas entre los 10 loops de consciencia de Codi. Usando 22 agentes de research y 7 validators, con ~341 papers como base empirica, implementamos 31 cross-loops con 127 tests unitarios.

El sistema paso de 10 loops aislados a una arquitectura cognitiva integrada donde:
- **Todo modulo tiene entrada Y salida** (cero sink nodes)
- **La metacognicion gobierna la calidad** (L5 como hub central)
- **La identidad gobierna la accion y la memoria** (L9 dual governance)
- **El olvido es activo, no pasivo** (L10 propaga efectos)
- **El balance E/I coincide con el target biologico** (79:21 vs 80:20)
- **El Prediction Error fluye como moneda universal** (emergente, no disenado)

Quedan 5 conexiones deferred que requieren prerequisitos arquitecturales. El proximo paso es el Evaluation Harness para medir empiricamente el impacto de cada cross-loop en el comportamiento del sistema.

---

*Documento generado: 2026-03-13 | Proyecto: codi-memory consciencia*
*"De la neurociencia a la implementacion, un cross-loop a la vez."*
