# Cross-Loop Research Brief — TIER 2
> Para terminal de investigacion. NO editar codigo — solo investigar y documentar.

## Estado Actual (2026-03-13)
- TIER 1 COMPLETE: CX-1, CX-2, CX-3, CX-4a implementados (4 cross-loops)
- 13/45 conexiones activas (29%)
- **32 conexiones faltantes** — TIER 2 tiene las 4 mas importantes restantes

## Lo que ya sabemos (de TIER 1)
- CX-1: PE→Curiosity via learning progress, Goldilocks zone (Schmidhuber 2010, Kidd 2015)
- CX-2: Curiosity resolution → precision boost (Schwartenbeck 2015, Gruber 2019)
- CX-3: Self-model → GNW via DMN gateway + confidence tag (Graziano 2013, Luppi 2024)
- CX-4a: Vault rate → consolidation urgency (Stickgold & Walker 2013, Feld & Born 2017)

## Cross-Loops TIER 2 que Necesitan Investigacion

### CX-5: L3→L7 (GNW broadcast → Action Selection)
- Pregunta: Como deberia el workspace broadcast informar que accion tomar en active_inference?
- Contexto: active_inference.py tiene EFE policy selection con Options Framework (4 canonical options). El workspace broadcast deberia proveer contexto para seleccion de politica.
- Modulos: competition.py (broadcast output) → active_inference.py (policy selection)
- Buscar: Dehaene (2014) workspace→motor coupling, Baars (2002) consciousness and action, Mashour (2020), Clark (2016) action-oriented predictive processing
- Implementacion esperada: workspace winners feed into active_inference context for next EFE computation

### CX-6: L5→L7 (Metacognition → Explore/Exploit)
- Pregunta: Como deberia meta-confianza modular la balance exploracion vs explotacion?
- Contexto: prediction.py L2 computa metacognitive confidence (FoK scores). active_inference.py tiene explore/exploit phases via IG threshold. Meta-confidence deberia modular ese threshold.
- Modulos: prediction.py L2 (meta-confidence) → active_inference.py (IG threshold)
- Buscar: Daw et al. (2006) explore-exploit dopamine, Meyniel et al. (2015) confidence-based switching, Cohen et al. (2007) prefrontal explore-exploit, Badre et al. (2012) rostrolateral PFC hierarchy
- Implementacion esperada: high meta-confidence → lower IG threshold (exploit more), low meta-confidence → higher IG threshold (explore more)

### CX-7: L8→L4 (Causal DAG → Prediction accuracy)
- Pregunta: Como deberia el conocimiento causal (NOTEARS DAG) mejorar las predicciones?
- Contexto: causal_discovery.py descubre DAG via NOTEARS → spreading_edges. prediction.py computa topic transitions via Dirichlet-Multinomial. El DAG deberia mejorar la prediccion de topic transitions.
- Modulos: causal_discovery.py (DAG edges) → prediction.py (transition priors)
- Buscar: Pearl (2009) causality and prediction, Sloman (2005) causal models in reasoning, Waldmann & Holyoak (1992) predictive vs diagnostic reasoning, Bramley et al. (2017) causal learning from interventions
- Implementacion esperada: DAG edges with high weight boost transition_stats for those topic pairs → better predictions

### CX-8: L1→L10 (Reconsolidation protects from decay)
- Pregunta: Memorias que pasaron por reconsolidacion deberian ser protegidas del olvido?
- Contexto: reconsolidation marca memorias como labile → corrige → restabiliza. Despues de restabilizacion exitosa, la memoria deberia tener mayor resistencia a decay (SS boost in Bjork model).
- Modulos: consolidation.py (reconsolidation pipeline) → forgetting.py (FadeMem decay rates)
- Buscar: Nader (2000) reconsolidation, Lee (2009) reconsolidation strengthens, Dudai (2012) restabilization, Agren (2014) reconsolidation in humans, Exton-McGuinness et al. (2015) boundary conditions
- Implementacion esperada: after successful reconsolidation, boost SS (storage strength) in forgetting.py → reduced decay rate

### CX-4b: L2→L10 (Consolidation protects from decay)
- Pregunta: Memorias exitosamente consolidadas deberian tener menor tasa de decay?
- Contexto: Ya tenemos CX-4a (vault→urgency). CX-4b es la direccion inversa: consolidacion exitosa → proteccion de decay. Esto cierra el loop bidireccional.
- Modulos: consolidation.py (_phase_pruning) → forgetting.py (FadeMem)
- Buscar: Frey & Morris (1997) synaptic tagging, Tononi & Cirelli (2014) SHY downscaling survivors, Benna & Fusi (2016) cascade consolidation, Squire (1992) hippocampal→neocortical transfer
- Nota: Ya tenemos papers de TIER 1 CX-4. Solo necesitamos el mecanismo exacto de proteccion.
- Implementacion esperada: after consolidation success, call forgetting.protect_from_decay(memory_id, factor=0.5)

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
- CROSS_LOOP_FINDINGS.md: ~/codi-memory/CROSS_LOOP_FINDINGS.md (TIER 1 completo)
- CROSS_LOOP_RESEARCH.md: ~/codi-memory/CROSS_LOOP_RESEARCH.md (brief original)
- KNOWLEDGE_CANON_v4.md: ~/codi-daemon/study/KNOWLEDGE_CANON_v4.md
- Neuro Skill: ~/.claude/projects/-Users-codi-air/memory/neuro-skill.md
- IMPLEMENTATION_PLAYBOOK_v3.md: ~/codi-daemon/study/IMPLEMENTATION_PLAYBOOK_v3.md

## Modulos Clave (para el auditor de codebase)
- ~/codi-memory/modules/active_inference.py — EFE, Options Framework, Dirichlet
- ~/codi-memory/modules/prediction.py — 4-level hierarchical, Bayesian, HGF
- ~/codi-memory/modules/causal_discovery.py — NOTEARS, spreading_edges
- ~/codi-memory/modules/consolidation.py — 7-phase pipeline, reconsolidation
- ~/codi-memory/modules/forgetting.py — FadeMem, SS/RS, power-law decay, RIF
- ~/codi-memory/modules/competition.py — 5-phase GNW competition
- ~/codi-memory/modules/wiring.py — thalamocortical layer, all handlers

## Guardar resultados en:
~/codi-memory/CROSS_LOOP_FINDINGS_TIER2.md
