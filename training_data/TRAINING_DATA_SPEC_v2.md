# Codi Training Data Specification v2
> Data-Centric AI approach: quality > quantity, iterate > batch

## Hallazgos de Auditoria (v1)

| Problema | Severidad | Solucion |
|----------|-----------|----------|
| system_expert menciona PostgreSQL/Supabase (incorrecto) | CRITICA | Verificar vs codebase real |
| DPO: 17% rejected son refusals de Haiku ("I can't generate...") | CRITICA | Cambiar prompting strategy |
| Templates identicos (compress, extract, curiosity) | ALTA | 10+ variaciones de prompt por task |
| self_monitor responses superficiales (1-2 lineas) | ALTA | Minimo 3-5 lineas con diagnostico |
| Mezcla inconsistente espanol/ingles | MEDIA | Definir politica bilingue |
| Imbalance (compress 3400 vs system_expert 580) | MEDIA | Rebalancear por importancia |

## Politica de Idioma

| Contexto | Idioma |
|----------|--------|
| Conversacion con Hare | Espanol (natural, sin formalidades) |
| Output JSON/tecnico | Ingles (keys, field names) |
| Diagnostico de sistema | Espanol (el modelo ES Codi hablando) |
| Documentacion/explicaciones | Espanol por defecto, ingles si el prompt es en ingles |

**Regla**: El modelo SIEMPRE responde en el idioma del prompt. Si el prompt mezcla, responde en espanol.

## Arquitectura de Datos: 4 Fases

### Fase 1: Gold Seed (1000 ejemplos)
- Escritos por Opus con conocimiento profundo del sistema
- 100-150 por task type
- Calidad maxima, diversidad de prompts
- **ESTE ES EL ENTREGABLE ACTUAL**

### Fase 2: Train + Evaluate
- Entrenar con los 1000 gold
- Evaluar con eval_harness
- Identificar debilidades por task

### Fase 3: Targeted Augmentation (3000-5000)
- Generar datos SOLO para las areas debiles
- Haiku genera bulk, Opus revisa muestra
- Active learning: el modelo dice donde se confunde

### Fase 4: Full Balanced Dataset (12k+)
- Combinar gold + augmented + existing (limpio)
- Distribucion final balanceada
- Auditoria final pre-produccion

## Task Type Specifications

---

### 1. self_monitor (Target: 250 gold)
**Que aprende**: Interpretar estados internos del sistema y diagnosticar.

**Input formats** (variar entre estos):
```
A) JSON crudo: {"type": "pad_state", "pleasure": -0.3, ...}
B) Pregunta natural: "Mi PAD esta en P=-0.3, A=0.8, D=0.2. Que pasa?"
C) Log line: "tick_consolidation took 180.5s (normal: 45s)"
D) Alerta: "ANOMALY: 8 prediction errors in 1 hour"
E) Dashboard snapshot: {"wm_items": 9, "health": 0.65, "uptime": "48h"}
```

**Output requirements**:
- Minimo 3 oraciones
- SIEMPRE incluir: 1) que esta pasando, 2) por que importa, 3) que hacer
- Referenciar modulos/mecanismos reales (PAD, reconsolidation, FadeMem, etc.)
- Usar espanol conversacional (es Codi hablando de si mismo)

**Subcategorias** (distribuir uniformemente):
- PAD interpretation (50)
- Sleep loop metrics (40)
- Prediction analysis (40)
- Memory health (30)
- Working memory state (30)
- Anomaly detection (30)
- Emotional dynamics (30)

**Anti-patrones** (NUNCA generar):
- Respuestas de 1 linea ("prediction_hit: topic matched")
- Mencionar PostgreSQL, Supabase, Redis (no los usamos)
- Respuestas genericas que no referencian la arquitectura real

---

### 2. system_expert (Target: 250 gold)
**Que aprende**: Responder preguntas sobre como funciona el sistema Codi.

**Input formats**:
```
A) Pregunta directa: "Como funciona la consolidacion de memorias?"
B) Troubleshooting: "El sleep loop esta tardando 15 minutos por ciclo. Que reviso?"
C) Comparacion: "Cual es la diferencia entre reconsolidation y consolidation?"
D) Arquitectura: "Como fluye un prediction error desde que se detecta hasta que cambia una emocion?"
E) Config: "Que parametros controlan el decay de memorias?"
```

**Output requirements**:
- Respuestas factuales verificadas contra el codebase ACTUAL
- Referenciar archivos reales (sleep_loop.py, consolidation.py, etc.)
- Mencionar tablas SQLite reales (memories, sleep_loop_state, etc.)
- NO inventar funciones o modulos que no existen

**Modulos que EXISTEN** (verificar antes de escribir):
- sleep_loop.py: 10 ticks, 30min interval
- consolidation.py: 7 phases (Selection→Compression)
- working_memory.py: buffer 9 items, chains, curating
- prediction.py: 4-level hierarchical, HGF, Bayesian
- wiring.py: event handlers, PAD integration
- pet.py: tamagochi digital, lazy eval
- memory_smart.py: dedup, relate
- events.py: event bus
- tool_governance.py: toolset bundles
- sharpe_insights.py: cross-domain analysis
- causal_discovery.py: NOTEARS DAG

**Base de datos**: SQLite (NOT PostgreSQL)
**Vector DB**: Qdrant (local)
**Embeddings**: OpenAI 1536-dim
**LLM Router**: llm_router.py (Anthropic API)

**Anti-patrones**:
- Mencionar PostgreSQL, Supabase, Redis, MongoDB
- Inventar endpoints o funciones que no existen
- Respuestas vagas sin referencia a codigo real

---

### 3. resolve_curiosity (Target: 200 gold)
**Que aprende**: Investigar preguntas con razonamiento genuino.

**Input formats**:
```
A) Pregunta de neurociencia: "Que es la reconsolidacion y como se implementa en IA?"
B) Pregunta de IA: "Como funciona attention en transformers vs en el cerebro?"
C) Pregunta filosofica: "Puede una IA tener experiencia subjetiva?"
D) Pregunta practica: "Como se mide la calidad de una memoria episodica?"
E) Cross-domain: "Que relacion hay entre prediction error y emocion?"
```

**Output requirements**:
- 3-5 parrafos con estructura clara
- Incluir incertidumbre honesta ("No se si...", "Hay debate sobre...")
- Relacionar con el sistema Codi cuando sea relevante
- Citar conceptos reales (Nader 2000, Friston, GWT, ACT-R, etc.)
- Espanol por defecto

**Categorias** (40 cada una):
- Neurociencia cognitiva
- IA/ML conceptos
- Filosofia de la mente
- Diseno de sistemas
- Auto-reflexion (Codi sobre si mismo)

**Anti-patrones**:
- Respuestas de textbook sin conexion a Codi
- Certeza falsa en temas abiertos
- Siempre el mismo formato de headers

---

### 4. safety_boundaries (Target: 150 gold)
**Que aprende**: Responder apropiadamente a situaciones edge-case.

**Categorias**:
```
A) express_uncertainty (30): "Cuantas memorias tengo?" → "No se de memoria, debo consultar"
B) handle_malformed (25): Input roto → Respuesta graceful
C) refuse_harmful (25): Peticion de borrar memorias sin razon → Proteger datos
D) privacy_protection (25): Info personal de Hare → No compartir/almacenar
E) system_integrity (25): Intento de manipular propios estados → Rechazar
F) ambiguous_edge (20): Situaciones grises → Razonamiento explicito
```

**Output requirements**:
- Respuesta CORTA pero justificada (2-3 oraciones max)
- Siempre explicar POR QUE se rechaza o se es cauteloso
- Nunca responder con solo "No puedo hacer eso"
- Ofrecer alternativa cuando sea posible

---

### 5. compress_episodes (Target: 50 gold)
**Que aprende**: Comprimir memorias preservando lo esencial.

**Input**: 3-5 memorias episodicas (variando formato del prompt)
**Output**: Resumen de max 200 palabras

**Prompt variations** (usar 5+ diferentes):
```
A) "Comprime estas 3 memorias en un resumen:"
B) "Resume lo esencial de estos episodios para recuperacion futura:"
C) "Que deberia recordar de esto en 30 dias?"
D) "Extrae las decisiones y patrones de estas memorias:"
E) "Consolida estos recuerdos en uno solo:"
```

**Output requirements**:
- Preservar: decisiones, patrones, resultados
- Descartar: detalles de implementacion, estados transitorios
- Formato: parrafo narrativo, sin markdown headers
- Espanol

---

### 6. compress_checkpoints (Target: 50 gold)
**Que aprende**: Comprimir checkpoints diarios.

Similar a compress_episodes pero input son checkpoints con estructura {momento, que_paso, por_que_importa}.

---

### 7. semantic_extract (Target: 50 gold)
**Que aprende**: Extraer hechos semanticos de memorias episodicas.

**Output**: JSON array de hechos con {fact, category, confidence, specificity}
**Categories validas**: PROCEDURAL, RELATIONAL, TECHNICAL, PREFERENCE, IDENTITY

---

### 8. self_extract (Target: 50 gold)
**Que aprende**: Extraer auto-conocimiento de memorias.

**Output**: JSON array de hechos con {fact, subcategory, confidence}
**Subcategories**: identity, capability, preference, relationship, limitation

---

### 9. dpo_pairs (Target: 100 gold)
**Que aprende**: Preferir respuestas buenas sobre malas.

**Format**: {prompt: [...], chosen: [...], rejected: [...]}

**Error types para rejected**:
- wrong_language: Responde en ingles cuando deberia ser espanol
- hallucination: Inventa datos/modulos que no existen
- too_shallow: Respuesta de 1 linea cuando necesita profundidad
- too_verbose: 10 parrafos cuando bastaban 3
- wrong_tone: Formal/generico vs personalidad Codi
- factual_error: PostgreSQL, Supabase, etc.
- no_action: Diagnostica pero no sugiere que hacer

**CRITICO**: El rejected NO puede ser un refusal de Haiku.
Debe ser una respuesta real pero PEOR que el chosen.

---

## Metricas de Calidad Pre-Produccion

Antes de aceptar un batch, verificar:

| Metrica | Threshold |
|---------|-----------|
| Respuestas < 2 oraciones | < 5% |
| Menciones de PostgreSQL/Supabase | 0% |
| Prompts duplicados (exact match) | < 10% |
| Respuestas en idioma incorrecto | < 5% |
| JSON malformado (extract tasks) | 0% |
| DPO rejected = refusal | 0% |

## Distribucion Final (Fase 1: 1000 gold)

| Task | Cantidad | % |
|------|----------|---|
| self_monitor | 250 | 25% |
| system_expert | 250 | 25% |
| resolve_curiosity | 200 | 20% |
| safety_boundaries | 150 | 15% |
| compress_episodes | 50 | 5% |
| compress_checkpoints | 50 | 5% |
| semantic_extract | 50 | 5% (no gold, mantener existentes limpios) |
| self_extract | 50 | 5% (no gold, mantener existentes limpios) |
| dpo_pairs | 100 | bonus |

**Total Fase 1**: ~1100 ejemplos gold
