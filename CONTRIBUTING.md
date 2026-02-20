# Contributing

Gracias por contribuir a Codi Memory.

## TL;DR

1. Python 3.14+ con venv
2. Instala deps
3. Corre tests
4. Abre PR con descripcion clara y checks verdes

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Necesitas tambien:
- **Qdrant Cloud** (o local) con colecciones `codi_memories` y `codi_semantic`
- **OpenAI API key** para embeddings y consolidation LLM
- **Variables de entorno** en `.env` (ver `.env.example` si existe)

## Running Tests

```bash
# Suite completa (386 tests, ~40s)
./venv/bin/pytest tests/ -v

# Health signal rapido: continuity + parity (42 tests, ~25s)
./venv/bin/pytest -m "continuity or parity" -v

# Solo continuity battery (31 tests)
./venv/bin/pytest -m continuity -v

# Solo parity snapshots (11 tests)
./venv/bin/pytest -m parity -v

# Behavioral battery (20 tests)
./venv/bin/pytest -m battery -v
```

Ver `docs/testing.md` para mas detalle sobre markers, fixtures y debug tips.

## Branching & PRs

- Branch name: `feature/<short-name>` o `fix/<short-name>`
- PRs pequenos > PRs gigantes
- Si es refactor grande: requiere parity/contract tests antes de tocar produccion
- Todo PR debe linkear un issue cuando aplique

## Definition of Done

- [ ] Tests pasan localmente (`pytest tests/ -q`)
- [ ] Health suite pasa (`pytest -m "continuity or parity" -q`)
- [ ] No hay cambios de schema sin migracion + plan de backfill (ver `DECISIONS.md`)
- [ ] No se degrada performance sin justificar (ver `docs/SLO.md`)
- [ ] Si se agrega una MCP tool, se incluye en el conteo y en `ARCHITECTURE.md`

## Style & Hygiene

- Funciones pequenas y modulos cohesivos
- Evitar circular imports: usar lazy imports donde aplique (patron `__getattr__`)
- **Cero `CREATE TABLE` fuera de migraciones**: toda tabla nueva va en `migrations/` o `migrations_prospective/`
- Tests que validen mecanismos, no solo "no crashea"
- Si vas a mockear, **parchea donde se consume, no donde se re-exporta** (fachada)

## Issues & Labels

| Label | Uso |
|-------|-----|
| `phase/0`..`phase/5` | Fase del roadmap |
| `type/epic`, `type/task`, `type/gate` | Tipo de issue |
| `area/perf`, `area/continuity`, `area/docs` | Dominio |
| `prio/critical`, `prio/high`, `prio/medium` | Prioridad |

## Seguridad Operativa

- No bloquear lifecycle/hooks por errores: wrap en `try/except` donde sea necesario
- Evitar operaciones pesadas sin budget enforcement (ver Sleep Loop ticks en `ARCHITECTURE.md` sec 14)
- Write-path (mem0/OpenAI) es sincrono y lento (~30-80s). No agregar mas llamadas sin considerar el impacto en UX

## Key Documentation

| Doc | Que contiene |
|-----|-------------|
| `ARCHITECTURE.md` | Mapa completo del sistema (15 secciones) |
| `docs/SLO.md` | Service Level Objectives (3 capas) |
| `docs/testing.md` | Guia de testing (markers, fixtures, debug) |
| `DECISIONS.md` | Registro de decisiones arquitectonicas |
| `docs/perf/baseline_2026-02-16.md` | Baseline de performance (413 calls) |
