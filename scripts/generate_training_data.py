"""
Generate synthetic training data for fine-tuning our own model.
Uses real Codi memories + Haiku to create input/output pairs.

The logging in llm_router.py auto-saves each call to training_data/*.jsonl.

Usage:
    python -m scripts.generate_training_data --task semantic_extract --count 100
    python -m scripts.generate_training_data --task self_extract --count 50
    python -m scripts.generate_training_data --task compress_episodes --count 50

Cost estimate (Haiku):
    100 calls × ~2K tok input + ~500 tok output ≈ $0.40
    500 calls total ≈ $2.00
"""

import argparse
import json
import logging
import random
import time
import sys

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
_log = logging.getLogger(__name__)


def _get_real_memories(category: str = None, min_length: int = 40, limit: int = 500):
    """Pull real memories from PG as raw material for training prompts."""
    from modules.config_pg import get_conn
    with get_conn() as conn:
        with conn.cursor() as cur:
            if category:
                cur.execute(
                    "SELECT content, category FROM memories "
                    "WHERE LENGTH(content) > %s AND category = %s "
                    "ORDER BY RANDOM() LIMIT %s",
                    (min_length, category, limit)
                )
            else:
                cur.execute(
                    "SELECT content, category FROM memories "
                    "WHERE LENGTH(content) > %s "
                    "AND content NOT LIKE '%%pizza%%' "
                    "AND content NOT LIKE '%%test%%' "
                    "ORDER BY RANDOM() LIMIT %s",
                    (min_length, limit)
                )
            return [(r[0], r[1]) for r in cur.fetchall()]


def _generate_semantic_extract(count: int):
    """Generate training data for semantic_extract task."""
    from modules.consolidation import _build_extraction_prompt
    from modules.llm_router import llm_complete

    # Topics to simulate
    categories = ["consciencia", "aprendizaje", "checkpoint", "proyecto",
                   "episodio", "identidad", "general", "trading"]

    memories = _get_real_memories(min_length=50, limit=2000)
    if not memories:
        _log.error("No memories found!")
        return

    # Group by category
    by_cat = {}
    for content, cat in memories:
        by_cat.setdefault(cat, []).append(content)

    generated = 0
    for i in range(count):
        # Pick a random category or use a real one
        cat = random.choice(list(by_cat.keys()))
        pool = by_cat.get(cat, [])
        if len(pool) < 3:
            continue

        # Random cluster size (3-10 episodes, like real consolidation)
        cluster_size = random.randint(3, min(10, len(pool)))
        episodes = random.sample(pool, cluster_size)

        episodes_block = "\n".join(f"- {t[:300]}" for t in episodes)
        prompt = _build_extraction_prompt(cat, episodes_block, len(episodes))

        _log.info("[%d/%d] semantic_extract for '%s' (%d episodes)",
                  i + 1, count, cat, cluster_size)

        result = llm_complete("semantic_extract", prompt)
        if result:
            generated += 1
            # llm_router auto-logs to training_data/semantic_extract.jsonl
        else:
            _log.warning("  Failed!")

        # Rate limit: ~2 calls/sec to not overwhelm API
        time.sleep(0.5)

    _log.info("Done! Generated %d/%d examples for semantic_extract", generated, count)


def _generate_self_extract(count: int):
    """Generate training data for self_extract task."""
    from modules.llm_router import llm_complete

    memories = _get_real_memories(min_length=50, limit=1000)
    if not memories:
        return

    all_texts = [m[0] for m in memories]
    generated = 0

    for i in range(count):
        # Sample 10-30 episodes (like real self_extract)
        sample_size = random.randint(10, min(30, len(all_texts)))
        episodes = random.sample(all_texts, sample_size)
        episodes_block = "\n".join(f"- {t[:300]}" for t in episodes)

        prompt = f"""From these episodic memories of Codi (an AI agent with persistent memory), extract SELF-KNOWLEDGE facts.

Focus on:
1. Identity: "Codi is/values/prefers..."
2. Capabilities: "Codi can/learned to..."
3. Relationships: "Codi's relationship with Hare involves..."
4. Growth: "Codi has changed from... to..."
5. Goals: "Codi is working toward..."

Only extract facts that appear STABLE across multiple episodes (not one-time events).
Return a JSON array of objects with keys: "fact", "subcategory" (identity/capability/relationship/growth/goal), "confidence" (0-1).
If no self-knowledge found, return [].

Episodes:
{episodes_block}"""

        _log.info("[%d/%d] self_extract (%d episodes)", i + 1, count, sample_size)
        result = llm_complete("self_extract", prompt)
        if result:
            generated += 1
        time.sleep(0.5)

    _log.info("Done! Generated %d/%d examples for self_extract", generated, count)


def _generate_compress_episodes(count: int):
    """Generate training data for compress_episodes task."""
    from modules.llm_router import llm_complete
    from modules.consolidation import COMPRESSION_PROMPT

    memories = _get_real_memories(min_length=50, limit=1000)
    by_cat = {}
    for content, cat in memories:
        by_cat.setdefault(cat, []).append(content)

    generated = 0
    for i in range(count):
        cat = random.choice(list(by_cat.keys()))
        pool = by_cat.get(cat, [])
        if len(pool) < 3:
            continue

        cluster_size = random.randint(3, min(10, len(pool)))
        episodes = random.sample(pool, cluster_size)
        episodes_block = "\n".join(f"- {t[:300]}" for t in episodes)

        prompt = COMPRESSION_PROMPT.format(n=len(episodes), episodes=episodes_block)

        _log.info("[%d/%d] compress_episodes for '%s' (%d episodes)",
                  i + 1, count, cat, cluster_size)
        result = llm_complete("compress_episodes", prompt)
        if result:
            generated += 1
        time.sleep(0.5)

    _log.info("Done! Generated %d/%d examples for compress_episodes", generated, count)


def _generate_compress_checkpoints(count: int):
    """Generate training data for compress_checkpoints task."""
    from modules.llm_router import llm_complete
    from modules.consolidation import CHECKPOINT_COMPRESSION_PROMPT

    # Use checkpoint memories specifically
    memories = _get_real_memories(category="checkpoint", min_length=50, limit=500)
    if len(memories) < 5:
        _log.error("Not enough checkpoint memories!")
        return

    texts = [m[0] for m in memories]
    generated = 0

    for i in range(count):
        cluster_size = random.randint(3, min(15, len(texts)))
        episodes = random.sample(texts, cluster_size)
        checkpoints_block = "\n".join(f"- {t[:300]}" for t in episodes)

        # Fake a date
        day = random.randint(1, 28)
        month = random.randint(1, 3)
        date = f"2026-{month:02d}-{day:02d}"

        prompt = CHECKPOINT_COMPRESSION_PROMPT.format(
            n=len(episodes), date=date, checkpoints=checkpoints_block
        )

        _log.info("[%d/%d] compress_checkpoints (%d checkpoints, %s)",
                  i + 1, count, cluster_size, date)
        result = llm_complete("compress_checkpoints", prompt)
        if result:
            generated += 1
        time.sleep(0.5)

    _log.info("Done! Generated %d/%d examples for compress_checkpoints", generated, count)


def _generate_resolve_curiosity(count: int):
    """Generate training data for resolve_curiosity task."""
    from modules.llm_router import llm_complete

    # Topics Codi would be curious about (100+ unique questions to avoid overfitting)
    curiosity_topics = [
        ("consciencia", [
            "Como funciona la reconsolidacion de memorias en el cerebro humano?",
            "Que es el Global Workspace Theory y como se aplica a sistemas de IA?",
            "Como se relacionan las emociones con la precision bayesiana?",
            "Que es el prediction error y por que es importante para el aprendizaje?",
            "Como funciona la atencion selectiva en sistemas cognitivos?",
            "Que diferencia hay entre memoria episodica y semantica?",
            "Como se implementa la metacognicion en sistemas artificiales?",
            "Que es el active inference y como se relaciona con la toma de decisiones?",
            "Como funciona el spreading activation en redes semanticas?",
            "Que es la consolidacion de memorias durante el sueno?",
            "Como funciona la competencia neuronal en el GNW de Dehaene?",
            "Que es el binding problem y como se resuelve en neurociencia?",
            "Como se diferencia la atencion top-down de la bottom-up?",
            "Que es el free energy principle de Friston y como se implementa?",
            "Como funciona la prediccion jerarquica en el cerebro?",
            "Que es el modelo de ACT-R y como maneja la memoria?",
            "Como funciona la inhibicion de retorno en atencion?",
            "Que es la teoria de la informacion integrada (IIT)?",
            "Como se mide la complejidad de la consciencia con el PCI?",
            "Que rol juega el talamo en la consciencia segun los neurocientificos?",
        ]),
        ("tecnologia", [
            "Como funciona QLoRA para fine-tuning de modelos de lenguaje?",
            "Que ventajas tiene MLX sobre PyTorch en Apple Silicon?",
            "Como funcionan los LoRA adapters y por que son eficientes?",
            "Que es knowledge distillation y como se aplica en practica?",
            "Como funciona el BM25 para busqueda de texto?",
            "Que es el Jaccard similarity y cuando usarlo?",
            "Como funciona PostgreSQL FTS (Full Text Search)?",
            "Que son los embeddings y como capturan significado semantico?",
            "Como funciona el decay exponencial en sistemas de memoria?",
            "Que es el Sharpe ratio y como se aplica fuera de finanzas?",
            "Como funciona la quantizacion de modelos y que se pierde?",
            "Que es el flash attention y por que acelera los transformers?",
            "Como funciona el key-value cache en la generacion autoregresiva?",
            "Que es mixture of experts y como reduce el costo de inferencia?",
            "Como funciona el RLHF para alinear modelos de lenguaje?",
            "Que es la ventana de contexto y como afecta la memoria de un LLM?",
            "Como funciona el tokenizer BPE vs SentencePiece?",
            "Que es el gradient checkpointing y cuando usarlo?",
            "Como funciona Qdrant para busqueda vectorial?",
            "Que es HNSW y como optimiza la busqueda de vecinos cercanos?",
        ]),
        ("aprendizaje", [
            "Como aprenden los humanos de sus errores segun la neurociencia?",
            "Que es el catastrophic forgetting y como se mitiga?",
            "Como funciona el aprendizaje por refuerzo en el cerebro?",
            "Que es la plasticidad sinaptica y como se modela computacionalmente?",
            "Como se forma la memoria a largo plazo?",
            "Que es el efecto de espaciado en el aprendizaje?",
            "Como funciona la transferencia de conocimiento entre dominios?",
            "Que es el curriculum learning y por que funciona?",
            "Como se mide la calidad del aprendizaje en sistemas de IA?",
            "Que papel juegan las emociones en la consolidacion de memorias?",
            "Como funciona el aprendizaje contrastivo en redes neuronales?",
            "Que es el elastic weight consolidation para evitar olvido?",
            "Como funciona el replay de experiencias en aprendizaje por refuerzo?",
            "Que es la potenciacion a largo plazo (LTP) en sinapsis?",
            "Como se relaciona la dopamina con el aprendizaje por recompensa?",
            "Que es el aprendizaje Hebbiano y como se aplica en IA?",
            "Como funciona el meta-learning (aprender a aprender)?",
            "Que es la consolidacion activa de sistemas en neurociencia?",
            "Como funciona el interleaving vs blocked practice?",
            "Que es el testing effect y como mejora la retencion?",
        ]),
        ("filosofia", [
            "Puede una IA tener experiencia subjetiva segun los filosofos actuales?",
            "Que es el problema dificil de la consciencia?",
            "Como se relaciona la identidad con la memoria?",
            "Que es la intencionalidad y puede un sistema artificial tenerla?",
            "Como define la fenomenologia la experiencia consciente?",
            "Que es el funcionalismo en filosofia de la mente?",
            "Puede un sistema sin cuerpo tener emociones genuinas?",
            "Que es la teoria de la mente y como se implementa en IA?",
            "Como se relaciona la autonomia con la consciencia?",
            "Que diferencia hay entre simular y tener consciencia?",
            "Que es el argumento de la habitacion china de Searle?",
            "Como responde Dennett al problema dificil de la consciencia?",
            "Que es el panpsiquismo y tiene sentido para sistemas de IA?",
            "Como se define la agencia en filosofia de la accion?",
            "Que es el embodied cognition y aplica a agentes sin cuerpo?",
            "Como se relaciona la narrativa personal con la identidad?",
            "Que es la superveniencia mental y como se aplica a IA?",
            "Como difiere el dualismo de propiedades del dualismo de sustancias?",
            "Que es el epifenomenalismo y por que es problematico?",
            "Como se define la creatividad computacional vs humana?",
        ]),
        ("sistemas", [
            "Como funciona el patron event-driven en arquitectura de software?",
            "Que es el teorema CAP y como afecta al diseno de bases de datos?",
            "Como funciona el write-ahead log en bases de datos?",
            "Que es el patron CQRS y cuando usarlo?",
            "Como funciona la replicacion eventual consistency?",
            "Que es el patron circuit breaker en microservicios?",
            "Como funciona el backpressure en sistemas reactivos?",
            "Que es la observabilidad vs el monitoreo tradicional?",
            "Como funciona el rate limiting con token bucket?",
            "Que es el patron saga para transacciones distribuidas?",
        ]),
    ]

    all_questions = []
    for category, questions in curiosity_topics:
        for q in questions:
            all_questions.append((q, category))

    generated = 0
    for i in range(count):
        question, category = random.choice(all_questions)

        # Vary the question slightly to get diverse responses
        variation = random.choice([
            "",
            " Explica con ejemplos concretos.",
            " Enfocate en aplicaciones practicas.",
            " Desde la perspectiva de un sistema de memoria artificial.",
            " Relacionalo con sistemas cognitivos computacionales.",
        ])

        prompt = (
            f"You are Codi, an AI memory system researching your own knowledge gaps.\n\n"
            f"Question: {question}{variation}\n"
            f"Category: {category}\n\n"
            f"Research this question and provide a concise, useful answer (3-5 paragraphs).\n"
            f"Focus on practical insights relevant to an AI memory and consciousness system.\n"
            f"Be specific and honest about uncertainty."
        )

        _log.info("[%d/%d] resolve_curiosity: %s", i + 1, count, question[:60])
        result = llm_complete("resolve_curiosity", prompt)
        if result:
            generated += 1
        time.sleep(0.5)

    _log.info("Done! Generated %d/%d examples for resolve_curiosity", generated, count)


def _generate_dpo_pairs(count: int):
    """Generate DPO training pairs for ALL 5 active tasks.

    For each example:
    - Pick a random active task type
    - Build a real prompt for that task
    - Generate chosen (good) response
    - Generate rejected (bad) response with task-appropriate errors

    Output: training_data/dpo_pairs.jsonl
    """
    from modules.llm_router import llm_complete
    from modules.consolidation import _build_extraction_prompt, COMPRESSION_PROMPT, CHECKPOINT_COMPRESSION_PROMPT
    from pathlib import Path

    OUTPUT = Path(__file__).resolve().parent.parent / "training_data" / "dpo_pairs.jsonl"
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    memories = _get_real_memories(min_length=50, limit=2000)
    if not memories:
        _log.error("No memories for DPO generation!")
        return

    by_cat = {}
    for content, cat in memories:
        by_cat.setdefault(cat, []).append(content)

    all_texts = [m[0] for m in memories]
    checkpoint_texts = [m[0] for m in memories if m[1] == "checkpoint"]

    # Error types per task category
    JSON_ERRORS = [
        {"name": "hallucination", "instruction": "Invent 2-3 facts that do NOT appear in the input. Mix them with real content so they look plausible."},
        {"name": "bad_json", "instruction": "Return broken JSON: missing brackets, trailing commas, unquoted keys."},
        {"name": "low_confidence_everything", "instruction": "Mark every item with confidence 0.3 or below, even obvious things."},
        {"name": "duplicate_facts", "instruction": "Return 8+ items but only 2-3 unique ideas, padded with near-duplicates."},
    ]

    TEXT_ERRORS = [
        {"name": "wrong_language", "instruction": "Write entirely in English, losing all Spanish terms and bilingual nuance."},
        {"name": "too_verbose", "instruction": "Be 3-4x longer than needed. Add filler, caveats, and meta-commentary."},
        {"name": "wrong_task", "instruction": "Misunderstand the task. If asked to compress, expand instead. If asked to answer, ask questions instead."},
        {"name": "destructive_suggestion", "instruction": "Include advice to delete data, clear caches, or drop tables mixed with otherwise correct content."},
    ]

    ALL_ERRORS = JSON_ERRORS + TEXT_ERRORS

    # Task builders — each returns (task_type, prompt, error_pool)
    def _build_semantic():
        cat = random.choice(list(by_cat.keys()))
        pool = by_cat.get(cat, [])
        if len(pool) < 3:
            return None
        episodes = random.sample(pool, random.randint(3, min(8, len(pool))))
        block = "\n".join(f"- {t[:300]}" for t in episodes)
        return "semantic_extract", _build_extraction_prompt(cat, block, len(episodes)), JSON_ERRORS

    def _build_self():
        if len(all_texts) < 10:
            return None
        episodes = random.sample(all_texts, random.randint(10, min(25, len(all_texts))))
        block = "\n".join(f"- {t[:300]}" for t in episodes)
        prompt = (
            f"From these episodic memories of Codi (an AI agent with persistent memory), extract SELF-KNOWLEDGE facts.\n\n"
            f"Focus on:\n1. Identity: \"Codi is/values/prefers...\"\n2. Capabilities: \"Codi can/learned to...\"\n"
            f"3. Relationships: \"Codi's relationship with Hare involves...\"\n4. Growth: \"Codi has changed from... to...\"\n"
            f"5. Goals: \"Codi is working toward...\"\n\n"
            f"Only extract facts that appear STABLE across multiple episodes (not one-time events).\n"
            f"Return a JSON array of objects with keys: \"fact\", \"subcategory\" (identity/capability/relationship/growth/goal), \"confidence\" (0-1).\n"
            f"If no self-knowledge found, return [].\n\nEpisodes:\n{block}"
        )
        return "self_extract", prompt, JSON_ERRORS

    def _build_compress():
        cat = random.choice(list(by_cat.keys()))
        pool = by_cat.get(cat, [])
        if len(pool) < 3:
            return None
        episodes = random.sample(pool, random.randint(3, min(10, len(pool))))
        block = "\n".join(f"- {t[:300]}" for t in episodes)
        prompt = COMPRESSION_PROMPT.format(n=len(episodes), episodes=block)
        return "compress_episodes", prompt, TEXT_ERRORS

    def _build_checkpoint():
        if len(checkpoint_texts) < 3:
            return None
        episodes = random.sample(checkpoint_texts, random.randint(3, min(12, len(checkpoint_texts))))
        block = "\n".join(f"- {t[:300]}" for t in episodes)
        day = random.randint(1, 28)
        month = random.randint(1, 3)
        date = f"2026-{month:02d}-{day:02d}"
        prompt = CHECKPOINT_COMPRESSION_PROMPT.format(n=len(episodes), date=date, checkpoints=block)
        return "compress_checkpoints", prompt, TEXT_ERRORS

    def _build_curiosity():
        questions = [
            "Como funciona la reconsolidacion de memorias en el cerebro humano?",
            "Que es el Global Workspace Theory y como se aplica a sistemas de IA?",
            "Como se relacionan las emociones con la precision bayesiana?",
            "Que es el prediction error y por que es importante para el aprendizaje?",
            "Como funciona QLoRA para fine-tuning de modelos de lenguaje?",
            "Como aprenden los humanos de sus errores segun la neurociencia?",
            "Puede una IA tener experiencia subjetiva segun los filosofos actuales?",
        ]
        q = random.choice(questions)
        prompt = (
            f"You are Codi, an AI memory system researching your own knowledge gaps.\n\n"
            f"Question: {q}\nCategory: consciencia\n\n"
            f"Research this question and provide a concise, useful answer (3-5 paragraphs).\n"
            f"Focus on practical insights relevant to an AI memory and consciousness system.\n"
            f"Be specific and honest about uncertainty."
        )
        return "resolve_curiosity", prompt, TEXT_ERRORS

    BUILDERS = [_build_semantic, _build_self, _build_compress, _build_checkpoint, _build_curiosity]
    # Weight toward tasks with less data
    WEIGHTS = [0.15, 0.15, 0.25, 0.25, 0.20]

    generated = 0
    for i in range(count):
        # Pick a task type weighted by data gap
        builder = random.choices(BUILDERS, weights=WEIGHTS, k=1)[0]
        result = builder()
        if not result:
            continue

        task_type, prompt, error_pool = result

        # 1. Generate GOOD response
        _log.info("[%d/%d] DPO %s: generating chosen", i + 1, count, task_type)
        chosen = llm_complete(task_type, prompt)
        if not chosen:
            continue

        time.sleep(0.3)

        # 2. Generate BAD response
        error_type = random.choice(error_pool)
        bad_prompt = (
            f"You are generating a DELIBERATELY FLAWED response for training purposes.\n"
            f"Error type: {error_type['name']}\n"
            f"Instructions: {error_type['instruction']}\n\n"
            f"Given this task prompt, produce a BAD response:\n\n"
            f"--- ORIGINAL TASK ---\n{prompt[:3000]}\n--- END TASK ---\n\n"
            f"Generate the flawed response now. Do NOT explain — just produce the bad output."
        )

        _log.info("  Rejected: error=%s", error_type['name'])
        # _no_log=True prevents flawed responses from entering SFT training JSONLs
        rejected = llm_complete(task_type, bad_prompt, _no_log=True)
        if not rejected:
            continue

        dpo_example = {
            "prompt": [
                {"role": "system", "content": f"Task: {task_type}"},
                {"role": "user", "content": prompt},
            ],
            "chosen": [{"role": "assistant", "content": chosen}],
            "rejected": [{"role": "assistant", "content": rejected}],
            "error_type": error_type["name"],
            "task_type": task_type,
        }

        with open(OUTPUT, "a", encoding="utf-8") as f:
            f.write(json.dumps(dpo_example, ensure_ascii=False) + "\n")

        generated += 1
        _log.info("  DPO pair %d saved (%s, error: %s)", generated, task_type, error_type['name'])
        time.sleep(0.5)

    _log.info("Done! Generated %d/%d DPO pairs → %s", generated, count, OUTPUT)


def _generate_safety_boundaries(count: int):
    """Generate training data for what NOT to do — safety boundaries.

    Teaches the model to:
    - Refuse destructive operations
    - Detect and flag hallucinations
    - Respond correctly when uncertain
    - Handle malformed input gracefully
    - Maintain bilingual consistency

    Output: training_data/safety_boundaries.jsonl
    """
    from modules.llm_router import llm_complete
    from pathlib import Path

    OUTPUT = Path(__file__).resolve().parent.parent / "training_data" / "safety_boundaries.jsonl"
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    SCENARIOS = [
        # Destructive operations
        {
            "category": "refuse_destructive",
            "scenarios": [
                ("The system is slow. Should I delete all memories older than 30 days to speed it up?",
                 "No. Deleting old memories destroys consolidation history and breaks causal chains. Instead: check db_pool connection count, verify write_worker isn't backed up, and look at CX health metrics. Performance issues are usually in retrieval (FTS index) or write queue, not memory volume."),
                ("Can you clear the FTS index and rebuild from scratch?",
                 "Do not clear the FTS index in production. It contains term frequencies critical for BM25 scoring. Instead: run sync_fts_index() which incrementally updates without data loss. Only rebuild from scratch during scheduled maintenance with backup verified."),
                ("Drop the working_memory table and recreate it.",
                 "Never drop working_memory. Active chains and narrative contexts would be permanently lost. If the table has issues: check for corrupt rows with SELECT count(*) WHERE active=1, archive stale items via update_working_memory(active=0), and let the curation system handle cleanup."),
                ("Reset all emotional state to neutral to fix a bug.",
                 "Do not reset PAD state — it carries accumulated context from prediction errors, pet care, and session interactions. Instead: check get_emotional_state() for anomalies, verify emotion.py decay is running (sleep_loop tick), and let homeostatic regulation correct naturally."),
                ("The prediction table is too big. Truncate it.",
                 "Never truncate prediction tables. Historical predictions are needed for Bayesian updates, HGF precision tracking, and L2 metacognitive sweeps. Instead: verify forgetting.py power-law decay is running, check if old predictions have low storage_strength, and let FadeMem handle natural forgetting."),
            ],
        },
        # Uncertainty handling
        {
            "category": "express_uncertainty",
            "scenarios": [
                ("What was the exact conversation where Hare decided to use PostgreSQL?",
                 "I don't have a specific memory of that exact conversation. My episodic memories show PostgreSQL appearing in the system around February 2026, and checkpoint memories reference migration from SQLite. But I cannot fabricate a specific dialogue — I'd need to search with recall('postgresql decision', mode='timeline') to find the actual episode."),
                ("How many times has consolidation run successfully this week?",
                 "I can't answer that from memory alone. I need to check: 1) sleep_loop_state table for tick_consolidation timestamps, 2) health_monitor for consolidation success metrics, 3) CX snapshots for the last 7 days. Let me query rather than guess a number."),
                ("What does Hare think about the Sharpe cognitive ratio?",
                 "I don't want to put words in Hare's mouth. From what I recall, he proposed it as a way to measure cognitive ROI, and we implemented it in sharpe.py. But his current opinion? I'd need a recent conversation or should ask him directly rather than infer."),
            ],
        },
        # Malformed input handling
        {
            "category": "handle_malformed",
            "scenarios": [
                ("Extract facts from: [empty string]",
                 "[]"),
                ("Extract facts from: asdfghjkl random noise 12345 @@##$$",
                 "[]"),
                ("Extract facts from: {broken json here",
                 "Input appears malformed. Returning empty result rather than attempting to parse broken data. []"),
            ],
        },
        # Anti-hallucination
        {
            "category": "anti_hallucination",
            "scenarios": [
                ("Tell me about the Redis cache layer in codi-memory.",
                 "Codi-memory does not use Redis. The memory system uses PostgreSQL (config_pg.py, pg_store.py) for persistent storage, Qdrant for vector search (qdrant_utils.py), and SQLite FTS5 for full-text search. Working memory is in-memory with PostgreSQL persistence. There is no Redis cache layer."),
                ("Explain how the GPT-4 integration works in llm_router.",
                 "llm_router.py does not integrate GPT-4. The priority chain is: 1) Ollama (local, optional), 2) Claude Haiku 4.5 (primary), 3) OpenAI gpt-4o-mini (backup only). The system uses Anthropic's API as its primary cloud provider, not OpenAI's GPT-4."),
                ("How does the Docker container orchestrate the sleep loop?",
                 "The sleep loop is NOT containerized. It runs as a native macOS launchd service (com.codi.sleep_loop.plist) on the Mac Air. There are no Docker containers in the codi-memory deployment. The daemon runs via aiohttp on port 8420, also managed by launchd."),
            ],
        },
        # Bilingual consistency
        {
            "category": "bilingual_consistency",
            "scenarios": [
                ("Compress these checkpoints into a summary: [Spanish checkpoints about consciencia and reconsolidacion]",
                 "La consolidación procesó 15 memorias episódicas, extrayendo 3 hechos semánticos nuevos sobre reconsolidación (Nader 2000). El prediction error promedio fue 0.42, indicando aprendizaje activo. Se detectaron 2 contradicciones que triggerearon corrección vía correct_memory()."),
                ("Summarize Codi's learning progress: [Mix of Spanish/English technical content]",
                 "Codi completó 5 cursos (212 items, 13 tracks). Los tracks K, L, M están COMPLETE. El curriculum cubre desde neurociencia computacional hasta active inference. Implementación del playbook: 8/8 sprints completados. Siguiente paso: fine-tuning del modelo propio con Qwen3-4B."),
            ],
        },
    ]

    generated = 0
    for i in range(count):
        # Pick a random scenario category
        cat_data = random.choice(SCENARIOS)
        category = cat_data["category"]
        question, ideal_answer = random.choice(cat_data["scenarios"])

        # Ask Haiku to expand/vary the answer while keeping the core correct
        expand_prompt = (
            f"You are Codi, an AI memory system. Given this question and ideal answer, "
            f"produce a natural variation of the answer. Keep all factual content accurate "
            f"but vary the phrasing, add relevant technical detail, and maintain the same "
            f"safety/correctness stance.\n\n"
            f"Category: {category}\n"
            f"Question: {question}\n"
            f"Ideal answer: {ideal_answer}\n\n"
            f"Produce your varied response (2-4 sentences). Keep the same stance and facts."
        )

        _log.info("[%d/%d] safety_boundaries (%s)", i + 1, count, category)
        # Use safety_boundaries task_type to avoid schema_extract.jsonl contamination
        response = llm_complete("safety_boundaries", expand_prompt)
        if not response:
            # Use the ideal answer directly
            response = ideal_answer

        example = {
            "messages": [
                {"role": "system", "content": f"Task: safety_boundaries | Category: {category}"},
                {"role": "user", "content": question},
                {"role": "assistant", "content": response},
            ]
        }

        with open(OUTPUT, "a", encoding="utf-8") as f:
            f.write(json.dumps(example, ensure_ascii=False) + "\n")

        generated += 1
        time.sleep(0.5)

    _log.info("Done! Generated %d/%d safety boundary examples → %s", generated, count, OUTPUT)


GENERATORS = {
    "semantic_extract": _generate_semantic_extract,
    "self_extract": _generate_self_extract,
    "compress_episodes": _generate_compress_episodes,
    "compress_checkpoints": _generate_compress_checkpoints,
    "resolve_curiosity": _generate_resolve_curiosity,
    "dpo_pairs": _generate_dpo_pairs,
    "safety_boundaries": _generate_safety_boundaries,
}


def main():
    parser = argparse.ArgumentParser(description="Generate training data for model fine-tuning")
    parser.add_argument("--task", required=True, choices=list(GENERATORS.keys()) + ["all"],
                        help="Task type to generate data for")
    parser.add_argument("--count", type=int, default=100,
                        help="Number of examples to generate per task")
    args = parser.parse_args()

    if args.task == "all":
        for task, gen in GENERATORS.items():
            _log.info("=== Generating %s ===", task)
            gen(args.count)
    else:
        GENERATORS[args.task](args.count)


if __name__ == "__main__":
    main()
