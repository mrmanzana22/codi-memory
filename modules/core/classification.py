"""
core/classification.py — Topic classification for codi-memory.

Pure function + keyword dictionaries. No state, no DB.
"""

KNOWN_PROJECTS = [
    "trading", "fullempaques", "consciencia", "n8n",
    "kraken", "memoria", "pilas", "portal-aliados-mrmanzana",
]

TOPIC_KEYWORDS = {
    "n8n": ["n8n", "workflow", "automatiz", "nodo"],
    "trading": ["trading", "kraken", "cripto", "bitcoin", "mercado"],
    "fullempaques": ["fullempaques", "produccion", "fabrica", "empaque"],
    "memoria": ["memoria", "recuerdo", "recordar", "qdrant"],
    "codigo": ["codigo", "python", "javascript", "programar", "server.py"],
    "proyecto": ["proyecto", "implementar", "desarrollar", "feature"],
    "configuracion": ["config", "variable", "entorno", "setup", "easypanel"],
    "consciencia": [
        "consciencia", "consciente", "self-model", "prediccion",
        "consciousness", "awareness", "metacognicion", "metacognition",
        "self_model", "sleep_loop", "reconsolidacion", "reconsolidation",
        "preturn", "butlin", "gwt", "gnw", "fok_calibration",
    ],
}

TRIGGER_PRIORITY_ORDER = [
    "proyecto_nuevo", "fullempaques", "automatizacion", "trading", "mi_entrenamiento",
]


def classify_topic(text: str) -> str:
    """Classify text into a known topic using keyword matching.

    Loewenstein 1994: Curiosity is driven by information gaps in specific
    DOMAINS, not random words.

    Returns topic string or 'general' if no match.
    """
    text_lower = text.lower()
    scores = {}
    for topic, keywords in TOPIC_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw in text_lower)
        if score > 0:
            scores[topic] = score
    if not scores:
        return "general"
    return max(scores, key=scores.get)
