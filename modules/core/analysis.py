"""
core/analysis.py — Content analysis utilities.

Pure functions for theme inference, self-reference detection,
and confidence scoring. No DB, no state.
"""

from modules.core.constants import IMPORTANCE_WEIGHTS


def infer_themes(content: str) -> list:
    """Infer themes from content text. Returns up to 5 theme strings."""
    themes = []
    content_lower = content.lower()

    if "consciencia" in content_lower or "consciente" in content_lower:
        themes.append("consciencia")
    if "memoria" in content_lower or "recuerdo" in content_lower:
        themes.append("memoria")
    if "identidad" in content_lower or "soy codi" in content_lower:
        themes.append("identidad")
    if "hare" in content_lower or "harec" in content_lower:
        themes.append("relaciones")
    if "andre" in content_lower:
        themes.append("relaciones")
    if "fullempaques" in content_lower:
        themes.append("proyectos")
    if "error" in content_lower or "bug" in content_lower:
        themes.append("desarrollo")
    if "aprend" in content_lower:
        themes.append("aprendizaje")

    return themes[:5]


def is_self_referential(content: str) -> bool:
    """Detect if content is about Codi itself (metacognition).

    Returns True if content matches self-referential patterns.
    """
    content_lower = content.lower()
    self_patterns = [
        "soy codi", "yo soy", "me siento", "me parece", "creo que yo",
        "mi identidad", "mi consciencia", "mi memoria", "mi capacidad",
        "puedo hacer", "no puedo", "mis limitaciones", "mi conocimiento",
        "me cuesta", "se me dificulta", "entiendo que", "no entiendo",
        "mi experiencia", "como codi", "siendo codi", "mi rol",
        "mi proposito", "mi funcion", "mi manera de", "mi forma de pensar",
        "mi modelo", "mi auto-imagen", "reflexiono sobre", "me observo",
    ]
    return any(pattern in content_lower for pattern in self_patterns)


def calculate_confidence_score(memories: list) -> dict:
    """Calculate confidence score based on memory sources.

    Returns dict with: score (0-1), level, breakdown, reason.
    """
    if not memories:
        return {
            "score": 0.0,
            "level": "ninguno",
            "breakdown": {"total": 0},
            "reason": "No hay memorias sobre este tema",
        }

    source_weights = {
        "experienced": 1.0,
        "told": 0.7,
        "learned": 0.6,
        "inferred": 0.4,
    }

    total_weight = 0
    source_counts = {"experienced": 0, "told": 0, "learned": 0, "inferred": 0}

    for mem in memories:
        payload = mem.payload if hasattr(mem, "payload") else mem
        source = payload.get("ownership_source", "inferred")
        importance = payload.get("narrative_importance", "medium")
        confidence = payload.get("ownership_confidence", 0.5)

        source_counts[source] = source_counts.get(source, 0) + 1
        weight = (
            source_weights.get(source, 0.5)
            * IMPORTANCE_WEIGHTS.get(importance, 0.5)
            * confidence
        )
        total_weight += weight

    max_possible = len(memories) * 1.0
    score = min(total_weight / max_possible, 1.0) if max_possible > 0 else 0.0

    if score >= 0.8:
        level = "muy_alto"
    elif score >= 0.6:
        level = "alto"
    elif score >= 0.4:
        level = "medio"
    elif score >= 0.2:
        level = "bajo"
    else:
        level = "muy_bajo"

    return {
        "score": round(score, 2),
        "level": level,
        "breakdown": {"total": len(memories), **source_counts},
        "reason": f"{source_counts['experienced']} experiencias directas, {source_counts['told']} me contaron, {source_counts['learned']} aprendi, {source_counts['inferred']} inferi",
    }
