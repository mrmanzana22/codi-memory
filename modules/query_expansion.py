"""
Query Expansion Module
======================
Expands user queries to improve BM25 recall without hurting precision.

Three strategies:
  1. Static alias table (bilingual technical terms)
  2. Acronym/abbreviation expansion
  3. Working memory context priming

Design: Expansion only affects the BM25/FTS channel.
Vector search already handles semantic similarity naturally.

Neuroscience basis:
  - Anderson 1983: richer cues activate more associated nodes
  - Tulving & Thomson 1973: retrieval succeeds when cue overlaps encoding
  - Morris et al. 1977: transfer-appropriate processing
"""

# Strategy 1: Bilingual aliases — technical terms ES<->EN
# Each key triggers OR-appending its aliases to the FTS query
ALIASES: dict[str, list[str]] = {
    # Core modules
    "prediction": ["HGF", "L0", "L1", "L2", "L3", "prediccion", "Bayesian", "forecast"],
    "prediccion": ["prediction", "HGF", "Bayesian", "forecast"],
    "active inference": ["EFE", "dirichlet", "options", "inferencia activa", "free energy"],
    "inferencia activa": ["active inference", "EFE", "dirichlet", "free energy"],
    "consolidation": ["sleep", "clustering", "consolidacion", "semantic extraction"],
    "consolidacion": ["consolidation", "sleep", "clustering", "semantic extraction"],
    "reconsolidation": ["contradiction", "prediction error", "Nader", "labile", "reconsolidacion"],
    "reconsolidacion": ["reconsolidation", "contradiction", "prediction error", "Nader"],
    "consciousness": ["GNW", "workspace", "ignition", "consciencia", "awareness", "loops"],
    "consciencia": ["consciousness", "GNW", "workspace", "ignition", "awareness", "loops"],
    "spreading": ["activation", "graph", "edges", "propagacion", "ACT-R"],
    "metacognition": ["FOK", "feeling knowing", "calibration", "metamemoria"],
    "metamemoria": ["metacognition", "FOK", "feeling knowing", "calibration"],
    "counterfactual": ["what if", "SCM", "causal", "contrafactual"],
    "contrafactual": ["counterfactual", "what if", "SCM", "causal"],
    "NOTEARS": ["causal discovery", "DAG", "Lagrangian", "descubrimiento causal"],
    "forgetting": ["FadeMem", "decay", "vault", "dormant", "olvido"],
    "olvido": ["forgetting", "FadeMem", "decay", "vault", "dormant"],
    "dedup": ["deduplication", "BMR", "Bayes factor", "duplicado"],
    "emotion": ["PAD", "arousal", "valence", "emocion", "AC"],
    "emocion": ["emotion", "PAD", "arousal", "valence", "AC"],
    "working memory": ["WM", "memoria trabajo", "buffer", "chain"],
    "memoria trabajo": ["working memory", "WM", "buffer"],
    "self model": ["self_model", "automodelo", "identity", "reflect"],

    # Projects
    "cerebro operativo": ["consciousness implementation", "brain", "cognitive system",
                          "implementacion consciencia", "sistema cognitivo", "5 layers"],
    "fullempaques": ["empaques", "packaging", "boxes", "cajas", "produccion"],
    "trading": ["Kraken", "crypto", "exchange", "cripto", "order"],
    "daemon": ["launchd", "background", "service", "sleep loop", "codi-daemon"],
    "hook": ["PostToolUse", "Stop", "SessionStart", "edit_tracker", "session_capture"],
    "nous": ["nous-lang", "lenguaje cognitivo", "transpile"],

    # Infrastructure
    "memoria": ["memory", "recuerdo", "recall", "retrieval"],
    "memory": ["memoria", "recuerdo", "recall", "retrieval"],
    "aprendizaje": ["learning", "training", "curso", "track"],
    "learning": ["aprendizaje", "training", "curso"],
    "proyecto": ["project", "feature", "sistema"],
    "project": ["proyecto", "feature", "sistema"],
    "error": ["bug", "falla", "fix", "solucion"],
    "bug": ["error", "falla", "fix", "solucion"],
    "decision": ["elegimos", "decidimos", "approach", "tradeoff"],
    "implementar": ["implement", "build", "construir", "deploy", "implementacion"],
    "implement": ["implementar", "construir", "deploy", "implementacion"],
    "busqueda": ["search", "recall", "retrieval", "query"],
    "search": ["busqueda", "recall", "retrieval", "query"],
    "guardar": ["save", "store", "remember", "persist"],
    "save": ["guardar", "store", "remember", "persist"],

    # People and context
    "Hare": ["Harec", "CEO", "jefe"],
    "Sebastian": ["dev", "developer", "equipo"],
    "valora": ["values", "autonomy"],
}

# Strategy 2: Acronym expansion
ACRONYMS: dict[str, str] = {
    "PE": "prediction error",
    "GNW": "global workspace",
    "FOK": "feeling knowing",
    "WM": "working memory",
    "EFE": "expected free energy",
    "BMR": "Bayesian model reduction",
    "SS": "storage strength",
    "RS": "retrieval strength",
    "CX": "cross loop",
    "PAD": "pleasure arousal dominance",
    "RIF": "retrieval induced forgetting",
    "HGF": "hierarchical gaussian filter",
    "DAG": "directed acyclic graph",
    "FTS": "full text search",
    "BM25": "keyword search",
    "ACT-R": "activation retrieval",
    "TCM": "temporal context model",
}

# Spanish stopwords to filter before expansion
_ES_STOPS = frozenset({
    "que", "de", "en", "el", "la", "los", "las", "un", "una",
    "por", "con", "para", "como", "del", "al", "es", "se", "no",
    "lo", "le", "ya", "si", "su", "muy", "mas", "pero", "sin",
    "ser", "fue", "esto", "eso", "este", "esta", "hay", "tiene",
})


def expand_query_for_fts(query: str) -> set[str]:
    """Expand a query with synonyms, bilingual terms, and acronyms.

    Returns a SET of additional terms to OR-append to the FTS query.
    Only called from the FTS path, never affects vector search.

    Args:
        query: The original user query string.

    Returns:
        Set of expansion terms (may be empty).
    """
    expansions: set[str] = set()
    q_lower = query.lower()

    # Strategy 1: Alias matching (phrase-level)
    for trigger, aliases in ALIASES.items():
        if trigger.lower() in q_lower:
            for alias in aliases:
                # Only add terms >1 char that aren't already in query
                if len(alias) > 1 and alias.lower() not in q_lower:
                    expansions.add(alias)

    # Strategy 2: Acronym expansion (word-level)
    words = query.split()
    for word in words:
        upper = word.upper().rstrip(".,;:!?")
        if upper in ACRONYMS and ACRONYMS[upper].lower() not in q_lower:
            for term in ACRONYMS[upper].split():
                if len(term) > 1:
                    expansions.add(term)

    return expansions


def build_expanded_tsquery(query: str) -> str:
    """Build a complete FTS tsquery string with expansions.

    Handles: stopword removal, AND/OR logic, alias expansion, acronym expansion.
    Replaces the inline _FTS_ALIASES logic in pg_store.py.

    Args:
        query: The original user query string.

    Returns:
        A tsquery-compatible string ready for to_tsquery('english', ...).
    """
    # Filter stopwords
    terms = [w for w in query.split() if len(w) > 1 and w.lower() not in _ES_STOPS]

    if not terms:
        return query

    # AND for <=2 words (precise), OR for 3+ (broad recall)
    if len(terms) <= 2:
        base_query = " & ".join(terms)
    else:
        base_query = " | ".join(terms)

    # Get expansions
    expansions = expand_query_for_fts(query)

    if expansions:
        # Sanitize expansion terms for tsquery (remove special chars)
        safe_expansions = []
        for term in expansions:
            # Keep only alphanumeric and basic chars, split multi-word
            for subterm in term.split():
                cleaned = "".join(c for c in subterm if c.isalnum() or c == '_')
                if len(cleaned) > 1:
                    safe_expansions.append(cleaned)

        if safe_expansions:
            # Cap expansions to prevent query explosion
            safe_expansions = safe_expansions[:15]
            base_query = f"({base_query}) | " + " | ".join(safe_expansions)

    return base_query
