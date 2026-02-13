import json
import re
from typing import Any, Dict, List, Optional

from modules.memory_core import (
    search_memory,
    search_by_theme,
    search_by_ownership,
    get_project_timeline,
)
from modules.memory_smart import add_memory_smart
from modules.working_memory import push_to_working_memory, get_working_memory
from modules.consciousness import get_workspace_state, despertar_codi, search_by_emotion
from modules.maintenance import _ver_recordatorios_externos


# ============================================================
# INTERFACE LAYER (Fase 2.5) - 3 macro-tools
# recall / remember / context_snapshot
# ============================================================

VALID_IMPORTANCE = {"critical", "high", "medium", "low", "auto"}
VALID_RECALL_MODES = {"auto", "memory", "theme", "ownership", "emotion", "timeline"}
VALID_SNAPSHOT_LEVELS = {"light", "full"}


def _json_response(pretty: str, **extra: Any) -> str:
    payload: Dict[str, Any] = {"pretty": pretty}
    payload.update(extra)
    return json.dumps(payload, ensure_ascii=False)


def _importance_to_relevance(importance: str) -> float:
    # relevance is 0..1 for working memory
    m = {
        "critical": 0.95,
        "high": 0.80,
        "medium": 0.60,
        "low": 0.40,
    }
    return m.get(importance, 0.60)


def _auto_importance_from_text(content: str) -> str:
    c = content.lower()
    # Very simple heuristic; keeps the API stable and avoids heavy NLP here.
    if any(k in c for k in ["urgente", "important", "importante", "recuerda", "no olvidar", "deadline", "hoy", "mañana"]):
        return "high"
    if len(content) >= 200:
        return "high"
    if len(content) <= 60:
        return "medium"
    return "medium"


def recall(query: str, mode: str = "auto", limit: int = 8) -> str:
    """
    Macro-tool: unifica formas de buscar para reducir ambiguedad.

    Args:
        query: Lo que quieres recordar
        mode: auto|memory|theme|ownership|emotion|timeline
        limit: Maximo resultados (se clamp a 20)

    Returns:
        JSON string con 'pretty' + 'results' estructurados
    """
    mode = (mode or "auto").strip().lower()
    if mode not in VALID_RECALL_MODES:
        return _json_response(
            f"Modo invalido: {mode}. Usa: {', '.join(sorted(VALID_RECALL_MODES))}.",
            results=[],
            count=0,
        )
    try:
        limit = int(limit)
    except Exception:
        limit = 8
    limit = max(1, min(20, limit))

    results: List[Dict[str, Any]] = []
    pretty_lines: List[str] = [f"# RECALL\n\n**Query:** {query}\n**Mode:** {mode}\n"]

    # Heuristic routing
    q = (query or "").strip()
    q_low = q.lower()

    def add_result(source: str, text: str, meta: Optional[Dict[str, Any]] = None):
        entry = {"source": source, "text": text}
        if meta:
            entry["meta"] = meta
        results.append(entry)

    if mode == "timeline":
        # Expect: "timeline: proyecto X" or just query as project name
        proj = q
        if ":" in q:
            proj = q.split(":", 1)[1].strip()
        out = get_project_timeline(proj, limit=max(20, limit))
        add_result("get_project_timeline", out, {"project": proj})
        pretty_lines.append("## Timeline\n" + out)
        return _json_response("\n".join(pretty_lines), results=results, count=len(results))

    if mode == "theme":
        theme = q
        if q_low.startswith("tema:") or q_low.startswith("theme:"):
            theme = q.split(":", 1)[1].strip()
        out = search_by_theme(theme, limit=max(10, limit))
        add_result("search_by_theme", out, {"theme": theme})
        pretty_lines.append("## Por tema\n" + out)
        return _json_response("\n".join(pretty_lines), results=results, count=len(results))

    if mode == "ownership":
        # Minimal parser: allow "source=learned importance=high"
        src = None
        imp = None
        min_conf = 0.0
        m_src = re.search(r"\bsource\s*=\s*(experienced|told|learned|inferred)\b", q_low)
        if m_src:
            src = m_src.group(1)
        m_imp = re.search(r"\bimportance\s*=\s*(critical|high|medium|low)\b", q_low)
        if m_imp:
            imp = m_imp.group(1)
        m_conf = re.search(r"\bmin_confidence\s*=\s*(0\.\d+|1\.0|1)\b", q_low)
        if m_conf:
            try:
                min_conf = float(m_conf.group(1))
            except Exception:
                min_conf = 0.0
        out = search_by_ownership(source=src, min_confidence=min_conf, importance=imp, limit=max(10, limit))
        add_result("search_by_ownership", out, {"source": src, "importance": imp, "min_confidence": min_conf})
        pretty_lines.append("## Por ownership\n" + out)
        return _json_response("\n".join(pretty_lines), results=results, count=len(results))

    if mode == "emotion":
        emo = q_low.strip()
        if ":" in q_low:
            emo = q_low.split(":", 1)[1].strip()
        out = search_by_emotion(emo, threshold=0.3, limit=max(10, limit))
        add_result("search_by_emotion", out, {"emotion": emo})
        pretty_lines.append("## Por emocion\n" + out)
        return _json_response("\n".join(pretty_lines), results=results, count=len(results))

    # mode auto or memory
    # 1) Try working memory quick scan (cheap)
    wm_raw = None
    try:
        wm_raw = get_working_memory()
        wm = json.loads(wm_raw)
        wm_items = wm.get("items", [])
        hits = []
        for it in wm_items:
            content = (it.get("content") or "")
            if q_low and q_low in content.lower():
                hits.append(it)
        if hits:
            add_result("working_memory_hits", json.dumps(hits, ensure_ascii=False), {"hits": len(hits)})
            pretty_lines.append("## Working Memory (matches)\n" + "\n".join([f"- {h.get('content','')}" for h in hits[:10]]))
    except Exception:
        pass

    # 2) General hybrid memory search (safe default)
    out = search_memory(q, limit=limit)
    add_result("search_memory", out, {"limit": limit})
    pretty_lines.append("## Long-term memory\n" + out)

    # 3) If auto, optionally add specialized views when signal is strong
    if mode == "auto":
        # theme cues
        if any(k in q_low for k in ["tema:", "theme:", "proyecto", "fase ", "roadmap", "feature"]):
            try:
                theme = q.split(":", 1)[1].strip() if ":" in q else q
                out2 = search_by_theme(theme, limit=max(10, limit))
                add_result("search_by_theme", out2, {"theme": theme})
                pretty_lines.append("\n## (Auto) Por tema\n" + out2)
            except Exception:
                pass
        # ownership cues
        if any(k in q_low for k in ["experienced", "told", "learned", "inferred", "source="]):
            try:
                out3 = search_by_ownership(limit=max(10, limit))
                add_result("search_by_ownership", out3, {})
                pretty_lines.append("\n## (Auto) Ownership\n" + out3)
            except Exception:
                pass
        # emotion cues
        if any(k in q_low for k in ["anxious", "hostile", "relaxed", "bored", "exuberant", "dependent", "docile", "disdainful", "emocion:"]):
            try:
                emo = q_low.split(":", 1)[1].strip() if "emocion:" in q_low else q_low
                out4 = search_by_emotion(emo, threshold=0.3, limit=max(10, limit))
                add_result("search_by_emotion", out4, {"emotion": emo})
                pretty_lines.append("\n## (Auto) Emocion\n" + out4)
            except Exception:
                pass

    return _json_response("\n".join(pretty_lines), results=results, count=len(results))


def remember(content: str, importance: str = "auto", topic: str = "general",
             source: str = "interaction", long_term: bool = True) -> str:
    """
    Macro-tool: unifica guardar (working memory + long-term) sin ambiguedad.

    Siempre:
      - push_to_working_memory(content, topic, relevance)

    Long-term (por defecto):
      - add_memory_smart(content, category=topic, source=..., importance=...)

    Args:
        content: Lo que quieres recordar
        importance: auto|critical|high|medium|low
        topic: topic/category
        source: interaction by default
        long_term: si False, solo working memory

    Returns:
        JSON string con resultados de working + long-term
    """
    imp = (importance or "auto").strip().lower()
    if imp not in VALID_IMPORTANCE:
        imp = "auto"
    if imp == "auto":
        imp = _auto_importance_from_text(content)

    relevance = _importance_to_relevance(imp)

    # 1) Working memory
    wm_res = push_to_working_memory(content=content, topic=topic, relevance=relevance, occurred_at=None, source=source)

    lt_res = None
    lt_enabled = bool(long_term)

    # 2) Long-term decision: if low importance and short, skip unless explicitly requested
    if lt_enabled:
        if imp == "low" and len(content) < 120 and not re.search(r"\brecuerda\b|\bno olvidar\b", content.lower()):
            lt_enabled = False

    if lt_enabled:
        # Map source to memory_smart expected values when possible
        ms_source = "experienced"
        if source in ("reflection", "prediction", "consolidation"):
            ms_source = "inferred"
        # memory_smart expects: critical/high/medium/low
        lt_res = add_memory_smart(content=content, category=topic, source=ms_source, importance=imp)

    pretty_lines = [
        "# REMEMBER",
        f"**Topic:** {topic}",
        f"**Importance:** {imp}",
        "",
        "## Working Memory",
    ]
    try:
        wm_j = json.loads(wm_res)
        pretty_lines.append(wm_j.get("pretty", str(wm_j)))
    except Exception:
        pretty_lines.append(str(wm_res))

    if lt_res is not None:
        pretty_lines.append("\n## Long-term (add_memory_smart)\n" + str(lt_res))
    else:
        pretty_lines.append("\n## Long-term\n*No se consolido a long-term (por decision de importance/long_term).*")

    return _json_response(
        "\n".join(pretty_lines),
        topic=topic,
        importance=imp,
        relevance=relevance,
        working_memory=wm_res,
        long_term=lt_res,
        long_term_enabled=lt_enabled,
    )


def context_snapshot(level: str = "light") -> str:
    """
    Macro-tool: devuelve estado en una sola llamada.
    - light: working memory + workspace + recordatorios externos
    - full: despertar_codi()

    Returns JSON string con pretty + componentes.
    """
    lvl = (level or "light").strip().lower()
    if lvl not in VALID_SNAPSHOT_LEVELS:
        lvl = "light"

    if lvl == "full":
        out = despertar_codi()
        return _json_response(out, level="full")

    # light
    pretty_lines = ["# CONTEXT SNAPSHOT (light)\n"]

    wm_raw = ""
    wm_pretty = ""
    try:
        wm_raw = get_working_memory()
        wm_j = json.loads(wm_raw)
        wm_pretty = wm_j.get("pretty", "")
        pretty_lines.append("## Working Memory\n" + (wm_pretty or "*Vacia.*"))
    except Exception:
        pretty_lines.append("## Working Memory\n*No disponible.*")

    ws = ""
    try:
        ws = get_workspace_state()
        pretty_lines.append("\n## Global Workspace\n" + ws)
    except Exception:
        pretty_lines.append("\n## Global Workspace\n*No disponible.*")

    rec = ""
    try:
        rec = _ver_recordatorios_externos()
        if rec and rec.strip():
            pretty_lines.append("\n## Recordatorios externos\n" + rec)
    except Exception:
        pass

    return _json_response(
        "\n".join(pretty_lines),
        level="light",
        working_memory=wm_raw,
        workspace=ws,
        recordatorios=rec,
    )


def register_tools(mcp):
    """Registra las 3 macro-tools de interfaz en el servidor MCP."""
    mcp.tool()(recall)
    mcp.tool()(remember)
    mcp.tool()(context_snapshot)
