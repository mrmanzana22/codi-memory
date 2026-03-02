import atexit
import json
import logging
import os
import re
import threading
from typing import Any, Dict, List, Optional

_logger = logging.getLogger(__name__)

# Proposal #57: Thread-local flag to prevent WM double-push from remember()
_remember_ctx = threading.local()

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
from modules.tracing import new_trace_id, get_trace_id


# ============================================================
# INTERFACE LAYER (Fase 2.5) - 3 macro-tools
# recall / remember / context_snapshot
# ============================================================

# Write mode: sync (default) | shadow (sync + enqueue) | async (enqueue only)
# Reads from: 1) env var CODI_WRITE_MODE, 2) file .write_mode in project root
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_WRITE_MODE_FILE = os.path.join(_PROJECT_ROOT, ".write_mode")
_REMEMBER_MODE_FILE = os.path.join(_PROJECT_ROOT, ".remember_mode")

# Sync-compare thread limits (module-level for monkeypatching in tests)
_SYNC_COMPARE_SEMAPHORE = threading.BoundedSemaphore(5)
_SYNC_COMPARE_SEMAPHORE_CAPACITY = 5
_SYNC_COMPARE_TIMEOUT = 90  # seconds
_SYNC_COMPARE_ACTIVE = 0
_SYNC_COMPARE_LOCK = threading.Lock()
_SYNC_COMPARE_THREADS: list = []  # track live bg threads for drain

_DRAIN_TIMEOUT = 10  # max seconds to wait for compare threads on shutdown


def drain_sync_compares(timeout: float = 0) -> int:
    """Wait for in-flight sync-compare threads to finish.

    Called by atexit or manually.  Returns number of threads that
    were still alive after timeout (0 = clean drain).
    """
    t = timeout or _DRAIN_TIMEOUT
    deadline = __import__("time").time() + t
    alive = 0
    for th in list(_SYNC_COMPARE_THREADS):
        remaining = max(0, deadline - __import__("time").time())
        if remaining <= 0:
            break
        th.join(timeout=remaining)
        if th.is_alive():
            alive += 1
    # Clean up dead refs
    _SYNC_COMPARE_THREADS[:] = [th for th in _SYNC_COMPARE_THREADS if th.is_alive()]
    if alive:
        _logger.warning("[drain] %d sync-compare threads still alive after %.1fs", alive, t)
    else:
        _logger.info("[drain] all sync-compare threads drained cleanly")
    return alive


atexit.register(drain_sync_compares)

def _get_write_mode() -> str:
    mode = os.environ.get("CODI_WRITE_MODE", "").strip().lower()
    if mode:
        return mode
    try:
        with open(_WRITE_MODE_FILE) as f:
            return f.read().strip().lower() or "sync"
    except FileNotFoundError:
        return "sync"


VALID_WRITE_MODES = {"sync", "async", "shadow", "dual_ack"}


def _get_remember_mode() -> str:
    """Get write mode for remember(), with per-tool override.

    Resolution order:
      1. CODI_REMEMBER_MODE env var (if set and valid)
      2. .remember_mode file in project root (if exists)
      3. _get_write_mode() (global fallback)
    """
    mode = os.environ.get("CODI_REMEMBER_MODE", "").strip().lower()
    if mode and mode in VALID_WRITE_MODES:
        return mode
    try:
        with open(_REMEMBER_MODE_FILE) as f:
            fmode = f.read().strip().lower()
            if fmode and fmode in VALID_WRITE_MODES:
                return fmode
    except FileNotFoundError:
        pass
    return _get_write_mode()

VALID_IMPORTANCE = {"critical", "high", "medium", "low", "auto"}
VALID_RECALL_MODES = {"auto", "memory", "theme", "ownership", "emotion", "timeline", "counterfactual"}
VALID_SNAPSHOT_LEVELS = {"light", "full"}


def _json_response(pretty: str, **extra: Any) -> str:
    payload: Dict[str, Any] = {"pretty": pretty}
    payload.update(extra)
    tid = get_trace_id()
    if tid:
        payload["trace_id"] = tid
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


def _text_importance_signal(content: str) -> float:
    """Convert text to continuous importance prior signal (0.0-1.0).

    J.1.4: Replaces discrete heuristic with continuous signal used as
    Beta-Binomial prior. Craik & Lockhart 1972: depth of processing.
    """
    c = content.lower()
    score = 0.50  # neutral prior
    # Urgency / high-importance markers
    if any(k in c for k in ["urgente", "important", "importante", "recuerda",
                              "no olvidar", "deadline", "critico", "critical"]):
        score += 0.25
    # Length = depth of processing (Craik & Lockhart 1972)
    if len(content) >= 200:
        score += 0.15
    elif len(content) >= 100:
        score += 0.05
    # Short or trivial = shallow trace
    stripped = content.strip()
    if _is_trivial_content(stripped):
        score -= 0.30
    elif len(content) <= 50:
        score -= 0.15
    return min(1.0, max(0.0, score))


def _auto_importance_from_text(content: str, access_count: int = 0,
                                decay_count: int = 0) -> str:
    """Bayesian importance: Beta-Binomial posterior (J.1.4).

    Prior: informed by text heuristic (Craik & Lockhart 1972 depth of processing)
    Update: each access → alpha += 1 (successful retrieval strengthens importance)
             each decay without access → beta += 1 (forgetting signal)
    Posterior mean: alpha / (alpha + beta)

    Bjork & Bjork 1992: retrieval success is the signal of storage strength.
    Kahneman & Tversky 1973: availability heuristic (accessed = more available).

    Args:
        content: Memory text (for text-based prior)
        access_count: Number of successful retrievals (default 0 for new memories)
        decay_count: Number of decay cycles without access (default 0)
    """
    # Step 1: Text signal as informed prior (not Beta(1,1) flat)
    text_signal = _text_importance_signal(content)
    # Weight text as equivalent to 5 prior "observations"
    N_PRIOR = 5
    alpha_prior = max(0.5, text_signal * N_PRIOR)
    beta_prior = max(0.5, (1.0 - text_signal) * N_PRIOR)

    # Step 2: Update with access history
    alpha_post = alpha_prior + access_count
    beta_post = beta_prior + decay_count

    # Posterior mean
    posterior_mean = alpha_post / (alpha_post + beta_post)

    # Step 3: Map to importance level
    if posterior_mean >= 0.75:
        return "critical"
    if posterior_mean >= 0.55:
        return "high"
    if posterior_mean >= 0.35:
        return "medium"
    return "low"


def _is_trivial_content(text: str) -> bool:
    """Detect trivial content: dates, numbers, short status lines."""
    import re
    # Pure date/time patterns
    if re.match(r'^(Date:\s*)?[\d\-/T:\.Z\s]+$', text):
        return True
    # Pure numbers or IDs
    if re.match(r'^[a-f0-9\-]{8,}$', text):
        return True
    # Very short status like "OK", "done", "listo"
    if len(text) <= 15 and not any(c.isupper() and i > 0 for i, c in enumerate(text)):
        return True
    return False


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
    new_trace_id()
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

    if mode == "counterfactual":
        try:
            from modules.counterfactual import run_counterfactual
            cf_result = run_counterfactual(q)
            if "error" in cf_result:
                out = cf_result.get("message", "Not a counterfactual query.")
                add_result("counterfactual_error", out, {"error": cf_result["error"]})
                pretty_lines.append("## Counterfactual\n" + out)
            else:
                prediction = cf_result.get("prediction", {})
                narrative = prediction.get("narrative", "")
                add_result("counterfactual", narrative, {
                    "intervention_target": cf_result["parsed"]["intervention_target"],
                    "factual_memories": prediction.get("factual_memories", 0),
                    "alt_memories": prediction.get("alt_memories", 0),
                    "confidence": prediction.get("confidence", 0),
                })
                pretty_lines.append("## Counterfactual Analysis\n" + narrative)
        except Exception as e:
            err_msg = f"Counterfactual error: {str(e)[:100]}"
            add_result("counterfactual_error", err_msg, {})
            pretty_lines.append("## Counterfactual\n" + err_msg)
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

    # Spotlight conditional update on strong recall signal
    try:
        from modules.spotlight import should_update_from_recall, get_spotlight, update_spotlight_from_text, set_spotlight
        if results and should_update_from_recall(results):
            combined = " | ".join(r.get("text", "")[:100] for r in results[:3])
            updated = update_spotlight_from_text(get_spotlight(), combined, "recall")
            set_spotlight(updated)
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
    new_trace_id()
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

        write_mode = _get_remember_mode()

        if write_mode == "async":
            # Async: enqueue only, return immediate ACK
            from modules.write_queue import enqueue_write_job, compute_dedupe_key
            dedupe_key = compute_dedupe_key("remember", content)
            enqueue_result = enqueue_write_job(
                kind="remember",
                payload={"content": content, "category": topic, "source": ms_source, "importance": imp},
                priority=3 if imp in ("critical", "high") else 5,
                dedupe_key=dedupe_key,
                session_id=get_trace_id(),
            )
            preview = content[:120] + "..." if len(content) > 120 else content
            lt_res = json.dumps({
                "action": "queued",
                "mode": write_mode,
                "job_id": enqueue_result["job_id"],
                "kind": "remember",
                "preview": preview,
                "tags": [topic, imp],
                "eta_seconds": 30,
                "dedupe_hit": enqueue_result["dedupe_hit"],
                "check": f"write_status('{enqueue_result['job_id']}')",
                "cancel": f"cancel_write('{enqueue_result['job_id']}')",
            }, ensure_ascii=False)
        elif write_mode == "dual_ack":
            # Dual-ACK: async fires first (immediate ACK), background sync compares
            from modules.write_queue import enqueue_write_job, compute_dedupe_key
            from modules.dual_compare import (
                record_async_intent, record_sync_result,
                compute_request_fingerprint, update_sync_compare_status,
            )

            trace_id = get_trace_id()
            dedupe_key = compute_dedupe_key("remember", content)
            fingerprint = compute_request_fingerprint("remember", content, trace_id)

            # 1. Enqueue async write job
            enqueue_result = enqueue_write_job(
                kind="remember",
                payload={"content": content, "category": topic, "source": ms_source, "importance": imp},
                priority=3 if imp in ("critical", "high") else 5,
                dedupe_key=dedupe_key,
                session_id=trace_id,
            )
            job_id = enqueue_result["job_id"]

            # 2. Record async intent in dual_compare_log
            record_async_intent(
                fingerprint=fingerprint,
                trace_id=trace_id,
                kind="remember",
                async_job_id=job_id,
                write_mode="dual_ack",
            )

            # 3. Build immediate ACK (same shape as async)
            preview = content[:120] + "..." if len(content) > 120 else content
            lt_res = json.dumps({
                "action": "queued",
                "mode": "dual_ack",
                "job_id": job_id,
                "kind": "remember",
                "preview": preview,
                "tags": [topic, imp],
                "eta_seconds": 30,
                "dedupe_hit": enqueue_result["dedupe_hit"],
                "check": f"write_status('{job_id}')",
                "cancel": f"cancel_write('{job_id}')",
            }, ensure_ascii=False)

            # 4. Launch background sync compare with semaphore + timeout
            def _bg_sync_compare():
                import modules.interface as _iface
                sem = _iface._SYNC_COMPARE_SEMAPHORE
                timeout_val = _iface._SYNC_COMPARE_TIMEOUT
                acquired = sem.acquire(blocking=False)
                if not acquired:
                    update_sync_compare_status(
                        job_id, "skipped_capacity",
                        error="semaphore full",
                    )
                    return
                try:
                    with _iface._SYNC_COMPARE_LOCK:
                        _iface._SYNC_COMPARE_ACTIVE += 1
                    result_box = {}

                    def _inner():
                        try:
                            _remember_ctx.wm_pushed = True
                            result_box["output"] = add_memory_smart(
                                content=content, category=topic,
                                source=ms_source, importance=imp,
                            )
                        except Exception as exc:
                            result_box["error"] = str(exc)
                        finally:
                            _remember_ctx.wm_pushed = False

                    inner_thread = threading.Thread(target=_inner, daemon=True)
                    inner_thread.start()
                    inner_thread.join(timeout=timeout_val)

                    if inner_thread.is_alive():
                        update_sync_compare_status(
                            job_id, "timeout",
                            error=f"timeout after {timeout_val}s",
                        )
                        return

                    if "error" in result_box:
                        update_sync_compare_status(
                            job_id, "error",
                            error=result_box["error"],
                        )
                        return

                    record_sync_result(
                        fingerprint=fingerprint,
                        trace_id=trace_id,
                        kind="remember",
                        raw_output=result_box.get("output"),
                        async_job_id=job_id,
                        async_status=enqueue_result["status"],
                    )
                    update_sync_compare_status(job_id, "ok")
                finally:
                    with _iface._SYNC_COMPARE_LOCK:
                        _iface._SYNC_COMPARE_ACTIVE -= 1
                    sem.release()

            bg_thread = threading.Thread(target=_bg_sync_compare, daemon=True,
                                        name=f"sync-compare-{job_id[:8]}")
            bg_thread.start()
            _SYNC_COMPARE_THREADS.append(bg_thread)
            # Prune dead threads to prevent unbounded list growth
            _SYNC_COMPARE_THREADS[:] = [t for t in _SYNC_COMPARE_THREADS if t.is_alive()]

        elif write_mode == "shadow":
            # Shadow: sync first, then enqueue + record dual compare
            _remember_ctx.wm_pushed = True
            try:
                lt_res = add_memory_smart(content=content, category=topic, source=ms_source, importance=imp)
            finally:
                _remember_ctx.wm_pushed = False
            try:
                from modules.write_queue import enqueue_write_job, compute_dedupe_key
                from modules.dual_compare import record_sync_result, compute_request_fingerprint
                trace_id = get_trace_id()
                dedupe_key = compute_dedupe_key("remember", content)
                fingerprint = compute_request_fingerprint("remember", content, trace_id)
                enqueue_result = enqueue_write_job(
                    kind="remember",
                    payload={"content": content, "category": topic, "source": ms_source, "importance": imp},
                    dedupe_key=dedupe_key,
                    session_id=trace_id,
                )
                record_sync_result(
                    fingerprint=fingerprint,
                    trace_id=trace_id,
                    kind="remember",
                    raw_output=lt_res,
                    async_job_id=enqueue_result["job_id"],
                    async_status=enqueue_result["status"],
                )
            except Exception as e:
                from modules.secret_redact import redact_secrets
                _logger.error("dual enqueue failed: %s: %s", type(e).__name__, redact_secrets(str(e)))
        else:
            # Sync (default): current behavior
            _remember_ctx.wm_pushed = True
            try:
                lt_res = add_memory_smart(content=content, category=topic, source=ms_source, importance=imp)
            finally:
                _remember_ctx.wm_pushed = False

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
    new_trace_id()
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

    # Goals (Sprint 15)
    goals_str = ""
    try:
        from modules.goals import get_context_goals
        gctx = get_context_goals(limit=5)
        if gctx and gctx.get("goals"):
            lines_g = [f"- [{g['level']}][{g['priority']}] {g['title']} (act={g['activation']})"
                       for g in gctx["goals"]]
            goals_str = f"Interference={gctx['interference_level']} | {gctx['above_threshold']}/{gctx['total_active']} above\n" + "\n".join(lines_g)
            pretty_lines.append("\n## Active Goals\n" + goals_str)
    except Exception:
        pass

    return _json_response(
        "\n".join(pretty_lines),
        level="light",
        working_memory=wm_raw,
        workspace=ws,
        recordatorios=rec,
        goals=goals_str,
    )


def get_runtime_flags() -> str:
    """Return current runtime configuration for debugging.

    Shows resolved write/remember modes, flag file paths,
    and sync-compare thread state.
    """
    write_mode = _get_write_mode()
    remember_mode = _get_remember_mode()

    import modules.interface as _iface
    with _iface._SYNC_COMPARE_LOCK:
        active = _iface._SYNC_COMPARE_ACTIVE

    flags = {
        "write_mode": write_mode,
        "write_mode_file": _WRITE_MODE_FILE,
        "write_mode_file_exists": os.path.exists(_WRITE_MODE_FILE),
        "remember_mode": remember_mode,
        "remember_mode_file": _REMEMBER_MODE_FILE,
        "remember_mode_file_exists": os.path.exists(_REMEMBER_MODE_FILE),
        "sync_compare": {
            "timeout_seconds": _iface._SYNC_COMPARE_TIMEOUT,
            "semaphore_capacity": _iface._SYNC_COMPARE_SEMAPHORE_CAPACITY,
            "active_threads": active,
        },
        "env_overrides": {
            "CODI_WRITE_MODE": os.environ.get("CODI_WRITE_MODE", "(not set)"),
            "CODI_REMEMBER_MODE": os.environ.get("CODI_REMEMBER_MODE", "(not set)"),
        },
    }

    lines = [
        "# RUNTIME FLAGS",
        f"- write_mode: **{write_mode}** (file exists: {flags['write_mode_file_exists']})",
        f"- remember_mode: **{remember_mode}** (file exists: {flags['remember_mode_file_exists']})",
        f"- sync_compare: timeout={flags['sync_compare']['timeout_seconds']}s, "
        f"capacity={flags['sync_compare']['semaphore_capacity']}, "
        f"active={active}",
        f"- env: CODI_WRITE_MODE={flags['env_overrides']['CODI_WRITE_MODE']}, "
        f"CODI_REMEMBER_MODE={flags['env_overrides']['CODI_REMEMBER_MODE']}",
    ]

    return _json_response("\n".join(lines), flags=flags)


def register_tools(mcp):
    """Registra las 3 macro-tools de interfaz en el servidor MCP."""
    mcp.tool()(recall)
    mcp.tool()(remember)
    mcp.tool()(context_snapshot)
    mcp.tool()(get_runtime_flags)
