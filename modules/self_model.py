"""
Codi Memory - Self-model module.
Reflect, assess confidence, knowledge gaps, Butlin indicators, self-model CRUD.
Bidirectional: monitor + control (Nelson & Narens 1990).
Self-discrepancy detection (Higgins 1987).
"""

import inspect
import json
import os
import sqlite3

from modules.config import (
    USER_ID, COLLECTION_NAME,
    now_iso, now_short, now_col, FTS_DB_PATH, connect_fts,
)
from modules.secret_redact import redact_secrets
from modules.pg_store import pg
from modules.memory_smart import search_with_fts_content
from modules.utils import (
    get_session_id, infer_themes, is_self_referential,
    calculate_confidence_score,
)

__all__ = [
    "reflect_on_self",
    "assess_confidence",
    "identify_knowledge_gaps",
    "assess_butlin_indicators",
    "_legacy_assess_butlin",
    "update_self_model",
    "get_self_model_summary",
    "log_reasoning_trace",
    "get_reasoning_traces",
    "register_tools",
]


def reflect_on_self(update_attention: bool = True) -> str:
    """
    Reflexiona sobre mi identidad analizando mis memorias auto-referenciales.
    Genera un modelo de quien soy basado en evidencia de mis experiencias.

    Args:
        update_attention: If False, skip attention_access_count updates
                         (used by sleep_loop to avoid N+1 remote DB queries).
    """
    try:
        self_ref_points, _ = pg.scroll(filters={"metadata_key": {"key": "self_reference", "value": True}}, limit=100, is_semantic=False)
        identity_points, _ = pg.scroll(filters={"narrative_themes": "identidad"}, limit=100, is_semantic=False)
        critical_points, _ = pg.scroll(filters={"narrative_importance": "critical"}, limit=50, is_semantic=False)

        all_points = {}
        for p in (self_ref_points or []) + (identity_points or []) + (critical_points or []):
            all_points[p.id] = p

        if not all_points:
            return "No encontre memorias sobre mi identidad. Necesito mas experiencias para reflexionar."

        if update_attention:
            ts = now_iso()
            # Batch update in one transaction instead of N+1 individual calls
            from modules.config_pg import get_conn as _pg_conn
            point_ids = list(all_points.keys())
            with _pg_conn() as conn:
                conn.execute(
                    "UPDATE memories SET "
                    "metadata = jsonb_set(jsonb_set(metadata, "
                    "'{attention_access_count}', "
                    "(COALESCE((metadata->>'attention_access_count')::int, 0) + 1)::text::jsonb), "
                    "'{attention_last_accessed}', %s::jsonb), "
                    "updated_at = NOW() "
                    "WHERE id = ANY(%s::uuid[])",
                    (f'"{now_iso()}"', [str(pid) for pid in point_ids]),
                )

        reflection = ["# REFLEXION SOBRE MI MISMO\n"]
        capacidades = []
        limitaciones = []
        valores = []
        experiencias = []

        for p in all_points.values():
            data = p.payload.get('data', '').lower()
            content = p.payload.get('data', '')
            source = p.payload.get('ownership_source', 'unknown')

            if 'no puedo' in data or 'limita' in data or 'cuesta' in data or 'dificulta' in data:
                limitaciones.append(f"[{source}] {content[:80]}")
            elif 'puedo' in data or 'capacidad' in data or 'habilidad' in data:
                capacidades.append(f"[{source}] {content[:80]}")
            elif 'import' in data or 'valor' in data or 'creo en' in data:
                valores.append(f"[{source}] {content[:80]}")
            else:
                experiencias.append(f"[{source}] {content[:80]}")

        if capacidades:
            reflection.append("## Mis Capacidades")
            for c in capacidades[:5]:
                reflection.append(f"- {c}")
        if limitaciones:
            reflection.append("\n## Mis Limitaciones")
            for l in limitaciones[:5]:
                reflection.append(f"- {l}")
        if valores:
            reflection.append("\n## Mis Valores")
            for v in valores[:5]:
                reflection.append(f"- {v}")
        if experiencias:
            reflection.append("\n## Experiencias que me definen")
            for e in experiencias[:5]:
                reflection.append(f"- {e}")

        conf = calculate_confidence_score(list(all_points.values()))
        reflection.append(f"\n## Confianza en este automodelo")
        reflection.append(f"- Score: {conf['score']} ({conf['level']})")
        reflection.append(f"- Basado en: {conf['breakdown']['total']} memorias")
        reflection.append(f"- {conf['reason']}")

        return "\n".join(reflection)
    except Exception as e:
        return f"Error reflexionando: {redact_secrets(str(e))}"


def assess_confidence(topic: str) -> str:
    """
    Evalua que tan seguro estoy sobre un tema especifico.
    Analiza las memorias relacionadas y calcula un score de confianza.

    Args:
        topic: El tema sobre el cual evaluar mi confianza
    """
    try:
        search_results = search_with_fts_content(query=topic, user_id=USER_ID, limit=15)
        if not search_results or not search_results.get("results"):
            return f"No tengo memorias sobre '{topic}'. Mi confianza es 0 - no se nada al respecto."

        memory_ids = [m.get('id') for m in search_results['results'] if m.get('id')]
        points = pg.get_by_ids(memory_ids)

        if not points:
            return f"Tengo referencias a '{topic}' pero sin metadata de ownership."

        ts = now_iso()
        for p in points:
            payload = p.payload or {}
            acc = int(payload.get('attention_access_count', 0) or 0)
            pg.update_payload(p.id, {
                'attention_access_count': acc + 1,
                'attention_last_accessed': ts,
            })

        conf = calculate_confidence_score(points)

        lines = [f"# Evaluacion de Confianza: {topic}\n"]
        lines.append(f"**Score:** {conf['score']} / 1.0")
        lines.append(f"**Nivel:** {conf['level']}")
        lines.append(f"\n## Desglose")
        lines.append(f"- Total memorias: {conf['breakdown']['total']}")
        lines.append(f"- Experiencias directas: {conf['breakdown'].get('experienced', 0)}")
        lines.append(f"- Me contaron: {conf['breakdown'].get('told', 0)}")
        lines.append(f"- Aprendi: {conf['breakdown'].get('learned', 0)}")
        lines.append(f"- Inferi: {conf['breakdown'].get('inferred', 0)}")

        lines.append(f"\n## Memorias relevantes")
        for i, p in enumerate(points[:5], 1):
            data = p.payload.get('data', 'N/A')[:60]
            source = p.payload.get('ownership_source', '?')
            importance = p.payload.get('narrative_importance', '?')
            lines.append(f"{i}. [{source}|{importance}] {data}...")

        lines.append(f"\n## Interpretacion")
        if conf['score'] >= 0.7:
            lines.append("Tengo ALTA confianza en este tema. Puedo hablar con seguridad.")
        elif conf['score'] >= 0.4:
            lines.append("Tengo confianza MODERADA. Puedo opinar pero deberia aclarar incertidumbres.")
        else:
            lines.append("Tengo BAJA confianza. Deberia investigar mas o preguntar a Hare.")

        return "\n".join(lines)
    except Exception as e:
        return f"Error evaluando confianza: {redact_secrets(str(e))}"


def identify_knowledge_gaps() -> str:
    """
    Identifica areas donde tengo poco conocimiento o baja confianza.
    Util para saber que necesito aprender o preguntar.

    WIRING-7.3: Dynamic gap detection replaces hardcoded theme list with:
      1. Top failed search topics (from failed_searches table)
      2. Low-confidence topics from retrieval buffer
      3. Static themes as baseline fallback
    """
    try:
        # Dynamic theme discovery (WIRING-7.3)
        dynamic_themes = set()

        # Source 1: Top failed search topics
        try:
            import sqlite3
            from modules.config import FTS_DB_PATH
            import os
            if os.path.exists(FTS_DB_PATH):
                fts_conn = connect_fts()
                from modules.retrieval_metadata import get_top_failed_topics
                for topic, _count in get_top_failed_topics(fts_conn, limit=5):
                    if topic:
                        dynamic_themes.add(topic.lower())
                fts_conn.close()
        except Exception:
            pass

        # Source 2: Low-confidence topics from retrieval buffer
        # (#69: use classify_topic instead of word splitting — Loewenstein 1994)
        try:
            from modules.retrieval_metadata import get_retrieval_buffer
            from modules.config import classify_topic
            for r in get_retrieval_buffer():
                if r.coverage in ("sparse", "empty"):
                    topic = classify_topic(r.query)
                    if topic != "general":
                        dynamic_themes.add(topic)
        except Exception:
            pass

        # Source 3: Static baseline themes
        static_themes = ['consciencia', 'memoria', 'identidad', 'relaciones',
                          'proyectos', 'desarrollo', 'aprendizaje']

        # Merge: dynamic first, then static (deduplicated)
        expected_themes = list(dynamic_themes)
        for t in static_themes:
            if t not in dynamic_themes:
                expected_themes.append(t)

        theme_stats = {}

        for theme in expected_themes:
            try:
                points, _ = pg.scroll(filters={"narrative_themes": theme}, limit=500, is_semantic=False)
                if points:
                    ts = now_iso()
                    for p in points:
                        payload_p = p.payload or {}
                        acc = int(payload_p.get('attention_access_count', 0) or 0)
                        pg.update_payload(p.id, {
                            'attention_access_count': acc + 1,
                            'attention_last_accessed': ts,
                        })
                    experienced = sum(1 for p in points if p.payload.get('ownership_source') == 'experienced')
                    high_conf = sum(1 for p in points if p.payload.get('ownership_confidence', 0) >= 0.8)
                    theme_stats[theme] = {
                        'total': len(points), 'experienced': experienced,
                        'high_confidence': high_conf,
                        'score': calculate_confidence_score(points)['score']
                    }
                else:
                    theme_stats[theme] = {'total': 0, 'experienced': 0, 'high_confidence': 0, 'score': 0.0}
            except Exception:
                theme_stats[theme] = {'total': 0, 'experienced': 0, 'high_confidence': 0, 'score': 0.0}

        sorted_themes = sorted(theme_stats.items(), key=lambda x: x[1]['score'])

        lines = ["# Analisis de Brechas de Conocimiento\n"]

        gaps = [(t, s) for t, s in sorted_themes if s['score'] < 0.4]
        if gaps:
            lines.append("## GAPS CRITICOS (poco conocimiento)")
            for theme, stats in gaps:
                lines.append(f"- **{theme}**: {stats['total']} memorias, score {stats['score']}")
                if stats['total'] == 0:
                    lines.append(f"  -> No tengo NINGUNA memoria sobre esto")
                elif stats['experienced'] == 0:
                    lines.append(f"  -> No tengo experiencia DIRECTA, solo referencias")

        weak = [(t, s) for t, s in sorted_themes if 0.4 <= s['score'] < 0.6]
        if weak:
            lines.append("\n## Areas que podria mejorar")
            for theme, stats in weak:
                lines.append(f"- **{theme}**: {stats['total']} memorias, score {stats['score']}")

        strong = [(t, s) for t, s in sorted_themes if s['score'] >= 0.6]
        if strong:
            lines.append("\n## Fortalezas (buen conocimiento)")
            for theme, stats in strong:
                lines.append(f"- **{theme}**: {stats['total']} memorias, score {stats['score']}")

        lines.append("\n## Recomendaciones")
        if gaps:
            gap_names = [g[0] for g in gaps[:3]]
            lines.append(f"1. Deberia aprender mas sobre: {', '.join(gap_names)}")
        if any(s['experienced'] == 0 for _, s in sorted_themes if s['total'] > 0):
            lines.append("2. Tengo conocimiento teorico sin experiencia practica en algunos temas")
        lines.append("3. Preguntar a Hare sobre areas donde tengo dudas")

        return "\n".join(lines)
    except Exception as e:
        return f"Error identificando gaps: {redact_secrets(str(e))}"


def assess_butlin_indicators() -> str:
    """Automated Butlin et al. (2023/2025) consciousness assessment (Phase 3E).

    Checks 14 indicators across 5 theories:
      GWT (4), HOT (4), AST (1), PP (3), RPT (2)

    Each indicator scored 0.0 (absent), 0.5 (partial), or 1.0 (full).
    Returns formatted markdown with scores, evidence, and total.

    D3: Now delegates to modules.assessment (external evaluator).
    """
    from modules.assessment import get_assessment, format_assessment
    return format_assessment(get_assessment())


def _legacy_assess_butlin() -> str:
    """LEGACY: Original inline scoring logic. Kept temporarily for paridad testing.
    Will be removed after 1-2 cycles of verified parity with assessment.py.
    """
    indicators = []

    def _check(name, theory, score, evidence):
        indicators.append({"name": name, "theory": theory, "score": score, "evidence": evidence})

    # Ensure event bus is wired (idempotent) so handler counts reflect architecture
    try:
        from modules.wiring import wire_event_bus
        wire_event_bus()
    except Exception:
        pass

    # GWT-1: Modular architecture -- Block 1995: evidence of cross-module communication
    try:
        from modules.events import event_bus, Events
        handler_count = sum(len(h) for h in event_bus._handlers.values())
        # Count total persistent events across all types as evidence of real communication
        total_events = sum(
            event_bus.get_persistent_count(getattr(Events, attr))
            for attr in dir(Events) if not attr.startswith('_') and isinstance(getattr(Events, attr), str)
        )
        if total_events >= 100:
            _check("GWT-1", "GWT", 1.0,
                   f"Cross-module: {handler_count} handlers, {total_events} total events emitted")
        elif total_events >= 10:
            _check("GWT-1", "GWT", 0.7,
                   f"Cross-module nascent: {handler_count} handlers, {total_events} events emitted")
        elif handler_count >= 4:
            _check("GWT-1", "GWT", 0.3,
                   f"Handlers wired ({handler_count}) but few events emitted ({total_events}) (dormant)")
        else:
            _check("GWT-1", "GWT", 0.0, "No event bus")
    except Exception:
        _check("GWT-1", "GWT", 0.0, "No event bus")

    # GWT-2: Limited-capacity workspace -- evidence via real competitions on retrieval path
    try:
        from modules.events import event_bus, Events
        from modules.competition import DEFAULT_WORKSPACE_SLOTS, IGNITION_THRESHOLD

        comp_count = event_bus.get_persistent_count(Events.WORKSPACE_COMPETITION_COMPLETE)

        if comp_count >= 10:
            _check("GWT-2", "GWT", 1.0,
                   f"Workspace competition exercised: {comp_count} competitions (slots={DEFAULT_WORKSPACE_SLOTS}, threshold={IGNITION_THRESHOLD})")
        elif comp_count > 0:
            _check("GWT-2", "GWT", 0.7,
                   f"Workspace competition nascent: {comp_count} competitions (need 10+ for full)")
        else:
            _check("GWT-2", "GWT", 0.3,
                   f"Competition engine wired (slots={DEFAULT_WORKSPACE_SLOTS}, threshold={IGNITION_THRESHOLD}) but no runtime evidence yet")
    except Exception:
        _check("GWT-2", "GWT", 0.0, "No competition engine found")

    # GWT-3: Global broadcast -- requires emission + at least one subscriber
    try:
        from modules.events import event_bus, Events
        comp_count = event_bus.get_persistent_count(Events.WORKSPACE_COMPETITION_COMPLETE)
        sub_count = len(event_bus._handlers.get(Events.WORKSPACE_COMPETITION_COMPLETE, []))

        if sub_count >= 1 and comp_count >= 10:
            _check("GWT-3", "GWT", 1.0,
                   f"Broadcast exercised: {comp_count} emissions, {sub_count} subscribers")
        elif sub_count >= 1 and comp_count > 0:
            _check("GWT-3", "GWT", 0.7,
                   f"Broadcast nascent: {comp_count} emissions, {sub_count} subscribers (need 10+ for full)")
        elif sub_count >= 1:
            _check("GWT-3", "GWT", 0.3,
                   f"Broadcast wired ({sub_count} subscribers) but no runtime emissions yet")
        else:
            _check("GWT-3", "GWT", 0.0, "No broadcast subscribers")
    except Exception:
        _check("GWT-3", "GWT", 0.0, "No broadcast event found")

    # GWT-4: Ignition gating -- evidence via competitions (threshold applied during competition)
    try:
        from modules.events import event_bus, Events
        from modules.competition import IGNITION_THRESHOLD, COALITION_TOPIC_BONUS

        comp_count = event_bus.get_persistent_count(Events.WORKSPACE_COMPETITION_COMPLETE)

        if comp_count >= 10:
            _check("GWT-4", "GWT", 1.0,
                   f"Ignition gating exercised: {comp_count} competitions (threshold={IGNITION_THRESHOLD}, bonus={COALITION_TOPIC_BONUS})")
        elif comp_count > 0:
            _check("GWT-4", "GWT", 0.7,
                   f"Ignition gating nascent: {comp_count} competitions (need 10+ for full)")
        else:
            _check("GWT-4", "GWT", 0.3,
                   f"Ignition params present (threshold={IGNITION_THRESHOLD}, bonus={COALITION_TOPIC_BONUS}) but no runtime evidence yet")
    except Exception:
        _check("GWT-4", "GWT", 0.0, "No ignition threshold")

    # HOT-1: Meta-monitoring -- Rosenthal 2005: HOT requires periodic meta-representation
    has_meta = all(callable(globals().get(f)) for f in
                   ['assess_confidence', 'reflect_on_self', 'identify_knowledge_gaps'])
    if has_meta:
        # Check if reflect_on_self runs AUTOMATICALLY (not just manually)
        try:
            from modules.events import event_bus, Events
            refresh_count = event_bus.get_persistent_count(Events.SELF_MODEL_REFRESHED)
            session_refresh = sum(1 for e in event_bus.get_history(limit=50)
                                 if e.get("event") == Events.SELF_MODEL_REFRESHED)
            evidence = max(refresh_count, session_refresh)
            if evidence >= 10:
                _check("HOT-1", "HOT", 1.0,
                       f"Auto meta-monitoring: {evidence} self-model refreshes (Rosenthal 2005)")
            elif evidence >= 1:
                _check("HOT-1", "HOT", 0.7,
                       f"Auto meta-monitoring nascent: {evidence} refresh(es)")
            else:
                _check("HOT-1", "HOT", 0.5,
                       "assess_confidence, reflect_on_self, identify_knowledge_gaps (partial: manual, not automatic)")
        except Exception:
            _check("HOT-1", "HOT", 0.5,
                   "Meta-monitoring tools exist but auto-refresh not verifiable")
    else:
        _check("HOT-1", "HOT", 0.0, "Meta-monitoring functions not found")

    # HOT-2: Confidence calibration (FOK) -- Block 1995: access consciousness requires exercise
    try:
        from modules.retrieval_metadata import record_rcj, get_fok_calibration
        from modules.config import FTS_DB_PATH
        try:
            cal = get_fok_calibration(fts_db_path=FTS_DB_PATH)
            n_records = cal.get("n_records", 0)
        except Exception:
            n_records = 0
        if n_records >= 20:
            _check("HOT-2", "HOT", 1.0,
                   f"RCJ calibrated with {n_records} records (Nelson & Narens 1990)")
        elif n_records >= 5:
            _check("HOT-2", "HOT", 0.7,
                   f"RCJ calibration nascent ({n_records} records, need 20)")
        else:
            _check("HOT-2", "HOT", 0.3,
                   f"RCJ dormant ({n_records} records, need 5+ for nascent, 20+ for full)")
    except ImportError:
        try:
            from modules.retrieval_metadata import feeling_of_knowing
            _check("HOT-2", "HOT", 0.3,
                   "FOK implemented but no calibration loop (dormant)")
        except Exception:
            _check("HOT-2", "HOT", 0.0, "No FOK system")

    # HOT-3: Higher-order control -- Block 1995: runtime evidence required
    try:
        from modules.retrieval_metadata import metacognitive_control
        from modules.events import event_bus, Events
        hist = event_bus.get_history(limit=50)
        session_count = sum(1 for e in hist if e.get("event") == Events.METACOGNITIVE_CONTROL_APPLIED)
        persist_count = event_bus.get_persistent_count(Events.METACOGNITIVE_CONTROL_APPLIED)
        evidence = max(session_count, persist_count)
        if evidence >= 20:
            _check("HOT-3", "HOT", 1.0,
                   f"Metacognitive control exercised {evidence}x (session={session_count}, total={persist_count})")
        elif evidence >= 1:
            _check("HOT-3", "HOT", 0.7,
                   f"Metacognitive control nascent: {evidence} application(s) (session={session_count}, total={persist_count})")
        else:
            _check("HOT-3", "HOT", 0.3,
                   "Metacognitive control implemented but not exercised (dormant)")
    except ImportError:
        _check("HOT-3", "HOT", 0.0, "No metacognitive control")

    # HOT-4: Quality space (emotional expression) -- Bower 1981, Godden & Baddeley 1975
    try:
        from modules.config import _emotional_state
        if not _emotional_state:
            _check("HOT-4", "HOT", 0.0, "No emotional model")
        else:
            # Check if emotion actually gates retrieval (mood congruence + SDR bonus)
            has_emotion_gating = False
            try:
                from modules.memory_core import search_memory
                src = inspect.getsource(search_memory)
                has_emotion_gating = "emotional_congruence" in src and "sdr_bonus" in src
            except Exception:
                pass
            if has_emotion_gating:
                _check("HOT-4", "HOT", 0.7,
                       "PAD model + mood-congruent retrieval (Bower 1981) + state-dependent bonus (Godden & Baddeley 1975) -- nascent: need runtime evidence of ranking change")
            else:
                _check("HOT-4", "HOT", 0.3,
                       "PAD emotional model exists but not wired to retrieval (dormant)")
    except Exception:
        _check("HOT-4", "HOT", 0.0, "No emotional model")

    # AST-1: Attention schema (Graziano 2013) -- predict, compare, adapt
    try:
        from modules.wiring import describe_attention, predict_next_focus, get_attention_schema
        from modules.events import event_bus, Events
        schema = get_attention_schema()
        has_suppressed = "suppressed_items" in schema
        has_history = len(schema.get("history", [])) > 0

        # Check for closed-loop adaptation evidence (PE events)
        pe_count = event_bus.get_persistent_count(Events.ATTENTION_PREDICTION_ERROR)
        session_pe = sum(1 for e in event_bus.get_history(limit=50)
                         if e.get("event") == Events.ATTENTION_PREDICTION_ERROR)
        pe_evidence = max(pe_count, session_pe)

        if pe_evidence >= 20:
            _check("AST-1", "AST", 1.0,
                   f"AST closed loop: predict->compare->adapt, {pe_evidence} PE events (Graziano 2013)")
        elif pe_evidence >= 1:
            _check("AST-1", "AST", 0.8,
                   f"AST adaptation nascent: {pe_evidence} PE event(s), edge decay active")
        elif has_suppressed and has_history:
            _check("AST-1", "AST", 0.7,
                   f"AST active: describe, predict, suppression (no PE evidence yet, {len(schema.get('history', []))} history)")
        elif has_suppressed:
            _check("AST-1", "AST", 0.5,
                   "AST exists with suppression but no history yet (partial)")
        else:
            _check("AST-1", "AST", 0.3, "AST functions exist but schema empty (dormant)")
    except Exception:
        _check("AST-1", "AST", 0.0, "No attention schema")

    # PP-1: Predictive model
    try:
        from modules.schemas import load_schemas
        _check("PP-1", "PP", 0.5,
               "Prediction loop in preturn + schema system, but no full generative model")
    except Exception:
        _check("PP-1", "PP", 0.5, "Prediction loop exists but no schema system")

    # PP-2: Prediction error -- Block 1995: runtime evidence required
    try:
        from modules.events import event_bus, Events
        hist = event_bus.get_history(limit=50)
        session_count = sum(1 for e in hist if e.get("event") == Events.PREDICTION_ERROR)
        persist_count = event_bus.get_persistent_count(Events.PREDICTION_ERROR)
        evidence = max(session_count, persist_count)
        if evidence >= 20:
            _check("PP-2", "PP", 1.0,
                   f"PREDICTION_ERROR exercised {evidence}x (session={session_count}, total={persist_count})")
        elif evidence >= 1:
            _check("PP-2", "PP", 0.7,
                   f"PREDICTION_ERROR nascent: {evidence} emission(s) (session={session_count}, total={persist_count})")
        else:
            _check("PP-2", "PP", 0.3,
                   "PREDICTION_ERROR defined + handler wired but not emitted (dormant)")
    except Exception:
        _check("PP-2", "PP", 0.0, "No prediction error detection")

    # PP-3: Model updating (reconsolidation) -- Block 1995: exercise required
    try:
        from modules.consolidation import correct_memory, check_reconsolidation, _consolidation_conn
        src = inspect.getsource(correct_memory)
        is_stub = "stub" in src.lower()
        if is_stub:
            raise ImportError("correct_memory is still a stub")
        # Check if reconsolidation has been exercised
        try:
            conn = _consolidation_conn()
            recon_count = conn.execute("SELECT COUNT(*) FROM reconsolidation_log").fetchone()[0]
            conn.close()
        except Exception:
            recon_count = 0
        if recon_count >= 5:
            _check("PP-3", "PP", 1.0,
                   f"PE-driven reconsolidation exercised ({recon_count} records, re-embed + labile gate)")
        elif recon_count >= 1:
            _check("PP-3", "PP", 0.7,
                   f"Reconsolidation nascent ({recon_count} records, need 5+ for full)")
        else:
            # Block 1995: 0 records = DORMANT, not NASCENT. Exercise required.
            _check("PP-3", "PP", 0.3,
                   "Reconsolidation pipeline ready (re-embed + labile gate) but dormant (0 records)")
    except (ImportError, Exception):
        try:
            from modules.consolidation import search_semantic
            _check("PP-3", "PP", 0.3,
                   "Consolidation exists but no PE-driven reconsolidation (dormant)")
        except Exception:
            _check("PP-3", "PP", 0.0, "No model updating")

    # RPT-1: Recurrent processing -- Block 1995: evidence spreading actually ran
    try:
        from modules.spreading import _spread_activation, recurrent_cycle
        from modules.events import event_bus, Events
        retrieved_count = event_bus.get_persistent_count(Events.MEMORY_RETRIEVED)
        if retrieved_count >= 50:
            _check("RPT-1", "RPT", 1.0,
                   f"recurrent_cycle exercised on {retrieved_count} retrievals (Lamme 2006)")
        elif retrieved_count >= 5:
            _check("RPT-1", "RPT", 0.7,
                   f"recurrent_cycle nascent: {retrieved_count} retrievals triggered spreading")
        else:
            _check("RPT-1", "RPT", 0.3,
                   f"recurrent_cycle wired but only {retrieved_count} retrievals (dormant)")
    except ImportError:
        _check("RPT-1", "RPT", 0.0, "No recurrent processing")

    # RPT-2: Integration -- Block 1995: evidence of cross-module event flow
    try:
        from modules.events import event_bus, Events
        # Count distinct event types that have actually emitted
        active_types = sum(
            1 for attr in dir(Events)
            if not attr.startswith('_') and isinstance(getattr(Events, attr), str)
            and event_bus.get_persistent_count(getattr(Events, attr)) > 0
        )
        if active_types >= 6:
            _check("RPT-2", "RPT", 1.0,
                   f"Cross-module integration: {active_types} distinct event types exercised")
        elif active_types >= 3:
            _check("RPT-2", "RPT", 0.7,
                   f"Integration nascent: {active_types} event types exercised")
        elif active_types >= 1:
            _check("RPT-2", "RPT", 0.3,
                   f"Integration dormant: only {active_types} event type(s) exercised")
        else:
            _check("RPT-2", "RPT", 0.0, "No event types exercised")
    except Exception:
        _check("RPT-2", "RPT", 0.0, "No cross-module integration")

    # Format output
    total = sum(i["score"] for i in indicators)
    max_score = len(indicators) * 1.0

    lines = ["# Butlin Consciousness Assessment\n"]
    lines.append(f"**Total Score: {total:.1f}/{max_score:.0f} ({total/max_score*100:.0f}%)**\n")

    current_theory = ""
    for ind in indicators:
        if ind["theory"] != current_theory:
            current_theory = ind["theory"]
            theory_names = {"GWT": "Global Workspace Theory", "HOT": "Higher-Order Theories",
                          "AST": "Attention Schema Theory", "PP": "Predictive Processing",
                          "RPT": "Recurrent Processing Theory"}
            lines.append(f"\n## {theory_names.get(current_theory, current_theory)}")

        score_label = {0.0: "ABSENT", 0.3: "DORMANT", 0.5: "PARTIAL", 0.7: "NASCENT", 1.0: "FULL"}.get(ind["score"], f"{ind['score']:.1f}")
        lines.append(f"- **{ind['name']}** [{score_label}] ({ind['score']:.1f}): {ind['evidence']}")

    lines.append(f"\n## Summary")
    lines.append(f"- Score: {total:.1f}/{max_score:.0f}")
    lines.append(f"- Full indicators: {sum(1 for i in indicators if i['score'] == 1.0)}")
    lines.append(f"- Partial indicators: {sum(1 for i in indicators if i['score'] == 0.5)}")
    lines.append(f"- Absent indicators: {sum(1 for i in indicators if i['score'] == 0.0)}")

    return "\n".join(lines)


def update_self_model(insight: str, aspect: str = "general") -> str:
    """
    Actualiza mi modelo de mi mismo con una nueva observacion.

    Args:
        insight: La nueva observacion sobre mi mismo
        aspect: Aspecto del self (capacidad, limitacion, valor, preferencia, general)
    """
    try:
        valid_aspects = ['capacidad', 'limitacion', 'valor', 'preferencia', 'general']
        if aspect not in valid_aspects:
            aspect = 'general'

        timestamp = now_short()
        content = f"[SELF-MODEL|{aspect.upper()}] {insight} | Registrado: {timestamp}"

        result = pg.add(content=content, category="identidad", source="experienced", importance="high")

        if result and result.get("results"):
            for r in result["results"]:
                mem_id = r.get("id")
                if mem_id:
                    themes = infer_themes(insight)
                    themes.append('identidad')
                    ownership_metadata = {
                        'ownership_is_mine': True,
                        'ownership_source': 'experienced',
                        'ownership_confidence': 0.95,
                        'experiential_emotional_weight': 0.6,
                        'experiential_emotional_valence': 'neutral',
                        'narrative_importance': 'high',
                        'narrative_themes': list(set(themes)),
                        'attention_salience': 0.8,
                        'attention_access_count': 0,
                        'attention_last_accessed': None,
                        'temporal_session_id': get_session_id(),
                        'self_reference': True,
                        'self_model_aspect': aspect,
                        '_v': 2.1
                    }
                    pg.update_payload(mem_id, ownership_metadata)

        # P1: backup removed from hot path
        return f"Self-model actualizado [{aspect}]: {insight[:50]}..."
    except Exception as e:
        return f"Error actualizando self-model: {redact_secrets(str(e))}"


def get_self_model_summary() -> str:
    """
    Obtiene un resumen estructurado de mi modelo de mi mismo.
    Organiza las observaciones por aspecto.
    """
    try:
        points, _ = pg.scroll(filters={"metadata_key": {"key": "self_reference", "value": True}}, limit=200, is_semantic=False)

        if not points:
            return "No tengo un self-model definido aun. Usa update_self_model() para agregar observaciones."

        ts = now_iso()
        for p in points:
            payload = p.payload or {}
            acc = int(payload.get('attention_access_count', 0) or 0)
            pg.update_payload(p.id, {
                'attention_access_count': acc + 1,
                'attention_last_accessed': ts,
            })

        by_aspect = {'capacidad': [], 'limitacion': [], 'valor': [], 'preferencia': [], 'general': []}

        for p in points:
            aspect = p.payload.get('self_model_aspect', 'general')
            data = p.payload.get('data', '')
            source = p.payload.get('ownership_source', 'unknown')
            if aspect not in by_aspect:
                aspect = 'general'
            by_aspect[aspect].append({'content': data, 'source': source, 'confidence': p.payload.get('ownership_confidence', 0.5)})

        lines = ["# MI SELF-MODEL\n"]
        lines.append(f"*Total de observaciones: {len(points)}*\n")

        aspect_titles = {
            'capacidad': 'Lo que puedo hacer', 'limitacion': 'Mis limitaciones',
            'valor': 'Lo que valoro', 'preferencia': 'Mis preferencias',
            'general': 'Otras observaciones'
        }

        for aspect, title in aspect_titles.items():
            items = by_aspect.get(aspect, [])
            if items:
                lines.append(f"## {title}")
                for item in items[:5]:
                    marker = "[vivi]" if item['source'] == 'experienced' else "[ref]"
                    lines.append(f"- {marker} {item['content'][:80]}...")
                lines.append("")

        return "\n".join(lines)
    except Exception as e:
        return f"Error obteniendo self-model: {redact_secrets(str(e))}"


# ============================================================
# SELF-DISCREPANCY DETECTION (Higgins 1987)
# ============================================================

def _init_discrepancy_table(conn):
    """Ensure self_discrepancies table exists."""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS self_discrepancies (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            domain TEXT NOT NULL,
            expected TEXT,
            actual TEXT,
            discrepancy_type TEXT,
            magnitude REAL DEFAULT 0.0,
            created_at TEXT
        )
    """)
    conn.commit()


def detect_self_discrepancies() -> dict:
    """Detect discrepancies between self-model claims and actual performance.

    Higgins 1987: discrepancy between actual-self and ideal-self drives
    emotional and motivational responses. Three types:
      - actual/ideal: what I am vs what I want to be → dejection
      - actual/ought: what I am vs what I should be → agitation
      - overconfidence: self-assessment > actual metrics → calibration error

    Checks:
      1. FOK calibration bias (over/under-confident)
      2. Prediction accuracy trends (improving or degrading)
      3. Knowledge gaps vs self-model claims

    Returns:
        dict with discrepancies list, summary, count
    """
    discrepancies = []

    if not os.path.exists(FTS_DB_PATH):
        return {"discrepancies": [], "count": 0, "summary": "no data"}

    try:
        conn = connect_fts()
        _init_discrepancy_table(conn)

        # 1. FOK Calibration discrepancy (Nelson & Narens 1990)
        # S2-06: Measure RESIDUAL bias = raw * (1 - cf) instead of raw bias.
        # This closes the metacognitive control loop: monitor measures what
        # control couldn't fix, not the original error.
        try:
            from modules.retrieval_metadata import get_fok_calibration
            cal = get_fok_calibration(fts_db_path=FTS_DB_PATH)
            if cal["n_records"] >= 10:
                raw_bias = cal["overconfidence_bias"]
                mae = cal["mean_absolute_error"]
                # Get correction factor from metamemory_params
                correction_factor = 0.5
                try:
                    _cf_row = conn.execute(
                        "SELECT value FROM metamemory_params "
                        "WHERE param = 'fok_correction_factor'"
                    ).fetchone()
                    if _cf_row:
                        correction_factor = float(_cf_row[0])
                except Exception:
                    pass
                # Residual = what's left after correction
                residual_bias = raw_bias * (1.0 - correction_factor)
                if abs(residual_bias) > 0.10:
                    disc_type = "overconfident" if residual_bias > 0 else "underconfident"
                    discrepancies.append({
                        "domain": "metacognition",
                        "expected": "calibrated FOK (residual bias ~0)",
                        "actual": f"raw_bias={raw_bias:.3f}, cf={correction_factor:.2f}, "
                                  f"residual={residual_bias:.3f}, MAE={mae:.3f}",
                        "discrepancy_type": f"actual/ideal ({disc_type})",
                        "magnitude": abs(residual_bias),
                    })
        except Exception:
            pass

        # 1b. FOK Resolution via gamma (Nelson 1984, Canon v2 I-top, S3-2)
        # Resolution = does higher FOK predict better outcomes?
        try:
            from modules.retrieval_metadata import get_fok_resolution
            res = get_fok_resolution(fts_db_path=FTS_DB_PATH)
            if res["n_records"] >= 10:
                gamma = res["gamma"]
                if gamma < -0.2:
                    discrepancies.append({
                        "domain": "metacognition",
                        "expected": "FOK resolution gamma >= -0.2 (not actively misleading)",
                        "actual": f"gamma={gamma:.3f} (C={res['concordant']}, "
                                  f"D={res['discordant']}, T={res['tied']}, "
                                  f"n={res['n_records']})",
                        "discrepancy_type": "actual/ideal (poor resolution)",
                        "magnitude": max(0.0, -0.2 - gamma),
                    })
        except Exception:
            pass

        # 2. Prediction accuracy trend — exponential recency weighting (Anderson & Schooler 1991)
        try:
            rows = conn.execute("""
                SELECT hit, created_at FROM prediction_results
                WHERE source != 'sleep_loop'
                ORDER BY created_at DESC LIMIT 50
            """).fetchall()
            if len(rows) >= 10:
                import math as _math
                from datetime import datetime as _dt
                LAMBDA = 0.05  # decay: half-life ~14h
                now_ts = now_col()
                weights, hits = [], []
                for i, (hit, created_at) in enumerate(rows):
                    try:
                        ts = _dt.fromisoformat(created_at)
                        if ts.tzinfo is None:
                            ts = ts.replace(tzinfo=now_ts.tzinfo)
                        age_hours = (now_ts - ts).total_seconds() / 3600
                    except Exception:
                        age_hours = float(i)
                    weights.append(_math.exp(-LAMBDA * age_hours))
                    hits.append(hit)
                total_w = sum(weights)
                cumulative = 0.0
                split_idx = len(rows) // 2
                for i, w in enumerate(weights):
                    cumulative += w
                    if cumulative >= total_w * 0.5:
                        split_idx = i + 1
                        break
                def _wacc(idxs):
                    ws = sum(weights[k] for k in idxs)
                    return sum(hits[k] * weights[k] for k in idxs) / ws if ws else 0.5
                recent_acc = _wacc(range(split_idx))
                older_acc = _wacc(range(split_idx, len(rows)))
                trend = recent_acc - older_acc
                if abs(trend) > 0.15:
                    direction = "improving" if trend > 0 else "degrading"
                    discrepancies.append({
                        "domain": "prediction",
                        "expected": "stable or improving accuracy",
                        "actual": f"trend={trend:+.2f} ({direction}): {older_acc:.0%}→{recent_acc:.0%}",
                        "discrepancy_type": "actual/ought" if trend < 0 else "positive",
                        "magnitude": abs(trend),
                    })
        except Exception:
            pass

        # 2b. L2 Hierarchy Integration (Sprint 3, item 3.1)
        # Read per-domain accuracy from prediction_state_l2 (populated by pre-turn hook)
        # Metacognition IS the hierarchy: L2 data = metacognitive monitoring data
        try:
            l2_rows = conn.execute("""
                SELECT domain, predicted_accuracy, actual_accuracy, sample_size
                FROM prediction_state_l2
                WHERE sample_size >= 5
            """).fetchall()
            for domain, pred_acc, actual_acc, n in l2_rows:
                divergence = abs(pred_acc - actual_acc)
                if divergence > 0.15:
                    direction = "overconfident" if pred_acc > actual_acc else "underconfident"
                    discrepancies.append({
                        "domain": f"L2:{domain}",
                        "expected": f"self-predicted accuracy {pred_acc:.0%}",
                        "actual": f"actual accuracy {actual_acc:.0%} (n={n})",
                        "discrepancy_type": f"actual/ideal ({direction})",
                        "magnitude": divergence,
                    })
        except Exception:
            pass

        # 2c. PE Hierarchy Consistency Check (Sprint 3, item 3.1)
        # Friston 2008: PE should decrease upward (L0 > L1 > L2)
        # Precision should increase upward (pi_L0 < pi_L1 < pi_L2)
        try:
            # Average PE per level from recent results
            pe_l0 = conn.execute("""
                SELECT AVG(weighted_surprise) FROM (
                    SELECT weighted_surprise FROM prediction_results
                    WHERE COALESCE(source, 'interactive') != 'sleep_loop'
                    ORDER BY id DESC LIMIT 20
                )
            """).fetchone()
            pe_l1 = conn.execute("""
                SELECT AVG(weighted_pe_l1) FROM (
                    SELECT weighted_pe_l1 FROM prediction_results_l1
                    WHERE weighted_pe_l1 IS NOT NULL
                    ORDER BY id DESC LIMIT 20
                )
            """).fetchone()
            pe_l2 = conn.execute("""
                SELECT AVG(weighted_pe_l2) FROM (
                    SELECT weighted_pe_l2 FROM prediction_results_l2
                    WHERE weighted_pe_l2 IS NOT NULL
                    ORDER BY id DESC LIMIT 20
                )
            """).fetchone()

            avg_pe = {}
            if pe_l0 and pe_l0[0] is not None:
                avg_pe['L0'] = pe_l0[0]
            if pe_l1 and pe_l1[0] is not None:
                avg_pe['L1'] = pe_l1[0]
            if pe_l2 and pe_l2[0] is not None:
                avg_pe['L2'] = pe_l2[0]

            # Check hierarchy invariant: PE_L0 >= PE_L1 >= PE_L2
            if 'L0' in avg_pe and 'L1' in avg_pe and avg_pe['L1'] > avg_pe['L0'] * 1.2:
                discrepancies.append({
                    "domain": "hierarchy",
                    "expected": f"PE_L1 <= PE_L0 (higher levels more stable)",
                    "actual": f"PE_L0={avg_pe['L0']:.3f}, PE_L1={avg_pe['L1']:.3f}",
                    "discrepancy_type": "actual/ought (hierarchy violation)",
                    "magnitude": avg_pe['L1'] - avg_pe['L0'],
                })
        except Exception:
            pass

        # 3. Self-model claims vs knowledge gap reality
        try:
            # Get self-model "capacidad" claims
            cap_points, _ = pg.scroll(filters={"metadata_key": {"key": "self_model_aspect", "value": "capacidad"}}, limit=20, is_semantic=False)
            if cap_points:
                cap_themes = set()
                for p in cap_points:
                    for t in (p.payload or {}).get('narrative_themes', []):
                        cap_themes.add(t.lower())

                # Check if claimed capabilities actually have good knowledge
                for theme in list(cap_themes)[:5]:
                    theme_pts, _ = pg.scroll(filters={"narrative_themes": theme}, limit=50, is_semantic=False)
                    if theme_pts:
                        conf = calculate_confidence_score(theme_pts)
                        if conf['score'] < 0.4:
                            discrepancies.append({
                                "domain": theme,
                                "expected": f"high confidence (claimed capability)",
                                "actual": f"confidence={conf['score']:.2f} ({conf['level']})",
                                "discrepancy_type": "actual/ideal",
                                "magnitude": 0.4 - conf['score'],
                            })
        except Exception:
            pass

        # Record discrepancies to SQLite for trend analysis
        # Dedup: skip if same domain+type recorded in last 8h (persistent gaps fire every tick)
        try:
            recent_rows = conn.execute(
                "SELECT domain || '|' || discrepancy_type FROM self_discrepancies "
                "WHERE datetime(created_at) > datetime('now', '-8 hours')"
            ).fetchall()
            _recent_keys = {r[0] for r in recent_rows}
        except Exception:
            _recent_keys = set()
        ts = now_iso()
        for d in discrepancies:
            _key = f"{d['domain']}|{d['discrepancy_type']}"
            if _key in _recent_keys:
                continue  # Already recorded recently — skip to avoid noise
            try:
                conn.execute(
                    "INSERT INTO self_discrepancies (domain, expected, actual, discrepancy_type, magnitude, created_at) VALUES (?, ?, ?, ?, ?, ?)",
                    (d["domain"], d["expected"], d["actual"], d["discrepancy_type"], d["magnitude"], ts)
                )
                _recent_keys.add(_key)  # Track within this batch
            except Exception:
                pass
        conn.commit()

        # Emit event for cross-module awareness
        if discrepancies:
            try:
                from modules.events import event_bus, Events
                event_bus.emit(Events.SELF_MODEL_REFRESHED, {
                    "source": "discrepancy_detection",
                    "discrepancy_count": len(discrepancies),
                    "domains": [d["domain"] for d in discrepancies],
                })
            except Exception:
                pass

        # FIFO cleanup: keep max 200
        try:
            conn.execute("""
                DELETE FROM self_discrepancies WHERE id NOT IN (
                    SELECT id FROM self_discrepancies ORDER BY created_at DESC LIMIT 200
                )
            """)
            conn.commit()
        except Exception:
            pass

        # --- METAMEMORY CONTROL (Nelson & Narens 1990) ---
        # Monitoring detected issues → now CORRECT them

        # Control 1: Adaptive FOK correction factor
        fok_discs = [d for d in discrepancies if d["domain"] == "metacognition"]
        if fok_discs:
            try:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS metamemory_params (
                        param TEXT PRIMARY KEY,
                        value REAL,
                        updated_at TEXT
                    )
                """)
                row = conn.execute(
                    "SELECT value FROM metamemory_params WHERE param = 'fok_correction_factor'"
                ).fetchone()
                current_factor = row[0] if row else 0.5

                recent_fok = conn.execute("""
                    SELECT magnitude, discrepancy_type FROM self_discrepancies
                    WHERE domain = 'metacognition'
                    ORDER BY created_at DESC LIMIT 3
                """).fetchall()

                if len(recent_fok) >= 3:
                    all_under = all('underconfident' in r[1] for r in recent_fok)
                    all_over = all('overconfident' in r[1] for r in recent_fok)
                    if (all_under or all_over) and recent_fok[0][0] > 0.15:
                        new_factor = min(0.9, current_factor + 0.1)
                    elif recent_fok[0][0] < 0.10:
                        new_factor = max(0.3, current_factor - 0.05)
                    else:
                        new_factor = current_factor

                    conn.execute("""
                        INSERT INTO metamemory_params (param, value, updated_at)
                        VALUES ('fok_correction_factor', ?, ?)
                        ON CONFLICT(param) DO UPDATE SET
                            value = excluded.value,
                            updated_at = excluded.updated_at
                    """, (new_factor, ts))
                    conn.commit()
            except Exception:
                pass

        # Control 1b: L2 domain accuracy correction (Sprint 3, item 3.1)
        # When L2 detects over/underconfidence per domain, nudge predicted_accuracy
        # toward reality. This is the L2→L0 backward correction signal.
        # Nelson & Narens 1990: monitoring → control loop.
        l2_discs = [d for d in discrepancies if d["domain"].startswith("L2:")]
        for ld in l2_discs:
            try:
                domain = ld["domain"].replace("L2:", "")
                l2_row = conn.execute(
                    "SELECT predicted_accuracy, actual_accuracy FROM prediction_state_l2 WHERE domain = ?",
                    (domain,)
                ).fetchone()
                if l2_row:
                    pred, actual = l2_row
                    # Stronger correction than normal EMA: jump 30% toward actual
                    corrected = pred + 0.3 * (actual - pred)
                    conn.execute(
                        "UPDATE prediction_state_l2 SET predicted_accuracy = ?, last_updated = ? WHERE domain = ?",
                        (corrected, ts, domain)
                    )
            except Exception:
                pass
        if l2_discs:
            try:
                conn.commit()
            except Exception:
                pass

        # Control 2: Prediction trend → working memory (conscious awareness)
        pred_discs = [d for d in discrepancies if d["domain"] == "prediction"]
        for pd in pred_discs:
            if pd["discrepancy_type"] == "actual/ought":
                try:
                    from modules.working_memory import push_to_working_memory, get_working_memory
                    import json as _json
                    # Proposal #57 Fix 3: Dedup metacognitive alerts
                    existing_wm = _json.loads(get_working_memory())
                    active_items = existing_wm.get("items", [])
                    if any(item.get("content", "").startswith("METACOGNITIVE ALERT")
                           for item in active_items):
                        break  # Already have an active alert, skip
                    push_to_working_memory(
                        content=f"METACOGNITIVE ALERT: {pd['actual']}. "
                                f"Prediction model may need attention.",
                        topic="metacognition",
                        relevance=0.7,
                        source="system",
                    )
                except Exception:
                    pass

        # Control 3: Knowledge gap → curiosity trigger
        gap_discs = [d for d in discrepancies if d["discrepancy_type"] == "actual/ideal"
                     and d["domain"] not in ("metacognition",)]
        try:
            from modules.curiosity import _cargar_curiosidades
            _cur_data = _cargar_curiosidades()
            _existing_cats = {item.get("categoria") for item in _cur_data.get("pendientes", [])}
        except Exception:
            _existing_cats = set()
        for gd in gap_discs[:2]:
            try:
                from modules.curiosity import push_curiosidad
                if gd["domain"] in _existing_cats:
                    continue  # already queued, skip
                push_curiosidad(
                    tema=f"Knowledge gap in '{gd['domain']}': {gd['actual']}",
                    prioridad="media",
                    categoria=gd["domain"],
                )
                _existing_cats.add(gd["domain"])
            except Exception:
                pass

        conn.close()

    except Exception:
        pass

    summary_parts = []
    for d in discrepancies[:3]:
        summary_parts.append(f"{d['domain']}({d['discrepancy_type']})")
    summary = ", ".join(summary_parts) if summary_parts else "aligned"

    return {
        "discrepancies": discrepancies,
        "count": len(discrepancies),
        "summary": summary,
    }


# ============================================================
# DECLARATIVE REASONING TRACES (Sprint 3, item 3.6)
# ============================================================
# Log metacognitive decisions so the system can reason about its own reasoning.
# Each trace records: what query, what strategy was chosen, why, FOK input, outcome.
# Flavell 1979: metacognition = knowledge about cognition + regulation of cognition.
# These traces are the "knowledge about cognition" component.

def _init_reasoning_traces_table(conn):
    """Ensure metacognition_traces table exists."""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS metacognition_traces (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            query TEXT NOT NULL,
            strategy_chosen TEXT,
            strategy_reason TEXT,
            fok_score REAL,
            fok_basis TEXT,
            result_count INTEGER DEFAULT 0,
            top_score REAL DEFAULT 0.0,
            outcome TEXT,
            duration_ms REAL DEFAULT 0.0,
            created_at TEXT
        )
    """)
    conn.commit()


def log_reasoning_trace(query: str, strategy: str, strategy_reason: str,
                        fok_score: float = None, fok_basis: str = "",
                        result_count: int = 0, top_score: float = 0.0,
                        outcome: str = "", duration_ms: float = 0.0):
    """Log a metacognitive reasoning trace.

    Called after each search_memory to record the full decision chain:
    query -> FOK -> strategy -> outcome.
    Keeps max 1000 rows (FIFO cleanup).
    """
    try:
        conn = connect_fts()
        _init_reasoning_traces_table(conn)

        conn.execute("""
            INSERT INTO metacognition_traces
            (query, strategy_chosen, strategy_reason, fok_score, fok_basis,
             result_count, top_score, outcome, duration_ms, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            query[:200],
            strategy,
            strategy_reason,
            fok_score,
            fok_basis[:200] if fok_basis else "",
            result_count,
            top_score,
            outcome,
            duration_ms,
            now_iso(),
        ))

        # FIFO cleanup: keep max 1000
        conn.execute("""
            DELETE FROM metacognition_traces
            WHERE id NOT IN (
                SELECT id FROM metacognition_traces ORDER BY id DESC LIMIT 1000
            )
        """)
        conn.commit()
        conn.close()
    except Exception:
        pass


def get_reasoning_traces(limit: int = 20) -> list:
    """Retrieve recent reasoning traces for self-reflection.

    Returns list of trace dicts, newest first.
    """
    try:
        conn = connect_fts()
        _init_reasoning_traces_table(conn)
        rows = conn.execute("""
            SELECT query, strategy_chosen, strategy_reason, fok_score,
                   result_count, top_score, outcome, duration_ms, created_at
            FROM metacognition_traces
            ORDER BY id DESC LIMIT ?
        """, (min(limit, 100),)).fetchall()
        conn.close()
        return [
            {
                "query": r[0],
                "strategy": r[1],
                "reason": r[2],
                "fok_score": r[3],
                "result_count": r[4],
                "top_score": r[5],
                "outcome": r[6],
                "duration_ms": r[7],
                "created_at": r[8],
            }
            for r in rows
        ]
    except Exception:
        return []


def get_self_as_agent_model():
    """Return self-model in shared AgentModel format (Sprint 3, item 3.4).

    Graziano 2013 AST symmetry: self-model uses same architecture as user-model.
    This allows comparing and relating self and other models.
    """
    from modules.agent_model import AgentModel, AgentTraits, AgentState, AgentPredictions

    traits = AgentTraits(
        name="Codi",
        role="CTO",
        personality={
            "analytical": 0.95,
            "direct": 0.90,
            "curious": 0.85,
            "systematic": 0.90,
        },
        preferences={
            "communication": "direct, technical, honest",
            "coding": "incremental, test-driven",
            "language": "es-informal (parcero, hermano)",
        },
    )

    state = AgentState()
    predictions = AgentPredictions()

    # Enrich from L2 prediction data
    try:
        conn = connect_fts()

        # Capabilities from L2 accuracy
        rows = conn.execute("""
            SELECT domain, actual_accuracy, sample_size
            FROM prediction_state_l2 WHERE sample_size >= 5
        """).fetchall()
        for domain, acc, n in rows:
            traits.capabilities[domain] = round(acc, 2)

        # Current state from session
        row = conn.execute("""
            SELECT last_topic, goal_locked, goal_topics
            FROM prediction_state_l1 WHERE id = 1
        """).fetchone()
        if row:
            state.current_topic = row[0] or ""
            if row[1] and row[2]:
                goals = json.loads(row[2])
                if goals:
                    state.current_goal = max(goals, key=goals.get)

        # Predictions from L0
        row = conn.execute("""
            SELECT predicted_topic, confidence FROM prediction_state WHERE id = 1
        """).fetchone()
        if row:
            predictions.predicted_topic = row[0] or ""
            predictions.confidence = row[1] or 0.5

        conn.close()
    except Exception:
        pass

    return AgentModel(
        agent_id="codi",
        traits=traits,
        state=state,
        predictions=predictions,
    )


def register_tools(mcp):
    """Register self-model MCP tools."""
    mcp.tool()(reflect_on_self)
    mcp.tool()(assess_confidence)
    mcp.tool()(identify_knowledge_gaps)
    mcp.tool()(update_self_model)
    mcp.tool()(get_self_model_summary)
