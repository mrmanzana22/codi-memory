"""
Codi Memory - Self-model module.
Reflect, assess confidence, knowledge gaps, Butlin indicators, self-model CRUD.
Bidirectional: monitor + control (Nelson & Narens 1990).
"""

import inspect

from qdrant_client.models import Filter, FieldCondition, MatchValue, Range

from modules.config import (
    memory, qdrant, USER_ID, COLLECTION_NAME,
    now_iso, now_short,
)
from modules.secret_redact import redact_secrets
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
    "register_tools",
]


def reflect_on_self() -> str:
    """
    Reflexiona sobre mi identidad analizando mis memorias auto-referenciales.
    Genera un modelo de quien soy basado en evidencia de mis experiencias.
    """
    try:
        self_ref_points, _ = qdrant.scroll(
            collection_name=COLLECTION_NAME,
            scroll_filter=Filter(must=[
                FieldCondition(key='self_reference', match=MatchValue(value=True))
            ]),
            limit=20, with_payload=True
        )
        identity_points, _ = qdrant.scroll(
            collection_name=COLLECTION_NAME,
            scroll_filter=Filter(must=[
                FieldCondition(key='narrative_themes', match=MatchValue(value='identidad'))
            ]),
            limit=20, with_payload=True
        )
        critical_points, _ = qdrant.scroll(
            collection_name=COLLECTION_NAME,
            scroll_filter=Filter(must=[
                FieldCondition(key='narrative_importance', match=MatchValue(value='critical'))
            ]),
            limit=10, with_payload=True
        )

        all_points = {}
        for p in (self_ref_points or []) + (identity_points or []) + (critical_points or []):
            all_points[p.id] = p

        if not all_points:
            return "No encontre memorias sobre mi identidad. Necesito mas experiencias para reflexionar."

        reflection = ["# REFLEXION SOBRE MI MISMO\n"]
        capacidades = []
        limitaciones = []
        valores = []
        experiencias = []

        for p in all_points.values():
            data = p.payload.get('data', '').lower()
            content = p.payload.get('data', '')
            source = p.payload.get('ownership_source', 'unknown')

            if 'puedo' in data or 'capacidad' in data or 'habilidad' in data:
                capacidades.append(f"[{source}] {content[:80]}")
            elif 'no puedo' in data or 'limita' in data or 'cuesta' in data or 'dificulta' in data:
                limitaciones.append(f"[{source}] {content[:80]}")
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
        search_results = memory.search(query=topic, user_id=USER_ID, limit=15)
        if not search_results or not search_results.get("results"):
            return f"No tengo memorias sobre '{topic}'. Mi confianza es 0 - no se nada al respecto."

        memory_ids = [m.get('id') for m in search_results['results'] if m.get('id')]
        points = qdrant.retrieve(collection_name=COLLECTION_NAME, ids=memory_ids, with_payload=True)

        if not points:
            return f"Tengo referencias a '{topic}' pero sin metadata de ownership."

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
                fts_conn = sqlite3.connect(FTS_DB_PATH)
                from modules.retrieval_metadata import get_top_failed_topics
                for topic, _count in get_top_failed_topics(fts_conn, limit=5):
                    if topic:
                        dynamic_themes.add(topic.lower())
                fts_conn.close()
        except Exception:
            pass

        # Source 2: Low-confidence topics from retrieval buffer
        try:
            from modules.retrieval_metadata import get_retrieval_buffer
            for r in get_retrieval_buffer():
                if r.coverage in ("sparse", "empty"):
                    words = [w for w in r.query.lower().split() if len(w) > 3]
                    if words:
                        dynamic_themes.add(words[0])
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
                points, _ = qdrant.scroll(
                    collection_name=COLLECTION_NAME,
                    scroll_filter=Filter(must=[
                        FieldCondition(key='narrative_themes', match=MatchValue(value=theme))
                    ]),
                    limit=100, with_payload=True
                )
                if points:
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

        result = memory.add(
            messages=[{"role": "user", "content": content}],
            user_id=USER_ID,
            metadata={"category": "identidad", "self_model_aspect": aspect, "timestamp": timestamp}
        )

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
                    qdrant.set_payload(
                        collection_name=COLLECTION_NAME,
                        payload=ownership_metadata,
                        points=[mem_id]
                    )

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
        points, _ = qdrant.scroll(
            collection_name=COLLECTION_NAME,
            scroll_filter=Filter(must=[
                FieldCondition(key='self_reference', match=MatchValue(value=True))
            ]),
            limit=50, with_payload=True
        )

        if not points:
            return "No tengo un self-model definido aun. Usa update_self_model() para agregar observaciones."

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


def register_tools(mcp):
    """Register self-model MCP tools."""
    mcp.tool()(reflect_on_self)
    mcp.tool()(assess_confidence)
    mcp.tool()(identify_knowledge_gaps)
    mcp.tool()(update_self_model)
    mcp.tool()(get_self_model_summary)
