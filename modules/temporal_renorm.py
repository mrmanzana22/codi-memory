"""
Temporal Renormalization — Multi-level Memory Consolidation.
Sprint 12 — Canon v4 refs: PN-21, XC-28, M-INV-06.

Implements RGM block-spin transformation for temporal hierarchies:
  Episode → Event → Narrative → Theme

Each level is a coarse-graining of the level below:
  - Events: cluster related episodes within temporal window + topic coherence
  - Narratives: sequence of events with causal links (NOTEARS edges)
  - Themes: recurring abstract patterns across narratives

References:
  - Friston 2025, "Renormalization Group for Generative Models" (RGM)
  - Wilson 1971, "Renormalization Group and Critical Phenomena"
  - Diekelmann & Born 2010, "Sleep consolidation" (multi-stage)
"""

import json
import uuid
import logging
import sqlite3
from datetime import datetime, timedelta
from collections import defaultdict
from modules.config import now_iso

_logger = logging.getLogger(__name__)

# Config
EVENT_TIME_WINDOW_HOURS = 4     # Max time span for episodes in same event
EVENT_MIN_EPISODES = 2          # Min episodes to form an event
NARRATIVE_MIN_EVENTS = 2        # Min events to form a narrative
THEME_MIN_NARRATIVES = 2        # Min narratives for theme extraction


def _get_fts_db():
    """Get path to FTS DB (instance-aware)."""
    from modules.config import FTS_DB_PATH
    return FTS_DB_PATH


def ensure_renorm_tables(conn: sqlite3.Connection):
    """Create temporal renormalization tables if they don't exist."""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS renorm_events (
            event_id TEXT PRIMARY KEY,
            topic TEXT,
            time_start TEXT,
            time_end TEXT,
            episode_ids TEXT,
            summary TEXT,
            episode_count INTEGER DEFAULT 0,
            created_at TEXT
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS renorm_narratives (
            narrative_id TEXT PRIMARY KEY,
            theme TEXT,
            event_ids TEXT,
            causal_chain TEXT,
            time_start TEXT,
            time_end TEXT,
            event_count INTEGER DEFAULT 0,
            created_at TEXT
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS renorm_themes (
            theme_id TEXT PRIMARY KEY,
            pattern TEXT,
            narrative_ids TEXT,
            frequency INTEGER DEFAULT 0,
            first_seen TEXT,
            last_seen TEXT,
            created_at TEXT
        )
    """)
    conn.commit()


def extract_events(clusters: list, conn: sqlite3.Connection = None) -> list:
    """Phase E1: Extract events from episode clusters.

    An event groups related episodes within a temporal window + topic coherence.
    This is the first renormalization: episodes → events.

    Args:
        clusters: List of cluster dicts from _phase_clustering
                  Each has: topic, episode_ids, centroid, etc.
        conn: FTS DB connection (for persisting events)

    Returns:
        List of event dicts: [{event_id, topic, episode_ids, time_range, summary}]
    """
    if not clusters:
        return []

    events = []
    now = now_iso()

    for cluster in clusters:
        episode_ids = cluster.get("episode_ids", [])
        topic = cluster.get("topic", "general")

        if len(episode_ids) < EVENT_MIN_EPISODES:
            continue

        # Sprint 6 FIX-14: use actual episode timestamps, not now()
        timestamps = cluster.get("timestamps", [])
        time_start = min(timestamps) if timestamps else now
        time_end = max(timestamps) if timestamps else now

        event_id = f"evt_{uuid.uuid4().hex[:8]}"
        event = {
            "event_id": event_id,
            "topic": topic,
            "episode_ids": episode_ids,
            "episode_count": len(episode_ids),
            "time_start": time_start,
            "time_end": time_end,
            "summary": f"Event: {len(episode_ids)} episodes about {topic} ({str(time_start)[:10]}—{str(time_end)[:10]})",
        }
        events.append(event)

        # Persist
        if conn:
            try:
                conn.execute("""
                    INSERT OR REPLACE INTO renorm_events
                    (event_id, topic, time_start, time_end, episode_ids,
                     summary, episode_count, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    event_id, topic, time_start, time_end,
                    json.dumps(episode_ids),
                    event["summary"],
                    len(episode_ids),
                    now,
                ))
            except Exception as e:
                _logger.warning("Failed to persist event: %s", e)

    if conn:
        conn.commit()

    _logger.info("Extracted %d events from %d clusters", len(events), len(clusters))
    return events


def extract_narratives(events: list, conn: sqlite3.Connection = None) -> list:
    """Phase E2: Extract narratives from events.

    A narrative is a sequence of events with causal or temporal links.
    Groups events by topic and links them chronologically.
    Uses NOTEARS causal edges if available.

    Args:
        events: List of event dicts from extract_events
        conn: FTS DB connection

    Returns:
        List of narrative dicts
    """
    if len(events) < NARRATIVE_MIN_EVENTS:
        return []

    # Group events by topic
    by_topic = defaultdict(list)
    for event in events:
        by_topic[event["topic"]].append(event)

    narratives = []
    now = now_iso()

    for topic, topic_events in by_topic.items():
        if len(topic_events) < NARRATIVE_MIN_EVENTS:
            continue

        # Sort by time
        topic_events.sort(key=lambda e: e.get("time_start", ""))

        narrative_id = f"nar_{uuid.uuid4().hex[:8]}"
        event_ids = [e["event_id"] for e in topic_events]

        # Look for causal links between events (via spreading_edges / NOTEARS)
        causal_chain = []
        if conn:
            try:
                for i, evt in enumerate(topic_events[:-1]):
                    next_evt = topic_events[i + 1]
                    # Check if any episodes have causal edges
                    for ep_a in evt["episode_ids"][:3]:
                        for ep_b in next_evt["episode_ids"][:3]:
                            # Sprint 6 FIX-14: spreading_edges uses from_id/to_id, not source_id/target_id
                            edge = conn.execute("""
                                SELECT edge_type FROM spreading_edges
                                WHERE from_id = ? AND to_id = ?
                            """, (str(ep_a), str(ep_b))).fetchone()
                            if edge:
                                causal_chain.append({
                                    "from": evt["event_id"],
                                    "to": next_evt["event_id"],
                                    "type": edge[0],
                                })
                                break
                        if causal_chain and causal_chain[-1]["from"] == evt["event_id"]:
                            break
            except Exception:
                pass  # spreading_edges table might not exist

        narrative = {
            "narrative_id": narrative_id,
            "theme": topic,
            "event_ids": event_ids,
            "event_count": len(event_ids),
            "causal_chain": causal_chain,
            "time_start": topic_events[0].get("time_start", now),
            "time_end": topic_events[-1].get("time_end", now),
        }
        narratives.append(narrative)

        # Persist
        if conn:
            try:
                conn.execute("""
                    INSERT OR REPLACE INTO renorm_narratives
                    (narrative_id, theme, event_ids, causal_chain,
                     time_start, time_end, event_count, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    narrative_id, topic,
                    json.dumps(event_ids),
                    json.dumps(causal_chain),
                    narrative["time_start"],
                    narrative["time_end"],
                    len(event_ids),
                    now,
                ))
            except Exception as e:
                _logger.warning("Failed to persist narrative: %s", e)

    if conn:
        conn.commit()

    _logger.info("Extracted %d narratives from %d events", len(narratives), len(events))
    return narratives


def extract_themes(narratives: list, conn: sqlite3.Connection = None) -> list:
    """Phase E3: Extract themes from narratives.

    A theme is a recurring abstract pattern across multiple narratives.
    Maps to RGM's highest renormalization level.

    Args:
        narratives: List of narrative dicts
        conn: FTS DB connection

    Returns:
        List of theme dicts
    """
    # Cross-run accumulation: each consolidation produces ~1 narrative,
    # but themes require >= THEME_MIN_NARRATIVES. Load persisted narratives
    # so themes can emerge across runs.
    all_narratives = list(narratives)
    if conn:
        try:
            current_ids = {n.get("narrative_id") for n in narratives}
            db_rows = conn.execute(
                "SELECT narrative_id, theme FROM renorm_narratives"
            ).fetchall()
            for row in db_rows:
                if row[0] not in current_ids:
                    all_narratives.append({"narrative_id": row[0], "theme": row[1]})
        except Exception:
            pass  # table may not exist yet

    if len(all_narratives) < THEME_MIN_NARRATIVES:
        return []

    # Sprint 6 FIX-14: merge similar topics into abstract themes
    # Instead of using raw topic names, cluster related topics together.
    # E.g., "consciencia" + "identidad" → theme "self_and_consciousness"
    _TOPIC_CLUSTERS = {
        "self_and_consciousness": {"consciencia", "identidad", "self", "consciousness"},
        "technical_work": {"codigo", "desarrollo", "trading", "kraken", "n8n"},
        "learning_and_growth": {"aprendizaje", "memoria", "learning"},
        "relationships": {"relaciones", "fullempaques", "personal"},
    }
    def _theme_for_topic(topic):
        t = topic.lower() if topic else "general"
        for theme, topics in _TOPIC_CLUSTERS.items():
            if t in topics or any(kw in t for kw in topics):
                return theme
        return t  # fallback to raw topic

    by_theme = defaultdict(list)
    for narrative in all_narratives:
        abstract_theme = _theme_for_topic(narrative["theme"])
        by_theme[abstract_theme].append(narrative)

    themes = []
    now = now_iso()

    for theme_name, theme_narratives in by_theme.items():
        if len(theme_narratives) < THEME_MIN_NARRATIVES:
            continue

        theme_id = f"thm_{uuid.uuid4().hex[:8]}"
        narrative_ids = [n["narrative_id"] for n in theme_narratives]

        theme = {
            "theme_id": theme_id,
            "pattern": theme_name,
            "narrative_ids": narrative_ids,
            "frequency": len(narrative_ids),
            "first_seen": theme_narratives[0].get("time_start", now),
            "last_seen": theme_narratives[-1].get("time_end", now),
        }
        themes.append(theme)

        # Persist or update existing
        if conn:
            try:
                existing = conn.execute(
                    "SELECT theme_id, frequency FROM renorm_themes WHERE pattern = ?",
                    (theme_name,)
                ).fetchone()

                if existing:
                    conn.execute("""
                        UPDATE renorm_themes SET
                            frequency = ?,
                            narrative_ids = ?,
                            last_seen = ?
                        WHERE theme_id = ?
                    """, (len(narrative_ids), json.dumps(narrative_ids), now, existing[0]))
                else:
                    conn.execute("""
                        INSERT INTO renorm_themes
                        (theme_id, pattern, narrative_ids, frequency,
                         first_seen, last_seen, created_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                    """, (
                        theme_id, theme_name,
                        json.dumps(narrative_ids),
                        len(narrative_ids),
                        theme["first_seen"],
                        now, now,
                    ))
            except Exception as e:
                _logger.warning("Failed to persist theme: %s", e)

    if conn:
        conn.commit()

    _logger.info("Extracted %d themes from %d narratives", len(themes), len(narratives))
    return themes


def run_temporal_renormalization(clusters: list) -> dict:
    """Execute full temporal renormalization pipeline.

    Episode clusters → Events → Narratives → Themes.

    Called from consolidation.py after clustering phase.

    Returns:
        Dict with counts: {events, narratives, themes}
    """
    import os
    from modules.config import connect_fts

    fts_path = _get_fts_db()
    if not os.path.exists(fts_path):
        return {"events": 0, "narratives": 0, "themes": 0}

    try:
        conn = connect_fts(fts_path)
        try:
            ensure_renorm_tables(conn)

            # E1: Episodes → Events
            events = extract_events(clusters, conn)

            # E2: Events → Narratives
            narratives = extract_narratives(events, conn)

            # E3: Narratives → Themes
            themes = extract_themes(narratives, conn)
        finally:
            conn.close()

        return {
            "events": len(events),
            "narratives": len(narratives),
            "themes": len(themes),
        }
    except Exception as e:
        _logger.error("Temporal renormalization failed: %s", e)
        return {"events": 0, "narratives": 0, "themes": 0}


def get_hierarchy_stats(conn: sqlite3.Connection = None) -> dict:
    """Get counts at each level of the temporal hierarchy."""
    import os
    from modules.config import connect_fts

    close_conn = False
    if conn is None:
        fts_path = _get_fts_db()
        if not os.path.exists(fts_path):
            return {"events": 0, "narratives": 0, "themes": 0}
        conn = connect_fts(fts_path)
        close_conn = True

    try:
        ensure_renorm_tables(conn)
        events = conn.execute("SELECT COUNT(*) FROM renorm_events").fetchone()[0]
        narratives = conn.execute("SELECT COUNT(*) FROM renorm_narratives").fetchone()[0]
        themes = conn.execute("SELECT COUNT(*) FROM renorm_themes").fetchone()[0]
        return {"events": events, "narratives": narratives, "themes": themes}
    except Exception:
        return {"events": 0, "narratives": 0, "themes": 0}
    finally:
        if close_conn:
            conn.close()
