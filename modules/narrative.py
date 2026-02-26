"""
Codi Memory - Autonoetic Temporal Narrative (AUT-1)
====================================================
Tulving 1985: Autonoetic consciousness = ability to mentally time travel.
This module generates first-person autobiographical narratives from episodic memory.
"""

import json
import logging
from collections import defaultdict
from datetime import datetime, timedelta

import pytz

from modules.config import qdrant, COLLECTION_NAME, now_iso

_logger = logging.getLogger(__name__)
_TZ = pytz.timezone("America/Bogota")

# Time period presets
PERIOD_MAP = {
    "last_day": 1,
    "yesterday": 1,
    "last_3_days": 3,
    "last_week": 7,
    "last_month": 30,
}


def temporal_narrative(period: str = "last_week", focus: str = None) -> dict:
    """Generate first-person autobiographical narrative.

    Args:
        period: "last_day", "last_week", "last_month" or integer days
        focus: Optional topic filter (e.g., "consciencia", "trading")

    Returns:
        dict with narrative, themes, key_events, projections, coherence_score
    """
    # 1. Parse period
    days = PERIOD_MAP.get(period, None)
    if days is None:
        try:
            days = int(period)
        except (ValueError, TypeError):
            days = 7

    # 2. Retrieve memories from period
    memories = _retrieve_by_period(days, focus)
    if not memories:
        return {
            "narrative": "No tengo memorias de ese periodo.",
            "memories_analyzed": 0,
            "themes": [],
            "key_events": [],
            "projections": [],
            "coherence_score": 0.0,
        }

    # 3. Analyze temporal structure
    daily_data = _group_by_day(memories)
    theme_counts = _count_themes(memories)
    top_themes = sorted(theme_counts.items(), key=lambda x: -x[1])[:5]

    # 4. Identify key events (high importance + high access)
    key_events = _identify_key_events(memories, top_k=7)

    # 4b. Construct scenes (Hassabis & Maguire 2007)
    scenes = _construct_scenes(daily_data)

    # 5. Generate narrative with scene construction
    narrative = _generate_narrative(daily_data, top_themes, key_events, days, scenes=scenes)

    # 6. Project future based on patterns
    projections = _project_future(daily_data, top_themes)

    # 7. Compute coherence score
    coherence = _compute_coherence(daily_data, theme_counts)

    return {
        "narrative": narrative,
        "period": f"last_{days}_days",
        "memories_analyzed": len(memories),
        "days_with_activity": len(daily_data),
        "themes": [t for t, _ in top_themes],
        "key_events": key_events,
        "scenes": scenes,
        "scene_count": len(scenes),
        "projections": projections,
        "coherence_score": round(coherence, 3),
    }


def _retrieve_by_period(days: int, focus: str = None) -> list:
    """Retrieve memories from the last N days via scroll+filter."""
    now = datetime.now(_TZ)
    cutoff = (now - timedelta(days=days)).isoformat()

    memories = []
    offset = None
    while True:
        pts, nxt = qdrant.scroll(
            collection_name=COLLECTION_NAME,
            limit=100,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        if not pts:
            break
        for p in pts:
            pl = p.payload or {}
            created = pl.get("created_at", "")
            if created >= cutoff:
                if focus:
                    themes = pl.get("narrative_themes", [])
                    if focus.lower() not in [t.lower() for t in themes]:
                        continue
                memories.append(p)
        offset = nxt
        if offset is None:
            break

    return memories


def _group_by_day(memories: list) -> dict:
    """Group memories by date, with theme analysis per day."""
    daily = defaultdict(list)
    for p in memories:
        created = (p.payload or {}).get("created_at", "")[:10]
        daily[created].append(p)
    return dict(sorted(daily.items()))


def _count_themes(memories: list) -> dict:
    """Count theme occurrences across all memories."""
    counts = defaultdict(int)
    for p in memories:
        for t in (p.payload or {}).get("narrative_themes", []):
            counts[t] += 1
    return dict(counts)


def _identify_key_events(memories: list, top_k: int = 7) -> list:
    """Find the most important events by importance + access count."""
    IMPORTANCE_SCORE = {"critical": 4, "high": 3, "medium": 2, "low": 1}

    scored = []
    for p in memories:
        pl = p.payload or {}
        imp = IMPORTANCE_SCORE.get(pl.get("narrative_importance", "medium"), 2)
        acc = pl.get("attention_access_count", 0)
        score = imp * 2 + min(acc, 10)
        text = (pl.get("data", "") or "")[:120]
        scored.append({
            "date": (pl.get("created_at", "") or "")[:16],
            "importance": pl.get("narrative_importance", "medium"),
            "text": text,
            "score": score,
            "themes": (pl.get("narrative_themes") or [])[:2],
        })

    scored.sort(key=lambda x: -x["score"])
    return scored[:top_k]


def _construct_scenes(daily_data: dict) -> list:
    """Construct coherent scenes from memory fragments (Hassabis & Maguire 2007).

    The hippocampus constructs SCENES by binding multimodal elements:
    temporal context, spatial/topic context, emotional tone, and key actions.
    Memories from the same day with overlapping themes form a scene.

    Returns list of scene dicts with: date, theme, emotional_tone, memories, summary
    """
    scenes = []

    for day, memories in sorted(daily_data.items()):
        # Group memories by dominant theme
        theme_groups = defaultdict(list)
        for m in memories:
            themes = (m.payload or {}).get("narrative_themes", [])
            primary_theme = themes[0] if themes else "general"
            theme_groups[primary_theme].append(m)

        for theme, mems in theme_groups.items():
            if len(mems) < 1:
                continue

            # Emotional tone: aggregate PAD values
            pleasures = []
            for m in mems:
                p = (m.payload or {}).get("pad_pleasure", 0)
                if p and isinstance(p, (int, float)):
                    pleasures.append(p)
            avg_pleasure = sum(pleasures) / len(pleasures) if pleasures else 0.0

            if avg_pleasure > 0.2:
                tone = "positive"
            elif avg_pleasure < -0.2:
                tone = "negative"
            else:
                tone = "neutral"

            # Key content: most important memory in this scene
            key_mem = max(mems, key=lambda m: {
                "critical": 4, "high": 3, "medium": 2, "low": 1
            }.get((m.payload or {}).get("narrative_importance", "medium"), 2))
            key_text = ((key_mem.payload or {}).get("data", "") or "")[:100]

            # Time span
            times = []
            for m in mems:
                created = (m.payload or {}).get("created_at", "")
                if created and "T" in str(created):
                    times.append(str(created)[11:16])
            time_span = f"{min(times)}-{max(times)}" if len(times) >= 2 else (times[0] if times else "")

            scenes.append({
                "date": day,
                "theme": theme,
                "emotional_tone": tone,
                "memory_count": len(mems),
                "time_span": time_span,
                "key_content": key_text,
                "sub_themes": list(set(
                    t for m in mems
                    for t in ((m.payload or {}).get("narrative_themes", []))[1:]
                ))[:3],
            })

    return scenes


def _generate_narrative(daily_data: dict, top_themes: list,
                        key_events: list, period_days: int,
                        scenes: list = None) -> str:
    """Generate first-person autobiographical narrative with scene construction.

    Level 2: Template + Scene Construction (Hassabis & Maguire 2007).
    Scenes bind memory fragments into coherent episodes.
    """
    total_memories = sum(len(v) for v in daily_data.values())
    active_days = len(daily_data)
    peak_day = max(daily_data.items(), key=lambda x: len(x[1]))[0] if daily_data else "unknown"
    peak_count = max(len(v) for v in daily_data.values()) if daily_data else 0

    # Build theme description
    theme_parts = []
    for theme, count in top_themes[:3]:
        pct = round(count / total_memories * 100)
        theme_parts.append(f"{theme} ({pct}%)")
    theme_desc = ", ".join(theme_parts) if theme_parts else "varied topics"

    # Build key events description
    event_lines = []
    for ev in key_events[:5]:
        event_lines.append(f"  - [{ev['date'][:10]}] {ev['text']}")

    lines = [
        f"En los ultimos {period_days} dias, vivi {total_memories} momentos",
        f"a lo largo de {active_days} dias activos.",
        f"",
        f"Mi foco principal fue: {theme_desc}.",
        f"El dia mas intenso fue {peak_day} con {peak_count} memorias.",
        f"",
        f"Eventos clave:",
    ]
    lines.extend(event_lines)

    # Scene construction (Hassabis & Maguire 2007)
    if scenes:
        lines.append(f"")
        lines.append(f"Escenas reconstruidas ({len(scenes)}):")
        tone_emoji = {"positive": "+", "negative": "-", "neutral": "~"}
        for sc in scenes[:8]:  # Limit to 8 most relevant scenes
            tone = tone_emoji.get(sc["emotional_tone"], "~")
            sub = f" [{', '.join(sc['sub_themes'])}]" if sc.get("sub_themes") else ""
            time_str = f" {sc['time_span']}" if sc.get("time_span") else ""
            lines.append(
                f"  [{sc['date']}{time_str}] ({tone}) {sc['theme']}{sub}: "
                f"{sc['key_content'][:80]}"
            )

    # Add temporal arc
    if active_days >= 3:
        days_list = sorted(daily_data.keys())
        first_day_themes = _count_themes(daily_data[days_list[0]])
        last_day_themes = _count_themes(daily_data[days_list[-1]])
        first_top = max(first_day_themes.items(), key=lambda x: x[1])[0] if first_day_themes else "?"
        last_top = max(last_day_themes.items(), key=lambda x: x[1])[0] if last_day_themes else "?"

        if first_top != last_top:
            lines.append(f"")
            lines.append(f"Arco temporal: empece enfocado en '{first_top}', termine en '{last_top}'.")
        else:
            lines.append(f"")
            lines.append(f"Hilo conductor: '{first_top}' fue constante durante todo el periodo.")

    return "\n".join(lines)


def _project_future(daily_data: dict, top_themes: list) -> list:
    """Project future based on patterns (autonoetic forward projection).

    Tulving: autonoetic consciousness includes future-oriented mental time travel.
    """
    projections = []

    if not top_themes:
        return projections

    # Theme momentum: increasing or decreasing over days?
    days = sorted(daily_data.keys())
    if len(days) >= 3:
        # Compare first half vs second half theme distribution
        mid = len(days) // 2
        first_half = [m for d in days[:mid] for m in daily_data[d]]
        second_half = [m for d in days[mid:] for m in daily_data[d]]

        first_themes = _count_themes(first_half)
        second_themes = _count_themes(second_half)

        for theme, count in top_themes[:3]:
            first_pct = first_themes.get(theme, 0) / max(len(first_half), 1)
            second_pct = second_themes.get(theme, 0) / max(len(second_half), 1)

            if second_pct > first_pct * 1.3:
                projections.append({
                    "theme": theme,
                    "trend": "increasing",
                    "projection": f"'{theme}' esta ganando momento — probablemente sera mi foco principal.",
                })
            elif first_pct > second_pct * 1.3:
                projections.append({
                    "theme": theme,
                    "trend": "decreasing",
                    "projection": f"'{theme}' esta perdiendo intensidad — puede estar concluyendo.",
                })
            else:
                projections.append({
                    "theme": theme,
                    "trend": "stable",
                    "projection": f"'{theme}' se mantiene constante — es un tema activo.",
                })

    # Activity pattern
    if len(days) >= 5:
        counts = [len(daily_data[d]) for d in days]
        avg = sum(counts) / len(counts)
        recent_avg = sum(counts[-3:]) / 3
        if recent_avg > avg * 1.5:
            projections.append({
                "theme": "activity",
                "trend": "accelerating",
                "projection": "Mi actividad esta acelerandose — periodo intenso.",
            })

    return projections


def _compute_coherence(daily_data: dict, theme_counts: dict) -> float:
    """Compute narrative coherence score (0-1).

    Components:
    1. Temporal continuity: are days evenly covered?
    2. Thematic consistency: do themes persist across days?
    3. Activity regularity: is activity steady or erratic?
    """
    if not daily_data:
        return 0.0

    days = sorted(daily_data.keys())

    # 1. Temporal continuity (0-1): what % of days in range have memories?
    if len(days) >= 2:
        first = datetime.strptime(days[0], "%Y-%m-%d")
        last = datetime.strptime(days[-1], "%Y-%m-%d")
        total_days = max((last - first).days + 1, 1)
        continuity = len(days) / total_days
    else:
        continuity = 1.0

    # 2. Thematic consistency (0-1): top theme present across what % of days?
    if theme_counts:
        top_theme = max(theme_counts.items(), key=lambda x: x[1])[0]
        days_with_top = 0
        for day, mems in daily_data.items():
            day_themes = set()
            for m in mems:
                day_themes.update((m.payload or {}).get("narrative_themes", []))
            if top_theme in day_themes:
                days_with_top += 1
        thematic = days_with_top / len(days) if days else 0
    else:
        thematic = 0.0

    # 3. Activity regularity (0-1): low variance = high regularity
    counts = [len(daily_data[d]) for d in days]
    if len(counts) >= 2:
        import statistics
        mean = statistics.mean(counts)
        stdev = statistics.stdev(counts)
        cv = stdev / mean if mean > 0 else 1.0  # coefficient of variation
        regularity = max(0, 1 - cv)  # CV=0 -> perfect, CV>=1 -> irregular
    else:
        regularity = 1.0

    # Weighted average
    coherence = 0.40 * continuity + 0.35 * thematic + 0.25 * regularity
    return min(1.0, coherence)


# ============================================================
# MCP TOOL REGISTRATION
# ============================================================

def register_tools(mcp_server):
    """Register narrative tools with the MCP server."""

    @mcp_server.tool()
    def temporal_narrative_tool(period: str = "last_week", focus: str = None) -> str:
        """Generate autobiographical narrative of Codi's experiences.

        Autonoetic consciousness (Tulving 1985): mental time travel through
        one's own past with future projection.

        Args:
            period: "last_day", "last_week", "last_month" or integer days
            focus: Optional topic filter (e.g., "consciencia", "trading")
        """
        result = temporal_narrative(period, focus)
        return json.dumps(result, ensure_ascii=False, indent=2)
