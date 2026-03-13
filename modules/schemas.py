"""
Codi Memory - Schema System Phase 1 (Phase 3A)
===============================================
Bartlett 1932, Rumelhart 1980, Gilboa & Marlatte 2017.

Schemas = abstracted knowledge templates from repeated experience.
mPFC stores schemas that guide encoding (congruent info encoded better)
and retrieval (schema-driven reconstruction).

Design: Pure logic, SQLite-only for persistence. No Qdrant.
"""

import json
import sqlite3
import os
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional
from modules.config import now_iso


# ============================================================
# CONSTANTS
# ============================================================

SCHEMA_CONGRUENCE_MAX_BONUS = 0.06   # Max bonus for schema-congruent memories
SCHEMA_MIN_FACTS_FOR_EXTRACTION = 3  # Need at least 3 facts to form a schema
SCHEMA_MATCH_THRESHOLD = 0.3         # Min confidence to count as a match

from modules.config import FTS_DB_PATH  # instance-aware


# ============================================================
# DATA TYPES
# ============================================================

@dataclass
class SchemaSlot:
    """A slot in a schema template."""
    name: str                    # e.g., "project", "action", "outcome"
    slot_type: str               # "entity" | "action" | "context" | "outcome"
    typical_values: list = field(default_factory=list)  # Common fillers
    required: bool = True


@dataclass
class SchemaTemplate:
    """An abstracted knowledge template (mPFC schema)."""
    name: str                    # e.g., "debugging_session", "deployment_flow"
    description: str
    topic: str                   # Parent topic (e.g., "fullempaques", "trading")
    slots: List[SchemaSlot] = field(default_factory=list)
    instance_count: int = 0      # How many times this pattern was observed
    created_at: str = field(default_factory=lambda: now_iso())
    last_matched: str = ""
    schema_id: int = 0


# ============================================================
# SQLITE PERSISTENCE
# ============================================================

def init_schemas_table(conn: sqlite3.Connection) -> None:
    """Validate schemas table exists (created by migrations)."""
    from modules.migrations import ensure_schema_ready
    ensure_schema_ready(conn, ["schemas"])


def save_schema(conn: sqlite3.Connection, schema: SchemaTemplate) -> int:
    """Save or update a schema in SQLite. Returns schema_id."""
    slots_json = json.dumps([
        {"name": s.name, "slot_type": s.slot_type,
         "typical_values": s.typical_values, "required": s.required}
        for s in schema.slots
    ])

    # Check if schema with same name+topic exists
    existing = conn.execute(
        "SELECT id, instance_count FROM schemas WHERE name = ? AND topic = ?",
        (schema.name, schema.topic)
    ).fetchone()

    if existing:
        conn.execute(
            "UPDATE schemas SET description = ?, instance_count = ?, slots_json = ?, last_matched = ? WHERE id = ?",
            (schema.description, existing[1] + 1, slots_json, now_iso(), existing[0])
        )
        conn.commit()
        return existing[0]
    else:
        cursor = conn.execute(
            "INSERT INTO schemas (name, description, topic, slots_json, instance_count, created_at) VALUES (?, ?, ?, ?, ?, ?)",
            (schema.name, schema.description, schema.topic, slots_json, max(1, schema.instance_count), schema.created_at)
        )
        conn.commit()
        return cursor.lastrowid


def load_schemas(conn: sqlite3.Connection, topic: Optional[str] = None) -> List[SchemaTemplate]:
    """Load schemas from SQLite, optionally filtered by topic."""
    if topic:
        rows = conn.execute(
            "SELECT id, name, description, topic, slots_json, instance_count, created_at, last_matched FROM schemas WHERE topic = ?",
            (topic,)
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT id, name, description, topic, slots_json, instance_count, created_at, last_matched FROM schemas"
        ).fetchall()

    schemas = []
    for row in rows:
        slots_data = json.loads(row[4])
        slots = [SchemaSlot(**s) for s in slots_data]
        schemas.append(SchemaTemplate(
            schema_id=row[0],
            name=row[1],
            description=row[2] or "",
            topic=row[3],
            slots=slots,
            instance_count=row[5],
            created_at=row[6],
            last_matched=row[7] or "",
        ))
    return schemas


# ============================================================
# SCHEMA EXTRACTION
# ============================================================

def extract_schemas_from_facts(facts: list) -> List[SchemaTemplate]:
    """Extract schema templates from a list of semantic facts.

    Groups facts by topic, finds recurring patterns within each topic.
    A schema is formed when 3+ facts share a topic and have common
    structural elements (entities, actions).

    Args:
        facts: List of dicts with keys: topic, fact, confidence, evidence_count

    Returns:
        List of newly extracted SchemaTemplate objects.
    """
    if not facts:
        return []

    # Group by topic
    by_topic = defaultdict(list)
    for f in facts:
        topic = f.get("topic", "general")
        if topic:
            by_topic[topic].append(f)

    schemas = []
    for topic, topic_facts in by_topic.items():
        if len(topic_facts) < SCHEMA_MIN_FACTS_FOR_EXTRACTION:
            continue

        # Extract common action words
        all_words = []
        for f in topic_facts:
            text = f.get("fact", "")
            words = [w.lower() for w in text.split() if len(w) > 3]
            all_words.extend(words)

        word_counts = Counter(all_words)
        common_words = [w for w, c in word_counts.most_common(10) if c >= 2]

        if not common_words:
            continue

        # Build slots from common patterns
        slots = []
        # Entity slot: topic itself
        slots.append(SchemaSlot(
            name="topic",
            slot_type="entity",
            typical_values=[topic],
            required=True,
        ))
        # Action slot: most common verbs/actions
        action_words = common_words[:3]
        if action_words:
            slots.append(SchemaSlot(
                name="action",
                slot_type="action",
                typical_values=action_words,
                required=True,
            ))
        # Context slot: fact texts as examples
        example_texts = [f.get("fact", "")[:80] for f in topic_facts[:5]]
        slots.append(SchemaSlot(
            name="context",
            slot_type="context",
            typical_values=example_texts,
            required=False,
        ))

        schema = SchemaTemplate(
            name=f"{topic}_pattern",
            description=f"Recurring pattern in {topic} ({len(topic_facts)} facts)",
            topic=topic,
            slots=slots,
            instance_count=len(topic_facts),
        )
        schemas.append(schema)

    return schemas


# ============================================================
# SCHEMA MATCHING
# ============================================================

def match_schema(content: str, schemas: List[SchemaTemplate]) -> List[tuple]:
    """Check if content matches any known schemas.

    Returns list of (schema, confidence) tuples, sorted by confidence descending.
    Confidence is based on keyword overlap between content and schema slot values.
    """
    if not content or not schemas:
        return []

    content_words = set(w.lower() for w in content.split() if len(w) > 3)
    matches = []

    for schema in schemas:
        # Collect all typical values from slots
        schema_words = set()
        for slot in schema.slots:
            for val in slot.typical_values:
                schema_words.update(w.lower() for w in str(val).split() if len(w) > 3)

        if not schema_words:
            continue

        # Jaccard-like overlap
        overlap = len(content_words & schema_words)
        total = len(content_words | schema_words)
        confidence = overlap / total if total > 0 else 0.0

        if confidence >= SCHEMA_MATCH_THRESHOLD:
            matches.append((schema, confidence))

    matches.sort(key=lambda x: x[1], reverse=True)
    return matches


# ============================================================
# CONGRUENCE BONUS
# ============================================================

def schema_congruence_bonus(match_confidence: float) -> float:
    """Compute encoding bonus for schema-congruent memories.

    Schema-congruent info is encoded more easily (Bartlett 1932).
    Bonus scales linearly with match confidence.

    Args:
        match_confidence: 0.0-1.0 from match_schema()

    Returns:
        Bonus value 0.0 to SCHEMA_CONGRUENCE_MAX_BONUS
    """
    return min(SCHEMA_CONGRUENCE_MAX_BONUS, match_confidence * SCHEMA_CONGRUENCE_MAX_BONUS)
