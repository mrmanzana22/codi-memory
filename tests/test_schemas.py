#!/usr/bin/env python3
"""
Unit tests for Schema System Phase 1 (Phase 3A).
Run: python3 -m pytest tests/test_schemas.py -v
"""

import sys
import os
import sqlite3
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from modules.schemas import (
    SchemaSlot,
    SchemaTemplate,
    extract_schemas_from_facts,
    match_schema,
    schema_congruence_bonus,
    init_schemas_table,
    save_schema,
    load_schemas,
    SCHEMA_CONGRUENCE_MAX_BONUS,
    SCHEMA_MIN_FACTS_FOR_EXTRACTION,
    SCHEMA_MATCH_THRESHOLD,
)


def _make_facts(topic, count, prefix="fact about"):
    """Helper to create semantic facts with common words."""
    return [
        {"topic": topic, "fact": f"{prefix} {topic} item {i} with implementation details",
         "confidence": 0.8, "evidence_count": 2}
        for i in range(count)
    ]


# ============================================================
# SCHEMA EXTRACTION
# ============================================================

class TestSchemaExtraction:
    def test_extract_from_enough_facts(self):
        """3+ facts on same topic -> schema extracted."""
        facts = _make_facts("trading", 5)
        schemas = extract_schemas_from_facts(facts)
        assert len(schemas) >= 1
        assert schemas[0].topic == "trading"
        assert schemas[0].instance_count == 5

    def test_no_extraction_below_threshold(self):
        """Fewer than SCHEMA_MIN_FACTS_FOR_EXTRACTION -> no schema."""
        facts = _make_facts("rare_topic", 2)
        schemas = extract_schemas_from_facts(facts)
        assert len(schemas) == 0

    def test_empty_facts(self):
        """Empty input -> empty output."""
        assert extract_schemas_from_facts([]) == []

    def test_multiple_topics(self):
        """Facts from 2 topics -> 2 schemas."""
        facts = _make_facts("trading", 4) + _make_facts("fullempaques", 3)
        schemas = extract_schemas_from_facts(facts)
        topics = [s.topic for s in schemas]
        assert "trading" in topics
        assert "fullempaques" in topics

    def test_schema_has_slots(self):
        """Extracted schema should have topic and action slots."""
        facts = _make_facts("trading", 5)
        schemas = extract_schemas_from_facts(facts)
        slot_names = [s.name for s in schemas[0].slots]
        assert "topic" in slot_names
        assert "action" in slot_names


# ============================================================
# SCHEMA MATCHING
# ============================================================

class TestSchemaMatching:
    def test_match_congruent_content(self):
        """Content with overlapping words matches schema."""
        schema = SchemaTemplate(
            name="test", description="", topic="trading",
            slots=[SchemaSlot(name="action", slot_type="action",
                              typical_values=["implementation", "details", "trading"])]
        )
        matches = match_schema("implementation of trading details", [schema])
        assert len(matches) >= 1
        assert matches[0][1] > SCHEMA_MATCH_THRESHOLD

    def test_no_match_unrelated(self):
        """Completely unrelated content -> no match."""
        schema = SchemaTemplate(
            name="test", description="", topic="trading",
            slots=[SchemaSlot(name="action", slot_type="action",
                              typical_values=["kubernetes", "docker", "deployment"])]
        )
        matches = match_schema("fullempaques plegadiza cotizacion", [schema])
        assert len(matches) == 0

    def test_empty_inputs(self):
        """Empty content or schemas -> empty matches."""
        assert match_schema("", []) == []
        assert match_schema("test", []) == []

    def test_matches_sorted_by_confidence(self):
        """Multiple matches should be sorted descending by confidence."""
        s1 = SchemaTemplate(
            name="weak", description="", topic="a",
            slots=[SchemaSlot(name="x", slot_type="action", typical_values=["alpha"])]
        )
        s2 = SchemaTemplate(
            name="strong", description="", topic="b",
            slots=[SchemaSlot(name="x", slot_type="action",
                              typical_values=["alpha", "beta", "gamma"])]
        )
        matches = match_schema("alpha beta gamma delta", [s1, s2])
        if len(matches) >= 2:
            assert matches[0][1] >= matches[1][1]


# ============================================================
# CONGRUENCE BONUS
# ============================================================

class TestCongruenceBonus:
    def test_zero_confidence(self):
        assert schema_congruence_bonus(0.0) == 0.0

    def test_full_confidence(self):
        assert schema_congruence_bonus(1.0) == pytest.approx(SCHEMA_CONGRUENCE_MAX_BONUS)

    def test_partial_confidence(self):
        bonus = schema_congruence_bonus(0.5)
        assert 0 < bonus < SCHEMA_CONGRUENCE_MAX_BONUS

    def test_never_exceeds_max(self):
        assert schema_congruence_bonus(2.0) <= SCHEMA_CONGRUENCE_MAX_BONUS


# ============================================================
# SQLITE PERSISTENCE
# ============================================================

class TestSQLitePersistence:
    def _setup_db(self):
        """Create temp DB with migrations applied."""
        f = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        db_path = f.name
        f.close()
        from modules.migrations import apply_migrations
        apply_migrations(db_path, migrations_dir=os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "migrations"))
        return db_path

    def test_save_and_load(self):
        db_path = self._setup_db()
        try:
            conn = sqlite3.connect(db_path)
            init_schemas_table(conn)

            schema = SchemaTemplate(
                name="test_schema", description="A test", topic="testing",
                slots=[SchemaSlot(name="entity", slot_type="entity", typical_values=["foo", "bar"])],
                instance_count=3,
            )
            sid = save_schema(conn, schema)
            assert sid > 0

            loaded = load_schemas(conn, topic="testing")
            assert len(loaded) == 1
            assert loaded[0].name == "test_schema"
            assert loaded[0].instance_count == 3
            assert len(loaded[0].slots) == 1
            assert loaded[0].slots[0].typical_values == ["foo", "bar"]
            conn.close()
        finally:
            os.unlink(db_path)

    def test_save_increments_count(self):
        db_path = self._setup_db()
        try:
            conn = sqlite3.connect(db_path)
            init_schemas_table(conn)

            schema = SchemaTemplate(name="dup", description="", topic="t",
                                    slots=[SchemaSlot(name="x", slot_type="action")])
            save_schema(conn, schema)
            save_schema(conn, schema)  # Second save increments count

            loaded = load_schemas(conn, topic="t")
            assert len(loaded) == 1
            assert loaded[0].instance_count == 2
            conn.close()
        finally:
            os.unlink(db_path)

    def test_load_all(self):
        db_path = self._setup_db()
        try:
            conn = sqlite3.connect(db_path)
            init_schemas_table(conn)

            for topic in ["a", "b", "c"]:
                schema = SchemaTemplate(name=f"s_{topic}", description="", topic=topic,
                                        slots=[SchemaSlot(name="x", slot_type="entity")])
                save_schema(conn, schema)

            all_schemas = load_schemas(conn)
            assert len(all_schemas) == 3
            conn.close()
        finally:
            os.unlink(db_path)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
