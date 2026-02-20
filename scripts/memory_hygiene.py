#!/usr/bin/env python3
"""
Memory Hygiene Check: validates memory metadata integrity.

Connects to Qdrant, scrolls all memories, and checks for:
  1. Missing required fields (ownership_source, created_at, etc.)
  2. Invalid field values (confidence out of range, unknown sources)
  3. Schema version mismatches
  4. Orphan relations (related_to pointing to nonexistent IDs)

Outputs a PASS/FAIL verdict similar to dual_report.py.

Usage:
  python scripts/memory_hygiene.py [--collection codi_memories] [--limit 1000]
"""

import argparse
import sys
import os
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ============================================================
# FIELD DEFINITIONS
# ============================================================

REQUIRED_FIELDS = [
    "data",
    "ownership_source",
    "created_at",
    "narrative_importance",
]

VALID_SOURCES = {"experienced", "told", "learned", "inferred", "unknown"}
VALID_IMPORTANCE = {"critical", "high", "medium", "low"}
CURRENT_SCHEMA_VERSION = 4.0


# ============================================================
# CHECKS
# ============================================================

class HygieneResult:
    """Accumulates check results."""

    def __init__(self):
        self.total_points = 0
        self.missing_fields: list[dict] = []
        self.invalid_sources: list[dict] = []
        self.invalid_importance: list[dict] = []
        self.bad_confidence: list[dict] = []
        self.missing_created_at: list[dict] = []
        self.old_schema: list[dict] = []
        self.orphan_relations: list[dict] = []
        self.empty_data: list[dict] = []

    @property
    def total_issues(self) -> int:
        return (
            len(self.missing_fields) +
            len(self.invalid_sources) +
            len(self.invalid_importance) +
            len(self.bad_confidence) +
            len(self.missing_created_at) +
            len(self.old_schema) +
            len(self.orphan_relations) +
            len(self.empty_data)
        )


def check_point(point_id: str, payload: dict, all_ids: set) -> list[dict]:
    """Check a single memory point for hygiene issues.

    Returns list of issue dicts (empty if clean).
    """
    issues = []
    pid = str(point_id)

    # Check required fields
    for field in REQUIRED_FIELDS:
        if field not in payload or payload[field] is None:
            issues.append({
                "id": pid,
                "check": "missing_field",
                "field": field,
            })

    # Check data not empty
    data = payload.get("data", "")
    if not data or (isinstance(data, str) and not data.strip()):
        issues.append({
            "id": pid,
            "check": "empty_data",
        })

    # Validate ownership_source
    source = payload.get("ownership_source", "")
    if source and source not in VALID_SOURCES:
        issues.append({
            "id": pid,
            "check": "invalid_source",
            "value": source,
        })

    # Validate narrative_importance
    importance = payload.get("narrative_importance", "")
    if importance and importance not in VALID_IMPORTANCE:
        issues.append({
            "id": pid,
            "check": "invalid_importance",
            "value": importance,
        })

    # Validate ownership_confidence range
    confidence = payload.get("ownership_confidence")
    if confidence is not None:
        try:
            conf_val = float(confidence)
            if conf_val < 0 or conf_val > 1:
                issues.append({
                    "id": pid,
                    "check": "bad_confidence",
                    "value": conf_val,
                })
        except (ValueError, TypeError):
            issues.append({
                "id": pid,
                "check": "bad_confidence",
                "value": str(confidence),
            })

    # Validate created_at is parseable
    created_at = payload.get("created_at", "")
    if created_at:
        try:
            if "T" in str(created_at):
                datetime.fromisoformat(str(created_at).replace("Z", "+00:00"))
        except (ValueError, TypeError):
            issues.append({
                "id": pid,
                "check": "bad_created_at",
                "value": str(created_at)[:50],
            })

    # Check schema version
    version = payload.get("_v")
    if version is not None:
        try:
            if float(version) < CURRENT_SCHEMA_VERSION:
                issues.append({
                    "id": pid,
                    "check": "old_schema",
                    "value": version,
                })
        except (ValueError, TypeError):
            pass

    # Check orphan relations
    related = payload.get("related_to")
    if related and isinstance(related, str) and related not in all_ids:
        issues.append({
            "id": pid,
            "check": "orphan_relation",
            "related_to": related,
        })

    return issues


def run_hygiene(collection_name: str = "codi_memories",
                limit: int = 1000,
                qdrant_client=None) -> HygieneResult:
    """Run hygiene checks on all memories in a collection.

    Args:
        collection_name: Qdrant collection to check.
        limit: Max points to scan (0 = all).
        qdrant_client: Optional pre-built Qdrant client (for testing).

    Returns:
        HygieneResult with all findings.
    """
    result = HygieneResult()

    if qdrant_client is None:
        try:
            from modules.config import get_qdrant_client
            qdrant_client = get_qdrant_client()
        except Exception as e:
            print(f"ERROR: Cannot connect to Qdrant: {e}")
            return result

    # Scroll all points
    all_points = []
    offset = None
    batch_size = min(limit, 100) if limit > 0 else 100

    while True:
        try:
            kwargs = {
                "collection_name": collection_name,
                "limit": batch_size,
                "with_payload": True,
            }
            if offset:
                kwargs["offset"] = offset

            points, next_offset = qdrant_client.scroll(**kwargs)
            all_points.extend(points)

            if not next_offset or (limit > 0 and len(all_points) >= limit):
                break
            offset = next_offset
        except Exception as e:
            print(f"ERROR scrolling collection '{collection_name}': {e}")
            break

    if limit > 0:
        all_points = all_points[:limit]

    result.total_points = len(all_points)

    # Build ID set for orphan detection
    all_ids = {str(p.id) for p in all_points}

    # Check each point
    for p in all_points:
        payload = p.payload or {}
        issues = check_point(p.id, payload, all_ids)

        for issue in issues:
            check = issue["check"]
            if check == "missing_field":
                result.missing_fields.append(issue)
            elif check == "empty_data":
                result.empty_data.append(issue)
            elif check == "invalid_source":
                result.invalid_sources.append(issue)
            elif check == "invalid_importance":
                result.invalid_importance.append(issue)
            elif check == "bad_confidence":
                result.bad_confidence.append(issue)
            elif check in ("bad_created_at", "missing_created_at"):
                result.missing_created_at.append(issue)
            elif check == "old_schema":
                result.old_schema.append(issue)
            elif check == "orphan_relation":
                result.orphan_relations.append(issue)

    return result


# ============================================================
# REPORT FORMAT
# ============================================================

def _section(title: str) -> str:
    return f"\n{'=' * 50}\n{title}\n{'=' * 50}"


def format_report(result: HygieneResult, collection: str) -> str:
    """Format hygiene results as readable report."""
    lines = []
    lines.append("MEMORY HYGIENE REPORT")
    lines.append(f"Collection: {collection}")
    lines.append(f"Points scanned: {result.total_points}")
    lines.append(f"Total issues: {result.total_issues}")

    # Section 1: Volume
    lines.append(_section("VOLUME"))
    lines.append(f"  Total points: {result.total_points}")
    clean = result.total_points - len(set(
        i["id"] for cat in [
            result.missing_fields, result.invalid_sources,
            result.invalid_importance, result.bad_confidence,
            result.missing_created_at, result.old_schema,
            result.orphan_relations, result.empty_data,
        ] for i in cat
    ))
    lines.append(f"  Clean points: {clean}")
    pct = (clean / result.total_points * 100) if result.total_points > 0 else 100
    lines.append(f"  Health: {pct:.1f}%")

    # Section 2: Missing fields
    lines.append(_section("MISSING REQUIRED FIELDS"))
    if result.missing_fields:
        from collections import Counter
        field_counts = Counter(i["field"] for i in result.missing_fields)
        for field, count in field_counts.most_common():
            lines.append(f"  {field}: {count} points")
    else:
        lines.append("  None (all required fields present)")

    # Section 3: Invalid values
    lines.append(_section("INVALID VALUES"))
    if result.invalid_sources:
        lines.append(f"  Invalid ownership_source: {len(result.invalid_sources)}")
        for i in result.invalid_sources[:5]:
            lines.append(f"    - {i['id'][:12]}... value='{i['value']}'")
    if result.invalid_importance:
        lines.append(f"  Invalid narrative_importance: {len(result.invalid_importance)}")
        for i in result.invalid_importance[:5]:
            lines.append(f"    - {i['id'][:12]}... value='{i['value']}'")
    if result.bad_confidence:
        lines.append(f"  Bad ownership_confidence: {len(result.bad_confidence)}")
        for i in result.bad_confidence[:5]:
            lines.append(f"    - {i['id'][:12]}... value={i['value']}")
    if not (result.invalid_sources or result.invalid_importance or result.bad_confidence):
        lines.append("  None (all values valid)")

    # Section 4: Empty data
    lines.append(_section("EMPTY DATA"))
    if result.empty_data:
        lines.append(f"  Points with empty data: {len(result.empty_data)}")
    else:
        lines.append("  None")

    # Section 5: Schema version
    lines.append(_section("SCHEMA VERSION"))
    if result.old_schema:
        lines.append(f"  Points with old schema: {len(result.old_schema)}")
        from collections import Counter
        ver_counts = Counter(i["value"] for i in result.old_schema)
        for ver, count in ver_counts.most_common():
            lines.append(f"    v{ver}: {count} points")
    else:
        lines.append(f"  All at v{CURRENT_SCHEMA_VERSION} (or unversioned)")

    # Section 6: Orphan relations
    lines.append(_section("ORPHAN RELATIONS"))
    if result.orphan_relations:
        lines.append(f"  Orphan relations: {len(result.orphan_relations)}")
        for i in result.orphan_relations[:5]:
            lines.append(f"    - {i['id'][:12]}... -> {i['related_to'][:12]}...")
    else:
        lines.append("  None (all relations valid)")

    # Section 7: Gate verdict
    lines.append(_section("VERDICT"))

    # FAIL conditions
    fail_reasons = []
    if result.empty_data:
        fail_reasons.append(f"empty_data={len(result.empty_data)}")
    if result.missing_fields:
        # Only fail on truly critical missing fields
        critical_missing = [i for i in result.missing_fields if i["field"] == "data"]
        if critical_missing:
            fail_reasons.append(f"missing_data_field={len(critical_missing)}")
    if result.bad_confidence:
        fail_reasons.append(f"bad_confidence={len(result.bad_confidence)}")

    # WARNING conditions (don't fail)
    warnings = []
    if result.old_schema:
        pct_old = len(result.old_schema) / max(result.total_points, 1) * 100
        warnings.append(f"old_schema={len(result.old_schema)} ({pct_old:.0f}%)")
    if result.orphan_relations:
        warnings.append(f"orphan_relations={len(result.orphan_relations)}")
    if result.invalid_sources:
        warnings.append(f"invalid_sources={len(result.invalid_sources)}")

    if fail_reasons:
        lines.append(f"  VERDICT: FAIL ({', '.join(fail_reasons)})")
    else:
        lines.append("  VERDICT: PASS")

    if warnings:
        lines.append(f"  WARNINGS: {', '.join(warnings)}")

    return "\n".join(lines)


# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Memory hygiene check")
    parser.add_argument("--collection", default="codi_memories",
                        help="Qdrant collection to check")
    parser.add_argument("--limit", type=int, default=0,
                        help="Max points to scan (0 = all)")
    args = parser.parse_args()

    result = run_hygiene(collection_name=args.collection, limit=args.limit)
    report = format_report(result, args.collection)
    print(report)

    # Exit code: 0 = PASS, 1 = FAIL
    has_fails = bool(result.empty_data or result.bad_confidence or
                     any(i["field"] == "data" for i in result.missing_fields))
    return 1 if has_fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
