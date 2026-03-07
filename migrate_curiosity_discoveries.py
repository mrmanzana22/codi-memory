#!/usr/bin/env python3
"""One-shot migration: save existing curiosity discoveries to long-term memory.

The curiosity system resolved 42 discoveries but only saved them truncated
to preguntas_curiosidad.json. This migrates them to long-term memory via
add_memory_smart() so they're searchable and integrated.

Run once: python3 migrate_curiosity_discoveries.py
"""
import json
import os
import sys
import time

# Setup path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from modules.memory_smart import add_memory_smart


def main():
    json_path = os.path.join(os.path.dirname(__file__), "preguntas_curiosidad.json")
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    exploradas = data.get("exploradas", [])
    discoveries = [e for e in exploradas if e.get("outcome") == "discovery"]

    print(f"Found {len(discoveries)} discoveries to migrate")

    saved = 0
    skipped = 0
    errors = 0

    for d in discoveries:
        question = d.get("pregunta", "")
        description = d.get("outcome_description", "")
        category = d.get("categoria", "general")
        source_type = d.get("source", "manual")

        if not description:
            skipped += 1
            continue

        content = (
            f"[CURIOSITY DISCOVERY] Q: {question} | "
            f"A: {description} | source: {source_type}"
        )

        try:
            result = add_memory_smart(
                content=content,
                category=category,
                source="learned",
                importance="medium",
            )
            result_str = str(result)
            if "skipped_duplicate" in result_str:
                skipped += 1
                print(f"  SKIP #{d['id']}: duplicate")
            else:
                saved += 1
                print(f"  SAVE #{d['id']}: {question[:50]}...")
        except Exception as e:
            errors += 1
            print(f"  ERR  #{d['id']}: {e}")

        time.sleep(0.2)  # Don't hammer the store

    print(f"\nDone: {saved} saved, {skipped} skipped (dup/empty), {errors} errors")


if __name__ == "__main__":
    main()
