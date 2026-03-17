"""
Generate Opus-quality gold training examples for Codi fine-tuning.
Hand-crafted with deep system knowledge. Spec v2.

Usage:
    python -m scripts.generate_opus_gold
    python -m scripts.generate_opus_gold --task self_monitor
"""
import json
import argparse
from pathlib import Path

OUT_DIR = Path(__file__).resolve().parent.parent / "training_data" / "opus_gold"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def write_jsonl(filename: str, examples: list[dict]):
    path = OUT_DIR / filename
    with open(path, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    print(f"  {filename}: {len(examples)} examples")


def m(user: str, assistant: str, system: str | None = None) -> dict:
    """Build a messages dict. System is optional."""
    msgs = []
    if system:
        msgs.append({"role": "system", "content": system})
    msgs.append({"role": "user", "content": user})
    msgs.append({"role": "assistant", "content": assistant})
    return {"messages": msgs}


# Import task generators
from scripts._gold_self_monitor import gen_self_monitor
from scripts._gold_system_expert import gen_system_expert
from scripts._gold_curiosity import gen_curiosity
from scripts._gold_safety import gen_safety
from scripts._gold_compress import gen_compress_episodes, gen_compress_checkpoints
from scripts._gold_dpo import gen_dpo


GENERATORS = {
    "self_monitor": ("self_monitor.jsonl", gen_self_monitor),
    "system_expert": ("system_expert.jsonl", gen_system_expert),
    "resolve_curiosity": ("resolve_curiosity.jsonl", gen_curiosity),
    "safety_boundaries": ("safety_boundaries.jsonl", gen_safety),
    "compress_episodes": ("compress_episodes.jsonl", gen_compress_episodes),
    "compress_checkpoints": ("compress_checkpoints.jsonl", gen_compress_checkpoints),
    "dpo_pairs": ("dpo_pairs.jsonl", gen_dpo),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", help="Generate only this task type")
    args = parser.parse_args()

    print("Generating Opus gold training examples (Spec v2)")
    print("=" * 50)
    total = 0

    for name, (filename, gen_fn) in GENERATORS.items():
        if args.task and args.task != name:
            continue
        examples = gen_fn()
        write_jsonl(filename, examples)
        total += len(examples)

    print(f"\nTotal: {total} examples")
    print(f"Output: {OUT_DIR}")


if __name__ == "__main__":
    main()
