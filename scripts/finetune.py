"""
Fine-tune Qwen3-4B on Codi's training data using MLX-LM QLoRA.

Pipeline:
    1. Split training data (eval_harness split)
    2. Merge per-task splits into single train/valid/test.jsonl
    3. Run MLX-LM LoRA training
    4. Test with eval_harness eval

Usage:
    # Full pipeline: split + merge + train
    python -m scripts.finetune

    # Just merge (if splits already exist)
    python -m scripts.finetune --skip-split

    # Just train (if merged files already exist)
    python -m scripts.finetune --skip-split --skip-merge

    # Dry run (show what would happen)
    python -m scripts.finetune --dry-run
"""

import argparse
import json
import logging
import os
import random
import re
import subprocess
import sys
from pathlib import Path

RANDOM_SEED = 42

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
_log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
TRAINING_DATA_DIR = PROJECT_ROOT / "training_data"
EVAL_DATA_DIR = PROJECT_ROOT / "eval_data"
GOLD_DATA_DIR = TRAINING_DATA_DIR / "opus_gold"
MERGED_DIR = PROJECT_ROOT / "finetune_data"
ADAPTER_DIR = PROJECT_ROOT / "adapters" / "codi-v1"

# Model — 4-bit quantized Qwen3-4B
MODEL = "mlx-community/Qwen3-4B-Instruct-2507-4bit"

# ---------------------------------------------------------------------------
# Step 1: Split (delegates to eval_harness)
# ---------------------------------------------------------------------------
def step_split(min_examples: int = 50):
    """Split training_data/*.jsonl into eval_data/{train,valid,test}/."""
    _log.info("=== STEP 1: Split training data ===")
    venv_python = PROJECT_ROOT / "venv" / "bin" / "python3"
    cmd = [
        str(venv_python), "-m", "scripts.eval_harness", "split",
        "--input", str(TRAINING_DATA_DIR),
        "--output", str(EVAL_DATA_DIR),
        "--min-examples", str(min_examples),
    ]
    _log.info("Running: %s", " ".join(cmd))
    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT), capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print(result.stderr)
    if result.returncode != 0:
        _log.error("Split failed with code %d", result.returncode)
        sys.exit(1)


# ---------------------------------------------------------------------------
# Step 2: Merge per-task splits into single files for MLX-LM
# ---------------------------------------------------------------------------
def _is_contaminated(text: str) -> bool:
    """Check if text mentions wrong DB tech (Codi uses SQLite, not Postgres)."""
    return bool(re.search(
        r'postgres|supabase|pg_catalog|pg_stat|mongodb|mongo_client|redis\.client',
        text, re.IGNORECASE
    ))


def step_merge():
    """Merge eval_data/{train,valid,test}/*.jsonl + gold into finetune_data/."""
    _log.info("=== STEP 2: Merge splits for MLX-LM ===")
    MERGED_DIR.mkdir(parents=True, exist_ok=True)
    random.seed(RANDOM_SEED)

    # Max chars per example (~1536 tokens * 3 chars/token for multilingual)
    MAX_CHARS = 4500
    truncated_count = 0
    contaminated_count = 0

    for split in ("train", "valid", "test"):
        split_dir = EVAL_DATA_DIR / split
        if not split_dir.exists():
            _log.error("Split dir not found: %s", split_dir)
            sys.exit(1)

        all_lines = []
        for jsonl_file in sorted(split_dir.glob("*.jsonl")):
            task_name = jsonl_file.stem
            with open(jsonl_file, "r", encoding="utf-8") as f:
                lines = [line.strip() for line in f if line.strip()]

            # DPO pairs: convert to chat format for SFT
            if task_name == "dpo_pairs":
                converted = []
                for line in lines:
                    try:
                        d = json.loads(line)
                        messages = []
                        prompt_msgs = d.get("prompt", [])
                        if isinstance(prompt_msgs, list):
                            messages.extend(prompt_msgs)
                        for msg in d.get("chosen", []):
                            messages.append(msg)
                        if len(messages) >= 2:
                            converted.append(json.dumps({"messages": messages}, ensure_ascii=False))
                    except Exception:
                        pass
                lines = converted
                _log.info("  %s/%s: %d DPO pairs -> %d chat examples",
                          split, task_name, len(lines), len(converted))

            # Filter contaminated examples (wrong DB tech)
            before_contam = len(lines)
            lines = [l for l in lines if not _is_contaminated(l)]
            contam_removed = before_contam - len(lines)
            if contam_removed:
                contaminated_count += contam_removed
                _log.info("  %s/%s: removed %d contaminated examples",
                          split, task_name, contam_removed)

            # Filter out sequences that are too long (would cause NaN)
            filtered = []
            for line in lines:
                try:
                    d = json.loads(line)
                    total_chars = sum(len(m.get("content", "")) for m in d.get("messages", []))
                    if total_chars > MAX_CHARS:
                        truncated_count += 1
                        continue
                    filtered.append(line)
                except Exception:
                    pass
            lines = filtered

            all_lines.extend(lines)
            _log.info("  %s/%s: %d examples", split, task_name, len(lines))

        # Inject gold examples into TRAIN split only
        if split == "train" and GOLD_DATA_DIR.exists():
            gold_count = 0
            for jsonl_file in sorted(GOLD_DATA_DIR.glob("*.jsonl")):
                task_name = f"gold_{jsonl_file.stem}"
                with open(jsonl_file, "r", encoding="utf-8") as f:
                    lines = [line.strip() for line in f if line.strip()]

                # DPO gold: convert to chat format
                if jsonl_file.stem == "dpo_pairs":
                    converted = []
                    for line in lines:
                        try:
                            d = json.loads(line)
                            messages = []
                            prompt_msgs = d.get("prompt", [])
                            if isinstance(prompt_msgs, list):
                                messages.extend(prompt_msgs)
                            for msg in d.get("chosen", []):
                                messages.append(msg)
                            if len(messages) >= 2:
                                converted.append(json.dumps({"messages": messages}, ensure_ascii=False))
                        except Exception:
                            pass
                    lines = converted

                # Length filter (gold should already be within limits)
                filtered = []
                for line in lines:
                    try:
                        d = json.loads(line)
                        total_chars = sum(len(m.get("content", "")) for m in d.get("messages", []))
                        if total_chars > MAX_CHARS:
                            truncated_count += 1
                            continue
                        filtered.append(line)
                    except Exception:
                        pass
                lines = filtered

                all_lines.extend(lines)
                gold_count += len(lines)
                _log.info("  %s/%s: %d gold examples", split, task_name, len(lines))

            _log.info("  Gold total injected into train: %d", gold_count)

        # Shuffle for training
        random.shuffle(all_lines)

        out_path = MERGED_DIR / f"{split}.jsonl"
        with open(out_path, "w", encoding="utf-8") as f:
            for line in all_lines:
                f.write(line + "\n")

        _log.info("  => %s: %d total examples", split, len(all_lines))

    if truncated_count:
        _log.warning("  Filtered %d examples exceeding %d chars (~1536 tokens)", truncated_count, MAX_CHARS)
    if contaminated_count:
        _log.warning("  Filtered %d contaminated examples (wrong DB tech)", contaminated_count)


# ---------------------------------------------------------------------------
# Step 3: Train with MLX-LM LoRA
# ---------------------------------------------------------------------------
def step_train(iters: int = 1000, batch_size: int = 2, learning_rate: float = 1e-4,
               num_layers: int = 16, max_seq_length: int = 2048,
               steps_per_eval: int = 100, save_every: int = 200):
    """Run QLoRA fine-tuning via MLX-LM."""
    _log.info("=== STEP 3: QLoRA Fine-tuning ===")

    ADAPTER_DIR.mkdir(parents=True, exist_ok=True)

    venv_python = PROJECT_ROOT / "venv" / "bin" / "python3"
    cmd = [
        str(venv_python), "-m", "mlx_lm", "lora",
        "--model", MODEL,
        "--data", str(MERGED_DIR),
        "--train",
        "--fine-tune-type", "lora",
        "--batch-size", str(batch_size),
        "--iters", str(iters),
        "--learning-rate", str(learning_rate),
        "--num-layers", str(num_layers),
        "--max-seq-length", str(max_seq_length),
        "--adapter-path", str(ADAPTER_DIR),
        "--steps-per-report", "10",
        "--steps-per-eval", str(steps_per_eval),
        "--save-every", str(save_every),
        "--mask-prompt",
        "--grad-checkpoint",
    ]

    _log.info("Running: %s", " ".join(cmd))
    _log.info("Model: %s", MODEL)
    _log.info("Data: %s", MERGED_DIR)
    _log.info("Adapters: %s", ADAPTER_DIR)
    _log.info("Config: iters=%d, batch=%d, lr=%s, layers=%d, max_seq=%d",
              iters, batch_size, learning_rate, num_layers, max_seq_length)

    # Run training (streams output)
    process = subprocess.Popen(cmd, cwd=str(PROJECT_ROOT))
    process.wait()

    if process.returncode != 0:
        _log.error("Training failed with code %d", process.returncode)
        sys.exit(1)

    _log.info("Training complete! Adapters saved to %s", ADAPTER_DIR)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Fine-tune Qwen3-4B on Codi training data")
    parser.add_argument("--skip-split", action="store_true", help="Skip the split step")
    parser.add_argument("--skip-merge", action="store_true", help="Skip the merge step")
    parser.add_argument("--skip-train", action="store_true", help="Skip training (just prep data)")
    parser.add_argument("--dry-run", action="store_true", help="Show what would happen")
    parser.add_argument("--iters", type=int, default=1000, help="Training iterations")
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--num-layers", type=int, default=16, help="LoRA layers")
    parser.add_argument("--max-seq-length", type=int, default=1536, help="Max sequence length")
    parser.add_argument("--steps-per-eval", type=int, default=100, help="Validation interval (0=disable)")
    parser.add_argument("--save-every", type=int, default=200, help="Checkpoint save interval")
    parser.add_argument("--min-examples", type=int, default=50, help="Min examples per task")
    args = parser.parse_args()

    _log.info("Codi Fine-tuning Pipeline")
    _log.info("========================")

    if args.dry_run:
        _log.info("[DRY RUN] Would:")
        if not args.skip_split:
            _log.info("  1. Split %s -> %s (min %d examples)", TRAINING_DATA_DIR, EVAL_DATA_DIR, args.min_examples)
        if not args.skip_merge:
            _log.info("  2. Merge %s/{train,valid,test}/*.jsonl -> %s/{train,valid,test}.jsonl", EVAL_DATA_DIR, MERGED_DIR)
        if not args.skip_train:
            _log.info("  3. Train %s with QLoRA (%d iters, batch %d, lr %s)",
                       MODEL, args.iters, args.batch_size, args.lr)
            _log.info("     Adapters -> %s", ADAPTER_DIR)
        return

    if not args.skip_split:
        step_split(args.min_examples)

    if not args.skip_merge:
        step_merge()

    if not args.skip_train:
        step_train(
            iters=args.iters,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            num_layers=args.num_layers,
            max_seq_length=args.max_seq_length,
            steps_per_eval=args.steps_per_eval if args.steps_per_eval > 0 else 99999,
            save_every=args.save_every,
        )

    _log.info("Pipeline complete!")


if __name__ == "__main__":
    main()
