"""
train_safe.py — MLX LoRA training with Metal memory limits.
Prevents GPU OOM by capping Metal memory allocation.
"""
import mlx.core as mx

# Set Metal memory limits BEFORE any model loading
MEMORY_LIMIT_GB = 10
CACHE_LIMIT_MB = 512
WIRED_LIMIT_GB = 10

mx.set_memory_limit(int(MEMORY_LIMIT_GB * 1024**3))
mx.set_cache_limit(int(CACHE_LIMIT_MB * 1024**2))
mx.set_wired_limit(int(WIRED_LIMIT_GB * 1024**3))

print(f"[SAFE] Memory limit: {MEMORY_LIMIT_GB}GB")
print(f"[SAFE] Cache limit: {CACHE_LIMIT_MB}MB")
print(f"[SAFE] Wired limit: {WIRED_LIMIT_GB}GB")
print(f"[SAFE] Active memory: {mx.get_active_memory() / 1024**2:.0f}MB")
print(f"[SAFE] Peak memory: {mx.get_peak_memory() / 1024**2:.0f}MB")

# Run training via CLI entry point
import sys
sys.argv = [
    "mlx_lm", "lora",
    "--model", "mlx-community/Qwen3-4B-Instruct-2507-4bit",
    "--data", "./finetune_data_v2",
    "--adapter-path", "./adapters/codi-v2",
    "--resume-adapter-file", "./adapters/codi-v2/0000600_adapters.safetensors",
    "--train",
    "--batch-size", "1",
    "--max-seq-length", "768",
    "--num-layers", "16",
    "--iters", "2000",
    "--save-every", "200",
    "--steps-per-report", "50",
    "--steps-per-eval", "99999",
    "--learning-rate", "5e-5",
    "--grad-checkpoint",
]

from mlx_lm.cli import main
main()
