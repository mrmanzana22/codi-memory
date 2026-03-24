# Codi Fine-Tuning Guide
> Setup, configuracion y troubleshooting para entrenar adapters LoRA en Apple Silicon.
> Ultima actualizacion: 2026-03-23

---

## Hardware

| Componente | Air M4 (actual) | Pro M4 Pro (Hare) |
|------------|------------------|--------------------|
| RAM | 16GB unified | 48GB unified |
| GPU cores | 10 | 18 |
| Config segura | batch=1, seq=768 | batch=2, seq=1536 |
| Training peak | ~3.87GB | ~8-10GB |
| Riesgo OOM | Alto sin limits | Ninguno |

---

## Pre-requisitos

```bash
# Python 3.12+ con mlx-lm
pip3 install mlx-lm

# Verificar instalacion
python3 -c "import mlx.core as mx; print(f'MLX {mx.__version__}')"
python3 -c "from mlx_lm import lora; print('mlx-lm OK')"
```

---

## Estructura de archivos

```
~/codi-memory/
  finetune_data_v2/          # Training data (JSONL, chat format)
    train.jsonl
    valid.jsonl
    test.jsonl
  adapters/
    codi-v1/                  # Adapter v1 (completado)
    codi-v2/                  # Adapter v2 (en progreso)
      adapter_config.json
      adapters.safetensors    # Ultimo checkpoint
      0000200_adapters.safetensors
      0000400_adapters.safetensors
      ...
  train_safe.py               # Wrapper con memory limits
```

---

## Modelo base

```
mlx-community/Qwen3-4B-Instruct-2507-4bit
```
- 4B params, quantizado a 4-bit (QLoRA automatico)
- Se descarga automatico de HuggingFace la primera vez
- Cache local en ~/.cache/huggingface/

---

## Configuracion de training

### adapter_config.json
```json
{
  "model": "mlx-community/Qwen3-4B-Instruct-2507-4bit",
  "fine_tune_type": "lora",
  "batch_size": 1,
  "max_seq_length": 768,
  "num_layers": 16,
  "iters": 2000,
  "learning_rate": 5e-05,
  "save_every": 200,
  "steps_per_report": 50,
  "steps_per_eval": 99999,
  "grad_checkpoint": true,
  "mask_prompt": false,
  "lora_parameters": {
    "rank": 8,
    "dropout": 0.0,
    "scale": 20.0
  }
}
```

### Parametros criticos

| Parametro | Valor | Por que |
|-----------|-------|---------|
| `batch_size` | 1 | Minimo, esencial para 16GB |
| `max_seq_length` | 768 | Balance memoria/calidad. En 48GB usar 1536 |
| `grad_checkpoint` | true | Recomputa intermedios en backward, ahorra ~40% GPU mem |
| `mask_prompt` | false | **CRITICO: `true` causa NaN con data v2** |
| `num_layers` | 16 | Todas las capas. Bajar a 8 si OOM persiste |
| `save_every` | 200 | Checkpoints frecuentes para recovery |
| `steps_per_eval` | 99999 | Skip eval durante training (ahorra memoria) |

---

## OBLIGATORIO: Memory Limits (Air M4 16GB)

### El problema
MLX por defecto:
- Wires 75% de RAM (12GB) — memoria que no se puede swapear
- Cache ilimitado — tensores liberados NO se devuelven al sistema
- Metal no puede reclamar memoria para otros procesos
- Resultado: Metal OOM crash aleatorio (SIGABRT en com.Metal.CompletionQueueDispatch)

### La solucion: train_safe.py

```python
"""
train_safe.py — MLX LoRA training con memory limits.
SIEMPRE usar este script en el Air M4 16GB.
"""
import mlx.core as mx

# Memory limits — ANTES de cargar modelo
mx.set_memory_limit(10 * 1024**3)   # 10GB techo total
mx.set_cache_limit(512 * 1024**2)   # 512MB cache max (CLAVE)
mx.set_wired_limit(10 * 1024**3)    # 10GB wired max

# Verificar
print(f"[SAFE] Memory limit: 10GB")
print(f"[SAFE] Cache limit: 512MB")
print(f"[SAFE] Wired limit: 10GB")

# Lanzar training via CLI
import sys
sys.argv = [
    "mlx_lm", "lora",
    "--model", "mlx-community/Qwen3-4B-Instruct-2507-4bit",
    "--data", "./finetune_data_v2",
    "--adapter-path", "./adapters/codi-v2",
    "--resume-adapter-file", "./adapters/codi-v2/CHECKPOINT.safetensors",  # <-- CAMBIAR
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
```

### Por que funciona
- `set_cache_limit(512MB)`: Obliga a MLX a devolver tensores al sistema en vez de cachearlos
- `set_memory_limit(10GB)`: Techo total, falla graceful en vez de kernel panic
- `set_wired_limit(10GB)`: Reduce paginas wired, deja margen para sistema

---

## Lanzar training

### En Air M4 16GB (SIEMPRE con train_safe.py)
```bash
# 1. Matar daemon y procesos que usen GPU
launchctl bootout gui/$(id -u) ~/Library/LaunchAgents/com.codi.daemon.plist 2>/dev/null
launchctl bootout gui/$(id -u) ~/Library/LaunchAgents/com.seb.daemon.plist 2>/dev/null
launchctl disable gui/$(id -u)/com.codi.daemon
launchctl disable gui/$(id -u)/com.seb.daemon

# 2. Cerrar Chrome y apps graficas (usan GPU memory)

# 3. Editar train_safe.py: cambiar CHECKPOINT al ultimo checkpoint disponible

# 4. Lanzar
cd ~/codi-memory
/opt/homebrew/bin/python3 train_safe.py > /tmp/codi-v2-train.log 2>&1 &
echo "PID: $!"

# 5. Monitorear
tail -f /tmp/codi-v2-train.log
# O periodicamente:
grep "^Iter" /tmp/codi-v2-train.log | tail -5
```

### En Pro M4 Pro 48GB (directo, sin limits)
```bash
# No necesita memory limits — sobra memoria
cd ~/codi-memory
python3 -m mlx_lm lora \
  --model mlx-community/Qwen3-4B-Instruct-2507-4bit \
  --data ./finetune_data_v2 \
  --adapter-path ./adapters/codi-v2 \
  --resume-adapter-file ./adapters/codi-v2/CHECKPOINT.safetensors \
  --train \
  --batch-size 2 \
  --max-seq-length 1536 \
  --num-layers 16 \
  --iters 2000 \
  --save-every 200 \
  --steps-per-report 50 \
  --steps-per-eval 99999 \
  --learning-rate 5e-5 \
  --grad-checkpoint \
  > /tmp/codi-v2-train.log 2>&1 &
```

---

## Resumir desde checkpoint

Si el training se interrumpe, resume desde el ultimo checkpoint:

```bash
# Ver checkpoints disponibles
ls -la ~/codi-memory/adapters/codi-v2/*.safetensors

# Agregar flag al comando:
--resume-adapter-file ./adapters/codi-v2/0000800_adapters.safetensors
```

**Nota:** MLX reinicia el contador de iters desde 1 al resumir, pero los weights continuan donde quedaron.

---

## Post-training: Swap adapter

```bash
# 1. Verificar que el adapter final existe
ls -la ~/codi-memory/adapters/codi-v2/adapters.safetensors

# 2. Actualizar llm_router.py para usar codi-v2
# Cambiar adapter path de codi-v1 a codi-v2

# 3. Revivir daemon
launchctl enable gui/$(id -u)/com.codi.daemon
launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.codi.daemon.plist
launchctl enable gui/$(id -u)/com.seb.daemon
launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.seb.daemon.plist

# 4. Verificar daemon
curl http://127.0.0.1:8420/health
```

---

## Troubleshooting

### Metal OOM crash (SIGABRT)
- **Causa:** Cache de MLX acumula memoria, Metal no encuentra bloque contiguo
- **Solucion:** Usar train_safe.py con memory limits
- **Extra:** Cerrar Chrome, Screen Sharing, cualquier app que use GPU

### NaN en loss
- **Causa:** `--mask-prompt` con data mal formateada
- **Solucion:** NO usar `--mask-prompt` con data v2

### Training muy lento (< 0.1 it/sec)
- **Causa:** Swap activo, memoria comprimida
- **Solucion:** Cerrar mas apps, reducir `--max-seq-length` a 512

### Daemon revive solo despues de matarlo
- **Causa:** Multiples plists con KeepAlive=1 (com.codi.daemon + com.seb.daemon)
- **Solucion:** `launchctl disable gui/$(id -u)/com.codi.daemon` Y `com.seb.daemon`

### Val loss sube al resumir
- **Normal:** El optimizer state se resetea al resumir. Loss sube brevemente y luego baja.

---

## Historial de training codi-v2

| Fecha | Config | Iters completadas | Loss final | Notas |
|-------|--------|-------------------|------------|-------|
| 2026-03-22 | batch=1, seq=768, sin limits | 0-550 | 1.334 | Crash Metal OOM iter 550 |
| 2026-03-23 run1 | batch=1, seq=768, sin limits | 400-700 | 1.158 | Crash Metal OOM iter 700 |
| 2026-03-23 run2 | batch=1, seq=768, sin limits | 400-50 | 1.073 | Crash Metal OOM iter 50 |
| 2026-03-23 safe | batch=1, seq=768, **con limits** | 600-2000 | TBD | En progreso, estable |

---

## Data format (finetune_data_v2)

```jsonl
{"messages": [
  {"role": "system", "content": "Eres Codi..."},
  {"role": "user", "content": "..."},
  {"role": "assistant", "content": "..."}
]}
```

- ~12K ejemplos (train + valid + test)
- Formato chat (messages con 3 turnos)
- Secuencias largas se truncan a max_seq_length
