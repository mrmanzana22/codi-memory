#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

mkdir -p logs
./venv/bin/python scripts/dual_canary.py >> logs/dual_canary.log 2>&1
