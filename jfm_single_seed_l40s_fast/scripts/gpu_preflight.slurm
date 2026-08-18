#!/bin/bash
set -euo pipefail

ROOT="${JFM_ROOT:-${SLURM_SUBMIT_DIR:?}}"
OUTPUT_ROOT="${JFM_OUTPUT_ROOT:-$ROOT/run_output}"
PREFLIGHT_DIR="$OUTPUT_ROOT/l40s/preflight"
mkdir -p "$PREFLIGHT_DIR"
cd "$ROOT"

echo "route=l40s host=$(hostname) job=${SLURM_JOB_ID:-none}"
nvidia-smi --query-gpu=name,memory.total,memory.free,compute_cap --format=csv
GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
[[ "$GPU_NAME" == *L40S* ]] || { echo "Expected L40S, got $GPU_NAME" >&2; exit 3; }

COMMON=(
  --kn 20 --rt 0.5 --particles 80000000
  --steps 4 --sample-start 2 --sample-every 1
  --time-blocks 2 --progress-every 1 --require-free-gb 40.0
)

mamba run --no-capture-output -n dsmc-gpu python -u \
  solver/JFM_hs_dsmc_quarter.py "${COMMON[@]}" \
  --seed 900000001 --output "$PREFLIGHT_DIR/hs80m"
mamba run --no-capture-output -n dsmc-gpu python -u \
  solver/JFM_bgk_shakhov_quarter.py --mode SHAKHOV "${COMMON[@]}" \
  --seed 900000007 --output "$PREFLIGHT_DIR/shakhov80m"

echo "[OK] exact-L40S 80M memory preflight passed"
