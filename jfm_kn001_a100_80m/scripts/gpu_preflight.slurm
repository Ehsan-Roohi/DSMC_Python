#!/bin/bash
set -euo pipefail

ROOT="${JFM_ROOT:-${SLURM_SUBMIT_DIR:?}}"
OUTPUT_ROOT="${JFM_OUTPUT_ROOT:-$ROOT/run_output}"
PREFLIGHT_DIR="$OUTPUT_ROOT/preflight"
mkdir -p "$PREFLIGHT_DIR"
cd "$ROOT"

echo "host=$(hostname) job=${SLURM_JOB_ID:-none}"
nvidia-smi --query-gpu=name,memory.total,memory.free,compute_cap --format=csv

mamba run -n dsmc-gpu python solver/JFM_bgk_shakhov_quarter.py \
  --mode SHAKHOV --kn 0.01 --rt 0.5 --particles 80000000 \
  --steps 4 --sample-start 2 --sample-every 1 --time-blocks 2 \
  --seed 900000007 --progress-every 1 --require-free-gb 60 \
  --output "$PREFLIGHT_DIR"

test -s "$PREFLIGHT_DIR/ThermalCavity_SHAKHOV_Kn0.01_RT0p5_quarter_seed900000007_metrics.json"
test -s "$PREFLIGHT_DIR/ThermalCavity_SHAKHOV_Kn0.01_RT0p5_quarter_seed900000007_raw.npz"
echo "[OK] 80M Shakhov memory/import preflight passed on A100-80GB"
