#!/bin/bash
set -euo pipefail

ROOT="${JFM_ROOT:-${SLURM_SUBMIT_DIR:?}}"
OUTPUT_ROOT="${JFM_OUTPUT_ROOT:-$ROOT/run_output}"
CASE_FILE="${JFM_CASE_TABLE:-$ROOT/cases/kn001_a100_80m.csv}"
OUTPUT_DIR="$OUTPUT_ROOT/runs"
cd "$ROOT"
mkdir -p "$OUTPUT_DIR"

ROW="$((SLURM_ARRAY_TASK_ID + 2))"
LINE="$(awk -F, -v row="$ROW" 'NR==row {print; exit}' "$CASE_FILE")"
if [[ -z "$LINE" ]]; then
  echo "No case row for array task $SLURM_ARRAY_TASK_ID" >&2
  exit 2
fi
IFS=, read -r CASE_ID MODEL KN RT SEED GROUP <<< "$LINE"
RT_TAG="${RT//./p}"
STEM="ThermalCavity_${MODEL}_Kn${KN}_RT${RT_TAG}_quarter_seed${SEED}"

echo "case=$CASE_ID model=$MODEL Kn=$KN RT=$RT seed=$SEED group=$GROUP"
echo "particles=80000000 steps=5000000 sample_window=100000:5000000:2 time_blocks=49"
echo "host=$(hostname) job=$SLURM_JOB_ID task=$SLURM_ARRAY_TASK_ID"
nvidia-smi --query-gpu=name,memory.total,memory.free,compute_cap --format=csv

if [[ -s "$OUTPUT_DIR/${STEM}_metrics.json" && -s "$OUTPUT_DIR/${STEM}_raw.npz" ]]; then
  echo "[SKIP] Complete outputs already exist for $STEM"
  exit 0
fi

mamba run -n dsmc-gpu python solver/JFM_bgk_shakhov_quarter.py \
  --mode "$MODEL" --kn "$KN" --rt "$RT" --particles 80000000 \
  --steps 5000000 --sample-start 100000 --sample-every 2 \
  --time-blocks 49 --seed "$SEED" --progress-every 10000 \
  --require-free-gb 60 --output "$OUTPUT_DIR"

test -s "$OUTPUT_DIR/${STEM}_metrics.json"
test -s "$OUTPUT_DIR/${STEM}_raw.npz"
echo "[OK] $CASE_ID complete"
