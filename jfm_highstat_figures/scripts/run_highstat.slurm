#!/bin/bash
set -euo pipefail

ROOT="${JFM_ROOT:-${SLURM_SUBMIT_DIR:?}}"
ROUTE="${JFM_ROUTE:?JFM_ROUTE is required}"
OUTPUT_ROOT="${JFM_OUTPUT_ROOT:-$ROOT/run_output}"
cd "$ROOT"

CASE_FILE="${JFM_CASE_TABLE:?JFM_CASE_TABLE is required}"
PARTICLES=80000000
ACTIVE_TASKS=21
case "$ROUTE" in
  vram48) REQUIRE_FREE=40.0 ;;
  vram80) REQUIRE_FREE=60.0 ;;
  *) echo "Unknown route: $ROUTE" >&2; exit 2 ;;
esac

if (( SLURM_ARRAY_TASK_ID >= ACTIVE_TASKS )); then
  echo "[SKIP] task $SLURM_ARRAY_TASK_ID is unused in $ROUTE mode"
  exit 0
fi

ROW="$((SLURM_ARRAY_TASK_ID + 2))"
LINE="$(awk -F, -v row="$ROW" 'NR==row {print; exit}' "$CASE_FILE")"
if [[ -z "$LINE" ]]; then
  echo "No case-table row for task $SLURM_ARRAY_TASK_ID in $CASE_FILE" >&2
  exit 2
fi
IFS=, read -r CASE_ID MODEL KN RT SEED FIGURE <<< "$LINE"
RT_TAG="${RT//./p}"
OUTPUT_DIR="$OUTPUT_ROOT/$ROUTE/runs"
mkdir -p "$OUTPUT_DIR"

if [[ "$MODEL" == "HS" ]]; then
  STEM="ThermalCavity_HS_DSMC_Kn${KN}_RT${RT_TAG}_quarter_seed${SEED}"
else
  STEM="ThermalCavity_${MODEL}_Kn${KN}_RT${RT_TAG}_quarter_seed${SEED}"
fi

echo "case=$CASE_ID route=$ROUTE figure=$FIGURE model=$MODEL Kn=$KN RT=$RT seed=$SEED"
echo "particles=$PARTICLES steps=5000000 sample_window=100000:5000000:2"
echo "host=$(hostname) job=$SLURM_JOB_ID task=$SLURM_ARRAY_TASK_ID"
nvidia-smi --query-gpu=name,memory.total,memory.free,compute_cap --format=csv

if [[ -s "$OUTPUT_DIR/${STEM}_metrics.json" && \
      -s "$OUTPUT_DIR/${STEM}_raw.npz" ]]; then
  echo "[SKIP] Complete outputs already exist for $STEM"
  exit 0
fi

COMMON=(
  --kn "$KN" --rt "$RT" --particles "$PARTICLES"
  --steps 5000000 --sample-start 100000 --sample-every 2
  --time-blocks 8 --seed "$SEED" --progress-every 10000
  --require-free-gb "$REQUIRE_FREE" --output "$OUTPUT_DIR"
)

if [[ "$MODEL" == "HS" ]]; then
  mamba run -n dsmc-gpu python solver/JFM_hs_dsmc_quarter.py \
    "${COMMON[@]}"
else
  mamba run -n dsmc-gpu python solver/JFM_bgk_shakhov_quarter.py \
    --mode "$MODEL" "${COMMON[@]}"
fi

test -s "$OUTPUT_DIR/${STEM}_metrics.json"
test -s "$OUTPUT_DIR/${STEM}_raw.npz"
echo "[OK] $CASE_ID complete"
