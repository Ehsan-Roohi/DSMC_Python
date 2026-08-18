#!/bin/bash
#SBATCH --job-name=jfm-k001-hvy
#SBATCH --partition=gpu
#SBATCH --qos=long
#SBATCH --gpus=1
#SBATCH --constraint=2080ti
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=168:00:00
#SBATCH --array=0-8%9
#SBATCH --output=slurm-kn001-heavy-%A_%a.out
#SBATCH --error=slurm-kn001-heavy-%A_%a.err

set -euo pipefail

ROOT="${SLURM_SUBMIT_DIR:?Submit from the package root}"
cd "$ROOT"
CASE_FILE="$ROOT/cases/kn001_heavy.csv"
OUTPUT_ROOT="${JFM_OUTPUT_ROOT:-$ROOT/run_output}"
OUTPUT_DIR="$OUTPUT_ROOT/kn001_heavy"

test -r "$CASE_FILE"
mkdir -p "$OUTPUT_DIR"
test -w "$OUTPUT_DIR"

ROW="$((SLURM_ARRAY_TASK_ID + 2))"
LINE="$(awk -F, -v row="$ROW" 'NR==row {print; exit}' "$CASE_FILE")"
if [[ -z "$LINE" ]]; then
  echo "No case-table row for array task $SLURM_ARRAY_TASK_ID" >&2
  exit 2
fi
IFS=, read -r CASE_ID MODEL KN RT SEED GROUP <<< "$LINE"
RT_TAG="${RT//./p}"

if [[ "$MODEL" == "HS" ]]; then
  STEM="ThermalCavity_HS_DSMC_Kn${KN}_RT${RT_TAG}_quarter_seed${SEED}"
else
  STEM="ThermalCavity_${MODEL}_Kn${KN}_RT${RT_TAG}_quarter_seed${SEED}"
fi

echo "case=$CASE_ID model=$MODEL Kn=$KN RT=$RT seed=$SEED group=$GROUP"
echo "host=$(hostname) job=${SLURM_JOB_ID} task=${SLURM_ARRAY_TASK_ID}"
echo "output=$OUTPUT_DIR"
nvidia-smi -L

if [[ -s "$OUTPUT_DIR/${STEM}_metrics.json" &&
      -s "$OUTPUT_DIR/${STEM}_raw.npz" ]]; then
  echo "[SKIP] Complete outputs already exist for $STEM"
  exit 0
fi

mamba run -n dsmc-gpu python -c \
  'import cupy; p=cupy.cuda.runtime.getDeviceProperties(0); print("GPU:",p["name"]); print("free,total:",cupy.cuda.runtime.memGetInfo())'

COMMON=(
  --kn "$KN"
  --rt "$RT"
  --particles 22000000
  --steps 2000000
  --sample-start 400000
  --sample-every 2
  --time-blocks 4
  --seed "$SEED"
  --progress-every 10000
  --require-free-gb 8.5
  --output "$OUTPUT_DIR"
)

if [[ "$MODEL" == "HS" ]]; then
  mamba run -n dsmc-gpu python JFM_hs_dsmc_quarter_22m.py \
    "${COMMON[@]}"
else
  mamba run -n dsmc-gpu python JFM_bgk_shakhov_quarter_22m.py \
    --mode "$MODEL" "${COMMON[@]}"
fi

test -s "$OUTPUT_DIR/${STEM}_metrics.json"
test -s "$OUTPUT_DIR/${STEM}_raw.npz"
echo "[OK] $CASE_ID complete"
