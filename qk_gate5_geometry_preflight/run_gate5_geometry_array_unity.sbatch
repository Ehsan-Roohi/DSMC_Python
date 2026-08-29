#!/usr/bin/env bash
#SBATCH --job-name=qk_g5_geo
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=16:00:00
#SBATCH --array=0-2%3
#SBATCH --output=/project/pi_roohie_umass_edu/Combustion/QK_GATE5_COUPLED/runs/geometry-array-%A_%a.out
#SBATCH --error=/project/pi_roohie_umass_edu/Combustion/QK_GATE5_COUPLED/runs/geometry-array-%A_%a.err
set -euo pipefail

BASE=/project/pi_roohie_umass_edu/Combustion/QK_GATE5_COUPLED
: "${GATE5_GEOMETRY_RELEASE_ID:?GATE5_GEOMETRY_RELEASE_ID is required}"
PAYLOAD="$BASE/releases/$GATE5_GEOMETRY_RELEASE_ID"
PASS_MARKER="$BASE/GEOMETRY_SMOKE_PASS_${GATE5_GEOMETRY_RELEASE_ID}.env"
test -s "$PASS_MARKER"
GROUP="$BASE/runs/geometry_preflight_${SLURM_ARRAY_JOB_ID}"
mkdir -p "$GROUP/tasks" "$GROUP/cases"
module load gcc/12.2.0 2>/dev/null || module load gcc 2>/dev/null || true
PYTHON_BIN=/work/pi_roohie_umass_edu/roohie_umass_edu/.conda/envs/dsmc-gpu/bin/python
if [[ ! -x "$PYTHON_BIN" ]]; then PYTHON_BIN="$HOME/.conda/envs/dsmc-gpu/bin/python"; fi
if [[ ! -x "$PYTHON_BIN" ]]; then echo "ERROR: dsmc-gpu Python missing" >&2; exit 3; fi
export PYTHON_BIN
"$PAYLOAD/run_gate5_geometry_case.sh" "$GROUP" "$SLURM_ARRAY_TASK_ID" 2>&1 | \
  tee "$GROUP/task_${SLURM_ARRAY_TASK_ID}.log"
echo "UNITY_QK_GATE5_GEOMETRY_CASE_PASS task=$SLURM_ARRAY_TASK_ID"
