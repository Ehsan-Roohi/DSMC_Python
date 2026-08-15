#!/usr/bin/env bash
#SBATCH --job-name=moh-mv17b-eval
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00

set -euo pipefail
: "${MV17B_PROJECT_ROOT:?}"
: "${MV17B_PAYLOAD_ROOT:?}"
: "${MV17B_OUTPUT_ROOT:?}"
: "${MV17B_PYTHON:?}"
CAMPAIGN="${MV17B_OUTPUT_ROOT}/campaign"
exec > >(tee "${CAMPAIGN}/logs/analyze_${SLURM_JOB_ID}.out") 2>&1
cd "${MV17B_PROJECT_ROOT}"
export PATH="$(dirname "${MV17B_PYTHON}"):${PATH}"
export PYTHONPATH="${MV17B_PROJECT_ROOT}:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export PYTHONUNBUFFERED=1
export MPLCONFIGDIR="${TMPDIR:-/tmp}/mv17b-mpl-${SLURM_JOB_ID}"
mkdir -p "${MPLCONFIGDIR}"
test "$(find "${CAMPAIGN}/cases" -name RUN_STATUS.env -type f | wc -l)" -eq 12
for STATUS_FILE in "${CAMPAIGN}"/cases/*/results/RUN_STATUS.env; do
  grep -qx 'STATUS=LOCKED_WINDOW_COMPLETE' "${STATUS_FILE}"
done
"${MV17B_PYTHON}" -m vgdsmc.mohammadzadeh_mv17b_fresh_cylinder_confirmation analyze \
  --campaign-root "${CAMPAIGN}" \
  --output-root "${MV17B_OUTPUT_ROOT}"
test -s "${MV17B_OUTPUT_ROOT}/analysis/summary.json"
echo "MV17B_ANALYSIS_PASS output=${MV17B_OUTPUT_ROOT}"
