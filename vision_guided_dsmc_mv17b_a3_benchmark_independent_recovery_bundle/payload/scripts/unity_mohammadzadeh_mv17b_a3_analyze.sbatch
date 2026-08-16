#!/usr/bin/env bash
#SBATCH --job-name=moh-mv17b-a3-eval
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00

set -euo pipefail
: "${MV17B_PROJECT_ROOT:?}"
: "${MV17B_A3_PAYLOAD_ROOT:?}"
: "${MV17B_OUTPUT_ROOT:?}"
: "${MV17B_PYTHON:?}"
CAMPAIGN="${MV17B_OUTPUT_ROOT}/campaign"
exec > >(tee "${CAMPAIGN}/logs/analyze_a3_${SLURM_JOB_ID}.out") 2>&1
cd "${MV17B_PROJECT_ROOT}"
export PATH="$(dirname "${MV17B_PYTHON}"):${PATH}"
export PYTHONPATH="${MV17B_PROJECT_ROOT}:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export PYTHONUNBUFFERED=1
export MPLCONFIGDIR="${TMPDIR:-/tmp}/mv17b-a3-mpl-${SLURM_JOB_ID}"
mkdir -p "${MPLCONFIGDIR}"
test ! -e "${MV17B_OUTPUT_ROOT}/analysis"
"${MV17B_PYTHON}" -m vgdsmc.mohammadzadeh_mv17b_a3_benchmark_independent_recovery verify-ready \
  --output-root "${MV17B_OUTPUT_ROOT}"
"${MV17B_PYTHON}" -m vgdsmc.mohammadzadeh_mv17b_fresh_cylinder_confirmation analyze \
  --campaign-root "${CAMPAIGN}" \
  --output-root "${MV17B_OUTPUT_ROOT}"
test -s "${MV17B_OUTPUT_ROOT}/analysis/summary.json"
echo "MV17B_A3_ANALYSIS_PASS output=${MV17B_OUTPUT_ROOT}"
