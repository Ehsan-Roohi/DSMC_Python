#!/usr/bin/env bash
#SBATCH --job-name=moh-mv15b
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00

set -euo pipefail
: "${MV15B_PROJECT_ROOT:?}"
: "${MV15B_PAYLOAD_ROOT:?}"
: "${MV15B_OUTPUT_ROOT:?}"
: "${MV15B_MV9_OUTPUT_ROOT:?}"
: "${MV15B_MV15A_OUTPUT_ROOT:?}"
: "${MV15B_PYTHON:?}"
test -x "${MV15B_PYTHON}"

mkdir -p "${MV15B_PROJECT_ROOT}/logs"
exec > >(tee "${MV15B_PROJECT_ROOT}/logs/moh_mv15b_budget_${SLURM_JOB_ID}.out") 2>&1
cd "${MV15B_PROJECT_ROOT}"
export PATH="$(dirname "${MV15B_PYTHON}"):${PATH}"
export PYTHONPATH="${MV15B_PROJECT_ROOT}:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MPLCONFIGDIR="${TMPDIR:-/tmp}/mv15b-matplotlib-${SLURM_JOB_ID}"
mkdir -p "${MPLCONFIGDIR}"

"${MV15B_PYTHON}" -c 'import importlib.util, numpy, torch; print("MV15B_RUNTIME_PASS", "numpy="+numpy.__version__, "torch="+torch.__version__, "scipy="+str(importlib.util.find_spec("scipy") is not None))'
"${MV15B_PYTHON}" "${MV15B_PAYLOAD_ROOT}/tests/test_mohammadzadeh_mv15b_data_consistent_budget.py"
"${MV15B_PYTHON}" -m vgdsmc.mohammadzadeh_mv15b_data_consistent_budget verify-lock
"${MV15B_PYTHON}" -m vgdsmc.mohammadzadeh_mv15b_data_consistent_budget predict \
  --mv9-output-root "${MV15B_MV9_OUTPUT_ROOT}" \
  --mv15a-output-root "${MV15B_MV15A_OUTPUT_ROOT}" \
  --output-root "${MV15B_OUTPUT_ROOT}" \
  --batch-size "${MV15B_BATCH_SIZE:-8}"
touch "${MV15B_OUTPUT_ROOT}/PREDICTION_LOCK_PASS"
echo "MV15B_PREDICTION_LOCK_PASS output=${MV15B_OUTPUT_ROOT}"
