#!/usr/bin/env bash
#SBATCH --job-name=moh-mv15a
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=08:00:00

set -euo pipefail
: "${MV15A_PROJECT_ROOT:?}"
: "${MV15A_PAYLOAD_ROOT:?}"
: "${MV15A_OUTPUT_ROOT:?}"
: "${MV15A_MV9_OUTPUT_ROOT:?}"
: "${MV15A_MV14_OUTPUT_ROOT:?}"
: "${MV15A_PYTHON:?}"
test -x "${MV15A_PYTHON}"

mkdir -p "${MV15A_PROJECT_ROOT}/logs"
exec > >(tee "${MV15A_PROJECT_ROOT}/logs/moh_mv15a_spectral_${SLURM_JOB_ID}.out") 2>&1
cd "${MV15A_PROJECT_ROOT}"
export PATH="$(dirname "${MV15A_PYTHON}"):${PATH}"
export PYTHONPATH="${MV15A_PROJECT_ROOT}:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

"${MV15A_PYTHON}" -c 'import importlib.util, numpy, torch; print("MV15A_RUNTIME_PASS", "numpy="+numpy.__version__, "torch="+torch.__version__, "scipy="+str(importlib.util.find_spec("scipy") is not None))'
"${MV15A_PYTHON}" "${MV15A_PAYLOAD_ROOT}/tests/test_mohammadzadeh_mv15a_spectral_information_audit.py"
"${MV15A_PYTHON}" -m vgdsmc.mohammadzadeh_mv15a_spectral_information_audit verify-lock
"${MV15A_PYTHON}" -m vgdsmc.mohammadzadeh_mv15a_spectral_information_audit predict \
  --mv9-output-root "${MV15A_MV9_OUTPUT_ROOT}" \
  --mv14-output-root "${MV15A_MV14_OUTPUT_ROOT}" \
  --output-root "${MV15A_OUTPUT_ROOT}" \
  --batch-size "${MV15A_BATCH_SIZE:-8}"
touch "${MV15A_OUTPUT_ROOT}/PREDICTION_LOCK_PASS"
echo "MV15A_PREDICTION_LOCK_PASS output=${MV15A_OUTPUT_ROOT}"
