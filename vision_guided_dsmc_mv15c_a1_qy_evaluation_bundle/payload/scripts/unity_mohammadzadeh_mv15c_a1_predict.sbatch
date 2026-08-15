#!/usr/bin/env bash
#SBATCH --job-name=moh-mv15c-a1
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00

set -euo pipefail
: "${MV15C_A1_PROJECT_ROOT:?}"
: "${MV15C_A1_PAYLOAD_ROOT:?}"
: "${MV15C_A1_OUTPUT_ROOT:?}"
: "${MV15C_A1_PYTHON:?}"
test -x "${MV15C_A1_PYTHON}"
test -s "${MV15C_A1_OUTPUT_ROOT}/mv15c_a1_amendment_lock_manifest.json"

exec > >(tee "${MV15C_A1_PROJECT_ROOT}/logs/moh_mv15c_a1_predict_${SLURM_JOB_ID}.out") 2>&1
cd "${MV15C_A1_PROJECT_ROOT}"
export PATH="$(dirname "${MV15C_A1_PYTHON}"):${PATH}"
export PYTHONPATH="${MV15C_A1_PROJECT_ROOT}:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export PYTHONUNBUFFERED=1

"${MV15C_A1_PYTHON}" -c 'import numpy, torch; print("MV15C_A1_RUNTIME_PASS", "numpy="+numpy.__version__, "torch="+torch.__version__)'
"${MV15C_A1_PYTHON}" "${MV15C_A1_PAYLOAD_ROOT}/tests/test_mohammadzadeh_mv15c_a1_qy_evaluation.py"
"${MV15C_A1_PYTHON}" -m vgdsmc.mohammadzadeh_mv15c_a1_qy_evaluation predict \
  --output-root "${MV15C_A1_OUTPUT_ROOT}" \
  --batch-size "${MV15C_A1_BATCH_SIZE:-8}"
test -s "${MV15C_A1_OUTPUT_ROOT}/mv15c_a1_prediction_attestation_manifest.json"
echo "MV15C_A1_PREDICTION_PASS output=${MV15C_A1_OUTPUT_ROOT}"

