#!/usr/bin/env bash
#SBATCH --job-name=moh-mv15c-ref
#SBATCH --partition=cpu
#SBATCH --array=0-7%4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=24:00:00

set -euo pipefail
: "${MV15C_PROJECT_ROOT:?}"
: "${MV15C_OUTPUT_ROOT:?}"
: "${MV15C_PYTHON:?}"
test -x "${MV15C_PYTHON}"
test -s "${MV15C_OUTPUT_ROOT}/source_lock_manifest.json"

mkdir -p "${MV15C_PROJECT_ROOT}/logs"
exec > >(tee "${MV15C_PROJECT_ROOT}/logs/moh_mv15c_ref_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.out") 2>&1
cd "${MV15C_PROJECT_ROOT}"
export PATH="$(dirname "${MV15C_PYTHON}"):${PATH}"
export PYTHONPATH="${MV15C_PROJECT_ROOT}:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export PYTHONUNBUFFERED=1

"${MV15C_PYTHON}" -m vgdsmc.mohammadzadeh_mv15c_fresh_b3_confirmation \
  run-reference \
  --task-index "${SLURM_ARRAY_TASK_ID}" \
  --output-root "${MV15C_OUTPUT_ROOT}"
echo "MV15C_REFERENCE_PASS task=${SLURM_ARRAY_TASK_ID} output=${MV15C_OUTPUT_ROOT}"
