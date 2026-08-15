#!/usr/bin/env bash
#SBATCH --job-name=moh-mv16a-post
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00

set -euo pipefail
: "${MV16A_PROJECT_ROOT:?}"
: "${MV16A_PAYLOAD_ROOT:?}"
: "${MV16A_OUTPUT_ROOT:?}"
: "${MV16A_PYTHON:?}"
: "${MV16A_PREDICT_JOB_ID:?}"
test -x "${MV16A_PYTHON}"
test -s "${MV16A_OUTPUT_ROOT}/prediction_manifest.json"

exec > >(tee "${MV16A_PROJECT_ROOT}/logs/moh_mv16a_cylinder_post_${SLURM_JOB_ID}.out") 2>&1
cd "${MV16A_PROJECT_ROOT}"
export PATH="$(dirname "${MV16A_PYTHON}"):${PATH}"
export PYTHONPATH="${MV16A_PROJECT_ROOT}:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export PYTHONUNBUFFERED=1
export MPLCONFIGDIR="${TMPDIR:-/tmp}/mv16a-mpl-${SLURM_JOB_ID}"
mkdir -p "${MPLCONFIGDIR}"

"${MV16A_PYTHON}" -m vgdsmc.mohammadzadeh_mv16a_frozen_cylinder_transfer post \
  --output-root "${MV16A_OUTPUT_ROOT}"
sacct -j "${MV16A_PREDICT_JOB_ID},${SLURM_JOB_ID}" -X -P \
  --format=JobID,JobIDRaw,JobName,State,ExitCode,ElapsedRaw,CPUTimeRAW,MaxRSS,NodeList \
  > "${MV16A_OUTPUT_ROOT}/mv16a_slurm_accounting.psv" || true
"${MV16A_PYTHON}" -m vgdsmc.mohammadzadeh_mv16a_frozen_cylinder_transfer package \
  --output-root "${MV16A_OUTPUT_ROOT}" \
  --return-directory "${MV16A_PROJECT_ROOT}"
echo "MV16A_CYLINDER_POST_PASS output=${MV16A_OUTPUT_ROOT}"
