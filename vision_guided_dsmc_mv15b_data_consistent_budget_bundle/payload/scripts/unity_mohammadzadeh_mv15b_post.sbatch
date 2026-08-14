#!/usr/bin/env bash
#SBATCH --job-name=moh-mv15b-post
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=02:00:00

set -euo pipefail
: "${MV15B_PROJECT_ROOT:?}"
: "${MV15B_PAYLOAD_ROOT:?}"
: "${MV15B_OUTPUT_ROOT:?}"
: "${MV15B_PREDICT_JOB_ID:?}"
: "${MV15B_PYTHON:?}"
test -x "${MV15B_PYTHON}"
test -f "${MV15B_OUTPUT_ROOT}/PREDICTION_LOCK_PASS"

exec > >(tee "${MV15B_PROJECT_ROOT}/logs/moh_mv15b_post_${SLURM_JOB_ID}.out") 2>&1
cd "${MV15B_PROJECT_ROOT}"
export PATH="$(dirname "${MV15B_PYTHON}"):${PATH}"
export PYTHONPATH="${MV15B_PROJECT_ROOT}:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MPLCONFIGDIR="${TMPDIR:-/tmp}/mv15b-matplotlib-${SLURM_JOB_ID}"
mkdir -p "${MPLCONFIGDIR}"

"${MV15B_PYTHON}" -m vgdsmc.mohammadzadeh_mv15b_data_consistent_budget post \
  --output-root "${MV15B_OUTPUT_ROOT}"
sacct -j "${MV15B_PREDICT_JOB_ID},${SLURM_JOB_ID}" -X -P \
  --format=JobIDRaw,JobName,State,ExitCode,ElapsedRaw,CPUTimeRAW,MaxRSS,NodeList \
  > "${MV15B_OUTPUT_ROOT}/slurm_accounting.psv" || true
"${MV15B_PYTHON}" -m vgdsmc.mohammadzadeh_mv15b_data_consistent_budget package \
  --output-root "${MV15B_OUTPUT_ROOT}" \
  --return-directory "${MV15B_PROJECT_ROOT}"
echo "MV15B_POST_PASS output=${MV15B_OUTPUT_ROOT}"
