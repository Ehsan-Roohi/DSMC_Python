#!/usr/bin/env bash
#SBATCH --job-name=moh-mv15a-post
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=04:00:00

set -euo pipefail
: "${MV15A_PROJECT_ROOT:?}"
: "${MV15A_PAYLOAD_ROOT:?}"
: "${MV15A_OUTPUT_ROOT:?}"
: "${MV15A_PREDICT_JOB_ID:?}"
: "${MV15A_PYTHON:?}"
test -x "${MV15A_PYTHON}"
test -f "${MV15A_OUTPUT_ROOT}/PREDICTION_LOCK_PASS"

exec > >(tee "${MV15A_PROJECT_ROOT}/logs/moh_mv15a_post_${SLURM_JOB_ID}.out") 2>&1
cd "${MV15A_PROJECT_ROOT}"
export PATH="$(dirname "${MV15A_PYTHON}"):${PATH}"
export PYTHONPATH="${MV15A_PROJECT_ROOT}:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

"${MV15A_PYTHON}" -m vgdsmc.mohammadzadeh_mv15a_spectral_information_audit post \
  --output-root "${MV15A_OUTPUT_ROOT}"
sacct -j "${MV15A_PREDICT_JOB_ID},${SLURM_JOB_ID}" -X -P \
  --format=JobIDRaw,JobName,State,ExitCode,ElapsedRaw,CPUTimeRAW,MaxRSS,NodeList \
  > "${MV15A_OUTPUT_ROOT}/slurm_accounting.psv" || true
"${MV15A_PYTHON}" -m vgdsmc.mohammadzadeh_mv15a_spectral_information_audit package \
  --output-root "${MV15A_OUTPUT_ROOT}" \
  --return-directory "${MV15A_PROJECT_ROOT}"
echo "MV15A_POST_PASS output=${MV15A_OUTPUT_ROOT}"
