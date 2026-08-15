#!/usr/bin/env bash
#SBATCH --job-name=moh-mv15c-a1-post
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=02:00:00

set -euo pipefail
: "${MV15C_A1_PROJECT_ROOT:?}"
: "${MV15C_A1_OUTPUT_ROOT:?}"
: "${MV15C_A1_PYTHON:?}"
: "${MV15C_A1_PREDICT_JOB_ID:?}"
: "${MV15C_A1_ORIGINAL_REFERENCE_JOB_ID:?}"
test -x "${MV15C_A1_PYTHON}"
test -s "${MV15C_A1_OUTPUT_ROOT}/mv15c_a1_prediction_attestation_manifest.json"

exec > >(tee "${MV15C_A1_PROJECT_ROOT}/logs/moh_mv15c_a1_post_${SLURM_JOB_ID}.out") 2>&1
cd "${MV15C_A1_PROJECT_ROOT}"
export PATH="$(dirname "${MV15C_A1_PYTHON}"):${PATH}"
export PYTHONPATH="${MV15C_A1_PROJECT_ROOT}:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MPLCONFIGDIR="${TMPDIR:-/tmp}/mv15c-a1-matplotlib-${SLURM_JOB_ID}"
mkdir -p "${MPLCONFIGDIR}"

"${MV15C_A1_PYTHON}" -m vgdsmc.mohammadzadeh_mv15c_a1_qy_evaluation post \
  --output-root "${MV15C_A1_OUTPUT_ROOT}"
sacct -j "${MV15C_A1_ORIGINAL_REFERENCE_JOB_ID},${MV15C_A1_PREDICT_JOB_ID},${SLURM_JOB_ID}" -X -P \
  --format=JobID,JobIDRaw,JobName,State,ExitCode,ElapsedRaw,CPUTimeRAW,MaxRSS,NodeList \
  > "${MV15C_A1_OUTPUT_ROOT}/mv15c_a1_slurm_accounting.psv" || true
"${MV15C_A1_PYTHON}" -m vgdsmc.mohammadzadeh_mv15c_a1_qy_evaluation package \
  --output-root "${MV15C_A1_OUTPUT_ROOT}" \
  --return-directory "${MV15C_A1_PROJECT_ROOT}"
echo "MV15C_A1_POST_PASS output=${MV15C_A1_OUTPUT_ROOT}"

