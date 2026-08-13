#!/usr/bin/env bash
#SBATCH --job-name=moh-mv12-post
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=03:00:00

set -euo pipefail
: "${MV12_PROJECT_ROOT:?}"
: "${MV12_OUTPUT_ROOT:?}"
: "${MV12_PAYLOAD_ROOT:?}"
: "${MV12_PREDICT_JOB_ID:?}"
: "${MV12_VENV_DIR:?}"
test -x "${MV12_VENV_DIR}/bin/python"

test -f "${MV12_OUTPUT_ROOT}/PREDICTION_LOCK_PASS"
exec > >(tee "${MV12_PROJECT_ROOT}/logs/moh_mv12_post_${SLURM_JOB_ID}.out") 2>&1
cd "${MV12_PROJECT_ROOT}"
source "${MV12_VENV_DIR}/bin/activate"
export PYTHONPATH="${MV12_PROJECT_ROOT}:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

python -m vgdsmc.mohammadzadeh_mv12_sage_qy post \
  --output-root "${MV12_OUTPUT_ROOT}"
sacct -j "${MV12_PREDICT_JOB_ID},${SLURM_JOB_ID}" -X -P \
  --format=JobIDRaw,JobName,State,ExitCode,ElapsedRaw,CPUTimeRAW,MaxRSS,NodeList \
  > "${MV12_OUTPUT_ROOT}/slurm_accounting.psv" || true
python -m vgdsmc.mohammadzadeh_mv12_sage_qy package \
  --output-root "${MV12_OUTPUT_ROOT}" \
  --return-directory "${MV12_PROJECT_ROOT}"

echo "MV12_POST_PASS output=${MV12_OUTPUT_ROOT}"
