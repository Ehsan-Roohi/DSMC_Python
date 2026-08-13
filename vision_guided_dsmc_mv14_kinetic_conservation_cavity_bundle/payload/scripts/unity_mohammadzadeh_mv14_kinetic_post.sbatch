#!/usr/bin/env bash
#SBATCH --job-name=moh-mv14-post
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=04:00:00

set -euo pipefail
: "${MV14_PROJECT_ROOT:?}"
: "${MV14_OUTPUT_ROOT:?}"
: "${MV14_PAYLOAD_ROOT:?}"
: "${MV14_PREDICT_JOB_ID:?}"
: "${MV14_VENV_DIR:?}"
test -x "${MV14_VENV_DIR}/bin/python"
test -f "${MV14_OUTPUT_ROOT}/PREDICTION_LOCK_PASS"

exec > >(tee "${MV14_PROJECT_ROOT}/logs/moh_mv14_post_${SLURM_JOB_ID}.out") 2>&1
cd "${MV14_PROJECT_ROOT}"
source "${MV14_VENV_DIR}/bin/activate"
export PYTHONPATH="${MV14_PROJECT_ROOT}:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

python -m vgdsmc.mohammadzadeh_mv14_kinetic_conservation_cavity post \
  --output-root "${MV14_OUTPUT_ROOT}"
sacct -j "${MV14_PREDICT_JOB_ID},${SLURM_JOB_ID}" -X -P \
  --format=JobIDRaw,JobName,State,ExitCode,ElapsedRaw,CPUTimeRAW,MaxRSS,NodeList \
  > "${MV14_OUTPUT_ROOT}/slurm_accounting.psv" || true
python -m vgdsmc.mohammadzadeh_mv14_kinetic_conservation_cavity package \
  --output-root "${MV14_OUTPUT_ROOT}" \
  --return-directory "${MV14_PROJECT_ROOT}"

echo "MV14_POST_PASS output=${MV14_OUTPUT_ROOT}"
