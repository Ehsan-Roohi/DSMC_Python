#!/usr/bin/env bash
#SBATCH --job-name=moh-mv14-kin
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=08:00:00

set -euo pipefail
: "${MV14_PROJECT_ROOT:?}"
: "${MV14_PAYLOAD_ROOT:?}"
: "${MV14_OUTPUT_ROOT:?}"
: "${MV14_MV9_OUTPUT_ROOT:?}"
: "${MV14_MV12_OUTPUT_ROOT:?}"
: "${MV14_VENV_DIR:?}"
test -x "${MV14_VENV_DIR}/bin/python"

mkdir -p "${MV14_PROJECT_ROOT}/logs"
exec > >(tee "${MV14_PROJECT_ROOT}/logs/moh_mv14_kinetic_${SLURM_JOB_ID}.out") 2>&1
cd "${MV14_PROJECT_ROOT}"
source "${MV14_VENV_DIR}/bin/activate"
export PYTHONPATH="${MV14_PROJECT_ROOT}:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

python -c 'import torch; print("MV14_JOB_TORCH_PASS", torch.__version__)'
python "${MV14_PAYLOAD_ROOT}/tests/test_mohammadzadeh_mv14_kinetic_conservation_cavity.py"
python -m vgdsmc.mohammadzadeh_mv14_kinetic_conservation_cavity verify-lock
python -m vgdsmc.mohammadzadeh_mv14_kinetic_conservation_cavity predict \
  --mv9-output-root "${MV14_MV9_OUTPUT_ROOT}" \
  --mv12-output-root "${MV14_MV12_OUTPUT_ROOT}" \
  --output-root "${MV14_OUTPUT_ROOT}" \
  --batch-size "${MV14_BATCH_SIZE:-8}"

touch "${MV14_OUTPUT_ROOT}/PREDICTION_LOCK_PASS"
echo "MV14_PREDICTION_LOCK_PASS output=${MV14_OUTPUT_ROOT}"
