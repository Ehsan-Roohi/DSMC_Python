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
: "${MV15A_VENV_DIR:?}"
test -x "${MV15A_VENV_DIR}/bin/python"

mkdir -p "${MV15A_PROJECT_ROOT}/logs"
exec > >(tee "${MV15A_PROJECT_ROOT}/logs/moh_mv15a_spectral_${SLURM_JOB_ID}.out") 2>&1
cd "${MV15A_PROJECT_ROOT}"
source "${MV15A_VENV_DIR}/bin/activate"
export PYTHONPATH="${MV15A_PROJECT_ROOT}:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

python -c 'import scipy, torch; print("MV15A_RUNTIME_PASS", scipy.__version__, torch.__version__)'
python "${MV15A_PAYLOAD_ROOT}/tests/test_mohammadzadeh_mv15a_spectral_information_audit.py"
python -m vgdsmc.mohammadzadeh_mv15a_spectral_information_audit verify-lock
python -m vgdsmc.mohammadzadeh_mv15a_spectral_information_audit predict \
  --mv9-output-root "${MV15A_MV9_OUTPUT_ROOT}" \
  --mv14-output-root "${MV15A_MV14_OUTPUT_ROOT}" \
  --output-root "${MV15A_OUTPUT_ROOT}" \
  --batch-size "${MV15A_BATCH_SIZE:-8}"
touch "${MV15A_OUTPUT_ROOT}/PREDICTION_LOCK_PASS"
echo "MV15A_PREDICTION_LOCK_PASS output=${MV15A_OUTPUT_ROOT}"
