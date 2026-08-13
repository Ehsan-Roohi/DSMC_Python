#!/usr/bin/env bash
#SBATCH --job-name=moh-mv12-sage
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=06:00:00

set -euo pipefail
: "${MV12_PROJECT_ROOT:?}"
: "${MV12_OUTPUT_ROOT:?}"
: "${MV12_MV10_OUTPUT_ROOT:?}"
: "${MV12_PAYLOAD_ROOT:?}"

mkdir -p "${MV12_PROJECT_ROOT}/logs"
exec > >(tee "${MV12_PROJECT_ROOT}/logs/moh_mv12_sage_${SLURM_JOB_ID}.out") 2>&1
cd "${MV12_PROJECT_ROOT}"
export PYTHONPATH="${MV12_PROJECT_ROOT}:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

python3 "${MV12_PAYLOAD_ROOT}/tests/test_mohammadzadeh_mv12_sage_qy.py"
python3 -m vgdsmc.mohammadzadeh_mv12_sage_qy verify-lock
python3 -m vgdsmc.mohammadzadeh_mv12_sage_qy predict \
  --mv10-output-root "${MV12_MV10_OUTPUT_ROOT}" \
  --output-root "${MV12_OUTPUT_ROOT}" \
  --batch-size "${MV12_BATCH_SIZE:-8}"

touch "${MV12_OUTPUT_ROOT}/PREDICTION_LOCK_PASS"
echo "MV12_PREDICTION_LOCK_PASS output=${MV12_OUTPUT_ROOT}"
