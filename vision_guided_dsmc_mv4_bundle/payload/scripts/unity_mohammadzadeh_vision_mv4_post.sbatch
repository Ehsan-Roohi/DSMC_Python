#!/usr/bin/env bash
#SBATCH --job-name=moh_mv4_post
#SBATCH --account=pi_roohie_umass_edu
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=20G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

REPO_ROOT="${MV4_REPO_ROOT:?exported by submit script}"
OUTPUT_ROOT="${MV4_OUTPUT_ROOT:?exported by submit script}"
VENV_DIR="${MV4_VENV_DIR:?exported by submit script}"

export MPLBACKEND=Agg
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-2}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-2}"
export PYTHONUNBUFFERED=1

cd "${REPO_ROOT}"
source "${VENV_DIR}/bin/activate"
python -m vgdsmc.mohammadzadeh_vision_mv4 --output-dir "${OUTPUT_ROOT}" --aggregate
python -m vgdsmc.mohammadzadeh_vision_mv4 --output-dir "${OUTPUT_ROOT}" --verify-only > "${OUTPUT_ROOT}/verification.json"
python -m vgdsmc.mohammadzadeh_vision_mv4 --output-dir "${OUTPUT_ROOT}" --package

echo "Return these files:"
echo "${OUTPUT_ROOT}/MOHAMMADZADEH_MV4_JCP_RETURN_BUNDLE.tar.gz"
echo "${OUTPUT_ROOT}/MOHAMMADZADEH_MV4_JCP_RETURN_BUNDLE.tar.gz.sha256"
echo "${OUTPUT_ROOT}/summary.json"
echo "${OUTPUT_ROOT}/verification.json"
