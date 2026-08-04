#!/usr/bin/env bash
#SBATCH --job-name=m3_qy_post
#SBATCH --account=pi_roohie_umass_edu
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

REPO_ROOT="${M3_REPO_ROOT:?M3_REPO_ROOT must be exported by the submit script}"
OUTPUT_ROOT="${M3_OUTPUT_ROOT:?M3_OUTPUT_ROOT must be exported by the submit script}"
ENSEMBLE_DIR="${OUTPUT_ROOT}/ensemble"
RETURN_DIR="${OUTPUT_ROOT}/return_bundle"

export MPLBACKEND=Agg
export PYTHONUNBUFFERED=1

cd "${REPO_ROOT}"
source "${REPO_ROOT}/.venv-m3/bin/activate"
python -m vgdsmc.mohammadzadeh_qy_ensemble \
  --input-root "${OUTPUT_ROOT}" \
  --output-dir "${ENSEMBLE_DIR}"
python -m vgdsmc.mohammadzadeh_qy_ensemble \
  --input-root "${OUTPUT_ROOT}" \
  --output-dir "${ENSEMBLE_DIR}" \
  --verify-only | tee "${ENSEMBLE_DIR}/verification.json"

mkdir -p "${RETURN_DIR}/ensemble"
cp "${ENSEMBLE_DIR}"/summary.json \
   "${ENSEMBLE_DIR}"/artifact_manifest.json \
   "${ENSEMBLE_DIR}"/ensemble_fields.npz \
   "${ENSEMBLE_DIR}"/line_profiles_m3.png \
   "${ENSEMBLE_DIR}"/contours_m3.png \
   "${ENSEMBLE_DIR}"/verification.json \
   "${RETURN_DIR}/ensemble/"
for seed in 91901 91902 91903 91904 91905 91906 91907 91908; do
  mkdir -p "${RETURN_DIR}/seed_${seed}"
  cp "${OUTPUT_ROOT}/seed_${seed}/summary.json" \
     "${OUTPUT_ROOT}/seed_${seed}/artifact_manifest.json" \
     "${OUTPUT_ROOT}/seed_${seed}/fields.npz" \
     "${OUTPUT_ROOT}/seed_${seed}/block_fields.npz" \
     "${RETURN_DIR}/seed_${seed}/"
done
tar -C "${RETURN_DIR}" -czf "${OUTPUT_ROOT}/M3_QY100_RETURN_BUNDLE.tar.gz" .
sha256sum "${OUTPUT_ROOT}/M3_QY100_RETURN_BUNDLE.tar.gz" \
  > "${OUTPUT_ROOT}/M3_QY100_RETURN_BUNDLE.sha256"
echo "Return these two files:"
echo "${OUTPUT_ROOT}/M3_QY100_RETURN_BUNDLE.tar.gz"
echo "${OUTPUT_ROOT}/M3_QY100_RETURN_BUNDLE.sha256"
