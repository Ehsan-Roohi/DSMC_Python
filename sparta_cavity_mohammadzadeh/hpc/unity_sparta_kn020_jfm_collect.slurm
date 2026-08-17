#!/usr/bin/env bash
#SBATCH --job-name=sp-k020-pack
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/sp-k020-pack-%j.out
#SBATCH --error=logs/sp-k020-pack-%j.err

set -euo pipefail

ROOT_DIR="${SPARTA_CASE_ROOT:-${SLURM_SUBMIT_DIR:-}}"
RESULTS_BASE="${SPARTA_RESULTS_BASE:?SPARTA_RESULTS_BASE was not exported}"
BUNDLE_DIR="${SPARTA_BUNDLE_DIR:?SPARTA_BUNDLE_DIR was not exported}"
ARRAY_JOB_ID="${SPARTA_ARRAY_JOB_ID:?SPARTA_ARRAY_JOB_ID was not exported}"
BUILD_JOB_ID="${SPARTA_BUILD_JOB_ID:?SPARTA_BUILD_JOB_ID was not exported}"
if [[ -z "${ROOT_DIR}" || ! -d "${ROOT_DIR}" ]]; then
  echo "SPARTA_CASE_ROOT/SLURM_SUBMIT_DIR does not identify the code directory." >&2
  exit 2
fi
ROOT_DIR="$(cd "${ROOT_DIR}" && pwd)"
ARRAY_ROOT="${RESULTS_BASE}/array_${ARRAY_JOB_ID}"
ANALYSIS_DIR="${ARRAY_ROOT}/analysis"
MANIFEST="${ARRAY_ROOT}/manifest.txt"
BUNDLE="${BUNDLE_DIR}/SPARTA_KN020_JFM_${ARRAY_JOB_ID}.tar.gz"
SEEDS=(104729 130363)

mkdir -p "${ARRAY_ROOT}" "${ANALYSIS_DIR}" "${ARRAY_ROOT}/scheduler_logs" "${BUNDLE_DIR}"

set +e
python3 "${ROOT_DIR}/scripts/postprocess_jfm_kn020.py" \
  "${ARRAY_ROOT}" --output "${ANALYSIS_DIR}" --seeds "${SEEDS[@]}"
POST_STATUS=$?
set -e

shopt -s nullglob
for path in \
  "${ROOT_DIR}"/logs/sp-k020-build-"${BUILD_JOB_ID}".* \
  "${ROOT_DIR}"/logs/sp-k020-jfm-"${ARRAY_JOB_ID}"_*.out \
  "${ROOT_DIR}"/logs/sp-k020-jfm-"${ARRAY_JOB_ID}"_*.err \
  "${ROOT_DIR}"/logs/sp-k020-pack-"${SLURM_JOB_ID}".*; do
  cp -p -- "${path}" "${ARRAY_ROOT}/scheduler_logs/"
done

mkdir -p "${ARRAY_ROOT}/provenance"
cp -p -- \
  "${ROOT_DIR}/scripts/generate_jfm_kn020_case.py" \
  "${ROOT_DIR}/scripts/postprocess_jfm_kn020.py" \
  "${ROOT_DIR}/hpc/unity_sparta_kn020_jfm_build.slurm" \
  "${ROOT_DIR}/hpc/unity_sparta_kn020_jfm_array.slurm" \
  "${ROOT_DIR}/hpc/unity_sparta_kn020_jfm_collect.slurm" \
  "${ARRAY_ROOT}/provenance/"

COMPLETE_COUNT=0
{
  printf 'SPARTA DSMC JFM matched high-Kn return bundle\n'
  printf 'kn=0.20\nkn_convention=gu_lambda_over_L\n'
  printf 'configuration=N160_ppc256_2seeds_warmup40000_samples_per_cell20000\n'
  printf 'build_job_id=%s\narray_job_id=%s\ncollector_job_id=%s\n' \
    "${BUILD_JOB_ID}" "${ARRAY_JOB_ID}" "${SLURM_JOB_ID}"
  printf 'code_commit=%s\n' "$(git -C "${ROOT_DIR}/.." rev-parse HEAD)"
  printf 'sparta_commit=%s\n' "$(git -C "${ROOT_DIR}/third_party/sparta" rev-parse HEAD)"
  printf 'created_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  for seed in "${SEEDS[@]}"; do
    if grep -q '^status=complete$' "${ARRAY_ROOT}/seed_${seed}/unity_run_metadata.txt" 2>/dev/null; then
      printf 'seed_%s=complete\n' "${seed}"
      COMPLETE_COUNT=$((COMPLETE_COUNT + 1))
    elif [[ -d "${ARRAY_ROOT}/seed_${seed}" ]]; then
      printf 'seed_%s=incomplete\n' "${seed}"
    else
      printf 'seed_%s=missing\n' "${seed}"
    fi
  done
  printf 'complete_seed_count=%s\npostprocess_exit=%s\n' "${COMPLETE_COUNT}" "${POST_STATUS}"
} > "${MANIFEST}"

tar \
  --exclude='seed_*/grid.checkpoint.*' \
  --exclude='seed_*/restart.kn020.1' \
  --exclude='seed_*/restart.kn020.2' \
  -czf "${BUNDLE}" -C "${ARRAY_ROOT}" .
tar -tzf "${BUNDLE}" >/dev/null
sha256sum "${BUNDLE}" > "${BUNDLE}.sha256"

echo "RETURN THIS FILE: ${BUNDLE}"
echo "SHA256 FILE: ${BUNDLE}.sha256"
echo "Complete seeds: ${COMPLETE_COUNT}/${#SEEDS[@]}"

if (( POST_STATUS != 0 || COMPLETE_COUNT != ${#SEEDS[@]} )); then
  echo "A diagnostic bundle was created, but the two-seed ensemble is incomplete." >&2
  exit 6
fi
