#!/usr/bin/env bash
#SBATCH --job-name=sparta-hq-pack
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:20:00
#SBATCH --output=logs/sparta-hq-pack-%j.out
#SBATCH --error=logs/sparta-hq-pack-%j.err

set -euo pipefail

ROOT_DIR="${SPARTA_CASE_ROOT:-${SLURM_SUBMIT_DIR:-}}"
if [[ -z "${ROOT_DIR}" || ! -d "${ROOT_DIR}" ]]; then
  echo "SPARTA_CASE_ROOT/SLURM_SUBMIT_DIR does not identify the case directory." >&2
  exit 2
fi
ROOT_DIR="$(cd "${ROOT_DIR}" && pwd)"
ARRAY_JOB_ID="${SPARTA_ARRAY_JOB_ID:?SPARTA_ARRAY_JOB_ID was not exported by the bootstrap script}"
BUILD_JOB_ID="${SPARTA_BUILD_JOB_ID:?SPARTA_BUILD_JOB_ID was not exported by the bootstrap script}"
SEEDS=(20260803 20260819 20260831)
FINAL_DUMP_NAME="grid.final.00160000"
MANIFEST_REL="runs/SPARTA_KN01_HQ_RESULTS_${ARRAY_JOB_ID}.manifest.txt"
BUNDLE_REL="runs/SPARTA_KN01_HQ_RESULTS_${ARRAY_JOB_ID}.tar.gz"
MANIFEST="${ROOT_DIR}/${MANIFEST_REL}"
BUNDLE="${ROOT_DIR}/${BUNDLE_REL}"
INCLUDE_PATHS=("${MANIFEST_REL}")
COMPLETE_COUNT=0

mkdir -p "${ROOT_DIR}/logs" "${ROOT_DIR}/runs"
{
  printf 'SPARTA Kn=0.1 high-statistics return bundle\n'
  printf 'configuration=grid200x200_ppc128_warmup40000_average160000_stride10\n'
  printf 'build_job_id=%s\n' "${BUILD_JOB_ID}"
  printf 'array_job_id=%s\n' "${ARRAY_JOB_ID}"
  printf 'collector_job_id=%s\n' "${SLURM_JOB_ID}"
  printf 'created_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
} > "${MANIFEST}"

for SEED in "${SEEDS[@]}"; do
  RUN_REL="runs/hq_${ARRAY_JOB_ID}_kn01_seed_${SEED}"
  RUN_DIR="${ROOT_DIR}/${RUN_REL}"
  if [[ -s "${RUN_DIR}/${FINAL_DUMP_NAME}" && -s "${RUN_DIR}/log.cavity" ]] && \
     grep -q '^status=complete$' "${RUN_DIR}/unity_run_metadata.txt" 2>/dev/null; then
    printf 'seed_%s=complete\n' "${SEED}" >> "${MANIFEST}"
    INCLUDE_PATHS+=("${RUN_REL}")
    COMPLETE_COUNT=$((COMPLETE_COUNT + 1))
  elif [[ -d "${RUN_DIR}" ]]; then
    printf 'seed_%s=incomplete\n' "${SEED}" >> "${MANIFEST}"
    INCLUDE_PATHS+=("${RUN_REL}")
  else
    printf 'seed_%s=missing\n' "${SEED}" >> "${MANIFEST}"
  fi
done

shopt -s nullglob
for LOG_PATH in \
  "${ROOT_DIR}"/logs/sparta-hq-build-"${BUILD_JOB_ID}".* \
  "${ROOT_DIR}"/logs/sparta-kn01-hq-"${ARRAY_JOB_ID}"_*.out \
  "${ROOT_DIR}"/logs/sparta-kn01-hq-"${ARRAY_JOB_ID}"_*.err; do
  INCLUDE_PATHS+=("${LOG_PATH#"${ROOT_DIR}/"}")
done

printf 'complete_seed_count=%s\n' "${COMPLETE_COUNT}" >> "${MANIFEST}"
tar -czf "${BUNDLE}" -C "${ROOT_DIR}" "${INCLUDE_PATHS[@]}"
tar -tzf "${BUNDLE}" >/dev/null
sha256sum "${BUNDLE}" > "${BUNDLE}.sha256"

echo "RETURN THIS FILE: ${BUNDLE}"
echo "SHA256 FILE: ${BUNDLE}.sha256"
echo "Complete HQ seeds: ${COMPLETE_COUNT}/${#SEEDS[@]}"

if (( COMPLETE_COUNT != ${#SEEDS[@]} )); then
  echo "The bundle was created, but one or more HQ jobs did not finish." >&2
  exit 6
fi
