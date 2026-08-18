#!/usr/bin/env bash
#SBATCH --job-name=sparta-pack
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:20:00
#SBATCH --output=logs/sparta-pack-%j.out
#SBATCH --error=logs/sparta-pack-%j.err

set -euo pipefail

ROOT_DIR="${SPARTA_CASE_ROOT:-${SLURM_SUBMIT_DIR:-}}"
if [[ -z "${ROOT_DIR}" || ! -d "${ROOT_DIR}" ]]; then
  echo "SPARTA_CASE_ROOT/SLURM_SUBMIT_DIR does not identify the case directory." >&2
  exit 2
fi
ROOT_DIR="$(cd "${ROOT_DIR}" && pwd)"
ARRAY_JOB_ID="${SPARTA_ARRAY_JOB_ID:?SPARTA_ARRAY_JOB_ID was not exported by the bootstrap script}"
SEEDS=(20260803 20260819 20260831)
MANIFEST_REL="runs/SPARTA_KN01_RESULTS_${ARRAY_JOB_ID}.manifest.txt"
BUNDLE_REL="runs/SPARTA_KN01_RESULTS_${ARRAY_JOB_ID}.tar.gz"
MANIFEST="${ROOT_DIR}/${MANIFEST_REL}"
BUNDLE="${ROOT_DIR}/${BUNDLE_REL}"
INCLUDE_PATHS=("${MANIFEST_REL}")
COMPLETE_COUNT=0

mkdir -p "${ROOT_DIR}/logs" "${ROOT_DIR}/runs"
{
  printf 'SPARTA Kn=0.1 production return bundle\n'
  printf 'array_job_id=%s\n' "${ARRAY_JOB_ID}"
  printf 'collector_job_id=%s\n' "${SLURM_JOB_ID}"
  printf 'created_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
} > "${MANIFEST}"

for SEED in "${SEEDS[@]}"; do
  RUN_REL="runs/production_kn01_seed_${SEED}"
  RUN_DIR="${ROOT_DIR}/${RUN_REL}"
  if [[ -s "${RUN_DIR}/grid.final.00026000" && -s "${RUN_DIR}/log.cavity" ]]; then
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
  "${ROOT_DIR}"/logs/sparta-kn01-"${ARRAY_JOB_ID}"_*.out \
  "${ROOT_DIR}"/logs/sparta-kn01-"${ARRAY_JOB_ID}"_*.err; do
  INCLUDE_PATHS+=("${LOG_PATH#"${ROOT_DIR}/"}")
done

printf 'complete_seed_count=%s\n' "${COMPLETE_COUNT}" >> "${MANIFEST}"
tar -czf "${BUNDLE}" -C "${ROOT_DIR}" "${INCLUDE_PATHS[@]}"

echo "RETURN THIS FILE: ${BUNDLE}"
echo "Complete production seeds: ${COMPLETE_COUNT}/${#SEEDS[@]}"

if (( COMPLETE_COUNT != ${#SEEDS[@]} )); then
  echo "The bundle was created, but one or more production jobs did not finish." >&2
  exit 6
fi
