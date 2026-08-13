#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p "${ROOT_DIR}/logs" "${ROOT_DIR}/runs"

BUILD_SUBMISSION="$(sbatch --parsable \
  --export="ALL,SPARTA_TUTORIAL_ROOT=${ROOT_DIR}" \
  "${ROOT_DIR}/hpc/unity_sparta_build.slurm")"
BUILD_JOB_ID="${BUILD_SUBMISSION%%;*}"
ARRAY_SUBMISSION="$(sbatch --parsable \
  --dependency="afterok:${BUILD_JOB_ID}" \
  --kill-on-invalid-dep=yes \
  --export="ALL,SPARTA_TUTORIAL_ROOT=${ROOT_DIR}" \
  "${ROOT_DIR}/hpc/unity_sparta_array.slurm")"
ARRAY_JOB_ID="${ARRAY_SUBMISSION%%;*}"

STATE_FILE="${ROOT_DIR}/LAST_SPARTA_TUTORIAL_JOBS.env"
{
  printf 'BUILD_JOB_ID=%q\n' "${BUILD_JOB_ID}"
  printf 'ARRAY_JOB_ID=%q\n' "${ARRAY_JOB_ID}"
  printf 'SPARTA_TUTORIAL_ROOT=%q\n' "${ROOT_DIR}"
} > "${STATE_FILE}"

echo "Build job: ${BUILD_JOB_ID}"
echo "Array job: ${ARRAY_JOB_ID}"
echo "State file: ${STATE_FILE}"
echo "Monitor with:"
echo "  source '${STATE_FILE}'"
echo '  squeue -j "${BUILD_JOB_ID},${ARRAY_JOB_ID}"'

