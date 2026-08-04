#!/usr/bin/env bash
set -euo pipefail

REPOSITORY_URL="https://github.com/Ehsan-Roohi/DSMC_Python.git"
BRANCH="${DSMC_BOOK_BRANCH:-agent/validated-dsmc-cavity}"
BASE_DIR="${UNITY_SPARTA_BASE:-/project/pi_roohie_umass_edu/DSMC_CAVITY_BOOK}"
REPO_DIR="${BASE_DIR}/DSMC_Python"
ROOT_DIR="${REPO_DIR}/sparta_cavity_mohammadzadeh"
OPENMPI_MODULE="${UNITY_OPENMPI_MODULE:-openmpi/5.0.3}"

for COMMAND_NAME in git sbatch; do
  if ! command -v "${COMMAND_NAME}" >/dev/null 2>&1; then
    echo "Required Unity command is unavailable: ${COMMAND_NAME}" >&2
    exit 2
  fi
done

mkdir -p "${BASE_DIR}"
if [[ -d "${REPO_DIR}/.git" ]]; then
  if [[ -n "$(git -C "${REPO_DIR}" status --porcelain --untracked-files=no)" ]]; then
    echo "The existing clone has tracked changes; refusing to update it: ${REPO_DIR}" >&2
    exit 3
  fi
  git -C "${REPO_DIR}" fetch origin "${BRANCH}"
  if git -C "${REPO_DIR}" show-ref --verify --quiet "refs/heads/${BRANCH}"; then
    git -C "${REPO_DIR}" switch "${BRANCH}"
  else
    git -C "${REPO_DIR}" switch --create "${BRANCH}" --track "origin/${BRANCH}"
  fi
  git -C "${REPO_DIR}" pull --ff-only origin "${BRANCH}"
elif [[ -e "${REPO_DIR}" ]]; then
  echo "Path exists but is not a Git clone: ${REPO_DIR}" >&2
  exit 3
else
  git clone --branch "${BRANCH}" --single-branch "${REPOSITORY_URL}" "${REPO_DIR}"
fi

cd "${ROOT_DIR}"
mkdir -p logs runs

BUILD_SUBMISSION="$(
  sbatch --parsable \
    --export="ALL,SPARTA_CASE_ROOT=${ROOT_DIR},UNITY_OPENMPI_MODULE=${OPENMPI_MODULE}" \
    hpc/unity_sparta_build.slurm
)"
BUILD_JOB_ID="${BUILD_SUBMISSION%%;*}"
ARRAY_SUBMISSION="$(
  sbatch --parsable \
    --dependency="afterok:${BUILD_JOB_ID}" \
    --kill-on-invalid-dep=yes \
    --export="ALL,SPARTA_CASE_ROOT=${ROOT_DIR},UNITY_OPENMPI_MODULE=${OPENMPI_MODULE}" \
    hpc/unity_sparta_kn01_array.slurm
)"
ARRAY_JOB_ID="${ARRAY_SUBMISSION%%;*}"
COLLECT_SUBMISSION="$(
  sbatch --parsable \
    --dependency="afterany:${ARRAY_JOB_ID}" \
    --export="ALL,SPARTA_CASE_ROOT=${ROOT_DIR},SPARTA_ARRAY_JOB_ID=${ARRAY_JOB_ID}" \
    hpc/unity_sparta_kn01_collect.slurm
)"
COLLECT_JOB_ID="${COLLECT_SUBMISSION%%;*}"
STATE_FILE="${BASE_DIR}/LAST_SPARTA_KN01_JOBS.env"
BUNDLE="${ROOT_DIR}/runs/SPARTA_KN01_RESULTS_${ARRAY_JOB_ID}.tar.gz"

{
  printf 'BUILD_JOB_ID=%q\n' "${BUILD_JOB_ID}"
  printf 'ARRAY_JOB_ID=%q\n' "${ARRAY_JOB_ID}"
  printf 'COLLECT_JOB_ID=%q\n' "${COLLECT_JOB_ID}"
  printf 'SPARTA_CASE_ROOT=%q\n' "${ROOT_DIR}"
  printf 'SPARTA_RETURN_BUNDLE=%q\n' "${BUNDLE}"
} > "${STATE_FILE}"

echo
echo "Submitted the SPARTA build and three independent Kn=0.1 production runs."
echo "Build job:     ${BUILD_JOB_ID}"
echo "Array job:     ${ARRAY_JOB_ID}"
echo "Collector job: ${COLLECT_JOB_ID} (runs after every array task exits)"
echo
echo "Monitor:"
echo "  source '${STATE_FILE}'"
echo '  squeue -j "${BUILD_JOB_ID},${ARRAY_JOB_ID},${COLLECT_JOB_ID}"'
echo '  sacct -j "${BUILD_JOB_ID},${ARRAY_JOB_ID},${COLLECT_JOB_ID}" --format=JobID,State,Elapsed,ExitCode'
echo
echo "Follow seed 20260803 after it starts:"
echo "  tail -f '${ROOT_DIR}/logs/sparta-kn01-${ARRAY_JOB_ID}_0.out'"
echo
echo "When the collector finishes, send this file back:"
echo "  ${BUNDLE}"
