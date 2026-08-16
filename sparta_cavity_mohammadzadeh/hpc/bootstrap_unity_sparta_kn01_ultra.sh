#!/usr/bin/env bash
set -euo pipefail

REPOSITORY_URL="https://github.com/Ehsan-Roohi/DSMC_Python.git"
BRANCH="${DSMC_BOOK_BRANCH:-agent/sparta-kn01-ultra}"
BASE_DIR="${UNITY_SPARTA_BASE:-/project/pi_roohie_umass_edu/DSMC_CAVITY_BOOK}"
# Reuse the dedicated HQ checkout and its compiled SPARTA binary when present.
REPO_DIR="${BASE_DIR}/DSMC_Python_sparta_kn01_hq"
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
    echo "The dedicated SPARTA checkout has tracked changes; refusing to update it: ${REPO_DIR}" >&2
    exit 3
  fi
  # An existing --single-branch checkout may not have a fetch refspec for this
  # branch.  Fetch it explicitly into the corresponding remote-tracking ref.
  git -C "${REPO_DIR}" fetch origin \
    "+refs/heads/${BRANCH}:refs/remotes/origin/${BRANCH}"
  if git -C "${REPO_DIR}" show-ref --verify --quiet "refs/heads/${BRANCH}"; then
    git -C "${REPO_DIR}" switch "${BRANCH}"
  else
    # The remote branch may be outside this clone's single-branch fetch
    # configuration, so create directly from the fetched ref without --track.
    git -C "${REPO_DIR}" switch --create "${BRANCH}" "refs/remotes/origin/${BRANCH}"
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
    hpc/unity_sparta_hq_build.slurm
)"
BUILD_JOB_ID="${BUILD_SUBMISSION%%;*}"
ARRAY_SUBMISSION="$(
  sbatch --parsable \
    --dependency="afterok:${BUILD_JOB_ID}" \
    --kill-on-invalid-dep=yes \
    --export="ALL,SPARTA_CASE_ROOT=${ROOT_DIR},UNITY_OPENMPI_MODULE=${OPENMPI_MODULE}" \
    hpc/unity_sparta_kn01_ultra_array.slurm
)"
ARRAY_JOB_ID="${ARRAY_SUBMISSION%%;*}"
COLLECT_SUBMISSION="$(
  sbatch --parsable \
    --dependency="afterany:${ARRAY_JOB_ID}" \
    --export="ALL,SPARTA_CASE_ROOT=${ROOT_DIR},SPARTA_ARRAY_JOB_ID=${ARRAY_JOB_ID},SPARTA_BUILD_JOB_ID=${BUILD_JOB_ID}" \
    hpc/unity_sparta_kn01_ultra_collect.slurm
)"
COLLECT_JOB_ID="${COLLECT_SUBMISSION%%;*}"
STATE_FILE="${BASE_DIR}/LAST_SPARTA_KN01_ULTRA_JOBS.env"
BUNDLE="${ROOT_DIR}/runs/SPARTA_KN01_ULTRA_RESULTS_${ARRAY_JOB_ID}.tar.gz"

{
  printf 'ULTRA_BUILD_JOB_ID=%q\n' "${BUILD_JOB_ID}"
  printf 'ULTRA_ARRAY_JOB_ID=%q\n' "${ARRAY_JOB_ID}"
  printf 'ULTRA_COLLECT_JOB_ID=%q\n' "${COLLECT_JOB_ID}"
  printf 'ULTRA_CAMPAIGN_ROOT=%q\n' "${ROOT_DIR}"
  printf 'ULTRA_RETURN_BUNDLE=%q\n' "${BUNDLE}"
} > "${STATE_FILE}"

echo
echo "Submitted SPARTA Kn=0.1 Ultra ensemble."
echo "Configuration: 200x200, 128 PPC, 40000 warmup + 640000 averaging steps, 3 seeds."
echo "Running-average fields and restart checkpoints are written every 80000 steps."
echo "Build/test job: ${BUILD_JOB_ID}"
echo "Array job:      ${ARRAY_JOB_ID}"
echo "Collector job:  ${COLLECT_JOB_ID}"
echo "State file:     ${STATE_FILE}"
echo
echo "Monitor:"
echo "  source '${STATE_FILE}'"
echo '  squeue -j "${ULTRA_BUILD_JOB_ID},${ULTRA_ARRAY_JOB_ID},${ULTRA_COLLECT_JOB_ID}"'
echo '  sacct -X -j "${ULTRA_BUILD_JOB_ID},${ULTRA_ARRAY_JOB_ID},${ULTRA_COLLECT_JOB_ID}" --format=JobID%22,JobName%20,State,ExitCode,Elapsed,NodeList%22'
echo
echo "When the collector finishes, upload:"
echo "  ${BUNDLE}"
