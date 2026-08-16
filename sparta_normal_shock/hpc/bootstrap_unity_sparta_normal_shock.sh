#!/usr/bin/env bash
set -euo pipefail

REPOSITORY_URL="https://github.com/Ehsan-Roohi/DSMC_Python.git"
BRANCH="${DSMC_BOOK_BRANCH:-agent/sparta-normal-shock-book}"
BASE_DIR="${UNITY_SPARTA_BASE:-/project/pi_roohie_umass_edu/DSMC_CAVITY_BOOK}"
REPO_DIR="${BASE_DIR}/DSMC_Python_sparta_normal_shock"
ROOT_DIR="${REPO_DIR}/sparta_normal_shock"
OPENMPI_MODULE="${UNITY_OPENMPI_MODULE:-openmpi/5.0.3}"

for command_name in git sbatch; do
  command -v "${command_name}" >/dev/null 2>&1 || {
    echo "Required Unity command is unavailable: ${command_name}" >&2
    exit 2
  }
done
mkdir -p "${BASE_DIR}"
if [[ -d "${REPO_DIR}/.git" ]]; then
  [[ -z "$(git -C "${REPO_DIR}" status --porcelain --untracked-files=no)" ]] || {
    echo "Dedicated checkout has tracked changes; refusing to update: ${REPO_DIR}" >&2
    exit 3
  }
  git -C "${REPO_DIR}" fetch origin \
    "+refs/heads/${BRANCH}:refs/remotes/origin/${BRANCH}"
  if git -C "${REPO_DIR}" show-ref --verify --quiet "refs/heads/${BRANCH}"; then
    git -C "${REPO_DIR}" switch "${BRANCH}"
  else
    git -C "${REPO_DIR}" switch --create "${BRANCH}" "refs/remotes/origin/${BRANCH}"
  fi
  git -C "${REPO_DIR}" pull --ff-only origin "${BRANCH}"
elif [[ -e "${REPO_DIR}" ]]; then
  echo "Path exists but is not a Git checkout: ${REPO_DIR}" >&2
  exit 3
else
  git clone --branch "${BRANCH}" --single-branch "${REPOSITORY_URL}" "${REPO_DIR}"
fi

cd "${ROOT_DIR}"
mkdir -p logs runs
BUILD_SUBMISSION="$(sbatch --parsable \
  --export="ALL,SPARTA_CASE_ROOT=${ROOT_DIR},UNITY_OPENMPI_MODULE=${OPENMPI_MODULE}" \
  hpc/unity_sparta_shock_build.slurm)"
BUILD_JOB_ID="${BUILD_SUBMISSION%%;*}"
ARRAY_SUBMISSION="$(sbatch --parsable \
  --dependency="afterok:${BUILD_JOB_ID}" --kill-on-invalid-dep=yes \
  --export="ALL,SPARTA_CASE_ROOT=${ROOT_DIR},UNITY_OPENMPI_MODULE=${OPENMPI_MODULE}" \
  hpc/unity_sparta_shock_array.slurm)"
ARRAY_JOB_ID="${ARRAY_SUBMISSION%%;*}"
COLLECT_SUBMISSION="$(sbatch --parsable \
  --dependency="afterany:${ARRAY_JOB_ID}" \
  --export="ALL,SPARTA_CASE_ROOT=${ROOT_DIR},SPARTA_ARRAY_JOB_ID=${ARRAY_JOB_ID},SPARTA_BUILD_JOB_ID=${BUILD_JOB_ID}" \
  hpc/unity_sparta_shock_collect.slurm)"
COLLECT_JOB_ID="${COLLECT_SUBMISSION%%;*}"
STATE_FILE="${BASE_DIR}/LAST_SPARTA_NORMAL_SHOCK_JOBS.env"
BUNDLE="${ROOT_DIR}/runs/SPARTA_NORMAL_SHOCK_RESULTS_${ARRAY_JOB_ID}.tar.gz"
{
  printf 'SHOCK_BUILD_JOB_ID=%q\n' "${BUILD_JOB_ID}"
  printf 'SHOCK_ARRAY_JOB_ID=%q\n' "${ARRAY_JOB_ID}"
  printf 'SHOCK_COLLECT_JOB_ID=%q\n' "${COLLECT_JOB_ID}"
  printf 'SHOCK_CAMPAIGN_ROOT=%q\n' "${ROOT_DIR}"
  printf 'SHOCK_RETURN_BUNDLE=%q\n' "${BUNDLE}"
} > "${STATE_FILE}"

echo
echo "Submitted the SPARTA steady normal-shock campaign."
echo "Cases: Mach 2.5, 3, 5; three seeds each; nx=600; upstream PPC=64."
echo "Build/smoke job: ${BUILD_JOB_ID}"
echo "Production array: ${ARRAY_JOB_ID}"
echo "Collector job: ${COLLECT_JOB_ID}"
echo "State file: ${STATE_FILE}"
echo
echo "Monitor:"
echo "  source '${STATE_FILE}'"
echo '  squeue -j "${SHOCK_BUILD_JOB_ID},${SHOCK_ARRAY_JOB_ID},${SHOCK_COLLECT_JOB_ID}"'
echo '  sacct -X -j "${SHOCK_BUILD_JOB_ID},${SHOCK_ARRAY_JOB_ID},${SHOCK_COLLECT_JOB_ID}" --format=JobID%22,JobName%20,State,ExitCode,Elapsed,NodeList%22'
echo
echo "When the collector completes, upload:"
echo "  ${BUNDLE}"

