#!/usr/bin/env bash
set -euo pipefail

REPOSITORY_URL="https://github.com/Ehsan-Roohi/DSMC_Python.git"
BRANCH="${DSMC_BOOK_BRANCH:-agent/sparta-normal-shock-v2}"
BASE_DIR="${UNITY_SPARTA_BASE:-/project/pi_roohie_umass_edu/DSMC_CAVITY_BOOK}"
REPO_DIR="${BASE_DIR}/DSMC_Python_sparta_normal_shock_v2"
ROOT_DIR="${REPO_DIR}/sparta_normal_shock"
OPENMPI_MODULE="${UNITY_OPENMPI_MODULE:-openmpi/5.0.3}"
OPENMPI_PML="${UNITY_OPENMPI_PML:-ob1}"

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
  --export="ALL,SPARTA_CASE_ROOT=${ROOT_DIR},UNITY_OPENMPI_MODULE=${OPENMPI_MODULE},UNITY_OPENMPI_PML=${OPENMPI_PML}" \
  hpc/unity_sparta_shock_build.slurm)"
BUILD_JOB_ID="${BUILD_SUBMISSION%%;*}"
ARRAY_SUBMISSION="$(sbatch --parsable \
  --dependency="afterok:${BUILD_JOB_ID}" --kill-on-invalid-dep=yes \
  --export="ALL,SPARTA_CASE_ROOT=${ROOT_DIR},UNITY_OPENMPI_MODULE=${OPENMPI_MODULE},UNITY_OPENMPI_PML=${OPENMPI_PML}" \
  hpc/unity_sparta_shock_array.slurm)"
ARRAY_JOB_ID="${ARRAY_SUBMISSION%%;*}"
COLLECT_SUBMISSION="$(sbatch --parsable \
  --dependency="afterany:${ARRAY_JOB_ID}" \
  --export="ALL,SPARTA_CASE_ROOT=${ROOT_DIR},SPARTA_V2_ARRAY_JOB_ID=${ARRAY_JOB_ID},SPARTA_V2_BUILD_JOB_ID=${BUILD_JOB_ID}" \
  hpc/unity_sparta_shock_collect.slurm)"
COLLECT_JOB_ID="${COLLECT_SUBMISSION%%;*}"
STATE_FILE="${BASE_DIR}/LAST_SPARTA_NORMAL_SHOCK_V2_JOBS.env"
BUNDLE="${ROOT_DIR}/runs/SPARTA_NORMAL_SHOCK_V2_RESULTS_${ARRAY_JOB_ID}.tar.gz"
MANIFEST="${ROOT_DIR}/runs/SPARTA_NORMAL_SHOCK_V2_RESULTS_${ARRAY_JOB_ID}.manifest.txt"
{
  printf 'SHOCK_V2_BUILD_JOB_ID=%q\n' "${BUILD_JOB_ID}"
  printf 'SHOCK_V2_ARRAY_JOB_ID=%q\n' "${ARRAY_JOB_ID}"
  printf 'SHOCK_V2_COLLECT_JOB_ID=%q\n' "${COLLECT_JOB_ID}"
  printf 'SHOCK_V2_CAMPAIGN_ROOT=%q\n' "${ROOT_DIR}"
  printf 'SHOCK_V2_RETURN_BUNDLE=%q\n' "${BUNDLE}"
  printf 'SHOCK_V2_MANIFEST=%q\n' "${MANIFEST}"
} > "${STATE_FILE}"

echo
echo "Submitted the SPARTA steady normal-shock v2 campaign."
echo "Cases: Mach 2.5, 3, 5; three seeds each; x/lambda1=[-30,30]; nx=1200; upstream PPC=64."
echo "Sampling: warmup=80000; cumulative average=320000; stride=10."
echo "Build/smoke job: ${BUILD_JOB_ID}"
echo "Production array: ${ARRAY_JOB_ID}"
echo "Collector job: ${COLLECT_JOB_ID}"
echo "State file: ${STATE_FILE}"
echo
echo "Monitor:"
echo "  source '${STATE_FILE}'"
echo '  IDS="${SHOCK_V2_BUILD_JOB_ID},${SHOCK_V2_ARRAY_JOB_ID},${SHOCK_V2_COLLECT_JOB_ID}"'
echo '  squeue -j "${IDS}" 2>/dev/null || true'
echo '  sacct -X -j "${IDS}" --format=JobID%22,JobName%20,State,ExitCode,Elapsed,NodeList%22'
echo
echo "When the collector finishes, inspect validated_member_count in the manifest and upload:"
echo "  ${BUNDLE}"
