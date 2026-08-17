#!/usr/bin/env bash
set -euo pipefail

REPOSITORY_URL="https://github.com/Ehsan-Roohi/DSMC_Python.git"
BRANCH="${DSMC_KN020_BRANCH:-agent/sparta-kn020-jfm}"
BASE_DIR="${UNITY_SPARTA_BASE:-/project/pi_roohie_umass_edu/DSMC_CAVITY_BOOK}"
REPO_DIR="${BASE_DIR}/DSMC_Python_sparta_kn020_jfm"
ROOT_DIR="${REPO_DIR}/sparta_cavity_mohammadzadeh"
CAMPAIGN="${JFM_CAMPAIGN:-/project/pi_roohie_umass_edu/CavityColdToHotIdentify/JFM_Five_Run_Campaign_20260802}"
RESULTS_BASE="${CAMPAIGN}/results/run7_dsmc_kn020_sparta"
OPENMPI_MODULE="${UNITY_OPENMPI_MODULE:-openmpi/5.0.3}"
MAX_PARALLEL="${DSMC_KN020_MAX_PARALLEL:-4}"

if ! [[ "${MAX_PARALLEL}" =~ ^[1-8]$ ]]; then
  echo "DSMC_KN020_MAX_PARALLEL must be an integer from 1 through 8." >&2
  exit 2
fi
for command_name in git sbatch; do
  if ! command -v "${command_name}" >/dev/null 2>&1; then
    echo "Required Unity command is unavailable: ${command_name}" >&2
    exit 2
  fi
done

mkdir -p "${BASE_DIR}" "${RESULTS_BASE}" "${CAMPAIGN}"
if [[ -d "${REPO_DIR}/.git" ]]; then
  if [[ -n "$(git -C "${REPO_DIR}" status --porcelain)" ]]; then
    echo "The dedicated Kn=0.20 checkout has changes; refusing to update: ${REPO_DIR}" >&2
    exit 3
  fi
  git -C "${REPO_DIR}" fetch origin \
    "+refs/heads/${BRANCH}:refs/remotes/origin/${BRANCH}"
  if git -C "${REPO_DIR}" show-ref --verify --quiet "refs/heads/${BRANCH}"; then
    git -C "${REPO_DIR}" switch "${BRANCH}"
  else
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
mkdir -p logs

BUILD_SUBMISSION="$(
  sbatch --parsable \
    --export="ALL,SPARTA_CASE_ROOT=${ROOT_DIR},UNITY_OPENMPI_MODULE=${OPENMPI_MODULE}" \
    hpc/unity_sparta_kn020_jfm_build.slurm
)"
BUILD_JOB_ID="${BUILD_SUBMISSION%%;*}"
ARRAY_SUBMISSION="$(
  sbatch --parsable \
    --dependency="afterok:${BUILD_JOB_ID}" \
    --kill-on-invalid-dep=yes \
    --array="0-7%${MAX_PARALLEL}" \
    --export="ALL,SPARTA_CASE_ROOT=${ROOT_DIR},SPARTA_RESULTS_BASE=${RESULTS_BASE},UNITY_OPENMPI_MODULE=${OPENMPI_MODULE}" \
    hpc/unity_sparta_kn020_jfm_array.slurm
)"
ARRAY_JOB_ID="${ARRAY_SUBMISSION%%;*}"
COLLECT_SUBMISSION="$(
  sbatch --parsable \
    --dependency="afterany:${ARRAY_JOB_ID}" \
    --export="ALL,SPARTA_CASE_ROOT=${ROOT_DIR},SPARTA_RESULTS_BASE=${RESULTS_BASE},SPARTA_BUNDLE_DIR=${CAMPAIGN},SPARTA_ARRAY_JOB_ID=${ARRAY_JOB_ID},SPARTA_BUILD_JOB_ID=${BUILD_JOB_ID}" \
    hpc/unity_sparta_kn020_jfm_collect.slurm
)"
COLLECT_JOB_ID="${COLLECT_SUBMISSION%%;*}"
STATE_FILE="${CAMPAIGN}/LAST_SPARTA_KN020_JFM_JOBS.env"
BUNDLE="${CAMPAIGN}/SPARTA_KN020_JFM_${ARRAY_JOB_ID}.tar.gz"

{
  printf 'KN020_BUILD_JOB_ID=%q\n' "${BUILD_JOB_ID}"
  printf 'KN020_ARRAY_JOB_ID=%q\n' "${ARRAY_JOB_ID}"
  printf 'KN020_COLLECT_JOB_ID=%q\n' "${COLLECT_JOB_ID}"
  printf 'KN020_CODE_ROOT=%q\n' "${ROOT_DIR}"
  printf 'KN020_RESULTS_ROOT=%q\n' "${RESULTS_BASE}/array_${ARRAY_JOB_ID}"
  printf 'KN020_RETURN_BUNDLE=%q\n' "${BUNDLE}"
} > "${STATE_FILE}"

echo
echo "Submitted matched SPARTA DSMC Kn=0.20 ensemble."
echo "Configuration: N160, 128 PPC, 40000 warmup, 8501 samples/cell, 8 seeds."
echo "Parallel seeds: ${MAX_PARALLEL}/8"
echo "Build/test job: ${BUILD_JOB_ID}"
echo "Array job:      ${ARRAY_JOB_ID}"
echo "Collector job:  ${COLLECT_JOB_ID}"
echo "State file:     ${STATE_FILE}"
echo
echo "Monitor:"
echo "  source '${STATE_FILE}'; squeue -j \"\${KN020_BUILD_JOB_ID},\${KN020_ARRAY_JOB_ID},\${KN020_COLLECT_JOB_ID}\""
echo "Final bundle: ${BUNDLE}"
