#!/usr/bin/env bash
set -euo pipefail

REPOSITORY_URL="https://github.com/Ehsan-Roohi/DSMC_Python.git"
BRANCH="${DSMC_MAXWELL_BRANCH:-agent/maxwell-matched-antifourier}"
BASE_DIR="${UNITY_SPARTA_BASE:-/project/pi_roohie_umass_edu/DSMC_CAVITY_BOOK}"
REPO_DIR="${BASE_DIR}/DSMC_Python_sparta_maxwell_kngu005_020_jfm"
ROOT_DIR="${REPO_DIR}/sparta_cavity_mohammadzadeh"
CAMPAIGN="${JFM_MAXWELL_CAMPAIGN:-/project/pi_roohie_umass_edu/CavityColdToHotIdentify/JFM_Maxwell_Matched_Campaign_20260817}"
RESULTS_BASE="${CAMPAIGN}/results"
OPENMPI_MODULE="${UNITY_OPENMPI_MODULE:-openmpi/5.0.3}"
MAX_PARALLEL="${DSMC_MAXWELL_MAX_PARALLEL:-2}"
SEED="${MAXWELL_SINGLE_SEED:-104729}"

if ! [[ "${MAX_PARALLEL}" =~ ^[1-2]$ ]]; then
  echo "DSMC_MAXWELL_MAX_PARALLEL must be 1 or 2." >&2
  exit 2
fi
if ! [[ "${SEED}" =~ ^[1-9][0-9]*$ ]]; then
  echo "MAXWELL_SINGLE_SEED must be a positive integer." >&2
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
  if [[ -n "$(git -C "${REPO_DIR}" status --porcelain --untracked-files=no)" ]]; then
    echo "The dedicated Maxwell-VSS checkout has tracked changes: ${REPO_DIR}" >&2
    exit 3
  fi
  git -C "${REPO_DIR}" fetch origin \
    "+refs/heads/${BRANCH}:refs/remotes/origin/${BRANCH}"
  if git -C "${REPO_DIR}" show-ref --verify --quiet "refs/heads/${BRANCH}"; then
    git -C "${REPO_DIR}" switch "${BRANCH}"
    git -C "${REPO_DIR}" pull --ff-only origin "${BRANCH}"
  else
    git -C "${REPO_DIR}" switch --create "${BRANCH}" "refs/remotes/origin/${BRANCH}"
  fi
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
    hpc/unity_sparta_maxwell_kngu005_020_jfm_build.slurm
)"
BUILD_JOB_ID="${BUILD_SUBMISSION%%;*}"
RUN_SUBMISSION="$(
  sbatch --parsable \
    --dependency="afterok:${BUILD_JOB_ID}" \
    --kill-on-invalid-dep=yes \
    --array="0-1%${MAX_PARALLEL}" \
    --export="ALL,SPARTA_CASE_ROOT=${ROOT_DIR},SPARTA_RESULTS_BASE=${RESULTS_BASE},UNITY_OPENMPI_MODULE=${OPENMPI_MODULE},MAXWELL_SINGLE_SEED=${SEED}" \
    hpc/unity_sparta_maxwell_kngu005_020_jfm_single.slurm
)"
RUN_JOB_ID="${RUN_SUBMISSION%%;*}"
COLLECT_SUBMISSION="$(
  sbatch --parsable \
    --dependency="afterany:${RUN_JOB_ID}" \
    --export="ALL,SPARTA_CASE_ROOT=${ROOT_DIR},SPARTA_RESULTS_BASE=${RESULTS_BASE},SPARTA_BUNDLE_DIR=${CAMPAIGN},SPARTA_RUN_JOB_ID=${RUN_JOB_ID},SPARTA_BUILD_JOB_ID=${BUILD_JOB_ID},MAXWELL_SINGLE_SEED=${SEED}" \
    hpc/unity_sparta_maxwell_kngu005_020_jfm_collect.slurm
)"
COLLECT_JOB_ID="${COLLECT_SUBMISSION%%;*}"
STATE_FILE="${CAMPAIGN}/LAST_SPARTA_MAXWELL_KNGU005_020_JFM_JOBS.env"
BUNDLE="${CAMPAIGN}/SPARTA_MAXWELL_KNGU005_020_JFM_${RUN_JOB_ID}_SEED${SEED}_TO_ANALYZE.zip"

{
  printf 'MAXWELL_BUILD_JOB_ID=%q\n' "${BUILD_JOB_ID}"
  printf 'MAXWELL_RUN_JOB_ID=%q\n' "${RUN_JOB_ID}"
  printf 'MAXWELL_COLLECT_JOB_ID=%q\n' "${COLLECT_JOB_ID}"
  printf 'MAXWELL_CODE_ROOT=%q\n' "${ROOT_DIR}"
  printf 'MAXWELL_RESULTS_ROOT=%q\n' "${RESULTS_BASE}/campaign_${RUN_JOB_ID}"
  printf 'MAXWELL_RETURN_BUNDLE=%q\n' "${BUNDLE}"
} > "${STATE_FILE}"

echo
echo "Submitted matched Maxwell-VSS SPARTA DSMC cases at Kn_Gu=0.05 and 0.20."
echo "Collision parameters: omega=1, alpha=2.140."
echo "Configuration per case: N160, 256 PPC, 40000 warmup, 20000 samples/cell, one seed (${SEED})."
echo "Production dump: 15 fields (q/T, direct momentum flux, diagnostic instantaneous-COM B1)."
echo "Parallel cases: ${MAX_PARALLEL}/2"
echo "Build/test job: ${BUILD_JOB_ID}"
echo "Run array job:  ${RUN_JOB_ID}"
echo "Collector job:  ${COLLECT_JOB_ID}"
echo "State file:     ${STATE_FILE}"
echo
echo "Monitor:"
echo "  source '${STATE_FILE}'; squeue -j \"\${MAXWELL_BUILD_JOB_ID},\${MAXWELL_RUN_JOB_ID},\${MAXWELL_COLLECT_JOB_ID}\""
echo "Final ZIP: ${BUNDLE}"
