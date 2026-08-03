#!/usr/bin/env bash

set -euo pipefail

REPOSITORY="https://github.com/Ehsan-Roohi/DSMC_Python.git"
BRANCH="${DSMC_CAVITY_BRANCH:-agent/validated-dsmc-cavity}"
WORK_ROOT="${DSMC_CAVITY_WORK_ROOT:-/project/pi_roohie_umass_edu/DSMC_CAVITY_BOOK}"
REPO_DIR="${WORK_ROOT}/DSMC_Python"
CONFIG_PATH="${DSMC_CAVITY_CONFIG:-configs/unity_kn01_model_matrix.toml}"
OUTPUT_ROOT="${DSMC_CAVITY_OUTPUT:-results/unity_kn01_model_matrix}"

mkdir -p "${WORK_ROOT}"
if [[ -d "${REPO_DIR}/.git" ]]; then
  if [[ -n "$(git -C "${REPO_DIR}" status --porcelain)" ]]; then
    echo "Existing checkout has local changes; refusing to overwrite: ${REPO_DIR}" >&2
    exit 2
  fi
  git -C "${REPO_DIR}" fetch --depth 1 origin "${BRANCH}"
  git -C "${REPO_DIR}" checkout -B "${BRANCH}" FETCH_HEAD
else
  git clone --depth 1 --single-branch --branch "${BRANCH}" \
    "${REPOSITORY}" "${REPO_DIR}"
fi

cd "${REPO_DIR}/validated_cavity_dsmc"
mkdir -p logs "${OUTPUT_ROOT}"
ARRAY_SUBMISSION="$(sbatch --parsable hpc/unity_gpu_model_array.slurm "${CONFIG_PATH}" "${OUTPUT_ROOT}")"
ARRAY_JOB="${ARRAY_SUBMISSION%%;*}"
COMPARE_SUBMISSION="$(sbatch --parsable --dependency="afterok:${ARRAY_JOB}" hpc/unity_compare_results.slurm "${OUTPUT_ROOT}")"
COMPARE_JOB="${COMPARE_SUBMISSION%%;*}"

echo "GPU array job: ${ARRAY_JOB} (NTC-PreScan, SBT, GBT, SSBT, SGBT)"
echo "Dependent comparison job: ${COMPARE_JOB}"
echo "Monitor: squeue -j ${ARRAY_JOB},${COMPARE_JOB}"
echo "Final table: ${REPO_DIR}/validated_cavity_dsmc/${OUTPUT_ROOT}/comparison.csv"
echo "Final plot: ${REPO_DIR}/validated_cavity_dsmc/${OUTPUT_ROOT}/comparison.png"
