#!/usr/bin/env bash
#SBATCH --job-name=moh-mv17b-a2-prep
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=02:00:00

set -euo pipefail
: "${MV17B_PROJECT_ROOT:?}"
: "${MV17B_A2_PAYLOAD_ROOT:?}"
: "${MV17B_OUTPUT_ROOT:?}"
: "${MV17B_PYTHON:?}"

CAMPAIGN="${MV17B_OUTPUT_ROOT}/campaign"
BUILD="${CAMPAIGN}/build"
SOURCE="${BUILD}/DS2V_BIRD_MV17B.F90"
PATCHED_SOURCE="${BUILD}/DS2V_BIRD_MV17B_A2.F90"
PATCH_REPORT="${BUILD}/MV17B_A2_SOURCE_PATCH_REPORT.json"
EXE="${BUILD}/ds2v_bird_mv17b_a2"
LOG="${BUILD}/mv17b_a2_build.log"
test -s "${MV17B_OUTPUT_ROOT}/MV17B_A2_LOCKED_WINDOW_RECOVERY_LOCK.json"
test ! -e "${MV17B_OUTPUT_ROOT}/analysis"
test -s "${SOURCE}"
test ! -e "${PATCHED_SOURCE}"
test ! -e "${EXE}"
exec > >(tee "${CAMPAIGN}/logs/prepare_a2_${SLURM_JOB_ID}.out") 2>&1

cd "${MV17B_PROJECT_ROOT}"
export PATH="$(dirname "${MV17B_PYTHON}"):${PATH}"
export PYTHONPATH="${MV17B_PROJECT_ROOT}:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export PYTHONUNBUFFERED=1

"${MV17B_PYTHON}" -m vgdsmc.mohammadzadeh_mv17b_a2_locked_window_recovery patch-source \
  --source "${SOURCE}" \
  --output "${PATCHED_SOURCE}" \
  --report "${PATCH_REPORT}"
gfortran \
  -O2 -g -std=legacy -fdec -fno-automatic \
  -fallow-argument-mismatch -ffree-line-length-none -fno-range-check \
  -fbacktrace -fcheck=bounds \
  "${PATCHED_SOURCE}" -o "${EXE}" > "${LOG}" 2>&1
test -x "${EXE}"
test "$(grep -c 'MV17B_A2_LOCKED_WINDOW_EXTENSION_BEGIN' "${PATCHED_SOURCE}")" -eq 1
test "$(grep -c 'MV17B_A2_LOCKED_WINDOW_EXTENSION_END' "${PATCHED_SOURCE}")" -eq 1
test "$(grep -cF 'DO WHILE ((TIME < TLIM) .OR. (NOUT < 116))' "${PATCHED_SOURCE}")" -eq 1
test "$(sed -n '6p' "${CAMPAIGN}/input/interactive_input.txt")" = "3"
{
  gfortran --version | head -n 2
  echo "flags=-O2 -g -std=legacy -fdec -fno-automatic -fallow-argument-mismatch -ffree-line-length-none -fno-range-check -fbacktrace -fcheck=bounds"
  echo "fresh_IRUN=3"
  echo "locked_last_NOUT=116"
  echo "source_change=termination_guard_only"
  echo "base_source_sha256=$(sha256sum "${SOURCE}" | awk '{print $1}')"
  echo "extended_source_sha256=$(sha256sum "${PATCHED_SOURCE}" | awk '{print $1}')"
  echo "extended_executable_sha256=$(sha256sum "${EXE}" | awk '{print $1}')"
} > "${BUILD}/MV17B_A2_BUILD_CONFIGURATION.txt"
touch "${CAMPAIGN}/MV17B_A2_PREPARE_PASS"
echo "MV17B_A2_PREPARE_PASS fresh_IRUN=3 termination_guard_only=YES"
