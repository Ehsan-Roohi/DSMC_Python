#!/usr/bin/env bash
#SBATCH --job-name=moh-mv17b-a3-prep
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=02:00:00

set -euo pipefail
: "${MV17B_PROJECT_ROOT:?}"
: "${MV17B_A3_PAYLOAD_ROOT:?}"
: "${MV17B_OUTPUT_ROOT:?}"
: "${MV17B_PYTHON:?}"

CAMPAIGN="${MV17B_OUTPUT_ROOT}/campaign"
BUILD="${CAMPAIGN}/build"
SOURCE="${BUILD}/DS2V_BIRD_MV17B.F90"
PATCHED_SOURCE="${BUILD}/DS2V_BIRD_MV17B_A3.F90"
PATCH_REPORT="${BUILD}/MV17B_A3_SOURCE_PATCH_REPORT.json"
EXE="${BUILD}/ds2v_bird_mv17b_a3"
LOG="${BUILD}/mv17b_a3_build.log"
test -s "${MV17B_OUTPUT_ROOT}/MV17B_A3_BENCHMARK_INDEPENDENT_RECOVERY_LOCK.json"
test ! -e "${MV17B_OUTPUT_ROOT}/analysis"
test -s "${SOURCE}"
test ! -e "${PATCHED_SOURCE}"
test ! -e "${EXE}"
exec > >(tee "${CAMPAIGN}/logs/prepare_a3_${SLURM_JOB_ID}.out") 2>&1

cd "${MV17B_PROJECT_ROOT}"
export PATH="$(dirname "${MV17B_PYTHON}"):${PATH}"
export PYTHONPATH="${MV17B_PROJECT_ROOT}:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export PYTHONUNBUFFERED=1

"${MV17B_PYTHON}" -m vgdsmc.mohammadzadeh_mv17b_a3_benchmark_independent_recovery patch-source \
  --source "${SOURCE}" \
  --output "${PATCHED_SOURCE}" \
  --report "${PATCH_REPORT}"
gfortran \
  -O2 -g -std=legacy -fdec -fno-automatic \
  -fallow-argument-mismatch -ffree-line-length-none -fno-range-check \
  -fbacktrace -fcheck=bounds \
  "${PATCHED_SOURCE}" -o "${EXE}" > "${LOG}" 2>&1
test -x "${EXE}"
test "$(grep -c 'MV17B_A3_FIXED_NOUT_LOOP_BEGIN' "${PATCHED_SOURCE}")" -eq 1
test "$(grep -c 'MV17B_A3_FIXED_NOUT_LOOP_END' "${PATCHED_SOURCE}")" -eq 1
test "$(grep -c 'MV17B_A3_BENCHMARK_STOP_DISABLED_BEGIN' "${PATCHED_SOURCE}")" -eq 1
test "$(grep -c 'MV17B_A3_BENCHMARK_STOP_DISABLED_END' "${PATCHED_SOURCE}")" -eq 1
test "$(grep -cF 'DO WHILE (NOUT < 116)' "${PATCHED_SOURCE}")" -eq 1
! grep -Fq 'IF (SUM_heat_ER.LE.Heat_er) CONV_PAR=1' "${PATCHED_SOURCE}"
test "$(grep -cF 'IF (CONV_PAR.EQ.1) GO TO 2512' "${PATCHED_SOURCE}")" -eq 1
test "$(sed -n '6p' "${CAMPAIGN}/input/interactive_input.txt")" = "3"
{
  gfortran --version | head -n 2
  echo "flags=-O2 -g -std=legacy -fdec -fno-automatic -fallow-argument-mismatch -ffree-line-length-none -fno-range-check -fbacktrace -fcheck=bounds"
  echo "fresh_IRUN=3"
  echo "locked_last_NOUT=116"
  echo "source_change=fixed_NOUT_loop_plus_benchmark_stop_disable"
  echo "source_replacement_count=2"
  echo "benchmark_values_used_for_termination=false"
  echo "base_source_sha256=$(sha256sum "${SOURCE}" | awk '{print $1}')"
  echo "extended_source_sha256=$(sha256sum "${PATCHED_SOURCE}" | awk '{print $1}')"
  echo "extended_executable_sha256=$(sha256sum "${EXE}" | awk '{print $1}')"
} > "${BUILD}/MV17B_A3_BUILD_CONFIGURATION.txt"
touch "${CAMPAIGN}/MV17B_A3_PREPARE_PASS"
echo "MV17B_A3_PREPARE_PASS fresh_IRUN=3 fixed_nout=116 benchmark_stop=DISABLED"
