#!/usr/bin/env bash
#SBATCH --job-name=sparta-hq-build
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=00:45:00
#SBATCH --output=logs/sparta-hq-build-%j.out
#SBATCH --error=logs/sparta-hq-build-%j.err

set -euo pipefail

ROOT_DIR="${SPARTA_CASE_ROOT:-${SLURM_SUBMIT_DIR:-}}"
if [[ -z "${ROOT_DIR}" || ! -d "${ROOT_DIR}" ]]; then
  echo "SPARTA_CASE_ROOT/SLURM_SUBMIT_DIR does not identify the case directory." >&2
  exit 2
fi
ROOT_DIR="$(cd "${ROOT_DIR}" && pwd)"
SPARTA_BIN="${ROOT_DIR}/third_party/sparta/src/spa_mpi"
OPENMPI_MODULE="${UNITY_OPENMPI_MODULE:-openmpi/5.0.3}"
BUILD_JOBS="${SLURM_CPUS_PER_TASK:-8}"

module purge
module load "${OPENMPI_MODULE}"
module list

for COMMAND_NAME in git make g++ mpicxx python3; do
  if ! command -v "${COMMAND_NAME}" >/dev/null 2>&1; then
    echo "Required build command is unavailable after loading ${OPENMPI_MODULE}: ${COMMAND_NAME}" >&2
    exit 2
  fi
done

cd "${ROOT_DIR}"
python3 -m unittest discover -s tests -v

if [[ -x "${SPARTA_BIN}" ]]; then
  echo "Reusing existing MPI-enabled SPARTA executable: ${SPARTA_BIN}"
elif [[ -d "${ROOT_DIR}/third_party/sparta/src" ]]; then
  make -C "${ROOT_DIR}/third_party/sparta/src" -j"${BUILD_JOBS}" mpi
else
  BUILD_JOBS="${BUILD_JOBS}" bash "${ROOT_DIR}/scripts/install_sparta_linux.sh" mpi
fi

if [[ ! -x "${SPARTA_BIN}" ]]; then
  echo "SPARTA build did not create ${SPARTA_BIN}" >&2
  exit 3
fi

HELP_FILE="${ROOT_DIR}/logs/sparta-hq-build-${SLURM_JOB_ID}-help.txt"
set +e
"${SPARTA_BIN}" -help > "${HELP_FILE}" 2>&1
HELP_STATUS=$?
set -e
if ! grep -q "^Usage: sparta" "${HELP_FILE}"; then
  echo "SPARTA executable did not produce a valid help banner (exit ${HELP_STATUS})." >&2
  exit 4
fi

# Exercise the generated thermal/grid diagnostic before releasing the HQ array.
# This catches input syntax or output-schema regressions in seconds rather than
# after a multi-hour production member has started.
SMOKE_DIR="${ROOT_DIR}/logs/sparta-thermal-smoke-${SLURM_JOB_ID}"
mkdir "${SMOKE_DIR}"
python3 "${ROOT_DIR}/scripts/generate_case.py" \
  --level smoke \
  --kn 0.1 \
  --seed 20260803 \
  --output "${SMOKE_DIR}" > "${SMOKE_DIR}/generator.out"
(
  cd "${SMOKE_DIR}"
  "${SPARTA_BIN}" < in.cavity > screen.out 2> screen.err
)
SMOKE_DUMP="${SMOKE_DIR}/grid.final.00000030"
if [[ ! -s "${SMOKE_DUMP}" ]] || \
   ! grep -Fq 'f_flowavg[5]' "${SMOKE_DUMP}" || \
   grep -Fq 'f_flowavg[6]' "${SMOKE_DUMP}"; then
  echo "Thermal-temperature smoke test did not produce the required five-column dump." >&2
  exit 5
fi

echo "SPARTA tests, help banner, and thermal-temperature smoke case verified."
echo "READY: ${SPARTA_BIN}"
