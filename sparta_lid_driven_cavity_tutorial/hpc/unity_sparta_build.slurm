#!/usr/bin/env bash
#SBATCH --job-name=sparta-build
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/sparta-build-%j.out
#SBATCH --error=logs/sparta-build-%j.err

set -euo pipefail

ROOT_DIR="${SPARTA_TUTORIAL_ROOT:-${SLURM_SUBMIT_DIR:-}}"
if [[ -z "${ROOT_DIR}" || ! -d "${ROOT_DIR}" ]]; then
  echo "SPARTA_TUTORIAL_ROOT/SLURM_SUBMIT_DIR does not identify the tutorial directory." >&2
  exit 2
fi
ROOT_DIR="$(cd "${ROOT_DIR}" && pwd)"
SPARTA_BIN="${ROOT_DIR}/third_party/sparta/src/spa_mpi"
OPENMPI_MODULE="${UNITY_OPENMPI_MODULE:-openmpi/5.0.3}"
BUILD_JOBS="${SLURM_CPUS_PER_TASK:-8}"

module purge
module load "${OPENMPI_MODULE}"
module list

for command_name in git make g++ mpicxx mpirun; do
  if ! command -v "${command_name}" >/dev/null 2>&1; then
    echo "Required build command is unavailable: ${command_name}" >&2
    exit 2
  fi
done

if [[ -x "${SPARTA_BIN}" ]]; then
  echo "Reusing MPI-enabled SPARTA executable: ${SPARTA_BIN}"
elif [[ -d "${ROOT_DIR}/third_party/sparta/src" ]]; then
  make -C "${ROOT_DIR}/third_party/sparta/src" -j"${BUILD_JOBS}" mpi
else
  BUILD_JOBS="${BUILD_JOBS}" bash "${ROOT_DIR}/scripts/install_sparta_linux.sh" mpi
fi

if [[ ! -x "${SPARTA_BIN}" ]]; then
  echo "SPARTA build did not create ${SPARTA_BIN}" >&2
  exit 3
fi

HELP_FILE="${ROOT_DIR}/logs/sparta-build-${SLURM_JOB_ID}-help.txt"
set +e
"${SPARTA_BIN}" -help > "${HELP_FILE}" 2>&1
HELP_STATUS=$?
set -e
if ! grep -q "^Usage: sparta" "${HELP_FILE}"; then
  echo "SPARTA executable did not produce a valid help banner." >&2
  exit 4
fi
echo "SPARTA help banner verified (exit status ${HELP_STATUS})."
echo "READY: ${SPARTA_BIN}"

