#!/usr/bin/env bash
#SBATCH --job-name=sp2-shock-build
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=00:45:00
#SBATCH --output=logs/sp2-shock-build-%j.out
#SBATCH --error=logs/sp2-shock-build-%j.err

set -euo pipefail

ROOT_DIR="${SPARTA_CASE_ROOT:-${SLURM_SUBMIT_DIR:-}}"
[[ -n "${ROOT_DIR}" && -d "${ROOT_DIR}" ]] || {
  echo "SPARTA_CASE_ROOT/SLURM_SUBMIT_DIR does not identify the case directory." >&2
  exit 2
}
ROOT_DIR="$(cd "${ROOT_DIR}" && pwd)"
SPARTA_BIN="${ROOT_DIR}/third_party/sparta/src/spa_mpi"
OPENMPI_MODULE="${UNITY_OPENMPI_MODULE:-openmpi/5.0.3}"
BUILD_JOBS="${SLURM_CPUS_PER_TASK:-8}"

module purge
module load "${OPENMPI_MODULE}"
module list
for command_name in git make g++ mpicxx python3; do
  command -v "${command_name}" >/dev/null 2>&1 || {
    echo "Required command unavailable after loading ${OPENMPI_MODULE}: ${command_name}" >&2
    exit 2
  }
done

cd "${ROOT_DIR}"
python3 -m unittest discover -s tests -v
if [[ -x "${SPARTA_BIN}" ]]; then
  echo "Reusing MPI SPARTA executable: ${SPARTA_BIN}"
elif [[ -d "${ROOT_DIR}/third_party/sparta/src" ]]; then
  make -C "${ROOT_DIR}/third_party/sparta/src" -j"${BUILD_JOBS}" mpi
else
  BUILD_JOBS="${BUILD_JOBS}" bash "${ROOT_DIR}/scripts/install_sparta_linux.sh" mpi
fi
[[ -x "${SPARTA_BIN}" ]] || { echo "Missing ${SPARTA_BIN}" >&2; exit 3; }

HELP_FILE="${ROOT_DIR}/logs/sp2-shock-build-${SLURM_JOB_ID}-help.txt"
set +e
"${SPARTA_BIN}" -help > "${HELP_FILE}" 2>&1
HELP_STATUS=$?
set -e
grep -q '^Usage: sparta' "${HELP_FILE}" || {
  echo "SPARTA help preflight failed with status ${HELP_STATUS}." >&2
  exit 4
}

bash "${ROOT_DIR}/scripts/run_smoke.sh" "${SPARTA_BIN}"
{
  printf 'status=READY\n'
  printf 'sparta_binary=%s\n' "${SPARTA_BIN}"
  printf 'sparta_source_commit=%s\n' "$(git -C "${ROOT_DIR}/third_party/sparta" rev-parse HEAD)"
  printf 'sparta_help_exit_status=%s\n' "${HELP_STATUS}"
  printf 'sparta_help_signature=pass\n'
  printf 'verified_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
} > "${ROOT_DIR}/logs/BUILD_READY_${SLURM_JOB_ID}.txt"
echo "SPARTA tests and real M=3 normal-shock smoke case passed."
