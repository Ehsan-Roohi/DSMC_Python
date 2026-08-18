#!/usr/bin/env bash
set -euo pipefail

LEVEL="${1:-student}"
MODE="${2:-serial}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_DIR="${ROOT_DIR}/runs/${LEVEL}_kn01"
DEFAULT_BIN="${ROOT_DIR}/third_party/sparta/src/spa_${MODE}"
SPARTA_BIN="${SPARTA_BIN:-${DEFAULT_BIN}}"

if [[ "${LEVEL}" != "smoke" && "${LEVEL}" != "student" && "${LEVEL}" != "production" ]]; then
  echo "Usage: $0 [smoke|student|production] [serial|mpi]" >&2
  exit 2
fi
if [[ "${MODE}" != "serial" && "${MODE}" != "mpi" ]]; then
  echo "Usage: $0 [smoke|student|production] [serial|mpi]" >&2
  exit 2
fi
if [[ ! -x "${SPARTA_BIN}" ]]; then
  echo "SPARTA binary not found: ${SPARTA_BIN}" >&2
  echo "Run: bash scripts/install_sparta_linux.sh ${MODE}" >&2
  exit 3
fi

python3 "${ROOT_DIR}/scripts/generate_case.py" --level "${LEVEL}" --output "${RUN_DIR}"

pushd "${RUN_DIR}" >/dev/null
export MPLCONFIGDIR="${RUN_DIR}/.matplotlib"
mkdir -p "${MPLCONFIGDIR}"
if [[ "${MODE}" == "mpi" ]]; then
  MPI_RANKS="${MPI_RANKS:-4}"
  mpirun -np "${MPI_RANKS}" "${SPARTA_BIN}" -in in.cavity -log log.cavity
else
  "${SPARTA_BIN}" -in in.cavity -log log.cavity
fi
popd >/dev/null

if [[ "${SPARTA_SKIP_POST:-0}" != "1" ]]; then
  set +e
  python3 "${ROOT_DIR}/scripts/postprocess.py" "${RUN_DIR}"
  POST_STATUS=$?
  set -e
  if [[ ${POST_STATUS} -eq 2 && "${LEVEL}" != "production" ]]; then
    echo "Expected: ${LEVEL} is a learning/smoke run, not a publication validation." >&2
  elif [[ ${POST_STATUS} -ne 0 ]]; then
    exit "${POST_STATUS}"
  fi
fi

echo "Run directory: ${RUN_DIR}"
