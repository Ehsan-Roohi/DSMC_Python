#!/usr/bin/env bash
set -euo pipefail

LEVEL="${1:-tutorial}"
MODE="${2:-serial}"
SEED="${SEED:-20260803}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_DIR="${RUN_DIR:-${ROOT_DIR}/runs/${LEVEL}_kn01_seed_${SEED}}"
DEFAULT_BIN="${ROOT_DIR}/third_party/sparta/src/spa_${MODE}"
SPARTA_BIN="${SPARTA_BIN:-${DEFAULT_BIN}}"

if [[ "${LEVEL}" != "smoke" && "${LEVEL}" != "tutorial" && "${LEVEL}" != "production" ]]; then
  echo "Usage: $0 [smoke|tutorial|production] [serial|mpi]" >&2
  exit 2
fi
if [[ "${MODE}" != "serial" && "${MODE}" != "mpi" ]]; then
  echo "Usage: $0 [smoke|tutorial|production] [serial|mpi]" >&2
  exit 2
fi
if [[ ! -x "${SPARTA_BIN}" ]]; then
  echo "SPARTA binary not found: ${SPARTA_BIN}" >&2
  echo "Build it with: bash scripts/install_sparta_linux.sh ${MODE}" >&2
  exit 3
fi
if [[ -e "${RUN_DIR}" ]]; then
  echo "Refusing to overwrite existing run directory: ${RUN_DIR}" >&2
  exit 4
fi

python3 "${ROOT_DIR}/scripts/generate_case.py" \
  --level "${LEVEL}" --kn 0.1 --seed "${SEED}" --output "${RUN_DIR}"

pushd "${RUN_DIR}" >/dev/null
export MPLCONFIGDIR="${RUN_DIR}/.matplotlib"
mkdir -p "${MPLCONFIGDIR}"
if [[ "${MODE}" == "mpi" ]]; then
  MPI_RANKS="${MPI_RANKS:-4}"
  mpirun -n "${MPI_RANKS}" "${SPARTA_BIN}" < in.cavity
else
  "${SPARTA_BIN}" -in in.cavity -log log.cavity
fi
if [[ -s log.sparta && ! -e log.cavity ]]; then
  mv -- log.sparta log.cavity
fi
popd >/dev/null

python3 "${ROOT_DIR}/scripts/postprocess.py" "${RUN_DIR}"
echo "Completed run directory: ${RUN_DIR}"

