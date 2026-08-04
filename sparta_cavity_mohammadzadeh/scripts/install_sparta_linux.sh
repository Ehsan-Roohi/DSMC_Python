#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-serial}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
INSTALL_DIR="${SPARTA_INSTALL_DIR:-${ROOT_DIR}/third_party/sparta}"
SPARTA_REF="${SPARTA_REF:-912c9e163c38ea5c3562d039e65215f6e2a4f3f8}"
BUILD_JOBS="${BUILD_JOBS:-4}"

if [[ "${MODE}" != "serial" && "${MODE}" != "mpi" ]]; then
  echo "Usage: $0 [serial|mpi]" >&2
  exit 2
fi

for command_name in git make g++; do
  if ! command -v "${command_name}" >/dev/null 2>&1; then
    echo "Missing ${command_name}. On Ubuntu run:" >&2
    echo "  sudo apt update && sudo apt install -y git build-essential" >&2
    exit 3
  fi
done

if [[ "${MODE}" == "mpi" ]]; then
  for command_name in mpicxx mpirun; do
    if ! command -v "${command_name}" >/dev/null 2>&1; then
      echo "Missing ${command_name}. On Ubuntu run:" >&2
      echo "  sudo apt install -y openmpi-bin libopenmpi-dev" >&2
      exit 3
    fi
  done
fi

if [[ -e "${INSTALL_DIR}" ]]; then
  echo "Refusing to overwrite existing path: ${INSTALL_DIR}" >&2
  echo "Set SPARTA_INSTALL_DIR to a new empty path or reuse its existing binary." >&2
  exit 4
fi

mkdir -p "$(dirname "${INSTALL_DIR}")"
git clone https://github.com/sparta/sparta.git "${INSTALL_DIR}"
git -C "${INSTALL_DIR}" checkout --detach "${SPARTA_REF}"
make -C "${INSTALL_DIR}/src" -j"${BUILD_JOBS}" "${MODE}"

BINARY="${INSTALL_DIR}/src/spa_${MODE}"
if [[ ! -x "${BINARY}" ]]; then
  echo "SPARTA build did not create ${BINARY}" >&2
  exit 5
fi

echo "SPARTA ${MODE} binary: ${BINARY}"
echo "Pinned source commit: $(git -C "${INSTALL_DIR}" rev-parse HEAD)"
