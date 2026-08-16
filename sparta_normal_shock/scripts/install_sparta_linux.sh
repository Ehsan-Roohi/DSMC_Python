#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-serial}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
INSTALL_DIR="${SPARTA_INSTALL_DIR:-${ROOT_DIR}/third_party/sparta}"
SPARTA_REF="${SPARTA_REF:-51021386a11c1d045de3fad0c98218e34bc09fc6}"
BUILD_JOBS="${BUILD_JOBS:-4}"

if [[ "${MODE}" != "serial" && "${MODE}" != "mpi" ]]; then
  echo "Usage: $0 [serial|mpi]" >&2
  exit 2
fi
for command_name in git make g++; do
  command -v "${command_name}" >/dev/null 2>&1 || {
    echo "Missing required build command: ${command_name}" >&2
    exit 3
  }
done
if [[ "${MODE}" == "mpi" ]]; then
  command -v mpicxx >/dev/null 2>&1 || { echo "Missing mpicxx" >&2; exit 3; }
fi
if [[ -e "${INSTALL_DIR}" ]]; then
  echo "Refusing to overwrite existing path: ${INSTALL_DIR}" >&2
  exit 4
fi
mkdir -p "$(dirname "${INSTALL_DIR}")"
git clone https://github.com/sparta/sparta.git "${INSTALL_DIR}"
git -C "${INSTALL_DIR}" checkout --detach "${SPARTA_REF}"
make -C "${INSTALL_DIR}/src" -j"${BUILD_JOBS}" "${MODE}"
BINARY="${INSTALL_DIR}/src/spa_${MODE}"
[[ -x "${BINARY}" ]] || { echo "SPARTA build did not create ${BINARY}" >&2; exit 5; }
echo "SPARTA ${MODE} binary: ${BINARY}"
echo "Pinned source commit: $(git -C "${INSTALL_DIR}" rev-parse HEAD)"

