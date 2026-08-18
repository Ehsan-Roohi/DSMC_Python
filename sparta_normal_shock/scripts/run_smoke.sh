#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SPARTA_BIN="${1:-${SPARTA_BIN:-${ROOT_DIR}/third_party/sparta/src/spa_serial}}"
[[ -x "${SPARTA_BIN}" ]] || { echo "Usage: $0 /absolute/path/to/spa_serial" >&2; exit 2; }
SMOKE_DIR="$(mktemp -d "${TMPDIR:-/tmp}/sparta-normal-shock-smoke.XXXXXX")"
trap 'rm -rf -- "${SMOKE_DIR}"' EXIT
python3 "${ROOT_DIR}/scripts/generate_case.py" \
  --level smoke --mach 3 --seed 20260803 --output "${SMOKE_DIR}" > "${SMOKE_DIR}/generator.out"
(
  cd "${SMOKE_DIR}"
  "${SPARTA_BIN}" < in.shock > screen.out 2> screen.err
)
DUMP="${SMOKE_DIR}/profile.final.00000400"
[[ -s "${DUMP}" ]] || { echo "Smoke run did not create ${DUMP}" >&2; exit 3; }
grep -Fq 'f_avg[10]' "${DUMP}" || { echo "Ten-field dump schema missing" >&2; exit 4; }
python3 "${ROOT_DIR}/scripts/postprocess.py" single "${SMOKE_DIR}" >/dev/null
[[ -s "${SMOKE_DIR}/validation_metrics.json" ]] || exit 5
echo "SPARTA_NORMAL_SHOCK_SMOKE_PASS dump=${DUMP}"

