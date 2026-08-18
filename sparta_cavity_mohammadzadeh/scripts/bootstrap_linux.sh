#!/usr/bin/env bash
set -euo pipefail

REPOSITORY="https://github.com/Ehsan-Roohi/DSMC_Python.git"
BRANCH="${DSMC_BOOK_BRANCH:-agent/validated-dsmc-cavity}"
DESTINATION="${DSMC_BOOK_DIR:-${PWD}/DSMC_Python}"

for command_name in git make g++ python3; do
  if ! command -v "${command_name}" >/dev/null 2>&1; then
    echo "Missing prerequisite: ${command_name}" >&2
    echo "Install the Linux prerequisites listed in README.md, then try again." >&2
    exit 2
  fi
done

if [[ -e "${DESTINATION}" ]]; then
  echo "Refusing to overwrite existing path: ${DESTINATION}" >&2
  echo "Set DSMC_BOOK_DIR to a new directory or use the existing checkout." >&2
  exit 3
fi

git clone --branch "${BRANCH}" --single-branch "${REPOSITORY}" "${DESTINATION}"
CASE_ROOT="${DESTINATION}/sparta_cavity_mohammadzadeh"
cd "${CASE_ROOT}"

bash scripts/install_sparta_linux.sh serial
python3 -m unittest discover -s tests -v
bash scripts/run_case.sh smoke serial

echo
echo "SPARTA smoke workflow completed. This is not a publication validation."
echo "Next classroom run:"
echo "  cd ${CASE_ROOT}"
echo "  bash scripts/run_case.sh student serial"
