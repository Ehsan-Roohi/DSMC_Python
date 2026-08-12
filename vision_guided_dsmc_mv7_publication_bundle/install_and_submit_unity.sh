#!/usr/bin/env bash
set -euo pipefail

BUNDLE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_TARGET="/project/pi_roohie_umass_edu/DSMC_Python_M3_QY/vision_guided_dsmc"
TARGET_ROOT="${1:-${MV7_PUBLICATION_TARGET_ROOT:-${DEFAULT_TARGET}}}"

test -d "${TARGET_ROOT}"
test -f "${TARGET_ROOT}/LAST_MOHAMMADZADEH_MV7_JCP_JOB.env"
test -f "${BUNDLE_ROOT}/payload/scripts/make_mv7_jcp_publication_suite.py"
test -f "${BUNDLE_ROOT}/payload/scripts/unity_mohammadzadeh_mv7_publication.sbatch"
test -f "${BUNDLE_ROOT}/payload/tests/test_make_mv7_jcp_publication_suite.py"

if [[ -f "${TARGET_ROOT}/LAST_MOHAMMADZADEH_MV7_PUBLICATION_JOB.env" \
      && "${MV7_PUBLICATION_ALLOW_NEW_RUN:-0}" != "1" ]]; then
    echo "Refusing duplicate MV7 publication submission." >&2
    echo "Inspect LAST_MOHAMMADZADEH_MV7_PUBLICATION_JOB.env first, or set MV7_PUBLICATION_ALLOW_NEW_RUN=1 intentionally." >&2
    exit 2
fi

install -D -m 0644 \
    "${BUNDLE_ROOT}/payload/scripts/make_mv7_jcp_publication_suite.py" \
    "${TARGET_ROOT}/scripts/make_mv7_jcp_publication_suite.py"
install -D -m 0644 \
    "${BUNDLE_ROOT}/payload/tests/test_make_mv7_jcp_publication_suite.py" \
    "${TARGET_ROOT}/tests/test_make_mv7_jcp_publication_suite.py"
install -D -m 0755 \
    "${BUNDLE_ROOT}/payload/scripts/unity_mohammadzadeh_mv7_publication.sbatch" \
    "${TARGET_ROOT}/scripts/unity_mohammadzadeh_mv7_publication.sbatch"

cd "${TARGET_ROOT}"
source LAST_MOHAMMADZADEH_MV7_JCP_JOB.env
PYTHON_BIN="python3"
if [[ -n "${MV7_VENV_DIR:-}" && -x "${MV7_VENV_DIR}/bin/python" ]]; then
    PYTHON_BIN="${MV7_VENV_DIR}/bin/python"
fi
"${PYTHON_BIN}" -m py_compile scripts/make_mv7_jcp_publication_suite.py
"${PYTHON_BIN}" -m pytest -q tests/test_make_mv7_jcp_publication_suite.py
mkdir -p logs

JOB_ID="$(sbatch --parsable \
    --export="ALL,MV7_REPO_ROOT=${TARGET_ROOT}" \
    scripts/unity_mohammadzadeh_mv7_publication.sbatch)"

ENV_FILE="${TARGET_ROOT}/LAST_MOHAMMADZADEH_MV7_PUBLICATION_JOB.env"
printf 'MV7_PUBLICATION_JOB_ID=%q\nMV7_REPO_ROOT=%q\n' \
    "${JOB_ID}" "${TARGET_ROOT}" > "${ENV_FILE}"

echo "Submitted MV7 JCP publication job: ${JOB_ID}"
echo "Saved: ${ENV_FILE}"
echo "Check: squeue -j ${JOB_ID}"
echo "Log:   ${TARGET_ROOT}/logs/moh_mv7_pub_${JOB_ID}.out"
