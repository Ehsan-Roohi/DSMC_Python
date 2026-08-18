#!/usr/bin/env bash
# Bootstrap and submit the development-only JCP Phase-0 hierarchy audit.

set -euo pipefail

JCP1_CODE_COMMIT=1a3ee336f7a4c5488b9d5a602446841b9fcb2347
JCP1_RAW="https://raw.githubusercontent.com/Ehsan-Roohi/DSMC_Python/${JCP1_CODE_COMMIT}"
JCP1_ORIGINAL_REPO=/project/pi_roohie_umass_edu/DSMC_Python_M3_QY/vision_guided_dsmc
JCP1_WORK=/project/pi_roohie_umass_edu/DSMC_Python_M3_QY/JCP1
JCP1_SOURCE="${JCP1_WORK}/src"

test -d "${JCP1_ORIGINAL_REPO}/vgdsmc"
test -d "${JCP1_ORIGINAL_REPO}/reference_data"
test -f "${JCP1_ORIGINAL_REPO}/LAST_MOHAMMADZADEH_MV15C_FRESH_B3_RESULT.env"
test ! -e "${JCP1_WORK}"

set -a
source "${JCP1_ORIGINAL_REPO}/LAST_MOHAMMADZADEH_MV15C_FRESH_B3_RESULT.env"
set +a
: "${MV15C_OUTPUT_ROOT:?MV15C output pointer is incomplete}"
test -d "${MV15C_OUTPUT_ROOT}/references"
test -f "${MV15C_OUTPUT_ROOT}/locked_fresh_predictions.npz"

JCP1_VENV=
if test -f "${JCP1_ORIGINAL_REPO}/LAST_MOHAMMADZADEH_MV9_HEAT_FLUX_JOB.env"; then
    set -a
    source "${JCP1_ORIGINAL_REPO}/LAST_MOHAMMADZADEH_MV9_HEAT_FLUX_JOB.env"
    set +a
    JCP1_VENV="${MV9_VENV_DIR:-}"
fi
if [[ -z "${JCP1_VENV}" || ! -x "${JCP1_VENV}/bin/python" ]]; then
    JCP1_VENV="${JCP1_ORIGINAL_REPO}/.venv-mv1"
fi
JCP1_PYTHON="${JCP1_VENV}/bin/python"
test -x "${JCP1_PYTHON}"

mkdir -p "${JCP1_SOURCE}" "${JCP1_SOURCE}/tests" "${JCP1_WORK}/logs"
cp -a "${JCP1_ORIGINAL_REPO}/vgdsmc" "${JCP1_SOURCE}/"
cp -a "${JCP1_ORIGINAL_REPO}/reference_data" "${JCP1_SOURCE}/"
curl -fsSL "${JCP1_RAW}/vgdsmc/jcp_phase0.py" -o "${JCP1_SOURCE}/vgdsmc/jcp_phase0.py"
curl -fsSL "${JCP1_RAW}/tests/test_jcp_phase0.py" -o "${JCP1_SOURCE}/tests/test_jcp_phase0.py"
curl -fsSL "${JCP1_RAW}/scripts/unity_jcp1.sbatch" -o "${JCP1_WORK}/unity_jcp1.sbatch"

cd "${JCP1_SOURCE}"
PYTHONPATH="${JCP1_SOURCE}" "${JCP1_PYTHON}" -m py_compile \
    vgdsmc/jcp_phase0.py tests/test_jcp_phase0.py
PYTHONPATH="${JCP1_SOURCE}" "${JCP1_PYTHON}" -m unittest -q tests.test_jcp_phase0

cd "${JCP1_WORK}"
JCP1_JOB_ID="$(sbatch --parsable \
    --export="ALL,JCP1_REPO_ROOT=${JCP1_SOURCE},JCP1_MV15C_ROOT=${MV15C_OUTPUT_ROOT},JCP1_WORK_ROOT=${JCP1_WORK},JCP1_PYTHON=${JCP1_PYTHON},JCP1_SCRIPT=${JCP1_SOURCE}/vgdsmc/jcp_phase0.py" \
    unity_jcp1.sbatch)"

printf 'JCP1_JOB_ID=%q\nJCP1_WORK=%q\nJCP1_ARCHIVE=%q\nJCP1_CHECKSUM=%q\nJCP1_CODE_COMMIT=%q\n' \
    "${JCP1_JOB_ID}" "${JCP1_WORK}" "${JCP1_WORK}/JCP1.zip" \
    "${JCP1_WORK}/JCP1.zip.sha256" "${JCP1_CODE_COMMIT}" \
    > "${JCP1_WORK}/LAST_JCP1.env"

echo "JCP1_JOB_ID=${JCP1_JOB_ID}"
echo "MONITOR=squeue -j ${JCP1_JOB_ID}"
echo "WHEN_COMPLETE_UPLOAD=${JCP1_WORK}/JCP1.zip ${JCP1_WORK}/JCP1.zip.sha256"
