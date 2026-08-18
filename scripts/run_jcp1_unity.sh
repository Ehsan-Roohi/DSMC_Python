#!/usr/bin/env bash
# Bootstrap and submit the development-only JCP Phase-0 hierarchy audit.

set -euo pipefail

JCP1_CODE_COMMIT=1a3ee336f7a4c5488b9d5a602446841b9fcb2347
JCP1_RAW="https://raw.githubusercontent.com/Ehsan-Roohi/DSMC_Python/${JCP1_CODE_COMMIT}"
JCP1_ORIGINAL_REPO=/project/pi_roohie_umass_edu/DSMC_Python_M3_QY/vision_guided_dsmc
JCP1_WORK=/project/pi_roohie_umass_edu/DSMC_Python_M3_QY/JCP1
JCP1_SOURCE="${JCP1_WORK}/src"

trap 'JCP1_RC=$?; echo "JCP1_BOOTSTRAP_FAILED rc=${JCP1_RC} line=${LINENO} command=${BASH_COMMAND}" >&2; exit "${JCP1_RC}"' ERR

[[ -d "${JCP1_ORIGINAL_REPO}/vgdsmc" ]] || { echo "MISSING=${JCP1_ORIGINAL_REPO}/vgdsmc" >&2; exit 2; }
[[ -d "${JCP1_ORIGINAL_REPO}/reference_data" ]] || { echo "MISSING=${JCP1_ORIGINAL_REPO}/reference_data" >&2; exit 2; }

if [[ -f "${JCP1_WORK}/JCP1.zip" && -f "${JCP1_WORK}/JCP1.zip.sha256" ]]; then
    cd "${JCP1_WORK}"
    sha256sum -c JCP1.zip.sha256
    echo "JCP1_ALREADY_COMPLETE=1"
    echo "UPLOAD=${JCP1_WORK}/JCP1.zip ${JCP1_WORK}/JCP1.zip.sha256"
    exit 0
fi

if [[ -f "${JCP1_WORK}/LAST_JCP1.env" ]]; then
    source "${JCP1_WORK}/LAST_JCP1.env"
    if [[ -n "$(squeue -h -j "${JCP1_JOB_ID}" 2>/dev/null)" ]]; then
        echo "JCP1_ALREADY_SUBMITTED=${JCP1_JOB_ID}"
        echo "MONITOR=squeue -j ${JCP1_JOB_ID}"
        exit 0
    fi
    JCP1_OLD_STATE="$(sacct -X -n -j "${JCP1_JOB_ID}" --format=State 2>/dev/null | awk 'NF {print $1; exit}')"
    echo "JCP1_PREVIOUS_JOB=${JCP1_JOB_ID} state=${JCP1_OLD_STATE:-UNKNOWN}"
    JCP1_WORK=/project/pi_roohie_umass_edu/DSMC_Python_M3_QY/JCP1R
    JCP1_SOURCE="${JCP1_WORK}/src"
    [[ ! -e "${JCP1_WORK}" ]] || { echo "REFUSING_OVERWRITE=${JCP1_WORK}" >&2; exit 3; }
fi

MV15C_OUTPUT_ROOT=
if [[ -f "${JCP1_ORIGINAL_REPO}/LAST_MOHAMMADZADEH_MV15C_A1_QY_RESULT.env" ]]; then
    set -a
    source "${JCP1_ORIGINAL_REPO}/LAST_MOHAMMADZADEH_MV15C_A1_QY_RESULT.env"
    set +a
    MV15C_OUTPUT_ROOT="${MV15C_A1_OUTPUT_ROOT:-}"
elif [[ -f "${JCP1_ORIGINAL_REPO}/LAST_MOHAMMADZADEH_MV15C_FRESH_B3_RESULT.env" ]]; then
    set -a
    source "${JCP1_ORIGINAL_REPO}/LAST_MOHAMMADZADEH_MV15C_FRESH_B3_RESULT.env"
    set +a
fi
[[ -n "${MV15C_OUTPUT_ROOT}" ]] || { echo "MISSING_MV15C_RESULT_POINTER=1" >&2; exit 4; }
[[ -d "${MV15C_OUTPUT_ROOT}/references" ]] || { echo "MISSING=${MV15C_OUTPUT_ROOT}/references" >&2; exit 4; }
[[ -f "${MV15C_OUTPUT_ROOT}/locked_fresh_predictions.npz" ]] || { echo "MISSING=${MV15C_OUTPUT_ROOT}/locked_fresh_predictions.npz" >&2; exit 4; }

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
[[ -x "${JCP1_PYTHON}" ]] || { echo "MISSING_PYTHON=${JCP1_PYTHON}" >&2; exit 5; }

mkdir -p "${JCP1_SOURCE}" "${JCP1_SOURCE}/tests" "${JCP1_WORK}/logs"
mkdir -p "${JCP1_SOURCE}/vgdsmc" "${JCP1_SOURCE}/reference_data"
cp -a "${JCP1_ORIGINAL_REPO}/vgdsmc/." "${JCP1_SOURCE}/vgdsmc/"
cp -a "${JCP1_ORIGINAL_REPO}/reference_data/." "${JCP1_SOURCE}/reference_data/"
curl --retry 3 -fsSL "${JCP1_RAW}/vgdsmc/jcp_phase0.py" -o "${JCP1_SOURCE}/vgdsmc/jcp_phase0.py"
curl --retry 3 -fsSL "${JCP1_RAW}/tests/test_jcp_phase0.py" -o "${JCP1_SOURCE}/tests/test_jcp_phase0.py"
curl --retry 3 -fsSL "${JCP1_RAW}/scripts/unity_jcp1.sbatch" -o "${JCP1_WORK}/unity_jcp1.sbatch"
touch "${JCP1_SOURCE}/tests/__init__.py"

cd "${JCP1_SOURCE}"
PYTHONPATH="${JCP1_SOURCE}" "${JCP1_PYTHON}" -m py_compile \
    vgdsmc/jcp_phase0.py tests/test_jcp_phase0.py
PYTHONPATH="${JCP1_SOURCE}" "${JCP1_PYTHON}" tests/test_jcp_phase0.py -q
echo "JCP1_PREFLIGHT_PASS=1"

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
