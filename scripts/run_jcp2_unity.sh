#!/usr/bin/env bash
set -Eeuo pipefail

JCP2_CODE_COMMIT=189a098d09974b6c8afd905e87d33651d8c83de5
JCP2_RAW="https://raw.githubusercontent.com/Ehsan-Roohi/DSMC_Python/${JCP2_CODE_COMMIT}"
JCP2_ORIGINAL_REPO=/project/pi_roohie_umass_edu/DSMC_Python_M3_QY/vision_guided_dsmc
JCP2_WORK=/project/pi_roohie_umass_edu/DSMC_Python_M3_QY/JCP2
JCP2_SOURCE="${JCP2_WORK}/src"

trap 'JCP2_RC=$?; echo "JCP2_BOOTSTRAP_FAILED rc=${JCP2_RC} line=${LINENO} command=${BASH_COMMAND}" >&2; exit "${JCP2_RC}"' ERR

[[ -d "${JCP2_ORIGINAL_REPO}/vgdsmc" ]] || { echo "MISSING=${JCP2_ORIGINAL_REPO}/vgdsmc" >&2; exit 2; }
[[ -d "${JCP2_ORIGINAL_REPO}/reference_data" ]] || { echo "MISSING=${JCP2_ORIGINAL_REPO}/reference_data" >&2; exit 2; }

if [[ -f "${JCP2_WORK}/JCP2.zip" && -f "${JCP2_WORK}/JCP2.zip.sha256" ]]; then
    cd "${JCP2_WORK}"
    sha256sum -c JCP2.zip.sha256
    echo "JCP2_ALREADY_COMPLETE=1"
    echo "UPLOAD=${JCP2_WORK}/JCP2.zip ${JCP2_WORK}/JCP2.zip.sha256"
    exit 0
fi

if [[ -f "${JCP2_WORK}/LAST_JCP2.env" ]]; then
    source "${JCP2_WORK}/LAST_JCP2.env"
    JCP2_ACTIVE=0
    for JCP2_ID in "${JCP2_EVAL_JOB_ID:-}" "${JCP2_REFERENCE_JOB_ID:-}" "${JCP2_PREDICTION_JOB_ID:-}" "${JCP2_SCORE_JOB_ID:-}"; do
        if [[ -n "${JCP2_ID}" && -n "$(squeue -h -j "${JCP2_ID}" 2>/dev/null)" ]]; then
            JCP2_ACTIVE=1
        fi
    done
    if [[ "${JCP2_ACTIVE}" == 1 ]]; then
        echo "JCP2_ALREADY_SUBMITTED=1"
        echo "MONITOR=squeue -j ${JCP2_EVAL_JOB_ID},${JCP2_REFERENCE_JOB_ID},${JCP2_PREDICTION_JOB_ID},${JCP2_SCORE_JOB_ID}"
        exit 0
    fi
    echo "JCP2_RESUMING_AFTER_INACTIVE_JOB_SET=1"
fi

MV15C_OUTPUT_ROOT=
if [[ -f "${JCP2_ORIGINAL_REPO}/LAST_MOHAMMADZADEH_MV15C_A1_QY_RESULT.env" ]]; then
    set +u
    source "${JCP2_ORIGINAL_REPO}/LAST_MOHAMMADZADEH_MV15C_A1_QY_RESULT.env"
    set -u
    MV15C_OUTPUT_ROOT="${MV15C_A1_OUTPUT_ROOT:-}"
elif [[ -f "${JCP2_ORIGINAL_REPO}/LAST_MOHAMMADZADEH_MV15C_FRESH_B3_RESULT.env" ]]; then
    set +u
    source "${JCP2_ORIGINAL_REPO}/LAST_MOHAMMADZADEH_MV15C_FRESH_B3_RESULT.env"
    set -u
fi
[[ -n "${MV15C_OUTPUT_ROOT}" ]] || { echo "MISSING_MV15C_RESULT_POINTER=1" >&2; exit 3; }
[[ -f "${MV15C_OUTPUT_ROOT}/submission_lock.json" ]] || { echo "MISSING=${MV15C_OUTPUT_ROOT}/submission_lock.json" >&2; exit 3; }
[[ -f "${MV15C_OUTPUT_ROOT}/locked_fresh_predictions.npz" ]] || { echo "MISSING=${MV15C_OUTPUT_ROOT}/locked_fresh_predictions.npz" >&2; exit 3; }

JCP2_VENV=
if [[ -f "${JCP2_ORIGINAL_REPO}/LAST_MOHAMMADZADEH_MV9_HEAT_FLUX_JOB.env" ]]; then
    set +u
    source "${JCP2_ORIGINAL_REPO}/LAST_MOHAMMADZADEH_MV9_HEAT_FLUX_JOB.env"
    set -u
    JCP2_VENV="${MV9_VENV_DIR:-}"
fi
if [[ -z "${JCP2_VENV}" || ! -x "${JCP2_VENV}/bin/python" ]]; then
    JCP2_VENV="${JCP2_ORIGINAL_REPO}/.venv-mv1"
fi
JCP2_PYTHON="${JCP2_VENV}/bin/python"
[[ -x "${JCP2_PYTHON}" ]] || { echo "MISSING_PYTHON=${JCP2_PYTHON}" >&2; exit 4; }

JCP2_MV9_ROOT="$("${JCP2_PYTHON}" -c 'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["mv9_output_root"])' "${MV15C_OUTPUT_ROOT}/submission_lock.json")"
[[ -f "${JCP2_MV9_ROOT}/dataset.npz" ]] || { echo "MISSING=${JCP2_MV9_ROOT}/dataset.npz" >&2; exit 4; }

mkdir -p "${JCP2_SOURCE}/vgdsmc" "${JCP2_SOURCE}/reference_data/mohammadzadeh_2012" "${JCP2_SOURCE}/scripts" "${JCP2_SOURCE}/tests" "${JCP2_WORK}/logs" "${JCP2_WORK}/runs"
cp -a "${JCP2_ORIGINAL_REPO}/vgdsmc/." "${JCP2_SOURCE}/vgdsmc/"
cp -a "${JCP2_ORIGINAL_REPO}/reference_data/." "${JCP2_SOURCE}/reference_data/"

JCP2_FILES=(
  vgdsmc/jcp_phase0.py
  vgdsmc/jcp_phase1_cavity.py
  vgdsmc/moment_sampling.py
  vgdsmc/mohammadzadeh_production.py
  vgdsmc/ntc_checkpoint.py
  reference_data/mohammadzadeh_2012/jcp2_cavity_protocol_v1.json
  reference_data/mohammadzadeh_2012/jcp2_cavity_seed_bank_v1.json
  reference_data/mohammadzadeh_2012/jcp2_cavity_lock_v1.json
  scripts/unity_jcp2_task.sbatch
  scripts/unity_jcp2_predict.sbatch
  scripts/unity_jcp2_score.sbatch
  scripts/run_jcp2_unity.sh
  tests/test_jcp_phase1_cavity.py
  tests/test_moment_sampling.py
  tests/test_ntc_checkpoint.py
)
for JCP2_FILE in "${JCP2_FILES[@]}"; do
    mkdir -p "${JCP2_SOURCE}/$(dirname "${JCP2_FILE}")"
    curl --retry 3 -fsSL "${JCP2_RAW}/${JCP2_FILE}" -o "${JCP2_SOURCE}/${JCP2_FILE}"
done
touch "${JCP2_SOURCE}/tests/__init__.py"

cd "${JCP2_SOURCE}"
PYTHONPATH="${JCP2_SOURCE}" "${JCP2_PYTHON}" -m py_compile \
  vgdsmc/jcp_phase0.py vgdsmc/jcp_phase1_cavity.py \
  vgdsmc/moment_sampling.py vgdsmc/mohammadzadeh_production.py vgdsmc/ntc_checkpoint.py
PYTHONPATH="${JCP2_SOURCE}" "${JCP2_PYTHON}" tests/test_jcp_phase1_cavity.py -q
PYTHONPATH="${JCP2_SOURCE}" "${JCP2_PYTHON}" -m vgdsmc.jcp_phase1_cavity verify-lock >/dev/null
echo "JCP2_PREFLIGHT_PASS=1"

JCP2_RUN_ROOT="${JCP2_WORK}/runs"
JCP2_PREDICTION_ROOT="${JCP2_WORK}/predictions"
JCP2_SCORE_ROOT="${JCP2_WORK}/score"
JCP2_COMMON="ALL,JCP2_REPO_ROOT=${JCP2_SOURCE},JCP2_RUN_ROOT=${JCP2_RUN_ROOT},JCP2_PYTHON=${JCP2_PYTHON}"
cd "${JCP2_WORK}"
JCP2_EVAL_JOB_ID="$(sbatch --parsable --job-name=j2-eval --array=0-11%4 --export="${JCP2_COMMON},JCP2_GROUP=evaluation" "${JCP2_SOURCE}/scripts/unity_jcp2_task.sbatch")"
JCP2_REFERENCE_JOB_ID="$(sbatch --parsable --job-name=j2-ref --array=0-24%6 --export="${JCP2_COMMON},JCP2_GROUP=reference" "${JCP2_SOURCE}/scripts/unity_jcp2_task.sbatch")"
JCP2_PREDICTION_JOB_ID="$(sbatch --parsable --dependency="afterok:${JCP2_EVAL_JOB_ID}" --export="${JCP2_COMMON},JCP2_PREDICTION_ROOT=${JCP2_PREDICTION_ROOT},JCP2_MV9_ROOT=${JCP2_MV9_ROOT},JCP2_MV15C_ROOT=${MV15C_OUTPUT_ROOT}" "${JCP2_SOURCE}/scripts/unity_jcp2_predict.sbatch")"
JCP2_SCORE_JOB_ID="$(sbatch --parsable --dependency="afterok:${JCP2_PREDICTION_JOB_ID}:${JCP2_REFERENCE_JOB_ID}" --export="${JCP2_COMMON},JCP2_PREDICTION_ROOT=${JCP2_PREDICTION_ROOT},JCP2_SCORE_ROOT=${JCP2_SCORE_ROOT},JCP2_WORK_ROOT=${JCP2_WORK}" "${JCP2_SOURCE}/scripts/unity_jcp2_score.sbatch")"

printf 'JCP2_EVAL_JOB_ID=%q\nJCP2_REFERENCE_JOB_ID=%q\nJCP2_PREDICTION_JOB_ID=%q\nJCP2_SCORE_JOB_ID=%q\nJCP2_WORK=%q\nJCP2_ARCHIVE=%q\nJCP2_CHECKSUM=%q\nJCP2_CODE_COMMIT=%q\n' \
  "${JCP2_EVAL_JOB_ID}" "${JCP2_REFERENCE_JOB_ID}" "${JCP2_PREDICTION_JOB_ID}" "${JCP2_SCORE_JOB_ID}" \
  "${JCP2_WORK}" "${JCP2_WORK}/JCP2.zip" "${JCP2_WORK}/JCP2.zip.sha256" "${JCP2_CODE_COMMIT}" \
  > "${JCP2_WORK}/LAST_JCP2.env"

echo "JCP2_EVAL_JOB_ID=${JCP2_EVAL_JOB_ID}"
echo "JCP2_REFERENCE_JOB_ID=${JCP2_REFERENCE_JOB_ID}"
echo "JCP2_PREDICTION_JOB_ID=${JCP2_PREDICTION_JOB_ID}"
echo "JCP2_SCORE_JOB_ID=${JCP2_SCORE_JOB_ID}"
echo "MONITOR=squeue -j ${JCP2_EVAL_JOB_ID},${JCP2_REFERENCE_JOB_ID},${JCP2_PREDICTION_JOB_ID},${JCP2_SCORE_JOB_ID}"
echo "WHEN_COMPLETE_UPLOAD=${JCP2_WORK}/JCP2.zip ${JCP2_WORK}/JCP2.zip.sha256"
