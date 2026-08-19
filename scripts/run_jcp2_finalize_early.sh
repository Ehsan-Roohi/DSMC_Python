#!/usr/bin/env bash
set -Eeuo pipefail

JCP2_WORK="${JCP2_RECOVERY_WORK_OVERRIDE:-/project/pi_roohie_umass_edu/DSMC_Python_M3_QY/JCP2}"
JCP2_SOURCE="${JCP2_WORK}/src"
JCP2_RUN_ROOT="${JCP2_WORK}/runs"
JCP2_PREDICTION_ROOT="${JCP2_WORK}/predictions"
JCP2_SCORE_ROOT="${JCP2_WORK}/score"
JCP2_ORIGINAL_REPO="${JCP2_RECOVERY_ORIGINAL_REPO_OVERRIDE:-/project/pi_roohie_umass_edu/DSMC_Python_M3_QY/vision_guided_dsmc}"
JCP2_ENV="${JCP2_WORK}/LAST_JCP2.env"
JCP2_RECOVERY_ENV="${JCP2_WORK}/LAST_JCP2_RECOVERY.env"
JCP2_RECOVERY_CODE_COMMIT=a9b66ecc851580aedffffbb482008a0863630452
JCP2_RECOVERY_RAW="${JCP2_RECOVERY_RAW_OVERRIDE:-https://raw.githubusercontent.com/Ehsan-Roohi/DSMC_Python/${JCP2_RECOVERY_CODE_COMMIT}}"

trap 'JCP2_RC=$?; echo "JCP2_RECOVERY_FAILED rc=${JCP2_RC} line=${LINENO} command=${BASH_COMMAND}" >&2; exit "${JCP2_RC}"' ERR

for JCP2_COMMAND in squeue sacct sbatch scancel curl; do
    command -v "${JCP2_COMMAND}" >/dev/null || {
        echo "MISSING_COMMAND=${JCP2_COMMAND}" >&2
        exit 2
    }
done
[[ -f "${JCP2_ENV}" ]] || { echo "MISSING=${JCP2_ENV}" >&2; exit 2; }
[[ -d "${JCP2_SOURCE}/vgdsmc" ]] || { echo "MISSING=${JCP2_SOURCE}/vgdsmc" >&2; exit 2; }

# shellcheck disable=SC1090
source "${JCP2_ENV}"
: "${JCP2_EVAL_JOB_ID:?}"
: "${JCP2_REFERENCE_JOB_ID:?}"
: "${JCP2_PREDICTION_JOB_ID:?}"
: "${JCP2_SCORE_JOB_ID:?}"

jcp2_queue_state() {
    squeue -h -j "$1" -o '%T' 2>/dev/null | head -n 1 || true
}

jcp2_accounting_state() {
    local state
    state="$(sacct -n -X -j "$1" --format=State -P 2>/dev/null | head -n 1 | cut -d'|' -f1 || true)"
    state="${state%%+*}"
    state="${state%% *}"
    printf '%s\n' "${state}"
}

jcp2_job_state() {
    local state
    state="$(jcp2_queue_state "$1")"
    if [[ -z "${state}" ]]; then
        state="$(jcp2_accounting_state "$1")"
    fi
    printf '%s\n' "${state:-UNKNOWN}"
}

jcp2_terminal_failure() {
    case "$1" in
        CANCELLED|FAILED|TIMEOUT|OUT_OF_MEMORY|NODE_FAIL|BOOT_FAIL|DEADLINE|PREEMPTED)
            return 0
            ;;
        *)
            return 1
            ;;
    esac
}

if [[ -s "${JCP2_WORK}/JCP2.zip" && -s "${JCP2_WORK}/JCP2.zip.sha256" ]]; then
    cd "${JCP2_WORK}"
    sha256sum -c JCP2.zip.sha256
    echo "JCP2_ALREADY_COMPLETE=1"
    echo "UPLOAD=${JCP2_WORK}/JCP2.zip ${JCP2_WORK}/JCP2.zip.sha256"
    exit 0
fi

JCP2_RECOVERY_PREDICTION_JOB_ID=
JCP2_RECOVERY_SCORE_JOB_ID=
if [[ -f "${JCP2_RECOVERY_ENV}" ]]; then
    set +u
    # shellcheck disable=SC1090
    source "${JCP2_RECOVERY_ENV}"
    set -u
fi

if [[ -n "${JCP2_RECOVERY_SCORE_JOB_ID}" ]]; then
    JCP2_SCORE_INITIAL_STATE="$(jcp2_job_state "${JCP2_RECOVERY_SCORE_JOB_ID}")"
    case "${JCP2_SCORE_INITIAL_STATE}" in
        RUNNING|COMPLETING|CONFIGURING|PENDING)
            echo "JCP2_RECOVERY_ALREADY_ACTIVE=1"
            echo "JCP2_SCORE_JOB_ID=${JCP2_RECOVERY_SCORE_JOB_ID}"
            echo "JCP2_SCORE_STATE=${JCP2_SCORE_INITIAL_STATE}"
            echo "MONITOR=squeue -j ${JCP2_RECOVERY_SCORE_JOB_ID},${JCP2_RECOVERY_PREDICTION_JOB_ID:-${JCP2_PREDICTION_JOB_ID}}"
            exit 0
            ;;
        COMPLETED)
            echo "JCP2_SCORE_COMPLETED_BUT_ARCHIVE_MISSING=1" >&2
            exit 3
            ;;
    esac
fi

JCP2_VENV=
if [[ -f "${JCP2_ORIGINAL_REPO}/LAST_MOHAMMADZADEH_MV9_HEAT_FLUX_JOB.env" ]]; then
    set +u
    # shellcheck disable=SC1091
    source "${JCP2_ORIGINAL_REPO}/LAST_MOHAMMADZADEH_MV9_HEAT_FLUX_JOB.env"
    set -u
    JCP2_VENV="${MV9_VENV_DIR:-}"
fi
if [[ -z "${JCP2_VENV}" || ! -x "${JCP2_VENV}/bin/python" ]]; then
    JCP2_VENV="${JCP2_ORIGINAL_REPO}/.venv-mv1"
fi
JCP2_PYTHON="${JCP2_VENV}/bin/python"
[[ -x "${JCP2_PYTHON}" ]] || { echo "MISSING_PYTHON=${JCP2_PYTHON}" >&2; exit 4; }

JCP2_RECOVERY_FILES=(
    vgdsmc/jcp_phase1_recovery.py
    scripts/unity_jcp2_predict_recovery.sbatch
    scripts/unity_jcp2_score_recovery.sbatch
)
for JCP2_RECOVERY_FILE in "${JCP2_RECOVERY_FILES[@]}"; do
    mkdir -p "${JCP2_SOURCE}/$(dirname "${JCP2_RECOVERY_FILE}")"
    curl --retry 3 -fsSL "${JCP2_RECOVERY_RAW}/${JCP2_RECOVERY_FILE}" -o "${JCP2_SOURCE}/${JCP2_RECOVERY_FILE}"
done
PYTHONPATH="${JCP2_SOURCE}" "${JCP2_PYTHON}" -m py_compile "${JCP2_SOURCE}/vgdsmc/jcp_phase1_recovery.py"

JCP2_READINESS_FILE="$(mktemp "${TMPDIR:-/tmp}/jcp2-recovery-readiness.XXXXXX")"
trap 'rm -f "${JCP2_READINESS_FILE:-}"' EXIT
if ! JCP2_SELECTION_AUDIT_ROOT="${JCP2_WORK}" PYTHONPATH="${JCP2_SOURCE}" "${JCP2_PYTHON}" - "${JCP2_WORK}" >"${JCP2_READINESS_FILE}" <<'PY'
import sys
from pathlib import Path

from vgdsmc import jcp_phase1_cavity as jcp2
from vgdsmc import jcp_phase1_recovery as recovery

work = Path(sys.argv[1])
evaluation = recovery.qc_audit(work / "runs", "evaluation", 8)
evaluation_seeds = [int(record["seed"]) for record in evaluation["accepted"]]

prediction_dir = work / "predictions"
prediction_path = prediction_dir / "predictions.npz"
prediction_lock_path = prediction_dir / "prediction_lock.json"
has_prediction = prediction_path.is_file()
has_lock = prediction_lock_path.is_file()
if has_prediction != has_lock:
    raise ValueError("JCP2 prediction directory is partial")
if not has_prediction and prediction_dir.exists() and any(prediction_dir.iterdir()):
    raise ValueError("JCP2 prediction directory contains unrecognized partial output")

print("JCP2_QC_READY=1")
print("JCP2_READY_EVALUATION_COUNT=8")
print("JCP2_SELECTED_EVALUATION_SEEDS=" + ",".join(map(str, evaluation_seeds)))

if has_prediction:
    prediction_lock = jcp2._json(prediction_lock_path)
    if prediction_lock.get("status") != "predictions_locked_before_reference_interface":
        raise ValueError("JCP2 prediction lock has an invalid status")
    if jcp2._sha256(prediction_path) != prediction_lock.get("prediction_sha256"):
        raise ValueError("JCP2 prediction hash does not match its lock")
    locked_seeds = [int(seed) for seed in prediction_lock.get("selected_evaluation_seeds", [])]
    if evaluation_seeds != locked_seeds:
        raise ValueError("locked prediction seeds differ from the preregistered QC selection")
    references = recovery.qc_audit(work / "runs", "reference", 20)
    reference_seeds = [int(record["seed"]) for record in references["accepted"]]
    print("JCP2_PREDICTION_READY=1")
    print("JCP2_PREDICTION_SHA256=" + str(prediction_lock["prediction_sha256"]))
    print("JCP2_READY_REFERENCE_COUNT=20")
    print("JCP2_SELECTED_REFERENCE_SEEDS=" + ",".join(map(str, reference_seeds)))
else:
    print("JCP2_PREDICTION_READY=0")
    print("JCP2_REFERENCE_AUDIT=deferred_until_prediction_lock")
PY
then
    echo "JCP2_QC_READY=0" >&2
    echo "Eight QC-pass evaluation units are not available, or locked artifacts are invalid." >&2
    exit 10
fi
cat "${JCP2_READINESS_FILE}"
# The readiness file contains only script-generated numeric, comma-separated,
# hash, and 0/1 values.
# shellcheck disable=SC1090
source "${JCP2_READINESS_FILE}"

if [[ -d "${JCP2_SCORE_ROOT}" ]] && find "${JCP2_SCORE_ROOT}" -mindepth 1 -print -quit | grep -q .; then
    echo "REFUSING_PARTIAL_SCORE_DIRECTORY=${JCP2_SCORE_ROOT}" >&2
    exit 4
fi

jcp2_prepare_prediction_context() {
    JCP2_MV15C_ROOT=
    MV15C_A1_OUTPUT_ROOT=
    MV15C_OUTPUT_ROOT=
    if [[ -f "${JCP2_ORIGINAL_REPO}/LAST_MOHAMMADZADEH_MV15C_A1_QY_RESULT.env" ]]; then
        set +u
        # shellcheck disable=SC1091
        source "${JCP2_ORIGINAL_REPO}/LAST_MOHAMMADZADEH_MV15C_A1_QY_RESULT.env"
        set -u
        JCP2_MV15C_ROOT="${MV15C_A1_OUTPUT_ROOT:-${MV15C_OUTPUT_ROOT:-}}"
    elif [[ -f "${JCP2_ORIGINAL_REPO}/LAST_MOHAMMADZADEH_MV15C_FRESH_B3_RESULT.env" ]]; then
        set +u
        # shellcheck disable=SC1091
        source "${JCP2_ORIGINAL_REPO}/LAST_MOHAMMADZADEH_MV15C_FRESH_B3_RESULT.env"
        set -u
        JCP2_MV15C_ROOT="${MV15C_A1_OUTPUT_ROOT:-${MV15C_OUTPUT_ROOT:-}}"
    fi
    [[ -n "${JCP2_MV15C_ROOT}" ]] || {
        echo "MISSING_MV15C_RESULT_POINTER=1" >&2
        return 11
    }
    [[ -f "${JCP2_MV15C_ROOT}/submission_lock.json" ]] || {
        echo "MISSING=${JCP2_MV15C_ROOT}/submission_lock.json" >&2
        return 11
    }
    [[ -f "${JCP2_MV15C_ROOT}/locked_fresh_predictions.npz" ]] || {
        echo "MISSING=${JCP2_MV15C_ROOT}/locked_fresh_predictions.npz" >&2
        return 11
    }
    JCP2_MV9_ROOT="$("${JCP2_PYTHON}" -c 'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["mv9_output_root"])' "${JCP2_MV15C_ROOT}/submission_lock.json")"
    [[ -f "${JCP2_MV9_ROOT}/dataset.npz" ]] || {
        echo "MISSING=${JCP2_MV9_ROOT}/dataset.npz" >&2
        return 11
    }
}

JCP2_FINAL_PREDICTION_JOB_ID=
JCP2_PREDICTION_ACTION=already_locked
if [[ "${JCP2_PREDICTION_READY}" != 1 ]]; then
    JCP2_SUBMIT_RECOVERY_PREDICTION=0
    if [[ -n "${JCP2_RECOVERY_PREDICTION_JOB_ID}" ]]; then
        JCP2_PREDICTION_STATE="$(jcp2_job_state "${JCP2_RECOVERY_PREDICTION_JOB_ID}")"
        case "${JCP2_PREDICTION_STATE}" in
            RUNNING|COMPLETING|CONFIGURING|PENDING)
                JCP2_FINAL_PREDICTION_JOB_ID="${JCP2_RECOVERY_PREDICTION_JOB_ID}"
                JCP2_PREDICTION_ACTION=recovery_existing_active
                ;;
            COMPLETED)
                echo "JCP2_PREDICTION_COMPLETED_BUT_ARTIFACTS_MISSING=1" >&2
                exit 11
                ;;
            *)
                if ! jcp2_terminal_failure "${JCP2_PREDICTION_STATE}"; then
                    echo "JCP2_RECOVERY_PREDICTION_STATE_UNKNOWN=${JCP2_PREDICTION_STATE}" >&2
                    exit 11
                fi
                JCP2_SUBMIT_RECOVERY_PREDICTION=1
                ;;
        esac
    else
        JCP2_ORIGINAL_PREDICTION_STATE="$(jcp2_job_state "${JCP2_PREDICTION_JOB_ID}")"
        case "${JCP2_ORIGINAL_PREDICTION_STATE}" in
            RUNNING|COMPLETING|CONFIGURING|PENDING)
                scancel "${JCP2_PREDICTION_JOB_ID}"
                echo "JCP2_CANCELLED_BUGGY_ORIGINAL_PREDICTION=${JCP2_PREDICTION_JOB_ID}"
                JCP2_SUBMIT_RECOVERY_PREDICTION=1
                ;;
            COMPLETED)
                echo "JCP2_PREDICTION_COMPLETED_BUT_ARTIFACTS_MISSING=1" >&2
                exit 11
                ;;
            *)
                if ! jcp2_terminal_failure "${JCP2_ORIGINAL_PREDICTION_STATE}"; then
                    echo "JCP2_ORIGINAL_PREDICTION_STATE_UNKNOWN=${JCP2_ORIGINAL_PREDICTION_STATE}" >&2
                    exit 11
                fi
                JCP2_SUBMIT_RECOVERY_PREDICTION=1
                ;;
        esac
    fi
    if [[ "${JCP2_SUBMIT_RECOVERY_PREDICTION}" == 1 ]]; then
        jcp2_prepare_prediction_context
        cd "${JCP2_WORK}"
        JCP2_FINAL_PREDICTION_JOB_ID="$(sbatch --parsable --job-name=j2-pred-e \
            --export="ALL,JCP2_REPO_ROOT=${JCP2_SOURCE},JCP2_RUN_ROOT=${JCP2_RUN_ROOT},JCP2_PREDICTION_ROOT=${JCP2_PREDICTION_ROOT},JCP2_MV9_ROOT=${JCP2_MV9_ROOT},JCP2_MV15C_ROOT=${JCP2_MV15C_ROOT},JCP2_WORK_ROOT=${JCP2_WORK},JCP2_PYTHON=${JCP2_PYTHON}" \
            "${JCP2_SOURCE}/scripts/unity_jcp2_predict_recovery.sbatch")"
        JCP2_PREDICTION_ACTION=recovery_replacement_submitted
    fi
fi

JCP2_SUBMIT_RECOVERY_SCORE=0
if [[ -n "${JCP2_RECOVERY_SCORE_JOB_ID}" ]]; then
    JCP2_SCORE_STATE="$(jcp2_job_state "${JCP2_RECOVERY_SCORE_JOB_ID}")"
    case "${JCP2_SCORE_STATE}" in
        RUNNING|COMPLETING|CONFIGURING|PENDING)
            JCP2_FINAL_SCORE_JOB_ID="${JCP2_RECOVERY_SCORE_JOB_ID}"
            JCP2_SCORE_ACTION=recovery_existing_active
            ;;
        COMPLETED)
            echo "JCP2_SCORE_COMPLETED_BUT_ARCHIVE_MISSING=1" >&2
            exit 12
            ;;
        *)
            if ! jcp2_terminal_failure "${JCP2_SCORE_STATE}"; then
                echo "JCP2_RECOVERY_SCORE_STATE_UNKNOWN=${JCP2_SCORE_STATE}" >&2
                exit 12
            fi
            JCP2_SUBMIT_RECOVERY_SCORE=1
            ;;
    esac
else
    JCP2_ORIGINAL_SCORE_STATE="$(jcp2_job_state "${JCP2_SCORE_JOB_ID}")"
    case "${JCP2_ORIGINAL_SCORE_STATE}" in
        RUNNING|COMPLETING|CONFIGURING|PENDING)
            scancel "${JCP2_SCORE_JOB_ID}"
            echo "JCP2_CANCELLED_BUGGY_ORIGINAL_SCORE=${JCP2_SCORE_JOB_ID}"
            JCP2_SUBMIT_RECOVERY_SCORE=1
            ;;
        COMPLETED)
            echo "JCP2_SCORE_COMPLETED_BUT_ARCHIVE_MISSING=1" >&2
            exit 12
            ;;
        *)
            if ! jcp2_terminal_failure "${JCP2_ORIGINAL_SCORE_STATE}"; then
                echo "JCP2_ORIGINAL_SCORE_STATE_UNKNOWN=${JCP2_ORIGINAL_SCORE_STATE}" >&2
                exit 12
            fi
            JCP2_SUBMIT_RECOVERY_SCORE=1
            ;;
    esac
fi

if [[ "${JCP2_SUBMIT_RECOVERY_SCORE}" == 1 ]]; then
    JCP2_SCORE_DEPENDENCY=()
    if [[ "${JCP2_PREDICTION_READY}" != 1 ]]; then
        JCP2_SCORE_DEPENDENCY=(--dependency="afterok:${JCP2_FINAL_PREDICTION_JOB_ID}")
    fi
    cd "${JCP2_WORK}"
    JCP2_FINAL_SCORE_JOB_ID="$(sbatch --parsable --job-name=j2-score-e \
        "${JCP2_SCORE_DEPENDENCY[@]}" \
        --export="ALL,JCP2_REPO_ROOT=${JCP2_SOURCE},JCP2_RUN_ROOT=${JCP2_RUN_ROOT},JCP2_PREDICTION_ROOT=${JCP2_PREDICTION_ROOT},JCP2_SCORE_ROOT=${JCP2_SCORE_ROOT},JCP2_WORK_ROOT=${JCP2_WORK},JCP2_PYTHON=${JCP2_PYTHON}" \
        "${JCP2_SOURCE}/scripts/unity_jcp2_score_recovery.sbatch")"
    JCP2_SCORE_ACTION=recovery_replacement_submitted
fi

printf 'JCP2_RECOVERY_PREDICTION_JOB_ID=%q\nJCP2_RECOVERY_SCORE_JOB_ID=%q\nJCP2_PREDICTION_ACTION=%q\nJCP2_SCORE_ACTION=%q\nJCP2_RECOVERY_UTC=%q\n' \
    "${JCP2_FINAL_PREDICTION_JOB_ID}" "${JCP2_FINAL_SCORE_JOB_ID}" \
    "${JCP2_PREDICTION_ACTION}" "${JCP2_SCORE_ACTION}" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    > "${JCP2_RECOVERY_ENV}"

JCP2_PREDICTION_AFTER=locked
if [[ -n "${JCP2_FINAL_PREDICTION_JOB_ID}" ]]; then
    JCP2_PREDICTION_AFTER="$(jcp2_job_state "${JCP2_FINAL_PREDICTION_JOB_ID}")"
fi
JCP2_SCORE_AFTER="$(jcp2_job_state "${JCP2_FINAL_SCORE_JOB_ID}")"

echo "JCP2_RECOVERY_SUBMITTED=1"
echo "JCP2_PREDICTION_ACTION=${JCP2_PREDICTION_ACTION}"
echo "JCP2_PREDICTION_JOB_ID=${JCP2_FINAL_PREDICTION_JOB_ID:-ARTIFACT_ALREADY_LOCKED}"
echo "JCP2_PREDICTION_STATE=${JCP2_PREDICTION_AFTER}"
echo "JCP2_SCORE_ACTION=${JCP2_SCORE_ACTION}"
echo "JCP2_SCORE_JOB_ID=${JCP2_FINAL_SCORE_JOB_ID}"
echo "JCP2_SCORE_STATE=${JCP2_SCORE_AFTER}"
echo "JCP2_REFERENCE_SPARE_24_UNTOUCHED=1"
echo "MONITOR=squeue -j ${JCP2_FINAL_SCORE_JOB_ID},${JCP2_FINAL_PREDICTION_JOB_ID:-${JCP2_PREDICTION_JOB_ID}},${JCP2_REFERENCE_JOB_ID}"
echo "WHEN_COMPLETE_UPLOAD=${JCP2_WORK}/JCP2.zip ${JCP2_WORK}/JCP2.zip.sha256"
