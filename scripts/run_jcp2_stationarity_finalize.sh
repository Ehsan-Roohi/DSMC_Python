#!/usr/bin/env bash
set -Eeuo pipefail

JCP2_REPAIR_CODE_COMMIT=__JCP2_REPAIR_CODE_COMMIT__
JCP2_REPAIR_RAW="${JCP2_REPAIR_RAW_OVERRIDE:-https://raw.githubusercontent.com/Ehsan-Roohi/DSMC_Python/${JCP2_REPAIR_CODE_COMMIT}}"
JCP2_WORK="${JCP2_REPAIR_WORK_OVERRIDE:-/project/pi_roohie_umass_edu/DSMC_Python_M3_QY/JCP2}"
JCP2_SOURCE="${JCP2_WORK}/src"
JCP2_RUN_ROOT="${JCP2_WORK}/runs"
JCP2_PREDICTION_ROOT="${JCP2_WORK}/predictions"
JCP2_SCORE_ROOT="${JCP2_WORK}/score"
JCP2_ORIGINAL_REPO="${JCP2_REPAIR_ORIGINAL_REPO_OVERRIDE:-/project/pi_roohie_umass_edu/DSMC_Python_M3_QY/vision_guided_dsmc}"
JCP2_EVALUATION_REPAIR_AUDIT="${JCP2_WORK}/evaluation_stationarity_implementation_repair_audit.json"
JCP2_REFERENCE_REPAIR_AUDIT="${JCP2_WORK}/reference_stationarity_implementation_repair_audit.json"
JCP2_REPAIR_ENV="${JCP2_WORK}/LAST_JCP2_STATIONARITY_FINALIZE.env"

trap 'JCP2_RC=$?; echo "JCP2_STATIONARITY_FINALIZE_FAILED rc=${JCP2_RC} line=${LINENO} command=${BASH_COMMAND}" >&2; exit "${JCP2_RC}"' ERR

for JCP2_COMMAND in curl python sbatch squeue sha256sum; do
    command -v "${JCP2_COMMAND}" >/dev/null || {
        echo "MISSING_COMMAND=${JCP2_COMMAND}" >&2
        exit 2
    }
done
[[ -d "${JCP2_SOURCE}/vgdsmc" ]] || { echo "MISSING=${JCP2_SOURCE}/vgdsmc" >&2; exit 2; }

jcp2_final_archive_valid() {
    python - "${JCP2_WORK}/JCP2.zip" <<'PY'
import sys
import zipfile
from pathlib import Path

path = Path(sys.argv[1])
required = {
    "summary.json", "metrics.csv", "reference_stats.npz",
    "prediction_lock.json", "manifest.json", "predictions.npz",
}
if not path.is_file() or not zipfile.is_zipfile(path):
    raise SystemExit(1)
with zipfile.ZipFile(path) as stream:
    if not required.issubset(stream.namelist()) or stream.testzip() is not None:
        raise SystemExit(1)
PY
}

if [[ -s "${JCP2_WORK}/JCP2.zip" && -s "${JCP2_WORK}/JCP2.zip.sha256" ]] \
    && jcp2_final_archive_valid \
    && (cd "${JCP2_WORK}" && sha256sum -c JCP2.zip.sha256); then
    echo "JCP2_ALREADY_COMPLETE=1"
    echo "UPLOAD=${JCP2_WORK}/JCP2.zip ${JCP2_WORK}/JCP2.zip.sha256"
    exit 0
fi

jcp2_queue_state() {
    squeue -h -j "$1" -o '%T' 2>/dev/null | head -n 1 || true
}

JCP2_REPAIR_PREDICTION_JOB_ID=
JCP2_REPAIR_REFERENCE_JOB_ID=
JCP2_REPAIR_SCORE_JOB_ID=
if [[ -f "${JCP2_REPAIR_ENV}" ]]; then
    set +u
    # shellcheck disable=SC1090
    source "${JCP2_REPAIR_ENV}"
    set -u
    for JCP2_ID in \
        "${JCP2_REPAIR_PREDICTION_JOB_ID:-}" \
        "${JCP2_REPAIR_REFERENCE_JOB_ID:-}" \
        "${JCP2_REPAIR_SCORE_JOB_ID:-}"; do
        if [[ -n "${JCP2_ID}" ]]; then
            case "$(jcp2_queue_state "${JCP2_ID}")" in
                RUNNING|COMPLETING|CONFIGURING|PENDING)
                    echo "JCP2_STATIONARITY_FINALIZE_ALREADY_ACTIVE=1"
                    echo "MONITOR=squeue -j ${JCP2_REPAIR_PREDICTION_JOB_ID},${JCP2_REPAIR_REFERENCE_JOB_ID},${JCP2_REPAIR_SCORE_JOB_ID}"
                    exit 0
                    ;;
            esac
        fi
    done
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
[[ -x "${JCP2_PYTHON}" ]] || { echo "MISSING_PYTHON=${JCP2_PYTHON}" >&2; exit 3; }

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
[[ -n "${JCP2_MV15C_ROOT}" ]] || { echo "MISSING_MV15C_RESULT_POINTER=1" >&2; exit 3; }
[[ -f "${JCP2_MV15C_ROOT}/submission_lock.json" ]] || { echo "MISSING=${JCP2_MV15C_ROOT}/submission_lock.json" >&2; exit 3; }
[[ -f "${JCP2_MV15C_ROOT}/locked_fresh_predictions.npz" ]] || { echo "MISSING=${JCP2_MV15C_ROOT}/locked_fresh_predictions.npz" >&2; exit 3; }
JCP2_MV9_ROOT="$("${JCP2_PYTHON}" -c 'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["mv9_output_root"])' "${JCP2_MV15C_ROOT}/submission_lock.json")"
[[ -f "${JCP2_MV9_ROOT}/dataset.npz" ]] || { echo "MISSING=${JCP2_MV9_ROOT}/dataset.npz" >&2; exit 3; }

JCP2_REPAIR_FILES=(
    vgdsmc/jcp_phase1_stationarity_repair.py
    tests/test_jcp_phase1_stationarity_repair.py
    scripts/unity_jcp2_reference_stationarity_repair.sbatch
    scripts/unity_jcp2_predict_recovery.sbatch
    scripts/unity_jcp2_score_recovery.sbatch
)
for JCP2_FILE in "${JCP2_REPAIR_FILES[@]}"; do
    mkdir -p "${JCP2_SOURCE}/$(dirname "${JCP2_FILE}")"
    curl --retry 3 -fsSL "${JCP2_REPAIR_RAW}/${JCP2_FILE}" -o "${JCP2_SOURCE}/${JCP2_FILE}"
done
touch "${JCP2_SOURCE}/tests/__init__.py"

PYTHONPATH="${JCP2_SOURCE}" "${JCP2_PYTHON}" -m py_compile \
    "${JCP2_SOURCE}/vgdsmc/jcp_phase1_stationarity_repair.py"
PYTHONPATH="${JCP2_SOURCE}" "${JCP2_PYTHON}" \
    "${JCP2_SOURCE}/tests/test_jcp_phase1_stationarity_repair.py"
PYTHONPATH="${JCP2_SOURCE}" "${JCP2_PYTHON}" \
    -m vgdsmc.jcp_phase1_stationarity_repair \
    --run-root "${JCP2_RUN_ROOT}" \
    --group evaluation \
    --audit "${JCP2_EVALUATION_REPAIR_AUDIT}" \
    --apply > "${JCP2_WORK}/evaluation_stationarity_implementation_repair_stdout.json"

"${JCP2_PYTHON}" - "${JCP2_EVALUATION_REPAIR_AUDIT}" <<'PY'
import json
import sys

audit = json.load(open(sys.argv[1], encoding="utf-8"))
print(f"JCP2_EVALUATION_PASSING={audit['passing_count']}")
print("JCP2_EVALUATION_SELECTED=" + ",".join(map(str, audit["selected_seeds"])))
print("JCP2_EVALUATION_STATIONARITY_REPAIR_READY=" + str(int(audit["selection_complete"])))
PY

for JCP2_OUTPUT_ROOT in "${JCP2_PREDICTION_ROOT}" "${JCP2_SCORE_ROOT}"; do
    if [[ -d "${JCP2_OUTPUT_ROOT}" ]] && find "${JCP2_OUTPUT_ROOT}" -mindepth 1 -print -quit | grep -q .; then
        JCP2_BACKUP_TAG="$(date -u +%Y%m%dT%H%M%SZ)-$$"
        mv "${JCP2_OUTPUT_ROOT}" "${JCP2_OUTPUT_ROOT}.pre-stationarity-repair-${JCP2_BACKUP_TAG}"
        echo "JCP2_PARTIAL_OUTPUT_PRESERVED=${JCP2_OUTPUT_ROOT}.pre-stationarity-repair-${JCP2_BACKUP_TAG}"
    fi
done
mkdir -p "${JCP2_WORK}/logs"

JCP2_COMMON="ALL,JCP2_REPO_ROOT=${JCP2_SOURCE},JCP2_RUN_ROOT=${JCP2_RUN_ROOT},JCP2_PYTHON=${JCP2_PYTHON}"
cd "${JCP2_WORK}"
JCP2_REPAIR_PREDICTION_JOB_ID="$(sbatch --parsable --job-name=j2-pred-s \
    --export="${JCP2_COMMON},JCP2_PREDICTION_ROOT=${JCP2_PREDICTION_ROOT},JCP2_MV9_ROOT=${JCP2_MV9_ROOT},JCP2_MV15C_ROOT=${JCP2_MV15C_ROOT},JCP2_WORK_ROOT=${JCP2_WORK}" \
    "${JCP2_SOURCE}/scripts/unity_jcp2_predict_recovery.sbatch")"
JCP2_REPAIR_REFERENCE_JOB_ID="$(sbatch --parsable --job-name=j2-ref-qc \
    --dependency="afterok:${JCP2_REPAIR_PREDICTION_JOB_ID}" \
    --export="${JCP2_COMMON},JCP2_PREDICTION_ROOT=${JCP2_PREDICTION_ROOT},JCP2_REFERENCE_REPAIR_AUDIT=${JCP2_REFERENCE_REPAIR_AUDIT}" \
    "${JCP2_SOURCE}/scripts/unity_jcp2_reference_stationarity_repair.sbatch")"
JCP2_REPAIR_SCORE_JOB_ID="$(sbatch --parsable --job-name=j2-score-s \
    --dependency="afterok:${JCP2_REPAIR_REFERENCE_JOB_ID}" \
    --export="${JCP2_COMMON},JCP2_PREDICTION_ROOT=${JCP2_PREDICTION_ROOT},JCP2_SCORE_ROOT=${JCP2_SCORE_ROOT},JCP2_WORK_ROOT=${JCP2_WORK}" \
    "${JCP2_SOURCE}/scripts/unity_jcp2_score_recovery.sbatch")"

printf 'JCP2_REPAIR_PREDICTION_JOB_ID=%q\nJCP2_REPAIR_REFERENCE_JOB_ID=%q\nJCP2_REPAIR_SCORE_JOB_ID=%q\nJCP2_REPAIR_CODE_COMMIT=%q\nJCP2_REPAIR_UTC=%q\n' \
    "${JCP2_REPAIR_PREDICTION_JOB_ID}" "${JCP2_REPAIR_REFERENCE_JOB_ID}" "${JCP2_REPAIR_SCORE_JOB_ID}" \
    "${JCP2_REPAIR_CODE_COMMIT}" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    > "${JCP2_REPAIR_ENV}"

echo "JCP2_STATIONARITY_FINALIZE_SUBMITTED=1"
echo "JCP2_PREDICTION_JOB_ID=${JCP2_REPAIR_PREDICTION_JOB_ID}"
echo "JCP2_REFERENCE_REPAIR_JOB_ID=${JCP2_REPAIR_REFERENCE_JOB_ID}"
echo "JCP2_SCORE_JOB_ID=${JCP2_REPAIR_SCORE_JOB_ID}"
echo "MONITOR=squeue -j ${JCP2_REPAIR_PREDICTION_JOB_ID},${JCP2_REPAIR_REFERENCE_JOB_ID},${JCP2_REPAIR_SCORE_JOB_ID}"
echo "WHEN_COMPLETE_UPLOAD=${JCP2_WORK}/JCP2.zip ${JCP2_WORK}/JCP2.zip.sha256"
