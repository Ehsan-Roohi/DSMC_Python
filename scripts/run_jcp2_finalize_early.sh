#!/usr/bin/env bash
set -Eeuo pipefail

JCP2_WORK=/project/pi_roohie_umass_edu/DSMC_Python_M3_QY/JCP2
JCP2_SOURCE="${JCP2_WORK}/src"
JCP2_ORIGINAL_REPO=/project/pi_roohie_umass_edu/DSMC_Python_M3_QY/vision_guided_dsmc
JCP2_ENV="${JCP2_WORK}/LAST_JCP2.env"

trap 'JCP2_RC=$?; echo "JCP2_EARLY_FINALIZER_FAILED rc=${JCP2_RC} line=${LINENO} command=${BASH_COMMAND}" >&2; exit "${JCP2_RC}"' ERR

command -v squeue >/dev/null || { echo "MISSING_COMMAND=squeue" >&2; exit 2; }
command -v scontrol >/dev/null || { echo "MISSING_COMMAND=scontrol" >&2; exit 2; }
[[ -f "${JCP2_ENV}" ]] || { echo "MISSING=${JCP2_ENV}" >&2; exit 2; }
[[ -d "${JCP2_SOURCE}/vgdsmc" ]] || { echo "MISSING=${JCP2_SOURCE}/vgdsmc" >&2; exit 2; }

# shellcheck disable=SC1090
source "${JCP2_ENV}"
: "${JCP2_EVAL_JOB_ID:?}"
: "${JCP2_REFERENCE_JOB_ID:?}"
: "${JCP2_PREDICTION_JOB_ID:?}"
: "${JCP2_SCORE_JOB_ID:?}"

if [[ -s "${JCP2_WORK}/JCP2.zip" && -s "${JCP2_WORK}/JCP2.zip.sha256" ]]; then
    cd "${JCP2_WORK}"
    sha256sum -c JCP2.zip.sha256
    echo "JCP2_ALREADY_COMPLETE=1"
    echo "UPLOAD=${JCP2_WORK}/JCP2.zip ${JCP2_WORK}/JCP2.zip.sha256"
    exit 0
fi

JCP2_SCORE_QUEUE_STATE="$(squeue -h -j "${JCP2_SCORE_JOB_ID}" -o '%T' 2>/dev/null | head -n 1)"
case "${JCP2_SCORE_QUEUE_STATE}" in
    RUNNING|COMPLETING|CONFIGURING)
        echo "JCP2_SCORE_ALREADY_ACTIVE=1"
        echo "JCP2_SCORE_JOB_ID=${JCP2_SCORE_JOB_ID}"
        echo "MONITOR=squeue -j ${JCP2_SCORE_JOB_ID}"
        exit 0
        ;;
    PENDING)
        ;;
    "")
        JCP2_ACCOUNTING_STATE="$(sacct -n -X -j "${JCP2_SCORE_JOB_ID}" --format=State -P 2>/dev/null | head -n 1 | cut -d'|' -f1 | cut -d'+' -f1)"
        if [[ "${JCP2_ACCOUNTING_STATE}" == COMPLETED ]]; then
            echo "JCP2_SCORE_COMPLETED_BUT_ARCHIVE_MISSING=1" >&2
        else
            echo "JCP2_SCORE_NOT_PENDING state=${JCP2_ACCOUNTING_STATE:-UNKNOWN}" >&2
        fi
        exit 3
        ;;
    *)
        echo "JCP2_SCORE_NOT_RELEASABLE state=${JCP2_SCORE_QUEUE_STATE}" >&2
        exit 3
        ;;
esac

if [[ -d "${JCP2_WORK}/score" ]] && find "${JCP2_WORK}/score" -mindepth 1 -print -quit | grep -q .; then
    echo "REFUSING_PARTIAL_SCORE_DIRECTORY=${JCP2_WORK}/score" >&2
    exit 4
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

JCP2_READINESS_FILE="$(mktemp "${TMPDIR:-/tmp}/jcp2-early-readiness.XXXXXX")"
trap 'rm -f "${JCP2_READINESS_FILE:-}"' EXIT
if ! PYTHONPATH="${JCP2_SOURCE}" "${JCP2_PYTHON}" - "${JCP2_WORK}" >"${JCP2_READINESS_FILE}" <<'PY'
import sys
from pathlib import Path

from vgdsmc import jcp_phase1_cavity as jcp2

work = Path(sys.argv[1])
prediction_dir = work / "predictions"
prediction_path = prediction_dir / "predictions.npz"
prediction_lock_path = prediction_dir / "prediction_lock.json"

if not prediction_path.is_file() or not prediction_lock_path.is_file():
    raise FileNotFoundError("locked JCP2 prediction artifacts are not complete")
prediction_lock = jcp2._json(prediction_lock_path)
if prediction_lock.get("status") != "predictions_locked_before_reference_interface":
    raise ValueError("JCP2 prediction lock has an invalid status")
if jcp2._sha256(prediction_path) != prediction_lock.get("prediction_sha256"):
    raise ValueError("JCP2 prediction hash does not match its lock")

evaluation = jcp2._qc_selected(work / "runs", "evaluation", 8)
references = jcp2._qc_selected(work / "runs", "reference", 20)
evaluation_seeds = [int(jcp2._json(path / "summary.json")["seed"]) for path in evaluation]
reference_seeds = [int(jcp2._json(path / "summary.json")["seed"]) for path in references]
if evaluation_seeds != [int(seed) for seed in prediction_lock.get("selected_evaluation_seeds", [])]:
    raise ValueError("locked prediction seeds differ from the preregistered QC selection")

print("JCP2_EARLY_READY=1")
print("JCP2_READY_EVALUATION_COUNT=8")
print("JCP2_READY_REFERENCE_COUNT=20")
print("JCP2_SELECTED_EVALUATION_SEEDS=" + ",".join(map(str, evaluation_seeds)))
print("JCP2_SELECTED_REFERENCE_SEEDS=" + ",".join(map(str, reference_seeds)))
print("JCP2_PREDICTION_SHA256=" + str(prediction_lock["prediction_sha256"]))
PY
then
    echo "JCP2_EARLY_READY=0" >&2
    echo "The locked prediction, eight QC-pass evaluation units, or twenty QC-pass references are not ready yet." >&2
    exit 10
fi
cat "${JCP2_READINESS_FILE}"

# Recheck immediately before changing the dependency so a scheduler transition
# cannot create a second scoring job or a competing writer.
JCP2_SCORE_QUEUE_STATE="$(squeue -h -j "${JCP2_SCORE_JOB_ID}" -o '%T' 2>/dev/null | head -n 1)"
if [[ "${JCP2_SCORE_QUEUE_STATE}" != PENDING ]]; then
    echo "JCP2_SCORE_STATE_CHANGED=${JCP2_SCORE_QUEUE_STATE:-LEFT_QUEUE}"
    echo "No dependency was changed. Re-run this finalizer to inspect the new state."
    exit 0
fi

# The readiness checks above reproduce the frozen scorer's input gates.  Once
# they pass, the last reference-array spare is no longer an input dependency.
scontrol update JobId="${JCP2_SCORE_JOB_ID}" Dependency=

printf 'JCP2_EARLY_RELEASED=1\nJCP2_SCORE_JOB_ID=%q\nJCP2_RELEASED_UTC=%q\n' \
    "${JCP2_SCORE_JOB_ID}" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    > "${JCP2_WORK}/LAST_JCP2_EARLY_RELEASE.env"

JCP2_AFTER="$(squeue -h -j "${JCP2_SCORE_JOB_ID}" -o '%T|%r' 2>/dev/null | head -n 1)"
echo "JCP2_SCORE_DEPENDENCY_RELEASED=1"
echo "JCP2_SCORE_JOB_ID=${JCP2_SCORE_JOB_ID}"
echo "JCP2_SCORE_STATE=${JCP2_AFTER:-LEFT_QUEUE_CHECK_SACCT}"
echo "JCP2_REFERENCE_SPARE_24_UNTOUCHED=1"
echo "MONITOR=squeue -j ${JCP2_SCORE_JOB_ID},${JCP2_REFERENCE_JOB_ID}"
echo "WHEN_COMPLETE_UPLOAD=${JCP2_WORK}/JCP2.zip ${JCP2_WORK}/JCP2.zip.sha256"
