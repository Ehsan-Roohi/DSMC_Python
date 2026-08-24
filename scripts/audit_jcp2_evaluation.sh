#!/usr/bin/env bash
set -u

JCP2_WORK=/project/pi_roohie_umass_edu/DSMC_Python_M3_QY/JCP2
JCP2_ENV="${JCP2_WORK}/LAST_JCP2.env"

[[ -f "${JCP2_ENV}" ]] || {
    echo "MISSING=${JCP2_ENV}" >&2
    exit 2
}
# shellcheck disable=SC1090
source "${JCP2_ENV}"

echo "JCP2_EVAL_JOB_ID=${JCP2_EVAL_JOB_ID}"
echo "JCP2_REFERENCE_JOB_ID=${JCP2_REFERENCE_JOB_ID}"
echo "JCP2_PREDICTION_JOB_ID=${JCP2_PREDICTION_JOB_ID}"
echo "JCP2_SCORE_JOB_ID=${JCP2_SCORE_JOB_ID}"

echo "=== EVALUATION ARRAY ACCOUNTING ==="
sacct -X -j "${JCP2_EVAL_JOB_ID}" --format=JobID%24,JobName%16,State%22,Elapsed,ExitCode,NodeList || true

echo "=== EVALUATION ARTIFACT COUNTS ==="
JCP2_EVAL_MANIFESTS="$(find "${JCP2_WORK}/runs/evaluation" -type f -name artifact_manifest.json 2>/dev/null | wc -l)"
JCP2_EVAL_SUMMARIES="$(find "${JCP2_WORK}/runs/evaluation" -type f -name summary.json 2>/dev/null | wc -l)"
echo "artifact_manifests=${JCP2_EVAL_MANIFESTS}"
echo "summaries=${JCP2_EVAL_SUMMARIES}"

echo "=== EVALUATION DIRECTORIES ==="
find "${JCP2_WORK}/runs/evaluation" -mindepth 1 -maxdepth 2 -printf '%y\t%TY-%Tm-%TdT%TH:%TM:%TS\t%s\t%p\n' 2>/dev/null | sort || true

echo "=== NONEMPTY EVALUATION ERROR LOGS ==="
JCP2_NONEMPTY_ERRORS=0
shopt -s nullglob
for JCP2_LOG in "${JCP2_WORK}"/logs/j2-eval_"${JCP2_EVAL_JOB_ID}"_*.err; do
    if [[ -s "${JCP2_LOG}" ]]; then
        JCP2_NONEMPTY_ERRORS=$((JCP2_NONEMPTY_ERRORS + 1))
        echo "--- ${JCP2_LOG} ---"
        tail -n 60 "${JCP2_LOG}"
    fi
done
echo "nonempty_error_logs=${JCP2_NONEMPTY_ERRORS}"

echo "=== EVALUATION STDOUT TAILS ==="
for JCP2_LOG in "${JCP2_WORK}"/logs/j2-eval_"${JCP2_EVAL_JOB_ID}"_*.out; do
    echo "--- ${JCP2_LOG} ---"
    tail -n 12 "${JCP2_LOG}"
done

echo "JCP2_EVALUATION_AUDIT_COMPLETE=1"
