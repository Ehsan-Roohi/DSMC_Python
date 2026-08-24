#!/usr/bin/env bash
set -Eeuo pipefail

JCP2_WORK=/project/pi_roohie_umass_edu/DSMC_Python_M3_QY/JCP2
JCP2_SOURCE="${JCP2_WORK}/src"
JCP2_ENV="${JCP2_WORK}/LAST_JCP2.env"
JCP2_STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
JCP2_AUDIT_DIR="${JCP2_WORK}/salvage_audit_${JCP2_STAMP}"
JCP2_REPORT="${JCP2_AUDIT_DIR}/report.txt"
JCP2_ARCHIVE="${JCP2_WORK}/JCP2_SALVAGE_AUDIT_${JCP2_STAMP}.tar.gz"

trap 'JCP2_RC=$?; echo "JCP2_SALVAGE_AUDIT_FAILED rc=${JCP2_RC} line=${LINENO} command=${BASH_COMMAND}" >&2; exit "${JCP2_RC}"' ERR

[[ -f "${JCP2_ENV}" ]] || { echo "MISSING=${JCP2_ENV}" >&2; exit 2; }
# shellcheck disable=SC1090
source "${JCP2_ENV}"
mkdir -p "${JCP2_AUDIT_DIR}/source"

{
    echo "JCP2_SALVAGE_AUDIT_UTC=${JCP2_STAMP}"
    echo "JCP2_EVAL_JOB_ID=${JCP2_EVAL_JOB_ID}"
    echo "JCP2_REFERENCE_JOB_ID=${JCP2_REFERENCE_JOB_ID}"
    echo "JCP2_PREDICTION_JOB_ID=${JCP2_PREDICTION_JOB_ID}"
    echo "JCP2_SCORE_JOB_ID=${JCP2_SCORE_JOB_ID}"

    echo "=== SLURM ACCOUNTING: EVALUATION ==="
    sacct -X -j "${JCP2_EVAL_JOB_ID}" --format=JobID%24,JobName%16,State%22,Elapsed,ExitCode,NodeList || true
    echo "=== SLURM ACCOUNTING: REFERENCE ==="
    sacct -X -j "${JCP2_REFERENCE_JOB_ID}" --format=JobID%24,JobName%16,State%22,Elapsed,ExitCode,NodeList || true

    echo "=== ARTIFACT INVENTORY ==="
    for JCP2_GROUP in evaluation reference; do
        echo "group=${JCP2_GROUP}"
        echo "manifests=$(find "${JCP2_WORK}/runs/${JCP2_GROUP}" -type f -name artifact_manifest.json 2>/dev/null | wc -l)"
        echo "summaries=$(find "${JCP2_WORK}/runs/${JCP2_GROUP}" -type f -name summary.json 2>/dev/null | wc -l)"
        echo "checkpoints=$(find "${JCP2_WORK}/runs/${JCP2_GROUP}" -type f -name 'checkpoint*.npz' 2>/dev/null | wc -l)"
    done

    echo "=== CHECKPOINT AND NUMPY FILE LOCATIONS ==="
    find "${JCP2_WORK}" -type f \( -name 'checkpoint*.npz' -o -name '*.npz.tmp' -o -name 'predictions.npz' \) -printf '%TY-%Tm-%TdT%TH:%TM:%TS\t%s\t%p\n' 2>/dev/null | sort || true

    echo "=== RUN DIRECTORY INVENTORY ==="
    find "${JCP2_WORK}/runs" -maxdepth 4 -printf '%y\t%TY-%Tm-%TdT%TH:%TM:%TS\t%s\t%p\n' 2>/dev/null | sort || true

    echo "=== NONEMPTY REFERENCE ERROR LOG TAILS ==="
    shopt -s nullglob
    for JCP2_LOG in "${JCP2_WORK}"/logs/j2-ref_"${JCP2_REFERENCE_JOB_ID}"_*.err; do
        if [[ -s "${JCP2_LOG}" ]]; then
            echo "--- ${JCP2_LOG} ---"
            tail -n 30 "${JCP2_LOG}"
        fi
    done

    echo "=== ENGINE CHECKPOINT/EVALUATION CALL SITES ==="
    rg -n 'checkpoint|evaluate_mohammadzadeh_fields|output_dir|artifact_manifest' "${JCP2_SOURCE}/vgdsmc/mohammadzadeh_spatial_refinement.py" || true

    echo "=== ENGINE LINES 280-410 ==="
    nl -ba "${JCP2_SOURCE}/vgdsmc/mohammadzadeh_spatial_refinement.py" | sed -n '280,410p'

    echo "=== VALIDATOR LINES 90-175 ==="
    nl -ba "${JCP2_SOURCE}/vgdsmc/mohammadzadeh_validation.py" | sed -n '90,175p'
} > "${JCP2_REPORT}" 2>&1

for JCP2_FILE in mohammadzadeh_spatial_refinement.py mohammadzadeh_validation.py mohammadzadeh_production.py jcp_phase1_cavity.py ntc_checkpoint.py; do
    [[ -f "${JCP2_SOURCE}/vgdsmc/${JCP2_FILE}" ]] || { echo "MISSING_SOURCE=${JCP2_FILE}" >&2; exit 3; }
    cp "${JCP2_SOURCE}/vgdsmc/${JCP2_FILE}" "${JCP2_AUDIT_DIR}/source/${JCP2_FILE}"
done
cp "${JCP2_ENV}" "${JCP2_AUDIT_DIR}/LAST_JCP2.env"

tar -C "${JCP2_AUDIT_DIR}" -czf "${JCP2_ARCHIVE}" .
sha256sum "${JCP2_ARCHIVE}" > "${JCP2_ARCHIVE}.sha256"

echo "JCP2_SALVAGE_AUDIT_COMPLETE=1"
echo "UPLOAD=${JCP2_ARCHIVE} ${JCP2_ARCHIVE}.sha256"
