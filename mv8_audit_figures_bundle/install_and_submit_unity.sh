#!/usr/bin/env bash
set -euo pipefail

BUNDLE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_TARGET="/project/pi_roohie_umass_edu/DSMC_Python_M3_QY/vision_guided_dsmc"
TARGET_ROOT="${1:-${MV8_TARGET_ROOT:-${DEFAULT_TARGET}}}"

test -d "${TARGET_ROOT}"
test -f "${TARGET_ROOT}/LAST_MOHAMMADZADEH_MV8_KINETIC_JOB.env"

install -D -m 0755 \
  "${BUNDLE_ROOT}/payload/scripts/make_mohammadzadeh_mv8_audit_figures.py" \
  "${TARGET_ROOT}/scripts/make_mohammadzadeh_mv8_audit_figures.py"
install -D -m 0755 \
  "${BUNDLE_ROOT}/payload/scripts/unity_mohammadzadeh_mv8_audit_figures.sbatch" \
  "${TARGET_ROOT}/scripts/unity_mohammadzadeh_mv8_audit_figures.sbatch"

cd "${TARGET_ROOT}"
set -a
source LAST_MOHAMMADZADEH_MV8_KINETIC_JOB.env
set +a

test -x "${MV8_VENV_DIR}/bin/python"
test -f "${MV8_OUTPUT_ROOT}/dataset.npz"
test -f "${MV8_OUTPUT_ROOT}/assembly_summary.json"
"${MV8_VENV_DIR}/bin/python" -m py_compile scripts/make_mohammadzadeh_mv8_audit_figures.py

ENV_FILE="${TARGET_ROOT}/LAST_MOHAMMADZADEH_MV8_AUDIT_FIGURES_JOB.env"
if [[ -f "${ENV_FILE}" && "${MV8_AUDIT_FIGURES_ALLOW_NEW_RUN:-0}" != "1" ]]; then
  echo "Refusing duplicate MV8 audit-figure submission; inspect ${ENV_FILE}" >&2
  echo "Set MV8_AUDIT_FIGURES_ALLOW_NEW_RUN=1 only for an intentional rerun" >&2
  exit 3
fi

RUN_TAG="${MV8_AUDIT_FIGURE_TAG:-$(date -u +%Y%m%dT%H%M%SZ)}"
FIGURE_OUTPUT="${MV8_OUTPUT_ROOT}/audit_physical_figures_${RUN_TAG}"
test ! -e "${FIGURE_OUTPUT}"
mkdir -p logs

JOB_ID="$(sbatch --parsable \
  --export="ALL,MV8_REPO_ROOT=${TARGET_ROOT},MV8_OUTPUT_ROOT=${MV8_OUTPUT_ROOT},MV8_VENV_DIR=${MV8_VENV_DIR},MV8_AUDIT_FIGURE_OUTPUT=${FIGURE_OUTPUT}" \
  scripts/unity_mohammadzadeh_mv8_audit_figures.sbatch)"

printf 'MV8_AUDIT_FIGURE_JOB_ID=%q\nMV8_AUDIT_FIGURE_OUTPUT=%q\nMV8_AUDIT_FIGURE_ARCHIVE=%q\n' \
  "${JOB_ID}" "${FIGURE_OUTPUT}" \
  "${FIGURE_OUTPUT}/MOHAMMADZADEH_MV8_AUDIT_PHYSICAL_FIGURES.tar.gz" > "${ENV_FILE}"

echo "Submitted MV8 audit-only physical figures: ${JOB_ID}"
echo "Results: ${FIGURE_OUTPUT}"
echo "Saved: ${ENV_FILE}"
echo "Monitor: squeue -j ${JOB_ID}"

