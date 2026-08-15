#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${MV17A_PROJECT_ROOT:-/project/pi_roohie_umass_edu/DSMC_Python_M3_QY/vision_guided_dsmc}"
PAYLOAD_ROOT="${PROJECT_ROOT}/mv17a_cylinder_native_crossfit"
SOURCE_POINTER="${PROJECT_ROOT}/LAST_MOHAMMADZADEH_MV16B_JCP_EVIDENCE_RESULT.env"
JOB_POINTER="${PROJECT_ROOT}/LAST_MOHAMMADZADEH_MV17A_CYLINDER_NATIVE_JOB.env"
RESULT_POINTER="${PROJECT_ROOT}/LAST_MOHAMMADZADEH_MV17A_CYLINDER_NATIVE_RESULT.env"

command -v sbatch >/dev/null
test -s "${SOURCE_POINTER}"
source "${SOURCE_POINTER}"
: "${MV16B_OUTPUT_ROOT:?}"
test -s "${MV16B_OUTPUT_ROOT}/artifact_manifest.json"
test -s "${MV16B_OUTPUT_ROOT}/summary.json"
for seed in 20260813 32452843 49979687 67867967; do
  test -s "${MV16B_OUTPUT_ROOT}/cylinder_native_fields_seed_${seed}.npz"
done

PYTHON_BIN="${MV17A_PYTHON:-}"
if [[ -z "${PYTHON_BIN}" || ! -x "${PYTHON_BIN}" ]]; then
  echo "MV17A_SUBMIT_ERROR: MV17A_PYTHON is absent or not executable" >&2
  exit 2
fi
if [[ -e "${JOB_POINTER}" && "${MV17A_ALLOW_RESUBMIT:-0}" != "1" ]]; then
  echo "MV17A_SUBMIT_ERROR: refusing duplicate submission; inspect ${JOB_POINTER}" >&2
  exit 3
fi
if [[ -e "${RESULT_POINTER}" ]]; then
  echo "MV17A_SUBMIT_ERROR: result pointer already exists; refusing overwrite" >&2
  exit 4
fi

STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUTPUT_ROOT="${MV17A_OUTPUT_ROOT:-${PROJECT_ROOT}/results/mohammadzadeh_2012/mv17a_cylinder_native_crossfit/run_${STAMP}}"
test ! -e "${OUTPUT_ROOT}"
mkdir -p "${PROJECT_ROOT}/logs"

cd "${PROJECT_ROOT}"
export PATH="$(dirname "${PYTHON_BIN}"):${PATH}"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
"${PYTHON_BIN}" -m vgdsmc.mohammadzadeh_mv17a_cylinder_native_crossfit verify

EXPORTS="ALL,MV17A_PROJECT_ROOT=${PROJECT_ROOT},MV17A_PAYLOAD_ROOT=${PAYLOAD_ROOT},MV17A_OUTPUT_ROOT=${OUTPUT_ROOT},MV17A_SOURCE_ROOT=${MV16B_OUTPUT_ROOT},MV17A_PYTHON=${PYTHON_BIN}"
AUDIT_JOB="$(sbatch --parsable \
  --export="${EXPORTS}" \
  --output="${PROJECT_ROOT}/logs/moh_mv17a_audit_%j.slurm.out" \
  --error="${PROJECT_ROOT}/logs/moh_mv17a_audit_%j.slurm.err" \
  "${PAYLOAD_ROOT}/scripts/unity_mohammadzadeh_mv17a_audit.sbatch")"
AUDIT_JOB="${AUDIT_JOB%%;*}"
POST_JOB="$(sbatch --parsable --dependency="afterok:${AUDIT_JOB}" \
  --export="${EXPORTS},MV17A_AUDIT_JOB_ID=${AUDIT_JOB}" \
  --output="${PROJECT_ROOT}/logs/moh_mv17a_post_%j.slurm.out" \
  --error="${PROJECT_ROOT}/logs/moh_mv17a_post_%j.slurm.err" \
  "${PAYLOAD_ROOT}/scripts/unity_mohammadzadeh_mv17a_post.sbatch")"
POST_JOB="${POST_JOB%%;*}"

TEMPORARY="${JOB_POINTER}.tmp"
{
  printf 'MV17A_OUTPUT_ROOT=%s\n' "${OUTPUT_ROOT}"
  printf 'MV17A_SOURCE_ROOT=%s\n' "${MV16B_OUTPUT_ROOT}"
  printf 'MV17A_PYTHON=%s\n' "${PYTHON_BIN}"
  printf 'MV17A_AUDIT_JOB_ID=%s\n' "${AUDIT_JOB}"
  printf 'MV17A_POST_JOB_ID=%s\n' "${POST_JOB}"
  printf 'MV17A_JOB_IDS=%s,%s\n' "${AUDIT_JOB}" "${POST_JOB}"
  printf 'MV17A_DSMC_RERUN=false\n'
  printf 'MV17A_NEURAL_TRAINING=false\n'
} > "${TEMPORARY}"
mv "${TEMPORARY}" "${JOB_POINTER}"
echo "MV17A_CYLINDER_NATIVE_SUBMITTED output=${OUTPUT_ROOT} audit=${AUDIT_JOB} post=${POST_JOB} dsmc_rerun=NO neural_training=NO"

