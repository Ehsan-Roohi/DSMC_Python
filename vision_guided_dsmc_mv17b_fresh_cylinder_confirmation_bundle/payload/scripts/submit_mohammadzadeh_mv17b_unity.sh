#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${MV17B_PROJECT_ROOT:-/project/pi_roohie_umass_edu/DSMC_Python_M3_QY/vision_guided_dsmc}"
PAYLOAD_ROOT="${PROJECT_ROOT}/mv17b_fresh_cylinder_confirmation"
MV16B_POINTER="${PROJECT_ROOT}/LAST_MOHAMMADZADEH_MV16B_JCP_EVIDENCE_RESULT.env"
MV17A_POINTER="${PROJECT_ROOT}/LAST_MOHAMMADZADEH_MV17A_CYLINDER_NATIVE_RESULT.env"
JOB_POINTER="${PROJECT_ROOT}/LAST_MOHAMMADZADEH_MV17B_FRESH_CYLINDER_JOB.env"
RESULT_POINTER="${PROJECT_ROOT}/LAST_MOHAMMADZADEH_MV17B_FRESH_CYLINDER_RESULT.env"

command -v sbatch >/dev/null
test -d "${PROJECT_ROOT}"
test -d "${PAYLOAD_ROOT}"
test -s "${MV16B_POINTER}"
test -s "${MV17A_POINTER}"
source "${MV16B_POINTER}"
source "${MV17A_POINTER}"
: "${MV16B_OUTPUT_ROOT:?}"
: "${MV17A_OUTPUT_ROOT:?}"
test -s "${MV16B_OUTPUT_ROOT}/artifact_manifest.json"
test -s "${MV17A_OUTPUT_ROOT}/artifact_manifest.json"
test "$(jq -r '.decision' "${MV17A_OUTPUT_ROOT}/summary.json")" = \
  "MV17A_retrospective_cylinder_native_crossfit_supports_freezing_for_fresh_confirmation"
test "$(jq -r '.all_gates_pass' "${MV17A_OUTPUT_ROOT}/summary.json")" = "true"

PYTHON_BIN="${MV17B_PYTHON:-}"
if [[ -z "${PYTHON_BIN}" || ! -x "${PYTHON_BIN}" ]]; then
  echo "MV17B_SUBMIT_ERROR: MV17B_PYTHON is absent or not executable" >&2
  exit 2
fi
if [[ -e "${JOB_POINTER}" && "${MV17B_ALLOW_RESUBMIT:-0}" != "1" ]]; then
  echo "MV17B_SUBMIT_ERROR: refusing duplicate submission; inspect ${JOB_POINTER}" >&2
  exit 3
fi
if [[ -e "${RESULT_POINTER}" ]]; then
  echo "MV17B_SUBMIT_ERROR: result pointer already exists; refusing overwrite" >&2
  exit 4
fi

DS2V_PROJECT="${MV17B_DS2V_PROJECT:-/project/pi_roohie_umass_edu/Ab-initio-shock/ABINITIO_SHOCK_TESTS_v2}"
SOURCE="${MV17B_DS2V_SOURCE:-}"
if [[ -z "${SOURCE}" ]]; then
  EXACT="${DS2V_PROJECT}/DS2V_BIRD_M10_FRESH_ONLY_20260722_234904/source/Plasma_Calculations2.bird_m10_fresh.F90"
  if [[ -s "${EXACT}" ]]; then
    SOURCE="${EXACT}"
  else
    SOURCE=$(find "${DS2V_PROJECT}" -maxdepth 3 -type f \
      \( -name 'Plasma_Calculations2.bird_m10_fresh.F90' -o -name 'Plasma_Calculations2.cfsafe.F90' \) \
      -printf '%T@ %p\n' 2>/dev/null | sort -nr | sed -n '1s/^[^ ]* //p')
  fi
fi
if [[ -z "${SOURCE}" || ! -s "${SOURCE}" ]]; then
  echo "MV17B_SUBMIT_ERROR: corrected Bird DS2V source was not found" >&2
  echo "Set MV17B_DS2V_SOURCE=/absolute/path/to/source.F90 and rerun" >&2
  exit 5
fi

STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUTPUT_ROOT="${MV17B_OUTPUT_ROOT:-${PROJECT_ROOT}/results/mohammadzadeh_2012/mv17b_fresh_cylinder_confirmation/run_${STAMP}}"
test ! -e "${OUTPUT_ROOT}"
mkdir -p "${OUTPUT_ROOT}/campaign/logs" "${PROJECT_ROOT}/logs"
cat > "${OUTPUT_ROOT}/seed_table.tsv" <<'EOF'
pair_01_observation	pair_01	observation	171701
pair_01_reference	pair_01	reference	171702
pair_02_observation	pair_02	observation	171703
pair_02_reference	pair_02	reference	171704
pair_03_observation	pair_03	observation	171705
pair_03_reference	pair_03	reference	171706
pair_04_observation	pair_04	observation	171707
pair_04_reference	pair_04	reference	171708
pair_05_observation	pair_05	observation	171709
pair_05_reference	pair_05	reference	171710
pair_06_observation	pair_06	observation	171711
pair_06_reference	pair_06	reference	171712
EOF

cd "${PROJECT_ROOT}"
export PATH="$(dirname "${PYTHON_BIN}"):${PATH}"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
"${PYTHON_BIN}" -m vgdsmc.mohammadzadeh_mv17b_fresh_cylinder_confirmation verify

EXPORTS="ALL,MV17B_PROJECT_ROOT=${PROJECT_ROOT},MV17B_PAYLOAD_ROOT=${PAYLOAD_ROOT},MV17B_OUTPUT_ROOT=${OUTPUT_ROOT},MV17B_MV16B_ROOT=${MV16B_OUTPUT_ROOT},MV17B_MV17A_ROOT=${MV17A_OUTPUT_ROOT},MV17B_DS2V_SOURCE=${SOURCE},MV17B_PYTHON=${PYTHON_BIN}"
PREP_JOB=$(sbatch --parsable --export="${EXPORTS}" \
  --output="${OUTPUT_ROOT}/campaign/logs/prepare_%j.slurm.out" \
  --error="${OUTPUT_ROOT}/campaign/logs/prepare_%j.slurm.err" \
  "${PAYLOAD_ROOT}/scripts/unity_mohammadzadeh_mv17b_prepare.sbatch")
PREP_JOB=${PREP_JOB%%;*}
ARRAY_JOB=$(sbatch --parsable --dependency="afterok:${PREP_JOB}" \
  --array="0-11%${MV17B_ARRAY_CONCURRENCY:-4}" --export="${EXPORTS}" \
  --output="${OUTPUT_ROOT}/campaign/logs/run_%A_%a.slurm.out" \
  --error="${OUTPUT_ROOT}/campaign/logs/run_%A_%a.slurm.err" \
  "${PAYLOAD_ROOT}/scripts/unity_mohammadzadeh_mv17b_run_array.sbatch")
ARRAY_JOB=${ARRAY_JOB%%;*}
ANALYZE_JOB=$(sbatch --parsable --dependency="afterok:${ARRAY_JOB}" --export="${EXPORTS}" \
  --output="${OUTPUT_ROOT}/campaign/logs/analyze_%j.slurm.out" \
  --error="${OUTPUT_ROOT}/campaign/logs/analyze_%j.slurm.err" \
  "${PAYLOAD_ROOT}/scripts/unity_mohammadzadeh_mv17b_analyze.sbatch")
ANALYZE_JOB=${ANALYZE_JOB%%;*}
POST_JOB=$(sbatch --parsable --dependency="afterok:${ANALYZE_JOB}" --export="${EXPORTS}" \
  --output="${OUTPUT_ROOT}/campaign/logs/post_%j.slurm.out" \
  --error="${OUTPUT_ROOT}/campaign/logs/post_%j.slurm.err" \
  "${PAYLOAD_ROOT}/scripts/unity_mohammadzadeh_mv17b_post.sbatch")
POST_JOB=${POST_JOB%%;*}
JOB_IDS="${PREP_JOB},${ARRAY_JOB},${ANALYZE_JOB},${POST_JOB}"

cat > "${OUTPUT_ROOT}/SUBMISSION.env" <<EOF
MV17B_OUTPUT_ROOT=${OUTPUT_ROOT}
MV17B_CAMPAIGN_ROOT=${OUTPUT_ROOT}/campaign
MV17B_MV16B_ROOT=${MV16B_OUTPUT_ROOT}
MV17B_MV17A_ROOT=${MV17A_OUTPUT_ROOT}
MV17B_DS2V_SOURCE=${SOURCE}
MV17B_PYTHON=${PYTHON_BIN}
MV17B_PREP_JOB_ID=${PREP_JOB}
MV17B_ARRAY_JOB_ID=${ARRAY_JOB}
MV17B_ANALYZE_JOB_ID=${ANALYZE_JOB}
MV17B_POST_JOB_ID=${POST_JOB}
MV17B_JOB_IDS=${JOB_IDS}
MV17B_FRESH_DSMC_TRAJECTORIES=12
MV17B_NEURAL_TRAINING=false
MV17B_FRESH_PARAMETER_SELECTION=false
EOF
TEMPORARY="${JOB_POINTER}.tmp"
cp "${OUTPUT_ROOT}/SUBMISSION.env" "${TEMPORARY}"
mv "${TEMPORARY}" "${JOB_POINTER}"
echo "MV17B_FRESH_CYLINDER_SUBMITTED output=${OUTPUT_ROOT} prep=${PREP_JOB} array=${ARRAY_JOB} analyze=${ANALYZE_JOB} post=${POST_JOB} trajectories=12 training=NO"

