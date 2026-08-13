#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${MV11_PROJECT_ROOT:-/project/pi_roohie_umass_edu/DSMC_Python_M3_QY/vision_guided_dsmc}"
PAYLOAD_ROOT="${PROJECT_ROOT}/mv11_ds2v_cylinder"
DS2V_PROJECT="${MV11_DS2V_PROJECT:-/project/pi_roohie_umass_edu/Ab-initio-shock/ABINITIO_SHOCK_TESTS_v2}"
command -v sbatch >/dev/null
test -d "${PROJECT_ROOT}"
test -d "${PAYLOAD_ROOT}"

SOURCE="${MV11_DS2V_SOURCE:-}"
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
  echo "ERROR: corrected Bird DS2V source was not found." >&2
  echo "Set MV11_DS2V_SOURCE=/absolute/path/to/source.F90 and rerun." >&2
  exit 2
fi

STAMP=$(date +%Y%m%d_%H%M%S)
CAMPAIGN_ROOT="${PROJECT_ROOT}/mv11_ds2v_cylinder_runs/MV11_DS2V_CYLINDER_${STAMP}"
mkdir -p "${CAMPAIGN_ROOT}/logs"
cat > "${CAMPAIGN_ROOT}/seed_table.tsv" <<'EOF'
seed_20260813	20260813
seed_32452843	32452843
seed_49979687	49979687
seed_67867967	67867967
EOF

EXPORTS="ALL,MV11_PROJECT_ROOT=${PROJECT_ROOT},MV11_CAMPAIGN_ROOT=${CAMPAIGN_ROOT},MV11_DS2V_SOURCE=${SOURCE},MV11_PAYLOAD_ROOT=${PAYLOAD_ROOT}"
PREP_JOB=$(sbatch --parsable --export="${EXPORTS}" \
  --output="${CAMPAIGN_ROOT}/logs/prepare_%j.slurm.out" \
  --error="${CAMPAIGN_ROOT}/logs/prepare_%j.slurm.err" \
  "${PAYLOAD_ROOT}/scripts/unity_mv11_prepare.sbatch")
PREP_JOB=${PREP_JOB%%;*}
ARRAY_JOB=$(sbatch --parsable --dependency="afterok:${PREP_JOB}" --array="0-3%${MV11_ARRAY_CONCURRENCY:-2}" \
  --export="${EXPORTS}" \
  --output="${CAMPAIGN_ROOT}/logs/run_%A_%a.slurm.out" \
  --error="${CAMPAIGN_ROOT}/logs/run_%A_%a.slurm.err" \
  "${PAYLOAD_ROOT}/scripts/unity_mv11_run_array.sbatch")
ARRAY_JOB=${ARRAY_JOB%%;*}
POST_JOB=$(sbatch --parsable --dependency="afterany:${ARRAY_JOB}" --export="${EXPORTS}" \
  --output="${CAMPAIGN_ROOT}/logs/post_%j.slurm.out" \
  --error="${CAMPAIGN_ROOT}/logs/post_%j.slurm.err" \
  "${PAYLOAD_ROOT}/scripts/unity_mv11_post.sbatch")
POST_JOB=${POST_JOB%%;*}

cat > "${CAMPAIGN_ROOT}/SUBMISSION.env" <<EOF
MV11_CAMPAIGN_ROOT=${CAMPAIGN_ROOT}
MV11_DS2V_SOURCE=${SOURCE}
MV11_PREP_JOB_ID=${PREP_JOB}
MV11_ARRAY_JOB_ID=${ARRAY_JOB}
MV11_POST_JOB_ID=${POST_JOB}
MV11_JOB_IDS=${PREP_JOB},${ARRAY_JOB},${POST_JOB}
EOF

cat > "${PROJECT_ROOT}/LAST_MV11_DS2V_CYLINDER_JOB.env" <<EOF
MV11_CAMPAIGN_ROOT=${CAMPAIGN_ROOT}
MV11_DS2V_SOURCE=${SOURCE}
MV11_PREP_JOB_ID=${PREP_JOB}
MV11_ARRAY_JOB_ID=${ARRAY_JOB}
MV11_POST_JOB_ID=${POST_JOB}
MV11_JOB_IDS=${PREP_JOB},${ARRAY_JOB},${POST_JOB}
EOF
echo "MV11_SUBMITTED campaign=${CAMPAIGN_ROOT} prep=${PREP_JOB} array=${ARRAY_JOB} post=${POST_JOB}"
