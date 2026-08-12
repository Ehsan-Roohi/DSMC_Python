#!/usr/bin/env bash
set -euo pipefail

BUNDLE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_TARGET="/project/pi_roohie_umass_edu/DSMC_Python_M3_QY/vision_guided_dsmc"
TARGET_ROOT="${1:-${MV6_PUBLICATION_TARGET_ROOT:-${DEFAULT_TARGET}}}"

test -d "${TARGET_ROOT}"
test -f "${BUNDLE_ROOT}/payload/scripts/make_mv6_publication_suite.py"
test -f "${BUNDLE_ROOT}/payload/scripts/unity_mohammadzadeh_mv6_publication_figures.sbatch"

if [[ ! -f "${TARGET_ROOT}/LAST_MOHAMMADZADEH_MV6_POSTFIX_JOB.env" \
      && ! -f "${TARGET_ROOT}/LAST_MOHAMMADZADEH_MV5_REPAIR_MV6_JOB.env" ]]; then
    echo "No completed MV6 environment file found in ${TARGET_ROOT}" >&2
    exit 2
fi

install -D -m 0644 \
    "${BUNDLE_ROOT}/payload/scripts/make_mv6_publication_suite.py" \
    "${TARGET_ROOT}/scripts/make_mv6_publication_suite.py"
install -D -m 0755 \
    "${BUNDLE_ROOT}/payload/scripts/unity_mohammadzadeh_mv6_publication_figures.sbatch" \
    "${TARGET_ROOT}/scripts/unity_mohammadzadeh_mv6_publication_figures.sbatch"

cd "${TARGET_ROOT}"
python3 -m py_compile scripts/make_mv6_publication_suite.py
mkdir -p logs

JOB_ID="$(sbatch --parsable \
    --export="ALL,MV6_REPO_ROOT=${TARGET_ROOT}" \
    scripts/unity_mohammadzadeh_mv6_publication_figures.sbatch)"

ENV_FILE="${TARGET_ROOT}/LAST_MOHAMMADZADEH_MV6_PUBLICATION_JOB.env"
printf 'MV6_PUBLICATION_JOB_ID=%q\nMV6_REPO_ROOT=%q\n' \
    "${JOB_ID}" "${TARGET_ROOT}" > "${ENV_FILE}"

echo "Submitted MV6 publication-figure job: ${JOB_ID}"
echo "Saved: ${ENV_FILE}"
echo "Check: squeue -j ${JOB_ID}"
echo "Log:   ${TARGET_ROOT}/logs/moh_mv6_pub_${JOB_ID}.out"
