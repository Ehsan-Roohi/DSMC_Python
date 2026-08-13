#!/usr/bin/env bash
#SBATCH --job-name=mv11-cyl-post
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=02:00:00

set -euo pipefail
: "${MV11_PROJECT_ROOT:?}"
: "${MV11_CAMPAIGN_ROOT:?}"
: "${MV11_PAYLOAD_ROOT:?}"
mkdir -p "${MV11_CAMPAIGN_ROOT}/analysis" "${MV11_CAMPAIGN_ROOT}/logs"
exec > >(tee "${MV11_CAMPAIGN_ROOT}/logs/post_${SLURM_JOB_ID}.out") 2>&1
command -v zip

python3 "${MV11_PAYLOAD_ROOT}/tools/analyze_mv11_cylinder.py" \
  --campaign-root "${MV11_CAMPAIGN_ROOT}" \
  --output-dir "${MV11_CAMPAIGN_ROOT}/analysis"

{
  for STATUS_FILE in "${MV11_CAMPAIGN_ROOT}"/cases/seed_*/results/RUN_STATUS.env; do
    [[ -f "${STATUS_FILE}" ]] || continue
    echo "===== ${STATUS_FILE#${MV11_CAMPAIGN_ROOT}/} ====="
    cat "${STATUS_FILE}"
  done
} > "${MV11_CAMPAIGN_ROOT}/analysis/run_statuses.txt"
if [[ -s "${MV11_CAMPAIGN_ROOT}/SUBMISSION.env" ]]; then
  source "${MV11_CAMPAIGN_ROOT}/SUBMISSION.env"
  sacct -j "${MV11_JOB_IDS}" -X \
    --format=JobIDRaw,JobName%24,State,ExitCode,Elapsed,Start,End,NodeList%20,MaxRSS \
    > "${MV11_CAMPAIGN_ROOT}/analysis/slurm_accounting.txt" 2>&1 || true
fi

STAMP=$(basename "${MV11_CAMPAIGN_ROOT}" | sed 's/^MV11_DS2V_CYLINDER_//')
BUNDLE="${MV11_PROJECT_ROOT}/MV11_DS2V_CYLINDER_ANALYSIS_BUNDLE_${STAMP}.zip"
MANIFEST="${MV11_CAMPAIGN_ROOT}/analysis/DELIVERY_MANIFEST_SHA256.txt"
MANIFEST_TMP=$(mktemp "${MV11_CAMPAIGN_ROOT}/delivery_manifest.XXXXXX")

find \
  "${MV11_CAMPAIGN_ROOT}/analysis" \
  "${MV11_CAMPAIGN_ROOT}/logs" \
  "${MV11_CAMPAIGN_ROOT}/build/PATCH_REPORT.json" \
  "${MV11_CAMPAIGN_ROOT}/build/BUILD_CONFIGURATION.txt" \
  "${MV11_CAMPAIGN_ROOT}/INPUT_SOURCE_EXECUTABLE_SHA256.txt" \
  "${MV11_CAMPAIGN_ROOT}/mv11_cylinder_protocol.json" \
  "${MV11_CAMPAIGN_ROOT}/seed_table.tsv" \
  "${MV11_CAMPAIGN_ROOT}/SUBMISSION.env" \
  -type f ! -path "${MANIFEST}" -print0 | sort -z | xargs -0 sha256sum > "${MANIFEST_TMP}"
mv "${MANIFEST_TMP}" "${MANIFEST}"

rm -f "${BUNDLE}"
(
  cd "${MV11_CAMPAIGN_ROOT}"
  zip -q -r "${BUNDLE}" \
    analysis logs \
    build/PATCH_REPORT.json build/BUILD_CONFIGURATION.txt \
    INPUT_SOURCE_EXECUTABLE_SHA256.txt mv11_cylinder_protocol.json seed_table.tsv SUBMISSION.env \
    -x 'analysis/*.tmp'
)
sha256sum "${BUNDLE}" > "${BUNDLE}.sha256"

SUMMARY="${MV11_CAMPAIGN_ROOT}/analysis/mv11_cylinder_summary.json"
PASS=$(python3 -c 'import json,sys; print(str(json.load(open(sys.argv[1]))["analysis_pass"]).lower())' "${SUMMARY}")
cat > "${MV11_PROJECT_ROOT}/LAST_MV11_DS2V_CYLINDER_RESULT.env" <<EOF
MV11_CAMPAIGN_ROOT=${MV11_CAMPAIGN_ROOT}
MV11_ANALYSIS_BUNDLE=${BUNDLE}
MV11_ANALYSIS_BUNDLE_SHA256=$(sha256sum "${BUNDLE}" | awk '{print $1}')
MV11_ANALYSIS_PASS=${PASS}
MV11_POST_JOB_ID=${SLURM_JOB_ID}
EOF
echo "MV11_POST_PASS bundle=${BUNDLE} analysis_pass=${PASS}"
