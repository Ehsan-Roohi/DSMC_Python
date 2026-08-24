#!/bin/bash
#SBATCH --job-name=j11-score
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=02:30:00

set -Eeuo pipefail
: "${JCP11_WORK:?}"
: "${JCP11_CODE:?}"
: "${JCP11_REFERENCE:?}"
export MPLCONFIGDIR="${JCP11_WORK}/.matplotlib"
mkdir -p "${MPLCONFIGDIR}"

python "${JCP11_CODE}/scripts/jcp11_support_validation.py" score-pack \
  --rule-archive "${JCP11_WORK}/JCP11_SUPPORT_RULE_LOCK.zip" \
  --loo-calibration "${JCP11_WORK}/JCP11_DEVELOPMENT_LOO_CALIBRATION.csv" \
  --specificity-archive "${JCP11_WORK}/JCP11_M10_SPECIFICITY.zip" \
  --specificity-summary "${JCP11_WORK}/JCP11_M10_SPECIFICITY_SUMMARY.json" \
  --specificity-ledger "${JCP11_WORK}/JCP11_M10_SPECIFICITY_LEDGER.csv" \
  --evaluation "${JCP11_WORK}/JCP11_M12_EVALUATION.zip" \
  --decision-lock "${JCP11_WORK}/JCP11_M12_DECISION_LOCK.zip" \
  --reference "${JCP11_REFERENCE}" \
  --protocol "${JCP11_CODE}/reference_data/mohammadzadeh_2012/jcp11_support_validation_protocol.json" \
  --output "${JCP11_WORK}"
(cd "${JCP11_WORK}" && sha256sum -c JCP11_SUPPORT_VALIDATION.zip.sha256)
echo "JCP11_SUPPORT_VALIDATION_COMPLETE=1"
echo "UPLOAD=${JCP11_WORK}/JCP11_SUPPORT_VALIDATION.zip ${JCP11_WORK}/JCP11_SUPPORT_VALIDATION.zip.sha256"
