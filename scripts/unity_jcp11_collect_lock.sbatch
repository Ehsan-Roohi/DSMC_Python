#!/bin/bash
#SBATCH --job-name=j11-lock
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:30:00

set -Eeuo pipefail
: "${JCP11_WORK:?}"
: "${JCP11_CODE:?}"
: "${JCP11_MODEL_LOCK:?}"
: "${JCP11_RULE_LOCK:?}"

python "${JCP11_CODE}/scripts/jcp11_support_validation.py" collect-lock \
  --work "${JCP11_WORK}" \
  --model-lock "${JCP11_MODEL_LOCK}" \
  --rule "${JCP11_RULE_LOCK}" \
  --protocol "${JCP11_CODE}/reference_data/mohammadzadeh_2012/jcp11_support_validation_protocol.json" \
  --output "${JCP11_WORK}"
(cd "${JCP11_WORK}" && sha256sum -c \
  JCP11_M12_EVALUATION.zip.sha256 JCP11_M12_DECISION_LOCK.zip.sha256)
echo "JCP11_FRESH_DECISIONS_LOCKED=1"
