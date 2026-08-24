#!/bin/bash
#SBATCH --job-name=j7-m12-pack
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=01:00:00

set -Eeuo pipefail
: "${JCP7_WORK:?}"
: "${JCP7_CODE:?}"
python "${JCP7_CODE}/scripts/collect_jcp7_m12_evaluation.py" \
  --work "${JCP7_WORK}" \
  --protocol "${JCP7_CODE}/reference_data/mohammadzadeh_2012/jcp7_m12_evaluation_protocol.json" \
  --archive "${JCP7_WORK}/JCP7_M12_EVALUATION.zip"
cd "${JCP7_WORK}"
sha256sum -c JCP7_M12_EVALUATION.zip.sha256
echo "JCP7_M12_EVALUATION_LOCKED=1"
