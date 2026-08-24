#!/bin/bash
#SBATCH --job-name=j4-m8-pack
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=02:00:00

set -Eeuo pipefail
: "${JCP4_WORK:?}"
: "${JCP4_CODE:?}"

python "${JCP4_CODE}/scripts/collect_jcp4_m8_reference.py" \
  --work "${JCP4_WORK}" \
  --protocol "${JCP4_CODE}/reference_data/mohammadzadeh_2012/jcp4_m8_reference_protocol.json" \
  --archive "${JCP4_WORK}/JCP4_M8_REFERENCE.zip"

cd "${JCP4_WORK}"
sha256sum -c JCP4_M8_REFERENCE.zip.sha256
echo "JCP4_M8_REFERENCE_COMPLETE=1"
echo "UPLOAD_ZIP=${JCP4_WORK}/JCP4_M8_REFERENCE.zip"
echo "UPLOAD_SHA=${JCP4_WORK}/JCP4_M8_REFERENCE.zip.sha256"
