#!/bin/bash
#SBATCH --job-name=j5-m10-audit
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=01:00:00

set -Eeuo pipefail
: "${JCP5_M10_ROOT:?}"
: "${JCP5_JCP4_ARCHIVE:?}"
: "${JCP5_WORK:?}"
: "${JCP5_CODE:?}"

python "${JCP5_CODE}/scripts/audit_jcp5_m10_assets.py" \
  --m10-root "${JCP5_M10_ROOT}" \
  --jcp4-archive "${JCP5_JCP4_ARCHIVE}" \
  --output "${JCP5_WORK}/JCP5_M10_AUDIT.zip"
cd "${JCP5_WORK}"
sha256sum -c JCP5_M10_AUDIT.zip.sha256
echo "JCP5_M10_AUDIT_COMPLETE=1"
echo "UPLOAD_ZIP=${JCP5_WORK}/JCP5_M10_AUDIT.zip"
echo "UPLOAD_SHA=${JCP5_WORK}/JCP5_M10_AUDIT.zip.sha256"
