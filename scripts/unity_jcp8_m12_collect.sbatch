#!/bin/bash
#SBATCH --job-name=j8-m12-pack
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=02:00:00
set -Eeuo pipefail
: "${JCP8_WORK:?}"; : "${JCP8_CODE:?}"
python "${JCP8_CODE}/scripts/collect_jcp8_m12_reference.py" --work "${JCP8_WORK}" --protocol "${JCP8_CODE}/reference_data/mohammadzadeh_2012/jcp8_m12_reference_score_protocol.json" --archive "${JCP8_WORK}/JCP8_M12_REFERENCE.zip"
cd "${JCP8_WORK}"; sha256sum -c JCP8_M12_REFERENCE.zip.sha256
echo "JCP8_REFERENCE_LOCK_COMPLETE=1"
