#!/bin/bash
#SBATCH --job-name=j9-close
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

set -euo pipefail

: "${JCP9_ROOT:?}"
: "${JCP9_PREDICTION:?}"
: "${JCP9_REFERENCE:?}"
: "${JCP9_SCORE:?}"

python "$JCP9_ROOT/code/jcp9_closeout_existing_m12.py" \
  --prediction "$JCP9_PREDICTION" \
  --reference "$JCP9_REFERENCE" \
  --prospective-score "$JCP9_SCORE" \
  --protocol "$JCP9_ROOT/code/jcp9_m12_closeout_protocol.json" \
  --output "$JCP9_ROOT"

cd "$JCP9_ROOT"
sha256sum -c JCP9_M12_CLOSEOUT.zip.sha256

