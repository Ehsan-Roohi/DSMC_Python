#!/usr/bin/env bash
#SBATCH --job-name=qk_g5_geo_sum
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:15:00
#SBATCH --output=/project/pi_roohie_umass_edu/Combustion/QK_GATE5_COUPLED/runs/geometry-summary-%j.out
#SBATCH --error=/project/pi_roohie_umass_edu/Combustion/QK_GATE5_COUPLED/runs/geometry-summary-%j.err
set -euo pipefail

BASE=/project/pi_roohie_umass_edu/Combustion/QK_GATE5_COUPLED
: "${GATE5_GEOMETRY_RELEASE_ID:?GATE5_GEOMETRY_RELEASE_ID is required}"
: "${GEOMETRY_ARRAY_JOB_ID:?GEOMETRY_ARRAY_JOB_ID is required}"
PAYLOAD="$BASE/releases/$GATE5_GEOMETRY_RELEASE_ID"
GROUP="$BASE/runs/geometry_preflight_${GEOMETRY_ARRAY_JOB_ID}"
PYTHON_BIN=/work/pi_roohie_umass_edu/roohie_umass_edu/.conda/envs/dsmc-gpu/bin/python
if [[ ! -x "$PYTHON_BIN" ]]; then PYTHON_BIN="$HOME/.conda/envs/dsmc-gpu/bin/python"; fi
if [[ ! -x "$PYTHON_BIN" ]]; then echo "ERROR: dsmc-gpu Python missing" >&2; exit 3; fi
export PYTHON_BIN
"$PAYLOAD/run_gate5_geometry_summary.sh" "$GROUP" 2>&1 | \
  tee "$GROUP/unity_gate5_geometry_summary.log"
echo "UNITY_QK_GATE5_GEOMETRY_SUMMARY_PASS"
echo "RESULT_DIR=$GROUP"
