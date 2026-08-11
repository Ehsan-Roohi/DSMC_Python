#!/usr/bin/env bash
#SBATCH --job-name=qk_g5_geo_smk
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=/project/pi_roohie_umass_edu/Combustion/QK_GATE5_COUPLED/runs/geometry-smoke-%j.out
#SBATCH --error=/project/pi_roohie_umass_edu/Combustion/QK_GATE5_COUPLED/runs/geometry-smoke-%j.err
set -euo pipefail

BASE=/project/pi_roohie_umass_edu/Combustion/QK_GATE5_COUPLED
: "${GATE5_GEOMETRY_RELEASE_ID:?GATE5_GEOMETRY_RELEASE_ID is required}"
PAYLOAD="$BASE/releases/$GATE5_GEOMETRY_RELEASE_ID"
RESULT="$BASE/runs/geometry_smoke_${SLURM_JOB_ID}"
mkdir -p "$RESULT"
module load gcc/12.2.0 2>/dev/null || module load gcc 2>/dev/null || true
PYTHON_BIN=/work/pi_roohie_umass_edu/roohie_umass_edu/.conda/envs/dsmc-gpu/bin/python
if [[ ! -x "$PYTHON_BIN" ]]; then PYTHON_BIN="$HOME/.conda/envs/dsmc-gpu/bin/python"; fi
if [[ ! -x "$PYTHON_BIN" ]]; then echo "ERROR: dsmc-gpu Python missing" >&2; exit 3; fi
export PYTHON_BIN
command -v gfortran
"$PAYLOAD/run_gate5_geometry_smoke.sh" "$RESULT" 2>&1 | \
  tee "$RESULT/unity_gate5_geometry_smoke.log"
echo "UNITY_QK_GATE5_GEOMETRY_SMOKE_PASS"
echo "RESULT_DIR=$RESULT"
PASS_MARKER="$BASE/GEOMETRY_SMOKE_PASS_${GATE5_GEOMETRY_RELEASE_ID}.env"
PASS_TMP="${PASS_MARKER}.job_${SLURM_JOB_ID}.tmp"
cat > "$PASS_TMP" <<EOF
GEOMETRY_SMOKE_JOB_ID=$SLURM_JOB_ID
GEOMETRY_SMOKE_RESULT_DIR=$RESULT
GEOMETRY_RELEASE_ID=$GATE5_GEOMETRY_RELEASE_ID
EOF
mv "$PASS_TMP" "$PASS_MARKER"
