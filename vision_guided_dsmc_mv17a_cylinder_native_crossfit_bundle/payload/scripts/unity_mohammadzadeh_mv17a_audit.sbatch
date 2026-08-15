#!/usr/bin/env bash
#SBATCH --job-name=moh-mv17a-audit
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00

set -euo pipefail
: "${MV17A_PROJECT_ROOT:?}"
: "${MV17A_PAYLOAD_ROOT:?}"
: "${MV17A_OUTPUT_ROOT:?}"
: "${MV17A_SOURCE_ROOT:?}"
: "${MV17A_PYTHON:?}"
test -x "${MV17A_PYTHON}"
test ! -e "${MV17A_OUTPUT_ROOT}"
test -s "${MV17A_SOURCE_ROOT}/artifact_manifest.json"

exec > >(tee "${MV17A_PROJECT_ROOT}/logs/moh_mv17a_audit_${SLURM_JOB_ID}.out") 2>&1
cd "${MV17A_PROJECT_ROOT}"
export PATH="$(dirname "${MV17A_PYTHON}"):${PATH}"
export PYTHONPATH="${MV17A_PROJECT_ROOT}:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export PYTHONUNBUFFERED=1
export MPLCONFIGDIR="${TMPDIR:-/tmp}/mv17a-mpl-${SLURM_JOB_ID}"
mkdir -p "${MPLCONFIGDIR}"

"${MV17A_PYTHON}" -c 'import matplotlib,numpy,scipy; print("MV17A_RUNTIME_PASS", "numpy="+numpy.__version__, "scipy="+scipy.__version__)'
"${MV17A_PYTHON}" "${MV17A_PAYLOAD_ROOT}/tests/test_mohammadzadeh_mv17a_cylinder_native_crossfit.py"
"${MV17A_PYTHON}" -m vgdsmc.mohammadzadeh_mv17a_cylinder_native_crossfit run \
  --mv16b-root "${MV17A_SOURCE_ROOT}" \
  --output-root "${MV17A_OUTPUT_ROOT}"
test -s "${MV17A_OUTPUT_ROOT}/summary.json"
echo "MV17A_AUDIT_PASS output=${MV17A_OUTPUT_ROOT} dsmc_rerun=NO neural_training=NO"

