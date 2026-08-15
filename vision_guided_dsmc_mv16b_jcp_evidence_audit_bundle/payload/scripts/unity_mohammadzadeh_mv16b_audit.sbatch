#!/usr/bin/env bash
#SBATCH --job-name=moh-mv16b-audit
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00

set -euo pipefail
: "${MV16B_PROJECT_ROOT:?}"
: "${MV16B_PAYLOAD_ROOT:?}"
: "${MV16B_OUTPUT_ROOT:?}"
: "${MV16B_PYTHON:?}"
: "${MV16B_MV15C_ROOT:?}"
: "${MV16B_MV16A_ROOT:?}"
: "${MV16B_CAMPAIGN_ROOT:?}"
test -x "${MV16B_PYTHON}"
test ! -e "${MV16B_OUTPUT_ROOT}"

exec > >(tee "${MV16B_PROJECT_ROOT}/logs/moh_mv16b_audit_${SLURM_JOB_ID}.out") 2>&1
cd "${MV16B_PROJECT_ROOT}"
export PATH="$(dirname "${MV16B_PYTHON}"):${PATH}"
export PYTHONPATH="${MV16B_PROJECT_ROOT}:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export PYTHONUNBUFFERED=1
export MPLCONFIGDIR="${TMPDIR:-/tmp}/mv16b-mpl-${SLURM_JOB_ID}"
mkdir -p "${MPLCONFIGDIR}"

"${MV16B_PYTHON}" -c 'import matplotlib,numpy,scipy,torch; print("MV16B_RUNTIME_PASS", "numpy="+numpy.__version__, "scipy="+scipy.__version__, "torch="+torch.__version__)'
"${MV16B_PYTHON}" "${MV16B_PAYLOAD_ROOT}/tests/test_mohammadzadeh_mv16b_jcp_evidence_audit.py"
"${MV16B_PYTHON}" -m vgdsmc.mohammadzadeh_mv16b_jcp_evidence_audit run \
  --mv15c-root "${MV16B_MV15C_ROOT}" \
  --mv16a-root "${MV16B_MV16A_ROOT}" \
  --campaign-root "${MV16B_CAMPAIGN_ROOT}" \
  --output-root "${MV16B_OUTPUT_ROOT}" \
  --batch-size "${MV16B_BATCH_SIZE:-4}"
test -s "${MV16B_OUTPUT_ROOT}/summary.json"
echo "MV16B_AUDIT_PASS output=${MV16B_OUTPUT_ROOT} dsmc_rerun=NO training=NO"

