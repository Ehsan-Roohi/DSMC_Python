#!/usr/bin/env bash
#SBATCH --job-name=sp-mm-build
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=00:50:00
#SBATCH --output=logs/sp-mm-build-%j.out
#SBATCH --error=logs/sp-mm-build-%j.err

set -euo pipefail

ROOT_DIR="${SPARTA_CASE_ROOT:-${SLURM_SUBMIT_DIR:-}}"
if [[ -z "${ROOT_DIR}" || ! -d "${ROOT_DIR}" ]]; then
  echo "SPARTA_CASE_ROOT/SLURM_SUBMIT_DIR does not identify the code directory." >&2
  exit 2
fi
ROOT_DIR="$(cd "${ROOT_DIR}" && pwd)"
SPARTA_BIN="${ROOT_DIR}/third_party/sparta/src/spa_mpi"
SPARTA_REF="912c9e163c38ea5c3562d039e65215f6e2a4f3f8"
OPENMPI_MODULE="${UNITY_OPENMPI_MODULE:-openmpi/5.0.3}"
BUILD_JOBS="${SLURM_CPUS_PER_TASK:-8}"

module purge
module load "${OPENMPI_MODULE}"
module list

for command_name in git make g++ mpicxx python3; do
  if ! command -v "${command_name}" >/dev/null 2>&1; then
    echo "Required command is unavailable after loading ${OPENMPI_MODULE}: ${command_name}" >&2
    exit 2
  fi
done

cd "${ROOT_DIR}"
mkdir -p logs
python3 -m unittest discover -s tests -v

if [[ -x "${SPARTA_BIN}" ]]; then
  echo "Reusing existing MPI-enabled SPARTA executable: ${SPARTA_BIN}"
elif [[ -d "${ROOT_DIR}/third_party/sparta/src" ]]; then
  make -C "${ROOT_DIR}/third_party/sparta/src" -j"${BUILD_JOBS}" mpi
else
  SPARTA_REF="${SPARTA_REF}" BUILD_JOBS="${BUILD_JOBS}" \
    bash "${ROOT_DIR}/scripts/install_sparta_linux.sh" mpi
fi

if [[ ! -x "${SPARTA_BIN}" ]]; then
  echo "SPARTA build did not create ${SPARTA_BIN}" >&2
  exit 3
fi
SOURCE_COMMIT="$(git -C "${ROOT_DIR}/third_party/sparta" rev-parse HEAD)"
if [[ "${SOURCE_COMMIT}" != "${SPARTA_REF}" ]]; then
  echo "SPARTA source mismatch: ${SOURCE_COMMIT}; expected ${SPARTA_REF}" >&2
  exit 4
fi

HELP_FILE="${ROOT_DIR}/logs/sp-mm-build-${SLURM_JOB_ID}-help.txt"
set +e
"${SPARTA_BIN}" -help > "${HELP_FILE}" 2>&1
HELP_STATUS=$?
set -e
if ! grep -q '^Usage: sparta' "${HELP_FILE}"; then
  echo "SPARTA did not produce a valid help banner (exit ${HELP_STATUS})." >&2
  exit 5
fi
for compute_style in 'pflux/grid' 'sonine/grid' 'thermal/grid' 'eflux/grid'; do
  if ! grep -Fq "${compute_style}" "${HELP_FILE}"; then
    echo "Pinned SPARTA does not advertise required compute style: ${compute_style}" >&2
    exit 5
  fi
done

# Validate both full production contracts without running them, then execute a
# tiny parser/data-path smoke case for each Knudsen number.
for case_spec in '005 0.05' '020 0.20'; do
  read -r case_tag kn_gu <<< "${case_spec}"
  CONTRACT_DIR="${ROOT_DIR}/logs/sp-mm-contract-${case_tag}-${SLURM_JOB_ID}"
  python3 "${ROOT_DIR}/scripts/generate_jfm_maxwell_kngu020_case.py" \
    --seed 104729 --kn-gu "${kn_gu}" --output "${CONTRACT_DIR}" \
    > "${CONTRACT_DIR}.generator.out"
  python3 "${ROOT_DIR}/scripts/validate_jfm_maxwell_kngu_case.py" \
    "${CONTRACT_DIR}" --kn-gu "${kn_gu}"

  SMOKE_DIR="${ROOT_DIR}/logs/sp-mm-smoke-${case_tag}-${SLURM_JOB_ID}"
  python3 "${ROOT_DIR}/scripts/generate_jfm_maxwell_kngu020_case.py" \
    --seed 20260817 --kn-gu "${kn_gu}" --nx 12 --ppc 4 \
    --warmup 20 --sample 30 --stride 2 --checkpoint 0 \
    --output "${SMOKE_DIR}" > "${SMOKE_DIR}.generator.out"
  (
    cd "${SMOKE_DIR}"
    "${SPARTA_BIN}" < in.cavity > screen.out 2> screen.err
  )
  SMOKE_DUMP="${SMOKE_DIR}/grid.final.00000030"
  if [[ ! -s "${SMOKE_DUMP}" ]] || \
     ! grep -Fq 'f_fieldavg[15]' "${SMOKE_DUMP}" || \
     grep -Fq 'f_fieldavg[16]' "${SMOKE_DUMP}"; then
    echo "Maxwell-VSS Kn_Gu=${kn_gu} smoke case did not produce the 15-field schema." >&2
    exit 6
  fi
done

echo "Pinned SPARTA, both exact Kn_Gu contracts, and the 15-field anti-Fourier schema passed."
echo "READY: ${SPARTA_BIN}"
