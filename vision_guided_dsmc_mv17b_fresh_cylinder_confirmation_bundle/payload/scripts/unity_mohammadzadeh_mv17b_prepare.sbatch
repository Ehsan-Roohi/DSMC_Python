#!/usr/bin/env bash
#SBATCH --job-name=moh-mv17b-prep
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=02:00:00

set -euo pipefail
: "${MV17B_PROJECT_ROOT:?}"
: "${MV17B_PAYLOAD_ROOT:?}"
: "${MV17B_OUTPUT_ROOT:?}"
: "${MV17B_MV16B_ROOT:?}"
: "${MV17B_MV17A_ROOT:?}"
: "${MV17B_DS2V_SOURCE:?}"
: "${MV17B_PYTHON:?}"

CAMPAIGN="${MV17B_OUTPUT_ROOT}/campaign"
BUILD="${CAMPAIGN}/build"
INPUT="${CAMPAIGN}/input"
mkdir -p "${BUILD}" "${INPUT}" "${CAMPAIGN}/logs"
if find "${CAMPAIGN}/cases" -type f -print -quit 2>/dev/null | grep -q .; then
  echo "MV17B_PREP_ERROR: fresh case data existed before the model freeze" >&2
  exit 20
fi
exec > >(tee "${CAMPAIGN}/logs/prepare_${SLURM_JOB_ID}.out") 2>&1

cd "${MV17B_PROJECT_ROOT}"
export PATH="$(dirname "${MV17B_PYTHON}"):${PATH}"
export PYTHONPATH="${MV17B_PROJECT_ROOT}:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export PYTHONUNBUFFERED=1

"${MV17B_PYTHON}" "${MV17B_PAYLOAD_ROOT}/tests/test_mohammadzadeh_mv17b_fresh_cylinder_confirmation.py"
"${MV17B_PYTHON}" -m vgdsmc.mohammadzadeh_mv17b_fresh_cylinder_confirmation freeze \
  --mv16b-root "${MV17B_MV16B_ROOT}" \
  --mv17a-root "${MV17B_MV17A_ROOT}" \
  --output-root "${MV17B_OUTPUT_ROOT}"
test -s "${MV17B_OUTPUT_ROOT}/frozen/FROZEN_MODEL_PASS"

SOURCE_COPY="${BUILD}/DS2V_BIRD_BASE.F90"
PATCHED_SOURCE="${BUILD}/DS2V_BIRD_MV17B.F90"
PATCH_REPORT="${BUILD}/PATCH_REPORT.json"
EXE="${BUILD}/ds2v_bird_mv17b"
install -m 0644 "${MV17B_DS2V_SOURCE}" "${SOURCE_COPY}"
"${MV17B_PYTHON}" "${MV17B_PROJECT_ROOT}/mv11_ds2v_cylinder/patcher/patch_ds2v_mv11.py" \
  "${SOURCE_COPY}" --out "${PATCHED_SOURCE}" --report "${PATCH_REPORT}"
gfortran \
  -O2 -g -std=legacy -fdec -fno-automatic \
  -fallow-argument-mismatch -ffree-line-length-none -fno-range-check \
  -fbacktrace -fcheck=bounds \
  "${PATCHED_SOURCE}" -o "${EXE}" > "${BUILD}/build.log" 2>&1
test -x "${EXE}"

NX="${MV17B_NX:-194}"
NY="${MV17B_NY:-100}"
PARTICLES="${MV17B_PARTICLES:-1500000}"
sed \
  -e "s/@NX@/${NX}/" \
  -e "s/@NY@/${NY}/" \
  -e "s/@PARTICLES@/${PARTICLES}/" \
  "${MV17B_PROJECT_ROOT}/mv11_ds2v_cylinder/case/DS2VD_M10_ARGON_CYLINDER.template" \
  > "${INPUT}/DS2VD.DAT"
install -m 0644 "${MV17B_PROJECT_ROOT}/mv11_ds2v_cylinder/case/interactive_input.txt" "${INPUT}/interactive_input.txt"
install -m 0644 "${MV17B_PROJECT_ROOT}/mv11_ds2v_cylinder/case/HEAT-BENCH.TXT" "${INPUT}/HEAT-BENCH.TXT"

"${MV17B_PYTHON}" - "${INPUT}/DS2VD.DAT" <<'PY'
import math
import sys
from pathlib import Path
values = [line.strip() for line in Path(sys.argv[1]).read_text().splitlines() if line.strip()]
if len(values) != 83:
    raise SystemExit(f"MV17B deck must have 83 entries, found {len(values)}")
f = lambda line: float(values[line - 1].replace("D", "E"))
checks = {17: -0.2, 18: 0.65, 20: 0.4, 35: 4.247e20, 36: 200.0,
          37: 2634.1, 51: 0.3048, 59: 500.0}
for line, target in checks.items():
    if not math.isclose(f(line), target, rel_tol=1e-12, abs_tol=1e-12):
        raise SystemExit(f"MV17B deck line {line} mismatch")
mass = f(31)
mach = f(37) / math.sqrt((5.0 / 3.0) * 1.380649e-23 * f(36) / mass)
if not 9.95 <= mach <= 10.05:
    raise SystemExit(f"MV17B Mach preflight failed: {mach}")
print(f"MV17B_DECK_PREFLIGHT_PASS Mach={mach:.8f}")
PY

{
  gfortran --version | head -n 2
  echo "flags=-O2 -g -std=legacy -fdec -fno-automatic -fallow-argument-mismatch -ffree-line-length-none -fno-range-check -fbacktrace -fcheck=bounds"
  echo "NX=${NX}"
  echo "NY=${NY}"
  echo "PARTICLES=${PARTICLES}"
  echo "LOCKED_LAST_NOUT=116"
  echo "FRESH_TRAJECTORIES=12"
} > "${BUILD}/BUILD_CONFIGURATION.txt"
sha256sum \
  "${SOURCE_COPY}" "${PATCHED_SOURCE}" "${EXE}" "${INPUT}/DS2VD.DAT" \
  "${INPUT}/interactive_input.txt" "${INPUT}/HEAT-BENCH.TXT" \
  "${MV17B_OUTPUT_ROOT}/frozen/mv17b_frozen_cylinder_model.npz" \
  > "${BUILD}/INPUT_SOURCE_MODEL_EXECUTABLE_SHA256.txt"
touch "${CAMPAIGN}/PREPARE_PASS"
echo "MV17B_PREPARE_PASS model_frozen_before_fresh_DSMC=YES"

