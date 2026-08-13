#!/usr/bin/env bash
#SBATCH --job-name=mv11-cyl-prep
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=01:00:00

set -euo pipefail
: "${MV11_PROJECT_ROOT:?}"
: "${MV11_CAMPAIGN_ROOT:?}"
: "${MV11_DS2V_SOURCE:?}"
: "${MV11_PAYLOAD_ROOT:?}"

mkdir -p "${MV11_CAMPAIGN_ROOT}/build" "${MV11_CAMPAIGN_ROOT}/input" "${MV11_CAMPAIGN_ROOT}/logs"
exec > >(tee "${MV11_CAMPAIGN_ROOT}/logs/prepare_${SLURM_JOB_ID}.out") 2>&1

echo "[prepare] $(date -Is) host=$(hostname)"
test -s "${MV11_DS2V_SOURCE}"
command -v python3
command -v gfortran
python3 "${MV11_PAYLOAD_ROOT}/tests/test_mv11_ds2v_cylinder.py"

SOURCE_COPY="${MV11_CAMPAIGN_ROOT}/build/DS2V_BIRD_BASE.F90"
PATCHED_SOURCE="${MV11_CAMPAIGN_ROOT}/build/DS2V_BIRD_MV11.F90"
PATCH_REPORT="${MV11_CAMPAIGN_ROOT}/build/PATCH_REPORT.json"
EXE="${MV11_CAMPAIGN_ROOT}/build/ds2v_bird_mv11"

install -m 0644 "${MV11_DS2V_SOURCE}" "${SOURCE_COPY}"
python3 "${MV11_PAYLOAD_ROOT}/patcher/patch_ds2v_mv11.py" \
  "${SOURCE_COPY}" --out "${PATCHED_SOURCE}" --report "${PATCH_REPORT}"

gfortran \
  -O2 -g -std=legacy -fdec -fno-automatic \
  -fallow-argument-mismatch -ffree-line-length-none -fno-range-check \
  -fbacktrace -fcheck=bounds \
  "${PATCHED_SOURCE}" -o "${EXE}" \
  > "${MV11_CAMPAIGN_ROOT}/build/build.log" 2>&1
test -x "${EXE}"

NX="${MV11_NX:-194}"
NY="${MV11_NY:-100}"
PARTICLES="${MV11_PARTICLES:-1500000}"
sed \
  -e "s/@NX@/${NX}/" \
  -e "s/@NY@/${NY}/" \
  -e "s/@PARTICLES@/${PARTICLES}/" \
  "${MV11_PAYLOAD_ROOT}/case/DS2VD_M10_ARGON_CYLINDER.template" \
  > "${MV11_CAMPAIGN_ROOT}/input/DS2VD.DAT"
install -m 0644 "${MV11_PAYLOAD_ROOT}/case/interactive_input.txt" "${MV11_CAMPAIGN_ROOT}/input/interactive_input.txt"
install -m 0644 "${MV11_PAYLOAD_ROOT}/case/HEAT-BENCH.TXT" "${MV11_CAMPAIGN_ROOT}/input/HEAT-BENCH.TXT"
install -m 0644 "${MV11_PAYLOAD_ROOT}/case/mv11_cylinder_protocol.json" "${MV11_CAMPAIGN_ROOT}/mv11_cylinder_protocol.json"

python3 - "${MV11_CAMPAIGN_ROOT}/input/DS2VD.DAT" <<'PY'
import math
import sys
from pathlib import Path

values = [line.strip() for line in Path(sys.argv[1]).read_text().splitlines() if line.strip()]
if len(values) != 83:
    raise SystemExit(f"deck must have 83 entries, found {len(values)}")
f = lambda line: float(values[line - 1].replace("D", "E"))
checks = {17: -0.2, 18: 0.65, 20: 0.4, 35: 4.247e20, 36: 200.0,
          37: 2634.1, 51: 0.3048, 59: 500.0}
for line, target in checks.items():
    if not math.isclose(f(line), target, rel_tol=1e-12, abs_tol=1e-12):
        raise SystemExit(f"deck line {line} mismatch")
mass = f(31)
mach = f(37) / math.sqrt((5.0 / 3.0) * 1.380649e-23 * f(36) / mass)
if not 9.95 <= mach <= 10.05:
    raise SystemExit(f"Mach preflight failed: {mach}")
print(f"MV11_DECK_PREFLIGHT_PASS Mach={mach:.8f}")
PY

{
  gfortran --version | head -n 2
  echo "flags=-O2 -g -std=legacy -fdec -fno-automatic -fallow-argument-mismatch -ffree-line-length-none -fno-range-check -fbacktrace -fcheck=bounds"
  echo "NX=${NX}"
  echo "NY=${NY}"
  echo "PARTICLES=${PARTICLES}"
} > "${MV11_CAMPAIGN_ROOT}/build/BUILD_CONFIGURATION.txt"

sha256sum \
  "${SOURCE_COPY}" "${PATCHED_SOURCE}" "${EXE}" \
  "${MV11_CAMPAIGN_ROOT}/input/DS2VD.DAT" \
  "${MV11_CAMPAIGN_ROOT}/input/HEAT-BENCH.TXT" \
  "${MV11_CAMPAIGN_ROOT}/mv11_cylinder_protocol.json" \
  > "${MV11_CAMPAIGN_ROOT}/INPUT_SOURCE_EXECUTABLE_SHA256.txt"

touch "${MV11_CAMPAIGN_ROOT}/PREPARE_PASS"
echo "MV11_PREPARE_PASS"
