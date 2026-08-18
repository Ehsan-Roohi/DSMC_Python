#!/bin/bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

for command_name in sbatch mamba python awk sha256sum; do
  command -v "$command_name" >/dev/null || {
    echo "Missing required command: $command_name" >&2
    exit 2
  }
done

python -m py_compile \
  solver/JFM_hs_dsmc_quarter.py \
  solver/JFM_bgk_shakhov_quarter.py \
  tools/summarize_highstat.py

for script in scripts/*.sh scripts/*.slurm submit_from_github.sh; do
  bash -n "$script"
done

python tests/test_bundle.py
python tests/test_summary_e2e.py
mamba run -n dsmc-gpu python -c \
  'import cupy, numba, numpy, matplotlib; print("dsmc-gpu imports OK")'

mkdir -p run_output slurm
test -w run_output
echo "[OK] local preflight passed"
