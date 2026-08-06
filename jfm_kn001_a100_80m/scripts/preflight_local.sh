#!/bin/bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

python -m py_compile solver/JFM_bgk_shakhov_quarter.py tools/summarize_six_seed.py tests/test_bundle.py tests/test_summary_e2e.py
python tests/test_bundle.py
python tests/test_summary_e2e.py
mamba run -n dsmc-gpu python -c 'import cupy; print("dsmc-gpu imports OK")'
echo "[OK] local preflight passed"
