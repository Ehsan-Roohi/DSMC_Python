#!/usr/bin/env python3
from __future__ import annotations

import csv
import hashlib
import importlib.util
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
EXPECTED_SHA = {
    "solver/JFM_hs_dsmc_quarter.py": "d234a93910403bcf4429ab2b20767a2a7d35349b27dab3016d257abfa5881f6f",
    "solver/JFM_bgk_shakhov_quarter.py": "c20d1a5fac0ab7e019e695a68e7ddecc64571f63102a8a957f61d766d6b4db55",
}
for relative, expected in EXPECTED_SHA.items():
    assert hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() == expected

with (ROOT / "cases/remaining4.csv").open(newline="") as handle:
    remaining = list(csv.DictReader(handle))
with (ROOT / "cases/final7.csv").open(newline="") as handle:
    final = list(csv.DictReader(handle))
assert len(remaining) == 4 and len(final) == 7
assert {int(row["seed"]) for row in final} == {104729}
assert {row["source"] for row in final} == {"existing_vram48", "new_l40s"}

run = (ROOT / "scripts/run_l40s.slurm").read_text()
submit = (ROOT / "scripts/submit_l40s.sh").read_text()
assert "--steps 3000000 --sample-start 100000 --sample-every 2" in run
assert "--time-blocks 29" in run
assert "--no-capture-output" in run and "python -u" in run
assert "--constraint=l40s" in submit and "--array=0-3%4" in submit
assert "--qos=long" not in submit
assert "62606478:62611438:62611907" in submit

spec = importlib.util.spec_from_file_location("summary", ROOT / "tools/summarize_single.py")
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(module)
blocks = np.array([[[1.0]], [[3.0]], [[5.0]]])
sd, se = module.block_statistics(blocks)
assert np.allclose(sd, 2.0)
assert np.allclose(se, 2.0 / np.sqrt(3.0))
print("[OK] single-seed L40S bundle and solver provenance validated")
