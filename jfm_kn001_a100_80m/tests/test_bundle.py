#!/usr/bin/env python3
from __future__ import annotations

import csv
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXPECTED_SOLVER_SHA256 = "c20d1a5fac0ab7e019e695a68e7ddecc64571f63102a8a957f61d766d6b4db55"


def main() -> None:
    table = ROOT / "cases" / "kn001_a100_80m.csv"
    with table.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 12
    assert {row["model"] for row in rows} == {"BGK", "SHAKHOV"}
    assert all(float(row["kn"]) == 0.01 for row in rows)
    assert all(float(row["rt"]) == 0.5 for row in rows)
    for model in ("BGK", "SHAKHOV"):
        seeds = [int(row["seed"]) for row in rows if row["model"] == model]
        assert len(seeds) == 6 and len(set(seeds)) == 6
    solver = ROOT / "solver" / "JFM_bgk_shakhov_quarter.py"
    digest = hashlib.sha256(solver.read_bytes()).hexdigest()
    assert digest == EXPECTED_SOLVER_SHA256, digest
    slurm = (ROOT / "scripts" / "submit.sh").read_text(encoding="utf-8")
    assert "--constraint=a100-80g" in slurm
    assert "--qos" not in slurm
    run = (ROOT / "scripts" / "run_a100.slurm").read_text(encoding="utf-8")
    for token in ("--particles 80000000", "--steps 5000000", "--sample-start 100000", "--time-blocks 49"):
        assert token in run, token
    print("[OK] 12-case A100 bundle and solver provenance validated")


if __name__ == "__main__":
    main()
