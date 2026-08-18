#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CASE_TABLE = ROOT / "cases" / "high48_80m.csv"


def stem(row):
    kn = f"{float(row['kn']):g}"
    rt = f"{float(row['rt']):g}".replace(".", "p")
    if row["model"] == "HS":
        prefix = "ThermalCavity_HS_DSMC"
    else:
        prefix = f"ThermalCavity_{row['model']}"
    return f"{prefix}_Kn{kn}_RT{rt}_quarter_seed{int(row['seed'])}"


with tempfile.TemporaryDirectory(prefix="jfm-summary-test-") as tmp:
    base = Path(tmp)
    input_dir = base / "runs"
    output_dir = base / "summary"
    mpl_dir = base / "mpl"
    input_dir.mkdir()
    mpl_dir.mkdir()
    with CASE_TABLE.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    x = np.array([-0.25, 0.25])
    y = np.array([-0.25, 0.25])
    for index, row in enumerate(rows):
        value = 1.0e-4 * (1 + index % 3)
        shape = (2, 2)
        name = stem(row)
        np.savez_compressed(
            input_dir / f"{name}_raw.npz",
            x=x, y=y,
            ux=np.full(shape, value),
            uy=np.full(shape, -0.5 * value),
            T=np.full(shape, float(row["rt"]) + 0.1),
            rho=np.ones(shape),
        )
        metrics = {
            "particles": 80_000_000,
            "steps": 5_000_000,
            "profile_samples": 2_450_000,
            "last_block_velocity_rmse_vs_all_samples": value / 10,
        }
        (input_dir / f"{name}_metrics.json").write_text(
            json.dumps(metrics), encoding="utf-8"
        )
    env = dict(os.environ, MPLCONFIGDIR=str(mpl_dir))
    subprocess.run(
        [
            "python", str(ROOT / "tools" / "summarize_highstat.py"),
            "--input", str(input_dir), "--output", str(output_dir),
            "--case-table", str(CASE_TABLE), "--route", "test",
        ],
        check=True,
        env=env,
    )
    manifest = json.loads(
        (output_dir / "ALL_HIGHSTAT_ENSEMBLES.json").read_text(encoding="utf-8")
    )
    assert len(manifest) == 7
    assert len(list(output_dir.glob("*_RAW_UNFILTERED.dat"))) == 7
    assert len(list(output_dir.glob("*_RAW_UNFILTERED.npz"))) == 7
    assert len(list(output_dir.glob("*_profile_y0p25.csv"))) == 7
    assert len(list(output_dir.glob("*_diagnostic.png"))) == 7
print("[OK] synthetic end-to-end high-statistics summary test passed")
