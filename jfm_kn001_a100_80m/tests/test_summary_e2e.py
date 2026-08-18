#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SEEDS = (104729, 1299709, 15485863, 32452843, 49979687, 67867967)


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="jfm-kn001-sixseed-") as tmp:
        base = Path(tmp)
        runs = base / "runs"
        summary = base / "summary"
        runs.mkdir()
        x = np.linspace(-0.45, 0.45, 8)
        y = np.linspace(-0.45, 0.45, 8)
        xx, yy = np.meshgrid(x, y)
        for model_index, model in enumerate(("BGK", "SHAKHOV"), start=1):
            for seed_index, seed in enumerate(SEEDS):
                scale = model_index * 1.0e-4 * (1.0 + 0.01 * seed_index)
                ux = scale * xx * (1.0 - yy * yy)
                uy = -scale * yy * (1.0 - xx * xx)
                temp = 0.75 + 0.20 * yy
                rho = 1.0 - 0.10 * yy
                blocks = np.stack(
                    [
                        np.stack([field * (0.98 + 0.005 * b) for b in range(5)])
                        for field in (ux, uy)
                    ]
                )
                ux_blocks, uy_blocks = blocks
                t_blocks = np.stack([temp for _ in range(5)])
                rho_blocks = np.stack([rho for _ in range(5)])
                stem = runs / f"ThermalCavity_{model}_Kn0.01_RT0p5_quarter_seed{seed}"
                np.savez_compressed(
                    str(stem) + "_raw.npz",
                    x=x, y=y, ux=ux, uy=uy, T=temp, rho=rho,
                    ux_time_blocks=ux_blocks, uy_time_blocks=uy_blocks,
                    T_time_blocks=t_blocks, rho_time_blocks=rho_blocks,
                    samples_per_time_block=np.full(5, 100, dtype=np.int64),
                )
                metrics = {
                    "solver_version": "synthetic-test",
                    "particles": 80_000_000,
                    "steps": 5_000_000,
                    "profile_samples": 2_450_000,
                    "last_block_velocity_rmse_vs_all_samples": 1.0e-7,
                }
                Path(str(stem) + "_metrics.json").write_text(
                    json.dumps(metrics), encoding="utf-8"
                )
        subprocess.run(
            [
                sys.executable, str(ROOT / "tools" / "summarize_six_seed.py"),
                "--input", str(runs), "--output", str(summary),
                "--case-table", str(ROOT / "cases" / "kn001_a100_80m.csv"),
                "--no-plots",
            ],
            check=True,
        )
        with (summary / "ALL_ENSEMBLES_summary.csv").open(
            newline="", encoding="utf-8"
        ) as handle:
            rows = list(csv.DictReader(handle))
        assert len(rows) == 2
        assert all(int(row["number_of_independent_runs"]) == 6 for row in rows)
        for model in ("BGK", "SHAKHOV"):
            path = summary / f"{model}_Kn0.01_RT0p5_RAW_UNFILTERED_six_seed_ensemble.npz"
            with np.load(path, allow_pickle=False) as data:
                assert data["kinetic_energy_cross_seed_triple_values"].shape == (20,)
                assert data["kinetic_energy_cross_seed_pair_values"].shape == (15,)
                assert data["kinetic_energy_cross_seed_time_blocks"].shape == (5,)
                assert float(data["kinetic_energy_cross_seed_noise_unbiased"]) > 0.0
        print("[OK] synthetic six-seed U-statistic and summary test passed")


if __name__ == "__main__":
    main()
