#!/usr/bin/env python3
from __future__ import annotations

import csv
import importlib.util
from pathlib import Path
import tempfile

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def load_summarizer():
    path = ROOT / "tools" / "summarize_ensembles.py"
    spec = importlib.util.spec_from_file_location("summarize_ensembles", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_case_table() -> None:
    with (ROOT / "cases" / "kn001_heavy.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 9
    assert {row["model"] for row in rows} == {"HS", "BGK", "SHAKHOV"}
    assert {row["kn"] for row in rows} == {"0.01"}
    assert {row["rt"] for row in rows} == {"0.5"}
    assert {int(row["seed"]) for row in rows} == {42, 271828, 314159}
    keys = {(row["model"], row["seed"]) for row in rows}
    assert len(keys) == 9


def test_cross_seed_definition_is_fully_independent() -> None:
    # Synthetic fields make the three expected terms exactly auditable.
    ux = np.asarray([[[1.0]], [[2.0]], [[4.0]]])
    uy = np.asarray([[[0.5]], [[1.0]], [[2.0]]])
    rho = np.asarray([[[10.0]], [[20.0]], [[30.0]]])
    pairs = np.asarray([
        np.sum(rho[2] * (ux[0] * ux[1] + uy[0] * uy[1])),
        np.sum(rho[1] * (ux[0] * ux[2] + uy[0] * uy[2])),
        np.sum(rho[0] * (ux[1] * ux[2] + uy[1] * uy[2])),
    ])
    assert np.array_equal(pairs, np.asarray([75.0, 100.0, 100.0]))
    assert np.mean(pairs) == 275.0 / 3.0


def test_expected_seed_contract() -> None:
    module = load_summarizer()
    assert module.EXPECTED_SEEDS == (42, 271828, 314159)


def test_summarizer_writes_third_seed_density_estimator() -> None:
    module = load_summarizer()
    with tempfile.TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        output = root / "summary"
        output.mkdir()
        x = np.asarray([0.0, 1.0])
        y = np.asarray([0.0, 1.0])
        ux_values = (1.0, 2.0, 4.0)
        uy_values = (0.5, 1.0, 2.0)
        rho_values = (10.0, 20.0, 30.0)
        runs = []
        for seed, ux_value, uy_value, rho_value in zip(
            (42, 271828, 314159), ux_values, uy_values, rho_values
        ):
            path = root / f"seed_{seed}.npz"
            np.savez_compressed(
                path,
                x=x,
                y=y,
                ux=np.full((2, 2), ux_value),
                uy=np.full((2, 2), uy_value),
                T=np.ones((2, 2)),
                rho=np.full((2, 2), rho_value),
            )
            runs.append(
                {
                    "path": path,
                    "seed": seed,
                    "metrics": {
                        "particles": 22_000_000,
                        "steps": 2_000_000,
                        "profile_samples": 800_000,
                        "solver_version": "synthetic-test",
                        "last_block_velocity_rmse_vs_all_samples": 0.0,
                    },
                }
            )
        summary = module.summarize_group(
            ("HS", 0.01, 0.5), runs, output, False
        )
        # Four unit-area cells multiply the single-cell values [75,100,100].
        assert np.allclose(
            summary["kinetic_energy_cross_seed_pair_values"],
            np.asarray([300.0, 400.0, 400.0]),
        )
        assert np.isclose(
            summary["kinetic_energy_cross_seed_noise_unbiased"],
            1100.0 / 3.0,
        )
        assert summary["kinetic_energy_quality_status"] == "PROVISIONAL"


if __name__ == "__main__":
    test_case_table()
    test_cross_seed_definition_is_fully_independent()
    test_expected_seed_contract()
    test_summarizer_writes_third_seed_density_estimator()
    print("[OK] bundle tests passed")
