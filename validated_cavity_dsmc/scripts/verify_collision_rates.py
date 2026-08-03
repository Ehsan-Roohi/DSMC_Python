"""Monte Carlo check that each BT trial generator is collision-rate unbiased."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dsmc_cavity.collisions import generate_bt_trials
from dsmc_cavity.config import SimulationConfig
from dsmc_cavity.physics import vhs_sigma_g_scalar


MODELS = ("sbt", "gbt", "ssbt", "sgbt")


def sigma_g(config, first: int, second: int, velocity: np.ndarray) -> float:
    return vhs_sigma_g_scalar(
        np.linalg.norm(velocity[first] - velocity[second]),
        config.diameter_ref,
        config.temperature_ref,
        config.viscosity_index,
        config.mass,
    )


def verify(samples: int, particles: int) -> list[dict[str, float | str | bool]]:
    initial = np.random.default_rng(90210)
    position = initial.random((particles, 2)) * 1.0e-6
    velocity = initial.normal(0.0, 300.0, size=(particles, 3))
    groups = [np.arange(particles)]
    config = SimulationConfig(
        nx=1,
        ny=1,
        particles_per_cell=particles,
        steps=2,
        warmup_steps=1,
        gbt_fraction=0.5,
    )
    exact = sum(
        sigma_g(config, i, j, velocity)
        for i in range(particles)
        for j in range(i + 1, particles)
    )
    rows = []
    for model in MODELS:
        estimates = np.empty(samples)
        for sample in range(samples):
            trials = generate_bt_trials(
                model, groups, position, config, np.random.default_rng(sample + 17)
            )
            estimates[sample] = sum(
                multiplier * sigma_g(config, int(i), int(j), velocity)
                for i, j, multiplier in zip(
                    trials.first, trials.second, trials.multiplier
                )
            )
        ratio = estimates.mean() / exact
        standard_error = estimates.std(ddof=1) / np.sqrt(samples) / exact
        rows.append(
            {
                "model": model,
                "samples": samples,
                "particles": particles,
                "mean_rate_ratio_to_all_pairs": float(ratio),
                "standard_error": float(standard_error),
                "z_score_from_unity": float((ratio - 1.0) / standard_error),
                "pass_abs_bias_le_0p01": bool(abs(ratio - 1.0) <= 0.01),
            }
        )
    return rows


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--samples", type=int, default=5000)
    p.add_argument("--particles", type=int, default=20)
    p.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results" / "collision_rate_verification.csv",
    )
    args = p.parse_args()
    rows = verify(args.samples, args.particles)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        print(row)
    return 0 if all(row["pass_abs_bias_le_0p01"] for row in rows) else 2


if __name__ == "__main__":
    raise SystemExit(main())
