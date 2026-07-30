from __future__ import annotations

from dataclasses import asdict, dataclass
import argparse
import json
from pathlib import Path
import numpy as np

from .vhs_sbt import (
    KB,
    MASS_AR,
    VHSParameters,
    sbt_collide_cell,
    vhs_cross_section,
)


@dataclass(frozen=True)
class FrequencyValidationConfig:
    particles: int = 80
    temperature: float = 300.0
    fnum: float = 550.0
    dt: float = 1.0e-9
    cell_volume: float = 1.0e-18
    trials: int = 5000
    seed: int = 1


@dataclass(frozen=True)
class RelaxationValidationConfig:
    particles: int = 200
    temperatures: tuple[float, float, float] = (
        600.0,
        150.0,
        150.0,
    )
    fnum: float = 3000.0
    dt: float = 1.0e-9
    cell_volume: float = 1.0e-18
    sweeps: int = 100
    seed: int = 42


def max_candidate_probability(
    velocities: np.ndarray,
    fnum: float,
    dt: float,
    cell_volume: float,
    params: VHSParameters = VHSParameters(),
) -> float:
    count = len(velocities)
    maximum = 0.0
    for first in range(count):
        for second in range(first + 1, count):
            speed = float(
                np.linalg.norm(
                    velocities[first] - velocities[second]
                )
            )
            probability = (
                (count - 1)
                * fnum
                * dt
                * float(vhs_cross_section(speed, params))
                * speed
                / cell_volume
            )
            maximum = max(maximum, probability)
    return maximum


def exact_pair_expectation(
    velocities: np.ndarray,
    fnum: float,
    dt: float,
    cell_volume: float,
    params: VHSParameters = VHSParameters(),
) -> float:
    """Expected accepted SBT count before probability clipping.

    Sequential SBT chooses one random partner for each first particle and
    multiplies its acceptance probability by the number of remaining partners.
    Averaging over partner selection therefore recovers the sum over all
    unordered molecular pairs.
    """
    total = 0.0
    for first in range(len(velocities)):
        relative = velocities[first + 1 :] - velocities[first]
        if len(relative) == 0:
            continue
        speeds = np.linalg.norm(relative, axis=1)
        total += float(
            np.sum(vhs_cross_section(speeds, params) * speeds)
        )
    return fnum * dt * total / cell_volume


def directional_temperatures(
    velocities: np.ndarray,
    mass: float = MASS_AR,
) -> np.ndarray:
    centered = velocities - np.mean(velocities, axis=0)
    return mass * np.mean(centered**2, axis=0) / KB


def anisotropy(temperatures: np.ndarray) -> float:
    temperatures = np.asarray(temperatures, dtype=float)
    return float(
        (temperatures.max() - temperatures.min())
        / temperatures.mean()
    )


def run_frequency_validation(
    cfg: FrequencyValidationConfig = FrequencyValidationConfig(),
) -> dict[str, object]:
    rng = np.random.default_rng(cfg.seed)
    thermal_std = np.sqrt(KB * cfg.temperature / MASS_AR)
    velocities = rng.normal(
        0.0,
        thermal_std,
        (cfg.particles, 3),
    )
    expected = exact_pair_expectation(
        velocities,
        cfg.fnum,
        cfg.dt,
        cfg.cell_volume,
    )
    counts = np.empty(cfg.trials)
    for trial in range(cfg.trials):
        _, accepted = sbt_collide_cell(
            velocities,
            cfg.fnum,
            cfg.dt,
            cfg.cell_volume,
            np.random.default_rng(cfg.seed + 10_000 + trial),
        )
        counts[trial] = accepted
    measured = float(counts.mean())
    standard_error = float(
        counts.std(ddof=1) / np.sqrt(cfg.trials)
    )
    relative_error = float(abs(measured - expected) / expected)
    return {
        "config": asdict(cfg),
        "expected_collisions_per_sweep": expected,
        "measured_collisions_per_sweep": measured,
        "standard_error": standard_error,
        "relative_error": relative_error,
        "z_score": float(
            (measured - expected) / standard_error
        ),
        "maximum_initial_candidate_probability": (
            max_candidate_probability(
                velocities,
                cfg.fnum,
                cfg.dt,
                cfg.cell_volume,
            )
        ),
    }


def run_relaxation_validation(
    cfg: RelaxationValidationConfig = RelaxationValidationConfig(),
) -> dict[str, object]:
    rng = np.random.default_rng(cfg.seed)
    thermal_std = np.sqrt(
        KB * np.asarray(cfg.temperatures) / MASS_AR
    )
    velocities = (
        rng.normal(size=(cfg.particles, 3)) * thermal_std
    )
    velocities -= velocities.mean(axis=0)
    initial_temperatures = directional_temperatures(velocities)
    initial_mean_velocity = velocities.mean(axis=0).copy()
    initial_energy = float(np.sum(velocities**2))
    accepted_collisions = 0

    for sweep in range(cfg.sweeps):
        velocities, accepted = sbt_collide_cell(
            velocities,
            cfg.fnum,
            cfg.dt,
            cfg.cell_volume,
            np.random.default_rng(cfg.seed + 1000 + sweep),
        )
        accepted_collisions += accepted

    final_temperatures = directional_temperatures(velocities)
    final_energy = float(np.sum(velocities**2))
    return {
        "config": asdict(cfg),
        "initial_directional_temperatures": (
            initial_temperatures.tolist()
        ),
        "final_directional_temperatures": (
            final_temperatures.tolist()
        ),
        "initial_anisotropy": anisotropy(
            initial_temperatures
        ),
        "final_anisotropy": anisotropy(final_temperatures),
        "anisotropy_ratio": (
            anisotropy(final_temperatures)
            / anisotropy(initial_temperatures)
        ),
        "total_temperature_relative_change": float(
            abs(
                final_temperatures.sum()
                - initial_temperatures.sum()
            )
            / initial_temperatures.sum()
        ),
        "velocity_energy_relative_change": float(
            abs(final_energy - initial_energy) / initial_energy
        ),
        "mean_velocity_absolute_change": float(
            np.max(
                np.abs(
                    velocities.mean(axis=0)
                    - initial_mean_velocity
                )
            )
        ),
        "accepted_collisions": accepted_collisions,
    }


def run_validation(output: str | Path) -> dict[str, object]:
    result = {
        "frequency": run_frequency_validation(),
        "anisotropic_relaxation": (
            run_relaxation_validation()
        ),
    }
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(
        json.dumps(result, indent=2),
        encoding="utf-8",
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        default="outputs/stage5_collision_validation",
    )
    args = parser.parse_args()
    print(json.dumps(run_validation(args.output), indent=2))


if __name__ == "__main__":
    main()
