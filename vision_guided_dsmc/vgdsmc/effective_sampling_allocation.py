from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import argparse
import json
import numpy as np

from .dvm_shakhov_corrected import (
    ShakhovReferenceConfig,
    solve_shakhov_reference,
)
from .physical_adaptive import field_error
from .sampling_allocation import (
    collect_physical_snapshots,
    estimate_from_sampling_counts,
    exact_sampling_counts,
    full_trajectory_mean,
)
from .sbt_solver import run_physical_cavity
from .vhs_model import KB, MASS_AR, PhysicalCavityConfig


@dataclass(frozen=True)
class EffectiveSamplingConfig:
    nx: int = 6
    ny: int = 6
    particles_per_cell: int = 20
    warm_steps: int = 40
    pilot_steps: int = 40
    evaluation_steps: int = 80
    base_samples_per_cell: int = 20
    minimum_samples: int = 15
    maximum_samples: int = 25
    nv: int = 6
    dvm_max_steps: int = 900
    seeds: tuple[int, ...] = (
        399,
        411,
        422,
        433,
        444,
        455,
        466,
        477,
        488,
        499,
    )
    conditions: tuple[tuple[float, float], ...] = (
        (0.05, 40.0),
        (0.10, 40.0),
        (0.20, 40.0),
    )


def lag1_autocorrelation(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 3 or values.shape[0] < 3:
        raise ValueError("values must have shape (time, ny, nx) with time >= 3")
    first = values[:-1]
    second = values[1:]
    first_centered = first - np.mean(first, axis=0)
    second_centered = second - np.mean(second, axis=0)
    covariance = np.mean(first_centered * second_centered, axis=0)
    denominator = np.sqrt(
        np.mean(first_centered**2, axis=0)
        * np.mean(second_centered**2, axis=0)
    )
    correlation = np.divide(
        covariance,
        denominator,
        out=np.zeros_like(covariance),
        where=denominator > 1.0e-20,
    )
    return np.clip(correlation, -0.5, 0.95)


def _normalized_error_series(
    snapshots: dict[str, np.ndarray],
    cfg: PhysicalCavityConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    temperature = np.asarray(snapshots["T"], dtype=np.float64)
    density = np.asarray(snapshots["rho"], dtype=np.float64)
    speed = np.hypot(
        np.asarray(snapshots["u"], dtype=np.float64),
        np.asarray(snapshots["v"], dtype=np.float64),
    )
    temperature_scale = np.maximum(
        np.mean(np.abs(temperature), axis=0),
        1.0e-12,
    )
    thermal_speed = np.sqrt(2.0 * KB * cfg.t0 / MASS_AR)
    return (
        temperature / temperature_scale[None],
        density,
        speed / max(float(thermal_speed), 1.0e-12),
    )


def composite_variance_priority(
    snapshots: dict[str, np.ndarray],
    cfg: PhysicalCavityConfig,
    *,
    autocorrelation_corrected: bool,
) -> np.ndarray:
    """Return the Neyman score implied by the field-error metric.

    The error metric weights are 0.45 for relative temperature, 0.30 for
    normalized density, and 0.25 for speed scaled by thermal velocity.  The
    optimal independent-sample allocation is proportional to the square root of
    the weighted local variance.  With autocorrelation correction, each field
    variance is multiplied by ``(1 + rho_1)/(1 - rho_1)`` using pilot lag-one
    correlation, clipped to a stable finite range.
    """
    temperature, density, speed = _normalized_error_series(snapshots, cfg)
    components = []
    for values, weight in (
        (temperature, 0.45),
        (density, 0.30),
        (speed, 0.25),
    ):
        variance = np.var(values, axis=0, ddof=1)
        if autocorrelation_corrected:
            rho1 = lag1_autocorrelation(values)
            inflation = np.clip((1.0 + rho1) / (1.0 - rho1), 0.25, 20.0)
            variance = variance * inflation
        components.append(weight**2 * variance)
    effective_standard_deviation = np.sqrt(np.maximum(sum(components), 0.0))
    mean = max(float(np.mean(effective_standard_deviation)), 1.0e-14)
    return effective_standard_deviation / mean


def _paired_statistics(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    mean = float(np.mean(values))
    standard_error = (
        float(np.std(values, ddof=1) / np.sqrt(len(values)))
        if len(values) > 1
        else 0.0
    )
    return {
        "mean": mean,
        "median": float(np.median(values)),
        "standard_error": standard_error,
        "ci95_low": mean - 1.96 * standard_error,
        "ci95_high": mean + 1.96 * standard_error,
    }


def _condition_summaries(
    rows: list[dict[str, object]],
    policy: str,
) -> dict[str, dict[str, float | int]]:
    output: dict[str, dict[str, float | int]] = {}
    for knudsen in sorted({float(row["knudsen"]) for row in rows}):
        selected = [
            row
            for row in rows
            if row["policy"] == policy
            and float(row["knudsen"]) == knudsen
        ]
        ratios = np.asarray(
            [float(row["sampling_error_ratio"]) for row in selected]
        )
        statistics = _paired_statistics(ratios)
        output[f"Kn_{knudsen:.2f}"] = {
            **statistics,
            "run_count": len(selected),
            "improved_runs": int(np.sum(ratios < 1.0)),
            "mean_improvement_percent": 100.0 * (1.0 - statistics["mean"]),
        }
    return output


def run_effective_sampling_benchmark(
    output_dir: str | Path,
    cfg: EffectiveSamplingConfig = EffectiveSamplingConfig(),
) -> dict[str, object]:
    if cfg.maximum_samples > cfg.evaluation_steps:
        raise ValueError("maximum_samples exceeds available evaluation snapshots")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    references: dict[tuple[float, float], dict[str, np.ndarray]] = {}
    reference_diagnostics: dict[str, dict[str, float]] = {}
    for knudsen, delta_temperature in cfg.conditions:
        reference_cfg = ShakhovReferenceConfig(
            nx=cfg.nx,
            ny=cfg.ny,
            nv=cfg.nv,
            knudsen=knudsen,
            t_left=300.0 + delta_temperature,
            t_right=300.0 - delta_temperature,
            t_top=300.0,
            t_bottom=300.0,
            max_steps=cfg.dvm_max_steps,
        )
        reference = solve_shakhov_reference(reference_cfg)
        references[(knudsen, delta_temperature)] = reference
        reference_diagnostics[f"kn{knudsen:.3f}_dt{delta_temperature:.0f}"] = {
            "iterations": float(reference["iterations"]),
            "final_residual": float(reference["residual_history"][-1]),
        }

    rows: list[dict[str, object]] = []
    saved_maps: dict[str, np.ndarray] = {}
    policies = {
        "weighted_raw_variance": False,
        "weighted_effective_variance": True,
    }
    for knudsen, delta_temperature in cfg.conditions:
        reference = references[(knudsen, delta_temperature)]
        for seed in cfg.seeds:
            simulation_cfg = PhysicalCavityConfig(
                nx=cfg.nx,
                ny=cfg.ny,
                particles_per_cell=cfg.particles_per_cell,
                knudsen=knudsen,
                t_left=300.0 + delta_temperature,
                t_right=300.0 - delta_temperature,
                t_top=300.0,
                t_bottom=300.0,
                steps=cfg.warm_steps,
                sample_start=max(1, cfg.warm_steps // 2),
                seed=seed,
            )
            _, warm_state, warm_diagnostics = run_physical_cavity(
                simulation_cfg,
                return_state=True,
            )
            pilot_snapshots, pilot_state, pilot_diagnostics = (
                collect_physical_snapshots(
                    warm_state.copy(),
                    simulation_cfg,
                    cfg.pilot_steps,
                    seed=seed + 30000,
                )
            )
            evaluation_snapshots, _, evaluation_diagnostics = (
                collect_physical_snapshots(
                    pilot_state.copy(),
                    simulation_cfg,
                    cfg.evaluation_steps,
                    seed=seed + 40000,
                )
            )
            full_mean = full_trajectory_mean(evaluation_snapshots)
            uniform_counts = np.full(
                (cfg.ny, cfg.nx),
                cfg.base_samples_per_cell,
                dtype=np.int64,
            )
            selection_seed = seed + 50000
            uniform_estimate = estimate_from_sampling_counts(
                evaluation_snapshots,
                uniform_counts,
                seed=selection_seed,
            )
            uniform_sampling_error = field_error(uniform_estimate, full_mean)
            uniform_dvm_error = field_error(uniform_estimate, reference)
            full_dvm_error = field_error(full_mean, reference)
            case_key = f"kn{knudsen:.3f}_dt{delta_temperature:.0f}_s{seed}"

            for policy_name, autocorrelation_corrected in policies.items():
                priority = composite_variance_priority(
                    pilot_snapshots,
                    simulation_cfg,
                    autocorrelation_corrected=autocorrelation_corrected,
                )
                counts = exact_sampling_counts(
                    priority,
                    cfg.base_samples_per_cell,
                    cfg.minimum_samples,
                    cfg.maximum_samples,
                )
                estimate = estimate_from_sampling_counts(
                    evaluation_snapshots,
                    counts,
                    seed=selection_seed,
                )
                sampling_error = field_error(estimate, full_mean)
                dvm_error = field_error(estimate, reference)
                saved_maps[f"priority_{policy_name}_{case_key}"] = priority.astype(
                    np.float32
                )
                saved_maps[f"counts_{policy_name}_{case_key}"] = counts.astype(
                    np.int16
                )
                rows.append(
                    {
                        "knudsen": knudsen,
                        "delta_temperature": delta_temperature,
                        "seed": seed,
                        "policy": policy_name,
                        "uniform_observations": int(uniform_counts.sum()),
                        "adaptive_observations": int(counts.sum()),
                        "observation_ratio": float(
                            counts.sum() / uniform_counts.sum()
                        ),
                        "minimum_samples": int(counts.min()),
                        "maximum_samples": int(counts.max()),
                        "mean_absolute_fractional_count_change": float(
                            np.mean(
                                np.abs(
                                    counts / cfg.base_samples_per_cell - 1.0
                                )
                            )
                        ),
                        "priority_std": float(np.std(priority)),
                        "uniform_sampling_error": uniform_sampling_error,
                        "adaptive_sampling_error": sampling_error,
                        "sampling_error_ratio": sampling_error
                        / max(uniform_sampling_error, 1.0e-14),
                        "uniform_dvm_error": uniform_dvm_error,
                        "adaptive_dvm_error": dvm_error,
                        "dvm_error_ratio": dvm_error
                        / max(uniform_dvm_error, 1.0e-14),
                        "full_trajectory_dvm_error": full_dvm_error,
                        "warm_collisions": float(
                            warm_diagnostics["accepted_collisions"]
                        ),
                        "pilot_collisions": float(
                            pilot_diagnostics["accepted_collisions"]
                        ),
                        "evaluation_collisions": float(
                            evaluation_diagnostics["accepted_collisions"]
                        ),
                    }
                )

    policy_summaries: dict[str, dict[str, object]] = {}
    for policy_name in policies:
        selected = [row for row in rows if row["policy"] == policy_name]
        sampling_ratios = np.asarray(
            [float(row["sampling_error_ratio"]) for row in selected]
        )
        dvm_ratios = np.asarray(
            [float(row["dvm_error_ratio"]) for row in selected]
        )
        sampling_statistics = _paired_statistics(sampling_ratios)
        dvm_statistics = _paired_statistics(dvm_ratios)
        condition_summaries = _condition_summaries(rows, policy_name)
        improving_conditions = sum(
            float(metrics["mean"]) < 1.0
            for metrics in condition_summaries.values()
        )
        policy_summaries[policy_name] = {
            "sampling_error": {
                **sampling_statistics,
                "run_count": len(selected),
                "improved_runs": int(np.sum(sampling_ratios < 1.0)),
                "mean_improvement_percent": 100.0
                * (1.0 - sampling_statistics["mean"]),
            },
            "dvm_error": {
                **dvm_statistics,
                "improved_runs": int(np.sum(dvm_ratios < 1.0)),
                "mean_improvement_percent": 100.0
                * (1.0 - dvm_statistics["mean"]),
            },
            "condition_summaries": condition_summaries,
            "improving_condition_count": improving_conditions,
            "primary_success": bool(
                sampling_statistics["ci95_high"] < 1.0
                and improving_conditions >= 2
            ),
            "mean_absolute_fractional_count_change": float(
                np.mean(
                    [
                        float(row["mean_absolute_fractional_count_change"])
                        for row in selected
                    ]
                )
            ),
        }

    summary: dict[str, object] = {
        "stage": 21,
        "description": (
            "Field-error-weighted Neyman sampling with and without lag-one autocorrelation correction"
        ),
        "configuration": {
            "nx": cfg.nx,
            "ny": cfg.ny,
            "particles_per_cell": cfg.particles_per_cell,
            "warm_steps": cfg.warm_steps,
            "pilot_steps": cfg.pilot_steps,
            "evaluation_steps": cfg.evaluation_steps,
            "base_samples_per_cell": cfg.base_samples_per_cell,
            "minimum_samples": cfg.minimum_samples,
            "maximum_samples": cfg.maximum_samples,
            "observations_per_estimator": cfg.nx
            * cfg.ny
            * cfg.base_samples_per_cell,
            "seeds": list(cfg.seeds),
            "conditions": [list(value) for value in cfg.conditions],
        },
        "reference_diagnostics": reference_diagnostics,
        "rows": rows,
        "policy_summaries": policy_summaries,
        "success_rule": (
            "The upper 95% confidence bound of sampling-error ratio must be "
            "below one and at least two Knudsen-condition means must be below one."
        ),
        "scope_guard": (
            "This is the final fixed-trajectory sampling-feasibility test. "
            "It does not alter particle dynamics and does not claim wall-clock speedup."
        ),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    np.savez_compressed(output_dir / "effective_sampling_maps.npz", **saved_maps)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Stage 21 effective-variance sampling allocation"
    )
    parser.add_argument("--output-dir", default="outputs/stage21_effective_sampling")
    parser.add_argument("--nx", type=int, default=6)
    parser.add_argument("--ny", type=int, default=6)
    parser.add_argument("--ppc", type=int, default=20)
    parser.add_argument("--warm-steps", type=int, default=40)
    parser.add_argument("--pilot-steps", type=int, default=40)
    parser.add_argument("--evaluation-steps", type=int, default=80)
    parser.add_argument("--base-samples", type=int, default=20)
    parser.add_argument("--minimum-samples", type=int, default=15)
    parser.add_argument("--maximum-samples", type=int, default=25)
    parser.add_argument("--nv", type=int, default=6)
    parser.add_argument("--dvm-max-steps", type=int, default=900)
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=list(EffectiveSamplingConfig().seeds),
    )
    args = parser.parse_args()
    summary = run_effective_sampling_benchmark(
        args.output_dir,
        EffectiveSamplingConfig(
            nx=args.nx,
            ny=args.ny,
            particles_per_cell=args.ppc,
            warm_steps=args.warm_steps,
            pilot_steps=args.pilot_steps,
            evaluation_steps=args.evaluation_steps,
            base_samples_per_cell=args.base_samples,
            minimum_samples=args.minimum_samples,
            maximum_samples=args.maximum_samples,
            nv=args.nv,
            dvm_max_steps=args.dvm_max_steps,
            seeds=tuple(args.seeds),
        ),
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
