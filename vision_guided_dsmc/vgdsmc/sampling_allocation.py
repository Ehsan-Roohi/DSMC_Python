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
from .physical_adaptive import exact_budget_ppc, field_error
from .sbt_solver import (
    collide_vhs_sbt,
    run_physical_cavity,
    sample_physical_state,
)
from .vhs_model import (
    PhysicalCavityConfig,
    PhysicalParticleState,
    apply_diffuse_walls,
)


@dataclass(frozen=True)
class SamplingAllocationConfig:
    nx: int = 6
    ny: int = 6
    particles_per_cell: int = 20
    warm_steps: int = 40
    pilot_steps: int = 20
    evaluation_steps: int = 60
    base_samples_per_cell: int = 20
    nv: int = 6
    dvm_max_steps: int = 900
    seeds: tuple[int, ...] = (
        288,
        299,
        311,
        322,
        333,
        344,
        355,
        366,
        377,
        388,
    )
    conditions: tuple[tuple[float, float], ...] = (
        (0.05, 40.0),
        (0.10, 40.0),
        (0.20, 40.0),
    )
    policies: tuple[tuple[str, int, int], ...] = (
        ("mild", 15, 25),
        ("strong", 10, 30),
    )


def collect_physical_snapshots(
    state: PhysicalParticleState,
    cfg: PhysicalCavityConfig,
    steps: int,
    seed: int,
) -> tuple[dict[str, np.ndarray], PhysicalParticleState, dict[str, float]]:
    """Advance one fixed DSMC trajectory and retain instantaneous cell moments."""
    if steps <= 0:
        raise ValueError("steps must be positive")
    rng = np.random.default_rng(seed)
    stored: dict[str, list[np.ndarray]] = {
        "T": [],
        "rho": [],
        "u": [],
        "v": [],
        "w": [],
    }
    collisions = 0
    for _ in range(steps):
        state.pos += state.vel[:, :2] * cfg.dt
        apply_diffuse_walls(state, cfg, rng)
        collisions += collide_vhs_sbt(state, cfg, rng)
        fields = sample_physical_state(state, cfg)
        for name in stored:
            stored[name].append(np.asarray(fields[name], dtype=np.float64))
    snapshots = {
        name: np.stack(values, axis=0)
        for name, values in stored.items()
    }
    diagnostics = {
        "accepted_collisions": float(collisions),
        "collisions_per_particle_step": float(
            collisions / max(len(state.pos) * steps, 1)
        ),
        "dt": float(cfg.dt),
    }
    return snapshots, state, diagnostics


def _robust_unit_scale(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    low, high = np.percentile(values, [5.0, 95.0])
    if high - low <= 1.0e-14:
        return np.zeros_like(values)
    return np.clip((values - low) / (high - low), 0.0, 1.0)


def pilot_variance_priority(
    snapshots: dict[str, np.ndarray],
) -> np.ndarray:
    """Estimate a dimensionless local Neyman-allocation priority.

    The score uses pilot temporal standard deviations of temperature, normalized
    density, and speed.  It is computed before the evaluation trajectory, so no
    evaluation samples leak into the allocation decision.
    """
    required = {"T", "rho", "u", "v"}
    if not required.issubset(snapshots):
        raise ValueError("Snapshots must contain T, rho, u, and v")
    shape = np.asarray(snapshots["T"]).shape
    if len(shape) != 3 or shape[0] < 2:
        raise ValueError("Snapshots must have shape (time, ny, nx) with time >= 2")
    if any(np.asarray(snapshots[name]).shape != shape for name in required):
        raise ValueError("All snapshot fields must share a common shape")

    temperature = np.asarray(snapshots["T"], dtype=np.float64)
    density = np.asarray(snapshots["rho"], dtype=np.float64)
    speed = np.hypot(
        np.asarray(snapshots["u"], dtype=np.float64),
        np.asarray(snapshots["v"], dtype=np.float64),
    )
    relative_temperature_std = np.std(temperature, axis=0, ddof=1) / np.maximum(
        np.mean(np.abs(temperature), axis=0),
        1.0e-12,
    )
    relative_density_std = np.std(density, axis=0, ddof=1) / np.maximum(
        np.mean(np.abs(density), axis=0),
        1.0e-12,
    )
    speed_std = np.std(speed, axis=0, ddof=1)

    priority = (
        0.50 * _robust_unit_scale(relative_temperature_std)
        + 0.30 * _robust_unit_scale(relative_density_std)
        + 0.20 * _robust_unit_scale(speed_std)
    )
    return np.asarray(priority, dtype=np.float64)


def exact_sampling_counts(
    priority: np.ndarray,
    base_samples_per_cell: int,
    minimum_samples: int,
    maximum_samples: int,
) -> np.ndarray:
    priority = np.asarray(priority, dtype=np.float64)
    if priority.ndim != 2 or not np.isfinite(priority).all():
        raise ValueError("priority must be a finite two-dimensional array")
    if not 1 <= minimum_samples <= base_samples_per_cell <= maximum_samples:
        raise ValueError(
            "Require 1 <= minimum <= base_samples_per_cell <= maximum"
        )
    return exact_budget_ppc(
        priority,
        base_samples_per_cell,
        budget_ratio=1.0,
        min_ppc=minimum_samples,
        max_ppc=maximum_samples,
    )


def _nested_sample_indices(
    available_steps: int,
    cell_index: int,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed + 104729 * (cell_index + 1))
    return rng.permutation(available_steps)


def estimate_from_sampling_counts(
    snapshots: dict[str, np.ndarray],
    sample_counts: np.ndarray,
    seed: int,
) -> dict[str, np.ndarray]:
    """Build cell estimators from a shared nested temporal-sample ordering."""
    counts = np.asarray(sample_counts, dtype=np.int64)
    if counts.ndim != 2 or np.any(counts < 1):
        raise ValueError("sample_counts must be a positive two-dimensional map")
    first = np.asarray(next(iter(snapshots.values())))
    if first.ndim != 3 or first.shape[1:] != counts.shape:
        raise ValueError("Snapshot and sample-count shapes do not match")
    available_steps = first.shape[0]
    if int(counts.max()) > available_steps:
        raise ValueError("Requested samples exceed available trajectory snapshots")

    output = {
        name: np.zeros(counts.shape, dtype=np.float64)
        for name in ("T", "rho", "u", "v", "w")
    }
    ny, nx = counts.shape
    for j in range(ny):
        for i in range(nx):
            cell_index = j * nx + i
            order = _nested_sample_indices(available_steps, cell_index, seed)
            selected = order[: int(counts[j, i])]
            for name in output:
                values = np.asarray(snapshots[name], dtype=np.float64)
                output[name][j, i] = float(np.mean(values[selected, j, i]))
    return output


def full_trajectory_mean(
    snapshots: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    return {
        name: np.mean(np.asarray(values, dtype=np.float64), axis=0)
        for name, values in snapshots.items()
        if name in {"T", "rho", "u", "v", "w"}
    }


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


def _condition_summary(
    rows: list[dict[str, object]],
    policy_name: str,
) -> dict[str, dict[str, float | int]]:
    output: dict[str, dict[str, float | int]] = {}
    for knudsen in sorted({float(row["knudsen"]) for row in rows}):
        selected = [
            row
            for row in rows
            if row["policy"] == policy_name
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


def run_sampling_allocation_benchmark(
    output_dir: str | Path,
    cfg: SamplingAllocationConfig = SamplingAllocationConfig(),
) -> dict[str, object]:
    if cfg.base_samples_per_cell > cfg.evaluation_steps:
        raise ValueError("base_samples_per_cell cannot exceed evaluation_steps")
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
            "mean_temperature": float(np.mean(reference["T"])),
        }

    rows: list[dict[str, object]] = []
    saved_maps: dict[str, np.ndarray] = {}
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
            priority = pilot_variance_priority(pilot_snapshots)
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
            saved_maps[f"priority_{case_key}"] = priority.astype(np.float32)

            for policy_name, minimum_samples, maximum_samples in cfg.policies:
                if maximum_samples > cfg.evaluation_steps:
                    raise ValueError(
                        f"Policy {policy_name} requests more samples than available"
                    )
                counts = exact_sampling_counts(
                    priority,
                    cfg.base_samples_per_cell,
                    minimum_samples,
                    maximum_samples,
                )
                adaptive_estimate = estimate_from_sampling_counts(
                    evaluation_snapshots,
                    counts,
                    seed=selection_seed,
                )
                adaptive_sampling_error = field_error(
                    adaptive_estimate,
                    full_mean,
                )
                adaptive_dvm_error = field_error(adaptive_estimate, reference)
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
                        "adaptive_sampling_error": adaptive_sampling_error,
                        "sampling_error_ratio": adaptive_sampling_error
                        / max(uniform_sampling_error, 1.0e-14),
                        "sampling_improvement_percent": 100.0
                        * (
                            1.0
                            - adaptive_sampling_error
                            / max(uniform_sampling_error, 1.0e-14)
                        ),
                        "uniform_dvm_error": uniform_dvm_error,
                        "adaptive_dvm_error": adaptive_dvm_error,
                        "dvm_error_ratio": adaptive_dvm_error
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
    for policy_name, _, _ in cfg.policies:
        selected = [row for row in rows if row["policy"] == policy_name]
        sampling_ratios = np.asarray(
            [float(row["sampling_error_ratio"]) for row in selected]
        )
        dvm_ratios = np.asarray(
            [float(row["dvm_error_ratio"]) for row in selected]
        )
        sampling_statistics = _paired_statistics(sampling_ratios)
        dvm_statistics = _paired_statistics(dvm_ratios)
        condition_summaries = _condition_summary(rows, policy_name)
        improving_conditions = sum(
            float(value["mean"]) < 1.0
            for value in condition_summaries.values()
        )
        primary_success = (
            sampling_statistics["ci95_high"] < 1.0
            and improving_conditions >= 2
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
            "primary_success": bool(primary_success),
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
        "stage": 20,
        "description": (
            "Pilot-variance matched-observation sampling allocation on fixed DSMC trajectories"
        ),
        "configuration": {
            "nx": cfg.nx,
            "ny": cfg.ny,
            "particles_per_cell": cfg.particles_per_cell,
            "warm_steps": cfg.warm_steps,
            "pilot_steps": cfg.pilot_steps,
            "evaluation_steps": cfg.evaluation_steps,
            "base_samples_per_cell": cfg.base_samples_per_cell,
            "observations_per_estimator": cfg.nx
            * cfg.ny
            * cfg.base_samples_per_cell,
            "seeds": list(cfg.seeds),
            "conditions": [list(value) for value in cfg.conditions],
            "policies": [list(value) for value in cfg.policies],
        },
        "reference_diagnostics": reference_diagnostics,
        "rows": rows,
        "policy_summaries": policy_summaries,
        "primary_endpoint": (
            "paired field-error ratio relative to the full mean of the same "
            "evaluation trajectory"
        ),
        "secondary_endpoint": (
            "paired field-error ratio relative to the deterministic Shakhov-DVM reference"
        ),
        "success_rule": (
            "For a predeclared policy, the upper 95% confidence bound of the "
            "sampling-error ratio must be below one and at least two of three "
            "Knudsen-condition means must be below one."
        ),
        "scope_guard": (
            "This stage evaluates estimator allocation only. It does not claim "
            "DSMC wall-clock speedup because particle motion and collisions are "
            "identical for all estimators."
        ),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    np.savez_compressed(output_dir / "sampling_maps.npz", **saved_maps)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Stage 20 matched-budget temporal sampling allocation"
    )
    parser.add_argument("--output-dir", default="outputs/stage20_sampling")
    parser.add_argument("--nx", type=int, default=6)
    parser.add_argument("--ny", type=int, default=6)
    parser.add_argument("--ppc", type=int, default=20)
    parser.add_argument("--warm-steps", type=int, default=40)
    parser.add_argument("--pilot-steps", type=int, default=20)
    parser.add_argument("--evaluation-steps", type=int, default=60)
    parser.add_argument("--base-samples", type=int, default=20)
    parser.add_argument("--nv", type=int, default=6)
    parser.add_argument("--dvm-max-steps", type=int, default=900)
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=list(SamplingAllocationConfig().seeds),
    )
    args = parser.parse_args()
    summary = run_sampling_allocation_benchmark(
        args.output_dir,
        SamplingAllocationConfig(
            nx=args.nx,
            ny=args.ny,
            particles_per_cell=args.ppc,
            warm_steps=args.warm_steps,
            pilot_steps=args.pilot_steps,
            evaluation_steps=args.evaluation_steps,
            base_samples_per_cell=args.base_samples,
            nv=args.nv,
            dvm_max_steps=args.dvm_max_steps,
            seeds=tuple(args.seeds),
        ),
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
