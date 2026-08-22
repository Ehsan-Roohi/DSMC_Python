from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import argparse
import json
import math
import numpy as np

from .dvm_shakhov_corrected import (
    ShakhovReferenceConfig,
    solve_shakhov_reference,
)


@dataclass(frozen=True)
class DVMConvergenceConfig:
    knudsen: float = 0.10
    t_left: float = 340.0
    t_right: float = 260.0
    t_top: float = 300.0
    t_bottom: float = 300.0
    velocity_extent: float = 5.0
    spatial_levels: tuple[int, ...] = (6, 8, 10)
    spatial_reference: int = 12
    spatial_nv: int = 8
    velocity_levels: tuple[int, ...] = (6, 8, 10)
    velocity_reference: int = 12
    velocity_grid: int = 8
    max_steps: int = 1400
    tolerance: float = 3.0e-6


def interpolate_cell_center_field(
    field: np.ndarray,
    new_ny: int,
    new_nx: int,
) -> np.ndarray:
    """Bilinearly interpolate a uniform cell-center field to another grid."""
    field = np.asarray(field, dtype=np.float64)
    if field.ndim != 2:
        raise ValueError("field must be two-dimensional")
    old_ny, old_nx = field.shape
    if min(old_ny, old_nx, new_ny, new_nx) < 2:
        raise ValueError("all grid dimensions must be at least two")
    if (old_ny, old_nx) == (new_ny, new_nx):
        return field.copy()

    old_x = (np.arange(old_nx, dtype=np.float64) + 0.5) / old_nx
    old_y = (np.arange(old_ny, dtype=np.float64) + 0.5) / old_ny
    new_x = (np.arange(new_nx, dtype=np.float64) + 0.5) / new_nx
    new_y = (np.arange(new_ny, dtype=np.float64) + 0.5) / new_ny

    x_interpolated = np.stack(
        [np.interp(new_x, old_x, row) for row in field],
        axis=0,
    )
    output = np.stack(
        [
            np.interp(new_y, old_y, x_interpolated[:, column])
            for column in range(new_nx)
        ],
        axis=1,
    )
    return output


def _rms(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    return float(np.sqrt(np.mean(values**2)))


def field_difference_metrics(
    candidate: dict[str, np.ndarray],
    reference: dict[str, np.ndarray],
    reference_config: ShakhovReferenceConfig,
) -> dict[str, float]:
    """Return normalized RMS errors after aligning candidate to reference grid."""
    required = ("T", "rho", "u", "v", "qx", "qy")
    if not all(name in candidate and name in reference for name in required):
        raise ValueError("candidate and reference must contain T, rho, u, v, qx, qy")
    reference_shape = np.asarray(reference["T"]).shape
    if len(reference_shape) != 2:
        raise ValueError("reference fields must be two-dimensional")
    aligned = {
        name: interpolate_cell_center_field(
            np.asarray(candidate[name], dtype=np.float64),
            reference_shape[0],
            reference_shape[1],
        )
        for name in required
    }
    reference_fields = {
        name: np.asarray(reference[name], dtype=np.float64)
        for name in required
    }

    temperature_scale = max(_rms(reference_fields["T"]), 1.0e-14)
    density_scale = max(_rms(reference_fields["rho"]), 1.0e-14)
    velocity_scale = max(float(reference_config.velocity_scale), 1.0e-14)
    heat_flux_reference = np.hypot(
        reference_fields["qx"],
        reference_fields["qy"],
    )
    heat_flux_scale = max(
        _rms(heat_flux_reference),
        reference_config.velocity_scale**3 * 1.0e-8,
        1.0e-14,
    )

    temperature_error = _rms(aligned["T"] - reference_fields["T"]) / temperature_scale
    density_error = _rms(aligned["rho"] - reference_fields["rho"]) / density_scale
    velocity_error = _rms(
        np.hypot(
            aligned["u"] - reference_fields["u"],
            aligned["v"] - reference_fields["v"],
        )
    ) / velocity_scale
    heat_flux_error = _rms(
        np.hypot(
            aligned["qx"] - reference_fields["qx"],
            aligned["qy"] - reference_fields["qy"],
        )
    ) / heat_flux_scale
    composite = (
        0.35 * temperature_error
        + 0.20 * density_error
        + 0.20 * velocity_error
        + 0.25 * heat_flux_error
    )
    return {
        "temperature_relative_rms": temperature_error,
        "density_relative_rms": density_error,
        "velocity_thermal_rms": velocity_error,
        "heat_flux_relative_rms": heat_flux_error,
        "composite_error": composite,
    }


def _is_monotone_nonincreasing(values: list[float], relative_tolerance: float = 0.02) -> bool:
    for previous, current in zip(values, values[1:]):
        if current > previous * (1.0 + relative_tolerance) + 1.0e-14:
            return False
    return True


def _observed_orders(
    levels: tuple[int, ...],
    errors: list[float],
) -> list[float | None]:
    output: list[float | None] = []
    for first_level, second_level, first_error, second_error in zip(
        levels,
        levels[1:],
        errors,
        errors[1:],
    ):
        if first_error <= 1.0e-15 or second_error <= 1.0e-15:
            output.append(None)
            continue
        denominator = math.log(second_level / first_level)
        output.append(
            float(math.log(first_error / second_error) / denominator)
            if abs(denominator) > 1.0e-15
            else None
        )
    return output


def _run_key(nx: int, ny: int, nv: int) -> str:
    return f"nx{nx}_ny{ny}_nv{nv}"


def _solve_case(
    nx: int,
    ny: int,
    nv: int,
    cfg: DVMConvergenceConfig,
) -> tuple[dict[str, np.ndarray | float | int], ShakhovReferenceConfig]:
    solver_config = ShakhovReferenceConfig(
        nx=nx,
        ny=ny,
        nv=nv,
        velocity_extent=cfg.velocity_extent,
        knudsen=cfg.knudsen,
        t_left=cfg.t_left,
        t_right=cfg.t_right,
        t_top=cfg.t_top,
        t_bottom=cfg.t_bottom,
        max_steps=cfg.max_steps,
        tolerance=cfg.tolerance,
    )
    return solve_shakhov_reference(solver_config), solver_config


def _sequence_summary(
    levels: tuple[int, ...],
    metrics: list[dict[str, float]],
) -> dict[str, object]:
    metric_names = tuple(metrics[0])
    monotonicity = {
        name: _is_monotone_nonincreasing([row[name] for row in metrics])
        for name in metric_names
    }
    observed_orders = {
        name: _observed_orders(levels, [row[name] for row in metrics])
        for name in metric_names
    }
    return {
        "levels": list(levels),
        "metrics": metrics,
        "monotone_nonincreasing_with_2pct_tolerance": monotonicity,
        "observed_orders": observed_orders,
    }


def run_dvm_convergence_study(
    output_dir: str | Path,
    cfg: DVMConvergenceConfig = DVMConvergenceConfig(),
) -> dict[str, object]:
    if len(cfg.spatial_levels) < 2 or len(cfg.velocity_levels) < 2:
        raise ValueError("at least two non-reference levels are required per sequence")
    if max(cfg.spatial_levels) >= cfg.spatial_reference:
        raise ValueError("spatial_reference must exceed all spatial_levels")
    if max(cfg.velocity_levels) >= cfg.velocity_reference:
        raise ValueError("velocity_reference must exceed all velocity_levels")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cases: dict[tuple[int, int, int], tuple[dict[str, object], ShakhovReferenceConfig]] = {}

    requested = {
        *[(level, level, cfg.spatial_nv) for level in cfg.spatial_levels],
        (cfg.spatial_reference, cfg.spatial_reference, cfg.spatial_nv),
        *[(cfg.velocity_grid, cfg.velocity_grid, level) for level in cfg.velocity_levels],
        (cfg.velocity_grid, cfg.velocity_grid, cfg.velocity_reference),
    }
    for nx, ny, nv in sorted(requested):
        result, solver_config = _solve_case(nx, ny, nv, cfg)
        cases[(nx, ny, nv)] = (result, solver_config)

    spatial_reference_result, spatial_reference_config = cases[
        (cfg.spatial_reference, cfg.spatial_reference, cfg.spatial_nv)
    ]
    spatial_metrics = []
    for level in cfg.spatial_levels:
        candidate, _ = cases[(level, level, cfg.spatial_nv)]
        spatial_metrics.append(
            field_difference_metrics(
                candidate,
                spatial_reference_result,
                spatial_reference_config,
            )
        )

    velocity_reference_result, velocity_reference_config = cases[
        (cfg.velocity_grid, cfg.velocity_grid, cfg.velocity_reference)
    ]
    velocity_metrics = []
    for level in cfg.velocity_levels:
        candidate, _ = cases[(cfg.velocity_grid, cfg.velocity_grid, level)]
        velocity_metrics.append(
            field_difference_metrics(
                candidate,
                velocity_reference_result,
                velocity_reference_config,
            )
        )

    run_diagnostics: dict[str, dict[str, float | int]] = {}
    field_artifacts: dict[str, np.ndarray] = {}
    for (nx, ny, nv), (result, _) in sorted(cases.items()):
        key = _run_key(nx, ny, nv)
        residual_history = np.asarray(result["residual_history"], dtype=np.float64)
        run_diagnostics[key] = {
            "nx": nx,
            "ny": ny,
            "nv": nv,
            "iterations": int(result["iterations"]),
            "final_residual": float(residual_history[-1]),
            "mean_temperature": float(np.mean(result["T"])),
            "left_temperature": float(np.mean(result["T"][:, 0])),
            "right_temperature": float(np.mean(result["T"][:, -1])),
            "mean_density": float(np.mean(result["rho"])),
            "maximum_speed": float(np.max(np.hypot(result["u"], result["v"]))),
            "mean_heat_flux_x": float(np.mean(result["qx"])),
        }
        for name in ("T", "rho", "u", "v", "qx", "qy"):
            field_artifacts[f"{name}_{key}"] = np.asarray(result[name])

    spatial_summary = _sequence_summary(cfg.spatial_levels, spatial_metrics)
    spatial_summary["reference"] = {
        "grid": [cfg.spatial_reference, cfg.spatial_reference],
        "nv": cfg.spatial_nv,
    }
    velocity_summary = _sequence_summary(cfg.velocity_levels, velocity_metrics)
    velocity_summary["reference"] = {
        "grid": [cfg.velocity_grid, cfg.velocity_grid],
        "nv": cfg.velocity_reference,
    }

    summary: dict[str, object] = {
        "stage": 22,
        "description": "Internal spatial- and velocity-grid convergence study for corrected Shakhov-DVM",
        "configuration": {
            "knudsen": cfg.knudsen,
            "wall_temperatures": {
                "left": cfg.t_left,
                "right": cfg.t_right,
                "top": cfg.t_top,
                "bottom": cfg.t_bottom,
            },
            "velocity_extent": cfg.velocity_extent,
            "max_steps": cfg.max_steps,
            "tolerance": cfg.tolerance,
        },
        "run_count": len(cases),
        "run_diagnostics": run_diagnostics,
        "spatial_convergence": spatial_summary,
        "velocity_convergence": velocity_summary,
        "interpretation_guard": (
            "This is internal discretization convergence, not external physical validation. "
            "Non-monotone quantities are reported rather than forced to pass."
        ),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    np.savez_compressed(output_dir / "fields.npz", **field_artifacts)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Stage 22 corrected Shakhov-DVM convergence study"
    )
    parser.add_argument("--output-dir", default="outputs/stage22_dvm_convergence")
    parser.add_argument("--kn", type=float, default=0.10)
    parser.add_argument("--max-steps", type=int, default=1400)
    parser.add_argument("--tolerance", type=float, default=3.0e-6)
    parser.add_argument("--spatial-levels", nargs="+", type=int, default=[6, 8, 10])
    parser.add_argument("--spatial-reference", type=int, default=12)
    parser.add_argument("--spatial-nv", type=int, default=8)
    parser.add_argument("--velocity-levels", nargs="+", type=int, default=[6, 8, 10])
    parser.add_argument("--velocity-reference", type=int, default=12)
    parser.add_argument("--velocity-grid", type=int, default=8)
    args = parser.parse_args()
    summary = run_dvm_convergence_study(
        args.output_dir,
        DVMConvergenceConfig(
            knudsen=args.kn,
            max_steps=args.max_steps,
            tolerance=args.tolerance,
            spatial_levels=tuple(args.spatial_levels),
            spatial_reference=args.spatial_reference,
            spatial_nv=args.spatial_nv,
            velocity_levels=tuple(args.velocity_levels),
            velocity_reference=args.velocity_reference,
            velocity_grid=args.velocity_grid,
        ),
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
