from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import argparse
import json
import numpy as np

from .dvm_convergence import (
    _sequence_summary,
    field_difference_metrics,
    interpolate_cell_center_field,
)
from .dvm_shakhov_corrected import (
    ShakhovReferenceConfig,
    solve_shakhov_reference,
)


@dataclass(frozen=True)
class CombinedDVMReferenceConfig:
    knudsen: float = 0.10
    t_left: float = 340.0
    t_right: float = 260.0
    t_top: float = 300.0
    t_bottom: float = 300.0
    velocity_extent: float = 5.0
    spatial_levels: tuple[int, ...] = (10, 12, 14)
    velocity_levels: tuple[int, ...] = (10, 12)
    max_steps: int = 1800
    tolerance: float = 2.0e-6


def _run_key(grid: int, nv: int) -> str:
    return f"nx{grid}_ny{grid}_nv{nv}"


def _midline_rows(field: np.ndarray) -> np.ndarray:
    field = np.asarray(field, dtype=np.float64)
    if field.ndim != 2:
        raise ValueError("field must be two-dimensional")
    ny = field.shape[0]
    if ny % 2:
        return field[ny // 2].copy()
    return 0.5 * (field[ny // 2 - 1] + field[ny // 2])


def _midline_columns(field: np.ndarray) -> np.ndarray:
    field = np.asarray(field, dtype=np.float64)
    if field.ndim != 2:
        raise ValueError("field must be two-dimensional")
    nx = field.shape[1]
    if nx % 2:
        return field[:, nx // 2].copy()
    return 0.5 * (field[:, nx // 2 - 1] + field[:, nx // 2])


def extract_comparison_profiles(fields: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    required = ("T", "rho", "u", "v", "qx", "qy")
    if not all(name in fields for name in required):
        raise ValueError("fields must contain T, rho, u, v, qx, qy")
    shape = np.asarray(fields["T"]).shape
    if len(shape) != 2 or min(shape) < 2:
        raise ValueError("all fields must be two-dimensional with at least two cells")
    for name in required:
        if np.asarray(fields[name]).shape != shape:
            raise ValueError("all fields must have the same shape")
    ny, nx = shape
    return {
        "x": (np.arange(nx, dtype=np.float64) + 0.5) / nx,
        "y": (np.arange(ny, dtype=np.float64) + 0.5) / ny,
        "T_horizontal_centerline": _midline_rows(fields["T"]),
        "T_vertical_centerline": _midline_columns(fields["T"]),
        "rho_horizontal_centerline": _midline_rows(fields["rho"]),
        "rho_vertical_centerline": _midline_columns(fields["rho"]),
        "u_horizontal_centerline": _midline_rows(fields["u"]),
        "v_vertical_centerline": _midline_columns(fields["v"]),
        "normal_heat_flux_left": np.asarray(fields["qx"], dtype=np.float64)[:, 0].copy(),
        "normal_heat_flux_right": -np.asarray(fields["qx"], dtype=np.float64)[:, -1].copy(),
        "normal_heat_flux_bottom": np.asarray(fields["qy"], dtype=np.float64)[0].copy(),
        "normal_heat_flux_top": -np.asarray(fields["qy"], dtype=np.float64)[-1].copy(),
    }


def _relative_rms(candidate: np.ndarray, reference: np.ndarray, floor: float = 1.0e-14) -> float:
    candidate = np.asarray(candidate, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    numerator = float(np.sqrt(np.mean((candidate - reference) ** 2)))
    denominator = max(float(np.sqrt(np.mean(reference**2))), floor)
    return numerator / denominator


def profile_difference_metrics(
    candidate: dict[str, np.ndarray],
    reference: dict[str, np.ndarray],
) -> dict[str, float]:
    reference_shape = np.asarray(reference["T"]).shape
    aligned = {
        name: interpolate_cell_center_field(
            np.asarray(candidate[name], dtype=np.float64),
            reference_shape[0],
            reference_shape[1],
        )
        for name in ("T", "rho", "u", "v", "qx", "qy")
    }
    candidate_profiles = extract_comparison_profiles(aligned)
    reference_profiles = extract_comparison_profiles(reference)
    thermal_speed_scale = max(
        float(np.sqrt(np.mean(reference["u"] ** 2 + reference["v"] ** 2))),
        1.0,
    )
    velocity_error = float(
        np.sqrt(
            np.mean(
                np.concatenate(
                    [
                        candidate_profiles["u_horizontal_centerline"]
                        - reference_profiles["u_horizontal_centerline"],
                        candidate_profiles["v_vertical_centerline"]
                        - reference_profiles["v_vertical_centerline"],
                    ]
                )
                ** 2
            )
        )
        / thermal_speed_scale
    )
    wall_candidate = np.concatenate(
        [
            candidate_profiles["normal_heat_flux_left"],
            candidate_profiles["normal_heat_flux_right"],
            candidate_profiles["normal_heat_flux_bottom"],
            candidate_profiles["normal_heat_flux_top"],
        ]
    )
    wall_reference = np.concatenate(
        [
            reference_profiles["normal_heat_flux_left"],
            reference_profiles["normal_heat_flux_right"],
            reference_profiles["normal_heat_flux_bottom"],
            reference_profiles["normal_heat_flux_top"],
        ]
    )
    return {
        "temperature_horizontal_centerline_relative_rms": _relative_rms(
            candidate_profiles["T_horizontal_centerline"],
            reference_profiles["T_horizontal_centerline"],
        ),
        "temperature_vertical_centerline_relative_rms": _relative_rms(
            candidate_profiles["T_vertical_centerline"],
            reference_profiles["T_vertical_centerline"],
        ),
        "density_centerline_relative_rms": 0.5
        * (
            _relative_rms(
                candidate_profiles["rho_horizontal_centerline"],
                reference_profiles["rho_horizontal_centerline"],
            )
            + _relative_rms(
                candidate_profiles["rho_vertical_centerline"],
                reference_profiles["rho_vertical_centerline"],
            )
        ),
        "velocity_centerline_reference_speed_rms": velocity_error,
        "wall_normal_heat_flux_relative_rms": _relative_rms(
            wall_candidate,
            wall_reference,
        ),
    }


def _solve_case(
    grid: int,
    nv: int,
    cfg: CombinedDVMReferenceConfig,
) -> tuple[dict[str, object], ShakhovReferenceConfig]:
    solver_config = ShakhovReferenceConfig(
        nx=grid,
        ny=grid,
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


def run_combined_dvm_reference_study(
    output_dir: str | Path,
    cfg: CombinedDVMReferenceConfig = CombinedDVMReferenceConfig(),
) -> dict[str, object]:
    if len(cfg.spatial_levels) < 3:
        raise ValueError("at least three spatial levels are required")
    if len(cfg.velocity_levels) < 2:
        raise ValueError("at least two velocity levels are required")
    if tuple(sorted(set(cfg.spatial_levels))) != cfg.spatial_levels:
        raise ValueError("spatial levels must be unique and increasing")
    if tuple(sorted(set(cfg.velocity_levels))) != cfg.velocity_levels:
        raise ValueError("velocity levels must be unique and increasing")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cases: dict[tuple[int, int], tuple[dict[str, object], ShakhovReferenceConfig]] = {}
    for grid in cfg.spatial_levels:
        for nv in cfg.velocity_levels:
            cases[(grid, nv)] = _solve_case(grid, nv, cfg)

    finest_grid = cfg.spatial_levels[-1]
    finest_nv = cfg.velocity_levels[-1]
    finest_result, finest_config = cases[(finest_grid, finest_nv)]

    run_diagnostics: dict[str, dict[str, float | int]] = {}
    canonical_errors: dict[str, dict[str, float]] = {}
    profile_errors: dict[str, dict[str, float]] = {}
    field_artifacts: dict[str, np.ndarray] = {}
    profile_artifacts: dict[str, np.ndarray] = {}

    for (grid, nv), (result, _) in sorted(cases.items()):
        key = _run_key(grid, nv)
        residual_history = np.asarray(result["residual_history"], dtype=np.float64)
        run_diagnostics[key] = {
            "grid": grid,
            "nv": nv,
            "iterations": int(result["iterations"]),
            "final_residual": float(residual_history[-1]),
            "mean_temperature": float(np.mean(result["T"])),
            "left_temperature": float(np.mean(result["T"][:, 0])),
            "right_temperature": float(np.mean(result["T"][:, -1])),
            "maximum_speed": float(np.max(np.hypot(result["u"], result["v"]))),
            "mean_heat_flux_x": float(np.mean(result["qx"])),
        }
        canonical_errors[key] = field_difference_metrics(
            result,
            finest_result,
            finest_config,
        )
        profile_errors[key] = profile_difference_metrics(result, finest_result)
        for name in ("T", "rho", "u", "v", "qx", "qy"):
            field_artifacts[f"{name}_{key}"] = np.asarray(result[name])
        for name, value in extract_comparison_profiles(result).items():
            profile_artifacts[f"{name}_{key}"] = np.asarray(value)

    spatial_sequences: dict[str, object] = {}
    for nv in cfg.velocity_levels:
        reference_result, reference_config = cases[(finest_grid, nv)]
        levels = cfg.spatial_levels[:-1]
        metrics = [
            field_difference_metrics(cases[(grid, nv)][0], reference_result, reference_config)
            for grid in levels
        ]
        sequence = _sequence_summary(levels, metrics)
        sequence["reference"] = {"grid": finest_grid, "nv": nv}
        spatial_sequences[f"Nv_{nv}"] = sequence

    velocity_increments: dict[str, dict[str, object]] = {}
    coarse_nv = cfg.velocity_levels[-2]
    for grid in cfg.spatial_levels:
        reference_result, reference_config = cases[(grid, finest_nv)]
        candidate_result, _ = cases[(grid, coarse_nv)]
        velocity_increments[f"grid_{grid}"] = {
            "candidate_nv": coarse_nv,
            "reference_nv": finest_nv,
            "field_metrics": field_difference_metrics(
                candidate_result,
                reference_result,
                reference_config,
            ),
            "profile_metrics": profile_difference_metrics(
                candidate_result,
                reference_result,
            ),
        }

    ranked_candidates = sorted(
        (
            {
                "case": key,
                "composite_error": values["composite_error"],
                "heat_flux_relative_rms": values["heat_flux_relative_rms"],
                "wall_normal_heat_flux_relative_rms": profile_errors[key][
                    "wall_normal_heat_flux_relative_rms"
                ],
            }
            for key, values in canonical_errors.items()
            if key != _run_key(finest_grid, finest_nv)
        ),
        key=lambda row: row["composite_error"],
    )

    summary: dict[str, object] = {
        "stage": 23,
        "description": (
            "Combined high-resolution spatial/velocity Shakhov-DVM reference matrix "
            "with centerline and wall-normal heat-flux outputs"
        ),
        "configuration": {
            "knudsen": cfg.knudsen,
            "wall_temperatures": {
                "left": cfg.t_left,
                "right": cfg.t_right,
                "top": cfg.t_top,
                "bottom": cfg.t_bottom,
            },
            "velocity_extent": cfg.velocity_extent,
            "spatial_levels": list(cfg.spatial_levels),
            "velocity_levels": list(cfg.velocity_levels),
            "max_steps": cfg.max_steps,
            "tolerance": cfg.tolerance,
        },
        "run_count": len(cases),
        "canonical_reference": {
            "grid": [finest_grid, finest_grid],
            "nv": finest_nv,
            "status": "finest_internal_reference_not_external_validation",
        },
        "run_diagnostics": run_diagnostics,
        "errors_relative_to_canonical": canonical_errors,
        "profile_errors_relative_to_canonical": profile_errors,
        "spatial_sequences": spatial_sequences,
        "velocity_increments": velocity_increments,
        "candidate_rank_by_composite_error": ranked_candidates,
        "external_validation_contract": {
            "centerlines": [
                "T_horizontal_centerline",
                "T_vertical_centerline",
                "rho_horizontal_centerline",
                "rho_vertical_centerline",
                "u_horizontal_centerline",
                "v_vertical_centerline",
            ],
            "wall_profiles": [
                "normal_heat_flux_left",
                "normal_heat_flux_right",
                "normal_heat_flux_bottom",
                "normal_heat_flux_top",
            ],
            "artifact": "profiles.npz",
        },
        "interpretation_guard": (
            "The finest 14x14, Nv=12 case is an internal numerical reference. "
            "It is not a validated physical benchmark until compared with an independent code, "
            "published data, or experiment."
        ),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    np.savez_compressed(output_dir / "fields.npz", **field_artifacts)
    np.savez_compressed(output_dir / "profiles.npz", **profile_artifacts)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Stage 23 combined high-resolution Shakhov-DVM study"
    )
    parser.add_argument("--output-dir", default="outputs/stage23_combined_dvm")
    parser.add_argument("--kn", type=float, default=0.10)
    parser.add_argument("--max-steps", type=int, default=1800)
    parser.add_argument("--tolerance", type=float, default=2.0e-6)
    parser.add_argument("--spatial-levels", nargs="+", type=int, default=[10, 12, 14])
    parser.add_argument("--velocity-levels", nargs="+", type=int, default=[10, 12])
    args = parser.parse_args()
    summary = run_combined_dvm_reference_study(
        args.output_dir,
        CombinedDVMReferenceConfig(
            knudsen=args.kn,
            max_steps=args.max_steps,
            tolerance=args.tolerance,
            spatial_levels=tuple(args.spatial_levels),
            velocity_levels=tuple(args.velocity_levels),
        ),
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
