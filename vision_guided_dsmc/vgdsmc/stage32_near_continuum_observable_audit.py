from __future__ import annotations

from pathlib import Path
import argparse
import json
import math
import numpy as np

from .linear_sidewall_validation import (
    LinearSidewallConfig,
    TABLE3_UY_RATIO_0P1,
    TABLE3_Y,
    TABLE6_QAV_RATIO_0P1,
    _relative_rms,
)
from .reduced_spherical_solver import solve_reduced_case
from .velocity_quadrature_audit import spherical_product


STAGE32_GRIDS = ((12, 12), (18, 18), (24, 24))
STAGE32_OBSERVABLES = (
    "boundary_mixture",
    "adjacent_cell_center",
    "linear_extrapolated_wall",
)
STAGE32_KNUDSEN = 0.1
STAGE32_RATIO = 0.1
STAGE32_QUADRATURE = "spherical_matched_r16_mu12_phi24"


def validate_stage32_design(
    grids: tuple[tuple[int, int], ...],
    max_steps: int,
    tolerance: float,
) -> None:
    if len(grids) < 2:
        raise ValueError("at least two spatial grids are required")
    previous_points = 0
    for nx, ny in grids:
        if nx < 3 or ny < 3:
            raise ValueError("all spatial dimensions must be at least three")
        points = nx * ny
        if points <= previous_points:
            raise ValueError("spatial grids must increase monotonically")
        previous_points = points
    if max_steps <= 0:
        raise ValueError("max_steps must be positive")
    if tolerance <= 0.0:
        raise ValueError("tolerance must be positive")


def wall_observable_profiles(
    result: dict[str, object],
    cfg: LinearSidewallConfig,
) -> dict[str, np.ndarray]:
    """Return three fixed definitions of the left-wall tangential gas velocity.

    The boundary-mixture value is the full incoming/outgoing wall distribution.
    The adjacent-cell value is the first cell-center macroscopic velocity. The
    extrapolated value is the linear one-sided limit from the first two cells.
    All values use the paper velocity scale already returned by the solver.
    """
    boundary = np.asarray(result["left_wall_velocity"], dtype=np.float64)
    vertical_velocity = np.asarray(result["v"], dtype=np.float64)
    if vertical_velocity.shape != (cfg.ny, cfg.nx):
        raise ValueError("result velocity field does not match the configuration")
    if boundary.shape != (cfg.ny,):
        raise ValueError("boundary velocity profile does not match ny")
    adjacent = vertical_velocity[:, 0]
    extrapolated = 1.5 * vertical_velocity[:, 0] - 0.5 * vertical_velocity[:, 1]
    y_centers = (np.arange(cfg.ny, dtype=np.float64) + 0.5) / cfg.ny
    return {
        "boundary_mixture": np.interp(TABLE3_Y, y_centers, boundary),
        "adjacent_cell_center": np.interp(TABLE3_Y, y_centers, adjacent),
        "linear_extrapolated_wall": np.interp(TABLE3_Y, y_centers, extrapolated),
    }


def observable_metrics(profile: np.ndarray, reference: np.ndarray) -> dict[str, float]:
    profile = np.asarray(profile, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    if profile.shape != reference.shape:
        raise ValueError("profile and reference must have matching shapes")
    return {
        "relative_rms": _relative_rms(profile, reference),
        "sign_agreement": float(np.mean(np.sign(profile) == np.sign(reference))),
        "relative_l1": float(
            np.mean(np.abs(profile - reference))
            / max(float(np.mean(np.abs(reference))), 1.0e-14)
        ),
    }


def relative_profile_change(coarse: np.ndarray, fine: np.ndarray) -> float:
    coarse = np.asarray(coarse, dtype=np.float64)
    fine = np.asarray(fine, dtype=np.float64)
    if coarse.shape != fine.shape:
        raise ValueError("profiles must have matching shapes")
    return float(
        np.linalg.norm(fine - coarse)
        / max(float(np.linalg.norm(fine)), 1.0e-14)
    )


def stage32_decision(
    rows: list[dict[str, object]],
    profile_changes: dict[str, float],
) -> str:
    finest = rows[-1]
    observables = finest["observables"]
    assert isinstance(observables, dict)
    boundary = observables["boundary_mixture"]
    candidates = [
        (name, observables[name])
        for name in ("adjacent_cell_center", "linear_extrapolated_wall")
    ]
    for name, metrics in candidates:
        if (
            float(metrics["relative_rms"])
            <= 0.85 * float(boundary["relative_rms"])
            and float(metrics["sign_agreement"])
            >= float(boundary["sign_agreement"])
            and profile_changes[name] <= 0.10
        ):
            return "adopt_interior_or_extrapolated_wall_observable_and_cross_validate"

    coarsest = rows[0]
    coarse_observables = coarsest["observables"]
    assert isinstance(coarse_observables, dict)
    best_coarse = min(
        float(coarse_observables[name]["relative_rms"])
        for name in STAGE32_OBSERVABLES
    )
    best_fine = min(
        float(observables[name]["relative_rms"])
        for name in STAGE32_OBSERVABLES
    )
    q_reduction = float(finest["qav_relative_error"]) <= 0.85 * float(
        coarsest["qav_relative_error"]
    )
    velocity_reduction = best_fine <= 0.85 * best_coarse
    if q_reduction and velocity_reduction:
        return "spatial_refinement_materially_reduces_near_continuum_error"
    return "observable_and_spatial_resolution_do_not_explain_kn0p1_error_audit_model_limit"


def run_stage32(
    output_dir: str | Path,
    *,
    grids: tuple[tuple[int, int], ...] = STAGE32_GRIDS,
    max_steps: int = 9000,
    tolerance: float = 3.0e-5,
) -> dict[str, object]:
    """Audit Kn0=0.1 wall observables and spatial convergence with frozen physics."""
    validate_stage32_design(grids, max_steps, tolerance)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    quadrature = spherical_product(
        16, 12, 24, 5.0, STAGE32_QUADRATURE
    )
    reference_velocity = TABLE3_UY_RATIO_0P1[STAGE32_KNUDSEN]
    reference_qav = TABLE6_QAV_RATIO_0P1[STAGE32_KNUDSEN]
    rows: list[dict[str, object]] = []
    arrays: dict[str, np.ndarray] = {}
    profiles_by_grid: dict[tuple[int, int], dict[str, np.ndarray]] = {}

    for nx, ny in grids:
        cfg = LinearSidewallConfig(
            nx=nx,
            ny=ny,
            nv=19,
            velocity_extent=5.0,
            kn0=STAGE32_KNUDSEN,
            cold_hot_ratio=STAGE32_RATIO,
            max_steps=max_steps,
            tolerance=tolerance,
            check_interval=100,
            minimum_steps=1200,
        )
        result = solve_reduced_case(cfg, quadrature)
        profiles = wall_observable_profiles(result, cfg)
        profiles_by_grid[(nx, ny)] = profiles
        bottom_heat_flux = np.asarray(result["bottom_heat_flux"], dtype=np.float64)
        predicted_qav = float(np.mean(bottom_heat_flux))
        residual = np.asarray(result["residual_history"], dtype=np.float64)
        row = {
            "grid": [nx, ny],
            "point_count": quadrature.point_count,
            "iterations": int(result["iterations"]),
            "converged": bool(result["converged"]),
            "final_change": float(residual[-1]) if residual.size else float("nan"),
            "predicted_qav": predicted_qav,
            "literature_qav": reference_qav,
            "qav_relative_error": abs(predicted_qav - reference_qav) / reference_qav,
            "wall_mass_balance_relative_error": float(
                result["wall_mass_balance_relative_error"]
            ),
            "minimum_distribution": float(result["minimum_distribution"]),
            "minimum_temperature": float(np.min(np.asarray(result["T"]))),
            "maximum_temperature": float(np.max(np.asarray(result["T"]))),
            "work_proxy": int(result["iterations"])
            * nx
            * ny
            * quadrature.point_count,
            "observables": {
                name: observable_metrics(profiles[name], reference_velocity)
                for name in STAGE32_OBSERVABLES
            },
        }
        rows.append(row)
        key = f"{nx}x{ny}"
        for name in ("T", "rho", "u", "v", "qx", "qy", "bottom_heat_flux", "residual_history"):
            arrays[f"{name}_{key}"] = np.asarray(result[name])
        for name, profile in profiles.items():
            arrays[f"table_velocity_{name}_{key}"] = profile

    penultimate = grids[-2]
    finest = grids[-1]
    profile_changes = {
        name: relative_profile_change(
            profiles_by_grid[penultimate][name], profiles_by_grid[finest][name]
        )
        for name in STAGE32_OBSERVABLES
    }
    decision = stage32_decision(rows, profile_changes)
    finest_observables = rows[-1]["observables"]
    assert isinstance(finest_observables, dict)
    best_name = min(
        STAGE32_OBSERVABLES,
        key=lambda name: float(finest_observables[name]["relative_rms"]),
    )
    summary = {
        "stage": 32,
        "description": (
            "Fixed-physics near-continuum audit of wall-velocity observable "
            "definitions and spatial convergence"
        ),
        "configuration": {
            "kn0": STAGE32_KNUDSEN,
            "cold_hot_ratio": STAGE32_RATIO,
            "grids": [list(grid) for grid in grids],
            "quadrature": STAGE32_QUADRATURE,
            "velocity_point_count": quadrature.point_count,
            "transport": "first-order upwind",
            "collision": "Shakhov",
            "max_steps": max_steps,
            "tolerance": tolerance,
            "physical_parameter_retuning": False,
            "observable_definitions": list(STAGE32_OBSERVABLES),
        },
        "rows": rows,
        "finest_grid_profile_changes_from_penultimate": profile_changes,
        "best_finest_grid_observable": best_name,
        "all_cases_converged": all(bool(row["converged"]) for row in rows),
        "decision": decision,
        "interpretation_guard": (
            "Stage 32 changes only spatial resolution and the post-processing "
            "definition of the Table-3 wall velocity. Knudsen number, collision "
            "model, wall temperatures, viscosity law, Prandtl number, quadrature, "
            "and normalization remain fixed. Negative and nonconverged results are retained."
        ),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    np.savez_compressed(output_dir / "fields_and_profiles.npz", **arrays)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Stage 32 Kn0=0.1 observable and spatial-convergence audit"
    )
    parser.add_argument("--output-dir", default="outputs/stage32_observable_audit")
    parser.add_argument("--max-steps", type=int, default=9000)
    parser.add_argument("--tolerance", type=float, default=3.0e-5)
    args = parser.parse_args()
    print(
        json.dumps(
            run_stage32(
                args.output_dir,
                max_steps=args.max_steps,
                tolerance=args.tolerance,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
