from __future__ import annotations

from pathlib import Path
import argparse
import json
import numpy as np

from .linear_sidewall_validation import (
    LinearSidewallConfig,
    TABLE3_UY_RATIO_0P1,
    TABLE6_QAV_RATIO_0P1,
)
from .reduced_spherical_solver import solve_reduced_case
from .stage32_near_continuum_observable_audit import (
    STAGE32_KNUDSEN,
    STAGE32_QUADRATURE,
    STAGE32_RATIO,
    observable_metrics,
    relative_profile_change,
    wall_observable_profiles,
)
from .velocity_quadrature_audit import spherical_product


STAGE33_GRIDS = ((24, 24), (30, 30), (36, 36))
STAGE33_OBSERVABLE = "linear_extrapolated_wall"


def validate_stage33_design(
    grids: tuple[tuple[int, int], ...], max_steps: int, tolerance: float
) -> None:
    if len(grids) != 3:
        raise ValueError("Stage 33 requires exactly three grids")
    previous = 0
    for nx, ny in grids:
        if nx != ny or nx < 8:
            raise ValueError("Stage 33 requires square grids of at least 8x8")
        if nx * ny <= previous:
            raise ValueError("grids must increase monotonically")
        previous = nx * ny
    if max_steps <= 0 or tolerance <= 0.0:
        raise ValueError("max_steps and tolerance must be positive")


def linear_h_extrapolation(
    grid_sizes: np.ndarray, values: np.ndarray
) -> tuple[np.ndarray, np.ndarray, float]:
    """Fit value(h)=limit+slope*h and return limit, slope, and R^2.

    Values may be scalar or have trailing profile dimensions. The reported R^2
    is computed over all fitted entries, so it is a compact audit diagnostic,
    not an uncertainty estimate.
    """
    n = np.asarray(grid_sizes, dtype=np.float64)
    y = np.asarray(values, dtype=np.float64)
    if n.ndim != 1 or n.size < 2 or np.any(n <= 0.0):
        raise ValueError("grid_sizes must be a positive one-dimensional sequence")
    if y.shape[0] != n.size:
        raise ValueError("values must have one leading entry per grid")
    h = 1.0 / n
    design = np.stack([np.ones_like(h), h], axis=1)
    flat = y.reshape(n.size, -1)
    coefficients, _, _, _ = np.linalg.lstsq(design, flat, rcond=None)
    fitted = design @ coefficients
    residual = float(np.sum((flat - fitted) ** 2))
    centered = float(np.sum((flat - np.mean(flat, axis=0, keepdims=True)) ** 2))
    r2 = 1.0 if centered <= 1.0e-30 else 1.0 - residual / centered
    limit = coefficients[0].reshape(y.shape[1:])
    slope = coefficients[1].reshape(y.shape[1:])
    return limit, slope, float(r2)


def strictly_decreasing(values: list[float], relative_tolerance: float = 0.01) -> bool:
    return all(
        current <= previous * (1.0 + relative_tolerance) + 1.0e-14
        for previous, current in zip(values, values[1:])
    )


def stage33_decision(
    rows: list[dict[str, object]],
    extrapolated_q_error: float,
    extrapolated_velocity_metrics: dict[str, float],
    finest_profile_change: float,
) -> str:
    all_converged = all(bool(row["converged"]) for row in rows)
    q_monotone = strictly_decreasing(
        [float(row["qav_relative_error"]) for row in rows]
    )
    velocity_monotone = strictly_decreasing(
        [float(row["velocity_metrics"]["relative_rms"]) for row in rows]
    )
    if (
        all_converged
        and q_monotone
        and velocity_monotone
        and extrapolated_q_error <= 0.10
        and extrapolated_velocity_metrics["relative_rms"] <= 0.25
        and extrapolated_velocity_metrics["sign_agreement"] >= 0.9
    ):
        return "spatial_truncation_explains_most_error_advance_second_order_spherical_kn0p1"
    if all_converged and q_monotone and velocity_monotone and finest_profile_change <= 0.15:
        return "asymptotic_limit_retains_model_or_benchmark_discrepancy"
    return "not_yet_asymptotic_extend_spatial_sequence_without_retuning"


def run_stage33(
    output_dir: str | Path,
    *,
    grids: tuple[tuple[int, int], ...] = STAGE33_GRIDS,
    max_steps: int = 12000,
    tolerance: float = 2.0e-5,
) -> dict[str, object]:
    """Run a frozen-physics high-resolution spatial extrapolation at Kn0=0.1."""
    validate_stage33_design(grids, max_steps, tolerance)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    quadrature = spherical_product(16, 12, 24, 5.0, STAGE32_QUADRATURE)
    reference_velocity = TABLE3_UY_RATIO_0P1[STAGE32_KNUDSEN]
    reference_qav = TABLE6_QAV_RATIO_0P1[STAGE32_KNUDSEN]

    rows: list[dict[str, object]] = []
    profiles: list[np.ndarray] = []
    q_values: list[float] = []
    arrays: dict[str, np.ndarray] = {}

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
            minimum_steps=1500,
        )
        result = solve_reduced_case(cfg, quadrature)
        profile = wall_observable_profiles(result, cfg)[STAGE33_OBSERVABLE]
        metrics = observable_metrics(profile, reference_velocity)
        bottom_heat_flux = np.asarray(result["bottom_heat_flux"], dtype=np.float64)
        qav = float(np.mean(bottom_heat_flux))
        residual = np.asarray(result["residual_history"], dtype=np.float64)
        row = {
            "grid": [nx, ny],
            "iterations": int(result["iterations"]),
            "converged": bool(result["converged"]),
            "final_change": float(residual[-1]) if residual.size else float("nan"),
            "predicted_qav": qav,
            "literature_qav": reference_qav,
            "qav_relative_error": abs(qav - reference_qav) / reference_qav,
            "velocity_metrics": metrics,
            "wall_mass_balance_relative_error": float(
                result["wall_mass_balance_relative_error"]
            ),
            "minimum_distribution": float(result["minimum_distribution"]),
            "minimum_temperature": float(np.min(np.asarray(result["T"]))),
            "maximum_temperature": float(np.max(np.asarray(result["T"]))),
            "work_proxy": int(result["iterations"]) * nx * ny * quadrature.point_count,
        }
        rows.append(row)
        profiles.append(profile)
        q_values.append(qav)
        key = f"{nx}x{ny}"
        arrays[f"table_velocity_{key}"] = profile
        arrays[f"bottom_heat_flux_{key}"] = bottom_heat_flux
        arrays[f"residual_history_{key}"] = residual
        arrays[f"T_{key}"] = np.asarray(result["T"])

    sizes = np.asarray([nx for nx, _ in grids], dtype=np.float64)
    q_limit_array, q_slope_array, q_r2 = linear_h_extrapolation(
        sizes, np.asarray(q_values, dtype=np.float64)
    )
    profile_limit, profile_slope, profile_r2 = linear_h_extrapolation(
        sizes, np.stack(profiles, axis=0)
    )
    q_limit = float(q_limit_array)
    q_slope = float(q_slope_array)
    extrapolated_q_error = abs(q_limit - reference_qav) / reference_qav
    extrapolated_velocity_metrics = observable_metrics(profile_limit, reference_velocity)
    finest_profile_change = relative_profile_change(profiles[-2], profiles[-1])
    decision = stage33_decision(
        rows,
        extrapolated_q_error,
        extrapolated_velocity_metrics,
        finest_profile_change,
    )

    summary = {
        "stage": 33,
        "description": (
            "Frozen-physics high-resolution spatial extrapolation for the "
            "Kn0=0.1 spherical-quadrature cavity"
        ),
        "configuration": {
            "kn0": STAGE32_KNUDSEN,
            "cold_hot_ratio": STAGE32_RATIO,
            "grids": [list(grid) for grid in grids],
            "quadrature": STAGE32_QUADRATURE,
            "velocity_point_count": quadrature.point_count,
            "transport": "first-order upwind",
            "collision": "Shakhov",
            "observable": STAGE33_OBSERVABLE,
            "max_steps": max_steps,
            "tolerance": tolerance,
            "physical_parameter_retuning": False,
        },
        "rows": rows,
        "linear_in_h_extrapolation": {
            "predicted_qav_h0": q_limit,
            "qav_slope": q_slope,
            "qav_fit_r2": q_r2,
            "qav_relative_error_h0": extrapolated_q_error,
            "velocity_profile_h0": profile_limit.tolist(),
            "velocity_profile_slope": profile_slope.tolist(),
            "velocity_fit_r2": profile_r2,
            "velocity_metrics_h0": extrapolated_velocity_metrics,
        },
        "finest_profile_change_30x30_to_36x36": finest_profile_change,
        "all_cases_converged": all(bool(row["converged"]) for row in rows),
        "qav_error_monotone": strictly_decreasing(
            [float(row["qav_relative_error"]) for row in rows]
        ),
        "velocity_error_monotone": strictly_decreasing(
            [float(row["velocity_metrics"]["relative_rms"]) for row in rows]
        ),
        "decision": decision,
        "interpretation_guard": (
            "Only spatial resolution is changed. Knudsen number, wall temperatures, "
            "Shakhov model, viscosity law, Prandtl number, spherical quadrature, "
            "transport order, normalization, and wall observable remain fixed. "
            "The h-to-zero fit is a diagnostic extrapolation, not an uncertainty bound."
        ),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    arrays["velocity_profile_h0"] = profile_limit
    arrays["velocity_profile_slope"] = profile_slope
    np.savez_compressed(output_dir / "fields_and_profiles.npz", **arrays)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Stage 33 Kn0=0.1 high-resolution spatial extrapolation"
    )
    parser.add_argument("--output-dir", default="outputs/stage33_spatial_extrapolation")
    parser.add_argument("--max-steps", type=int, default=12000)
    parser.add_argument("--tolerance", type=float, default=2.0e-5)
    args = parser.parse_args()
    print(
        json.dumps(
            run_stage33(
                args.output_dir,
                max_steps=args.max_steps,
                tolerance=args.tolerance,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
