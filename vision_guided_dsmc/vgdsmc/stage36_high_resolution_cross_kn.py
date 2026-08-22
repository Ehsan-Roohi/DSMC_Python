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
from .stage32_near_continuum_observable_audit import (
    observable_metrics,
    relative_profile_change,
    wall_observable_profiles,
)
from .stage34_velocity_scale_consistency import solve_reduced_case_with_mapping
from .velocity_quadrature_audit import spherical_product


STAGE36_RATIO = 0.1
STAGE36_QUADRATURE = "spherical_matched_r16_mu12_phi24"
STAGE36_OBSERVABLE = "linear_extrapolated_wall"
STAGE36_CASES = (
    ("kn0p1_24x24", 0.1, (24, 24)),
    ("kn0p1_36x36", 0.1, (36, 36)),
    ("kn1p0_24x24", 1.0, (24, 24)),
    ("kn10p0_24x24", 10.0, (24, 24)),
)

# Exact Stage 35 paper-consistent 12x12 spherical baselines. These values are
# retained verbatim and are not recomputed or replaced by the high-resolution
# runs, so any degradation remains visible.
STAGE35_BASELINES = {
    0.1: {
        "grid": [12, 12],
        "iterations": 1600,
        "converged": True,
        "final_change": 1.885076686490572e-05,
        "predicted_qav": 0.07954802746383147,
        "literature_qav": 0.072,
        "qav_relative_error": 0.10483371477543713,
        "best_wall_observable": "linear_extrapolated_wall",
        "wall_velocity_relative_rms": 1.9402797092765636,
        "wall_velocity_sign_agreement": 0.6,
        "wall_mass_balance_relative_error": 2.396774623369399e-16,
    },
    1.0: {
        "grid": [12, 12],
        "iterations": 1500,
        "converged": True,
        "final_change": 1.876206721151963e-05,
        "predicted_qav": 0.1532141835930005,
        "literature_qav": 0.148,
        "qav_relative_error": 0.03523097022297649,
        "best_wall_observable": "linear_extrapolated_wall",
        "wall_velocity_relative_rms": 0.30127320602229624,
        "wall_velocity_sign_agreement": 1.0,
        "wall_mass_balance_relative_error": 1.600569125474438e-16,
    },
    10.0: {
        "grid": [12, 12],
        "iterations": 4100,
        "converged": True,
        "final_change": 2.8348192986360488e-05,
        "predicted_qav": 0.18323830533804605,
        "literature_qav": 0.178,
        "qav_relative_error": 0.02942868167441608,
        "best_wall_observable": "linear_extrapolated_wall",
        "wall_velocity_relative_rms": 0.26087668938553804,
        "wall_velocity_sign_agreement": 1.0,
        "wall_mass_balance_relative_error": 1.4029124140826564e-16,
    },
}


def validate_stage36_design(
    cases: tuple[tuple[str, float, tuple[int, int]], ...],
    max_steps: int,
    tolerance: float,
) -> None:
    if cases != STAGE36_CASES:
        raise ValueError("Stage 36 case matrix is fixed and may not be retuned")
    names = [name for name, _, _ in cases]
    if len(names) != len(set(names)):
        raise ValueError("Stage 36 case names must be unique")
    for _, kn0, (nx, ny) in cases:
        if kn0 not in STAGE35_BASELINES:
            raise ValueError("Stage 36 supports only Kn0=0.1, 1, and 10")
        if nx != ny or nx < 24:
            raise ValueError("Stage 36 requires square grids of at least 24x24")
    if max_steps <= 0 or tolerance <= 0.0:
        raise ValueError("max_steps and tolerance must be positive")


def case_key(kn0: float, grid: tuple[int, int]) -> str:
    token = str(float(kn0)).replace(".", "p")
    return f"kn{token}_{grid[0]}x{grid[1]}"


def stage36_decision(
    rows: list[dict[str, object]],
    low_kn_profile_change_24_to_36: float,
) -> str:
    if not all(bool(row["converged"]) for row in rows):
        return "high_resolution_nonconvergence_requires_numerical_stability_audit_stage37"

    by_case = {str(row["case"]): row for row in rows}
    low = by_case["kn0p1_36x36"]
    middle = by_case["kn1p0_24x24"]
    high = by_case["kn10p0_24x24"]

    def supported(row: dict[str, object], q_limit: float, v_limit: float) -> bool:
        return (
            float(row["qav_relative_error"]) <= q_limit
            and float(row["velocity_metrics"]["relative_rms"]) <= v_limit
            and float(row["velocity_metrics"]["sign_agreement"]) >= 0.9
        )

    low_supported = supported(low, 0.10, 0.50) and low_kn_profile_change_24_to_36 <= 0.20
    middle_supported = supported(middle, 0.06, 0.35)
    high_supported = supported(high, 0.06, 0.35)

    if low_supported and middle_supported and high_supported:
        return "cross_kn_quantitative_support_advance_independent_reference_stage37"
    if middle_supported and high_supported and not low_supported:
        return "high_kn_supported_low_kn_requires_transport_or_collision_audit_stage37"
    return "high_resolution_retains_cross_kn_discrepancy_stage37_model_observable_audit"


def run_stage36(
    output_dir: str | Path,
    *,
    cases: tuple[tuple[str, float, tuple[int, int]], ...] = STAGE36_CASES,
    max_steps: int = 14000,
    tolerance: float = 2.0e-5,
) -> dict[str, object]:
    """Run paper-consistent high-resolution spherical cases without retuning."""
    validate_stage36_design(cases, max_steps, tolerance)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    quadrature = spherical_product(16, 12, 24, 5.0, STAGE36_QUADRATURE)

    rows: list[dict[str, object]] = []
    profiles: dict[str, np.ndarray] = {}
    arrays: dict[str, np.ndarray] = {}

    for name, kn0, grid in cases:
        nx, ny = grid
        cfg = LinearSidewallConfig(
            nx=nx,
            ny=ny,
            nv=19,
            velocity_extent=5.0,
            kn0=kn0,
            cold_hot_ratio=STAGE36_RATIO,
            max_steps=max_steps,
            tolerance=tolerance,
            check_interval=100,
            minimum_steps=1500,
        )
        result = solve_reduced_case_with_mapping(
            cfg, quadrature, mapping="paper_consistent_c0"
        )
        profile = wall_observable_profiles(result, cfg)[STAGE36_OBSERVABLE]
        reference_velocity = TABLE3_UY_RATIO_0P1[kn0]
        velocity_metrics = observable_metrics(profile, reference_velocity)
        bottom_flux = np.asarray(result["bottom_heat_flux"], dtype=np.float64)
        residual = np.asarray(result["residual_history"], dtype=np.float64)
        qav = float(np.mean(bottom_flux))
        reference_qav = TABLE6_QAV_RATIO_0P1[kn0]
        q_error = abs(qav - reference_qav) / reference_qav
        baseline = STAGE35_BASELINES[kn0]
        row = {
            "case": name,
            "kn0": kn0,
            "grid": [nx, ny],
            "iterations": int(result["iterations"]),
            "converged": bool(result["converged"]),
            "final_change": float(residual[-1]) if residual.size else float("nan"),
            "predicted_qav": qav,
            "literature_qav": reference_qav,
            "qav_relative_error": q_error,
            "velocity_observable": STAGE36_OBSERVABLE,
            "velocity_metrics": velocity_metrics,
            "wall_mass_balance_relative_error": float(
                result["wall_mass_balance_relative_error"]
            ),
            "minimum_distribution": float(result["minimum_distribution"]),
            "minimum_temperature": float(np.min(np.asarray(result["T"]))),
            "maximum_temperature": float(np.max(np.asarray(result["T"]))),
            "work_proxy": int(result["iterations"]) * nx * ny * quadrature.point_count,
            "comparison_to_stage35_12x12": {
                "qav_error_ratio": q_error / float(baseline["qav_relative_error"]),
                "velocity_error_ratio": float(velocity_metrics["relative_rms"])
                / float(baseline["wall_velocity_relative_rms"]),
                "sign_agreement_change": float(velocity_metrics["sign_agreement"])
                - float(baseline["wall_velocity_sign_agreement"]),
            },
        }
        rows.append(row)
        profiles[name] = profile
        key = case_key(kn0, grid)
        arrays[f"table_velocity_{key}"] = profile
        arrays[f"bottom_heat_flux_{key}"] = bottom_flux
        arrays[f"residual_history_{key}"] = residual
        arrays[f"T_{key}"] = np.asarray(result["T"])
        arrays[f"rho_{key}"] = np.asarray(result["rho"])

    low_kn_profile_change = relative_profile_change(
        profiles["kn0p1_24x24"], profiles["kn0p1_36x36"]
    )
    decision = stage36_decision(rows, low_kn_profile_change)
    summary = {
        "stage": 36,
        "description": (
            "Paper-consistent high-resolution cross-Knudsen spherical-quadrature "
            "validation following the exact Stage 35 covariance pass"
        ),
        "configuration": {
            "cases": [
                {"name": name, "kn0": kn0, "grid": list(grid)}
                for name, kn0, grid in cases
            ],
            "cold_hot_ratio": STAGE36_RATIO,
            "quadrature": STAGE36_QUADRATURE,
            "velocity_point_count": quadrature.point_count,
            "relaxation_mapping": "sqrt(2)*Kn0/sqrt(pi) in c0 coordinates",
            "transport": "first-order upwind",
            "collision": "Shakhov",
            "wall_observable": STAGE36_OBSERVABLE,
            "max_steps": max_steps,
            "tolerance": tolerance,
            "physical_parameter_retuning": False,
        },
        "stage35_12x12_baselines": STAGE35_BASELINES,
        "rows": rows,
        "low_kn_profile_change_24x24_to_36x36": low_kn_profile_change,
        "all_cases_converged": all(bool(row["converged"]) for row in rows),
        "decision": decision,
        "interpretation_guard": (
            "Only spatial resolution and the predeclared case matrix change. The "
            "paper-consistent relaxation mapping, Knudsen numbers, wall temperatures, "
            "Shakhov model, viscosity law, Prandtl number, spherical quadrature, "
            "transport order, normalization, and wall observable remain fixed. "
            "Negative changes relative to the exact Stage 35 baselines are retained."
        ),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    np.savez_compressed(output_dir / "fields_and_profiles.npz", **arrays)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Stage 36 corrected-mapping high-resolution cross-Kn audit"
    )
    parser.add_argument(
        "--output-dir", default="outputs/stage36_high_resolution_cross_kn"
    )
    parser.add_argument("--max-steps", type=int, default=14000)
    parser.add_argument("--tolerance", type=float, default=2.0e-5)
    args = parser.parse_args()
    print(
        json.dumps(
            run_stage36(
                args.output_dir,
                max_steps=args.max_steps,
                tolerance=args.tolerance,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
