from __future__ import annotations

from pathlib import Path
import argparse
import json
import math
import numpy as np

from .linear_sidewall_validation import LinearSidewallConfig
from .stage41_projected_polar_operator_audit import (
    STAGE41_CORRECTION_FLOOR,
    STAGE41_FINE_RULE,
    STAGE41_PRANDTL,
    mapped_polar_quadrature,
)
from .stage42_projected_polar_heated_cavity_pilot import (
    STAGE42_CHECK_INTERVAL,
    STAGE42_COLLISION,
    STAGE42_KNUDSEN,
    STAGE42_MAX_ITERATIONS,
    STAGE42_MINIMUM_ITERATIONS,
    STAGE42_RATIO,
    STAGE42_RELAXATION_MAPPING,
    STAGE42_SOURCE_RELAXATION,
    STAGE42_TOLERANCE,
    STAGE42_TRANSPORT,
    solve_stage42_pilot,
)


STAGE42_COMPLETED_ENDPOINT = {
    "workflow_run_id": 30731231176,
    "workflow_job_id": 91451917655,
    "workflow_conclusion": "success",
    "regression_tests_passed": 44,
    "regression_tests_failed": 0,
    "artifact_id": 8828768222,
    "artifact_size_bytes": 29149,
    "artifact_sha256": "4d9181f5e01407ebb15da0b3a52539d3e536a62e6b38271b905cc5d1be646552",
    "head_sha": "eb6447ef81c5dd925f163caf314f4e2a75d77372",
    "grid": [8, 8],
    "polar_rule": [32, 96],
    "iterations": 500,
    "converged": True,
    "final_change": 6.200682328705298e-12,
    "predicted_qav": 0.08915883055151556,
    "literature_qav": 0.072,
    "qav_relative_error": 0.2383170909932718,
    "velocity_relative_rms": 4.342029169274625,
    "velocity_relative_l1": 4.959838161846327,
    "velocity_sign_agreement": 0.2,
    "wall_mass_balance_relative_error": 1.446888721362172e-16,
    "maximum_phi_clipped_weight_fraction": 0.0017852355264458837,
    "maximum_psi_clipped_weight_fraction": 0.004012338285911871,
    "decision": "projected_polar_pilot_converged_stage43_resolution_sequence",
}

STAGE43_GRIDS = ((8, 8), (12, 12), (16, 16))
STAGE43_RULE = STAGE41_FINE_RULE
STAGE43_KNUDSEN = STAGE42_KNUDSEN
STAGE43_RATIO = STAGE42_RATIO
STAGE43_MAX_ITERATIONS = STAGE42_MAX_ITERATIONS
STAGE43_TOLERANCE = STAGE42_TOLERANCE
STAGE43_SOURCE_RELAXATION = STAGE42_SOURCE_RELAXATION
STAGE43_REPRODUCTION_ATOL = 1.0e-12


def validate_stage43_design(
    grids: tuple[tuple[int, int], ...],
    rule: tuple[int, int],
    kn0: float,
    cold_hot_ratio: float,
    max_iterations: int,
    tolerance: float,
    source_relaxation: float,
) -> None:
    if grids != STAGE43_GRIDS:
        raise ValueError("Stage 43 is fixed to the 8x8, 12x12 and 16x16 spatial sequence")
    if any(nx != ny for nx, ny in grids):
        raise ValueError("Stage 43 uses square physical grids")
    if rule != STAGE43_RULE:
        raise ValueError("Stage 43 retains the Stage 41 fine 32x96 polar rule")
    if kn0 != STAGE43_KNUDSEN:
        raise ValueError("Stage 43 is fixed to Kn0=0.1")
    if cold_hot_ratio != STAGE43_RATIO:
        raise ValueError("Stage 43 is fixed to Tcold/Thot=0.1")
    if max_iterations != STAGE43_MAX_ITERATIONS:
        raise ValueError("Stage 43 retains the preregistered 3000-iteration horizon")
    if tolerance != STAGE43_TOLERANCE:
        raise ValueError("Stage 43 retains tolerance=2e-5")
    if source_relaxation != STAGE43_SOURCE_RELAXATION:
        raise ValueError("Stage 43 does not tune source-iteration relaxation")


def build_stage43_config(grid: tuple[int, int]) -> LinearSidewallConfig:
    if grid not in STAGE43_GRIDS:
        raise ValueError("grid is not part of the frozen Stage 43 sequence")
    return LinearSidewallConfig(
        nx=grid[0],
        ny=grid[1],
        kn0=STAGE43_KNUDSEN,
        cold_hot_ratio=STAGE43_RATIO,
        viscosity_exponent=0.5,
        prandtl=STAGE41_PRANDTL,
        max_steps=STAGE43_MAX_ITERATIONS,
        cfl=0.2,
        tolerance=STAGE43_TOLERANCE,
        check_interval=STAGE42_CHECK_INTERVAL,
        minimum_steps=STAGE42_MINIMUM_ITERATIONS,
        positivity_floor=1.0e-30,
    )


def _case_is_stable(case: dict[str, object]) -> bool:
    return bool(
        case["finite"]
        and float(case["minimum_phi"]) > 0.0
        and float(case["minimum_psi"]) > 0.0
        and float(case["wall_mass_balance_relative_error"]) < 1.0e-10
    )


def reproduce_stage42_endpoint(case: dict[str, object]) -> dict[str, float | bool]:
    checks = {
        "predicted_qav_absolute_difference": abs(
            float(case["predicted_qav"]) - STAGE42_COMPLETED_ENDPOINT["predicted_qav"]
        ),
        "qav_relative_error_absolute_difference": abs(
            float(case["qav_relative_error"])
            - STAGE42_COMPLETED_ENDPOINT["qav_relative_error"]
        ),
        "velocity_relative_rms_absolute_difference": abs(
            float(case["velocity_metrics"]["relative_rms"])
            - STAGE42_COMPLETED_ENDPOINT["velocity_relative_rms"]
        ),
        "velocity_relative_l1_absolute_difference": abs(
            float(case["velocity_metrics"]["relative_l1"])
            - STAGE42_COMPLETED_ENDPOINT["velocity_relative_l1"]
        ),
        "velocity_sign_agreement_absolute_difference": abs(
            float(case["velocity_metrics"]["sign_agreement"])
            - STAGE42_COMPLETED_ENDPOINT["velocity_sign_agreement"]
        ),
        "wall_mass_balance_absolute_difference": abs(
            float(case["wall_mass_balance_relative_error"])
            - STAGE42_COMPLETED_ENDPOINT["wall_mass_balance_relative_error"]
        ),
    }
    checks["passed"] = bool(
        int(case["iterations"]) == STAGE42_COMPLETED_ENDPOINT["iterations"]
        and bool(case["converged"]) is STAGE42_COMPLETED_ENDPOINT["converged"]
        and all(
            float(value) <= STAGE43_REPRODUCTION_ATOL
            for key, value in checks.items()
            if key != "passed"
        )
    )
    return checks


def linear_h_extrapolation(
    grid_sizes: np.ndarray,
    qav: np.ndarray,
    table_velocity: np.ndarray,
    literature_qav: float,
    literature_velocity: np.ndarray,
) -> dict[str, object]:
    grid_sizes = np.asarray(grid_sizes, dtype=np.float64)
    qav = np.asarray(qav, dtype=np.float64)
    table_velocity = np.asarray(table_velocity, dtype=np.float64)
    literature_velocity = np.asarray(literature_velocity, dtype=np.float64)
    if grid_sizes.ndim != 1 or grid_sizes.size < 3:
        raise ValueError("at least three grid sizes are required")
    if qav.shape != grid_sizes.shape:
        raise ValueError("qav must have one value per grid")
    if table_velocity.shape[0] != grid_sizes.size:
        raise ValueError("table_velocity must have one row per grid")
    h = 1.0 / grid_sizes
    design = np.column_stack([h, np.ones_like(h)])
    q_coeff, _, _, _ = np.linalg.lstsq(design, qav, rcond=None)
    q_fit = design @ q_coeff
    q_sse = float(np.sum((qav - q_fit) ** 2))
    q_sst = float(np.sum((qav - np.mean(qav)) ** 2))
    q_r2 = 1.0 if q_sst == 0.0 and q_sse == 0.0 else 1.0 - q_sse / max(q_sst, 1.0e-300)

    profile_coeff, _, _, _ = np.linalg.lstsq(design, table_velocity, rcond=None)
    profile_fit = design @ profile_coeff
    profile_sse = float(np.sum((table_velocity - profile_fit) ** 2))
    centered = table_velocity - np.mean(table_velocity, axis=0, keepdims=True)
    profile_sst = float(np.sum(centered**2))
    profile_r2 = (
        1.0
        if profile_sst == 0.0 and profile_sse == 0.0
        else 1.0 - profile_sse / max(profile_sst, 1.0e-300)
    )
    extrapolated_qav = float(q_coeff[1])
    extrapolated_velocity = np.asarray(profile_coeff[1])
    return {
        "model": "observable(h)=slope*h+intercept with h=1/N",
        "grid_sizes": grid_sizes.astype(int).tolist(),
        "h": h.tolist(),
        "qav_slope": float(q_coeff[0]),
        "extrapolated_qav": extrapolated_qav,
        "qav_fit_r2": float(q_r2),
        "extrapolated_qav_relative_error": abs(extrapolated_qav - literature_qav)
        / max(abs(literature_qav), 1.0e-14),
        "extrapolated_table_velocity": extrapolated_velocity.tolist(),
        "velocity_fit_r2": float(profile_r2),
        "extrapolated_velocity_relative_rms": float(
            np.linalg.norm(extrapolated_velocity - literature_velocity)
            / max(float(np.linalg.norm(literature_velocity)), 1.0e-14)
        ),
    }


def _nonincreasing(values: list[float], relative_tolerance: float = 1.0e-12) -> bool:
    return all(
        second <= first * (1.0 + relative_tolerance)
        for first, second in zip(values[:-1], values[1:])
    )


def stage43_decision(cases: list[dict[str, object]], reproduction_passed: bool) -> str:
    if not reproduction_passed:
        return "stage42_reproduction_blocker"
    if len(cases) != len(STAGE43_GRIDS):
        return "stage43_resolution_blocker"
    if not all(_case_is_stable(case) and bool(case["converged"]) for case in cases):
        return "stage43_resolution_blocker"
    q_errors = [float(case["qav_relative_error"]) for case in cases]
    v_errors = [float(case["velocity_metrics"]["relative_rms"]) for case in cases]
    q_improves = _nonincreasing(q_errors)
    v_improves = _nonincreasing(v_errors)
    if q_improves and v_improves:
        finest_q_change = abs(float(cases[-1]["predicted_qav"]) - float(cases[-2]["predicted_qav"])) / max(
            abs(float(cases[-1]["predicted_qav"])), 1.0e-14
        )
        finest_profile_change = float(
            np.linalg.norm(
                np.asarray(cases[-1]["table_velocity"])
                - np.asarray(cases[-2]["table_velocity"])
            )
            / max(float(np.linalg.norm(cases[-1]["table_velocity"])), 1.0e-14)
        )
        if finest_q_change <= 0.05 and finest_profile_change <= 0.10:
            return "projected_polar_spatial_sequence_converging_stage44_cross_kn_extension"
        return "projected_polar_spatial_improvement_not_converged_stage44_finer_grid"
    if q_improves and not v_improves:
        return "projected_polar_heat_flux_improves_velocity_discrepancy_stage44_wall_observable_audit"
    if v_improves and not q_improves:
        return "projected_polar_velocity_improves_heat_flux_discrepancy_stage44_heat_flux_definition_audit"
    return "projected_polar_nonmonotonic_stage44_space_velocity_coupling_audit"


def _serializable_case(result: dict[str, object], grid: tuple[int, int]) -> dict[str, object]:
    return {
        "grid": list(grid),
        "iterations": int(result["iterations"]),
        "converged": bool(result["converged"]),
        "final_change": float(result["final_change"]),
        "predicted_qav": float(result["predicted_qav"]),
        "literature_qav": float(result["literature_qav"]),
        "qav_relative_error": float(result["qav_relative_error"]),
        "velocity_metrics": {
            key: float(value) for key, value in result["velocity_metrics"].items()
        },
        "wall_mass_balance_relative_error": float(result["wall_mass_balance_relative_error"]),
        "minimum_phi": float(result["minimum_phi"]),
        "minimum_psi": float(result["minimum_psi"]),
        "maximum_phi_clipped_weight_fraction": float(result["maximum_phi_clipped_weight_fraction"]),
        "maximum_psi_clipped_weight_fraction": float(result["maximum_psi_clipped_weight_fraction"]),
        "finite": bool(result["finite"]),
        "work_proxy": int(result["work_proxy"]),
        "table_velocity": np.asarray(result["table_velocity"]).tolist(),
    }


def run_stage43(
    output_dir: str | Path,
    *,
    grids: tuple[tuple[int, int], ...] = STAGE43_GRIDS,
    rule: tuple[int, int] = STAGE43_RULE,
    kn0: float = STAGE43_KNUDSEN,
    cold_hot_ratio: float = STAGE43_RATIO,
    max_iterations: int = STAGE43_MAX_ITERATIONS,
    tolerance: float = STAGE43_TOLERANCE,
    source_relaxation: float = STAGE43_SOURCE_RELAXATION,
) -> dict[str, object]:
    validate_stage43_design(
        grids, rule, kn0, cold_hot_ratio, max_iterations, tolerance,
        source_relaxation,
    )
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    quadrature = mapped_polar_quadrature(*rule)
    raw_results: list[dict[str, object]] = []
    cases: list[dict[str, object]] = []
    for grid in grids:
        cfg = build_stage43_config(grid)
        raw = solve_stage42_pilot(cfg, quadrature, source_relaxation)
        raw_results.append(raw)
        cases.append(_serializable_case(raw, grid))

    reproduction = reproduce_stage42_endpoint(raw_results[0])
    decision = stage43_decision(raw_results, bool(reproduction["passed"]))
    all_valid = bool(
        reproduction["passed"]
        and all(_case_is_stable(case) and bool(case["converged"]) for case in raw_results)
    )
    extrapolation: dict[str, object] | None = None
    if all_valid:
        extrapolation = linear_h_extrapolation(
            np.asarray([grid[0] for grid in grids]),
            np.asarray([case["predicted_qav"] for case in raw_results]),
            np.asarray([case["table_velocity"] for case in raw_results]),
            float(raw_results[0]["literature_qav"]),
            np.asarray(raw_results[0]["table_velocity"])
            - np.asarray(raw_results[0]["table_velocity"])
            + np.asarray(
                # The literature vector is reconstructed from prediction and
                # the public metric source inside the Stage 42 solver below.
                __import__(
                    "vgdsmc.linear_sidewall_validation",
                    fromlist=["TABLE3_UY_RATIO_0P1"],
                ).TABLE3_UY_RATIO_0P1[kn0]
            ),
        )

    q_errors = [float(case["qav_relative_error"]) for case in raw_results]
    v_errors = [float(case["velocity_metrics"]["relative_rms"]) for case in raw_results]
    summary = {
        "stage": 43,
        "description": (
            "Frozen-physics 8x8, 12x12 and 16x16 spatial-resolution sequence "
            "for the projected phi/psi mapped-polar heated-cavity solver"
        ),
        "retained_stage42_endpoint": STAGE42_COMPLETED_ENDPOINT,
        "stage42_reproduction": reproduction,
        "configuration": {
            "kn0": kn0,
            "cold_hot_ratio": cold_hot_ratio,
            "grids": [list(grid) for grid in grids],
            "polar_rule": {
                "radial_nodes": rule[0],
                "angular_nodes": rule[1],
                "point_count": quadrature.point_count,
            },
            "radial_mapping": "r=s*(1+x)/(1-x)",
            "radial_scale": quadrature.radial_scale,
            "prandtl": STAGE41_PRANDTL,
            "shakhov_correction_floor": STAGE41_CORRECTION_FLOOR,
            "transport_iteration": STAGE42_TRANSPORT,
            "collision": STAGE42_COLLISION,
            "relaxation_mapping": STAGE42_RELAXATION_MAPPING,
            "source_relaxation": source_relaxation,
            "max_iterations": max_iterations,
            "minimum_iterations": STAGE42_MINIMUM_ITERATIONS,
            "check_interval": STAGE42_CHECK_INTERVAL,
            "tolerance": tolerance,
            "physical_parameter_retuning": False,
            "velocity_quadrature_retuning": False,
            "source_resolution_reproduction": False,
        },
        "cases": cases,
        "monotonicity": {
            "qav_error_nonincreasing": _nonincreasing(q_errors),
            "velocity_rms_nonincreasing": _nonincreasing(v_errors),
            "qav_errors": q_errors,
            "velocity_rms_errors": v_errors,
        },
        "linear_h_extrapolation": extrapolation,
        "decision": decision,
        "interpretation_guard": (
            "Stage 43 changes only physical-grid resolution. The polar rule, "
            "Knudsen number, wall temperatures, Shakhov model, Prandtl number, "
            "viscosity law, relaxation mapping, source relaxation, positivity floor, "
            "transport order, stopping criteria and observable definitions remain fixed."
        ),
        "scientific_conclusion": (
            "All finite-grid improvements, degradations, nonmonotonic trends and "
            "nonconvergence are retained. A projected-polar cross-Knudsen claim is "
            "allowed only if both Table 6 heat flux and Table 3 velocity improve "
            "monotonically and the finest-grid changes are small."
        ),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    arrays: dict[str, np.ndarray] = {
        "quadrature_vx": quadrature.vx,
        "quadrature_vy": quadrature.vy,
        "quadrature_weight": quadrature.weight,
    }
    for grid, result in zip(grids, raw_results):
        label = f"n{grid[0]}"
        for key in (
            "T", "rho", "u", "v", "qx", "qy", "left_wall_velocity",
            "table_velocity", "bottom_heat_flux", "residual_history",
        ):
            arrays[f"{label}_{key}"] = np.asarray(result[key])
    np.savez_compressed(output_dir / "fields_and_profiles.npz", **arrays)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        default="results/stage43_projected_polar_resolution_sequence",
    )
    args = parser.parse_args()
    print(json.dumps(run_stage43(args.output_dir), indent=2))


if __name__ == "__main__":
    main()
