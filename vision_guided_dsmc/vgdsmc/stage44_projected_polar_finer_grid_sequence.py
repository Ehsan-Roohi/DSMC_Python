from __future__ import annotations

from pathlib import Path
import argparse
import json
import math
import numpy as np

from .linear_sidewall_validation import LinearSidewallConfig, TABLE3_UY_RATIO_0P1
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
from .stage43_projected_polar_resolution_sequence import linear_h_extrapolation


STAGE43_COMPLETED_ENDPOINT = {
    "workflow_run_id": 30733957628,
    "workflow_job_id": 91459088187,
    "workflow_conclusion": "success",
    "tests_passed": 53,
    "tests_failed": 0,
    "artifact_id": 8830544995,
    "artifact_size_bytes": 46042,
    "artifact_sha256": "70bb2a08a1e49cfe31091ce91cf26c286c7652ae7c341855a4ec1dc81de528dd",
    "source_head_sha": "27f33f1be007cf81ea97a1cc452c248e07d83470",
    "decision": "projected_polar_spatial_improvement_not_converged_stage44_finer_grid",
}

STAGE43_FINEST_CASE = {
    "grid": [16, 16],
    "iterations": 500,
    "converged": True,
    "final_change": 1.9979318251275857e-06,
    "predicted_qav": 0.0814053328666412,
    "literature_qav": 0.072,
    "qav_relative_error": 0.13062962314779453,
    "velocity_metrics": {
        "relative_rms": 2.0834768337668104,
        "relative_l1": 2.357676714858226,
        "sign_agreement": 0.2,
    },
    "wall_mass_balance_relative_error": 1.4780919171369843e-16,
    "maximum_phi_clipped_weight_fraction": 0.0019625644653364113,
    "maximum_psi_clipped_weight_fraction": 0.004751847239489797,
    "table_velocity": [
        0.0044983536901870985,
        0.0033595034323805695,
        0.0025388603631742845,
        0.0019255545960640092,
        0.0015045702395754773,
        0.0012841160966272694,
        0.0013499756163744478,
        0.0019093348441055612,
        0.003117631398644011,
        0.005108457373763799,
    ],
}

STAGE44_GRIDS = ((20, 20), (24, 24))
STAGE44_SEQUENCE_GRIDS = (16, 20, 24)
STAGE44_RULE = STAGE41_FINE_RULE
STAGE44_KNUDSEN = STAGE42_KNUDSEN
STAGE44_RATIO = STAGE42_RATIO
STAGE44_MAX_ITERATIONS = STAGE42_MAX_ITERATIONS
STAGE44_TOLERANCE = STAGE42_TOLERANCE
STAGE44_SOURCE_RELAXATION = STAGE42_SOURCE_RELAXATION


def validate_stage44_design(
    grids: tuple[tuple[int, int], ...],
    rule: tuple[int, int],
    kn0: float,
    cold_hot_ratio: float,
    max_iterations: int,
    tolerance: float,
    source_relaxation: float,
) -> None:
    if grids != STAGE44_GRIDS:
        raise ValueError("Stage 44 is fixed to new 20x20 and 24x24 grids")
    if any(nx != ny for nx, ny in grids):
        raise ValueError("Stage 44 uses square physical grids")
    if rule != STAGE44_RULE:
        raise ValueError("Stage 44 retains the fixed 32x96 mapped-polar rule")
    if kn0 != STAGE44_KNUDSEN:
        raise ValueError("Stage 44 remains fixed at Kn0=0.1")
    if cold_hot_ratio != STAGE44_RATIO:
        raise ValueError("Stage 44 remains fixed at Tcold/Thot=0.1")
    if max_iterations != STAGE44_MAX_ITERATIONS:
        raise ValueError("Stage 44 retains the 3000-iteration horizon")
    if tolerance != STAGE44_TOLERANCE:
        raise ValueError("Stage 44 retains tolerance=2e-5")
    if source_relaxation != STAGE44_SOURCE_RELAXATION:
        raise ValueError("Stage 44 does not tune source relaxation")


def build_stage44_config(grid: tuple[int, int]) -> LinearSidewallConfig:
    if grid not in STAGE44_GRIDS:
        raise ValueError("grid is not part of the frozen Stage 44 design")
    return LinearSidewallConfig(
        nx=grid[0],
        ny=grid[1],
        kn0=STAGE44_KNUDSEN,
        cold_hot_ratio=STAGE44_RATIO,
        viscosity_exponent=0.5,
        prandtl=STAGE41_PRANDTL,
        max_steps=STAGE44_MAX_ITERATIONS,
        cfl=0.2,
        tolerance=STAGE44_TOLERANCE,
        check_interval=STAGE42_CHECK_INTERVAL,
        minimum_steps=STAGE42_MINIMUM_ITERATIONS,
        positivity_floor=1.0e-30,
    )


def _stable(case: dict[str, object]) -> bool:
    return bool(
        case["finite"]
        and float(case["minimum_phi"]) > 0.0
        and float(case["minimum_psi"]) > 0.0
        and float(case["wall_mass_balance_relative_error"]) < 1.0e-10
    )


def _nonincreasing(values: list[float], relative_tolerance: float = 1.0e-12) -> bool:
    return all(
        second <= first * (1.0 + relative_tolerance)
        for first, second in zip(values[:-1], values[1:])
    )


def stage44_decision(
    retained_16: dict[str, object],
    new_cases: list[dict[str, object]],
) -> str:
    if len(new_cases) != len(STAGE44_GRIDS):
        return "stage44_finer_grid_blocker"
    if not all(_stable(case) and bool(case["converged"]) for case in new_cases):
        return "stage44_finer_grid_blocker"
    sequence = [retained_16, *new_cases]
    q_errors = [float(case["qav_relative_error"]) for case in sequence]
    v_errors = [float(case["velocity_metrics"]["relative_rms"]) for case in sequence]
    q_improves = _nonincreasing(q_errors)
    v_improves = _nonincreasing(v_errors)
    if q_improves and v_improves:
        q_change = abs(
            float(new_cases[-1]["predicted_qav"])
            - float(new_cases[-2]["predicted_qav"])
        ) / max(abs(float(new_cases[-1]["predicted_qav"])), 1.0e-14)
        profile_change = float(
            np.linalg.norm(
                np.asarray(new_cases[-1]["table_velocity"])
                - np.asarray(new_cases[-2]["table_velocity"])
            )
            / max(
                float(np.linalg.norm(np.asarray(new_cases[-1]["table_velocity"]))),
                1.0e-14,
            )
        )
        if q_change <= 0.03 and profile_change <= 0.10:
            return "projected_polar_finer_grid_converging_stage45_cross_kn_extension"
        return "projected_polar_finer_grid_improving_not_converged_stage45_32x32_confirmation"
    if q_improves and not v_improves:
        return "projected_polar_heat_flux_only_improves_stage45_wall_observable_audit"
    if v_improves and not q_improves:
        return "projected_polar_velocity_only_improves_stage45_heat_flux_definition_audit"
    return "projected_polar_finer_grid_nonmonotonic_stage45_space_velocity_coupling_audit"


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
        "wall_mass_balance_relative_error": float(
            result["wall_mass_balance_relative_error"]
        ),
        "minimum_phi": float(result["minimum_phi"]),
        "minimum_psi": float(result["minimum_psi"]),
        "maximum_phi_clipped_weight_fraction": float(
            result["maximum_phi_clipped_weight_fraction"]
        ),
        "maximum_psi_clipped_weight_fraction": float(
            result["maximum_psi_clipped_weight_fraction"]
        ),
        "finite": bool(result["finite"]),
        "work_proxy": int(result["work_proxy"]),
        "table_velocity": np.asarray(result["table_velocity"]).tolist(),
    }


def run_stage44(
    output_dir: str | Path,
    *,
    grids: tuple[tuple[int, int], ...] = STAGE44_GRIDS,
    rule: tuple[int, int] = STAGE44_RULE,
    kn0: float = STAGE44_KNUDSEN,
    cold_hot_ratio: float = STAGE44_RATIO,
    max_iterations: int = STAGE44_MAX_ITERATIONS,
    tolerance: float = STAGE44_TOLERANCE,
    source_relaxation: float = STAGE44_SOURCE_RELAXATION,
) -> dict[str, object]:
    validate_stage44_design(
        grids,
        rule,
        kn0,
        cold_hot_ratio,
        max_iterations,
        tolerance,
        source_relaxation,
    )
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    quadrature = mapped_polar_quadrature(*rule)
    raw_results: list[dict[str, object]] = []
    cases: list[dict[str, object]] = []
    for grid in grids:
        cfg = build_stage44_config(grid)
        raw = solve_stage42_pilot(cfg, quadrature, source_relaxation)
        raw_results.append(raw)
        cases.append(_serializable_case(raw, grid))

    decision = stage44_decision(STAGE43_FINEST_CASE, raw_results)
    combined_q = np.asarray(
        [STAGE43_FINEST_CASE["predicted_qav"], *[r["predicted_qav"] for r in raw_results]],
        dtype=np.float64,
    )
    combined_v = np.asarray(
        [STAGE43_FINEST_CASE["table_velocity"], *[r["table_velocity"] for r in raw_results]],
        dtype=np.float64,
    )
    extrapolation = linear_h_extrapolation(
        np.asarray(STAGE44_SEQUENCE_GRIDS, dtype=np.float64),
        combined_q,
        combined_v,
        float(STAGE43_FINEST_CASE["literature_qav"]),
        np.asarray(TABLE3_UY_RATIO_0P1[STAGE44_KNUDSEN]),
    )
    q_errors = [
        float(STAGE43_FINEST_CASE["qav_relative_error"]),
        *[float(r["qav_relative_error"]) for r in raw_results],
    ]
    v_errors = [
        float(STAGE43_FINEST_CASE["velocity_metrics"]["relative_rms"]),
        *[float(r["velocity_metrics"]["relative_rms"]) for r in raw_results],
    ]
    q_change_20_to_24 = abs(
        float(raw_results[-1]["predicted_qav"])
        - float(raw_results[-2]["predicted_qav"])
    ) / max(abs(float(raw_results[-1]["predicted_qav"])), 1.0e-14)
    profile_change_20_to_24 = float(
        np.linalg.norm(
            np.asarray(raw_results[-1]["table_velocity"])
            - np.asarray(raw_results[-2]["table_velocity"])
        )
        / max(
            float(np.linalg.norm(np.asarray(raw_results[-1]["table_velocity"]))),
            1.0e-14,
        )
    )
    summary = {
        "stage": 44,
        "description": (
            "Frozen-physics 20x20 and 24x24 continuation of the Stage 43 "
            "projected phi/psi mapped-polar spatial-resolution sequence"
        ),
        "retained_stage43_endpoint": STAGE43_COMPLETED_ENDPOINT,
        "retained_stage43_finest_case": STAGE43_FINEST_CASE,
        "configuration": {
            "kn0": kn0,
            "cold_hot_ratio": cold_hot_ratio,
            "new_grids": [list(grid) for grid in grids],
            "combined_sequence_grids": list(STAGE44_SEQUENCE_GRIDS),
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
        "new_cases": cases,
        "combined_monotonicity": {
            "qav_error_nonincreasing": _nonincreasing(q_errors),
            "velocity_rms_nonincreasing": _nonincreasing(v_errors),
            "qav_errors": q_errors,
            "velocity_rms_errors": v_errors,
            "qav_change_20x20_to_24x24": q_change_20_to_24,
            "profile_change_20x20_to_24x24": profile_change_20_to_24,
        },
        "linear_h_extrapolation": extrapolation,
        "decision": decision,
        "interpretation_guard": (
            "Stage 44 changes only physical-grid resolution. Knudsen number, "
            "temperatures, projected Shakhov model, Prandtl number, viscosity law, "
            "relaxation mapping, mapped-polar velocity rule, source relaxation, "
            "positivity floor, transport order, stopping criteria and observables "
            "remain frozen."
        ),
        "scientific_conclusion": (
            "The Stage 43 trend is advanced only if both average heat-flux error "
            "and wall-velocity RMS continue to decrease on 20x20 and 24x24 grids. "
            "A cross-Knudsen extension requires small finest-grid changes; otherwise "
            "the unresolved positive or negative endpoint is retained."
        ),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    arrays: dict[str, np.ndarray] = {
        "retained_table_velocity_16x16": np.asarray(
            STAGE43_FINEST_CASE["table_velocity"], dtype=np.float64
        )
    }
    for grid, result in zip(grids, raw_results):
        token = f"{grid[0]}x{grid[1]}"
        for key in (
            "T", "rho", "u", "v", "qx", "qy", "left_wall_velocity",
            "table_velocity", "bottom_heat_flux", "residual_history",
        ):
            arrays[f"{key}_{token}"] = np.asarray(result[key], dtype=np.float64)
    np.savez_compressed(output_dir / "fields_and_profiles.npz", **arrays)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    summary = run_stage44(args.output_dir)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
