from __future__ import annotations

from pathlib import Path
import argparse
import json
import numpy as np

from .linear_sidewall_validation import LinearSidewallConfig, TABLE3_UY_RATIO_0P1
from .stage41_projected_polar_operator_audit import (
    STAGE41_CORRECTION_FLOOR,
    STAGE41_PRANDTL,
    mapped_polar_quadrature,
)
from .stage42_projected_polar_heated_cavity_pilot import (
    STAGE42_CHECK_INTERVAL,
    STAGE42_COLLISION,
    STAGE42_MINIMUM_ITERATIONS,
    STAGE42_RELAXATION_MAPPING,
    STAGE42_TRANSPORT,
    solve_stage42_pilot,
)
from .stage43_projected_polar_resolution_sequence import linear_h_extrapolation
from .stage45_projected_polar_32x32_confirmation import (
    STAGE44_RETAINED_CASES,
    STAGE45_KNUDSEN,
    STAGE45_MAX_ITERATIONS,
    STAGE45_RATIO,
    STAGE45_RULE,
    STAGE45_SOURCE_RELAXATION,
    STAGE45_TOLERANCE,
)


STAGE45_COMPLETED_ENDPOINT = {
    "workflow_run_id": 30751714392,
    "workflow_job_id": 91506805495,
    "workflow_conclusion": "success",
    "tests_passed": 41,
    "tests_failed": 0,
    "artifact_id": 8834940840,
    "artifact_size_bytes": 37009,
    "artifact_sha256": "3d0e7eacd03986c90646a08c9175a6de435e406ac94d8a6c4db76d140ccfd8c7",
    "source_head_sha": "6b96df929af9389542314625ca5ba46f5ed3c612",
    "decision": (
        "projected_polar_32x32_improving_not_converged_"
        "stage46_40x40_confirmation"
    ),
}

STAGE45_RETAINED_32X32_CASE = {
    "grid": [32, 32],
    "iterations": 850,
    "converged": True,
    "final_change": 9.852351250450031e-06,
    "predicted_qav": 0.07765042067403488,
    "literature_qav": 0.072,
    "qav_relative_error": 0.07847806491715122,
    "velocity_metrics": {
        "relative_rms": 0.9048587311194247,
        "relative_l1": 1.0065974563171856,
        "sign_agreement": 0.6,
    },
    "wall_mass_balance_relative_error": 2.2052024216374906e-16,
    "minimum_phi": 1.0e-30,
    "minimum_psi": 1.0e-30,
    "maximum_phi_clipped_weight_fraction": 0.002682339256022005,
    "maximum_psi_clipped_weight_fraction": 0.004964130817052622,
    "finite": True,
    "work_proxy": 2673868800,
    "table_velocity": [
        0.00307454895088456,
        0.0018761778363374521,
        0.0008824958468704334,
        0.0000878520709762072,
        -0.0005384053530629294,
        -0.0009789664797633854,
        -0.001134986651797936,
        -0.0008231556679703857,
        0.0002712966187804899,
        0.0025473100665375356,
    ],
}

STAGE46_GRID = (40, 40)
STAGE46_SEQUENCE_GRIDS = (20, 24, 32, 40)
STAGE46_RULE = STAGE45_RULE
STAGE46_KNUDSEN = STAGE45_KNUDSEN
STAGE46_RATIO = STAGE45_RATIO
STAGE46_MAX_ITERATIONS = STAGE45_MAX_ITERATIONS
STAGE46_TOLERANCE = STAGE45_TOLERANCE
STAGE46_SOURCE_RELAXATION = STAGE45_SOURCE_RELAXATION


def validate_stage46_design(
    grid: tuple[int, int],
    rule: tuple[int, int],
    kn0: float,
    cold_hot_ratio: float,
    max_iterations: int,
    tolerance: float,
    source_relaxation: float,
) -> None:
    if grid != STAGE46_GRID:
        raise ValueError("Stage 46 is fixed to one 40x40 confirmation grid")
    if rule != STAGE46_RULE:
        raise ValueError("Stage 46 retains the fixed 32x96 mapped-polar rule")
    if kn0 != STAGE46_KNUDSEN:
        raise ValueError("Stage 46 remains fixed at Kn0=0.1")
    if cold_hot_ratio != STAGE46_RATIO:
        raise ValueError("Stage 46 remains fixed at Tcold/Thot=0.1")
    if max_iterations != STAGE46_MAX_ITERATIONS:
        raise ValueError("Stage 46 retains the 3000-iteration horizon")
    if tolerance != STAGE46_TOLERANCE:
        raise ValueError("Stage 46 retains tolerance=2e-5")
    if source_relaxation != STAGE46_SOURCE_RELAXATION:
        raise ValueError("Stage 46 does not tune source relaxation")


def build_stage46_config() -> LinearSidewallConfig:
    return LinearSidewallConfig(
        nx=STAGE46_GRID[0],
        ny=STAGE46_GRID[1],
        kn0=STAGE46_KNUDSEN,
        cold_hot_ratio=STAGE46_RATIO,
        viscosity_exponent=0.5,
        prandtl=STAGE41_PRANDTL,
        max_steps=STAGE46_MAX_ITERATIONS,
        cfl=0.2,
        tolerance=STAGE46_TOLERANCE,
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


def stage46_decision(
    retained_32: dict[str, object],
    new_40: dict[str, object],
) -> str:
    if not bool(new_40["converged"]) or not _stable(new_40):
        return "stage46_40x40_blocker"

    q_improves = float(new_40["qav_relative_error"]) <= float(
        retained_32["qav_relative_error"]
    )
    v_improves = float(new_40["velocity_metrics"]["relative_rms"]) <= float(
        retained_32["velocity_metrics"]["relative_rms"]
    )
    q_change = abs(
        float(new_40["predicted_qav"]) - float(retained_32["predicted_qav"])
    ) / max(abs(float(new_40["predicted_qav"])), 1.0e-14)
    profile_change = float(
        np.linalg.norm(
            np.asarray(new_40["table_velocity"])
            - np.asarray(retained_32["table_velocity"])
        )
        / max(float(np.linalg.norm(np.asarray(new_40["table_velocity"]))), 1.0e-14)
    )

    if q_improves and v_improves:
        if q_change <= 0.03 and profile_change <= 0.10:
            return "projected_polar_40x40_converging_stage47_cross_kn_extension"
        return (
            "projected_polar_40x40_improving_not_converged_"
            "stage47_48x48_confirmation"
        )
    if q_improves and not v_improves:
        return (
            "projected_polar_40x40_heat_flux_only_improves_"
            "stage47_wall_observable_audit"
        )
    if v_improves and not q_improves:
        return (
            "projected_polar_40x40_velocity_only_improves_"
            "stage47_heat_flux_definition_audit"
        )
    return (
        "projected_polar_40x40_nonmonotonic_"
        "stage47_space_velocity_coupling_audit"
    )


def _serializable_case(result: dict[str, object]) -> dict[str, object]:
    return {
        "grid": list(STAGE46_GRID),
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


def run_stage46(
    output_dir: str | Path,
    *,
    grid: tuple[int, int] = STAGE46_GRID,
    rule: tuple[int, int] = STAGE46_RULE,
    kn0: float = STAGE46_KNUDSEN,
    cold_hot_ratio: float = STAGE46_RATIO,
    max_iterations: int = STAGE46_MAX_ITERATIONS,
    tolerance: float = STAGE46_TOLERANCE,
    source_relaxation: float = STAGE46_SOURCE_RELAXATION,
) -> dict[str, object]:
    validate_stage46_design(
        grid,
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
    cfg = build_stage46_config()
    raw = solve_stage42_pilot(cfg, quadrature, source_relaxation)
    case = _serializable_case(raw)
    retained_32 = STAGE45_RETAINED_32X32_CASE

    q_change = abs(
        float(case["predicted_qav"]) - float(retained_32["predicted_qav"])
    ) / max(abs(float(case["predicted_qav"])), 1.0e-14)
    profile_change = float(
        np.linalg.norm(
            np.asarray(case["table_velocity"])
            - np.asarray(retained_32["table_velocity"])
        )
        / max(float(np.linalg.norm(np.asarray(case["table_velocity"]))), 1.0e-14)
    )

    retained_cases = (
        STAGE44_RETAINED_CASES[0],
        STAGE44_RETAINED_CASES[1],
        STAGE45_RETAINED_32X32_CASE,
    )
    combined_q = np.asarray(
        [item["predicted_qav"] for item in retained_cases]
        + [case["predicted_qav"]],
        dtype=np.float64,
    )
    combined_v = np.asarray(
        [item["table_velocity"] for item in retained_cases]
        + [case["table_velocity"]],
        dtype=np.float64,
    )
    extrapolation = linear_h_extrapolation(
        np.asarray(STAGE46_SEQUENCE_GRIDS, dtype=np.float64),
        combined_q,
        combined_v,
        float(case["literature_qav"]),
        np.asarray(TABLE3_UY_RATIO_0P1[STAGE46_KNUDSEN]),
    )
    decision = stage46_decision(retained_32, raw)

    summary = {
        "stage": 46,
        "description": (
            "Frozen-physics 40x40 confirmation of the improving but unresolved "
            "Stage 45 projected phi/psi mapped-polar spatial sequence"
        ),
        "retained_stage45_endpoint": STAGE45_COMPLETED_ENDPOINT,
        "retained_cases": list(retained_cases),
        "configuration": {
            "kn0": kn0,
            "cold_hot_ratio": cold_hot_ratio,
            "grid": list(grid),
            "combined_sequence_grids": list(STAGE46_SEQUENCE_GRIDS),
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
        },
        "new_case": case,
        "finest_grid_changes": {
            "qav_change_32x32_to_40x40": q_change,
            "profile_change_32x32_to_40x40": profile_change,
            "qav_error_improves": (
                float(case["qav_relative_error"])
                <= float(retained_32["qav_relative_error"])
            ),
            "velocity_rms_improves": (
                float(case["velocity_metrics"]["relative_rms"])
                <= float(retained_32["velocity_metrics"]["relative_rms"])
            ),
        },
        "linear_h_extrapolation": extrapolation,
        "decision": decision,
        "interpretation_guard": (
            "Stage 46 changes only physical-grid resolution. Knudsen number, "
            "temperatures, projected Shakhov model, Prandtl number, viscosity law, "
            "relaxation mapping, mapped-polar velocity rule, source relaxation, "
            "positivity floor, transport order, stopping criteria and observables "
            "remain frozen. Extrapolation remains diagnostic and is not validation."
        ),
        "scientific_conclusion": (
            "Cross-Knudsen extension is permitted only if the 40x40 result is "
            "stable, both benchmark errors do not worsen, and the 32x32-to-40x40 "
            "heat-flux and wall-profile changes are at most 3% and 10%. Otherwise "
            "the positive, mixed or negative endpoint is retained without retuning."
        ),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    arrays: dict[str, np.ndarray] = {}
    for name in (
        "T",
        "rho",
        "u",
        "v",
        "qx",
        "qy",
        "phi",
        "psi",
        "left_wall_velocity",
        "table_velocity",
        "bottom_heat_flux",
        "residual_history",
    ):
        if name in raw:
            arrays[name] = np.asarray(raw[name])
    np.savez_compressed(output_dir / "fields_and_profiles.npz", **arrays)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Stage 46 projected-polar 40x40 confirmation"
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/stage46_projected_polar_40x40_confirmation",
    )
    args = parser.parse_args()
    print(json.dumps(run_stage46(args.output_dir), indent=2))


if __name__ == "__main__":
    main()
