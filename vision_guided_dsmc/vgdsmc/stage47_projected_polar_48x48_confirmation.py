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
from .stage46_projected_polar_40x40_confirmation import (
    STAGE45_RETAINED_32X32_CASE,
)


STAGE46_COMPLETED_ENDPOINT = {
    "workflow_run_id": 30753260337,
    "workflow_job_id": 91510943443,
    "workflow_conclusion": "success",
    "tests_passed": 45,
    "tests_failed": 0,
    "artifact_id": 8835645496,
    "artifact_size_bytes": 56107,
    "artifact_sha256": "384af3378996f74f7f17905f064c86c9aa702cbb2f86549cacef3998f24886e8",
    "source_head_sha": "9a7d48b7925b404513f1e2b40cff3c9588c60ce0",
    "decision": (
        "projected_polar_40x40_improving_not_converged_"
        "stage47_48x48_confirmation"
    ),
}

STAGE46_RETAINED_40X40_CASE = {
    "grid": [40, 40],
    "iterations": 1025,
    "converged": True,
    "final_change": 1.2884848201499821e-05,
    "predicted_qav": 0.07691741440108028,
    "literature_qav": 0.072,
    "qav_relative_error": 0.06829742223722624,
    "velocity_metrics": {
        "relative_rms": 0.6767910267583875,
        "relative_l1": 0.7455075134434416,
        "sign_agreement": 0.8,
    },
    "wall_mass_balance_relative_error": 1.8272734223688737e-16,
    "minimum_phi": 1.0e-30,
    "minimum_psi": 1.0e-30,
    "maximum_phi_clipped_weight_fraction": 0.002831966864582501,
    "maximum_psi_clipped_weight_fraction": 0.004914176009276124,
    "finite": True,
    "work_proxy": 5038080000,
    "table_velocity": [
        0.0027800270702341757,
        0.0015922435966530667,
        0.0005708568915356494,
        -0.0002564131195976637,
        -0.0009161317429350799,
        -0.0013933468141399737,
        -0.0015931924002603986,
        -0.001332742785050453,
        -0.00029288504235082066,
        0.001983402475669999,
    ],
}

STAGE47_GRID = (48, 48)
STAGE47_SEQUENCE_GRIDS = (20, 24, 32, 40, 48)
STAGE47_RULE = STAGE45_RULE
STAGE47_KNUDSEN = STAGE45_KNUDSEN
STAGE47_RATIO = STAGE45_RATIO
STAGE47_MAX_ITERATIONS = STAGE45_MAX_ITERATIONS
STAGE47_TOLERANCE = STAGE45_TOLERANCE
STAGE47_SOURCE_RELAXATION = STAGE45_SOURCE_RELAXATION


def validate_stage47_design(
    grid: tuple[int, int],
    rule: tuple[int, int],
    kn0: float,
    cold_hot_ratio: float,
    max_iterations: int,
    tolerance: float,
    source_relaxation: float,
) -> None:
    if grid != STAGE47_GRID:
        raise ValueError("Stage 47 is fixed to one 48x48 confirmation grid")
    if rule != STAGE47_RULE:
        raise ValueError("Stage 47 retains the fixed 32x96 mapped-polar rule")
    if kn0 != STAGE47_KNUDSEN:
        raise ValueError("Stage 47 remains fixed at Kn0=0.1")
    if cold_hot_ratio != STAGE47_RATIO:
        raise ValueError("Stage 47 remains fixed at Tcold/Thot=0.1")
    if max_iterations != STAGE47_MAX_ITERATIONS:
        raise ValueError("Stage 47 retains the 3000-iteration horizon")
    if tolerance != STAGE47_TOLERANCE:
        raise ValueError("Stage 47 retains tolerance=2e-5")
    if source_relaxation != STAGE47_SOURCE_RELAXATION:
        raise ValueError("Stage 47 does not tune source relaxation")


def build_stage47_config() -> LinearSidewallConfig:
    return LinearSidewallConfig(
        nx=STAGE47_GRID[0],
        ny=STAGE47_GRID[1],
        kn0=STAGE47_KNUDSEN,
        cold_hot_ratio=STAGE47_RATIO,
        viscosity_exponent=0.5,
        prandtl=STAGE41_PRANDTL,
        max_steps=STAGE47_MAX_ITERATIONS,
        cfl=0.2,
        tolerance=STAGE47_TOLERANCE,
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


def stage47_decision(
    retained_40: dict[str, object],
    new_48: dict[str, object],
) -> str:
    if not bool(new_48["converged"]) or not _stable(new_48):
        return "stage47_48x48_blocker"

    q_improves = float(new_48["qav_relative_error"]) <= float(
        retained_40["qav_relative_error"]
    )
    v_improves = float(new_48["velocity_metrics"]["relative_rms"]) <= float(
        retained_40["velocity_metrics"]["relative_rms"]
    )
    q_change = abs(
        float(new_48["predicted_qav"]) - float(retained_40["predicted_qav"])
    ) / max(abs(float(new_48["predicted_qav"])), 1.0e-14)
    profile_change = float(
        np.linalg.norm(
            np.asarray(new_48["table_velocity"])
            - np.asarray(retained_40["table_velocity"])
        )
        / max(float(np.linalg.norm(np.asarray(new_48["table_velocity"]))), 1.0e-14)
    )

    if q_improves and v_improves:
        if q_change <= 0.03 and profile_change <= 0.10:
            return "projected_polar_48x48_converging_stage48_cross_kn_extension"
        return (
            "projected_polar_48x48_improving_not_converged_"
            "stage48_56x56_confirmation"
        )
    if q_improves and not v_improves:
        return (
            "projected_polar_48x48_heat_flux_only_improves_"
            "stage48_wall_observable_audit"
        )
    if v_improves and not q_improves:
        return (
            "projected_polar_48x48_velocity_only_improves_"
            "stage48_heat_flux_definition_audit"
        )
    return (
        "projected_polar_48x48_nonmonotonic_"
        "stage48_space_velocity_coupling_audit"
    )


def _serializable_case(result: dict[str, object]) -> dict[str, object]:
    return {
        "grid": list(STAGE47_GRID),
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


def run_stage47(
    output_dir: str | Path,
    *,
    grid: tuple[int, int] = STAGE47_GRID,
    rule: tuple[int, int] = STAGE47_RULE,
    kn0: float = STAGE47_KNUDSEN,
    cold_hot_ratio: float = STAGE47_RATIO,
    max_iterations: int = STAGE47_MAX_ITERATIONS,
    tolerance: float = STAGE47_TOLERANCE,
    source_relaxation: float = STAGE47_SOURCE_RELAXATION,
) -> dict[str, object]:
    validate_stage47_design(
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
    cfg = build_stage47_config()
    raw = solve_stage42_pilot(cfg, quadrature, source_relaxation)
    case = _serializable_case(raw)
    retained_40 = STAGE46_RETAINED_40X40_CASE

    q_change = abs(
        float(case["predicted_qav"]) - float(retained_40["predicted_qav"])
    ) / max(abs(float(case["predicted_qav"])), 1.0e-14)
    profile_change = float(
        np.linalg.norm(
            np.asarray(case["table_velocity"])
            - np.asarray(retained_40["table_velocity"])
        )
        / max(float(np.linalg.norm(np.asarray(case["table_velocity"]))), 1.0e-14)
    )

    retained_cases = (
        STAGE44_RETAINED_CASES[0],
        STAGE44_RETAINED_CASES[1],
        STAGE45_RETAINED_32X32_CASE,
        STAGE46_RETAINED_40X40_CASE,
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
        np.asarray(STAGE47_SEQUENCE_GRIDS, dtype=np.float64),
        combined_q,
        combined_v,
        float(case["literature_qav"]),
        np.asarray(TABLE3_UY_RATIO_0P1[STAGE47_KNUDSEN]),
    )
    decision = stage47_decision(retained_40, raw)

    summary = {
        "stage": 47,
        "description": (
            "Frozen-physics 48x48 confirmation of the improving but unresolved "
            "Stage 46 projected phi/psi mapped-polar spatial sequence"
        ),
        "retained_stage46_endpoint": STAGE46_COMPLETED_ENDPOINT,
        "retained_cases": list(retained_cases),
        "configuration": {
            "kn0": kn0,
            "cold_hot_ratio": cold_hot_ratio,
            "grid": list(grid),
            "combined_sequence_grids": list(STAGE47_SEQUENCE_GRIDS),
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
            "qav_change_40x40_to_48x48": q_change,
            "profile_change_40x40_to_48x48": profile_change,
            "qav_error_improves": (
                float(case["qav_relative_error"])
                <= float(retained_40["qav_relative_error"])
            ),
            "velocity_rms_improves": (
                float(case["velocity_metrics"]["relative_rms"])
                <= float(retained_40["velocity_metrics"]["relative_rms"])
            ),
        },
        "linear_h_extrapolation": extrapolation,
        "decision": decision,
        "interpretation_guard": (
            "Stage 47 changes only physical-grid resolution. Knudsen number, "
            "temperatures, projected Shakhov model, Prandtl number, viscosity law, "
            "relaxation mapping, mapped-polar velocity rule, source relaxation, "
            "positivity floor, transport order, stopping criteria and observables "
            "remain frozen. Extrapolation remains diagnostic and is not validation."
        ),
        "scientific_conclusion": (
            "Cross-Knudsen extension is permitted only if the 48x48 result is "
            "stable, both benchmark errors do not worsen, and the 40x40-to-48x48 "
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
        description="Run Stage 47 projected-polar 48x48 confirmation"
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/stage47_projected_polar_48x48_confirmation",
    )
    args = parser.parse_args()
    print(json.dumps(run_stage47(args.output_dir), indent=2))


if __name__ == "__main__":
    main()
