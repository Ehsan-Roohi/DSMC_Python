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
from .stage46_projected_polar_40x40_confirmation import STAGE45_RETAINED_32X32_CASE
from .stage47_projected_polar_48x48_confirmation import STAGE46_RETAINED_40X40_CASE


STAGE47_COMPLETED_ENDPOINT = {
    "workflow_run_id": 30755677448,
    "workflow_job_id": 91517302666,
    "workflow_conclusion": "success",
    "tests_passed": 49,
    "tests_failed": 0,
    "artifact_id": 8838257601,
    "artifact_size_bytes": 78522,
    "artifact_sha256": "58ef46902d6e80f3e2c198fa6d33f0f9209331da08e14664b9c4e22b1ddb0227",
    "source_head_sha": "10eba2a345c8d6b0942bce6bbed2c0d030dc005a",
    "decision": (
        "projected_polar_48x48_improving_not_converged_"
        "stage48_56x56_confirmation"
    ),
}

STAGE47_RETAINED_48X48_CASE = {
    "grid": [48, 48],
    "iterations": 1200,
    "converged": True,
    "final_change": 1.48828827501446e-05,
    "predicted_qav": 0.07643318669591011,
    "literature_qav": 0.072,
    "qav_relative_error": 0.06157203744319602,
    "velocity_metrics": {
        "relative_rms": 0.5272365313931947,
        "relative_l1": 0.5745980077213692,
        "sign_agreement": 0.8,
    },
    "wall_mass_balance_relative_error": 2.1351859343697615e-16,
    "minimum_phi": 1.0e-30,
    "minimum_psi": 1.0e-30,
    "maximum_phi_clipped_weight_fraction": 0.002952525420025651,
    "maximum_psi_clipped_weight_fraction": 0.004908712341779989,
    "finite": True,
    "work_proxy": 8493465600,
    "table_velocity": [
        0.0025841673664771764,
        0.0014070949476494127,
        0.0003687553431730256,
        -0.00047945415584926143,
        -0.0011607680412823252,
        -0.0016605378969302414,
        -0.001885070415301367,
        -0.0016557618579879196,
        -0.0006589371340178195,
        0.0015838397879820172,
    ],
}

STAGE48_GRID = (56, 56)
STAGE48_SEQUENCE_GRIDS = (20, 24, 32, 40, 48, 56)
STAGE48_RULE = STAGE45_RULE
STAGE48_KNUDSEN = STAGE45_KNUDSEN
STAGE48_RATIO = STAGE45_RATIO
STAGE48_MAX_ITERATIONS = STAGE45_MAX_ITERATIONS
STAGE48_TOLERANCE = STAGE45_TOLERANCE
STAGE48_SOURCE_RELAXATION = STAGE45_SOURCE_RELAXATION


def validate_stage48_design(
    grid: tuple[int, int],
    rule: tuple[int, int],
    kn0: float,
    cold_hot_ratio: float,
    max_iterations: int,
    tolerance: float,
    source_relaxation: float,
) -> None:
    if grid != STAGE48_GRID:
        raise ValueError("Stage 48 is fixed to one 56x56 confirmation grid")
    if rule != STAGE48_RULE:
        raise ValueError("Stage 48 retains the fixed 32x96 mapped-polar rule")
    if kn0 != STAGE48_KNUDSEN:
        raise ValueError("Stage 48 remains fixed at Kn0=0.1")
    if cold_hot_ratio != STAGE48_RATIO:
        raise ValueError("Stage 48 remains fixed at Tcold/Thot=0.1")
    if max_iterations != STAGE48_MAX_ITERATIONS:
        raise ValueError("Stage 48 retains the 3000-iteration horizon")
    if tolerance != STAGE48_TOLERANCE:
        raise ValueError("Stage 48 retains tolerance=2e-5")
    if source_relaxation != STAGE48_SOURCE_RELAXATION:
        raise ValueError("Stage 48 does not tune source relaxation")


def build_stage48_config() -> LinearSidewallConfig:
    return LinearSidewallConfig(
        nx=STAGE48_GRID[0],
        ny=STAGE48_GRID[1],
        kn0=STAGE48_KNUDSEN,
        cold_hot_ratio=STAGE48_RATIO,
        viscosity_exponent=0.5,
        prandtl=STAGE41_PRANDTL,
        max_steps=STAGE48_MAX_ITERATIONS,
        cfl=0.2,
        tolerance=STAGE48_TOLERANCE,
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


def stage48_decision(
    retained_48: dict[str, object],
    new_56: dict[str, object],
) -> str:
    if not bool(new_56["converged"]) or not _stable(new_56):
        return "stage48_56x56_blocker"

    q_improves = float(new_56["qav_relative_error"]) <= float(
        retained_48["qav_relative_error"]
    )
    v_improves = float(new_56["velocity_metrics"]["relative_rms"]) <= float(
        retained_48["velocity_metrics"]["relative_rms"]
    )
    q_change = abs(
        float(new_56["predicted_qav"]) - float(retained_48["predicted_qav"])
    ) / max(abs(float(new_56["predicted_qav"])), 1.0e-14)
    profile_change = float(
        np.linalg.norm(
            np.asarray(new_56["table_velocity"])
            - np.asarray(retained_48["table_velocity"])
        )
        / max(float(np.linalg.norm(np.asarray(new_56["table_velocity"]))), 1.0e-14)
    )

    if q_improves and v_improves:
        if q_change <= 0.03 and profile_change <= 0.10:
            return "projected_polar_56x56_converging_stage49_cross_kn_extension"
        return (
            "projected_polar_56x56_improving_not_converged_"
            "stage49_64x64_confirmation"
        )
    if q_improves and not v_improves:
        return (
            "projected_polar_56x56_heat_flux_only_improves_"
            "stage49_wall_observable_audit"
        )
    if v_improves and not q_improves:
        return (
            "projected_polar_56x56_velocity_only_improves_"
            "stage49_heat_flux_definition_audit"
        )
    return (
        "projected_polar_56x56_nonmonotonic_"
        "stage49_space_velocity_coupling_audit"
    )


def _serializable_case(result: dict[str, object]) -> dict[str, object]:
    return {
        "grid": list(STAGE48_GRID),
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


def run_stage48(
    output_dir: str | Path,
    *,
    grid: tuple[int, int] = STAGE48_GRID,
    rule: tuple[int, int] = STAGE48_RULE,
    kn0: float = STAGE48_KNUDSEN,
    cold_hot_ratio: float = STAGE48_RATIO,
    max_iterations: int = STAGE48_MAX_ITERATIONS,
    tolerance: float = STAGE48_TOLERANCE,
    source_relaxation: float = STAGE48_SOURCE_RELAXATION,
) -> dict[str, object]:
    validate_stage48_design(
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
    cfg = build_stage48_config()
    raw = solve_stage42_pilot(cfg, quadrature, source_relaxation)
    case = _serializable_case(raw)
    retained_48 = STAGE47_RETAINED_48X48_CASE

    q_change = abs(
        float(case["predicted_qav"]) - float(retained_48["predicted_qav"])
    ) / max(abs(float(case["predicted_qav"])), 1.0e-14)
    profile_change = float(
        np.linalg.norm(
            np.asarray(case["table_velocity"])
            - np.asarray(retained_48["table_velocity"])
        )
        / max(float(np.linalg.norm(np.asarray(case["table_velocity"]))), 1.0e-14)
    )

    retained_cases = (
        STAGE44_RETAINED_CASES[0],
        STAGE44_RETAINED_CASES[1],
        STAGE45_RETAINED_32X32_CASE,
        STAGE46_RETAINED_40X40_CASE,
        STAGE47_RETAINED_48X48_CASE,
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
        np.asarray(STAGE48_SEQUENCE_GRIDS, dtype=np.float64),
        combined_q,
        combined_v,
        float(case["literature_qav"]),
        np.asarray(TABLE3_UY_RATIO_0P1[STAGE48_KNUDSEN]),
    )
    decision = stage48_decision(retained_48, raw)

    summary = {
        "stage": 48,
        "description": (
            "Frozen-physics 56x56 confirmation of the improving but unresolved "
            "Stage 47 projected phi/psi mapped-polar spatial sequence"
        ),
        "retained_stage47_endpoint": STAGE47_COMPLETED_ENDPOINT,
        "retained_cases": list(retained_cases),
        "configuration": {
            "kn0": kn0,
            "cold_hot_ratio": cold_hot_ratio,
            "grid": list(grid),
            "combined_sequence_grids": list(STAGE48_SEQUENCE_GRIDS),
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
            "qav_change_48x48_to_56x56": q_change,
            "profile_change_48x48_to_56x56": profile_change,
            "qav_error_improves": (
                float(case["qav_relative_error"])
                <= float(retained_48["qav_relative_error"])
            ),
            "velocity_rms_improves": (
                float(case["velocity_metrics"]["relative_rms"])
                <= float(retained_48["velocity_metrics"]["relative_rms"])
            ),
        },
        "linear_h_extrapolation": extrapolation,
        "decision": decision,
        "interpretation_guard": (
            "Stage 48 changes only physical-grid resolution. Knudsen number, "
            "temperatures, projected Shakhov model, Prandtl number, viscosity law, "
            "relaxation mapping, mapped-polar velocity rule, source relaxation, "
            "positivity floor, transport order, stopping criteria and observables "
            "remain frozen. Extrapolation remains diagnostic and is not validation."
        ),
        "scientific_conclusion": (
            "Cross-Knudsen extension is permitted only if the 56x56 result is "
            "stable, both benchmark errors do not worsen, and the 48x48-to-56x56 "
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
        description="Run Stage 48 projected-polar 56x56 confirmation"
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/stage48_projected_polar_56x56_confirmation",
    )
    args = parser.parse_args()
    print(json.dumps(run_stage48(args.output_dir), indent=2))


if __name__ == "__main__":
    main()
