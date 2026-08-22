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
from .stage44_projected_polar_finer_grid_sequence import (
    STAGE44_KNUDSEN,
    STAGE44_MAX_ITERATIONS,
    STAGE44_RATIO,
    STAGE44_RULE,
    STAGE44_SOURCE_RELAXATION,
    STAGE44_TOLERANCE,
)


STAGE44_COMPLETED_ENDPOINT = {
    "workflow_run_id": 30743552934,
    "workflow_job_id": 91485129929,
    "workflow_conclusion": "success",
    "tests_passed": 37,
    "tests_failed": 0,
    "artifact_id": 8832382991,
    "artifact_size_bytes": 37290,
    "artifact_sha256": "3eabd271c4b05ccdd6c04d462398ecf08cb70411ef1b977424171a619df6de34",
    "source_head_sha": "947cabc1b75636c8e19a9d29304ca07d5fb29f52",
    "decision": "projected_polar_finer_grid_improving_not_converged_stage45_32x32_confirmation",
}

STAGE44_RETAINED_CASES = (
    {
        "grid": [20, 20],
        "iterations": 550,
        "converged": True,
        "final_change": 9.611691218588951e-06,
        "predicted_qav": 0.07988685618966523,
        "literature_qav": 0.072,
        "qav_relative_error": 0.10953966930090611,
        "velocity_metrics": {
            "relative_rms": 1.6137584554918936,
            "relative_l1": 1.818079614086768,
            "sign_agreement": 0.2,
        },
        "wall_mass_balance_relative_error": 1.7584297059707913e-16,
        "maximum_phi_clipped_weight_fraction": 0.0022314458914989635,
        "maximum_psi_clipped_weight_fraction": 0.004820725128282489,
        "table_velocity": [
            0.003940044894081905,
            0.002763769303288723,
            0.0018634912771260097,
            0.0011776544004352262,
            0.000668866859999322,
            0.0003570509290675451,
            0.00034081425109317717,
            0.0008011750957768484,
            0.0020113343949879113,
            0.004152457620959318,
        ],
    },
    {
        "grid": [24, 24],
        "iterations": 625,
        "converged": True,
        "final_change": 1.0299366825117229e-05,
        "predicted_qav": 0.07888585433050331,
        "literature_qav": 0.072,
        "qav_relative_error": 0.09563686570143495,
        "velocity_metrics": {
            "relative_rms": 1.2973287076686175,
            "relative_l1": 1.4558592163053783,
            "sign_agreement": 0.4,
        },
        "wall_mass_balance_relative_error": 1.5240787796522437e-16,
        "maximum_phi_clipped_weight_fraction": 0.0024081694616501876,
        "maximum_psi_clipped_weight_fraction": 0.004957436431617976,
        "table_velocity": [
            0.003558702245332935,
            0.0023641564964287007,
            0.0014250942410337542,
            0.0006881894511479978,
            0.00012813867599237625,
            -0.00024311044353719075,
            -0.00032870356385294596,
            0.00007266449491364512,
            0.0012259803339745023,
            0.003466449234811846,
        ],
    },
)

STAGE45_GRID = (32, 32)
STAGE45_SEQUENCE_GRIDS = (20, 24, 32)
STAGE45_RULE = STAGE44_RULE
STAGE45_KNUDSEN = STAGE44_KNUDSEN
STAGE45_RATIO = STAGE44_RATIO
STAGE45_MAX_ITERATIONS = STAGE44_MAX_ITERATIONS
STAGE45_TOLERANCE = STAGE44_TOLERANCE
STAGE45_SOURCE_RELAXATION = STAGE44_SOURCE_RELAXATION


def validate_stage45_design(
    grid: tuple[int, int],
    rule: tuple[int, int],
    kn0: float,
    cold_hot_ratio: float,
    max_iterations: int,
    tolerance: float,
    source_relaxation: float,
) -> None:
    if grid != STAGE45_GRID:
        raise ValueError("Stage 45 is fixed to one 32x32 confirmation grid")
    if rule != STAGE45_RULE:
        raise ValueError("Stage 45 retains the fixed 32x96 mapped-polar rule")
    if kn0 != STAGE45_KNUDSEN:
        raise ValueError("Stage 45 remains fixed at Kn0=0.1")
    if cold_hot_ratio != STAGE45_RATIO:
        raise ValueError("Stage 45 remains fixed at Tcold/Thot=0.1")
    if max_iterations != STAGE45_MAX_ITERATIONS:
        raise ValueError("Stage 45 retains the 3000-iteration horizon")
    if tolerance != STAGE45_TOLERANCE:
        raise ValueError("Stage 45 retains tolerance=2e-5")
    if source_relaxation != STAGE45_SOURCE_RELAXATION:
        raise ValueError("Stage 45 does not tune source relaxation")


def build_stage45_config() -> LinearSidewallConfig:
    return LinearSidewallConfig(
        nx=STAGE45_GRID[0],
        ny=STAGE45_GRID[1],
        kn0=STAGE45_KNUDSEN,
        cold_hot_ratio=STAGE45_RATIO,
        viscosity_exponent=0.5,
        prandtl=STAGE41_PRANDTL,
        max_steps=STAGE45_MAX_ITERATIONS,
        cfl=0.2,
        tolerance=STAGE45_TOLERANCE,
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


def stage45_decision(
    retained_24: dict[str, object],
    new_32: dict[str, object],
) -> str:
    if not bool(new_32["converged"]) or not _stable(new_32):
        return "stage45_32x32_blocker"

    q_improves = float(new_32["qav_relative_error"]) <= float(
        retained_24["qav_relative_error"]
    )
    v_improves = float(new_32["velocity_metrics"]["relative_rms"]) <= float(
        retained_24["velocity_metrics"]["relative_rms"]
    )
    q_change = abs(
        float(new_32["predicted_qav"]) - float(retained_24["predicted_qav"])
    ) / max(abs(float(new_32["predicted_qav"])), 1.0e-14)
    profile_change = float(
        np.linalg.norm(
            np.asarray(new_32["table_velocity"])
            - np.asarray(retained_24["table_velocity"])
        )
        / max(float(np.linalg.norm(np.asarray(new_32["table_velocity"]))), 1.0e-14)
    )

    if q_improves and v_improves:
        if q_change <= 0.03 and profile_change <= 0.10:
            return "projected_polar_32x32_converging_stage46_cross_kn_extension"
        return (
            "projected_polar_32x32_improving_not_converged_"
            "stage46_40x40_confirmation"
        )
    if q_improves and not v_improves:
        return "projected_polar_32x32_heat_flux_only_improves_stage46_wall_observable_audit"
    if v_improves and not q_improves:
        return "projected_polar_32x32_velocity_only_improves_stage46_heat_flux_definition_audit"
    return "projected_polar_32x32_nonmonotonic_stage46_space_velocity_coupling_audit"


def _serializable_case(result: dict[str, object]) -> dict[str, object]:
    return {
        "grid": list(STAGE45_GRID),
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


def run_stage45(
    output_dir: str | Path,
    *,
    grid: tuple[int, int] = STAGE45_GRID,
    rule: tuple[int, int] = STAGE45_RULE,
    kn0: float = STAGE45_KNUDSEN,
    cold_hot_ratio: float = STAGE45_RATIO,
    max_iterations: int = STAGE45_MAX_ITERATIONS,
    tolerance: float = STAGE45_TOLERANCE,
    source_relaxation: float = STAGE45_SOURCE_RELAXATION,
) -> dict[str, object]:
    validate_stage45_design(
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
    cfg = build_stage45_config()
    raw = solve_stage42_pilot(cfg, quadrature, source_relaxation)
    case = _serializable_case(raw)
    retained_24 = STAGE44_RETAINED_CASES[-1]

    q_change = abs(
        float(case["predicted_qav"]) - float(retained_24["predicted_qav"])
    ) / max(abs(float(case["predicted_qav"])), 1.0e-14)
    profile_change = float(
        np.linalg.norm(
            np.asarray(case["table_velocity"])
            - np.asarray(retained_24["table_velocity"])
        )
        / max(float(np.linalg.norm(np.asarray(case["table_velocity"]))), 1.0e-14)
    )
    combined_q = np.asarray(
        [
            STAGE44_RETAINED_CASES[0]["predicted_qav"],
            STAGE44_RETAINED_CASES[1]["predicted_qav"],
            case["predicted_qav"],
        ],
        dtype=np.float64,
    )
    combined_v = np.asarray(
        [
            STAGE44_RETAINED_CASES[0]["table_velocity"],
            STAGE44_RETAINED_CASES[1]["table_velocity"],
            case["table_velocity"],
        ],
        dtype=np.float64,
    )
    extrapolation = linear_h_extrapolation(
        np.asarray(STAGE45_SEQUENCE_GRIDS, dtype=np.float64),
        combined_q,
        combined_v,
        float(case["literature_qav"]),
        np.asarray(TABLE3_UY_RATIO_0P1[STAGE45_KNUDSEN]),
    )
    decision = stage45_decision(retained_24, raw)
    summary = {
        "stage": 45,
        "description": (
            "Frozen-physics 32x32 confirmation of the improving but unresolved "
            "Stage 44 projected phi/psi mapped-polar spatial sequence"
        ),
        "retained_stage44_endpoint": STAGE44_COMPLETED_ENDPOINT,
        "retained_stage44_cases": list(STAGE44_RETAINED_CASES),
        "configuration": {
            "kn0": kn0,
            "cold_hot_ratio": cold_hot_ratio,
            "grid": list(grid),
            "combined_sequence_grids": list(STAGE45_SEQUENCE_GRIDS),
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
            "qav_change_24x24_to_32x32": q_change,
            "profile_change_24x24_to_32x32": profile_change,
            "qav_error_improves": (
                float(case["qav_relative_error"])
                <= float(retained_24["qav_relative_error"])
            ),
            "velocity_rms_improves": (
                float(case["velocity_metrics"]["relative_rms"])
                <= float(retained_24["velocity_metrics"]["relative_rms"])
            ),
        },
        "linear_h_extrapolation": extrapolation,
        "decision": decision,
        "interpretation_guard": (
            "Stage 45 changes only physical-grid resolution. Knudsen number, "
            "temperatures, projected Shakhov model, Prandtl number, viscosity law, "
            "relaxation mapping, mapped-polar velocity rule, source relaxation, "
            "positivity floor, transport order, stopping criteria and observables "
            "remain frozen. Extrapolation remains diagnostic and is not validation."
        ),
        "scientific_conclusion": (
            "A cross-Knudsen extension is permitted only if the 32x32 result remains "
            "stable, both benchmark errors do not worsen, and the 24x24-to-32x32 "
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
        description="Run Stage 45 projected-polar 32x32 confirmation"
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/stage45_projected_polar_32x32_confirmation",
    )
    args = parser.parse_args()
    print(json.dumps(run_stage45(args.output_dir), indent=2))


if __name__ == "__main__":
    main()
