from __future__ import annotations

from pathlib import Path
import argparse
import json
import numpy as np

from .linear_sidewall_validation import LinearSidewallConfig
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
from .stage49_projected_polar_64x64_confirmation import (
    STAGE49_GRID,
    STAGE49_MAX_ITERATIONS,
    STAGE49_RATIO,
    STAGE49_RULE,
    STAGE49_SOURCE_RELAXATION,
    STAGE49_TOLERANCE,
)


STAGE49_COMPLETED_ENDPOINT = {
    "workflow_run_id": 30767671512,
    "workflow_job_id": 91549183689,
    "workflow_conclusion": "success",
    "tests_passed": 57,
    "tests_failed": 0,
    "test_duration_seconds": 0.25,
    "artifact_id": 8841966863,
    "artifact_size_bytes": 136455,
    "artifact_sha256": "1955c345571e08dd338c324aef9febf2098ddc728395fd4ab08b77420e91023c",
    "source_head_sha": "2bd8ae07fd8aaf007e724bba1fc254ca5cd8166b",
    "decision": "projected_polar_64x64_converging_stage50_cross_kn_extension",
}

STAGE49_RETAINED_0P1_CASE = {
    "kn0": 0.1,
    "grid": [64, 64],
    "iterations": 1550,
    "converged": True,
    "final_change": 1.673947904379247e-05,
    "predicted_qav": 0.07583282640214684,
    "literature_qav": 0.072,
    "qav_relative_error": 0.05323370002981724,
    "velocity_metrics": {
        "relative_rms": 0.34777209032506057,
        "relative_l1": 0.36774617638417995,
        "sign_agreement": 0.8,
    },
    "wall_mass_balance_relative_error": 2.051199878905107e-16,
    "minimum_phi": 1.0e-30,
    "minimum_psi": 1.0e-30,
    "maximum_phi_clipped_weight_fraction": 0.003122144949217257,
    "maximum_psi_clipped_weight_fraction": 0.005236378288354882,
    "finite": True,
    "work_proxy": 19503513600,
    "table_velocity": [
        0.002335752789234165,
        0.0011821028253378354,
        0.00012299908472434076,
        -0.0007505126886615177,
        -0.0014556823929926354,
        -0.001979819462718378,
        -0.002233084664853311,
        -0.0020386338039931505,
        -0.001096805758787821,
        0.0010910284516402902,
    ],
}

STAGE50_GRID = STAGE49_GRID
STAGE50_RULE = STAGE49_RULE
STAGE50_KNUDSEN_NUMBERS = (1.0, 10.0)
STAGE50_RATIO = STAGE49_RATIO
STAGE50_MAX_ITERATIONS = STAGE49_MAX_ITERATIONS
STAGE50_TOLERANCE = STAGE49_TOLERANCE
STAGE50_SOURCE_RELAXATION = STAGE49_SOURCE_RELAXATION
STAGE50_QAV_RELATIVE_ERROR_SCREEN = 0.10
STAGE50_VELOCITY_RELATIVE_RMS_SCREEN = 0.50
STAGE50_SIGN_AGREEMENT_SCREEN = 0.80


def validate_stage50_design(
    grid: tuple[int, int],
    rule: tuple[int, int],
    knudsen_numbers: tuple[float, ...],
    cold_hot_ratio: float,
    max_iterations: int,
    tolerance: float,
    source_relaxation: float,
) -> None:
    if grid != STAGE50_GRID:
        raise ValueError("Stage 50 retains the converged-screen 64x64 physical grid")
    if rule != STAGE50_RULE:
        raise ValueError("Stage 50 retains the fixed 32x96 mapped-polar rule")
    if knudsen_numbers != STAGE50_KNUDSEN_NUMBERS:
        raise ValueError("Stage 50 is fixed to the preregistered Kn0=1 and 10 extension")
    if cold_hot_ratio != STAGE50_RATIO:
        raise ValueError("Stage 50 retains Tcold/Thot=0.1")
    if max_iterations != STAGE50_MAX_ITERATIONS:
        raise ValueError("Stage 50 retains the 3000-iteration horizon")
    if tolerance != STAGE50_TOLERANCE:
        raise ValueError("Stage 50 retains tolerance=2e-5")
    if source_relaxation != STAGE50_SOURCE_RELAXATION:
        raise ValueError("Stage 50 does not tune source relaxation")


def build_stage50_config(kn0: float) -> LinearSidewallConfig:
    if kn0 not in STAGE50_KNUDSEN_NUMBERS:
        raise ValueError("Stage 50 configuration is defined only for Kn0=1 and 10")
    return LinearSidewallConfig(
        nx=STAGE50_GRID[0],
        ny=STAGE50_GRID[1],
        kn0=kn0,
        cold_hot_ratio=STAGE50_RATIO,
        viscosity_exponent=0.5,
        prandtl=STAGE41_PRANDTL,
        max_steps=STAGE50_MAX_ITERATIONS,
        cfl=0.2,
        tolerance=STAGE50_TOLERANCE,
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


def stage50_decision(cases: list[dict[str, object]]) -> str:
    if len(cases) != 3 or [float(case["kn0"]) for case in cases] != [0.1, 1.0, 10.0]:
        raise ValueError("Stage 50 decision requires the exact Kn0=0.1, 1, 10 sequence")
    if any(not _stable(case) for case in cases):
        return (
            "stage50_cross_kn_numerical_blocker_"
            "stage51_projected_operator_stability_audit"
        )
    if any(not bool(case["converged"]) for case in cases):
        return (
            "stage50_cross_kn_stable_nonconverged_"
            "stage51_fixed_point_convergence_audit"
        )

    heat_flux_consistent = all(
        float(case["qav_relative_error"]) <= STAGE50_QAV_RELATIVE_ERROR_SCREEN
        for case in cases
    )
    velocity_consistent = all(
        float(case["velocity_metrics"]["relative_rms"])
        <= STAGE50_VELOCITY_RELATIVE_RMS_SCREEN
        and float(case["velocity_metrics"]["sign_agreement"])
        >= STAGE50_SIGN_AGREEMENT_SCREEN
        for case in cases
    )
    if heat_flux_consistent and velocity_consistent:
        return (
            "projected_polar_cross_kn_consistent_"
            "stage51_velocity_resolution_confirmation"
        )
    if heat_flux_consistent:
        return (
            "projected_polar_cross_kn_heat_flux_consistent_velocity_unresolved_"
            "stage51_wall_profile_audit"
        )
    if velocity_consistent:
        return (
            "projected_polar_cross_kn_velocity_consistent_heat_flux_unresolved_"
            "stage51_heat_flux_definition_audit"
        )
    return (
        "projected_polar_cross_kn_mixed_or_negative_"
        "stage51_space_velocity_coupling_audit"
    )


def _serializable_case(kn0: float, result: dict[str, object]) -> dict[str, object]:
    return {
        "kn0": float(kn0),
        "grid": list(STAGE50_GRID),
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


def _save_case_arrays(
    output_dir: Path,
    kn0: float,
    raw: dict[str, object],
) -> None:
    arrays: dict[str, np.ndarray] = {}
    for name in (
        "T",
        "rho",
        "u",
        "v",
        "qx",
        "qy",
        "left_wall_velocity",
        "table_velocity",
        "bottom_heat_flux",
        "residual_history",
    ):
        if name in raw:
            arrays[name] = np.asarray(raw[name])
    tag = str(int(kn0)) if float(kn0).is_integer() else str(kn0).replace(".", "p")
    np.savez_compressed(output_dir / f"kn{tag}_fields_and_profiles.npz", **arrays)


def run_stage50(
    output_dir: str | Path,
    *,
    grid: tuple[int, int] = STAGE50_GRID,
    rule: tuple[int, int] = STAGE50_RULE,
    knudsen_numbers: tuple[float, ...] = STAGE50_KNUDSEN_NUMBERS,
    cold_hot_ratio: float = STAGE50_RATIO,
    max_iterations: int = STAGE50_MAX_ITERATIONS,
    tolerance: float = STAGE50_TOLERANCE,
    source_relaxation: float = STAGE50_SOURCE_RELAXATION,
) -> dict[str, object]:
    validate_stage50_design(
        grid,
        rule,
        knudsen_numbers,
        cold_hot_ratio,
        max_iterations,
        tolerance,
        source_relaxation,
    )
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    quadrature = mapped_polar_quadrature(*rule)

    new_cases: list[dict[str, object]] = []
    for kn0 in knudsen_numbers:
        cfg = build_stage50_config(kn0)
        raw = solve_stage42_pilot(cfg, quadrature, source_relaxation)
        _save_case_arrays(output_dir, kn0, raw)
        new_cases.append(_serializable_case(kn0, raw))

    all_cases = [STAGE49_RETAINED_0P1_CASE, *new_cases]
    decision = stage50_decision(all_cases)
    summary = {
        "stage": 50,
        "description": (
            "Frozen-operator 64x64 projected-polar Shakhov extension from the "
            "converged-screen Kn0=0.1 endpoint to Kn0=1 and 10"
        ),
        "retained_stage49_endpoint": STAGE49_COMPLETED_ENDPOINT,
        "retained_kn0_0p1_case": STAGE49_RETAINED_0P1_CASE,
        "configuration": {
            "knudsen_numbers": list(knudsen_numbers),
            "cold_hot_ratio": cold_hot_ratio,
            "grid": list(grid),
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
            "qav_relative_error_screen": STAGE50_QAV_RELATIVE_ERROR_SCREEN,
            "velocity_relative_rms_screen": STAGE50_VELOCITY_RELATIVE_RMS_SCREEN,
            "sign_agreement_screen": STAGE50_SIGN_AGREEMENT_SCREEN,
            "physical_parameter_retuning": False,
            "velocity_quadrature_retuning": False,
            "transport_retuning": False,
        },
        "new_cases": new_cases,
        "cross_kn_metrics": {
            "all_converged": all(bool(case["converged"]) for case in all_cases),
            "all_stable": all(_stable(case) for case in all_cases),
            "maximum_qav_relative_error": max(
                float(case["qav_relative_error"]) for case in all_cases
            ),
            "maximum_velocity_relative_rms": max(
                float(case["velocity_metrics"]["relative_rms"])
                for case in all_cases
            ),
            "minimum_sign_agreement": min(
                float(case["velocity_metrics"]["sign_agreement"])
                for case in all_cases
            ),
            "maximum_wall_mass_balance_relative_error": max(
                float(case["wall_mass_balance_relative_error"])
                for case in all_cases
            ),
        },
        "decision": decision,
        "interpretation_guard": (
            "Stage 50 changes only the benchmark Knudsen number from the retained "
            "Kn0=0.1 endpoint to the published Kn0=1 and 10 cases. The 64x64 grid, "
            "32x96 mapped-polar rule, projected Shakhov model, Prandtl number, "
            "viscosity law, relaxation mapping, source relaxation, positivity floor, "
            "first-order transport, wall treatment, stopping criteria and observables "
            "remain frozen. The preregistered screens assess consistency and do not "
            "constitute external validation."
        ),
        "scientific_conclusion": (
            "Every converged, nonconverged, mixed or unstable endpoint is retained. "
            "No failed physical parameter, collision parameter, transport setting or "
            "velocity quadrature is retuned after observing the results."
        ),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Stage 50 projected-polar 64x64 cross-Knudsen extension"
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/stage50_projected_polar_cross_kn_extension",
    )
    args = parser.parse_args()
    print(json.dumps(run_stage50(args.output_dir), indent=2))


if __name__ == "__main__":
    main()
