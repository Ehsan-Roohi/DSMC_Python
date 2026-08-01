from __future__ import annotations

from pathlib import Path
import argparse
import json
import math
import numpy as np

from .linear_sidewall_validation import (
    LinearSidewallConfig,
    TABLE3_UY_RATIO_0P1,
)
from .stage32_near_continuum_observable_audit import observable_metrics
from .stage37_low_kn_transport_audit import (
    STAGE37_CFL,
    STAGE37_GRID,
    STAGE37_KNUDSEN,
    STAGE37_OBSERVABLE,
    STAGE37_QUADRATURE,
    STAGE37_RATIO,
    transport_case_metrics,
)
from .stage38_transport_collision_interaction_audit import solve_strang_reduced_case
from .velocity_quadrature_audit import spherical_product


STAGE39_GRID = STAGE37_GRID
STAGE39_CFL = STAGE37_CFL
STAGE39_MODELS = (
    ("shakhov_pr_2_over_3", 2.0 / 3.0),
    ("bgk_pr_1", 1.0),
)
STAGE39_COORDINATE_VARIANTS = (
    "direct",
    "reverse_y",
    "flip_tangential_sign",
    "reverse_y_and_flip_sign",
)

# Exact completed Stage 38 endpoint. Stage 39 reruns the canonical first-order
# Strang/Shakhov arm and verifies deterministic reproduction before interpreting
# any collision-model or benchmark-coordinate sensitivity.
STAGE38_COMPLETED_ENDPOINT = {
    "workflow_run_id": 30714512141,
    "workflow_job_id": 91407706438,
    "workflow_conclusion": "success",
    "tests_passed": 54,
    "tests_failed": 0,
    "artifact_id": 8823968646,
    "artifact_size_bytes": 15586,
    "artifact_sha256": "b9d17b291c2cefe5ead6b0953de2547296ec617ed2855c825d8c15610b06159d",
    "first_order_strang_explicit": {
        "iterations": 5500,
        "converged": True,
        "final_change": 1.3511402530752559e-05,
        "predicted_qav": 0.0726173364328985,
        "literature_qav": 0.072,
        "qav_relative_error": 0.008574117123590345,
        "velocity_relative_rms": 1.0091453484689894,
        "velocity_sign_agreement": 0.9,
        "velocity_relative_l1": 0.7591131331922547,
        "wall_mass_balance_relative_error": 2.3230252114543756e-16,
    },
    "muscl_strang_explicit": {
        "iterations": 5900,
        "converged": True,
        "predicted_qav": 0.04329230501615015,
        "qav_relative_error": 0.39871798588680346,
        "velocity_relative_rms": 2.7865958679189817,
        "velocity_sign_agreement": 1.0,
    },
    "decision": "no_coupling_rescue_stage39_collision_model_or_benchmark_audit",
}


def validate_stage39_design(
    grid: tuple[int, int],
    cfl: float,
    models: tuple[tuple[str, float], ...],
    max_steps: int,
    tolerance: float,
) -> None:
    if grid != STAGE39_GRID:
        raise ValueError("Stage 39 is fixed to the Stage 38 24x24 endpoint")
    if cfl != STAGE39_CFL:
        raise ValueError("Stage 39 retains CFL=0.20 without retuning")
    if models != STAGE39_MODELS:
        raise ValueError("Stage 39 compares only canonical Shakhov and BGK limits")
    if max_steps <= 0 or tolerance <= 0.0:
        raise ValueError("max_steps and tolerance must be positive")


def profile_coordinate_variants(profile: np.ndarray) -> dict[str, np.ndarray]:
    """Return preregistered Table-3 coordinate/sign interpretations.

    These transformations are reported as a benchmark-convention audit only.
    They never replace the direct observable silently and are not used to tune
    the physical solution.
    """
    direct = np.asarray(profile, dtype=np.float64)
    if direct.ndim != 1:
        raise ValueError("profile must be one-dimensional")
    return {
        "direct": direct.copy(),
        "reverse_y": direct[::-1].copy(),
        "flip_tangential_sign": (-direct).copy(),
        "reverse_y_and_flip_sign": (-direct[::-1]).copy(),
    }


def coordinate_variant_metrics(profile: np.ndarray) -> dict[str, dict[str, float]]:
    reference = TABLE3_UY_RATIO_0P1[STAGE37_KNUDSEN]
    return {
        name: observable_metrics(candidate, reference)
        for name, candidate in profile_coordinate_variants(profile).items()
    }


def _ratio(numerator: float, denominator: float) -> float:
    return float(numerator) / max(float(denominator), 1.0e-14)


def stage39_decision(
    shakhov: dict[str, object],
    bgk: dict[str, object],
    reproduction: dict[str, object],
) -> str:
    if not bool(reproduction["within_tolerance"]):
        return "stage39_reproduction_mismatch_blocker"
    if not bool(shakhov["converged"]) or not bool(bgk["converged"]):
        return "stage39_nonconvergence_stage40_numerical_stability_audit"

    shakhov_variants = shakhov["coordinate_variant_metrics"]
    bgk_variants = bgk["coordinate_variant_metrics"]
    assert isinstance(shakhov_variants, dict)
    assert isinstance(bgk_variants, dict)
    direct_s = shakhov_variants["direct"]
    direct_b = bgk_variants["direct"]

    for variant in STAGE39_COORDINATE_VARIANTS[1:]:
        candidate_s = shakhov_variants[variant]
        candidate_b = bgk_variants[variant]
        if (
            _ratio(candidate_s["relative_rms"], direct_s["relative_rms"]) <= 0.50
            and _ratio(candidate_b["relative_rms"], direct_b["relative_rms"]) <= 0.50
            and float(candidate_s["sign_agreement"]) >= 0.90
            and float(candidate_b["sign_agreement"]) >= 0.90
        ):
            return "benchmark_coordinate_convention_flag_stage40_source_table_audit"

    q_ratio = _ratio(bgk["qav_relative_error"], shakhov["qav_relative_error"])
    v_ratio = _ratio(
        bgk["velocity_metrics"]["relative_rms"],
        shakhov["velocity_metrics"]["relative_rms"],
    )
    sign_ok = float(bgk["velocity_metrics"]["sign_agreement"]) >= float(
        shakhov["velocity_metrics"]["sign_agreement"]
    )
    if q_ratio <= 0.80 and v_ratio <= 0.80 and sign_ok:
        return "canonical_bgk_sensitivity_stage40_independent_collision_model_audit"
    if q_ratio <= 0.80 or q_ratio >= 1.25 or v_ratio <= 0.80 or v_ratio >= 1.25:
        return "mixed_collision_model_sensitivity_stage40_external_benchmark_audit"
    return "collision_model_and_simple_conventions_do_not_explain_stage40_external_reference_audit"


def run_stage39(
    output_dir: str | Path,
    *,
    grid: tuple[int, int] = STAGE39_GRID,
    cfl: float = STAGE39_CFL,
    models: tuple[tuple[str, float], ...] = STAGE39_MODELS,
    max_steps: int = 16000,
    tolerance: float = 2.0e-5,
) -> dict[str, object]:
    validate_stage39_design(grid, cfl, models, max_steps, tolerance)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    quadrature = spherical_product(16, 12, 24, 5.0, STAGE37_QUADRATURE)

    rows: list[dict[str, object]] = []
    arrays: dict[str, np.ndarray] = {}
    for model_name, prandtl in models:
        cfg = LinearSidewallConfig(
            nx=grid[0],
            ny=grid[1],
            nv=19,
            velocity_extent=5.0,
            kn0=STAGE37_KNUDSEN,
            cold_hot_ratio=STAGE37_RATIO,
            viscosity_exponent=0.5,
            prandtl=prandtl,
            max_steps=max_steps,
            cfl=cfl,
            tolerance=tolerance,
            check_interval=100,
            minimum_steps=1800,
        )
        result = solve_strang_reduced_case(
            cfg,
            quadrature,
            transport_order="first_order_upwind",
        )
        row, profiles = transport_case_metrics(
            result,
            cfg,
            f"first_order_strang_{model_name}",
            quadrature,
        )
        selected_profile = np.asarray(profiles[STAGE37_OBSERVABLE], dtype=np.float64)
        row["collision_model"] = "Shakhov" if prandtl != 1.0 else "BGK"
        row["prandtl"] = prandtl
        row["coordinate_variant_metrics"] = coordinate_variant_metrics(selected_profile)
        rows.append(row)

        short = "shakhov" if prandtl != 1.0 else "bgk"
        for field in (
            "T",
            "rho",
            "u",
            "v",
            "qx",
            "qy",
            "bottom_heat_flux",
            "residual_history",
        ):
            arrays[f"{field}_{short}"] = np.asarray(result[field], dtype=np.float64)
        for name, profile in profiles.items():
            arrays[f"table_velocity_{name}_{short}"] = np.asarray(
                profile, dtype=np.float64
            )
        for name, profile in profile_coordinate_variants(selected_profile).items():
            arrays[f"table_velocity_variant_{name}_{short}"] = profile

    shakhov, bgk = rows
    retained = STAGE38_COMPLETED_ENDPOINT["first_order_strang_explicit"]
    reproduction = {
        "qav_absolute_difference": abs(
            float(shakhov["predicted_qav"]) - float(retained["predicted_qav"])
        ),
        "qav_error_absolute_difference": abs(
            float(shakhov["qav_relative_error"])
            - float(retained["qav_relative_error"])
        ),
        "velocity_rms_absolute_difference": abs(
            float(shakhov["velocity_metrics"]["relative_rms"])
            - float(retained["velocity_relative_rms"])
        ),
        "iterations_difference": int(shakhov["iterations"])
        - int(retained["iterations"]),
    }
    reproduction["within_tolerance"] = bool(
        reproduction["qav_absolute_difference"] <= 1.0e-12
        and reproduction["qav_error_absolute_difference"] <= 1.0e-12
        and reproduction["velocity_rms_absolute_difference"] <= 1.0e-12
        and reproduction["iterations_difference"] == 0
    )

    comparison = {
        "bgk_to_shakhov_qav_error_ratio": _ratio(
            bgk["qav_relative_error"], shakhov["qav_relative_error"]
        ),
        "bgk_to_shakhov_velocity_rms_ratio": _ratio(
            bgk["velocity_metrics"]["relative_rms"],
            shakhov["velocity_metrics"]["relative_rms"],
        ),
        "bgk_minus_shakhov_sign_agreement": (
            float(bgk["velocity_metrics"]["sign_agreement"])
            - float(shakhov["velocity_metrics"]["sign_agreement"])
        ),
        "bgk_minus_shakhov_qav": (
            float(bgk["predicted_qav"]) - float(shakhov["predicted_qav"])
        ),
    }
    decision = stage39_decision(shakhov, bgk, reproduction)
    summary = {
        "stage": 39,
        "description": (
            "Frozen-numerics canonical Shakhov-versus-BGK collision-model audit "
            "with explicit Table-3 coordinate/sign convention checks"
        ),
        "configuration": {
            "kn0": STAGE37_KNUDSEN,
            "cold_hot_ratio": STAGE37_RATIO,
            "grid": list(grid),
            "quadrature": STAGE37_QUADRATURE,
            "velocity_point_count": quadrature.point_count,
            "relaxation_mapping": "sqrt(2)*Kn0/sqrt(pi) in c0 coordinates",
            "transport": "first_order_upwind",
            "coupling": "strang_explicit_half_collision",
            "collision_models": [
                {"name": name, "prandtl": prandtl} for name, prandtl in models
            ],
            "wall_observable": STAGE37_OBSERVABLE,
            "coordinate_variants": list(STAGE39_COORDINATE_VARIANTS),
            "cfl": cfl,
            "max_steps": max_steps,
            "tolerance": tolerance,
            "physical_parameter_retuning": False,
        },
        "retained_stage38_endpoint": STAGE38_COMPLETED_ENDPOINT,
        "stage38_shakhov_reproduction": reproduction,
        "rows": rows,
        "comparison": comparison,
        "decision": decision,
        "interpretation_guard": (
            "Knudsen number, temperatures, corrected relaxation mapping, viscosity "
            "law, velocity quadrature, grid, first-order transport, symmetric "
            "coupling, walls, normalization, CFL, stopping rule and positivity floor "
            "are frozen. The only kinetic-model change is the canonical BGK limit "
            "Pr=1 versus Shakhov Pr=2/3. Coordinate and sign variants are all reported "
            "as post-processing diagnostics and never replace the direct Table-3 "
            "comparison silently. Negative and mixed outcomes are retained."
        ),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    np.savez_compressed(output_dir / "fields_and_profiles.npz", **arrays)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Stage 39 collision-model and benchmark-convention audit"
    )
    parser.add_argument(
        "--output-dir", default="outputs/stage39_collision_model_benchmark_audit"
    )
    parser.add_argument("--max-steps", type=int, default=16000)
    parser.add_argument("--tolerance", type=float, default=2.0e-5)
    args = parser.parse_args()
    print(
        json.dumps(
            run_stage39(
                args.output_dir,
                max_steps=args.max_steps,
                tolerance=args.tolerance,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
