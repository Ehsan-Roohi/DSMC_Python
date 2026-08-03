from __future__ import annotations

from pathlib import Path
import argparse
import gc
import hashlib
import json
import math
from typing import Mapping

import numpy as np

from .stage41_projected_polar_operator_audit import mapped_polar_quadrature
from .stage42_projected_polar_heated_cavity_pilot import solve_stage42_pilot
from .stage52_velocity_resolution_audit import (
    STAGE52_BASELINE_CASE,
    STAGE52_GRID,
    STAGE52_KNUDSEN,
    STAGE52_RATIO,
    STAGE52_SOURCE_RELAXATION,
    build_stage52_config,
)

STAGE52_COMPLETED_ENDPOINT = {
    "workflow_run_id": 30787293128,
    "workflow_job_id": 91603227075,
    "workflow_conclusion": "success",
    "tests_passed": 79,
    "tests_failed": 0,
    "test_duration_seconds": 0.49,
    "artifact_id": 8851319553,
    "artifact_size_bytes": 393526,
    "artifact_sha256": "3c496189886de4b764ff13f7bfd737415de874329b4ecd191c270d90c97fca4b",
    "source_head_sha": "9633e52fa3cbe7c2bfb2932eabe20de9d5d5657b",
    "summary_sha256": "2e707e3220ad4a54c16e861895ad042cc517c1439a928148046f4be5d968ff10",
    "radial_refined_sha256": "425902be29e47697ec32245948a117bae193d7741686b4708e516274c38d1917",
    "angular_refined_sha256": "7fd46b6c815a39ae84ecce780125e580b691f43847b37a51f1b75ca5a5e2f632",
    "coupled_refined_sha256": "267241a2d2aac192aaced3687fc2c448adf0066348739b4d0c20621a322b7556",
    "decision": "velocity_resolution_small_or_mixed_effect_stage53_radial_mapping_tail_audit",
}

STAGE53_GRID = STAGE52_GRID
STAGE53_KNUDSEN = STAGE52_KNUDSEN
STAGE53_RATIO = STAGE52_RATIO
STAGE53_RULE = (32, 96)
STAGE53_BASELINE_SCALE = 1.0
STAGE53_AUDIT_SCALES = (
    ("compressed_tail", 0.5),
    ("expanded_tail", 2.0),
)
STAGE53_SOURCE_RELAXATION = STAGE52_SOURCE_RELAXATION
STAGE53_MIN_ERROR_REDUCTION = 0.10
STAGE53_MAX_VELOCITY_RMS_DEGRADATION = 0.10
STAGE53_MAX_TAIL_MOMENT_ERROR = 1.0e-3
STAGE53_TAIL_TEMPERATURES = (0.1, 1.0)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_stage53_design(
    grid=STAGE53_GRID,
    kn0=STAGE53_KNUDSEN,
    rule=STAGE53_RULE,
    baseline_scale=STAGE53_BASELINE_SCALE,
    audit_scales=STAGE53_AUDIT_SCALES,
    cold_hot_ratio=STAGE53_RATIO,
    source_relaxation=STAGE53_SOURCE_RELAXATION,
    tail_temperatures=STAGE53_TAIL_TEMPERATURES,
) -> None:
    expected = (
        grid == STAGE53_GRID,
        kn0 == STAGE53_KNUDSEN,
        rule == STAGE53_RULE,
        baseline_scale == STAGE53_BASELINE_SCALE,
        audit_scales == STAGE53_AUDIT_SCALES,
        cold_hot_ratio == STAGE53_RATIO,
        source_relaxation == STAGE53_SOURCE_RELAXATION,
        tail_temperatures == STAGE53_TAIL_TEMPERATURES,
    )
    if not all(expected):
        raise ValueError(
            "Stage 53 is frozen; only the two preregistered radial mapping scales may differ"
        )


def build_stage53_config():
    return build_stage52_config()


def maxwellian_tail_moment_audit(quadrature) -> dict[str, object]:
    radius2 = np.asarray(quadrature.radius, dtype=np.float64) ** 2
    weight = np.asarray(quadrature.weight, dtype=np.float64)
    temperatures = []
    maximum_error = 0.0
    for temperature in STAGE53_TAIL_TEMPERATURES:
        raw = np.exp(-radius2 / (2.0 * temperature)) / (
            2.0 * math.pi * temperature
        )
        exact = {
            "mass": 1.0,
            "radial_second": 2.0 * temperature,
            "radial_fourth": 8.0 * temperature**2,
            "radial_sixth": 48.0 * temperature**3,
        }
        computed = {
            "mass": float(np.sum(raw * weight)),
            "radial_second": float(np.sum(radius2 * raw * weight)),
            "radial_fourth": float(np.sum(radius2**2 * raw * weight)),
            "radial_sixth": float(np.sum(radius2**3 * raw * weight)),
        }
        relative_errors = {
            key: abs(computed[key] - exact[key]) / abs(exact[key])
            for key in exact
        }
        maximum_error = max(maximum_error, max(relative_errors.values()))
        temperatures.append(
            {
                "temperature": temperature,
                "computed": computed,
                "exact": exact,
                "relative_errors": relative_errors,
            }
        )
    return {
        "radial_scale": float(quadrature.radial_scale),
        "radial_nodes": int(quadrature.radial_nodes),
        "angular_nodes": int(quadrature.angular_nodes),
        "point_count": int(quadrature.point_count),
        "minimum_radius": float(np.min(quadrature.radius)),
        "maximum_radius": float(np.max(quadrature.radius)),
        "temperatures": temperatures,
        "maximum_relative_error": float(maximum_error),
        "passes_preregistered_tail_closure": bool(
            maximum_error <= STAGE53_MAX_TAIL_MOMENT_ERROR
        ),
    }


def _stable(case: Mapping[str, object]) -> bool:
    return bool(
        case.get("finite")
        and float(case.get("minimum_phi", 0.0)) > 0.0
        and float(case.get("minimum_psi", 0.0)) > 0.0
        and float(case.get("wall_mass_balance_relative_error", math.inf)) < 1.0e-10
    )


def _case(
    name: str,
    scale: float,
    raw: Mapping[str, object],
    tail_audit: Mapping[str, object],
) -> dict[str, object]:
    return {
        "name": name,
        "kn0": STAGE53_KNUDSEN,
        "grid": list(STAGE53_GRID),
        "polar_rule": {
            "radial_nodes": STAGE53_RULE[0],
            "angular_nodes": STAGE53_RULE[1],
            "point_count": STAGE53_RULE[0] * STAGE53_RULE[1],
            "radial_scale": scale,
        },
        "tail_moment_audit": dict(tail_audit),
        "iterations": int(raw["iterations"]),
        "converged": bool(raw["converged"]),
        "final_change": float(raw["final_change"]),
        "predicted_qav": float(raw["predicted_qav"]),
        "literature_qav": float(raw["literature_qav"]),
        "qav_relative_error": float(raw["qav_relative_error"]),
        "velocity_metrics": {
            key: float(value) for key, value in raw["velocity_metrics"].items()
        },
        "wall_mass_balance_relative_error": float(
            raw["wall_mass_balance_relative_error"]
        ),
        "minimum_phi": float(raw["minimum_phi"]),
        "minimum_psi": float(raw["minimum_psi"]),
        "maximum_phi_clipped_weight_fraction": float(
            raw["maximum_phi_clipped_weight_fraction"]
        ),
        "maximum_psi_clipped_weight_fraction": float(
            raw["maximum_psi_clipped_weight_fraction"]
        ),
        "finite": bool(raw["finite"]),
        "work_proxy": int(raw["work_proxy"]),
        "table_velocity": np.asarray(raw["table_velocity"]).tolist(),
    }


def compare_to_baseline(
    baseline: Mapping[str, object],
    case: Mapping[str, object],
) -> dict[str, object]:
    q0 = float(baseline["qav_relative_error"])
    q1 = float(case["qav_relative_error"])
    v0 = float(baseline["velocity_metrics"]["relative_rms"])
    v1 = float(case["velocity_metrics"]["relative_rms"])
    p0 = np.asarray(baseline["table_velocity"], dtype=np.float64)
    p1 = np.asarray(case["table_velocity"], dtype=np.float64)
    q_reduction = (q0 - q1) / max(q0, 1.0e-300)
    v_change = (v1 - v0) / max(v0, 1.0e-300)
    tail_passes = bool(
        case["tail_moment_audit"]["passes_preregistered_tail_closure"]
    )
    material = bool(
        _stable(case)
        and case["converged"]
        and tail_passes
        and q_reduction >= STAGE53_MIN_ERROR_REDUCTION
        and v_change <= STAGE53_MAX_VELOCITY_RMS_DEGRADATION
        and float(case["velocity_metrics"]["sign_agreement"])
        >= float(baseline["velocity_metrics"]["sign_agreement"])
    )
    return {
        "qav_error_reduction_fraction": q_reduction,
        "qav_relative_change": abs(
            float(case["predicted_qav"]) - float(baseline["predicted_qav"])
        )
        / abs(float(baseline["predicted_qav"])),
        "velocity_rms_change_fraction": v_change,
        "table_velocity_profile_change": float(
            np.linalg.norm(p1 - p0) / max(np.linalg.norm(p0), 1.0e-300)
        ),
        "sign_agreement_change": float(
            case["velocity_metrics"]["sign_agreement"]
        )
        - float(baseline["velocity_metrics"]["sign_agreement"]),
        "phi_clipping_change": float(
            case["maximum_phi_clipped_weight_fraction"]
        )
        - float(baseline["maximum_phi_clipped_weight_fraction"]),
        "psi_clipping_change": float(
            case["maximum_psi_clipped_weight_fraction"]
        )
        - float(baseline["maximum_psi_clipped_weight_fraction"]),
        "tail_moment_maximum_relative_error": float(
            case["tail_moment_audit"]["maximum_relative_error"]
        ),
        "tail_moment_closure_passes": tail_passes,
        "stable": _stable(case),
        "materially_improves": material,
    }


def stage53_decision(cases, comparisons) -> str:
    names = [name for name, _ in STAGE53_AUDIT_SCALES]
    if [case.get("name") for case in cases] != names:
        raise ValueError("Stage 53 requires the exact two-arm mapping-scale sequence")
    if any(not _stable(case) for case in cases):
        return "stage53_radial_mapping_tail_numerical_blocker"
    if any(_stable(case) and not case["converged"] for case in cases):
        return (
            "stage53_radial_mapping_tail_stable_nonconverged_"
            "stage54_fixed_point_audit"
        )
    improved = [
        name for name in names if comparisons[name]["materially_improves"]
    ]
    if improved:
        best = min(
            improved,
            key=lambda name: next(
                case for case in cases if case["name"] == name
            )["qav_relative_error"],
        )
        return f"{best}_materially_improves_stage54_cross_kn_confirmation"
    largest_q_change = max(
        comparisons[name]["qav_relative_change"] for name in names
    )
    if largest_q_change >= 0.02:
        return (
            "radial_mapping_changes_solution_without_material_benchmark_"
            "improvement_stage54_projected_collision_moment_audit"
        )
    return (
        "radial_mapping_tail_does_not_explain_cross_kn_heat_flux_"
        "stage54_projected_collision_moment_audit"
    )


def _validate_stage52_artifact(stage52_dir: Path) -> tuple[dict, dict]:
    summary_path = stage52_dir / "summary.json"
    checks = (
        sha256_file(summary_path) == STAGE52_COMPLETED_ENDPOINT["summary_sha256"],
        sha256_file(stage52_dir / "radial_refined_fields_and_profiles.npz")
        == STAGE52_COMPLETED_ENDPOINT["radial_refined_sha256"],
        sha256_file(stage52_dir / "angular_refined_fields_and_profiles.npz")
        == STAGE52_COMPLETED_ENDPOINT["angular_refined_sha256"],
        sha256_file(stage52_dir / "coupled_refined_fields_and_profiles.npz")
        == STAGE52_COMPLETED_ENDPOINT["coupled_refined_sha256"],
    )
    if not all(checks):
        raise ValueError("Stage 52 artifact checksum mismatch")
    summary = json.loads(summary_path.read_text())
    if (
        summary.get("stage") != 52
        or summary.get("decision") != STAGE52_COMPLETED_ENDPOINT["decision"]
    ):
        raise ValueError("Stage 52 provenance or decision mismatch")
    baseline = summary["baseline_case"]
    exact_checks = (
        int(baseline["iterations"]) == STAGE52_BASELINE_CASE["iterations"],
        bool(baseline["converged"]) is True,
        math.isclose(
            float(baseline["predicted_qav"]),
            STAGE52_BASELINE_CASE["predicted_qav"],
            rel_tol=0.0,
            abs_tol=1.0e-15,
        ),
        math.isclose(
            float(baseline["qav_relative_error"]),
            STAGE52_BASELINE_CASE["qav_relative_error"],
            rel_tol=0.0,
            abs_tol=1.0e-15,
        ),
        np.allclose(
            np.asarray(baseline["table_velocity"]),
            np.asarray(STAGE52_BASELINE_CASE["table_velocity"]),
            rtol=0.0,
            atol=1.0e-15,
        ),
    )
    if not all(exact_checks):
        raise ValueError("Stage 52 retained baseline mismatch")
    return summary, baseline


def run_stage53(stage52_artifact_dir, output_dir, **design):
    validate_stage53_design(**design)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    retained_summary, baseline = _validate_stage52_artifact(
        Path(stage52_artifact_dir)
    )

    baseline_quadrature = mapped_polar_quadrature(
        *STAGE53_RULE, radial_scale=STAGE53_BASELINE_SCALE
    )
    baseline_tail_audit = maxwellian_tail_moment_audit(baseline_quadrature)
    del baseline_quadrature

    cases = []
    comparisons = {}
    for name, scale in STAGE53_AUDIT_SCALES:
        quadrature = mapped_polar_quadrature(
            *STAGE53_RULE, radial_scale=scale
        )
        tail_audit = maxwellian_tail_moment_audit(quadrature)
        raw = solve_stage42_pilot(
            build_stage53_config(), quadrature, STAGE53_SOURCE_RELAXATION
        )
        case = _case(name, scale, raw, tail_audit)
        arrays = {
            key: np.asarray(raw[key])
            for key in (
                "T",
                "rho",
                "u",
                "v",
                "qx",
                "qy",
                "table_velocity",
                "bottom_heat_flux",
                "residual_history",
            )
            if key in raw
        }
        np.savez_compressed(out / f"{name}_fields_and_profiles.npz", **arrays)
        cases.append(case)
        comparisons[name] = compare_to_baseline(baseline, case)
        del raw, quadrature
        gc.collect()

    summary = {
        "stage": 53,
        "description": (
            "Preregistered factor-of-two radial mapping tail audit at Kn0=10 "
            "after velocity point-count refinement had only a small or mixed effect"
        ),
        "retained_stage52_endpoint": STAGE52_COMPLETED_ENDPOINT,
        "retained_stage52_decision": retained_summary["decision"],
        "configuration": {
            "kn0": STAGE53_KNUDSEN,
            "cold_hot_ratio": STAGE53_RATIO,
            "grid": list(STAGE53_GRID),
            "radial_nodes": STAGE53_RULE[0],
            "angular_nodes": STAGE53_RULE[1],
            "point_count": STAGE53_RULE[0] * STAGE53_RULE[1],
            "baseline_radial_scale": STAGE53_BASELINE_SCALE,
            "audit_scales": [
                {"name": name, "radial_scale": scale}
                for name, scale in STAGE53_AUDIT_SCALES
            ],
            "tail_temperatures": list(STAGE53_TAIL_TEMPERATURES),
            "maximum_tail_moment_relative_error": (
                STAGE53_MAX_TAIL_MOMENT_ERROR
            ),
            "minimum_material_error_reduction": STAGE53_MIN_ERROR_REDUCTION,
            "maximum_velocity_rms_degradation": (
                STAGE53_MAX_VELOCITY_RMS_DEGRADATION
            ),
            "source_relaxation": STAGE53_SOURCE_RELAXATION,
            "physical_parameter_retuning": False,
            "collision_parameter_retuning": False,
            "transport_retuning": False,
            "wall_model_retuning": False,
            "normalization_retuning": False,
            "velocity_point_count_retuning": False,
            "radial_mapping_scale_is_preregistered_audit_variable": True,
        },
        "baseline_case": baseline,
        "baseline_tail_moment_audit": baseline_tail_audit,
        "audit_cases": cases,
        "comparisons_to_baseline": comparisons,
        "decision": stage53_decision(cases, comparisons),
        "interpretation_guard": (
            "The two factor-of-two mapping scales were fixed before execution and "
            "both outcomes are retained. Physics, grid, velocity point count, "
            "transport, collision model, walls, normalization, relaxation and "
            "stopping criteria remain frozen."
        ),
        "scientific_conclusion": (
            "A mapping scale is materially beneficial only with at least 10% "
            "heat-flux error reduction, no more than 10% velocity-RMS degradation, "
            "no sign loss, stable convergence and preregistered Maxwellian tail-"
            "moment closure. Passing this screen would require cross-Knudsen "
            "confirmation and would not by itself establish external validation."
        ),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage52-artifact-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            run_stage53(args.stage52_artifact_dir, args.output_dir),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
