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
from .stage50_projected_polar_cross_kn_extension import (
    STAGE50_GRID,
    STAGE50_KNUDSEN_NUMBERS,
    STAGE50_MAX_ITERATIONS,
    STAGE50_RATIO,
    STAGE50_SOURCE_RELAXATION,
    STAGE50_TOLERANCE,
    build_stage50_config,
)

STAGE51_COMPLETED_ENDPOINT = STAGE51_ENDPOINT = {
    "workflow_run_id": 30781991799,
    "workflow_job_id": 91588361434,
    "workflow_conclusion": "success",
    "tests_passed": 71,
    "tests_failed": 0,
    "test_duration_seconds": 0.45,
    "artifact_id": 8845163403,
    "artifact_size_bytes": 6247,
    "artifact_sha256": "95cbcb84bb57386a8d88d191566af4a2138ca62f49c3304737fd45aa7b89f974",
    "source_head_sha": "a27646f23858aa57e9e241506d32565398220347",
    "summary_sha256": "8a8b239c9e95f1215eb33fba34652065180ab5e8bd9119baeb1060f3365f94a1",
    "profiles_sha256": "22c68653b6aee23da52ca29253438149547912f434edf0fc57a2a36404ba0f4f",
    "decision": "heat_flux_definition_and_location_do_not_explain_cross_kn_error_stage52_velocity_resolution_audit",
}
STAGE50_ARTIFACT = {
    "workflow_run_id": 30776673982,
    "artifact_id": 8843553740,
    "artifact_size_bytes": 264879,
    "artifact_sha256": "51540b50188b5cf3ba95464141d9b18d3ebdd3465c6e4ffc0909a51cdecd97c2",
    "summary_sha256": "f2b71365e4fd7cd94157bdf888c1678a4609423168ba74934988860dd66bcf5d",
    "kn10_sha256": "0d922ddfae58b26cd7e088e1ceadeec3109cd3e60d3e7f50057e4103d30359a0",
    "kn10_fields_sha256": "0d922ddfae58b26cd7e088e1ceadeec3109cd3e60d3e7f50057e4103d30359a0",
}
STAGE52_GRID = STAGE50_GRID
STAGE52_KNUDSEN = 10.0
STAGE52_RATIO = STAGE50_RATIO
STAGE52_BASELINE_RULE = (32, 96)
STAGE52_RULES = (
    ("radial_refined", (40, 96)),
    ("angular_refined", (32, 120)),
    ("coupled_refined", (40, 120)),
)
STAGE52_MAX_ITERATIONS = STAGE50_MAX_ITERATIONS
STAGE52_TOLERANCE = STAGE50_TOLERANCE
STAGE52_SOURCE_RELAXATION = STAGE50_SOURCE_RELAXATION
STAGE52_MIN_ERROR_REDUCTION = 0.10
STAGE52_MAX_VELOCITY_RMS_DEGRADATION = 0.10

STAGE52_BASELINE_CASE = {
    "kn0": 10.0,
    "grid": [64, 64],
    "iterations": 775,
    "converged": True,
    "final_change": 1.50744313059874e-05,
    "predicted_qav": 0.22492827563466516,
    "literature_qav": 0.178,
    "qav_relative_error": 0.26364199794755716,
    "velocity_metrics": {
        "relative_rms": 0.4249110488734276,
        "relative_l1": 0.45068357867877606,
        "sign_agreement": 1.0,
    },
    "wall_mass_balance_relative_error": 1.9744774409718493e-16,
    "minimum_phi": 1.0e-30,
    "minimum_psi": 1.0e-30,
    "maximum_phi_clipped_weight_fraction": 0.012995473412961285,
    "maximum_psi_clipped_weight_fraction": 0.027552143031269263,
    "finite": True,
    "work_proxy": 9751756800,
    "table_velocity": [
        0.0018562878870007674,
        0.0018394734535706495,
        0.001694124064748957,
        0.0014997569706574063,
        0.001284032953647522,
        0.0010605795191220215,
        0.0008388104502277677,
        0.0006295560333769755,
        0.0004462564896183598,
        0.00031152244959190424,
    ],
}


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_stage52_design(
    grid=STAGE52_GRID,
    kn0=STAGE52_KNUDSEN,
    baseline_rule=STAGE52_BASELINE_RULE,
    rules=STAGE52_RULES,
    cold_hot_ratio=STAGE52_RATIO,
    max_iterations=STAGE52_MAX_ITERATIONS,
    tolerance=STAGE52_TOLERANCE,
    source_relaxation=STAGE52_SOURCE_RELAXATION,
) -> None:
    expected = (
        grid == STAGE52_GRID,
        kn0 == STAGE52_KNUDSEN,
        baseline_rule == STAGE52_BASELINE_RULE,
        rules == STAGE52_RULES,
        cold_hot_ratio == STAGE52_RATIO,
        max_iterations == STAGE52_MAX_ITERATIONS,
        tolerance == STAGE52_TOLERANCE,
        source_relaxation == STAGE52_SOURCE_RELAXATION,
    )
    if not all(expected):
        raise ValueError(
            "Stage 52 design is frozen; only preregistered quadrature resolution may change"
        )


def build_stage52_config():
    if STAGE52_KNUDSEN not in STAGE50_KNUDSEN_NUMBERS:
        raise RuntimeError("Kn0=10 is not a retained Stage 50 case")
    return build_stage50_config(STAGE52_KNUDSEN)


def _stable(case: Mapping[str, object]) -> bool:
    return bool(
        case.get("finite")
        and float(case.get("minimum_phi", 0.0)) > 0.0
        and float(case.get("minimum_psi", 0.0)) > 0.0
        and float(case.get("wall_mass_balance_relative_error", math.inf)) < 1.0e-10
    )


def _case(
    name: str,
    rule: tuple[int, int],
    raw: Mapping[str, object],
) -> dict[str, object]:
    return {
        "name": name,
        "kn0": STAGE52_KNUDSEN,
        "grid": list(STAGE52_GRID),
        "polar_rule": {
            "radial_nodes": rule[0],
            "angular_nodes": rule[1],
            "point_count": rule[0] * rule[1],
        },
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
    p0 = np.asarray(baseline["table_velocity"])
    p1 = np.asarray(case["table_velocity"])
    q_reduction = (q0 - q1) / max(q0, 1.0e-300)
    v_change = (v1 - v0) / max(v0, 1.0e-300)
    material = bool(
        _stable(case)
        and case["converged"]
        and q_reduction >= STAGE52_MIN_ERROR_REDUCTION
        and v_change <= STAGE52_MAX_VELOCITY_RMS_DEGRADATION
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
        "stable": _stable(case),
        "materially_improves": material,
    }


def stage52_decision(cases, comparisons) -> str:
    names = [name for name, _ in STAGE52_RULES]
    if [case.get("name") for case in cases] != names:
        raise ValueError("Stage 52 requires the exact three-arm sequence")
    if any(not _stable(case) for case in cases):
        return "stage52_velocity_resolution_numerical_blocker"
    if any(_stable(case) and not case["converged"] for case in cases):
        return (
            "stage52_velocity_resolution_stable_nonconverged_"
            "stage53_fixed_point_audit"
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
        return f"{best}_materially_improves_stage53_cross_kn_confirmation"
    if any(
        comparisons[name]["qav_error_reduction_fraction"] > 0.0
        for name in names
    ):
        return (
            "velocity_resolution_small_or_mixed_effect_"
            "stage53_radial_mapping_tail_audit"
        )
    return (
        "velocity_point_count_does_not_explain_cross_kn_heat_flux_"
        "stage53_projected_collision_moment_audit"
    )


def _validate_inputs(
    stage50_dir: Path,
    stage51_dir: Path,
) -> tuple[dict, dict]:
    p50 = stage50_dir / "summary.json"
    p51 = stage51_dir / "summary.json"
    checks = (
        sha256_file(p50) == STAGE50_ARTIFACT["summary_sha256"],
        sha256_file(stage50_dir / "kn10_fields_and_profiles.npz")
        == STAGE50_ARTIFACT["kn10_sha256"],
        sha256_file(p51) == STAGE51_ENDPOINT["summary_sha256"],
        sha256_file(stage51_dir / "heat_flux_profiles.npz")
        == STAGE51_ENDPOINT["profiles_sha256"],
    )
    if not all(checks):
        raise ValueError("Stage 50/51 artifact checksum mismatch")
    s50 = json.loads(p50.read_text())
    s51 = json.loads(p51.read_text())
    if (
        s50.get("stage") != 50
        or s51.get("stage") != 51
        or s51.get("decision") != STAGE51_ENDPOINT["decision"]
    ):
        raise ValueError("Stage 50/51 provenance or decision mismatch")
    baseline = next(
        case for case in s50["new_cases"] if float(case["kn0"]) == 10.0
    )
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
        math.isclose(
            float(baseline["velocity_metrics"]["relative_rms"]),
            STAGE52_BASELINE_CASE["velocity_metrics"]["relative_rms"],
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
        raise ValueError("Stage 50 Kn10 baseline mismatch")
    return baseline, s51


def run_stage52(
    stage50_artifact_dir,
    stage51_artifact_dir,
    output_dir,
    **design,
):
    validate_stage52_design(**design)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    baseline, _ = _validate_inputs(
        Path(stage50_artifact_dir), Path(stage51_artifact_dir)
    )
    cases = []
    comparisons = {}
    for name, rule in STAGE52_RULES:
        quadrature = mapped_polar_quadrature(*rule)
        raw = solve_stage42_pilot(
            build_stage52_config(), quadrature, STAGE52_SOURCE_RELAXATION
        )
        case = _case(name, rule, raw)
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
        np.savez_compressed(
            out / f"{name}_fields_and_profiles.npz", **arrays
        )
        cases.append(case)
        comparisons[name] = compare_to_baseline(baseline, case)
        del raw, quadrature
        gc.collect()
    summary = {
        "stage": 52,
        "description": (
            "Orthogonal mapped-polar velocity-resolution audit at the worst "
            "Stage 50 cross-Knudsen endpoint"
        ),
        "retained_stage51_endpoint": STAGE51_ENDPOINT,
        "retained_stage50_artifact": STAGE50_ARTIFACT,
        "configuration": {
            "kn0": STAGE52_KNUDSEN,
            "cold_hot_ratio": STAGE52_RATIO,
            "grid": list(STAGE52_GRID),
            "baseline_rule": {
                "radial_nodes": 32,
                "angular_nodes": 96,
                "point_count": 3072,
            },
            "audit_rules": [
                {
                    "name": name,
                    "radial_nodes": rule[0],
                    "angular_nodes": rule[1],
                    "point_count": rule[0] * rule[1],
                }
                for name, rule in STAGE52_RULES
            ],
            "minimum_material_error_reduction": STAGE52_MIN_ERROR_REDUCTION,
            "maximum_velocity_rms_degradation": (
                STAGE52_MAX_VELOCITY_RMS_DEGRADATION
            ),
            "max_iterations": STAGE52_MAX_ITERATIONS,
            "tolerance": STAGE52_TOLERANCE,
            "source_relaxation": STAGE52_SOURCE_RELAXATION,
            "physical_parameter_retuning": False,
            "collision_parameter_retuning": False,
            "transport_retuning": False,
            "wall_model_retuning": False,
            "normalization_retuning": False,
            "radial_mapping_scale_retuning": False,
            "quadrature_resolution_is_preregistered_audit_variable": True,
        },
        "baseline_case": baseline,
        "audit_cases": cases,
        "comparisons_to_baseline": comparisons,
        "decision": stage52_decision(cases, comparisons),
        "interpretation_guard": (
            "Only mapped-polar radial/angular node counts change. Physics, grid, "
            "mapping scale, transport, collision model, walls, normalization and "
            "stopping criteria remain frozen; all outcomes are retained."
        ),
        "scientific_conclusion": (
            "Material improvement requires at least 10% Table-6 error reduction, "
            "no more than 10% Table-3 RMS degradation, no sign loss, convergence "
            "and stability. No retrospective retuning is allowed."
        ),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage50-artifact-dir", required=True)
    parser.add_argument("--stage51-artifact-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            run_stage52(
                args.stage50_artifact_dir,
                args.stage51_artifact_dir,
                args.output_dir,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
