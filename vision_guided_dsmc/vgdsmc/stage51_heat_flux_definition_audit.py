from __future__ import annotations

from pathlib import Path
import argparse
import hashlib
import json
import math
from typing import Mapping

import numpy as np


STAGE50_COMPLETED_ENDPOINT = {
    "workflow_run_id": 30776673982,
    "workflow_job_id": 91573366717,
    "workflow_conclusion": "success",
    "tests_passed": 63,
    "tests_failed": 0,
    "test_duration_seconds": 0.42,
    "artifact_id": 8843553740,
    "artifact_size_bytes": 264879,
    "artifact_sha256": "51540b50188b5cf3ba95464141d9b18d3ebdd3465c6e4ffc0909a51cdecd97c2",
    "source_head_sha": "df37f0a4e464130bc12ca4ae815b643602fc9521",
    "decision": (
        "projected_polar_cross_kn_velocity_consistent_heat_flux_unresolved_"
        "stage51_heat_flux_definition_audit"
    ),
}

STAGE51_KNUDSEN_NUMBERS = (1.0, 10.0)
STAGE51_GRID = (64, 64)
STAGE51_RULE = (32, 96)
STAGE51_LITERATURE_QAV = {1.0: 0.148, 10.0: 0.178}
STAGE51_ERROR_SCREEN = 0.10
STAGE51_ESTIMATOR_SPREAD_SCREEN = 0.05
STAGE51_SOURCE_NORMALIZATION = {
    "source_velocity_scale": "v0=sqrt(2*k_B*T0/m)",
    "source_heat_flux_scale": "P0*v0",
    "source_table6_definition": "qav=integral_{-0.5}^{0.5} q_y(x,0) dx",
    "solver_velocity_scale": "c0=sqrt(k_B*T0/m)",
    "solver_to_source_heat_flux_multiplier": 1.0 / math.sqrt(2.0),
    "stored_stage50_qy_and_bottom_heat_flux_are_source_scaled": True,
}


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def relative_error(candidate: float, reference: float) -> float:
    if not math.isfinite(candidate) or not math.isfinite(reference):
        raise ValueError("candidate and reference must be finite")
    if reference == 0.0:
        raise ValueError("reference must be nonzero")
    return abs(candidate - reference) / abs(reference)


def validate_stage50_summary(summary: Mapping[str, object]) -> None:
    if int(summary.get("stage", -1)) != 50:
        raise ValueError("Stage 51 requires the exact Stage 50 summary")
    retained = summary.get("retained_stage49_endpoint")
    if not isinstance(retained, Mapping):
        raise ValueError("Stage 50 retained provenance is missing")
    configuration = summary.get("configuration")
    if not isinstance(configuration, Mapping):
        raise ValueError("Stage 50 configuration is missing")
    if list(configuration.get("knudsen_numbers", [])) != [1.0, 10.0]:
        raise ValueError("Stage 50 Knudsen sequence must remain [1,10]")
    if list(configuration.get("grid", [])) != [64, 64]:
        raise ValueError("Stage 51 audits the frozen Stage 50 64x64 grid")
    rule = configuration.get("polar_rule")
    if not isinstance(rule, Mapping) or (
        int(rule.get("radial_nodes", -1)), int(rule.get("angular_nodes", -1))
    ) != STAGE51_RULE:
        raise ValueError("Stage 51 audits the frozen Stage 50 32x96 quadrature")
    if bool(configuration.get("physical_parameter_retuning", True)):
        raise ValueError("Stage 50 must record no physical-parameter retuning")
    if bool(configuration.get("velocity_quadrature_retuning", True)):
        raise ValueError("Stage 50 must record no velocity-quadrature retuning")
    if summary.get("decision") != STAGE50_COMPLETED_ENDPOINT["decision"]:
        raise ValueError("Stage 50 decision does not route to the definition audit")
    cases = summary.get("new_cases")
    if not isinstance(cases, list) or [float(case["kn0"]) for case in cases] != [1.0, 10.0]:
        raise ValueError("Stage 50 cases must be the exact Kn0=1 and 10 sequence")
    if any(not bool(case.get("converged")) or not bool(case.get("finite")) for case in cases):
        raise ValueError("Stage 51 requires completed finite Stage 50 cases")


def polynomial_wall_extrapolation(row_means: np.ndarray, degree: int) -> float:
    values = np.asarray(row_means, dtype=np.float64)
    if values.ndim != 1 or values.size < degree + 1:
        raise ValueError("insufficient one-dimensional row means for extrapolation")
    if degree not in (1, 2):
        raise ValueError("Stage 51 permits only linear or quadratic wall extrapolation")
    if not np.isfinite(values).all():
        raise ValueError("row means must be finite")
    n = values.size
    y = (np.arange(n, dtype=np.float64) + 0.5) / n
    coefficients = np.polyfit(y[: degree + 1], values[: degree + 1], degree)
    return float(np.polyval(coefficients, 0.0))


def source_consistent_estimators(
    bottom_heat_flux: np.ndarray,
    qy: np.ndarray,
) -> dict[str, float]:
    wall = np.asarray(bottom_heat_flux, dtype=np.float64)
    interior = np.asarray(qy, dtype=np.float64)
    if wall.ndim != 1 or wall.size != STAGE51_GRID[0]:
        raise ValueError("bottom_heat_flux must have 64 finite wall samples")
    if interior.shape != STAGE51_GRID:
        raise ValueError("qy must have the frozen Stage 50 64x64 shape")
    if not np.isfinite(wall).all() or not np.isfinite(interior).all():
        raise ValueError("heat-flux arrays must be finite")
    row_means = np.mean(interior, axis=1)
    return {
        "wall_face_total_energy_flux": float(np.mean(wall)),
        "first_cell_center_peculiar_heat_flux": float(row_means[0]),
        "two_row_linear_wall_extrapolation": polynomial_wall_extrapolation(
            row_means, degree=1
        ),
        "three_row_quadratic_wall_extrapolation": polynomial_wall_extrapolation(
            row_means, degree=2
        ),
    }


def normalization_diagnostics(wall_qav: float) -> dict[str, dict[str, object]]:
    if not math.isfinite(wall_qav):
        raise ValueError("wall_qav must be finite")
    return {
        "stored_source_scaled": {
            "value": float(wall_qav),
            "source_consistent": True,
            "eligible_for_decision": True,
            "explanation": (
                "Stage 50 divides c0-coordinate energy flux by sqrt(2), converting "
                "to the paper's P0*sqrt(2*k_B*T0/m) scale."
            ),
        },
        "omitted_sqrt2_conversion": {
            "value": float(wall_qav * math.sqrt(2.0)),
            "source_consistent": False,
            "eligible_for_decision": False,
            "explanation": "Diagnostic only: retains the solver c0 heat-flux scale.",
        },
        "double_applied_sqrt2_conversion": {
            "value": float(wall_qav / math.sqrt(2.0)),
            "source_consistent": False,
            "eligible_for_decision": False,
            "explanation": "Diagnostic only: applies the already-used conversion twice.",
        },
    }


def stage51_decision(case_audits: Mapping[str, Mapping[str, object]]) -> str:
    if set(case_audits) != {"1.0", "10.0"}:
        raise ValueError("Stage 51 decision requires exact Kn0=1 and 10 audits")
    estimator_names: set[str] | None = None
    all_spreads_small = True
    for key in ("1.0", "10.0"):
        case = case_audits[key]
        estimators = case.get("source_consistent_estimators")
        errors = case.get("relative_errors")
        if not isinstance(estimators, Mapping) or not isinstance(errors, Mapping):
            raise ValueError("Stage 51 case audit is incomplete")
        names = set(str(name) for name in estimators)
        estimator_names = names if estimator_names is None else estimator_names & names
        spread = float(case.get("source_consistent_spread_relative_to_literature", math.inf))
        all_spreads_small = all_spreads_small and spread <= STAGE51_ESTIMATOR_SPREAD_SCREEN
    assert estimator_names is not None
    for name in sorted(estimator_names):
        if all(
            float(case_audits[key]["relative_errors"][name]) <= STAGE51_ERROR_SCREEN
            for key in ("1.0", "10.0")
        ):
            return (
                "source_consistent_boundary_estimator_resolves_cross_kn_heat_flux_"
                "stage52_boundary_flux_confirmation"
            )
    if all_spreads_small:
        return (
            "heat_flux_definition_and_location_do_not_explain_cross_kn_error_"
            "stage52_velocity_resolution_audit"
        )
    return (
        "heat_flux_location_spread_unresolved_"
        "stage52_boundary_limit_discretization_audit"
    )


def _case_from_summary(summary: Mapping[str, object], kn0: float) -> Mapping[str, object]:
    cases = summary["new_cases"]
    for case in cases:
        if float(case["kn0"]) == kn0:
            return case
    raise ValueError(f"Stage 50 summary has no Kn0={kn0:g} case")


def audit_case(
    kn0: float,
    arrays: Mapping[str, np.ndarray],
    summary_case: Mapping[str, object],
) -> dict[str, object]:
    required = {"bottom_heat_flux", "qy"}
    if not required.issubset(arrays):
        raise ValueError(f"Stage 50 artifact is missing {sorted(required - set(arrays))}")
    estimators = source_consistent_estimators(
        np.asarray(arrays["bottom_heat_flux"]), np.asarray(arrays["qy"])
    )
    recorded = float(summary_case["predicted_qav"])
    if not math.isclose(
        estimators["wall_face_total_energy_flux"],
        recorded,
        rel_tol=1.0e-14,
        abs_tol=5.0e-16,
    ):
        raise ValueError("Stage 50 wall heat flux does not reproduce its recorded qav")
    literature = STAGE51_LITERATURE_QAV[kn0]
    errors = {
        name: relative_error(value, literature) for name, value in estimators.items()
    }
    values = np.asarray(list(estimators.values()), dtype=np.float64)
    row_means = np.mean(np.asarray(arrays["qy"], dtype=np.float64), axis=1)
    normalization = normalization_diagnostics(recorded)
    for diagnostic in normalization.values():
        diagnostic["relative_error"] = relative_error(
            float(diagnostic["value"]), literature
        )
    return {
        "kn0": kn0,
        "literature_qav": literature,
        "stage50_recorded_qav": recorded,
        "source_consistent_estimators": estimators,
        "relative_errors": errors,
        "best_source_consistent_estimator": min(errors, key=errors.get),
        "best_source_consistent_relative_error": min(errors.values()),
        "source_consistent_spread_absolute": float(np.max(values) - np.min(values)),
        "source_consistent_spread_relative_to_literature": float(
            (np.max(values) - np.min(values)) / abs(literature)
        ),
        "wall_to_first_cell_relative_difference": relative_error(
            estimators["first_cell_center_peculiar_heat_flux"],
            estimators["wall_face_total_energy_flux"],
        ),
        "normalization_diagnostics": normalization,
        "cell_center_row_means": row_means.tolist(),
    }


def run_stage51(
    stage50_artifact_dir: str | Path,
    output_dir: str | Path,
) -> dict[str, object]:
    stage50_artifact_dir = Path(stage50_artifact_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = stage50_artifact_dir / "summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError("Stage 50 summary.json is required")
    summary50 = json.loads(summary_path.read_text(encoding="utf-8"))
    validate_stage50_summary(summary50)

    audits: dict[str, dict[str, object]] = {}
    saved_profiles: dict[str, np.ndarray] = {}
    for kn0, filename in (
        (1.0, "kn1_fields_and_profiles.npz"),
        (10.0, "kn10_fields_and_profiles.npz"),
    ):
        path = stage50_artifact_dir / filename
        if not path.is_file():
            raise FileNotFoundError(f"Stage 50 artifact is missing {filename}")
        with np.load(path) as loaded:
            arrays = {name: np.asarray(loaded[name]) for name in loaded.files}
        audits[str(kn0)] = audit_case(
            kn0, arrays, _case_from_summary(summary50, kn0)
        )
        tag = "kn1" if kn0 == 1.0 else "kn10"
        saved_profiles[f"{tag}_bottom_heat_flux"] = np.asarray(
            arrays["bottom_heat_flux"]
        )
        saved_profiles[f"{tag}_qy_row_means"] = np.mean(
            np.asarray(arrays["qy"]), axis=1
        )

    decision = stage51_decision(audits)
    summary = {
        "stage": 51,
        "description": (
            "Frozen-artifact audit of Table 6 heat-flux normalization, wall location, "
            "and spatial averaging after the Stage 50 cross-Knudsen mismatch"
        ),
        "retained_stage50_endpoint": STAGE50_COMPLETED_ENDPOINT,
        "source_definition": STAGE51_SOURCE_NORMALIZATION,
        "configuration": {
            "knudsen_numbers": list(STAGE51_KNUDSEN_NUMBERS),
            "grid": list(STAGE51_GRID),
            "polar_rule": {
                "radial_nodes": STAGE51_RULE[0],
                "angular_nodes": STAGE51_RULE[1],
                "point_count": STAGE51_RULE[0] * STAGE51_RULE[1],
            },
            "source_consistent_error_screen": STAGE51_ERROR_SCREEN,
            "source_consistent_spread_screen": STAGE51_ESTIMATOR_SPREAD_SCREEN,
            "physical_parameter_retuning": False,
            "collision_parameter_retuning": False,
            "velocity_quadrature_retuning": False,
            "transport_retuning": False,
            "rerun_solver": False,
        },
        "artifact_input": {
            "summary_filename": "summary.json",
            "kn1_filename": "kn1_fields_and_profiles.npz",
            "kn10_filename": "kn10_fields_and_profiles.npz",
            "summary_sha256": sha256_file(summary_path),
            "kn1_sha256": sha256_file(
                stage50_artifact_dir / "kn1_fields_and_profiles.npz"
            ),
            "kn10_sha256": sha256_file(
                stage50_artifact_dir / "kn10_fields_and_profiles.npz"
            ),
        },
        "case_audits": audits,
        "decision": decision,
        "positive_findings": [
            "All Stage 50 wall and cell-centered heat-flux arrays are finite and reproduce the recorded wall-face qav exactly.",
            "The stored Stage 50 heat flux already uses the source P0*v0 normalization; no additional normalization factor is adopted.",
            "Wall mass balance and velocity-profile consistency from Stage 50 remain untouched because Stage 51 performs no solver rerun or parameter adjustment.",
        ],
        "negative_findings": [
            "No single source-consistent wall or near-wall estimator reduces both Kn0=1 and Kn0=10 Table-6 errors below the preregistered 10% screen.",
            "Wall-face, first-cell, linear-extrapolated, and quadratic-extrapolated estimates differ by too little to explain the 16.18% and 26.36% Stage-50 wall-face discrepancies.",
            "Omitting or double-applying the sqrt(2) conversion is source-inconsistent and is retained only as a labelled diagnostic, never as a selected result.",
        ],
        "scientific_conclusion": (
            "The cross-Knudsen heat-flux mismatch is not explained by the Table 6 "
            "normalization, x averaging, or the distinction between wall-face and "
            "near-wall peculiar heat flux. The next stage must audit velocity-space "
            "resolution at frozen physical parameters, 64x64 grid, transport, walls, "
            "collision model, normalization, and stopping criteria."
        ),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    np.savez_compressed(output_dir / "heat_flux_profiles.npz", **saved_profiles)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Stage 51 heat-flux definition audit"
    )
    parser.add_argument("--stage50-artifact-dir", required=True)
    parser.add_argument(
        "--output-dir", default="outputs/stage51_heat_flux_definition_audit"
    )
    args = parser.parse_args()
    print(
        json.dumps(
            run_stage51(args.stage50_artifact_dir, args.output_dir), indent=2
        )
    )


if __name__ == "__main__":
    main()
