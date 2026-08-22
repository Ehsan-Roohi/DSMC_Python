from __future__ import annotations

from pathlib import Path
import argparse, hashlib, json
import numpy as np

from .linear_sidewall_validation import LinearSidewallConfig
from .stage41_projected_polar_operator_audit import (
    STAGE41_CORRECTION_FLOOR, STAGE41_PRANDTL, mapped_polar_quadrature,
)
from .stage42_projected_polar_heated_cavity_pilot import solve_stage42_pilot
from .stage57_conservative_solver_pilot import (
    STAGE57_CHECK_INTERVAL, STAGE57_KNUDSEN, STAGE57_MAX_ITERATIONS,
    STAGE57_MAX_OBSERVABLE_ERROR_DEGRADATION, STAGE57_MINIMUM_ITERATIONS,
    STAGE57_RADIAL_SCALE, STAGE57_RATIO, STAGE57_RULE,
    STAGE57_SOURCE_RELAXATION, STAGE57_TOLERANCE, _compact_result,
    compare_arms, solve_conservative_stage57, stage57_decision,
)

STAGE57_COMPLETED_ENDPOINT = {
    "workflow_run_id": 30859493733,
    "workflow_job_id": 91838121463,
    "workflow_conclusion": "success",
    "tests_passed": 71,
    "tests_failed": 0,
    "test_duration_seconds": 0.38,
    "artifact_id": 8875278804,
    "artifact_size_bytes": 22166,
    "artifact_sha256": "17f4a660e1c528308bc7ece461fdecb22e93a51f01d910220236b9aae33f8210",
    "source_head_sha": "5f997fb49afb558c06464b2d50e499c3793bd03e",
    "summary_sha256": "28b0d71ac69a3a3532769d4fd501eef7c8985d6a0457e0f5534c94b64a7cf78e",
    "baseline_fields_sha256": "c4f9d5fd556f6bd3904bc79d12a45cc67eb46df1d2fd2837702a2dd8de35e437",
    "conservative_fields_sha256": "14ec32791020930ac606b15860759d1707666e8670774e719bfcf32c201af1f3",
    "decision": "stage57_conservative_solver_pilot_passes_stage58_frozen_64x64_confirmation",
}
STAGE58_GRID = (64, 64)
STAGE58_KNUDSEN = STAGE57_KNUDSEN
STAGE58_RATIO = STAGE57_RATIO
STAGE58_RULE = STAGE57_RULE
STAGE58_RADIAL_SCALE = STAGE57_RADIAL_SCALE
STAGE58_SOURCE_RELAXATION = STAGE57_SOURCE_RELAXATION
STAGE58_MAX_ITERATIONS = STAGE57_MAX_ITERATIONS
STAGE58_MINIMUM_ITERATIONS = STAGE57_MINIMUM_ITERATIONS
STAGE58_CHECK_INTERVAL = STAGE57_CHECK_INTERVAL
STAGE58_TOLERANCE = STAGE57_TOLERANCE


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_stage58_design(
    grid=STAGE58_GRID, kn0=STAGE58_KNUDSEN,
    cold_hot_ratio=STAGE58_RATIO, rule=STAGE58_RULE,
    radial_scale=STAGE58_RADIAL_SCALE,
    source_relaxation=STAGE58_SOURCE_RELAXATION,
    max_iterations=STAGE58_MAX_ITERATIONS,
    minimum_iterations=STAGE58_MINIMUM_ITERATIONS,
    check_interval=STAGE58_CHECK_INTERVAL,
    tolerance=STAGE58_TOLERANCE,
    correction_floor=STAGE41_CORRECTION_FLOOR,
) -> None:
    actual = (grid, kn0, cold_hot_ratio, rule, radial_scale, source_relaxation,
              max_iterations, minimum_iterations, check_interval, tolerance,
              correction_floor)
    expected = (STAGE58_GRID, STAGE58_KNUDSEN, STAGE58_RATIO, STAGE58_RULE,
                STAGE58_RADIAL_SCALE, STAGE58_SOURCE_RELAXATION,
                STAGE58_MAX_ITERATIONS, STAGE58_MINIMUM_ITERATIONS,
                STAGE58_CHECK_INTERVAL, STAGE58_TOLERANCE,
                STAGE41_CORRECTION_FLOOR)
    if actual != expected:
        raise ValueError(
            "Stage 58 is frozen to one paired 64x64 Kn0=10 confirmation, "
            "the 40x96 expanded-tail rule at radial scale 2.0, the retained "
            "0.05 correction floor, source relaxation 1.0 and existing stopping rules."
        )


def _validate_stage57_artifact(root: Path) -> dict[str, object]:
    expected = {
        "summary.json": STAGE57_COMPLETED_ENDPOINT["summary_sha256"],
        "baseline_clipped_fields_and_profiles.npz":
            STAGE57_COMPLETED_ENDPOINT["baseline_fields_sha256"],
        "conservative_fields_and_profiles.npz":
            STAGE57_COMPLETED_ENDPOINT["conservative_fields_sha256"],
    }
    for name, checksum in expected.items():
        path = root / name
        if not path.is_file() or sha256_file(path) != checksum:
            raise ValueError(f"Stage 57 artifact checksum mismatch: {name}")
    summary = json.loads((root / "summary.json").read_text())
    if summary.get("stage") != 57 or summary.get("decision") != STAGE57_COMPLETED_ENDPOINT["decision"]:
        raise ValueError("Stage 57 artifact endpoint mismatch")
    return summary


def build_stage58_config() -> LinearSidewallConfig:
    return LinearSidewallConfig(
        nx=64, ny=64, kn0=STAGE58_KNUDSEN,
        cold_hot_ratio=STAGE58_RATIO, viscosity_exponent=0.5,
        prandtl=STAGE41_PRANDTL, max_steps=STAGE58_MAX_ITERATIONS,
        cfl=0.2, tolerance=STAGE58_TOLERANCE,
        check_interval=STAGE58_CHECK_INTERVAL,
        minimum_steps=STAGE58_MINIMUM_ITERATIONS,
        positivity_floor=1.0e-30,
    )


def stage58_decision(baseline, conservative, comparison, projection) -> str:
    retained = stage57_decision(baseline, conservative, comparison, projection)
    mapping = {
        "stage57_conservative_solver_pilot_passes_stage58_frozen_64x64_confirmation":
            "stage58_frozen_64x64_confirmation_passes_independent_reference_review_before_any_extension",
        "stage57_conservative_solver_stable_but_observables_degrade_requires_review_before_full_resolution":
            "stage58_conservative_confirmation_stable_but_observables_degrade_requires_review_without_retuning",
        "stage57_stable_nonconverged_blocker_without_parameter_retuning":
            "stage58_stable_nonconverged_blocker_without_parameter_retuning",
        "stage57_conservative_projection_in_solver_blocker_requires_review":
            "stage58_conservative_projection_closure_blocker_requires_review",
        "stage57_conservative_solver_numerical_blocker_requires_review":
            "stage58_conservative_solver_numerical_blocker_requires_review",
        "stage57_retained_clipped_baseline_numerical_blocker":
            "stage58_retained_clipped_baseline_numerical_blocker",
        "stage57_conservative_solver_nonfinite_blocker_requires_review":
            "stage58_nonfinite_blocker_requires_review",
    }
    return mapping[retained]


def _save_fields(path: Path, raw: dict[str, object]) -> None:
    keys = ("T", "rho", "u", "v", "qx", "qy", "left_wall_velocity",
            "table_velocity", "bottom_heat_flux", "residual_history")
    np.savez_compressed(path, **{key: np.asarray(raw[key]) for key in keys})


def _resolution_changes(retained57, conservative58) -> dict[str, float]:
    prior = retained57["conservative_projection"]
    p0 = np.asarray(prior["table_velocity"], dtype=float)
    p1 = np.asarray(conservative58["table_velocity"], dtype=float)
    return {
        "conservative_qav_relative_change_16x16_to_64x64":
            abs(float(conservative58["predicted_qav"]) - float(prior["predicted_qav"]))
            / max(abs(float(prior["predicted_qav"])), 1e-300),
        "conservative_qav_error_change_fraction_16x16_to_64x64":
            (float(conservative58["qav_relative_error"]) - float(prior["qav_relative_error"]))
            / max(float(prior["qav_relative_error"]), 1e-300),
        "conservative_velocity_rms_change_fraction_16x16_to_64x64":
            (float(conservative58["velocity_metrics"]["relative_rms"])
             - float(prior["velocity_metrics"]["relative_rms"]))
            / max(float(prior["velocity_metrics"]["relative_rms"]), 1e-300),
        "conservative_table_velocity_profile_change_16x16_to_64x64":
            float(np.linalg.norm(p1 - p0) / max(np.linalg.norm(p0), 1e-300)),
    }


def run_stage58(stage57_artifact_dir: str | Path, output_dir: str | Path, **design):
    validate_stage58_design(**design)
    retained57 = _validate_stage57_artifact(Path(stage57_artifact_dir))
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    cfg = build_stage58_config()
    quadrature = mapped_polar_quadrature(*STAGE58_RULE, radial_scale=STAGE58_RADIAL_SCALE)

    baseline_raw = solve_stage42_pilot(cfg, quadrature, STAGE58_SOURCE_RELAXATION)
    baseline = _compact_result(baseline_raw)
    _save_fields(out / "baseline_clipped_fields_and_profiles.npz", baseline_raw)
    del baseline_raw

    conservative_raw = solve_conservative_stage57(cfg, quadrature)
    conservative = _compact_result(conservative_raw)
    projection = {key: float(value) for key, value in
                  conservative_raw["projection_diagnostics"].items()}
    _save_fields(out / "conservative_fields_and_profiles.npz", conservative_raw)
    comparison = compare_arms(baseline, conservative)
    resolution = _resolution_changes(retained57, conservative)
    decision = stage58_decision(baseline, conservative, comparison, projection)

    summary = {
        "stage": 58,
        "description": "Frozen paired 64x64 confirmation of the Stage-57 bounded conservative positivity projection against the retained clipped projected-Shakhov operator at Kn0=10.",
        "retained_stage57_endpoint": STAGE57_COMPLETED_ENDPOINT,
        "retained_stage57_decision": retained57["decision"],
        "configuration": {
            "kn0": 10.0, "cold_hot_ratio": 0.1, "grid": [64, 64],
            "radial_nodes": 40, "angular_nodes": 96, "point_count": 3840,
            "radial_scale": 2.0, "prandtl": STAGE41_PRANDTL,
            "retained_correction_floor": STAGE41_CORRECTION_FLOOR,
            "source_relaxation": 1.0, "max_iterations": STAGE58_MAX_ITERATIONS,
            "minimum_iterations": STAGE58_MINIMUM_ITERATIONS,
            "check_interval": STAGE58_CHECK_INTERVAL, "tolerance": STAGE58_TOLERANCE,
            "maximum_observable_error_degradation": STAGE57_MAX_OBSERVABLE_ERROR_DEGRADATION,
            "physical_parameter_retuning": False,
            "collision_parameter_retuning": False, "transport_retuning": False,
            "wall_model_retuning": False, "normalization_retuning": False,
            "correction_floor_retuning": False, "source_relaxation_retuning": False,
            "velocity_quadrature_retuning": False,
            "paired_retained_clipped_baseline": True,
            "cross_knudsen_extension_permitted": False,
            "resolution_is_confirmation_not_parameter_search": True,
        },
        "baseline_clipped": baseline,
        "conservative_projection": conservative,
        "projection_diagnostics": projection,
        "paired_comparison": comparison,
        "resolution_change_from_stage57_pilot": resolution,
        "decision": decision,
        "interpretation_guard": "The only design change from Stage 57 is the preregistered grid increase from 16x16 to 64x64. Physics, walls, normalization, 40x96 quadrature, radial scale 2.0, correction floor, relaxation, positivity floor and stopping rules are frozen. Both positive and negative endpoints are retained without retuning.",
        "scientific_conclusion": "A pass establishes only stable moment-conservative operation at the frozen 64x64 confirmation resolution without material paired-observable degradation. It is not validation and does not authorize cross-Knudsen extension; independent-reference review is required.",
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage57-artifact-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    print(json.dumps(run_stage58(args.stage57_artifact_dir, args.output_dir),
                     indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
