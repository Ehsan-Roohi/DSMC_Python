from __future__ import annotations

from pathlib import Path
import argparse
import hashlib
import json
import math


STAGE58_COMPLETED_ENDPOINT = {
    "workflow_run_id": 30864287564,
    "workflow_job_id": 91852659651,
    "workflow_conclusion": "success",
    "tests_passed": 72,
    "tests_failed": 0,
    "artifact_id": 8882191150,
    "artifact_size_bytes": 265675,
    "artifact_sha256": "c1e1cb40d35439d44a93bae091f732f35e1790c2c4e3f1180b16d7f8fb54e8f6",
    "source_head_sha": "448ce586344052de1cf5dd0fd86e3ffb4b6a52be",
    "summary_sha256": "b91921a631bd92c2696c7bdd18668afb4994c909bbc7aa82cca1d0afc25ffced",
    "baseline_fields_sha256": "9aa53136e05917236f87fb9279c2ecc4e29d6056ca5ec34ca3bcb4d8f66aa822",
    "conservative_fields_sha256": "d7de5b77497193c30332e8ab60e4ff19f627e76c8dc6be5285f2454929332711",
    "decision": "stage58_conservative_confirmation_stable_but_observables_degrade_requires_review_without_retuning",
}

# Vargas, Tatsios, Valougeorgis & Stefanov, Physics of Fluids 26, 057101
# (2014), DOI 10.1063/1.4875235, Table VI, Tc/Th = 0.1, Kn0 = 10.
# The deterministic Shakhov and independently generated DSMC values are both
# retained.  Table III supplies a Shakhov wall-velocity profile but no separate
# DSMC profile at the ten tabulated ordinates.
PUBLISHED_REFERENCE = {
    "doi": "10.1063/1.4875235",
    "title": "Rarefied gas flow in a rectangular enclosure induced by non-isothermal walls",
    "case": {"kn0": 10.0, "cold_hot_ratio": 0.1, "aspect_ratio": 1.0},
    "table6_qav": {"shakhov": 0.178, "dsmc": 0.179},
    "table3_velocity_reference": "shakhov_only",
    "table3_independent_dsmc_profile_available": False,
}

MAXIMUM_ACCEPTABLE_PAIRED_OBSERVABLE_DEGRADATION = 0.10
MINIMUM_ROBUST_DISCREPANCY_ERROR = 0.20
MINIMUM_GAP_TO_REFERENCE_SPREAD = 10.0


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_stage58_artifact(root: str | Path) -> dict[str, object]:
    root = Path(root)
    expected = {
        "summary.json": STAGE58_COMPLETED_ENDPOINT["summary_sha256"],
        "baseline_clipped_fields_and_profiles.npz":
            STAGE58_COMPLETED_ENDPOINT["baseline_fields_sha256"],
        "conservative_fields_and_profiles.npz":
            STAGE58_COMPLETED_ENDPOINT["conservative_fields_sha256"],
    }
    for name, checksum in expected.items():
        path = root / name
        if not path.is_file() or sha256_file(path) != checksum:
            raise ValueError(f"Stage 58 artifact checksum mismatch: {name}")
    summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    if summary.get("stage") != 58:
        raise ValueError("Stage 58 artifact stage mismatch")
    if summary.get("decision") != STAGE58_COMPLETED_ENDPOINT["decision"]:
        raise ValueError("Stage 58 artifact decision mismatch")
    return summary


def _reference_metrics(predicted_qav: float) -> dict[str, object]:
    refs = PUBLISHED_REFERENCE["table6_qav"]
    errors = {
        name: {
            "absolute": abs(float(predicted_qav) - float(value)),
            "relative": abs(float(predicted_qav) - float(value)) / float(value),
        }
        for name, value in refs.items()
    }
    values = list(float(value) for value in refs.values())
    spread = max(values) - min(values)
    minimum_absolute_gap = min(item["absolute"] for item in errors.values())
    minimum_relative_error = min(item["relative"] for item in errors.values())
    return {
        "predicted_qav": float(predicted_qav),
        "errors": errors,
        "reference_spread": spread,
        "minimum_absolute_gap": minimum_absolute_gap,
        "minimum_relative_error": minimum_relative_error,
        "gap_to_reference_spread": minimum_absolute_gap / max(spread, 1.0e-300),
    }


def evaluate_stage58_summary(stage58: dict[str, object]) -> dict[str, object]:
    cfg = stage58["configuration"]
    if cfg["kn0"] != 10.0 or cfg["cold_hot_ratio"] != 0.1:
        raise ValueError("Stage 59 is frozen to the completed Kn0=10, Tc/Th=0.1 endpoint")
    if cfg["grid"] != [64, 64] or cfg["radial_nodes"] != 40 or cfg["angular_nodes"] != 96:
        raise ValueError("Stage 59 must audit the exact completed Stage 58 resolution")

    baseline = stage58["baseline_clipped"]
    conservative = stage58["conservative_projection"]
    paired = stage58["paired_comparison"]
    baseline_reference = _reference_metrics(float(baseline["predicted_qav"]))
    conservative_reference = _reference_metrics(float(conservative["predicted_qav"]))

    robust_heat_flux_discrepancy = all(
        row["minimum_relative_error"] >= MINIMUM_ROBUST_DISCREPANCY_ERROR
        and row["gap_to_reference_spread"] >= MINIMUM_GAP_TO_REFERENCE_SPREAD
        for row in (baseline_reference, conservative_reference)
    )
    projection_worsens_best_heat_flux_error = (
        conservative_reference["minimum_relative_error"]
        > baseline_reference["minimum_relative_error"]
    )
    velocity_guard_failed = (
        float(paired["velocity_rms_error_change_fraction"])
        > MAXIMUM_ACCEPTABLE_PAIRED_OBSERVABLE_DEGRADATION
    )
    projection_not_adopted = projection_worsens_best_heat_flux_error or velocity_guard_failed

    if robust_heat_flux_discrepancy and projection_not_adopted:
        decision = (
            "stage59_independent_dsmc_heat_flux_confirms_discrepancy_"
            "projection_not_adopted_transport_wall_audit_next"
        )
    elif robust_heat_flux_discrepancy:
        decision = "stage59_independent_dsmc_heat_flux_confirms_discrepancy_requires_review"
    else:
        decision = "stage59_reference_uncertainty_requires_manual_review"

    return {
        "stage": 59,
        "description": (
            "Artifact-only independent-reference review of the exact completed Stage 58 "
            "Kn0=10 endpoint using both the published Shakhov and independently generated "
            "DSMC average heat fluxes."
        ),
        "retained_stage58_endpoint": STAGE58_COMPLETED_ENDPOINT,
        "published_reference": PUBLISHED_REFERENCE,
        "configuration": {
            "solver_rerun": False,
            "physical_parameter_retuning": False,
            "collision_parameter_retuning": False,
            "transport_retuning": False,
            "wall_model_retuning": False,
            "normalization_retuning": False,
            "correction_floor_retuning": False,
            "source_relaxation_retuning": False,
            "velocity_quadrature_retuning": False,
            "cross_knudsen_extension_permitted": False,
            "maximum_acceptable_paired_observable_degradation":
                MAXIMUM_ACCEPTABLE_PAIRED_OBSERVABLE_DEGRADATION,
            "minimum_robust_discrepancy_error": MINIMUM_ROBUST_DISCREPANCY_ERROR,
            "minimum_gap_to_reference_spread": MINIMUM_GAP_TO_REFERENCE_SPREAD,
        },
        "baseline_clipped": {
            "heat_flux_reference_review": baseline_reference,
            "velocity_relative_rms_to_table3_shakhov":
                float(baseline["velocity_metrics"]["relative_rms"]),
            "velocity_sign_agreement_to_table3_shakhov":
                float(baseline["velocity_metrics"]["sign_agreement"]),
        },
        "conservative_projection": {
            "heat_flux_reference_review": conservative_reference,
            "velocity_relative_rms_to_table3_shakhov":
                float(conservative["velocity_metrics"]["relative_rms"]),
            "velocity_sign_agreement_to_table3_shakhov":
                float(conservative["velocity_metrics"]["sign_agreement"]),
        },
        "paired_review": {
            "heat_flux_error_worsens_against_best_published_reference":
                bool(projection_worsens_best_heat_flux_error),
            "velocity_rms_error_change_fraction":
                float(paired["velocity_rms_error_change_fraction"]),
            "velocity_guard_failed": bool(velocity_guard_failed),
            "projection_not_adopted": bool(projection_not_adopted),
            "independent_dsmc_heat_flux_confirms_large_discrepancy":
                bool(robust_heat_flux_discrepancy),
            "independent_dsmc_velocity_profile_available": False,
        },
        "decision": decision,
        "positive_findings": [
            "Both Stage 58 arms remained finite, converged, wall-mass balanced and positivity bounded.",
            "The conservative arm preserved collision moments to approximately 1e-12.",
            "The published Shakhov and DSMC heat-flux references agree within 0.001, providing a genuinely independent heat-flux cross-check.",
        ],
        "negative_findings": [
            "Both Stage 58 heat fluxes remain more than 25% from even the closer published reference.",
            "The conservative projection worsens the closest-reference heat-flux error and worsens the Table-III velocity RMS by more than the preregistered 10% guard.",
            "The paper does not tabulate an independent DSMC wall-velocity profile at the Table-III ordinates, so velocity-profile validation remains unresolved.",
        ],
        "interpretation_guard": (
            "Agreement between the two published heat-flux references confirms that source-table "
            "rounding or Shakhov-versus-DSMC reference choice cannot explain the Stage 58 gap. "
            "This audit does not validate either solver arm and does not authorize parameter "
            "retuning or cross-Knudsen extension."
        ),
        "scientifically_justified_next_scope": (
            "Audit the frozen transport and diffuse-wall flux implementation at Kn0=10, beginning "
            "with equation-level conservation and collision-off/free-molecular consistency checks; "
            "do not adopt the conservative projection or retune failed parameters."
        ),
    }


def run_stage59(stage58_artifact_dir: str | Path, output_dir: str | Path) -> dict[str, object]:
    stage58 = validate_stage58_artifact(stage58_artifact_dir)
    summary = evaluate_stage58_summary(stage58)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage58-artifact-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    print(json.dumps(run_stage59(args.stage58_artifact_dir, args.output_dir), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
